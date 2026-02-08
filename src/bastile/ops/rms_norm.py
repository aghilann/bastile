"""
CuTile RMSNorm - ported from TileGym with full backward support and autotuning.

Key optimizations:
1. Two forward variants: gather-based and static persistent
2. Full backward kernel with gradient computation for dx and dw
3. Autotuning to select best mode and tile sizes
4. Float32 for numerical stability
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple

from ..registry import register_patch
from ..autotune import autotune

import cuda.tile as ct

ConstInt = ct.Constant[int]
ConstFloat = ct.Constant[float]


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


@dataclass
class RMSNormConfig:
    """Configuration for RMSNorm kernel autotuning."""
    use_static_persistent: bool
    tile_size_m: int  # For persistent mode
    tile_size_n: int  # Tile size for columns
    
    def __hash__(self):
        return hash((self.use_static_persistent, self.tile_size_m, self.tile_size_n))


def rms_norm_search_space(M: int, N: int, num_sms: int):
    """Generate search space for RMSNorm autotuning."""
    TILE_SIZE_N = next_power_of_2(N)
    
    # Gather mode configs (one block per row)
    for tile_n in [256, 512, 1024, 2048, 4096]:
        if tile_n >= N:
            yield RMSNormConfig(
                use_static_persistent=False,
                tile_size_m=1,
                tile_size_n=min(tile_n, TILE_SIZE_N),
            )
    
    # Static persistent mode configs (for larger batches)
    for tile_m in [2, 4, 8, 16]:
        yield RMSNormConfig(
            use_static_persistent=True,
            tile_size_m=tile_m,
            tile_size_n=TILE_SIZE_N,
        )


@ct.kernel(occupancy=2)
def rms_norm_backward_kernel(
    dx,           # [M, N] output gradient w.r.t. input
    dy,           # [M, N] upstream gradient
    x,            # [M, N] input tensor
    weight,       # [N] weight tensor
    rstd,         # [M] reciprocal std from forward
    temp_buffer,  # [M, N] temporary buffer for dw computation
    N: ConstInt,
    TILE_SIZE: ConstInt,
):
    """
    Compute input gradients for RMSNorm backward pass.
    """
    row_idx = ct.bid(0)

    input_row = ct.load(x, index=(row_idx, 0), shape=(1, TILE_SIZE), padding_mode=ct.PaddingMode.ZERO)
    gradient_row = ct.load(dy, index=(row_idx, 0), shape=(1, TILE_SIZE), padding_mode=ct.PaddingMode.ZERO)

    input_row = ct.astype(input_row, ct.float32)
    gradient_row = ct.astype(gradient_row, ct.float32)

    inv_std_row = ct.load(rstd, index=(row_idx,), shape=(1,), padding_mode=ct.PaddingMode.ZERO)
    inv_std_row = ct.reshape(inv_std_row, (1, 1))

    weight_vector = ct.load(weight, index=(0,), shape=(TILE_SIZE,), padding_mode=ct.PaddingMode.ZERO)
    weight_vector = ct.astype(weight_vector, ct.float32)
    weight_vector = ct.reshape(weight_vector, (1, TILE_SIZE))

    c1 = input_row * gradient_row
    c2 = c1 * inv_std_row
    ct.store(temp_buffer, index=(row_idx, 0), tile=c2)

    weighted_gradient_product = c1 * weight_vector
    weighted_gradient_sum = ct.sum(weighted_gradient_product, axis=1, keepdims=True)

    inv_std_cubed = inv_std_row * inv_std_row * inv_std_row
    norm_factor = ct.full((1, 1), float(N), dtype=ct.float32)
    normalization_correction_coeff = input_row * inv_std_cubed / norm_factor
    normalization_correction = normalization_correction_coeff * weighted_gradient_sum

    scaled_gradient = gradient_row * weight_vector * inv_std_row

    input_gradient_row = scaled_gradient - normalization_correction

    input_gradient_row = ct.astype(input_gradient_row, dx.dtype)

    ct.store(dx, index=(row_idx, 0), tile=input_gradient_row)


@ct.kernel
def rms_norm_kernel_gather(
    x,       # [M, N] input tensor
    w,       # [N] weight tensor
    out,     # [M, N] output tensor
    rstd,    # [M] reciprocal std output
    N: ConstInt,
    eps: ConstFloat,
    TILE_SIZE: ConstInt,
):
    """
    RMSNorm forward kernel using gather/scatter (for smaller batches).
    One block per row, processes columns in tiles.
    """
    row = ct.bid(0)
    _rms = ct.full((TILE_SIZE,), 0.0, dtype=ct.float32)
    num_tiles = (N + TILE_SIZE - 1) // TILE_SIZE
    offsets = ct.arange(TILE_SIZE, dtype=ct.int32)

    for j in range(num_tiles):
        offs = j * TILE_SIZE + offsets
        xj = ct.gather(x, (row, offs), check_bounds=True, padding_value=0.0, latency=1)
        xj = ct.astype(xj, ct.float32)
        _rms = _rms + xj * xj

    rms = ct.rsqrt(ct.sum(_rms, axis=0, keepdims=False) / N + eps)
    ct.scatter(rstd, row, rms, check_bounds=True)

    for j in range(num_tiles):
        offs = j * TILE_SIZE + offsets
        wj = ct.gather(w, offs, check_bounds=True, padding_value=1.0, latency=1)
        wj = ct.astype(wj, ct.float32)
        xj = ct.gather(x, (row, offs), check_bounds=True, padding_value=0.0, latency=1)
        xj = ct.astype(xj, ct.float32)
        yj = xj * rms * wj
        yj = ct.astype(yj, x.dtype)
        ct.scatter(out, (row, offs), yj, check_bounds=True, latency=1)


@ct.kernel
def rms_norm_kernel_static_persistent(
    X,          # [M, N] input tensor
    Y,          # [M, N] output tensor
    W,          # [N] weight tensor
    Rstd,       # [M] reciprocal std output
    TILE_SIZE_M: ConstInt,
    TILE_SIZE_N: ConstInt,
    N: ConstInt,
    eps: ConstFloat,
):
    """
    Static persistent RMSNorm kernel for larger batches.
    Each block processes multiple rows for better efficiency.
    """
    bid = ct.bid(0)
    M = X.shape[0]
    upper_bound = (M + TILE_SIZE_M - 1) // TILE_SIZE_M

    w = ct.load(W, index=(0,), shape=(TILE_SIZE_N,), padding_mode=ct.PaddingMode.ZERO)
    w = ct.astype(w, ct.float32)

    num_tile_blocks = ct.num_blocks(0)
    for current_bid in range(bid, upper_bound, num_tile_blocks):
        x = ct.load(
            X,
            index=(current_bid, 0),
            shape=(TILE_SIZE_M, TILE_SIZE_N),
            latency=10,
            padding_mode=ct.PaddingMode.ZERO,
        )
        x = ct.astype(x, ct.float32)

        x_squared = x * x
        x2_sum = ct.sum(x_squared, axis=1, keepdims=True)

        N_f32 = ct.full((TILE_SIZE_M, 1), float(N), dtype=ct.float32)
        variance = x2_sum / N_f32

        eps_tensor = ct.full((TILE_SIZE_M, 1), eps, dtype=ct.float32)
        variance_eps = variance + eps_tensor
        rsqrt_var = ct.rsqrt(variance_eps)

        rstd_flat = ct.reshape(rsqrt_var, (TILE_SIZE_M,))
        row_offsets = ct.arange(TILE_SIZE_M, dtype=ct.int32) + current_bid * TILE_SIZE_M
        ct.scatter(Rstd, row_offsets, rstd_flat, check_bounds=True)

        x_normalized = x * rsqrt_var

        w_broadcasted = ct.reshape(w, (1, TILE_SIZE_N))
        y = x_normalized * w_broadcasted

        y = ct.astype(y, X.dtype)
        ct.store(
            Y,
            index=(current_bid, 0),
            tile=y,
            allow_tma=False,
            latency=3,
        )


def rms_norm_backward(
    x: torch.Tensor,
    dy: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CuTile RMSNorm backward pass."""
    x = x.contiguous()
    dy = dy.contiguous()
    weight = weight.contiguous()
    rstd = rstd.contiguous()

    x_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    dy = dy.reshape(-1, dy.shape[-1])

    M, N = x.shape

    dx = torch.empty_like(x)
    temp_buffer = torch.empty(x.shape, device=x.device, dtype=torch.float32)

    TILE_SIZE_N = next_power_of_2(N)

    grid_dx = (M,)
    ct.launch(
        torch.cuda.current_stream(),
        grid_dx,
        rms_norm_backward_kernel,
        (dx, dy, x, weight, rstd, temp_buffer, N, TILE_SIZE_N),
    )

    dw = temp_buffer[:, :N].to(torch.float32).sum(dim=0).to(weight.dtype)

    return dx.view(*x_shape), dw


def _run_rms_norm_with_config(
    x_flat: torch.Tensor,
    weight: torch.Tensor,
    y: torch.Tensor,
    rstd: torch.Tensor,
    N: int,
    eps: float,
    config: RMSNormConfig,
    num_sms: int,
):
    """Run RMSNorm with a specific config."""
    M = x_flat.shape[0]
    
    if config.use_static_persistent:
        grid_size = min(num_sms, ((M + config.tile_size_m - 1) // config.tile_size_m))
        grid = (grid_size,)
        
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            rms_norm_kernel_static_persistent,
            (x_flat, y, weight, rstd, config.tile_size_m, config.tile_size_n, N, eps),
        )
    else:
        grid = (M,)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            rms_norm_kernel_gather,
            (x_flat, weight, y, rstd, N, eps, config.tile_size_n),
        )


class RMSNormFunction(torch.autograd.Function):
    """CuTile RMSNorm with full backward support and autotuning."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        use_static_persistent: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        RMSNorm forward pass with autotuning.
        """
        x = x.contiguous()
        weight = weight.contiguous()

        x_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        M, N = x_flat.shape

        y = torch.empty_like(x_flat)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)

        NUM_SMS = torch.cuda.get_device_properties(x.device).multi_processor_count

        # If mode is forced, use it directly
        if use_static_persistent is not None:
            TILE_SIZE_N = next_power_of_2(N)
            if use_static_persistent:
                if TILE_SIZE_N <= 1024:
                    TILE_SIZE_M = 16
                elif TILE_SIZE_N >= 16384:
                    TILE_SIZE_M = 2
                else:
                    TILE_SIZE_M = 4
                config = RMSNormConfig(True, TILE_SIZE_M, TILE_SIZE_N)
            else:
                MAX_FUSED_SIZE = 4096 // x.element_size()
                TILE_SIZE = min(MAX_FUSED_SIZE, TILE_SIZE_N)
                config = RMSNormConfig(False, 1, TILE_SIZE)
        else:
            # Autotune
            key = (M, N, str(x.dtype))
            
            def run_with_config(cfg):
                _run_rms_norm_with_config(x_flat, weight, y, rstd, N, eps, cfg, NUM_SMS)
            
            config = autotune(
                kernel_name="rms_norm",
                run_fn=run_with_config,
                search_space=list(rms_norm_search_space(M, N, NUM_SMS)),
                key=str(key),
                max_iter=8,
            )

        # Run with selected config
        _run_rms_norm_with_config(x_flat, weight, y, rstd, N, eps, config, NUM_SMS)

        ctx.save_for_backward(x, weight, rstd)
        ctx.eps = eps

        return y.view(*x_shape)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        """RMSNorm backward pass."""
        x, weight, rstd = ctx.saved_tensors
        dx, dw = rms_norm_backward(x, dy, weight, rstd)
        return dx, dw, None, None


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    use_static_persistent: Optional[bool] = None,
) -> torch.Tensor:
    """
    Apply CuTile RMSNorm with autotuning.
    """
    return RMSNormFunction.apply(x, weight, eps, use_static_persistent)


class CuTileRMSNorm(nn.Module):
    """Drop-in replacement for Qwen3RMSNorm using CuTile kernels with autotuning."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return RMSNormFunction.apply(
            hidden_states, self.weight, self.variance_epsilon, None
        )

    def extra_repr(self) -> str:
        return f"{self.weight.shape[0]}, eps={self.variance_epsilon}"


# Register patches
register_patch(
    name="rms_norm_qwen3",
    description="CuTile RMSNorm with autotuning for Qwen3",
    target_module="transformers.models.qwen3.modeling_qwen3",
    target_attr="Qwen3RMSNorm",
    replacement=CuTileRMSNorm,
    has_backward=True,
    priority=10,
    models=["qwen3"],
)

register_patch(
    name="rms_norm_gpt_oss",
    description="CuTile RMSNorm with autotuning for GPT-OSS",
    target_module="transformers.models.gpt_oss.modeling_gpt_oss",
    target_attr="GptOssRMSNorm",
    replacement=CuTileRMSNorm,
    has_backward=True,
    priority=10,
    models=["gpt_oss"],
)
