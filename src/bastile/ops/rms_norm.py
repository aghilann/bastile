"""
RMSNorm with CuTile kernel - supports forward and backward passes.

Based on TileGym's RMSNorm implementation.
"""

import torch
import torch.nn as nn
import cuda.tile as ct

from ..registry import register_patch


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


# ============================================================================
# CuTile Kernels
# ============================================================================

@ct.kernel(occupancy=2)
def rms_norm_backward_kernel(
    dx,
    dy,
    x,
    weight,
    Rstd,
    temp_buffer,
    TILE_SIZE: ct.Constant[int],
):
    """Compute input gradients for RMSNorm backward pass."""
    row_idx = ct.bid(0)
    M, N = x.shape

    input_row = ct.load(x, index=(row_idx, 0), shape=(1, TILE_SIZE), padding_mode=ct.PaddingMode.ZERO)
    gradient_row = ct.load(dy, index=(row_idx, 0), shape=(1, TILE_SIZE), padding_mode=ct.PaddingMode.ZERO)
    inv_std_row = ct.load(Rstd, index=(row_idx,), shape=(1,), padding_mode=ct.PaddingMode.ZERO)
    inv_std_row = ct.reshape(inv_std_row, (1, 1))
    weight_vector = ct.load(weight, index=(0,), shape=(TILE_SIZE,), padding_mode=ct.PaddingMode.ZERO)
    weight_vector = ct.reshape(weight_vector, (1, TILE_SIZE))

    c1 = input_row * gradient_row
    c2 = c1 * inv_std_row
    ct.store(temp_buffer, index=(row_idx, 0), tile=ct.astype(c2, temp_buffer.dtype))

    weighted_gradient_product = c1 * weight_vector
    weighted_gradient_sum = ct.sum(weighted_gradient_product, axis=1, keepdims=True)

    inv_std_cubed = inv_std_row * inv_std_row * inv_std_row
    norm_factor = ct.full((1, 1), N * 1.0, dtype=ct.float32)
    normalization_correction_coeff = input_row * inv_std_cubed / norm_factor
    normalization_correction = normalization_correction_coeff * weighted_gradient_sum

    scaled_gradient = gradient_row * weight_vector * inv_std_row
    input_gradient_row = scaled_gradient - normalization_correction
    input_gradient_row = ct.astype(input_gradient_row, dx.dtype)
    ct.store(dx, index=(row_idx, 0), tile=input_gradient_row)


@ct.kernel
def rms_norm_kernel_gather(
    x,
    w,
    out,
    Rstd,
    N: ct.Constant[int],
    eps: ct.Constant[float],
    TILE_SIZE: ct.Constant[int],
):
    """Standard RMSNorm kernel with gather/scatter."""
    row = ct.bid(0)
    _rms = ct.full((TILE_SIZE,), 0.0, dtype=ct.float32)
    num_tiles = ct.cdiv(N, TILE_SIZE)
    offsets = ct.arange(TILE_SIZE, dtype=ct.int32)

    for j in range(0, num_tiles):
        offs = j * TILE_SIZE + offsets
        xj = ct.gather(x, (row, offs), latency=1)
        xj = ct.astype(xj, ct.float32)
        _rms += xj * xj

    rms = ct.rsqrt(ct.sum(_rms, axis=0, keepdims=False) / N + eps)
    ct.scatter(Rstd, row, rms)

    for j in range(0, num_tiles):
        offs = j * TILE_SIZE + offsets
        wj = ct.gather(w, offs, latency=1)
        wj = ct.astype(wj, ct.float32)
        xj = ct.gather(x, (row, offs), latency=1)
        xj = ct.astype(xj, ct.float32)
        yj = xj * rms * wj
        yj = ct.astype(yj, x.dtype)
        ct.scatter(out, (row, offs), yj, latency=1)


def rms_norm_backward(
    x: torch.Tensor,
    dy: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    dw = torch.empty_like(weight)
    temp_buffer = torch.empty(x.shape, device=x.device, dtype=torch.float32)

    dx = dx.detach()
    dw = dw.detach()

    TILE_SIZE_N = next_power_of_2(N)
    grid_dx = (M,)
    ct.launch(
        torch.cuda.current_stream(),
        grid_dx,
        rms_norm_backward_kernel,
        (dx, dy, x, weight, rstd, temp_buffer, TILE_SIZE_N),
    )

    dw = temp_buffer[:, :N].to(torch.float32).sum(dim=0).to(weight.dtype)
    return dx.view(*x_shape), dw


class CuTileRMSNorm(torch.autograd.Function):
    """RMSNorm with CuTile forward and backward."""
    
    @staticmethod
    def forward(ctx, x, weight, eps):
        x = x.contiguous()
        weight = weight.contiguous()
        x_arg = x.reshape(-1, x.shape[-1])
        
        y = torch.empty_like(x_arg)
        M, N = x_arg.shape
        
        rstd = torch.empty((M,), dtype=torch.float32, device="cuda")
        MAX_FUSED_SIZE = 4096 // x.element_size()
        TILE_SIZE = min(MAX_FUSED_SIZE, next_power_of_2(N))
        
        grid = (M,)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            rms_norm_kernel_gather,
            (x_arg.detach(), weight.detach(), y.detach(), rstd, N, eps, TILE_SIZE),
        )
        
        ctx.save_for_backward(x, weight, rstd)
        ctx.eps = eps
        return y.view(*x.shape)
    
    @staticmethod
    def backward(ctx, dy):
        x, weight, rstd = ctx.saved_tensors
        dx, dw = rms_norm_backward(x, dy, weight, rstd)
        return dx, dw, None


class BastileRMSNorm(nn.Module):
    """Drop-in replacement for Qwen3RMSNorm using CuTile."""
    
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size
    
    def forward(self, hidden_states):
        return CuTileRMSNorm.apply(hidden_states, self.weight, self.variance_epsilon)
    
    def extra_repr(self):
        return f"{self.hidden_size}, eps={self.variance_epsilon}"


# ============================================================================
# Register patches for Qwen3
# ============================================================================

register_patch(
    name="rms_norm_qwen3",
    description="CuTile RMSNorm for Qwen3 models",
    target_module="transformers.models.qwen3.modeling_qwen3",
    target_attr="Qwen3RMSNorm",
    replacement=BastileRMSNorm,
    has_backward=True,
    priority=10,
    models=["qwen3"],
)

# GPT-OSS models
register_patch(
    name="rms_norm_gpt_oss",
    description="CuTile RMSNorm for GPT-OSS models",
    target_module="transformers.models.gpt_oss.modeling_gpt_oss",
    target_attr="GptOssRMSNorm",
    replacement=BastileRMSNorm,
    has_backward=True,
    priority=10,
    models=["gpt_oss"],
)
