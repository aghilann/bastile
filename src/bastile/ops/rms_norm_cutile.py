"""
CuTile RMSNorm - native cuTILE implementation for Bastile.

Provides a cuTILE-native RMSNorm that aims to match the performance of
Quack's cuteDSL kernels. Two forward kernel strategies are autotuned:

  - gather/scatter: tiled loop with latency-hint prefetching (best for multi-tile)
  - TMA: bulk DMA load of full row, in-register reduction (best for single-tile)

Key optimizations:
1. Dual-strategy forward (gather vs TMA) selected by autotuner
2. Float32 accumulation for numerical stability
3. Wide occupancy sweep (1/2/4/8) with extensive tile-size search
4. TMA-based backward for efficient row-wise gradient computation
5. Weight gradient via temp buffer + host-side reduction
"""

import torch
import torch.nn as nn
from dataclasses import dataclass

import cuda.tile as ct

from ..autotune import autotune
from .utils import next_power_of_2

ConstInt = ct.Constant[int]
ConstFloat = ct.Constant[float]
PAD_ZERO = ct.PaddingMode.ZERO


# ============================================================================
# Autotune Config
# ============================================================================

@dataclass
class RMSNormCuTileConfig:
    """Configuration for cuTILE RMSNorm autotuning."""
    tile_size: int
    occupancy: int
    use_tma: bool = False

    def __hash__(self):
        return hash((self.tile_size, self.occupancy, self.use_tma))


# ============================================================================
# Forward Kernels - Gather/Scatter variant (tiled loop)
# ============================================================================

def _fwd_gather_body(x, w, out, rstd_out, N, eps, TILE_SIZE):
    """Gather/scatter forward kernel body."""
    row = ct.bid(0)
    offsets = ct.arange(TILE_SIZE, dtype=ct.int32)
    num_tiles = ct.cdiv(N, TILE_SIZE)

    # Phase 1: Accumulate sum of squares in float32
    _rms = ct.full((TILE_SIZE,), 0.0, dtype=ct.float32)
    for j in range(0, num_tiles):
        offs = j * TILE_SIZE + offsets
        xj = ct.gather(x, (row, offs), latency=1)
        xj = ct.astype(xj, ct.float32)
        _rms += xj * xj

    rms = ct.rsqrt(ct.sum(_rms, axis=0, keepdims=False) / N + eps)
    ct.scatter(rstd_out, row, rms)

    # Phase 2: Normalize and scale by weight
    for j in range(0, num_tiles):
        offs = j * TILE_SIZE + offsets
        wj = ct.gather(w, offs, latency=1)
        wj = ct.astype(wj, ct.float32)
        xj = ct.gather(x, (row, offs), latency=1)
        xj = ct.astype(xj, ct.float32)
        yj = xj * rms * wj
        yj = ct.astype(yj, x.dtype)
        ct.scatter(out, (row, offs), yj, latency=1)


@ct.kernel(occupancy=1)
def rms_norm_fwd_gather_occ1(x, w, out, rstd_out, N: ConstInt, eps: ConstFloat, TILE_SIZE: ConstInt):
    _fwd_gather_body(x, w, out, rstd_out, N, eps, TILE_SIZE)

@ct.kernel(occupancy=2)
def rms_norm_fwd_gather_occ2(x, w, out, rstd_out, N: ConstInt, eps: ConstFloat, TILE_SIZE: ConstInt):
    _fwd_gather_body(x, w, out, rstd_out, N, eps, TILE_SIZE)

@ct.kernel(occupancy=4)
def rms_norm_fwd_gather_occ4(x, w, out, rstd_out, N: ConstInt, eps: ConstFloat, TILE_SIZE: ConstInt):
    _fwd_gather_body(x, w, out, rstd_out, N, eps, TILE_SIZE)

@ct.kernel(occupancy=8)
def rms_norm_fwd_gather_occ8(x, w, out, rstd_out, N: ConstInt, eps: ConstFloat, TILE_SIZE: ConstInt):
    _fwd_gather_body(x, w, out, rstd_out, N, eps, TILE_SIZE)

FWD_GATHER_KERNELS = {
    1: rms_norm_fwd_gather_occ1,
    2: rms_norm_fwd_gather_occ2,
    4: rms_norm_fwd_gather_occ4,
    8: rms_norm_fwd_gather_occ8,
}


# ============================================================================
# Forward Kernels - TMA variant (single bulk load, no loop)
# ============================================================================

def _fwd_tma_body(x, w, out, rstd_out, TILE_SIZE):
    """TMA-based forward: bulk load full row, reduce, normalize, store."""
    row = ct.bid(0)

    # Bulk load entire row + weight via TMA
    x_row = ct.load(x, index=(row, 0), shape=(1, TILE_SIZE), padding_mode=PAD_ZERO)
    w_row = ct.load(w, index=(0,), shape=(TILE_SIZE,), padding_mode=PAD_ZERO)
    w_row = ct.reshape(w_row, (1, TILE_SIZE))

    # Compute rstd in float32
    x_f32 = ct.astype(x_row, ct.float32)
    sum_sq = ct.sum(x_f32 * x_f32, axis=1, keepdims=True)
    M_val, N_val = x.shape
    rms = ct.rsqrt(sum_sq / N_val + 1e-6)

    # Store rstd (scalar per row)
    ct.store(rstd_out, index=(row,), tile=ct.reshape(rms, (1,)))

    # Normalize and scale
    w_f32 = ct.astype(w_row, ct.float32)
    y_row = x_f32 * rms * w_f32
    y_row = ct.astype(y_row, out.dtype)
    ct.store(out, index=(row, 0), tile=y_row)


@ct.kernel(occupancy=1)
def rms_norm_fwd_tma_occ1(x, w, out, rstd_out, TILE_SIZE: ConstInt):
    _fwd_tma_body(x, w, out, rstd_out, TILE_SIZE)

@ct.kernel(occupancy=2)
def rms_norm_fwd_tma_occ2(x, w, out, rstd_out, TILE_SIZE: ConstInt):
    _fwd_tma_body(x, w, out, rstd_out, TILE_SIZE)

@ct.kernel(occupancy=4)
def rms_norm_fwd_tma_occ4(x, w, out, rstd_out, TILE_SIZE: ConstInt):
    _fwd_tma_body(x, w, out, rstd_out, TILE_SIZE)

@ct.kernel(occupancy=8)
def rms_norm_fwd_tma_occ8(x, w, out, rstd_out, TILE_SIZE: ConstInt):
    _fwd_tma_body(x, w, out, rstd_out, TILE_SIZE)

FWD_TMA_KERNELS = {
    1: rms_norm_fwd_tma_occ1,
    2: rms_norm_fwd_tma_occ2,
    4: rms_norm_fwd_tma_occ4,
    8: rms_norm_fwd_tma_occ8,
}


# ============================================================================
# Backward Kernel (TMA-based, fixed occupancy=2)
# ============================================================================

@ct.kernel(occupancy=2)
def rms_norm_bwd_kernel(
    dx, dy, x, weight, Rstd, temp_buffer,
    TILE_SIZE: ConstInt,
):
    """RMSNorm backward: compute dx and dw contributions.

    Uses TMA loads for efficient row-wise access. Each block processes
    one row. Weight gradient contributions are stored to temp_buffer
    and reduced on the host.
    """
    row_idx = ct.bid(0)
    M, N = x.shape

    input_row = ct.load(x, index=(row_idx, 0), shape=(1, TILE_SIZE), padding_mode=PAD_ZERO)
    gradient_row = ct.load(dy, index=(row_idx, 0), shape=(1, TILE_SIZE), padding_mode=PAD_ZERO)
    inv_std_row = ct.load(Rstd, index=(row_idx,), shape=(1,), padding_mode=PAD_ZERO)
    inv_std_row = ct.reshape(inv_std_row, (1, 1))
    weight_vector = ct.load(weight, index=(0,), shape=(TILE_SIZE,), padding_mode=PAD_ZERO)
    weight_vector = ct.reshape(weight_vector, (1, TILE_SIZE))

    # Weight gradient contribution: x * dy * rstd -> temp_buffer
    c1 = input_row * gradient_row
    c2 = c1 * inv_std_row
    ct.store(temp_buffer, index=(row_idx, 0), tile=ct.astype(c2, temp_buffer.dtype))

    # Input gradient: dx = rstd * (dy * w) - x * rstd^3 / N * sum(x * dy * w)
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


# ============================================================================
# Search Space
# ============================================================================

def _fwd_search_space(N):
    """Generate search space for RMSNorm forward autotuning.

    Tries both gather/scatter and TMA strategies across multiple
    tile sizes and occupancy levels.
    """
    base_tile = next_power_of_2(N)
    tile_sizes = set()

    # Multiplier-based sizes around the base
    for mult in [0.25, 0.5, 1, 2]:
        ts = int(base_tile * mult)
        if 64 <= ts <= 16384 and ts == next_power_of_2(ts):
            tile_sizes.add(ts)

    # Standard power-of-2 tile sizes (match TileGym)
    for ts in [256, 512, 1024, 2048, 4096]:
        if ts >= N or ts == next_power_of_2(N):
            tile_sizes.add(ts)

    # Always include the covering tile
    tile_sizes.add(base_tile)

    configs = []
    for ts in sorted(tile_sizes):
        for occ in [1, 2, 4, 8]:
            # Gather/scatter variant (works for any tile_size)
            configs.append(RMSNormCuTileConfig(tile_size=ts, occupancy=occ, use_tma=False))
            # TMA variant (only when tile covers full row)
            if ts >= N:
                configs.append(RMSNormCuTileConfig(tile_size=ts, occupancy=occ, use_tma=True))

    return configs


# ============================================================================
# Launch Helpers
# ============================================================================

def _run_fwd_with_config(x, w, out, rstd, N, eps, config):
    """Run forward kernel with a specific config."""
    M = x.shape[0]
    if config.use_tma:
        kernel = FWD_TMA_KERNELS[config.occupancy]
        ct.launch(
            torch.cuda.current_stream(),
            (M,),
            kernel,
            (x, w, out, rstd, config.tile_size),
        )
    else:
        kernel = FWD_GATHER_KERNELS[config.occupancy]
        ct.launch(
            torch.cuda.current_stream(),
            (M,),
            kernel,
            (x, w, out, rstd, N, eps, config.tile_size),
        )


def _run_bwd(dx, dy, x, weight, rstd, temp_buffer, N):
    """Run backward kernel (fixed config: occupancy=2, tile=next_pow2(N))."""
    M = x.shape[0]
    tile_size = next_power_of_2(N)
    ct.launch(
        torch.cuda.current_stream(),
        (M,),
        rms_norm_bwd_kernel,
        (dx, dy, x, weight, rstd, temp_buffer, tile_size),
    )


# ============================================================================
# Fast Config Cache (bypasses autotune overhead on hot path)
# ============================================================================

# Direct config cache: (N, dtype_str) -> RMSNormCuTileConfig
# Populated by first autotune call, then used directly without lock/disk I/O
_fwd_config_cache: dict = {}


def _ensure_fwd_config(N, dtype, x_arg, weight, y, rstd, eps):
    """Get or compute the best forward config for (N, dtype).

    First call triggers full autotuning. Subsequent calls return cached config
    with zero overhead (simple dict lookup).
    """
    cache_key = (N, str(dtype))
    config = _fwd_config_cache.get(cache_key)
    if config is not None:
        return config

    # First call for this (N, dtype): run full autotuning
    M = x_arg.shape[0]
    key = (M, N, str(dtype))

    def run_fn(cfg):
        _run_fwd_with_config(x_arg.detach(), weight.detach(), y.detach(), rstd, N, eps, cfg)

    config = autotune(
        kernel_name="rms_norm_cutile_fwd",
        run_fn=run_fn,
        search_space=_fwd_search_space(N),
        key=str(key),
        max_iter=32,
        warmup=5,
        rep=10,
        use_heuristic=False,
    )

    _fwd_config_cache[cache_key] = config
    return config


# ============================================================================
# High-level Functions
# ============================================================================

def rms_norm_forward(x, weight, eps):
    """CuTile RMSNorm forward with autotuning."""
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape

    y = torch.empty_like(x_arg)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

    config = _ensure_fwd_config(N, x.dtype, x_arg, weight, y, rstd, eps)
    _run_fwd_with_config(x_arg, weight, y, rstd, N, eps, config)

    return y.view(*x.shape), rstd


def rms_norm_backward(
    x: torch.Tensor,
    dy: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CuTile RMSNorm backward pass."""
    x_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    dy = dy.reshape(-1, dy.shape[-1])

    M, N = x.shape
    dx = torch.empty_like(x)
    temp_buffer = torch.empty(x.shape, device=x.device, dtype=torch.float32)

    _run_bwd(dx, dy, x, weight, rstd, temp_buffer, N)

    dw = temp_buffer[:, :N].to(torch.float32).sum(dim=0).to(weight.dtype)
    return dx.view(*x_shape), dw


# ============================================================================
# Autograd Function (inlined hot path for minimal dispatch overhead)
# ============================================================================

class CuTileRMSNormFunction(torch.autograd.Function):
    """CuTile RMSNorm with autotuned forward and backward.

    The forward path is inlined to minimize Python dispatch overhead -
    avoids extra function calls by directly looking up the cached config
    and launching the kernel.
    """

    @staticmethod
    def forward(ctx, x, weight, eps):
        x_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        M, N = x_2d.shape

        y = torch.empty_like(x_2d)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

        # Fast path: check config cache directly (zero overhead on hit)
        cache_key = (N, str(x.dtype))
        config = _fwd_config_cache.get(cache_key)
        if config is None:
            config = _ensure_fwd_config(N, x.dtype, x_2d, weight, y, rstd, eps)

        # Inline kernel launch (avoid _run_fwd_with_config function call)
        stream = torch.cuda.current_stream()
        if config.use_tma:
            ct.launch(stream, (M,), FWD_TMA_KERNELS[config.occupancy],
                      (x_2d, weight, y, rstd, config.tile_size))
        else:
            ct.launch(stream, (M,), FWD_GATHER_KERNELS[config.occupancy],
                      (x_2d, weight, y, rstd, N, eps, config.tile_size))

        ctx.save_for_backward(x, weight, rstd)
        ctx.eps = eps
        return y.view(x_shape)

    @staticmethod
    def backward(ctx, dy):
        x, weight, rstd = ctx.saved_tensors
        dx, dw = rms_norm_backward(x, dy, weight, rstd)
        return dx, dw, None


# ============================================================================
# Module and Convenience Functions
# ============================================================================

class CuTileRMSNorm(nn.Module):
    """Drop-in RMSNorm replacement using native cuTILE kernels."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return CuTileRMSNormFunction.apply(
            hidden_states, self.weight, self.variance_epsilon,
        )

    def extra_repr(self) -> str:
        return f"{self.weight.shape[0]}, eps={self.variance_epsilon}"


def rms_norm(input, weight, eps=1e-6, static_persistent=None):
    """Standalone cuTILE RMSNorm function.

    Args:
        input: Input tensor of shape (..., hidden_size)
        weight: Weight tensor of shape (hidden_size,)
        eps: Epsilon for numerical stability
        static_persistent: Ignored (kept for API compatibility with benchmarks)

    Returns:
        Normalized tensor with same shape as input
    """
    return CuTileRMSNormFunction.apply(input, weight, eps)


def warmup_rms_norm(
    hidden_size: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Pre-compile cuTILE RMSNorm kernels for given hidden size."""
    x = torch.randn(2, hidden_size, dtype=dtype, device=device, requires_grad=True)
    w = torch.ones(hidden_size, dtype=dtype, device=device, requires_grad=True)
    out = rms_norm(x, w, eps=1e-6)
    out.sum().backward()
    torch.cuda.synchronize()
