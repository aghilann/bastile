"""
CuTile RMSNorm - native cuTILE implementation for Bastile.

Optimized for NVIDIA Blackwell (B200) using patterns from TileGym:

Forward kernels:
  1. Static-persistent 2D TMA — persistent grid-stride loop over multi-row
     tiles, weight pre-loaded once, allow_tma=False stores (+30%), latency
     hints on loads (+2%) and stores (+3%). Best for large M (training).
  2. Gather/scatter 1D — tiled column loop with latency-hint prefetching.
     Best for small M or non-power-of-2 N.
  3. TMA 1-row — bulk DMA load of full row, in-register reduction.
     Best for small M with power-of-2 N.

Backward:
  TMA-based row-wise dx + temp_buffer for dw reduction.

Key Blackwell-specific optimizations:
  - allow_tma=False on ct.store (bypasses TMA store path → +30% on B200)
  - latency hints on ct.load (latency=10) and ct.store (latency=3)
  - TILE_SIZE_M sweep (2/4/8/16) for 2D persistent kernels
  - Occupancy sweep (1/2/4/8) compiled as separate kernel objects
  - Persistent scheduling: grid = NUM_SMS, blocks stride over rows
"""

import torch
import torch.nn as nn
from dataclasses import dataclass

import cuda.tile as ct

from ..autotune import autotune
from ..registry import register_patch
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
    use_persistent: bool = False
    tile_size_m: int = 1  # rows per tile (only for persistent)

    def __hash__(self):
        return hash((self.tile_size, self.occupancy, self.use_tma,
                     self.use_persistent, self.tile_size_m))


# ============================================================================
# Forward Kernels - Static Persistent 2D TMA (Blackwell-optimized)
# ============================================================================

def _fwd_persistent_body(X, W, Y, Rstd, TILE_SIZE_M, TILE_SIZE_N, eps):
    """Static persistent 2D RMSNorm: each block processes multiple row-tiles.

    Adapted from TileGym's rms_norm_kernel_static_persistent with
    Blackwell-specific store hints. Also saves rstd for backward pass.
    """
    bid = ct.bid(0)
    M = X.shape[0]
    N = X.shape[1]

    upper_bound = (M + TILE_SIZE_M - 1) // TILE_SIZE_M

    # Pre-load weight once (shared across all tiles this block processes)
    w = ct.load(W, index=(0,), shape=(TILE_SIZE_N,), padding_mode=PAD_ZERO)
    w = ct.astype(w, ct.float32)

    # Persistent grid-stride loop
    num_tile_blocks = ct.num_blocks(0)
    for current_bid in range(bid, upper_bound, num_tile_blocks):
        # Load 2D input tile [TILE_SIZE_M, TILE_SIZE_N]
        x = ct.load(
            X,
            index=(current_bid, 0),
            shape=(TILE_SIZE_M, TILE_SIZE_N),
            latency=10,  # prefetch hint (+2% on B200)
            padding_mode=PAD_ZERO,
        )
        x = ct.astype(x, ct.float32)

        # RMS computation: rsqrt(mean(x^2) + eps)
        x_squared = ct.mul(x, x)
        x2_sum = ct.sum(x_squared, axis=1, keepdims=True)
        N_f32 = ct.full((TILE_SIZE_M, 1), N * 1.0, dtype=ct.float32)
        variance = ct.truediv(x2_sum, N_f32)
        eps_t = ct.full((TILE_SIZE_M, 1), eps, dtype=ct.float32)
        rsqrt_var = ct.rsqrt(ct.add(variance, eps_t))

        # Save rstd for backward pass (1 float32 per row, negligible BW)
        rstd_flat = ct.reshape(rsqrt_var, (TILE_SIZE_M,))
        ct.store(Rstd, index=(current_bid,), tile=rstd_flat, allow_tma=False)

        # Normalize and scale by weight
        x_normalized = ct.mul(x, rsqrt_var)
        w_broadcasted = ct.reshape(w, (1, TILE_SIZE_N))
        y = ct.mul(x_normalized, w_broadcasted)

        # Store result
        y = ct.astype(y, X.dtype)
        ct.store(
            Y,
            index=(current_bid, 0),
            tile=y,
            allow_tma=False,  # bypass TMA store (+30% on B200)
            latency=3,  # store hint (+3% on B200)
        )


@ct.kernel(occupancy=1)
def rms_norm_fwd_persistent_occ1(X, W, Y, Rstd, TILE_SIZE_M: ConstInt, TILE_SIZE_N: ConstInt, eps: ConstFloat):
    _fwd_persistent_body(X, W, Y, Rstd, TILE_SIZE_M, TILE_SIZE_N, eps)

@ct.kernel(occupancy=2)
def rms_norm_fwd_persistent_occ2(X, W, Y, Rstd, TILE_SIZE_M: ConstInt, TILE_SIZE_N: ConstInt, eps: ConstFloat):
    _fwd_persistent_body(X, W, Y, Rstd, TILE_SIZE_M, TILE_SIZE_N, eps)

@ct.kernel(occupancy=4)
def rms_norm_fwd_persistent_occ4(X, W, Y, Rstd, TILE_SIZE_M: ConstInt, TILE_SIZE_N: ConstInt, eps: ConstFloat):
    _fwd_persistent_body(X, W, Y, Rstd, TILE_SIZE_M, TILE_SIZE_N, eps)

@ct.kernel(occupancy=8)
def rms_norm_fwd_persistent_occ8(X, W, Y, Rstd, TILE_SIZE_M: ConstInt, TILE_SIZE_N: ConstInt, eps: ConstFloat):
    _fwd_persistent_body(X, W, Y, Rstd, TILE_SIZE_M, TILE_SIZE_N, eps)

FWD_PERSISTENT_KERNELS = {
    1: rms_norm_fwd_persistent_occ1,
    2: rms_norm_fwd_persistent_occ2,
    4: rms_norm_fwd_persistent_occ4,
    8: rms_norm_fwd_persistent_occ8,
}


# ============================================================================
# Forward Kernels - Gather/Scatter variant (tiled loop, 1 row per block)
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
# Forward Kernels - TMA variant (single bulk load per row)
# ============================================================================

def _fwd_tma_body(x, w, out, rstd_out, TILE_SIZE):
    """TMA-based forward: bulk load full row, reduce, normalize, store."""
    row = ct.bid(0)

    x_row = ct.load(x, index=(row, 0), shape=(1, TILE_SIZE),
                    padding_mode=PAD_ZERO, latency=10)
    w_row = ct.load(w, index=(0,), shape=(TILE_SIZE,), padding_mode=PAD_ZERO)
    w_row = ct.reshape(w_row, (1, TILE_SIZE))

    x_f32 = ct.astype(x_row, ct.float32)
    sum_sq = ct.sum(x_f32 * x_f32, axis=1, keepdims=True)
    M_val, N_val = x.shape
    rms = ct.rsqrt(sum_sq / N_val + 1e-6)

    ct.store(rstd_out, index=(row,), tile=ct.reshape(rms, (1,)))

    w_f32 = ct.astype(w_row, ct.float32)
    y_row = x_f32 * rms * w_f32
    y_row = ct.astype(y_row, out.dtype)
    ct.store(out, index=(row, 0), tile=y_row,
             allow_tma=False, latency=3)


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

    input_row = ct.load(x, index=(row_idx, 0), shape=(1, TILE_SIZE),
                        padding_mode=PAD_ZERO, latency=10)
    gradient_row = ct.load(dy, index=(row_idx, 0), shape=(1, TILE_SIZE),
                           padding_mode=PAD_ZERO, latency=10)
    inv_std_row = ct.load(Rstd, index=(row_idx,), shape=(1,), padding_mode=PAD_ZERO)
    inv_std_row = ct.reshape(inv_std_row, (1, 1))
    weight_vector = ct.load(weight, index=(0,), shape=(TILE_SIZE,), padding_mode=PAD_ZERO)
    weight_vector = ct.reshape(weight_vector, (1, TILE_SIZE))

    # Weight gradient contribution: x * dy * rstd -> temp_buffer
    c1 = input_row * gradient_row
    c2 = c1 * inv_std_row
    ct.store(temp_buffer, index=(row_idx, 0), tile=ct.astype(c2, temp_buffer.dtype),
             allow_tma=False, latency=3)

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
    ct.store(dx, index=(row_idx, 0), tile=input_gradient_row,
             allow_tma=False, latency=3)


# ============================================================================
# Search Space
# ============================================================================

def _fwd_search_space(M, N):
    """Generate search space for RMSNorm forward autotuning.

    Includes:
      1. Static persistent 2D TMA kernels (best for large M)
      2. Gather/scatter kernels (flexible tile sizes)
      3. TMA single-row kernels (best for small M, pow2 N)
    """
    configs = []
    base_tile = next_power_of_2(N)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # --- Strategy 1: Static Persistent 2D (the big win for large M) ---
    TILE_SIZE_N = base_tile
    for tile_m in [2, 4, 8, 16]:
        # Heuristic: skip very large tile_m if N is huge (register pressure)
        if tile_m * TILE_SIZE_N > 16384 * 16:
            continue
        # Skip small tile_m for moderate N (e.g. 1024)
        if TILE_SIZE_N <= 1024 and tile_m < 8:
            continue
        for occ in [1, 2, 4]:
            configs.append(RMSNormCuTileConfig(
                tile_size=TILE_SIZE_N, occupancy=occ,
                use_persistent=True, tile_size_m=tile_m,
            ))

    # --- Strategy 2: Gather/scatter (flexible) ---
    tile_sizes_g = set()
    for mult in [0.5, 1, 2]:
        ts = int(base_tile * mult)
        if 256 <= ts <= 16384 and ts == next_power_of_2(ts):
            tile_sizes_g.add(ts)
    tile_sizes_g.add(base_tile)

    for ts in sorted(tile_sizes_g):
        for occ in [1, 2, 4]:
            configs.append(RMSNormCuTileConfig(
                tile_size=ts, occupancy=occ, use_tma=False,
            ))

    # --- Strategy 3: TMA single-row (best for small M, pow2 N) ---
    if base_tile >= N:
        for occ in [1, 2, 4]:
            configs.append(RMSNormCuTileConfig(
                tile_size=base_tile, occupancy=occ, use_tma=True,
            ))

    return configs


# ============================================================================
# Launch Helpers
# ============================================================================

def _run_fwd_with_config(x, w, out, rstd, N, eps, config):
    """Run forward kernel with a specific config."""
    M = x.shape[0]
    stream = torch.cuda.current_stream()

    if config.use_persistent:
        # Static persistent: grid = min(NUM_SMS, ceil(M / tile_m))
        global _num_sms_cache
        if _num_sms_cache == 0:
            _num_sms_cache = torch.cuda.get_device_properties("cuda").multi_processor_count
        num_row_tiles = (M + config.tile_size_m - 1) // config.tile_size_m
        grid_size = min(_num_sms_cache, num_row_tiles)
        kernel = FWD_PERSISTENT_KERNELS[config.occupancy]
        ct.launch(
            stream, (grid_size,), kernel,
            (x, w, out, rstd, config.tile_size_m, config.tile_size, eps),
        )
    elif config.use_tma:
        kernel = FWD_TMA_KERNELS[config.occupancy]
        ct.launch(stream, (M,), kernel,
                  (x, w, out, rstd, config.tile_size))
    else:
        kernel = FWD_GATHER_KERNELS[config.occupancy]
        ct.launch(stream, (M,), kernel,
                  (x, w, out, rstd, N, eps, config.tile_size))


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

_fwd_config_cache: dict = {}
_num_sms_cache: int = 0  # Cached once on first use
_stream_cache = None      # Cached CUDA stream


def _heuristic_config(M, N):
    """Pick a good default config without timing (zero compile overhead).

    Tuned for Blackwell (B200, 160 SMs). Uses autotuning data:
      - Small M (≤ 4*SMS): TMA occ=4 (pow2 N) or gather occ=1
      - Medium M (≤ 16*SMS): persistent tile_m=8 occ=1
      - Large M (> 16*SMS): persistent tile_m=4 occ=1
      - HIGH PADDING (>20% waste): always gather — avoids loading zeros
        via TMA, loops over real columns only (like Quack does).
    """
    global _num_sms_cache
    if _num_sms_cache == 0:
        _num_sms_cache = torch.cuda.get_device_properties("cuda").multi_processor_count
    NUM_SMS = _num_sms_cache
    TILE_N = next_power_of_2(N)

    # Padding ratio: how much bandwidth is wasted on zeros
    # e.g. N=5120 → TILE_N=8192 → padding_ratio = 0.375 (37.5% waste)
    padding_ratio = (TILE_N - N) / TILE_N
    high_padding = padding_ratio > 0.20  # >20% waste threshold

    if high_padding:
        # Non-power-of-2 N with significant padding: use gather kernel
        # which loops over actual columns in small tiles, zero waste.
        # Pick tile_size as largest power-of-2 ≤ N for minimal loop iters.
        gather_tile = next_power_of_2(N) // 2  # e.g. N=5120 → tile=4096
        gather_tile = max(256, min(gather_tile, 4096))  # clamp to [256, 4096]
        return RMSNormCuTileConfig(
            tile_size=gather_tile, occupancy=2,
        )

    if M <= NUM_SMS * 4:
        # Small/medium M (up to ~4 waves): persistent scheduling overhead
        # not worthwhile; use TMA or gather with one block per row.
        if TILE_N == N:
            return RMSNormCuTileConfig(
                tile_size=TILE_N, occupancy=4, use_tma=True,
            )
        else:
            return RMSNormCuTileConfig(
                tile_size=TILE_N, occupancy=1,
            )
    elif M <= NUM_SMS * 16:
        # Medium M (short prefill / small-batch training)
        # Cap tile elements to ~32K to avoid register spills on large N.
        # tile_m=8 for N≤4096, tile_m=4 for N=8192, tile_m=2 for N=16384+
        tile_m = max(2, min(8, 32768 // TILE_N))
        if TILE_N <= 1024:
            tile_m = 16
        return RMSNormCuTileConfig(
            tile_size=TILE_N, occupancy=1,
            use_persistent=True, tile_size_m=tile_m,
        )
    else:
        # Large M (long prefill / training): more waves, smaller tile_m
        if TILE_N <= 1024:
            tile_m = 16
        elif TILE_N >= 16384:
            tile_m = 2
        else:
            tile_m = 4
        return RMSNormCuTileConfig(
            tile_size=TILE_N, occupancy=1,
            use_persistent=True, tile_size_m=tile_m,
        )


def _ensure_fwd_config(M, N, dtype, x_arg, weight, y, rstd, eps):
    """Get or compute the best forward config for (M, N, dtype).

    Uses a fast heuristic by default. Set BASTILE_AUTOTUNE=1 env var
    to enable full autotuning on first call.
    """
    import os

    cache_key = (M, N, dtype)
    config = _fwd_config_cache.get(cache_key)
    if config is not None:
        return config

    full_tune = os.environ.get("BASTILE_AUTOTUNE", "0") == "1"

    if full_tune:
        key = (M, N, dtype)

        def run_fn(cfg):
            _run_fwd_with_config(x_arg.detach(), weight.detach(), y.detach(), rstd, N, eps, cfg)

        config = autotune(
            kernel_name="rms_norm_cutile_fwd",
            run_fn=run_fn,
            search_space=_fwd_search_space(M, N),
            key=str(key),
            max_iter=48,
            warmup=5,
            rep=10,
            use_heuristic=False,
        )
    else:
        config = _heuristic_config(M, N)

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

    config = _ensure_fwd_config(M, N, x.dtype, x_arg, weight, y, rstd, eps)
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

    The forward path is maximally inlined to minimize Python dispatch
    overhead. All device properties are cached; the hot path is a single
    dict lookup + ct.launch.
    """

    @staticmethod
    def forward(ctx, x, weight, eps):
        global _num_sms_cache, _stream_cache

        x_shape = x.shape
        N = x_shape[-1]
        x_2d = x.reshape(-1, N)
        M = x_2d.shape[0]

        y = torch.empty_like(x_2d)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

        # Fast path: dict lookup with dtype (hashable, avoids str())
        config = _fwd_config_cache.get((M, N, x.dtype))
        if config is None:
            config = _ensure_fwd_config(M, N, x.dtype, x_2d, weight, y, rstd, eps)

        # Cached stream (valid for default stream usage)
        if _stream_cache is None:
            _stream_cache = torch.cuda.current_stream()

        # Inline kernel launch — all branches pre-resolved
        if config.use_persistent:
            if _num_sms_cache == 0:
                _num_sms_cache = torch.cuda.get_device_properties("cuda").multi_processor_count
            tile_m = config.tile_size_m
            grid_size = (M + tile_m - 1) // tile_m
            if grid_size > _num_sms_cache:
                grid_size = _num_sms_cache
            ct.launch(_stream_cache, (grid_size,),
                      FWD_PERSISTENT_KERNELS[config.occupancy],
                      (x_2d, weight, y, rstd, tile_m,
                       config.tile_size, eps))
        elif config.use_tma:
            ct.launch(_stream_cache, (M,), FWD_TMA_KERNELS[config.occupancy],
                      (x_2d, weight, y, rstd, config.tile_size))
        else:
            ct.launch(_stream_cache, (M,), FWD_GATHER_KERNELS[config.occupancy],
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


register_patch(
    name="rms_norm_qwen3",
    description="CuTile RMSNorm for Qwen3 (autotuned persistent/gather/TMA with full backward)",
    target_module="transformers.models.qwen3.modeling_qwen3",
    target_attr="Qwen3RMSNorm",
    replacement=CuTileRMSNorm,
    has_backward=True,
    priority=10,
    models=["qwen3"],
)
