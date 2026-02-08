"""
CuTile RMSNorm - optimized for NVIDIA B200 with full backward support.

Key optimizations:
1. gather/scatter memory access (faster than ct.load/ct.store for row-wise ops)
2. Single-tile processing (no loops when TILE_SIZE >= N)
3. flush_to_zero=True for fast computation
4. Single pass over data (reads x once for both RMS and normalization)
5. Full backward pass with gradient computation for dx and dw
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from ..registry import register_patch
from .utils import next_power_of_2

import cuda.tile as ct

ConstInt = ct.Constant[int]
ConstFloat = ct.Constant[float]


# ============================================================================
# Forward kernel
# ============================================================================

@ct.kernel
def rms_norm_forward_kernel(
    x, w, out, rstd,
    N: ConstInt,
    eps: ConstFloat,
    TILE_SIZE: ConstInt,
):
    """RMSNorm forward using gather/scatter with fast math.

    Each block processes one row. Reads x and weight once,
    computes inverse RMS, normalizes, and writes output.
    """
    bid = ct.bid(0)
    offsets = ct.arange(TILE_SIZE, dtype=ct.int32)

    # Single gather for row and weight
    x_tile = ct.gather(x, (bid, offsets), check_bounds=True, padding_value=0.0)
    w_tile = ct.gather(w, offsets, check_bounds=True, padding_value=1.0)

    # Cast to float32 for numerical stability
    x_f32 = ct.astype(x_tile, ct.float32)
    w_f32 = ct.astype(w_tile, ct.float32)

    # Compute inverse RMS with fast math
    x_sq = ct.mul(x_f32, x_f32, flush_to_zero=True)
    inv_rms = ct.rsqrt(ct.sum(x_sq, axis=0, keepdims=False) / N + eps)

    # Store rstd for backward pass
    ct.scatter(rstd, bid, inv_rms, check_bounds=True)

    # Normalize: y = x * rstd * weight
    y = ct.mul(ct.mul(x_f32, inv_rms, flush_to_zero=True), w_f32, flush_to_zero=True)
    y = ct.astype(y, x.dtype)

    ct.scatter(out, (bid, offsets), y, check_bounds=True)


# ============================================================================
# Backward kernel
# ============================================================================

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


# ============================================================================
# Launch functions
# ============================================================================

def _rms_norm_forward(
    x_flat: torch.Tensor,
    weight: torch.Tensor,
    y: torch.Tensor,
    rstd: torch.Tensor,
    N: int,
    eps: float,
):
    """Launch optimized gather-based RMSNorm forward kernel."""
    M = x_flat.shape[0]
    TILE_SIZE = next_power_of_2(N)
    grid = (M,)

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        rms_norm_forward_kernel,
        (x_flat, weight, y, rstd, N, eps, TILE_SIZE),
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


# ============================================================================
# Autograd wrapper
# ============================================================================

class RMSNormFunction(torch.autograd.Function):
    """CuTile RMSNorm with full backward support."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
        use_static_persistent: Optional[bool] = None,
    ) -> torch.Tensor:
        x = x.contiguous()
        weight = weight.contiguous()

        x_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        M, N = x_flat.shape

        y = torch.empty_like(x_flat)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)

        _rms_norm_forward(x_flat, weight, y, rstd, N, eps)

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
    """Apply CuTile RMSNorm."""
    return RMSNormFunction.apply(x, weight, eps, use_static_persistent)


# ============================================================================
# Module
# ============================================================================

class CuTileRMSNorm(nn.Module):
    """Drop-in replacement for Qwen3RMSNorm using CuTile kernels."""

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
    description="CuTile RMSNorm for Qwen3",
    target_module="transformers.models.qwen3.modeling_qwen3",
    target_attr="Qwen3RMSNorm",
    replacement=CuTileRMSNorm,
    has_backward=True,
    priority=10,
    models=["qwen3"],
)

register_patch(
    name="rms_norm_gpt_oss",
    description="CuTile RMSNorm for GPT-OSS",
    target_module="transformers.models.gpt_oss.modeling_gpt_oss",
    target_attr="GptOssRMSNorm",
    replacement=CuTileRMSNorm,
    has_backward=True,
    priority=10,
    models=["gpt_oss"],
)
