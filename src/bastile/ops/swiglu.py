"""
CuTile SwiGLU (SiLU * x) - optimized for NVIDIA B200 with full backward support.

Key optimizations:
1. gather/scatter memory access (faster than ct.load/ct.store for row-wise ops)
2. flush_to_zero=True for fast exponential computation
3. Approximate rounding for division (fast reciprocal)
4. SM-aware tile sizing (from TileGym approach)
5. Full backward pass with recomputation (no extra memory)
"""

import torch
import torch.nn as nn
from typing import Tuple

from ..registry import register_patch
from .utils import next_power_of_2, ceildiv

import cuda.tile as ct
from cuda.tile._numeric_semantics import RoundingMode as RMd

ConstInt = ct.Constant[int]


# ============================================================================
# Forward kernel
# ============================================================================

@ct.kernel
def swiglu_forward_kernel(
    gate, up, output,
    TILE_SIZE: ConstInt,
):
    """SwiGLU forward: silu(gate) * up using gather/scatter.

    Each block processes one row. Uses flush_to_zero and approximate
    reciprocal for fast sigmoid computation on Blackwell.
    """
    bid = ct.bid(0)
    offsets = ct.arange(TILE_SIZE, dtype=ct.int32)

    gate_tile = ct.gather(gate, (bid, offsets), check_bounds=True, padding_value=0.0)
    up_tile = ct.gather(up, (bid, offsets), check_bounds=True, padding_value=0.0)

    # Compute sigmoid in float32 for numerical stability
    gate_f32 = ct.astype(gate_tile, ct.float32)
    denom = ct.add(1.0, ct.exp(-gate_f32), flush_to_zero=True)
    sig = ct.truediv(1.0, denom, flush_to_zero=True, rounding_mode=RMd.APPROX)

    # SiLU(gate) * up
    silu_gate = ct.mul(gate_f32, sig, flush_to_zero=True)
    result = ct.astype(silu_gate, gate.dtype) * up_tile

    ct.scatter(output, (bid, offsets), result, check_bounds=True)


# ============================================================================
# Backward kernel
# ============================================================================

@ct.kernel
def swiglu_backward_kernel(
    grad_output, gate, up, grad_gate, grad_up,
    TILE_SIZE: ConstInt,
):
    """SwiGLU backward with gather/scatter and recomputation.

    d(silu(g)*u)/dg = u * sigmoid(g) * (1 + g * (1 - sigmoid(g)))
    d(silu(g)*u)/du = silu(g)
    """
    bid = ct.bid(0)
    offsets = ct.arange(TILE_SIZE, dtype=ct.int32)

    dy_tile = ct.gather(grad_output, (bid, offsets), check_bounds=True, padding_value=0.0)
    gate_tile = ct.gather(gate, (bid, offsets), check_bounds=True, padding_value=0.0)
    up_tile = ct.gather(up, (bid, offsets), check_bounds=True, padding_value=0.0)

    # Compute in float32
    dy_f32 = ct.astype(dy_tile, ct.float32)
    gate_f32 = ct.astype(gate_tile, ct.float32)
    up_f32 = ct.astype(up_tile, ct.float32)

    # Recompute sigmoid exactly (no fast-math for backward numerical accuracy)
    sig_g = ct.truediv(1.0, ct.add(1.0, ct.exp(-gate_f32)))
    silu_g = gate_f32 * sig_g

    # d(silu)/d(gate) = sigmoid(g) * (1 + g * (1 - sigmoid(g)))
    dsilu_dgate = sig_g * (1.0 + gate_f32 * (1.0 - sig_g))

    dg = ct.astype(dy_f32 * dsilu_dgate * up_f32, grad_output.dtype)
    du = ct.astype(dy_f32 * silu_g, grad_output.dtype)

    ct.scatter(grad_gate, (bid, offsets), dg, check_bounds=True)
    ct.scatter(grad_up, (bid, offsets), du, check_bounds=True)


# ============================================================================
# Launch functions
# ============================================================================

def swiglu_forward_cutile(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """CuTile-accelerated SwiGLU forward.

    Uses gather/scatter pattern from TileGym's silu_and_mul with fast math
    (flush_to_zero, approximate reciprocal). One block per row with
    SM-aware tile sizing.
    """
    ori_shape = gate.shape
    n_cols = ori_shape[-1]
    gate = gate.view(-1, n_cols)
    up = up.view(-1, n_cols)
    output = torch.empty_like(gate)
    n_rows = gate.shape[0]

    TILE_SIZE = next_power_of_2(n_cols)
    grid = (n_rows,)

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        swiglu_forward_kernel,
        (gate.data, up.data, output.data, TILE_SIZE),
    )

    return output.view(*ori_shape)


def swiglu_backward_cutile(
    grad_output: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CuTile-accelerated SwiGLU backward."""
    ori_shape = grad_output.shape
    n_cols = ori_shape[-1]
    grad_output = grad_output.reshape(-1, n_cols).contiguous()
    gate = gate.reshape(-1, n_cols).contiguous()
    up = up.reshape(-1, n_cols).contiguous()

    grad_gate = torch.empty_like(gate)
    grad_up = torch.empty_like(up)
    n_rows = grad_output.shape[0]

    TILE_SIZE = next_power_of_2(n_cols)
    grid = (n_rows,)

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        swiglu_backward_kernel,
        (grad_output, gate, up, grad_gate, grad_up, TILE_SIZE),
    )

    return grad_gate.view(*ori_shape), grad_up.view(*ori_shape)


# ============================================================================
# Autograd wrapper
# ============================================================================

class SwiGLUFunction(torch.autograd.Function):
    """CuTile SwiGLU with full backward support."""

    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        output = swiglu_forward_cutile(gate, up)
        ctx.save_for_backward(gate, up)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        gate, up = ctx.saved_tensors
        grad_gate, grad_up = swiglu_backward_cutile(grad_output, gate, up)
        return grad_gate, grad_up


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Apply CuTile SwiGLU: silu(gate) * up"""
    return SwiGLUFunction.apply(gate, up)


# ============================================================================
# MLP Module
# ============================================================================

class CuTileSwiGLUMLP(nn.Module):
    """Drop-in replacement for Qwen3MLP using CuTile SwiGLU."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        if hasattr(config, 'hidden_act') and config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(SwiGLUFunction.apply(self.gate_proj(x), self.up_proj(x)))


# Register patch
register_patch(
    name="swiglu_qwen3",
    description="CuTile SwiGLU MLP with fast math for Qwen3",
    target_module="transformers.models.qwen3.modeling_qwen3",
    target_attr="Qwen3MLP",
    replacement=CuTileSwiGLUMLP,
    has_backward=True,
    priority=10,
    models=["qwen3"],
)
