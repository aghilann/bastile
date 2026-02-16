"""
CuTile MoE Expert Gate for GPT-OSS - optimized for NVIDIA B200.

Replaces GptOssExperts._apply_gate which uses a custom gating function:
  gate, up = gate_up[..., ::2], gate_up[..., 1::2]
  gate = clamp(gate, max=7.0)
  up = clamp(up, -7.0, 7.0)
  glu = gate * sigmoid(gate * 1.702)
  output = (up + 1) * glu

Key optimizations:
1. Forward: stride-2 gather from interleaved input + saves contiguous gate/up
   for backward (fused into same kernel, no extra Python copies)
2. Backward: all contiguous memory access (reads saved gate/up, writes grad_gate/up)
3. flush_to_zero + approximate reciprocal for fast sigmoid
4. Re-interleave gradients via fast PyTorch stride-copy
"""

import torch
from typing import Tuple

from .utils import next_power_of_2

import cuda.tile as ct
from cuda.tile._numeric_semantics import RoundingMode as RMd

ConstInt = ct.Constant[int]

# Constants baked into kernels
ALPHA = 1.702
LIMIT = 7.0


# ============================================================================
# Forward kernel: stride-2 gather + save contiguous gate/up for backward
# ============================================================================

@ct.kernel
def moe_gate_forward_kernel(
    gate_up, output, saved_gate, saved_up,
    TILE_SIZE: ConstInt,
):
    """MoE gate forward: stride-2 gather + contiguous save.

    gate_up:     (rows, 2*intermediate_size) interleaved input
    output:      (rows, intermediate_size)   gating result
    saved_gate:  (rows, intermediate_size)   de-interleaved gate for backward
    saved_up:    (rows, intermediate_size)   de-interleaved up for backward
    """
    bid = ct.bid(0)
    offsets = ct.arange(TILE_SIZE, dtype=ct.int32)

    # Stride-2 gather from interleaved gate_up
    gate_tile = ct.gather(gate_up, (bid, offsets * 2), check_bounds=True, padding_value=0.0)
    up_tile = ct.gather(gate_up, (bid, offsets * 2 + 1), check_bounds=True, padding_value=0.0)

    # Save raw gate/up contiguously for backward
    ct.scatter(saved_gate, (bid, offsets), gate_tile, check_bounds=True)
    ct.scatter(saved_up, (bid, offsets), up_tile, check_bounds=True)

    # Compute in float32
    gate_f32 = ct.astype(gate_tile, ct.float32)
    up_f32 = ct.astype(up_tile, ct.float32)

    # Clamp: gate max=7.0, up [-7.0, 7.0]
    gate_c = ct.minimum(gate_f32, LIMIT)
    up_c = ct.maximum(ct.minimum(up_f32, LIMIT), -LIMIT)

    # sigmoid(gate_c * alpha) using fast math
    neg_scaled = ct.mul(-ALPHA, gate_c, flush_to_zero=True)
    denom = ct.add(1.0, ct.exp(neg_scaled), flush_to_zero=True)
    sig = ct.truediv(1.0, denom, flush_to_zero=True, rounding_mode=RMd.APPROX)

    # glu = gate_c * sig, output = (up_c + 1) * glu
    glu = ct.mul(gate_c, sig, flush_to_zero=True)
    result = ct.mul(ct.add(up_c, 1.0), glu, flush_to_zero=True)

    ct.scatter(output, (bid, offsets), ct.astype(result, gate_up.dtype), check_bounds=True)


# ============================================================================
# Backward kernel: all contiguous memory access
# ============================================================================

@ct.kernel
def moe_gate_backward_kernel(
    grad_output, gate, up, grad_gate, grad_up,
    TILE_SIZE: ConstInt,
):
    """MoE gate backward with fully contiguous memory access.

    All inputs/outputs are (rows, intermediate_size).
    gate/up are the pre-saved contiguous tensors from forward.
    """
    bid = ct.bid(0)
    offsets = ct.arange(TILE_SIZE, dtype=ct.int32)

    # All contiguous reads
    dy_tile = ct.gather(grad_output, (bid, offsets), check_bounds=True, padding_value=0.0)
    gate_tile = ct.gather(gate, (bid, offsets), check_bounds=True, padding_value=0.0)
    up_tile = ct.gather(up, (bid, offsets), check_bounds=True, padding_value=0.0)

    # Compute in float32
    dy_f32 = ct.astype(dy_tile, ct.float32)
    gate_f32 = ct.astype(gate_tile, ct.float32)
    up_f32 = ct.astype(up_tile, ct.float32)

    # Recompute clamped values
    gate_c = ct.minimum(gate_f32, LIMIT)
    up_c = ct.maximum(ct.minimum(up_f32, LIMIT), -LIMIT)

    # Recompute sigmoid with fast math
    neg_scaled = ct.mul(-ALPHA, gate_c, flush_to_zero=True)
    denom = ct.add(1.0, ct.exp(neg_scaled), flush_to_zero=True)
    sig = ct.truediv(1.0, denom, flush_to_zero=True, rounding_mode=RMd.APPROX)

    # glu = gate_c * sig
    glu = ct.mul(gate_c, sig, flush_to_zero=True)

    # d_output/d_gate_c = (up_c + 1) * sig * (1 + alpha * gate_c * (1 - sig))
    dsig_term = ct.mul(ALPHA, ct.mul(gate_c, ct.add(1.0, -sig), flush_to_zero=True), flush_to_zero=True)
    d_gate_c = ct.mul(dy_f32, ct.mul(ct.add(up_c, 1.0), ct.mul(sig, ct.add(1.0, dsig_term), flush_to_zero=True), flush_to_zero=True), flush_to_zero=True)

    # d_output/d_up_c = glu
    d_up_c = ct.mul(dy_f32, glu, flush_to_zero=True)

    # Clamp masks
    gate_mask = ct.astype(ct.less_equal(gate_f32, LIMIT), ct.float32)
    up_ge_neg = ct.astype(ct.greater_equal(up_f32, -LIMIT), ct.float32)
    up_le_pos = ct.astype(ct.less_equal(up_f32, LIMIT), ct.float32)
    up_mask = up_ge_neg * up_le_pos

    # All contiguous writes
    ct.scatter(grad_gate, (bid, offsets), ct.astype(d_gate_c * gate_mask, grad_output.dtype), check_bounds=True)
    ct.scatter(grad_up, (bid, offsets), ct.astype(d_up_c * up_mask, grad_output.dtype), check_bounds=True)


# ============================================================================
# Launch functions
# ============================================================================

def moe_gate_forward_cutile(gate_up: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CuTile forward: returns (output, saved_gate, saved_up)."""
    ori_shape = gate_up.shape
    n_cols_interleaved = ori_shape[-1]
    n_cols = n_cols_interleaved // 2

    gate_up_2d = gate_up.reshape(-1, n_cols_interleaved)
    n_rows = gate_up_2d.shape[0]

    output = torch.empty(n_rows, n_cols, dtype=gate_up.dtype, device=gate_up.device)
    saved_gate = torch.empty(n_rows, n_cols, dtype=gate_up.dtype, device=gate_up.device)
    saved_up = torch.empty(n_rows, n_cols, dtype=gate_up.dtype, device=gate_up.device)

    TILE_SIZE = next_power_of_2(n_cols)
    grid = (n_rows,)

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        moe_gate_forward_kernel,
        (gate_up_2d.data, output.data, saved_gate.data, saved_up.data, TILE_SIZE),
    )

    out_shape = ori_shape[:-1] + (n_cols,)
    return output.view(*out_shape), saved_gate.view(*out_shape), saved_up.view(*out_shape)


def moe_gate_backward_cutile(
    grad_output: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CuTile backward: all contiguous access."""
    ori_shape = grad_output.shape
    n_cols = ori_shape[-1]

    grad_output_2d = grad_output.reshape(-1, n_cols).contiguous()
    gate_2d = gate.reshape(-1, n_cols).contiguous()
    up_2d = up.reshape(-1, n_cols).contiguous()
    n_rows = grad_output_2d.shape[0]

    grad_gate = torch.empty_like(gate_2d)
    grad_up = torch.empty_like(up_2d)

    TILE_SIZE = next_power_of_2(n_cols)
    grid = (n_rows,)

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        moe_gate_backward_kernel,
        (grad_output_2d, gate_2d, up_2d, grad_gate, grad_up, TILE_SIZE),
    )

    return grad_gate.view(*ori_shape), grad_up.view(*ori_shape)


# ============================================================================
# Autograd wrapper
# ============================================================================

class MoEGateFunction(torch.autograd.Function):
    """CuTile MoE expert gate with full backward support.

    Forward: stride-2 gather from interleaved gate_up, saves contiguous
    gate/up tensors in the same kernel pass (no extra Python copies).
    Backward: fully contiguous reads/writes, then re-interleave grads.
    """

    @staticmethod
    def forward(ctx, gate_up: torch.Tensor) -> torch.Tensor:
        output, saved_gate, saved_up = moe_gate_forward_cutile(gate_up)
        ctx.save_for_backward(saved_gate, saved_up)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        gate, up = ctx.saved_tensors
        grad_gate, grad_up = moe_gate_backward_cutile(grad_output, gate, up)

        # Re-interleave gradients into original layout
        grad_gate_up = torch.empty(
            *grad_gate.shape[:-1], grad_gate.shape[-1] * 2,
            dtype=grad_gate.dtype, device=grad_gate.device,
        )
        grad_gate_up[..., ::2] = grad_gate
        grad_gate_up[..., 1::2] = grad_up
        return grad_gate_up


def moe_gate(gate_up: torch.Tensor) -> torch.Tensor:
    """Apply CuTile MoE expert gate: fused interleaved gather + gating."""
    return MoEGateFunction.apply(gate_up)


# ============================================================================
# Patch function (replaces GptOssExperts._apply_gate)
# ============================================================================

def moe_gate_apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for GptOssExperts._apply_gate using CuTile."""
    return MoEGateFunction.apply(gate_up)
