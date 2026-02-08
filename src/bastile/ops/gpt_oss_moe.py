"""
CuTile kernels for GPT-OSS MoE (Mixture of Experts) - optimized for NVIDIA B200.

GPT-OSS uses a custom GEGLU activation:
  gate = gate.clamp(max=limit)
  up = up.clamp(min=-limit, max=limit)
  glu = gate * sigmoid(gate * alpha)
  output = (up + 1) * glu

Key optimizations:
1. gather/scatter memory access with fast math
2. flush_to_zero=True for fast sigmoid computation
3. Approximate rounding for division (fast reciprocal)
4. One block per row with inner column loop for cache-friendly strided access
5. Full backward pass with recomputation (no extra memory)
"""

import torch
import torch.nn as nn
from typing import Optional

from ..registry import register_patch
from .utils import next_power_of_2

import cuda.tile as ct
from cuda.tile._numeric_semantics import RoundingMode as RMd

# Constants from GPT-OSS
ALPHA = 1.702
LIMIT = 7.0

ConstInt = ct.Constant[int]
ConstFloat = ct.Constant[float]


# ============================================================================
# Forward kernel
# ============================================================================

@ct.kernel
def geglu_forward_kernel(
    gate_up, output,
    expert_dim: ConstInt,
    alpha: ConstFloat, limit: ConstFloat,
    BLOCK_SIZE: ConstInt,
):
    """GEGLU forward: (up + 1) * gate * sigmoid(gate * alpha) with fast math.

    One block per row with inner column loop. Uses 2D gather/scatter
    for bounds safety with stride-2 access for interleaved gate/up.
    """
    bid = ct.bid(0)

    for col_start in range(0, expert_dim, BLOCK_SIZE):
        offsets = col_start + ct.arange(BLOCK_SIZE, dtype=ct.int32)

        # 2D gather with stride-2 for interleaved layout
        gate_cols = offsets * 2
        up_cols = offsets * 2 + 1

        gate = ct.gather(gate_up, (bid, gate_cols), check_bounds=True, padding_value=0.0)
        up = ct.gather(gate_up, (bid, up_cols), check_bounds=True, padding_value=0.0)

        gate = ct.astype(gate, ct.float32)
        up = ct.astype(up, ct.float32)

        # Clamp
        gate = ct.minimum(gate, limit)
        up = ct.maximum(ct.minimum(up, limit), -limit)

        # Sigmoid(gate * alpha) with fast math
        denom = ct.add(1.0, ct.exp(-(gate * alpha)), flush_to_zero=True)
        sigmoid_val = ct.truediv(1.0, denom, flush_to_zero=True, rounding_mode=RMd.APPROX)

        # glu = gate * sigmoid(gate * alpha)
        glu = ct.mul(gate, sigmoid_val, flush_to_zero=True)

        # output = (up + 1) * glu
        result = ct.mul(up + 1.0, glu, flush_to_zero=True)
        result = ct.astype(result, output.dtype)

        # 2D scatter (bounds-safe)
        ct.scatter(output, (bid, offsets), result, check_bounds=True)


# ============================================================================
# Backward kernel
# ============================================================================

@ct.kernel
def geglu_backward_kernel(
    grad_output, gate_up, grad_gate_up,
    expert_dim: ConstInt,
    alpha: ConstFloat, limit: ConstFloat,
    BLOCK_SIZE: ConstInt,
):
    """GEGLU backward with gather/scatter and fast math.

    d(output)/d(gate) = (up + 1) * sigmoid(g*a) * (1 + g*a*(1-sigmoid(g*a)))
    d(output)/d(up) = glu
    """
    bid = ct.bid(0)

    for col_start in range(0, expert_dim, BLOCK_SIZE):
        offsets = col_start + ct.arange(BLOCK_SIZE, dtype=ct.int32)

        # Gather grad_output
        grad_out = ct.gather(grad_output, (bid, offsets), check_bounds=True, padding_value=0.0)
        grad_out = ct.astype(grad_out, ct.float32)

        # Gather gate/up from interleaved layout
        gate_cols = offsets * 2
        up_cols = offsets * 2 + 1
        gate_orig = ct.gather(gate_up, (bid, gate_cols), check_bounds=True, padding_value=0.0)
        up_orig = ct.gather(gate_up, (bid, up_cols), check_bounds=True, padding_value=0.0)
        gate_orig = ct.astype(gate_orig, ct.float32)
        up_orig = ct.astype(up_orig, ct.float32)

        # Recompute clamped values
        gate = ct.minimum(gate_orig, limit)
        up = ct.maximum(ct.minimum(up_orig, limit), -limit)

        # Recompute sigmoid with fast math
        denom = ct.add(1.0, ct.exp(-(gate * alpha)), flush_to_zero=True)
        sigmoid_val = ct.truediv(1.0, denom, flush_to_zero=True, rounding_mode=RMd.APPROX)

        glu = ct.mul(gate, sigmoid_val, flush_to_zero=True)

        # d_up = grad_out * glu
        d_up = ct.mul(grad_out, glu, flush_to_zero=True)

        # d_glu/d_gate = sigmoid * (1 + g*a*(1-sigmoid))
        one_minus_sig = 1.0 - sigmoid_val
        term = (gate * alpha) * one_minus_sig
        d_glu_d_gate = ct.mul(sigmoid_val, 1.0 + term, flush_to_zero=True)
        d_gate = ct.mul(ct.mul(grad_out, up + 1.0, flush_to_zero=True), d_glu_d_gate, flush_to_zero=True)

        d_gate = ct.astype(d_gate, grad_gate_up.dtype)
        d_up = ct.astype(d_up, grad_gate_up.dtype)

        # 2D scatter with stride-2 for interleaved layout
        ct.scatter(grad_gate_up, (bid, gate_cols), d_gate, check_bounds=True)
        ct.scatter(grad_gate_up, (bid, up_cols), d_up, check_bounds=True)


# ============================================================================
# Launch functions
# ============================================================================

def geglu_forward_cutile(gate_up: torch.Tensor) -> torch.Tensor:
    """CuTile-accelerated GEGLU forward."""
    num_tokens = gate_up.shape[0]
    expert_dim = gate_up.shape[1] // 2

    output = torch.empty(
        (num_tokens, expert_dim),
        dtype=gate_up.dtype,
        device=gate_up.device,
    )

    if num_tokens > 0:
        # Use smaller block size for cache-friendly strided access
        BLOCK_SIZE = min(next_power_of_2(expert_dim), 256)
        grid = (num_tokens,)

        ct.launch(
            torch.cuda.current_stream(),
            grid,
            geglu_forward_kernel,
            (gate_up, output, expert_dim, ALPHA, LIMIT, BLOCK_SIZE),
        )

    return output


def geglu_backward_cutile(
    grad_output: torch.Tensor,
    gate_up: torch.Tensor,
) -> torch.Tensor:
    """CuTile-accelerated GEGLU backward."""
    num_tokens = gate_up.shape[0]
    expert_dim = gate_up.shape[1] // 2

    grad_gate_up = torch.empty_like(gate_up)

    if num_tokens > 0:
        BLOCK_SIZE = min(next_power_of_2(expert_dim), 256)
        grid = (num_tokens,)

        ct.launch(
            torch.cuda.current_stream(),
            grid,
            geglu_backward_kernel,
            (grad_output, gate_up, grad_gate_up, expert_dim, ALPHA, LIMIT, BLOCK_SIZE),
        )

    return grad_gate_up


# ============================================================================
# Autograd wrapper
# ============================================================================

class GEGLUFunction(torch.autograd.Function):
    """CuTile GEGLU with full backward support."""

    @staticmethod
    def forward(ctx, gate_up: torch.Tensor) -> torch.Tensor:
        output = geglu_forward_cutile(gate_up)
        ctx.save_for_backward(gate_up)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        gate_up, = ctx.saved_tensors
        grad_gate_up = geglu_backward_cutile(grad_output, gate_up)
        return grad_gate_up


def geglu_activation(gate_up: torch.Tensor) -> torch.Tensor:
    """Apply CuTile GEGLU activation to interleaved gate/up tensor."""
    return GEGLUFunction.apply(gate_up)


# ============================================================================
# MoE Module
# ============================================================================

class BastileGptOssExperts(nn.Module):
    """Optimized GPT-OSS Experts module with CuTile GEGLU activation."""

    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size

        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim)
        )
        self.gate_up_proj_bias = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.expert_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.expert_dim, self.hidden_size)
        )
        self.down_proj_bias = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size)
        )
        self.alpha = ALPHA
        self.limit = LIMIT

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_indices=None,
        routing_weights=None,
    ) -> torch.Tensor:
        """Forward pass with fused GEGLU activation."""
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        num_experts = routing_weights.shape[1]

        next_states = torch.zeros_like(hidden_states)

        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                router_indices, num_classes=num_experts
            )
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)), 0
            ).nonzero()

        for expert_idx in expert_hit[:]:
            expert_idx = expert_idx[0]
            with torch.no_grad():
                _, token_idx = torch.where(expert_mask[expert_idx])

            current_state = hidden_states[token_idx]

            gate_up = (
                current_state @ self.gate_up_proj[expert_idx]
                + self.gate_up_proj_bias[expert_idx]
            )

            gated_output = geglu_activation(gate_up)

            out = (
                gated_output @ self.down_proj[expert_idx]
                + self.down_proj_bias[expert_idx]
            )

            weighted_output = out * routing_weights[token_idx, expert_idx, None]
            next_states.index_add_(
                0, token_idx, weighted_output.to(hidden_states.dtype)
            )

        next_states = next_states.view(batch_size, -1, self.hidden_size)

        return next_states


# Register the patch
register_patch(
    name="moe_experts_gpt_oss",
    description="CuTile Fused GEGLU MoE Experts for GPT-OSS models",
    target_module="transformers.models.gpt_oss.modeling_gpt_oss",
    target_attr="GptOssExperts",
    replacement=BastileGptOssExperts,
    has_backward=True,
    priority=10,
    models=["gpt_oss"],
)
