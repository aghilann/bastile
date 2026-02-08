"""
CuTile kernels for GPT-OSS MoE (Mixture of Experts) with autotuning.

GPT-OSS uses a custom GEGLU activation:
  gate = gate.clamp(max=limit)
  up = up.clamp(min=-limit, max=limit)
  glu = gate * sigmoid(gate * alpha)
  output = (up + 1) * glu

This module provides:
1. Fused GEGLU activation kernel with autotuning (including occupancy)
2. Fused expert forward (gate_up projection + GEGLU + down projection)
3. Backward passes for training
"""

import torch
import torch.nn as nn
from typing import Optional

from ..registry import register_patch
from ..autotune import autotune, default_key
from .configs import GEGLUConfig

import cuda.tile as ct

# Constants from GPT-OSS
ALPHA = 1.702
LIMIT = 7.0

ConstInt = ct.Constant[int]
ConstFloat = ct.Constant[float]


def geglu_search_space():
    """Generate search space for GEGLU autotuning with occupancy."""
    for block_size in [64, 128, 256, 512, 1024]:
        for occupancy in [1, 2, 4, 8]:
            yield GEGLUConfig(block_size=block_size, occupancy=occupancy, use_float32=True)


# ============================================================================
# Forward kernel variants with different occupancy levels
# ============================================================================

def _geglu_forward_impl(gate_up_ptr, output_ptr, num_tokens, expert_dim, alpha, limit, BLOCK_SIZE):
    """Shared GEGLU forward implementation."""
    pid = ct.bid(0)
    num_programs = ct.num_blocks(0)
    
    for row_idx in range(pid, num_tokens, num_programs):
        for col_start in range(0, expert_dim, BLOCK_SIZE):
            col_offsets = col_start + ct.arange(BLOCK_SIZE, dtype=ct.int32)
            
            gate_indices = row_idx * (2 * expert_dim) + col_offsets * 2
            up_indices = gate_indices + 1
            
            gate = ct.gather(gate_up_ptr, gate_indices, check_bounds=True, padding_value=0.0)
            up = ct.gather(gate_up_ptr, up_indices, check_bounds=True, padding_value=0.0)
            
            gate = ct.astype(gate, ct.float32)
            up = ct.astype(up, ct.float32)
            
            gate = ct.minimum(gate, limit)
            up = ct.maximum(ct.minimum(up, limit), -limit)
            
            sigmoid_input = ct.mul(gate, ct.full((BLOCK_SIZE,), alpha, dtype=ct.float32))
            neg_sigmoid_input = ct.sub(ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32), sigmoid_input)
            exp_neg = ct.exp(neg_sigmoid_input)
            one = ct.full((BLOCK_SIZE,), 1.0, dtype=ct.float32)
            sigmoid_val = ct.truediv(one, ct.add(one, exp_neg))
            
            glu = ct.mul(gate, sigmoid_val)
            up_plus_one = ct.add(up, one)
            result = ct.mul(up_plus_one, glu)
            
            result = ct.astype(result, output_ptr.dtype)
            
            out_indices = row_idx * expert_dim + col_offsets
            ct.scatter(output_ptr, out_indices, result, check_bounds=True)


@ct.kernel(occupancy=1)
def geglu_forward_kernel_occ1(
    gate_up_ptr, output_ptr,
    num_tokens: ConstInt, expert_dim: ConstInt,
    alpha: ConstFloat, limit: ConstFloat, BLOCK_SIZE: ConstInt,
):
    """GEGLU forward with occupancy=1."""
    _geglu_forward_impl(gate_up_ptr, output_ptr, num_tokens, expert_dim, alpha, limit, BLOCK_SIZE)


@ct.kernel(occupancy=2)
def geglu_forward_kernel_occ2(
    gate_up_ptr, output_ptr,
    num_tokens: ConstInt, expert_dim: ConstInt,
    alpha: ConstFloat, limit: ConstFloat, BLOCK_SIZE: ConstInt,
):
    """GEGLU forward with occupancy=2."""
    _geglu_forward_impl(gate_up_ptr, output_ptr, num_tokens, expert_dim, alpha, limit, BLOCK_SIZE)


@ct.kernel(occupancy=4)
def geglu_forward_kernel_occ4(
    gate_up_ptr, output_ptr,
    num_tokens: ConstInt, expert_dim: ConstInt,
    alpha: ConstFloat, limit: ConstFloat, BLOCK_SIZE: ConstInt,
):
    """GEGLU forward with occupancy=4."""
    _geglu_forward_impl(gate_up_ptr, output_ptr, num_tokens, expert_dim, alpha, limit, BLOCK_SIZE)


@ct.kernel(occupancy=8)
def geglu_forward_kernel_occ8(
    gate_up_ptr, output_ptr,
    num_tokens: ConstInt, expert_dim: ConstInt,
    alpha: ConstFloat, limit: ConstFloat, BLOCK_SIZE: ConstInt,
):
    """GEGLU forward with occupancy=8."""
    _geglu_forward_impl(gate_up_ptr, output_ptr, num_tokens, expert_dim, alpha, limit, BLOCK_SIZE)


GEGLU_FORWARD_KERNELS = {
    1: geglu_forward_kernel_occ1,
    2: geglu_forward_kernel_occ2,
    4: geglu_forward_kernel_occ4,
    8: geglu_forward_kernel_occ8,
}


# ============================================================================
# Backward kernel variants with different occupancy levels
# ============================================================================

def _geglu_backward_impl(grad_output_ptr, gate_up_ptr, grad_gate_up_ptr, num_tokens, expert_dim, alpha, limit, BLOCK_SIZE):
    """Shared GEGLU backward implementation."""
    pid = ct.bid(0)
    num_programs = ct.num_blocks(0)
    
    for row_idx in range(pid, num_tokens, num_programs):
        for col_start in range(0, expert_dim, BLOCK_SIZE):
            col_offsets = col_start + ct.arange(BLOCK_SIZE, dtype=ct.int32)
            
            grad_out_indices = row_idx * expert_dim + col_offsets
            grad_out = ct.gather(grad_output_ptr, grad_out_indices, check_bounds=True, padding_value=0.0)
            grad_out = ct.astype(grad_out, ct.float32)
            
            gate_indices = row_idx * (2 * expert_dim) + col_offsets * 2
            up_indices = gate_indices + 1
            
            gate_orig = ct.gather(gate_up_ptr, gate_indices, check_bounds=True, padding_value=0.0)
            up_orig = ct.gather(gate_up_ptr, up_indices, check_bounds=True, padding_value=0.0)
            gate_orig = ct.astype(gate_orig, ct.float32)
            up_orig = ct.astype(up_orig, ct.float32)
            
            gate = ct.minimum(gate_orig, limit)
            up = ct.maximum(ct.minimum(up_orig, limit), -limit)
            
            alpha_tensor = ct.full((BLOCK_SIZE,), alpha, dtype=ct.float32)
            sigmoid_input = ct.mul(gate, alpha_tensor)
            zero = ct.full((BLOCK_SIZE,), 0.0, dtype=ct.float32)
            neg_sigmoid_input = ct.sub(zero, sigmoid_input)
            exp_neg = ct.exp(neg_sigmoid_input)
            one = ct.full((BLOCK_SIZE,), 1.0, dtype=ct.float32)
            sigmoid_val = ct.truediv(one, ct.add(one, exp_neg))
            
            glu = ct.mul(gate, sigmoid_val)
            
            d_up = ct.mul(grad_out, glu)
            
            one_minus_sig = ct.sub(one, sigmoid_val)
            term = ct.mul(ct.mul(gate, alpha_tensor), one_minus_sig)
            d_glu_d_gate = ct.mul(sigmoid_val, ct.add(one, term))
            up_plus_one = ct.add(up, one)
            d_gate = ct.mul(ct.mul(grad_out, up_plus_one), d_glu_d_gate)
            
            d_gate = ct.astype(d_gate, grad_gate_up_ptr.dtype)
            d_up = ct.astype(d_up, grad_gate_up_ptr.dtype)
            
            ct.scatter(grad_gate_up_ptr, gate_indices, d_gate, check_bounds=True)
            ct.scatter(grad_gate_up_ptr, up_indices, d_up, check_bounds=True)


@ct.kernel(occupancy=1)
def geglu_backward_kernel_occ1(
    grad_output_ptr, gate_up_ptr, grad_gate_up_ptr,
    num_tokens: ConstInt, expert_dim: ConstInt,
    alpha: ConstFloat, limit: ConstFloat, BLOCK_SIZE: ConstInt,
):
    """GEGLU backward with occupancy=1."""
    _geglu_backward_impl(grad_output_ptr, gate_up_ptr, grad_gate_up_ptr, num_tokens, expert_dim, alpha, limit, BLOCK_SIZE)


@ct.kernel(occupancy=2)
def geglu_backward_kernel_occ2(
    grad_output_ptr, gate_up_ptr, grad_gate_up_ptr,
    num_tokens: ConstInt, expert_dim: ConstInt,
    alpha: ConstFloat, limit: ConstFloat, BLOCK_SIZE: ConstInt,
):
    """GEGLU backward with occupancy=2."""
    _geglu_backward_impl(grad_output_ptr, gate_up_ptr, grad_gate_up_ptr, num_tokens, expert_dim, alpha, limit, BLOCK_SIZE)


@ct.kernel(occupancy=4)
def geglu_backward_kernel_occ4(
    grad_output_ptr, gate_up_ptr, grad_gate_up_ptr,
    num_tokens: ConstInt, expert_dim: ConstInt,
    alpha: ConstFloat, limit: ConstFloat, BLOCK_SIZE: ConstInt,
):
    """GEGLU backward with occupancy=4."""
    _geglu_backward_impl(grad_output_ptr, gate_up_ptr, grad_gate_up_ptr, num_tokens, expert_dim, alpha, limit, BLOCK_SIZE)


@ct.kernel(occupancy=8)
def geglu_backward_kernel_occ8(
    grad_output_ptr, gate_up_ptr, grad_gate_up_ptr,
    num_tokens: ConstInt, expert_dim: ConstInt,
    alpha: ConstFloat, limit: ConstFloat, BLOCK_SIZE: ConstInt,
):
    """GEGLU backward with occupancy=8."""
    _geglu_backward_impl(grad_output_ptr, gate_up_ptr, grad_gate_up_ptr, num_tokens, expert_dim, alpha, limit, BLOCK_SIZE)


GEGLU_BACKWARD_KERNELS = {
    1: geglu_backward_kernel_occ1,
    2: geglu_backward_kernel_occ2,
    4: geglu_backward_kernel_occ4,
    8: geglu_backward_kernel_occ8,
}


# ============================================================================
# Kernel launch functions
# ============================================================================

def geglu_forward_cutile(
    gate_up: torch.Tensor,
    config: GEGLUConfig,
) -> torch.Tensor:
    """Launch CuTile GEGLU forward kernel."""
    num_tokens = gate_up.shape[0]
    expert_dim = gate_up.shape[1] // 2
    
    output = torch.empty(
        (num_tokens, expert_dim),
        dtype=gate_up.dtype,
        device=gate_up.device,
    )
    
    if num_tokens > 0:
        gate_up_flat = gate_up.reshape(-1)
        output_flat = output.reshape(-1)
        
        NUM_SM = torch.cuda.get_device_properties(gate_up.device).multi_processor_count
        num_programs = min(NUM_SM * config.occupancy, num_tokens)
        grid = (num_programs,)
        
        kernel = GEGLU_FORWARD_KERNELS[config.occupancy]
        
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            kernel,
            (
                gate_up_flat,
                output_flat,
                num_tokens,
                expert_dim,
                ALPHA,
                LIMIT,
                config.block_size,
            ),
        )
    
    return output


def geglu_backward_cutile(
    grad_output: torch.Tensor,
    gate_up: torch.Tensor,
    config: GEGLUConfig,
) -> torch.Tensor:
    """Launch CuTile GEGLU backward kernel."""
    num_tokens = gate_up.shape[0]
    expert_dim = gate_up.shape[1] // 2
    
    grad_gate_up = torch.empty_like(gate_up)
    
    if num_tokens > 0:
        gate_up_flat = gate_up.reshape(-1)
        grad_output_flat = grad_output.reshape(-1)
        grad_gate_up_flat = grad_gate_up.reshape(-1)
        
        NUM_SM = torch.cuda.get_device_properties(gate_up.device).multi_processor_count
        num_programs = min(NUM_SM * config.occupancy, num_tokens)
        grid = (num_programs,)
        
        kernel = GEGLU_BACKWARD_KERNELS[config.occupancy]
        
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            kernel,
            (
                grad_output_flat,
                gate_up_flat,
                grad_gate_up_flat,
                num_tokens,
                expert_dim,
                ALPHA,
                LIMIT,
                config.block_size,
            ),
        )
    
    return grad_gate_up


class GEGLUFunction(torch.autograd.Function):
    """PyTorch autograd wrapper for autotuned CuTile GEGLU."""
    
    _cached_config: Optional[GEGLUConfig] = None
    
    @staticmethod
    def forward(ctx, gate_up: torch.Tensor) -> torch.Tensor:
        """Forward pass with autotuning."""
        num_tokens = gate_up.shape[0]
        expert_dim = gate_up.shape[1] // 2
        
        key = (num_tokens, expert_dim, str(gate_up.dtype))
        
        def run_with_config(cfg):
            geglu_forward_cutile(gate_up, cfg)
        
        config = autotune(
            kernel_name="geglu_forward",
            run_fn=run_with_config,
            search_space=geglu_search_space(),
            key=key,
            max_iter=20,  # More iterations for occupancy tuning
            use_heuristic=False,  # Actually benchmark configs on B200
        )
        
        GEGLUFunction._cached_config = config
        
        output = geglu_forward_cutile(gate_up, config)
        
        ctx.save_for_backward(gate_up)
        ctx.config = config
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        gate_up, = ctx.saved_tensors
        config = ctx.config
        
        grad_gate_up = geglu_backward_cutile(grad_output, gate_up, config)
        
        return grad_gate_up


def geglu_activation(gate_up: torch.Tensor) -> torch.Tensor:
    """Apply autotuned GEGLU activation to interleaved gate/up tensor."""
    return GEGLUFunction.apply(gate_up)


class BastileGptOssExperts(nn.Module):
    """Optimized GPT-OSS Experts module with autotuned CuTile GEGLU activation."""
    
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
    description="Autotuned CuTile Fused GEGLU MoE Experts for GPT-OSS models",
    target_module="transformers.models.gpt_oss.modeling_gpt_oss",
    target_attr="GptOssExperts",
    replacement=BastileGptOssExperts,
    has_backward=True,
    priority=10,
    models=["gpt_oss"],
)
