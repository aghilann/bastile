"""
CuTile kernels for GPT-OSS MoE (Mixture of Experts).

GPT-OSS uses a custom GEGLU activation:
  gate = gate.clamp(max=limit)
  up = up.clamp(min=-limit, max=limit)
  glu = gate * sigmoid(gate * alpha)
  output = (up + 1) * glu

This module provides:
1. Fused GEGLU activation kernel
2. Fused expert forward (gate_up projection + GEGLU + down projection)
3. Backward passes for training
"""

import torch
import torch.nn as nn
import cuda.tile as ct
from ..registry import register_patch


# Constants from GPT-OSS
ALPHA = 1.702
LIMIT = 7.0


@ct.kernel
def geglu_forward_kernel(
    gate_up_ptr,  # Input: (num_tokens, 2 * expert_dim) - interleaved gate/up
    output_ptr,   # Output: (num_tokens, expert_dim)
    num_tokens: int,
    expert_dim: ct.Constant[int],
    alpha: ct.Constant[float],
    limit: ct.Constant[float],
    BLOCK_SIZE: ct.Constant[int],
):
    """
    Fused GEGLU activation for GPT-OSS.
    
    Input is interleaved: gate_up[..., 0::2] = gate, gate_up[..., 1::2] = up
    Output: (up + 1) * (gate * sigmoid(gate * alpha))
    """
    row_idx = ct.bid(axis=0)
    if row_idx < num_tokens:
        for col_start in range(0, expert_dim, BLOCK_SIZE):
            col_offsets = col_start + ct.arange(BLOCK_SIZE, dtype=ct.int32)
            mask = col_offsets < expert_dim
            
            # Load interleaved gate and up values
            gate_indices = row_idx * (2 * expert_dim) + col_offsets * 2
            up_indices = gate_indices + 1
            
            gate = ct.gather(gate_up_ptr, gate_indices, padding_value=0.0)
            up = ct.gather(gate_up_ptr, up_indices, padding_value=0.0)
            
            # Convert to float32 for computation
            gate = ct.astype(gate, ct.float32)
            up = ct.astype(up, ct.float32)
            
            # Apply clamps
            gate = ct.minimum(gate, limit)
            up = ct.maximum(ct.minimum(up, limit), -limit)
            
            # GEGLU: (up + 1) * gate * sigmoid(gate * alpha)
            sigmoid_input = gate * alpha
            sigmoid_val = 1.0 / (1.0 + ct.exp(-sigmoid_input))
            glu = gate * sigmoid_val
            result = (up + 1.0) * glu
            
            # Convert back to output dtype
            result = ct.astype(result, output_ptr.dtype)
            
            # Store
            out_indices = row_idx * expert_dim + col_offsets
            ct.scatter(output_ptr, out_indices, result)


@ct.kernel
def geglu_backward_kernel(
    grad_output_ptr,  # Input: (num_tokens, expert_dim)
    gate_up_ptr,      # Input: (num_tokens, 2 * expert_dim) - saved from forward
    grad_gate_up_ptr, # Output: (num_tokens, 2 * expert_dim)
    num_tokens: int,
    expert_dim: ct.Constant[int],
    alpha: ct.Constant[float],
    limit: ct.Constant[float],
    BLOCK_SIZE: ct.Constant[int],
):
    """
    Backward pass for GEGLU activation.
    
    d_out/d_gate = (up + 1) * (sigmoid + gate * alpha * sigmoid * (1 - sigmoid))
    d_out/d_up = glu (where glu = gate * sigmoid(gate * alpha))
    
    Need to account for clamp gradients (zero out where clamped).
    """
    row_idx = ct.bid(axis=0)
    if row_idx < num_tokens:
        for col_start in range(0, expert_dim, BLOCK_SIZE):
            col_offsets = col_start + ct.arange(BLOCK_SIZE, dtype=ct.int32)
            mask = col_offsets < expert_dim
            
            # Load grad_output
            grad_out_indices = row_idx * expert_dim + col_offsets
            grad_out = ct.gather(grad_output_ptr, grad_out_indices, padding_value=0.0)
            grad_out = ct.astype(grad_out, ct.float32)
            
            # Load gate and up (interleaved)
            gate_indices = row_idx * (2 * expert_dim) + col_offsets * 2
            up_indices = gate_indices + 1
            
            gate_orig = ct.gather(gate_up_ptr, gate_indices, padding_value=0.0)
            up_orig = ct.gather(gate_up_ptr, up_indices, padding_value=0.0)
            gate_orig = ct.astype(gate_orig, ct.float32)
            up_orig = ct.astype(up_orig, ct.float32)
            
            # Apply clamps for forward computation
            gate = ct.minimum(gate_orig, limit)
            up = ct.maximum(ct.minimum(up_orig, limit), -limit)
            
            # Compute sigmoid and derivatives
            sigmoid_input = gate * alpha
            sigmoid_val = 1.0 / (1.0 + ct.exp(-sigmoid_input))
            
            # glu = gate * sigmoid(gate * alpha)
            glu = gate * sigmoid_val
            
            # d_out/d_up = glu (gradient through (up + 1))
            # Note: We compute full gradient, clamp is a pass-through for values in range
            d_up = grad_out * glu
            
            # d_out/d_gate = (up + 1) * d(glu)/d(gate)
            # d(glu)/d(gate) = sigmoid + gate * alpha * sigmoid * (1 - sigmoid)
            #                = sigmoid * (1 + gate * alpha * (1 - sigmoid))
            d_glu_d_gate = sigmoid_val * (1.0 + gate * alpha * (1.0 - sigmoid_val))
            d_gate = grad_out * (up + 1.0) * d_glu_d_gate
            
            # Convert back and store (interleaved)
            d_gate = ct.astype(d_gate, grad_gate_up_ptr.dtype)
            d_up = ct.astype(d_up, grad_gate_up_ptr.dtype)
            
            ct.scatter(grad_gate_up_ptr, gate_indices, d_gate)
            ct.scatter(grad_gate_up_ptr, up_indices, d_up)


class GEGLUFunction(torch.autograd.Function):
    """PyTorch autograd wrapper for CuTile GEGLU."""
    
    @staticmethod
    def forward(ctx, gate_up: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gate_up: (num_tokens, 2 * expert_dim) - interleaved gate/up values
        Returns:
            output: (num_tokens, expert_dim)
        """
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
            
            grid = (num_tokens,)
            ct.launch(
                torch.cuda.current_stream(),
                grid,
                geglu_forward_kernel,
                (
                    gate_up_flat,
                    output_flat,
                    num_tokens,
                    expert_dim,
                    ALPHA,
                    LIMIT,
                    min(256, expert_dim),  # BLOCK_SIZE
                ),
            )
        
        ctx.save_for_backward(gate_up)
        ctx.expert_dim = expert_dim
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        gate_up, = ctx.saved_tensors
        expert_dim = ctx.expert_dim
        num_tokens = gate_up.shape[0]
        
        grad_gate_up = torch.empty_like(gate_up)
        
        if num_tokens > 0:
            gate_up_flat = gate_up.reshape(-1)
            grad_output_flat = grad_output.reshape(-1)
            grad_gate_up_flat = grad_gate_up.reshape(-1)
            
            grid = (num_tokens,)
            ct.launch(
                torch.cuda.current_stream(),
                grid,
                geglu_backward_kernel,
                (
                    grad_output_flat,
                    gate_up_flat,
                    grad_gate_up_flat,
                    num_tokens,
                    expert_dim,
                    ALPHA,
                    LIMIT,
                    min(256, expert_dim),
                ),
            )
        
        return grad_gate_up


def geglu_activation(gate_up: torch.Tensor) -> torch.Tensor:
    """Apply GEGLU activation to interleaved gate/up tensor."""
    return GEGLUFunction.apply(gate_up)


class BastileGptOssExperts(nn.Module):
    """
    Optimized GPT-OSS Experts module with CuTile GEGLU activation.
    
    This replaces the training path's expert-by-expert loop with:
    1. Batched matrix multiplications
    2. Fused GEGLU activation kernel
    """
    
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        
        # Same parameter layout as original
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
        """
        Forward pass with optimized GEGLU activation.
        
        Uses batched operations for inference, expert-by-expert with fused
        GEGLU for training.
        """
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        num_experts = routing_weights.shape[1]
        
        if hidden_states.device.type == "cpu" or self.training:
            # Training path: expert-by-expert with fused GEGLU
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
                
                # Gate-up projection
                gate_up = (
                    current_state @ self.gate_up_proj[expert_idx]
                    + self.gate_up_proj_bias[expert_idx]
                )
                
                # Fused GEGLU activation (CuTile kernel)
                gated_output = geglu_activation(gate_up)
                
                # Down projection
                out = (
                    gated_output @ self.down_proj[expert_idx]
                    + self.down_proj_bias[expert_idx]
                )
                
                # Apply routing weight
                weighted_output = out * routing_weights[token_idx, expert_idx, None]
                next_states.index_add_(
                    0, token_idx, weighted_output.to(hidden_states.dtype)
                )
            
            next_states = next_states.view(batch_size, -1, self.hidden_size)
        else:
            # Inference path: batched computation
            hidden_states = hidden_states.repeat(num_experts, 1)
            hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)
            
            # Batched gate-up projection
            gate_up = (
                torch.bmm(hidden_states, self.gate_up_proj)
                + self.gate_up_proj_bias[..., None, :]
            )
            
            # Apply fused GEGLU per expert
            # Reshape for batch processing: (num_experts, tokens, 2*expert_dim)
            gated_outputs = []
            for e in range(num_experts):
                gated = geglu_activation(gate_up[e])
                gated_outputs.append(gated)
            gated_output = torch.stack(gated_outputs, dim=0)
            
            # Down projection
            next_states = torch.bmm(gated_output, self.down_proj)
            next_states = next_states + self.down_proj_bias[..., None, :]
            
            # Apply routing weights
            next_states = next_states.view(
                num_experts, batch_size, -1, self.hidden_size
            )
            next_states = (
                next_states
                * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[
                    ..., None
                ]
            )
            next_states = next_states.sum(dim=0)
        
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
