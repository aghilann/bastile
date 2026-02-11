"""
Fused Linear Cross-Entropy Loss for Bastile.

Avoids materializing the full logits tensor [BT, V] by chunking the computation.
For each chunk, computes: logits_chunk = hidden @ weight.T, then applies the
existing CuTile cross-entropy kernel. Gradients are pre-computed during forward,
making backward nearly free.

Key optimizations:
1. Never materializes full [BT, V] logits (saves ~512MB for typical Qwen3 configs)
2. Reuses CuTile cross-entropy kernel (online softmax + in-place gradients)
3. Pre-computes grad_input and grad_weight during forward
4. Backward just scales stored gradients

Reference: Liger Kernel's fused_linear_cross_entropy.py
"""

import torch
import torch.nn as nn

from ..registry import register_patch
from .utils import next_power_of_2, ceildiv
from .cross_entropy import cross_entropy_forward_cutile


class FusedLinearCrossEntropyFunction(torch.autograd.Function):
    """Fused linear projection + cross-entropy with chunked computation.

    Avoids materializing full [BT, V] logits tensor. Instead:
    1. Chunks BT dimension into manageable pieces
    2. For each chunk: matmul → CuTile CE kernel → accumulate gradients
    3. Backward just scales pre-computed gradients
    """

    @staticmethod
    def forward(
        ctx,
        _input,       # [BT, H] hidden states
        weight,       # [V, H] lm_head weight
        target,       # [BT] target labels
        bias=None,    # [V] optional bias
        ignore_index=-100,
    ):
        input_requires_grad = _input.requires_grad

        BT, H = _input.shape
        V = weight.shape[0]

        # Calculate chunk size to bound memory usage
        inc_factor = ceildiv(V, H)
        chunk_size = next_power_of_2(ceildiv(BT, inc_factor))
        num_chunks = ceildiv(BT, chunk_size)

        # Allocate outputs
        grad_input = torch.zeros_like(_input) if input_requires_grad else None
        grad_weight = torch.zeros_like(weight, dtype=torch.float32) if (input_requires_grad and weight.requires_grad) else None
        grad_bias = torch.zeros_like(bias, dtype=torch.float32) if (bias is not None and input_requires_grad) else None

        loss_1d = torch.zeros(BT, dtype=torch.float32, device=_input.device)

        # Count non-ignored tokens (GPU tensor, avoid CPU sync)
        n_non_ignore = torch.clamp_min((target != ignore_index).sum(), 1)

        for chunk_id in range(num_chunks):
            start_idx = chunk_id * chunk_size
            end_idx = min((chunk_id + 1) * chunk_size, BT)
            _input_chunk = _input[start_idx:end_idx]

            # Compute logits chunk (stays in input dtype, e.g. bf16)
            logits_chunk = _input_chunk @ weight.t()
            if bias is not None:
                logits_chunk = logits_chunk + bias

            target_chunk = target[start_idx:end_idx]

            # CuTile CE kernel needs float32 logits (stores gradients in-place)
            logits_chunk = logits_chunk.float()

            # CuTile cross-entropy: computes loss + overwrites logits with gradients
            chunk_losses = cross_entropy_forward_cutile(
                logits_chunk, target_chunk, ignore_index
            )
            loss_1d[start_idx:end_idx] = chunk_losses

            # logits_chunk now contains raw softmax gradients (float32)
            if input_requires_grad:
                # grad_input = grad_logits @ weight (cast grad to input dtype for fast matmul)
                grad_input[start_idx:end_idx] = logits_chunk.to(_input.dtype) @ weight

            if grad_weight is not None:
                # Accumulate in float32: grad_logits.T @ input (bf16 matmul then convert)
                torch.addmm(grad_weight, logits_chunk.t(), _input_chunk.float(), out=grad_weight)

            if grad_bias is not None:
                grad_bias += logits_chunk.sum(dim=0)

        # Mean loss
        loss = loss_1d.sum() / n_non_ignore

        # Scale gradients by 1/n_non_ignore
        if input_requires_grad and grad_input is not None:
            grad_input.div_(n_non_ignore)
        if grad_weight is not None:
            grad_weight.div_(n_non_ignore)
            grad_weight = grad_weight.to(weight.dtype)
        if grad_bias is not None:
            grad_bias.div_(n_non_ignore)
            grad_bias = grad_bias.to(bias.dtype)

        ctx.save_for_backward(
            grad_input.detach() if grad_input is not None else None,
            grad_weight.detach() if grad_weight is not None else None,
            grad_bias.detach() if grad_bias is not None else None,
        )

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight, grad_bias = ctx.saved_tensors

        # Scale pre-computed gradients by upstream gradient
        # For CE as final layer, grad_output is scalar 1.0, skip multiply
        if grad_input is not None and grad_output.ne(1.0).any():
            grad_input = grad_input * grad_output
            if grad_weight is not None:
                grad_weight = grad_weight * grad_output
            if grad_bias is not None:
                grad_bias = grad_bias * grad_output

        return (
            grad_input,   # _input
            grad_weight,  # weight
            None,         # target
            grad_bias,    # bias
            None,         # ignore_index
        )


def fused_linear_cross_entropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias=None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute fused linear projection + cross-entropy loss.

    Args:
        hidden_states: [BT, H] or [B, T, H] hidden states
        weight: [V, H] language model head weight
        target: [BT] or [B, T] target labels
        bias: optional [V] bias
        ignore_index: index to ignore in target

    Returns:
        Scalar loss value
    """
    # Flatten if needed
    if hidden_states.ndim == 3:
        B, T, H = hidden_states.shape
        hidden_states = hidden_states.view(-1, H)
        target = target.view(-1)

    return FusedLinearCrossEntropyFunction.apply(
        hidden_states, weight, target, bias, ignore_index
    )
