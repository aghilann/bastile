"""
CuTile Fused Cross-Entropy Loss - optimized for NVIDIA B200.

Fuses softmax, loss computation, and gradient storage into a single forward pass
using the online softmax algorithm. Gradients are stored in-place in the logits
tensor, making backward nearly free.

Key optimizations:
1. Online softmax (2 passes over vocab): avoids materializing full softmax
2. In-place gradient storage: saves memory (no separate gradient tensor)
3. Fused forward+gradient: single kernel replaces softmax + NLL + backward
4. Fast math: flush_to_zero on Blackwell
5. Zero CPU-GPU sync: all normalization uses tensor ops (no .item() calls)
"""

import torch
import torch.nn as nn
from typing import Optional

from ..registry import register_patch
from .utils import next_power_of_2

import cuda.tile as ct

ConstInt = ct.Constant[int]
ConstFloat = ct.Constant[float]


# ============================================================================
# Forward kernel
# ============================================================================

@ct.kernel
def cross_entropy_forward_kernel(
    logits,          # [BT, V] float32 - will be OVERWRITTEN with gradients
    targets,         # [BT] int64 - target class indices
    losses,          # [BT] float32 - per-token loss output
    V: ConstInt,
    BLOCK_SIZE: ConstInt,
):
    """Fused cross-entropy: online softmax + loss + raw gradient in one kernel.

    Each block processes one token (one row of the logits matrix).
    Uses online softmax to compute log-sum-exp in a single pass,
    then a second pass to store raw softmax gradients in-place.

    Gradients stored are raw: softmax[i] for all i, softmax[target] - 1 for target.
    Normalization by n_non_ignore happens in Python via autograd (avoids kernel
    recompilation and CPU-GPU sync).
    """
    bid = ct.bid(0)  # token index

    # Load target for this token
    target = ct.gather(targets, bid, check_bounds=True, padding_value=0)

    # Load the logit at the target position (needed for loss = lse - x[target])
    ori_x_target = ct.gather(logits, (bid, target), check_bounds=True, padding_value=0.0)
    ori_x_target = ct.astype(ori_x_target, ct.float32)

    # ---- Pass 1: Online softmax to find max (m) and sum (d) ----
    # Initialize from first block to get proper scalar types from reductions
    offsets_0 = ct.arange(BLOCK_SIZE, dtype=ct.int32)
    x_0 = ct.gather(logits, (bid, offsets_0), check_bounds=True, padding_value=-1e30)
    x_0 = ct.astype(x_0, ct.float32)

    m = ct.max(x_0, axis=0)  # scalar: running max
    d = ct.sum(ct.exp(ct.sub(x_0, m, flush_to_zero=True)), axis=0)

    for col_start in range(BLOCK_SIZE, V, BLOCK_SIZE):
        offsets = col_start + ct.arange(BLOCK_SIZE, dtype=ct.int32)
        x_block = ct.gather(logits, (bid, offsets), check_bounds=True, padding_value=-1e30)
        x_block = ct.astype(x_block, ct.float32)

        block_max = ct.max(x_block, axis=0)
        m_new = ct.maximum(m, block_max)
        d = ct.add(
            ct.mul(d, ct.exp(ct.sub(m, m_new, flush_to_zero=True)), flush_to_zero=True),
            ct.sum(ct.exp(ct.sub(x_block, m_new, flush_to_zero=True)), axis=0),
            flush_to_zero=True,
        )
        m = m_new

    # logsumexp = m + log(d)
    lse = ct.add(m, ct.log(d), flush_to_zero=True)

    # Per-token loss: lse - x[target]
    token_loss = ct.sub(lse, ori_x_target, flush_to_zero=True)
    ct.scatter(losses, bid, token_loss, check_bounds=True)

    # ---- Pass 2: Compute raw softmax gradients and overwrite logits ----
    for col_start in range(0, V, BLOCK_SIZE):
        offsets = col_start + ct.arange(BLOCK_SIZE, dtype=ct.int32)
        x_block = ct.gather(logits, (bid, offsets), check_bounds=True, padding_value=-1e30)
        x_block = ct.astype(x_block, ct.float32)

        # Raw softmax: exp(x_i - lse)
        grad_block = ct.exp(ct.sub(x_block, lse, flush_to_zero=True))

        grad_block = ct.astype(grad_block, logits.dtype)
        ct.scatter(logits, (bid, offsets), grad_block, check_bounds=True)

    # Fix the target position: subtract 1.0
    # grad[target] should be softmax[target] - 1.0
    target_grad = ct.gather(logits, (bid, target), check_bounds=True, padding_value=0.0)
    target_grad = ct.astype(target_grad, ct.float32)
    target_grad = ct.sub(target_grad, 1.0, flush_to_zero=True)
    target_grad = ct.astype(target_grad, logits.dtype)
    ct.scatter(logits, (bid, target), target_grad, check_bounds=True)


# ============================================================================
# Launch functions
# ============================================================================

def cross_entropy_forward_cutile(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """CuTile cross-entropy forward: returns per-token losses, overwrites logits with raw gradients."""
    BT, V = logits.shape

    losses = torch.zeros(BT, dtype=torch.float32, device=logits.device)

    if BT > 0:
        BLOCK_SIZE = min(next_power_of_2(V), 32768)
        grid = (BT,)

        ct.launch(
            torch.cuda.current_stream(),
            grid,
            cross_entropy_forward_kernel,
            (logits, targets, losses, V, BLOCK_SIZE),
        )

        # Zero out ignored tokens (all GPU ops, no CPU sync)
        ignore_mask = targets == ignore_index
        losses.masked_fill_(ignore_mask, 0.0)
        logits.masked_fill_(ignore_mask.unsqueeze(1), 0.0)

    return losses


# ============================================================================
# Autograd wrapper
# ============================================================================

class CuTileCrossEntropyFunction(torch.autograd.Function):
    """Fused cross-entropy with in-place gradient storage.

    Always returns sum of per-token losses (un-normalized).
    Normalization by n_non_ignore/num_items_in_batch happens outside this
    Function via tensor division, so autograd handles gradient scaling
    automatically. This avoids CPU-GPU sync for n_non_ignore computation.
    """

    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        logits = logits.contiguous()
        targets = targets.contiguous()

        input_requires_grad = logits.requires_grad

        # Launch kernel: overwrites logits with raw softmax gradients
        losses = cross_entropy_forward_cutile(logits, targets, ignore_index)

        # Sum of per-token losses (normalization happens outside)
        loss = losses.sum()

        if input_requires_grad:
            ctx.save_for_backward(logits.detach())

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        (grad_logits,) = ctx.saved_tensors
        # grad_output includes normalization from the division outside this Function.
        # Just scale stored raw gradients by grad_output.
        if grad_output.ndim == 0:
            grad_logits.mul_(grad_output)
        else:
            grad_logits.mul_(grad_output.unsqueeze(-1))
        return grad_logits, None, None


# ============================================================================
# Drop-in replacement for fixed_cross_entropy
# ============================================================================

def cutile_fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    """Drop-in replacement for transformers.loss.loss_utils.fixed_cross_entropy.

    Zero CPU-GPU sync: all normalization uses tensor operations.
    Gradient scaling handled by autograd through the division node.
    """
    # Compute sum of per-token losses (un-normalized)
    loss_sum = CuTileCrossEntropyFunction.apply(source, target, ignore_index)

    if num_items_in_batch is not None:
        # Training path: normalize by num_items_in_batch
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss_sum.device)
        loss = loss_sum / num_items_in_batch
    else:
        # Eval/test path: normalize by count of non-ignored tokens
        n_non_ignore = torch.clamp_min((target != ignore_index).sum(), 1)
        loss = loss_sum / n_non_ignore

    return loss


# ============================================================================
# Register patch
# ============================================================================

register_patch(
    name="cross_entropy",
    description="CuTile fused cross-entropy loss with in-place gradient",
    target_module="transformers.loss.loss_utils",
    target_attr="fixed_cross_entropy",
    replacement=cutile_fixed_cross_entropy,
    has_backward=True,
    priority=10,
    models=[],  # Universal - works for all causal LM models
)
