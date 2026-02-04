"""
Rotary Position Embedding (RoPE) with gradient support.

Uses PyTorch implementation with autograd.
"""

import torch

from ..registry import register_patch


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Apply rotary position embeddings to query and key tensors.
    
    HuggingFace-compatible implementation with gradient support.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


# ============================================================================
# Register patches for Qwen3
# ============================================================================

register_patch(
    name="rope_qwen3",
    description="RoPE for Qwen3 models (PyTorch with autograd)",
    target_module="transformers.models.qwen3.modeling_qwen3",
    target_attr="apply_rotary_pos_emb",
    replacement=apply_rotary_pos_emb,
    has_backward=True,
    priority=10,
    models=["qwen3"],
)

# Note: GPT-OSS uses a different RoPE convention (partial rotation)
# so we don't patch it - the native implementation is compatible
