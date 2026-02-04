"""
Unit tests for RoPE (Rotary Position Embedding).

Verifies forward and backward passes work correctly.
"""

import torch


def test_forward_preserves_shape():
    """RoPE should preserve input shapes."""
    from bastile.ops.rope import apply_rotary_pos_emb
    
    batch_size = 2
    num_heads = 8
    seq_len = 64
    head_dim = 64
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
    # HuggingFace passes cos/sin with shape (batch, seq_len, head_dim)
    cos = torch.randn(batch_size, seq_len, head_dim, device="cuda")
    sin = torch.randn(batch_size, seq_len, head_dim, device="cuda")
    
    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
    
    assert q_out.shape == q.shape, f"Q shape mismatch: {q_out.shape} vs {q.shape}"
    assert k_out.shape == k.shape, f"K shape mismatch: {k_out.shape} vs {k.shape}"
    print("✓ RoPE preserves shapes")


def test_forward_values():
    """RoPE should correctly apply rotary embeddings."""
    from bastile.ops.rope import apply_rotary_pos_emb, rotate_half
    
    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 32
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
    cos = torch.randn(batch_size, seq_len, head_dim, device="cuda")
    sin = torch.randn(batch_size, seq_len, head_dim, device="cuda")
    
    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
    
    # Manual reference computation
    cos_expanded = cos.unsqueeze(1)  # (batch, 1, seq, head_dim)
    sin_expanded = sin.unsqueeze(1)
    q_ref = (q * cos_expanded) + (rotate_half(q) * sin_expanded)
    k_ref = (k * cos_expanded) + (rotate_half(k) * sin_expanded)
    
    torch.testing.assert_close(q_out, q_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(k_out, k_ref, rtol=1e-5, atol=1e-5)
    print("✓ RoPE forward values are correct")


def test_backward_produces_gradients():
    """RoPE should produce valid gradients."""
    from bastile.ops.rope import apply_rotary_pos_emb
    
    batch_size = 2
    num_heads = 4
    seq_len = 32
    head_dim = 64
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", requires_grad=True)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", requires_grad=True)
    cos = torch.randn(batch_size, seq_len, head_dim, device="cuda")
    sin = torch.randn(batch_size, seq_len, head_dim, device="cuda")
    
    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
    loss = q_out.sum() + k_out.sum()
    loss.backward()
    
    assert q.grad is not None, "No Q gradient"
    assert k.grad is not None, "No K gradient"
    assert torch.isfinite(q.grad).all(), "Non-finite Q gradients"
    assert torch.isfinite(k.grad).all(), "Non-finite K gradients"
    print("✓ RoPE backward produces valid gradients")


def run_all():
    print("=" * 60)
    print("RoPE Tests")
    print("=" * 60)
    test_forward_preserves_shape()
    test_forward_values()
    test_backward_produces_gradients()
    print("✓ All RoPE tests passed\n")


if __name__ == "__main__":
    run_all()
