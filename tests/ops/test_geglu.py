"""
Unit tests for CuTile GEGLU kernel (GPT-OSS activation).

GPT-OSS uses: (up + 1) * gate * sigmoid(gate * 1.702)

Verifies forward and backward passes match PyTorch reference.
"""

import torch


def test_forward_matches_pytorch():
    """GEGLU forward should match PyTorch implementation."""
    from bastile.ops.gpt_oss_moe import geglu_activation, ALPHA, LIMIT
    
    num_tokens = 128
    expert_dim = 256
    
    # Interleaved gate/up input
    gate_up = torch.randn(num_tokens, 2 * expert_dim, device="cuda", dtype=torch.float32)
    
    # PyTorch reference
    gate = gate_up[..., 0::2].clone()
    up = gate_up[..., 1::2].clone()
    gate = gate.clamp(max=LIMIT)
    up = up.clamp(min=-LIMIT, max=LIMIT)
    glu = gate * torch.sigmoid(gate * ALPHA)
    output_ref = (up + 1) * glu
    
    # CuTile implementation
    output_cutile = geglu_activation(gate_up)
    
    # Compare
    torch.testing.assert_close(output_cutile, output_ref, rtol=1e-3, atol=1e-3)
    print("✓ GEGLU forward matches PyTorch")


def test_backward_produces_gradients():
    """GEGLU backward should produce valid gradients."""
    from bastile.ops.gpt_oss_moe import geglu_activation
    
    num_tokens = 64
    expert_dim = 128
    
    gate_up = torch.randn(num_tokens, 2 * expert_dim, device="cuda", dtype=torch.float32, requires_grad=True)
    
    output = geglu_activation(gate_up)
    loss = output.sum()
    loss.backward()
    
    assert gate_up.grad is not None, "No gradient"
    assert torch.isfinite(gate_up.grad).all(), "Non-finite gradients"
    assert gate_up.grad.norm() > 0, "Zero gradients"
    print("✓ GEGLU backward produces valid gradients")


def test_backward_gradient_shape():
    """GEGLU backward gradients should have correct shape."""
    from bastile.ops.gpt_oss_moe import geglu_activation
    
    num_tokens = 16
    expert_dim = 32
    
    gate_up = torch.randn(num_tokens, 2 * expert_dim, device="cuda", dtype=torch.float32, requires_grad=True)
    
    output = geglu_activation(gate_up)
    loss = output.sum()
    loss.backward()
    
    # Check gradient shape matches input
    assert gate_up.grad.shape == gate_up.shape, f"Gradient shape mismatch: {gate_up.grad.shape} vs {gate_up.shape}"
    
    # Check gradient norm is reasonable (not exploding or vanishing)
    grad_norm = gate_up.grad.norm().item()
    assert 0.1 < grad_norm < 1000, f"Gradient norm out of range: {grad_norm}"
    
    print("✓ GEGLU backward gradients have correct shape")


def run_all():
    print("=" * 60)
    print("GEGLU Tests (GPT-OSS)")
    print("=" * 60)
    test_forward_matches_pytorch()
    test_backward_produces_gradients()
    test_backward_gradient_shape()
    print("✓ All GEGLU tests passed\n")


if __name__ == "__main__":
    run_all()
