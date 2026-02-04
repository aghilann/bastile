"""
Unit tests for CuTile RMSNorm kernel.

Verifies forward and backward passes match PyTorch reference.
"""

import torch


def test_forward_matches_pytorch():
    """RMSNorm forward should match PyTorch implementation."""
    from bastile.ops.rms_norm import CuTileRMSNorm
    
    hidden_size = 1024
    eps = 1e-6
    batch_size = 4
    seq_len = 128
    
    # Create input
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float32)
    weight = torch.ones(hidden_size, device="cuda", dtype=torch.float32)
    
    # PyTorch reference
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed_ref = x * torch.rsqrt(variance + eps)
    output_ref = weight * x_normed_ref
    
    # CuTile implementation
    output_cutile = CuTileRMSNorm.apply(x, weight, eps)
    
    # Compare
    torch.testing.assert_close(output_cutile, output_ref, rtol=1e-3, atol=1e-3)
    print("✓ RMSNorm forward matches PyTorch")


def test_backward_matches_pytorch():
    """RMSNorm backward should produce valid gradients."""
    from bastile.ops.rms_norm import CuTileRMSNorm
    
    hidden_size = 512
    eps = 1e-6
    batch_size = 2
    seq_len = 64
    
    # CuTile path
    x_cutile = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float32, requires_grad=True)
    weight_cutile = torch.ones(hidden_size, device="cuda", dtype=torch.float32, requires_grad=True)
    
    output_cutile = CuTileRMSNorm.apply(x_cutile, weight_cutile, eps)
    loss_cutile = output_cutile.sum()
    loss_cutile.backward()
    
    # Check gradients exist and are finite
    assert x_cutile.grad is not None, "No gradient for input"
    assert weight_cutile.grad is not None, "No gradient for weight"
    assert torch.isfinite(x_cutile.grad).all(), "Non-finite input gradient"
    assert torch.isfinite(weight_cutile.grad).all(), "Non-finite weight gradient"
    
    # PyTorch reference path
    x_ref = x_cutile.detach().clone().requires_grad_(True)
    weight_ref = weight_cutile.detach().clone().requires_grad_(True)
    
    variance = x_ref.pow(2).mean(-1, keepdim=True)
    x_normed = x_ref * torch.rsqrt(variance + eps)
    output_ref = weight_ref * x_normed
    loss_ref = output_ref.sum()
    loss_ref.backward()
    
    # Compare gradients
    torch.testing.assert_close(x_cutile.grad, x_ref.grad, rtol=1e-2, atol=1e-2)
    print("✓ RMSNorm backward matches PyTorch")


def run_all():
    print("=" * 60)
    print("RMSNorm Tests")
    print("=" * 60)
    test_forward_matches_pytorch()
    test_backward_matches_pytorch()
    print("✓ All RMSNorm tests passed\n")


if __name__ == "__main__":
    run_all()
