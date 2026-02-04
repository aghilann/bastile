"""
Unit tests for CuTile SwiGLU kernel.

Verifies forward and backward passes match PyTorch reference.
"""

import torch


def test_forward_matches_pytorch():
    """SwiGLU forward should match PyTorch implementation."""
    from bastile.ops.swiglu import SwiGLUFunction
    
    batch_size = 4
    seq_len = 64
    hidden_size = 512
    
    gate = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float32)
    up = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float32)
    
    # PyTorch reference: SiLU(gate) * up
    output_ref = torch.nn.functional.silu(gate) * up
    
    # CuTile implementation
    output_cutile = SwiGLUFunction.apply(gate, up)
    
    # Compare
    torch.testing.assert_close(output_cutile, output_ref, rtol=1e-3, atol=1e-3)
    print("✓ SwiGLU forward matches PyTorch")


def test_backward_matches_pytorch():
    """SwiGLU backward should match PyTorch gradients."""
    from bastile.ops.swiglu import SwiGLUFunction
    
    batch_size = 2
    seq_len = 32
    hidden_size = 256
    
    # CuTile path
    gate_cutile = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float32, requires_grad=True)
    up_cutile = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float32, requires_grad=True)
    
    output_cutile = SwiGLUFunction.apply(gate_cutile, up_cutile)
    loss_cutile = output_cutile.sum()
    loss_cutile.backward()
    
    # PyTorch reference path
    gate_ref = gate_cutile.detach().clone().requires_grad_(True)
    up_ref = up_cutile.detach().clone().requires_grad_(True)
    
    output_ref = torch.nn.functional.silu(gate_ref) * up_ref
    loss_ref = output_ref.sum()
    loss_ref.backward()
    
    # Compare gradients
    torch.testing.assert_close(gate_cutile.grad, gate_ref.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(up_cutile.grad, up_ref.grad, rtol=1e-2, atol=1e-2)
    print("✓ SwiGLU backward matches PyTorch")


def run_all():
    print("=" * 60)
    print("SwiGLU Tests")
    print("=" * 60)
    test_forward_matches_pytorch()
    test_backward_matches_pytorch()
    print("✓ All SwiGLU tests passed\n")


if __name__ == "__main__":
    run_all()
