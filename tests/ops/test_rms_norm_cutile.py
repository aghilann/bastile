"""
Unit tests for cuTILE-native RMSNorm kernel.

Verifies forward and backward passes match PyTorch reference across
multiple dtypes and hidden sizes.
"""

import torch
import pytest


def pytorch_rms_norm(x, weight, eps):
    """PyTorch reference RMSNorm implementation."""
    x_float = x.float()
    variance = x_float.pow(2).mean(-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(variance + eps)
    return (weight.float() * x_normed).to(x.dtype)


# ============================================================================
# Forward Tests
# ============================================================================

@pytest.mark.parametrize("hidden_size", [512, 1024, 2048, 3072, 4096])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_forward_correctness(hidden_size, dtype):
    """Forward output should match PyTorch reference."""
    from bastile.ops.rms_norm_cutile import rms_norm

    batch_size, seq_len = 4, 128
    eps = 1e-6

    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
    weight = torch.ones(hidden_size, device="cuda", dtype=dtype)

    output = rms_norm(x, weight, eps=eps)
    expected = pytorch_rms_norm(x, weight, eps)

    rtol = 1e-2 if dtype != torch.float32 else 1e-3
    atol = 1e-2 if dtype != torch.float32 else 1e-3
    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)


def test_forward_2d_input():
    """Forward should handle 2D input (M, N) directly."""
    from bastile.ops.rms_norm_cutile import rms_norm

    M, N = 256, 2048
    eps = 1e-6

    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(N, device="cuda", dtype=torch.bfloat16)

    output = rms_norm(x, weight, eps=eps)
    expected = pytorch_rms_norm(x, weight, eps)

    torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)


def test_forward_large_batch():
    """Forward should handle large batch sizes."""
    from bastile.ops.rms_norm_cutile import rms_norm

    batch_size, seq_len, hidden_size = 32, 512, 2048
    eps = 1e-6

    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(hidden_size, device="cuda", dtype=torch.bfloat16)

    output = rms_norm(x, weight, eps=eps)
    expected = pytorch_rms_norm(x, weight, eps)

    torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)


# ============================================================================
# Backward Tests
# ============================================================================

@pytest.mark.parametrize("hidden_size", [512, 1024, 2048, 4096])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_backward_correctness(hidden_size, dtype):
    """Backward gradients should match PyTorch reference."""
    from bastile.ops.rms_norm_cutile import rms_norm

    batch_size, seq_len = 2, 64
    eps = 1e-6

    # cuTILE path
    x_ct = torch.randn(
        batch_size, seq_len, hidden_size,
        device="cuda", dtype=dtype, requires_grad=True,
    )
    w_ct = torch.ones(hidden_size, device="cuda", dtype=dtype, requires_grad=True)

    out_ct = rms_norm(x_ct, w_ct, eps=eps)
    loss_ct = out_ct.sum()
    loss_ct.backward()

    assert x_ct.grad is not None, "No gradient for input"
    assert w_ct.grad is not None, "No gradient for weight"
    assert torch.isfinite(x_ct.grad).all(), "Non-finite input gradient"
    assert torch.isfinite(w_ct.grad).all(), "Non-finite weight gradient"

    # PyTorch reference path
    x_ref = x_ct.detach().clone().requires_grad_(True)
    w_ref = w_ct.detach().clone().requires_grad_(True)

    out_ref = pytorch_rms_norm(x_ref, w_ref, eps)
    loss_ref = out_ref.sum()
    loss_ref.backward()

    rtol = 1e-2 if dtype != torch.float32 else 5e-3
    atol = 1e-2 if dtype != torch.float32 else 5e-3
    torch.testing.assert_close(x_ct.grad, x_ref.grad, rtol=rtol, atol=atol)
    torch.testing.assert_close(w_ct.grad, w_ref.grad, rtol=rtol, atol=atol)


def test_backward_weight_gradient_accumulation():
    """Weight gradient should correctly accumulate across all rows."""
    from bastile.ops.rms_norm_cutile import rms_norm

    hidden_size = 1024
    eps = 1e-6

    x = torch.randn(8, 64, hidden_size, device="cuda", dtype=torch.float32, requires_grad=True)
    w = torch.randn(hidden_size, device="cuda", dtype=torch.float32, requires_grad=True)

    out = rms_norm(x, w, eps=eps)
    out.sum().backward()

    # Reference
    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)
    out_ref = pytorch_rms_norm(x_ref, w_ref, eps)
    out_ref.sum().backward()

    torch.testing.assert_close(w.grad, w_ref.grad, rtol=5e-3, atol=5e-3)


# ============================================================================
# Module Tests
# ============================================================================

def test_module_forward():
    """CuTileRMSNorm module should work as drop-in replacement."""
    from bastile.ops.rms_norm_cutile import CuTileRMSNorm

    hidden_size = 2048
    eps = 1e-6

    module = CuTileRMSNorm(hidden_size, eps=eps).cuda()
    x = torch.randn(4, 128, hidden_size, device="cuda", dtype=torch.bfloat16)

    output = module(x)
    expected = pytorch_rms_norm(x, module.weight, eps)

    torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)


def test_module_backward():
    """CuTileRMSNorm module backward should produce valid gradients."""
    from bastile.ops.rms_norm_cutile import CuTileRMSNorm

    hidden_size = 2048
    eps = 1e-6

    module = CuTileRMSNorm(hidden_size, eps=eps).cuda()
    x = torch.randn(2, 64, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    output = module(x)
    output.sum().backward()

    assert x.grad is not None
    assert module.weight.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(module.weight.grad).all()


# ============================================================================
# Simple runner (for standalone execution)
# ============================================================================

def run_all():
    print("=" * 60)
    print("cuTILE RMSNorm Tests")
    print("=" * 60)

    for hs in [512, 1024, 2048, 4096]:
        for dt in [torch.float32, torch.bfloat16]:
            test_forward_correctness(hs, dt)
            print(f"  Forward  {dt} hidden={hs} OK")

    test_forward_2d_input()
    print("  Forward 2D input OK")

    test_forward_large_batch()
    print("  Forward large batch OK")

    for hs in [512, 1024, 2048, 4096]:
        for dt in [torch.float32, torch.bfloat16]:
            test_backward_correctness(hs, dt)
            print(f"  Backward {dt} hidden={hs} OK")

    test_backward_weight_gradient_accumulation()
    print("  Backward weight gradient accumulation OK")

    test_module_forward()
    print("  Module forward OK")

    test_module_backward()
    print("  Module backward OK")

    print("All cuTILE RMSNorm tests passed\n")


if __name__ == "__main__":
    run_all()
