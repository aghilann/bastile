"""
Unit tests for CuTile Fused Linear Cross-Entropy kernel.

Verifies forward loss and backward gradients match PyTorch reference
(F.linear + F.cross_entropy).
"""

import torch
import torch.nn.functional as F


def _pytorch_reference(x, weight, target, ignore_index=-100, reduction="mean"):
    """PyTorch baseline: materialise full logits, then cross-entropy."""
    logits = F.linear(x, weight)
    return F.cross_entropy(logits, target, ignore_index=ignore_index, reduction=reduction)


def test_forward_loss_matches_pytorch():
    """Fused LCE forward loss should match PyTorch F.linear + F.cross_entropy."""
    from bastile.ops.fused_linear_cross_entropy import fused_linear_cross_entropy

    H, V = 256, 1024
    BT = 32
    torch.manual_seed(42)

    x = torch.randn(BT, H, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn(V, H, device="cuda", dtype=torch.float32, requires_grad=True)
    target = torch.randint(0, V, (BT,), device="cuda")

    loss_ref = _pytorch_reference(x.detach().requires_grad_(True), weight.detach().requires_grad_(True), target)
    loss_cutile = fused_linear_cross_entropy(x, weight, target)

    torch.testing.assert_close(loss_cutile, loss_ref, rtol=1e-3, atol=1e-3)
    print("  forward loss matches PyTorch (float32)")


def test_forward_loss_bf16():
    """Fused LCE forward loss should match PyTorch in bfloat16."""
    from bastile.ops.fused_linear_cross_entropy import fused_linear_cross_entropy

    H, V = 256, 1024
    BT = 64
    torch.manual_seed(42)

    x = torch.randn(BT, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(V, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(0, V, (BT,), device="cuda")

    loss_ref = _pytorch_reference(
        x.detach().float().requires_grad_(True), weight.detach().float().requires_grad_(True), target
    )
    loss_cutile = fused_linear_cross_entropy(x, weight, target)

    torch.testing.assert_close(loss_cutile.float(), loss_ref, rtol=5e-2, atol=5e-2)
    print("  forward loss matches PyTorch (bfloat16)")


def test_backward_gradients():
    """Fused LCE backward should produce gradients close to PyTorch."""
    from bastile.ops.fused_linear_cross_entropy import fused_linear_cross_entropy

    H, V = 128, 512
    BT = 16
    torch.manual_seed(42)

    # PyTorch reference
    x_ref = torch.randn(BT, H, device="cuda", dtype=torch.float32, requires_grad=True)
    w_ref = torch.randn(V, H, device="cuda", dtype=torch.float32, requires_grad=True)
    target = torch.randint(0, V, (BT,), device="cuda")

    loss_ref = _pytorch_reference(x_ref, w_ref, target)
    loss_ref.backward()

    # CuTile path
    x_ct = x_ref.detach().clone().requires_grad_(True)
    w_ct = w_ref.detach().clone().requires_grad_(True)

    loss_ct = fused_linear_cross_entropy(x_ct, w_ct, target)
    loss_ct.backward()

    assert x_ct.grad is not None, "No gradient for hidden_states"
    assert w_ct.grad is not None, "No gradient for weight"
    assert torch.isfinite(x_ct.grad).all(), "Non-finite hidden_states gradient"
    assert torch.isfinite(w_ct.grad).all(), "Non-finite weight gradient"

    torch.testing.assert_close(x_ct.grad, x_ref.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(w_ct.grad, w_ref.grad, rtol=1e-2, atol=1e-2)
    print("  backward gradients match PyTorch (float32)")


def test_ignore_index():
    """Fused LCE should correctly ignore target tokens with ignore_index."""
    from bastile.ops.fused_linear_cross_entropy import fused_linear_cross_entropy

    H, V = 128, 512
    BT = 32
    torch.manual_seed(42)

    x = torch.randn(BT, H, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn(V, H, device="cuda", dtype=torch.float32, requires_grad=True)
    target = torch.randint(0, V, (BT,), device="cuda")
    target[::3] = -100  # Mark every 3rd token as ignored

    loss_ref = _pytorch_reference(
        x.detach().requires_grad_(True),
        weight.detach().requires_grad_(True),
        target,
        ignore_index=-100,
    )
    loss_ct = fused_linear_cross_entropy(x, weight, target, ignore_index=-100)

    torch.testing.assert_close(loss_ct, loss_ref, rtol=1e-3, atol=1e-3)
    print("  ignore_index handled correctly")


def test_3d_input():
    """Fused LCE should handle 3D (B, T, H) input correctly."""
    from bastile.ops.fused_linear_cross_entropy import fused_linear_cross_entropy

    H, V = 128, 512
    B, T = 2, 16
    torch.manual_seed(42)

    x = torch.randn(B, T, H, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn(V, H, device="cuda", dtype=torch.float32, requires_grad=True)
    target = torch.randint(0, V, (B, T), device="cuda")

    loss_ref = _pytorch_reference(
        x.detach().view(-1, H).requires_grad_(True),
        weight.detach().requires_grad_(True),
        target.view(-1),
    )
    loss_ct = fused_linear_cross_entropy(x, weight, target)

    torch.testing.assert_close(loss_ct, loss_ref, rtol=1e-3, atol=1e-3)
    print("  3D input (B, T, H) handled correctly")


def test_v_chunked_mode():
    """V-chunked (memory-efficient) mode should match PyTorch."""
    from bastile.ops.fused_linear_cross_entropy import fused_linear_cross_entropy

    H, V = 128, 512
    BT = 32
    torch.manual_seed(42)

    x = torch.randn(BT, H, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn(V, H, device="cuda", dtype=torch.float32, requires_grad=True)
    target = torch.randint(0, V, (BT,), device="cuda")

    loss_ref = _pytorch_reference(
        x.detach().requires_grad_(True),
        weight.detach().requires_grad_(True),
        target,
    )
    loss_ct = fused_linear_cross_entropy(x, weight, target, v_chunk_size=128)

    torch.testing.assert_close(loss_ct, loss_ref, rtol=1e-2, atol=1e-2)
    print("  V-chunked mode matches PyTorch")


def run_all():
    print("=" * 60)
    print("Fused Linear Cross-Entropy Tests")
    print("=" * 60)
    test_forward_loss_matches_pytorch()
    test_forward_loss_bf16()
    test_backward_gradients()
    test_ignore_index()
    test_3d_input()
    test_v_chunked_mode()
    print("All Fused LCE tests passed\n")


if __name__ == "__main__":
    run_all()
