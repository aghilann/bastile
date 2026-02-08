"""
Unit tests for CuTile fused cross-entropy kernel.

Verifies forward loss matches PyTorch and backward produces correct gradients.
"""

import torch


def test_forward_matches_pytorch():
    """Cross-entropy forward should match PyTorch nn.functional.cross_entropy."""
    from bastile.ops.cross_entropy import cutile_fixed_cross_entropy

    BT = 256
    V = 1024

    logits = torch.randn(BT, V, device="cuda", dtype=torch.float32)
    targets = torch.randint(0, V, (BT,), device="cuda")

    # PyTorch reference
    loss_ref = torch.nn.functional.cross_entropy(logits, targets, reduction="mean")

    # CuTile implementation (clone because it modifies logits in-place)
    loss_cutile = cutile_fixed_cross_entropy(logits.clone(), targets)

    torch.testing.assert_close(loss_cutile, loss_ref, rtol=1e-4, atol=1e-4)
    print("OK Cross-entropy forward matches PyTorch")


def test_forward_with_ignore_index():
    """Cross-entropy should correctly handle ignore_index tokens."""
    from bastile.ops.cross_entropy import cutile_fixed_cross_entropy

    BT = 128
    V = 512

    logits = torch.randn(BT, V, device="cuda", dtype=torch.float32)
    targets = torch.randint(0, V, (BT,), device="cuda")
    # Set ~25% of targets to ignore_index
    targets[::4] = -100

    loss_ref = torch.nn.functional.cross_entropy(
        logits, targets, ignore_index=-100, reduction="mean"
    )
    loss_cutile = cutile_fixed_cross_entropy(logits.clone(), targets, ignore_index=-100)

    torch.testing.assert_close(loss_cutile, loss_ref, rtol=1e-4, atol=1e-4)
    print("OK Cross-entropy with ignore_index matches PyTorch")


def test_forward_with_num_items_in_batch():
    """Cross-entropy with num_items_in_batch (sum reduction / N)."""
    from bastile.ops.cross_entropy import cutile_fixed_cross_entropy

    BT = 64
    V = 256

    logits = torch.randn(BT, V, device="cuda", dtype=torch.float32)
    targets = torch.randint(0, V, (BT,), device="cuda")
    num_items = torch.tensor(32.0, device="cuda")

    loss_ref = (
        torch.nn.functional.cross_entropy(logits, targets, reduction="sum") / num_items
    )
    loss_cutile = cutile_fixed_cross_entropy(
        logits.clone(), targets, num_items_in_batch=num_items
    )

    torch.testing.assert_close(loss_cutile, loss_ref, rtol=1e-4, atol=1e-4)
    print("OK Cross-entropy with num_items_in_batch matches PyTorch")


def test_backward_matches_pytorch():
    """Cross-entropy backward should match PyTorch gradients."""
    from bastile.ops.cross_entropy import cutile_fixed_cross_entropy

    BT = 128
    V = 512

    # PyTorch reference
    logits_ref = torch.randn(BT, V, device="cuda", dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, V, (BT,), device="cuda")

    loss_ref = torch.nn.functional.cross_entropy(logits_ref, targets, reduction="mean")
    loss_ref.backward()

    # CuTile path (use same logit values)
    logits_cutile = logits_ref.detach().clone().requires_grad_(True)
    loss_cutile = cutile_fixed_cross_entropy(logits_cutile, targets)
    loss_cutile.backward()

    torch.testing.assert_close(logits_cutile.grad, logits_ref.grad, rtol=1e-3, atol=1e-3)
    print("OK Cross-entropy backward matches PyTorch")


def test_backward_with_ignore_index():
    """Cross-entropy backward with ignore_index should match PyTorch."""
    from bastile.ops.cross_entropy import cutile_fixed_cross_entropy

    BT = 128
    V = 512

    logits_ref = torch.randn(BT, V, device="cuda", dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, V, (BT,), device="cuda")
    targets[::3] = -100

    loss_ref = torch.nn.functional.cross_entropy(
        logits_ref, targets, ignore_index=-100, reduction="mean"
    )
    loss_ref.backward()

    logits_cutile = logits_ref.detach().clone().requires_grad_(True)
    loss_cutile = cutile_fixed_cross_entropy(logits_cutile, targets, ignore_index=-100)
    loss_cutile.backward()

    torch.testing.assert_close(logits_cutile.grad, logits_ref.grad, rtol=1e-3, atol=1e-3)
    print("OK Cross-entropy backward with ignore_index matches PyTorch")


def test_large_vocab():
    """Test with large vocabulary (realistic LLM size)."""
    from bastile.ops.cross_entropy import cutile_fixed_cross_entropy

    BT = 64
    V = 32000

    logits = torch.randn(BT, V, device="cuda", dtype=torch.float32)
    targets = torch.randint(0, V, (BT,), device="cuda")

    loss_ref = torch.nn.functional.cross_entropy(logits, targets, reduction="mean")
    loss_cutile = cutile_fixed_cross_entropy(logits.clone(), targets)

    torch.testing.assert_close(loss_cutile, loss_ref, rtol=1e-3, atol=1e-3)
    print("OK Cross-entropy with large vocab (32000) matches PyTorch")


def run_all():
    print("=" * 60)
    print("Cross-Entropy Tests")
    print("=" * 60)
    test_forward_matches_pytorch()
    test_forward_with_ignore_index()
    test_forward_with_num_items_in_batch()
    test_backward_matches_pytorch()
    test_backward_with_ignore_index()
    test_large_vocab()
    print("OK All cross-entropy tests passed\n")


if __name__ == "__main__":
    run_all()
