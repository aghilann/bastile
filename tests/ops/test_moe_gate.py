"""
Unit tests for CuTile MoE expert gate kernel.

Verifies forward and backward passes match the PyTorch reference
implementation from GptOssExperts._apply_gate:
  gate, up = gate_up[..., ::2], gate_up[..., 1::2]
  gate = gate.clamp(max=7.0)
  up = up.clamp(-7.0, 7.0)
  glu = gate * sigmoid(gate * 1.702)
  output = (up + 1) * glu
"""

import torch


ALPHA = 1.702
LIMIT = 7.0


def pytorch_moe_gate(gate_up: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation of GPT-OSS _apply_gate."""
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(min=None, max=LIMIT)
    up = up.clamp(min=-LIMIT, max=LIMIT)
    glu = gate * torch.sigmoid(gate * ALPHA)
    return (up + 1) * glu


def test_forward_matches_pytorch():
    """MoE gate forward should match PyTorch implementation."""
    from bastile.ops.moe_gate import MoEGateFunction

    # Test with GPT-OSS dimensions: intermediate_size=2880
    for n_cols in [256, 2880]:
        gate_up = torch.randn(32, 2 * n_cols, device="cuda", dtype=torch.bfloat16)

        output_ref = pytorch_moe_gate(gate_up.float()).to(torch.bfloat16)
        output_cutile = MoEGateFunction.apply(gate_up)

        torch.testing.assert_close(output_cutile, output_ref, rtol=1e-2, atol=1e-2)
        print(f"  forward n_cols={n_cols}: max_diff={torch.max(torch.abs(output_cutile.float() - output_ref.float())):.6f}")

    print("  forward matches PyTorch")


def test_forward_edge_cases():
    """Test clamping behavior at boundaries."""
    from bastile.ops.moe_gate import MoEGateFunction

    n_cols = 128
    # Create values that exercise clamping
    gate_up = torch.zeros(4, 2 * n_cols, device="cuda", dtype=torch.bfloat16)
    # Set gate values (even indices) to exceed 7.0
    gate_up[:, ::2] = 10.0
    # Set up values (odd indices) to exceed boundaries
    gate_up[:, 1::2] = -10.0

    output_ref = pytorch_moe_gate(gate_up.float()).to(torch.bfloat16)
    output_cutile = MoEGateFunction.apply(gate_up)

    torch.testing.assert_close(output_cutile, output_ref, rtol=1e-2, atol=1e-2)
    print("  edge cases (clamping) match PyTorch")


def test_backward_matches_pytorch():
    """MoE gate backward should match PyTorch gradients."""
    from bastile.ops.moe_gate import MoEGateFunction

    n_cols = 256

    # CuTile path
    gate_up_cutile = torch.randn(16, 2 * n_cols, device="cuda", dtype=torch.float32, requires_grad=True)
    output_cutile = MoEGateFunction.apply(gate_up_cutile)
    loss_cutile = output_cutile.sum()
    loss_cutile.backward()

    # PyTorch reference path
    gate_up_ref = gate_up_cutile.detach().clone().requires_grad_(True)
    output_ref = pytorch_moe_gate(gate_up_ref)
    loss_ref = output_ref.sum()
    loss_ref.backward()

    torch.testing.assert_close(gate_up_cutile.grad, gate_up_ref.grad, rtol=1e-2, atol=1e-2)
    max_diff = torch.max(torch.abs(gate_up_cutile.grad - gate_up_ref.grad))
    print(f"  backward: max_grad_diff={max_diff:.6f}")
    print("  backward matches PyTorch")


def test_backward_bf16():
    """MoE gate backward with bfloat16 inputs."""
    from bastile.ops.moe_gate import MoEGateFunction

    n_cols = 2880

    # CuTile path (bf16)
    gate_up_cutile = torch.randn(8, 2 * n_cols, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    output_cutile = MoEGateFunction.apply(gate_up_cutile)
    loss_cutile = output_cutile.sum()
    loss_cutile.backward()

    # PyTorch reference path (bf16)
    gate_up_ref = gate_up_cutile.detach().clone().requires_grad_(True)
    output_ref = pytorch_moe_gate(gate_up_ref)
    loss_ref = output_ref.sum()
    loss_ref.backward()

    torch.testing.assert_close(gate_up_cutile.grad, gate_up_ref.grad, rtol=5e-2, atol=5e-2)
    max_diff = torch.max(torch.abs(gate_up_cutile.grad.float() - gate_up_ref.grad.float()))
    print(f"  backward bf16: max_grad_diff={max_diff:.6f}")
    print("  backward bf16 matches PyTorch")


def run_all():
    print("=" * 60)
    print("MoE Expert Gate Tests")
    print("=" * 60)
    test_forward_matches_pytorch()
    test_forward_edge_cases()
    test_backward_matches_pytorch()
    test_backward_bf16()
    print("All MoE gate tests passed\n")


if __name__ == "__main__":
    run_all()
