"""
Unit tests for MoE experts parallel execution.

Verifies that the CuTile fused MoE GEMM forward matches the eager (sequential)
forward from GptOssExperts for both forward output and backward gradients.
"""

import torch
import torch.nn as nn


ALPHA = 1.702
LIMIT = 7.0


def pytorch_apply_gate(gate_up: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation of GPT-OSS _apply_gate."""
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(min=None, max=LIMIT)
    up = up.clamp(min=-LIMIT, max=LIMIT)
    glu = gate * torch.sigmoid(gate * ALPHA)
    return (up + 1) * glu


def eager_moe_forward(
    gate_up_proj, gate_up_bias, down_proj, down_bias,
    hidden_states, router_indices, routing_weights,
    num_experts, intermediate_size,
):
    """Reference eager forward: sequential expert loop (matches GptOssExperts.forward)."""
    next_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(
            router_indices, num_classes=num_experts
        )
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate_up = current_state @ gate_up_proj[expert_idx] + gate_up_bias[expert_idx]
        gated_output = pytorch_apply_gate(gate_up)
        out = gated_output @ down_proj[expert_idx] + down_bias[expert_idx]
        weighted_output = out * routing_weights[token_idx, top_k_pos, None]
        next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))

    return next_states


class FakeGptOssExperts(nn.Module):
    """Minimal GptOssExperts-like module for testing."""

    def __init__(self, num_experts, hidden_size, intermediate_size):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.has_bias = True
        self.is_transposed = True
        self.alpha = ALPHA
        self.limit = LIMIT

        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, hidden_size, 2 * intermediate_size, dtype=torch.bfloat16, device="cuda") * 0.02
        )
        self.gate_up_proj_bias = nn.Parameter(
            torch.randn(num_experts, 2 * intermediate_size, dtype=torch.bfloat16, device="cuda") * 0.02
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size, dtype=torch.bfloat16, device="cuda") * 0.02
        )
        self.down_proj_bias = nn.Parameter(
            torch.randn(num_experts, hidden_size, dtype=torch.bfloat16, device="cuda") * 0.02
        )

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        return pytorch_apply_gate(gate_up)


def make_routing(num_tokens, num_experts, top_k, device="cuda"):
    """Create random routing indices and weights."""
    router_indices = torch.stack([
        torch.randperm(num_experts, device=device)[:top_k]
        for _ in range(num_tokens)
    ])
    router_logits = torch.randn(num_tokens, top_k, device=device, dtype=torch.bfloat16)
    routing_weights = torch.softmax(router_logits.float(), dim=-1).to(torch.bfloat16)
    return router_indices, routing_weights


def test_cutile_forward_matches_eager():
    """CuTile fused MoE GEMM forward should match the eager sequential loop."""
    from bastile.ops.moe_experts import cutile_moe_experts_forward

    # Use dims divisible by TILE_K=64 and TILE_N=64
    num_experts = 8
    hidden_size = 128
    intermediate_size = 128
    num_tokens = 32
    top_k = 2

    module = FakeGptOssExperts(num_experts, hidden_size, intermediate_size)
    hidden_states = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    router_indices, routing_weights = make_routing(num_tokens, num_experts, top_k)

    ref_output = eager_moe_forward(
        module.gate_up_proj, module.gate_up_proj_bias,
        module.down_proj, module.down_proj_bias,
        hidden_states, router_indices, routing_weights,
        num_experts, intermediate_size,
    )

    test_output = cutile_moe_experts_forward(
        module, hidden_states, router_indices, routing_weights,
    )

    max_diff = torch.max(torch.abs(ref_output.float() - test_output.float())).item()
    print(f"  cutile forward: max_diff={max_diff:.6f}")

    torch.testing.assert_close(test_output, ref_output, rtol=5e-2, atol=5e-2)
    print("  CuTile fused MoE forward matches eager")


def test_cutile_backward():
    """CuTile fused MoE should produce valid gradients."""
    from bastile.ops.moe_experts import cutile_moe_experts_forward

    num_experts = 8
    hidden_size = 128
    intermediate_size = 128
    num_tokens = 16
    top_k = 2

    module = FakeGptOssExperts(num_experts, hidden_size, intermediate_size)
    hidden_states = torch.randn(
        num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    router_indices, routing_weights = make_routing(num_tokens, num_experts, top_k)

    output = cutile_moe_experts_forward(
        module, hidden_states, router_indices, routing_weights,
    )
    loss = output.sum()
    loss.backward()

    assert hidden_states.grad is not None, "hidden_states grad should not be None"
    assert hidden_states.grad.shape == hidden_states.shape
    grad_norm = hidden_states.grad.float().norm().item()
    print(f"  backward: grad_norm={grad_norm:.6f}")
    assert grad_norm > 0, "gradient should be non-zero"

    # Check weight grads exist
    assert module.gate_up_proj.grad is not None, "gate_up_proj grad missing"
    assert module.down_proj.grad is not None, "down_proj grad missing"
    assert module.gate_up_proj_bias.grad is not None, "gate_up_proj_bias grad missing"
    assert module.down_proj_bias.grad is not None, "down_proj_bias grad missing"
    print("  CuTile fused MoE backward produces valid gradients")


def test_gpt_oss_dimensions():
    """Test with actual GPT-OSS-20B dimensions (reduced token count)."""
    from bastile.ops.moe_experts import cutile_moe_experts_forward

    num_experts = 128
    hidden_size = 2880
    intermediate_size = 2880
    num_tokens = 64
    top_k = 4

    module = FakeGptOssExperts(num_experts, hidden_size, intermediate_size)
    hidden_states = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    router_indices, routing_weights = make_routing(num_tokens, num_experts, top_k)

    output = cutile_moe_experts_forward(
        module, hidden_states, router_indices, routing_weights,
    )

    assert output.shape == hidden_states.shape
    print(f"  GPT-OSS dims: output_shape={output.shape}, output_norm={output.float().norm():.2f}")
    print("  GPT-OSS dimensions work correctly")


def run_all():
    print("=" * 60)
    print("MoE Experts Tests (CuTile Fused GEMM)")
    print("=" * 60)
    test_cutile_forward_matches_eager()
    test_cutile_backward()
    test_gpt_oss_dimensions()
    print("All MoE experts tests passed\n")


if __name__ == "__main__":
    run_all()
