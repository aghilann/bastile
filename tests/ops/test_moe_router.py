"""Test fused MoE router and load-balancing loss correctness."""

import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, "/workspace/bastile/src")


def reference_load_balancing_loss(gate_logits, num_experts, top_k):
    """Reference implementation matching HuggingFace's load_balancing_loss_func."""
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat(
        [layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0
    )

    routing_weights = F.softmax(concatenated_gate_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = F.one_hot(selected_experts, num_experts)

    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
    router_prob_per_expert = torch.mean(routing_weights, dim=0)

    loss = num_experts * torch.sum(tokens_per_expert * router_prob_per_expert)
    return loss


def test_fused_load_balancing_loss():
    """Test that fused implementation matches reference."""
    from bastile.ops.moe_router import fused_load_balancing_loss

    torch.manual_seed(42)
    num_experts = 128
    top_k = 4
    num_tokens = 4096
    num_layers = 4

    # Create random router logits for each layer
    gate_logits = tuple(
        torch.randn(num_tokens, num_experts, device="cuda", dtype=torch.bfloat16)
        for _ in range(num_layers)
    )

    # Reference
    ref_loss = reference_load_balancing_loss(gate_logits, num_experts, top_k)

    # Fused
    fused_loss = fused_load_balancing_loss(gate_logits, num_experts, top_k)

    ref_val = ref_loss.float().item()
    fused_val = fused_loss.float().item()

    print(f"Reference loss: {ref_val:.6f}")
    print(f"Fused loss:     {fused_val:.6f}")
    print(f"Abs diff:       {abs(ref_val - fused_val):.6f}")
    print(f"Rel diff:       {abs(ref_val - fused_val) / max(abs(ref_val), 1e-8):.4%}")

    # The fused kernel uses atomic_add which can have minor floating point
    # ordering differences, so we use a relaxed tolerance
    assert abs(ref_val - fused_val) / max(abs(ref_val), 1e-8) < 0.05, \
        f"Loss mismatch: ref={ref_val}, fused={fused_val}"
    print("PASSED: fused load-balancing loss matches reference\n")


def test_fused_router_forward():
    """Test that fused router forward matches reference."""
    from bastile.ops.moe_router import FusedRouterFunction

    torch.manual_seed(42)
    num_experts = 128
    top_k = 4
    hidden_size = 2880
    num_tokens = 4096

    hidden_states = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(num_experts, hidden_size, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(num_experts, device="cuda", dtype=torch.bfloat16)

    # Reference: standard router
    ref_logits = F.linear(hidden_states, weight, bias)
    ref_top_values, ref_indices = torch.topk(ref_logits, top_k, dim=-1)
    ref_scores = F.softmax(ref_top_values, dim=-1, dtype=ref_top_values.dtype)

    # Fused
    fused_logits, fused_scores, fused_indices = FusedRouterFunction.apply(
        hidden_states, weight, bias, top_k
    )

    # Check logits match exactly (same linear)
    logits_match = torch.allclose(ref_logits, fused_logits, atol=1e-3, rtol=1e-2)
    print(f"Logits match:  {logits_match}")
    print(f"  max diff: {(ref_logits - fused_logits).abs().max().item():.6f}")

    # Check indices match
    indices_match = torch.equal(ref_indices, fused_indices)
    print(f"Indices match: {indices_match}")

    # Check scores match
    scores_match = torch.allclose(ref_scores, fused_scores, atol=1e-3, rtol=1e-2)
    print(f"Scores match:  {scores_match}")
    print(f"  max diff: {(ref_scores - fused_scores).abs().max().item():.6f}")

    assert logits_match and indices_match and scores_match, "Router forward mismatch"
    print("PASSED: fused router forward matches reference\n")


def test_fused_router_backward():
    """Test that fused router backward produces valid gradients."""
    from bastile.ops.moe_router import FusedRouterFunction

    torch.manual_seed(42)
    num_experts = 128
    top_k = 4
    hidden_size = 2880
    num_tokens = 512  # smaller for backward test

    hidden_states = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn(num_experts, hidden_size, device="cuda", dtype=torch.float32, requires_grad=True)
    bias = torch.randn(num_experts, device="cuda", dtype=torch.float32, requires_grad=True)

    # Reference backward
    ref_h = hidden_states.detach().clone().requires_grad_(True)
    ref_w = weight.detach().clone().requires_grad_(True)
    ref_b = bias.detach().clone().requires_grad_(True)
    ref_logits = F.linear(ref_h, ref_w, ref_b)
    ref_top_values, ref_indices = torch.topk(ref_logits, top_k, dim=-1)
    ref_scores = F.softmax(ref_top_values, dim=-1)
    ref_loss = ref_scores.sum() + ref_logits.mean()
    ref_loss.backward()

    # Fused backward
    fused_h = hidden_states.detach().clone().requires_grad_(True)
    fused_w = weight.detach().clone().requires_grad_(True)
    fused_b = bias.detach().clone().requires_grad_(True)
    fused_logits, fused_scores, fused_indices = FusedRouterFunction.apply(
        fused_h, fused_w, fused_b, top_k
    )
    fused_loss = fused_scores.sum() + fused_logits.mean()
    fused_loss.backward()

    # Compare gradients
    h_match = torch.allclose(ref_h.grad, fused_h.grad, atol=1e-3, rtol=1e-2)
    w_match = torch.allclose(ref_w.grad, fused_w.grad, atol=1e-3, rtol=1e-2)
    b_match = torch.allclose(ref_b.grad, fused_b.grad, atol=1e-3, rtol=1e-2)

    print(f"d_hidden match: {h_match}  (max diff: {(ref_h.grad - fused_h.grad).abs().max().item():.6f})")
    print(f"d_weight match: {w_match}  (max diff: {(ref_w.grad - fused_w.grad).abs().max().item():.6f})")
    print(f"d_bias match:   {b_match}  (max diff: {(ref_b.grad - fused_b.grad).abs().max().item():.6f})")

    assert h_match and w_match and b_match, "Router backward mismatch"
    print("PASSED: fused router backward matches reference\n")


if __name__ == "__main__":
    print("=" * 60)
    print("  Testing Fused MoE Router")
    print("=" * 60 + "\n")

    test_fused_router_forward()
    test_fused_router_backward()
    test_fused_load_balancing_loss()

    print("All tests passed!")
