"""Test different approaches for computing weight grads."""
import torch


def time_fn(label, fn, warmup=10, rep=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep): fn()
    e.record(); e.synchronize()
    us = s.elapsed_time(e) / rep * 1000
    print(f"  {label}: {us:.0f} us")
    return us


def main():
    ne = 128
    h = 2880
    inter = 2880
    inter_2x = 2 * inter
    num_tokens = 256
    top_k = 4
    S = num_tokens * top_k
    device = "cuda"
    dtype = torch.bfloat16

    # Simulate sorted data
    flat_ids = torch.stack([torch.randperm(ne, device=device)[:top_k] for _ in range(num_tokens)]).reshape(-1)
    perm = torch.argsort(flat_ids)
    counts = torch.histc(flat_ids[perm].int().float(), bins=ne, min=0, max=ne-1)
    offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)

    hidden_s = torch.randn(S, h, device=device, dtype=dtype)
    d_gate_up_s = torch.randn(S, inter_2x, device=device, dtype=dtype)
    gated_s = torch.randn(S, inter, device=device, dtype=dtype)
    grad_weighted_s = torch.randn(S, h, device=device, dtype=dtype)

    gate_up_proj = torch.randn(ne, h, inter_2x, device=device, dtype=dtype)
    gate_up_bias = torch.randn(ne, inter_2x, device=device, dtype=dtype)
    down_proj = torch.randn(ne, inter, h, device=device, dtype=dtype)
    down_bias = torch.randn(ne, h, device=device, dtype=dtype)
    expert_of_sorted = flat_ids[perm].long()

    print("=" * 70)
    print(f"  Weight Grad Approaches  (ne={ne}, h={h}, inter={inter}, S={S})")
    print("=" * 70)

    # Approach 1: Current re-forward + autograd
    def reforward_approach():
        with torch.enable_grad():
            gup_w = gate_up_proj.detach().requires_grad_(True)
            gup_b = gate_up_bias.detach().requires_grad_(True)
            recomp = torch._grouped_mm(hidden_s.detach(), gup_w, offs=offsets)
            recomp = recomp + gup_b[expert_of_sorted]
            torch.autograd.grad(recomp, (gup_w, gup_b), grad_outputs=d_gate_up_s.detach())

            dp_w = down_proj.detach().requires_grad_(True)
            dp_b = down_bias.detach().requires_grad_(True)
            recomp2 = torch._grouped_mm(gated_s.detach(), dp_w, offs=offsets)
            recomp2 = recomp2 + dp_b[expert_of_sorted]
            torch.autograd.grad(recomp2, (dp_w, dp_b), grad_outputs=grad_weighted_s.detach())

    time_fn("Approach 1: re-forward + autograd (current)", reforward_approach)

    # Approach 2: Direct _grouped_mm for dW
    # dW_gate_up[e] = hidden_s[e].T @ d_gate_up_s[e]
    # Using _grouped_mm: input=(S, K).T per group @ grad=(S, N) per group
    # We can compute this as: _grouped_mm(hidden_s, d_gate_up_s, offs, transposed=True)
    # But _grouped_mm may not support this directly.
    # Instead: use the fact that A.T @ B per group = _grouped_mm(A.T, B)
    # But A.T isn't contiguous per group...

    # Approach 2a: Scatter to per-expert buffers, then bmm
    max_tok = int(counts.max().item())
    # Pad to max_tok per expert, then bmm
    def padded_bmm_approach():
        # Build padded per-expert tensors
        offs_list = torch.cat([torch.zeros(1, device=device, dtype=torch.int32), offsets])
        # hidden: (ne, max_tok, h), d_gate_up: (ne, max_tok, inter_2x)
        h_padded = torch.zeros(ne, max_tok, h, device=device, dtype=dtype)
        g_padded = torch.zeros(ne, max_tok, inter_2x, device=device, dtype=dtype)

        for e in range(ne):
            start = int(offs_list[e].item())
            end = int(offs_list[e+1].item())
            n = end - start
            if n > 0:
                h_padded[e, :n] = hidden_s[start:end]
                g_padded[e, :n] = d_gate_up_s[start:end]

        # bmm: (ne, h, max_tok) @ (ne, max_tok, inter_2x) -> (ne, h, inter_2x)
        d_gup = torch.bmm(h_padded.transpose(1, 2), g_padded)

        # bias: (ne, inter_2x) = sum over tokens
        d_gub = g_padded.sum(dim=1)

        # Similarly for down
        ga_padded = torch.zeros(ne, max_tok, inter, device=device, dtype=dtype)
        gw_padded = torch.zeros(ne, max_tok, h, device=device, dtype=dtype)
        for e in range(ne):
            start = int(offs_list[e].item())
            end = int(offs_list[e+1].item())
            n = end - start
            if n > 0:
                ga_padded[e, :n] = gated_s[start:end]
                gw_padded[e, :n] = grad_weighted_s[start:end]
        d_dp = torch.bmm(ga_padded.transpose(1, 2), gw_padded)
        d_db = gw_padded.sum(dim=1)

    time_fn("Approach 2a: padded bmm (Python loop)", padded_bmm_approach)

    # Approach 2b: Vectorized scatter + bmm (no Python loop)
    def vectorized_bmm_approach():
        offs_with_zero = torch.cat([torch.zeros(1, device=device, dtype=torch.int32), offsets])
        expert_starts = offs_with_zero[:-1].long()

        # Compute per-token offset within its expert group
        token_positions = torch.arange(S, device=device)
        token_expert = expert_of_sorted
        within_expert_offset = token_positions - expert_starts[token_expert]

        # Scatter into padded buffers
        h_padded = torch.zeros(ne, max_tok, h, device=device, dtype=dtype)
        g_padded = torch.zeros(ne, max_tok, inter_2x, device=device, dtype=dtype)
        h_padded[token_expert, within_expert_offset] = hidden_s
        g_padded[token_expert, within_expert_offset] = d_gate_up_s

        d_gup = torch.bmm(h_padded.transpose(1, 2), g_padded)
        d_gub = g_padded.sum(dim=1)

        ga_padded = torch.zeros(ne, max_tok, inter, device=device, dtype=dtype)
        gw_padded = torch.zeros(ne, max_tok, h, device=device, dtype=dtype)
        ga_padded[token_expert, within_expert_offset] = gated_s
        gw_padded[token_expert, within_expert_offset] = grad_weighted_s
        d_dp = torch.bmm(ga_padded.transpose(1, 2), gw_padded)
        d_db = gw_padded.sum(dim=1)

    time_fn("Approach 2b: vectorized scatter + bmm", vectorized_bmm_approach)

    # Approach 3: Direct _grouped_mm with transposed input
    # _grouped_mm(A, B, offs) computes output[group] = A[group] @ B[expert]
    # For dW: we want dW[expert] = A[group].T @ grad[group]
    # This is _grouped_mm with A transposed per-group, which isn't directly supported.
    # But we can use: _grouped_mm(grad.T_per_group, A_per_group) — same problem.
    # Try: construct transposed grouped input manually
    def direct_grouped_mm_approach():
        # For gate_up: dW[e] = hidden_s[e].T @ d_gate_up_s[e]
        # = (h, S_e) @ (S_e, inter_2x) -> (h, inter_2x)
        # _grouped_mm expects (S_total, K) @ (E, K, N) -> (S_total, N)
        # We need the TRANSPOSE: (S_total, K).T @ (S_total, N) -> (E, K, N)
        # This is not what _grouped_mm computes.

        # Simplest: use index_add_ for the bias, and _grouped_mm backward for weights
        # Actually, _grouped_mm(A, B, offs) internally uses CUTLASS grouped GEMM.
        # The backward of _grouped_mm w.r.t. B gives us dB[e] = A[group_e].T @ dOut[group_e]
        # which is EXACTLY what we want.

        # So: create a fake forward, then use autograd for just the weight grad
        # But wait, _grouped_mm backward for weights is buggy (we saw this earlier).
        # Let's test if it works with the specific shapes we need.
        pass

    # Approach 4: Simple bias via index scatter_add, weights via approach 2b bmm
    def hybrid_approach():
        offs_with_zero = torch.cat([torch.zeros(1, device=device, dtype=torch.int32), offsets])
        expert_starts = offs_with_zero[:-1].long()
        token_positions = torch.arange(S, device=device)
        within_expert_offset = token_positions - expert_starts[expert_of_sorted]

        # Gate-up weight grad: bmm
        h_padded = torch.zeros(ne, max_tok, h, device=device, dtype=dtype)
        g_padded = torch.zeros(ne, max_tok, inter_2x, device=device, dtype=dtype)
        h_padded[expert_of_sorted, within_expert_offset] = hidden_s
        g_padded[expert_of_sorted, within_expert_offset] = d_gate_up_s
        d_gup = torch.bmm(h_padded.transpose(1, 2), g_padded)

        # Gate-up bias grad: scatter_add
        d_gub = torch.zeros(ne, inter_2x, device=device, dtype=torch.float32)
        d_gub.scatter_add_(0, expert_of_sorted.unsqueeze(1).expand(-1, inter_2x),
                           d_gate_up_s.float())
        d_gub = d_gub.to(dtype)

        # Down weight grad: bmm
        ga_padded = torch.zeros(ne, max_tok, inter, device=device, dtype=dtype)
        gw_padded = torch.zeros(ne, max_tok, h, device=device, dtype=dtype)
        ga_padded[expert_of_sorted, within_expert_offset] = gated_s
        gw_padded[expert_of_sorted, within_expert_offset] = grad_weighted_s
        d_dp = torch.bmm(ga_padded.transpose(1, 2), gw_padded)

        # Down bias grad: scatter_add
        d_db = torch.zeros(ne, h, device=device, dtype=torch.float32)
        d_db.scatter_add_(0, expert_of_sorted.unsqueeze(1).expand(-1, h),
                          grad_weighted_s.float())
        d_db = d_db.to(dtype)

    time_fn("Approach 4: vectorized scatter + bmm + scatter_add bias", hybrid_approach)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
