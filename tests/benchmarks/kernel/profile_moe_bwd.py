"""Quick profiling script for MoE backward breakdown."""

import torch
import torch.nn as nn
import time

ALPHA = 1.702
LIMIT = 7.0


def pytorch_apply_gate(gate_up):
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(min=None, max=LIMIT)
    up = up.clamp(min=-LIMIT, max=LIMIT)
    glu = gate * torch.sigmoid(gate * ALPHA)
    return (up + 1) * glu


class FakeGptOssExperts(nn.Module):
    def __init__(self, num_experts, hidden_size, intermediate_size, dtype=torch.bfloat16):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.has_bias = True
        self.is_transposed = True
        self.alpha = ALPHA
        self.limit = LIMIT
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, hidden_size, 2 * intermediate_size, dtype=dtype, device="cuda") * 0.02
        )
        self.gate_up_proj_bias = nn.Parameter(
            torch.randn(num_experts, 2 * intermediate_size, dtype=dtype, device="cuda") * 0.02
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size, dtype=dtype, device="cuda") * 0.02
        )
        self.down_proj_bias = nn.Parameter(
            torch.randn(num_experts, hidden_size, dtype=dtype, device="cuda") * 0.02
        )

    def _apply_gate(self, gate_up):
        return pytorch_apply_gate(gate_up)


def make_routing(num_tokens, num_experts, top_k, device="cuda"):
    router_indices = torch.stack([
        torch.randperm(num_experts, device=device)[:top_k]
        for _ in range(num_tokens)
    ])
    router_logits = torch.randn(num_tokens, top_k, device=device, dtype=torch.bfloat16)
    routing_weights = torch.softmax(router_logits.float(), dim=-1).to(torch.bfloat16)
    return router_indices, routing_weights


def time_op(label, fn, warmup=3, rep=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    end.synchronize()
    ms = start.elapsed_time(end) / rep
    print(f"  {label}: {ms*1000:.0f} us")
    return ms


def main():
    from bastile.ops.moe_experts import (
        cutile_moe_experts_forward, moe_align_block_size, _launch_fused_moe_gemm,
    )
    from bastile.ops.moe_gate import moe_gate_backward_cutile
    from bastile.ops.configs import MoEGemmConfig

    num_experts = 128
    hidden_size = 2880
    inter_size = 2880
    inter_2x = 2 * inter_size
    num_tokens = 256
    top_k = 4
    S = num_tokens * top_k

    module = FakeGptOssExperts(num_experts, hidden_size, inter_size)
    hidden_states = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ri, rw = make_routing(num_tokens, num_experts, top_k)

    # Forward (to populate saved tensors)
    out = cutile_moe_experts_forward(module, hidden_states, ri, rw)
    torch.cuda.synchronize()

    # Simulate backward inputs
    grad_output = torch.randn_like(hidden_states)
    cfg = MoEGemmConfig(tile_m=128, tile_n=64, tile_k=32, group_m=8)

    print("=" * 70)
    print(f"  MoE Backward Breakdown  (tok={num_tokens}, exp={num_experts}, top_k={top_k})")
    print(f"  S={S}, hidden={hidden_size}, inter={inter_size}")
    print("=" * 70)

    # --- Time individual operations ---

    # 1. Weight transpose + contiguous
    print("\n-- Weight transpose costs --")
    time_op("down_proj.T.contiguous()",
            lambda: module.down_proj.transpose(-2, -1).contiguous())
    time_op("gate_up_proj.T.contiguous()",
            lambda: module.gate_up_proj.transpose(-2, -1).contiguous())

    # 2. moe_align_block_size
    print("\n-- Routing alignment --")
    time_op("moe_align_block_size",
            lambda: moe_align_block_size(ri, cfg.tile_m, num_experts))

    # 3. Grad expansion + weight multiply
    print("\n-- Grad preparation --")
    time_op("grad_exp + routing_weight mul",
            lambda: (grad_output.unsqueeze(1).expand(-1, top_k, -1).reshape(S, hidden_size)
                     * rw.reshape(-1, 1)))

    # 4. Fused GEMM (the kernel itself)
    sorted_ids, block_experts, n_post_pad = moe_align_block_size(ri, cfg.tile_m, num_experts)
    weights_flat = rw.reshape(-1)
    down_proj_t = module.down_proj.transpose(-2, -1).contiguous()
    grad_weighted = (grad_output.unsqueeze(1).expand(-1, top_k, -1).reshape(S, hidden_size)
                     * rw.reshape(-1, 1)).contiguous()

    print("\n-- Fused GEMM kernels --")
    d_gated_buf = torch.zeros(S + cfg.tile_m, inter_size, dtype=torch.bfloat16, device="cuda")
    time_op("fused GEMM: d_gated (down dX)",
            lambda: _launch_fused_moe_gemm(
                grad_weighted, down_proj_t, d_gated_buf, None,
                weights_flat, sorted_ids, block_experts, n_post_pad,
                N=inter_size, K=hidden_size, top_k=1,
                mul_routed_weight=False, has_bias=False, cfg=cfg,
            ))

    gate_up_proj_t = module.gate_up_proj.transpose(-2, -1).contiguous()
    d_gate_up = torch.randn(S, inter_2x, dtype=torch.bfloat16, device="cuda")
    d_hidden_buf = torch.zeros(S + cfg.tile_m, hidden_size, dtype=torch.bfloat16, device="cuda")
    time_op("fused GEMM: d_hidden (gate_up dX)",
            lambda: _launch_fused_moe_gemm(
                d_gate_up, gate_up_proj_t, d_hidden_buf, None,
                weights_flat, sorted_ids, block_experts, n_post_pad,
                N=hidden_size, K=inter_2x, top_k=1,
                mul_routed_weight=False, has_bias=False, cfg=cfg,
            ))

    # 5. Compare: _grouped_mm path
    print("\n-- grouped_mm baseline (what we replaced) --")
    flat_ids = ri.reshape(-1)
    perm = torch.argsort(flat_ids)
    counts = torch.histc(flat_ids[perm].int().float(), bins=num_experts, min=0, max=num_experts - 1)
    offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)
    grad_down_s = grad_weighted[perm].contiguous()

    time_op("argsort + histc + cumsum",
            lambda: (torch.argsort(flat_ids),
                     torch.histc(flat_ids[perm].int().float(), bins=num_experts, min=0, max=num_experts - 1)))
    time_op("grad[perm].contiguous()",
            lambda: grad_weighted[perm].contiguous())
    time_op("_grouped_mm (d_gated)",
            lambda: torch._grouped_mm(grad_down_s, module.down_proj.transpose(-2, -1), offs=offsets))
    time_op("result[inv_perm]",
            lambda: torch._grouped_mm(grad_down_s, module.down_proj.transpose(-2, -1), offs=offsets)[torch.argsort(perm)])

    # 6. Weight grads (re-forward trick)
    print("\n-- Weight grads (re-forward) --")
    hidden_s = hidden_states[torch.arange(num_tokens, device="cuda").unsqueeze(1).expand(-1, top_k).reshape(-1)[perm]].contiguous().detach()
    d_gate_up_s = d_gate_up[perm].contiguous().detach()

    def weight_grad_step():
        gup_w = module.gate_up_proj.detach().requires_grad_(True)
        gup_b = module.gate_up_proj_bias.detach().requires_grad_(True)
        gate_up_recomp = torch._grouped_mm(hidden_s, gup_w, offs=offsets)
        gate_up_recomp = gate_up_recomp + gup_b[flat_ids[perm].long()]
        torch.autograd.grad(gate_up_recomp, (gup_w, gup_b), grad_outputs=d_gate_up_s)

    time_op("weight grad (gate_up, re-forward+autograd)", weight_grad_step)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
