"""Profile MoE GEMM kernel to identify bottleneck."""
import torch
from bastile.ops.moe_experts import (
    fused_moe_gemm_kernel, _launch_fused_moe_gemm, moe_align_block_size,
    grouped_mm_moe_forward,
)
from bastile.ops.configs import MoEGemmConfig
import torch.nn as nn

ALPHA, LIMIT = 1.702, 7.0

class FakeModule(nn.Module):
    def __init__(self, ne, h, inter, dtype=torch.bfloat16):
        super().__init__()
        self.num_experts = ne
        self.has_bias = True
        self.is_transposed = True
        self.alpha = ALPHA
        self.limit = LIMIT
        self.gate_up_proj = nn.Parameter(torch.randn(ne, h, 2*inter, dtype=dtype, device="cuda") * 0.02)
        self.gate_up_proj_bias = nn.Parameter(torch.randn(ne, 2*inter, dtype=dtype, device="cuda") * 0.02)
        self.down_proj = nn.Parameter(torch.randn(ne, inter, h, dtype=dtype, device="cuda") * 0.02)
        self.down_proj_bias = nn.Parameter(torch.randn(ne, h, dtype=dtype, device="cuda") * 0.02)
    def _apply_gate(self, g):
        gate, up = g[..., ::2], g[..., 1::2]
        gate = gate.clamp(max=LIMIT); up = up.clamp(-LIMIT, LIMIT)
        return (up + 1) * gate * torch.sigmoid(gate * ALPHA)


def time_fn(label, fn, warmup=20, rep=50):
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
    ne, h, inter = 128, 2880, 2880
    inter_2x = 2 * inter
    top_k = 4

    print("=" * 80)
    print("  MoE GEMM Bottleneck Analysis")
    print("=" * 80)

    for num_tokens in [256, 1024]:
        S = num_tokens * top_k

        module = FakeModule(ne, h, inter)
        hidden = torch.randn(num_tokens, h, device="cuda", dtype=torch.bfloat16)
        ri = torch.stack([torch.randperm(ne, device="cuda")[:top_k] for _ in range(num_tokens)])
        rw = torch.softmax(torch.randn(num_tokens, top_k, device="cuda", dtype=torch.bfloat16).float(), dim=-1).bfloat16()

        print(f"\n-- num_tokens={num_tokens}, S={S} --")

        # Test different tile configs
        configs = [
            MoEGemmConfig(64, 64, 64, 8),
            MoEGemmConfig(64, 64, 64, 32),
            MoEGemmConfig(64, 32, 64, 8),
            MoEGemmConfig(128, 64, 64, 8),
            MoEGemmConfig(128, 32, 64, 8),
            MoEGemmConfig(64, 64, 32, 8),
            MoEGemmConfig(256, 64, 64, 8),
        ]

        weights_flat = rw.reshape(-1)
        for cfg in configs:
            sorted_ids, block_experts, n_post_pad = moe_align_block_size(ri, cfg.tile_m, ne)
            gate_up_out = torch.zeros(S + cfg.tile_m, inter_2x, dtype=torch.bfloat16, device="cuda")

            us = time_fn(
                f"tile=({cfg.tile_m},{cfg.tile_n},{cfg.tile_k}) grp={cfg.group_m}",
                lambda: _launch_fused_moe_gemm(
                    hidden, module.gate_up_proj, gate_up_out, module.gate_up_proj_bias,
                    weights_flat, sorted_ids, block_experts, n_post_pad,
                    N=inter_2x, K=h, top_k=top_k,
                    mul_routed_weight=False, has_bias=True, cfg=cfg,
                ),
            )

        # Compare grouped_mm
        flat_ids = ri.reshape(-1)
        perm = torch.argsort(flat_ids)
        counts = torch.histc(flat_ids[perm].int().float(), bins=ne, min=0, max=ne-1)
        offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)
        tok_idx = torch.arange(num_tokens, device="cuda").unsqueeze(1).expand(-1, top_k).reshape(-1)
        hidden_s = hidden[tok_idx[perm]].contiguous()

        time_fn(
            "grouped_mm (CUTLASS baseline)",
            lambda: torch._grouped_mm(hidden_s, module.gate_up_proj, offs=offsets),
        )

        # Compute arithmetic intensity
        # gate_up GEMM: (S, h) @ (E, h, 2*inter) -> (S, 2*inter)
        # FLOPs = 2 * S * h * 2*inter (per expert subset, but total across all experts)
        flops = 2 * S * h * inter_2x
        # Memory: read A (S*h*2), read B (one expert h*2inter*2 per token), write C (S*2inter*2)
        mem_bytes = S * h * 2 + ne * h * inter_2x * 2 + S * inter_2x * 2
        ai = flops / mem_bytes
        print(f"  Arithmetic intensity: {ai:.1f} FLOP/byte")
        print(f"  Total FLOPs: {flops/1e9:.1f} GFLOP")
        print(f"  Total memory: {mem_bytes/1e9:.1f} GB")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
