"""
MoE Experts Kernel Benchmark.

Compares eager (sequential loop) vs grouped_mm (HF baseline) vs CuTile fused GEMM
for MoE expert execution. Tests both forward-only and forward+backward (training).

Shapes match GPT-OSS-20B: num_experts=128, top_k=4, hidden=2880, intermediate=2880.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass

from ..utils import (
    benchmark_fn,
    clear_cuda_state,
    print_header,
    print_gpu_info,
    format_speedup,
)


ALPHA = 1.702
LIMIT = 7.0


def pytorch_apply_gate(gate_up: torch.Tensor) -> torch.Tensor:
    """PyTorch reference: GPT-OSS _apply_gate."""
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(min=None, max=LIMIT)
    up = up.clamp(min=-LIMIT, max=LIMIT)
    glu = gate * torch.sigmoid(gate * ALPHA)
    return (up + 1) * glu


class FakeGptOssExperts(nn.Module):
    """Minimal GptOssExperts for benchmarking."""

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

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        return pytorch_apply_gate(gate_up)


def eager_forward(module, hidden_states, router_indices, routing_weights):
    """Eager (sequential) forward - matches GptOssExperts.forward."""
    next_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(
            router_indices, num_classes=module.num_experts
        )
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == module.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate_up = current_state @ module.gate_up_proj[expert_idx] + module.gate_up_proj_bias[expert_idx]
        gated_output = module._apply_gate(gate_up)
        out = gated_output @ module.down_proj[expert_idx] + module.down_proj_bias[expert_idx]
        weighted_output = out * routing_weights[token_idx, top_k_pos, None]
        next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
    return next_states


def make_routing(num_tokens, num_experts, top_k, device="cuda"):
    """Create random routing indices and weights."""
    router_indices = torch.stack([
        torch.randperm(num_experts, device=device)[:top_k]
        for _ in range(num_tokens)
    ])
    router_logits = torch.randn(num_tokens, top_k, device=device, dtype=torch.bfloat16)
    routing_weights = torch.softmax(router_logits.float(), dim=-1).to(torch.bfloat16)
    return router_indices, routing_weights


@dataclass
class MoEConfig:
    num_tokens: int
    num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    dtype: torch.dtype

    def __str__(self) -> str:
        dtype_str = str(self.dtype).split('.')[-1]
        return f"tok={self.num_tokens} exp={self.num_experts} top{self.top_k} h={self.hidden_size} {dtype_str}"


def run_forward_benchmark(config, grouped_mm_fn, cutile_fn):
    """Benchmark forward: eager vs grouped_mm vs CuTile."""
    clear_cuda_state()
    module = FakeGptOssExperts(
        config.num_experts, config.hidden_size, config.intermediate_size, config.dtype
    )
    hidden_states = torch.randn(
        config.num_tokens, config.hidden_size, device="cuda", dtype=config.dtype
    )
    ri, rw = make_routing(config.num_tokens, config.num_experts, config.top_k)

    grouped_lat = benchmark_fn(
        lambda: grouped_mm_fn(module, hidden_states, ri, rw), warmup=10, iterations=30,
    )
    cutile_lat = benchmark_fn(
        lambda: cutile_fn(module, hidden_states, ri, rw), warmup=10, iterations=30,
    )
    return {"grouped_mm": grouped_lat, "cutile": cutile_lat}


def run_fwd_bwd_benchmark(config, grouped_mm_fn, cutile_fn):
    """Benchmark forward+backward: grouped_mm vs CuTile."""
    clear_cuda_state()
    module = FakeGptOssExperts(
        config.num_experts, config.hidden_size, config.intermediate_size, config.dtype
    )
    ri, rw = make_routing(config.num_tokens, config.num_experts, config.top_k)

    def grouped_fwd_bwd():
        h = torch.randn(config.num_tokens, config.hidden_size, device="cuda",
                         dtype=config.dtype, requires_grad=True)
        out = grouped_mm_fn(module, h, ri, rw)
        out.sum().backward()

    def cutile_fwd_bwd():
        h = torch.randn(config.num_tokens, config.hidden_size, device="cuda",
                         dtype=config.dtype, requires_grad=True)
        out = cutile_fn(module, h, ri, rw)
        out.sum().backward()

    grouped_lat = benchmark_fn(grouped_fwd_bwd, warmup=10, iterations=30)
    cutile_lat = benchmark_fn(cutile_fwd_bwd, warmup=10, iterations=30)
    return {"grouped_mm": grouped_lat, "cutile": cutile_lat}


def main():
    from bastile.ops.moe_experts import grouped_mm_moe_forward, cutile_moe_experts_forward
    from bastile.autotune import clear_cache, _autotune_cache

    print("=" * 110)
    print("  MoE Experts Benchmark: grouped_mm (HF baseline) vs CuTile Fused GEMM (autotuned)")
    print("  GPT-OSS-20B: 128 experts, top-4 routing, hidden=2880, intermediate=2880")
    print("=" * 110)

    print_gpu_info()

    # Clear autotune cache to force fresh autotuning
    clear_cache()
    print("\n  Autotune cache cleared — will autotune tile sizes for each config.")

    # Warmup both implementations
    print("  Warming up...")
    module = FakeGptOssExperts(16, 128, 128)
    h = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ri, rw = make_routing(32, 16, 2)
    out = grouped_mm_moe_forward(module, h, ri, rw)
    out.sum().backward()
    h2 = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out2 = cutile_moe_experts_forward(module, h2, ri, rw)
    out2.sum().backward()
    torch.cuda.synchronize()
    del module, h, h2, ri, rw, out, out2
    clear_cuda_state()
    print("  Warmup complete.\n")

    configs = [
        MoEConfig(64, 128, 4, 2880, 2880, torch.bfloat16),
        MoEConfig(128, 128, 4, 2880, 2880, torch.bfloat16),
        MoEConfig(256, 128, 4, 2880, 2880, torch.bfloat16),
        MoEConfig(512, 128, 4, 2880, 2880, torch.bfloat16),
        MoEConfig(1024, 128, 4, 2880, 2880, torch.bfloat16),
        MoEConfig(4096, 128, 4, 2880, 2880, torch.bfloat16),
    ]

    # Forward only
    print_header("FORWARD ONLY", 110)
    print(f"\n  {'Config':<55} {'Grouped':>10} {'CuTile':>10} {'Speedup':>10}")
    print("  " + "-" * 90)

    fwd_results = []
    for config in configs:
        try:
            result = run_forward_benchmark(config, grouped_mm_moe_forward, cutile_moe_experts_forward)
            fwd_results.append(result)
            g, c = result["grouped_mm"], result["cutile"]
            speedup = g / c
            print(f"  {str(config):<55} {g:>8.0f}us {c:>8.0f}us {format_speedup(speedup):>10}")
        except Exception as ex:
            print(f"  {str(config):<55} ERROR: {ex}")

    # Show autotuned configs
    print(f"\n  Autotuned configs:")
    for k, v in _autotune_cache.items():
        if "moe_experts" in k:
            cfg = v.config
            print(f"    {k}: tile_m={cfg.tile_m} tile_n={cfg.tile_n} tile_k={cfg.tile_k} group_m={cfg.group_m}")

    # Forward + Backward
    print_header("FORWARD + BACKWARD (training)", 110)
    print(f"\n  {'Config':<55} {'Grouped':>10} {'CuTile':>10} {'Speedup':>10}")
    print("  " + "-" * 90)

    bwd_results = []
    for config in configs:
        try:
            result = run_fwd_bwd_benchmark(config, grouped_mm_moe_forward, cutile_moe_experts_forward)
            bwd_results.append(result)
            g, c = result["grouped_mm"], result["cutile"]
            speedup = g / c
            print(f"  {str(config):<55} {g:>8.0f}us {c:>8.0f}us {format_speedup(speedup):>10}")
        except Exception as ex:
            print(f"  {str(config):<55} ERROR: {ex}")

    # Summary
    print_header("SUMMARY", 110)
    if fwd_results:
        speedups = [r["grouped_mm"] / r["cutile"] for r in fwd_results]
        print(f"\n  Forward (CuTile vs grouped_mm):")
        print(f"    Average Speedup: {sum(speedups)/len(speedups):.2f}x")
        print(f"    Max Speedup:     {max(speedups):.2f}x")
        print(f"    Min Speedup:     {min(speedups):.2f}x")

    if bwd_results:
        speedups = [r["grouped_mm"] / r["cutile"] for r in bwd_results]
        print(f"\n  Forward + Backward (CuTile vs grouped_mm):")
        print(f"    Average Speedup: {sum(speedups)/len(speedups):.2f}x")
        print(f"    Max Speedup:     {max(speedups):.2f}x")
        print(f"    Min Speedup:     {min(speedups):.2f}x")

    print("\n" + "=" * 110)


if __name__ == "__main__":
    main()
