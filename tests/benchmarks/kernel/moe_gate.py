"""
MoE Expert Gate Kernel Benchmark.

Compares CuTile MoE gate vs PyTorch reference for GPT-OSS _apply_gate.
Tests both forward and forward+backward (training) performance.

Shapes match GPT-OSS-20B: intermediate_size=2880, with varying token counts
to simulate different seq_len * batch_size * top_k combinations.
"""

import torch
from dataclasses import dataclass
from typing import List, Tuple

from ..utils import (
    benchmark_fn,
    clear_cuda_state,
    print_header,
    print_gpu_info,
    format_speedup,
)


ALPHA = 1.702
LIMIT = 7.0


def pytorch_moe_gate(gate_up: torch.Tensor) -> torch.Tensor:
    """PyTorch reference: GPT-OSS _apply_gate."""
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(min=None, max=LIMIT)
    up = up.clamp(min=-LIMIT, max=LIMIT)
    glu = gate * torch.sigmoid(gate * ALPHA)
    return (up + 1) * glu


@dataclass
class MoEGateConfig:
    num_tokens: int  # batch * seq_len * top_k routed tokens
    intermediate_size: int
    dtype: torch.dtype

    @property
    def gate_up_shape(self) -> Tuple[int, int]:
        return (self.num_tokens, 2 * self.intermediate_size)

    def __str__(self) -> str:
        dtype_str = str(self.dtype).split('.')[-1]
        return f"({self.num_tokens}, 2x{self.intermediate_size}) {dtype_str}"


def run_forward_benchmark(config: MoEGateConfig, cutile_fn) -> dict:
    """Benchmark forward pass only."""
    clear_cuda_state()
    gate_up = torch.randn(*config.gate_up_shape, device="cuda", dtype=config.dtype)

    pytorch_lat = benchmark_fn(lambda: pytorch_moe_gate(gate_up))
    cutile_lat = benchmark_fn(lambda: cutile_fn(gate_up))

    return {"pytorch": pytorch_lat, "cutile": cutile_lat}


def run_fwd_bwd_benchmark(config: MoEGateConfig, cutile_fn) -> dict:
    """Benchmark forward + backward pass (training)."""
    clear_cuda_state()

    def pytorch_fwd_bwd():
        gate_up = torch.randn(*config.gate_up_shape, device="cuda", dtype=config.dtype, requires_grad=True)
        out = pytorch_moe_gate(gate_up)
        out.sum().backward()

    def cutile_fwd_bwd():
        gate_up = torch.randn(*config.gate_up_shape, device="cuda", dtype=config.dtype, requires_grad=True)
        out = cutile_fn(gate_up)
        out.sum().backward()

    pytorch_lat = benchmark_fn(pytorch_fwd_bwd)
    cutile_lat = benchmark_fn(cutile_fwd_bwd)

    return {"pytorch": pytorch_lat, "cutile": cutile_lat}


def jit_warmup(cutile_fn):
    """Pre-compile kernel variants."""
    print("  JIT compiling kernels...")
    for n_tok in [64, 256, 1024]:
        for dtype in [torch.bfloat16, torch.float32]:
            gate_up = torch.randn(n_tok, 2 * 2880, device="cuda", dtype=dtype, requires_grad=True)
            out = cutile_fn(gate_up)
            out.sum().backward()
            torch.cuda.synchronize()
    print("  JIT compilation complete.\n")


def main():
    from bastile.ops.moe_gate import MoEGateFunction

    def cutile_fn(gate_up):
        return MoEGateFunction.apply(gate_up)

    print("=" * 110)
    print("  MoE Expert Gate Kernel Benchmark: CuTile vs PyTorch")
    print("  GPT-OSS _apply_gate: interleaved gate/up, clamp, scaled sigmoid, gating")
    print("=" * 110)

    print_gpu_info()
    jit_warmup(cutile_fn)

    # GPT-OSS dimensions: intermediate_size=2880
    # num_tokens = seq_len * num_experts_per_tok (top-4) for tokens routed to each expert
    # In practice, each expert sees a subset. We test various token counts.
    configs = [
        # Small (few tokens routed to expert)
        MoEGateConfig(16, 2880, torch.bfloat16),
        MoEGateConfig(64, 2880, torch.bfloat16),
        MoEGateConfig(128, 2880, torch.bfloat16),
        MoEGateConfig(256, 2880, torch.bfloat16),
        # Medium
        MoEGateConfig(512, 2880, torch.bfloat16),
        MoEGateConfig(1024, 2880, torch.bfloat16),
        # Larger (seq_len=4096 * top_k=4 tokens distributed across 128 experts)
        MoEGateConfig(2048, 2880, torch.bfloat16),
        MoEGateConfig(4096, 2880, torch.bfloat16),
        # float32
        MoEGateConfig(256, 2880, torch.float32),
        MoEGateConfig(1024, 2880, torch.float32),
    ]

    # Forward only
    print_header("FORWARD ONLY", 110)
    print(f"\n  {'Config':<40} {'PyTorch':>10} {'CuTile':>10} {'Speedup':>10}")
    print("  " + "-" * 80)

    fwd_results = []
    for config in configs:
        try:
            result = run_forward_benchmark(config, cutile_fn)
            fwd_results.append(result)
            py, ct = result["pytorch"], result["cutile"]
            speedup = py / ct
            print(f"  {str(config):<40} {py:>8.1f}us {ct:>8.1f}us {format_speedup(speedup):>10}")
        except Exception as e:
            print(f"  {str(config):<40} ERROR: {e}")

    # Forward + Backward
    print_header("FORWARD + BACKWARD (training)", 110)
    print(f"\n  {'Config':<40} {'PyTorch':>10} {'CuTile':>10} {'Speedup':>10}")
    print("  " + "-" * 80)

    bwd_results = []
    for config in configs:
        try:
            result = run_fwd_bwd_benchmark(config, cutile_fn)
            bwd_results.append(result)
            py, ct = result["pytorch"], result["cutile"]
            speedup = py / ct
            print(f"  {str(config):<40} {py:>8.1f}us {ct:>8.1f}us {format_speedup(speedup):>10}")
        except Exception as e:
            print(f"  {str(config):<40} ERROR: {e}")

    # Summary
    print_header("SUMMARY", 110)
    if fwd_results:
        fwd_speedups = [r["pytorch"] / r["cutile"] for r in fwd_results]
        print(f"\n  Forward:")
        print(f"    Average Speedup: {sum(fwd_speedups)/len(fwd_speedups):.2f}x")
        print(f"    Max Speedup:     {max(fwd_speedups):.2f}x")
        print(f"    Min Speedup:     {min(fwd_speedups):.2f}x")

    if bwd_results:
        bwd_speedups = [r["pytorch"] / r["cutile"] for r in bwd_results]
        print(f"\n  Forward + Backward:")
        print(f"    Average Speedup: {sum(bwd_speedups)/len(bwd_speedups):.2f}x")
        print(f"    Max Speedup:     {max(bwd_speedups):.2f}x")
        print(f"    Min Speedup:     {min(bwd_speedups):.2f}x")

    print("\n" + "=" * 110)


if __name__ == "__main__":
    main()
