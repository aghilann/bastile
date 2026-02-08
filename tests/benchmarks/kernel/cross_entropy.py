"""
Cross-Entropy Kernel Benchmark.

Compares CuTile fused cross-entropy vs Liger Kernel vs PyTorch reference.

Measures both:
1. Forward-only latency (note: CuTile/Liger fuse gradient computation into forward)
2. Forward + backward latency (the real comparison - fused vs separate passes)
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


@dataclass
class CrossEntropyConfig:
    batch_tokens: int  # BT = batch_size * seq_len (flattened)
    vocab_size: int
    ignore_pct: float  # percentage of targets set to ignore_index

    def __str__(self) -> str:
        return f"(BT={self.batch_tokens}, V={self.vocab_size}, ign={self.ignore_pct:.0%})"


def pytorch_cross_entropy_fwd(logits, targets):
    return torch.nn.functional.cross_entropy(
        logits, targets, ignore_index=-100, reduction="mean"
    )


def pytorch_cross_entropy_fwd_bwd(logits, targets):
    """PyTorch CE forward + backward (separate passes)."""
    logits.requires_grad_(True)
    loss = torch.nn.functional.cross_entropy(
        logits, targets, ignore_index=-100, reduction="mean"
    )
    loss.backward()
    return loss


def run_single_benchmark(config, cutile_ce, liger_ce) -> dict:
    """Run benchmark for a single configuration."""
    clear_cuda_state()

    logits = torch.randn(
        config.batch_tokens, config.vocab_size,
        device="cuda", dtype=torch.float32,
    )
    targets = torch.randint(0, config.vocab_size, (config.batch_tokens,), device="cuda")

    if config.ignore_pct > 0:
        n_ignore = int(config.batch_tokens * config.ignore_pct)
        targets[:n_ignore] = -100

    # Forward-only benchmarks
    pytorch_fwd = benchmark_fn(lambda: pytorch_cross_entropy_fwd(logits.clone(), targets))
    liger_fwd = benchmark_fn(lambda: liger_ce(logits.clone(), targets))
    cutile_fwd = benchmark_fn(lambda: cutile_ce(logits.clone(), targets))

    # Forward + backward benchmarks (the real comparison)
    pytorch_fwd_bwd = benchmark_fn(
        lambda: pytorch_cross_entropy_fwd_bwd(logits.clone(), targets)
    )
    liger_fwd_bwd = benchmark_fn(lambda: (
        liger_ce(logits.clone().requires_grad_(True), targets).backward()
    ))
    cutile_fwd_bwd = benchmark_fn(lambda: (
        cutile_ce(logits.clone().requires_grad_(True), targets).backward()
    ))

    return {
        "pytorch_fwd": pytorch_fwd,
        "liger_fwd": liger_fwd,
        "cutile_fwd": cutile_fwd,
        "pytorch_fwd_bwd": pytorch_fwd_bwd,
        "liger_fwd_bwd": liger_fwd_bwd,
        "cutile_fwd_bwd": cutile_fwd_bwd,
    }


def jit_warmup(cutile_ce, liger_ce):
    """Pre-compile kernel variants for all benchmark configs."""
    print("JIT compiling kernels...")

    warmup_configs = [
        (64, 1024),
        (256, 1024),
        (256, 32000),
        (512, 32000),
        (1024, 32000),
        (2048, 32000),
        (4096, 32000),
    ]

    for bt, vocab in warmup_configs:
        logits = torch.randn(bt, vocab, device="cuda", dtype=torch.float32, requires_grad=True)
        targets = torch.randint(0, vocab, (bt,), device="cuda")
        for _ in range(3):
            loss = cutile_ce(logits.clone().requires_grad_(True), targets)
            loss.backward()
            loss = liger_ce(logits.clone().requires_grad_(True), targets)
            loss.backward()
        torch.cuda.synchronize()

    print("JIT compilation complete.\n")


def main():
    from bastile.ops.cross_entropy import cutile_fixed_cross_entropy
    from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction

    def liger_ce(logits, targets):
        loss, _, _ = LigerCrossEntropyFunction.apply(
            logits, targets, None, -100, 0.0, 0.0, "mean", None, False, False,
        )
        return loss

    print("=" * 120)
    print("Cross-Entropy Kernel Benchmark: CuTile vs Liger Kernel vs PyTorch")
    print("=" * 120)

    print_gpu_info()

    jit_warmup(cutile_fixed_cross_entropy, liger_ce)

    configs = [
        CrossEntropyConfig(256, 1024, 0.0),
        CrossEntropyConfig(512, 32000, 0.0),
        CrossEntropyConfig(1024, 32000, 0.0),
        CrossEntropyConfig(2048, 32000, 0.0),
        CrossEntropyConfig(4096, 32000, 0.0),
        CrossEntropyConfig(2048, 32000, 0.1),
        CrossEntropyConfig(2048, 32000, 0.25),
    ]

    # Forward-only table
    print_header("FORWARD ONLY (note: CuTile/Liger also compute gradients in this pass)", 120)
    print(f"{'Config':<40} {'PyTorch':>10} {'Liger':>10} {'CuTile':>10} {'CT/Py':>12} {'CT/Lg':>12}")
    print("-" * 120)

    all_results = []

    for config in configs:
        try:
            result = run_single_benchmark(config, cutile_fixed_cross_entropy, liger_ce)
            all_results.append(result)

            py = result["pytorch_fwd"]
            lg = result["liger_fwd"]
            ct_val = result["cutile_fwd"]

            print(f"{str(config):<40} {py:>8.1f}us {lg:>8.1f}us {ct_val:>8.1f}us "
                  f"{format_speedup(py/ct_val):>12} {format_speedup(lg/ct_val):>12}")
        except Exception as e:
            print(f"{str(config):<40} ERROR: {e}")

    # Forward + backward table
    print_header("FORWARD + BACKWARD (real-world comparison)", 120)
    print(f"{'Config':<40} {'PyTorch':>10} {'Liger':>10} {'CuTile':>10} {'CT/Py':>12} {'CT/Lg':>12}")
    print("-" * 120)

    for i, config in enumerate(configs):
        if i < len(all_results):
            r = all_results[i]
            py = r["pytorch_fwd_bwd"]
            lg = r["liger_fwd_bwd"]
            ct_val = r["cutile_fwd_bwd"]

            print(f"{str(config):<40} {py:>8.1f}us {lg:>8.1f}us {ct_val:>8.1f}us "
                  f"{format_speedup(py/ct_val):>12} {format_speedup(lg/ct_val):>12}")

    # Summary
    print_header("SUMMARY", 120)

    if all_results:
        # Forward + backward summary (the meaningful comparison)
        ct_vs_py_fwdbwd = [r["pytorch_fwd_bwd"] / r["cutile_fwd_bwd"] for r in all_results]
        ct_vs_lg_fwdbwd = [r["liger_fwd_bwd"] / r["cutile_fwd_bwd"] for r in all_results]
        lg_vs_py_fwdbwd = [r["pytorch_fwd_bwd"] / r["liger_fwd_bwd"] for r in all_results]

        print(f"\nForward + Backward (real-world comparison):")
        print(f"  CuTile vs PyTorch: avg {sum(ct_vs_py_fwdbwd)/len(ct_vs_py_fwdbwd):.2f}x "
              f"(range: {min(ct_vs_py_fwdbwd):.2f}x - {max(ct_vs_py_fwdbwd):.2f}x)")
        print(f"  CuTile vs Liger:   avg {sum(ct_vs_lg_fwdbwd)/len(ct_vs_lg_fwdbwd):.2f}x "
              f"(range: {min(ct_vs_lg_fwdbwd):.2f}x - {max(ct_vs_lg_fwdbwd):.2f}x)")
        print(f"  Liger vs PyTorch:  avg {sum(lg_vs_py_fwdbwd)/len(lg_vs_py_fwdbwd):.2f}x "
              f"(range: {min(lg_vs_py_fwdbwd):.2f}x - {max(lg_vs_py_fwdbwd):.2f}x)")

        avg_vs_liger = sum(ct_vs_lg_fwdbwd)/len(ct_vs_lg_fwdbwd)
        if avg_vs_liger >= 1.0:
            print(f"\n  Result: CuTile CE fwd+bwd is {(avg_vs_liger - 1) * 100:.1f}% FASTER than Liger")
        else:
            print(f"\n  Result: CuTile CE fwd+bwd is {(1 - avg_vs_liger) * 100:.1f}% SLOWER than Liger")

    print("\n" + "=" * 120)


if __name__ == "__main__":
    main()
