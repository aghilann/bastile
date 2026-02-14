"""
RMSNorm Kernel Benchmark: cuTILE vs Quack (cuteDSL) vs Liger vs PyTorch.

Compares the native cuTILE RMSNorm implementation against Quack's cuteDSL
kernels, Liger Kernel, and PyTorch reference. Reports latency, throughput,
bandwidth utilization, and relative speedups.
"""

import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional

from ..utils import (
    benchmark_fn,
    clear_cuda_state,
    get_peak_bandwidth,
    compute_throughput_gbps,
    compute_bandwidth_utilization,
    print_header,
    print_gpu_info,
    format_speedup,
)


@dataclass
class BenchConfig:
    batch_size: int
    seq_len: int
    hidden_size: int
    dtype: torch.dtype

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.batch_size, self.seq_len, self.hidden_size)

    @property
    def total_elements(self) -> int:
        return self.batch_size * self.seq_len * self.hidden_size

    @property
    def element_size(self) -> int:
        return self.dtype.itemsize

    @property
    def bytes_accessed(self) -> int:
        # Read input + read weight + write output = 2*M*N + N (weight amortized)
        return 2 * self.total_elements * self.element_size

    def __str__(self) -> str:
        dtype_str = str(self.dtype).split(".")[-1]
        return f"({self.batch_size}, {self.seq_len}, {self.hidden_size}) {dtype_str}"


def pytorch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """PyTorch reference RMSNorm."""
    x_float = x.float()
    variance = x_float.pow(2).mean(-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(variance + eps)
    return (weight.float() * x_normed).to(x.dtype)


def run_single_benchmark(
    config: BenchConfig,
    cutile_fn,
    quack_fn,
    liger_fn,
    peak_bw: float,
):
    """Run benchmark for a single configuration."""
    clear_cuda_state()

    x = torch.randn(*config.shape, device="cuda", dtype=config.dtype)
    weight = torch.ones(config.hidden_size, device="cuda", dtype=config.dtype)
    eps = 1e-6

    # PyTorch reference
    pytorch_lat = benchmark_fn(lambda: pytorch_rms_norm(x, weight, eps))
    pytorch_tp = compute_throughput_gbps(config.bytes_accessed, pytorch_lat)
    pytorch_util = compute_bandwidth_utilization(pytorch_tp, peak_bw)

    # Liger Kernel
    liger_lat = benchmark_fn(lambda: liger_fn(x, weight, eps))
    liger_tp = compute_throughput_gbps(config.bytes_accessed, liger_lat)
    liger_util = compute_bandwidth_utilization(liger_tp, peak_bw)

    # Quack (cuteDSL)
    quack_lat = benchmark_fn(lambda: quack_fn(x, weight, eps))
    quack_tp = compute_throughput_gbps(config.bytes_accessed, quack_lat)
    quack_util = compute_bandwidth_utilization(quack_tp, peak_bw)

    # cuTILE (native)
    cutile_lat = benchmark_fn(lambda: cutile_fn(x, weight, eps))
    cutile_tp = compute_throughput_gbps(config.bytes_accessed, cutile_lat)
    cutile_util = compute_bandwidth_utilization(cutile_tp, peak_bw)

    return {
        "pytorch": {"latency": pytorch_lat, "throughput": pytorch_tp, "util": pytorch_util},
        "liger": {"latency": liger_lat, "throughput": liger_tp, "util": liger_util},
        "quack": {"latency": quack_lat, "throughput": quack_tp, "util": quack_util},
        "cutile": {"latency": cutile_lat, "throughput": cutile_tp, "util": cutile_util},
    }


def jit_warmup(cutile_fn, quack_fn, liger_fn, all_configs):
    """Pre-compile all kernel variants for every dtype/shape in the benchmark."""
    print("JIT compiling kernels...")

    # Collect unique (hidden_size, dtype) pairs from all benchmark configs
    seen = set()
    warmup_shapes = []
    for cfg in all_configs:
        key = (cfg.hidden_size, cfg.dtype)
        if key not in seen:
            seen.add(key)
            warmup_shapes.append(key)

    for hidden, dtype in warmup_shapes:
        x = torch.randn(4, 256, hidden, device="cuda", dtype=dtype)
        w = torch.ones(hidden, device="cuda", dtype=dtype)

        # Warmup each implementation
        for fn in [cutile_fn, quack_fn, liger_fn]:
            for _ in range(3):
                _ = fn(x, w, 1e-6)
            torch.cuda.synchronize()

    print(f"JIT compilation complete ({len(warmup_shapes)} shape/dtype combos).\n")


COL_WIDTH = 125


def run_benchmark_section(
    title: str,
    configs: List[BenchConfig],
    cutile_fn,
    quack_fn,
    liger_fn,
    peak_bw: float,
) -> List[dict]:
    """Run benchmarks for a section of configs."""
    print_header(title, COL_WIDTH)
    hdr = (
        f"{'Config':<38} {'Provider':<10} {'Latency':>10} {'Thruput':>10} "
        f"{'BW%':>6} {'vs PyTorch':>12} {'vs Liger':>10} {'vs Quack':>10}"
    )
    print(hdr)
    print("-" * COL_WIDTH)

    results = []
    for config in configs:
        try:
            result = run_single_benchmark(
                config, cutile_fn, quack_fn, liger_fn, peak_bw,
            )
            results.append((config, result))

            py = result["pytorch"]
            lg = result["liger"]
            qk = result["quack"]
            ct = result["cutile"]

            # PyTorch
            print(
                f"{str(config):<38} {'PyTorch':<10} "
                f"{py['latency']:>8.1f}us {py['throughput']:>8.1f}GB/s "
                f"{py['util']:>5.1f}%"
            )

            # Liger
            lg_vs_py = py["latency"] / lg["latency"]
            print(
                f"{'':<38} {'Liger':<10} "
                f"{lg['latency']:>8.1f}us {lg['throughput']:>8.1f}GB/s "
                f"{lg['util']:>5.1f}% {format_speedup(lg_vs_py):>12}"
            )

            # Quack
            qk_vs_py = py["latency"] / qk["latency"]
            qk_vs_lg = lg["latency"] / qk["latency"]
            print(
                f"{'':<38} {'Quack':<10} "
                f"{qk['latency']:>8.1f}us {qk['throughput']:>8.1f}GB/s "
                f"{qk['util']:>5.1f}% {format_speedup(qk_vs_py):>12} "
                f"{format_speedup(qk_vs_lg):>10}"
            )

            # cuTILE
            ct_vs_py = py["latency"] / ct["latency"]
            ct_vs_lg = lg["latency"] / ct["latency"]
            ct_vs_qk = qk["latency"] / ct["latency"]
            print(
                f"{'':<38} {'cuTILE':<10} "
                f"{ct['latency']:>8.1f}us {ct['throughput']:>8.1f}GB/s "
                f"{ct['util']:>5.1f}% {format_speedup(ct_vs_py):>12} "
                f"{format_speedup(ct_vs_lg):>10} {format_speedup(ct_vs_qk):>10}"
            )
            print()
        except Exception as e:
            print(f"{str(config):<38} ERROR: {e}\n")

    return results


def main():
    # --- Import implementations ---
    from bastile.ops.rms_norm_cutile import rms_norm as cutile_rms_norm
    from bastile.ops.rms_norm import FastRMSNormFunction

    def quack_rms_norm(x, w, eps):
        return FastRMSNormFunction.apply(x, w, eps)

    from liger_kernel.ops import LigerRMSNormFunction

    def liger_rms_norm(x, w, eps):
        return LigerRMSNormFunction.apply(x, w, eps)

    print("=" * COL_WIDTH)
    print("RMSNorm Benchmark: cuTILE vs Quack (cuteDSL) vs Liger vs PyTorch")
    print("=" * COL_WIDTH)

    print_gpu_info()
    peak_bw = get_peak_bandwidth()

    # --- Define all configs upfront so we can warm up ALL dtypes ---
    bf16_configs = [
        BenchConfig(4, 256, 2048, torch.bfloat16),
        BenchConfig(4, 256, 4096, torch.bfloat16),
        BenchConfig(8, 512, 2048, torch.bfloat16),
        BenchConfig(8, 512, 4096, torch.bfloat16),
        BenchConfig(32, 512, 2048, torch.bfloat16),
        BenchConfig(32, 512, 4096, torch.bfloat16),
        BenchConfig(64, 512, 4096, torch.bfloat16),
    ]
    fp16_configs = [
        BenchConfig(4, 256, 2048, torch.float16),
        BenchConfig(8, 512, 4096, torch.float16),
        BenchConfig(32, 512, 2048, torch.float16),
        BenchConfig(32, 512, 4096, torch.float16),
    ]
    fp32_configs = [
        BenchConfig(4, 256, 2048, torch.float32),
        BenchConfig(4, 256, 4096, torch.float32),
        BenchConfig(32, 512, 2048, torch.float32),
    ]

    all_bench_configs = bf16_configs + fp16_configs + fp32_configs
    jit_warmup(cutile_rms_norm, quack_rms_norm, liger_rms_norm, all_bench_configs)

    all_results = []

    all_results.extend(
        run_benchmark_section(
            "BFLOAT16 BENCHMARKS",
            bf16_configs, cutile_rms_norm, quack_rms_norm, liger_rms_norm, peak_bw,
        )
    )

    all_results.extend(
        run_benchmark_section(
            "FLOAT16 BENCHMARKS",
            fp16_configs, cutile_rms_norm, quack_rms_norm, liger_rms_norm, peak_bw,
        )
    )

    all_results.extend(
        run_benchmark_section(
            "FLOAT32 BENCHMARKS",
            fp32_configs, cutile_rms_norm, quack_rms_norm, liger_rms_norm, peak_bw,
        )
    )

    # --- Summary ---
    print_header("SUMMARY", COL_WIDTH)

    if not all_results:
        print("No results collected.")
        return

    cutile_vs_quack = []
    cutile_vs_liger = []
    cutile_vs_pytorch = []
    quack_vs_liger = []

    for config, result in all_results:
        py_lat = result["pytorch"]["latency"]
        lg_lat = result["liger"]["latency"]
        qk_lat = result["quack"]["latency"]
        ct_lat = result["cutile"]["latency"]

        cutile_vs_pytorch.append(py_lat / ct_lat)
        cutile_vs_liger.append(lg_lat / ct_lat)
        cutile_vs_quack.append(qk_lat / ct_lat)
        quack_vs_liger.append(lg_lat / qk_lat)

    def print_stats(label, values):
        avg = sum(values) / len(values)
        print(f"\n{label}:")
        print(f"  Average: {format_speedup(avg)}")
        print(f"  Best:    {format_speedup(max(values))}")
        print(f"  Worst:   {format_speedup(min(values))}")

    print_stats("cuTILE vs PyTorch", cutile_vs_pytorch)
    print_stats("cuTILE vs Liger", cutile_vs_liger)
    print_stats("cuTILE vs Quack (cuteDSL)", cutile_vs_quack)
    print_stats("Quack vs Liger", quack_vs_liger)

    print("\n" + "=" * COL_WIDTH)


if __name__ == "__main__":
    main()
