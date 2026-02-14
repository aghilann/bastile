"""
RMSNorm Kernel Benchmark: Quack (cuteDSL) vs CuTile.

Micro-benchmark comparing forward and backward latencies of:
  1. Quack (cuteDSL) RMSNorm — the fast baseline
  2. CuTile RMSNorm — the native cuTILE implementation we're optimizing
  3. PyTorch reference — for context

Reports latency (µs), effective bandwidth (GB/s), and BW utilization (%).
"""

import torch
from dataclasses import dataclass
from typing import Optional, Tuple

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


# ============================================================================
# Config
# ============================================================================

@dataclass
class RMSNormConfig:
    M: int  # rows (batch * seq_len)
    N: int  # hidden_size
    dtype: torch.dtype

    @property
    def bytes_fwd(self) -> int:
        """Read x + read w + write y  (weight is amortised across rows)."""
        elem = self.dtype.itemsize
        return 2 * self.M * self.N * elem + self.N * elem

    @property
    def bytes_bwd(self) -> int:
        """Read x + dy + w + rstd + write dx + temp_buffer r/w + sum→dw."""
        elem = self.dtype.itemsize
        return (
            self.M * self.N * elem       # read x
            + self.M * self.N * elem     # read dy
            + self.N * elem              # read w
            + self.M * 4                 # read rstd (f32)
            + self.M * self.N * elem     # write dx
            + self.M * self.N * 4 * 2    # temp_buffer write + read (f32)
            + self.N * elem              # write dw
        )

    def __str__(self) -> str:
        dt = str(self.dtype).split(".")[-1]
        return f"M={self.M:<6d} N={self.N:<5d} {dt}"


# ============================================================================
# Reference Implementations
# ============================================================================

def pytorch_rms_norm_fwd(x, w, eps):
    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    return (x_f * torch.rsqrt(var + eps)).to(x.dtype) * w


# ============================================================================
# Forward Benchmark
# ============================================================================

def bench_forward(configs, quack_rms_norm, cutile_rms_norm, peak_bw):
    print_header("FORWARD PASS", 130)
    hdr = (
        f"{'Config':<30} {'Provider':<10} {'Latency':>10} "
        f"{'BW GB/s':>10} {'BW%':>6} {'vs PyTorch':>12} {'vs Quack':>10}"
    )
    print(hdr)
    print("-" * 130)

    all_speedups = []

    for cfg in configs:
        clear_cuda_state()
        x = torch.randn(cfg.M, cfg.N, device="cuda", dtype=cfg.dtype)
        w = torch.ones(cfg.N, device="cuda", dtype=cfg.dtype)
        eps = 1e-6

        # PyTorch
        lat_pt = benchmark_fn(lambda: pytorch_rms_norm_fwd(x, w, eps))
        bw_pt = compute_throughput_gbps(cfg.bytes_fwd, lat_pt)
        ut_pt = compute_bandwidth_utilization(bw_pt, peak_bw)

        # Quack
        lat_q = benchmark_fn(lambda: quack_rms_norm(x, w, eps))
        bw_q = compute_throughput_gbps(cfg.bytes_fwd, lat_q)
        ut_q = compute_bandwidth_utilization(bw_q, peak_bw)
        sp_q = lat_pt / lat_q

        # CuTile
        lat_ct = benchmark_fn(lambda: cutile_rms_norm(x, w, eps))
        bw_ct = compute_throughput_gbps(cfg.bytes_fwd, lat_ct)
        ut_ct = compute_bandwidth_utilization(bw_ct, peak_bw)
        sp_ct_pt = lat_pt / lat_ct
        sp_ct_q = lat_q / lat_ct

        all_speedups.append(sp_ct_q)

        tag = str(cfg)
        print(f"{tag:<30} {'PyTorch':<10} {lat_pt:>8.1f}us {bw_pt:>8.1f}GB/s {ut_pt:>5.1f}%")
        print(f"{'':<30} {'Quack':<10} {lat_q:>8.1f}us {bw_q:>8.1f}GB/s {ut_q:>5.1f}% {format_speedup(sp_q):>12}")
        print(f"{'':<30} {'CuTile':<10} {lat_ct:>8.1f}us {bw_ct:>8.1f}GB/s {ut_ct:>5.1f}% {format_speedup(sp_ct_pt):>12} {format_speedup(sp_ct_q):>10}")
        print()

    return all_speedups


# ============================================================================
# Backward Benchmark
# ============================================================================

def bench_backward(configs, quack_module_cls, cutile_module_cls, peak_bw):
    """Benchmark the backward pass through autograd."""
    print_header("BACKWARD PASS (fwd+bwd via autograd)", 130)
    hdr = (
        f"{'Config':<30} {'Provider':<10} {'Latency':>10} "
        f"{'BW GB/s':>10} {'BW%':>6} {'vs PyTorch':>12} {'vs Quack':>10}"
    )
    print(hdr)
    print("-" * 130)

    all_speedups = []

    for cfg in configs:
        clear_cuda_state()

        # Build modules
        pt_mod = torch.nn.RMSNorm(cfg.N, eps=1e-6).to(dtype=cfg.dtype, device="cuda")
        q_mod = quack_module_cls(cfg.N, eps=1e-6).to("cuda")
        ct_mod = cutile_module_cls(cfg.N, eps=1e-6).to("cuda")

        total_bytes = cfg.bytes_fwd + cfg.bytes_bwd

        def run_fwd_bwd(mod):
            x = torch.randn(cfg.M, cfg.N, device="cuda", dtype=cfg.dtype, requires_grad=True)
            y = mod(x)
            y.sum().backward()
            return

        lat_pt = benchmark_fn(lambda: run_fwd_bwd(pt_mod), warmup=30, iterations=80)
        bw_pt = compute_throughput_gbps(total_bytes, lat_pt)
        ut_pt = compute_bandwidth_utilization(bw_pt, peak_bw)

        lat_q = benchmark_fn(lambda: run_fwd_bwd(q_mod), warmup=30, iterations=80)
        bw_q = compute_throughput_gbps(total_bytes, lat_q)
        ut_q = compute_bandwidth_utilization(bw_q, peak_bw)
        sp_q = lat_pt / lat_q

        lat_ct = benchmark_fn(lambda: run_fwd_bwd(ct_mod), warmup=30, iterations=80)
        bw_ct = compute_throughput_gbps(total_bytes, lat_ct)
        ut_ct = compute_bandwidth_utilization(bw_ct, peak_bw)
        sp_ct_pt = lat_pt / lat_ct
        sp_ct_q = lat_q / lat_ct

        all_speedups.append(sp_ct_q)

        tag = str(cfg)
        print(f"{tag:<30} {'PyTorch':<10} {lat_pt:>8.1f}us {bw_pt:>8.1f}GB/s {ut_pt:>5.1f}%")
        print(f"{'':<30} {'Quack':<10} {lat_q:>8.1f}us {bw_q:>8.1f}GB/s {ut_q:>5.1f}% {format_speedup(sp_q):>12}")
        print(f"{'':<30} {'CuTile':<10} {lat_ct:>8.1f}us {bw_ct:>8.1f}GB/s {ut_ct:>5.1f}% {format_speedup(sp_ct_pt):>12} {format_speedup(sp_ct_q):>10}")
        print()

    return all_speedups


# ============================================================================
# Main
# ============================================================================

def main():
    # ---- imports ----
    from bastile.ops.rms_norm import (
        FastCuteDSLRMSNorm as QuackRMSNorm,
        FastRMSNormFunction,
    )
    from bastile.ops.rms_norm_cutile import (
        CuTileRMSNorm,
        rms_norm as cutile_rms_norm,
        warmup_rms_norm as cutile_warmup,
    )

    def quack_rms_norm(x, w, eps):
        return FastRMSNormFunction.apply(x, w, eps)

    print("=" * 130)
    print("  RMSNorm Micro-Benchmark: Quack (cuteDSL) vs CuTile vs PyTorch")
    print("=" * 130)
    print_gpu_info()
    peak_bw = get_peak_bandwidth()

    # ---- JIT warmup ----
    print("\nWarming up Quack kernels (forward only) …")
    for hs in [2048, 3584, 4096, 5120, 8192]:
        x = torch.randn(4, hs, dtype=torch.bfloat16, device="cuda")
        w = torch.ones(hs, dtype=torch.bfloat16, device="cuda")
        for _ in range(3):
            _ = quack_rms_norm(x, w, 1e-6)
        torch.cuda.synchronize()
    print("Warming up CuTile kernels …")
    for hs in [2048, 3584, 4096, 5120, 8192]:
        cutile_warmup(hs)
    print("Done.\n")

    # ---- Configs: cover realistic LLM hidden sizes ----
    configs = [
        # Small rows (decode-like)
        RMSNormConfig(M=256,   N=2048,  dtype=torch.bfloat16),
        RMSNormConfig(M=256,   N=3584,  dtype=torch.bfloat16),
        RMSNormConfig(M=256,   N=4096,  dtype=torch.bfloat16),
        RMSNormConfig(M=256,   N=5120,  dtype=torch.bfloat16),
        RMSNormConfig(M=256,   N=8192,  dtype=torch.bfloat16),
        # Medium rows (short prefill)
        RMSNormConfig(M=2048,  N=2048,  dtype=torch.bfloat16),
        RMSNormConfig(M=2048,  N=3584,  dtype=torch.bfloat16),
        RMSNormConfig(M=2048,  N=4096,  dtype=torch.bfloat16),
        RMSNormConfig(M=2048,  N=5120,  dtype=torch.bfloat16),
        RMSNormConfig(M=2048,  N=8192,  dtype=torch.bfloat16),
        # Large rows (long prefill / training)
        RMSNormConfig(M=8192,  N=3584,  dtype=torch.bfloat16),
        RMSNormConfig(M=8192,  N=4096,  dtype=torch.bfloat16),
        RMSNormConfig(M=16384, N=3584,  dtype=torch.bfloat16),
        RMSNormConfig(M=16384, N=4096,  dtype=torch.bfloat16),
        # FP16 for comparison
        RMSNormConfig(M=2048,  N=4096,  dtype=torch.float16),
        RMSNormConfig(M=8192,  N=4096,  dtype=torch.float16),
    ]

    fwd_speedups = bench_forward(configs, quack_rms_norm, cutile_rms_norm, peak_bw)

    # Note: Quack backward is currently broken (stride mismatch in quack library).
    # Skip the full fwd+bwd comparison; CuTile backward works fine independently.
    bwd_speedups = []

    # ---- Summary ----
    print_header("SUMMARY", 130)
    if fwd_speedups:
        avg = sum(fwd_speedups) / len(fwd_speedups)
        print(f"\nForward CuTile vs Quack:")
        print(f"  Average: {format_speedup(avg)}")
        print(f"  Best:    {format_speedup(max(fwd_speedups))}")
        print(f"  Worst:   {format_speedup(min(fwd_speedups))}")
    if bwd_speedups:
        avg = sum(bwd_speedups) / len(bwd_speedups)
        print(f"\nFwd+Bwd CuTile vs Quack:")
        print(f"  Average: {format_speedup(avg)}")
        print(f"  Best:    {format_speedup(max(bwd_speedups))}")
        print(f"  Worst:   {format_speedup(min(bwd_speedups))}")
    print("\n" + "=" * 130)


if __name__ == "__main__":
    main()
