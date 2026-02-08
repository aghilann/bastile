"""
RMSNorm Kernel Benchmark.

Compares CuTile RMSNorm vs Liger Kernel vs PyTorch reference.
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
class RMSNormConfig:
    batch_size: int
    seq_len: int
    hidden_size: int
    dtype: torch.dtype
    use_static_persistent: Optional[bool] = None

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.batch_size, self.seq_len, self.hidden_size)

    @property
    def total_elements(self) -> int:
        return self.batch_size * self.seq_len * self.hidden_size

    @property
    def element_size(self) -> int:
        if self.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            return 1
        return self.dtype.itemsize

    @property
    def bytes_accessed(self) -> int:
        return 2 * self.total_elements * self.element_size

    def __str__(self) -> str:
        dtype_str = str(self.dtype).split('.')[-1]
        mode = ""
        if self.use_static_persistent is True:
            mode = " [persistent]"
        elif self.use_static_persistent is False:
            mode = " [gather]"
        return f"({self.batch_size}, {self.seq_len}, {self.hidden_size}) {dtype_str}{mode}"


def pytorch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """PyTorch reference implementation of RMSNorm."""
    x_float = x.float()
    variance = x_float.pow(2).mean(-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(variance + eps)
    return (weight.float() * x_normed).to(x.dtype)


def run_single_benchmark(config: RMSNormConfig, cutile_rms_norm, liger_rms_norm, peak_bw: float):
    """Run benchmark for a single configuration."""
    clear_cuda_state()

    # Skip FP8 for Liger (not supported)
    skip_liger = config.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)

    if config.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        x = torch.randn(*config.shape, device="cuda", dtype=torch.float16).to(config.dtype)
        weight = torch.ones(config.hidden_size, device="cuda", dtype=torch.float16).to(config.dtype)
    else:
        x = torch.randn(*config.shape, device="cuda", dtype=config.dtype)
        weight = torch.ones(config.hidden_size, device="cuda", dtype=config.dtype)
    
    eps = 1e-6

    # Benchmark PyTorch
    pytorch_latency = benchmark_fn(lambda: pytorch_rms_norm(x, weight, eps))
    pytorch_throughput = compute_throughput_gbps(config.bytes_accessed, pytorch_latency)
    pytorch_util = compute_bandwidth_utilization(pytorch_throughput, peak_bw)

    # Benchmark Liger Kernel
    if not skip_liger:
        liger_latency = benchmark_fn(lambda: liger_rms_norm(x, weight, eps))
        liger_throughput = compute_throughput_gbps(config.bytes_accessed, liger_latency)
        liger_util = compute_bandwidth_utilization(liger_throughput, peak_bw)
    else:
        liger_latency = liger_throughput = liger_util = None

    # Benchmark CuTile
    cutile_latency = benchmark_fn(lambda: cutile_rms_norm(x, weight, eps, config.use_static_persistent))
    cutile_throughput = compute_throughput_gbps(config.bytes_accessed, cutile_latency)
    cutile_util = compute_bandwidth_utilization(cutile_throughput, peak_bw)

    return {
        "pytorch": {"latency": pytorch_latency, "throughput": pytorch_throughput, "util": pytorch_util},
        "liger": {"latency": liger_latency, "throughput": liger_throughput, "util": liger_util} if not skip_liger else None,
        "cutile": {"latency": cutile_latency, "throughput": cutile_throughput, "util": cutile_util},
    }


def jit_warmup(cutile_rms_norm, liger_rms_norm):
    """Pre-compile all kernel variants."""
    print("JIT compiling kernels...")
    
    warmup_configs = [
        (4, 256, 2048, torch.float16),
        (4, 256, 2048, torch.bfloat16),
        (32, 512, 2048, torch.float16),
        (32, 512, 2048, torch.bfloat16),
        (4, 256, 2048, torch.float32),
    ]
    
    for batch, seq, hidden, dtype in warmup_configs:
        x = torch.randn(batch, seq, hidden, device="cuda", dtype=dtype)
        w = torch.ones(hidden, device="cuda", dtype=dtype)
        # Warmup CuTile
        for mode in [False, True]:
            for _ in range(3):
                _ = cutile_rms_norm(x, w, 1e-6, mode)
        # Warmup Liger
        for _ in range(3):
            _ = liger_rms_norm(x, w, 1e-6)
        torch.cuda.synchronize()
    
    # Try FP8 warmup for CuTile
    try:
        for mode in [False, True]:
            x = torch.randn(4, 256, 2048, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
            w = torch.ones(2048, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
            for _ in range(3):
                _ = cutile_rms_norm(x, w, 1e-6, mode)
            torch.cuda.synchronize()
        print("  FP8 support: YES (CuTile only)")
    except Exception:
        print("  FP8 support: NO")
    
    print("JIT compilation complete.\n")


def run_benchmark_section(title: str, configs: List[RMSNormConfig], cutile_rms_norm, liger_rms_norm, peak_bw: float) -> List[dict]:
    """Run benchmarks for a section of configs."""
    print_header(title, 115)
    print(f"{'Config':<40} {'Provider':<10} {'Latency':>10} {'Thruput':>10} {'BW%':>6} {'vs PyTorch':>12} {'vs Liger':>10}")
    print("-" * 115)
    
    results = []
    for config in configs:
        try:
            result = run_single_benchmark(config, cutile_rms_norm, liger_rms_norm, peak_bw)
            results.append((config, result))
            
            py = result["pytorch"]
            lg = result["liger"]
            ct = result["cutile"]
            
            # Print PyTorch
            print(f"{str(config):<40} {'PyTorch':<10} {py['latency']:>8.1f}us {py['throughput']:>8.1f}GB/s {py['util']:>5.1f}%")
            
            # Print Liger if available
            if lg:
                speedup_vs_py = py['latency'] / lg['latency']
                print(f"{'':<40} {'Liger':<10} {lg['latency']:>8.1f}us {lg['throughput']:>8.1f}GB/s {lg['util']:>5.1f}% {format_speedup(speedup_vs_py):>12}")
            
            # Print CuTile
            speedup_vs_py = py['latency'] / ct['latency']
            speedup_vs_lg = lg['latency'] / ct['latency'] if lg else None
            lg_str = format_speedup(speedup_vs_lg) if speedup_vs_lg else "N/A"
            print(f"{'':<40} {'CuTile':<10} {ct['latency']:>8.1f}us {ct['throughput']:>8.1f}GB/s {ct['util']:>5.1f}% {format_speedup(speedup_vs_py):>12} {lg_str:>10}")
            print()
        except Exception as e:
            print(f"{str(config):<40} ERROR: {e}\n")
    
    return results


def main():
    from bastile.ops.rms_norm import rms_norm as cutile_rms_norm
    from liger_kernel.ops import LigerRMSNormFunction
    
    def liger_rms_norm(x, w, eps):
        return LigerRMSNormFunction.apply(x, w, eps)

    print("=" * 115)
    print("RMSNorm Kernel Benchmark: CuTile vs Liger Kernel vs PyTorch")
    print("=" * 115)

    print_gpu_info()
    peak_bw = get_peak_bandwidth()
    
    jit_warmup(cutile_rms_norm, liger_rms_norm)
    
    all_results = []

    # Section 1: FP16
    fp16_configs = [
        RMSNormConfig(8, 256, 2048, torch.float16, use_static_persistent=False),
        RMSNormConfig(16, 512, 2048, torch.float16, use_static_persistent=False),
        RMSNormConfig(8, 256, 4096, torch.float16, use_static_persistent=False),
        RMSNormConfig(32, 512, 2048, torch.float16, use_static_persistent=True),
        RMSNormConfig(32, 512, 4096, torch.float16, use_static_persistent=True),
    ]
    all_results.extend(run_benchmark_section("FLOAT16 BENCHMARKS", fp16_configs, cutile_rms_norm, liger_rms_norm, peak_bw))

    # Section 2: BFloat16
    bf16_configs = [
        RMSNormConfig(4, 256, 2048, torch.bfloat16, use_static_persistent=False),
        RMSNormConfig(4, 256, 4096, torch.bfloat16, use_static_persistent=False),
        RMSNormConfig(32, 512, 2048, torch.bfloat16, use_static_persistent=True),
        RMSNormConfig(32, 512, 4096, torch.bfloat16, use_static_persistent=True),
        RMSNormConfig(64, 512, 4096, torch.bfloat16, use_static_persistent=True),
    ]
    all_results.extend(run_benchmark_section("BFLOAT16 BENCHMARKS", bf16_configs, cutile_rms_norm, liger_rms_norm, peak_bw))

    # Section 3: Float32
    fp32_configs = [
        RMSNormConfig(4, 256, 2048, torch.float32, use_static_persistent=False),
        RMSNormConfig(4, 256, 4096, torch.float32, use_static_persistent=False),
        RMSNormConfig(32, 512, 2048, torch.float32, use_static_persistent=True),
        RMSNormConfig(32, 512, 4096, torch.float32, use_static_persistent=True),
    ]
    all_results.extend(run_benchmark_section("FLOAT32 BENCHMARKS", fp32_configs, cutile_rms_norm, liger_rms_norm, peak_bw))

    # Summary
    print_header("SUMMARY", 115)

    if all_results:
        cutile_vs_pytorch = []
        cutile_vs_liger = []
        liger_vs_pytorch = []
        
        for (config, result) in all_results:
            py = result["pytorch"]["latency"]
            ct = result["cutile"]["latency"]
            cutile_vs_pytorch.append(py / ct)
            
            if result["liger"]:
                lg = result["liger"]["latency"]
                cutile_vs_liger.append(lg / ct)
                liger_vs_pytorch.append(py / lg)

        print(f"\nCuTile vs PyTorch:")
        print(f"  Average Speedup: {sum(cutile_vs_pytorch)/len(cutile_vs_pytorch):.2f}x")
        print(f"  Max Speedup:     {max(cutile_vs_pytorch):.2f}x")
        print(f"  Min Speedup:     {min(cutile_vs_pytorch):.2f}x")
        
        if liger_vs_pytorch:
            print(f"\nLiger vs PyTorch:")
            print(f"  Average Speedup: {sum(liger_vs_pytorch)/len(liger_vs_pytorch):.2f}x")
            print(f"  Max Speedup:     {max(liger_vs_pytorch):.2f}x")
            print(f"  Min Speedup:     {min(liger_vs_pytorch):.2f}x")
        
        if cutile_vs_liger:
            print(f"\nCuTile vs Liger:")
            print(f"  Average Speedup: {sum(cutile_vs_liger)/len(cutile_vs_liger):.2f}x")
            print(f"  Max Speedup:     {max(cutile_vs_liger):.2f}x")
            print(f"  Min Speedup:     {min(cutile_vs_liger):.2f}x")

    print("\n" + "=" * 115)


if __name__ == "__main__":
    main()
