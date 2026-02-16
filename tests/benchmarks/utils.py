"""
Shared utilities for benchmarking.

Common functions used across kernel and e2e benchmarks.
"""

import gc
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch


def get_gpu_info() -> dict:
    """Get GPU information."""
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "total_memory_gb": props.total_memory / 1024**3,
        "multi_processor_count": props.multi_processor_count,
        "cuda_version": torch.version.cuda,
    }


def get_peak_bandwidth() -> float:
    """Get theoretical peak memory bandwidth for current GPU in GB/s."""
    props = torch.cuda.get_device_properties(0)
    name = props.name.lower()
    if "b200" in name:
        return 8000.0  # GB/s (B200 HBM3e)
    elif "h100" in name:
        return 3350.0
    elif "a100" in name:
        return 2000.0
    elif "4090" in name:
        return 1000.0
    else:
        return 1000.0


def clear_cuda_state():
    """Clear CUDA memory and garbage collect."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()


def reset_peak_memory():
    """Reset peak memory stats."""
    torch.cuda.reset_peak_memory_stats()


def get_peak_memory_gb() -> float:
    """Get peak memory allocated in GB."""
    return torch.cuda.max_memory_allocated() / 1024**3


def benchmark_fn(
    fn: Callable,
    warmup: int = 50,
    iterations: int = 100,
) -> float:
    """
    Benchmark a function using CUDA events.

    Args:
        fn: Function to benchmark
        warmup: Number of warmup iterations
        iterations: Number of timed iterations

    Returns:
        Median latency in microseconds
    """
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        fn()
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # ms to us

    times.sort()
    return times[len(times) // 2]  # Median


def benchmark_fn_with_stats(
    fn: Callable,
    warmup: int = 50,
    iterations: int = 100,
) -> dict:
    """
    Benchmark a function and return detailed statistics.

    Returns:
        Dict with median, mean, min, max, p95, p99 latencies in microseconds
    """
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        fn()
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # ms to us

    times.sort()
    n = len(times)

    return {
        "median_us": times[n // 2],
        "mean_us": sum(times) / n,
        "min_us": times[0],
        "max_us": times[-1],
        "p95_us": times[int(n * 0.95)],
        "p99_us": times[int(n * 0.99)],
    }


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        torch.cuda.synchronize()
        self.end_time = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    @property
    def elapsed_sec(self) -> float:
        return self.end_time - self.start_time


def compute_throughput_gbps(
    total_bytes: int,
    latency_us: float,
) -> float:
    """Compute throughput in GB/s."""
    return total_bytes / (latency_us * 1e-6) / 1e9


def compute_bandwidth_utilization(
    throughput_gbps: float,
    peak_bw_gbps: float | None = None,
) -> float:
    """Compute bandwidth utilization as percentage."""
    if peak_bw_gbps is None:
        peak_bw_gbps = get_peak_bandwidth()
    return (throughput_gbps / peak_bw_gbps) * 100


@dataclass
class KernelBenchmarkResult:
    """Result from a kernel benchmark."""

    config_str: str
    provider: str
    latency_us: float
    throughput_gbps: float | None = None
    bandwidth_util_pct: float | None = None

    def speedup_vs(self, other: "KernelBenchmarkResult") -> float:
        """Calculate speedup compared to another result."""
        return other.latency_us / self.latency_us


@dataclass
class E2EBenchmarkResult:
    """Result from an end-to-end training benchmark."""

    name: str
    iterations: int
    total_time_sec: float
    avg_iter_ms: float
    tokens_per_sec: float
    peak_memory_gb: float
    initial_loss: float
    final_loss: float
    loss_history: list[float] | None = None

    def speedup_vs(self, other: "E2EBenchmarkResult") -> float:
        """Calculate speedup compared to another result."""
        return self.tokens_per_sec / other.tokens_per_sec

    def memory_saved_vs(self, other: "E2EBenchmarkResult") -> float:
        """Calculate memory saved compared to another result in GB."""
        return other.peak_memory_gb - self.peak_memory_gb


def print_header(title: str, width: int = 80):
    """Print a section header."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_gpu_info():
    """Print GPU information."""
    info = get_gpu_info()
    print(f"\nGPU: {info['name']}")
    print(f"CUDA Version: {info['cuda_version']}")
    print(f"Peak Bandwidth: {get_peak_bandwidth():.0f} GB/s")


def format_speedup(speedup: float) -> str:
    """Format speedup as string."""
    if speedup >= 1.0:
        return f"{speedup:.2f}x"
    else:
        return f"{1 / speedup:.2f}x slower"
