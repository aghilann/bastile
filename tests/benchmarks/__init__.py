"""
Bastile Benchmarks

Structure:
- kernel/  : Individual kernel benchmarks vs PyTorch
- e2e/     : End-to-end training benchmarks with patched models
- utils.py : Shared utilities for benchmarking
"""

from . import kernel
from . import e2e
from .utils import (
    benchmark_fn,
    benchmark_fn_with_stats,
    get_gpu_info,
    get_peak_bandwidth,
    clear_cuda_state,
    Timer,
    KernelBenchmarkResult,
    E2EBenchmarkResult,
)


def run_all_kernel_benchmarks():
    """Run all kernel benchmarks."""
    kernel.run_all()


def run_all_e2e_benchmarks():
    """Run all e2e benchmarks."""
    e2e.run_all()


def run_all():
    """Run all benchmarks."""
    print("=" * 80)
    print("Bastile Complete Benchmark Suite")
    print("=" * 80)
    
    run_all_kernel_benchmarks()
    run_all_e2e_benchmarks()
    
    print("\n" + "=" * 80)
    print("All benchmarks complete!")
    print("=" * 80)


__all__ = [
    "kernel",
    "e2e",
    "benchmark_fn",
    "benchmark_fn_with_stats",
    "get_gpu_info",
    "get_peak_bandwidth",
    "clear_cuda_state",
    "Timer",
    "KernelBenchmarkResult",
    "E2EBenchmarkResult",
    "run_all_kernel_benchmarks",
    "run_all_e2e_benchmarks",
    "run_all",
]
