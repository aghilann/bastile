"""Bastile benchmark helpers and entrypoints."""

from .utils import (
    E2EBenchmarkResult,
    KernelBenchmarkResult,
    Timer,
    benchmark_fn,
    clear_cuda_state,
    get_gpu_info,
    get_peak_bandwidth,
)


def run_all_kernel_benchmarks():
    """Run all kernel benchmarks."""
    from . import kernel

    kernel.run_all()


def run_all_e2e_benchmarks():
    """Run all e2e benchmarks."""
    from . import e2e

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
    "E2EBenchmarkResult",
    "KernelBenchmarkResult",
    "Timer",
    "benchmark_fn",
    "clear_cuda_state",
    "get_gpu_info",
    "get_peak_bandwidth",
    "run_all",
    "run_all_e2e_benchmarks",
    "run_all_kernel_benchmarks",
]
