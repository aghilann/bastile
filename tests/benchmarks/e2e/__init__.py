"""
End-to-end benchmarks - full training runs with patched kernels.

Benchmarks:
- qwen_8b_seqlen: Qwen3-8B seq length sweep (PyTorch vs Liger vs Bastile)
- comparison_small: Qwen3-8B HuggingFace vs Liger vs Bastile (single config)
- profile_kernels: Kernel-level profiling with torch.profiler
"""

from .qwen_8b_seqlen import main as benchmark_qwen3_8b
from .comparison_small import main as benchmark_comparison_small


def run_all():
    """Run all e2e benchmarks."""
    print("=" * 80)
    print("Running All E2E Benchmarks")
    print("=" * 80)

    benchmark_qwen3_8b()

    print("\n" + "=" * 80)
    print("All E2E benchmarks complete!")
    print("=" * 80)


__all__ = [
    "benchmark_qwen3_8b",
    "benchmark_comparison_small",
    "run_all",
]
