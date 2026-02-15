"""
End-to-end benchmarks - full training runs with patched kernels.

Benchmarks:
- qwen_8b_seqlen: Qwen3-8B seq length sweep (PyTorch vs Liger vs Bastile)
- qwen_8b_fsdp: Qwen3-8B FSDP multi-GPU benchmark
"""

from .qwen_8b_seqlen import main as benchmark_qwen3_8b


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
    "run_all",
]
