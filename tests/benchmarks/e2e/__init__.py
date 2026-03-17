"""
End-to-end benchmarks - full training runs with patched kernels.

Benchmarks:
- qwen3_seqlen: Qwen3 seq length sweep (PyTorch vs Liger vs Bastile)
- qwen3_5_seqlen: Qwen3.5 seq length sweep (PyTorch vs Liger vs Bastile)
"""


def run_all():
    """Run all e2e benchmarks."""
    from .qwen3_5_seqlen import main as benchmark_qwen3_5
    from .qwen3_seqlen import main as benchmark_qwen3

    print("=" * 80)
    print("Running All E2E Benchmarks")
    print("=" * 80)

    benchmark_qwen3()
    benchmark_qwen3_5()

    print("\n" + "=" * 80)
    print("All E2E benchmarks complete!")
    print("=" * 80)


__all__ = [
    "benchmark_qwen3",
    "benchmark_qwen3_5",
    "run_all",
]


def benchmark_qwen3():
    from .qwen3_seqlen import main

    return main()


def benchmark_qwen3_5():
    from .qwen3_5_seqlen import main

    return main()
