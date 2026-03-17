"""Kernel benchmarks - individual kernel performance vs PyTorch."""


def benchmark_rms_norm():
    from .rms_norm import main

    return main()


def benchmark_rope():
    from .rope import main

    return main()


def benchmark_swiglu():
    from .swiglu import main

    return main()


def run_all():
    """Run all kernel benchmarks."""
    print("=" * 80)
    print("Running All Kernel Benchmarks")
    print("=" * 80)

    benchmark_rms_norm()
    benchmark_swiglu()
    benchmark_rope()

    print("\n" + "=" * 80)
    print("All kernel benchmarks complete!")
    print("=" * 80)


__all__ = [
    "benchmark_rms_norm",
    "benchmark_rope",
    "benchmark_swiglu",
    "run_all",
]
