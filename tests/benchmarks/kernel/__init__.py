"""
Kernel benchmarks - individual kernel performance vs PyTorch.

Benchmarks:
- rms_norm: RMSNorm kernel
- swiglu: SwiGLU activation kernel
- rope: Rotary Position Embedding
- bench_fused_lce: Fused Linear Cross-Entropy
"""

from .rms_norm import main as benchmark_rms_norm
from .rope import main as benchmark_rope
from .swiglu import main as benchmark_swiglu


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
