"""
End-to-end benchmarks - full training runs with patched kernels.

Benchmarks:
- qwen3: Qwen3 finetuning benchmark
- gpt_oss: GPT-OSS finetuning benchmark
- comparison: Qwen3 HuggingFace vs Liger vs Bastile
"""

from .qwen3 import main as benchmark_qwen3
from .gpt_oss import main as benchmark_gpt_oss
from .comparison import main as benchmark_comparison


def run_all():
    """Run all e2e benchmarks."""
    print("=" * 80)
    print("Running All E2E Benchmarks")
    print("=" * 80)
    
    benchmark_qwen3()
    benchmark_gpt_oss()
    benchmark_comparison()
    
    print("\n" + "=" * 80)
    print("All E2E benchmarks complete!")
    print("=" * 80)


__all__ = [
    "benchmark_qwen3",
    "benchmark_gpt_oss",
    "benchmark_comparison",
    "run_all",
]
