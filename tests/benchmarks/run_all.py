"""
Run all E2E benchmarks.
"""

import sys


def run_all():
    print("=" * 70)
    print("Bastile E2E Benchmarks")
    print("=" * 70)
    print()
    print("Available benchmarks:")
    print("  1. benchmark_e2e.py    - Compare Qwen3 and GPT-OSS (30s each)")
    print("  2. benchmark_qwen3.py  - Detailed Qwen3 benchmark (60s)")
    print("  3. benchmark_gpt_oss.py - Detailed GPT-OSS benchmark (60s)")
    print()
    print("Run individually with:")
    print("  python -m tests.benchmarks.benchmark_e2e")
    print("  python -m tests.benchmarks.benchmark_qwen3")
    print("  python -m tests.benchmarks.benchmark_gpt_oss")
    print()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        print("Running E2E benchmark...")
        from . import benchmark_e2e
        benchmark_e2e.main()


if __name__ == "__main__":
    run_all()
