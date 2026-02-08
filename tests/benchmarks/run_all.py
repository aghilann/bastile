"""
Run all benchmarks.

Usage:
    python -m tests.benchmarks.run_all [--kernel] [--e2e]
    
Options:
    --kernel  Run only kernel benchmarks
    --e2e     Run only e2e benchmarks
    (no args) Run all benchmarks
"""

import sys


def main():
    args = sys.argv[1:]
    
    if "--kernel" in args:
        from . import kernel
        kernel.run_all()
    elif "--e2e" in args:
        from . import e2e
        e2e.run_all()
    else:
        from . import run_all
        run_all()


if __name__ == "__main__":
    main()
