"""
Run all ops unit tests.
"""

from . import test_fused_linear_cross_entropy, test_rms_norm, test_rope, test_swiglu


def run_all():
    print("=" * 70)
    print("Bastile Ops Unit Tests")
    print("=" * 70)
    print()

    test_rms_norm.run_all()
    test_swiglu.run_all()
    test_rope.run_all()
    test_fused_linear_cross_entropy.run_all()

    print("=" * 70)
    print("ALL OPS TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    run_all()
