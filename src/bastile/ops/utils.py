"""Common utilities for kernel implementations."""


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def ceildiv(a: int, b: int) -> int:
    """Ceiling division: returns ceil(a / b)."""
    return -(a // -b)
