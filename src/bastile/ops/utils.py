"""Common utilities for kernel implementations."""

import torch

_sm_count: int = 0


def get_sm_count() -> int:
    """Return the number of streaming multiprocessors (cached after first call)."""
    global _sm_count
    if _sm_count == 0:
        _sm_count = torch.cuda.get_device_properties("cuda").multi_processor_count
    return _sm_count


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def ceildiv(a: int, b: int) -> int:
    """Ceiling division: returns ceil(a / b)."""
    return -(a // -b)
