"""Kernel warmup and cache management for Bastile."""

from pathlib import Path

import torch

_CACHE_DIR = Path.home() / ".cache" / "bastile"


def clear_cache():
    """Clear disk-based kernel caches."""
    if not _CACHE_DIR.exists():
        return
    for f in _CACHE_DIR.glob("*.json"):
        try:
            f.unlink()
        except OSError:
            pass


def warmup_all_kernels(
    batch_size: int = 4,
    seq_len: int = 512,
    hidden_size: int = 2048,
    intermediate_size: int = 5504,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """
    Warmup all registered CuTile kernels with typical tensor shapes.

    Call this once after bastile.apply() to pre-compile all kernels and
    avoid JIT compilation overhead during training.
    """
    if not torch.cuda.is_available():
        return

    try:
        from .ops.rms_norm import rms_norm
        from .ops.swiglu import swiglu

        BT = batch_size * seq_len

        x = torch.randn(BT, hidden_size, dtype=dtype, device=device)
        weight = torch.ones(hidden_size, dtype=dtype, device=device)
        for _ in range(3):
            _ = rms_norm(x, weight, eps=1e-6)
        torch.cuda.synchronize()

        gate = torch.randn(BT, intermediate_size, dtype=dtype, device=device)
        up = torch.randn(BT, intermediate_size, dtype=dtype, device=device)
        for _ in range(3):
            _ = swiglu(gate, up)
        torch.cuda.synchronize()

    except Exception as e:
        import logging

        logging.debug(f"Warmup skipped: {e}")
