"""
Bastile - Monkey-patch PyTorch with optimized CuTile kernels for training.

Usage:
    import bastile
    bastile.apply()  # Patches common PyTorch ops with CuTile kernels

    # Selectively:
    bastile.apply(rms_norm=True, swiglu=True, rope=True)

    # For HuggingFace models:
    bastile.apply_to_model(model)  # Patches a loaded model

    # Clear autotuning cache:
    bastile.clear_autotune_cache()
"""

from .autotune import clear_cache as clear_autotune_cache
from .autotune import warmup_all_kernels
from .core import apply, apply_to_model, get_patched_ops, reset
from .registry import list_patches, register_patch

__version__ = "0.1.0"
__all__ = [
    "apply",
    "apply_to_model",
    "clear_autotune_cache",
    "get_patched_ops",
    "list_patches",
    "register_patch",
    "reset",
    "warmup_all_kernels",
]
