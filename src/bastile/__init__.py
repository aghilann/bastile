"""
Bastile - Monkey-patch PyTorch with optimized CuTile kernels for training.

Usage:
    import bastile
    bastile.apply()  # Patches common PyTorch ops with CuTile kernels
    
    # Or selectively:
    bastile.apply(rms_norm=True, swiglu=True, rope=False)
    
    # For HuggingFace models:
    bastile.apply_to_model(model)  # Patches a loaded model
"""

from .core import apply, apply_to_model, reset, get_patched_ops
from .registry import register_patch, list_patches

__version__ = "0.1.0"
__all__ = [
    "apply",
    "apply_to_model", 
    "reset",
    "get_patched_ops",
    "register_patch",
    "list_patches",
]
