"""
Bastile Operations - kernel implementations with backward passes.

Supported models:
- Qwen3: RMSNorm (CuTile), SwiGLU (CuTile), RoPE (CuTile),
         Fused Linear Cross-Entropy (quack)

This module automatically registers all available patches when imported.
"""

from . import rms_norm_cutile as rms_norm  # CuTile RMSNorm (autotuned gather/TMA)
from . import swiglu  # CuTile SwiGLU
from . import rope  # CuTile RoPE

__all__ = ['rms_norm', 'swiglu', 'rope']
