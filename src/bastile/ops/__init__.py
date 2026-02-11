"""
Bastile Operations - kernel implementations with backward passes.

Supported models:
- Qwen3: RMSNorm (cuteDSL/quack), SwiGLU (Triton), RoPE (CuTile),
         Fused Linear Cross-Entropy (quack)

This module automatically registers all available patches when imported.
"""

from . import rms_norm  # cuteDSL RMSNorm (from quack)
from . import swiglu  # Triton SwiGLU
from . import rope  # CuTile RoPE

__all__ = ['rms_norm', 'swiglu', 'rope']
