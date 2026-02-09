"""
Bastile Operations - CuTile kernel implementations with backward passes.

Supported models:
- Qwen3: RMSNorm, SwiGLU, RoPE, Cross-Entropy

This module automatically registers all available patches when imported.
"""

from . import rms_norm
from . import swiglu
from . import rope
from . import cross_entropy

__all__ = ['rms_norm', 'swiglu', 'rope', 'cross_entropy']
