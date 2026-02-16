"""
Bastile Operations - kernel implementations with backward passes.

Supported models:
- Qwen3: RMSNorm (CuTile), SwiGLU (CuTile), RoPE (CuTile),
         Fused Linear Cross-Entropy (CuTile)

This module automatically registers all available patches when imported.
"""

from . import (
    rms_norm,  # CuTile RMSNorm (persistent fwd + persistent bwd)
    rope,  # CuTile RoPE
    swiglu,  # CuTile SwiGLU
)

__all__ = ["rms_norm", "rope", "swiglu"]
