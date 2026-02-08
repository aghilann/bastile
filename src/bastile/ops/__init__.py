"""
Bastile Operations - CuTile kernel implementations with backward passes.

Supported models:
- Qwen3: RMSNorm, SwiGLU, RoPE
- GPT-OSS: RMSNorm, Fused GEGLU MoE Experts

This module automatically registers all available patches when imported.
"""

from . import rms_norm
from . import swiglu
from . import rope
from . import gpt_oss_moe
from . import cross_entropy

__all__ = ['rms_norm', 'swiglu', 'rope', 'gpt_oss_moe', 'cross_entropy']
