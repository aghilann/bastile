"""
Bastile Operations - kernel implementations with backward passes.

Supported models:
- Qwen3: RMSNorm (CuTile), SwiGLU (CuTile), RoPE (CuTile),
         Fused Linear Cross-Entropy (quack)

This module automatically registers all available patches when imported.
"""

from . import rms_norm  # CuTile RMSNorm (persistent fwd + persistent bwd)
from . import swiglu  # CuTile SwiGLU
from . import rope  # CuTile RoPE
from . import moe_gate  # CuTile MoE expert gate (GPT-OSS)
from . import moe_experts  # Fused MoE experts (grouped_mm / CuTile GEMM)
from . import moe_router  # Fused MoE router (softmax+topk) + load-balancing loss

__all__ = ['rms_norm', 'swiglu', 'rope', 'moe_gate', 'moe_experts', 'moe_router']
