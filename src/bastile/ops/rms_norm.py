"""
Bastile RMSNorm - cuteDSL implementation from quack.

High-performance RMSNorm using cuteDSL kernels (NVIDIA CUTLASS DSL).
Uses quack's RMSNorm which leverages Thread Block Clusters, online reduction,
and SM-count-based persistent backward kernels for Blackwell GPUs.

Key optimizations (from quack):
1. cuteDSL kernel with Thread Block Clusters for cooperative reductions
2. SM-count-based persistent backward grid for efficient dW accumulation
3. Single-pass forward with RSTD caching
4. cp_async for overlapping gmem loads with computation
5. Vectorized memory access (128-bit aligned for Blackwell)
"""

import torch
import torch.nn as nn
from typing import Optional

from ..registry import register_patch

# Import quack's cuteDSL RMSNorm directly
from quack.rmsnorm import rmsnorm as quack_rmsnorm, RMSNormFunction as QuackRMSNormFunction


# ============================================================================
# Module - Drop-in replacement for Qwen3RMSNorm
# ============================================================================

class CuteDSLRMSNorm(nn.Module):
    """Drop-in replacement for Qwen3RMSNorm using quack's cuteDSL kernels.

    Uses quack's high-performance RMSNorm implementation which leverages:
    - cuteDSL (NVIDIA CUTLASS DSL) for kernel generation
    - Thread Block Clusters for cooperative reductions
    - SM-count-based persistent backward kernel
    - 128-bit vectorized memory access
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return quack_rmsnorm(hidden_states, self.weight, eps=self.variance_epsilon)

    def extra_repr(self) -> str:
        return f"{self.weight.shape[0]}, eps={self.variance_epsilon}"


# Register patch
register_patch(
    name="rms_norm_qwen3",
    description="cuteDSL RMSNorm for Qwen3 (from quack)",
    target_module="transformers.models.qwen3.modeling_qwen3",
    target_attr="Qwen3RMSNorm",
    replacement=CuteDSLRMSNorm,
    has_backward=True,
    priority=10,
    models=["qwen3"],
)
