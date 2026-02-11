"""
Bastile RMSNorm - Fast cuteDSL implementation via quack.

Wraps quack's RMSNorm with reduced CPU dispatch overhead by:
1. Directly invoking compiled CuTe DSL kernels (bypasses torch.library.custom_op dispatch)
2. Caching device properties (sm_count)
3. Minimizing Python overhead on the hot path (no assertions, no repeated lookups)

The compiled GPU kernels are identical to quack's - only the Python dispatch path is faster.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from ..registry import register_patch
from quack.rmsnorm import (
    rmsnorm as quack_rmsnorm,
    _rmsnorm_fwd,
    _rmsnorm_bwd,
    _get_sm_count,
)
from quack.cute_dsl_utils import torch2cute_dtype_map

_sm_count_cache: Dict[Tuple[int, torch.device], int] = {}


def _cached_sm_count(N: int, device: torch.device) -> int:
    key = (N, device)
    if key not in _sm_count_cache:
        _sm_count_cache[key] = _get_sm_count(N, device)
    return _sm_count_cache[key]


_fwd_cache = _rmsnorm_fwd.compile_cache
_bwd_cache = _rmsnorm_bwd.compile_cache


class FastRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, weight, eps):
        x_shape_og = hidden_states.shape
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        out = torch.empty_like(x)
        rstd = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)

        N = x.shape[1]
        dtype = torch2cute_dtype_map[x.dtype]
        w_dtype = torch2cute_dtype_map[weight.dtype]
        fwd_key = (dtype, dtype, None, w_dtype, None, None, N, True, False, False)

        kernel = _fwd_cache.get(fwd_key)
        if kernel is not None:
            kernel(x, weight, None, None, out, None, rstd, None, eps)
        else:
            _rmsnorm_fwd(x, weight, out, None, rstd, None, None, None, eps, False)

        ctx.save_for_backward(x, weight, rstd)
        ctx.x_shape_og = x_shape_og
        return out.reshape(x_shape_og)

    @staticmethod
    def backward(ctx, dout):
        x, weight, rstd = ctx.saved_tensors
        x_shape_og = ctx.x_shape_og
        N = x.shape[1]
        dout = dout.reshape(-1, dout.shape[-1])
        dx = torch.empty_like(x)
        device = x.device
        sm_count = _cached_sm_count(N, device)
        dw_partial = torch.empty(sm_count, N, device=device, dtype=torch.float32)

        dtype = torch2cute_dtype_map[x.dtype]
        w_dtype = torch2cute_dtype_map[weight.dtype]
        bwd_key = (N, dtype, dtype, dtype, w_dtype, False, None, None)

        kernel = _bwd_cache.get(bwd_key)
        if kernel is not None:
            kernel(x, weight, dout, None, rstd, dx, dw_partial, None, None, sm_count)
        else:
            _rmsnorm_bwd(x, weight, dout, rstd, dx, dw_partial, None, None, None, sm_count)

        dw = dw_partial.sum(dim=0).to(weight.dtype)
        return dx.view(x_shape_og), dw, None


class FastCuteDSLRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return FastRMSNormFunction.apply(
            hidden_states, self.weight, self.variance_epsilon
        )

    def extra_repr(self) -> str:
        return f"{self.weight.shape[0]}, eps={self.variance_epsilon}"


def warmup_rmsnorm(
    hidden_size: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    x = torch.randn(2, hidden_size, dtype=dtype, device=device, requires_grad=True)
    w = torch.ones(hidden_size, dtype=dtype, device=device, requires_grad=True)
    out = quack_rmsnorm(x, w, eps=1e-6)
    out.sum().backward()
    _cached_sm_count(hidden_size, torch.device(device))
    torch.cuda.synchronize()


register_patch(
    name="rms_norm_qwen3",
    description="Fast cuteDSL RMSNorm for Qwen3 (reduced CPU dispatch overhead)",
    target_module="transformers.models.qwen3.modeling_qwen3",
    target_attr="Qwen3RMSNorm",
    replacement=FastCuteDSLRMSNorm,
    has_backward=True,
    priority=10,
    models=["qwen3"],
)
