"""CuTile RMSNorm — persistent kernels for forward and backward."""

import torch
import torch.nn as nn
import cuda.tile as ct

from ..registry import register_patch
from .utils import next_power_of_2, get_sm_count

ConstInt = ct.Constant[int]
ConstFloat = ct.Constant[float]
PAD_ZERO = ct.PaddingMode.ZERO


@ct.kernel(occupancy=1)
def _rms_fwd(X, W, Y, Rstd, TILE_M: ConstInt, TILE_N: ConstInt, eps: ConstFloat):
    bid = ct.bid(0)
    M, N = X.shape[0], X.shape[1]
    blocks = ct.num_blocks(0)
    upper = (M + TILE_M - 1) // TILE_M

    w = ct.astype(ct.load(W, index=(0,), shape=(TILE_N,), padding_mode=PAD_ZERO), ct.float32)
    w = ct.reshape(w, (1, TILE_N))
    rcp = ct.full((TILE_M, 1), 1.0 / N, dtype=ct.float32)
    e = ct.full((TILE_M, 1), eps, dtype=ct.float32)

    for i in range(bid, upper, blocks):
        x = ct.astype(
            ct.load(X, index=(i, 0), shape=(TILE_M, TILE_N), latency=10, padding_mode=PAD_ZERO),
            ct.float32,
        )
        r = ct.rsqrt(ct.sum(x * x, axis=1, keepdims=True) * rcp + e)
        ct.store(Rstd, index=(i,), tile=ct.reshape(r, (TILE_M,)), allow_tma=False)
        ct.store(Y, index=(i, 0), tile=ct.astype(x * r * w, X.dtype), allow_tma=False, latency=3)


@ct.kernel(occupancy=1)
def _rms_bwd(dx, dy, x, weight, Rstd, dw_partial, TILE_M: ConstInt, TILE_N: ConstInt):
    bid = ct.bid(0)
    M, N = x.shape[0], x.shape[1]
    blocks = ct.num_blocks(0)
    upper = (M + TILE_M - 1) // TILE_M

    w = ct.astype(ct.load(weight, index=(0,), shape=(TILE_N,), padding_mode=PAD_ZERO), ct.float32)
    w = ct.reshape(w, (1, TILE_N))
    rcp = ct.full((TILE_M, 1), 1.0 / N, dtype=ct.float32)
    dw_acc = ct.full((1, TILE_N), 0.0, dtype=ct.float32)

    for i in range(bid, upper, blocks):
        xt = ct.astype(
            ct.load(x, index=(i, 0), shape=(TILE_M, TILE_N), padding_mode=PAD_ZERO, latency=10),
            ct.float32,
        )
        dyt = ct.astype(
            ct.load(dy, index=(i, 0), shape=(TILE_M, TILE_N), padding_mode=PAD_ZERO, latency=10),
            ct.float32,
        )
        r = ct.reshape(
            ct.load(Rstd, index=(i,), shape=(TILE_M,), padding_mode=PAD_ZERO),
            (TILE_M, 1),
        )
        xhat = xt * r
        wdy = dyt * w
        c = ct.sum(xhat * wdy, axis=1, keepdims=True) * rcp
        ct.store(dx, index=(i, 0), tile=ct.astype((wdy - xhat * c) * r, dx.dtype),
                 allow_tma=False, latency=3)
        dw_acc = dw_acc + ct.sum(dyt * xhat, axis=0, keepdims=True)

    ct.store(dw_partial, index=(bid, 0), tile=dw_acc, allow_tma=False)


_fwd_cfg: dict = {}  # (M, N, dtype) -> (tile_m, tile_n, grid)
_bwd_cfg: dict = {}  # (M, N) -> (tile_m, tile_n, grid, N)


def _fwd_tiles(M, N):
    sms = get_sm_count()
    T = next_power_of_2(N)
    if M <= sms * 4:
        return (1, T, min(sms, M))
    if T <= 1024:
        tm = 16
    elif T >= 16384:
        tm = 2
    else:
        tm = max(2, min(8, 32768 // T))
    return (tm, T, min(sms, (M + tm - 1) // tm))


def _bwd_tiles(M, N):
    T = next_power_of_2(N)
    if T > 4096:
        tm = 1
    elif T <= 2048 or (M >= 8192 and T <= 4096):
        tm = 4
    else:
        tm = 1
    sms = get_sm_count()
    tiles = (M + tm - 1) // tm
    g = min(sms, tiles)
    if tiles <= 64:
        g = min(g, 32)
    return (tm, T, g, N)


class CuTileRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        shape = x.shape
        N = shape[-1]
        x2 = x.reshape(-1, N)
        M = x2.shape[0]

        cfg = _fwd_cfg.get((M, N, x.dtype))
        if cfg is None:
            cfg = _fwd_tiles(M, N)
            _fwd_cfg[(M, N, x.dtype)] = cfg
        tm, tn, g = cfg

        stream = torch.cuda.current_stream()
        y = torch.empty_like(x2)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)
        ct.launch(stream, (g,), _rms_fwd, (x2, weight, y, rstd, tm, tn, eps))

        ctx.save_for_backward(x, weight, rstd)
        ctx.eps = eps
        return y.view(shape)

    @staticmethod
    def backward(ctx, dy):
        x, weight, rstd = ctx.saved_tensors
        shape = x.shape
        N = shape[-1]
        x2 = x.reshape(-1, N)
        M = x2.shape[0]
        dy2 = dy.reshape(-1, N)
        if not dy2.is_contiguous():
            dy2 = dy2.contiguous()

        cfg = _bwd_cfg.get((M, N))
        if cfg is None:
            cfg = _bwd_tiles(M, N)
            _bwd_cfg[(M, N)] = cfg
        tm, T, g, No = cfg

        stream = torch.cuda.current_stream()
        dx = torch.empty_like(x2)
        dwp = torch.empty((g, T), device=x.device, dtype=torch.float32)
        ct.launch(stream, (g,), _rms_bwd, (dx, dy2, x2, weight, rstd, dwp, tm, T))

        dw = dwp.sum(0)
        if T != No:
            dw = dw[:No]
        return dx.view(shape), dw.to(weight.dtype), None


class CuTileRMSNorm(nn.Module):
    """Drop-in RMSNorm using cuTILE persistent kernels."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return CuTileRMSNormFunction.apply(x, self.weight, self.variance_epsilon)

    def extra_repr(self) -> str:
        return f"{self.weight.shape[0]}, eps={self.variance_epsilon}"


def rms_norm(x, weight, eps=1e-6, **_kw):
    """Standalone cuTILE RMSNorm."""
    return CuTileRMSNormFunction.apply(x, weight, eps)


def warmup_rms_norm(hidden_size: int, dtype=torch.bfloat16, device="cuda"):
    """JIT-compile kernels for *hidden_size*."""
    x = torch.randn(2, hidden_size, dtype=dtype, device=device, requires_grad=True)
    w = torch.ones(hidden_size, dtype=dtype, device=device, requires_grad=True)
    rms_norm(x, w, 1e-6).sum().backward()
    torch.cuda.synchronize()


register_patch(
    name="rms_norm_qwen3",
    description="CuTile RMSNorm for Qwen3 (persistent fwd + persistent bwd)",
    target_module="transformers.models.qwen3.modeling_qwen3",
    target_attr="Qwen3RMSNorm",
    replacement=CuTileRMSNorm,
    has_backward=True,
    priority=10,
    models=["qwen3"],
)
