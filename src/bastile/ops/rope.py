"""CuTile RoPE (Rotary Position Embedding) with autotuning."""

import torch
from typing import Tuple

from ..registry import register_patch
from ..autotune import autotune
from .utils import next_power_of_2
from .configs import RoPEConfig

import cuda.tile as ct

ConstInt = ct.Constant[int]
PAD_ZERO = ct.PaddingMode.ZERO


def rope_search_space(n_q_head: int, n_kv_head: int, head_dim: int):
    tile_qh = next_power_of_2(n_q_head)
    tile_kh = next_power_of_2(n_kv_head)
    tile_hd = next_power_of_2(head_dim // 2)
    for occupancy in [1, 2, 4, 8]:
        yield RoPEConfig(
            tile_qh=tile_qh, tile_kh=tile_kh,
            tile_hd=tile_hd, occupancy=occupancy,
        )


def _rope_kernel_body(
    q, k, cos, sin,
    cos_bs: ConstInt,
    seq_len: ConstInt,
    TILE_QH: ConstInt,
    TILE_KH: ConstInt,
    TILE_HD: ConstInt,
):
    """Rotate q and k by (cos, sin). Pass -sin for inverse (backward)."""
    bid = ct.bid(0)
    batch_idx = bid // seq_len
    row_idx = bid % seq_len
    cos_batch_idx = 0 if cos_bs == 1 else batch_idx

    cos_row = ct.load(
        cos, index=(cos_batch_idx, row_idx, 0, 0),
        shape=(1, 1, 1, TILE_HD), padding_mode=PAD_ZERO,
    ).reshape((1, TILE_HD))
    sin_row = ct.load(
        sin, index=(cos_batch_idx, row_idx, 0, 0),
        shape=(1, 1, 1, TILE_HD), padding_mode=PAD_ZERO,
    ).reshape((1, TILE_HD))

    q1 = ct.load(
        q, index=(batch_idx, 0, row_idx, 0, 0),
        shape=(1, TILE_QH, 1, 1, TILE_HD), padding_mode=PAD_ZERO,
    ).reshape((TILE_QH, TILE_HD))
    q2 = ct.load(
        q, index=(batch_idx, 0, row_idx, 1, 0),
        shape=(1, TILE_QH, 1, 1, TILE_HD), padding_mode=PAD_ZERO,
    ).reshape((TILE_QH, TILE_HD))

    ct.store(q, index=(batch_idx, 0, row_idx, 0, 0),
             tile=(q1 * cos_row - q2 * sin_row).reshape((1, TILE_QH, 1, 1, TILE_HD)).astype(q.dtype))
    ct.store(q, index=(batch_idx, 0, row_idx, 1, 0),
             tile=(q2 * cos_row + q1 * sin_row).reshape((1, TILE_QH, 1, 1, TILE_HD)).astype(q.dtype))

    k1 = ct.load(
        k, index=(batch_idx, 0, row_idx, 0, 0),
        shape=(1, TILE_KH, 1, 1, TILE_HD), padding_mode=PAD_ZERO,
    ).reshape((TILE_KH, TILE_HD))
    k2 = ct.load(
        k, index=(batch_idx, 0, row_idx, 1, 0),
        shape=(1, TILE_KH, 1, 1, TILE_HD), padding_mode=PAD_ZERO,
    ).reshape((TILE_KH, TILE_HD))

    ct.store(k, index=(batch_idx, 0, row_idx, 0, 0),
             tile=(k1 * cos_row - k2 * sin_row).reshape((1, TILE_KH, 1, 1, TILE_HD)).astype(k.dtype))
    ct.store(k, index=(batch_idx, 0, row_idx, 1, 0),
             tile=(k2 * cos_row + k1 * sin_row).reshape((1, TILE_KH, 1, 1, TILE_HD)).astype(k.dtype))


ROPE_KERNELS = {occ: ct.kernel(_rope_kernel_body, occupancy=occ) for occ in [1, 2, 4, 8]}


def _launch_rope(q, k, cos, sin, batch_size: int, seq_len: int, config: RoPEConfig):
    ct.launch(
        torch.cuda.current_stream(),
        (batch_size * seq_len, 1, 1),
        ROPE_KERNELS[config.occupancy],
        (q, k, cos, sin, cos.shape[0], seq_len,
         config.tile_qh, config.tile_kh, config.tile_hd),
    )


def _reshape_for_rope(q, k, cos, sin):
    batch_size, n_q_head, seq_len, head_dim = q.shape
    n_kv_head = k.shape[1]
    q = q.reshape(batch_size, n_q_head, seq_len, 2, head_dim // 2)
    k = k.reshape(batch_size, n_kv_head, seq_len, 2, head_dim // 2)
    if cos.shape[-1] == head_dim:
        cos = cos.reshape(cos.shape[0], seq_len, 2, head_dim // 2)
        sin = sin.reshape(sin.shape[0], seq_len, 2, head_dim // 2)
    else:
        cos = cos.reshape(cos.shape[0], seq_len, 1, head_dim // 2)
        sin = sin.reshape(sin.shape[0], seq_len, 1, head_dim // 2)
    return q, k, cos, sin, batch_size, n_q_head, n_kv_head, seq_len, head_dim


def rope_forward_cutile(
    q: torch.Tensor, k: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q, k, cos, sin, bs, nqh, nkvh, sl, hd = _reshape_for_rope(q, k, cos, sin)
    key = str((bs, nqh, nkvh, sl, hd, str(q.dtype)))

    def run(cfg):
        _launch_rope(q, k, cos, sin, bs, sl, cfg)

    config = autotune(
        run_fn=run,
        search_space=list(rope_search_space(nqh, nkvh, hd)),
        key=key, kernel_name="rope_forward",
        max_iter=8, use_heuristic=False,
    )
    _launch_rope(q, k, cos, sin, bs, sl, config)
    return q.reshape(bs, nqh, sl, hd), k.reshape(bs, nkvh, sl, hd)


def rope_backward_cutile(
    dq: torch.Tensor, dk: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dq = dq.contiguous()
    dk = dk.contiguous()
    # Negate sin to reverse the rotation: forward(x, cos, -sin) == inverse
    dq, dk, cos, neg_sin, bs, nqh, nkvh, sl, hd = _reshape_for_rope(dq, dk, cos, -sin)
    key = str((bs, nqh, nkvh, sl, hd, str(dq.dtype)))

    def run(cfg):
        _launch_rope(dq, dk, cos, neg_sin, bs, sl, cfg)

    config = autotune(
        run_fn=run,
        search_space=list(rope_search_space(nqh, nkvh, hd)),
        key=key, kernel_name="rope_backward",
        max_iter=8, use_heuristic=False,
    )
    _launch_rope(dq, dk, cos, neg_sin, bs, sl, config)
    return dq.reshape(bs, nqh, sl, hd), dk.reshape(bs, nkvh, sl, hd)


class RoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        q_out, k_out = rope_forward_cutile(q, k, cos, sin)
        ctx.save_for_backward(cos, sin)
        return q_out, k_out

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        dq_out, dk_out = rope_backward_cutile(dq, dk, cos, sin)
        return dq_out, dk_out, None, None, None, None


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """HuggingFace-compatible rotary position embedding using CuTile."""
    return RoPEFunction.apply(q, k, cos, sin, position_ids, unsqueeze_dim)


register_patch(
    name="rope_qwen3",
    description="CuTile RoPE with autotuning for Qwen3 models",
    target_module="transformers.models.qwen3.modeling_qwen3",
    target_attr="apply_rotary_pos_emb",
    replacement=apply_rotary_pos_emb,
    has_backward=True,
    priority=10,
    models=["qwen3"],
)
