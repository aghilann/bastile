"""CuTile RoPE (Rotary Position Embedding) for Qwen3 models."""

import cuda.tile as ct
import torch

from ..registry import register_patch
from .utils import next_power_of_2

ConstInt = ct.Constant[int]
PAD_ZERO = ct.PaddingMode.ZERO


@ct.kernel
def _rope_kernel(
    q,
    k,
    cos,
    sin,
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
        cos,
        index=(cos_batch_idx, row_idx, 0, 0),
        shape=(1, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((1, TILE_HD))
    sin_row = ct.load(
        sin,
        index=(cos_batch_idx, row_idx, 0, 0),
        shape=(1, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((1, TILE_HD))

    q1 = ct.load(
        q,
        index=(batch_idx, 0, row_idx, 0, 0),
        shape=(1, TILE_QH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_QH, TILE_HD))
    q2 = ct.load(
        q,
        index=(batch_idx, 0, row_idx, 1, 0),
        shape=(1, TILE_QH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_QH, TILE_HD))

    ct.store(
        q,
        index=(batch_idx, 0, row_idx, 0, 0),
        tile=(q1 * cos_row - q2 * sin_row).reshape((1, TILE_QH, 1, 1, TILE_HD)).astype(q.dtype),
    )
    ct.store(
        q,
        index=(batch_idx, 0, row_idx, 1, 0),
        tile=(q2 * cos_row + q1 * sin_row).reshape((1, TILE_QH, 1, 1, TILE_HD)).astype(q.dtype),
    )

    k1 = ct.load(
        k,
        index=(batch_idx, 0, row_idx, 0, 0),
        shape=(1, TILE_KH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_KH, TILE_HD))
    k2 = ct.load(
        k,
        index=(batch_idx, 0, row_idx, 1, 0),
        shape=(1, TILE_KH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_KH, TILE_HD))

    ct.store(
        k,
        index=(batch_idx, 0, row_idx, 0, 0),
        tile=(k1 * cos_row - k2 * sin_row).reshape((1, TILE_KH, 1, 1, TILE_HD)).astype(k.dtype),
    )
    ct.store(
        k,
        index=(batch_idx, 0, row_idx, 1, 0),
        tile=(k2 * cos_row + k1 * sin_row).reshape((1, TILE_KH, 1, 1, TILE_HD)).astype(k.dtype),
    )


def _reshape_and_launch(q, k, cos, sin):
    """Reshape tensors for 2-part rotation and launch the kernel."""
    batch_size, n_q_head, seq_len, head_dim = q.shape
    n_kv_head = k.shape[1]
    half_hd = head_dim // 2
    q = q.reshape(batch_size, n_q_head, seq_len, 2, half_hd)
    k = k.reshape(batch_size, n_kv_head, seq_len, 2, half_hd)
    if cos.shape[-1] == head_dim:
        cos = cos.reshape(cos.shape[0], seq_len, 2, half_hd)
        sin = sin.reshape(sin.shape[0], seq_len, 2, half_hd)
    else:
        cos = cos.reshape(cos.shape[0], seq_len, 1, half_hd)
        sin = sin.reshape(sin.shape[0], seq_len, 1, half_hd)
    ct.launch(
        torch.cuda.current_stream(),
        (batch_size * seq_len, 1, 1),
        _rope_kernel,
        (
            q,
            k,
            cos,
            sin,
            cos.shape[0],
            seq_len,
            next_power_of_2(n_q_head),
            next_power_of_2(n_kv_head),
            next_power_of_2(half_hd),
        ),
    )
    return q.reshape(batch_size, n_q_head, seq_len, head_dim), k.reshape(batch_size, n_kv_head, seq_len, head_dim)


class RoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        q_out, k_out = _reshape_and_launch(q, k, cos, sin)
        ctx.save_for_backward(cos, sin)
        return q_out, k_out

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        dq_out, dk_out = _reshape_and_launch(dq.contiguous(), dk.contiguous(), cos, -sin)
        return dq_out, dk_out, None, None, None, None


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """HuggingFace-compatible rotary position embedding using CuTile."""
    return RoPEFunction.apply(q, k, cos, sin, position_ids, unsqueeze_dim)


register_patch(
    name="rope_qwen3",
    description="CuTile RoPE for Qwen3 models",
    target_module="transformers.models.qwen3.modeling_qwen3",
    target_attr="apply_rotary_pos_emb",
    replacement=apply_rotary_pos_emb,
    has_backward=True,
    priority=10,
    models=["qwen3"],
)
