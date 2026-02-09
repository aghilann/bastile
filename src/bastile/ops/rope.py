"""
CuTile RoPE (Rotary Position Embedding) - ported from TileGym with full backward support and autotuning.

Key optimizations:
1. CuTile forward kernel for efficient rotation computation
2. CuTile backward kernel with reversed rotation
3. Autotuning for optimal tile sizes and occupancy
4. In-place tensor reshaping for 2-part rotation
"""

import torch
from typing import Optional, Tuple

from ..registry import register_patch
from ..autotune import autotune
from .utils import next_power_of_2
from .configs import RoPEConfig

import cuda.tile as ct

ConstInt = ct.Constant[int]
PAD_ZERO = ct.PaddingMode.ZERO


def rope_search_space(n_q_head: int, n_kv_head: int, head_dim: int):
    """Generate search space for RoPE autotuning."""
    # Use power-of-2 sizes for better performance
    tile_qh = next_power_of_2(n_q_head)
    tile_kh = next_power_of_2(n_kv_head)
    tile_hd = next_power_of_2(head_dim // 2)  # Half head_dim for rotation
    
    # Try different occupancy levels
    for occupancy in [1, 2, 4, 8]:
        yield RoPEConfig(
            tile_qh=tile_qh,
            tile_kh=tile_kh,
            tile_hd=tile_hd,
            occupancy=occupancy,
        )


# ============================================================================
# Forward kernel variants with different occupancy levels
# ============================================================================

@ct.kernel(occupancy=1)
def rope_forward_kernel_occ1(
    q, k, cos, sin,
    cos_bs: ConstInt,
    seq_len: ConstInt,
    TILE_QH: ConstInt,
    TILE_KH: ConstInt,
    TILE_HD: ConstInt,
):
    """RoPE forward kernel with occupancy=1."""
    bid = ct.bid(0)
    batch_idx = bid // seq_len
    row_idx = bid % seq_len
    cos_batch_idx = 0 if cos_bs == 1 else batch_idx
    
    # Load cos and sin values (shared for all heads)
    cos_row = ct.load(
        cos, index=(cos_batch_idx, row_idx, 0, 0), 
        shape=(1, 1, 1, TILE_HD), 
        padding_mode=PAD_ZERO
    ).reshape((1, TILE_HD))
    sin_row = ct.load(
        sin, index=(cos_batch_idx, row_idx, 0, 0), 
        shape=(1, 1, 1, TILE_HD), 
        padding_mode=PAD_ZERO
    ).reshape((1, TILE_HD))
    
    # Process Q tensor
    q_tile_1 = ct.load(
        q, index=(batch_idx, 0, row_idx, 0, 0),
        shape=(1, TILE_QH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_QH, TILE_HD))
    q_tile_2 = ct.load(
        q, index=(batch_idx, 0, row_idx, 1, 0),
        shape=(1, TILE_QH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_QH, TILE_HD))
    
    # Rotation: y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
    
    ct.store(q, index=(batch_idx, 0, row_idx, 0, 0),
             tile=new_q_tile_1.reshape((1, TILE_QH, 1, 1, TILE_HD)).astype(q.dtype))
    ct.store(q, index=(batch_idx, 0, row_idx, 1, 0),
             tile=new_q_tile_2.reshape((1, TILE_QH, 1, 1, TILE_HD)).astype(q.dtype))
    
    # Process K tensor
    k_tile_1 = ct.load(
        k, index=(batch_idx, 0, row_idx, 0, 0),
        shape=(1, TILE_KH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_KH, TILE_HD))
    k_tile_2 = ct.load(
        k, index=(batch_idx, 0, row_idx, 1, 0),
        shape=(1, TILE_KH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_KH, TILE_HD))
    
    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
    
    ct.store(k, index=(batch_idx, 0, row_idx, 0, 0),
             tile=new_k_tile_1.reshape((1, TILE_KH, 1, 1, TILE_HD)).astype(k.dtype))
    ct.store(k, index=(batch_idx, 0, row_idx, 1, 0),
             tile=new_k_tile_2.reshape((1, TILE_KH, 1, 1, TILE_HD)).astype(k.dtype))


@ct.kernel(occupancy=2)
def rope_forward_kernel_occ2(
    q, k, cos, sin,
    cos_bs: ConstInt,
    seq_len: ConstInt,
    TILE_QH: ConstInt,
    TILE_KH: ConstInt,
    TILE_HD: ConstInt,
):
    """RoPE forward kernel with occupancy=2."""
    bid = ct.bid(0)
    batch_idx = bid // seq_len
    row_idx = bid % seq_len
    cos_batch_idx = 0 if cos_bs == 1 else batch_idx
    
    cos_row = ct.load(
        cos, index=(cos_batch_idx, row_idx, 0, 0), 
        shape=(1, 1, 1, TILE_HD), 
        padding_mode=PAD_ZERO
    ).reshape((1, TILE_HD))
    sin_row = ct.load(
        sin, index=(cos_batch_idx, row_idx, 0, 0), 
        shape=(1, 1, 1, TILE_HD), 
        padding_mode=PAD_ZERO
    ).reshape((1, TILE_HD))
    
    q_tile_1 = ct.load(
        q, index=(batch_idx, 0, row_idx, 0, 0),
        shape=(1, TILE_QH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_QH, TILE_HD))
    q_tile_2 = ct.load(
        q, index=(batch_idx, 0, row_idx, 1, 0),
        shape=(1, TILE_QH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_QH, TILE_HD))
    
    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
    
    ct.store(q, index=(batch_idx, 0, row_idx, 0, 0),
             tile=new_q_tile_1.reshape((1, TILE_QH, 1, 1, TILE_HD)).astype(q.dtype))
    ct.store(q, index=(batch_idx, 0, row_idx, 1, 0),
             tile=new_q_tile_2.reshape((1, TILE_QH, 1, 1, TILE_HD)).astype(q.dtype))
    
    k_tile_1 = ct.load(
        k, index=(batch_idx, 0, row_idx, 0, 0),
        shape=(1, TILE_KH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_KH, TILE_HD))
    k_tile_2 = ct.load(
        k, index=(batch_idx, 0, row_idx, 1, 0),
        shape=(1, TILE_KH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_KH, TILE_HD))
    
    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
    
    ct.store(k, index=(batch_idx, 0, row_idx, 0, 0),
             tile=new_k_tile_1.reshape((1, TILE_KH, 1, 1, TILE_HD)).astype(k.dtype))
    ct.store(k, index=(batch_idx, 0, row_idx, 1, 0),
             tile=new_k_tile_2.reshape((1, TILE_KH, 1, 1, TILE_HD)).astype(k.dtype))


@ct.kernel(occupancy=4)
def rope_forward_kernel_occ4(
    q, k, cos, sin,
    cos_bs: ConstInt,
    seq_len: ConstInt,
    TILE_QH: ConstInt,
    TILE_KH: ConstInt,
    TILE_HD: ConstInt,
):
    """RoPE forward kernel with occupancy=4."""
    bid = ct.bid(0)
    batch_idx = bid // seq_len
    row_idx = bid % seq_len
    cos_batch_idx = 0 if cos_bs == 1 else batch_idx
    
    cos_row = ct.load(
        cos, index=(cos_batch_idx, row_idx, 0, 0), 
        shape=(1, 1, 1, TILE_HD), 
        padding_mode=PAD_ZERO
    ).reshape((1, TILE_HD))
    sin_row = ct.load(
        sin, index=(cos_batch_idx, row_idx, 0, 0), 
        shape=(1, 1, 1, TILE_HD), 
        padding_mode=PAD_ZERO
    ).reshape((1, TILE_HD))
    
    q_tile_1 = ct.load(
        q, index=(batch_idx, 0, row_idx, 0, 0),
        shape=(1, TILE_QH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_QH, TILE_HD))
    q_tile_2 = ct.load(
        q, index=(batch_idx, 0, row_idx, 1, 0),
        shape=(1, TILE_QH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_QH, TILE_HD))
    
    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
    
    ct.store(q, index=(batch_idx, 0, row_idx, 0, 0),
             tile=new_q_tile_1.reshape((1, TILE_QH, 1, 1, TILE_HD)).astype(q.dtype))
    ct.store(q, index=(batch_idx, 0, row_idx, 1, 0),
             tile=new_q_tile_2.reshape((1, TILE_QH, 1, 1, TILE_HD)).astype(q.dtype))
    
    k_tile_1 = ct.load(
        k, index=(batch_idx, 0, row_idx, 0, 0),
        shape=(1, TILE_KH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_KH, TILE_HD))
    k_tile_2 = ct.load(
        k, index=(batch_idx, 0, row_idx, 1, 0),
        shape=(1, TILE_KH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_KH, TILE_HD))
    
    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
    
    ct.store(k, index=(batch_idx, 0, row_idx, 0, 0),
             tile=new_k_tile_1.reshape((1, TILE_KH, 1, 1, TILE_HD)).astype(k.dtype))
    ct.store(k, index=(batch_idx, 0, row_idx, 1, 0),
             tile=new_k_tile_2.reshape((1, TILE_KH, 1, 1, TILE_HD)).astype(k.dtype))


@ct.kernel(occupancy=8)
def rope_forward_kernel_occ8(
    q, k, cos, sin,
    cos_bs: ConstInt,
    seq_len: ConstInt,
    TILE_QH: ConstInt,
    TILE_KH: ConstInt,
    TILE_HD: ConstInt,
):
    """RoPE forward kernel with occupancy=8."""
    bid = ct.bid(0)
    batch_idx = bid // seq_len
    row_idx = bid % seq_len
    cos_batch_idx = 0 if cos_bs == 1 else batch_idx
    
    cos_row = ct.load(
        cos, index=(cos_batch_idx, row_idx, 0, 0), 
        shape=(1, 1, 1, TILE_HD), 
        padding_mode=PAD_ZERO
    ).reshape((1, TILE_HD))
    sin_row = ct.load(
        sin, index=(cos_batch_idx, row_idx, 0, 0), 
        shape=(1, 1, 1, TILE_HD), 
        padding_mode=PAD_ZERO
    ).reshape((1, TILE_HD))
    
    q_tile_1 = ct.load(
        q, index=(batch_idx, 0, row_idx, 0, 0),
        shape=(1, TILE_QH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_QH, TILE_HD))
    q_tile_2 = ct.load(
        q, index=(batch_idx, 0, row_idx, 1, 0),
        shape=(1, TILE_QH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_QH, TILE_HD))
    
    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
    
    ct.store(q, index=(batch_idx, 0, row_idx, 0, 0),
             tile=new_q_tile_1.reshape((1, TILE_QH, 1, 1, TILE_HD)).astype(q.dtype))
    ct.store(q, index=(batch_idx, 0, row_idx, 1, 0),
             tile=new_q_tile_2.reshape((1, TILE_QH, 1, 1, TILE_HD)).astype(q.dtype))
    
    k_tile_1 = ct.load(
        k, index=(batch_idx, 0, row_idx, 0, 0),
        shape=(1, TILE_KH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_KH, TILE_HD))
    k_tile_2 = ct.load(
        k, index=(batch_idx, 0, row_idx, 1, 0),
        shape=(1, TILE_KH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_KH, TILE_HD))
    
    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
    
    ct.store(k, index=(batch_idx, 0, row_idx, 0, 0),
             tile=new_k_tile_1.reshape((1, TILE_KH, 1, 1, TILE_HD)).astype(k.dtype))
    ct.store(k, index=(batch_idx, 0, row_idx, 1, 0),
             tile=new_k_tile_2.reshape((1, TILE_KH, 1, 1, TILE_HD)).astype(k.dtype))


ROPE_FORWARD_KERNELS = {
    1: rope_forward_kernel_occ1,
    2: rope_forward_kernel_occ2,
    4: rope_forward_kernel_occ4,
    8: rope_forward_kernel_occ8,
}


# ============================================================================
# Backward kernel variants with different occupancy levels
# ============================================================================

@ct.kernel(occupancy=1)
def rope_backward_kernel_occ1(
    dq, dk, cos, sin,
    cos_bs: ConstInt,
    seq_len: ConstInt,
    TILE_QH: ConstInt,
    TILE_KH: ConstInt,
    TILE_HD: ConstInt,
):
    """RoPE backward kernel with occupancy=1."""
    bid = ct.bid(0)
    batch_idx = bid // seq_len
    row_idx = bid % seq_len
    cos_batch_idx = 0 if cos_bs == 1 else batch_idx
    
    cos_row = ct.load(
        cos, index=(cos_batch_idx, row_idx, 0, 0), 
        shape=(1, 1, 1, TILE_HD), 
        padding_mode=PAD_ZERO
    ).reshape((1, TILE_HD))
    sin_row = ct.load(
        sin, index=(cos_batch_idx, row_idx, 0, 0), 
        shape=(1, 1, 1, TILE_HD), 
        padding_mode=PAD_ZERO
    ).reshape((1, TILE_HD))
    
    # Process dQ tensor (reverse rotation)
    dq_tile_1 = ct.load(
        dq, index=(batch_idx, 0, row_idx, 0, 0),
        shape=(1, TILE_QH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_QH, TILE_HD))
    dq_tile_2 = ct.load(
        dq, index=(batch_idx, 0, row_idx, 1, 0),
        shape=(1, TILE_QH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_QH, TILE_HD))
    
    # Reverse rotation: dy = [dx1, dx2] * [cos, cos] + [dx2, -dx1] * [sin, sin]
    new_dq_tile_1 = dq_tile_1 * cos_row + dq_tile_2 * sin_row
    new_dq_tile_2 = dq_tile_2 * cos_row - dq_tile_1 * sin_row
    
    ct.store(dq, index=(batch_idx, 0, row_idx, 0, 0),
             tile=new_dq_tile_1.reshape((1, TILE_QH, 1, 1, TILE_HD)).astype(dq.dtype))
    ct.store(dq, index=(batch_idx, 0, row_idx, 1, 0),
             tile=new_dq_tile_2.reshape((1, TILE_QH, 1, 1, TILE_HD)).astype(dq.dtype))
    
    # Process dK tensor
    dk_tile_1 = ct.load(
        dk, index=(batch_idx, 0, row_idx, 0, 0),
        shape=(1, TILE_KH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_KH, TILE_HD))
    dk_tile_2 = ct.load(
        dk, index=(batch_idx, 0, row_idx, 1, 0),
        shape=(1, TILE_KH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_KH, TILE_HD))
    
    new_dk_tile_1 = dk_tile_1 * cos_row + dk_tile_2 * sin_row
    new_dk_tile_2 = dk_tile_2 * cos_row - dk_tile_1 * sin_row
    
    ct.store(dk, index=(batch_idx, 0, row_idx, 0, 0),
             tile=new_dk_tile_1.reshape((1, TILE_KH, 1, 1, TILE_HD)).astype(dk.dtype))
    ct.store(dk, index=(batch_idx, 0, row_idx, 1, 0),
             tile=new_dk_tile_2.reshape((1, TILE_KH, 1, 1, TILE_HD)).astype(dk.dtype))


@ct.kernel(occupancy=2)
def rope_backward_kernel_occ2(
    dq, dk, cos, sin,
    cos_bs: ConstInt,
    seq_len: ConstInt,
    TILE_QH: ConstInt,
    TILE_KH: ConstInt,
    TILE_HD: ConstInt,
):
    """RoPE backward kernel with occupancy=2."""
    bid = ct.bid(0)
    batch_idx = bid // seq_len
    row_idx = bid % seq_len
    cos_batch_idx = 0 if cos_bs == 1 else batch_idx
    
    cos_row = ct.load(
        cos, index=(cos_batch_idx, row_idx, 0, 0), 
        shape=(1, 1, 1, TILE_HD), 
        padding_mode=PAD_ZERO
    ).reshape((1, TILE_HD))
    sin_row = ct.load(
        sin, index=(cos_batch_idx, row_idx, 0, 0), 
        shape=(1, 1, 1, TILE_HD), 
        padding_mode=PAD_ZERO
    ).reshape((1, TILE_HD))
    
    dq_tile_1 = ct.load(
        dq, index=(batch_idx, 0, row_idx, 0, 0),
        shape=(1, TILE_QH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_QH, TILE_HD))
    dq_tile_2 = ct.load(
        dq, index=(batch_idx, 0, row_idx, 1, 0),
        shape=(1, TILE_QH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_QH, TILE_HD))
    
    new_dq_tile_1 = dq_tile_1 * cos_row + dq_tile_2 * sin_row
    new_dq_tile_2 = dq_tile_2 * cos_row - dq_tile_1 * sin_row
    
    ct.store(dq, index=(batch_idx, 0, row_idx, 0, 0),
             tile=new_dq_tile_1.reshape((1, TILE_QH, 1, 1, TILE_HD)).astype(dq.dtype))
    ct.store(dq, index=(batch_idx, 0, row_idx, 1, 0),
             tile=new_dq_tile_2.reshape((1, TILE_QH, 1, 1, TILE_HD)).astype(dq.dtype))
    
    dk_tile_1 = ct.load(
        dk, index=(batch_idx, 0, row_idx, 0, 0),
        shape=(1, TILE_KH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_KH, TILE_HD))
    dk_tile_2 = ct.load(
        dk, index=(batch_idx, 0, row_idx, 1, 0),
        shape=(1, TILE_KH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_KH, TILE_HD))
    
    new_dk_tile_1 = dk_tile_1 * cos_row + dk_tile_2 * sin_row
    new_dk_tile_2 = dk_tile_2 * cos_row - dk_tile_1 * sin_row
    
    ct.store(dk, index=(batch_idx, 0, row_idx, 0, 0),
             tile=new_dk_tile_1.reshape((1, TILE_KH, 1, 1, TILE_HD)).astype(dk.dtype))
    ct.store(dk, index=(batch_idx, 0, row_idx, 1, 0),
             tile=new_dk_tile_2.reshape((1, TILE_KH, 1, 1, TILE_HD)).astype(dk.dtype))


@ct.kernel(occupancy=4)
def rope_backward_kernel_occ4(
    dq, dk, cos, sin,
    cos_bs: ConstInt,
    seq_len: ConstInt,
    TILE_QH: ConstInt,
    TILE_KH: ConstInt,
    TILE_HD: ConstInt,
):
    """RoPE backward kernel with occupancy=4."""
    bid = ct.bid(0)
    batch_idx = bid // seq_len
    row_idx = bid % seq_len
    cos_batch_idx = 0 if cos_bs == 1 else batch_idx
    
    cos_row = ct.load(
        cos, index=(cos_batch_idx, row_idx, 0, 0), 
        shape=(1, 1, 1, TILE_HD), 
        padding_mode=PAD_ZERO
    ).reshape((1, TILE_HD))
    sin_row = ct.load(
        sin, index=(cos_batch_idx, row_idx, 0, 0), 
        shape=(1, 1, 1, TILE_HD), 
        padding_mode=PAD_ZERO
    ).reshape((1, TILE_HD))
    
    dq_tile_1 = ct.load(
        dq, index=(batch_idx, 0, row_idx, 0, 0),
        shape=(1, TILE_QH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_QH, TILE_HD))
    dq_tile_2 = ct.load(
        dq, index=(batch_idx, 0, row_idx, 1, 0),
        shape=(1, TILE_QH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_QH, TILE_HD))
    
    new_dq_tile_1 = dq_tile_1 * cos_row + dq_tile_2 * sin_row
    new_dq_tile_2 = dq_tile_2 * cos_row - dq_tile_1 * sin_row
    
    ct.store(dq, index=(batch_idx, 0, row_idx, 0, 0),
             tile=new_dq_tile_1.reshape((1, TILE_QH, 1, 1, TILE_HD)).astype(dq.dtype))
    ct.store(dq, index=(batch_idx, 0, row_idx, 1, 0),
             tile=new_dq_tile_2.reshape((1, TILE_QH, 1, 1, TILE_HD)).astype(dq.dtype))
    
    dk_tile_1 = ct.load(
        dk, index=(batch_idx, 0, row_idx, 0, 0),
        shape=(1, TILE_KH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_KH, TILE_HD))
    dk_tile_2 = ct.load(
        dk, index=(batch_idx, 0, row_idx, 1, 0),
        shape=(1, TILE_KH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_KH, TILE_HD))
    
    new_dk_tile_1 = dk_tile_1 * cos_row + dk_tile_2 * sin_row
    new_dk_tile_2 = dk_tile_2 * cos_row - dk_tile_1 * sin_row
    
    ct.store(dk, index=(batch_idx, 0, row_idx, 0, 0),
             tile=new_dk_tile_1.reshape((1, TILE_KH, 1, 1, TILE_HD)).astype(dk.dtype))
    ct.store(dk, index=(batch_idx, 0, row_idx, 1, 0),
             tile=new_dk_tile_2.reshape((1, TILE_KH, 1, 1, TILE_HD)).astype(dk.dtype))


@ct.kernel(occupancy=8)
def rope_backward_kernel_occ8(
    dq, dk, cos, sin,
    cos_bs: ConstInt,
    seq_len: ConstInt,
    TILE_QH: ConstInt,
    TILE_KH: ConstInt,
    TILE_HD: ConstInt,
):
    """RoPE backward kernel with occupancy=8."""
    bid = ct.bid(0)
    batch_idx = bid // seq_len
    row_idx = bid % seq_len
    cos_batch_idx = 0 if cos_bs == 1 else batch_idx
    
    cos_row = ct.load(
        cos, index=(cos_batch_idx, row_idx, 0, 0), 
        shape=(1, 1, 1, TILE_HD), 
        padding_mode=PAD_ZERO
    ).reshape((1, TILE_HD))
    sin_row = ct.load(
        sin, index=(cos_batch_idx, row_idx, 0, 0), 
        shape=(1, 1, 1, TILE_HD), 
        padding_mode=PAD_ZERO
    ).reshape((1, TILE_HD))
    
    dq_tile_1 = ct.load(
        dq, index=(batch_idx, 0, row_idx, 0, 0),
        shape=(1, TILE_QH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_QH, TILE_HD))
    dq_tile_2 = ct.load(
        dq, index=(batch_idx, 0, row_idx, 1, 0),
        shape=(1, TILE_QH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_QH, TILE_HD))
    
    new_dq_tile_1 = dq_tile_1 * cos_row + dq_tile_2 * sin_row
    new_dq_tile_2 = dq_tile_2 * cos_row - dq_tile_1 * sin_row
    
    ct.store(dq, index=(batch_idx, 0, row_idx, 0, 0),
             tile=new_dq_tile_1.reshape((1, TILE_QH, 1, 1, TILE_HD)).astype(dq.dtype))
    ct.store(dq, index=(batch_idx, 0, row_idx, 1, 0),
             tile=new_dq_tile_2.reshape((1, TILE_QH, 1, 1, TILE_HD)).astype(dq.dtype))
    
    dk_tile_1 = ct.load(
        dk, index=(batch_idx, 0, row_idx, 0, 0),
        shape=(1, TILE_KH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_KH, TILE_HD))
    dk_tile_2 = ct.load(
        dk, index=(batch_idx, 0, row_idx, 1, 0),
        shape=(1, TILE_KH, 1, 1, TILE_HD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_KH, TILE_HD))
    
    new_dk_tile_1 = dk_tile_1 * cos_row + dk_tile_2 * sin_row
    new_dk_tile_2 = dk_tile_2 * cos_row - dk_tile_1 * sin_row
    
    ct.store(dk, index=(batch_idx, 0, row_idx, 0, 0),
             tile=new_dk_tile_1.reshape((1, TILE_KH, 1, 1, TILE_HD)).astype(dk.dtype))
    ct.store(dk, index=(batch_idx, 0, row_idx, 1, 0),
             tile=new_dk_tile_2.reshape((1, TILE_KH, 1, 1, TILE_HD)).astype(dk.dtype))


ROPE_BACKWARD_KERNELS = {
    1: rope_backward_kernel_occ1,
    2: rope_backward_kernel_occ2,
    4: rope_backward_kernel_occ4,
    8: rope_backward_kernel_occ8,
}


# ============================================================================
# Kernel launch functions
# ============================================================================

def _run_rope_forward_with_config(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    batch_size: int,
    n_q_head: int,
    n_kv_head: int,
    seq_len: int,
    head_dim: int,
    config: RoPEConfig,
):
    """Run RoPE forward with a specific config (in-place operation on reshaped tensors)."""
    n_row = batch_size * seq_len
    grid = (n_row, 1, 1)
    
    kernel = ROPE_FORWARD_KERNELS[config.occupancy]
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        kernel,
        (
            q,
            k,
            cos,
            sin,
            cos.shape[0],
            seq_len,
            config.tile_qh,
            config.tile_kh,
            config.tile_hd,
        ),
    )


def rope_forward_cutile(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CuTile-accelerated RoPE forward with autotuning."""
    batch_size, n_q_head, seq_len, head_dim = q.shape
    n_kv_head = k.shape[1]
    
    # Store original shapes
    original_cos_shape = cos.shape
    original_sin_shape = sin.shape
    
    # Reshape to split head dimension
    q = q.reshape(batch_size, n_q_head, seq_len, 2, head_dim // 2)
    k = k.reshape(batch_size, n_kv_head, seq_len, 2, head_dim // 2)
    
    # Reshape cos/sin if needed
    if cos.shape[-1] == head_dim:
        cos = cos.reshape(cos.shape[0], seq_len, 2, head_dim // 2)
        sin = sin.reshape(sin.shape[0], seq_len, 2, head_dim // 2)
    else:
        cos = cos.reshape(cos.shape[0], seq_len, 1, head_dim // 2)
        sin = sin.reshape(sin.shape[0], seq_len, 1, head_dim // 2)
    
    key = (batch_size, n_q_head, n_kv_head, seq_len, head_dim, str(q.dtype))
    
    def run_with_config(cfg):
        _run_rope_forward_with_config(q, k, cos, sin, batch_size, n_q_head, n_kv_head, seq_len, head_dim, cfg)
    
    config = autotune(
        kernel_name="rope_forward",
        run_fn=run_with_config,
        search_space=list(rope_search_space(n_q_head, n_kv_head, head_dim)),
        key=str(key),
        max_iter=8,
        use_heuristic=False,  # Actually benchmark on B200
    )
    
    _run_rope_forward_with_config(q, k, cos, sin, batch_size, n_q_head, n_kv_head, seq_len, head_dim, config)
    
    return (
        q.reshape(batch_size, n_q_head, seq_len, head_dim),
        k.reshape(batch_size, n_kv_head, seq_len, head_dim),
    )


def _run_rope_backward_with_config(
    dq: torch.Tensor,
    dk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    batch_size: int,
    n_q_head: int,
    n_kv_head: int,
    seq_len: int,
    head_dim: int,
    config: RoPEConfig,
):
    """Run RoPE backward with a specific config (in-place operation on reshaped tensors)."""
    n_row = batch_size * seq_len
    grid = (n_row, 1, 1)
    
    kernel = ROPE_BACKWARD_KERNELS[config.occupancy]
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        kernel,
        (
            dq,
            dk,
            cos,
            sin,
            cos.shape[0],
            seq_len,
            config.tile_qh,
            config.tile_kh,
            config.tile_hd,
        ),
    )


def rope_backward_cutile(
    dq: torch.Tensor,
    dk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CuTile-accelerated RoPE backward with autotuning."""
    batch_size, n_q_head, seq_len, head_dim = dq.shape
    n_kv_head = dk.shape[1]
    
    # Make contiguous
    dq = dq.contiguous()
    dk = dk.contiguous()
    
    # Reshape to split head dimension
    dq = dq.reshape(batch_size, n_q_head, seq_len, 2, head_dim // 2)
    dk = dk.reshape(batch_size, n_kv_head, seq_len, 2, head_dim // 2)
    
    # Reshape cos/sin if needed
    if cos.shape[-1] == head_dim:
        cos = cos.reshape(cos.shape[0], seq_len, 2, head_dim // 2)
        sin = sin.reshape(sin.shape[0], seq_len, 2, head_dim // 2)
    else:
        cos = cos.reshape(cos.shape[0], seq_len, 1, head_dim // 2)
        sin = sin.reshape(sin.shape[0], seq_len, 1, head_dim // 2)
    
    key = (batch_size, n_q_head, n_kv_head, seq_len, head_dim, str(dq.dtype))
    
    def run_with_config(cfg):
        _run_rope_backward_with_config(dq, dk, cos, sin, batch_size, n_q_head, n_kv_head, seq_len, head_dim, cfg)
    
    config = autotune(
        kernel_name="rope_backward",
        run_fn=run_with_config,
        search_space=list(rope_search_space(n_q_head, n_kv_head, head_dim)),
        key=str(key),
        max_iter=8,
        use_heuristic=False,
    )
    
    _run_rope_backward_with_config(dq, dk, cos, sin, batch_size, n_q_head, n_kv_head, seq_len, head_dim, config)
    
    return (
        dq.reshape(batch_size, n_q_head, seq_len, head_dim),
        dk.reshape(batch_size, n_kv_head, seq_len, head_dim),
    )


class RoPEFunction(torch.autograd.Function):
    """CuTile RoPE with full backward support and autotuning."""

    @staticmethod
    def forward(ctx, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """
        q size: (bsz, n_q_head, seq_len, head_dim)
        k size: (bsz, n_kv_head, seq_len, head_dim)
        cos size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        sin size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        """
        q_out, k_out = rope_forward_cutile(q, k, cos, sin)
        ctx.save_for_backward(cos, sin)
        return q_out, k_out

    @staticmethod
    def backward(ctx, dq, dk):
        """
        dq size: (bsz, n_q_head, seq_len, head_dim)
        dk size: (bsz, n_kv_head, seq_len, head_dim)
        """
        cos, sin = ctx.saved_tensors
        dq_out, dk_out = rope_backward_cutile(dq, dk, cos, sin)
        return dq_out, dk_out, None, None, None, None


def apply_rotary_pos_emb(
    q, k, cos, sin, 
    position_ids=None, 
    unsqueeze_dim=1,
):
    """
    Apply rotary position embeddings to query and key tensors using CuTile.
    
    HuggingFace-compatible implementation.
    
    Args:
        q: Query tensor [bsz, n_q_head, seq_len, head_dim]
        k: Key tensor [bsz, n_kv_head, seq_len, head_dim]
        cos: Cosine positional embedding
        sin: Sine positional embedding
        position_ids: Optional position IDs (unused, for HF compatibility)
        unsqueeze_dim: Dimension to unsqueeze cos/sin for head broadcasting
    """
    return RoPEFunction.apply(q, k, cos, sin, position_ids, unsqueeze_dim)


# Register patches for Qwen3
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
