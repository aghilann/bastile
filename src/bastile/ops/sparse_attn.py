"""
CuTile sparse multi-head attention — index-gather KV + online softmax.

Re-implements DeepSeek-V4's ``sparse_attn_kernel`` in cuTILE. For each
(batch, seq_pos), attends to only the top-k KV positions specified by an
index tensor, using online softmax (running max/sum) and an attn_sink bias.

Design:
  1. KV is pre-gathered by the host into [B, M, topk, D] to avoid
     per-element conditional loads inside the kernel.
  2. All heads for one (batch, seq_pos) are processed in a single block.
     Grid is (M, B) — ct.bid(0)=seq_pos, ct.bid(1)=batch.
  3. Q tile [H, D] is loaded once into registers, then KV tiles of
     BLOCK_K rows are streamed in a loop, with online softmax updating
     the running max/sum and output accumulator.
  4. An attn_sink (learnable per-head scalar) is added to the softmax
     denominator after the KV loop, matching the tilelang kernel's
     behaviour.
"""

from __future__ import annotations

import cuda.tile as ct
import torch
import torch.nn as nn

try:
    from cuda.tile._numeric_semantics import RoundingMode as RMd
    _APPROX_DIV = True
except ImportError:
    _APPROX_DIV = False

ConstInt = ct.Constant[int]
ConstFloat = ct.Constant[float]
PAD_ZERO = ct.PaddingMode.ZERO


@ct.kernel
def _sparse_attn_fwd_kernel(
    Q,                   # [B, M, H, D] — query activations (bf16/fp16)
    KV_gathered,         # [B, M, topk, D] — pre-gathered key/value cache
    O,                   # [B, M, H, D] — output
    attn_sink,           # [H] — per-head attn_sink bias (fp32)
    softmax_scale: ConstFloat,
    BLOCK_K: ConstInt,   # K-tile size (64)
    NUM_HEADS: ConstInt,
    HEAD_DIM: ConstInt,
    TOPK: ConstInt,
):
    bid_m = ct.bid(0)   # seq position index
    bid_b = ct.bid(1)   # batch index

    # Load Q for all heads at this (batch, seq_pos) — one HBM load, stays in registers
    q_tile = ct.reshape(
        ct.load(Q, index=(bid_b, bid_m, 0, 0), shape=(1, 1, NUM_HEADS, HEAD_DIM), padding_mode=PAD_ZERO),
        (NUM_HEADS, HEAD_DIM),
    )

    # Online softmax state — all fp32, per head
    m = ct.full((NUM_HEADS, 1), float("-inf"), dtype=ct.float32)
    l = ct.full((NUM_HEADS, 1), 0.0,            dtype=ct.float32)
    o = ct.full((NUM_HEADS, HEAD_DIM), 0.0,     dtype=ct.float32)

    num_k_blocks = ct.cdiv(TOPK, BLOCK_K)

    for t in range(num_k_blocks):
        ks = t * BLOCK_K

        # Load KV tile — [BLOCK_K, D] from the pre-gathered cache
        kv_tile = ct.reshape(
            ct.load(
                KV_gathered,
                index=(bid_b, bid_m, ks, 0),
                shape=(1, 1, BLOCK_K, HEAD_DIM),
                padding_mode=PAD_ZERO,
            ),
            (BLOCK_K, HEAD_DIM),
        )

        # Q @ K^T via tensor cores: (H, D) @ (D, BLOCK_K) → (H, BLOCK_K) fp32
        scores = ct.mma(
            q_tile,
            ct.transpose(kv_tile),
            ct.full((NUM_HEADS, BLOCK_K), 0.0, dtype=ct.float32),
        )
        scores = scores * softmax_scale

        # Online softmax update
        row_max = ct.max(scores, axis=1, keepdims=True)   # (H, 1)
        m_new = ct.maximum(m, row_max)

        exp_shift = ct.exp(m - m_new)                     # (H, 1) — rescale factor
        exp_scores = ct.exp(scores - m_new)                # (H, BLOCK_K)

        # Accumulate PV: cast exp_scores to input dtype for mma
        o = o * exp_shift + ct.mma(
            ct.astype(exp_scores, Q.dtype),
            kv_tile,
            ct.full((NUM_HEADS, HEAD_DIM), 0.0, dtype=ct.float32),
        )
        l = l * exp_shift + ct.sum(exp_scores, axis=1, keepdims=True)
        m = m_new

    # Add attn_sink bias to softmax denominator (vectorized)
    sink_vec = ct.reshape(ct.load(attn_sink, index=(0,), shape=(NUM_HEADS,), padding_mode=PAD_ZERO), (NUM_HEADS, 1))
    l = l + ct.exp(sink_vec - m)

    # Normalise and write back
    if _APPROX_DIV:
        o_norm = ct.truediv(o, l, flush_to_zero=True, rounding_mode=RMd.APPROX)
    else:
        o_norm = o / l
    out_tile = ct.reshape(ct.astype(o_norm, Q.dtype), (1, 1, NUM_HEADS, HEAD_DIM))
    ct.store(O, index=(bid_b, bid_m, 0, 0), tile=out_tile)


# ── Host-side launch ──────────────────────────────────────────────────────────

def _launch_sparse_attn(
    q: torch.Tensor,        # [B, M, H, D]
    kv_gathered: torch.Tensor,  # [B, M, topk, D]
    attn_sink: torch.Tensor,    # [H]
    softmax_scale: float,
) -> torch.Tensor:
    B, M, H, D = q.shape
    topk = kv_gathered.shape[2]

    if not _APPROX_DIV:
        # Without approximate division we use standard division
        pass

    o = torch.empty_like(q)

    BLOCK_K = 64
    grid = (M, B, 1)

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _sparse_attn_fwd_kernel,
        (q, kv_gathered, o, attn_sink, softmax_scale, BLOCK_K, H, D, topk),
    )
    return o


# ── Public API ─────────────────────────────────────────────────────────────────

def sparse_attn(
    q: torch.Tensor,             # [B, S, H, D]
    kv_states: torch.Tensor,     # [B, N, D] — key/value states for all N positions
    attn_sink: torch.Tensor,     # [H]
    topk_idxs: torch.Tensor,     # [B, S, topk] — int32 indices into kv_states
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Sparse multi-head attention — attends to top-k KV positions per query.

    Pre-gathers KV states from [B, N, D] into [B, S, topk, D], then launches
    the CuTile kernel (online softmax + attn_sink). N ≥ S during training.

    Args:
        q:              Query activations,        [batch, seq_len, num_heads, head_dim]
        kv_states:      Key/value states (shared), [batch, total_positions, head_dim]
        attn_sink:      Per-head attention sink bias, [num_heads] (fp32)
        topk_idxs:      Top-k position indices,   [batch, seq_len, topk] (int32)
        softmax_scale:  Scaling factor; defaults to 1/sqrt(head_dim)

    Returns:
        output:  [batch, seq_len, num_heads, head_dim]
    """
    if not (q.is_cuda and kv_states.is_cuda and attn_sink.is_cuda and topk_idxs.is_cuda):
        raise RuntimeError("sparse_attn requires CUDA tensors")

    B, S, H, D = q.shape
    topk = topk_idxs.shape[-1]

    if softmax_scale is None:
        softmax_scale = (1.0 / D) ** 0.5

    # Pad heads to 16 for warp efficiency (stripped after)
    pad_h = (16 - H % 16) % 16
    if pad_h:
        q = torch.cat([q, q.new_zeros(B, S, pad_h, D)], dim=2)
        attn_sink = torch.cat([attn_sink, attn_sink.new_zeros(pad_h)])
        H_padded = H + pad_h
    else:
        H_padded = H

    # Pre-gather KV states into [B, S, topk, D]
    kv_gathered = kv_states[
        torch.arange(B, device=kv_states.device)[:, None, None],
        topk_idxs.long(),
        :,
    ].contiguous()

    q_padded = q.contiguous()

    o_padded = _launch_sparse_attn(q_padded, kv_gathered, attn_sink, softmax_scale)

    if pad_h:
        o_padded = o_padded.narrow(2, 0, H).contiguous()

    return o_padded


# ── nn.Module drop-in ──────────────────────────────────────────────────────────

class CuTileSparseAttention(nn.Module):
    """Drop-in for the sparse-attention component of DSV4's MLA indexer.

    Stores ``attn_sink`` as a learnable parameter.
    """

    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_sink = nn.Parameter(torch.zeros(num_heads))

    def forward(
        self,
        q: torch.Tensor,
        kv_states: torch.Tensor,
        topk_idxs: torch.Tensor,
        softmax_scale: float | None = None,
    ) -> torch.Tensor:
        return sparse_attn(q, kv_states, self.attn_sink, topk_idxs, softmax_scale)

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, head_dim={self.head_dim}"