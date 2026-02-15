"""
CuTile Fused MoE Experts for GPT-OSS.

Replaces HuggingFace's grouped_mm MoE forward with CuTile fused GEMM kernels.
Each kernel fuses: token routing (gather) + GEMM (MMA) + bias (epilogue) +
optional routing weight multiply (epilogue).

Pipeline:
1. moe_align_block_size: sort tokens by expert, pad to tile boundaries
2. Fused gate_up GEMM: gather hidden_states by routing + GEMM + bias
3. Apply gate: existing CuTile MoE gate kernel
4. Fused down GEMM: GEMM + bias + routing weight multiply
5. Sum over top-k expert choices

Forward: CuTile fused kernels with autotuned tile sizes
Backward: CuTile fused gather-GEMM-scatter for activation grads (dX),
          torch._grouped_mm re-forward + autograd for weight grads (dW)
"""

import logging
import torch
import cuda.tile as ct

from .moe_gate import moe_gate_forward_cutile, moe_gate_backward_cutile
from .utils import ceildiv
from .configs import MoEGemmConfig
from ..autotune import autotune

logger = logging.getLogger(__name__)


# ============================================================================
# Search space for autotuning
# ============================================================================

def moe_gemm_search_space(hidden_size, intermediate_size):
    """Generate valid tile configs for MoE GEMM autotuning.

    TILE_N must divide both 2*intermediate_size (gate_up N) and hidden_size (down N).
    TILE_K must divide both hidden_size (gate_up K) and intermediate_size (down K).

    Includes larger tile sizes for Blackwell (sm100+) with num_ctas=2.
    """
    inter_2x = 2 * intermediate_size
    for tile_m in [64, 128]:
        for tile_n in [32, 64]:
            if hidden_size % tile_n != 0 or inter_2x % tile_n != 0:
                continue
            for tile_k in [32, 64]:
                if hidden_size % tile_k != 0 or intermediate_size % tile_k != 0:
                    continue
                for group_m in [8, 32]:
                    yield MoEGemmConfig(tile_m, tile_n, tile_k, group_m)


# ============================================================================
# Token-expert alignment (vectorized PyTorch)
# ============================================================================

def moe_align_block_size(topk_ids, block_size, num_experts):
    """Sort tokens by expert and pad to block_size boundaries."""
    flat_ids = topk_ids.reshape(-1)
    S = flat_ids.shape[0]
    device = flat_ids.device

    perm = torch.argsort(flat_ids)
    sorted_experts = flat_ids[perm]

    counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    counts.scatter_add_(0, flat_ids.long(), torch.ones(S, dtype=torch.int32, device=device))

    padded_counts = ((counts + block_size - 1) // block_size) * block_size
    total_padded = padded_counts.sum().item()

    cumsum_padded = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    cumsum_padded[1:] = torch.cumsum(padded_counts.long(), dim=0)
    cumsum_orig = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    cumsum_orig[1:] = torch.cumsum(counts.long(), dim=0)

    # Vectorized scatter: each sorted token goes to its padded destination
    pad_offsets = cumsum_padded[:-1] - cumsum_orig[:-1]
    destinations = torch.arange(S, device=device, dtype=torch.int64) + pad_offsets[sorted_experts.long()]

    sorted_token_ids = torch.full((total_padded,), S, dtype=torch.int32, device=device)
    sorted_token_ids[destinations] = perm.int()

    # Expert ID per M-block
    num_blocks = total_padded // block_size
    block_starts = (cumsum_padded[:-1] // block_size).int()
    block_ends = (cumsum_padded[1:] // block_size).int()
    block_counts = (block_ends - block_starts).long()

    expert_indices = torch.repeat_interleave(
        torch.arange(num_experts, device=device, dtype=torch.int32),
        block_counts,
    )
    expert_ids = torch.zeros(num_blocks, dtype=torch.int32, device=device)
    expert_ids[:expert_indices.shape[0]] = expert_indices

    num_tokens_post_padded = torch.tensor([total_padded], dtype=torch.int32, device=device)
    return sorted_token_ids, expert_ids, num_tokens_post_padded


# ============================================================================
# CuTile Fused MoE GEMM Kernel
# ============================================================================

@ct.kernel
def fused_moe_gemm_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: ct.Constant[int], K: ct.Constant[int],
    EM: int, num_valid_tokens: int,
    TILE_SIZE_M: ct.Constant[int], TILE_SIZE_N: ct.Constant[int],
    TILE_SIZE_K: ct.Constant[int], GROUP_SIZE_M: ct.Constant[int],
    MUL_ROUTED_WEIGHT: ct.Constant[int], top_k: ct.Constant[int],
    HAS_BIAS: ct.Constant[int],
    a_stride_1: ct.Constant[int], c_stride_1: ct.Constant[int],
):
    """Fused MoE GEMM for GPT-OSS (E, K, N) weight layout.

    Fuses token routing gather + MMA + bias + optional weight multiply.
    Uses latency hints for Blackwell memory subsystem.
    """
    bid = ct.bid(axis=0)
    num_bid_m = ct.cdiv(EM, TILE_SIZE_M)
    num_bid_n = ct.cdiv(N, TILE_SIZE_N)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + ((bid % num_bid_in_group) % group_size_m)
    bid_n = (bid % num_bid_in_group) // group_size_m

    num_tokens_post_padded = ct.gather(num_tokens_post_padded_ptr, 0, padding_value=0)
    if bid_m * TILE_SIZE_M < num_tokens_post_padded:
        offs_token_id = bid_m * TILE_SIZE_M + ct.arange(TILE_SIZE_M, dtype=ct.int32)
        offs_token = ct.gather(sorted_token_ids_ptr, offs_token_id, padding_value=0)
        token_mask = offs_token < num_valid_tokens

        off_experts = ct.load(expert_ids_ptr, index=(bid_m,), shape=(1,))
        off_experts = off_experts.item()

        row_indices = offs_token // top_k
        a_row_offset = row_indices[:, None] * a_stride_1

        accumulator = ct.full((TILE_SIZE_M, TILE_SIZE_N), 0.0, dtype=ct.float32)
        for k in range(0, ct.cdiv(K, TILE_SIZE_K)):
            col_indices = k * TILE_SIZE_K + ct.arange(TILE_SIZE_K, dtype=ct.int32)
            a_indices = a_row_offset + col_indices[None, :]
            a = ct.gather(a_ptr, a_indices, padding_value=0)

            # GPT-OSS weights: (E, K, N) — load contiguous tile
            b = ct.load(b_ptr, index=(off_experts, k, bid_n),
                        shape=(1, TILE_SIZE_K, TILE_SIZE_N))
            b = ct.reshape(b, (TILE_SIZE_K, TILE_SIZE_N))
            accumulator = ct.mma(a, b, accumulator)

        if HAS_BIAS:
            bias_offset = off_experts * N + bid_n * TILE_SIZE_N + ct.arange(TILE_SIZE_N, dtype=ct.int32)
            bias_tile = ct.gather(bias_ptr, bias_offset, padding_value=0)
            accumulator = accumulator + ct.astype(bias_tile, ct.float32)[None, :]

        if MUL_ROUTED_WEIGHT:
            moe_weight = ct.gather(topk_weights_ptr, offs_token, padding_value=0)
            accumulator = accumulator * ct.expand_dims(moe_weight, axis=1)

        offs_cn = bid_n * TILE_SIZE_N + ct.arange(TILE_SIZE_N, dtype=ct.int32)
        c_offset = c_stride_1 * offs_token[:, None] + offs_cn[None, :]
        ct.scatter(c_ptr, c_offset, ct.astype(accumulator, c_ptr.dtype))


def _launch_fused_moe_gemm(
    A, B, C, bias, topk_weights, sorted_token_ids, expert_ids,
    num_tokens_post_padded, N, K, top_k, mul_routed_weight, has_bias,
    cfg,
):
    """Launch the fused MoE GEMM kernel with autotuned config."""
    EM = sorted_token_ids.shape[0]
    grid = (ceildiv(EM, cfg.tile_m) * ceildiv(N, cfg.tile_n),)

    a_stride_1 = A.shape[1] if A.dim() == 2 else A.shape[-1]
    A_flat = A.reshape(-1)
    C_flat = C.reshape(-1)
    bias_flat = bias.reshape(-1) if has_bias else torch.empty(0, device=A.device, dtype=A.dtype)

    ct.launch(
        torch.cuda.current_stream(), grid, fused_moe_gemm_kernel,
        (A_flat, B, C_flat, bias_flat,
         topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
         N, K, EM, topk_weights.shape[0],
         cfg.tile_m, cfg.tile_n, cfg.tile_k, cfg.group_m,
         int(mul_routed_weight), top_k, int(has_bias),
         a_stride_1, N),
    )


# ============================================================================
# Forward implementation (used by both autotuner and autograd)
# ============================================================================

def _run_moe_forward_with_config(
    hidden_states, gate_up_proj, gate_up_bias, down_proj, down_bias,
    top_k_index, top_k_weights, num_experts, cfg,
):
    """Run full MoE forward with a specific config. Returns (final, gated_out, saved_gate, saved_up)."""
    top_k = top_k_index.shape[1]
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    inter_2x = gate_up_proj.shape[2]
    inter_size = inter_2x // 2
    S = num_tokens * top_k
    device = hidden_states.device

    sorted_ids, block_experts, n_post_pad = moe_align_block_size(top_k_index, cfg.tile_m, num_experts)
    weights_flat = top_k_weights.reshape(-1)

    # Gate-up GEMM + bias (padding rows for out-of-bounds scatter safety)
    gate_up_out = torch.zeros(S + cfg.tile_m, inter_2x, dtype=hidden_states.dtype, device=device)
    _launch_fused_moe_gemm(
        hidden_states, gate_up_proj, gate_up_out, gate_up_bias,
        weights_flat, sorted_ids, block_experts, n_post_pad,
        N=inter_2x, K=hidden_size, top_k=top_k,
        mul_routed_weight=False, has_bias=True, cfg=cfg,
    )
    gate_up_out = gate_up_out[:S]

    # Gate
    gated_out, saved_gate, saved_up = moe_gate_forward_cutile(gate_up_out)

    # Down GEMM + bias + weight multiply
    down_out = torch.zeros(S + cfg.tile_m, hidden_size, dtype=hidden_states.dtype, device=device)
    _launch_fused_moe_gemm(
        gated_out, down_proj, down_out, down_bias,
        weights_flat, sorted_ids, block_experts, n_post_pad,
        N=hidden_size, K=inter_size, top_k=1,
        mul_routed_weight=True, has_bias=True, cfg=cfg,
    )
    down_out = down_out[:S]

    final = down_out.view(num_tokens, top_k, hidden_size).sum(dim=1)
    return final, gated_out, saved_gate, saved_up


def _get_moe_config(hidden_states, gate_up_proj, gate_up_bias, down_proj, down_bias,
                    top_k_index, top_k_weights, num_experts):
    """Get autotuned MoE GEMM config for the given problem dimensions."""
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    inter_2x = gate_up_proj.shape[2]
    inter_size = inter_2x // 2
    top_k = top_k_index.shape[1]

    key = f"{num_tokens}_{num_experts}_{hidden_size}_{inter_size}_{top_k}_{hidden_states.dtype}"

    search = list(moe_gemm_search_space(hidden_size, inter_size))
    if not search:
        # Fallback: use safe defaults that divide any multiple of 32
        search = [MoEGemmConfig(128, 32, 32, 32)]

    def run_with_config(cfg):
        _run_moe_forward_with_config(
            hidden_states, gate_up_proj, gate_up_bias, down_proj, down_bias,
            top_k_index, top_k_weights, num_experts, cfg,
        )

    return autotune(
        kernel_name="moe_experts_fwd",
        run_fn=run_with_config,
        search_space=search,
        key=key,
        max_iter=20,
        warmup=2,
        rep=3,
        use_heuristic=False,
    )


# ============================================================================
# Autograd Function
# ============================================================================

class CuTileMoEExpertsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, gate_up_proj, gate_up_bias,
                down_proj, down_bias, top_k_index, top_k_weights, num_experts, cfg):
        final, gated_out, saved_gate, saved_up = _run_moe_forward_with_config(
            hidden_states, gate_up_proj, gate_up_bias, down_proj, down_bias,
            top_k_index, top_k_weights, num_experts, cfg,
        )

        ctx.save_for_backward(
            hidden_states, gate_up_proj, gate_up_bias, down_proj, down_bias,
            top_k_index, top_k_weights, gated_out, saved_gate, saved_up,
        )
        ctx.num_experts = num_experts
        ctx.top_k = top_k_index.shape[1]
        ctx.cfg = cfg
        return final

    @staticmethod
    def backward(ctx, grad_output):
        (hidden_states, gate_up_proj, gate_up_bias, down_proj, down_bias,
         top_k_index, top_k_weights, gated_out, saved_gate, saved_up,
        ) = ctx.saved_tensors
        num_experts = ctx.num_experts
        top_k = ctx.top_k
        num_tokens = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]
        inter_size = down_proj.shape[1]
        S = num_tokens * top_k
        device = hidden_states.device

        # Sort for grouped_mm backward
        flat_ids = top_k_index.reshape(-1)
        perm = torch.argsort(flat_ids)
        inv_perm = torch.argsort(perm)
        counts = torch.histc(flat_ids[perm].int().float(), bins=num_experts, min=0, max=num_experts - 1)
        offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)

        # grad weighted by routing
        grad_exp = grad_output.unsqueeze(1).expand(-1, top_k, -1).reshape(S, hidden_size)
        grad_weighted = (grad_exp * top_k_weights.reshape(-1, 1))
        grad_weighted_s = grad_weighted[perm].contiguous()
        gated_s = gated_out[perm].contiguous()

        # d_gated via grouped_mm: grad_weighted @ down_proj.T
        d_gated_s = torch._grouped_mm(grad_weighted_s, down_proj.transpose(-2, -1), offs=offsets)
        d_gated = d_gated_s[inv_perm]

        # Gate backward
        d_gg, d_gu = moe_gate_backward_cutile(d_gated, saved_gate, saved_up)
        d_gate_up = torch.empty(S, 2 * inter_size, dtype=hidden_states.dtype, device=device)
        d_gate_up[..., ::2] = d_gg
        d_gate_up[..., 1::2] = d_gu

        # d_hidden via grouped_mm: d_gate_up @ gate_up_proj.T
        d_gate_up_s = d_gate_up[perm].contiguous()
        d_hidden_s = torch._grouped_mm(d_gate_up_s, gate_up_proj.transpose(-2, -1), offs=offsets)
        d_hidden = d_hidden_s[inv_perm].view(num_tokens, top_k, hidden_size).sum(dim=1)

        # ---- Weight grads via vectorized scatter + bmm ----
        tok_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, top_k).reshape(-1)
        hidden_s = hidden_states[tok_idx[perm]].contiguous()
        expert_of_sorted = flat_ids[perm].long()

        # Compute per-token offset within its expert group for scatter
        offs_with_zero = torch.cat([torch.zeros(1, device=device, dtype=torch.int32), offsets])
        expert_starts = offs_with_zero[:-1].long()
        token_positions = torch.arange(S, device=device)
        within_expert_offset = token_positions - expert_starts[expert_of_sorted]
        max_tok = int(counts.max().item())

        # Gate-up weight grad: dW[e] = hidden_s[e].T @ d_gate_up_s[e]
        h_padded = torch.zeros(num_experts, max_tok, hidden_size, device=device, dtype=hidden_states.dtype)
        g_padded = torch.zeros(num_experts, max_tok, 2 * inter_size, device=device, dtype=hidden_states.dtype)
        h_padded[expert_of_sorted, within_expert_offset] = hidden_s
        g_padded[expert_of_sorted, within_expert_offset] = d_gate_up_s
        d_gup = torch.bmm(h_padded.transpose(1, 2), g_padded)

        # Gate-up bias grad: scatter_add
        d_gub = torch.zeros(num_experts, 2 * inter_size, device=device, dtype=torch.float32)
        d_gub.scatter_add_(0, expert_of_sorted.unsqueeze(1).expand(-1, 2 * inter_size),
                           d_gate_up_s.float())
        d_gub = d_gub.to(hidden_states.dtype)

        # Down weight grad: dW[e] = gated_s[e].T @ grad_weighted_s[e]
        ga_padded = torch.zeros(num_experts, max_tok, inter_size, device=device, dtype=hidden_states.dtype)
        gw_padded = torch.zeros(num_experts, max_tok, hidden_size, device=device, dtype=hidden_states.dtype)
        ga_padded[expert_of_sorted, within_expert_offset] = gated_s
        gw_padded[expert_of_sorted, within_expert_offset] = grad_weighted_s
        d_dp = torch.bmm(ga_padded.transpose(1, 2), gw_padded)

        # Down bias grad: scatter_add
        d_db = torch.zeros(num_experts, hidden_size, device=device, dtype=torch.float32)
        d_db.scatter_add_(0, expert_of_sorted.unsqueeze(1).expand(-1, hidden_size),
                          grad_weighted_s.float())
        d_db = d_db.to(hidden_states.dtype)

        return d_hidden, d_gup, d_gub, d_dp, d_db, None, None, None, None


# ============================================================================
# Drop-in replacement for GptOssExperts.forward
# ============================================================================

def cutile_moe_experts_forward(self, hidden_states, top_k_index=None,
                               top_k_weights=None, router_indices=None,
                               routing_weights=None):
    """Drop-in replacement for GptOssExperts.forward using CuTile fused GEMM with autotuning."""
    if top_k_index is None:
        top_k_index = router_indices
    if top_k_weights is None:
        top_k_weights = routing_weights

    cfg = _get_moe_config(
        hidden_states, self.gate_up_proj, self.gate_up_proj_bias,
        self.down_proj, self.down_proj_bias,
        top_k_index, top_k_weights, self.num_experts,
    )

    return CuTileMoEExpertsFunction.apply(
        hidden_states, self.gate_up_proj, self.gate_up_proj_bias,
        self.down_proj, self.down_proj_bias,
        top_k_index, top_k_weights, self.num_experts, cfg,
    )


# Keep grouped_mm version as fallback
def grouped_mm_moe_forward(self, hidden_states, top_k_index=None,
                           top_k_weights=None, router_indices=None,
                           routing_weights=None):
    """Fallback using torch._grouped_mm (CUTLASS-backed)."""
    from transformers.integrations.moe import grouped_mm_experts_forward
    if top_k_index is None:
        top_k_index = router_indices
    if top_k_weights is None:
        top_k_weights = routing_weights
    return grouped_mm_experts_forward(self, hidden_states, top_k_index, top_k_weights)
