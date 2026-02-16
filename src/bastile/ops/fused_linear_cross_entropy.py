"""
Fused Linear Cross-Entropy Loss.

Two forward implementations:

1. **BT-chunked** (default, fast):
   Materializes (chunk_size, V) logits per BT chunk. Cross-entropy uses a
   CuTile 2-pass online-softmax kernel (single launch → loss + dlogits).
   dx is pre-computed in the forward; dw is deferred to backward.

2. **V-chunked** (memory-efficient):
   Never materializes more than (bt_chunk, v_chunk) logits.  Uses torch.mm
   for small GEMMs + PyTorch streaming LSE.  Trades compute (logits
   recomputation in pass 2) for memory.  Best for large BT where full
   (BT, V) would OOM.

Architecture Notes:
  CuTile does not expose cluster APIs or warp primitives, so our CE
  kernel uses a per-row persistent loop with gather-based column tiles.
  The GEMMs use cuBLAS via torch.mm for peak TensorCore throughput.
"""

import cuda.tile as ct
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.amp import custom_fwd, custom_bwd

from .utils import get_sm_count

ConstInt = ct.Constant[int]

_ALIGN = 8  # GEMM alignment


# ═══════════════════════════════════════════════════════════════════════════
# CuTile Cross-Entropy Kernel  (2-pass online softmax)
#
# Pass 1 — streaming max + sum_exp over V tiles  (1 read of logits)
# Pass 2 — write dlogits = softmax probs          (1 read + 1 write)
#
# Total: 2 reads + 1 write = 3 × (M × V × elem_size).
# Saves one full read+write vs F.log_softmax (alloc) + torch.exp (write).
# Target fixup (dlogits[target] -= 1) is fused into pass 2.
# ═══════════════════════════════════════════════════════════════════════════

@ct.kernel(occupancy=4)
def _ce_online_kernel(
    logits,          # (M, V) bf16 — overwritten with softmax probs
    loss_out,        # (M,) float32
    target_logits,   # (M,) float32 — pre-extracted logits[row, target]
    n_rows: ConstInt,
    V: ConstInt,
    TILE_V: ConstInt,
):
    """2-pass online-softmax kernel.

    Pass 1: streaming max + sum_exp (reads logits once).
    Pass 2: writes probs = exp(x−max)/sum_exp (reads logits once, writes once).
    Loss = log(sum_exp) + max − target_logit (written per row).
    Target fixup and ignore_index handled in Python after this kernel.
    """
    pid = ct.bid(0)
    num_blocks = ct.num_blocks(0)
    num_chunks = ct.cdiv(V, TILE_V)
    col_base = ct.arange(TILE_V, dtype=ct.int32)

    for row in range(pid, n_rows, num_blocks):
        # ── Pass 1: Online softmax (streaming max + sum_exp) ──
        row_max = ct.full((1,), -1e30, dtype=ct.float32)
        sum_exp = ct.full((1,), 0.0, dtype=ct.float32)

        for c in range(num_chunks):
            cols = ct.add(ct.full((TILE_V,), c * TILE_V, dtype=ct.int32), col_base)
            chunk = ct.gather(logits, (row, cols),
                              check_bounds=True, padding_value=-1e30)
            chunk_f32 = ct.astype(chunk, ct.float32)

            # Online update: track (max, sum_exp) jointly
            c_max = ct.max(chunk_f32, 0, keepdims=True)
            new_max = ct.maximum(row_max, c_max)
            sum_exp = ct.mul(sum_exp, ct.exp(ct.sub(row_max, new_max)))
            exp_c = ct.exp(ct.sub(chunk_f32, new_max))
            sum_exp = ct.add(sum_exp, ct.sum(exp_c, 0, keepdims=True))
            row_max = new_max

        # ── Loss = log(sum_exp) + max − target_logit ──
        lse = ct.add(row_max, ct.log(sum_exp))
        tgt_logit = ct.load(target_logits, index=(row,), shape=(1,),
                            padding_mode=ct.PaddingMode.ZERO)
        tgt_logit = ct.astype(tgt_logit, ct.float32)
        loss = ct.sub(ct.reshape(lse, (1,)), tgt_logit)
        ct.store(loss_out, index=(row,), tile=loss, allow_tma=False)

        # ── Pass 2: Write probs = exp(x − max) / sum_exp ──
        inv_sum = ct.truediv(ct.full((1,), 1.0, dtype=ct.float32), sum_exp)

        for c in range(num_chunks):
            cols = ct.add(ct.full((TILE_V,), c * TILE_V, dtype=ct.int32), col_base)
            chunk = ct.gather(logits, (row, cols),
                              check_bounds=True, padding_value=-1e30)
            chunk_f32 = ct.astype(chunk, ct.float32)
            probs = ct.mul(ct.exp(ct.sub(chunk_f32, row_max)), inv_sum)
            ct.scatter(logits, (row, cols),
                       ct.astype(probs, logits.dtype), check_bounds=True)


def _ce_cutile(logits_chunk: Tensor, target_chunk: Tensor,
               loss_chunk: Tensor, ignore_index: int):
    """CuTile online-softmax CE: kernel computes loss + probs, Python does fixup."""
    M, V = logits_chunk.shape
    device = logits_chunk.device

    # Pre-extract target logits (small gather, negligible)
    valid = target_chunk != ignore_index
    safe_target = target_chunk.clamp(min=0)
    rows = torch.arange(M, device=device)
    target_logits = logits_chunk[rows, safe_target].float()
    target_logits[~valid] = 0.0

    # Launch CuTile kernel: 2-pass online softmax → loss + probs
    TILE_V = 4096
    sms = get_sm_count()
    grid = (min(sms * 4, M),)
    ct.launch(
        torch.cuda.current_stream(), grid, _ce_online_kernel,
        (logits_chunk, loss_chunk, target_logits, M, V, TILE_V),
    )

    # Target fixup: dlogits = probs − one_hot(target)
    logits_chunk[rows[valid], safe_target[valid]] -= 1.0
    if not valid.all():
        logits_chunk[~valid] = 0
    loss_chunk[~valid] = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# BT-chunked forward  (default — fast path)
# ═══════════════════════════════════════════════════════════════════════════

def _chunked_fwd(
    x: Tensor, weight: Tensor, target: Tensor,
    chunk_size: int, ignore_index: int,
):
    BT, H = x.shape
    V = weight.shape[0]
    device = x.device
    num_chunks = (BT + chunk_size - 1) // chunk_size

    loss = torch.empty(BT, device=device, dtype=torch.float32)
    logits_buf = torch.empty((chunk_size, V), device=device, dtype=x.dtype)
    dx = torch.empty_like(x)
    dw = torch.zeros(V, H, device=device, dtype=torch.float32) if num_chunks > 1 else None
    # Reusable buffer for dw GEMM output — avoids per-chunk allocation + .float() temp
    dw_mm_buf = torch.empty(V, H, device=device, dtype=x.dtype) if num_chunks > 1 else None
    last_dlogits = None
    last_x_chunk = None

    for i in range(num_chunks):
        s, e = i * chunk_size, min((i + 1) * chunk_size, BT)
        clen = e - s
        x_chunk = x[s:e]
        tgt_chunk = target[s:e]
        loss_chunk = loss[s:e]
        dx_chunk = dx[s:e]
        logits_chunk = logits_buf[:clen]

        # GEMM1: logits
        torch.mm(x_chunk, weight.mT, out=logits_chunk)

        # CE: loss + dlogits (CuTile 2-pass online softmax)
        _ce_cutile(logits_chunk, tgt_chunk, loss_chunk, ignore_index)

        # GEMM2: dx
        torch.mm(logits_chunk, weight, out=dx_chunk)

        # dw accumulation — write into reusable bf16 buffer, add to fp32 accumulator
        # Avoids: (1) per-chunk allocation of dw_c, (2) the 2.4GB .float() temporary
        if i == num_chunks - 1:
            last_dlogits = logits_chunk
            last_x_chunk = x_chunk
        else:
            torch.mm(logits_chunk.t(), x_chunk, out=dw_mm_buf)
            if i == 0:
                dw.copy_(dw_mm_buf)    # fp32 ← bf16, no temp
            else:
                dw.add_(dw_mm_buf)     # fp32 += bf16, in-place, no temp

    return loss, dx, dw, last_dlogits, last_x_chunk


# ═══════════════════════════════════════════════════════════════════════════
# V-chunked forward  (memory-efficient — fused GEMM epilogue approach)
#
# Tip 1: Fuse the GEMM epilogue — compute logits for a tile of V columns,
#         apply CE math immediately, never materialise [BT, V].
# Tip 2: Streaming log-sum-exp trick for online softmax.
# Tip 3: Capture the target logit while iterating V tiles.
# ═══════════════════════════════════════════════════════════════════════════

def _v_chunked_fwd(
    x: Tensor, weight: Tensor, target: Tensor,
    bt_chunk: int, v_chunk: int, ignore_index: int,
):
    """
    V-chunked: only (bt_chunk, v_chunk) logits materialised at a time.

    Pass 1 — streaming LSE: for each V-tile, torch.mm + update (max, sum_exp).
    Pass 2 — dlogits/dx/dw: recompute logits per V-tile, derive gradients.
    """
    BT, H = x.shape
    V = weight.shape[0]
    device = x.device
    bt_chunks = (BT + bt_chunk - 1) // bt_chunk
    v_tiles = (V + v_chunk - 1) // v_chunk

    loss = torch.empty(BT, device=device, dtype=torch.float32)
    dx = torch.zeros(BT, H, device=device, dtype=x.dtype)
    dw = torch.zeros(V, H, device=device, dtype=torch.float32)

    # Small reusable buffer (peak memory = bt_chunk × v_chunk × elem_size)
    logits_buf = torch.empty((bt_chunk, v_chunk), device=device, dtype=x.dtype)

    for bi in range(bt_chunks):
        bs, be = bi * bt_chunk, min((bi + 1) * bt_chunk, BT)
        clen = be - bs
        x_c = x[bs:be]                  # (clen, H)
        tgt_c = target[bs:be]            # (clen,)
        loss_c = loss[bs:be]             # (clen,)
        dx_c = dx[bs:be]                 # (clen, H)

        # Per-row streaming softmax state
        row_max = torch.full((clen, 1), -1e30, device=device, dtype=torch.float32)
        sum_exp = torch.zeros((clen, 1), device=device, dtype=torch.float32)
        target_logit = torch.zeros(clen, device=device, dtype=torch.float32)

        # ── Pass 1: Streaming LSE over V-tiles ──
        for vi in range(v_tiles):
            vs, ve = vi * v_chunk, min((vi + 1) * v_chunk, V)
            vlen = ve - vs
            w_tile = weight[vs:ve]               # (vlen, H)
            lt = logits_buf[:clen, :vlen]
            torch.mm(x_c, w_tile.mT, out=lt)     # small GEMM

            # Online softmax update (PyTorch ops on small tiles — fast)
            lt_f32 = lt.float()
            tile_max = lt_f32.max(dim=-1, keepdim=True).values
            new_max = torch.maximum(row_max, tile_max)
            sum_exp.mul_(torch.exp(row_max - new_max))
            sum_exp.add_(torch.exp(lt_f32 - new_max).sum(dim=-1, keepdim=True))
            row_max = new_max

            # Capture target logit
            in_range = (tgt_c >= vs) & (tgt_c < ve)
            if in_range.any():
                local_idx = tgt_c[in_range] - vs
                rows_in = torch.arange(clen, device=device)[in_range]
                target_logit[in_range] = lt_f32[rows_in, local_idx]

        # ── Loss ──
        lse = (row_max.squeeze(-1) + sum_exp.squeeze(-1).log())
        loss_c.copy_(lse - target_logit)
        ignored = tgt_c == ignore_index
        loss_c[ignored] = 0.0

        # ── Pass 2: Recompute logits → dlogits → accumulate dx, dw ──
        inv_sum = 1.0 / sum_exp          # (clen, 1) float32

        for vi in range(v_tiles):
            vs, ve = vi * v_chunk, min((vi + 1) * v_chunk, V)
            vlen = ve - vs
            w_tile = weight[vs:ve]
            lt = logits_buf[:clen, :vlen]
            torch.mm(x_c, w_tile.mT, out=lt)     # recompute

            # dlogits = softmax probs
            lt_f32 = lt.float()
            probs = torch.exp(lt_f32 - row_max) * inv_sum   # (clen, vlen) fp32

            # Target fixup: probs[row, tgt_col] -= 1
            in_range = (tgt_c >= vs) & (tgt_c < ve)
            if in_range.any():
                local_idx = tgt_c[in_range] - vs
                rows_in = torch.arange(clen, device=device)[in_range]
                probs[rows_in, local_idx] -= 1.0
            probs[ignored] = 0.0

            # Convert to input dtype for GEMMs
            dlogits = probs.to(x.dtype)           # (clen, vlen)

            # dx += dlogits @ W_tile             (accumulate over V-tiles)
            dx_c.add_(torch.mm(dlogits, w_tile))

            # dw[vs:ve] = dlogits.T @ x          (no accumulation needed — disjoint V)
            dw[vs:ve].copy_(torch.mm(dlogits.t(), x_c).float())

    return loss, dx, dw


# ═══════════════════════════════════════════════════════════════════════════
# Autograd Function
# ═══════════════════════════════════════════════════════════════════════════

class ChunkedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, x, weight, target, ignore_index, reduction, chunk_size,
                v_chunk_size):
        batch_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        BT = x_flat.shape[0]
        pad = (-BT) % _ALIGN
        if pad:
            x_flat = F.pad(x_flat, (0, 0, 0, pad))
            target_flat = F.pad(target.view(-1), (0, pad), value=ignore_index)
        else:
            target_flat = target.view(-1)

        if v_chunk_size is not None:
            # V-chunked (memory-efficient)
            loss, dx, dw = _v_chunked_fwd(
                x_flat, weight, target_flat, chunk_size, v_chunk_size, ignore_index
            )
            last_dlogits = last_x_chunk = None
        else:
            # BT-chunked (fast, default)
            loss, dx, dw, last_dlogits, last_x_chunk = _chunked_fwd(
                x_flat, weight, target_flat, chunk_size, ignore_index
            )

        if pad:
            loss = loss[:BT]
            dx = dx[:BT]

        loss_sum = loss.sum()
        n_valid = (target_flat[:BT] != ignore_index).sum().float()
        loss_scale = None if reduction == "sum" else (
            torch.tensor(1.0 / n_valid.item(), device=x.device, dtype=torch.float32)
            if n_valid > 0 else torch.tensor(0.0, device=x.device, dtype=torch.float32)
        )

        # For V-chunked, dw is fully computed; for BT-chunked, last chunk deferred
        if v_chunk_size is not None:
            # dw is complete — scale it now
            if loss_scale is not None:
                scaled = loss_sum * loss_scale
            else:
                scaled = loss_sum
            ctx.save_for_backward(dx, dw.to(weight.dtype),
                                  loss_scale if loss_scale is not None else torch.tensor(0.0, device=x.device))
            ctx.batch_shape = batch_shape
            ctx.weight_dtype = weight.dtype
            ctx.v_chunked = True
        else:
            ctx.save_for_backward(
                dx,
                dw if dw is not None else torch.tensor(0.0, device=x.device),
                last_dlogits if last_dlogits is not None else torch.tensor(0.0, device=x.device),
                last_x_chunk if last_x_chunk is not None else torch.tensor(0.0, device=x.device),
                loss_scale if loss_scale is not None else torch.tensor(0.0, device=x.device),
            )
            ctx.batch_shape = batch_shape
            ctx.weight_dtype = weight.dtype
            ctx.v_chunked = False
            ctx.has_scale = loss_scale is not None
            ctx.has_dw = dw is not None

        return loss_sum if loss_scale is None else loss_sum * loss_scale

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, dloss):
        if ctx.v_chunked:
            dx, dw, loss_scale = ctx.saved_tensors
            scale = dloss * loss_scale if loss_scale.item() != 0 else dloss
            dx.mul_(scale)
            dw.mul_(scale.to(dw.dtype))
            return dx.reshape(*ctx.batch_shape, dx.shape[-1]), dw, None, None, None, None, None
        else:
            dx, dw, last_dlogits, last_x_chunk, loss_scale = ctx.saved_tensors
            if ctx.has_scale:
                dloss = dloss * loss_scale
            dx.mul_(dloss)
            dx = dx.reshape(*ctx.batch_shape, dx.shape[-1])

            dw_last = torch.mm(last_dlogits.t(), last_x_chunk)  # (V, H) bf16
            if not ctx.has_dw:
                # Single chunk — dw_last is the only contribution
                dw_last.mul_(dloss.to(dw_last.dtype))
                dw = dw_last if dw_last.dtype == ctx.weight_dtype else dw_last.to(ctx.weight_dtype)
            else:
                # Multi-chunk: dw (fp32) += dloss * dw_last (bf16)
                # Avoid .float() temp by letting PyTorch upcast bf16→fp32 in-place
                dw.mul_(dloss)
                dw.add_(dw_last, alpha=float(dloss))  # fp32 += bf16, no 2.4GB temp
                # Write result back into dw_last buffer (bf16) to avoid new allocation
                dw_last.copy_(dw)
                dw = dw_last

            return dx, dw, None, None, None, None, None


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def fused_linear_cross_entropy(
    hidden_states: Tensor,
    weight: Tensor,
    target: Tensor,
    bias=None,
    ignore_index: int = -100,
    chunk_size: int = 4096,
    reduction: str = "mean",
    v_chunk_size: int | None = None,
) -> Tensor:
    """Chunked fused linear + cross-entropy.

    Args:
        v_chunk_size: If set, use V-chunked mode (memory-efficient).
                      Only (chunk_size, v_chunk_size) logits materialised.
    """
    if hidden_states.ndim == 3:
        B, T, H = hidden_states.shape
        hidden_states = hidden_states.view(-1, H)
        target = target.view(-1)

    if bias is not None:
        logits = F.linear(hidden_states, weight, bias)
        return F.cross_entropy(logits, target, ignore_index=ignore_index, reduction=reduction)

    return ChunkedLinearCrossEntropyFunction.apply(
        hidden_states, weight, target, ignore_index, reduction, chunk_size,
        v_chunk_size,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Model forward patch
# ═══════════════════════════════════════════════════════════════════════════

def bastile_lce_forward(
    self,
    input_ids=None, attention_mask=None, position_ids=None,
    past_key_values=None, inputs_embeds=None, labels=None,
    use_cache=None, output_attentions=None, output_hidden_states=None,
    cache_position=None, logits_to_keep=0, return_dict=None, **kwargs,
):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
        past_key_values=past_key_values, inputs_embeds=inputs_embeds,
        use_cache=use_cache, output_attentions=output_attentions,
        output_hidden_states=output_hidden_states, cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]
    logits = None
    loss = None

    if self.training and labels is not None:
        shift_hidden = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = fused_linear_cross_entropy(
            shift_hidden, self.lm_head.weight, shift_labels,
            bias=getattr(self.lm_head, 'bias', None), ignore_index=-100,
        )
    else:
        s = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, s, :])
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels,
                                     vocab_size=self.config.vocab_size, **kwargs)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output

    from transformers.modeling_outputs import CausalLMOutputWithPast
    return CausalLMOutputWithPast(
        loss=loss, logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Warmup
# ═══════════════════════════════════════════════════════════════════════════

def warmup_fused_lce(
    hidden_size: int = 4096, vocab_size: int = 151936,
    dtype: torch.dtype = torch.bfloat16, device: str = "cuda",
):
    """JIT-compile the CuTile CE kernel + warm cuBLAS caches."""
    x = torch.randn(8, hidden_size, dtype=dtype, device=device, requires_grad=True)
    w = torch.randn(vocab_size, hidden_size, dtype=dtype, device=device, requires_grad=True)
    t = torch.randint(0, vocab_size, (8,), device=device)
    loss = fused_linear_cross_entropy(x, w, t)
    loss.backward()
    torch.cuda.synchronize()
