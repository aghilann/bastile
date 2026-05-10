"""E2E benchmark: DeepSeek V4 decoder layer(s) at real dims, forward-only.

Creates a small DSV4 model (2-4 layers) with real Pro config dimensions,
patches CSA attention layers to use Bastile's cuTILE sparse_attn kernel
(replacing eager_attention_forward), and compares forward-pass wall time
and output correctness across varying sequence lengths.

Uses ``AutoModelForCausalLM.from_config`` to avoid downloading full weights.
Weights are randomly initialised on GPU.
"""

from __future__ import annotations

import argparse
import gc
import os
import types
import time

import torch
import torch.nn as nn

from bastile.ops.sparse_attn import sparse_attn
from tests.benchmarks.utils import (
    clear_cuda_state,
    print_gpu_info,
    print_header,
)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _make_dsv4_config(num_layers: int = 3, num_experts: int = 8):
    """Build a DeepseekV4Config that fits in a single consumer GPU.

    Uses ``from_pretrained`` to inherit the full structure (rope params,
    layer schedules, etc.) but overrides dimensions and expert count
    so the model is small enough for benchmarking. For 8 experts +
    3 layers the model is ~4.5 GB in bf16.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        "deepseek-ai/DeepSeek-V4-Pro",
        trust_remote_code=True,
    )
    # Shrink to Flash-class dimensions and reduce expert count
    config.hidden_size = 4096
    config.num_attention_heads = 64
    config.head_dim = 512
    config.q_lora_rank = 1024
    config.o_groups = 8
    config.o_lora_rank = 1024
    config.moe_intermediate_size = 2048
    config.num_hidden_layers = num_layers
    config.n_routed_experts = num_experts
    config.torch_dtype = "bfloat16"
    return config


def _materialize_model(config, device: torch.device):
    """Create model from config with random-weight initialisation on device.

    Uses ``to_empty`` to allocate directly on the target device (avoids meta-
    device issues with newer PyTorch), then normal-inits weights manually.
    """
    from transformers import AutoModelForCausalLM

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    model.to_empty(device=device)

    for name, param in model.named_parameters():
        if param.ndim >= 2:
            nn.init.normal_(param.data, std=config.initializer_range)
        elif "bias" not in name:
            nn.init.normal_(param.data, std=config.initializer_range)
        else:
            nn.init.zeros_(param.data)

    for name, buf in model.named_buffers():
        if buf.device.type == "meta":
            device_tensor = torch.empty_like(buf, device=device)
            device_tensor.zero_()
            buf.data.copy_(device_tensor)

    return model


# ---------------------------------------------------------------------------
# Patching: replace eager_attention_forward with sparse_attn for CSA layers
# ---------------------------------------------------------------------------

def _patch_attention_for_sparse(model, verbose: bool = True):
    """Replace eager_attention_forward with sparse_attn for CSA layers.

    For sliding_attention and HCA layers, eager is kept unchanged.
    For CSA layers, the attention call is replaced with sparse_attn using
    indices built from the attention_mask so that each query only attends
    to positions not masked out (same set eager sees).
    """
    patched = 0

    for _name, module in model.named_modules():
        if type(module).__name__ != "DeepseekV4Attention":
            continue
        if module.layer_type != "compressed_sparse_attention":
            continue
        _patch_single_csa_attention(module)
        patched += 1

    if verbose:
        print(f"  Patched {patched} DeepseekV4Attention CSA layers for sparse_attn")
    return model


def _patch_single_csa_attention(attn):
    """Monkey-patch a single DeepseekV4Attention CSA layer's forward."""
    original_forward = attn.forward

    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # ---- same as original until after kv concatenation ----
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
        q = self.q_b_norm(q)
        q = apply_rotary_pos_emb(q, cos, sin)

        kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)
        kv = apply_rotary_pos_emb(kv, cos, sin)

        if past_key_values is not None:
            kv = past_key_values.update(kv, kv, self.layer_idx)[0]

        if self.compressor is not None:
            compressed_kv = self.compressor(
                hidden_states, q_residual, position_ids, past_key_values, self.layer_idx
            )
            kv = torch.cat([kv, compressed_kv], dim=2)

        if isinstance(attention_mask, torch.Tensor) and kv.shape[2] > attention_mask.shape[-1]:
            attention_mask = torch.nn.functional.pad(
                attention_mask, (0, kv.shape[2] - attention_mask.shape[-1]), value=0.0
            )

        # ---- sparse_attn replaces eager_attention_forward ----
        B, H, S, D = q.shape
        T_total = kv.shape[2]  # sw + compressed positions
        kv_2d = kv.squeeze(0).contiguous()  # [B, T_total, D]
        q_2d = q.transpose(1, 2).contiguous()  # [B, S, H, D]

        # Build top-k indices from the attention mask so sparse_attn sees
        # exactly the same positions that eager would (causal sliding window
        # + all compressed entries).
        mask_2d = attention_mask.squeeze(0).squeeze(0)  # [S, T_total]
        # valid positions = mask value > -inf threshold (0 = visible, -inf = masked)
        valid_counts = (mask_2d > -1e10).sum(dim=-1)  # [S]
        max_topk = int(valid_counts.max().item())

        # Sort: valid positions (mask=0) sort before invalid (mask=-inf)
        sorted_idx = mask_2d.argsort(dim=-1, descending=True)  # [S, T_total]
        topk_idxs = sorted_idx[:, :max_topk].unsqueeze(0)  # [1, S, max_topk]

        # For queries with fewer valid positions than max_topk, pad tail
        # slots with index T_total (one-past-end) so PAD_ZERO gives KV=0
        # and those padded positions contribute ~0 weight (exp(0 - m) ≈ 0
        # once there's any real KV pushing m up).
        for s_idx in range(S):
            n = int(valid_counts[s_idx].item())
            if n < max_topk:
                topk_idxs[0, s_idx, n:] = T_total

        topk_idxs = topk_idxs.expand(B, -1, -1).contiguous().to(torch.int32)

        attn_output_sparse = sparse_attn(
            q_2d, kv_2d, self.sinks, topk_idxs, self.scaling
        )  # [B, S, H, D]
        attn_output = attn_output_sparse.transpose(1, 2).contiguous()  # [B, H, S, D]
        attn_weights = None  # sparse_attn doesn't return weights

        # ---- rest is same as original ----
        attn_output = apply_rotary_pos_emb(
            attn_output.transpose(1, 2), cos, -sin
        ).transpose(1, 2)

        grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)
        grouped = self.o_a_proj(grouped).flatten(2)
        output = self.o_b_proj(grouped)
        return output, attn_weights

    attn.forward = types.MethodType(patched_forward, attn)


# import apply_rotary_pos_emb from the model source
def _import_apply_rotary_pos_emb():
    from transformers.models.deepseek_v4.modular_deepseek_v4 import apply_rotary_pos_emb as _fn
    return _fn


apply_rotary_pos_emb = _import_apply_rotary_pos_emb()


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _create_inputs(config, S: int, device: torch.device, dtype: torch.dtype):
    B = 1
    input_ids = torch.randint(0, config.vocab_size - 1, (B, S), device=device)
    attention_mask = torch.ones(B, S, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _bench_forward(model, inputs, warmup_iters: int, timed_iters: int) -> float:
    """Run forward pass and return median iteration time in ms."""
    for _ in range(warmup_iters):
        with torch.no_grad():
            _ = model(**inputs)
    torch.cuda.synchronize()

    times = []
    for _ in range(timed_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            _ = model(**inputs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]  # median


def _check_correctness(ref_logits, test_logits, atol: float = 1e-2, rtol: float = 1e-2):
    """Compare patched vs unpatched model outputs."""
    torch.testing.assert_close(
        test_logits.float(), ref_logits.float(),
        rtol=rtol, atol=atol,
        msg="Patched model output diverges from reference",
    )
    print(f"  Correctness: outputs match within atol={atol}, rtol={rtol}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-layers", type=int, default=3,
                        help="Number of decoder layers (3+, to include at least one CSA layer)")
    parser.add_argument("--seq-lens", type=str,
                        default="128,256,512,1024,2048,4096",
                        help="Comma-separated sequence lengths")
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--timed-iters", type=int, default=15)
    parser.add_argument("--skip-correctness", action="store_true",
                        help="Skip correctness check (output comparison)")
    parser.add_argument("--correctness-seq-len", type=int, default=4096,
                        help="Sequence length used for correctness check")
    args = parser.parse_args()

    seq_lens = [int(s.strip()) for s in args.seq_lens.split(",") if s.strip()]
    dtype = torch.bfloat16
    device = torch.device("cuda")

    print_header("DeepSeek V4 sparse-attention E2E benchmark", width=80)
    print_gpu_info()
    print(f"  Layers: {args.num_layers}, dtype: {dtype}")
    print(f"  Seq lengths: {seq_lens}")
    print(f"  Warmup iters: {args.warmup_iters}, timed iters: {args.timed_iters}")
    print()

    torch.manual_seed(42)

    # --- Build model ---
    print("--- Building model ---")
    config = _make_dsv4_config(args.num_layers)
    print(f"  Config: {config.num_hidden_layers} layers, "
          f"hidden={config.hidden_size}, heads={config.num_attention_heads}, "
          f"head_dim={config.head_dim}")
    print(f"  Indexer: n_heads={config.index_n_heads}, head_dim={config.index_head_dim}, "
          f"topk={config.index_topk}")
    print(f"  Compress rates: {config.compress_rates}")
    print(f"  Layer types: {config.layer_types[:args.num_layers]}")

    # Unpatched (reference)
    model_eager = _materialize_model(config, device)
    model_eager.eval()
    param_count = sum(p.numel() for p in model_eager.parameters())
    print(f"  Parameters: {param_count:,} ({param_count * 2 / 1e9:.2f} GB bf16)")

    # Patched (Bastile sparse_attn)
    model_bastile = _materialize_model(config, device)
    model_bastile.eval()
    _patch_attention_for_sparse(model_bastile)

    # --- Correctness check ---
    if not args.skip_correctness:
        print("\n--- Correctness check ---")
        S_check = args.correctness_seq_len
        inputs_check = _create_inputs(config, S_check, device, dtype)
        print(f"  Running forward at S={S_check}...")

        with torch.no_grad():
            out_eager = model_eager(**inputs_check)
        with torch.no_grad():
            out_bastile = model_bastile(**inputs_check)

        _check_correctness(out_eager.logits, out_bastile.logits)
        del out_eager, out_bastile, inputs_check
        clear_cuda_state()

    # --- Benchmark sweep ---
    print("\n--- Benchmark sweep ---")
    _ = torch.cuda.max_memory_allocated()

    for S in seq_lens:
        inputs = _create_inputs(config, S, device, dtype)

        # Warmup both
        _bench_forward(model_eager, inputs, 1, 1)
        _bench_forward(model_bastile, inputs, 1, 1)
        torch.cuda.reset_peak_memory_stats()

        eager_ms = _bench_forward(model_eager, inputs, args.warmup_iters, args.timed_iters)
        bastile_ms = _bench_forward(model_bastile, inputs, args.warmup_iters, args.timed_iters)
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3

        speedup = eager_ms / bastile_ms if bastile_ms > 0 else float("inf")
        print(f"  S={S:5d}: eager={eager_ms:.2f}ms, bastile={bastile_ms:.2f}ms, "
              f"speedup={speedup:.2f}x, peak_mem={peak_mem:.2f}GB")

    del model_eager, model_bastile
    clear_cuda_state()
    gc.collect()


if __name__ == "__main__":
    main()