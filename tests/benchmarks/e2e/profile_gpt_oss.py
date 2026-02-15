"""
Profile GPT-OSS-20B to identify where time is spent per operation.

Uses torch.profiler to capture CUDA kernel-level timing for a full
forward + backward pass, then aggregates by operation category.
"""

import argparse
import importlib
import torch
import torch.profiler
from collections import defaultdict


def make_gpt_oss_20b_config(num_layers: int = 4):
    from transformers import GptOssConfig
    return GptOssConfig(
        vocab_size=201088,
        hidden_size=2880,
        intermediate_size=2880,
        num_hidden_layers=num_layers,
        num_attention_heads=64,
        num_key_value_heads=8,
        max_position_embeddings=131072,
        rope_theta=1000000.0,
        tie_word_embeddings=False,
        num_local_experts=128,
        num_experts_per_tok=4,
        sliding_window=128,
        attention_dropout=0.0,
        router_aux_loss_coef=0.9,
        output_router_logits=True,
    )


# ── Categorize CUDA kernels into logical ops ──
CATEGORIES = [
    # Attention
    ("Attention SDPA",      ["sdpa", "flash", "fmha", "efficient_attention", "mem_efficient",
                             "cutlass_fused_attention", "flash_fwd", "flash_bwd"]),
    ("Attention QKV Proj",  ["q_proj", "k_proj", "v_proj", "qkv"]),
    ("Attention Out Proj",  ["o_proj", "out_proj"]),

    # MoE
    ("MoE Router",          ["router", "topk", "top_k", "gating", "softmax"]),
    ("MoE grouped_mm",      ["grouped_mm", "grouped_gemm", "_grouped_mm"]),
    ("MoE Gate (SiLU)",     ["silu", "swiglu", "gelu_and_mul", "_apply_gate"]),
    ("MoE Sorting/Scatter", ["scatter", "gather", "sort", "argsort", "index_select",
                             "index_put", "index_add", "moe_align"]),

    # Normalization
    ("RMSNorm",             ["rms_norm", "rmsnorm", "layer_norm", "layernorm"]),

    # Embeddings / RoPE
    ("RoPE",                ["rotary", "rope"]),
    ("Embeddings",          ["embedding"]),

    # Cross-Entropy / Loss
    ("Cross-Entropy",       ["cross_entropy", "nll_loss", "log_softmax", "lm_head"]),

    # GEMM (general)
    ("GEMM (other)",        ["gemm", "cutlass", "cublas", "ampere_", "sm80_", "sm90_",
                             "volta_", "turing_", "_sgemm", "_hgemm"]),

    # Elementwise / Reduction
    ("Elementwise",         ["add_", "mul_", "copy_", "fill_", "cast_", "convert",
                             "elementwise", "binary", "unary"]),
    ("Reduction",           ["reduce", "sum", "mean", "amax", "all_reduce"]),

    # Optimizer
    ("Optimizer",           ["adam", "sgd", "adamw", "step", "fused_adam"]),

    # Memory
    ("Memory ops",          ["memcpy", "memset", "zero_", "ones_", "empty", "resize",
                             "allocat"]),
]


def categorize_kernel(name: str) -> str:
    name_lower = name.lower()
    for category, keywords in CATEGORIES:
        for kw in keywords:
            if kw.lower() in name_lower:
                return category
    return "Other"


def run_profile(num_layers: int = 4, seq_len: int = 4096, batch_size: int = 1,
                warmup: int = 3, profile_iters: int = 3):
    """Run profiling on vanilla PyTorch GPT-OSS."""

    # Ensure clean module state
    import transformers.models.gpt_oss.modeling_gpt_oss as gpt_oss_mod
    importlib.reload(gpt_oss_mod)

    config = make_gpt_oss_20b_config(num_layers=num_layers)

    print(f"\n{'='*90}")
    print(f"  Profiling GPT-OSS-20B (Vanilla PyTorch)")
    print(f"  {num_layers} layers, {config.num_local_experts} experts, top-{config.num_experts_per_tok}")
    print(f"  batch={batch_size}, seq_len={seq_len}, warmup={warmup}, profile_iters={profile_iters}")
    print(f"{'='*90}\n")

    model = gpt_oss_mod.GptOssForCausalLM(config).cuda().to(torch.bfloat16)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {total_params/1e9:.2f}B parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
    labels = input_ids.clone()
    labels[:, :1] = -100

    # Warmup
    print(f"  Warming up ({warmup} iters)...")
    for i in range(warmup):
        optimizer.zero_grad()
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
    print(f"  Warmup done.\n")

    # Profile
    print(f"  Profiling ({profile_iters} iters)...")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for i in range(profile_iters):
            optimizer.zero_grad()
            out = model(input_ids=input_ids, labels=labels)
            out.loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

    print(f"  Profiling done.\n")

    # ── Analyze results ──
    events = prof.key_averages()

    # 1. Raw top-N CUDA kernels by total time
    print(f"{'='*90}")
    print(f"  TOP 30 CUDA KERNELS BY TOTAL GPU TIME")
    print(f"{'='*90}\n")

    def cuda_time(e):
        """Get CUDA time from event, handling different profiler API versions."""
        if hasattr(e, 'self_device_time_total'):
            return e.self_device_time_total
        elif hasattr(e, 'self_cuda_time_total'):
            return e.self_cuda_time_total
        return 0

    cuda_events = [e for e in events if cuda_time(e) > 0]
    cuda_events.sort(key=lambda e: cuda_time(e), reverse=True)

    total_cuda_us = sum(cuda_time(e) for e in cuda_events)
    total_cuda_ms = total_cuda_us / 1000.0

    print(f"  {'Kernel':<60} {'Time(ms)':>10} {'%':>8} {'Calls':>8}")
    print(f"  {'-'*88}")
    for e in cuda_events[:30]:
        t_ms = cuda_time(e) / 1000.0
        pct = 100.0 * cuda_time(e) / total_cuda_us if total_cuda_us > 0 else 0
        print(f"  {e.key[:58]:<60} {t_ms:>9.1f} {pct:>7.1f}% {e.count:>8}")

    print(f"\n  Total CUDA time: {total_cuda_ms:.1f} ms ({profile_iters} iters)")
    print(f"  Per-iter: {total_cuda_ms/profile_iters:.1f} ms\n")

    # 2. Categorized breakdown
    print(f"{'='*90}")
    print(f"  CATEGORIZED BREAKDOWN")
    print(f"{'='*90}\n")

    cat_times = defaultdict(float)
    cat_counts = defaultdict(int)
    cat_kernels = defaultdict(list)

    for e in cuda_events:
        cat = categorize_kernel(e.key)
        t_ms = cuda_time(e) / 1000.0
        cat_times[cat] += t_ms
        cat_counts[cat] += e.count
        cat_kernels[cat].append((e.key, t_ms, e.count))

    # Sort categories by time
    sorted_cats = sorted(cat_times.items(), key=lambda x: x[1], reverse=True)

    print(f"  {'Category':<30} {'Time(ms)':>10} {'%':>8} {'Calls':>8}")
    print(f"  {'-'*58}")
    for cat, t_ms in sorted_cats:
        pct = 100.0 * t_ms / (total_cuda_ms) if total_cuda_ms > 0 else 0
        if pct < 0.1:
            continue
        print(f"  {cat:<30} {t_ms:>9.1f} {pct:>7.1f}% {cat_counts[cat]:>8}")

    print(f"\n  Total: {total_cuda_ms:.1f} ms")
    per_iter_ms = total_cuda_ms / profile_iters
    print(f"  Per-iter: {per_iter_ms:.1f} ms\n")

    # 3. Show top kernels per category for top categories
    print(f"{'='*90}")
    print(f"  TOP KERNELS PER CATEGORY (top 5 categories)")
    print(f"{'='*90}\n")

    for cat, t_ms in sorted_cats[:7]:
        pct = 100.0 * t_ms / total_cuda_ms if total_cuda_ms > 0 else 0
        print(f"  ── {cat} ({t_ms:.1f} ms, {pct:.1f}%) ──")
        kernels = sorted(cat_kernels[cat], key=lambda x: x[1], reverse=True)
        for kname, kt, kc in kernels[:5]:
            kpct = 100.0 * kt / total_cuda_ms if total_cuda_ms > 0 else 0
            print(f"     {kname[:65]:<67} {kt:>7.1f}ms  {kpct:>5.1f}%  x{kc}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Profile GPT-OSS-20B")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--profile-iters", type=int, default=3)
    args = parser.parse_args()

    run_profile(
        num_layers=args.layers,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        warmup=args.warmup,
        profile_iters=args.profile_iters,
    )


if __name__ == "__main__":
    main()
