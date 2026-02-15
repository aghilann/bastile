"""Profile PyTorch vs Bastile router/loss ops side-by-side."""

import importlib
import gc
import torch
import torch.profiler
from collections import defaultdict


def clear():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def make_config(num_layers=4):
    from transformers import GptOssConfig
    return GptOssConfig(
        vocab_size=201088, hidden_size=2880, intermediate_size=2880,
        num_hidden_layers=num_layers, num_attention_heads=64, num_key_value_heads=8,
        max_position_embeddings=131072, rope_theta=1000000.0,
        tie_word_embeddings=False, num_local_experts=128, num_experts_per_tok=4,
        sliding_window=128, attention_dropout=0.0,
        router_aux_loss_coef=0.9, output_router_logits=True,
    )


def profile_one(label, config, seq_len, setup_fn=None, warmup=3, profile_iters=3):
    import transformers.models.gpt_oss.modeling_gpt_oss as mod
    importlib.reload(mod)
    clear()

    if setup_fn:
        applied = setup_fn()
        print(f"  Patches: {applied}")

    mod = importlib.import_module("transformers.models.gpt_oss.modeling_gpt_oss")
    model = mod.GptOssForCausalLM(config).cuda().to(torch.bfloat16)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    ids = torch.randint(0, config.vocab_size, (1, seq_len), device="cuda")
    labels = ids.clone()

    for _ in range(warmup):
        opt.zero_grad()
        out = model(input_ids=ids, labels=labels)
        out.loss.backward()
        opt.step()
        torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(profile_iters):
            opt.zero_grad()
            out = model(input_ids=ids, labels=labels)
            out.loss.backward()
            opt.step()
            torch.cuda.synchronize()

    events = prof.key_averages()

    def cuda_time(e):
        return getattr(e, 'self_device_time_total', 0) or getattr(e, 'self_cuda_time_total', 0)

    cuda_events = [e for e in events if cuda_time(e) > 0]
    cuda_events.sort(key=lambda e: cuda_time(e), reverse=True)
    total_us = sum(cuda_time(e) for e in cuda_events)
    total_ms = total_us / 1000.0
    per_iter = total_ms / profile_iters

    print(f"\n  {label}: {per_iter:.1f} ms/iter  (total {total_ms:.1f} ms / {profile_iters} iters)\n")
    print(f"  {'Kernel':<65} {'ms':>8} {'%':>6} {'N':>6}")
    print(f"  {'-'*87}")
    for e in cuda_events[:20]:
        t = cuda_time(e) / 1000.0
        pct = 100.0 * cuda_time(e) / total_us
        print(f"  {e.key[:63]:<65} {t:>7.1f} {pct:>5.1f}% {e.count:>6}")

    # Keyword buckets
    buckets = {
        "Softmax (fwd+bwd)": ["softmax", "SoftMax"],
        "CatArrayBatchedCopy": ["CatArrayBatchedCopy"],
        "topk/sort": ["topk", "TopK", "radix_sort"],
        "one_hot": ["one_hot"],
        "atomic_add / CuTile router": ["atomic_add", "fused_router"],
        "Optimizer (multi_tensor)": ["multi_tensor"],
        "GEMM/CUTLASS": ["cutlass", "gemm", "cublas", "nvjet"],
        "indexing_backward": ["indexing_backward"],
        "elementwise_kernel": ["elementwise_kernel"],
    }
    print(f"\n  {'Bucket':<40} {'ms':>8} {'%':>6}")
    print(f"  {'-'*56}")
    for bname, kws in buckets.items():
        t = sum(cuda_time(e) for e in cuda_events
                if any(k.lower() in e.key.lower() for k in kws)) / 1000.0
        pct = 100.0 * t / total_ms if total_ms > 0 else 0
        if t > 0.1:
            print(f"  {bname:<40} {t:>7.1f} {pct:>5.1f}%")

    del model, opt
    clear()
    return per_iter


def setup_bastile():
    import bastile
    bastile.reset()
    return bastile.apply(
        rms_norm=True, swiglu=False, rope=True,
        fused_linear_cross_entropy=True,
        moe_experts=True, moe_router=True,
    )


def main():
    config = make_config(4)
    seq_len = 4096
    print(f"\n{'='*90}")
    print(f"  Router Profiling: PyTorch vs Bastile  (4 layers, seq_len={seq_len})")
    print(f"{'='*90}")

    print(f"\n--- PYTORCH ---")
    pt = profile_one("PyTorch", config, seq_len, warmup=3, profile_iters=3)

    print(f"\n\n--- BASTILE ---")
    ba = profile_one("Bastile", config, seq_len, setup_fn=setup_bastile, warmup=3, profile_iters=3)

    print(f"\n{'='*90}")
    print(f"  PyTorch: {pt:.1f} ms/iter")
    print(f"  Bastile: {ba:.1f} ms/iter")
    print(f"  Delta:   {(ba/pt - 1)*100:+.1f}%")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
