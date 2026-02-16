"""Quick A/B benchmark: PyTorch vs Bastile on GPT-OSS-20B."""

import importlib
import time
import torch
import gc


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


def bench(name, config, seq_len, setup_fn=None, warmup=3, duration=8.0):
    import transformers.models.gpt_oss.modeling_gpt_oss as mod
    importlib.reload(mod)
    clear()

    if setup_fn:
        applied = setup_fn()
        mod = importlib.import_module("transformers.models.gpt_oss.modeling_gpt_oss")

    model = mod.GptOssForCausalLM(config).cuda().to(torch.bfloat16)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

    ids = torch.randint(0, config.vocab_size, (1, seq_len), device="cuda")
    labels = ids.clone()

    # Warmup
    for _ in range(warmup):
        opt.zero_grad()
        out = model(input_ids=ids, labels=labels)
        out.loss.backward()
        opt.step()
        torch.cuda.synchronize()

    # Timed
    torch.cuda.reset_peak_memory_stats()
    iters = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration:
        opt.zero_grad()
        out = model(input_ids=ids, labels=labels)
        out.loss.backward()
        opt.step()
        torch.cuda.synchronize()
        iters += 1

    elapsed = time.perf_counter() - t0
    tok_s = iters * seq_len / elapsed
    ms_iter = elapsed / iters * 1000
    mem = torch.cuda.max_memory_allocated() / 1e9

    del model, opt
    clear()
    return tok_s, ms_iter, mem, iters


def setup_bastile():
    import bastile
    bastile.reset()
    return bastile.apply(
        rms_norm=True, swiglu=False, rope=True,
        fused_linear_cross_entropy=True,
        moe_experts=True, moe_router=True,
    )


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--seq-lens", type=str, default="256,1024,4096,8192")
    args = p.parse_args()

    config = make_config(args.layers)
    seq_lens = [int(s) for s in args.seq_lens.split(",")]

    print(f"\n  Quick Bench: GPT-OSS-20B ({args.layers} layers, 128 experts)")
    print(f"  Duration: {args.duration}s per run, seq_lens: {seq_lens}\n")
    print(f"  {'seq_len':>8}  {'PyTorch tok/s':>14}  {'Bastile tok/s':>14}  {'Speedup':>8}  {'PT mem':>8}  {'B mem':>8}")
    print(f"  {'-'*76}")

    for sl in seq_lens:
        pt_tok, pt_ms, pt_mem, pt_it = bench("PyTorch", config, sl, duration=args.duration)
        ba_tok, ba_ms, ba_mem, ba_it = bench("Bastile", config, sl, setup_fn=setup_bastile, duration=args.duration)
        spd = (ba_tok / pt_tok - 1) * 100
        print(f"  {sl:>8}  {pt_tok:>13,.0f}  {ba_tok:>13,.0f}  {spd:>+7.1f}%  {pt_mem:>7.1f}G  {ba_mem:>7.1f}G")

    print()


if __name__ == "__main__":
    main()
