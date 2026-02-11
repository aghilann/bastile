"""
Qwen3 0.6B Sequence Length Sweep: PyTorch vs Liger Kernel vs Bastile
Tests across seq_len = [1024, 2048, 4096, 8192] with batch_size=4.

Uses Qwen3 ~0.6B parameter config (24 layers, 896 hidden).
"""

import torch
import gc
import time
import importlib
import traceback
from typing import Optional, Callable, List, Dict

from ..utils import (
    clear_cuda_state,
    reset_peak_memory,
    get_peak_memory_gb,
    print_header,
    print_gpu_info,
    E2EBenchmarkResult,
)


SEQ_LENS = [1024, 2048, 4096, 8192]
BATCH_SIZE = 4
WARMUP_ITERS = 10
DURATION_SEC = 15.0


def reset_environment():
    clear_cuda_state()
    reset_peak_memory()
    import transformers.models.qwen3.modeling_qwen3 as qwen3_mod
    importlib.reload(qwen3_mod)


def make_qwen3_06b_config():
    from transformers import Qwen3Config
    return Qwen3Config(
        vocab_size=32000,
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        head_dim=64,
    )


def run_benchmark(
    name: str,
    config,
    setup_fn: Optional[Callable] = None,
    batch_size: int = BATCH_SIZE,
    seq_len: int = 2048,
    warmup_iters: int = WARMUP_ITERS,
    duration_sec: float = DURATION_SEC,
) -> Optional[E2EBenchmarkResult]:
    reset_environment()

    if setup_fn:
        print(f"  Applying {name} patches...")
        applied = setup_fn()
        if applied:
            print(f"  Patches applied: {applied}")

    from transformers import Qwen3ForCausalLM

    print(f"  Creating Qwen3 0.6B model...")
    try:
        model = Qwen3ForCausalLM(config).cuda().to(torch.bfloat16)
    except torch.cuda.OutOfMemoryError:
        print(f"  OOM creating model — skipping")
        clear_cuda_state()
        return None

    model.train()
    num_params = sum(p.numel() for p in model.parameters())
    tokens_per_iter = batch_size * seq_len
    print(f"  Model: {num_params / 1e9:.2f}B parameters")
    print(f"  Batch: {batch_size} x {seq_len} = {tokens_per_iter} tokens/iter")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
    attention_mask = torch.ones_like(input_ids)

    print(f"  Warming up ({warmup_iters} iterations)...")
    try:
        for _ in range(warmup_iters):
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
            optimizer.step()
    except torch.cuda.OutOfMemoryError:
        print(f"  OOM during warmup — skipping")
        del model, optimizer
        clear_cuda_state()
        return None

    torch.cuda.synchronize()
    reset_peak_memory()

    print(f"  Running benchmark for {duration_sec:.0f} seconds...")
    iterations = 0
    losses = []
    iter_times = []

    start_time = time.perf_counter()

    try:
        while True:
            iter_start = time.perf_counter()

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            iter_end = time.perf_counter()

            iter_times.append(iter_end - iter_start)
            losses.append(loss.item())
            iterations += 1

            elapsed = iter_end - start_time
            if elapsed >= duration_sec:
                break

            if iterations % 10 == 0:
                tps = (iterations * tokens_per_iter) / elapsed
                print(f"    Iter {iterations}: {tps:.0f} tok/s, loss={loss.item():.4f}")
    except torch.cuda.OutOfMemoryError:
        print(f"  OOM during benchmark after {iterations} iters — using partial results")
        if not iter_times:
            del model, optimizer
            clear_cuda_state()
            return None

    total_time = time.perf_counter() - start_time

    result = E2EBenchmarkResult(
        name=name,
        iterations=iterations,
        total_time_sec=total_time,
        avg_iter_ms=sum(iter_times) / len(iter_times) * 1000,
        tokens_per_sec=(iterations * tokens_per_iter) / total_time,
        peak_memory_gb=get_peak_memory_gb(),
        initial_loss=losses[0],
        final_loss=losses[-1],
        loss_history=losses,
    )

    print(f"  Complete: {result.iterations} iters, {result.tokens_per_sec:,.0f} tok/s, "
          f"{result.avg_iter_ms:.1f} ms/iter, {result.peak_memory_gb:.2f} GB")

    del model, optimizer
    clear_cuda_state()

    return result


def setup_liger():
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3
    apply_liger_kernel_to_qwen3(
        rope=True,
        rms_norm=True,
        swiglu=True,
        fused_linear_cross_entropy=True,
    )
    return "rope, rms_norm, swiglu, fused_linear_cross_entropy"


def setup_bastile():
    import bastile
    bastile.reset()
    applied = bastile.apply(
        rms_norm=True,
        swiglu=True,
        rope=True,
        fused_linear_cross_entropy=True,
    )
    return applied


def run_all_configs_for_seqlen(config, seq_len: int) -> Dict[str, Optional[E2EBenchmarkResult]]:
    results = {}

    print_header(f"seq_len={seq_len} | Bastile", 100)
    results["Bastile"] = run_benchmark(
        "Bastile", config, setup_fn=setup_bastile, seq_len=seq_len,
    )
    import bastile
    bastile.reset()

    print_header(f"seq_len={seq_len} | Liger Kernel", 100)
    reset_environment()
    try:
        results["Liger"] = run_benchmark(
            "Liger", config, setup_fn=setup_liger, seq_len=seq_len,
        )
    except Exception as e:
        print(f"  Error: {e}")
        traceback.print_exc()
        results["Liger"] = None

    print_header(f"seq_len={seq_len} | PyTorch (Baseline)", 100)
    reset_environment()
    results["PyTorch"] = run_benchmark(
        "PyTorch", config, setup_fn=None, seq_len=seq_len,
    )

    return results


def main():
    print("=" * 100)
    print("  Qwen3 0.6B Sequence Length Sweep: PyTorch vs Liger Kernel vs Bastile")
    print("=" * 100)
    print(f"\n  Model: Qwen3 ~0.6B (24 layers, 896 hidden), batch_size={BATCH_SIZE}, "
          f"warmup={WARMUP_ITERS}, duration={DURATION_SEC}s per run")
    print(f"  Sequence lengths: {SEQ_LENS}")
    print()

    print_gpu_info()

    config = make_qwen3_06b_config()

    all_results: Dict[int, Dict[str, Optional[E2EBenchmarkResult]]] = {}

    for seq_len in SEQ_LENS:
        print_header(f"SEQUENCE LENGTH = {seq_len}", 100)
        all_results[seq_len] = run_all_configs_for_seqlen(config, seq_len)

    # ========================================================================
    # Summary tables
    # ========================================================================
    print_header("RESULTS — TOKENS/SEC", 100)
    configs = ["PyTorch", "Liger", "Bastile"]

    print(f"\n  {'seq_len':>10}", end="")
    for c in configs:
        print(f"  {c:>14}", end="")
    print(f"  {'Bastile/Py':>12} {'Bastile/Lg':>12}")
    print(f"  {'-' * 88}")

    for seq_len in SEQ_LENS:
        res = all_results[seq_len]
        row = f"  {seq_len:>10}"
        tps = {}
        for c in configs:
            r = res.get(c)
            if r:
                row += f"  {r.tokens_per_sec:>12,.0f}  "
                tps[c] = r.tokens_per_sec
            else:
                row += f"  {'OOM':>12}  "
        if "Bastile" in tps and "PyTorch" in tps:
            row += f"  {tps['Bastile']/tps['PyTorch']:>10.2f}x"
        else:
            row += f"  {'—':>12}"
        if "Bastile" in tps and "Liger" in tps:
            row += f"  {tps['Bastile']/tps['Liger']:>10.2f}x"
        else:
            row += f"  {'—':>12}"
        print(row)

    print_header("RESULTS — MS/ITER", 100)

    print(f"\n  {'seq_len':>10}", end="")
    for c in configs:
        print(f"  {c:>14}", end="")
    print()
    print(f"  {'-' * 60}")

    for seq_len in SEQ_LENS:
        res = all_results[seq_len]
        row = f"  {seq_len:>10}"
        for c in configs:
            r = res.get(c)
            if r:
                row += f"  {r.avg_iter_ms:>12.1f}ms"
            else:
                row += f"  {'OOM':>14}"
        print(row)

    print_header("RESULTS — PEAK MEMORY (GB)", 100)

    print(f"\n  {'seq_len':>10}", end="")
    for c in configs:
        print(f"  {c:>14}", end="")
    print()
    print(f"  {'-' * 60}")

    for seq_len in SEQ_LENS:
        res = all_results[seq_len]
        row = f"  {seq_len:>10}"
        for c in configs:
            r = res.get(c)
            if r:
                row += f"  {r.peak_memory_gb:>12.2f}GB"
            else:
                row += f"  {'OOM':>14}"
        print(row)

    print_header("SUMMARY", 100)
    print()
    for seq_len in SEQ_LENS:
        res = all_results[seq_len]
        b = res.get("Bastile")
        l = res.get("Liger")
        p = res.get("PyTorch")

        parts = [f"  seq={seq_len}:"]
        if b and p:
            speedup = (b.tokens_per_sec / p.tokens_per_sec - 1) * 100
            parts.append(f"Bastile vs PyTorch {speedup:+.1f}%")
        if b and l:
            speedup = (b.tokens_per_sec / l.tokens_per_sec - 1) * 100
            parts.append(f"Bastile vs Liger {speedup:+.1f}%")
        if l and p:
            speedup = (l.tokens_per_sec / p.tokens_per_sec - 1) * 100
            parts.append(f"Liger vs PyTorch {speedup:+.1f}%")
        print("  ".join(parts))

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
