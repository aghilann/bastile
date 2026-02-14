"""
Qwen3 Benchmark: Raw HuggingFace vs Liger Kernel vs Bastile

Compares training performance across three configurations:
1. Raw HuggingFace (baseline)
2. Liger Kernel (Triton-based optimizations)
3. Bastile (CuTile-based optimizations)
"""

import torch
import gc
import time
import inspect
import importlib
from typing import Optional, Callable, List

from ..utils import (
    clear_cuda_state,
    reset_peak_memory,
    get_peak_memory_gb,
    print_header,
    print_gpu_info,
    E2EBenchmarkResult,
)


def reset_environment():
    """Reset CUDA and Python state between runs."""
    clear_cuda_state()
    reset_peak_memory()
    
    import transformers.models.qwen3.modeling_qwen3 as qwen3_mod
    importlib.reload(qwen3_mod)


def run_benchmark(
    name: str,
    setup_fn: Optional[Callable] = None,
    duration_sec: float = 15.0,
    batch_size: int = 1,
    seq_len: int = 4096,
) -> E2EBenchmarkResult:
    """Run training benchmark."""
    
    reset_environment()
    
    if setup_fn:
        setup_fn()
    
    # Import directly from the reloaded module to avoid stale cached references.
    # (from transformers import Qwen3ForCausalLM returns a cached class
    #  that doesn't reflect importlib.reload or monkey-patches)
    from transformers import Qwen3Config
    import transformers.models.qwen3.modeling_qwen3 as qwen3_mod
    Qwen3ForCausalLM = qwen3_mod.Qwen3ForCausalLM
    
    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        head_dim=128,
    )
    
    # Diagnostic: verify which implementations are active
    print(f"  RMSNorm: {qwen3_mod.Qwen3RMSNorm.__module__}.{qwen3_mod.Qwen3RMSNorm.__name__}")
    print(f"  MLP:     {qwen3_mod.Qwen3MLP.__module__}.{qwen3_mod.Qwen3MLP.__name__}")
    print(f"  RoPE:    {qwen3_mod.apply_rotary_pos_emb.__module__}.{qwen3_mod.apply_rotary_pos_emb.__name__}")
    print(f"  Forward: {inspect.getfile(Qwen3ForCausalLM.forward)}")
    
    model = Qwen3ForCausalLM(config)
    model = model.cuda().to(torch.bfloat16)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
    attention_mask = torch.ones_like(input_ids)
    tokens_per_iter = batch_size * seq_len
    
    # Warmup
    warmup_iters = 10
    for _ in range(warmup_iters):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        outputs.loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    reset_peak_memory()
    
    # Benchmark
    iterations = 0
    losses = []
    iter_times = []
    
    start_time = time.perf_counter()
    
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
        
        if iter_end - start_time >= duration_sec:
            break
    
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
    
    # Cleanup
    del model, optimizer
    clear_cuda_state()
    
    return result


def setup_liger():
    """Apply Liger Kernel patches."""
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3
    apply_liger_kernel_to_qwen3()


def setup_bastile():
    """Apply Bastile patches."""
    import bastile
    bastile.reset()
    bastile.apply(
        rms_norm=True, 
        swiglu=True, 
        rope=False,  # CuTile rope compiler fails on B200 sm_100
        fused_linear_cross_entropy=True,
    )


def main():
    print("=" * 75)
    print("Qwen3 Benchmark: HuggingFace vs Liger Kernel vs Bastile")
    print("=" * 75)
    print("\nConfig: Qwen3-8B (36 layers, 4096 hidden, 32 heads), batch=1, seq=4096, 15 sec each")
    
    print_gpu_info()
    
    results: List[E2EBenchmarkResult] = []
    
    # 1. Raw HuggingFace
    print("\n[1/3] Running Raw HuggingFace...")
    hf_result = run_benchmark("HuggingFace", setup_fn=None)
    results.append(hf_result)
    print(f"      {hf_result.iterations} iters, {hf_result.tokens_per_sec:.0f} tok/s, "
          f"{hf_result.peak_memory_gb:.2f} GB")
    
    # 2. Liger Kernel
    print("[2/3] Running Liger Kernel...")
    try:
        liger_result = run_benchmark("Liger", setup_fn=setup_liger)
        results.append(liger_result)
        print(f"      {liger_result.iterations} iters, {liger_result.tokens_per_sec:.0f} tok/s, "
              f"{liger_result.peak_memory_gb:.2f} GB")
    except Exception as e:
        print(f"      Error: {e}")
        liger_result = None
    
    # 3. Bastile
    print("[3/3] Running Bastile...")
    reset_environment()
    bastile_result = run_benchmark("Bastile", setup_fn=setup_bastile)
    results.append(bastile_result)
    print(f"      {bastile_result.iterations} iters, {bastile_result.tokens_per_sec:.0f} tok/s, "
          f"{bastile_result.peak_memory_gb:.2f} GB")
    
    # Reset Bastile
    import bastile
    bastile.reset()
    
    # Results table
    print_header("RESULTS", 75)
    
    baseline = hf_result
    
    print(f"\n  {'Config':<15} {'Tokens/sec':>12} {'Speedup':>10} {'Iter (ms)':>12} {'Memory':>10} {'Loss':>12}")
    print(f"  {'-'*73}")
    
    for r in results:
        speedup = r.tokens_per_sec / baseline.tokens_per_sec
        print(f"  {r.name:<15} {r.tokens_per_sec:>12,.0f} {speedup:>9.2f}x {r.avg_iter_ms:>12.1f} "
              f"{r.peak_memory_gb:>9.2f}GB {r.final_loss:>12.4f}")
    
    # Summary
    print_header("SUMMARY", 75)
    
    print(f"\n  Baseline: HuggingFace ({baseline.tokens_per_sec:,.0f} tokens/sec)")
    
    for r in results[1:]:
        speedup = (r.tokens_per_sec / baseline.tokens_per_sec - 1) * 100
        mem_saved = baseline.peak_memory_gb - r.peak_memory_gb
        sign = "+" if speedup >= 0 else ""
        print(f"  {r.name}: {sign}{speedup:.1f}% throughput, {mem_saved:+.2f} GB memory")
    
    print("\n" + "=" * 75)


if __name__ == "__main__":
    main()
