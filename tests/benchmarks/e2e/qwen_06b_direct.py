"""
Qwen3 0.5B Benchmark: PyTorch vs Liger Kernel vs Bastile
Direct (non-subprocess) benchmark to verify kernel application.

Uses Qwen3 model with ~0.5B parameters.
"""

import torch
import gc
import time
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
    
    # Reload Qwen3 module to clear any monkey patches
    import transformers.models.qwen3.modeling_qwen3 as qwen3_mod
    importlib.reload(qwen3_mod)


def run_benchmark(
    name: str,
    setup_fn: Optional[Callable] = None,
    duration_sec: float = 15.0,
    batch_size: int = 4,
    seq_len: int = 512,
    warmup_iterations: Optional[int] = None,
) -> E2EBenchmarkResult:
    """Run training benchmark with real Qwen 0.5B model."""
    
    reset_environment()
    
    if setup_fn:
        print(f"  Applying {name} patches...")
        applied = setup_fn()
        if applied:
            print(f"  Patches applied: {applied}")
    
    from transformers import Qwen3Config, Qwen3ForCausalLM
    
    # Qwen3 0.5B-like configuration
    config = Qwen3Config(
        vocab_size=32000,
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        head_dim=64,
    )
    
    print(f"  Creating Qwen3 model...")
    model = Qwen3ForCausalLM(config)
    model = model.cuda().to(torch.bfloat16)
    model.train()
    
    config = model.config
    print(f"  Model: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    print(f"  Config: {config.num_hidden_layers} layers, {config.hidden_size} hidden")
    print(f"  Batch: {batch_size} x {seq_len} = {batch_size * seq_len} tokens/iter")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
    attention_mask = torch.ones_like(input_ids)
    tokens_per_iter = batch_size * seq_len
    
    # Warmup (more iterations for kernel autotuning)
    if warmup_iterations is None:
        warmup_iters = 20 if setup_fn else 5
    else:
        warmup_iters = warmup_iterations
    print(f"  Warming up ({warmup_iters} iterations)...")
    for _ in range(warmup_iters):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        outputs.loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    reset_peak_memory()
    
    # Benchmark
    print(f"  Running benchmark for {duration_sec:.0f} seconds...")
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
        
        elapsed = iter_end - start_time
        if elapsed >= duration_sec:
            break
        
        if iterations % 10 == 0:
            tps = (iterations * tokens_per_iter) / elapsed
            print(f"    Iter {iterations}: {tps:.0f} tok/s, loss={loss.item():.4f}")
    
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
    
    print(f"  Complete: {result.iterations} iters, {result.tokens_per_sec:.0f} tok/s, "
          f"{result.avg_iter_ms:.1f} ms/iter, {result.peak_memory_gb:.2f} GB")
    
    # Cleanup
    del model, optimizer
    clear_cuda_state()
    
    return result


def setup_liger():
    """Apply Liger Kernel patches."""
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3
    apply_liger_kernel_to_qwen3(
        rope=True,
        rms_norm=True,
        swiglu=True,
        fused_linear_cross_entropy=True,
    )
    return "rope, rms_norm, swiglu, fused_linear_cross_entropy"


def setup_bastile():
    """Apply Bastile patches."""
    import bastile
    bastile.reset()
    applied = bastile.apply(
        rms_norm=True, 
        swiglu=True, 
        rope=True,           
        cross_entropy=True
    )
    return applied


def main():
    print("=" * 80)
    print("Qwen3 0.5B Direct Benchmark: PyTorch vs Liger Kernel vs Bastile")
    print("=" * 80)
    print("\nQwen3 ~0.5B model, batch=4, seq=512, 15 sec each")
    print("(Direct execution without subprocess isolation)")
    
    print_gpu_info()
    
    results: List[E2EBenchmarkResult] = []
    
    # 1. PyTorch Baseline
    print_header("Benchmark: PyTorch (Baseline)", 80)
    pytorch_result = run_benchmark("PyTorch", setup_fn=None)
    results.append(pytorch_result)
    
    # 2. Liger Kernel
    print_header("Benchmark: Liger Kernel", 80)
    try:
        liger_result = run_benchmark("Liger", setup_fn=setup_liger)
        results.append(liger_result)
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        liger_result = None
    
    # 3. Bastile
    print_header("Benchmark: Bastile", 80)
    reset_environment()
    bastile_result = run_benchmark("Bastile", setup_fn=setup_bastile, warmup_iterations=50)
    results.append(bastile_result)
    
    # Reset Bastile
    import bastile
    bastile.reset()
    
    # Results table
    print_header("RESULTS", 80)
    
    baseline = pytorch_result
    
    print(f"\n  {'Config':<15} {'Tokens/sec':>12} {'Speedup':>10} {'Iter (ms)':>12} {'Memory':>10} {'Loss':>12}")
    print(f"  {'-'*78}")
    
    for r in results:
        speedup = r.tokens_per_sec / baseline.tokens_per_sec
        print(f"  {r.name:<15} {r.tokens_per_sec:>12,.0f} {speedup:>9.2f}x {r.avg_iter_ms:>12.1f} "
              f"{r.peak_memory_gb:>9.2f}GB {r.final_loss:>12.4f}")
    
    # Summary
    print_header("SUMMARY", 80)
    
    print(f"\n  Baseline: PyTorch ({baseline.tokens_per_sec:,.0f} tokens/sec, {baseline.peak_memory_gb:.2f} GB)")
    
    for r in results[1:]:
        speedup = (r.tokens_per_sec / baseline.tokens_per_sec - 1) * 100
        mem_saved = baseline.peak_memory_gb - r.peak_memory_gb
        sign = "+" if speedup >= 0 else ""
        print(f"  {r.name}: {sign}{speedup:.1f}% throughput, {mem_saved:+.2f} GB memory")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
