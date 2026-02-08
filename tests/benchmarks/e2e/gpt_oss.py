"""
Benchmark GPT-OSS finetuning with and without Bastile.

Runs training with each configuration and compares throughput.
"""

import time
import torch
import gc
import importlib

from ..utils import (
    clear_cuda_state,
    reset_peak_memory,
    get_peak_memory_gb,
    print_header,
    print_gpu_info,
)


def benchmark_training(use_bastile: bool, duration_seconds: float = 15.0) -> dict:
    """
    Benchmark GPT-OSS training for a specified duration.
    
    Returns:
        dict with iterations, throughput, avg_time_per_iter
    """
    clear_cuda_state()
    
    if use_bastile:
        import bastile
        bastile.reset()
        # GPT-OSS gets Fused GEGLU MoE Experts only
        applied = bastile.apply(rms_norm=False, moe=True)
        bastile.warmup_all_kernels()
        print(f"  Bastile patches applied: {applied}")
    
    from transformers import AutoConfig, AutoModelForCausalLM
    
    config = AutoConfig.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)
    config.hidden_size = 1024
    config.intermediate_size = 2048
    config.num_hidden_layers = 4
    config.num_attention_heads = 16
    config.num_key_value_heads = 4
    
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model = model.cuda().to(torch.bfloat16)
    model.train()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {num_params / 1e6:.1f}M parameters")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    batch_size = 4
    seq_len = 512
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
    attention_mask = torch.ones_like(input_ids)
    
    print(f"  Batch: {batch_size} x {seq_len} = {batch_size * seq_len} tokens/iter")
    
    # Warmup
    warmup_iters = 10
    print(f"  Warming up ({warmup_iters} iterations)...")
    for _ in range(warmup_iters):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        outputs.loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    reset_peak_memory()
    
    # Benchmark
    print(f"  Running benchmark for {duration_seconds:.0f} seconds...")
    
    iterations = 0
    total_tokens = 0
    iter_times = []
    
    start_time = time.perf_counter()
    
    while True:
        iter_start = time.perf_counter()
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        outputs.loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize()
        iter_end = time.perf_counter()
        
        iter_times.append(iter_end - iter_start)
        iterations += 1
        total_tokens += batch_size * seq_len
        
        elapsed = iter_end - start_time
        if elapsed >= duration_seconds:
            break
        
        if iterations % 10 == 0:
            tokens_per_sec = total_tokens / elapsed
            print(f"    Iter {iterations}: {tokens_per_sec:.0f} tokens/sec, loss={outputs.loss.item():.4f}")
    
    total_time = time.perf_counter() - start_time
    avg_iter_time = sum(iter_times) / len(iter_times)
    tokens_per_sec = total_tokens / total_time
    peak_memory = get_peak_memory_gb()
    
    # Cleanup
    del model, optimizer
    clear_cuda_state()
    
    if use_bastile:
        import bastile
        bastile.reset()
    
    return {
        "iterations": iterations,
        "total_time": total_time,
        "avg_iter_time_ms": avg_iter_time * 1000,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_gb": peak_memory,
        "final_loss": outputs.loss.item(),
    }


def main():
    print("=" * 70)
    print("GPT-OSS Finetuning Benchmark: Bastile vs PyTorch")
    print("=" * 70)
    
    print_gpu_info()
    
    duration = 15.0
    
    # Benchmark WITHOUT Bastile
    print_header("Benchmark 1: PyTorch (No Bastile)", 70)
    
    pytorch_results = benchmark_training(use_bastile=False, duration_seconds=duration)
    
    print(f"\n  Results:")
    print(f"    Iterations: {pytorch_results['iterations']}")
    print(f"    Avg time/iter: {pytorch_results['avg_iter_time_ms']:.1f} ms")
    print(f"    Throughput: {pytorch_results['tokens_per_sec']:.0f} tokens/sec")
    print(f"    Peak memory: {pytorch_results['peak_memory_gb']:.2f} GB")
    
    # Clear and reload
    clear_cuda_state()
    reset_peak_memory()
    import transformers.models.gpt_oss.modeling_gpt_oss as gpt_oss_module
    importlib.reload(gpt_oss_module)
    
    # Benchmark WITH Bastile
    print_header("Benchmark 2: Bastile (CuTile Kernels)", 70)
    
    bastile_results = benchmark_training(use_bastile=True, duration_seconds=duration)
    
    print(f"\n  Results:")
    print(f"    Iterations: {bastile_results['iterations']}")
    print(f"    Avg time/iter: {bastile_results['avg_iter_time_ms']:.1f} ms")
    print(f"    Throughput: {bastile_results['tokens_per_sec']:.0f} tokens/sec")
    print(f"    Peak memory: {bastile_results['peak_memory_gb']:.2f} GB")
    
    # Comparison
    print_header("Comparison", 70)
    
    speedup = bastile_results['tokens_per_sec'] / pytorch_results['tokens_per_sec']
    iter_speedup = pytorch_results['avg_iter_time_ms'] / bastile_results['avg_iter_time_ms']
    memory_diff = bastile_results['peak_memory_gb'] - pytorch_results['peak_memory_gb']
    
    print(f"\n  PyTorch:  {pytorch_results['tokens_per_sec']:>8.0f} tokens/sec  |  {pytorch_results['avg_iter_time_ms']:>6.1f} ms/iter")
    print(f"  Bastile:  {bastile_results['tokens_per_sec']:>8.0f} tokens/sec  |  {bastile_results['avg_iter_time_ms']:>6.1f} ms/iter")
    print(f"  " + "-" * 50)
    print(f"  Speedup:  {speedup:>8.2f}x throughput    |  {iter_speedup:>6.2f}x faster")
    print(f"  Memory:   {memory_diff:>+8.2f} GB difference")
    
    if speedup > 1.0:
        print(f"\n  Bastile is {(speedup - 1) * 100:.1f}% faster!")
    elif speedup < 1.0:
        print(f"\n  Bastile is {(1 - speedup) * 100:.1f}% slower")
    else:
        print(f"\n  No significant difference")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
