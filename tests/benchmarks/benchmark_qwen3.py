"""
Benchmark Qwen3 finetuning with and without Bastile.

Runs 1 minute of training with each configuration and compares throughput.
"""

import time
import torch
import gc


def benchmark_training(use_bastile: bool, duration_seconds: float = 60.0):
    """
    Benchmark Qwen3 training for a specified duration.
    
    Returns:
        dict with iterations, throughput, avg_time_per_iter
    """
    # Clear any previous state
    torch.cuda.empty_cache()
    gc.collect()
    
    if use_bastile:
        import bastile
        bastile.reset()  # Ensure clean state
        applied = bastile.apply(rms_norm=True, swiglu=True, rope=True)
        print(f"  Bastile patches applied: {applied}")
    
    # Import after patching
    from transformers import Qwen3Config, Qwen3ForCausalLM
    
    # Create model config
    config = Qwen3Config(
        vocab_size=32000,
        hidden_size=1024,
        intermediate_size=2048,
        num_hidden_layers=4,
        num_attention_heads=16,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        head_dim=64,
    )
    
    # Create model
    model = Qwen3ForCausalLM(config)
    model = model.cuda().to(torch.bfloat16)
    model.train()
    
    # Verify patches are applied (or not)
    if use_bastile:
        for name, module in model.named_modules():
            if "input_layernorm" in name:
                assert "Bastile" in type(module).__name__, f"Bastile not applied to {name}"
                break
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {num_params / 1e6:.1f}M parameters")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    # Create data
    batch_size = 4
    seq_len = 512
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
    attention_mask = torch.ones_like(input_ids)
    
    print(f"  Batch: {batch_size} x {seq_len} = {batch_size * seq_len} tokens/iter")
    
    # Warmup
    print("  Warming up (5 iterations)...")
    for _ in range(5):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        outputs.loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    
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
        
        # Progress update every 10 iterations
        if iterations % 10 == 0:
            tokens_per_sec = total_tokens / elapsed
            print(f"    Iter {iterations}: {tokens_per_sec:.0f} tokens/sec, loss={outputs.loss.item():.4f}")
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Calculate statistics
    avg_iter_time = sum(iter_times) / len(iter_times)
    tokens_per_sec = total_tokens / total_time
    
    # Memory stats
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    
    # Cleanup
    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()
    
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
    print("Qwen3 Finetuning Benchmark: Bastile vs PyTorch")
    print("=" * 70)
    
    duration = 15.0  # 15 seconds each
    
    # Benchmark WITHOUT Bastile first
    print("\n" + "=" * 70)
    print("Benchmark 1: PyTorch (No Bastile)")
    print("=" * 70)
    
    pytorch_results = benchmark_training(use_bastile=False, duration_seconds=duration)
    
    print(f"\n  Results:")
    print(f"    Iterations: {pytorch_results['iterations']}")
    print(f"    Avg time/iter: {pytorch_results['avg_iter_time_ms']:.1f} ms")
    print(f"    Throughput: {pytorch_results['tokens_per_sec']:.0f} tokens/sec")
    print(f"    Peak memory: {pytorch_results['peak_memory_gb']:.2f} GB")
    
    # Clear GPU state
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    # Need to reimport transformers to get fresh classes
    import importlib
    import transformers.models.qwen3.modeling_qwen3 as qwen3_module
    importlib.reload(qwen3_module)
    
    # Benchmark WITH Bastile
    print("\n" + "=" * 70)
    print("Benchmark 2: Bastile (CuTile Kernels)")
    print("=" * 70)
    
    bastile_results = benchmark_training(use_bastile=True, duration_seconds=duration)
    
    print(f"\n  Results:")
    print(f"    Iterations: {bastile_results['iterations']}")
    print(f"    Avg time/iter: {bastile_results['avg_iter_time_ms']:.1f} ms")
    print(f"    Throughput: {bastile_results['tokens_per_sec']:.0f} tokens/sec")
    print(f"    Peak memory: {bastile_results['peak_memory_gb']:.2f} GB")
    
    # Comparison
    print("\n" + "=" * 70)
    print("Comparison")
    print("=" * 70)
    
    speedup = bastile_results['tokens_per_sec'] / pytorch_results['tokens_per_sec']
    iter_speedup = pytorch_results['avg_iter_time_ms'] / bastile_results['avg_iter_time_ms']
    memory_diff = bastile_results['peak_memory_gb'] - pytorch_results['peak_memory_gb']
    
    print(f"\n  PyTorch:  {pytorch_results['tokens_per_sec']:>8.0f} tokens/sec  |  {pytorch_results['avg_iter_time_ms']:>6.1f} ms/iter")
    print(f"  Bastile:  {bastile_results['tokens_per_sec']:>8.0f} tokens/sec  |  {bastile_results['avg_iter_time_ms']:>6.1f} ms/iter")
    print(f"  " + "-" * 50)
    print(f"  Speedup:  {speedup:>8.2f}x throughput    |  {iter_speedup:>6.2f}x faster")
    print(f"  Memory:   {memory_diff:>+8.2f} GB difference")
    
    if speedup > 1.0:
        print(f"\n  ✓ Bastile is {(speedup - 1) * 100:.1f}% faster!")
    elif speedup < 1.0:
        print(f"\n  ✗ Bastile is {(1 - speedup) * 100:.1f}% slower")
    else:
        print(f"\n  = No significant difference")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
