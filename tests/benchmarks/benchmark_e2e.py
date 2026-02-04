"""
End-to-End Benchmark: Bastile vs PyTorch for Qwen3 and GPT-OSS.

Measures:
- Loss convergence
- Peak memory usage
- Training time (tokens/sec, ms/iter)
"""

import torch
import gc
import time
import importlib
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class BenchmarkResult:
    model_type: str
    use_bastile: bool
    num_params: int
    iterations: int
    total_time_sec: float
    avg_iter_ms: float
    tokens_per_sec: float
    peak_memory_gb: float
    initial_loss: float
    final_loss: float
    loss_history: List[float]


def run_benchmark(
    model_type: str,
    use_bastile: bool,
    duration_sec: float = 30.0,
    batch_size: int = 4,
    seq_len: int = 256,
) -> BenchmarkResult:
    """
    Run training benchmark for a model.
    """
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    # Apply patches if needed
    if use_bastile:
        import bastile
        bastile.reset()
        bastile.apply(rms_norm=True, swiglu=True, rope=True, moe=True)
    
    # Create model
    if model_type == "qwen3":
        from transformers import Qwen3Config, Qwen3ForCausalLM
        config = Qwen3Config(
            vocab_size=32000,
            hidden_size=1024,
            intermediate_size=2816,
            num_hidden_layers=6,
            num_attention_heads=16,
            num_key_value_heads=4,
            hidden_act="silu",
            max_position_embeddings=2048,
            rms_norm_eps=1e-6,
            head_dim=64,
        )
        model = Qwen3ForCausalLM(config)
    else:  # gpt_oss
        from transformers import AutoConfig, AutoModelForCausalLM
        config = AutoConfig.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)
        config.hidden_size = 1024
        config.intermediate_size = 1536
        config.num_hidden_layers = 6
        config.num_attention_heads = 16
        config.num_key_value_heads = 4
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    
    model = model.cuda().to(torch.bfloat16)
    model.train()
    
    num_params = sum(p.numel() for p in model.parameters())
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create data
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
    attention_mask = torch.ones_like(input_ids)
    tokens_per_iter = batch_size * seq_len
    
    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        outputs.loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # Benchmark
    iterations = 0
    loss_history = []
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
        loss_history.append(loss.item())
        iterations += 1
        
        if iter_end - start_time >= duration_sec:
            break
    
    total_time = time.perf_counter() - start_time
    
    # Stats
    avg_iter_ms = sum(iter_times) / len(iter_times) * 1000
    tokens_per_sec = (iterations * tokens_per_iter) / total_time
    peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
    
    # Cleanup
    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()
    
    if use_bastile:
        import bastile
        bastile.reset()
    
    return BenchmarkResult(
        model_type=model_type,
        use_bastile=use_bastile,
        num_params=num_params,
        iterations=iterations,
        total_time_sec=total_time,
        avg_iter_ms=avg_iter_ms,
        tokens_per_sec=tokens_per_sec,
        peak_memory_gb=peak_memory_gb,
        initial_loss=loss_history[0],
        final_loss=loss_history[-1],
        loss_history=loss_history,
    )


def print_comparison(pytorch: BenchmarkResult, bastile: BenchmarkResult):
    """Print comparison between PyTorch and Bastile results."""
    
    speedup = bastile.tokens_per_sec / pytorch.tokens_per_sec
    iter_speedup = pytorch.avg_iter_ms / bastile.avg_iter_ms
    mem_saved = pytorch.peak_memory_gb - bastile.peak_memory_gb
    mem_pct = (mem_saved / pytorch.peak_memory_gb) * 100 if pytorch.peak_memory_gb > 0 else 0
    
    print(f"\n  {'Metric':<20} {'PyTorch':>15} {'Bastile':>15} {'Diff':>15}")
    print(f"  {'-'*65}")
    print(f"  {'Iterations':<20} {pytorch.iterations:>15} {bastile.iterations:>15} {bastile.iterations - pytorch.iterations:>+15}")
    print(f"  {'Tokens/sec':<20} {pytorch.tokens_per_sec:>15,.0f} {bastile.tokens_per_sec:>15,.0f} {speedup:>14.2f}x")
    print(f"  {'Avg iter (ms)':<20} {pytorch.avg_iter_ms:>15.1f} {bastile.avg_iter_ms:>15.1f} {iter_speedup:>14.2f}x")
    print(f"  {'Peak memory (GB)':<20} {pytorch.peak_memory_gb:>15.2f} {bastile.peak_memory_gb:>15.2f} {mem_saved:>+14.2f}")
    print(f"  {'Initial loss':<20} {pytorch.initial_loss:>15.4f} {bastile.initial_loss:>15.4f} {'-':>15}")
    print(f"  {'Final loss':<20} {pytorch.final_loss:>15.4f} {bastile.final_loss:>15.4f} {'-':>15}")
    
    print(f"\n  Summary: ", end="")
    if speedup >= 1.0:
        print(f"Bastile is {(speedup - 1) * 100:.1f}% faster", end="")
    else:
        print(f"Bastile is {(1 - speedup) * 100:.1f}% slower", end="")
    
    if mem_saved > 0:
        print(f", saves {mem_saved:.2f} GB ({mem_pct:.1f}%) memory")
    elif mem_saved < 0:
        print(f", uses {-mem_saved:.2f} GB more memory")
    else:
        print(", same memory usage")


def main():
    print("=" * 75)
    print("E2E Benchmark: Bastile vs PyTorch")
    print("=" * 75)
    print("\nConfig: 6 layers, 1024 hidden, batch=4, seq=256, 10 sec each")
    
    results: Dict[str, Dict[str, BenchmarkResult]] = {}
    
    for model_type in ["qwen3", "gpt_oss"]:
        print(f"\n{'='*75}")
        print(f"Model: {model_type.upper()}")
        print("=" * 75)
        
        results[model_type] = {}
        
        # Reload modules between runs
        if model_type == "qwen3":
            import transformers.models.qwen3.modeling_qwen3 as mod
        else:
            import transformers.models.gpt_oss.modeling_gpt_oss as mod
        importlib.reload(mod)
        
        # PyTorch baseline
        print("\n  [1/2] Running PyTorch baseline...")
        pytorch_result = run_benchmark(model_type, use_bastile=False, duration_sec=10)
        results[model_type]["pytorch"] = pytorch_result
        print(f"        {pytorch_result.iterations} iters, {pytorch_result.tokens_per_sec:.0f} tok/s, "
              f"{pytorch_result.peak_memory_gb:.2f} GB, loss {pytorch_result.initial_loss:.4f} -> {pytorch_result.final_loss:.4f}")
        
        # Reload modules
        importlib.reload(mod)
        
        # Bastile
        print("  [2/2] Running Bastile...")
        bastile_result = run_benchmark(model_type, use_bastile=True, duration_sec=10)
        results[model_type]["bastile"] = bastile_result
        print(f"        {bastile_result.iterations} iters, {bastile_result.tokens_per_sec:.0f} tok/s, "
              f"{bastile_result.peak_memory_gb:.2f} GB, loss {bastile_result.initial_loss:.4f} -> {bastile_result.final_loss:.4f}")
        
        # Comparison
        print_comparison(pytorch_result, bastile_result)
    
    # Final summary
    print("\n" + "=" * 75)
    print("FINAL SUMMARY")
    print("=" * 75)
    
    print(f"\n  {'Model':<12} {'Speedup':>12} {'Memory Saved':>15} {'Iterations':>20}")
    print(f"  {'-'*60}")
    
    for model_type in ["qwen3", "gpt_oss"]:
        py = results[model_type]["pytorch"]
        ba = results[model_type]["bastile"]
        speedup = ba.tokens_per_sec / py.tokens_per_sec
        mem_saved = py.peak_memory_gb - ba.peak_memory_gb
        iter_diff = ba.iterations - py.iterations
        
        print(f"  {model_type.upper():<12} {speedup:>11.2f}x {mem_saved:>+14.2f} GB {py.iterations:>8} -> {ba.iterations:<8} ({iter_diff:+})")
    
    print("\n" + "=" * 75)


if __name__ == "__main__":
    main()
