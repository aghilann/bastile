"""
Benchmark Qwen3 finetuning: PyTorch vs Liger Kernel vs Bastile.

Runs each configuration in a subprocess for clean isolation (Liger's
monkey-patching persists across importlib.reload).
"""

import sys
import time
import json
import torch
import gc
import subprocess

from ..utils import (
    clear_cuda_state,
    reset_peak_memory,
    get_peak_memory_gb,
    print_header,
    print_gpu_info,
)


# Standalone script that runs a single benchmark mode, outputs JSON result
_BENCHMARK_SCRIPT = r'''
import sys
import time
import json
import torch
import gc

BATCH_SIZE = 4
SEQ_LEN = 512

def clear():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()

def run(mode, duration=15.0):
    clear()

    if mode == "bastile":
        import bastile
        applied = bastile.apply(rms_norm=True, swiglu=True, rope=True, cross_entropy=True)
        print(f"  Patches applied: {applied}", file=sys.stderr)
    elif mode == "liger":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3
        apply_liger_kernel_to_qwen3(
            rope=True, rms_norm=True, swiglu=True,
            fused_linear_cross_entropy=True,
        )
        print("  Liger patches: rope, rms_norm, swiglu, fused_linear_cross_entropy", file=sys.stderr)

    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=32000, hidden_size=1024, intermediate_size=2048,
        num_hidden_layers=4, num_attention_heads=16, num_key_value_heads=4,
        hidden_act="silu", max_position_embeddings=2048, rms_norm_eps=1e-6,
        tie_word_embeddings=False, head_dim=64,
    )

    model = Qwen3ForCausalLM(config).cuda().to(torch.bfloat16)
    model.train()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {num_params / 1e6:.1f}M parameters", file=sys.stderr)
    print(f"  Batch: {BATCH_SIZE} x {SEQ_LEN} = {BATCH_SIZE * SEQ_LEN} tokens/iter", file=sys.stderr)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    input_ids = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device="cuda")
    labels = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device="cuda")
    attention_mask = torch.ones_like(input_ids)

    # Warmup
    print("  Warming up (5 iterations)...", file=sys.stderr)
    for _ in range(5):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        outputs.loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    print(f"  Running benchmark for {duration:.0f} seconds...", file=sys.stderr)

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
        total_tokens += BATCH_SIZE * SEQ_LEN

        elapsed = iter_end - start_time
        if elapsed >= duration:
            break

        if iterations % 100 == 0:
            tps = total_tokens / elapsed
            print(f"    Iter {iterations}: {tps:.0f} tok/s, loss={outputs.loss.item():.4f}", file=sys.stderr)

    total_time = time.perf_counter() - start_time
    avg_iter_time = sum(iter_times) / len(iter_times)
    tokens_per_sec = total_tokens / total_time
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3

    result = {
        "iterations": iterations,
        "total_time": total_time,
        "avg_iter_time_ms": avg_iter_time * 1000,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_gb": peak_memory,
        "final_loss": outputs.loss.item(),
    }
    print(json.dumps(result))

if __name__ == "__main__":
    run(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 15.0)
'''


def run_benchmark_subprocess(mode: str, duration: float = 15.0) -> dict:
    """Run a benchmark in a clean subprocess to avoid monkey-patch contamination."""
    import tempfile
    import os

    # Write script to temp file
    script_path = os.path.join(tempfile.gettempdir(), f"bench_{mode}.py")
    with open(script_path, "w") as f:
        f.write(_BENCHMARK_SCRIPT)

    result = subprocess.run(
        [sys.executable, script_path, mode, str(duration)],
        capture_output=True,
        text=True,
        cwd="/workspace/bastile",
        timeout=int(duration * 3 + 120),
    )

    # Print stderr (progress output)
    if result.stderr:
        for line in result.stderr.strip().split("\n"):
            print(line)

    if result.returncode != 0:
        print(f"  ERROR: subprocess exited with code {result.returncode}")
        if result.stderr:
            # Show last few error lines
            lines = result.stderr.strip().split("\n")
            for line in lines[-10:]:
                print(f"  {line}")
        return None

    # Parse JSON result from stdout
    stdout_lines = result.stdout.strip().split("\n")
    for line in reversed(stdout_lines):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)

    print("  ERROR: no JSON result found in output")
    return None


def main():
    print("=" * 80)
    print("Qwen3 Finetuning Benchmark: PyTorch vs Liger Kernel vs Bastile")
    print("=" * 80)

    print_gpu_info()
    print("\n  (Each benchmark runs in a clean subprocess for isolation)\n")

    duration = 15.0
    results = {}

    for mode, label in [
        ("pytorch", "PyTorch (Baseline)"),
        ("liger",   "Liger Kernel"),
        ("bastile", "Bastile (CuTile Kernels)"),
    ]:
        print_header(f"Benchmark: {label}", 80)
        r = run_benchmark_subprocess(mode, duration)
        if r:
            results[mode] = r
            print(f"\n  Results:")
            print(f"    Iterations: {r['iterations']}")
            print(f"    Avg time/iter: {r['avg_iter_time_ms']:.1f} ms")
            print(f"    Throughput: {r['tokens_per_sec']:.0f} tokens/sec")
            print(f"    Peak memory: {r['peak_memory_gb']:.2f} GB")
        else:
            print(f"\n  FAILED — skipping {mode}")

    # ---- Comparison ----
    if len(results) < 2:
        print("\nNot enough results for comparison.")
        return

    print_header("COMPARISON", 80)

    py = results.get("pytorch")
    lg = results.get("liger")
    bt = results.get("bastile")

    print(f"\n  {'':>20} {'Throughput':>14} {'ms/iter':>10} {'Memory':>10} {'vs PyTorch':>12} {'vs Liger':>12}")
    print(f"  {'-'*78}")

    if py:
        print(f"  {'PyTorch':<20} {py['tokens_per_sec']:>10.0f} t/s "
              f"{py['avg_iter_time_ms']:>8.1f}ms {py['peak_memory_gb']:>8.2f}GB "
              f"{'baseline':>12} {'':>12}")
    if lg:
        vs_py = f"{lg['tokens_per_sec']/py['tokens_per_sec']:>11.2f}x" if py else ""
        print(f"  {'Liger Kernel':<20} {lg['tokens_per_sec']:>10.0f} t/s "
              f"{lg['avg_iter_time_ms']:>8.1f}ms {lg['peak_memory_gb']:>8.2f}GB "
              f"{vs_py:>12} {'baseline':>12}")
    if bt:
        vs_py = f"{bt['tokens_per_sec']/py['tokens_per_sec']:>11.2f}x" if py else ""
        vs_lg = f"{bt['tokens_per_sec']/lg['tokens_per_sec']:>11.2f}x" if lg else ""
        print(f"  {'Bastile':<20} {bt['tokens_per_sec']:>10.0f} t/s "
              f"{bt['avg_iter_time_ms']:>8.1f}ms {bt['peak_memory_gb']:>8.2f}GB "
              f"{vs_py:>12} {vs_lg:>12}")

    print()
    if bt and py:
        print(f"  Bastile vs PyTorch: {(bt['tokens_per_sec']/py['tokens_per_sec'] - 1)*100:+.1f}% throughput, "
              f"{bt['peak_memory_gb'] - py['peak_memory_gb']:+.2f} GB memory")
    if bt and lg:
        print(f"  Bastile vs Liger:   {(bt['tokens_per_sec']/lg['tokens_per_sec'] - 1)*100:+.1f}% throughput, "
              f"{bt['peak_memory_gb'] - lg['peak_memory_gb']:+.2f} GB memory")
    if lg and py:
        print(f"  Liger vs PyTorch:   {(lg['tokens_per_sec']/py['tokens_per_sec'] - 1)*100:+.1f}% throughput, "
              f"{lg['peak_memory_gb'] - py['peak_memory_gb']:+.2f} GB memory")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
