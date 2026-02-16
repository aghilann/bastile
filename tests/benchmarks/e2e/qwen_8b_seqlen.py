"""
Qwen3-8B Sequence Length Sweep: PyTorch vs Liger vs Bastile

Full Qwen3-8B architecture (no pretrained weights loaded) benchmarked
across doubling sequence lengths.
batch_size=1 to maximize sequence length headroom on the GPU.

Each configuration (PyTorch, Liger, Bastile) runs as a separate subprocess
on its own GPU to ensure complete isolation — no monkey-patching conflicts.

Usage:
  # Auto-parallel on GPUs 0,1,2:
  python -u -m tests.benchmarks.e2e.qwen_8b_seqlen

  # Custom GPU assignment:
  python -u -m tests.benchmarks.e2e.qwen_8b_seqlen --gpus 2,3,4

  # Single-GPU sequential (if only 1 GPU available):
  python -u -m tests.benchmarks.e2e.qwen_8b_seqlen --sequential

  # Internal: run one phase (called by subprocess):
  python -u -m tests.benchmarks.e2e.qwen_8b_seqlen --phase pytorch --gpu 0
"""

import torch
import gc
import os
import sys
import json
import time
import inspect
import argparse
import importlib
import subprocess
import tempfile
from typing import Optional, Callable, List, Dict
from pathlib import Path

from ..utils import (
    clear_cuda_state,
    reset_peak_memory,
    get_peak_memory_gb,
    print_header,
    print_gpu_info,
    E2EBenchmarkResult,
)


SEQ_LENS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
BATCH_SIZE = 1
WARMUP_ITERS = 5
DURATION_SEC = 15.0


def reset_environment():
    clear_cuda_state()
    reset_peak_memory()
    import transformers.models.qwen3.modeling_qwen3 as qwen3_mod
    importlib.reload(qwen3_mod)


def make_qwen3_8b_config():
    """Qwen3-8B architecture config (randomly initialized, no checkpoint)."""
    from transformers import Qwen3Config
    return Qwen3Config(
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

    # Import directly from the module to avoid stale cached references
    import transformers.models.qwen3.modeling_qwen3 as qwen3_mod
    Qwen3ForCausalLM = qwen3_mod.Qwen3ForCausalLM

    # Verify what's actually being used
    print(f"  RMSNorm: {qwen3_mod.Qwen3RMSNorm.__module__}.{qwen3_mod.Qwen3RMSNorm.__name__}")
    print(f"  MLP:     {qwen3_mod.Qwen3MLP.__module__}.{qwen3_mod.Qwen3MLP.__name__}")
    print(f"  RoPE:    {qwen3_mod.apply_rotary_pos_emb.__module__}.{qwen3_mod.apply_rotary_pos_emb.__name__}")
    print(f"  Forward: {Qwen3ForCausalLM.forward.__module__}.{Qwen3ForCausalLM.forward.__name__}")

    print(f"  Creating Qwen3-8B model...")
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

            if iterations % 5 == 0:
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


# ============================================================================
# Setup functions for each configuration
# ============================================================================

def setup_liger():
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3
    applied = apply_liger_kernel_to_qwen3()
    return applied


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


PHASES = {
    "pytorch": ("PyTorch", None),
    "liger":   ("Liger",   setup_liger),
    "bastile": ("Bastile", setup_bastile),
}


def result_to_dict(r: Optional[E2EBenchmarkResult]) -> Optional[dict]:
    if r is None:
        return None
    return {
        "name": r.name,
        "iterations": r.iterations,
        "total_time_sec": r.total_time_sec,
        "avg_iter_ms": r.avg_iter_ms,
        "tokens_per_sec": r.tokens_per_sec,
        "peak_memory_gb": r.peak_memory_gb,
        "initial_loss": r.initial_loss,
        "final_loss": r.final_loss,
    }


def dict_to_result(d: Optional[dict]) -> Optional[E2EBenchmarkResult]:
    if d is None:
        return None
    return E2EBenchmarkResult(
        name=d["name"],
        iterations=d["iterations"],
        total_time_sec=d["total_time_sec"],
        avg_iter_ms=d["avg_iter_ms"],
        tokens_per_sec=d["tokens_per_sec"],
        peak_memory_gb=d["peak_memory_gb"],
        initial_loss=d["initial_loss"],
        final_loss=d["final_loss"],
        loss_history=[],
    )


# ============================================================================
# Single-phase runner (called by subprocess or directly)
# ============================================================================

def run_phase(phase: str, output_file: str):
    """Run one phase (pytorch/liger/bastile) across all seq lengths, write results to JSON."""
    name, setup_fn = PHASES[phase]
    config = make_qwen3_8b_config()

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    print(f"\n  [{phase.upper()}] Running on {gpu_name} (CUDA {torch.cuda.current_device()})")

    results: Dict[str, Optional[dict]] = {}

    for seq_len in SEQ_LENS:
        print_header(f"{name} | seq_len={seq_len}", 100)

        # Reset patches for each run
        if phase == "bastile":
            import bastile
            bastile.reset()

        r = run_benchmark(name, config, setup_fn=setup_fn, seq_len=seq_len)
        results[str(seq_len)] = result_to_dict(r)

        if phase == "bastile":
            import bastile
            bastile.reset()

    # Write results to JSON
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  [{phase.upper()}] Results written to {output_file}")


# ============================================================================
# Parallel orchestrator
# ============================================================================

def run_parallel(gpus: List[int]):
    """Spawn 3 subprocesses, one per config, each on its own GPU."""
    phases = ["pytorch", "liger", "bastile"]
    tmp_dir = tempfile.mkdtemp(prefix="bench_qwen8b_")

    print(f"\n  Launching 3 parallel benchmarks:")
    for phase, gpu in zip(phases, gpus):
        print(f"    {phase:>8} → GPU {gpu}")
    print(f"  Temp dir: {tmp_dir}\n")

    # Get the working directory (bastile repo root)
    cwd = os.getcwd()

    # Build subprocess commands
    procs = []
    output_files = []
    log_files = []

    for phase, gpu in zip(phases, gpus):
        output_file = os.path.join(tmp_dir, f"{phase}.json")
        log_file = os.path.join(tmp_dir, f"{phase}.log")
        output_files.append(output_file)
        log_files.append(log_file)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONUNBUFFERED"] = "1"

        cmd = [
            sys.executable, "-u", "-m",
            "tests.benchmarks.e2e.qwen_8b_seqlen",
            "--phase", phase,
            "--gpu", "0",  # always 0 since CUDA_VISIBLE_DEVICES remaps
            "--output", output_file,
        ]

        log_fh = open(log_file, "w")
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env,
            stdout=log_fh, stderr=subprocess.STDOUT,
        )
        procs.append((phase, proc, log_fh))
        print(f"  Started {phase} (PID {proc.pid}) → GPU {gpu}")

    # Wait for all to finish, streaming status
    print(f"\n  Waiting for all benchmarks to complete...")
    start = time.time()

    while any(p.poll() is None for _, p, _ in procs):
        time.sleep(30)
        elapsed = time.time() - start
        status = []
        for phase, proc, _ in procs:
            if proc.poll() is None:
                status.append(f"{phase}: running")
            else:
                status.append(f"{phase}: done (exit={proc.returncode})")
        print(f"  [{elapsed:.0f}s] {' | '.join(status)}")

    # Close log file handles
    for _, _, fh in procs:
        fh.close()

    elapsed = time.time() - start
    print(f"\n  All done in {elapsed:.0f}s")

    # Check for failures
    for phase, proc, _ in procs:
        if proc.returncode != 0:
            log_file = os.path.join(tmp_dir, f"{phase}.log")
            print(f"\n  ⚠ {phase} failed (exit={proc.returncode}). Log tail:")
            try:
                with open(log_file) as f:
                    lines = f.readlines()
                    for line in lines[-20:]:
                        print(f"    {line.rstrip()}")
            except Exception:
                pass

    # Load results
    all_results: Dict[str, Dict[int, Optional[E2EBenchmarkResult]]] = {}
    for phase, output_file in zip(phases, output_files):
        try:
            with open(output_file) as f:
                raw = json.load(f)
            all_results[phase] = {
                int(k): dict_to_result(v) for k, v in raw.items()
            }
        except FileNotFoundError:
            print(f"  ⚠ No results for {phase} (file not found)")
            all_results[phase] = {sl: None for sl in SEQ_LENS}

    # Print logs summary
    for phase in phases:
        log_file = os.path.join(tmp_dir, f"{phase}.log")
        print(f"\n  Full {phase} log: {log_file}")

    return all_results


def run_sequential():
    """Run all 3 phases sequentially on the current GPU."""
    config = make_qwen3_8b_config()

    all_results: Dict[str, Dict[int, Optional[E2EBenchmarkResult]]] = {}

    for phase_key, (name, setup_fn) in PHASES.items():
        print_header(f"PHASE: {name.upper()} (all sequence lengths)", 100)

        results: Dict[int, Optional[E2EBenchmarkResult]] = {}
        for seq_len in SEQ_LENS:
            print_header(f"{name} | seq_len={seq_len}", 100)

            if phase_key == "bastile":
                import bastile
                bastile.reset()

            results[seq_len] = run_benchmark(
                name, config, setup_fn=setup_fn, seq_len=seq_len,
            )

            if phase_key == "bastile":
                import bastile
                bastile.reset()

        all_results[phase_key] = results

    return all_results


# ============================================================================
# Results printing
# ============================================================================

def print_results(all_results: Dict[str, Dict[int, Optional[E2EBenchmarkResult]]]):
    """Print combined results tables for all 3 configurations."""

    pytorch = all_results.get("pytorch", {})
    liger = all_results.get("liger", {})
    bastile = all_results.get("bastile", {})

    def val_or_oom(r, attr):
        if r is None:
            return None
        return getattr(r, attr)

    # ── Tokens/sec ──
    print_header("RESULTS — TOKENS/SEC", 110)
    print(f"\n  {'seq_len':>10}  {'PyTorch':>14}  {'Liger':>14}  {'Bastile':>14}  {'Liger Δ%':>10}  {'Bastile Δ%':>12}")
    print(f"  {'-' * 90}")

    for sl in SEQ_LENS:
        p, l, b = pytorch.get(sl), liger.get(sl), bastile.get(sl)
        row = f"  {sl:>10}"
        for r in [p, l, b]:
            if r:
                row += f"  {r.tokens_per_sec:>12,.0f}  "
            else:
                row += f"  {'OOM':>12}  "
        if p and l:
            d = (l.tokens_per_sec / p.tokens_per_sec - 1) * 100
            row += f"  {d:>+8.1f}%"
        else:
            row += f"  {'—':>10}"
        if p and b:
            d = (b.tokens_per_sec / p.tokens_per_sec - 1) * 100
            row += f"  {d:>+10.1f}%"
        else:
            row += f"  {'—':>12}"
        print(row)

    # ── ms/iter ──
    print_header("RESULTS — MS/ITER", 110)
    print(f"\n  {'seq_len':>10}  {'PyTorch':>14}  {'Liger':>14}  {'Bastile':>14}")
    print(f"  {'-' * 60}")

    for sl in SEQ_LENS:
        p, l, b = pytorch.get(sl), liger.get(sl), bastile.get(sl)
        row = f"  {sl:>10}"
        for r in [p, l, b]:
            if r:
                row += f"  {r.avg_iter_ms:>12.1f}ms"
            else:
                row += f"  {'OOM':>14}"
        print(row)

    # ── Peak Memory ──
    print_header("RESULTS — PEAK MEMORY (GB)", 110)
    print(f"\n  {'seq_len':>10}  {'PyTorch':>14}  {'Liger':>14}  {'Bastile':>14}  {'Liger Saved':>14}  {'Bastile Saved':>16}")
    print(f"  {'-' * 95}")

    for sl in SEQ_LENS:
        p, l, b = pytorch.get(sl), liger.get(sl), bastile.get(sl)
        row = f"  {sl:>10}"
        for r in [p, l, b]:
            if r:
                row += f"  {r.peak_memory_gb:>12.2f}GB"
            else:
                row += f"  {'OOM':>14}"
        if p and l:
            row += f"  {p.peak_memory_gb - l.peak_memory_gb:>+12.2f}GB"
        else:
            row += f"  {'—':>14}"
        if p and b:
            row += f"  {p.peak_memory_gb - b.peak_memory_gb:>+14.2f}GB"
        else:
            row += f"  {'—':>16}"
        print(row)

    # ── Summary ──
    print_header("SUMMARY", 110)
    print()
    for sl in SEQ_LENS:
        p, l, b = pytorch.get(sl), liger.get(sl), bastile.get(sl)
        parts = [f"  seq={sl:>5}:"]
        if p and l:
            spd = (l.tokens_per_sec / p.tokens_per_sec - 1) * 100
            mem = p.peak_memory_gb - l.peak_memory_gb
            parts.append(f"Liger {spd:>+6.1f}% tput, {mem:>+.2f}GB mem")
        elif not l:
            parts.append("Liger OOM")
        if p and b:
            spd = (b.tokens_per_sec / p.tokens_per_sec - 1) * 100
            mem = p.peak_memory_gb - b.peak_memory_gb
            parts.append(f"| Bastile {spd:>+6.1f}% tput, {mem:>+.2f}GB mem")
        elif not b:
            parts.append("| Bastile OOM")
        if not p:
            parts = [f"  seq={sl:>5}: PyTorch OOM"]
        print(" ".join(parts))

    print("\n" + "=" * 110)


# ============================================================================
# Chart generation
# ============================================================================

def plot_results(
    all_results: Dict[str, Dict[int, Optional[E2EBenchmarkResult]]],
    assets_dir: Optional[str] = None,
):
    """Generate bar charts for throughput, latency, and memory, saving to assets/."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\n  matplotlib not installed — skipping chart generation")
        return

    if assets_dir is None:
        assets_dir = str(Path(__file__).resolve().parents[3] / "assets")
    os.makedirs(assets_dir, exist_ok=True)

    pytorch = all_results.get("pytorch", {})
    liger = all_results.get("liger", {})
    bastile_res = all_results.get("bastile", {})

    configs = [
        ("PyTorch", pytorch, "#5B8FF9"),
        ("Liger", liger, "#5AD8A6"),
        ("Bastile", bastile_res, "#F6BD16"),
    ]

    # Collect seq_lens that have at least one result
    active_seq_lens = [sl for sl in SEQ_LENS if any(
        cfg_data.get(sl) is not None for _, cfg_data, _ in configs
    )]
    if not active_seq_lens:
        print("\n  No results to plot")
        return

    x = np.arange(len(active_seq_lens))
    width = 0.25

    def _bar_chart(
        title: str,
        ylabel: str,
        filename: str,
        get_val,
        fmt_val=None,
        higher_is_better: bool = True,
    ):
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor("#FAFAFA")
        ax.set_facecolor("#FAFAFA")

        for i, (name, data, color) in enumerate(configs):
            vals = []
            for sl in active_seq_lens:
                r = data.get(sl)
                vals.append(get_val(r) if r is not None else 0)
            bars = ax.bar(x + (i - 1) * width, vals, width, label=name,
                          color=color, edgecolor="white", linewidth=0.5,
                          zorder=3)
            # Add value labels on bars
            for bar, v in zip(bars, vals):
                if v > 0:
                    label = fmt_val(v) if fmt_val else f"{v:.1f}"
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            label, ha="center", va="bottom", fontsize=7,
                            fontweight="bold", color="#333")

        ax.set_xlabel("Sequence Length", fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels([str(sl) for sl in active_seq_lens])
        ax.legend(frameon=True, fancybox=True, shadow=False, fontsize=10)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Mark OOM entries
        for i, (name, data, color) in enumerate(configs):
            for j, sl in enumerate(active_seq_lens):
                if data.get(sl) is None:
                    ax.text(x[j] + (i - 1) * width, 0, "OOM",
                            ha="center", va="bottom", fontsize=7,
                            color="red", fontweight="bold", rotation=90)

        fig.tight_layout()
        out_path = os.path.join(assets_dir, filename)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    print(f"\n  Generating charts → {assets_dir}/")

    _bar_chart(
        title="Qwen3-8B E2E Training — Throughput (tokens/sec)",
        ylabel="Tokens / sec",
        filename="bench_8b_throughput.png",
        get_val=lambda r: r.tokens_per_sec,
        fmt_val=lambda v: f"{v / 1000:.1f}K" if v >= 1000 else f"{v:.0f}",
        higher_is_better=True,
    )

    _bar_chart(
        title="Qwen3-8B E2E Training — Latency (ms / iteration)",
        ylabel="ms / iteration",
        filename="bench_8b_latency.png",
        get_val=lambda r: r.avg_iter_ms,
        fmt_val=lambda v: f"{v:.0f}" if v >= 10 else f"{v:.1f}",
        higher_is_better=False,
    )

    _bar_chart(
        title="Qwen3-8B E2E Training — Peak GPU Memory (GB)",
        ylabel="Peak Memory (GB)",
        filename="bench_8b_memory.png",
        get_val=lambda r: r.peak_memory_gb,
        fmt_val=lambda v: f"{v:.1f}",
        higher_is_better=False,
    )

    print("  Done!")


# ============================================================================
# Main entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["pytorch", "liger", "bastile"],
                        help="Run a single phase (used by subprocess)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index (used with --phase)")
    parser.add_argument("--output", type=str, default="",
                        help="Output JSON file (used with --phase)")
    parser.add_argument("--gpus", type=str, default="0,1,2",
                        help="Comma-separated GPU indices for parallel run (default: 0,1,2)")
    parser.add_argument("--sequential", action="store_true",
                        help="Run sequentially on current GPU instead of parallel")
    args = parser.parse_args()

    # Single-phase mode (called by subprocess)
    if args.phase:
        torch.cuda.set_device(args.gpu)
        run_phase(args.phase, args.output)
        return

    # Full benchmark
    print("=" * 110)
    print("  Qwen3-8B Sequence Length Sweep: PyTorch vs Liger vs Bastile")
    print("=" * 110)
    print(f"\n  Model: Qwen3-8B (36 layers, 4096 hidden, 32 heads), batch_size={BATCH_SIZE}")
    print(f"  Warmup: {WARMUP_ITERS} iters, Duration: {DURATION_SEC}s per run")
    print(f"  Sequence lengths: {SEQ_LENS}")
    print(f"  No pretrained weights — random init only")

    num_gpus = torch.cuda.device_count()
    print(f"\n  Available GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")

    if args.sequential or num_gpus < 3:
        if num_gpus < 3 and not args.sequential:
            print(f"\n  Only {num_gpus} GPU(s) available — falling back to sequential mode")
        print(f"\n  Mode: SEQUENTIAL (single GPU)")
        print_gpu_info()
        all_results = run_sequential()
    else:
        gpus = [int(g) for g in args.gpus.split(",")]
        print(f"\n  Mode: PARALLEL (GPUs {gpus})")
        all_results = run_parallel(gpus)

    print_results(all_results)
    plot_results(all_results)


if __name__ == "__main__":
    main()
