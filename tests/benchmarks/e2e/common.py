"""Shared helpers for end-to-end benchmark sweeps."""

from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path

import torch

from ..utils import E2EBenchmarkResult, clear_cuda_state, get_peak_memory_gb, print_gpu_info, print_header

# ---------------------------------------------------------------------------
# Gemma4 architecture configs
# ---------------------------------------------------------------------------

GEMMA4_CONFIGS = {
    "1b": dict(
        hidden_size=1152,
        intermediate_size=6912,
        num_hidden_layers=18,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=256,
    ),
    "4b": dict(
        hidden_size=2304,
        intermediate_size=9216,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
    ),
    "12b": dict(
        hidden_size=3840,
        intermediate_size=15360,
        num_hidden_layers=40,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=256,
    ),
    "27b": dict(
        hidden_size=5120,
        intermediate_size=16384,
        num_hidden_layers=62,
        num_attention_heads=32,
        num_key_value_heads=16,
        head_dim=128,
    ),
}


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--duration-sec", type=float, default=10.0)
    parser.add_argument("--seq-lens", type=str, default="256,512,1024,2048,4096,8192,16384,32768")
    parser.add_argument("--assets-dir", type=str, default="assets")
    parser.add_argument("--no-plot", action="store_true")


def parse_seq_lens(seq_lens: str) -> list[int]:
    return [int(part.strip()) for part in seq_lens.split(",") if part.strip()]


def print_run_header(
    *,
    benchmark_title: str,
    model_summary: str,
    batch_size: int,
    warmup_iters: int,
    duration_sec: float,
    seq_lens: list[int],
    extra_lines: list[str] | None = None,
) -> None:
    print_header(benchmark_title, width=110)
    print_gpu_info()
    print(f"Model: {model_summary}")
    print(f"Batch size: {batch_size}")
    print(f"Warmup iterations: {warmup_iters}")
    print(f"Timed duration: {duration_sec:.1f}s")
    print(f"Sequence lengths: {', '.join(str(seq_len) for seq_len in seq_lens)}")
    for line in extra_lines or []:
        print(line)


def _resolve_dtype(config) -> torch.dtype:
    torch_dtype = getattr(config, "torch_dtype", None) or getattr(config, "dtype", None)
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    if isinstance(torch_dtype, str):
        normalized = torch_dtype.removeprefix("torch.")
        return getattr(torch, normalized)
    return torch.bfloat16


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def _reset_bastile() -> None:
    try:
        import bastile
    except ImportError:
        return
    bastile.reset()


def _run_training_step(model, optimizer, input_ids, labels):
    optimizer.zero_grad(set_to_none=True)
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def run_benchmark(
    *,
    module_path: str,
    model_class_name: str,
    inspect_targets: tuple[tuple[str, str], ...],
    create_message: str,
    config,
    seq_len: int,
    phase_name: str,
    batch_size: int,
    warmup_iters: int,
    duration_sec: float,
) -> E2EBenchmarkResult:
    clear_cuda_state()

    module = importlib.import_module(module_path)
    model_class = getattr(module, model_class_name)
    print(f"\n[{phase_name}] Creating {create_message} at seq_len={seq_len}")
    for label, target_name in inspect_targets:
        target = getattr(module, target_name, None)
        if target is not None:
            print(f"  {label} {target_name}")

    device = torch.device("cuda")
    dtype = _resolve_dtype(config)
    model = model_class(config).to(device=device, dtype=dtype)
    model.train()

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_len),
        device=device,
        dtype=torch.long,
    )
    labels = input_ids.clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    torch.cuda.reset_peak_memory_stats()

    loss_history: list[float] = []
    for _ in range(warmup_iters):
        loss_history.append(_run_training_step(model, optimizer, input_ids, labels))
    torch.cuda.synchronize()

    start = time.perf_counter()
    iterations = 0
    while True:
        loss_history.append(_run_training_step(model, optimizer, input_ids, labels))
        iterations += 1
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if elapsed >= duration_sec:
            break

    total_time_sec = time.perf_counter() - start
    peak_memory_gb = get_peak_memory_gb()
    avg_iter_ms = (total_time_sec / iterations) * 1000.0
    tokens_per_sec = (batch_size * seq_len * iterations) / total_time_sec

    model = model.cpu()
    del optimizer, input_ids, labels, model
    clear_cuda_state()

    return E2EBenchmarkResult(
        name=phase_name,
        iterations=iterations,
        total_time_sec=total_time_sec,
        avg_iter_ms=avg_iter_ms,
        tokens_per_sec=tokens_per_sec,
        peak_memory_gb=peak_memory_gb,
        initial_loss=loss_history[0],
        final_loss=loss_history[-1],
        loss_history=loss_history,
    )


def run_suite(
    *,
    config,
    phases: dict[str, tuple[str, object | None]],
    seq_lens: list[int],
    run_one,
    batch_size: int,
    warmup_iters: int,
    duration_sec: float,
) -> dict[str, list[E2EBenchmarkResult | None]]:
    all_results = {phase_key: [] for phase_key in phases}

    for seq_len in seq_lens:
        for phase_key, (display_name, patch_fn) in phases.items():
            _reset_bastile()
            if patch_fn is not None:
                patch_fn()

            try:
                result = run_one(
                    config=config,
                    seq_len=seq_len,
                    phase_name=display_name,
                    batch_size=batch_size,
                    warmup_iters=warmup_iters,
                    duration_sec=duration_sec,
                )
            except Exception as exc:
                clear_cuda_state()
                if not _is_oom_error(exc):
                    raise
                print(f"[{display_name}] seq_len={seq_len} -> OOM")
                result = None
            finally:
                _reset_bastile()

            all_results[phase_key].append(result)

    return all_results


def _format_tokens_per_sec(result: E2EBenchmarkResult | None) -> str:
    if result is None:
        return "OOM"
    return f"{round(result.tokens_per_sec):,}"


def _format_ms(result: E2EBenchmarkResult | None) -> str:
    if result is None:
        return "OOM"
    return f"{result.avg_iter_ms:.1f}ms"


def _format_memory(result: E2EBenchmarkResult | None) -> str:
    if result is None:
        return "OOM"
    return f"{result.peak_memory_gb:.2f}GB"


def _format_delta(current: E2EBenchmarkResult | None, baseline: E2EBenchmarkResult | None) -> str:
    if current is None or baseline is None:
        return "-"
    delta_pct = ((current.tokens_per_sec / baseline.tokens_per_sec) - 1.0) * 100.0
    return f"{delta_pct:+.1f}%"


def _format_saved(current: E2EBenchmarkResult | None, baseline: E2EBenchmarkResult | None) -> str:
    if current is None or baseline is None:
        return "-"
    return f"{current.memory_saved_vs(baseline):+.2f}GB"


def print_results(all_results: dict[str, list[E2EBenchmarkResult | None]], seq_lens: list[int]) -> None:
    n = len(seq_lens)
    pytorch_results = all_results.get("pytorch", []) or ([None] * n)
    liger_results = all_results.get("liger", []) or ([None] * n)
    bastile_results = all_results.get("bastile", []) or ([None] * n)

    width = 110
    print("\n" + "=" * width)
    print("  RESULTS - TOKENS/SEC")
    print("=" * width)
    print()
    print(f"{'seq_len':>12} {'PyTorch':>15} {'Liger':>15} {'Bastile':>15} {'Liger d%':>12} {'Bastile d%':>14}")
    print("  " + "-" * 90)
    for seq_len, pytorch_result, liger_result, bastile_result in zip(
        seq_lens, pytorch_results, liger_results, bastile_results, strict=False
    ):
        print(
            f"{seq_len:>12} "
            f"{_format_tokens_per_sec(pytorch_result):>15} "
            f"{_format_tokens_per_sec(liger_result):>15} "
            f"{_format_tokens_per_sec(bastile_result):>15} "
            f"{_format_delta(liger_result, pytorch_result):>12} "
            f"{_format_delta(bastile_result, pytorch_result):>14}"
        )

    print("\n" + "=" * width)
    print("  RESULTS - MS/ITER")
    print("=" * width)
    print()
    print(f"{'seq_len':>12} {'PyTorch':>15} {'Liger':>15} {'Bastile':>15}")
    print("  " + "-" * 60)
    for seq_len, pytorch_result, liger_result, bastile_result in zip(
        seq_lens, pytorch_results, liger_results, bastile_results, strict=False
    ):
        print(
            f"{seq_len:>12} "
            f"{_format_ms(pytorch_result):>15} "
            f"{_format_ms(liger_result):>15} "
            f"{_format_ms(bastile_result):>15}"
        )

    print("\n" + "=" * width)
    print("  RESULTS - PEAK MEMORY (GB)")
    print("=" * width)
    print()
    print(f"{'seq_len':>12} {'PyTorch':>15} {'Liger':>15} {'Bastile':>15} {'Liger Saved':>15} {'Bastile Saved':>17}")
    print("  " + "-" * 93)
    for seq_len, pytorch_result, liger_result, bastile_result in zip(
        seq_lens, pytorch_results, liger_results, bastile_results, strict=False
    ):
        print(
            f"{seq_len:>12} "
            f"{_format_memory(pytorch_result):>15} "
            f"{_format_memory(liger_result):>15} "
            f"{_format_memory(bastile_result):>15} "
            f"{_format_saved(liger_result, pytorch_result):>15} "
            f"{_format_saved(bastile_result, pytorch_result):>17}"
        )

    print("\n" + "=" * width)
    print("  SUMMARY")
    print("=" * width)
    print()
    for seq_len, pytorch_result, liger_result, bastile_result in zip(
        seq_lens, pytorch_results, liger_results, bastile_results, strict=False
    ):
        pytorch_label = _format_tokens_per_sec(pytorch_result) + " tok/s" if pytorch_result else "OOM"
        liger_label = _format_tokens_per_sec(liger_result) + " tok/s" if liger_result else "OOM"
        if bastile_result is None:
            bastile_summary = "Bastile OOM"
        elif pytorch_result is None:
            bastile_summary = f"Bastile {_format_tokens_per_sec(bastile_result)} tok/s"
        else:
            bastile_summary = (
                f"Bastile {_format_delta(bastile_result, pytorch_result):>6} tput, "
                f"{_format_saved(bastile_result, pytorch_result)} mem"
            )
        print(f"  seq={seq_len:>5}: PyTorch {pytorch_label:>8} Liger {liger_label} | {bastile_summary}")
    print("\n" + "=" * width)


def _plot_metric(
    *,
    seq_lens: list[int],
    series: list[tuple[str, list[E2EBenchmarkResult | None], str]],
    title: str,
    y_label: str,
    filename: str,
    value_getter,
    value_formatter,
    assets_dir: str,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(seq_lens))
    width = 0.23 if len(series) >= 3 else 0.36

    fig, ax = plt.subplots(figsize=(17.85, 8.84), dpi=100)
    bars_by_series = []
    max_value = 0.0

    for index, (label, results, color) in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * width
        values = [value_getter(result) if result is not None else None for result in results]
        heights = [value or 0.0 for value in values]
        bars = ax.bar(x + offset, heights, width, label=label, color=color)
        bars_by_series.append((bars, values))
        valid_values = [value for value in values if value is not None]
        if valid_values:
            max_value = max(max_value, max(valid_values))

    ax.set_ylim(0, max_value * 1.18 if max_value > 0 else 1.0)
    ax.set_title(title, fontsize=27, fontweight="bold", pad=14)
    ax.set_xlabel("Sequence Length", fontsize=20, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=20, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(seq_len) for seq_len in seq_lens], fontsize=17)
    ax.tick_params(axis="y", labelsize=17, width=1.5, length=5)
    ax.grid(axis="y", alpha=0.3, linewidth=1)
    ax.set_axisbelow(True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.6)
    ax.spines["bottom"].set_linewidth(1.6)

    legend = ax.legend(loc="upper left", fontsize=16, frameon=True)
    legend.get_frame().set_alpha(0.9)

    y_offset = ax.get_ylim()[1] * 0.01
    for bars, values in bars_by_series:
        for bar, value in zip(bars, values, strict=False):
            x_pos = bar.get_x() + bar.get_width() / 2
            if value is None:
                ax.text(
                    x_pos,
                    ax.get_ylim()[1] * 0.008,
                    "OOM",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=12,
                    fontweight="bold",
                    color="red",
                )
                continue
            ax.text(
                x_pos,
                bar.get_height() + y_offset,
                value_formatter(value),
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                color="#444",
            )

    fig.tight_layout()
    output_path = Path(assets_dir) / filename
    print(f"  Saved: {output_path}")
    fig.savefig(output_path)
    plt.close(fig)


def plot_results(
    *,
    all_results: dict[str, list[E2EBenchmarkResult | None]],
    seq_lens: list[int],
    title_prefix: str,
    filename_prefix: str,
    assets_dir: str,
) -> None:
    colors = {
        "PyTorch": "#F54B26",
        "Liger": "#167CB3",
        "Bastile": "#5CDD73",
    }
    series = []
    if "pytorch" in all_results:
        series.append(("PyTorch", all_results["pytorch"], colors["PyTorch"]))
    if "liger" in all_results:
        series.append(("Liger", all_results["liger"], colors["Liger"]))
    if "bastile" in all_results:
        series.append(("Bastile", all_results["bastile"], colors["Bastile"]))

    assets_path = Path(assets_dir)
    assets_path.mkdir(parents=True, exist_ok=True)
    print(f"\n  Generating charts -> {assets_path.resolve()}/")

    _plot_metric(
        seq_lens=seq_lens,
        series=series,
        title=f"{title_prefix} - Throughput (tokens/sec)",
        y_label="Tokens / sec",
        filename=f"{filename_prefix}_throughput.png",
        value_getter=lambda result: result.tokens_per_sec,
        value_formatter=lambda value: f"{value / 1000:.1f}K" if value >= 1000 else f"{round(value)}",
        assets_dir=assets_dir,
    )
    _plot_metric(
        seq_lens=seq_lens,
        series=series,
        title=f"{title_prefix} - Latency (ms/iter)",
        y_label="ms / iter",
        filename=f"{filename_prefix}_latency.png",
        value_getter=lambda result: result.avg_iter_ms,
        value_formatter=lambda value: f"{value:.1f}",
        assets_dir=assets_dir,
    )
    _plot_metric(
        seq_lens=seq_lens,
        series=series,
        title=f"{title_prefix} - Peak GPU Memory (GB)",
        y_label="Peak Memory (GB)",
        filename=f"{filename_prefix}_memory.png",
        value_getter=lambda result: result.peak_memory_gb,
        value_formatter=lambda value: f"{value:.1f}",
        assets_dir=assets_dir,
    )
    print("  Done!")
