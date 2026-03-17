from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SEQ_LENS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

PYTORCH_TPUT = [361, 567, 945, 1377, 1693, None, None, None]
BASTILE_TPUT = [388, 705, 1113, 1641, 1995, None, None, None]

PYTORCH_LAT_MS = [709.5, 902.6, 1083.4, 1487.4, 2419.7, None, None, None]
BASTILE_LAT_MS = [660.5, 726.4, 919.9, 1248.1, 2053.2, None, None, None]

PYTORCH_MEM_GB = [83.77, 83.99, 84.43, 94.16, 138.21, None, None, None]
BASTILE_MEM_GB = [83.65, 83.75, 83.96, 86.43, 118.97, None, None, None]

TITLE_PREFIX = "Qwen3.5-9B E2E Training"
ASSETS_DIR = Path("assets")
FIGSIZE = (17.85, 8.84)
DPI = 100

COLORS = {
    "PyTorch": "#F54B26",
    "Bastile": "#5CDD73",
}


def _format_seq_len(seq_len: int) -> str:
    return str(seq_len)


def _format_throughput(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:.1f}K"
    return f"{round(value)}"


def _annotate_bars(ax, bars, values, formatter):
    ymax = ax.get_ylim()[1]
    offset = ymax * 0.01
    for bar, value in zip(bars, values):
        if value is None:
            continue
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        ax.text(x, y + offset, formatter(value), ha="center", va="bottom", fontsize=12, fontweight="bold", color="#444")


def _annotate_oom(ax, xs, values):
    ymax = ax.get_ylim()[1]
    y = ymax * 0.008
    for x, value in zip(xs, values):
        if value is None:
            ax.text(x, y, "OOM", ha="center", va="bottom", rotation=90, fontsize=12, fontweight="bold", color="red")


def _plot_metric(title_suffix: str, y_label: str, pytorch_values, bastile_values, filename: str, formatter):
    x = np.arange(len(SEQ_LENS))
    width = 0.36

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    pytorch_heights = [v or 0 for v in pytorch_values]
    bastile_heights = [v or 0 for v in bastile_values]

    bars_pt = ax.bar(x - width / 2, pytorch_heights, width, label="PyTorch", color=COLORS["PyTorch"])
    bars_bs = ax.bar(x + width / 2, bastile_heights, width, label="Bastile", color=COLORS["Bastile"])

    valid_values = [v for v in [*pytorch_values, *bastile_values] if v is not None]
    ymax = max(valid_values) * 1.18 if valid_values else 1
    ax.set_ylim(0, ymax)

    ax.set_title(f"{TITLE_PREFIX} - {title_suffix}", fontsize=27, fontweight="bold", pad=14)
    ax.set_xlabel("Sequence Length", fontsize=20, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=20, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([_format_seq_len(seq_len) for seq_len in SEQ_LENS], fontsize=17)
    ax.tick_params(axis="y", labelsize=17, width=1.5, length=5)
    ax.grid(axis="y", alpha=0.3, linewidth=1)
    ax.set_axisbelow(True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.6)
    ax.spines["bottom"].set_linewidth(1.6)

    legend = ax.legend(loc="upper left", fontsize=16, frameon=True)
    legend.get_frame().set_alpha(0.9)

    _annotate_bars(ax, bars_pt, pytorch_values, formatter)
    _annotate_bars(ax, bars_bs, bastile_values, formatter)
    _annotate_oom(ax, x - width / 2, pytorch_values)
    _annotate_oom(ax, x + width / 2, bastile_values)

    fig.tight_layout()
    out_path = ASSETS_DIR / filename
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    ASSETS_DIR.mkdir(exist_ok=True)
    _plot_metric(
        title_suffix="Throughput (tokens/sec)",
        y_label="Tokens / sec",
        pytorch_values=PYTORCH_TPUT,
        bastile_values=BASTILE_TPUT,
        filename="bench_qwen3_5_9b_throughput.png",
        formatter=_format_throughput,
    )
    _plot_metric(
        title_suffix="Latency (ms/iter)",
        y_label="ms / iter",
        pytorch_values=PYTORCH_LAT_MS,
        bastile_values=BASTILE_LAT_MS,
        filename="bench_qwen3_5_9b_latency.png",
        formatter=lambda value: f"{value:.1f}",
    )
    _plot_metric(
        title_suffix="Peak GPU Memory (GB)",
        y_label="Peak Memory (GB)",
        pytorch_values=PYTORCH_MEM_GB,
        bastile_values=BASTILE_MEM_GB,
        filename="bench_qwen3_5_9b_memory.png",
        formatter=lambda value: f"{value:.1f}",
    )


if __name__ == "__main__":
    main()
