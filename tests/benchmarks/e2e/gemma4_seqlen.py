"""Single-GPU Gemma4 sequence-length sweep: PyTorch vs Bastile."""

import argparse

from .common import (
    GEMMA4_CONFIGS,
    add_common_args,
    parse_seq_lens,
    plot_results,
    print_results,
    print_run_header,
    run_benchmark,
    run_suite,
)

MODULE_PATH = "transformers.models.gemma4.modeling_gemma4"
MODEL_CLASS_NAME = "Gemma4ForCausalLM"
INSPECT_TARGETS = (
    ("RMSNorm:", "Gemma4RMSNorm"),
    ("MLP:", "Gemma4TextMLP"),
    ("RoPE:", "apply_rotary_pos_emb"),
)
PHASES = {
    "bastile": (
        "Bastile",
        lambda: __import__("bastile").apply(rms_norm=True, swiglu=False, rope=False, model_type="gemma4"),
    ),
    "pytorch": ("PyTorch", None),
}

MODEL_SPECS = {
    "1b": {
        "summary": "Gemma4-1B (18 layers, 1152 hidden, 4 heads)",
        "create_message": "Gemma4-1B model",
        "title_prefix": "Gemma4-1B E2E Training",
        "filename_prefix": "bench_gemma4_1b",
    },
    "4b": {
        "summary": "Gemma4-4B (26 layers, 2304 hidden, 8 heads)",
        "create_message": "Gemma4-4B model",
        "title_prefix": "Gemma4-4B E2E Training",
        "filename_prefix": "bench_gemma4_4b",
    },
    "12b": {
        "summary": "Gemma4-12B (40 layers, 3840 hidden, 16 heads)",
        "create_message": "Gemma4-12B model",
        "title_prefix": "Gemma4-12B E2E Training",
        "filename_prefix": "bench_gemma4_12b",
    },
    "27b": {
        "summary": "Gemma4-27B (62 layers, 5120 hidden, 32 heads)",
        "create_message": "Gemma4-27B model",
        "title_prefix": "Gemma4-27B E2E Training",
        "filename_prefix": "bench_gemma4_27b",
    },
}


def make_config(model_size: str):
    from transformers import Gemma4TextConfig

    if model_size not in GEMMA4_CONFIGS:
        raise ValueError(f"Unsupported model size: {model_size}")
    return Gemma4TextConfig(
        **GEMMA4_CONFIGS[model_size],
        max_position_embeddings=131072,
        rms_norm_eps=1e-6,
        vocab_size=262144,
    )


def run_one(model_size: str, **kwargs):
    return run_benchmark(
        module_path=MODULE_PATH,
        model_class_name=MODEL_CLASS_NAME,
        inspect_targets=INSPECT_TARGETS,
        create_message=MODEL_SPECS[model_size]["create_message"],
        **kwargs,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=["1b", "4b", "12b", "27b"], default="4b")
    add_common_args(parser)
    args = parser.parse_args()

    seq_lens = parse_seq_lens(args.seq_lens)
    spec = MODEL_SPECS[args.model_size]
    print_run_header(
        benchmark_title=f"Gemma4-{args.model_size.upper()} Sequence Length Sweep: PyTorch vs Bastile",
        model_summary=spec["summary"],
        batch_size=args.batch_size,
        warmup_iters=args.warmup_iters,
        duration_sec=args.duration_sec,
        seq_lens=seq_lens,
    )
    all_results = run_suite(
        config=make_config(args.model_size),
        phases=PHASES,
        seq_lens=seq_lens,
        run_one=lambda **kwargs: run_one(args.model_size, **kwargs),
        batch_size=args.batch_size,
        warmup_iters=args.warmup_iters,
        duration_sec=args.duration_sec,
    )
    print_results(all_results, seq_lens)
    if not args.no_plot:
        plot_results(
            all_results=all_results,
            seq_lens=seq_lens,
            title_prefix=spec["title_prefix"],
            filename_prefix=spec["filename_prefix"],
            assets_dir=args.assets_dir,
        )


if __name__ == "__main__":
    main()
