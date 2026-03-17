"""Single-GPU Qwen3 sequence-length sweep."""

import argparse

from .common import (
    add_common_args,
    parse_seq_lens,
    plot_results,
    print_results,
    print_run_header,
    run_benchmark,
    run_suite,
)

MODULE_PATH = "transformers.models.qwen3.modeling_qwen3"
MODEL_CLASS_NAME = "Qwen3ForCausalLM"
INSPECT_TARGETS = (
    ("RMSNorm:", "Qwen3RMSNorm"),
    ("MLP:", "Qwen3MLP"),
    ("RoPE:", "apply_rotary_pos_emb"),
)
PHASES = {
    "bastile": (
        "Bastile",
        lambda: __import__("bastile").apply(rms_norm=True, swiglu=True, rope=True, fused_linear_cross_entropy=True),
    ),
    "pytorch": ("PyTorch", None),
    "liger": (
        "Liger",
        lambda: __import__(
            "liger_kernel.transformers", fromlist=["apply_liger_kernel_to_qwen3"]
        ).apply_liger_kernel_to_qwen3(),
    ),
}

MODEL_SPECS = {
    "0.6b": {
        "summary": "Qwen3-0.6B (28 layers, 1024 hidden, 16 heads)",
        "create_message": "Qwen3-0.6B model",
        "title_prefix": "Qwen3-0.6B E2E Training",
        "filename_prefix": "bench_qwen3_0_6b",
    },
    "8b": {
        "summary": "Qwen3-8B (36 layers, 4096 hidden, 32 heads)",
        "create_message": "Qwen3-8B model",
        "title_prefix": "Qwen3-8B E2E Training",
        "filename_prefix": "bench_8b",
    },
}


def make_config(model_size: str):
    from transformers import Qwen3Config

    if model_size == "0.6b":
        return Qwen3Config(
            architectures=["Qwen3ForCausalLM"],
            attention_bias=False,
            attention_dropout=0.0,
            bos_token_id=151643,
            eos_token_id=151645,
            head_dim=128,
            hidden_act="silu",
            hidden_size=1024,
            initializer_range=0.02,
            intermediate_size=3072,
            max_position_embeddings=40960,
            max_window_layers=28,
            model_type="qwen3",
            num_attention_heads=16,
            num_hidden_layers=28,
            num_key_value_heads=8,
            rms_norm_eps=1e-6,
            rope_scaling=None,
            rope_theta=1000000,
            sliding_window=None,
            tie_word_embeddings=True,
            torch_dtype="bfloat16",
            transformers_version="4.51.0",
            use_cache=True,
            use_sliding_window=False,
            vocab_size=151936,
        )

    if model_size == "8b":
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

    raise ValueError(f"Unsupported model size: {model_size}")


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
    parser.add_argument("--model-size", choices=["0.6b", "8b"], default="8b")
    add_common_args(parser)
    args = parser.parse_args()

    seq_lens = parse_seq_lens(args.seq_lens)
    spec = MODEL_SPECS[args.model_size]
    print_run_header(
        benchmark_title=f"Qwen3-{args.model_size.upper()} Sequence Length Sweep: PyTorch vs Liger vs Bastile",
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
