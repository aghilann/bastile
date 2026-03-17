"""Single-GPU Qwen3.5 sequence-length sweep."""

import argparse
import json

from .common import (
    add_common_args,
    parse_seq_lens,
    plot_results,
    print_results,
    print_run_header,
    run_benchmark,
    run_suite,
)

MODULE_PATH = "transformers.models.qwen3_5.modeling_qwen3_5"
MODEL_CLASS_NAME = "Qwen3_5ForCausalLM"
INSPECT_TARGETS = (
    ("RMSNorm:", "Qwen3_5RMSNorm"),
    ("MLP:", "Qwen3_5MLP"),
    ("RoPE:", "apply_rotary_pos_emb"),
)
_LAYER_TYPES_24 = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 6
_LAYER_TYPES_32 = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 8
MODEL_PRESETS: dict[str, dict] = {
    "0.8b": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
        "dtype": "bfloat16",
        "eos_token_id": 248044,
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 3584,
        "layer_types": _LAYER_TYPES_24,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "linear_value_head_dim": 128,
        "max_position_embeddings": 262144,
        "mlp_only_layers": [],
        "model_type": "qwen3_5_text",
        "mamba_ssm_dtype": "float32",
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "num_attention_heads": 8,
        "num_hidden_layers": 24,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
        },
        "tie_word_embeddings": True,
        "use_cache": True,
        "vocab_size": 248320,
    },
    "9b": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
        "dtype": "bfloat16",
        "eos_token_id": 248044,
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 12288,
        "layer_types": _LAYER_TYPES_32,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
        "max_position_embeddings": 262144,
        "mlp_only_layers": [],
        "model_type": "qwen3_5_text",
        "mamba_ssm_dtype": "float32",
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "num_attention_heads": 16,
        "num_hidden_layers": 32,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
        },
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": 248320,
    },
}
PHASES = {
    "bastile": (
        "Bastile",
        lambda: __import__("bastile").apply(
            rms_norm=True,
            swiglu=True,
            rope=True,
            fused_linear_cross_entropy=True,
            model_type="qwen3_5",
        ),
    ),
    "pytorch": ("PyTorch", None),
    "liger": (
        "Liger",
        lambda: (
            getattr(__import__("liger_kernel.transformers", fromlist=["*"]), "apply_liger_kernel_to_qwen3_5", None)
            or (_ for _ in ()).throw(
                RuntimeError("Liger does not expose apply_liger_kernel_to_qwen3_5 in this environment")
            )
        )(
            rope=True,
            rms_norm=True,
            swiglu=True,
            fused_linear_cross_entropy=True,
        ),
    ),
}


def load_text_config_dict(model_size: str, config_json: str | None) -> dict:
    if config_json:
        with open(config_json) as f:
            raw = json.load(f)
        return raw.get("text_config", raw)
    if model_size in MODEL_PRESETS:
        return MODEL_PRESETS[model_size]
    raise ValueError(
        f"No built-in config preset for model size '{model_size}'. "
        "Pass --config-json with the HF config.json or text_config JSON."
    )


def make_config(model_size: str, config_json: str | None):
    try:
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This environment does not include transformers.models.qwen3_5. "
            "Upgrade the pinned transformers version in the benchmark environment before running Qwen3.5 benchmarks."
        ) from exc

    return Qwen3_5TextConfig(**load_text_config_dict(model_size, config_json))


def run_one(**kwargs):
    return run_benchmark(
        module_path=MODULE_PATH,
        model_class_name=MODEL_CLASS_NAME,
        inspect_targets=INSPECT_TARGETS,
        create_message="Qwen3.5 model",
        **kwargs,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=["0.8b", "8b", "9b"], default="9b")
    parser.add_argument("--config-json", type=str, default=None)
    add_common_args(parser)
    args = parser.parse_args()

    if args.model_size == "8b" and args.config_json is None:
        raise SystemExit("Qwen3.5-8B requires --config-json because no built-in preset is registered.")

    seq_lens = parse_seq_lens(args.seq_lens)
    config = make_config(args.model_size, args.config_json)
    print_run_header(
        benchmark_title=f"Qwen3.5-{args.model_size.upper()} Sequence Length Sweep: PyTorch vs Liger vs Bastile",
        model_summary=(
            f"Qwen3.5-{args.model_size.upper()} "
            f"({config.num_hidden_layers} layers, {config.hidden_size} hidden, {config.num_attention_heads} attn heads)"
        ),
        batch_size=args.batch_size,
        warmup_iters=args.warmup_iters,
        duration_sec=args.duration_sec,
        seq_lens=seq_lens,
        extra_lines=[
            f"Config source: {args.config_json}"
            if args.config_json
            else f"Config source: built-in preset '{args.model_size}'"
        ],
    )
    all_results = run_suite(
        config=config,
        phases=PHASES,
        seq_lens=seq_lens,
        run_one=run_one,
        batch_size=args.batch_size,
        warmup_iters=args.warmup_iters,
        duration_sec=args.duration_sec,
    )
    print_results(all_results, seq_lens)
    if not args.no_plot:
        plot_results(
            all_results=all_results,
            seq_lens=seq_lens,
            title_prefix=f"Qwen3.5-{args.model_size.upper()} E2E Training",
            filename_prefix=f"bench_qwen3_5_{args.model_size.replace('.', '_')}",
            assets_dir=args.assets_dir,
        )


if __name__ == "__main__":
    main()
