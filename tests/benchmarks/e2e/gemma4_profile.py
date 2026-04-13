"""Torch profiler run for Gemma4 — shows where training time is spent."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.profiler

from .common import GEMMA4_CONFIGS


def _make_config(model_size: str):
    from transformers import Gemma4TextConfig

    return Gemma4TextConfig(
        **GEMMA4_CONFIGS[model_size],
        max_position_embeddings=131072,
        rms_norm_eps=1e-6,
        vocab_size=262144,
    )


# ---------------------------------------------------------------------------
# One training step
# ---------------------------------------------------------------------------

def _step(model, optimizer, input_ids, labels, *, skip_optimizer: bool = False):
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    outputs = model(input_ids=input_ids, labels=labels)
    outputs.loss.backward()
    if optimizer is not None and not skip_optimizer:
        optimizer.step()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Gemma4 torch profiler")
    parser.add_argument("--model-size", choices=list(GEMMA4_CONFIGS), default="1b",
                        help="Model size (default: 1b)")
    parser.add_argument("--seq-len", type=int, default=16384,
                        help="Sequence length (default: 16384)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=2,
                        help="Warm-up steps before profiling starts (default: 2)")
    parser.add_argument("--profile-iters", type=int, default=3,
                        help="Steps to profile (default: 3)")
    parser.add_argument("--output", type=str, default=None,
                        help="Chrome-trace output path (default: gemma4_<size>_profile.json)")
    parser.add_argument("--top-k", type=int, default=30,
                        help="Rows to show in the printed table (default: 30)")
    parser.add_argument("--no-optimizer", action="store_true",
                        help="Skip optimizer step (forward+backward only); saves ~2x model memory")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Enable gradient checkpointing to reduce activation memory")
    parser.add_argument("--lce", action="store_true",
                        help="Apply bastile fused linear cross-entropy (skips fp32 logits materialization)")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"gemma4_{args.model_size}_profile.json"

    device = torch.device("cuda")
    config = _make_config(args.model_size)
    dtype = torch.bfloat16

    if args.lce:
        import bastile
        bastile.apply(
            model_type="gemma4",
            rms_norm=True,
            swiglu=False,
            rope=False,
            fused_linear_cross_entropy=False,
        )
        print("  bastile fused LCE: applied")

    opt_note = "  [no optimizer step — forward+backward only]" if args.no_optimizer else ""
    print(f"Gemma4-{args.model_size.upper()}  seq_len={args.seq_len}  batch={args.batch_size}  dtype={dtype}{opt_note}")
    print(f"  {config.num_hidden_layers} layers, hidden={config.hidden_size}, "
          f"heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}")

    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
    model = Gemma4ForCausalLM(config).to(device=device, dtype=dtype)
    model.train()
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        print("  gradient checkpointing: enabled")

    input_ids = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len),
                              device=device, dtype=torch.long)
    labels = input_ids.clone()
    optimizer = None if args.no_optimizer else torch.optim.AdamW(
        model.parameters(), lr=1e-4, fused=True
    )

    # Warm-up (not profiled)
    print(f"\nWarm-up ({args.warmup_iters} steps)…", flush=True)
    for _ in range(args.warmup_iters):
        _step(model, optimizer, input_ids, labels)
    torch.cuda.synchronize()

    # Profile
    print(f"Profiling ({args.profile_iters} steps)…", flush=True)
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
        profile_memory=True,
    ) as prof:
        for _ in range(args.profile_iters):
            with torch.profiler.record_function("training_step"):
                _step(model, optimizer, input_ids, labels)
        torch.cuda.synchronize()

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    key_avgs = prof.key_averages()

    # PyTorch >= 2.x uses device_time_total (device-agnostic naming).
    # Fall back to cuda_time_total for older builds.
    def _self_device(e):
        return getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)

    def _device_total(e):
        return getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)

    # Sort by self device (CUDA) time (descending)
    sorted_avgs = sorted(key_avgs, key=_self_device, reverse=True)

    col_w = 52
    print(f"\n{'─' * 110}")
    print(f"  Top-{args.top_k} ops by self CUDA time  "
          f"(over {args.profile_iters} step(s), seq_len={args.seq_len})")
    print(f"{'─' * 110}")
    header = (
        f"{'Op':<{col_w}} "
        f"{'Calls':>6} "
        f"{'Self CPU':>10} "
        f"{'CPU total':>10} "
        f"{'Self CUDA':>10} "
        f"{'CUDA total':>11}"
    )
    print(header)
    print("─" * 110)

    def _fmt_us(us: float) -> str:
        if us >= 1_000_000:
            return f"{us / 1_000_000:.2f}s"
        if us >= 1_000:
            return f"{us / 1_000:.2f}ms"
        return f"{us:.1f}µs"

    for entry in sorted_avgs[: args.top_k]:
        name = entry.key
        if len(name) > col_w:
            name = name[: col_w - 1] + "…"
        print(
            f"{name:<{col_w}} "
            f"{entry.count:>6} "
            f"{_fmt_us(entry.self_cpu_time_total):>10} "
            f"{_fmt_us(entry.cpu_time_total):>10} "
            f"{_fmt_us(_self_device(entry)):>10} "
            f"{_fmt_us(_device_total(entry)):>11}"
        )

    print("─" * 110)

    # -----------------------------------------------------------------------
    # Chrome trace
    # -----------------------------------------------------------------------
    output_path = Path(args.output)
    prof.export_chrome_trace(str(output_path))
    print(f"\nChrome trace written to: {output_path.resolve()}")
    print("  Open it in Chrome at  chrome://tracing  or  https://ui.perfetto.dev")


if __name__ == "__main__":
    main()
