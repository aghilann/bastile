"""
Kernel-level profiling: PyTorch vs Liger Kernel vs Bastile
Profiles 5 training iterations (after warmup) and prints top CUDA kernels.
"""

import torch
import gc
import importlib
from typing import Optional, Callable

from ..utils import clear_cuda_state, reset_peak_memory, print_header, print_gpu_info


def reset_environment():
    clear_cuda_state()
    reset_peak_memory()
    import transformers.models.qwen3.modeling_qwen3 as qwen3_mod
    importlib.reload(qwen3_mod)


def profile_config(
    name: str,
    setup_fn: Optional[Callable] = None,
    warmup_iters: int = 50,
    profile_iters: int = 5,
    batch_size: int = 4,
    seq_len: int = 512,
):
    """Profile a single configuration and print top CUDA kernels."""

    reset_environment()

    if setup_fn:
        print(f"  Applying {name} patches...")
        applied = setup_fn()
        if applied:
            print(f"  Patches applied: {applied}")

    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=32000,
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        head_dim=64,
    )

    print(f"  Creating Qwen3 model...")
    model = Qwen3ForCausalLM(config).cuda().to(torch.bfloat16)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
    attention_mask = torch.ones_like(input_ids)

    # Warmup
    print(f"  Warming up ({warmup_iters} iterations)...")
    for _ in range(warmup_iters):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        outputs.loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # Profile
    print(f"  Profiling ({profile_iters} iterations)...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for _ in range(profile_iters):
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

    # Print top CUDA kernels by total CUDA time
    print(f"\n  Top 30 CUDA kernels by total GPU time ({name}):")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    # Also print self CUDA time to see where GPU time is actually spent (not in children)
    print(f"\n  Top 30 CUDA kernels by self GPU time ({name}):")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))

    # Cleanup
    del model, optimizer
    clear_cuda_state()

    return prof


def setup_liger():
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3
    apply_liger_kernel_to_qwen3(
        rope=True, rms_norm=True, swiglu=True, fused_linear_cross_entropy=True,
    )
    return "rope, rms_norm, swiglu, fused_linear_cross_entropy"


def setup_bastile():
    import bastile
    bastile.reset()
    applied = bastile.apply(rms_norm=True, swiglu=True, rope=True, fused_linear_cross_entropy=True)
    return applied


def main():
    print("=" * 120)
    print("  Kernel Profiling: PyTorch vs Liger Kernel vs Bastile (Qwen3 0.5B)")
    print("=" * 120)
    print_gpu_info()

    # 1. Bastile
    print_header("Profile: Bastile", 120)
    profile_config("Bastile", setup_fn=setup_bastile)

    import bastile
    bastile.reset()

    # 2. Liger
    print_header("Profile: Liger Kernel", 120)
    reset_environment()
    try:
        profile_config("Liger", setup_fn=setup_liger)
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

    # 3. PyTorch baseline
    print_header("Profile: PyTorch (Baseline)", 120)
    reset_environment()
    profile_config("PyTorch", setup_fn=None)

    print("\n" + "=" * 120)
    print("  Profiling complete.")
    print("=" * 120)


if __name__ == "__main__":
    main()
