"""
RoPE (Rotary Position Embedding) Kernel Benchmark.

Compares Bastile RoPE vs Liger Kernel vs PyTorch reference.

Note: Liger RoPE expects (bsz, n_head, seq_len, head_dim) format and
cos/sin with shape (1, seq_len, head_dim) or (bsz, seq_len, head_dim).
"""

from dataclasses import dataclass

import torch

from ..utils import (
    benchmark_fn,
    clear_cuda_state,
    format_speedup,
    print_gpu_info,
    print_header,
)


@dataclass
class RoPEConfig:
    batch_size: int
    num_heads: int
    seq_len: int
    head_dim: int
    dtype: torch.dtype

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (self.batch_size, self.num_heads, self.seq_len, self.head_dim)

    def __str__(self) -> str:
        dtype_str = str(self.dtype).split(".")[-1]
        return f"B{self.batch_size}_H{self.num_heads}_S{self.seq_len}_D{self.head_dim} {dtype_str}"


def rotate_half(x):
    """Rotate half the hidden dims."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def pytorch_rope(q, k, cos, sin, unsqueeze_dim=1):
    """PyTorch reference RoPE."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def run_single_benchmark(config: RoPEConfig, bastile_rope, liger_rope) -> dict:
    """Run benchmark for a single configuration."""
    clear_cuda_state()

    # Shape: (bsz, n_head, seq_len, head_dim)
    q = torch.randn(*config.shape, device="cuda", dtype=config.dtype)
    k = torch.randn(*config.shape, device="cuda", dtype=config.dtype)

    # HuggingFace format: (batch, seq_len, head_dim)
    cos = torch.randn(config.batch_size, config.seq_len, config.head_dim, device="cuda", dtype=config.dtype)
    sin = torch.randn(config.batch_size, config.seq_len, config.head_dim, device="cuda", dtype=config.dtype)

    pytorch_latency = benchmark_fn(lambda: pytorch_rope(q, k, cos, sin))
    bastile_latency = benchmark_fn(lambda: bastile_rope(q, k, cos, sin))

    # Liger expects cos/sin with shape that broadcasts correctly
    # For Liger: q, k are (bsz, n_head, seq_len, head_dim)
    # cos/sin should be (bsz, seq_len, head_dim // 2) for the half-rotation
    cos_liger = cos[..., : config.head_dim // 2].contiguous()
    sin_liger = sin[..., : config.head_dim // 2].contiguous()

    try:
        liger_latency = benchmark_fn(lambda: liger_rope(q.clone(), k.clone(), cos_liger, sin_liger))
    except Exception:
        # Fallback: try with full cos/sin
        liger_latency = benchmark_fn(lambda: liger_rope(q.clone(), k.clone(), cos, sin))

    return {
        "pytorch": pytorch_latency,
        "liger": liger_latency,
        "bastile": bastile_latency,
    }


def main():
    from liger_kernel.ops import LigerRopeFunction

    from bastile.ops.rope import apply_rotary_pos_emb as bastile_rope

    def liger_rope(q, k, cos, sin):
        return LigerRopeFunction.apply(q, k, cos, sin)

    print("=" * 100)
    print("RoPE Kernel Benchmark: Bastile vs Liger Kernel vs PyTorch")
    print("=" * 100)

    print_gpu_info()
    print()

    configs = [
        RoPEConfig(1, 32, 512, 128, torch.float16),
        RoPEConfig(4, 32, 512, 128, torch.float16),
        RoPEConfig(8, 32, 1024, 128, torch.float16),
        RoPEConfig(16, 32, 512, 128, torch.float16),
        RoPEConfig(4, 32, 512, 128, torch.bfloat16),
        RoPEConfig(8, 32, 1024, 128, torch.bfloat16),
        RoPEConfig(4, 16, 512, 64, torch.float16),
        RoPEConfig(4, 64, 512, 64, torch.float16),
        RoPEConfig(4, 32, 512, 128, torch.float32),
    ]

    print(f"{'Config':<45} {'PyTorch':>10} {'Liger':>10} {'Bastile':>10} {'Ba/Py':>10} {'Ba/Lg':>10}")
    print("-" * 100)

    all_results = []

    for config in configs:
        try:
            result = run_single_benchmark(config, bastile_rope, liger_rope)
            all_results.append(result)

            py = result["pytorch"]
            lg = result["liger"]
            ba = result["bastile"]

            speedup_vs_py = py / ba
            speedup_vs_lg = lg / ba

            print(
                f"{config!s:<45} {py:>8.1f}us {lg:>8.1f}us {ba:>8.1f}us "
                f"{format_speedup(speedup_vs_py):>10} {format_speedup(speedup_vs_lg):>10}"
            )
        except Exception as e:
            print(f"{config!s:<45} ERROR: {e}")

    # Summary
    print_header("SUMMARY", 100)

    if all_results:
        bastile_vs_pytorch = [r["pytorch"] / r["bastile"] for r in all_results]
        bastile_vs_liger = [r["liger"] / r["bastile"] for r in all_results]
        liger_vs_pytorch = [r["pytorch"] / r["liger"] for r in all_results]

        print("\nBastile vs PyTorch:")
        print(f"  Average Speedup: {sum(bastile_vs_pytorch) / len(bastile_vs_pytorch):.2f}x")
        print(f"  Max Speedup:     {max(bastile_vs_pytorch):.2f}x")
        print(f"  Min Speedup:     {min(bastile_vs_pytorch):.2f}x")

        print("\nLiger vs PyTorch:")
        print(f"  Average Speedup: {sum(liger_vs_pytorch) / len(liger_vs_pytorch):.2f}x")
        print(f"  Max Speedup:     {max(liger_vs_pytorch):.2f}x")
        print(f"  Min Speedup:     {min(liger_vs_pytorch):.2f}x")

        print("\nBastile vs Liger:")
        print(f"  Average Speedup: {sum(bastile_vs_liger) / len(bastile_vs_liger):.2f}x")
        print(f"  Max Speedup:     {max(bastile_vs_liger):.2f}x")
        print(f"  Min Speedup:     {min(bastile_vs_liger):.2f}x")

        avg_vs_liger = sum(bastile_vs_liger) / len(bastile_vs_liger)
        if avg_vs_liger >= 1.0:
            print(f"\n  Result: Bastile RoPE is {(avg_vs_liger - 1) * 100:.1f}% FASTER than Liger Kernel")
        else:
            print(f"\n  Result: Bastile RoPE is {(1 - avg_vs_liger) * 100:.1f}% SLOWER than Liger Kernel")

        print("\n  Note: Bastile RoPE uses CuTile kernel; Liger uses Triton kernel.")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
