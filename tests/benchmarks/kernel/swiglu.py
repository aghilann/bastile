"""
SwiGLU Kernel Benchmark.

Compares CuTile SwiGLU vs Liger Kernel vs PyTorch reference.
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
class SwiGLUConfig:
    batch_size: int
    seq_len: int
    hidden_size: int
    dtype: torch.dtype

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.batch_size, self.seq_len, self.hidden_size)

    def __str__(self) -> str:
        dtype_str = str(self.dtype).split(".")[-1]
        return f"({self.batch_size}, {self.seq_len}, {self.hidden_size}) {dtype_str}"


def pytorch_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """PyTorch reference: SiLU(gate) * up"""
    return torch.nn.functional.silu(gate) * up


def run_single_benchmark(config: SwiGLUConfig, cutile_swiglu, liger_swiglu) -> dict:
    """Run benchmark for a single configuration."""
    clear_cuda_state()

    gate = torch.randn(*config.shape, device="cuda", dtype=config.dtype)
    up = torch.randn(*config.shape, device="cuda", dtype=config.dtype)

    pytorch_latency = benchmark_fn(lambda: pytorch_swiglu(gate, up))
    liger_latency = benchmark_fn(lambda: liger_swiglu(gate, up))
    cutile_latency = benchmark_fn(lambda: cutile_swiglu(gate, up))

    return {
        "pytorch": pytorch_latency,
        "liger": liger_latency,
        "cutile": cutile_latency,
    }


def jit_warmup(cutile_swiglu, liger_swiglu):
    """Pre-compile kernel variants."""
    print("JIT compiling kernels...")

    warmup_configs = [
        (4, 256, 2048, torch.float16),
        (4, 256, 2048, torch.bfloat16),
        (4, 256, 2048, torch.float32),
        (32, 512, 4096, torch.bfloat16),
    ]

    for batch, seq, hidden, dtype in warmup_configs:
        gate = torch.randn(batch, seq, hidden, device="cuda", dtype=dtype)
        up = torch.randn(batch, seq, hidden, device="cuda", dtype=dtype)
        for _ in range(3):
            _ = cutile_swiglu(gate, up)
            _ = liger_swiglu(gate, up)
        torch.cuda.synchronize()

    print("JIT compilation complete.\n")


def main():
    from liger_kernel.ops import LigerSiLUMulFunction

    from bastile.ops.swiglu import swiglu as cutile_swiglu

    def liger_swiglu(gate, up):
        return LigerSiLUMulFunction.apply(gate, up)

    print("=" * 100)
    print("SwiGLU Kernel Benchmark: CuTile vs Liger Kernel vs PyTorch")
    print("=" * 100)

    print_gpu_info()

    jit_warmup(cutile_swiglu, liger_swiglu)

    configs = [
        SwiGLUConfig(1, 128, 1024, torch.float16),
        SwiGLUConfig(4, 256, 2048, torch.float16),
        SwiGLUConfig(4, 256, 4096, torch.float16),
        SwiGLUConfig(8, 256, 2048, torch.float16),
        SwiGLUConfig(16, 512, 2048, torch.float16),
        SwiGLUConfig(4, 256, 2048, torch.bfloat16),
        SwiGLUConfig(16, 512, 2048, torch.bfloat16),
        SwiGLUConfig(32, 512, 4096, torch.bfloat16),
        SwiGLUConfig(4, 256, 2048, torch.float32),
        SwiGLUConfig(16, 512, 2048, torch.float32),
    ]

    print(f"{'Config':<40} {'PyTorch':>10} {'Liger':>10} {'CuTile':>10} {'CT/Py':>10} {'CT/Lg':>10}")
    print("-" * 100)

    all_results = []

    for config in configs:
        try:
            result = run_single_benchmark(config, cutile_swiglu, liger_swiglu)
            all_results.append(result)

            py = result["pytorch"]
            lg = result["liger"]
            ct = result["cutile"]

            speedup_vs_py = py / ct
            speedup_vs_lg = lg / ct

            print(
                f"{config!s:<40} {py:>8.1f}us {lg:>8.1f}us {ct:>8.1f}us "
                f"{format_speedup(speedup_vs_py):>10} {format_speedup(speedup_vs_lg):>10}"
            )
        except Exception as e:
            print(f"{config!s:<40} ERROR: {e}")

    # Summary
    print_header("SUMMARY", 100)

    if all_results:
        cutile_vs_pytorch = [r["pytorch"] / r["cutile"] for r in all_results]
        cutile_vs_liger = [r["liger"] / r["cutile"] for r in all_results]
        liger_vs_pytorch = [r["pytorch"] / r["liger"] for r in all_results]

        print("\nCuTile vs PyTorch:")
        print(f"  Average Speedup: {sum(cutile_vs_pytorch) / len(cutile_vs_pytorch):.2f}x")
        print(f"  Max Speedup:     {max(cutile_vs_pytorch):.2f}x")
        print(f"  Min Speedup:     {min(cutile_vs_pytorch):.2f}x")

        print("\nLiger vs PyTorch:")
        print(f"  Average Speedup: {sum(liger_vs_pytorch) / len(liger_vs_pytorch):.2f}x")
        print(f"  Max Speedup:     {max(liger_vs_pytorch):.2f}x")
        print(f"  Min Speedup:     {min(liger_vs_pytorch):.2f}x")

        print("\nCuTile vs Liger:")
        print(f"  Average Speedup: {sum(cutile_vs_liger) / len(cutile_vs_liger):.2f}x")
        print(f"  Max Speedup:     {max(cutile_vs_liger):.2f}x")
        print(f"  Min Speedup:     {min(cutile_vs_liger):.2f}x")

        avg_vs_liger = sum(cutile_vs_liger) / len(cutile_vs_liger)
        if avg_vs_liger >= 1.0:
            print(f"\n  Result: CuTile SwiGLU is {(avg_vs_liger - 1) * 100:.1f}% FASTER than Liger Kernel")
        else:
            print(f"\n  Result: CuTile SwiGLU is {(1 - avg_vs_liger) * 100:.1f}% SLOWER than Liger Kernel")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
