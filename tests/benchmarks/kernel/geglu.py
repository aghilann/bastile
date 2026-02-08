"""
GEGLU Kernel Benchmark (GPT-OSS Activation).

GPT-OSS uses a custom GEGLU: (up + 1) * gate * sigmoid(gate * 1.702)
This is DIFFERENT from standard GEGLU (GELU * up).

Compares CuTile GEGLU vs Liger GEGLU (tanh approx) vs PyTorch reference.
Note: Liger uses standard GEGLU with tanh approximation, so comparison
is not perfectly fair - just shows relative kernel performance.
"""

import torch
from dataclasses import dataclass
from typing import Tuple

from ..utils import (
    benchmark_fn,
    clear_cuda_state,
    print_header,
    print_gpu_info,
    format_speedup,
    jit_warmup_kernel,
)


# Constants from GPT-OSS
ALPHA = 1.702
LIMIT = 7.0


@dataclass
class GEGLUConfig:
    num_tokens: int
    expert_dim: int
    dtype: torch.dtype

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.num_tokens, 2 * self.expert_dim)

    def __str__(self) -> str:
        dtype_str = str(self.dtype).split('.')[-1]
        return f"({self.num_tokens}, {self.expert_dim}) {dtype_str}"


def pytorch_geglu_gptoss(gate_up: torch.Tensor) -> torch.Tensor:
    """PyTorch reference GPT-OSS GEGLU activation."""
    gate = gate_up[..., 0::2].clone()
    up = gate_up[..., 1::2].clone()
    gate = gate.clamp(max=LIMIT)
    up = up.clamp(min=-LIMIT, max=LIMIT)
    glu = gate * torch.sigmoid(gate * ALPHA)
    return (up + 1) * glu


def run_single_benchmark(config: GEGLUConfig, cutile_geglu, liger_geglu) -> dict:
    """Run benchmark for a single configuration."""
    clear_cuda_state()

    gate_up = torch.randn(*config.shape, device="cuda", dtype=config.dtype)
    
    # For Liger, we need separate gate and up tensors
    gate = gate_up[..., 0::2].contiguous()
    up = gate_up[..., 1::2].contiguous()

    pytorch_latency = benchmark_fn(lambda: pytorch_geglu_gptoss(gate_up))
    cutile_latency = benchmark_fn(lambda: cutile_geglu(gate_up))
    liger_latency = benchmark_fn(lambda: liger_geglu(gate, up))

    return {
        "pytorch": pytorch_latency,
        "liger": liger_latency,
        "cutile": cutile_latency,
    }


def main():
    from bastile.ops.gpt_oss_moe import geglu_activation as cutile_geglu
    from liger_kernel.ops import LigerGELUMulFunction
    
    def liger_geglu(gate, up):
        return LigerGELUMulFunction.apply(gate, up)

    print("=" * 100)
    print("GEGLU Kernel Benchmark (GPT-OSS): CuTile vs Liger Kernel vs PyTorch")
    print("=" * 100)

    print_gpu_info()
    
    print("\nNote: GPT-OSS GEGLU = (up + 1) * gate * sigmoid(gate * 1.702)")
    print("      Liger GEGLU   = GELU(gate) * up (tanh approximation)")
    print("      These are different activations - comparing kernel speed only.\n")
    
    # JIT warmup
    def input_gen(cfg):
        num_tokens, expert_dim, dtype = cfg
        return torch.randn(num_tokens, 2 * expert_dim, device="cuda", dtype=dtype)
    
    warmup_configs = [
        (128, 256, torch.float16),
        (128, 256, torch.bfloat16),
        (512, 1024, torch.bfloat16),
        (1024, 2048, torch.float32),
    ]
    jit_warmup_kernel(cutile_geglu, input_gen, warmup_configs)
    
    # Also warmup Liger
    for cfg in warmup_configs:
        num_tokens, expert_dim, dtype = cfg
        gate = torch.randn(num_tokens, expert_dim, device="cuda", dtype=dtype)
        up = torch.randn(num_tokens, expert_dim, device="cuda", dtype=dtype)
        for _ in range(3):
            _ = liger_geglu(gate, up)
        torch.cuda.synchronize()
    
    configs = [
        GEGLUConfig(64, 256, torch.float16),
        GEGLUConfig(128, 256, torch.float16),
        GEGLUConfig(256, 512, torch.float16),
        GEGLUConfig(512, 1024, torch.float16),
        GEGLUConfig(1024, 1024, torch.float16),
        GEGLUConfig(2048, 2048, torch.float16),
        GEGLUConfig(128, 256, torch.bfloat16),
        GEGLUConfig(512, 1024, torch.bfloat16),
        GEGLUConfig(1024, 2048, torch.bfloat16),
        GEGLUConfig(128, 256, torch.float32),
        GEGLUConfig(512, 1024, torch.float32),
    ]

    print(f"{'Config':<35} {'PyTorch':>10} {'Liger':>10} {'CuTile':>10} {'CT/Py':>10} {'CT/Lg':>10}")
    print("-" * 100)

    all_results = []
    
    for config in configs:
        try:
            result = run_single_benchmark(config, cutile_geglu, liger_geglu)
            all_results.append(result)
            
            py = result["pytorch"]
            lg = result["liger"]
            ct = result["cutile"]
            
            speedup_vs_py = py / ct
            speedup_vs_lg = lg / ct
            
            print(f"{str(config):<35} {py:>8.1f}us {lg:>8.1f}us {ct:>8.1f}us "
                  f"{format_speedup(speedup_vs_py):>10} {format_speedup(speedup_vs_lg):>10}")
        except Exception as e:
            print(f"{str(config):<35} ERROR: {e}")

    # Summary
    print_header("SUMMARY", 100)
    
    if all_results:
        cutile_vs_pytorch = [r["pytorch"] / r["cutile"] for r in all_results]
        cutile_vs_liger = [r["liger"] / r["cutile"] for r in all_results]
        liger_vs_pytorch = [r["pytorch"] / r["liger"] for r in all_results]
        
        print(f"\nCuTile vs PyTorch:")
        print(f"  Average Speedup: {sum(cutile_vs_pytorch)/len(cutile_vs_pytorch):.2f}x")
        print(f"  Max Speedup:     {max(cutile_vs_pytorch):.2f}x")
        print(f"  Min Speedup:     {min(cutile_vs_pytorch):.2f}x")
        
        print(f"\nLiger vs PyTorch:")
        print(f"  Average Speedup: {sum(liger_vs_pytorch)/len(liger_vs_pytorch):.2f}x")
        print(f"  Max Speedup:     {max(liger_vs_pytorch):.2f}x")
        print(f"  Min Speedup:     {min(liger_vs_pytorch):.2f}x")
        
        print(f"\nCuTile vs Liger:")
        print(f"  Average Speedup: {sum(cutile_vs_liger)/len(cutile_vs_liger):.2f}x")
        print(f"  Max Speedup:     {max(cutile_vs_liger):.2f}x")
        print(f"  Min Speedup:     {min(cutile_vs_liger):.2f}x")
        
        avg_vs_liger = sum(cutile_vs_liger)/len(cutile_vs_liger)
        if avg_vs_liger >= 1.0:
            print(f"\n  Result: CuTile GEGLU is {(avg_vs_liger - 1) * 100:.1f}% FASTER than Liger Kernel")
        else:
            print(f"\n  Result: CuTile GEGLU is {(1 - avg_vs_liger) * 100:.1f}% SLOWER than Liger Kernel")
    
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
