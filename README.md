# Bastile

Optimized CuTile kernels for finetuning HuggingFace models.

## Performance Summary

| Model | Speedup | Memory Saved | Key Optimization |
|-------|---------|--------------|------------------|
| **GPT-OSS 20B** | **+16%** | 0.61 GB | Fused GEGLU MoE kernel |

**Key Insight**: Bastile excels at models with custom activations like GPT-OSS's GEGLU MoE. For 20%+ speedup on Qwen3, consider [Liger Kernel](https://github.com/linkedin/Liger-Kernel) which uses optimized Triton kernels.

## Installation

```bash
pip install bastile
# or
uv add bastile
```

**Requirements**: PyTorch nightly with CUDA 13.0+ support for CuTile.

## Quick Start

### GPT-OSS 20B (Recommended - 16% speedup)

```python
import bastile

# Apply patches BEFORE loading model
bastile.apply()

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b")

# Train as usual - Bastile automatically uses optimized GEGLU MoE kernel
```

### Qwen3

```python
import bastile

bastile.apply()

from transformers import Qwen3ForCausalLM
model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
```

## Optimizations

### GPT-OSS (CuTile Kernels)

| Operation | Speedup | Description |
|-----------|---------|-------------|
| **Fused GEGLU MoE** | +16% | Custom activation: `(up + 1) * gate * sigmoid(gate * 1.702)` |
| RMSNorm | ~0% | RSTD caching for backward pass |

### Qwen3

| Operation | Status | Notes |
|-----------|--------|-------|
| RMSNorm | Disabled by default | PyTorch is faster for small models |
| SwiGLU | Disabled by default | PyTorch is faster for small models |

## API

```python
import bastile

# Apply effective patches (moe by default for GPT-OSS)
# This automatically warms up kernels to avoid JIT overhead during training
bastile.apply()

# Selective patching
bastile.apply(
    rms_norm=False,  # Disabled - PyTorch is faster
    swiglu=False,    # Disabled - PyTorch is faster
    rope=False,      # Disabled - similar performance
    moe=True,        # Enabled - +16% for GPT-OSS
)

# Manual warmup (if needed before training loop)
bastile.warmup_all_kernels()

# Reset all patches
bastile.reset()

# Clear autotuning cache (for re-tuning)
bastile.clear_autotune_cache()

# Get applied patches
bastile.get_patched_ops()
```

## Autotuning

Bastile includes automatic kernel autotuning:

1. **First run**: Kernels are JIT-compiled and optimal configurations are cached
2. **Subsequent runs**: Cached configurations are used for faster execution
3. **Warmup**: Call `bastile.warmup_all_kernels()` to pre-compile kernels before training

## Benchmarks

```bash
# GPT-OSS benchmark (16% speedup)
uv run python -m tests.benchmarks.benchmark_gpt_oss

# Qwen3 comparison (HF vs Liger vs Bastile)
uv run python -m tests.benchmarks.benchmark_qwen3_comparison

# Run ops unit tests
uv run python -m tests.ops.run_all
```

## Why GPT-OSS?

GPT-OSS uses a **custom GEGLU activation** not found in standard libraries:

```python
# Standard SwiGLU: silu(gate) * up
# GPT-OSS GEGLU: (up + 1) * gate * sigmoid(gate * 1.702)
```

Bastile provides a fused CuTile kernel for this activation, giving significant speedup.

## Comparison with Liger Kernel

| Feature | Bastile | Liger Kernel |
|---------|---------|--------------|
| Backend | CuTile | Triton |
| Qwen3 speedup | ~0% | +20% |
| GPT-OSS speedup | +16% | N/A |

For maximum Qwen3 performance, use Liger Kernel. For GPT-OSS with custom activations, use Bastile.

## License

MIT
