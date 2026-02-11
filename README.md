# Bastile

Drop-in monkey-patch that replaces HuggingFace Qwen3 ops with optimized **CuTile** and **cuteDSL** kernels for training on NVIDIA Blackwell GPUs.

> **Requires**: NVIDIA Blackwell (B200 / B100) + CUDA Toolkit 13.1+

## Benchmarks

Qwen3-8B (36 layers, 4096 hidden, 32 heads) — single B200, batch_size=1, bf16, AdamW:

### Throughput (tokens/sec)

![Throughput](assets/bench_8b_throughput.png)

### Peak GPU Memory (GB)

![Memory](assets/bench_8b_memory.png)

### Latency (ms/iter)

![Latency](assets/bench_8b_latency.png)

Bastile's fused linear cross-entropy avoids materializing the full `[batch * seq_len, vocab_size]` logits tensor, which is the dominant memory cost at longer sequences. This is where the memory savings and throughput gains compound.

## Installation

```bash
pip install bastile
```

**Prerequisites:**
- NVIDIA Blackwell GPU (B200, B100, GB200)
- CUDA Toolkit **13.1+**
- PyTorch 2.4+ with CUDA support

```bash
# Inside a CUDA 13.1 container (e.g. baseten/gpu-dev:v8-cu13_1):
pip install bastile
```

## Quick Start

```python
import bastile

# Apply all patches BEFORE loading / creating the model
bastile.apply()

from transformers import Qwen3ForCausalLM

model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
model.train()

# Train as usual — Bastile automatically uses optimized kernels
```

Selective patching:

```python
bastile.apply(
    rms_norm=True,                   # cuteDSL RMSNorm (via quack)
    swiglu=True,                     # CuTile SwiGLU MLP
    rope=True,                       # CuTile RoPE with autotuning
    fused_linear_cross_entropy=True,  # Fused linear + CE (via quack)
)
```

Reset to original HuggingFace implementations:

```python
bastile.reset()
```

## Kernel Implementations

Bastile patches 4 operations in `transformers.models.qwen3.modeling_qwen3`. Each has a full forward **and** backward pass:

| Operation | Backend | Source | What it replaces |
|---|---|---|---|
| **RMSNorm** | cuteDSL (via [quack](https://github.com/pytorch-labs/quack)) | [`ops/rms_norm.py`](src/bastile/ops/rms_norm.py) | `Qwen3RMSNorm` |
| **SwiGLU MLP** | CuTile (`cuda.tile`) | [`ops/swiglu.py`](src/bastile/ops/swiglu.py) | `Qwen3MLP` |
| **RoPE** | CuTile (`cuda.tile`) | [`ops/rope.py`](src/bastile/ops/rope.py) | `apply_rotary_pos_emb` |
| **Fused Linear Cross-Entropy** | cuteDSL (via [quack](https://github.com/pytorch-labs/quack)) | [`ops/fused_linear_cross_entropy.py`](src/bastile/ops/fused_linear_cross_entropy.py) | `Qwen3ForCausalLM.forward` |

### RMSNorm — cuteDSL

Wraps quack's compiled cuteDSL kernels with reduced CPU dispatch overhead. Bypasses `torch.library.custom_op` dispatch by directly invoking the compiled kernel from a lookup cache, and caches SM counts to avoid repeated queries.

```
src/bastile/ops/rms_norm.py → patches Qwen3RMSNorm
```

### SwiGLU MLP — CuTile

Native CuTile kernels using `cuda.tile` with gather/scatter memory access. Uses `flush_to_zero` and approximate reciprocal for fast sigmoid on Blackwell. Full backward with recomputation (no extra activation memory).

```
src/bastile/ops/swiglu.py → patches Qwen3MLP
```

### RoPE — CuTile

CuTile rotary position embedding with occupancy-based autotuning (tests occupancy 1, 2, 4, 8 and caches the best). In-place rotation on reshaped tensors to minimize memory traffic.

```
src/bastile/ops/rope.py → patches apply_rotary_pos_emb
```

### Fused Linear Cross-Entropy — cuteDSL

Replaces the standard `lm_head(hidden_states) → logits → cross_entropy(logits, labels)` pipeline with quack's `chunked_linear_cross_entropy`. This **never materializes the full logits tensor** (`[batch * seq, 151936]` for Qwen3), instead computing cross-entropy in chunks of 4096. This is the single biggest memory saver at long sequence lengths.

```
src/bastile/ops/fused_linear_cross_entropy.py → patches Qwen3ForCausalLM.forward
```

## API Reference

```python
import bastile

bastile.apply()                  # Patch all ops
bastile.apply(rope=False)        # Patch everything except RoPE
bastile.reset()                  # Restore original implementations
bastile.get_patched_ops()        # List currently active patches
bastile.warmup_all_kernels()     # Pre-compile kernels (avoids JIT lag)
bastile.clear_autotune_cache()   # Re-run autotuning on next call
```

## Running Benchmarks

```bash
# Small model comparison (HuggingFace vs Liger vs Bastile)
make bench-small

# Qwen3-8B sequence length sweep (parallel on 3 GPUs)
make bench-8b

# Qwen3-8B sweep (sequential, single GPU)
make bench-8b-seq

# Kernel profiling with torch.profiler
make bench-profile
```

## Why CuTile instead of Triton?

Bastile uses NVIDIA's [CuTile](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-tile) (`cuda.tile`) and [cuteDSL](https://github.com/NVIDIA/cutlass) instead of Triton. On Blackwell (sm_100), CuTile generates native PTX through NVIDIA's own compiler toolchain, while Triton's code generation for sm_100 is still maturing. In our benchmarks, Triton-based kernels (Liger) often underperform raw PyTorch on B200, whereas CuTile kernels consistently match or beat it.

## License

MIT
