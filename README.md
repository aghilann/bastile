# Bastile

Drop-in monkey-patch that replaces HuggingFace Qwen3 ops with optimized **CuTile** kernels for training on NVIDIA Blackwell GPUs.

> **Requires**: NVIDIA Blackwell (B200 / B100) + CUDA Toolkit 13.1+

## Benchmarks

Current published benchmark focus is Qwen3. Bastile's optimized training path and the benchmark claims below are based on the Qwen3 kernels shown here.

Qwen3-8B (36 layers, 4096 hidden, 32 heads) — single B200, batch_size=1, bf16, AdamW:

### Throughput (tokens/sec)

![Throughput](assets/bench_8b_throughput.png)

### Peak GPU Memory (GB)

![Memory](assets/bench_8b_memory.png)

### Latency (ms/iter)

![Latency](assets/bench_8b_latency.png)

Bastile's fused linear cross-entropy avoids materializing the full `[batch * seq_len, vocab_size]` logits tensor, which is the dominant memory cost at longer sequences. This is where the memory savings and throughput gains compound.

### Qwen3.5 Status

Qwen3.5 support is still a work in progress.

The checked-in Qwen3.5 benchmark plots are provisional and are generated from the older Qwen3 optimization path, not from new Qwen3.5-specific kernel work. Treat them as placeholder reference numbers only, not as finalized Qwen3.5 performance claims.

Qwen3.5-9B (32 layers, 4096 hidden, 16 heads) — single B200, batch_size=1, bf16, AdamW:

### Qwen3.5 Throughput (tokens/sec)

![Qwen3.5 Throughput](assets/bench_qwen3_5_9b_throughput.png)

### Qwen3.5 Peak GPU Memory (GB)

![Qwen3.5 Memory](assets/bench_qwen3_5_9b_memory.png)

### Qwen3.5 Latency (ms/iter)

![Qwen3.5 Latency](assets/bench_qwen3_5_9b_latency.png)

## Installation

```bash
pip install bastile
```

**Prerequisites:**
- NVIDIA Blackwell GPU (B200, B100, GB200 or RTX 50 Series Chip)
- CUDA Toolkit **13.1+**
- PyTorch 2.9+ with CUDA support

```bash
# Inside a CUDA 13.1 container:
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
    rms_norm=True,                   # CuTile RMSNorm
    swiglu=True,                     # CuTile SwiGLU MLP
    rope=True,                       # CuTile RoPE
    fused_linear_cross_entropy=True,  # CuTile fused linear + CE
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
| **RMSNorm** | CuTile | [`ops/rms_norm.py`](src/bastile/ops/rms_norm.py) | `Qwen3RMSNorm` |
| **SwiGLU MLP** | CuTile | [`ops/swiglu.py`](src/bastile/ops/swiglu.py) | `Qwen3MLP` |
| **RoPE** | CuTile | [`ops/rope.py`](src/bastile/ops/rope.py) | `apply_rotary_pos_emb` |
| **Fused Linear Cross-Entropy** | CuTile | [`ops/fused_linear_cross_entropy.py`](src/bastile/ops/fused_linear_cross_entropy.py) | `Qwen3ForCausalLM.forward` |

### RMSNorm — CuTile

Native CuTile RMSNorm with persistent forward and backward kernels. Uses gather/scatter memory access with SM-aware tile sizing.

```
src/bastile/ops/rms_norm.py → patches Qwen3RMSNorm
```

### SwiGLU MLP — CuTile

Native CuTile kernels using `cuda.tile` with gather/scatter memory access. Uses `flush_to_zero` and approximate reciprocal for fast sigmoid on Blackwell. Full backward with recomputation (no extra activation memory).

```
src/bastile/ops/swiglu.py → patches Qwen3MLP
```

### RoPE — CuTile

CuTile rotary position embedding with compiler-optimized occupancy. In-place rotation on reshaped tensors to minimize memory traffic. Backward reuses the forward kernel by negating sin (inverse rotation identity).

```
src/bastile/ops/rope.py → patches apply_rotary_pos_emb
```

### Fused Linear Cross-Entropy — CuTile

Replaces the standard `lm_head(hidden_states) → logits → cross_entropy(logits, labels)` pipeline with a chunked approach. This **never materializes the full logits tensor** (`[batch * seq, 151936]` for Qwen3), instead computing cross-entropy in chunks. This is the single biggest memory saver at long sequence lengths.

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
bastile.clear_autotune_cache()   # Clear kernel caches
```

## Running Benchmarks

```bash
# Qwen3-8B sweep
make bench-3-8b

# Qwen3-0.6B sweep
make bench-0.6b

# Qwen3.5-9B sweep (WIP, placeholder numbers currently reflect older Qwen3 optimizations)
make bench-3.5-9b

# Kernel micro-benchmarks
make bench-rmsnorm
make bench-lce
```

## Why CuTile instead of Triton?

Bastile uses NVIDIA's [CuTile](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-tile) (`cuda.tile`) instead of Triton. On Blackwell (sm_100), CuTile generates native PTX through NVIDIA's own compiler toolchain, while Triton's code generation for sm_100 is still maturing. In our benchmarks, Triton-based kernels (Liger) often underperform raw PyTorch on B200, whereas CuTile kernels consistently match or beat it.

## License

MIT
