# Bastile

**Monkey-patch PyTorch with optimized CuTile kernels for Qwen3 finetuning**

Bastile automatically patches Qwen3 transformer operations with optimized CUDA kernels that support both forward and backward passes, enabling faster finetuning.

## Installation

```bash
# Requires PyTorch nightly with CUDA 13.0 support
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130
uv pip install cuda-tile
uv pip install -e .
```

## Quick Start

```python
import bastile

# Apply all optimized kernels before loading your model
bastile.apply()

# Load and finetune your model as usual
from transformers import Qwen3ForCausalLM
model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

# Training works with full gradient support
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss = model(input_ids, labels=labels).loss
loss.backward()  # Gradients flow through optimized kernels
optimizer.step()
```

## Selective Patching

```python
import bastile

# Apply only specific optimizations
bastile.apply(
    rms_norm=True,   # CuTile RMSNorm (forward + backward)
    swiglu=True,     # CuTile SwiGLU MLP (forward + backward)
    rope=True,       # RoPE with autograd support
)

# Check what's patched
print(bastile.get_patched_ops())

# Reset all patches
bastile.reset()
```

## Supported Operations

| Operation | Forward | Backward | Notes |
|-----------|---------|----------|-------|
| RMSNorm | CuTile | CuTile | Full CuTile implementation |
| SwiGLU MLP | CuTile | CuTile | Custom backward kernel |
| RoPE | PyTorch | Autograd | HuggingFace compatible |

## Supported Models

- **Qwen3** (RMSNorm, SwiGLU, RoPE) - Dense models
- **GPT-OSS 20B** (RMSNorm, Fused GEGLU MoE Experts) - OpenAI's open-weight MoE model
  - Custom CuTile kernel for GPT-OSS's GEGLU activation: `gate * sigmoid(gate * 1.702)`
  - **11.8% speedup** over PyTorch native implementation

## API Reference

### `bastile.apply(**kwargs) -> List[str]`

Apply kernel patches. Returns list of applied patch names.

```python
applied = bastile.apply(rms_norm=True, swiglu=True, rope=True)
# Returns: ['rms_norm_qwen3', 'swiglu_qwen3', 'rope_qwen3']
```

### `bastile.reset(names=None) -> List[str]`

Reset patches to original implementations.

```python
bastile.reset()  # Reset all
bastile.reset(['rms_norm_qwen3'])  # Reset specific patches
```

### `bastile.get_patched_ops() -> List[str]`

Get list of currently patched operations.

## Using Individual Kernels

```python
from bastile.ops.rms_norm import BastileRMSNorm
from bastile.ops.swiglu import swiglu, BastileSwiGLUMLP

# RMSNorm with gradient support
norm = BastileRMSNorm(hidden_size=4096).cuda()
x = torch.randn(2, 128, 4096, device="cuda", requires_grad=True)
y = norm(x)
y.sum().backward()  # Gradients work!

# SwiGLU with gradient support
a = torch.randn(2, 128, 2048, device="cuda", requires_grad=True)
b = torch.randn(2, 128, 2048, device="cuda", requires_grad=True)
c = swiglu(a, b)  # silu(a) * b
c.sum().backward()  # da and db computed
```

## Testing

```bash
cd /workspace/bastile
.venv/bin/python tests/test_qwen3_finetune.py
```

## Requirements

- Python 3.12+
- PyTorch nightly (2.8.0+ with CUDA 12.8/13.0)
- cuda-tile >= 1.1.0
- transformers >= 4.40.0

## Architecture

```
bastile/
├── src/bastile/
│   ├── __init__.py      # Main API (apply, reset, etc.)
│   ├── core.py          # Patching logic
│   ├── registry.py      # Patch registry
│   └── ops/
│       ├── rms_norm.py  # CuTile RMSNorm (forward + backward)
│       ├── swiglu.py    # CuTile SwiGLU (forward + backward)
│       └── rope.py      # RoPE with autograd
└── tests/
    └── test_qwen3_finetune.py
```

## License

MIT
