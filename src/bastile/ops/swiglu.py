"""
SwiGLU MLP with CuTile kernel - supports forward and backward passes.

SwiGLU: c = silu(a) * b where silu(x) = x * sigmoid(x)

Backward:
- da = dc * b * sigmoid(a) * (1 + a * (1 - sigmoid(a)))
- db = dc * silu(a)
"""

import torch
import torch.nn as nn
import cuda.tile as ct

from ..registry import register_patch


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


# ============================================================================
# CuTile Kernels
# ============================================================================

@ct.kernel
def swiglu_forward_kernel(a, b, c, TILE_SIZE: ct.Constant[int]):
    """Forward: c = silu(a) * b"""
    row = ct.bid(0)
    col = ct.bid(1)

    a_tile = ct.load(a, index=(row, col), shape=(1, TILE_SIZE), padding_mode=ct.PaddingMode.ZERO)
    b_tile = ct.load(b, index=(row, col), shape=(1, TILE_SIZE), padding_mode=ct.PaddingMode.ZERO)

    # silu(a) = a * sigmoid(a)
    a_f32 = ct.astype(a_tile, ct.float32)
    sigmoid_a = 1.0 / (1.0 + ct.exp(-a_f32))
    silu_a = a_f32 * sigmoid_a
    
    c_tile = ct.astype(silu_a, a.dtype) * b_tile
    ct.store(c, index=(row, col), tile=c_tile)


@ct.kernel
def swiglu_backward_kernel(
    dc, a, b, da, db,
    TILE_SIZE: ct.Constant[int]
):
    """
    Backward pass for SwiGLU: c = silu(a) * b
    
    da = dc * b * sigmoid(a) * (1 + a - a * sigmoid(a))
    db = dc * silu(a)
    """
    row = ct.bid(0)
    col = ct.bid(1)

    dc_tile = ct.load(dc, index=(row, col), shape=(1, TILE_SIZE), padding_mode=ct.PaddingMode.ZERO)
    a_tile = ct.load(a, index=(row, col), shape=(1, TILE_SIZE), padding_mode=ct.PaddingMode.ZERO)
    b_tile = ct.load(b, index=(row, col), shape=(1, TILE_SIZE), padding_mode=ct.PaddingMode.ZERO)

    # Convert to float32 for computation
    dc_f32 = ct.astype(dc_tile, ct.float32)
    a_f32 = ct.astype(a_tile, ct.float32)
    b_f32 = ct.astype(b_tile, ct.float32)

    # Compute sigmoid(a)
    sigmoid_a = 1.0 / (1.0 + ct.exp(-a_f32))
    
    # silu(a) = a * sigmoid(a)
    silu_a = a_f32 * sigmoid_a
    
    # dsigmoid/da = sigmoid(a) * (1 - sigmoid(a))
    # dsilu/da = sigmoid(a) + a * sigmoid(a) * (1 - sigmoid(a))
    #          = sigmoid(a) * (1 + a * (1 - sigmoid(a)))
    #          = sigmoid(a) * (1 + a - a * sigmoid(a))
    dsilu_da = sigmoid_a * (1.0 + a_f32 - a_f32 * sigmoid_a)
    
    # da = dc * b * dsilu/da
    da_tile = dc_f32 * b_f32 * dsilu_da
    
    # db = dc * silu(a)
    db_tile = dc_f32 * silu_a

    ct.store(da, index=(row, col), tile=ct.astype(da_tile, a.dtype))
    ct.store(db, index=(row, col), tile=ct.astype(db_tile, b.dtype))


def ceildiv(a, b):
    return -(a // -b)


def swiglu_forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Forward pass: c = silu(a) * b"""
    ori_shape = a.shape
    n_cols = ori_shape[-1]
    a = a.reshape(-1, n_cols).contiguous()
    b = b.reshape(-1, n_cols).contiguous()
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    TILE_N = ceildiv(NUM_SMS, max(n_rows, 1))
    TILE_SIZE = next_power_of_2(int(n_cols / max(TILE_N, 1)))
    TILE_SIZE = max(TILE_SIZE, 64)  # Minimum tile size
    
    grid = (n_rows, ceildiv(n_cols, TILE_SIZE), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        swiglu_forward_kernel,
        (a, b, c, TILE_SIZE),
    )
    return c.view(*ori_shape)


def swiglu_backward(
    dc: torch.Tensor, 
    a: torch.Tensor, 
    b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward pass for SwiGLU."""
    ori_shape = a.shape
    n_cols = ori_shape[-1]
    
    dc = dc.reshape(-1, n_cols).contiguous()
    a = a.reshape(-1, n_cols).contiguous()
    b = b.reshape(-1, n_cols).contiguous()
    
    da = torch.empty_like(a)
    db = torch.empty_like(b)
    n_rows = a.shape[0]

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    TILE_N = ceildiv(NUM_SMS, max(n_rows, 1))
    TILE_SIZE = next_power_of_2(int(n_cols / max(TILE_N, 1)))
    TILE_SIZE = max(TILE_SIZE, 64)
    
    grid = (n_rows, ceildiv(n_cols, TILE_SIZE), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        swiglu_backward_kernel,
        (dc, a, b, da, db, TILE_SIZE),
    )
    return da.view(*ori_shape), db.view(*ori_shape)


class SwiGLUFunction(torch.autograd.Function):
    """SwiGLU with CuTile forward and backward."""
    
    @staticmethod
    def forward(ctx, a, b):
        c = swiglu_forward(a, b)
        ctx.save_for_backward(a, b)
        return c
    
    @staticmethod
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        da, db = swiglu_backward(dc, a, b)
        return da, db


class BastileSwiGLUMLP(nn.Module):
    """Drop-in replacement for Qwen3MLP using CuTile SwiGLU."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        if hasattr(config, 'hidden_act') and config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")
    
    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        activated = SwiGLUFunction.apply(gate, up)
        return self.down_proj(activated)


# Standalone function for testing
def swiglu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute silu(a) * b with gradient support."""
    return SwiGLUFunction.apply(a, b)


# ============================================================================
# Register patches for Qwen3
# ============================================================================

register_patch(
    name="swiglu_qwen3",
    description="CuTile SwiGLU MLP for Qwen3 models",
    target_module="transformers.models.qwen3.modeling_qwen3",
    target_attr="Qwen3MLP",
    replacement=BastileSwiGLUMLP,
    has_backward=True,
    priority=10,
    models=["qwen3"],
)

# Note: GPT-OSS uses MoE with a custom GEGLU activation (gate * sigmoid(gate * alpha))
# which is different from standard SwiGLU. We don't patch it.
