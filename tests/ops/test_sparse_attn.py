"""CuTile sparse attention parity tests."""

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _eager_sparse_attn(q, kv_states, attn_sink, topk_idxs, softmax_scale):
    """Eager reference: gather KV @ topk_idxs, online softmax with attn_sink."""
    B, S, H, D = q.shape
    topk = topk_idxs.shape[-1]

    # Pre-gather KV
    kv_gathered = kv_states[
        torch.arange(B, device=kv_states.device)[:, None, None],
        topk_idxs.long(),
        :,
    ]  # [B, S, topk, D]

    # Compute per (batch, seq) — Q @ K^T, scale, softmax with attn_sink
    q_f32 = q.float()
    kv_f32 = kv_gathered.float()
    scores = q_f32 @ kv_f32.transpose(-2, -1) * softmax_scale  # [B, S, H, topk]

    m = scores.amax(dim=-1, keepdim=True)  # [B, S, H, 1]
    exp_scores = (scores - m).exp()  # [B, S, H, topk]
    l = exp_scores.sum(dim=-1, keepdim=True) + (attn_sink.float()[None, None, :, None] - m).exp()
    o = (exp_scores @ kv_f32) / l  # [B, S, H, D]
    return o.to(q.dtype)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 4, 8, 64, 4),        # tiny: B=1, S=4, H=8, D=64, topk=4
        (2, 16, 16, 128, 8),     # small: B=2, S=16, H=16, D=128, topk=8
        (4, 32, 32, 128, 32),    # medium: B=4, S=32, H=32, D=128, topk=32
        (1, 8, 12, 128, 16),     # odd heads: H=12 (padded to 16)
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_forward_matches_eager(shape, dtype):
    from bastile.ops.sparse_attn import sparse_attn

    B, S, H, D, topk = shape
    N = S * 2  # KV cache larger than seq_len (simulates real use)

    torch.manual_seed(0)
    device = torch.device("cuda")

    q = torch.randn(B, S, H, D, device=device, dtype=dtype) * 0.02
    kv_states = torch.randn(B, N, D, device=device, dtype=dtype) * 0.02
    attn_sink = torch.randn(H, device=device, dtype=torch.float32) * 0.02
    topk_idxs = torch.randint(0, N, (B, S, topk), device=device, dtype=torch.int32)

    softmax_scale = (1.0 / D) ** 0.5

    expected = _eager_sparse_attn(q, kv_states, attn_sink, topk_idxs, softmax_scale)
    got = sparse_attn(q, kv_states, attn_sink, topk_idxs, softmax_scale)

    if dtype == torch.float32:
        atol, rtol = 5e-2, 5e-2
    elif dtype == torch.float16:
        atol, rtol = 1e-1, 1e-1
    else:
        atol, rtol = 5e-2, 5e-2

    torch.testing.assert_close(got, expected, rtol=rtol, atol=atol,
                               msg=f"Mismatch at shape={shape}, dtype={dtype}")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_padding_stripped_correctly(dtype):
    """Heads padded to multiple of 16 are stripped back correctly."""
    from bastile.ops.sparse_attn import sparse_attn

    B, S, H, D, topk = 1, 4, 7, 64, 4  # H=7, pads to 16
    N = S * 2

    torch.manual_seed(1)
    device = torch.device("cuda")

    q = torch.randn(B, S, H, D, device=device, dtype=dtype) * 0.02
    kv_states = torch.randn(B, N, D, device=device, dtype=dtype) * 0.02
    attn_sink = torch.randn(H, device=device, dtype=torch.float32) * 0.02
    topk_idxs = torch.randint(0, N, (B, S, topk), device=device, dtype=torch.int32)

    softmax_scale = (1.0 / D) ** 0.5

    expected = _eager_sparse_attn(q, kv_states, attn_sink, topk_idxs, softmax_scale)
    got = sparse_attn(q, kv_states, attn_sink, topk_idxs, softmax_scale)

    # Output should have original head count
    assert got.shape == (B, S, H, D), f"Expected shape {(B, S, H, D)}, got {tuple(got.shape)}"

    if dtype == torch.float32:
        atol, rtol = 5e-2, 5e-2
    else:
        atol, rtol = 5e-2, 5e-2

    torch.testing.assert_close(got, expected, rtol=rtol, atol=atol)


def run_all():
    for shape in [(1, 4, 8, 64, 4), (2, 16, 16, 128, 8), (4, 32, 32, 128, 32), (1, 8, 12, 128, 16)]:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            test_forward_matches_eager(shape, dtype)
    for dtype in [torch.bfloat16, torch.float32]:
        test_padding_stripped_correctly(dtype)
    print("CuTile sparse attention parity")


if __name__ == "__main__":
    run_all()