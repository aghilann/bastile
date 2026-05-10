"""Kernel microbench: CuTile sparse attention vs PyTorch eager.

Compares ``sparse_attn`` (pre-gather KV + online softmax CuTile kernel)
against the eager gather + matmul + softmax + weighted sum.
"""

from __future__ import annotations

import argparse
import gc

import torch

from bastile.ops.sparse_attn import sparse_attn
from tests.benchmarks.utils import (
    clear_cuda_state,
    print_gpu_info,
    print_header,
)


def _eager_sparse_attn(q, kv_states, attn_sink, topk_idxs, softmax_scale):
    B, S, H, D = q.shape
    kv_gathered = kv_states[
        torch.arange(B, device=kv_states.device)[:, None, None],
        topk_idxs.long(),
        :,
    ]
    q_f32 = q.float()
    kv_f32 = kv_gathered.float()
    scores = q_f32 @ kv_f32.transpose(-2, -1) * softmax_scale
    m = scores.amax(dim=-1, keepdim=True)
    exp_scores = (scores - m).exp()
    l = exp_scores.sum(dim=-1, keepdim=True) + (attn_sink.float()[None, None, :, None] - m).exp()
    o = (exp_scores @ kv_f32) / l
    return o.to(q.dtype)


def _bench(fn, q, kv, attn_sink, topk_idxs, scale, warmup_iters, timed_iters):
    for _ in range(warmup_iters):
        _ = fn(q, kv, attn_sink, topk_idxs, scale)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(timed_iters):
        _ = fn(q, kv, attn_sink, topk_idxs, scale)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / timed_iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=2, help="Batch size")
    parser.add_argument("--S", type=int, default=32, help="Sequence length")
    parser.add_argument("--H", type=int, default=16, help="Number of heads")
    parser.add_argument("--D", type=int, default=128, help="Head dimension")
    parser.add_argument("--topk", type=int, default=32, help="Top-k positions")
    parser.add_argument("--N", type=int, default=1024, help="KV cache size")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--timed-iters", type=int, default=50)
    args = parser.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    print_header("Sparse Attention kernel microbench", width=80)
    print_gpu_info()
    print(f"B={args.B}, S={args.S}, H={args.H}, D={args.D}, topk={args.topk}, N={args.N}, dtype={args.dtype}")

    torch.manual_seed(0)
    device = torch.device("cuda")

    q = torch.randn(args.B, args.S, args.H, args.D, device=device, dtype=dtype) * 0.02
    kv_states = torch.randn(args.B, args.N, args.D, device=device, dtype=dtype) * 0.02
    attn_sink = torch.randn(args.H, device=device, dtype=torch.float32) * 0.02
    topk_idxs = torch.randint(0, args.N, (args.B, args.S, args.topk), device=device, dtype=torch.int32)

    scale = (1.0 / args.D) ** 0.5

    # Correctness smoke-test
    ref = _eager_sparse_attn(q, kv_states, attn_sink, topk_idxs, scale)
    out = sparse_attn(q, kv_states, attn_sink, topk_idxs, scale)
    if dtype == torch.bfloat16:
        atol, rtol = 5e-2, 5e-2
    elif dtype == torch.float16:
        atol, rtol = 1e-2, 1e-2
    else:
        atol, rtol = 1e-3, 1e-3
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
    print("  Correctness smoke-test passed")

    pytorch_ms = _bench(_eager_sparse_attn, q, kv_states, attn_sink, topk_idxs, scale, args.warmup_iters, args.timed_iters)
    bastile_ms = _bench(sparse_attn, q, kv_states, attn_sink, topk_idxs, scale, args.warmup_iters, args.timed_iters)

    speedup = pytorch_ms / bastile_ms if bastile_ms > 0 else float("inf")

    print(f"\n  PyTorch eager : {pytorch_ms:.3f} ms/iter")
    print(f"  Bastile fused : {bastile_ms:.3f} ms/iter")
    print(f"  Speedup       : {speedup:.2f}x")

    del q, kv_states, attn_sink, topk_idxs
    clear_cuda_state()
    gc.collect()


if __name__ == "__main__":
    main()