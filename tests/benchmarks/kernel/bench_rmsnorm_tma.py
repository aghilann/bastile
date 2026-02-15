"""
RMSNorm TMA Benchmark: compare forward-pass performance with and without TMA stores.

Usage (inside the bastile-dev container):
    CUDA_VISIBLE_DEVICES=0 uv run python3 -m tests.benchmarks.kernel.bench_rmsnorm_tma
"""

import torch
import cuda.tile as ct

from bastile.ops.utils import next_power_of_2

# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------
ConstInt = ct.Constant[int]
ConstFloat = ct.Constant[float]
PAD_ZERO = ct.PaddingMode.ZERO


# ---- Kernel WITHOUT TMA (allow_tma=False on stores) ----------------------
@ct.kernel(occupancy=1)
def _rms_fwd_no_tma(X, W, Y, Rstd, TILE_M: ConstInt, TILE_N: ConstInt, eps: ConstFloat):
    bid = ct.bid(0)
    M, N = X.shape[0], X.shape[1]
    blocks = ct.num_blocks(0)
    upper = (M + TILE_M - 1) // TILE_M

    w = ct.astype(ct.load(W, index=(0,), shape=(TILE_N,), padding_mode=PAD_ZERO), ct.float32)
    w = ct.reshape(w, (1, TILE_N))
    rcp = ct.full((TILE_M, 1), 1.0 / N, dtype=ct.float32)
    e = ct.full((TILE_M, 1), eps, dtype=ct.float32)

    for i in range(bid, upper, blocks):
        x = ct.astype(
            ct.load(X, index=(i, 0), shape=(TILE_M, TILE_N), latency=10, padding_mode=PAD_ZERO),
            ct.float32,
        )
        r = ct.rsqrt(ct.sum(x * x, axis=1, keepdims=True) * rcp + e)
        ct.store(Rstd, index=(i,), tile=ct.reshape(r, (TILE_M,)), allow_tma=False)
        ct.store(Y, index=(i, 0), tile=ct.astype(x * r * w, X.dtype), allow_tma=False, latency=3)


# ---- Kernel WITH TMA (allow_tma=True on stores, default load behaviour) --
@ct.kernel(occupancy=1)
def _rms_fwd_tma(X, W, Y, Rstd, TILE_M: ConstInt, TILE_N: ConstInt, eps: ConstFloat):
    bid = ct.bid(0)
    M, N = X.shape[0], X.shape[1]
    blocks = ct.num_blocks(0)
    upper = (M + TILE_M - 1) // TILE_M

    w = ct.astype(ct.load(W, index=(0,), shape=(TILE_N,), padding_mode=PAD_ZERO), ct.float32)
    w = ct.reshape(w, (1, TILE_N))
    rcp = ct.full((TILE_M, 1), 1.0 / N, dtype=ct.float32)
    e = ct.full((TILE_M, 1), eps, dtype=ct.float32)

    for i in range(bid, upper, blocks):
        x = ct.astype(
            ct.load(X, index=(i, 0), shape=(TILE_M, TILE_N), latency=10, padding_mode=PAD_ZERO),
            ct.float32,
        )
        r = ct.rsqrt(ct.sum(x * x, axis=1, keepdims=True) * rcp + e)
        ct.store(Rstd, index=(i,), tile=ct.reshape(r, (TILE_M,)), allow_tma=True)
        ct.store(Y, index=(i, 0), tile=ct.astype(x * r * w, X.dtype), allow_tma=True, latency=3)


# ---------------------------------------------------------------------------
# Tile configuration (same heuristic as bastile.ops.rms_norm)
# ---------------------------------------------------------------------------
def _fwd_tiles(M, N):
    sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    T = next_power_of_2(N)
    if M <= sms * 4:
        return (1, T, min(sms, M))
    if T <= 1024:
        tm = 16
    elif T >= 16384:
        tm = 2
    else:
        tm = max(2, min(8, 32768 // T))
    return (tm, T, min(sms, (M + tm - 1) // tm))


# ---------------------------------------------------------------------------
# Launcher wrappers
# ---------------------------------------------------------------------------
def launch_no_tma(x2, weight, eps):
    M, N = x2.shape
    tm, tn, g = _fwd_tiles(M, N)
    y = torch.empty_like(x2)
    rstd = torch.empty(M, dtype=torch.float32, device=x2.device)
    ct.launch(torch.cuda.current_stream(), (g,), _rms_fwd_no_tma, (x2, weight, y, rstd, tm, tn, eps))
    return y


def launch_tma(x2, weight, eps):
    M, N = x2.shape
    tm, tn, g = _fwd_tiles(M, N)
    y = torch.empty_like(x2)
    rstd = torch.empty(M, dtype=torch.float32, device=x2.device)
    ct.launch(torch.cuda.current_stream(), (g,), _rms_fwd_tma, (x2, weight, y, rstd, tm, tn, eps))
    return y


# ---------------------------------------------------------------------------
# PyTorch reference
# ---------------------------------------------------------------------------
def pytorch_rms_norm(x, weight, eps):
    x_float = x.float()
    variance = x_float.pow(2).mean(-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(variance + eps)
    return (weight.float() * x_normed).to(x.dtype)


# ---------------------------------------------------------------------------
# Benchmarking helpers
# ---------------------------------------------------------------------------
def benchmark_fn(fn, warmup=50, iterations=200):
    """Return median latency in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # ms -> us

    times.sort()
    return times[len(times) // 2]


def compute_gbps(M, N, dtype, latency_us):
    """Memory throughput: read x + read w + write y."""
    elem = torch.tensor([], dtype=dtype).element_size()
    total_bytes = (M * N * elem) + (N * elem) + (M * N * elem)
    return total_bytes / (latency_us * 1e-6) / 1e9


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = "cuda"
    eps = 1e-6
    dtype = torch.bfloat16

    props = torch.cuda.get_device_properties(0)
    print("=" * 100)
    print(f"  RMSNorm TMA Benchmark  —  GPU: {props.name}  |  SMs: {props.multi_processor_count}")
    print("=" * 100)

    configs = [
        # (M,     N)
        (  256,  2048),
        ( 1024,  2048),
        ( 4096,  2048),
        (16384,  2048),
        (  256,  4096),
        ( 1024,  4096),
        ( 4096,  4096),
        (16384,  4096),
        (  256,  8192),
        ( 1024,  8192),
        ( 4096,  8192),
        (16384,  8192),
        (  256, 16384),
        ( 1024, 16384),
        ( 4096, 16384),
    ]

    # -- JIT warmup ----------------------------------------------------------
    print("\nJIT compiling kernels (warmup)...")
    for M, N in configs[:3]:
        x = torch.randn(M, N, dtype=dtype, device=device)
        w = torch.ones(N, dtype=dtype, device=device)
        launch_no_tma(x, w, eps)
        launch_tma(x, w, eps)
    torch.cuda.synchronize()
    print("Done.\n")

    # -- Header --------------------------------------------------------------
    hdr = (
        f"{'M':>6}  {'N':>6}  "
        f"{'PyTorch':>10}  {'No-TMA':>10}  {'TMA':>10}  "
        f"{'PT GB/s':>9}  {'NoTMA GB/s':>11}  {'TMA GB/s':>9}  "
        f"{'NoTMA vs PT':>12}  {'TMA vs PT':>10}  {'TMA vs NoTMA':>13}"
    )
    print(hdr)
    print("-" * len(hdr))

    # -- Run -----------------------------------------------------------------
    summary_no_tma_vs_pt = []
    summary_tma_vs_pt = []
    summary_tma_vs_no_tma = []

    for M, N in configs:
        x = torch.randn(M, N, dtype=dtype, device=device)
        w = torch.ones(N, dtype=dtype, device=device)

        # Correctness quick-check
        ref = pytorch_rms_norm(x, w, eps)
        out_no = launch_no_tma(x, w, eps)
        out_yes = launch_tma(x, w, eps)
        torch.testing.assert_close(out_no, ref, atol=5e-2, rtol=0)
        torch.testing.assert_close(out_yes, ref, atol=5e-2, rtol=0)

        pt_us = benchmark_fn(lambda: pytorch_rms_norm(x, w, eps))
        no_us = benchmark_fn(lambda: launch_no_tma(x, w, eps))
        tm_us = benchmark_fn(lambda: launch_tma(x, w, eps))

        pt_gbs = compute_gbps(M, N, dtype, pt_us)
        no_gbs = compute_gbps(M, N, dtype, no_us)
        tm_gbs = compute_gbps(M, N, dtype, tm_us)

        sp_no_pt = pt_us / no_us
        sp_tm_pt = pt_us / tm_us
        sp_tm_no = no_us / tm_us

        summary_no_tma_vs_pt.append(sp_no_pt)
        summary_tma_vs_pt.append(sp_tm_pt)
        summary_tma_vs_no_tma.append(sp_tm_no)

        print(
            f"{M:>6}  {N:>6}  "
            f"{pt_us:>8.1f}us  {no_us:>8.1f}us  {tm_us:>8.1f}us  "
            f"{pt_gbs:>8.1f}  {no_gbs:>10.1f}  {tm_gbs:>8.1f}  "
            f"{sp_no_pt:>11.2f}x  {sp_tm_pt:>9.2f}x  {sp_tm_no:>12.2f}x"
        )

    # -- Summary -------------------------------------------------------------
    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)

    def stats(vals, label):
        avg = sum(vals) / len(vals)
        print(f"  {label}:  avg {avg:.2f}x  |  min {min(vals):.2f}x  |  max {max(vals):.2f}x")

    stats(summary_no_tma_vs_pt, "No-TMA  vs PyTorch")
    stats(summary_tma_vs_pt,    "TMA     vs PyTorch")
    stats(summary_tma_vs_no_tma, "TMA     vs No-TMA ")
    print("=" * 100)


if __name__ == "__main__":
    main()
