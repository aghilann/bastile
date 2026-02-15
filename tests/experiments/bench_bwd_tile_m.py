"""Quick benchmark: backward kernel with different tile_m for M=2048."""
import torch
import cuda.tile as ct
from bastile.ops.rms_norm_cutile import (
    rms_norm_forward, rms_norm_bwd_persistent,
    _bwd_grid_size, warmup_rms_norm,
)
from bastile.ops.utils import next_power_of_2


def bench_bwd_kernel(M, N, tile_m, warmup=30, iters=200):
    """Benchmark backward kernel with a specific tile_m."""
    eps = 1e-6
    dtype = torch.bfloat16
    TILE_N = next_power_of_2(N)

    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.ones(N, device="cuda", dtype=dtype)
    y, rstd = rms_norm_forward(x, w, eps)
    dy = torch.randn_like(y)

    x_2d = x.reshape(-1, N)
    dy_2d = dy.reshape(-1, N)
    grid_size = _bwd_grid_size(M, tile_m)
    dx = torch.empty_like(x_2d)
    dw_partial = torch.empty((grid_size, TILE_N), device="cuda", dtype=torch.float32)
    stream = torch.cuda.current_stream()

    # Warmup
    for _ in range(warmup):
        ct.launch(stream, (grid_size,), rms_norm_bwd_persistent,
                  (dx, dy_2d, x_2d, w, rstd, dw_partial, tile_m, TILE_N))
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        ct.launch(stream, (grid_size,), rms_norm_bwd_persistent,
                  (dx, dy_2d, x_2d, w, rstd, dw_partial, tile_m, TILE_N))
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000)
    times.sort()
    return times[len(times) // 2]


def main():
    print("Backward kernel tile_m benchmark\n")
    warmup_rms_norm(2048)
    warmup_rms_norm(4096)
    warmup_rms_norm(8192)
    torch.cuda.synchronize()

    shapes = [(2048, 2048), (2048, 4096), (2048, 8192), (8192, 4096)]

    for M, N in shapes:
        print(f"M={M}, N={N}, TILE_N={next_power_of_2(N)}:")
        for tm in [1, 2, 4, 8]:
            TILE_N = next_power_of_2(N)
            # Check if tile would be too large (max ~256KB)
            tile_bytes = tm * TILE_N * 4 * 4  # f32 × ~4 tiles in registers
            if tile_bytes > 256 * 1024:
                print(f"  tile_m={tm}: SKIP (tile too large: {tile_bytes//1024}KB)")
                continue
            try:
                us = bench_bwd_kernel(M, N, tm, warmup=20, iters=100)
                grid = _bwd_grid_size(M, tm)
                print(f"  tile_m={tm}: {us:>7.1f}µs  grid={grid}")
            except Exception as e:
                print(f"  tile_m={tm}: FAILED ({e})")
        print()


if __name__ == "__main__":
    main()
