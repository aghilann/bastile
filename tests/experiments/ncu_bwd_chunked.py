"""Profile chunked backward kernel with ncu to verify no register spilling.

Usage:
  CUDA_VISIBLE_DEVICES=N ncu --set detailed python3 -m tests.experiments.ncu_bwd_chunked [--chunk_n 1024]
"""

import argparse
import torch
import cuda.tile as ct

from bastile.ops.rms_norm_cutile import (
    rms_norm_forward, rms_norm_bwd_persistent,
    warmup_rms_norm, _bwd_grid_size, _bwd_tile_m,
)
from bastile.ops.utils import next_power_of_2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_n", type=int, default=1024)
    parser.add_argument("--tile_m", type=int, default=0)
    parser.add_argument("--M", type=int, default=2048)
    parser.add_argument("--N", type=int, default=4096)
    args = parser.parse_args()

    M, N = args.M, args.N
    tile_n = next_power_of_2(N)
    tile_m = args.tile_m if args.tile_m > 0 else _bwd_tile_m(M, N)
    chunk_n = args.chunk_n
    grid_size = _bwd_grid_size(M, tile_m)

    print(f"Profiling: M={M}, N={N}, tile_m={tile_m}, tile_n={tile_n}, chunk_n={chunk_n}, grid={grid_size}")

    # Warmup
    warmup_rms_norm(N)
    torch.cuda.synchronize()

    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(N, device="cuda", dtype=torch.bfloat16)
    y, rstd = rms_norm_forward(x, w, 1e-6)
    dy = torch.randn_like(y)
    dx = torch.empty_like(x)
    dw_partial = torch.zeros((grid_size, tile_n), device="cuda", dtype=torch.float32)
    stream = torch.cuda.current_stream()

    # Warmup chunked kernel
    for _ in range(5):
        dw_partial.zero_()
        ct.launch(stream, (grid_size,), rms_norm_bwd_chunked,
                  (dx, dy, x, w, rstd, dw_partial, tile_m, tile_n, chunk_n))
    torch.cuda.synchronize()

    # Profile region
    torch.cuda.nvtx.range_push(f"chunked_bwd_M{M}_N{N}_cn{chunk_n}")
    for _ in range(3):
        dw_partial.zero_()
        ct.launch(stream, (grid_size,), rms_norm_bwd_chunked,
                  (dx, dy, x, w, rstd, dw_partial, tile_m, tile_n, chunk_n))
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    # Also profile old kernel for comparison
    torch.cuda.nvtx.range_push(f"original_bwd_M{M}_N{N}")
    for _ in range(3):
        ct.launch(stream, (grid_size,), rms_norm_bwd_persistent,
                  (dx, dy, x, w, rstd, dw_partial, tile_m, tile_n))
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    print("Done.")


if __name__ == "__main__":
    main()
