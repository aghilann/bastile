"""Benchmark chunked backward kernel with configurable CHUNK_N and tile_m.

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m tests.experiments.bench_bwd_chunked --chunk_n 1024 --tile_m 2
  CUDA_VISIBLE_DEVICES=1 python -m tests.experiments.bench_bwd_chunked --chunk_n 512 --tile_m 2
  CUDA_VISIBLE_DEVICES=2 python -m tests.experiments.bench_bwd_chunked --chunk_n 1024 --tile_m 4
"""

import argparse
import torch
import cuda.tile as ct

from bastile.ops.rms_norm_cutile import (
    rms_norm_forward, rms_norm_bwd_persistent,
    warmup_rms_norm, _bwd_grid_size, _bwd_tile_m,
)
from bastile.ops.utils import next_power_of_2


def correctness_check(M, N, tile_m, chunk_n, eps=1e-6):
    """Verify chunked backward matches PyTorch reference."""
    torch.manual_seed(42)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(N, device="cuda", dtype=torch.bfloat16)

    # PyTorch reference
    x_pt = x.clone().requires_grad_(True)
    w_pt = w.clone().requires_grad_(True)
    x_f = x_pt.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    y_ref = (x_f * torch.rsqrt(var + eps)).to(torch.bfloat16) * w_pt
    y_ref.sum().backward()
    dx_ref = x_pt.grad.clone()
    dw_ref = w_pt.grad.clone()

    # CuTile forward (to get rstd)
    y, rstd = rms_norm_forward(x, w, eps)
    dy = torch.ones_like(y)

    # CuTile chunked backward
    tile_n = next_power_of_2(N)
    grid_size = _bwd_grid_size(M, tile_m)
    dx = torch.empty_like(x)
    dw_partial = torch.zeros((grid_size, tile_n), device="cuda", dtype=torch.float32)

    stream = torch.cuda.current_stream()
    if chunk_n > 0:
        ct.launch(stream, (grid_size,), rms_norm_bwd_chunked,
                  (dx, dy, x, w, rstd, dw_partial, tile_m, tile_n, chunk_n))
    else:
        ct.launch(stream, (grid_size,), rms_norm_bwd_persistent,
                  (dx, dy, x, w, rstd, dw_partial, tile_m, tile_n))
    torch.cuda.synchronize()

    dw = dw_partial[:, :N].sum(dim=0).to(w.dtype)

    # Check dx
    dx_err = (dx.float() - dx_ref.float()).abs().max().item()
    dx_rel = dx_err / (dx_ref.float().abs().max().item() + 1e-10)

    # Check dw
    dw_err = (dw.float() - dw_ref.float()).abs().max().item()
    dw_rel = dw_err / (dw_ref.float().abs().max().item() + 1e-10)

    ok = dx_rel < 0.02 and dw_rel < 0.02
    return ok, dx_rel, dw_rel


def benchmark_kernel(M, N, tile_m, chunk_n, warmup=20, iters=100):
    """Benchmark backward kernel time (GPU-only via CUDA events)."""
    eps = 1e-6
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(N, device="cuda", dtype=torch.bfloat16)
    y, rstd = rms_norm_forward(x, w, eps)
    dy = torch.randn_like(y)

    tile_n = next_power_of_2(N)
    grid_size = _bwd_grid_size(M, tile_m)
    dx = torch.empty_like(x)
    dw_partial = torch.empty((grid_size, tile_n), device="cuda", dtype=torch.float32)
    stream = torch.cuda.current_stream()

    use_chunked = chunk_n > 0

    # Warmup
    for _ in range(warmup):
        if use_chunked:
            dw_partial.zero_()
            ct.launch(stream, (grid_size,), rms_norm_bwd_chunked,
                      (dx, dy, x, w, rstd, dw_partial, tile_m, tile_n, chunk_n))
        else:
            ct.launch(stream, (grid_size,), rms_norm_bwd_persistent,
                      (dx, dy, x, w, rstd, dw_partial, tile_m, tile_n))
    torch.cuda.synchronize()

    # Benchmark kernel only (no dw_partial.sum)
    kernel_times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if use_chunked:
            dw_partial.zero_()
            ct.launch(stream, (grid_size,), rms_norm_bwd_chunked,
                      (dx, dy, x, w, rstd, dw_partial, tile_m, tile_n, chunk_n))
        else:
            ct.launch(stream, (grid_size,), rms_norm_bwd_persistent,
                      (dx, dy, x, w, rstd, dw_partial, tile_m, tile_n))
        end.record()
        torch.cuda.synchronize()
        kernel_times.append(start.elapsed_time(end) * 1000)  # µs

    kernel_times.sort()
    kernel_us = kernel_times[len(kernel_times) // 2]

    # Benchmark dw_partial.sum separately
    dw_times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        dw = dw_partial[:, :N].sum(dim=0).to(w.dtype)
        end.record()
        torch.cuda.synchronize()
        dw_times.append(start.elapsed_time(end) * 1000)
    dw_times.sort()
    dw_us = dw_times[len(dw_times) // 2]

    # Benchmark full backward (kernel + dw_sum)
    full_times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if use_chunked:
            dw_partial.zero_()
            ct.launch(stream, (grid_size,), rms_norm_bwd_chunked,
                      (dx, dy, x, w, rstd, dw_partial, tile_m, tile_n, chunk_n))
        else:
            ct.launch(stream, (grid_size,), rms_norm_bwd_persistent,
                      (dx, dy, x, w, rstd, dw_partial, tile_m, tile_n))
        dw = dw_partial[:, :N].sum(dim=0).to(w.dtype)
        end.record()
        torch.cuda.synchronize()
        full_times.append(start.elapsed_time(end) * 1000)
    full_times.sort()
    full_us = full_times[len(full_times) // 2]

    # PyTorch reference timing
    pt_mod = torch.nn.RMSNorm(N, eps=eps).to(dtype=torch.bfloat16, device="cuda")
    for _ in range(warmup):
        x_pt = torch.randn(M, N, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        y_pt = pt_mod(x_pt)
        y_pt.sum().backward()
    torch.cuda.synchronize()

    pt_times = []
    for _ in range(iters):
        x_pt = torch.randn(M, N, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        y_pt = pt_mod(x_pt)
        y_pt.sum().backward()
        end.record()
        torch.cuda.synchronize()
        pt_times.append(start.elapsed_time(end) * 1000)
    pt_times.sort()
    pt_fwdbwd_us = pt_times[len(pt_times) // 2]

    return {
        "kernel_us": kernel_us,
        "dw_sum_us": dw_us,
        "full_bwd_us": full_us,
        "pt_fwdbwd_us": pt_fwdbwd_us,
        "grid_size": grid_size,
        "tile_m": tile_m,
        "tile_n": tile_n,
        "chunk_n": chunk_n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_n", type=int, default=1024)
    parser.add_argument("--tile_m", type=int, default=0, help="0 = auto")
    parser.add_argument("--shapes", type=str, default="2048x4096,8192x4096,256x4096,2048x2048,2048x8192",
                        help="Comma-separated MxN shapes")
    args = parser.parse_args()

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Config: chunk_n={args.chunk_n}, tile_m={'auto' if args.tile_m == 0 else args.tile_m}")
    print()

    # Warmup all kernel variants
    print("Warming up kernels...")
    for hs in [2048, 4096, 8192]:
        warmup_rms_norm(hs)
    torch.cuda.synchronize()
    print("Done.\n")

    shapes = []
    for s in args.shapes.split(","):
        m, n = s.strip().split("x")
        shapes.append((int(m), int(n)))

    print(f"{'Shape':<16} {'Correct':>8} {'Kernel(µs)':>11} {'dw_sum(µs)':>11} "
          f"{'Full_bwd(µs)':>13} {'PT_fwdbwd(µs)':>14} {'vs_PT':>8} {'Grid':>5} {'Config':>20}")
    print("-" * 120)

    for M, N in shapes:
        tile_n = next_power_of_2(N)
        tile_m = args.tile_m if args.tile_m > 0 else _bwd_tile_m(M, N)
        chunk_n = args.chunk_n if tile_n > 1024 else 0

        # Correctness
        ok, dx_rel, dw_rel = correctness_check(M, N, tile_m, chunk_n)
        status = "PASS" if ok else f"FAIL(dx={dx_rel:.4f},dw={dw_rel:.4f})"

        if not ok:
            print(f"M={M:<6d} N={N:<5d} {status:>8}")
            continue

        # Benchmark
        r = benchmark_kernel(M, N, tile_m, chunk_n)
        speedup = r["pt_fwdbwd_us"] / r["full_bwd_us"] if r["full_bwd_us"] > 0 else 0
        config_str = f"tm={tile_m},tn={tile_n},cn={chunk_n}"

        print(f"M={M:<6d} N={N:<5d} {status:>8} {r['kernel_us']:>9.1f} {r['dw_sum_us']:>9.1f} "
              f"{r['full_bwd_us']:>11.1f} {r['pt_fwdbwd_us']:>12.1f} {speedup:>7.2f}x "
              f"{r['grid_size']:>5d} {config_str:>20}")


if __name__ == "__main__":
    main()
