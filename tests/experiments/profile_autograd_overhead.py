"""Profile where time is spent in the CuTile autograd fwd+bwd path.

Breaks down: tensor alloc, forward kernel, backward kernel, dw_sum, Python overhead.
"""

import torch
import time

from bastile.ops.rms_norm_cutile import (
    CuTileRMSNorm, CuTileRMSNormFunction, rms_norm_forward,
    _bwd_tile_m, _bwd_grid_size,
    rms_norm_bwd_persistent,
    warmup_rms_norm,
)
from bastile.ops.utils import next_power_of_2
import cuda.tile as ct


def profile_breakdown(M, N, warmup=20, iters=100):
    """Measure each component of the backward path separately."""
    eps = 1e-6
    dtype = torch.bfloat16

    # Pre-create test data
    x = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.ones(N, device="cuda", dtype=dtype)
    mod = CuTileRMSNorm(N, eps=eps).to("cuda")

    # Warm up all kernels and caches
    for _ in range(warmup):
        y = mod(x)
        y.sum().backward()
    torch.cuda.synchronize()

    # 1. Full fwd+bwd through autograd (what the benchmark measures)
    def run_full():
        xx = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=True)
        y = mod(xx)
        y.sum().backward()

    full_times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        run_full()
        e.record()
        torch.cuda.synchronize()
        full_times.append(s.elapsed_time(e) * 1000)
    full_times.sort()
    full_us = full_times[len(full_times) // 2]

    # 2. Forward kernel only (isolated)
    fwd_times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        y, rstd = rms_norm_forward(x.detach(), w, eps)
        e.record()
        torch.cuda.synchronize()
        fwd_times.append(s.elapsed_time(e) * 1000)
    fwd_times.sort()
    fwd_us = fwd_times[len(fwd_times) // 2]

    # 3. Backward kernel only (pre-allocated, no autograd)
    y, rstd = rms_norm_forward(x.detach(), w, eps)
    dy = torch.randn_like(y)
    tile_n = next_power_of_2(N)
    tile_m = _bwd_tile_m(M, N)
    grid_size = _bwd_grid_size(M, tile_m)
    dx = torch.empty_like(x.detach().reshape(-1, N))
    dw_partial = torch.empty((grid_size, tile_n), device="cuda", dtype=torch.float32)
    stream = torch.cuda.current_stream()

    kernel_times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        ct.launch(stream, (grid_size,), rms_norm_bwd_persistent,
                  (dx, dy, x.detach().reshape(-1, N), w, rstd, dw_partial, tile_m, tile_n))
        e.record()
        torch.cuda.synchronize()
        kernel_times.append(s.elapsed_time(e) * 1000)
    kernel_times.sort()
    kernel_us = kernel_times[len(kernel_times) // 2]

    # 4. dw_partial.sum() + cast (PyTorch)
    dw_sum_times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        dw = dw_partial[:, :N].sum(dim=0).to(w.dtype)
        e.record()
        torch.cuda.synchronize()
        dw_sum_times.append(s.elapsed_time(e) * 1000)
    dw_sum_times.sort()
    dw_sum_us = dw_sum_times[len(dw_sum_times) // 2]

    # 5. dx allocation
    alloc_times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        _dx = torch.empty(M, N, device="cuda", dtype=dtype)
        e.record()
        torch.cuda.synchronize()
        alloc_times.append(s.elapsed_time(e) * 1000)
    alloc_times.sort()
    alloc_us = alloc_times[len(alloc_times) // 2]

    # 6. .sum() for loss (y.sum() — part of the benchmark overhead)
    y_test = torch.randn(M, N, device="cuda", dtype=dtype)
    sum_times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        _ = y_test.sum()
        e.record()
        torch.cuda.synchronize()
        sum_times.append(s.elapsed_time(e) * 1000)
    sum_times.sort()
    sum_us = sum_times[len(sum_times) // 2]

    # PyTorch reference
    pt_mod = torch.nn.RMSNorm(N, eps=eps).to(dtype=dtype, device="cuda")
    for _ in range(warmup):
        xx = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=True)
        pt_mod(xx).sum().backward()
    torch.cuda.synchronize()

    pt_times = []
    for _ in range(iters):
        xx = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=True)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        pt_mod(xx).sum().backward()
        e.record()
        torch.cuda.synchronize()
        pt_times.append(s.elapsed_time(e) * 1000)
    pt_times.sort()
    pt_us = pt_times[len(pt_times) // 2]

    accounted = fwd_us + kernel_us + dw_sum_us + sum_us
    overhead = full_us - accounted

    return {
        "full_fwdbwd": full_us,
        "fwd_kernel": fwd_us,
        "bwd_kernel": kernel_us,
        "dw_sum": dw_sum_us,
        "dx_alloc": alloc_us,
        "y_sum": sum_us,
        "accounted": accounted,
        "overhead": overhead,
        "pytorch": pt_us,
        "grid": grid_size,
    }


def main():
    print("Profiling autograd overhead breakdown\n")

    print("Warming up...")
    for hs in [2048, 4096, 8192]:
        warmup_rms_norm(hs)
    torch.cuda.synchronize()
    print("Done.\n")

    shapes = [(256, 2048), (256, 4096), (2048, 2048), (2048, 4096), (2048, 8192), (8192, 4096), (16384, 4096)]

    print(f"{'Shape':<16} {'Full':>7} {'Fwd':>6} {'Bwd':>6} {'dwSum':>6} "
          f"{'Alloc':>6} {'ySum':>6} {'Accnt':>7} {'Ovrhd':>7} {'PT':>7} {'vs_PT':>7}")
    print("-" * 105)

    for M, N in shapes:
        r = profile_breakdown(M, N, warmup=30, iters=100)
        sp = r["pytorch"] / r["full_fwdbwd"]
        print(f"M={M:<6d}N={N:<5d} {r['full_fwdbwd']:>6.1f} {r['fwd_kernel']:>5.1f} "
              f"{r['bwd_kernel']:>5.1f} {r['dw_sum']:>5.1f} "
              f"{r['dx_alloc']:>5.1f} {r['y_sum']:>5.1f} {r['accounted']:>6.1f} "
              f"{r['overhead']:>6.1f} {r['pytorch']:>6.1f} {sp:>6.2f}x"
              f"  grid={r['grid']}")


if __name__ == "__main__":
    main()
