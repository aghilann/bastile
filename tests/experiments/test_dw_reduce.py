"""Compare dw reduction output: PyTorch sum approach on partial sums from backward kernel."""
import torch
from bastile.ops.rms_norm_cutile import (
    rms_norm_forward, warmup_rms_norm,
    _bwd_tile_m, _bwd_grid_size, _run_bwd,
)
from bastile.ops.utils import next_power_of_2


def main():
    torch.manual_seed(42)
    eps = 1e-6

    for N in [2048, 4096, 8192]:
        warmup_rms_norm(N)

    print("Verifying dw reduction (PyTorch sum on partial sums):")
    header = f"{'Shape':<28} {'dw_max':>10}"
    print(header)
    print("-" * len(header))

    for M, N in [(256, 2048), (2048, 4096), (2048, 8192), (8192, 4096)]:
        for dtype in [torch.bfloat16, torch.float16]:
            TILE_N = next_power_of_2(N)
            x = torch.randn(M, N, device="cuda", dtype=dtype)
            w = torch.ones(N, device="cuda", dtype=dtype)
            dy = torch.randn(M, N, device="cuda", dtype=dtype)

            y, rstd = rms_norm_forward(x, w, eps)

            x_2d = x.reshape(-1, N)
            dy_2d = dy.reshape(-1, N)
            TILE_M = _bwd_tile_m(M, N)
            grid_size = _bwd_grid_size(M, TILE_M)

            dx = torch.empty_like(x_2d)
            dw_partial = torch.empty(
                (grid_size, TILE_N), device="cuda", dtype=torch.float32
            )
            _run_bwd(dx, dy_2d, x_2d, w, rstd, dw_partial, N)
            torch.cuda.synchronize()

            dw = dw_partial[:, :N].sum(dim=0).to(dtype)

            tag = f"M={M:<6d}N={N:<5d} {str(dtype):<14s}"
            print(f"{tag} {dw.abs().max().item():>10.5f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
