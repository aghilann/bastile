#!/usr/bin/env python3
"""
Benchmark: Fused Linear Cross-Entropy — Speed + Peak Memory
  Qwen3-8B: H=4096, V=151936
  BT from 256 to 32768 (simulating seq_len from 256 to 32K with batch=1)
"""

import gc
import torch
import torch.nn.functional as F

from ..utils import benchmark_fn as _benchmark_fn_us, clear_cuda_state

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

H, V = 4096, 151936
BT_sizes = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
device = "cuda"
dtype = torch.bfloat16


def benchmark_fn(fn, warmup=5, iters=20):
    """Benchmark returning milliseconds (trimmed mean)."""
    return _benchmark_fn_us(fn, warmup=warmup, iterations=iters) / 1000.0


def measure_peak_memory(fn, warmup=3):
    """Run fn and return peak GPU memory allocated (MB)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    clear_cuda_state()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


print(f"{'='*120}")
print(f"Fused Linear Cross-Entropy Benchmark — H={H}, V={V}")
print(f"Weight matrix: {V*H*2/1024**2:.0f} MB (bf16)")
print(f"{'='*120}\n")

weight = torch.randn(V, H, dtype=dtype, device=device)

# ── Warmup ──
print("Warming up...")
x_w = torch.randn(8, H, dtype=dtype, device=device, requires_grad=True)
F.cross_entropy(F.linear(x_w, weight), torch.randint(0, V, (8,), device=device)).backward()
torch.cuda.synchronize()
print("  PyTorch ✓")

from bastile.ops.fused_linear_cross_entropy import fused_linear_cross_entropy as bastile_lce, warmup_fused_lce
warmup_fused_lce(H, V, dtype)
print("  Bastile CuTile-CE ✓")

from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
x_l = torch.randn(8, H, dtype=dtype, device=device, requires_grad=True)
w_l = weight.clone().requires_grad_(True)
loss_l, _, _ = LigerFusedLinearCrossEntropyFunction.apply(x_l, w_l, torch.randint(0, V, (8,), device=device))
loss_l.backward()
torch.cuda.synchronize()
print("  Liger ✓")
print()

# ══════════════════════════════════════════════════════════════════════════
# Speed benchmark
# ══════════════════════════════════════════════════════════════════════════
print("─── SPEED (forward + backward) ────────────────────────────────────────────────")
hdr = f"{'BT':>8} {'SeqLen':>8} | {'PyTorch':>10} {'Bastile':>10} {'Liger':>10} | {'Bast/PT':>8} {'Bast/Lg':>8}"
print(hdr)
print("─" * len(hdr))

speed_results = {}

for BT in BT_sizes:
    BT_padded = BT + ((-BT) % 8)
    seq_len = BT  # batch=1
    x = torch.randn(BT_padded, H, dtype=dtype, device=device)
    t = torch.randint(0, V, (BT_padded,), device=device)

    def run_pytorch():
        xp = x.detach().requires_grad_(True)
        wp = weight.clone().requires_grad_(True)
        F.cross_entropy(F.linear(xp, wp), t).backward()

    def run_bastile():
        xc = x.detach().requires_grad_(True)
        wc = weight.clone().requires_grad_(True)
        bastile_lce(xc, wc, t, chunk_size=4096).backward()

    def run_liger():
        xl = x.detach().requires_grad_(True)
        wl = weight.clone().requires_grad_(True)
        ll, _, _ = LigerFusedLinearCrossEntropyFunction.apply(xl, wl, t)
        ll.backward()

    results = {}
    for name, fn in [("PyTorch", run_pytorch), ("Bastile", run_bastile),
                      ("Liger", run_liger)]:
        try:
            results[name] = benchmark_fn(fn)
        except Exception as e:
            print(f"  {name} OOM/error @ BT={BT}: {type(e).__name__}")
            results[name] = float('inf')

    speed_results[BT] = results
    t_pt = results["PyTorch"]
    t_ba = results["Bastile"]
    t_lg = results["Liger"]

    def fmt(v):
        return f"{v:>8.2f}ms" if v < float('inf') else "     OOM"

    print(f"{BT:>8} {seq_len:>8} | {fmt(t_pt)} {fmt(t_ba)} {fmt(t_lg)} | "
          f"{t_ba/t_pt:>7.2f}x {t_ba/t_lg:>7.2f}x")

# ══════════════════════════════════════════════════════════════════════════
# Peak memory benchmark
# ══════════════════════════════════════════════════════════════════════════
print()
print("─── PEAK GPU MEMORY (forward + backward) ─────────────────────────────────────")
hdr2 = f"{'BT':>8} {'SeqLen':>8} | {'PyTorch':>10} {'Bastile':>10} {'Liger':>10} | {'Bast/PT':>8} {'Logits':>10}"
print(hdr2)
print("─" * len(hdr2))

for BT in BT_sizes:
    BT_padded = BT + ((-BT) % 8)
    seq_len = BT
    x = torch.randn(BT_padded, H, dtype=dtype, device=device)
    t = torch.randint(0, V, (BT_padded,), device=device)

    # Theoretical full logits size
    logits_size_mb = BT_padded * V * 2 / (1024 ** 2)

    def run_pytorch():
        xp = x.detach().requires_grad_(True)
        wp = weight.clone().requires_grad_(True)
        F.cross_entropy(F.linear(xp, wp), t).backward()

    def run_bastile():
        xc = x.detach().requires_grad_(True)
        wc = weight.clone().requires_grad_(True)
        bastile_lce(xc, wc, t, chunk_size=4096).backward()

    def run_liger():
        xl = x.detach().requires_grad_(True)
        wl = weight.clone().requires_grad_(True)
        ll, _, _ = LigerFusedLinearCrossEntropyFunction.apply(xl, wl, t)
        ll.backward()

    mem = {}
    for name, fn in [("PyTorch", run_pytorch), ("Bastile", run_bastile),
                      ("Liger", run_liger)]:
        try:
            mem[name] = measure_peak_memory(fn)
        except Exception as e:
            print(f"  {name} OOM/error @ BT={BT}: {type(e).__name__}")
            mem[name] = float('inf')

    m_pt = mem["PyTorch"]
    m_ba = mem["Bastile"]
    m_lg = mem["Liger"]

    def fmt_mem(v):
        if v == float('inf'):
            return "       OOM"
        if v > 1024:
            return f"{v/1024:>8.1f}GB"
        return f"{v:>8.0f}MB"

    ratio = m_ba / m_pt if m_pt > 0 and m_pt < float('inf') else float('inf')
    print(f"{BT:>8} {seq_len:>8} | {fmt_mem(m_pt)} {fmt_mem(m_ba)} {fmt_mem(m_lg)} | "
          f"{ratio:>7.2f}x {logits_size_mb:>8.0f}MB")

print()
print("Note: 'Logits' column shows theoretical full [BT, V] logits tensor size (bf16).")
print("      Chunked approaches (Bastile/Liger) avoid materializing this.")
print("\nDone!")
