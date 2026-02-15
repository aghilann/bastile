# Bastile CuTile RMSNorm Optimization — Full Context Briefing

## 1. Project Overview

**Repository:** `/workspace/bastile` (cloned from `https://github.com/aghilann/bastile`)

**Goal:** Replace the Quack-based (cuteDSL) RMSNorm in Bastile with a native **CuTile** (`cuda.tile`) implementation, and optimize it until performance matches or beats Quack on NVIDIA Blackwell (B200) GPUs.

**Current state:** Forward pass is ~1.05–1.07x slower than Quack on average (best configs match). Backward pass was just rewritten with Quack-style math but **has not been re-benchmarked yet** after the latest optimization.

---

## 2. Environment & Docker Setup

```bash
# Docker container (always running, GPU passthrough, bastile mounted)
docker run -d --gpus all --network=host --name bastile-dev \
  -v /workspace/bastile:/workspace/bastile \
  -w /workspace/bastile \
  baseten/gpu-dev:v8-cu13_1 sleep infinity

# Run commands inside container
docker exec bastile-dev bash -c "cd /workspace/bastile && \
  LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:\$LD_LIBRARY_PATH \
  uv run python3 -u -m tests.benchmarks.kernel.rms_norm_quack_vs_cutile" 2>&1

# CRITICAL: LD_LIBRARY_PATH must include /usr/local/cuda-13.0/compat
# otherwise PyTorch won't find libcuda.so inside the container
```

**Key dependencies (installed in `.venv` via `uv`):**
- `quack-kernels[cu13]` — Quack RMSNorm baseline (cuteDSL)
- `cuda-tile` — NVIDIA's tile-based GPU programming library
- `liger-kernel` — Another kernel library (used in e2e benchmarks)
- `torch 2.10.0` with CUDA 13.0

**GPU:** NVIDIA B200 (Blackwell), 160 SMs, 8000 GB/s peak bandwidth

---

## 3. File Map

### Core Implementation Files

| File | Purpose |
|------|---------|
| `src/bastile/ops/rms_norm_cutile.py` | **CuTile RMSNorm** — the file being optimized (640 lines). Contains 3 forward kernels + 1 backward kernel + autograd function + heuristic config selection. |
| `src/bastile/ops/rms_norm.py` | **Quack RMSNorm** wrapper — `FastRMSNormFunction` with inlined dispatch. Used as baseline. |
| `src/bastile/ops/__init__.py` | Currently imports `rms_norm_cutile as rms_norm` (CuTile is active, not Quack). |
| `src/bastile/ops/utils.py` | `next_power_of_2()` and `ceildiv()` utilities. |
| `src/bastile/autotune.py` | Autotuning infrastructure: `autotune()`, `_time_ms()`, disk cache. Imports from `rms_norm_cutile` for warmup. |
| `src/bastile/registry.py` | `register_patch()` — patches HuggingFace model classes with optimized ops. |

### Benchmark Files

| File | Purpose |
|------|---------|
| `tests/benchmarks/kernel/rms_norm_quack_vs_cutile.py` | **Main benchmark**: Forward (Quack vs CuTile vs PyTorch) + FWD+BWD (CuTile vs PyTorch). 16 configs covering M∈{256,2048,8192,16384}, N∈{2048,3584,4096,5120,8192}, bf16+fp16. |
| `Makefile` | `make bench-rmsnorm` target. Must prefix with `LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH`. |

### Reference Code (READ THESE for optimization inspiration)

| File | What to learn |
|------|---------------|
| `.venv/lib/python3.12/site-packages/quack/rmsnorm.py` | **Quack's backward kernel** — the gold standard. Study `RMSNormBackward` class (line 439–783) and `_get_sm_count()` (line 785). Key patterns: persistent grid-stride loop, dw_partial accumulation via shared memory reduction, sm_count multiplier for small N, double-buffered smem stages. |
| `TileGym/src/tilegym/ops/cutile/rms_norm.py` | CuTile RMSNorm patterns: static persistent kernels, TMA loads, `allow_tma=False` on stores, latency hints. |
| `TileGym/src/tilegym/ops/cutile/softmax.py` | Chunked softmax with TMA — shows how to handle large N with tiling. |
| `TileGym/CONTRIBUTING.md` | General CuTile optimization tips for Blackwell. |

---

## 4. Architecture of `rms_norm_cutile.py`

### 4.1 Forward Kernels (3 total, simplified from 12)

1. **`rms_norm_fwd_persistent`** (occupancy=1) — Static persistent 2D kernel. Grid = min(NUM_SMS, num_row_tiles). Each block processes multiple row-tiles via grid-stride loop. Weight pre-loaded once. Best for **large M** (training).

2. **`rms_norm_fwd_gather`** (occupancy=2) — 1D gather/scatter kernel. Loops over columns in TILE_SIZE chunks. **No padding waste** — only touches real columns. Best for **non-power-of-2 N** or moderate M.

3. **`rms_norm_fwd_tma`** (occupancy=4) — TMA single-row kernel. Bulk DMA load of full row. Best for **small M with power-of-2 N**.

### 4.2 Forward Heuristic (`_heuristic_config`)

```
High padding (>20% waste) → gather (avoids TMA loading zeros)
Small M (≤ 4×SMS) + pow2 N → TMA
Small M (≤ 4×SMS) + non-pow2 N → gather  
Medium M (≤ 16×SMS) → persistent, tile_m capped to 32K elements
Large M (> 16×SMS) → persistent, tile_m=4
```

### 4.3 Backward Kernel (1 persistent kernel)

**`rms_norm_bwd_persistent`** (occupancy=1) — Uses Quack's efficient math:
```
x_hat = x * rstd                        # normalized input, computed once
wdy   = dy * w                          # weighted gradient  
c1    = mean(x_hat * wdy)              # correction scalar per row
dx    = (wdy - x_hat * c1) * rstd      # input gradient
dw   += sum(dy * x_hat)                # accumulated across rows
```

Grid size uses Quack-style sm_count multiplier (`_bwd_grid_size()`):
- N≤256: 16× SMs, N≤1024: 8× SMs, N≤2048: 4× SMs, N≤4096: 2× SMs, else: 1× SMs

### 4.4 Autograd Function (`CuTileRMSNormFunction`)

Both forward and backward have **inlined hot paths** to minimize Python dispatch overhead:
- Config lookup is a single `dict.get()` with `(M, N, dtype)` as key
- Stream and NUM_SMS are cached globally
- `dtype` is used directly as dict key (hashable, avoids `str()` conversion)

---

## 5. Benchmark Results History

### Forward Pass Evolution

| Version | Average vs Quack | Best | Worst | Key Change |
|---------|-----------------|------|-------|------------|
| Initial (12 kernels, autotuning) | 1.31x slower | 1.08x slower | 1.65x slower | Heuristic picking wrong configs |
| Heuristic refinement | 1.26x slower | 1.06x slower | 1.59x slower | Better persistent tile_m caps |
| Python overhead reduction | 1.22x slower | 1.06x slower | 1.59x slower | Cached stream, dtype as key |
| Padding-aware routing | 1.09x slower | 1.02x faster | 1.48x slower | Gather for high-padding N |
| Simplified to 3 kernels | **1.05x slower** | **1.07x faster** | **1.19x slower** | Fixed occupancy per strategy |

### Best Forward Results (3-kernel version)

```
M=256    N=2048  CuTile  1.00x (matches Quack)
M=256    N=5120  CuTile  1.04x slower
M=2048   N=4096  CuTile  1.15–1.18x slower
M=8192   N=3584  CuTile  1.07x FASTER than Quack
M=8192   N=4096  CuTile  1.00–1.02x (matches Quack)
M=16384  N=4096  CuTile  1.12x slower

Average: 1.05–1.07x slower than Quack
```

### Backward Pass (FWD+BWD via autograd, CuTile vs PyTorch)

**Before persistent backward optimization:**
```
Average: 1.38x slower than PyTorch
Best:    1.02x (M=8192 N=4096)
Worst:   3.64x slower (M=256 N=2048)
```

**After first persistent backward + partial dw reduction:**
```
Average: 1.23x slower than PyTorch
Best:    1.08x FASTER (M=2048 N=8192)
Worst:   2.87x slower (M=256 N=2048)
```

**Latest (Quack-style math + grid sizing) — NOT YET BENCHMARKED:**
The backward kernel was just rewritten with:
1. Quack's efficient `x_hat/wdy/c1` formulation (fewer FLOPs)
2. Quack-style `_bwd_grid_size()` with sm_count multiplier
3. Conservative `_bwd_tile_m()` to avoid register pressure

---

## 6. Known Performance Bottlenecks

### 6.1 Python Dispatch Overhead (~5–8µs)

For small/fast kernels, the fixed cost of `torch.autograd.Function.__call__` + `ct.launch()` + dict lookups dominates. Measured overhead:
- Noop autograd: ~5.5µs
- `ct.launch` only: ~10.4µs
- Quack wrapper total: ~11.5µs
- CuTile wrapper total: ~14.4µs

**Impact:** For M=256 N=4096, raw kernel is ~6µs but wrapped is ~14µs. This ~8µs gap explains most of the difference at small M.

### 6.2 TMA Padding Waste for Non-Power-of-2 N

TMA loads `next_power_of_2(N)` columns. For N=5120, TILE_N=8192 → 37.5% wasted bandwidth. **Mitigated** by routing to gather kernel when padding > 20%.

### 6.3 Register Pressure for Large TILE_M × TILE_N

Persistent kernels with large tiles (e.g., tile_m=8 × TILE_N=8192 = 128KB in bf16 → 256KB in f32) cause register spilling. **Mitigated** by capping `tile_m` based on N.

### 6.4 Backward JIT Compilation Cost

Each unique `(TILE_M, TILE_N)` combination requires separate JIT compilation (~2–5s). First call is slow; subsequent calls hit the compile cache.

---

## 7. Key Optimization Patterns (from Quack + TileGym)

### From Quack's Backward Kernel (`rmsnorm.py:439–783`)

1. **Efficient math:** `x_hat = x * rstd`, `wdy = dy * w`, `c1 = mean(x_hat * wdy)`, `dx = (wdy - x_hat * c1) * rstd` — avoids computing `rstd³` and uses fewer multiply operations.

2. **sm_count multiplier:** For small N, uses more blocks than SMs (up to 16×) for better latency hiding and reduced wave quantization:
   ```python
   def _get_sm_count(N, device):
       mult = 16 if N<=256 else (8 if N<=1024 else (4 if N<=2048 else (2 if N<=4096 else 1)))
       return sm_count * mult
   ```

3. **Double-buffered smem:** Prefetches next row-tile while processing current one. Uses `stage ^= 1` ping-pong pattern.

4. **Shared memory dw reduction:** For multi-row tiles (`tiler_mn[0] > 1`), reduces dw across rows within a threadblock using shared memory before writing to global `dw_partial`.

5. **Cluster-based cross-CTA reduction:** Uses `cluster_n` for cooperative cross-CTA reductions on Hopper/Blackwell.

6. **Weight pre-loaded once:** Weight tensor loaded into registers before the persistent loop.

### From TileGym CuTile Patterns

1. **`allow_tma=False` on `ct.store()`** — Bypasses TMA store path, +30% on B200.
2. **Latency hints:** `latency=10` on `ct.load()`, `latency=3` on `ct.store()`.
3. **Static persistent scheduling:** `grid = NUM_SMS`, blocks stride via `range(bid, upper_bound, num_blocks)`.
4. **`padding_mode=PAD_ZERO`** on all loads to handle boundary conditions.

---

## 8. Quack Backward Stride Bug

The installed `quack-kernels` has a bug in `_rmsnorm_bwd` where `mdO.strides[1]` is mismatched during backward compilation. This affects calling `quack.rmsnorm._rmsnorm_bwd` directly.

**Workaround in benchmark:** The `FastRMSNormFunction` backward path (which calls `_rmsnorm_bwd` with properly-shaped tensors from `ctx.saved_tensors`) also triggers this bug in the current benchmark environment. As a result, the FWD+BWD benchmark currently only compares **CuTile vs PyTorch** (not vs Quack).

**Impact on optimization target:** Since we can't directly benchmark CuTile backward vs Quack backward, the target is to at least match or beat PyTorch's built-in RMSNorm backward.

---

## 9. Immediate Next Steps

### 9.1 Re-benchmark After Backward Optimization (HIGH PRIORITY)

The backward kernel was just rewritten with Quack-style math + grid sizing but hasn't been benchmarked:

```bash
docker exec bastile-dev bash -c "cd /workspace/bastile && \
  rm -rf ~/.cache/bastile && \
  LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:\$LD_LIBRARY_PATH \
  uv run python3 -u -m tests.benchmarks.kernel.rms_norm_quack_vs_cutile" 2>&1
```

### 9.2 Further Backward Optimization Ideas

If the backward is still slow after re-benchmarking:

1. **Double-buffered smem for backward** (like Quack): Prefetch next tile while processing current. CuTile's `ct.load` with `latency=10` partially achieves this, but explicit smem double-buffering could help.

2. **Gather-based backward for non-pow2 N:** Currently the backward always uses TMA-style loads (pad to pow2). For N=3584/5120, a gather-based backward could avoid 12–37% padding waste.

3. **Tune `_bwd_tile_m` per-shape:** Current heuristic is:
   - N≤1024: tile_m=4
   - N≤4096: tile_m=2
   - else: tile_m=1
   
   May need shape-specific tuning (e.g., M=256 with large tile_m wastes work).

4. **Reduce Python dispatch overhead in backward:** The backward currently calls `_bwd_grid_size()` and `_bwd_tile_m()` on every call. These could be cached per (N, M) pair.

5. **Study Quack's shared-memory dw reduction:** Quack's backward reduces dw across rows within a threadblock using shared memory before writing to global `dw_partial`. This avoids the host-side `dw_partial.sum(dim=0)`. In CuTile, the persistent kernel accumulates in registers, but for multi-row tiles, an intra-block smem reduction could improve bandwidth.

### 9.3 Forward Remaining Gaps

The main forward gap is at **M=2048 with N=5120/8192** (~1.2–1.5x slower). These are persistent kernels where:
- N=5120: Routed to gather (padding avoidance), but gather is inherently slower for large M
- N=8192: Persistent with tile_m capped due to register pressure

Potential fixes:
- Try **multi-tile gather** that processes several rows at once
- Try **loop-based persistent** that processes N in chunks (like Quack's internal loop) instead of loading the full padded row

---

## 10. Running Benchmarks

```bash
# Full benchmark (forward + backward, 16 configs)
docker exec bastile-dev bash -c "cd /workspace/bastile && \
  rm -rf ~/.cache/bastile && \
  LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:\$LD_LIBRARY_PATH \
  uv run python3 -u -m tests.benchmarks.kernel.rms_norm_quack_vs_cutile" 2>&1

# Quick forward-only timing for a specific shape
docker exec bastile-dev bash -c "cd /workspace/bastile && \
  LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:\$LD_LIBRARY_PATH \
  uv run python3 -u -c \"
import torch
from bastile.ops.rms_norm_cutile import rms_norm, _heuristic_config
from bastile.autotune import _time_ms

M, N = 2048, 4096
x = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
w = torch.ones(N, device='cuda', dtype=torch.bfloat16)
_ = rms_norm(x, w, 1e-6)  # warmup
t = _time_ms(lambda: rms_norm(x, w, 1e-6), warmup=10, rep=30)
cfg = _heuristic_config(M, N)
kind = 'persist' if cfg.use_persistent else ('tma' if cfg.use_tma else 'gather')
print(f'M={M} N={N} [{kind} tm={cfg.tile_size_m}]: {t*1000:.1f}µs')
\"" 2>&1

# Full autotuning (slower, tries all configs)
BASTILE_AUTOTUNE=1  # Set this env var before running

# Raw kernel vs wrapped overhead analysis
# See terminal 4 output for examples of measuring dispatch overhead
```

---

## 11. Important Implementation Details

### CuTile API Patterns

```python
import cuda.tile as ct

ConstInt = ct.Constant[int]
ConstFloat = ct.Constant[float]
PAD_ZERO = ct.PaddingMode.ZERO

# Kernel decorator with occupancy hint
@ct.kernel(occupancy=1)
def my_kernel(x, y, TILE_SIZE: ConstInt):
    bid = ct.bid(0)                              # block index
    num_blocks = ct.num_blocks(0)                # grid size
    
    # TMA 2D load
    tile = ct.load(x, index=(bid, 0), shape=(TILE_M, TILE_N),
                   padding_mode=PAD_ZERO, latency=10)
    
    # Gather 1D load (no padding waste)
    offsets = ct.arange(TILE_SIZE, dtype=ct.int32)
    vals = ct.gather(x, (row, offsets), latency=1)
    
    # Compute
    tile_f32 = ct.astype(tile, ct.float32)
    result = ct.sum(tile_f32, axis=1, keepdims=True)
    
    # Store (allow_tma=False is faster on Blackwell!)
    ct.store(y, index=(bid, 0), tile=result, allow_tma=False, latency=3)

# Launch
ct.launch(stream, (grid_size,), my_kernel, (x, y, tile_size))
```

### Autograd Integration

```python
class CuTileRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        # Inline everything for speed — no function calls on hot path
        config = _fwd_config_cache.get((M, N, x.dtype))  # single dict lookup
        ct.launch(...)
        ctx.save_for_backward(x, weight, rstd)
        return y

    @staticmethod  
    def backward(ctx, dy):
        # Also inlined
        ct.launch(...)
        dw = dw_partial[:, :N].sum(dim=0).to(weight.dtype)
        return dx, dw, None
```

### Config Dataclass

```python
@dataclass
class RMSNormCuTileConfig:
    tile_size: int          # TILE_N for persistent/TMA, column tile for gather
    use_tma: bool = False
    use_persistent: bool = False
    tile_size_m: int = 1    # rows per tile (persistent only)
```

---

## 12. Git State

The latest changes have been committed. The active branch contains:
- 3 forward kernels (persistent/gather/TMA)
- 1 backward kernel (persistent with Quack-style math)
- Heuristic config selection with padding-awareness
- Inlined autograd hot paths
- Full benchmark suite

The `__init__.py` currently routes to CuTile (`from . import rms_norm_cutile as rms_norm`). To switch back to Quack, change this to `from . import rms_norm`.
