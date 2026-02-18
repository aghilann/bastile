"""
Qwen3 8B FSDP Benchmark: PyTorch vs Liger Kernel vs Bastile
Finetunes the real pretrained Qwen/Qwen3-8B across 8 GPUs using FSDP.

Launch:
    torchrun --nproc_per_node=8 -m tests.benchmarks.e2e.qwen_8b_fsdp
"""

import gc
import importlib
import os
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp import (
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

MODEL_ID = "Qwen/Qwen3-8B"
BATCH_SIZE_PER_GPU = 4
SEQ_LEN = 2048
WARMUP_ITERS = 5
DURATION_SEC = 30.0


@dataclass
class FSDPBenchmarkResult:
    name: str
    iterations: int
    total_time_sec: float
    avg_iter_ms: float
    tokens_per_sec: float
    peak_memory_gb: float
    initial_loss: float
    final_loss: float


def rank0_print(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs, flush=True)


def print_header(title: str, width: int = 100):
    rank0_print("\n" + "=" * width)
    rank0_print(f"  {title}")
    rank0_print("=" * width)


def clear_cuda_state():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()


def reset_peak_memory():
    torch.cuda.reset_peak_memory_stats()


def get_peak_memory_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**3


def reset_environment():
    clear_cuda_state()
    reset_peak_memory()
    import transformers.models.qwen3.modeling_qwen3 as qwen3_mod

    importlib.reload(qwen3_mod)


def get_fsdp_wrap_policy():
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    return partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen3DecoderLayer},
    )


def get_mixed_precision():
    return MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )


def setup_bastile():
    import bastile

    bastile.reset()
    applied = bastile.apply(
        rms_norm=True,
        swiglu=True,
        rope=True,
        fused_linear_cross_entropy=True,
    )
    return applied


def setup_liger():
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3

    apply_liger_kernel_to_qwen3(
        rope=True,
        rms_norm=True,
        swiglu=True,
        fused_linear_cross_entropy=True,
    )
    return "rope, rms_norm, swiglu, fused_linear_cross_entropy"


def run_benchmark(
    name: str,
    setup_fn: Callable | None = None,
    batch_size: int = BATCH_SIZE_PER_GPU,
    seq_len: int = SEQ_LEN,
    warmup_iters: int = WARMUP_ITERS,
    duration_sec: float = DURATION_SEC,
) -> FSDPBenchmarkResult | None:
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    reset_environment()
    dist.barrier()

    if setup_fn:
        rank0_print(f"  Applying {name} patches...")
        applied = setup_fn()
        rank0_print(f"  Patches applied: {applied}")

    dist.barrier()

    from transformers import Qwen3ForCausalLM

    rank0_print(f"  Loading {MODEL_ID} (pretrained)...")
    model = Qwen3ForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    num_params = sum(p.numel() for p in model.parameters())
    rank0_print(f"  Model: {num_params / 1e9:.2f}B parameters")

    model = model.to(device)
    dist.barrier()

    rank0_print(f"  Wrapping with FSDP (FULL_SHARD, {world_size} GPUs)...")
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=get_mixed_precision(),
        auto_wrap_policy=get_fsdp_wrap_policy(),
        use_orig_params=True,
        device_id=device,
    )
    model.train()

    global_batch = batch_size * world_size
    tokens_per_iter = global_batch * seq_len
    rank0_print(
        f"  Batch: {batch_size}/GPU x {world_size} GPUs = {global_batch} global, "
        f"seq_len={seq_len}, {tokens_per_iter} tokens/iter"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    input_ids = torch.randint(0, 151936, (batch_size, seq_len), device=device)
    labels = torch.randint(0, 151936, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)

    rank0_print(f"  Warming up ({warmup_iters} iterations)...")
    try:
        for _ in range(warmup_iters):
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
            optimizer.step()
    except torch.cuda.OutOfMemoryError:
        rank0_print("  OOM during warmup — skipping")
        del model, optimizer
        clear_cuda_state()
        dist.barrier()
        return None

    torch.cuda.synchronize()
    dist.barrier()
    reset_peak_memory()

    rank0_print(f"  Running benchmark for {duration_sec:.0f} seconds...")
    iterations = 0
    losses = []
    iter_times = []

    start_time = time.perf_counter()

    try:
        while True:
            iter_start = time.perf_counter()

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            iter_end = time.perf_counter()

            iter_times.append(iter_end - iter_start)
            losses.append(loss.item())
            iterations += 1

            elapsed = iter_end - start_time
            if elapsed >= duration_sec:
                break

            if iterations % 5 == 0:
                tps = (iterations * tokens_per_iter) / elapsed
                rank0_print(f"    Iter {iterations}: {tps:,.0f} tok/s, loss={loss.item():.4f}")
    except torch.cuda.OutOfMemoryError:
        rank0_print(f"  OOM during benchmark after {iterations} iters — using partial results")
        if not iter_times:
            del model, optimizer
            clear_cuda_state()
            dist.barrier()
            return None

    total_time = time.perf_counter() - start_time
    peak_mem = get_peak_memory_gb()

    result = FSDPBenchmarkResult(
        name=name,
        iterations=iterations,
        total_time_sec=total_time,
        avg_iter_ms=sum(iter_times) / len(iter_times) * 1000,
        tokens_per_sec=(iterations * tokens_per_iter) / total_time,
        peak_memory_gb=peak_mem,
        initial_loss=losses[0],
        final_loss=losses[-1],
    )

    rank0_print(
        f"  Complete: {result.iterations} iters, {result.tokens_per_sec:,.0f} tok/s, "
        f"{result.avg_iter_ms:.1f} ms/iter, {result.peak_memory_gb:.2f} GB/GPU"
    )

    del model, optimizer
    clear_cuda_state()
    dist.barrier()

    return result


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    rank0_print("=" * 100)
    rank0_print("  Qwen3 8B FSDP Benchmark: PyTorch vs Liger Kernel vs Bastile")
    rank0_print("=" * 100)
    rank0_print(f"\n  Model: {MODEL_ID} (pretrained)")
    rank0_print(f"  GPUs: {world_size}x {torch.cuda.get_device_name(local_rank)}")
    rank0_print("  FSDP: FULL_SHARD, BFloat16MixedPrecision")
    rank0_print(f"  Batch: {BATCH_SIZE_PER_GPU}/GPU x {world_size} = {BATCH_SIZE_PER_GPU * world_size} global")
    rank0_print(f"  Seq len: {SEQ_LEN}, Warmup: {WARMUP_ITERS}, Duration: {DURATION_SEC}s per config")
    rank0_print()

    results: dict[str, FSDPBenchmarkResult | None] = {}

    # 1. Bastile
    print_header("Benchmark: Bastile")
    results["Bastile"] = run_benchmark("Bastile", setup_fn=setup_bastile)
    import bastile

    bastile.reset()
    dist.barrier()

    # 2. Liger
    print_header("Benchmark: Liger Kernel")
    reset_environment()
    dist.barrier()
    try:
        results["Liger"] = run_benchmark("Liger", setup_fn=setup_liger)
    except Exception as e:
        rank0_print(f"  Error: {e}")
        if rank == 0:
            traceback.print_exc()
        results["Liger"] = None
    dist.barrier()

    # 3. PyTorch
    print_header("Benchmark: PyTorch (Baseline)")
    reset_environment()
    dist.barrier()
    results["PyTorch"] = run_benchmark("PyTorch", setup_fn=None)
    dist.barrier()

    # Results (rank 0 only)
    if rank == 0:
        print_header("RESULTS")
        configs = ["PyTorch", "Liger", "Bastile"]

        print(f"\n  {'Config':<15} {'Tokens/sec':>14} {'Speedup':>10} {'ms/iter':>12} {'Mem/GPU':>10} {'Loss':>12}")
        print(f"  {'-' * 80}")

        baseline = results.get("PyTorch")
        for c in configs:
            r = results.get(c)
            if r is None:
                print(f"  {c:<15} {'FAILED':>14}")
                continue
            speedup = r.tokens_per_sec / baseline.tokens_per_sec if baseline else 0
            print(
                f"  {c:<15} {r.tokens_per_sec:>12,.0f}   {speedup:>8.2f}x {r.avg_iter_ms:>10.1f}ms "
                f"{r.peak_memory_gb:>8.2f}GB {r.final_loss:>12.4f}"
            )

        print_header("SUMMARY")
        b = results.get("Bastile")
        l = results.get("Liger")
        p = results.get("PyTorch")

        if p:
            print(f"\n  Baseline: PyTorch ({p.tokens_per_sec:,.0f} tok/s, {p.peak_memory_gb:.2f} GB/GPU)")
        if b and p:
            speedup = (b.tokens_per_sec / p.tokens_per_sec - 1) * 100
            mem_diff = b.peak_memory_gb - p.peak_memory_gb
            print(f"  Bastile vs PyTorch: {speedup:+.1f}% throughput, {mem_diff:+.2f} GB/GPU memory")
        if b and l:
            speedup = (b.tokens_per_sec / l.tokens_per_sec - 1) * 100
            mem_diff = b.peak_memory_gb - l.peak_memory_gb
            print(f"  Bastile vs Liger:   {speedup:+.1f}% throughput, {mem_diff:+.2f} GB/GPU memory")
        if l and p:
            speedup = (l.tokens_per_sec / p.tokens_per_sec - 1) * 100
            mem_diff = l.peak_memory_gb - p.peak_memory_gb
            print(f"  Liger vs PyTorch:   {speedup:+.1f}% throughput, {mem_diff:+.2f} GB/GPU memory")

        print("\n" + "=" * 100)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
