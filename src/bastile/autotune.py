"""
Autotuning infrastructure for Bastile kernels.

Provides caching and configuration selection for optimal kernel parameters.
Supports disk-based persistent caching to avoid re-tuning across sessions.
"""

import hashlib
import json
import os
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

import torch

T = TypeVar("T")

# Global cache for autotuned configurations
_autotune_cache: Dict[str, Any] = {}
_cache_lock = threading.Lock()

# Disk cache location
_CACHE_DIR = Path.home() / ".cache" / "bastile"


def _with_autotune_lock(fn: Callable[..., T]) -> Callable[..., T]:
    """Decorator for thread-safe cache access."""
    def wrapper(*args, **kwargs) -> T:
        with _cache_lock:
            return fn(*args, **kwargs)
    return wrapper


def _shape_dtype_key(tensor: torch.Tensor) -> str:
    """Generate a cache key from tensor shape and dtype."""
    return f"{tuple(tensor.shape)}_{tensor.dtype}"


def default_key(*tensors: torch.Tensor, **kwargs) -> str:
    """Generate a default cache key from tensors and kwargs."""
    parts = [_shape_dtype_key(t) for t in tensors]
    for k, v in sorted(kwargs.items()):
        parts.append(f"{k}={v}")
    return "_".join(parts)


def _time_ms(run_once: Callable, *, warmup: int = 3, rep: int = 5) -> float:
    """Time a function in milliseconds with proper warmup."""
    stream = torch.cuda.current_stream()
    stream.synchronize()

    # Warmup - run multiple times to ensure JIT compilation is done
    for _ in range(warmup):
        run_once()

    stream.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record(stream)
    for _ in range(rep):
        run_once()
    end.record(stream)
    end.synchronize()

    return start.elapsed_time(end) / max(1, rep)


@dataclass
class TunedResult:
    """Result of autotuning."""
    config: Any
    time_ms: float


@dataclass
class CacheEntry:
    """Cache entry for autotuned config."""
    config: Any
    time_ms: Optional[float] = None


def _get_disk_cache_path() -> Path:
    """Get path to disk cache file."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Include GPU info in cache key for portability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).replace(" ", "_")
        return _CACHE_DIR / f"autotune_cache_{gpu_name}.json"
    return _CACHE_DIR / "autotune_cache.json"


def _load_disk_cache() -> Dict[str, Any]:
    """Load cache from disk."""
    cache_path = _get_disk_cache_path()
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_disk_cache(cache: Dict[str, Any]):
    """Save cache to disk."""
    cache_path = _get_disk_cache_path()
    try:
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
    except IOError:
        pass  # Silent fail - disk cache is optional


def clear_cache():
    """Clear both memory and disk autotuning caches."""
    global _autotune_cache
    with _cache_lock:
        _autotune_cache.clear()
    # Also clear disk cache
    cache_path = _get_disk_cache_path()
    if cache_path.exists():
        try:
            cache_path.unlink()
        except IOError:
            pass


def autotune(
    run_fn: Callable[[Any], None],
    search_space: List[Any],
    key: str,
    *,
    kernel_name: str = "",
    max_iter: int = 10,
    warmup: int = 2,
    rep: int = 3,
    use_heuristic: bool = True,
    persist_to_disk: bool = True,
) -> Any:
    """
    Autotune a kernel by selecting the best config from the search space.

    Args:
        run_fn: Function that runs the kernel with a given config
        search_space: List of configs to try
        key: Cache key for this configuration
        kernel_name: Name of kernel (for disk cache organization)
        max_iter: Maximum configs to try (for large search spaces)
        warmup: Warmup iterations per config
        rep: Repetitions for timing
        use_heuristic: If True, pick middle config without timing (faster startup)
        persist_to_disk: If True, save results to disk cache

    Returns:
        Best config from the search space
    """
    global _autotune_cache

    full_key = f"{kernel_name}:{key}" if kernel_name else key

    # Convert generator to list if needed (do this early for disk cache template)
    if hasattr(search_space, '__iter__') and not hasattr(search_space, '__len__'):
        search_space = list(search_space)

    # Check memory cache first
    with _cache_lock:
        if full_key in _autotune_cache:
            return _autotune_cache[full_key].config

    # Check disk cache
    if persist_to_disk:
        disk_cache = _load_disk_cache()
        if full_key in disk_cache:
            config = disk_cache[full_key]
            # Try to reconstruct dataclass if search_space provides a template
            if isinstance(config, dict) and search_space:
                template = search_space[0]
                if hasattr(template, '__dataclass_fields__'):
                    # Reconstruct dataclass from dict
                    try:
                        config = type(template)(**config)
                    except (TypeError, ValueError):
                        pass  # Fall back to dict
            with _cache_lock:
                _autotune_cache[full_key] = CacheEntry(config=config)
            return config

    if not search_space:
        raise ValueError("Empty search space")

    # Use heuristic: pick middle config without timing
    if use_heuristic:
        best_config = search_space[len(search_space) // 2]
        with _cache_lock:
            _autotune_cache[full_key] = CacheEntry(config=best_config)

        # Persist to disk
        if persist_to_disk:
            disk_cache = _load_disk_cache()
            # Convert dataclass to dict if needed
            if hasattr(best_config, '__dict__'):
                disk_cache[full_key] = asdict(best_config) if hasattr(best_config, '__dataclass_fields__') else best_config.__dict__
            else:
                disk_cache[full_key] = best_config
            _save_disk_cache(disk_cache)

        return best_config

    # Full autotuning with timing
    best_config = search_space[0]
    best_time = float("inf")

    # Limit search space if too large
    configs_to_try = search_space[:max_iter] if len(search_space) > max_iter else search_space

    for config in configs_to_try:
        try:
            time_ms = _time_ms(lambda: run_fn(config), warmup=warmup, rep=rep)
            if time_ms < best_time:
                best_time = time_ms
                best_config = config
        except Exception:
            continue

    with _cache_lock:
        _autotune_cache[full_key] = CacheEntry(config=best_config, time_ms=best_time)

    # Persist to disk
    if persist_to_disk:
        disk_cache = _load_disk_cache()
        if hasattr(best_config, '__dict__'):
            disk_cache[full_key] = asdict(best_config) if hasattr(best_config, '__dataclass_fields__') else best_config.__dict__
        else:
            disk_cache[full_key] = best_config
        _save_disk_cache(disk_cache)

    return best_config


def warmup_all_kernels(
    batch_size: int = 4,
    seq_len: int = 512,
    hidden_size: int = 2048,
    vocab_size: int = 32000,
    intermediate_size: int = 5504,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """
    Warmup all registered CuTile kernels with typical tensor shapes.

    Call this once after bastile.apply() to pre-compile all kernels and
    avoid JIT compilation overhead during training.

    Args:
        batch_size: Batch size for warmup tensors
        seq_len: Sequence length for warmup tensors
        hidden_size: Hidden dimension (typical: 2048-4096)
        vocab_size: Vocabulary size (typical: 32000-152000)
        intermediate_size: MLP intermediate size
        dtype: Data type for warmup tensors
        device: Device for warmup tensors
    """
    if not torch.cuda.is_available():
        return

    # Import kernels
    try:
        from .ops.rms_norm import rms_norm, CuTileRMSNorm
        from .ops.swiglu import swiglu, CuTileSwiGLUMLP

        BT = batch_size * seq_len

        # Warmup RMSNorm
        x = torch.randn(BT, hidden_size, dtype=dtype, device=device)
        weight = torch.ones(hidden_size, dtype=dtype, device=device)
        for _ in range(3):
            _ = rms_norm(x, weight, eps=1e-6)
        torch.cuda.synchronize()

        # Warmup SwiGLU
        gate = torch.randn(BT, intermediate_size, dtype=dtype, device=device)
        up = torch.randn(BT, intermediate_size, dtype=dtype, device=device)
        for _ in range(3):
            _ = swiglu(gate, up)
        torch.cuda.synchronize()

    except Exception as e:
        # Silent fail - warmup is optional optimization
        import logging
        logging.debug(f"Warmup skipped: {e}")
