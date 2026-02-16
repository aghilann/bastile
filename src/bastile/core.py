"""
Core patching functionality for Bastile.

Handles applying and resetting CuTile kernel patches to PyTorch and HuggingFace.
"""

import importlib
import logging
from typing import Optional, List, Any

from .registry import get_registry, PatchInfo

logger = logging.getLogger(__name__)


def _import_module(module_path: str):
    """Dynamically import a module."""
    try:
        return importlib.import_module(module_path)
    except ImportError as e:
        logger.warning(f"Could not import {module_path}: {e}")
        return None


def _apply_patch(patch: PatchInfo) -> bool:
    """Apply a single patch. Returns True if successful."""
    if patch.is_applied:
        logger.debug(f"Patch '{patch.name}' already applied")
        return True
    
    module = _import_module(patch.target_module)
    if module is None:
        logger.warning(f"Could not apply patch '{patch.name}': module not found")
        return False
    
    if not hasattr(module, patch.target_attr):
        logger.warning(f"Could not apply patch '{patch.name}': {patch.target_attr} not found in {patch.target_module}")
        return False
    
    # Store original
    original = getattr(module, patch.target_attr)
    
    # Apply patch
    setattr(module, patch.target_attr, patch.replacement)
    
    # Mark as applied
    registry = get_registry()
    registry.mark_applied(patch.name, original)
    
    logger.info(f"Applied patch: {patch.name} ({patch.description})")
    return True


def _reset_patch(patch: PatchInfo) -> bool:
    """Reset a single patch. Returns True if successful."""
    if not patch.is_applied:
        logger.debug(f"Patch '{patch.name}' not applied, skipping reset")
        return True
    
    if patch.original is None:
        logger.warning(f"Cannot reset patch '{patch.name}': original not stored")
        return False
    
    module = _import_module(patch.target_module)
    if module is None:
        return False
    
    # Restore original
    setattr(module, patch.target_attr, patch.original)
    
    # Mark as reset
    registry = get_registry()
    registry.mark_reset(patch.name)
    
    logger.info(f"Reset patch: {patch.name}")
    return True


def apply(
    rms_norm: bool = True,
    swiglu: bool = True,
    rope: bool = True,
    fused_linear_cross_entropy: bool = True,
    model_type: Optional[str] = None,
    **kwargs,
) -> List[str]:
    """
    Apply CuTile kernel patches to PyTorch/HuggingFace.

    Args:
        rms_norm: Whether to patch RMSNorm (default: True)
        swiglu: Whether to patch SwiGLU/MLP (default: True)
        rope: Whether to patch RoPE (default: True)
        fused_linear_cross_entropy: Whether to use fused linear + CE (default: True)
        model_type: Optional model type filter (e.g., 'qwen3')

    Returns:
        List of applied patch names

    Example:
        >>> import bastile
        >>> bastile.apply()
    """
    from . import ops  # noqa: F401

    registry = get_registry()

    patch_filter = {
        'rms_norm': rms_norm,
        'swiglu': swiglu,
        'rope': rope,
    }

    applied = []

    # Get all patches, optionally filtered by model type
    if model_type:
        patches = registry.get_for_model(model_type)
    else:
        patches = [registry.get(name) for name in registry.list_all()]
        patches = [p for p in patches if p is not None]

    for patch in patches:
        # Find matching filter
        should_apply = True
        for key, enabled in patch_filter.items():
            if key in patch.name or patch.name.startswith(key.split('_')[0]):
                should_apply = enabled
                break

        if should_apply and _apply_patch(patch):
            applied.append(patch.name)

    # Apply fused linear cross-entropy if requested
    # This patches Qwen3ForCausalLM.forward to skip logits materialization
    if fused_linear_cross_entropy:
        try:
            from .ops.fused_linear_cross_entropy import bastile_lce_forward
            import transformers.models.qwen3.modeling_qwen3 as qwen3_module

            qwen3_module.Qwen3ForCausalLM.forward = bastile_lce_forward
            applied.append("fused_linear_cross_entropy")
            logger.info("Applied fused linear cross-entropy (skips logits materialization)")
        except Exception as e:
            logger.warning(f"Could not apply fused linear cross-entropy: {e}")

    # Warmup kernels to avoid JIT overhead during training
    try:
        from .autotune import warmup_all_kernels
        warmup_all_kernels()
    except Exception as e:
        logger.debug(f"Warmup skipped: {e}")

    return applied


def apply_to_model(model: Any, **kwargs) -> List[str]:
    """
    Apply patches relevant to a specific HuggingFace model.
    
    Args:
        model: A HuggingFace PreTrainedModel instance
        **kwargs: Same as apply()
    
    Returns:
        List of applied patch names
    """
    # Try to detect model type
    model_type = None
    if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
        model_type = model.config.model_type
    
    logger.info(f"Applying patches for model type: {model_type}")
    return apply(model_type=model_type, **kwargs)


def reset(names: Optional[List[str]] = None) -> List[str]:
    """
    Reset patches to original implementations.
    
    Args:
        names: Optional list of patch names to reset. If None, reset all.
    
    Returns:
        List of reset patch names
    """
    registry = get_registry()
    
    if names is None:
        names = registry.get_applied()
    
    reset_names = []
    for name in names:
        patch = registry.get(name)
        if patch and _reset_patch(patch):
            reset_names.append(name)
    
    return reset_names


def get_patched_ops() -> List[str]:
    """Get list of currently patched operations."""
    return get_registry().get_applied()
