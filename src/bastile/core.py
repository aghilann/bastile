"""
Core patching functionality for Bastile.

Handles applying and resetting CuTile kernel patches to PyTorch and HuggingFace.
"""

import importlib
import logging
from typing import Optional, List, Any

import torch

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
    
    # Handle dotted target_attr (e.g., "GptOssForCausalLM.forward")
    parts = patch.target_attr.split(".")
    obj = module
    for part in parts[:-1]:
        if not hasattr(obj, part):
            logger.warning(f"Could not apply patch '{patch.name}': {part} not found in {patch.target_module}")
            return False
        obj = getattr(obj, part)
    
    attr_name = parts[-1]
    if not hasattr(obj, attr_name):
        logger.warning(f"Could not apply patch '{patch.name}': {patch.target_attr} not found in {patch.target_module}")
        return False
    
    # Store original
    original = getattr(obj, attr_name)
    
    # Apply patch
    setattr(obj, attr_name, patch.replacement)
    
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
    
    # Handle dotted target_attr (e.g., "GptOssForCausalLM.forward")
    parts = patch.target_attr.split(".")
    obj = module
    for part in parts[:-1]:
        if not hasattr(obj, part):
            return False
        obj = getattr(obj, part)
    
    # Restore original
    setattr(obj, parts[-1], patch.original)
    
    # Mark as reset
    registry = get_registry()
    registry.mark_reset(patch.name)
    
    logger.info(f"Reset patch: {patch.name}")
    return True


def apply(
    rms_norm: bool = True,   # CuTile RMSNorm with full backward (from TileGym)
    swiglu: bool = True,     # CuTile SwiGLU with full backward (from TileGym)
    rope: bool = True,       # CuTile RoPE with autotuning
    cross_entropy: bool = False,  # Deprecated, unused (kept for API compat)
    fused_linear_cross_entropy: bool = True,  # Fused linear + CE via quack (skips logits)
    moe_experts: bool = True,  # CuTile fused MoE experts (includes gate) for GPT-OSS
    moe_router: bool = True,  # Fused MoE router + load-balancing loss for GPT-OSS
    model_type: Optional[str] = None,
) -> List[str]:
    """
    Apply CuTile kernel patches to PyTorch/HuggingFace.

    Args:
        rms_norm: Whether to patch RMSNorm (default: True)
        swiglu: Whether to patch SwiGLU/MLP (default: True)
        rope: Whether to patch RoPE (default: True)
        cross_entropy: Deprecated, no-op (kept for backwards compatibility)
        fused_linear_cross_entropy: Whether to use fused linear + CE (default: True)
        moe_experts: Whether to patch MoE experts with CuTile fused GEMM (default: True)
        moe_router: Whether to patch MoE router + load-balancing loss (default: True)
        model_type: Optional model type filter (e.g., 'qwen3')

    Returns:
        List of applied patch names

    Example:
        >>> import bastile
        >>> bastile.apply(fused_linear_cross_entropy=True)
    """
    # Import ops to register patches
    from . import ops  # noqa: F401

    registry = get_registry()

    # Build list of patches to apply based on flags
    patch_filter = {
        'rms_norm': rms_norm,
        'swiglu': swiglu,
        'rope': rope,
        'moe_router': moe_router,
        'moe_load_balancing': moe_router,
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
    # Patches ForCausalLM.forward to skip logits materialization
    if fused_linear_cross_entropy:
        # Qwen3
        try:
            from .ops.fused_linear_cross_entropy import bastile_lce_forward
            import transformers.models.qwen3.modeling_qwen3 as qwen3_module

            qwen3_module.Qwen3ForCausalLM.forward = bastile_lce_forward
            applied.append("fused_linear_cross_entropy_qwen3")
            logger.info("Applied fused linear cross-entropy for Qwen3")
        except Exception as e:
            logger.warning(f"Could not apply fused linear cross-entropy for Qwen3: {e}")

        # GPT-OSS
        try:
            from .ops.fused_linear_cross_entropy import bastile_gpt_oss_lce_forward
            import transformers.models.gpt_oss.modeling_gpt_oss as gpt_oss_module

            gpt_oss_module.GptOssForCausalLM.forward = bastile_gpt_oss_lce_forward
            applied.append("fused_linear_cross_entropy_gpt_oss")
            logger.info("Applied fused linear cross-entropy for GPT-OSS")
        except Exception as e:
            logger.warning(f"Could not apply fused linear cross-entropy for GPT-OSS: {e}")

    # Apply CuTile fused MoE experts for GPT-OSS
    # Replaces entire GptOssExperts.forward with fused GEMM kernels
    # (includes the gate kernel internally, so no separate moe_gate patch needed)
    if moe_experts:
        try:
            from .ops.moe_experts import cutile_moe_experts_forward
            import transformers.models.gpt_oss.modeling_gpt_oss as gpt_oss_module

            gpt_oss_module.GptOssExperts.forward = cutile_moe_experts_forward
            applied.append("moe_experts_gpt_oss")
            logger.info("Applied CuTile fused MoE experts for GPT-OSS")
        except Exception as e:
            logger.warning(f"Could not apply MoE experts for GPT-OSS: {e}")

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
