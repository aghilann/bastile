"""
Core patching functionality for Bastile.

Handles applying and resetting CuTile kernel patches to PyTorch and HuggingFace.
"""

import importlib
import logging
from typing import Any

import torch
from torch import nn

from .registry import PatchInfo, clone_patch, get_registry, register_patch

logger = logging.getLogger(__name__)


def _ensure_qwen3_5_compat_patches_registered() -> None:
    """Register Qwen3.5 aliases for the existing CuTile kernels."""
    registry = get_registry()

    if registry.get("rope_qwen3_5") is None:
        clone_patch(
            "rope_qwen3",
            name="rope_qwen3_5",
            description="CuTile RoPE for Qwen3.5 models",
            target_module="transformers.models.qwen3_5.modeling_qwen3_5",
            target_attr="apply_rotary_pos_emb",
            models=["qwen3_5"],
        )

    if registry.get("swiglu_qwen3_5") is None:
        clone_patch(
            "swiglu_qwen3",
            name="swiglu_qwen3_5",
            description="CuTile SwiGLU MLP with fast math for Qwen3.5",
            target_module="transformers.models.qwen3_5.modeling_qwen3_5",
            target_attr="Qwen3_5MLP",
            models=["qwen3_5"],
        )

    if registry.get("rms_norm_qwen3_5") is None:
        from .ops.rms_norm import rms_norm

        class CuTileQwen3_5RMSNorm(nn.Module):
            """Qwen3.5 RMSNorm wrapper backed by the existing CuTile kernel."""

            def __init__(self, dim: int, eps: float = 1e-6):
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.zeros(dim))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return rms_norm(x, 1.0 + self.weight, self.eps)

            def extra_repr(self) -> str:
                return f"{tuple(self.weight.shape)}, eps={self.eps}"

        register_patch(
            name="rms_norm_qwen3_5",
            description="CuTile RMSNorm for Qwen3.5 (preserves 1 + weight semantics)",
            target_module="transformers.models.qwen3_5.modeling_qwen3_5",
            target_attr="Qwen3_5RMSNorm",
            replacement=CuTileQwen3_5RMSNorm,
            has_backward=True,
            priority=10,
            models=["qwen3_5"],
        )


def _bastile_lce_forward_qwen3_5(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    logits_to_keep=0,
    **kwargs,
):
    from .ops.fused_linear_cross_entropy import fused_linear_cross_entropy

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    logits = None
    loss = None

    if self.training and labels is not None:
        shift_hidden = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = fused_linear_cross_entropy(
            shift_hidden,
            self.lm_head.weight,
            shift_labels,
            bias=getattr(self.lm_head, "bias", None),
            ignore_index=-100,
        )
    else:
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

    from transformers.modeling_outputs import CausalLMOutputWithPast

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


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
    model_type: str | None = None,
    **kwargs,
) -> list[str]:
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

    _ensure_qwen3_5_compat_patches_registered()

    registry = get_registry()

    patch_filter = {
        "rms_norm": rms_norm,
        "swiglu": swiglu,
        "rope": rope,
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
            if key in patch.name or patch.name.startswith(key.split("_")[0]):
                should_apply = enabled
                break

        if should_apply and _apply_patch(patch):
            applied.append(patch.name)

    # Apply fused linear cross-entropy if requested
    # This patches Qwen3/Qwen3.5 ForCausalLM.forward to skip logits materialization
    if fused_linear_cross_entropy:
        lce_targets = [
            ("qwen3", "transformers.models.qwen3.modeling_qwen3", "Qwen3ForCausalLM", "bastile_lce_forward"),
            (
                "qwen3_5",
                "transformers.models.qwen3_5.modeling_qwen3_5",
                "Qwen3_5ForCausalLM",
                "_bastile_lce_forward_qwen3_5",
            ),
        ]

        for target_model_type, module_path, class_name, forward_name in lce_targets:
            if model_type and model_type != target_model_type:
                continue
            try:
                module = importlib.import_module(module_path)
                replacement = globals()[forward_name]
                getattr(module, class_name).forward = replacement
                applied.append(f"fused_linear_cross_entropy_{target_model_type}")
                logger.info(
                    "Applied fused linear cross-entropy for %s (skips logits materialization)",
                    target_model_type,
                )
            except Exception as e:
                logger.warning(f"Could not apply fused linear cross-entropy for {target_model_type}: {e}")

    # Warmup kernels to avoid JIT overhead during training
    try:
        from .autotune import warmup_all_kernels

        warmup_all_kernels()
    except Exception as e:
        logger.debug(f"Warmup skipped: {e}")

    return applied


def apply_to_model(model: Any, **kwargs) -> list[str]:
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
    if hasattr(model, "config") and hasattr(model.config, "model_type"):
        model_type = model.config.model_type

    logger.info(f"Applying patches for model type: {model_type}")
    return apply(model_type=model_type, **kwargs)


def reset(names: list[str] | None = None) -> list[str]:
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


def get_patched_ops() -> list[str]:
    """Get list of currently patched operations."""
    return get_registry().get_applied()
