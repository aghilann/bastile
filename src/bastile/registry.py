"""
Registry for CuTile kernel patches.

Each patch is registered with:
- A unique name (e.g., 'rms_norm', 'swiglu')
- The target module/function to patch
- The replacement implementation
- Whether backward pass is supported
"""

from dataclasses import dataclass, field
from typing import Callable, Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class PatchInfo:
    """Information about a registered patch."""
    name: str
    description: str
    target_module: str  # e.g., "transformers.models.llama.modeling_llama"
    target_attr: str    # e.g., "LlamaRMSNorm"
    replacement: Any    # The replacement class/function
    has_backward: bool  # Whether backward pass is implemented
    original: Any = None  # Stores original for reset
    is_applied: bool = False
    priority: int = 0   # Higher priority patches are applied first
    models: List[str] = field(default_factory=list)  # Which model types this applies to


class PatchRegistry:
    """Central registry for all kernel patches."""
    
    def __init__(self):
        self._patches: Dict[str, PatchInfo] = {}
        self._applied: List[str] = []
    
    def register(
        self,
        name: str,
        description: str,
        target_module: str,
        target_attr: str,
        replacement: Any,
        has_backward: bool = True,
        priority: int = 0,
        models: Optional[List[str]] = None,
    ) -> None:
        """Register a new patch."""
        if name in self._patches:
            logger.warning(f"Patch '{name}' already registered, overwriting")
        
        self._patches[name] = PatchInfo(
            name=name,
            description=description,
            target_module=target_module,
            target_attr=target_attr,
            replacement=replacement,
            has_backward=has_backward,
            priority=priority,
            models=models or [],
        )
        logger.debug(f"Registered patch: {name}")
    
    def get(self, name: str) -> Optional[PatchInfo]:
        """Get a patch by name."""
        return self._patches.get(name)
    
    def list_all(self) -> List[str]:
        """List all registered patch names."""
        return list(self._patches.keys())
    
    def list_with_backward(self) -> List[str]:
        """List patches that have backward pass support."""
        return [name for name, patch in self._patches.items() if patch.has_backward]
    
    def get_for_model(self, model_type: str) -> List[PatchInfo]:
        """Get patches applicable to a specific model type."""
        applicable = []
        for patch in self._patches.values():
            if not patch.models or model_type in patch.models:
                applicable.append(patch)
        return sorted(applicable, key=lambda p: -p.priority)
    
    def mark_applied(self, name: str, original: Any) -> None:
        """Mark a patch as applied and store the original."""
        if name in self._patches:
            self._patches[name].original = original
            self._patches[name].is_applied = True
            if name not in self._applied:
                self._applied.append(name)
    
    def mark_reset(self, name: str) -> None:
        """Mark a patch as reset."""
        if name in self._patches:
            self._patches[name].is_applied = False
            if name in self._applied:
                self._applied.remove(name)
    
    def get_applied(self) -> List[str]:
        """Get list of currently applied patches."""
        return self._applied.copy()


# Global registry instance
_registry = PatchRegistry()


def register_patch(
    name: str,
    description: str,
    target_module: str,
    target_attr: str,
    replacement: Any,
    has_backward: bool = True,
    priority: int = 0,
    models: Optional[List[str]] = None,
) -> None:
    """Register a patch in the global registry."""
    _registry.register(
        name=name,
        description=description,
        target_module=target_module,
        target_attr=target_attr,
        replacement=replacement,
        has_backward=has_backward,
        priority=priority,
        models=models,
    )


def list_patches() -> List[str]:
    """List all registered patches."""
    return _registry.list_all()


def get_registry() -> PatchRegistry:
    """Get the global patch registry."""
    return _registry
