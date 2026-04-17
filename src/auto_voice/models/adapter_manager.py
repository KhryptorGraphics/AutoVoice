"""Unified Adapter Manager for Voice Conversion Pipelines.

Provides a single interface for loading, caching, and applying LoRA adapters
across both REALTIME and QUALITY pipelines.

Features:
- Profile-based adapter loading
- LRU caching for frequently used adapters
- Validation of adapter compatibility
- Integration with both pipeline types
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict
import json

import torch
import torch.nn as nn

from auto_voice.storage.paths import resolve_profiles_dir, resolve_trained_models_dir

logger = logging.getLogger(__name__)


@dataclass
class AdapterInfo:
    """Information about a loaded adapter."""
    profile_id: str
    profile_name: str
    path: Path
    version: str
    target_modules: List[str]
    rank: int
    alpha: int
    created_at: str
    sample_count: int
    training_epochs: int
    loss_final: float


@dataclass
class AdapterManagerConfig:
    """Configuration for AdapterManager."""
    adapters_dir: Path = field(default_factory=lambda: resolve_trained_models_dir())
    profiles_dir: Path = field(default_factory=lambda: resolve_profiles_dir())
    cache_size: int = 5  # Number of adapters to keep in memory
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    auto_validate: bool = True


class AdapterCache:
    """LRU cache for loaded adapters."""

    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self._cache: OrderedDict[str, Dict[str, torch.Tensor]] = OrderedDict()

    def get(self, profile_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get adapter from cache, moving to end (most recently used)."""
        if profile_id in self._cache:
            self._cache.move_to_end(profile_id)
            return self._cache[profile_id]
        return None

    def put(self, profile_id: str, adapter_state: Dict[str, torch.Tensor]) -> None:
        """Add adapter to cache, evicting oldest if necessary."""
        if profile_id in self._cache:
            self._cache.move_to_end(profile_id)
        else:
            if len(self._cache) >= self.max_size:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
                logger.debug(f"Evicted adapter {oldest} from cache")
            self._cache[profile_id] = adapter_state

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, profile_id: str) -> bool:
        return profile_id in self._cache


class AdapterManager:
    """Unified manager for voice adapter loading and application.

    Provides a single interface for both REALTIME and QUALITY pipelines
    to load and apply LoRA adapters trained on specific voice profiles.

    Usage:
        manager = AdapterManager()

        # Load adapter for a profile
        adapter = manager.load_adapter("profile-uuid")

        # Apply to model
        manager.apply_adapter(model, adapter)

        # Get profile info
        info = manager.get_adapter_info("profile-uuid")
    """

    def __init__(self, config: Optional[AdapterManagerConfig] = None):
        self.config = config or AdapterManagerConfig()
        self.device = torch.device(self.config.device)
        self._cache = AdapterCache(max_size=self.config.cache_size)
        self._adapter_info: Dict[str, AdapterInfo] = {}

        # Ensure directories exist
        self.config.adapters_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"AdapterManager initialized")
        logger.info(f"  Adapters dir: {self.config.adapters_dir}")
        logger.info(f"  Cache size: {self.config.cache_size}")

    def list_available_adapters(self) -> List[str]:
        """List all available adapter profile IDs."""
        adapters = []
        for path in self.config.adapters_dir.glob("*_adapter.pt"):
            profile_id = path.stem.replace("_adapter", "")
            adapters.append(profile_id)
        return adapters

    def has_adapter(self, profile_id: str) -> bool:
        """Check if an adapter exists for the given profile."""
        adapter_path = self.config.adapters_dir / f"{profile_id}_adapter.pt"
        return adapter_path.exists()

    def get_adapter_path(self, profile_id: str) -> Optional[Path]:
        """Get the path to an adapter file."""
        adapter_path = self.config.adapters_dir / f"{profile_id}_adapter.pt"
        return adapter_path if adapter_path.exists() else None

    def load_adapter(
        self,
        profile_id: str,
        use_cache: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Load adapter weights for a profile.

        Args:
            profile_id: UUID of the voice profile
            use_cache: Whether to use/update the LRU cache

        Returns:
            Dictionary of adapter state dict

        Raises:
            FileNotFoundError: If adapter doesn't exist
            ValueError: If adapter is invalid
        """
        # Check cache first
        if use_cache:
            cached = self._cache.get(profile_id)
            if cached is not None:
                logger.debug(f"Adapter {profile_id} loaded from cache")
                return cached

        # Load from disk
        adapter_path = self.config.adapters_dir / f"{profile_id}_adapter.pt"
        if not adapter_path.exists():
            raise FileNotFoundError(f"No adapter found for profile: {profile_id}")

        logger.info(f"Loading adapter from {adapter_path}")
        # weights_only=False for our trusted trained adapters (contain numpy scalars)
        state_dict = torch.load(adapter_path, map_location=self.device, weights_only=False)

        # Validate if configured
        if self.config.auto_validate:
            self._validate_adapter(state_dict, profile_id)

        # Cache the loaded adapter
        if use_cache:
            self._cache.put(profile_id, state_dict)

        return state_dict

    def _validate_adapter(self, state_dict: Dict[str, torch.Tensor], profile_id: str) -> None:
        """Validate adapter state dict structure."""
        if not state_dict:
            raise ValueError(f"Empty adapter state dict for {profile_id}")

        # Check for expected LoRA structure
        has_lora_a = any("lora_A" in k for k in state_dict.keys())
        has_lora_b = any("lora_B" in k for k in state_dict.keys())

        if not (has_lora_a and has_lora_b):
            logger.warning(f"Adapter {profile_id} may not be a standard LoRA adapter")

    def get_adapter_info(self, profile_id: str) -> Optional[AdapterInfo]:
        """Get metadata about an adapter."""
        # Check cached info
        if profile_id in self._adapter_info:
            return self._adapter_info[profile_id]

        # Load from profile metadata
        profile_path = self.config.profiles_dir / f"{profile_id}.json"
        if not profile_path.exists():
            return None

        try:
            with open(profile_path) as f:
                profile_data = json.load(f)

            adapter_path = self.config.adapters_dir / f"{profile_id}_adapter.pt"

            info = AdapterInfo(
                profile_id=profile_id,
                profile_name=profile_data.get("name", "Unknown"),
                path=adapter_path,
                version=profile_data.get("adapter_version", "1.0"),
                target_modules=profile_data.get("adapter_target_modules", ["content_proj", "output"]),
                rank=profile_data.get("adapter_rank", 8),
                alpha=profile_data.get("adapter_alpha", 16),
                created_at=profile_data.get("created_at", ""),
                sample_count=profile_data.get("sample_count", 0),
                training_epochs=profile_data.get("training_epochs", 0),
                loss_final=profile_data.get("loss_final", 0.0),
            )

            self._adapter_info[profile_id] = info
            return info

        except Exception as e:
            logger.warning(f"Failed to load adapter info for {profile_id}: {e}")
            return None

    def apply_adapter(
        self,
        model: nn.Module,
        adapter_state: Dict[str, torch.Tensor],
        target_modules: Optional[List[str]] = None
    ) -> None:
        """Apply adapter weights to a model.

        This method injects the LoRA weights into compatible layers
        of the target model.

        Args:
            model: The model to apply adapter to
            adapter_state: The adapter state dict
            target_modules: Specific modules to target (optional)
        """
        applied_count = 0

        for name, param in adapter_state.items():
            # Find matching parameter in model
            parts = name.split(".")

            # Navigate to the parameter
            obj = model
            found = True
            for part in parts[:-1]:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                elif hasattr(obj, 'lora_adapters') and part in obj.lora_adapters:
                    obj = obj.lora_adapters[part]
                else:
                    found = False
                    break

            if found and hasattr(obj, parts[-1]):
                target_param = getattr(obj, parts[-1])
                if isinstance(target_param, nn.Parameter):
                    target_param.data.copy_(param.to(target_param.device))
                    applied_count += 1

        if applied_count > 0:
            logger.info(f"Applied {applied_count} adapter parameters to model")
        else:
            logger.warning("No adapter parameters were applied - check module names")

    def remove_adapter(self, model: nn.Module) -> None:
        """Remove adapter from a model, restoring original weights."""
        if hasattr(model, 'lora_adapters'):
            # Zero out LoRA contributions
            for name, adapter in model.lora_adapters.items():
                if hasattr(adapter, 'lora_B'):
                    adapter.lora_B.data.zero_()
            logger.info("Adapter contributions zeroed out")

    def save_adapter(
        self,
        profile_id: str,
        model: nn.Module,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save adapter weights from a model.

        Args:
            profile_id: Profile ID to save adapter for
            model: Model with trained adapter weights
            metadata: Optional metadata to include

        Returns:
            Path to saved adapter file
        """
        adapter_state = {}

        # Extract LoRA parameters
        if hasattr(model, 'lora_adapters'):
            for name, adapter in model.lora_adapters.items():
                for param_name, param in adapter.named_parameters():
                    full_name = f"lora_adapters.{name}.{param_name}"
                    adapter_state[full_name] = param.data.cpu()

        # Also check for any other trainable parameters marked as adapter
        for name, param in model.named_parameters():
            if param.requires_grad and "lora" in name.lower():
                if name not in adapter_state:
                    adapter_state[name] = param.data.cpu()

        if not adapter_state:
            raise ValueError("No adapter parameters found in model")

        # Save
        adapter_path = self.config.adapters_dir / f"{profile_id}_adapter.pt"
        torch.save(adapter_state, adapter_path)

        logger.info(f"Saved adapter with {len(adapter_state)} parameters to {adapter_path}")

        # Update profile metadata if provided
        if metadata:
            profile_path = self.config.profiles_dir / f"{profile_id}.json"
            if profile_path.exists():
                with open(profile_path) as f:
                    profile_data = json.load(f)
                profile_data.update(metadata)
                with open(profile_path, 'w') as f:
                    json.dump(profile_data, f, indent=2)

        return adapter_path

    def clear_cache(self) -> None:
        """Clear the adapter cache."""
        self._cache.clear()
        self._adapter_info.clear()
        logger.info("Adapter cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_adapters": len(self._cache),
            "max_cache_size": self.config.cache_size,
            "cached_info": len(self._adapter_info),
        }


# Convenience functions for pipeline integration

def load_adapter_for_profile(profile_id: str, device: str = "cuda") -> Dict[str, torch.Tensor]:
    """Convenience function to load an adapter for a profile.

    Args:
        profile_id: UUID of the voice profile
        device: Device to load adapter to

    Returns:
        Adapter state dict
    """
    manager = AdapterManager(AdapterManagerConfig(device=device))
    return manager.load_adapter(profile_id)


def get_trained_profiles() -> List[Tuple[str, str]]:
    """Get list of profiles that have trained adapters.

    Returns:
        List of (profile_id, profile_name) tuples
    """
    manager = AdapterManager()
    trained = []

    for profile_id in manager.list_available_adapters():
        info = manager.get_adapter_info(profile_id)
        name = info.profile_name if info else profile_id
        trained.append((profile_id, name))

    return trained


# Global instance for shared use
_global_manager: Optional[AdapterManager] = None


def get_adapter_manager() -> AdapterManager:
    """Get or create global AdapterManager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = AdapterManager()
    return _global_manager
