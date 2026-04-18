"""Unified artifact manager for voice conversion model assets."""

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
ARTIFACT_PRIORITY = ("tensorrt", "full_model", "adapter")


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
    artifact_type: str = "adapter"
    artifact_types: List[str] = field(default_factory=list)


@dataclass
class ModelArtifact:
    """One profile artifact resolved by the manager."""

    profile_id: str
    artifact_type: str
    path: Path
    handle: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterManagerConfig:
    """Configuration for AdapterManager."""
    adapters_dir: Path = field(default_factory=lambda: resolve_trained_models_dir())
    profiles_dir: Path = field(default_factory=lambda: resolve_profiles_dir())
    tensorrt_dir: Optional[Path] = None
    cache_size: int = 5  # Number of adapters to keep in memory
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    auto_validate: bool = True


class AdapterCache:
    """LRU cache for loaded artifacts."""

    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self._cache: OrderedDict[str, Any] = OrderedDict()

    def _resolve_cache_key(self, cache_key: str) -> Optional[str]:
        if cache_key in self._cache:
            return cache_key

        if ":" in cache_key:
            return None

        suffix = f":{cache_key}"
        matches = [key for key in self._cache if key.endswith(suffix)]
        if len(matches) == 1:
            return matches[0]
        return None

    def get(self, cache_key: str) -> Optional[Any]:
        """Get item from cache, moving to end (most recently used)."""
        resolved_key = self._resolve_cache_key(cache_key)
        if resolved_key is not None:
            self._cache.move_to_end(resolved_key)
            return self._cache[resolved_key]
        return None

    def put(self, cache_key: str, value: Any) -> None:
        """Add item to cache, evicting oldest if necessary."""
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
        else:
            if len(self._cache) >= self.max_size:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
                logger.debug(f"Evicted cache entry {oldest}")
            self._cache[cache_key] = value

    def pop(self, cache_key: str) -> Optional[Any]:
        """Remove one cache entry if present."""
        resolved_key = self._resolve_cache_key(cache_key)
        if resolved_key is None:
            return None
        return self._cache.pop(resolved_key, None)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, cache_key: str) -> bool:
        return self._resolve_cache_key(cache_key) is not None


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
        self._active_artifact: Optional[ModelArtifact] = None
        self._active_cache_key: Optional[str] = None

        # Ensure directories exist
        self.config.adapters_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"AdapterManager initialized")
        logger.info(f"  Adapters dir: {self.config.adapters_dir}")
        if self.config.tensorrt_dir:
            logger.info(f"  TensorRT dir: {self.config.tensorrt_dir}")
        logger.info(f"  Cache size: {self.config.cache_size}")

    def list_available_adapters(self) -> List[str]:
        """List all available adapter profile IDs."""
        return sorted(
            artifact.profile_id
            for artifact in self.list_available_artifacts()
            if artifact.artifact_type == "adapter"
        )

    def _infer_profile_id_from_engine(self, path: Path) -> Optional[str]:
        stem = path.stem
        for marker in (
            "_tensorrt",
            "_engine",
            "_plan",
            "_trt",
            "_nvfp4",
            "_fp16",
            "_fp32",
            "_int8",
            "_int4",
        ):
            if stem.endswith(marker):
                candidate = stem[: -len(marker)]
                return candidate or None

        if "_" in stem:
            return stem.split("_", 1)[0] or None
        return stem or None

    def _collect_candidate_profile_ids(self, profile_id: Optional[str] = None) -> List[str]:
        if profile_id:
            return [profile_id]

        candidate_ids: List[str] = []

        def append_candidate(candidate: Optional[str]) -> None:
            if candidate and candidate not in candidate_ids:
                candidate_ids.append(candidate)

        for path in sorted(self.config.profiles_dir.glob("*.json")):
            append_candidate(path.stem)
        for path in sorted(self.config.profiles_dir.glob("*.npy")):
            append_candidate(path.stem)
        for path in sorted(self.config.adapters_dir.glob("*_adapter.pt")):
            append_candidate(path.stem.replace("_adapter", ""))
        for path in sorted(self.config.adapters_dir.glob("*_full_model.*")):
            append_candidate(path.stem.replace("_full_model", ""))
        for root in self._candidate_roots():
            if not root.exists():
                continue
            for pattern in ("*.engine", "*.plan", "*.trt"):
                for path in sorted(root.rglob(pattern)):
                    append_candidate(self._infer_profile_id_from_engine(path))

        return candidate_ids

    def _candidate_roots(self) -> List[Path]:
        roots = [self.config.adapters_dir]
        if self.config.tensorrt_dir is not None:
            roots.append(self.config.tensorrt_dir)
        return roots

    def get_full_model_path(self, profile_id: str) -> Optional[Path]:
        """Get the path to a profile full-model checkpoint."""
        for suffix in (".pt", ".pth"):
            path = self.config.adapters_dir / f"{profile_id}_full_model{suffix}"
            if path.exists():
                return path
        return None

    def get_tensorrt_engine_path(self, profile_id: str) -> Optional[Path]:
        """Get the first matching TensorRT engine for a profile."""
        patterns = (
            f"{profile_id}*.engine",
            f"{profile_id}*.plan",
            f"{profile_id}*.trt",
        )
        for root in self._candidate_roots():
            if not root.exists():
                continue
            for pattern in patterns:
                matches = sorted(root.rglob(pattern))
                if matches:
                    return matches[0]
        return None

    def get_embedding_path(self, profile_id: str) -> Path:
        """Get the profile embedding path."""
        return self.config.profiles_dir / f"{profile_id}.npy"

    def has_full_model(self, profile_id: str) -> bool:
        """Check if a full-model checkpoint exists for the profile."""
        return self.get_full_model_path(profile_id) is not None

    def has_tensorrt_engine(self, profile_id: str) -> bool:
        """Check if a TensorRT engine exists for the profile."""
        return self.get_tensorrt_engine_path(profile_id) is not None

    def get_available_artifact_types(self, profile_id: str) -> List[str]:
        """Return available artifact types for a profile in preferred order."""
        available = []
        if self.has_tensorrt_engine(profile_id):
            available.append("tensorrt")
        if self.has_full_model(profile_id):
            available.append("full_model")
        if self.has_adapter(profile_id):
            available.append("adapter")
        return available

    def list_available_artifacts(self, profile_id: Optional[str] = None) -> List[ModelArtifact]:
        """List all resolved artifacts, optionally for one profile."""
        artifacts: List[ModelArtifact] = []
        seen: set[tuple[str, str]] = set()
        candidate_ids = self._collect_candidate_profile_ids(profile_id)

        for candidate_id in candidate_ids:
            for artifact_type in self.get_available_artifact_types(candidate_id):
                path = self.get_artifact_path(candidate_id, artifact_type)
                key = (candidate_id, artifact_type)
                if path is not None and key not in seen:
                    artifacts.append(
                        ModelArtifact(
                            profile_id=candidate_id,
                            artifact_type=artifact_type,
                            path=path,
                        )
                    )
                    seen.add(key)

        artifacts.sort(
            key=lambda artifact: (
                artifact.profile_id,
                ARTIFACT_PRIORITY.index(artifact.artifact_type),
            )
        )
        return artifacts

    def has_adapter(self, profile_id: str) -> bool:
        """Check if an adapter exists for the given profile."""
        return self.get_adapter_path(profile_id) is not None

    def get_adapter_path(self, profile_id: str) -> Optional[Path]:
        """Get the path to an adapter file."""
        adapter_path = self.config.adapters_dir / f"{profile_id}_adapter.pt"
        return adapter_path if adapter_path.exists() else None

    def get_artifact_path(self, profile_id: str, artifact_type: str = "auto") -> Optional[Path]:
        """Resolve the artifact path for a profile and artifact type."""
        resolved_type = self._resolve_artifact_type(profile_id, artifact_type)
        if resolved_type == "adapter":
            return self.get_adapter_path(profile_id)
        if resolved_type == "full_model":
            return self.get_full_model_path(profile_id)
        if resolved_type == "tensorrt":
            return self.get_tensorrt_engine_path(profile_id)
        raise ValueError(f"Unsupported artifact type: {artifact_type}")

    def _resolve_artifact_type(self, profile_id: str, artifact_type: str) -> str:
        if artifact_type != "auto":
            return artifact_type

        available = self.get_available_artifact_types(profile_id)
        if not available:
            raise FileNotFoundError(f"No model artifact found for profile: {profile_id}")
        return available[0]

    def _cache_key(self, profile_id: str, artifact_type: str) -> str:
        return f"{artifact_type}:{profile_id}"

    def _load_tensorrt_engine(self, engine_path: Path) -> Any:
        """Load a TensorRT engine or fall back to raw bytes if deserialization fails."""
        with engine_path.open("rb") as handle:
            serialized = handle.read()

        try:
            import tensorrt as trt

            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            engine = runtime.deserialize_cuda_engine(serialized)
            return engine if engine is not None else serialized
        except Exception as exc:  # pragma: no cover - depends on system TRT
            logger.warning("Failed to deserialize TensorRT engine %s: %s", engine_path, exc)
            return serialized

    def load_artifact(
        self,
        profile_id: str,
        artifact_type: str = "auto",
        use_cache: bool = True,
    ) -> ModelArtifact:
        """Load any supported artifact type for a profile."""
        resolved_type = self._resolve_artifact_type(profile_id, artifact_type)
        cache_key = self._cache_key(profile_id, resolved_type)

        if use_cache:
            cached = self._cache.get(cache_key)
            if isinstance(cached, ModelArtifact):
                logger.debug("Artifact %s loaded from cache", cache_key)
                return cached

        artifact_path = self.get_artifact_path(profile_id, resolved_type)
        if artifact_path is None:
            if resolved_type == "adapter":
                raise FileNotFoundError(f"No adapter found for profile: {profile_id}")
            raise FileNotFoundError(
                f"No {resolved_type} artifact found for profile: {profile_id}"
            )

        if resolved_type == "tensorrt":
            handle = self._load_tensorrt_engine(artifact_path)
        else:
            handle = torch.load(artifact_path, map_location=self.device, weights_only=False)
            if resolved_type == "adapter" and self.config.auto_validate:
                self._validate_adapter(handle, profile_id)

        artifact = ModelArtifact(
            profile_id=profile_id,
            artifact_type=resolved_type,
            path=artifact_path,
            handle=handle,
            metadata={"artifact_types": self.get_available_artifact_types(profile_id)},
        )
        if use_cache:
            self._cache.put(cache_key, artifact)
        return artifact

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
        artifact = self.load_artifact(
            profile_id,
            artifact_type="adapter",
            use_cache=use_cache,
        )
        return artifact.handle

    def _validate_adapter(self, state_dict: Dict[str, torch.Tensor], profile_id: str) -> None:
        """Validate adapter state dict structure."""
        if not state_dict:
            raise ValueError(f"Empty adapter state dict for {profile_id}")

        # Check for expected LoRA structure
        has_lora_a = any("lora_A" in k for k in state_dict.keys())
        has_lora_b = any("lora_B" in k for k in state_dict.keys())

        if not (has_lora_a and has_lora_b):
            logger.warning(f"Adapter {profile_id} may not be a standard LoRA adapter")

    def load_speaker_embedding(
        self,
        profile_id: str,
        as_tensor: bool = False,
        normalize: bool = True,
    ) -> Any:
        """Load a profile speaker embedding from the voice profile store."""
        import numpy as np

        embedding_path = self.get_embedding_path(profile_id)
        if not embedding_path.exists():
            raise FileNotFoundError(
                f"No speaker embedding found for profile: {profile_id}"
            )

        embedding = np.asarray(np.load(embedding_path), dtype=np.float32)
        if embedding.shape != (256,):
            raise ValueError(
                f"Invalid embedding shape: {embedding.shape}, expected (256,)"
            )

        if normalize:
            norm = float(np.linalg.norm(embedding))
            if abs(norm - 1.0) > 0.01:
                logger.warning(
                    "Speaker embedding not L2-normalized (norm=%.4f), normalizing",
                    norm,
                )
                embedding = embedding / max(norm, 1e-8)

        if as_tensor:
            return torch.from_numpy(embedding).to(self.device)
        return embedding

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

            artifact_types = self.get_available_artifact_types(profile_id)
            artifact_type = artifact_types[0] if artifact_types else "adapter"
            artifact_path = self.get_artifact_path(profile_id, artifact_type) or (
                self.config.adapters_dir / f"{profile_id}_adapter.pt"
            )

            info = AdapterInfo(
                profile_id=profile_id,
                profile_name=profile_data.get("name", "Unknown"),
                path=artifact_path,
                version=profile_data.get("adapter_version", "1.0"),
                target_modules=profile_data.get("adapter_target_modules", ["content_proj", "output"]),
                rank=profile_data.get("adapter_rank", 8),
                alpha=profile_data.get("adapter_alpha", 16),
                created_at=profile_data.get("created_at", ""),
                sample_count=profile_data.get("sample_count", 0),
                training_epochs=profile_data.get("training_epochs", 0),
                loss_final=profile_data.get("loss_final", 0.0),
                artifact_type=artifact_type,
                artifact_types=artifact_types,
            )

            self._adapter_info[profile_id] = info
            return info

        except Exception as e:
            logger.warning(f"Failed to load adapter info for {profile_id}: {e}")
            return None

    def swap_artifact(
        self,
        profile_id: str,
        artifact_type: str = "auto",
        use_cache: bool = True,
    ) -> ModelArtifact:
        """Hot-swap the active artifact without restarting the process."""
        resolved_type = self._resolve_artifact_type(profile_id, artifact_type)
        next_key = self._cache_key(profile_id, resolved_type)
        if self._active_cache_key == next_key and self._active_artifact is not None:
            return self._active_artifact

        self.release_active_artifact()
        artifact = self.load_artifact(profile_id, resolved_type, use_cache=use_cache)
        self._active_artifact = artifact
        self._active_cache_key = next_key
        logger.info(
            "Activated %s artifact for profile %s from %s",
            artifact.artifact_type,
            profile_id,
            artifact.path,
        )
        return artifact

    def release_active_artifact(self) -> None:
        """Release the currently active artifact and free GPU-backed memory."""
        if self._active_artifact is None:
            return

        if self._active_cache_key is not None:
            self._cache.pop(self._active_cache_key)

        self._active_artifact = None
        self._active_cache_key = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Released active artifact")

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
        self._active_artifact = None
        self._active_cache_key = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Adapter cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_adapters": len(self._cache),
            "max_cache_size": self.config.cache_size,
            "cached_info": len(self._adapter_info),
            "active_artifact_type": (
                self._active_artifact.artifact_type if self._active_artifact else None
            ),
            "active_profile_id": (
                self._active_artifact.profile_id if self._active_artifact else None
            ),
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
