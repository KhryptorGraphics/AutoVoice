"""TensorRT engine rebuilding for fine-tuned models.

Task 7.4: Implement TensorRT engine rebuilding for fine-tuned models

Provides:
- Model checksum computation for version tracking
- Automatic rebuild detection after fine-tuning
- Engine caching with invalidation
- State persistence across sessions
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TRTEngineManager:
    """Manages TensorRT engines with automatic rebuilding for fine-tuned models.

    Tracks model checksums to detect when fine-tuning has changed parameters,
    triggering automatic ONNX export and TRT engine rebuild.
    """

    def __init__(
        self,
        cache_dir: str,
        precision: str = 'fp16',
    ):
        """Initialize TRT engine manager.

        Args:
            cache_dir: Directory for engine cache and metadata
            precision: TRT precision mode ('fp16' or 'fp32')
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.precision = precision

        # Engine tracking
        self.engines: Dict[str, str] = {}  # model_name -> engine_path
        self._registered_models: Dict[str, Dict[str, Any]] = {}
        self._current_engines: Dict[str, str] = {}

        # State file path
        self._state_file = self.cache_dir / 'engine_manager_state.json'

    def compute_model_checksum(self, model: nn.Module) -> str:
        """Compute SHA-256 checksum of model parameters.

        Creates a deterministic hash of all model parameters to detect
        when fine-tuning has modified the model.

        Args:
            model: PyTorch model to checksum

        Returns:
            64-character hex string (SHA-256 digest)
        """
        hasher = hashlib.sha256()

        # Hash each parameter's data
        for name, param in sorted(model.named_parameters()):
            # Include parameter name for ordering
            hasher.update(name.encode('utf-8'))
            # Include parameter data
            param_bytes = param.detach().cpu().numpy().tobytes()
            hasher.update(param_bytes)

        return hasher.hexdigest()

    def get_engine_path(self, model_name: str, model: nn.Module) -> Path:
        """Get engine path for a model, including checksum in filename.

        Args:
            model_name: Name identifier for the model
            model: PyTorch model instance

        Returns:
            Path to engine file (may not exist yet)
        """
        checksum = self.compute_model_checksum(model)
        short_checksum = checksum[:8]
        engine_filename = f"{model_name}_{short_checksum}_{self.precision}.engine"
        return self.cache_dir / engine_filename

    def needs_rebuild(self, model_name: str, model: nn.Module) -> bool:
        """Check if a model's TRT engine needs to be rebuilt.

        Args:
            model_name: Name identifier for the model
            model: Current PyTorch model instance

        Returns:
            True if engine needs to be rebuilt
        """
        current_checksum = self.compute_model_checksum(model)

        # Check if model is registered
        if model_name not in self._registered_models:
            return True

        registered = self._registered_models[model_name]

        # Check if checksum matches
        if registered.get('checksum') != current_checksum:
            return True

        # Check if engine file exists
        engine_path = registered.get('engine_path')
        if engine_path is None or not Path(engine_path).exists():
            return True

        return False

    def register_model(self, model_name: str, model: nn.Module) -> None:
        """Register a model for engine management.

        Args:
            model_name: Name identifier for the model
            model: PyTorch model instance
        """
        checksum = self.compute_model_checksum(model)

        self._registered_models[model_name] = {
            'checksum': checksum,
            'engine_path': None,
            'registered_at': time.time(),
        }

        logger.info(f"Registered model '{model_name}' with checksum {checksum[:8]}...")

    def _mark_engine_built(self, model_name: str, model: nn.Module) -> None:
        """Mark an engine as successfully built for a model.

        Args:
            model_name: Name identifier for the model
            model: PyTorch model instance
        """
        checksum = self.compute_model_checksum(model)
        engine_path = self.get_engine_path(model_name, model)

        if model_name not in self._registered_models:
            self._registered_models[model_name] = {}

        self._registered_models[model_name].update({
            'checksum': checksum,
            'engine_path': str(engine_path),
            'built_at': time.time(),
        })

        self._current_engines[model_name] = str(engine_path)

    def _store_engine_metadata(self, model_name: str, metadata: Dict[str, Any]) -> None:
        """Store engine metadata for a model.

        Args:
            model_name: Name identifier for the model
            metadata: Metadata dict to store
        """
        metadata_path = self.cache_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _get_engine_metadata(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get stored engine metadata for a model.

        Args:
            model_name: Name identifier for the model

        Returns:
            Metadata dict or None if not found
        """
        metadata_path = self.cache_dir / f"{model_name}_metadata.json"
        if not metadata_path.exists():
            return None

        with open(metadata_path, 'r') as f:
            return json.load(f)

    def cleanup_old_engines(self, keep_count: int = 3) -> List[str]:
        """Remove old engine files, keeping only the most recent.

        Args:
            keep_count: Number of engine versions to keep per model

        Returns:
            List of removed engine file paths
        """
        removed = []

        # Get all engine files
        engine_files = list(self.cache_dir.glob('*.engine'))

        # Group by model name (prefix before first underscore)
        model_engines: Dict[str, List[Path]] = {}
        for engine_path in engine_files:
            parts = engine_path.stem.split('_')
            if parts:
                model_name = parts[0]
                if model_name not in model_engines:
                    model_engines[model_name] = []
                model_engines[model_name].append(engine_path)

        # Keep only recent engines for each model
        for model_name, engines in model_engines.items():
            # Sort by modification time (newest first)
            engines.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            # Remove old engines
            for old_engine in engines[keep_count:]:
                # Don't remove current engine
                if str(old_engine) in self._current_engines.values():
                    continue
                try:
                    old_engine.unlink()
                    removed.append(str(old_engine))
                    logger.info(f"Removed old engine: {old_engine}")
                except Exception as e:
                    logger.warning(f"Failed to remove {old_engine}: {e}")

        return removed

    def save_state(self) -> None:
        """Save manager state to disk for persistence across sessions."""
        state = {
            'registered_models': self._registered_models,
            'current_engines': self._current_engines,
            'precision': self.precision,
        }

        with open(self._state_file, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved engine manager state to {self._state_file}")

    def load_state(self) -> bool:
        """Load manager state from disk.

        Returns:
            True if state was loaded successfully
        """
        if not self._state_file.exists():
            return False

        try:
            with open(self._state_file, 'r') as f:
                state = json.load(f)

            self._registered_models = state.get('registered_models', {})
            self._current_engines = state.get('current_engines', {})

            logger.info(f"Loaded engine manager state from {self._state_file}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
            return False

    def rebuild_engine(
        self,
        model_name: str,
        model: nn.Module,
        export_fn: callable,
        dynamic_shapes: Optional[Dict] = None,
    ) -> str:
        """Rebuild TRT engine for a model.

        Args:
            model_name: Name identifier for the model
            model: PyTorch model instance
            export_fn: Function to export model to ONNX (model, path) -> None
            dynamic_shapes: Optional dynamic shape profiles for TRT

        Returns:
            Path to built engine file

        Raises:
            RuntimeError: If build fails
        """
        from .trt_pipeline import TRTEngineBuilder

        # Export to ONNX
        onnx_path = self.cache_dir / f"{model_name}_temp.onnx"
        export_fn(model, str(onnx_path))

        # Build TRT engine
        engine_path = self.get_engine_path(model_name, model)
        builder = TRTEngineBuilder(precision=self.precision)

        try:
            builder.build_engine(str(onnx_path), str(engine_path), dynamic_shapes)
        finally:
            # Clean up ONNX file
            if onnx_path.exists():
                onnx_path.unlink()

        # Mark as built
        self._mark_engine_built(model_name, model)

        # Store metadata
        self._store_engine_metadata(model_name, {
            'model_name': model_name,
            'checksum': self.compute_model_checksum(model),
            'precision': self.precision,
            'build_time': time.time(),
            'engine_path': str(engine_path),
        })

        # Save state
        self.save_state()

        logger.info(f"Built TRT engine for '{model_name}': {engine_path}")
        return str(engine_path)
