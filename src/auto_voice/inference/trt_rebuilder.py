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

    Why rebuilding is needed:
    - LoRA fine-tuning modifies model parameters
    - TensorRT engines are static (baked with specific weights)
    - Using old engine with new weights causes incorrect inference
    - Each parameter change requires new ONNX export + TRT build

    How rebuild detection works:
    - Computes SHA-256 checksum of all model parameters
    - Compares against checksum from last build
    - Triggers rebuild if mismatch detected

    Usage:
        manager = TRTEngineManager(cache_dir='engines/', precision='fp16')
        manager.load_state()  # Restore from previous session
        manager.register_model('encoder', encoder_model)

        if manager.needs_rebuild('encoder', encoder_model):
            engine_path = manager.rebuild_engine('encoder', encoder_model, export_fn)
    """

    def __init__(
        self,
        cache_dir: str,
        precision: str = 'fp16',
    ):
        """Initialize TRT engine manager for automatic rebuild tracking.

        Sets up cache directory and state tracking for TensorRT engines.
        Call load_state() after initialization to restore previous session.

        Args:
            cache_dir: Directory for engine cache and metadata (created if missing)
            precision: TRT precision mode - 'fp16' for speed or 'fp32' for accuracy
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
        """Compute SHA-256 checksum of model parameters for change detection.

        Creates a deterministic hash of all model parameters to detect
        when fine-tuning has modified the model. The hash includes:
        - Parameter names (for ordering)
        - Parameter data (as bytes)

        Any change to model weights (LoRA fine-tuning, training, etc.) will
        produce a different checksum, triggering engine rebuild.

        Args:
            model: PyTorch model to checksum (on any device)

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

        Generates unique filename based on model parameters and precision:
        {model_name}_{checksum[:8]}_{precision}.engine

        The checksum ensures different model versions get different engine files,
        preventing accidental use of stale engines after fine-tuning.

        Args:
            model_name: Name identifier for the model
            model: PyTorch model instance (checksum computed from parameters)

        Returns:
            Path object to engine file (file may not exist yet)
        """
        checksum = self.compute_model_checksum(model)
        short_checksum = checksum[:8]
        engine_filename = f"{model_name}_{short_checksum}_{self.precision}.engine"
        return self.cache_dir / engine_filename

    def needs_rebuild(self, model_name: str, model: nn.Module) -> bool:
        """Check if a model's TRT engine needs to be rebuilt.

        Rebuilding is needed when:
        - Model is not yet registered (first-time use)
        - Model parameters have changed (fine-tuning detected via checksum)
        - Engine file is missing or was deleted

        This prevents using stale engines after LoRA fine-tuning or other
        parameter updates that would cause inference errors.

        Args:
            model_name: Name identifier for the model
            model: Current PyTorch model instance

        Returns:
            True if engine needs to be rebuilt, False if current engine is valid
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
        """Register a model for engine management and rebuild tracking.

        Establishes baseline checksum for detecting future fine-tuning changes.
        Must be called before first rebuild to enable automatic invalidation
        when model parameters change.

        Args:
            model_name: Name identifier for the model (e.g., 'encoder', 'vocoder')
            model: PyTorch model instance in current state
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

        Updates internal tracking to prevent redundant rebuilds. Stores:
        - Current model checksum (for detecting future fine-tuning)
        - Engine file path (for validation)
        - Build timestamp (for cleanup decisions)

        This allows needs_rebuild() to detect when an engine is valid.

        Args:
            model_name: Name identifier for the model
            model: PyTorch model instance (as built into engine)
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
        """Store engine metadata for a model to JSON file.

        Saves build information (checksum, precision, timestamp) separately from
        state file for debugging and auditing. Useful for tracking which model
        version produced a given engine.

        Args:
            model_name: Name identifier for the model
            metadata: Metadata dict to store (must be JSON-serializable)
        """
        metadata_path = self.cache_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _get_engine_metadata(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get stored engine metadata for a model from JSON file.

        Retrieves build information (checksum, precision, timestamp) for debugging
        and verification. Returns None if metadata file doesn't exist (engine never
        built or metadata deleted).

        Args:
            model_name: Name identifier for the model

        Returns:
            Metadata dict with keys: model_name, checksum, precision, build_time,
            engine_path; or None if no metadata file found
        """
        metadata_path = self.cache_dir / f"{model_name}_metadata.json"
        if not metadata_path.exists():
            return None

        with open(metadata_path, 'r') as f:
            return json.load(f)

    def cleanup_old_engines(self, keep_count: int = 3) -> List[str]:
        """Remove old engine files, keeping only the most recent versions.

        Why cleanup is needed:
        - Each fine-tuning iteration creates a new engine (different checksum)
        - Engine files are large (100MB-1GB+) and accumulate quickly
        - Only the latest N versions are needed for rollback

        Protects current engines (referenced in self._current_engines) from deletion
        even if they exceed the keep_count limit.

        Args:
            keep_count: Number of engine versions to keep per model (default: 3)

        Returns:
            List of removed engine file paths as strings
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
        """Save manager state to disk for persistence across sessions.

        Persists:
        - Registered models and their checksums
        - Current engine paths
        - Precision settings

        This prevents redundant rebuilds after server restarts by maintaining
        the model-to-engine mapping and invalidation state.
        """
        state = {
            'registered_models': self._registered_models,
            'current_engines': self._current_engines,
            'precision': self.precision,
        }

        with open(self._state_file, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved engine manager state to {self._state_file}")

    def load_state(self) -> bool:
        """Load manager state from disk to restore session.

        Restores model checksums and engine paths from previous session,
        allowing immediate validation without re-registration. If state
        file doesn't exist (first run), returns False gracefully.

        Returns:
            True if state was loaded successfully, False if no saved state found
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
        """Rebuild TRT engine for a model after fine-tuning or invalidation.

        Call this when needs_rebuild() returns True to regenerate the optimized
        TensorRT engine. This is required after:
        - LoRA fine-tuning changes model parameters
        - Model architecture updates
        - Precision mode changes
        - Engine file corruption or deletion

        The rebuild process:
        1. Exports PyTorch model to ONNX (temporary file)
        2. Builds TensorRT engine with specified precision
        3. Updates checksums and metadata
        4. Saves state for persistence

        Args:
            model_name: Name identifier for the model
            model: PyTorch model instance (current state)
            export_fn: Function to export model to ONNX (model, path) -> None
            dynamic_shapes: Optional dynamic shape profiles for TRT optimization

        Returns:
            Absolute path to built engine file as string

        Raises:
            RuntimeError: If ONNX export or TRT build fails
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
