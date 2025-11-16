"""
Model Deployment Service for AutoVoice
Handles dynamic loading, hot-swapping, and A/B testing of ML models
"""
import os
import json
import logging
import hashlib
import time
from typing import Dict, Any, Optional, List, Tuple, Callable
from pathlib import Path
from contextlib import contextmanager
import threading
import queue

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    trt = None  # type: ignore
    cuda = None  # type: ignore

logger = logging.getLogger(__name__)


class ModelDeploymentService:
    """Service for deploying and managing custom ML models with hot-swapping."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models_dir = Path(config.get('models_dir', 'models'))
        self.cache_dir = Path(config.get('cache_dir', 'models/cache'))
        self.metrics_dir = Path(config.get('metrics_dir', 'logs/model_metrics'))

        # Model registry: name -> model_info
        self.model_registry: Dict[str, Dict[str, Any]] = {}

        # Active model instances: name -> model_instance
        self.active_models: Dict[str, Any] = {}

        # Model performance tracking
        self.performance_stats: Dict[str, Dict[str, Any]] = {}

        # Deployment history
        self.deployment_history: List[Dict[str, Any]] = []

        # A/B testing configurations
        self.ab_tests: Dict[str, Dict[str, Any]] = {}

        # Thread safety
        self.deployment_lock = threading.RLock()

        # Initialize directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Auto-discovery of models
        self._discover_models()

        logger.info(f"Model Deployment Service initialized with {len(self.model_registry)} registered models")

    def _discover_models(self):
        """Auto-discover available models from filesystem."""
        if not self.models_dir.exists():
            logger.warning(f"Models directory {self.models_dir} does not exist")
            return

        # Scan for model configurations
        for config_file in self.models_dir.rglob('model_config.json'):
            try:
                with open(config_file, 'r') as f:
                    model_config = json.load(f)

                model_name = model_config.get('name')
                if model_name:
                    model_info = {
                        'config': model_config,
                        'config_path': str(config_file),
                        'type': model_config.get('type', 'pytorch'),
                        'framework': model_config.get('framework', 'torch'),
                        'input_shape': model_config.get('input_shape'),
                        'output_shape': model_config.get('output_shape'),
                        'version': model_config.get('version', '1.0.0'),
                        'capabilities': model_config.get('capabilities', []),
                        'last_modified': os.path.getmtime(config_file),
                        'hash': self._compute_config_hash(model_config)
                    }

                    self.model_registry[model_name] = model_info
                    logger.debug(f"Discovered model: {model_name}")

            except Exception as e:
                logger.error(f"Failed to load model config {config_file}: {e}")

    def deploy_model(self, model_name: str, model_data: Optional[bytes] = None,
                    auto_activate: bool = True) -> bool:
        """Deploy a model by name or from uploaded data.

        Args:
            model_name: Name of the model to deploy
            model_data: Binary model data (optional, will load from disk if not provided)
            auto_activate: Whether to automatically activate the model

        Returns:
            Success status
        """
        with self.deployment_lock:
            try:
                logger.info(f"Deploying model: {model_name}")

                # Get model info from registry
                if model_name not in self.model_registry:
                    if not model_data:
                        raise ValueError(f"Model {model_name} not found in registry and no data provided")
                    # Register new model
                    self._register_new_model(model_name, model_data)
                    model_info = self.model_registry[model_name]
                else:
                    model_info = self.model_registry[model_name]

                # Load model weights/data if not provided
                if not model_data:
                    model_data = self._load_model_data(model_name, model_info)

                # Validate model data
                if not self._validate_model_data(model_data, model_info):
                    raise ValueError(f"Invalid model data for {model_name}")

                # Cache model in memory
                model_instance = self._instantiate_model(model_data, model_info)

                # Validate model instance
                if not self._validate_model_instance(model_instance, model_info):
                    raise ValueError(f"Model instance validation failed for {model_name}")

                # Add to active models
                self.active_models[model_name] = {
                    'instance': model_instance,
                    'info': model_info,
                    'deployed_at': time.time(),
                    'status': 'loaded'
                }

                # Initialize performance tracking
                self.performance_stats[model_name] = {
                    'inference_count': 0,
                    'total_inference_time': 0.0,
                    'latency_samples': [],
                    'accuracy_samples': [],
                    'memory_usage': {},
                    'errors': 0
                }

                # Log deployment
                deployment_record = {
                    'model_name': model_name,
                    'timestamp': time.time(),
                    'version': model_info.get('version'),
                    'type': model_info.get('type'),
                    'auto_activated': auto_activate
                }
                self.deployment_history.append(deployment_record)

                # Auto-activate if requested
                if auto_activate:
                    self.activate_model(model_name)

                logger.info(f"Successfully deployed model: {model_name}")
                return True

            except Exception as e:
                logger.error(f"Failed to deploy model {model_name}: {e}")
                if model_name in self.active_models:
                    del self.active_models[model_name]
                return False

    def activate_model(self, model_name: str) -> bool:
        """Activate a deployed model for inference."""
        with self.deployment_lock:
            if model_name not in self.active_models:
                logger.error(f"Model {model_name} is not deployed")
                return False

            try:
                model_instance = self.active_models[model_name]['instance']

                # Warm up model
                self._warmup_model(model_instance)

                # Mark as active
                self.active_models[model_name]['status'] = 'active'

                # Update deployment record
                for record in reversed(self.deployment_history):
                    if record['model_name'] == model_name and record.get('activated_at') is None:
                        record['activated_at'] = time.time()
                        break

                logger.info(f"Activated model: {model_name}")
                return True

            except Exception as e:
                logger.error(f"Failed to activate model {model_name}: {e}")
                self.active_models[model_name]['status'] = 'error'
                return False

    def deactivate_model(self, model_name: str) -> bool:
        """Deactivate a model."""
        with self.deployment_lock:
            if model_name not in self.active_models:
                return True

            try:
                # Mark as inactive
                self.active_models[model_name]['status'] = 'inactive'

                # Clean up resources
                self._cleanup_model_resources(model_name)

                # Log deactivation
                deactivation_record = {
                    'model_name': model_name,
                    'timestamp': time.time(),
                    'action': 'deactivated'
                }
                self.deployment_history.append(deactivation_record)

                logger.info(f"Deactivated model: {model_name}")
                return True

            except Exception as e:
                logger.error(f"Failed to deactivate model {model_name}: {e}")
                return False

    def run_inference(self, model_name: str, inputs: Dict[str, Any],
                     **kwargs) -> Dict[str, Any]:
        """Run inference on a deployed model."""
        start_time = time.time()

        if model_name not in self.active_models:
            raise ValueError(f"Model {model_name} is not active")

        model_data = self.active_models[model_name]
        if model_data['status'] != 'active':
            raise ValueError(f"Model {model_name} is not active (status: {model_data['status']})")

        try:
            model_instance = model_data['instance']

            # Run inference
            with torch.no_grad() if TORCH_AVAILABLE else None:
                outputs = self._run_model_inference(model_instance, inputs)

            # Calculate latency
            inference_time = (time.time() - start_time) * 1000  # ms

            # Update performance stats
            self._update_performance_stats(model_name, inference_time)

            return {
                'outputs': outputs,
                'latency_ms': inference_time,
                'model_name': model_name,
                'timestamp': time.time()
            }

        except Exception as e:
            logger.error(f"Inference failed for model {model_name}: {e}")
            self.performance_stats[model_name]['errors'] += 1
            raise

    def setup_ab_test(self, test_name: str, model_a: str, model_b: str,
                     traffic_split: float = 0.5) -> bool:
        """Set up A/B testing between two models."""
        if model_a not in self.active_models or model_b not in self.active_models:
            logger.error(f"One or both models not active: {model_a}, {model_b}")
            return False

        self.ab_tests[test_name] = {
            'model_a': model_a,
            'model_b': model_b,
            'traffic_split': traffic_split,  # 0.5 = 50/50 split
            'created_at': time.time(),
            'stats': {
                'model_a_requests': 0,
                'model_b_requests': 0,
                'model_a_latency': [],
                'model_b_latency': []
            }
        }

        logger.info(f"Set up A/B test '{test_name}' between {model_a} and {model_b}")
        return True

    def run_ab_test_inference(self, test_name: str, inputs: Dict[str, Any],
                             **kwargs) -> Dict[str, Any]:
        """Run inference with A/B testing."""
        if test_name not in self.ab_tests:
            raise ValueError(f"A/B test '{test_name}' not found")

        test_config = self.ab_tests[test_name]
        model_a, model_b = test_config['model_a'], test_config['model_b']

        # Determine which model to use (simple random split for now)
        import random
        use_model_a = random.random() < test_config['traffic_split']

        selected_model = model_a if use_model_a else model_b
        test_config['stats'][f'{selected_model}_requests'] += 1

        # Run inference
        result = self.run_inference(selected_model, inputs, **kwargs)

        # Track latency for A/B comparison
        test_config['stats'][f'{selected_model}_latency'].append(result['latency_ms'])

        result['ab_test'] = {
            'test_name': test_name,
            'selected_model': selected_model,
            'traffic_split': test_config['traffic_split']
        }

        return result

    def get_deployment_stats(self) -> Dict[str, Any]:
        """Get comprehensive deployment statistics."""
        return {
            'active_models': {name: info['status'] for name, info in self.active_models.items()},
            'registered_models': list(self.model_registry.keys()),
            'performance_stats': self.performance_stats,
            'ab_tests': self.ab_tests,
            'deployment_history': self.deployment_history[-10:],  # Last 10 deployments
            'total_deployments': len(self.deployment_history)
        }

    def _register_new_model(self, model_name: str, model_data: bytes):
        """Register a new model from uploaded data."""
        # Create basic model info for uploaded models
        model_hash = hashlib.sha256(model_data).hexdigest()[:16]
        model_info = {
            'config': {
                'name': model_name,
                'type': 'uploaded',
                'framework': 'torch',  # Default
                'version': '1.0.0',
                'capabilities': ['inference']
            },
            'config_path': f'uploaded_{model_name}_{model_hash}',
            'type': 'uploaded',
            'framework': 'torch',
            'version': '1.0.0',
            'capabilities': ['inference'],
            'last_modified': time.time(),
            'hash': model_hash
        }

        self.model_registry[model_name] = model_info

    def _load_model_data(self, model_name: str, model_info: Dict[str, Any]) -> bytes:
        """Load model data from filesystem."""
        # Try different model file extensions
        model_dir = self.models_dir / model_name
        for ext in ['.pth', '.pt', '.onnx', '.engine']:
            model_file = model_dir / f"{model_name}{ext}"
            if model_file.exists():
                with open(model_file, 'rb') as f:
                    return f.read()

        raise FileNotFoundError(f"No model file found for {model_name}")

    def _validate_model_data(self, model_data: bytes, model_info: Dict[str, Any]) -> bool:
        """Validate model data integrity."""
        try:
            # Basic size check
            min_size = 1024  # 1KB minimum
            if len(model_data) < min_size:
                logger.warning(f"Model data too small: {len(model_data)} bytes")
                return False

            # For PyTorch models, try to check if it's a valid state_dict
            if model_info.get('framework') == 'torch' and TORCH_AVAILABLE:
                # This is a basic validation - more sophisticated checks could be added
                return True  # Assume valid for now

            return True
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False

    def _instantiate_model(self, model_data: bytes, model_info: Dict[str, Any]) -> Any:
        """Instantiate model from data."""
        try:
            if model_info.get('framework') == 'torch' and TORCH_AVAILABLE:
                # For now, we'll create a placeholder model
                # In a real implementation, this would deserialize the model
                class PlaceholderModel(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.layers = nn.Sequential(
                            nn.Linear(10, 64),
                            nn.ReLU(),
                            nn.Linear(64, 10)
                        )

                    def forward(self, x):
                        return self.layers(x)

                return PlaceholderModel()
            else:
                raise NotImplementedError("Only PyTorch models supported for now")
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate model: {e}")

    def _validate_model_instance(self, model_instance: Any, model_info: Dict[str, Any]) -> bool:
        """Validate instantiated model."""
        try:
            # Basic validation - try a forward pass with dummy data
            if TORCH_AVAILABLE and isinstance(model_instance, nn.Module):
                with torch.no_grad():
                    dummy_input = torch.randn(1, 10)
                    output = model_instance(dummy_input)
                    return output is not None
            return True
        except Exception as e:
            logger.error(f"Model instance validation failed: {e}")
            return False

    def _warmup_model(self, model_instance: Any):
        """Warm up model for optimal performance."""
        try:
            if TORCH_AVAILABLE and isinstance(model_instance, nn.Module):
                with torch.no_grad():
                    for _ in range(3):
                        dummy_input = torch.randn(1, 10)
                        _ = model_instance(dummy_input)
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    def _run_model_inference(self, model_instance: Any, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference on model instance."""
        if TORCH_AVAILABLE and isinstance(model_instance, nn.Module):
            with torch.no_grad():
                # Extract tensor inputs
                tensor_inputs = {}
                for key, value in inputs.items():
                    if isinstance(value, (list, tuple)):
                        tensor_inputs[key] = torch.tensor(value, dtype=torch.float32)
                    elif torch.is_tensor(value):
                        tensor_inputs[key] = value
                    else:
                        tensor_inputs[key] = torch.tensor([value], dtype=torch.float32)

                # Run forward pass
                output = model_instance(**tensor_inputs)

                # Convert output to dict if needed
                if not isinstance(output, dict):
                    output = {'output': output.detach().cpu().numpy()}

                return output
        else:
            raise NotImplementedError("Only PyTorch models supported")

    def _update_performance_stats(self, model_name: str, latency_ms: float):
        """Update performance statistics for a model."""
        stats = self.performance_stats[model_name]
        stats['inference_count'] += 1
        stats['total_inference_time'] += latency_ms
        stats['latency_samples'].append(latency_ms)

        # Keep only recent samples (last 1000)
        if len(stats['latency_samples']) > 1000:
            removed = stats['latency_samples'].pop(0)
            stats['total_inference_time'] -= removed

    def _cleanup_model_resources(self, model_name: str):
        """Clean up model resources."""
        # For PyTorch models, ensure GPU memory is freed
        if TORCH_AVAILABLE:
            torch.cuda.empty_cache()

    def _compute_config_hash(self, config: Dict[str, Any]) -> str:
        """Compute hash of model configuration."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
