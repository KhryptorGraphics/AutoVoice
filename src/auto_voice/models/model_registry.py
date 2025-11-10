"""
Model Registry for managing neural model loading, versioning, and caching.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, asdict
from enum import Enum
import yaml

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported model types."""
    HUBERT = "hubert"
    HIFIGAN = "hifigan"
    SPEAKER_ENCODER = "speaker_encoder"


@dataclass
class ModelConfig:
    """Configuration for a neural model."""
    name: str
    model_type: ModelType
    version: str
    url: Optional[str] = None
    local_path: Optional[str] = None
    sha256: Optional[str] = None
    config_url: Optional[str] = None
    requires_gpu: bool = False
    min_memory_gb: float = 4.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if isinstance(self.model_type, str):
            self.model_type = ModelType(self.model_type)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['model_type'] = self.model_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create from dictionary."""
        if 'model_type' in data and isinstance(data['model_type'], str):
            data['model_type'] = ModelType(data['model_type'])
        return cls(**data)


class ModelRegistry:
    """
    Central registry for managing neural models.

    Features:
    - Model downloading and caching
    - Version management
    - Lazy loading
    - Graceful fallback to mock models
    - Model warmup and initialization
    """

    def __init__(
        self,
        model_dir: str = 'models/',
        config_path: Optional[str] = None,
        use_mock: bool = False
    ):
        """
        Initialize the model registry.

        Args:
            model_dir: Directory to store downloaded models
            config_path: Path to models configuration YAML
            use_mock: Whether to use mock models (for testing)
        """
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.use_mock = use_mock
        self._models: Dict[str, Any] = {}
        self._configs: Dict[str, ModelConfig] = {}

        # Load model configurations
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent.parent / 'config' / 'models.yaml'
        self.config_path = Path(config_path)

        self._load_configs()

        # Model cache file
        self.cache_file = self.model_dir / '.model_cache.json'
        self._load_cache()

        logger.info(f"Model registry initialized with model_dir={self.model_dir}")

    def _load_configs(self):
        """Load model configurations from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Model config file not found: {self.config_path}")
            self._create_default_config()
            return

        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)

            for model_name, model_data in config_data.get('models', {}).items():
                config = ModelConfig.from_dict(model_data)
                self._configs[model_name] = config
                logger.info(f"Loaded config for model: {model_name}")

        except Exception as e:
            logger.error(f"Failed to load model configs: {e}")
            self._create_default_config()

    def _create_default_config(self):
        """Create default model configurations."""
        default_configs = {
            'hubert_base': ModelConfig(
                name='hubert_base',
                model_type=ModelType.HUBERT,
                version='1.0.0',
                url='https://huggingface.co/facebook/hubert-base-ls960/resolve/main/pytorch_model.bin',
                config_url='https://huggingface.co/facebook/hubert-base-ls960/resolve/main/config.json',
                requires_gpu=False,
                min_memory_gb=4.0,
                metadata={
                    'description': 'HuBERT base model for speech representation',
                    'sample_rate': 16000,
                    'hidden_size': 768
                }
            ),
            'hifigan_universal': ModelConfig(
                name='hifigan_universal',
                model_type=ModelType.HIFIGAN,
                version='1.0.0',
                url='https://huggingface.co/nvidia/hifigan/resolve/main/generator_v3',
                config_url='https://huggingface.co/nvidia/hifigan/resolve/main/config.json',
                requires_gpu=True,
                min_memory_gb=2.0,
                metadata={
                    'description': 'HiFi-GAN universal vocoder',
                    'sample_rate': 22050,
                    'hop_size': 256
                }
            ),
            'speaker_encoder': ModelConfig(
                name='speaker_encoder',
                model_type=ModelType.SPEAKER_ENCODER,
                version='1.0.0',
                url='https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/embedding_model.ckpt',
                requires_gpu=False,
                min_memory_gb=2.0,
                metadata={
                    'description': 'ECAPA-TDNN speaker encoder',
                    'embedding_dim': 192
                }
            )
        }

        self._configs = default_configs

        # Save to file
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            config_dict = {
                'models': {
                    name: config.to_dict()
                    for name, config in default_configs.items()
                }
            }
            yaml.dump(config_dict, f, default_flow_style=False)

        logger.info(f"Created default model config at {self.config_path}")

    def _load_cache(self):
        """Load model cache metadata."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self._cache = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load model cache: {e}")
                self._cache = {}
        else:
            self._cache = {}

    def _save_cache(self):
        """Save model cache metadata."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save model cache: {e}")

    def get_model_path(self, model_name: str) -> Path:
        """Get the local path for a model."""
        if model_name not in self._configs:
            raise ValueError(f"Unknown model: {model_name}")

        config = self._configs[model_name]

        # Check if local path is specified
        if config.local_path:
            return Path(config.local_path)

        # Otherwise, construct path in model directory
        return self.model_dir / f"{model_name}_v{config.version}"

    def is_model_downloaded(self, model_name: str) -> bool:
        """Check if a model is already downloaded."""
        model_path = self.get_model_path(model_name)
        return model_path.exists()

    def load_hubert(self, model_name: str = 'hubert_base') -> Any:
        """
        Load HuBERT model.

        Args:
            model_name: Name of the HuBERT model configuration

        Returns:
            Loaded HuBERT model
        """
        if model_name in self._models:
            return self._models[model_name]

        config = self._configs.get(model_name)
        if config is None or config.model_type != ModelType.HUBERT:
            raise ValueError(f"Invalid HuBERT model name: {model_name}")

        if self.use_mock:
            from .hubert_model import HuBERTModel
            model = HuBERTModel(use_mock=True)
            logger.info(f"Loaded mock HuBERT model: {model_name}")
        else:
            from .model_loader import ModelLoader
            loader = ModelLoader(self.model_dir)
            model = loader.load_hubert(config)
            logger.info(f"Loaded real HuBERT model: {model_name}")

        self._models[model_name] = model
        return model

    def load_hifigan(self, model_name: str = 'hifigan_universal') -> Any:
        """
        Load HiFi-GAN vocoder.

        Args:
            model_name: Name of the HiFi-GAN model configuration

        Returns:
            Loaded HiFi-GAN model
        """
        if model_name in self._models:
            return self._models[model_name]

        config = self._configs.get(model_name)
        if config is None or config.model_type != ModelType.HIFIGAN:
            raise ValueError(f"Invalid HiFi-GAN model name: {model_name}")

        if self.use_mock:
            from .hifigan_model import HiFiGANModel
            model = HiFiGANModel(use_mock=True)
            logger.info(f"Loaded mock HiFi-GAN model: {model_name}")
        else:
            from .model_loader import ModelLoader
            loader = ModelLoader(self.model_dir)
            model = loader.load_hifigan(config)
            logger.info(f"Loaded real HiFi-GAN model: {model_name}")

        self._models[model_name] = model
        return model

    def load_speaker_encoder(self, model_name: str = 'speaker_encoder') -> Any:
        """
        Load speaker encoder model.

        Args:
            model_name: Name of the speaker encoder model configuration

        Returns:
            Loaded speaker encoder model
        """
        if model_name in self._models:
            return self._models[model_name]

        config = self._configs.get(model_name)
        if config is None or config.model_type != ModelType.SPEAKER_ENCODER:
            raise ValueError(f"Invalid speaker encoder model name: {model_name}")

        if self.use_mock:
            from .speaker_encoder import SpeakerEncoderModel
            model = SpeakerEncoderModel(use_mock=True)
            logger.info(f"Loaded mock speaker encoder: {model_name}")
        else:
            from .model_loader import ModelLoader
            loader = ModelLoader(self.model_dir)
            model = loader.load_speaker_encoder(config)
            logger.info(f"Loaded real speaker encoder: {model_name}")

        self._models[model_name] = model
        return model

    def warmup_models(self, model_names: Optional[List[str]] = None):
        """
        Warmup models by loading them into memory.

        Args:
            model_names: List of model names to warmup, or None for all configured models
        """
        if model_names is None:
            model_names = list(self._configs.keys())

        for model_name in model_names:
            config = self._configs.get(model_name)
            if config is None:
                logger.warning(f"Unknown model for warmup: {model_name}")
                continue

            try:
                if config.model_type == ModelType.HUBERT:
                    self.load_hubert(model_name)
                elif config.model_type == ModelType.HIFIGAN:
                    self.load_hifigan(model_name)
                elif config.model_type == ModelType.SPEAKER_ENCODER:
                    self.load_speaker_encoder(model_name)

                logger.info(f"Warmed up model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to warmup model {model_name}: {e}")

    def list_models(self) -> List[str]:
        """List all configured model names."""
        return list(self._configs.keys())

    def get_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a specific model."""
        if model_name not in self._configs:
            raise ValueError(f"Unknown model: {model_name}")
        return self._configs[model_name]

    def clear_cache(self):
        """Clear all cached models from memory."""
        self._models.clear()
        logger.info("Cleared model cache")
