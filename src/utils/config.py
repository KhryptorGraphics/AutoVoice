"""Configuration management for AutoVoice."""

import json
import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model configuration."""
    model_type: str = "transformer"
    input_dim: int = 80
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    output_dim: int = 80
    dropout: float = 0.1
    num_speakers: int = 100
    num_styles: int = 10
    num_emotions: int = 7


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    gradient_clip: float = 1.0
    warmup_steps: int = 1000
    save_every: int = 5
    eval_every: int = 1
    mixed_precision: bool = True
    distributed: bool = False


@dataclass
class DataConfig:
    """Data configuration."""
    data_dir: str = "./data"
    metadata_file: str = "metadata.json"
    sample_rate: int = 44100
    segment_length: int = 16000
    n_mels: int = 80
    n_fft: int = 2048
    hop_length: int = 512
    augmentation_prob: float = 0.5


@dataclass
class InferenceConfig:
    """Inference configuration."""
    checkpoint_path: str = "./checkpoints/best_model.pth"
    vocoder_path: str = "./checkpoints/vocoder.pth"
    batch_size: int = 8
    streaming: bool = False
    chunk_size: int = 1024
    overlap: int = 256


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Load configuration from file.

        Args:
            config_path: Path to configuration file (JSON or YAML)

        Returns:
            Config object
        """
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                config_dict = json.load(f)
            elif config_path.endswith(('.yaml', '.yml')):
                config_dict = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path}")

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Config object
        """
        config = cls()

        # Update model config
        if 'model' in config_dict:
            for key, value in config_dict['model'].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)

        # Update training config
        if 'training' in config_dict:
            for key, value in config_dict['training'].items():
                if hasattr(config.training, key):
                    setattr(config.training, key, value)

        # Update data config
        if 'data' in config_dict:
            for key, value in config_dict['data'].items():
                if hasattr(config.data, key):
                    setattr(config.data, key, value)

        # Update inference config
        if 'inference' in config_dict:
            for key, value in config_dict['inference'].items():
                if hasattr(config.inference, key):
                    setattr(config.inference, key, value)

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Configuration dictionary
        """
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'inference': self.inference.__dict__
        }

    def save(self, config_path: str):
        """Save configuration to file.

        Args:
            config_path: Path to save configuration
        """
        config_dict = self.to_dict()

        with open(config_path, 'w') as f:
            if config_path.endswith('.json'):
                json.dump(config_dict, f, indent=2)
            elif config_path.endswith(('.yaml', '.yml')):
                yaml.dump(config_dict, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported config format: {config_path}")

    def validate(self):
        """Validate configuration."""
        # Check paths exist
        if not os.path.exists(self.data.data_dir):
            raise ValueError(f"Data directory does not exist: {self.data.data_dir}")

        # Check model parameters
        if self.model.hidden_dim % self.model.num_heads != 0:
            raise ValueError("Hidden dim must be divisible by num_heads")

        # Check training parameters
        if self.training.batch_size <= 0:
            raise ValueError("Batch size must be positive")

        if self.training.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

    def __str__(self) -> str:
        """String representation."""
        return json.dumps(self.to_dict(), indent=2)