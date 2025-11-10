"""
HuBERT model wrapper with mock mode support.
"""

import logging
from typing import Optional, Union
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class HuBERTModel:
    """
    HuBERT model for speech representation learning.

    Supports both real model loading and mock mode for testing.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        use_mock: bool = False,
        device: str = 'cpu'
    ):
        """
        Initialize HuBERT model.

        Args:
            model_path: Path to model weights
            config_path: Path to model configuration
            use_mock: Use mock implementation
            device: Device to load model on ('cpu' or 'cuda')
        """
        self.model_path = model_path
        self.config_path = config_path
        self.use_mock = use_mock
        self.device = device
        self.model = None

        if not use_mock:
            self._load_real_model()
        else:
            logger.info("Using mock HuBERT model")

    def _load_real_model(self):
        """Load the real HuBERT model."""
        try:
            # Import torch only when loading real model
            import torch
            from transformers import HubertModel, HubertConfig

            if self.config_path and Path(self.config_path).exists():
                config = HubertConfig.from_json_file(self.config_path)
            else:
                config = HubertConfig()

            if self.model_path and Path(self.model_path).exists():
                # Load from local path
                self.model = HubertModel.from_pretrained(
                    self.model_path,
                    config=config
                )
            else:
                # Load from HuggingFace Hub
                self.model = HubertModel.from_pretrained(
                    'facebook/hubert-base-ls960',
                    config=config
                )

            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Loaded real HuBERT model on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load real HuBERT model: {e}")
            logger.info("Falling back to mock mode")
            self.use_mock = True

    def extract_features(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> np.ndarray:
        """
        Extract features from audio.

        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of audio

        Returns:
            Extracted features
        """
        if self.use_mock:
            return self._mock_extract_features(audio)

        try:
            import torch

            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)

            # Extract features
            with torch.no_grad():
                outputs = self.model(audio_tensor)
                features = outputs.last_hidden_state

            return features.cpu().numpy()

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return self._mock_extract_features(audio)

    def _mock_extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Mock feature extraction for testing."""
        # Return random features with realistic shape
        # HuBERT typically outputs (batch, time_steps, hidden_size)
        time_steps = len(audio) // 320  # Rough approximation
        hidden_size = 768  # HuBERT base hidden size

        features = np.random.randn(1, time_steps, hidden_size).astype(np.float32)
        return features

    def __call__(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Make model callable."""
        return self.extract_features(audio, sample_rate)
