"""Voice model registry for managing speaker embeddings.

Provides functionality to list, load, and manage voice models for
karaoke voice conversion. Supports both pretrained models and
embeddings extracted from uploaded songs.
"""
import logging
import os
import uuid
from typing import Dict, Any, Optional, List

import torch

logger = logging.getLogger(__name__)

# Default models directory
DEFAULT_MODELS_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'models', 'voice_models'
)


def extract_speaker_embedding(
    audio: torch.Tensor,
    sample_rate: int = 24000,
) -> torch.Tensor:
    """Extract speaker embedding from audio.

    Uses mel-statistics approach (mean+std of 128 mel bands = 256 dims).

    Args:
        audio: Audio tensor of shape (samples,) or (channels, samples)
        sample_rate: Audio sample rate

    Returns:
        256-dimensional speaker embedding tensor
    """
    import torchaudio.transforms as T

    # Ensure mono
    if audio.dim() > 1:
        audio = audio.mean(dim=0)

    # Ensure float
    if audio.dtype != torch.float32:
        audio = audio.float()

    # Compute mel spectrogram
    n_mels = 128
    n_fft = 1024
    hop_length = 256

    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )

    mel_spec = mel_transform(audio)  # (n_mels, time)

    # Convert to log scale
    mel_spec = torch.log(mel_spec + 1e-9)

    # Compute statistics across time dimension
    mel_mean = mel_spec.mean(dim=1)  # (128,)
    mel_std = mel_spec.std(dim=1)    # (128,)

    # Concatenate mean and std for 256-dim embedding
    embedding = torch.cat([mel_mean, mel_std], dim=0)  # (256,)

    # L2 normalize
    embedding = embedding / (embedding.norm() + 1e-8)

    return embedding


class VoiceModelRegistry:
    """Registry for managing voice models (speaker embeddings).

    Supports:
    - Listing available pretrained models
    - Loading embeddings by model ID
    - Registering new models extracted from audio

    Args:
        models_dir: Directory containing pretrained model files
    """

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = models_dir or DEFAULT_MODELS_DIR

        # In-memory storage for extracted models
        # In production, this would be a database
        self._extracted_models: Dict[str, Dict[str, Any]] = {}

        # Scan for pretrained models
        self._pretrained_models: Dict[str, Dict[str, Any]] = {}
        self._scan_pretrained_models()

        logger.info(
            f"VoiceModelRegistry initialized with {len(self._pretrained_models)} "
            f"pretrained models"
        )

    def _scan_pretrained_models(self):
        """Scan models directory for pretrained speaker embeddings."""
        if not os.path.exists(self.models_dir):
            logger.info(f"Models directory does not exist: {self.models_dir}")
            return

        for filename in os.listdir(self.models_dir):
            if filename.endswith('.pt') or filename.endswith('.pth'):
                model_id = os.path.splitext(filename)[0]
                model_path = os.path.join(self.models_dir, filename)

                # Extract name from filename (replace underscores with spaces)
                name = model_id.replace('_', ' ').title()

                self._pretrained_models[model_id] = {
                    'id': model_id,
                    'name': name,
                    'type': 'pretrained',
                    'path': model_path,
                    'embedding_dim': 256,
                }

    def list_models(self) -> List[Dict[str, Any]]:
        """List all available voice models.

        Returns:
            List of model info dicts with id, name, type
        """
        models = []

        # Add pretrained models
        for model in self._pretrained_models.values():
            models.append({
                'id': model['id'],
                'name': model['name'],
                'type': model['type'],
            })

        # Add extracted models
        for model in self._extracted_models.values():
            models.append({
                'id': model['id'],
                'name': model['name'],
                'type': model['type'],
            })

        return models

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model info by ID.

        Args:
            model_id: Unique model identifier

        Returns:
            Model info dict or None if not found
        """
        if model_id in self._pretrained_models:
            model = self._pretrained_models[model_id]
            return {
                'id': model['id'],
                'name': model['name'],
                'type': model['type'],
                'embedding_dim': model['embedding_dim'],
            }

        if model_id in self._extracted_models:
            model = self._extracted_models[model_id]
            return {
                'id': model['id'],
                'name': model['name'],
                'type': model['type'],
                'embedding_dim': 256,
                'source_song_id': model.get('source_song_id'),
            }

        return None

    def get_embedding(self, model_id: str) -> Optional[torch.Tensor]:
        """Load speaker embedding for a model.

        Args:
            model_id: Unique model identifier

        Returns:
            Speaker embedding tensor (256-dim) or None if not found
        """
        # Check pretrained models
        if model_id in self._pretrained_models:
            model = self._pretrained_models[model_id]
            try:
                embedding = torch.load(model['path'], weights_only=True)
                if isinstance(embedding, dict):
                    embedding = embedding.get('embedding', embedding.get('speaker_embedding'))
                if embedding is not None:
                    return embedding.float()
            except Exception as e:
                logger.error(f"Failed to load embedding for {model_id}: {e}")
                return None

        # Check extracted models
        if model_id in self._extracted_models:
            model = self._extracted_models[model_id]
            return model['embedding'].clone()

        return None

    def register_extracted_model(
        self,
        name: str,
        embedding: torch.Tensor,
        source_song_id: Optional[str] = None,
    ) -> str:
        """Register a new voice model extracted from audio.

        Args:
            name: Display name for the model
            embedding: 256-dim speaker embedding tensor
            source_song_id: Optional ID of source song

        Returns:
            Generated model ID
        """
        model_id = f"extracted_{uuid.uuid4().hex[:8]}"

        # Ensure embedding is correct shape
        if embedding.dim() > 1:
            embedding = embedding.squeeze()
        if embedding.shape[0] != 256:
            raise ValueError(f"Embedding must be 256-dim, got {embedding.shape}")

        self._extracted_models[model_id] = {
            'id': model_id,
            'name': name,
            'type': 'extracted',
            'embedding': embedding.clone(),
            'source_song_id': source_song_id,
        }

        logger.info(f"Registered extracted model: {model_id} ({name})")

        return model_id

    def extract_and_register_from_audio(
        self,
        audio: torch.Tensor,
        name: str,
        sample_rate: int = 24000,
        source_song_id: Optional[str] = None,
    ) -> str:
        """Extract embedding from audio and register as a model.

        Args:
            audio: Audio tensor
            name: Display name for the model
            sample_rate: Audio sample rate
            source_song_id: Optional ID of source song

        Returns:
            Generated model ID
        """
        embedding = extract_speaker_embedding(audio, sample_rate)
        return self.register_extracted_model(name, embedding, source_song_id)
