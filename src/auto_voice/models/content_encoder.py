"""
Content Encoder for Singing Voice Conversion

Extracts speaker-independent linguistic/phonetic content from audio using
self-supervised learning models (HuBERT-Soft) or CNN-based fallback.
"""

from typing import Optional, Union
import logging
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio

logger = logging.getLogger(__name__)


class ContentEncodingError(Exception):
    """Exception raised for content encoding errors."""
    pass


class ContentEncoder(nn.Module):
    """
    Extract speaker-independent linguistic content from audio.

    Uses HuBERT-Soft SSL model via PyTorch Hub or falls back to CNN-based encoder.

    Args:
        encoder_type: Type of encoder ('hubert_soft' or 'cnn_fallback')
        output_dim: Dimension of output content features (default: 256)
        device: Device to run encoder on (default: None, auto-detect)
        use_torch_hub: Whether to load HuBERT-Soft from PyTorch Hub (default: True)

    Attributes:
        encoder_type: Type of encoder being used
        output_dim: Output feature dimension
        sample_rate: Expected input sample rate (16kHz for HuBERT-Soft)

    Example:
        >>> encoder = ContentEncoder(encoder_type='hubert_soft', device='cuda')
        >>> audio = torch.randn(1, 16000)  # 1 second at 16kHz
        >>> content = encoder(audio, sample_rate=16000)
        >>> print(content.shape)  # [1, 50, 256] (50 frames at 50Hz)
    """

    def __init__(
        self,
        encoder_type: str = 'hubert_soft',
        output_dim: int = 256,
        device: Optional[str] = None,
        use_torch_hub: bool = True,
        cnn_mel_config: Optional[dict] = None
    ):
        super().__init__()

        self.encoder_type = encoder_type
        self.output_dim = output_dim
        self.sample_rate = 16000  # HuBERT-Soft requires 16kHz
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.lock = threading.Lock()

        # Store CNN mel configuration
        self.cnn_mel_config = cnn_mel_config or {}

        # Initialize encoder
        self.hubert = None
        self.cnn_encoder = None

        # Cache for resamplers to avoid repeated instantiation
        self._resampler_cache = {}

        if encoder_type == 'hubert_soft' and use_torch_hub:
            try:
                logger.info("Loading HuBERT-Soft from PyTorch Hub...")
                self.hubert = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True)
                self.hubert = self.hubert.to(self.device)
                self.hubert.eval()
                logger.info("HuBERT-Soft loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load HuBERT-Soft: {e}. Falling back to CNN encoder.")
                self.encoder_type = 'cnn_fallback'

        if self.encoder_type == 'cnn_fallback' or self.hubert is None:
            logger.info("Initializing CNN-based content encoder (fallback)")
            self._init_cnn_encoder()

        # Move to device
        self.to(self.device)
        self.eval()

    def _init_cnn_encoder(self):
        """Initialize CNN-based fallback encoder with configurable mel parameters."""
        # Get mel parameters from config or use defaults
        n_mels = self.cnn_mel_config.get('n_mels', 80)
        n_fft = self.cnn_mel_config.get('n_fft', 1024)
        hop_length = self.cnn_mel_config.get('hop_length', 320)
        sample_rate = self.cnn_mel_config.get('sample_rate', 16000)

        # Update sample rate if provided in config
        self.sample_rate = sample_rate

        # Log configuration for visibility
        logger.info(
            f"ContentEncoder CNN fallback: n_fft={n_fft}, hop_length={hop_length}, "
            f"n_mels={n_mels}, sample_rate={sample_rate}, "
            f"frame_rate={sample_rate/hop_length:.1f}Hz"
        )

        # Simple CNN encoder that removes speaker information via instance norm
        self.cnn_encoder = nn.Sequential(
            # Input: mel-spectrogram [B, n_mels, T]
            nn.Conv1d(n_mels, 256, kernel_size=5, padding=2),
            nn.InstanceNorm1d(256),  # Remove speaker-specific statistics
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.InstanceNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, self.output_dim, kernel_size=5, padding=2),
            nn.InstanceNorm1d(self.output_dim),
            nn.ReLU(),
        )

        # Mel-spectrogram transform with configurable parameters
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

    def forward(self, audio: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """
        Extract content features from audio.

        Args:
            audio: Audio tensor [B, T] or [B, C, T]
            sample_rate: Sample rate of input audio

        Returns:
            Content features [B, T_frames, output_dim]

        Raises:
            ContentEncodingError: If encoding fails
        """
        try:
            with self.lock:
                # Handle different input dimensions
                if audio.dim() == 3:
                    # [B, C, T] - average across channels to get mono
                    audio = audio.mean(dim=1)  # [B, T]
                elif audio.dim() == 2:
                    # Treat as [B, T] batch by default
                    pass
                elif audio.dim() == 1:
                    # [T] - add batch dimension
                    audio = audio.unsqueeze(0)

                # Move to device
                audio = audio.to(self.device)

                # Resample to 16kHz if needed (with cached resampler)
                if sample_rate != self.sample_rate:
                    cache_key = (sample_rate, self.sample_rate, str(self.device))
                    if cache_key not in self._resampler_cache:
                        self._resampler_cache[cache_key] = torchaudio.transforms.Resample(
                            orig_freq=sample_rate,
                            new_freq=self.sample_rate
                        ).to(self.device)
                    resampler = self._resampler_cache[cache_key]
                    audio = resampler(audio)

                # Normalize to [-1, 1] per sample
                max_vals = audio.abs().amax(dim=-1, keepdim=True)
                max_vals = torch.clamp(max_vals, min=1e-8)
                audio = audio / max_vals

                # Extract content features
                if self.hubert is not None:
                    # Use HuBERT-Soft
                    with torch.no_grad():
                        content = self.hubert.units(audio)  # [B, T_frames, 256]
                else:
                    # Use CNN fallback
                    with torch.no_grad():
                        # Compute mel-spectrogram
                        mel = self.mel_transform(audio)  # [B, 80, T_frames]
                        mel = torch.log(mel + 1e-8)  # Log scale

                        # Pass through CNN encoder
                        content = self.cnn_encoder(mel)  # [B, output_dim, T_frames]
                        content = content.transpose(1, 2)  # [B, T_frames, output_dim]

                return content

        except Exception as e:
            raise ContentEncodingError(f"Content encoding failed: {str(e)}")

    def extract_content(
        self,
        audio: Union[torch.Tensor, np.ndarray, str],
        sample_rate: Optional[int] = None
    ) -> torch.Tensor:
        """
        High-level content extraction method.

        Args:
            audio: Audio as tensor, numpy array, or file path
            sample_rate: Sample rate (required for tensor/array inputs)

        Returns:
            Content features [B, T_frames, output_dim]
        """
        # Handle file path input
        if isinstance(audio, str):
            audio_tensor, sr = torchaudio.load(audio)
            sample_rate = sr
            # Average channels to mono: [C, T] -> [1, T]
            audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
        # Handle numpy array
        elif isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
            if sample_rate is None:
                raise ValueError("sample_rate required for numpy array input")
        else:
            audio_tensor = audio
            if sample_rate is None:
                sample_rate = self.sample_rate

        return self.forward(audio_tensor, sample_rate)

    def get_frame_rate(self) -> float:
        """
        Get frame rate of content features.

        Returns:
            Frame rate in Hz (depends on hop_length and sample_rate)
        """
        # Calculate frame rate from hop_length and sample_rate
        if self.encoder_type == 'cnn_fallback' and hasattr(self, 'mel_transform'):
            hop_length = self.cnn_mel_config.get('hop_length', 320)
            sample_rate = self.cnn_mel_config.get('sample_rate', 16000)
            return sample_rate / hop_length
        else:
            # HuBERT-Soft uses 320 samples per frame at 16kHz = 50Hz
            return 50.0
