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
        cnn_mel_config: Optional[dict] = None,
        gpu_resampling_enabled: bool = False
    ):
        super().__init__()

        self.encoder_type = encoder_type
        self.output_dim = output_dim
        self.sample_rate = 16000  # HuBERT-Soft requires 16kHz
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.lock = threading.Lock()

        # Store CNN mel configuration
        self.cnn_mel_config = cnn_mel_config or {}

        # GPU resampling support (disabled by default for compatibility)
        # When enabled, tries to use GPU for resampling if torchaudio build supports CUDA
        self.gpu_resampling_enabled = gpu_resampling_enabled

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
            # Handle different input dimensions (no lock needed)
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

            # Resample to 16kHz if needed
            # GPU resampling is optional: many torchaudio builds don't support CUDA resampling
            if sample_rate != self.sample_rate:
                cache_key = (sample_rate, self.sample_rate)  # Device-independent cache key
                if cache_key not in self._resampler_cache:
                    self._resampler_cache[cache_key] = torchaudio.transforms.Resample(
                        orig_freq=sample_rate,
                        new_freq=self.sample_rate
                    )
                resampler = self._resampler_cache[cache_key]

                # Try GPU resampling if enabled, otherwise fall back to CPU
                if self.gpu_resampling_enabled and self.device == 'cuda':
                    try:
                        # Attempt GPU resampling
                        resampler = resampler.to(self.device)
                        audio = resampler(audio)
                    except Exception as e:
                        # Fall back to CPU resampling if GPU fails
                        logger.debug(f"GPU resampling failed: {e}. Falling back to CPU.")
                        resampler = resampler.cpu()
                        audio_cpu = audio.cpu()
                        audio = resampler(audio_cpu).to(self.device)
                else:
                    # Default behavior: CPU resampling for compatibility
                    resampler = resampler.cpu()
                    audio_cpu = audio.cpu()
                    audio = resampler(audio_cpu).to(self.device)

            # Normalize to [-1, 1] per sample
            max_vals = audio.abs().amax(dim=-1, keepdim=True)
            max_vals = torch.clamp(max_vals, min=1e-8)
            audio = audio / max_vals

            # Extract content features
            if self.hubert is not None:
                # Use HuBERT-Soft (lock only HuBERT inference for thread safety)
                with torch.no_grad():
                    with self.lock:
                        content = self.hubert.units(audio)  # [B, T_frames, 256]
            else:
                # Use CNN fallback (no lock needed for pure PyTorch modules)
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

    def prepare_for_export(self):
        """
        Prepare model for ONNX export.

        Sets eval mode and handles HuBERT rejection.
        """
        self.eval()
        if self.hubert is not None:
            raise RuntimeError(
                "Cannot export HuBERT-based ContentEncoder to ONNX. "
                "HuBERT models contain operations not supported in ONNX export. "
                "Please use CNN fallback: initialize ContentEncoder with encoder_type='cnn_fallback' "
                "or force_cnn_fallback=True during export."
            )
        logger.info("ContentEncoder prepared for ONNX export (CNN fallback mode)")

    def export_to_onnx(
        self,
        onnx_path: str,
        opset_version: int = 17,
        input_sample: Optional[torch.Tensor] = None
    ) -> str:
        """
        Export ContentEncoder to ONNX format.

        Only supports CNN fallback encoder. HuBERT export will raise an error.
        Input must already be at self.sample_rate (16kHz). No runtime resampling in exported model.

        Args:
            onnx_path: Output path for ONNX model
            opset_version: ONNX opset version
            input_sample: Sample input tensor [1, T] at self.sample_rate (if None, uses default 1s audio at 16kHz)

        Returns:
            Path to exported ONNX model

        Raises:
            RuntimeError: If HuBERT encoder is active (not CNN fallback)

        Example:
            >>> encoder = ContentEncoder(encoder_type='cnn_fallback')
            >>> encoder.export_to_onnx('content_encoder.onnx')
        """
        # Ensure we can export (no HuBERT)
        self.prepare_for_export()

        # Create default input if not provided (1 second at self.sample_rate)
        if input_sample is None:
            input_sample = torch.randn(1, self.sample_rate)

        # Ensure input is 2D [B, T]
        if input_sample.dim() == 1:
            input_sample = input_sample.unsqueeze(0)

        device = next(self.parameters()).device
        input_sample = input_sample.to(device)

        # Define dynamic axes for variable-length audio and time frames
        dynamic_axes = {
            'input_audio': {0: 'batch_size', 1: 'audio_length'},
            'content_features': {0: 'batch_size', 1: 'time_frames'}
        }

        logger.info(f"Exporting ContentEncoder to ONNX: {onnx_path}")
        logger.info(f"Expected input sample rate: {self.sample_rate} Hz (no runtime resampling in exported model)")

        # Create wrapper to skip resampling during export
        class ContentEncoderExportWrapper(nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder

            def forward(self, audio):
                """Forward assuming audio is already at correct sample rate."""
                # Skip resampling logic - assume input is already at self.sample_rate
                with self.encoder.lock:
                    # Handle different input dimensions
                    if audio.dim() == 3:
                        audio = audio.mean(dim=1)
                    elif audio.dim() == 1:
                        audio = audio.unsqueeze(0)

                    # Move to device
                    audio = audio.to(self.encoder.device)

                    # Normalize to [-1, 1] per sample
                    max_vals = audio.abs().amax(dim=-1, keepdim=True)
                    max_vals = torch.clamp(max_vals, min=1e-8)
                    audio = audio / max_vals

                    # Extract content features (CNN only, no HuBERT)
                    with torch.no_grad():
                        mel = self.encoder.mel_transform(audio)
                        mel = torch.log(mel + 1e-8)
                        content = self.encoder.cnn_encoder(mel)
                        content = content.transpose(1, 2)

                    return content

        wrapper = ContentEncoderExportWrapper(self)
        wrapper.eval()

        try:
            torch.onnx.export(
                wrapper,
                input_sample,
                onnx_path,
                export_params=True,
                verbose=False,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input_audio'],
                output_names=['content_features'],
                dynamic_axes=dynamic_axes
            )
            logger.info(f"ContentEncoder exported successfully to {onnx_path}")
            return onnx_path
        except Exception as e:
            logger.error(f"ContentEncoder ONNX export failed: {e}")
            raise
