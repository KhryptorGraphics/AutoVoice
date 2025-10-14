"""Launcher for CUDA kernels."""

import torch
import numpy as np
from typing import Optional, Tuple
import logging

try:
    import auto_voice_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

logger = logging.getLogger(__name__)


class KernelLauncher:
    """Manages launching of custom CUDA kernels."""

    def __init__(self, device: torch.device):
        """Initialize kernel launcher.

        Args:
            device: Target device for kernel execution
        """
        self.device = device
        self.cuda_available = CUDA_AVAILABLE and device.type == 'cuda'

        if not self.cuda_available:
            logger.warning("CUDA kernels not available, using fallback implementations")

    def pitch_detection(self, audio: torch.Tensor, sample_rate: int = 44100) -> torch.Tensor:
        """Detect pitch using CUDA kernel.

        Args:
            audio: Audio tensor
            sample_rate: Sample rate

        Returns:
            Pitch tensor
        """
        if self.cuda_available:
            try:
                # Call CUDA kernel
                return auto_voice_cuda.pitch_detection_cuda(audio, sample_rate)
            except Exception as e:
                logger.warning(f"CUDA kernel failed: {e}, using fallback")

        # Fallback implementation
        return self._pitch_detection_fallback(audio, sample_rate)

    def spectrogram(self, audio: torch.Tensor, n_fft: int = 2048,
                   hop_length: int = 512) -> torch.Tensor:
        """Compute spectrogram using CUDA kernel.

        Args:
            audio: Audio tensor
            n_fft: FFT size
            hop_length: Hop length

        Returns:
            Spectrogram tensor
        """
        if self.cuda_available:
            try:
                # Call CUDA kernel
                return auto_voice_cuda.spectrogram_cuda(audio, n_fft, hop_length)
            except Exception as e:
                logger.warning(f"CUDA kernel failed: {e}, using fallback")

        # Fallback implementation
        return self._spectrogram_fallback(audio, n_fft, hop_length)

    def voice_synthesis(self, features: torch.Tensor,
                       model_params: torch.Tensor) -> torch.Tensor:
        """Synthesize voice using CUDA kernel.

        Args:
            features: Feature tensor
            model_params: Model parameters

        Returns:
            Synthesized audio
        """
        if self.cuda_available:
            try:
                # Call CUDA kernel
                return auto_voice_cuda.voice_synthesis_cuda(features, model_params)
            except Exception as e:
                logger.warning(f"CUDA kernel failed: {e}, using fallback")

        # Fallback implementation
        return self._voice_synthesis_fallback(features, model_params)

    def _pitch_detection_fallback(self, audio: torch.Tensor,
                                 sample_rate: int) -> torch.Tensor:
        """Fallback pitch detection implementation."""
        # Simple autocorrelation-based pitch detection
        audio_np = audio.cpu().numpy() if audio.is_cuda else audio.numpy()

        # Compute autocorrelation
        result = np.correlate(audio_np, audio_np, mode='full')
        result = result[len(result) // 2:]

        # Find first peak
        d = np.diff(result)
        start = np.where(d > 0)[0][0] if len(np.where(d > 0)[0]) > 0 else 0
        peak = np.argmax(result[start:]) + start if start < len(result) else 0

        # Convert to frequency
        frequency = sample_rate / peak if peak > 0 else 0.0

        return torch.tensor(frequency, device=self.device)

    def _spectrogram_fallback(self, audio: torch.Tensor, n_fft: int,
                            hop_length: int) -> torch.Tensor:
        """Fallback spectrogram implementation."""
        # Use PyTorch's STFT
        window = torch.hann_window(n_fft, device=self.device)
        spec = torch.stft(audio, n_fft=n_fft, hop_length=hop_length,
                         window=window, return_complex=True)

        # Convert to magnitude spectrogram
        magnitude = torch.abs(spec)
        return magnitude

    def _voice_synthesis_fallback(self, features: torch.Tensor,
                                 model_params: torch.Tensor) -> torch.Tensor:
        """Fallback voice synthesis implementation."""
        # Simple linear transformation as placeholder
        batch_size, feature_dim = features.shape
        output_dim = model_params.shape[0] // feature_dim

        # Reshape model params as weight matrix
        weights = model_params[:feature_dim * output_dim].view(feature_dim, output_dim)

        # Apply transformation
        output = torch.matmul(features, weights)

        # Apply tanh activation for audio range
        output = torch.tanh(output)

        return output.flatten()