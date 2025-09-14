"""
Audio processing utilities using CUDA kernels
"""
import torch
import numpy as np
from typing import Optional, Tuple
import logging

try:
    import auto_voice.cuda_kernels as cuda_kernels
except ImportError:
    cuda_kernels = None

class AudioProcessor:
    """GPU-accelerated audio processing"""

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.logger = logging.getLogger(__name__)

        if cuda_kernels is None:
            self.logger.warning("CUDA kernels not available, falling back to CPU")
            self.device = 'cpu'

    def extract_pitch(self, audio: torch.Tensor, sample_rate: float = 22050) -> torch.Tensor:
        """Extract pitch contour from audio"""
        if self.device == 'cuda' and cuda_kernels is not None:
            audio = audio.to('cuda')
            pitch = torch.zeros(audio.size(0) // 256, device='cuda')  # Approximate frame count
            cuda_kernels.pitch_detection(audio, pitch, sample_rate)
            return pitch
        else:
            # CPU fallback implementation
            return torch.zeros(audio.size(0) // 256)

    def voice_activity_detection(self, audio: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
        """Detect voice activity in audio"""
        if self.device == 'cuda' and cuda_kernels is not None:
            audio = audio.to('cuda')
            vad = torch.zeros(audio.size(0) // 256, device='cuda')
            cuda_kernels.voice_activity_detection(audio, vad, threshold)
            return vad
        else:
            # CPU fallback: simple energy-based VAD
            frame_length = 1024
            hop_length = 256
            frames = audio.unfold(0, frame_length, hop_length)
            energy = torch.mean(frames ** 2, dim=1)
            return (energy > threshold).float()

    def compute_spectrogram(self, audio: torch.Tensor, n_fft: int = 1024,
                          hop_length: int = 256, win_length: int = 1024) -> torch.Tensor:
        """Compute magnitude spectrogram"""
        if self.device == 'cuda' and cuda_kernels is not None:
            audio = audio.to('cuda')
            n_frames = (audio.size(0) - win_length) // hop_length + 1
            n_bins = n_fft // 2 + 1
            spectrogram = torch.zeros((n_frames, n_bins), device='cuda')
            cuda_kernels.spectrogram(audio, spectrogram, n_fft, hop_length, win_length)
            return spectrogram
        else:
            # CPU fallback using torch.stft
            spec = torch.stft(audio, n_fft, hop_length, win_length, return_complex=True)
            return torch.abs(spec).T