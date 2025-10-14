"""Audio processing utilities."""

import torch
import torchaudio
import numpy as np
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio loading, preprocessing, and augmentation."""

    def __init__(self, sample_rate: int = 44100, device: Optional[torch.device] = None):
        """Initialize audio processor.

        Args:
            sample_rate: Target sample rate
            device: Device for processing
        """
        self.sample_rate = sample_rate
        self.device = device or torch.device('cpu')

    def load_audio(self, path: str) -> Tuple[torch.Tensor, int]:
        """Load audio from file.

        Args:
            path: Path to audio file

        Returns:
            Audio tensor and sample rate
        """
        waveform, sr = torchaudio.load(path)

        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        return waveform.to(self.device), self.sample_rate

    def normalize_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Normalize audio to [-1, 1] range.

        Args:
            audio: Input audio tensor

        Returns:
            Normalized audio
        """
        max_val = torch.max(torch.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio

    def trim_silence(self, audio: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
        """Trim silence from audio.

        Args:
            audio: Input audio tensor
            threshold: Silence threshold

        Returns:
            Trimmed audio
        """
        # Find non-silent samples
        non_silent = torch.abs(audio) > threshold

        if torch.any(non_silent):
            # Find first and last non-silent sample
            indices = torch.where(non_silent.flatten())[0]
            start = indices[0]
            end = indices[-1] + 1
            return audio[..., start:end]

        return audio

    def add_noise(self, audio: torch.Tensor, noise_level: float = 0.005) -> torch.Tensor:
        """Add noise to audio for augmentation.

        Args:
            audio: Input audio tensor
            noise_level: Level of noise to add

        Returns:
            Audio with added noise
        """
        noise = torch.randn_like(audio) * noise_level
        return audio + noise

    def time_stretch(self, audio: torch.Tensor, rate: float = 1.0) -> torch.Tensor:
        """Apply time stretching to audio.

        Args:
            audio: Input audio tensor
            rate: Stretch rate (>1 for slower, <1 for faster)

        Returns:
            Time-stretched audio
        """
        if rate == 1.0:
            return audio

        # Use phase vocoder for time stretching
        spec = torch.stft(audio.flatten(), n_fft=2048, hop_length=512,
                         window=torch.hann_window(2048).to(self.device),
                         return_complex=True)

        # Stretch by modifying hop length
        stretched_length = int(spec.shape[-1] * rate)
        phase_advance = torch.linspace(0, spec.shape[-1] * np.pi,
                                      stretched_length).to(self.device)

        # Apply phase modification
        stretched_spec = torch.zeros((spec.shape[0], stretched_length),
                                    dtype=spec.dtype, device=self.device)

        for i in range(stretched_length):
            orig_idx = int(i / rate)
            if orig_idx < spec.shape[-1]:
                stretched_spec[:, i] = spec[:, orig_idx] * \
                                      torch.exp(1j * phase_advance[i])

        # Convert back to audio
        audio = torch.istft(stretched_spec, n_fft=2048, hop_length=512,
                           window=torch.hann_window(2048).to(self.device))

        return audio.unsqueeze(0) if audio.dim() == 1 else audio

    def pitch_shift(self, audio: torch.Tensor, semitones: int = 0) -> torch.Tensor:
        """Shift pitch of audio.

        Args:
            audio: Input audio tensor
            semitones: Number of semitones to shift

        Returns:
            Pitch-shifted audio
        """
        if semitones == 0:
            return audio

        # Calculate stretch factor
        factor = 2.0 ** (semitones / 12.0)

        # Time stretch then resample
        stretched = self.time_stretch(audio, 1.0 / factor)

        # Resample to original length
        orig_length = audio.shape[-1]
        indices = torch.linspace(0, stretched.shape[-1] - 1, orig_length).long()
        resampled = stretched[..., indices]

        return resampled

    def extract_segments(self, audio: torch.Tensor, segment_length: int,
                        hop_length: int) -> torch.Tensor:
        """Extract overlapping segments from audio.

        Args:
            audio: Input audio tensor
            segment_length: Length of each segment
            hop_length: Hop between segments

        Returns:
            Tensor of segments
        """
        num_segments = (audio.shape[-1] - segment_length) // hop_length + 1
        segments = []

        for i in range(num_segments):
            start = i * hop_length
            end = start + segment_length
            segments.append(audio[..., start:end])

        return torch.stack(segments)