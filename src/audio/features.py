"""Feature extraction for audio processing."""

import torch
import torchaudio
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract various audio features for voice processing."""

    def __init__(self, sample_rate: int = 44100, device: Optional[torch.device] = None):
        """Initialize feature extractor.

        Args:
            sample_rate: Sample rate
            device: Device for processing
        """
        self.sample_rate = sample_rate
        self.device = device or torch.device('cpu')

    def extract_mfcc(self, audio: torch.Tensor, n_mfcc: int = 13) -> torch.Tensor:
        """Extract MFCC features.

        Args:
            audio: Input audio tensor
            n_mfcc: Number of MFCC coefficients

        Returns:
            MFCC features
        """
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={'n_fft': 2048, 'hop_length': 512, 'n_mels': 128}
        ).to(self.device)

        return mfcc_transform(audio)

    def extract_mel_spectrogram(self, audio: torch.Tensor,
                               n_mels: int = 128) -> torch.Tensor:
        """Extract mel spectrogram.

        Args:
            audio: Input audio tensor
            n_mels: Number of mel bands

        Returns:
            Mel spectrogram
        """
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=n_mels
        ).to(self.device)

        return mel_transform(audio)

    def extract_pitch(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract pitch contour.

        Args:
            audio: Input audio tensor

        Returns:
            Pitch values
        """
        # Use autocorrelation for pitch detection
        audio_np = audio.cpu().numpy() if audio.is_cuda else audio.numpy()
        audio_flat = audio_np.flatten()

        # Frame-based pitch detection
        frame_length = 2048
        hop_length = 512
        num_frames = (len(audio_flat) - frame_length) // hop_length + 1

        pitches = []
        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_length
            frame = audio_flat[start:end]

            # Autocorrelation
            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(corr) // 2:]

            # Find first peak
            d = np.diff(corr)
            start_idx = np.where(d > 0)[0][0] if len(np.where(d > 0)[0]) > 0 else 0

            if start_idx < len(corr) - 1:
                peak = np.argmax(corr[start_idx:]) + start_idx
                freq = self.sample_rate / peak if peak > 0 else 0.0
            else:
                freq = 0.0

            pitches.append(freq)

        return torch.tensor(pitches, device=self.device)

    def extract_formants(self, audio: torch.Tensor,
                        num_formants: int = 3) -> torch.Tensor:
        """Extract formant frequencies.

        Args:
            audio: Input audio tensor
            num_formants: Number of formants to extract

        Returns:
            Formant frequencies
        """
        # Compute LPC coefficients
        audio_np = audio.cpu().numpy() if audio.is_cuda else audio.numpy()
        audio_flat = audio_np.flatten()

        # Use autocorrelation method for LPC
        order = 2 * num_formants + 2
        r = np.correlate(audio_flat, audio_flat, mode='full')
        r = r[len(r) // 2:len(r) // 2 + order + 1]

        # Solve Toeplitz system
        from scipy.linalg import toeplitz, solve
        R = toeplitz(r[:-1])
        try:
            lpc = solve(R, -r[1:])
        except:
            lpc = np.zeros(order)

        # Find roots and convert to frequencies
        roots = np.roots(np.concatenate(([1], lpc)))
        roots = roots[np.imag(roots) >= 0]  # Keep only positive frequency roots

        # Convert to frequencies
        angles = np.angle(roots)
        freqs = angles * self.sample_rate / (2 * np.pi)

        # Sort and select formants
        freqs = np.sort(freqs[freqs > 0])[:num_formants]

        # Pad if necessary
        if len(freqs) < num_formants:
            freqs = np.pad(freqs, (0, num_formants - len(freqs)))

        return torch.tensor(freqs, device=self.device)

    def extract_spectral_features(self, audio: torch.Tensor) -> dict:
        """Extract various spectral features.

        Args:
            audio: Input audio tensor

        Returns:
            Dictionary of spectral features
        """
        # Compute spectrogram
        spec = torch.stft(audio.flatten(), n_fft=2048, hop_length=512,
                         window=torch.hann_window(2048).to(self.device),
                         return_complex=True)
        magnitude = torch.abs(spec)

        # Frequency bins
        freqs = torch.linspace(0, self.sample_rate / 2,
                              magnitude.shape[0], device=self.device)

        # Spectral centroid
        centroid = torch.sum(freqs.unsqueeze(1) * magnitude, dim=0) / \
                  torch.sum(magnitude, dim=0).clamp(min=1e-10)

        # Spectral bandwidth
        bandwidth = torch.sqrt(
            torch.sum(((freqs.unsqueeze(1) - centroid.unsqueeze(0)) ** 2) * magnitude, dim=0) /
            torch.sum(magnitude, dim=0).clamp(min=1e-10)
        )

        # Spectral rolloff
        cumsum = torch.cumsum(magnitude, dim=0)
        total = cumsum[-1, :]
        rolloff_idx = torch.searchsorted(cumsum.T, 0.85 * total).float()
        rolloff = rolloff_idx * (self.sample_rate / 2) / magnitude.shape[0]

        return {
            'centroid': centroid,
            'bandwidth': bandwidth,
            'rolloff': rolloff
        }

    def extract_zero_crossing_rate(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract zero crossing rate.

        Args:
            audio: Input audio tensor

        Returns:
            Zero crossing rate
        """
        audio_flat = audio.flatten()
        signs = torch.sign(audio_flat)
        signs[signs == 0] = 1  # Handle zeros

        # Count zero crossings
        zero_crossings = torch.sum(torch.abs(torch.diff(signs)) > 1)

        # Normalize by length
        zcr = zero_crossings.float() / (len(audio_flat) - 1)

        return zcr

    def extract_all_features(self, audio: torch.Tensor) -> dict:
        """Extract all available features.

        Args:
            audio: Input audio tensor

        Returns:
            Dictionary of all features
        """
        features = {
            'mfcc': self.extract_mfcc(audio),
            'mel_spectrogram': self.extract_mel_spectrogram(audio),
            'pitch': self.extract_pitch(audio),
            'formants': self.extract_formants(audio),
            'zcr': self.extract_zero_crossing_rate(audio)
        }

        # Add spectral features
        spectral = self.extract_spectral_features(audio)
        features.update({f'spectral_{k}': v for k, v in spectral.items()})

        return features