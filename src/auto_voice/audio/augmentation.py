"""Data augmentation pipeline for training.

Provides pitch shifting, time stretching, and EQ augmentation
to increase effective training data diversity.
"""
import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class AugmentationPipeline:
    """Configurable audio augmentation pipeline for training.

    Each augmentation has an independent probability of being applied.
    Multiple augmentations can be applied to the same sample.

    Args:
        pitch_shift_prob: Probability of pitch shift (default 0.5)
        pitch_shift_range: Max semitones shift (default 2.0)
        time_stretch_prob: Probability of time stretch (default 0.3)
        time_stretch_range: Max speed factor deviation (default 0.1, meaning ±10%)
        eq_prob: Probability of EQ augmentation (default 0.3)
        eq_bands: Number of EQ bands to modify (default 3)
        eq_gain_range: Max gain in dB (default 6.0)
    """

    def __init__(self,
                 pitch_shift_prob: float = 0.5,
                 pitch_shift_range: float = 2.0,
                 time_stretch_prob: float = 0.3,
                 time_stretch_range: float = 0.1,
                 eq_prob: float = 0.3,
                 eq_bands: int = 3,
                 eq_gain_range: float = 6.0):
        self.pitch_shift_prob = pitch_shift_prob
        self.pitch_shift_range = pitch_shift_range
        self.time_stretch_prob = time_stretch_prob
        self.time_stretch_range = time_stretch_range
        self.eq_prob = eq_prob
        self.eq_bands = eq_bands
        self.eq_gain_range = eq_gain_range

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply random augmentations to audio.

        Args:
            audio: Audio array [T] (mono, float32)
            sr: Sample rate

        Returns:
            Augmented audio array [T] (same length as input)
        """
        augmented = audio.copy()
        original_length = len(augmented)

        if np.random.random() < self.pitch_shift_prob:
            augmented = self._pitch_shift(augmented, sr)

        if np.random.random() < self.time_stretch_prob:
            augmented = self._time_stretch(augmented, sr)

        if np.random.random() < self.eq_prob:
            augmented = self._eq(augmented, sr)

        # Ensure output length matches input (augmentations may change length)
        if len(augmented) > original_length:
            augmented = augmented[:original_length]
        elif len(augmented) < original_length:
            augmented = np.pad(augmented, (0, original_length - len(augmented)))

        # Ensure finite values
        augmented = np.nan_to_num(augmented, nan=0.0, posinf=0.0, neginf=0.0)

        return augmented.astype(np.float32)

    def _pitch_shift(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply random pitch shift within ±pitch_shift_range semitones."""
        import librosa
        n_steps = np.random.uniform(-self.pitch_shift_range, self.pitch_shift_range)
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

    def _time_stretch(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply random time stretch within ±time_stretch_range."""
        import librosa
        rate = 1.0 + np.random.uniform(-self.time_stretch_range, self.time_stretch_range)
        rate = max(0.5, min(2.0, rate))  # Safety clamp
        return librosa.effects.time_stretch(audio, rate=rate)

    def _eq(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply random bandpass EQ emphasis/attenuation."""
        from scipy.signal import butter, sosfilt

        nyquist = sr / 2.0
        min_freq = 80.0
        max_freq = min(nyquist * 0.9, 8000.0)

        for _ in range(self.eq_bands):
            # Random center frequency (log-uniform distribution)
            center = np.exp(np.random.uniform(np.log(min_freq), np.log(max_freq)))
            # Bandwidth: 0.5 to 2 octaves
            bandwidth_octaves = np.random.uniform(0.5, 2.0)
            low = center / (2 ** (bandwidth_octaves / 2))
            high = center * (2 ** (bandwidth_octaves / 2))

            low = max(20.0, min(low, nyquist - 10))
            high = max(low + 10, min(high, nyquist - 1))

            if high <= low or low >= nyquist or high >= nyquist:
                continue

            # Random gain
            gain_db = np.random.uniform(-self.eq_gain_range, self.eq_gain_range)
            gain_linear = 10 ** (gain_db / 20.0)

            # Bandpass filter
            try:
                sos = butter(2, [low / nyquist, high / nyquist], btype='band', output='sos')
                band = sosfilt(sos, audio)
                # Mix: add/subtract band content based on gain
                audio = audio + (gain_linear - 1.0) * band
            except (ValueError, RuntimeError):
                continue

        # Normalize to prevent clipping
        peak = np.abs(audio).max()
        if peak > 1.0:
            audio = audio / peak * 0.95

        return audio
