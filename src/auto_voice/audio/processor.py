"""Audio processing utilities."""
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Core audio processing operations."""

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def load(self, path: str, sr: Optional[int] = None, mono: bool = True) -> Tuple[np.ndarray, int]:
        """Load audio file."""
        import librosa
        target_sr = sr or self.sample_rate
        audio, sr_out = librosa.load(path, sr=target_sr, mono=mono)
        return audio, sr_out

    def save(self, path: str, audio: np.ndarray, sr: Optional[int] = None):
        """Save audio to file."""
        import soundfile as sf
        sf.write(path, audio, sr or self.sample_rate)

    def resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return audio
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    def normalize(self, audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
        """Normalize audio to peak amplitude."""
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio * (peak / max_val)
        return audio

    def trim_silence(self, audio: np.ndarray, threshold_db: float = -40) -> np.ndarray:
        """Trim silence from beginning and end."""
        import librosa
        trimmed, _ = librosa.effects.trim(audio, top_db=abs(threshold_db))
        return trimmed

    def to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Convert multi-channel audio to mono."""
        if audio.ndim == 1:
            return audio
        return np.mean(audio, axis=0)
