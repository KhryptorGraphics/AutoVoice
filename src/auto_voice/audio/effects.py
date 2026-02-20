"""Audio effects - pitch shifting, volume adjustment."""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def pitch_shift(audio: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Shift pitch by n_steps semitones.

    Args:
        audio: Input audio array
        sr: Sample rate
        n_steps: Number of semitones to shift (positive=up, negative=down)

    Returns:
        Pitch-shifted audio
    """
    if abs(n_steps) < 0.01:
        return audio

    try:
        import librosa
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    except Exception as e:
        logger.warning(f"Pitch shift failed: {e}")
        return audio


def volume_adjust(audio: np.ndarray, gain: float) -> np.ndarray:
    """Adjust audio volume.

    Args:
        audio: Input audio array
        gain: Volume multiplier (1.0 = no change)

    Returns:
        Volume-adjusted audio
    """
    return np.clip(audio * gain, -1.0, 1.0)


def fade_in(audio: np.ndarray, duration_samples: int) -> np.ndarray:
    """Apply linear fade-in."""
    if duration_samples <= 0 or duration_samples >= len(audio):
        return audio
    fade = np.linspace(0, 1, duration_samples)
    audio = audio.copy()
    audio[:duration_samples] *= fade
    return audio


def fade_out(audio: np.ndarray, duration_samples: int) -> np.ndarray:
    """Apply linear fade-out."""
    if duration_samples <= 0 or duration_samples >= len(audio):
        return audio
    fade = np.linspace(1, 0, duration_samples)
    audio = audio.copy()
    audio[-duration_samples:] *= fade
    return audio
