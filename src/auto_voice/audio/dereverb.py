"""Lightweight STFT-domain vocal de-reverberation.

Suppresses late reverberation on separated vocal stems before voice
conversion. Late-reverb magnitude is estimated per frequency bin with an
exponentially-decaying smoothing of past frames (Lebart-style), then removed
via spectral subtraction with a spectral floor. numpy/scipy/librosa only.
"""
import numpy as np
import librosa
from scipy.signal import lfilter

_N_FFT = 1024
_HOP = _N_FFT // 4
_DELAY_S = 0.06      # late reverb starts ~60ms after direct sound
_RT60_S = 0.5        # assumed reverb decay time
_FLOOR = 0.1         # spectral floor (-20 dB) to avoid musical noise


def is_available() -> bool:
    return True


def dereverb_vocals(audio: np.ndarray, sample_rate: int,
                    strength: float = 0.5) -> np.ndarray:
    """De-reverberate a mono vocal signal.

    Args:
        audio: Mono float32 numpy array.
        sample_rate: Sample rate in Hz.
        strength: 0..1. 0 returns input unchanged; higher is more aggressive.

    Returns:
        Mono float32 array, same length as input, clamped to [-1, 1].
    """
    if audio.ndim != 1:
        raise ValueError(f"dereverb_vocals expects mono 1D audio, got {audio.ndim}D")

    strength = float(np.clip(strength, 0.0, 1.0))
    n = audio.shape[0]
    # ponytail: passthrough for no-op strength or clips shorter than one STFT frame
    if strength == 0.0 or n < _N_FFT:
        return audio.astype(np.float32, copy=False)

    X = librosa.stft(audio.astype(np.float32), n_fft=_N_FFT, hop_length=_HOP)
    mag = np.abs(X)

    # Per-frame amplitude decay for the assumed RT60 (60 dB energy decay).
    a = 10.0 ** (-3.0 * _HOP / (sample_rate * _RT60_S))
    delay_frames = max(1, int(round(_DELAY_S * sample_rate / _HOP)))

    # Exponentially-smoothed past magnitude per bin, delayed and decayed:
    # late-reverb estimate R[t] = a^D * smooth(mag)[t - D]
    smoothed = lfilter([1.0 - a], [1.0, -a], mag, axis=1)
    late = np.zeros_like(mag)
    late[:, delay_frames:] = (a ** delay_frames) * smoothed[:, :-delay_frames]

    # Spectral subtraction with floor; no division, so silence stays NaN-free.
    k = 2.0 * strength
    out_mag = np.maximum(mag - k * late, _FLOOR * mag)

    y = librosa.istft(out_mag * np.exp(1j * np.angle(X)),
                      hop_length=_HOP, n_fft=_N_FFT, length=n)
    return np.clip(y, -1.0, 1.0).astype(np.float32)
