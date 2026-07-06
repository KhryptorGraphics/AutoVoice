"""Perceptual post-mix fixes for converted vocals.

Applies quality fixes to converted vocals using the original (pre-conversion)
source vocals as reference. All audio is mono float32 at a shared sample rate.
Length mismatches of a few frames are handled internally: processing happens on
the overlapping (min-length) region and the output is returned at the CONVERTED
array's length (any converted tail beyond the overlap is passed through as-is).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import fftconvolve

__all__ = ["transfer_loudness_envelope", "voicing_gated_passthrough"]

_EPS = 1e-8
_SILENCE_RMS = 1e-3        # ~-60 dBFS frame RMS floor
_ZCR_THRESHOLD = 0.1       # voiced tones ~0.03, sibilants/breath noise >0.2
_MATCH_MAX_GAIN_DB = 12.0  # clamp for consonant loudness matching
_GAIN_SMOOTH_FRAMES = 5    # frames of gain smoothing to avoid pumping


def _rms_frames(x: np.ndarray, frame_len: int, hop: int):
    """Short-term RMS envelope. Returns (rms, frame center sample indices)."""
    if x.shape[0] < frame_len:
        x = np.pad(x, (0, frame_len - x.shape[0]))
    frames = np.lib.stride_tricks.sliding_window_view(x, frame_len)[::hop]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    centers = np.arange(rms.shape[0]) * hop + frame_len // 2
    return rms, centers


def transfer_loudness_envelope(source_vocals: np.ndarray, converted_vocals: np.ndarray,
                               sample_rate: int, frame_ms: float = 50.0,
                               strength: float = 1.0, max_gain_db: float = 12.0) -> np.ndarray:
    """Map the converted vocals' loudness envelope toward the source's.

    A smooth per-frame gain ``source_rms / converted_rms`` (clamped to
    ``±max_gain_db``) is applied to the converted vocals. Source silence
    naturally yields strong attenuation; ``strength`` in [0, 1] interpolates
    (in dB) between unity gain and full transfer.
    """
    src = np.asarray(source_vocals, dtype=np.float32)
    cvt = np.asarray(converted_vocals, dtype=np.float32)
    out = cvt.copy()
    n = min(src.shape[0], cvt.shape[0])
    strength = float(np.clip(strength, 0.0, 1.0))
    if n == 0 or strength == 0.0:
        return out

    frame_len = max(4, int(round(sample_rate * frame_ms / 1000.0)))
    hop = max(1, frame_len // 4)
    s_rms, centers = _rms_frames(src[:n], frame_len, hop)
    c_rms, _ = _rms_frames(cvt[:n], frame_len, hop)

    # Ratio follows the source envelope: source silence -> strong attenuation,
    # clamped so converted material is never boosted/cut by more than max_gain_db.
    gain_db = strength * 20.0 * np.log10((s_rms + _EPS) / (c_rms + _EPS))
    gain_db = np.clip(gain_db, -max_gain_db, max_gain_db)
    gain_db = uniform_filter1d(gain_db, size=_GAIN_SMOOTH_FRAMES, mode="nearest")

    gain = np.interp(np.arange(n), centers, 10.0 ** (gain_db / 20.0))
    out[:n] = (cvt[:n].astype(np.float64) * gain).astype(np.float32)
    return out


def voicing_gated_passthrough(source_vocals: np.ndarray, converted_vocals: np.ndarray,
                              sample_rate: int, f0: np.ndarray, hop_length: int,
                              mix: float = 0.6, fade_ms: float = 8.0) -> np.ndarray:
    """Blend source consonants/sibilants/breaths back into the converted vocals.

    ``f0`` is the per-frame contour (Hz, 0/NaN = unvoiced) aligned to
    ``source_vocals`` with ``hop_length``. Only unvoiced frames where the source
    both has energy and is high-frequency dominant are passed through, with
    raised-cosine ``fade_ms`` ramps at region boundaries and local loudness
    matching so consonants don't jump out.
    """
    src = np.asarray(source_vocals, dtype=np.float32)
    cvt = np.asarray(converted_vocals, dtype=np.float32)
    out = cvt.copy()
    n = min(src.shape[0], cvt.shape[0])
    mix = float(np.clip(mix, 0.0, 1.0))
    f0 = np.asarray(f0, dtype=np.float64)
    if n == 0 or mix == 0.0 or f0.size == 0:
        return out

    # Frame gate: unvoiced AND source has energy AND high-frequency dominant
    # (consonants/sibilants/breaths -- not silence or tonal bleed).
    gate = np.zeros(f0.size, dtype=bool)
    for i in range(f0.size):
        if f0[i] > 0:  # NaN compares False -> treated as unvoiced
            continue
        seg = src[i * hop_length: min((i + 1) * hop_length, n)].astype(np.float64)
        if seg.size < 2:
            continue
        if np.sqrt(np.mean(seg ** 2)) < _SILENCE_RMS:
            continue
        zcr = float(np.mean(np.signbit(seg[:-1]) != np.signbit(seg[1:])))
        gate[i] = zcr > _ZCR_THRESHOLD

    m = np.repeat(gate.astype(np.float64), hop_length)[:n]
    if m.shape[0] < n:
        m = np.pad(m, (0, n - m.shape[0]))

    # Raised-cosine boundary ramps (~fade_ms) so region edges never click.
    fade = max(1, int(round(sample_rate * fade_ms / 1000.0)))
    win = np.hanning(2 * fade + 1)
    m = np.clip(fftconvolve(m, win / win.sum(), mode="same"), 0.0, 1.0)

    # Loudness-match the source to the local converted level before blending.
    frame_len = max(4, int(round(sample_rate * 0.02)))  # 20 ms
    hop = max(1, frame_len // 4)
    s_rms, centers = _rms_frames(src[:n], frame_len, hop)
    c_rms, _ = _rms_frames(cvt[:n], frame_len, hop)
    g_db = 20.0 * np.log10((c_rms + _EPS) / (s_rms + _EPS))
    g_db = np.clip(g_db, -_MATCH_MAX_GAIN_DB, _MATCH_MAX_GAIN_DB)
    g_db = uniform_filter1d(g_db, size=_GAIN_SMOOTH_FRAMES, mode="nearest")
    g = np.interp(np.arange(n), centers, 10.0 ** (g_db / 20.0))

    w = m * mix
    blend = (1.0 - w) * cvt[:n].astype(np.float64) + w * g * src[:n].astype(np.float64)
    out[:n] = blend.astype(np.float32)
    return out
