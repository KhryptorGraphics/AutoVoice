"""True-peak (dBTP) measurement and limiting.

A sample peak below 0 dBFS does not mean the signal is safe to deliver: a
reconstructed waveform can overshoot between the samples, and inter-sample
peaks are what a DAC reproduces. A vocal stem measured at 0.00 dBFS sample
peak was measured at +0.16 dBTP here — over full scale, and it clipped.

The existing guards in this codebase all compare ``np.abs(x).max()`` against
0.95 or 0.999. Those are *sample* peaks, so they let true peaks through.

This module measures the oversampled peak (4x, the EBU R128 / ITU-R BS.1770
recommendation) and can limit a signal to a true-peak ceiling by applying a
single constant gain — no compressor, no envelope follower, so nothing about
the timbre changes.

Why a constant gain rather than a limiter: the only defect measured on a
converted stem was a handful of samples at full scale (8 samples >= 0.999 in
235 s). A constant trim of a fraction of a dB removes that exactly and
preserves everything else bit-for-bit in shape. A dynamic limiter would
change the level envelope of the whole vocal to fix a peak.

The 4x oversampling uses scipy's ``resample_poly``, which is a polyphase FIR
filter: linear-phase, so the oversampled peak is not itself an artefact of
the method (an odd-length symmetric FIR has no phase error to shift the peak).
"""

from __future__ import annotations

import numpy as np

# ITU-R BS.1770 / EBU R128 recommend 4x oversampling for true-peak
# measurement. 2x misses peaks that fall between the 4x samples; 8x costs
# more than the accuracy is worth here.
OVERSAMPLE_FACTOR = 4

# Below this the signal is effectively silent and the measurement is noise.
_SILENT_FLOOR = 1e-9


def true_peak_dbfs(x: np.ndarray, oversample: int = OVERSAMPLE_FACTOR) -> float:
    """True peak of a mono or multi-channel signal in dBFS.

    Args:
        x: float array of any shape; multi-channel is interleaved along the
            last axis, which is how every stem in this codebase is stored.
        oversample: factor to oversample by before taking the peak. 1 gives the
            sample peak.

    Returns:
        The true peak in dBFS, or -999.0 for a silent signal.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return -999.0
    peak = _oversampled_peak(arr, oversample)
    if peak <= _SILENT_FLOOR:
        return -999.0
    return float(20.0 * np.log10(peak))


def _oversampled_peak(arr: np.ndarray, oversample: int) -> float:
    """Peak of the oversampled signal, worst channel over all channels."""
    if oversample <= 1:
        return float(np.abs(arr).max())
    if arr.ndim > 1:
        # Stem layout is (n, channels). Each channel is oversampled on its
        # own and the worst one wins: averaging them first would cancel a
        # peak against its opposite-polarity neighbour and understate it,
        # which is the one thing this measurement must not do.
        peaks = [
            _oversampled_peak(np.ascontiguousarray(arr[:, ch]), oversample)
            for ch in range(arr.shape[-1])
        ]
        return float(max(peaks))
    try:
        from scipy.signal import resample_poly
    except ImportError:  # pragma: no cover - scipy is a runtime dependency
        return float(np.abs(arr).max())
    up = resample_poly(arr, oversample, 1)
    return float(np.abs(up).max())


def limit_true_peak(
    x: np.ndarray,
    target_dbfs: float = -1.0,
    oversample: int = OVERSAMPLE_FACTOR,
) -> tuple[np.ndarray, float]:
    """Trim a signal so its true peak sits at or below ``target_dbfs``.

    A single constant gain is applied, so the waveform shape is preserved
    exactly. Nothing is done when the signal is already under the ceiling.

    Args:
        x: the signal to trim.
        target_dbfs: the true-peak ceiling in dBFS. -1.0 is the common
            streaming/delivery target; 0.0 means "never exceed full scale".
        oversample: factor used for the measurement (see ``true_peak_dbfs``).

    Returns:
        ``(trimmed, gain_db)`` where ``gain_db`` is 0.0 when no trim was
        needed and negative when a trim was applied.
    """
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0:
        return arr, 0.0
    current = true_peak_dbfs(arr, oversample)
    if current <= target_dbfs or current <= -999.0:
        return arr, 0.0
    gain_db = target_dbfs - current
    gain = 10.0 ** (gain_db / 20.0)
    return (arr * gain).astype(np.float32), float(gain_db)


__all__ = ["OVERSAMPLE_FACTOR", "true_peak_dbfs", "limit_true_peak"]
