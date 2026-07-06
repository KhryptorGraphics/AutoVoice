"""F0 contour post-processing utilities.

Contours are per-frame F0 in Hz where 0 (or NaN) marks unvoiced frames.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

__all__ = ["postprocess_f0", "semitone_shift_f0"]

_MIN_ISLAND_FRAMES = 3
_OCTAVE_TOL = 0.15  # |log2 deviation| within 1.0 +/- this counts as an octave jump


def _runs(mask: np.ndarray):
    """(start, end) index pairs of contiguous True runs."""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    d = np.diff(padded.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def postprocess_f0(f0: np.ndarray, voiced_confidence: np.ndarray | None = None,
                   median_kernel: int = 5, max_octave_jump_frames: int = 12) -> np.ndarray:
    """Clean an F0 contour: NaN handling, island removal, octave-jump
    correction, boundary-safe median filtering, and single-frame gap filling.

    Genuine vibrato (sub-semitone oscillation) is preserved.
    """
    # ponytail: voiced_confidence accepted for API compatibility, unused for now.
    f = np.asarray(f0, dtype=np.float64).copy()
    f[~np.isfinite(f)] = 0.0
    f[f < 0.0] = 0.0

    # (2) drop isolated voiced islands (< 3 frames, likely noise)
    for s, e in _runs(f > 0):
        if e - s < _MIN_ISLAND_FRAMES:
            f[s:e] = 0.0

    # (3) octave-jump correction inside contiguous voiced segments
    for s, e in _runs(f > 0):
        seg = f[s:e]
        if seg.size < _MIN_ISLAND_FRAMES:
            continue
        lg = np.log2(seg)
        # Wide running-median register reference; brief excursions don't move it.
        k = min(2 * max_octave_jump_frames + 1, seg.size)
        k -= 1 - k % 2  # force odd
        ref = median_filter(lg, size=k, mode="nearest")
        dev = lg - ref
        jump = np.abs(np.abs(dev) - 1.0) <= _OCTAVE_TOL
        for js, je in _runs(jump):
            # Only fix excursions that return to register within the horizon;
            # longer runs are genuine register changes and are left alone.
            if je - js <= max_octave_jump_frames:
                lg[js:je] -= np.round(dev[js:je])
        f[s:e] = 2.0 ** lg

    # (4) median filter INSIDE voiced segments only -- never across
    # voiced/unvoiced boundaries; small default kernel keeps vibrato intact.
    for s, e in _runs(f > 0):
        k = min(int(median_kernel), e - s)
        k -= 1 - k % 2
        if k >= 3:
            f[s:e] = median_filter(f[s:e], size=k, mode="nearest")

    # (5) fill single-frame unvoiced gaps inside voiced runs (linear interp)
    idx = np.flatnonzero((f[1:-1] == 0) & (f[:-2] > 0) & (f[2:] > 0)) + 1
    f[idx] = 0.5 * (f[idx - 1] + f[idx + 1])

    return f.astype(np.float32)


def semitone_shift_f0(f0: np.ndarray, semitones: float) -> np.ndarray:
    """Shift voiced F0 values by ``semitones``; unvoiced (0/NaN) frames stay 0."""
    f = np.asarray(f0, dtype=np.float32).copy()
    f[~np.isfinite(f)] = 0.0
    factor = np.float32(2.0 ** (float(semitones) / 12.0))
    return np.where(f > 0, f * factor, np.float32(0.0)).astype(np.float32)
