"""Lightweight neural source-filter style harmonic enhancement."""

from __future__ import annotations

import numpy as np


class NSFHarmonicEnhancer:
    """Inject a harmonic residual guided by an F0 contour."""

    def __init__(
        self,
        harmonic_strength: float = 0.12,
        max_harmonics: int = 6,
        blend: float = 0.2,
    ):
        self.harmonic_strength = float(max(harmonic_strength, 0.0))
        self.max_harmonics = int(max(max_harmonics, 1))
        self.blend = float(min(max(blend, 0.0), 1.0))

    def enhance(
        self,
        audio: np.ndarray,
        f0_contour: np.ndarray | None,
        sample_rate: int,
    ) -> np.ndarray:
        audio_np = np.asarray(audio, dtype=np.float32).copy()
        if audio_np.size == 0 or f0_contour is None or len(f0_contour) == 0:
            return audio_np

        frame_positions = np.linspace(
            0.0,
            max(audio_np.shape[0] - 1, 0),
            num=len(f0_contour),
            dtype=np.float32,
        )
        audio_positions = np.arange(audio_np.shape[0], dtype=np.float32)
        f0 = np.interp(audio_positions, frame_positions, np.asarray(f0_contour, dtype=np.float32))
        f0 = np.clip(f0, 0.0, sample_rate / 4)

        phase = np.cumsum((2 * np.pi * f0) / max(sample_rate, 1), dtype=np.float64)
        harmonic = np.zeros_like(audio_np, dtype=np.float32)
        for harmonic_index in range(1, self.max_harmonics + 1):
            amplitude = self.harmonic_strength / harmonic_index
            harmonic += amplitude * np.sin(phase * harmonic_index).astype(np.float32)

        enhanced = (1.0 - self.blend) * audio_np + self.blend * harmonic
        peak = float(np.max(np.abs(enhanced))) if enhanced.size else 0.0
        if peak > 0.98:
            enhanced *= 0.98 / peak
        return enhanced.astype(np.float32)
