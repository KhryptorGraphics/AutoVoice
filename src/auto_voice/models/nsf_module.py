"""Lightweight neural source-filter style harmonic enhancement."""

from __future__ import annotations

import numpy as np


class NSFHarmonicEnhancer:
    """Inject a harmonic residual guided by an F0 contour.

    This stays intentionally lightweight so it can run in the offline quality
    pipeline and in real-time-ish post-processing paths without introducing a
    heavyweight neural vocoder dependency.
    """

    def __init__(
        self,
        harmonic_strength: float = 0.12,
        max_harmonics: int = 6,
        blend: float = 0.2,
        noise_blend: float = 0.05,
        voice_characteristic: str = "neutral",
    ):
        strength_scale, harmonic_bonus = self._voice_characteristic_adjustment(voice_characteristic)
        self.harmonic_strength = float(max(harmonic_strength, 0.0) * strength_scale)
        self.max_harmonics = int(max(max_harmonics + harmonic_bonus, 1))
        self.blend = float(min(max(blend, 0.0), 1.0))
        self.noise_blend = float(min(max(noise_blend, 0.0), 1.0))
        self.voice_characteristic = voice_characteristic

    @staticmethod
    def _voice_characteristic_adjustment(voice_characteristic: str) -> tuple[float, int]:
        normalized = (voice_characteristic or "neutral").strip().lower()
        if normalized in {"male", "baritone", "tenor"}:
            return 1.15, -1
        if normalized in {"female", "alto", "soprano"}:
            return 0.95, 1
        return 1.0, 0

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

        noise_residual = np.zeros_like(audio_np, dtype=np.float32)
        if self.noise_blend > 0.0 and audio_np.size > 2:
            # Preserve some broadband texture by mixing a simple differentiator
            # residual back in. This is stable and cheap while still sounding
            # less synthetic than pure sinusoids.
            noise_residual[1:] = audio_np[1:] - audio_np[:-1]

        enhanced = (
            (1.0 - self.blend) * audio_np
            + self.blend * harmonic
            + self.noise_blend * noise_residual
        )
        peak = float(np.max(np.abs(enhanced))) if enhanced.size else 0.0
        if peak > 0.98:
            enhanced *= 0.98 / peak
        return enhanced.astype(np.float32)
