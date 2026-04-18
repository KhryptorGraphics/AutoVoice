"""Lightweight post-vocoder refinement inspired by higher-fidelity neural vocoders."""

from __future__ import annotations

import numpy as np


class PupuVocoderEnhancer:
    """Refine synthesized vocals with simple spectral tilt and transient lift."""

    def __init__(self, brightness: float = 0.08, transient_boost: float = 0.1):
        self.brightness = float(max(brightness, 0.0))
        self.transient_boost = float(max(transient_boost, 0.0))

    def refine(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        del sample_rate  # The current refinement is sample-rate agnostic.

        audio_np = np.asarray(audio, dtype=np.float32).copy()
        if audio_np.size < 4:
            return audio_np

        high_pass = np.diff(audio_np, prepend=audio_np[0]).astype(np.float32)
        transient = np.concatenate(([0.0], np.abs(np.diff(audio_np)))).astype(np.float32)
        transient /= float(np.max(transient)) + 1e-6

        refined = audio_np + (self.brightness * high_pass)
        refined += self.transient_boost * transient * np.sign(audio_np)

        peak = float(np.max(np.abs(refined))) if refined.size else 0.0
        if peak > 0.98:
            refined *= 0.98 / peak
        return refined.astype(np.float32)
