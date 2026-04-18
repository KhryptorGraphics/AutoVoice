"""Lightweight post-vocoder refinement with bounded anti-aliasing cleanup."""

from __future__ import annotations

import numpy as np


class PupuVocoderEnhancer:
    """Refine synthesized vocals while damping alias-heavy high-frequency residue.

    The enhancer remains intentionally lightweight so it can run in the post-processing
    path without changing the main vocoder architecture. The anti-alias stage performs
    zero-phase smoothing and attenuates only the high-frequency residue, which keeps
    the speech-band content that ECAPA2-style speaker embeddings rely on.
    """

    def __init__(
        self,
        brightness: float = 0.08,
        transient_boost: float = 0.1,
        anti_alias_strength: float = 0.35,
        speaker_guard_blend: float = 0.08,
    ):
        self.brightness = float(np.clip(brightness, 0.0, 1.0))
        self.transient_boost = float(np.clip(transient_boost, 0.0, 1.0))
        self.anti_alias_strength = float(np.clip(anti_alias_strength, 0.0, 1.0))
        self.speaker_guard_blend = float(np.clip(speaker_guard_blend, 0.0, 0.5))

    @staticmethod
    def _moving_average(signal: np.ndarray, kernel_size: int) -> np.ndarray:
        kernel_size = max(int(kernel_size), 1)
        if kernel_size == 1:
            return signal.astype(np.float32, copy=True)
        if kernel_size % 2 == 0:
            kernel_size += 1

        pad = kernel_size // 2
        padded = np.pad(signal, (pad, pad), mode='reflect')
        kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
        return np.convolve(padded, kernel, mode='valid').astype(np.float32)

    def _refine_channel(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        audio_np = np.asarray(audio, dtype=np.float32).copy()
        if audio_np.size < 8:
            return audio_np

        smoothing_kernel = max(3, int(sample_rate / 4000))
        if smoothing_kernel % 2 == 0:
            smoothing_kernel += 1

        low_band = self._moving_average(audio_np, smoothing_kernel)
        high_pass = audio_np - low_band

        transient = np.abs(np.diff(audio_np, prepend=audio_np[0])).astype(np.float32)
        transient /= float(np.max(transient)) + 1e-6

        refined = audio_np + (self.brightness * high_pass)
        refined += self.transient_boost * transient * np.sign(audio_np)

        # Zero-phase smoothing approximates a low-pass cleanup without shifting transients.
        smoothed_once = self._moving_average(refined, smoothing_kernel)
        smoothed = self._moving_average(smoothed_once[::-1], smoothing_kernel)[::-1]
        alias_residue = refined - smoothed

        residue_envelope = self._moving_average(np.abs(alias_residue), max(5, smoothing_kernel * 2))
        residue_scale = float(np.max(residue_envelope)) + 1e-6
        alias_mask = np.clip(residue_envelope / residue_scale, 0.0, 1.0)

        refined = refined - (self.anti_alias_strength * alias_mask * alias_residue)

        # Blend back a small amount of the original waveform to preserve speaker cues.
        if self.speaker_guard_blend > 0.0:
            refined = ((1.0 - self.speaker_guard_blend) * refined) + (
                self.speaker_guard_blend * audio_np
            )

        peak = float(np.max(np.abs(refined))) if refined.size else 0.0
        if peak > 0.98:
            refined *= 0.98 / peak
        return refined.astype(np.float32)

    def refine(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        audio_np = np.asarray(audio, dtype=np.float32)
        if audio_np.ndim == 1:
            return self._refine_channel(audio_np, sample_rate)
        if audio_np.ndim == 2:
            channels = [
                self._refine_channel(audio_np[..., channel], sample_rate)
                for channel in range(audio_np.shape[-1])
            ]
            return np.stack(channels, axis=-1).astype(np.float32)
        return audio_np.astype(np.float32, copy=True)
