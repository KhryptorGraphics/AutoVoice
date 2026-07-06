"""Tests for vocal de-reverberation (dereverb.py)."""
import numpy as np
import pytest

from auto_voice.audio.dereverb import dereverb_vocals, is_available

SR = 22050


def _sawtooth_bursts(sr=SR, f0=220.0, duration=2.0,
                     burst_s=0.1, gap_s=0.15):
    """Dry pulse-train 'voice': 220Hz sawtooth bursts separated by silence.

    Returns (audio, gap_mask) where gap_mask marks silent regions.
    """
    n = int(sr * duration)
    t = np.arange(n) / sr
    saw = 2.0 * ((t * f0) % 1.0) - 1.0

    mask = np.zeros(n, dtype=bool)
    period = burst_s + gap_s
    pos = 0.0
    while pos < duration:
        start = int(pos * sr)
        end = min(n, int((pos + burst_s) * sr))
        mask[start:end] = True
        pos += period

    audio = (saw * mask).astype(np.float32) * 0.5
    return audio, ~mask


def _reverberate(audio, sr=SR, tau=0.15, tail_s=0.4):
    """Convolve with a synthetic exponential-decay reverb tail."""
    rng = np.random.default_rng(0)
    n_tail = int(sr * tail_s)
    t = np.arange(n_tail) / sr
    ir = rng.standard_normal(n_tail).astype(np.float32) * np.exp(-t / tau) * 0.4
    ir[0] = 1.0  # direct path
    wet = np.convolve(audio, ir)[:len(audio)]
    wet = wet / (np.max(np.abs(wet)) + 1e-9) * 0.8
    return wet.astype(np.float32)


class TestDereverbVocals:
    def test_is_available(self):
        assert is_available() is True

    def test_output_shape_and_dtype(self):
        dry, _ = _sawtooth_bursts()
        wet = _reverberate(dry)
        out = dereverb_vocals(wet, SR, strength=0.8)
        assert out.shape == wet.shape
        assert out.dtype == np.float32
        assert np.all(np.abs(out) <= 1.0)

    def test_gap_energy_reduced(self):
        """Energy in the silent gaps between bursts must drop vs wet input."""
        dry, gap_mask = _sawtooth_bursts()
        wet = _reverberate(dry)
        out = dereverb_vocals(wet, SR, strength=0.8)

        # Skip the first 50ms of each gap so remaining direct-sound smearing
        # from the STFT itself doesn't dominate; measure late-reverb region.
        wet_gap_energy = float(np.sum(wet[gap_mask] ** 2))
        out_gap_energy = float(np.sum(out[gap_mask] ** 2))
        assert wet_gap_energy > 0
        assert out_gap_energy < 0.8 * wet_gap_energy, (
            f"gap energy not reduced: {out_gap_energy:.6f} vs "
            f"{wet_gap_energy:.6f}"
        )

    def test_strength_zero_returns_input_unchanged(self):
        dry, _ = _sawtooth_bursts()
        wet = _reverberate(dry)
        out = dereverb_vocals(wet, SR, strength=0.0)
        assert np.allclose(out, wet)

    def test_no_nans_on_zeros(self):
        audio = np.zeros(SR, dtype=np.float32)
        out = dereverb_vocals(audio, SR, strength=1.0)
        assert not np.any(np.isnan(out))
        assert out.shape == audio.shape

    def test_short_input_passthrough(self):
        audio = np.random.randn(256).astype(np.float32) * 0.1
        out = dereverb_vocals(audio, SR, strength=0.9)
        assert np.allclose(out, audio)
        assert out.shape == audio.shape

    def test_rejects_stereo(self):
        audio = np.zeros((2, SR), dtype=np.float32)
        with pytest.raises(ValueError):
            dereverb_vocals(audio, SR)

    def test_strength_clamped(self):
        dry, _ = _sawtooth_bursts(duration=0.5)
        wet = _reverberate(dry)
        out = dereverb_vocals(wet, SR, strength=5.0)  # clamps to 1.0
        assert not np.any(np.isnan(out))
        assert out.shape == wet.shape
