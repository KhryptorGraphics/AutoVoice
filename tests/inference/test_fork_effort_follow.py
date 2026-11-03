"""Tests for the source-keyed effort-following stage.

The stage exists because the converted voice flattens the difference between
soft and belted singing: its spectral balance changes only about two thirds as
much as the source's. These tests pin what would silently break that repair:
the contrast is transferred, loudness is left to the level stages, gaps where
the source is silent are untouched, the top band is matched to the source
only when asked to, and a zero strength leaves the chain's output alone.
"""
from __future__ import annotations

import numpy as np
import pytest

from auto_voice.inference.fork_effort_follow import band_weights, follow_effort

SR = 22050
SEG_S = 3.0


def _tone(f0: float, slope_db_per_octave: float, level: float,
          dur: float = SEG_S, sr: int = SR) -> np.ndarray:
    """Harmonics of f0 whose amplitude falls ``slope`` dB per octave."""
    t = np.arange(int(dur * sr)) / sr
    y = np.zeros_like(t)
    k = 1
    while f0 * k < 0.45 * sr:
        amp = 10 ** (slope_db_per_octave * np.log2(k) / 20.0)
        y += amp * np.sin(2 * np.pi * f0 * k * t + 0.37 * k)
        k += 1
    return (level * y / np.max(np.abs(y))).astype(np.float32)


def _tilt_db(y: np.ndarray, start_s: float, end_s: float) -> float:
    """Energy above 1 kHz relative to energy below it, over [start, end)."""
    seg = y[int(start_s * SR):int(end_s * SR)]
    P = np.abs(np.fft.rfft(seg)) ** 2
    f = np.fft.rfftfreq(len(seg), 1 / SR)
    return float(10 * np.log10(P[f >= 1000].sum() / P[(f > 50) & (f < 1000)].sum()))


def _top_share_db(y: np.ndarray, sr: int, start_s: float, end_s: float) -> float:
    """5-10 kHz share of the segment's 0.1-16 kHz energy, dB."""
    seg = y[int(start_s * sr):int(end_s * sr)]
    P = np.abs(np.fft.rfft(seg)) ** 2
    f = np.fft.rfftfreq(len(seg), 1 / sr)
    return float(10 * np.log10(P[(f >= 5000) & (f < 10000)].sum()
                               / P[(f >= 100) & (f < 16000)].sum()))


def _rms_db(y: np.ndarray, start_s: float, end_s: float) -> float:
    seg = y[int(start_s * SR):int(end_s * SR)]
    return float(20 * np.log10(np.sqrt(np.mean(seg.astype(np.float64) ** 2)) + 1e-12))


def _soft_then_belt(soft_slope: float, belt_slope: float, sr: int = SR) -> np.ndarray:
    return np.concatenate([_tone(300.0, soft_slope, 0.15, sr=sr),
                           _tone(300.0, belt_slope, 0.45, sr=sr)])


class TestBandWeights:
    def test_partition_of_unity(self):
        f = np.linspace(0, 44100 / 2, 2049)
        W = band_weights(f, (250.0, 500.0, 1000.0, 2000.0, 4000.0, 5040.0,
                             6350.0, 8000.0, 10079.0, 12699.0, 16000.0))
        np.testing.assert_allclose(W.sum(axis=0), 1.0, atol=1e-9)
        assert (W >= 0).all()


class TestContrastTransfer:
    def test_belt_soft_contrast_moves_toward_the_source(self):
        """A converted voice with the same tilt in soft and belted passages
        must take on the source's soft-versus-belt brightness contrast."""
        source = _soft_then_belt(-15.0, -3.0)
        converted = _soft_then_belt(-9.0, -9.0)
        out = follow_effort(source, converted, SR, strength=1.0, max_db=12.0)

        def contrast(y):
            return _tilt_db(y, 3.75, 5.25) - _tilt_db(y, 0.75, 2.25)

        src_c, before, after = contrast(source), contrast(converted), contrast(out)
        assert abs(before) < 1.0
        assert src_c > 15.0
        assert after > 0.7 * src_c, (src_c, before, after)

    def test_loudness_is_left_to_the_level_stages(self):
        """Spectral balance moves; each passage's level does not."""
        source = _soft_then_belt(-15.0, -3.0)
        converted = _soft_then_belt(-9.0, -9.0)
        out = follow_effort(source, converted, SR, strength=1.0, max_db=12.0)
        for a, b in ((0.75, 2.25), (3.75, 5.25)):
            assert abs(_rms_db(out, a, b) - _rms_db(converted, a, b)) < 1.0

    def test_identical_voices_are_untouched(self):
        """No deviation between source and converted means no correction."""
        source = _soft_then_belt(-15.0, -3.0)
        out = follow_effort(source, source.copy(), SR)
        np.testing.assert_allclose(out, source, atol=2e-4)


class TestTopBandModes:
    """Above ``match_above_hz`` the top band takes the SOURCE's per-frame
    share; without it every band keeps the converted's own average."""

    SR_FULL = 44100

    def _voices(self):
        sr = self.SR_FULL
        # Source: dark when soft, bright when belted. Converted: one fixed
        # tilt in between, so its top band is too bright in the soft passage
        # and too dull in the belt - the air restore's overshoot in miniature.
        return _soft_then_belt(-11.0, -7.0, sr), _soft_then_belt(-9.0, -9.0, sr)

    def test_match_pulls_the_top_band_to_the_source_per_passage(self):
        sr = self.SR_FULL
        source, converted = self._voices()
        out = follow_effort(source, converted, sr, match_above_hz=5000.0)
        for a, b in ((0.75, 2.25), (3.75, 5.25)):
            src = _top_share_db(source, sr, a, b)
            err_before = abs(_top_share_db(converted, sr, a, b) - src)
            err_after = abs(_top_share_db(out, sr, a, b) - src)
            assert err_before > 4.0
            assert err_after < 0.5 * err_before, (a, err_before, err_after)

    def test_without_match_the_top_band_keeps_its_own_average(self):
        """Deviation mode follows the source's swing but not its level: the
        converted's song-long top-band brightness is its own."""
        sr = self.SR_FULL
        source, converted = self._voices()
        out = follow_effort(source, converted, sr, match_above_hz=None)

        def mean_share(y):
            return np.mean([_top_share_db(y, sr, a, b)
                            for a, b in ((0.75, 2.25), (3.75, 5.25))])

        def swing(y):
            return _top_share_db(y, sr, 3.75, 5.25) - _top_share_db(y, sr, 0.75, 2.25)

        assert abs(mean_share(out) - mean_share(converted)) < 1.5
        assert swing(out) > swing(converted) + 0.5 * (swing(source) - swing(converted))


class TestFrameLevel:
    def test_every_frame_keeps_its_level(self):
        """Level dropout is judged frame by frame (> 6 dB under the source),
        so the stage may move spectral balance but not a frame's level - even
        where adjacent bands take large opposite gains. The first version
        normalised smoothed band energies and moved such frames 0.8 dB here
        (2.5 dB on real held notes), enough to create dropouts."""
        import librosa

        sr = 44100
        source = _soft_then_belt(-18.0, 0.0, sr)
        converted = _soft_then_belt(-9.0, -9.0, sr)
        out = follow_effort(source, converted, sr, max_db=14.0, match_above_hz=5000.0)

        def frame_db(y):
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            return 20 * np.log10(rms + 1e-9)

        before, after = frame_db(converted), frame_db(out)
        live = before > before.max() - 40.0
        assert np.max(np.abs(after - before)[live]) < 0.5


class TestNoOps:
    def test_zero_strength_returns_the_input(self):
        source = _soft_then_belt(-15.0, -3.0)
        converted = _soft_then_belt(-9.0, -9.0)
        out = follow_effort(source, converted, SR, strength=0.0)
        np.testing.assert_array_equal(out, converted)

    def test_source_gap_is_untouched(self):
        """Where the source is silent the converted keeps its own audio:
        the hum stage owns the gaps."""
        gap = np.zeros(int(SEG_S * SR), dtype=np.float32)
        source = np.concatenate([_tone(300.0, -3.0, 0.45), gap,
                                 _tone(300.0, -15.0, 0.15)])
        converted = np.concatenate([_tone(300.0, -9.0, 0.45),
                                    _tone(300.0, -9.0, 0.05),
                                    _tone(300.0, -9.0, 0.15)])
        out = follow_effort(source, converted, SR, strength=1.0, max_db=12.0)
        a, b = int(4.0 * SR), int(5.0 * SR)
        assert np.max(np.abs(out[a:b] - converted[a:b])) < 1e-4

    def test_short_input_returns_the_input(self):
        y = _tone(300.0, -9.0, 0.3, dur=0.2)
        out = follow_effort(y, y, SR)
        np.testing.assert_array_equal(out, y)


class TestBound:
    def test_correction_is_bounded(self):
        """A bad measurement must never move a band further than max_db."""
        source = _soft_then_belt(-30.0, 0.0)
        converted = _soft_then_belt(-9.0, -9.0)
        out = follow_effort(source, converted, SR, strength=1.0, max_db=3.0)
        for a, b in ((0.75, 2.25), (3.75, 5.25)):
            moved = abs(_tilt_db(out, a, b) - _tilt_db(converted, a, b))
            assert moved < 2 * 3.0 + 1.0


class TestChainStage:
    """The stage as the fork correction chain runs it, every other stage off."""

    @staticmethod
    def _pipeline(**config):
        from auto_voice.inference.singing_conversion_pipeline import (
            SingingConversionPipeline,
        )
        p = SingingConversionPipeline.__new__(SingingConversionPipeline)
        p.config = {
            'fork_hq_hum_suppress': False,
            'fork_hq_subharmonic_suppress': False,
            'fork_hq_voice_polish': False,
            'fork_hq_hf_detonalize': False,
            'fork_hq_air_cap_db': 0.0,
            'fork_hq_extend_bandwidth': False,
            'fork_hq_inharmonic_notch': False,
            **config,
        }
        p._explicit_settings = {}
        p._fill_lead_dropouts = lambda converted, source, sr: converted
        return p

    def test_zero_strength_leaves_the_chain_output_untouched(self):
        source = _soft_then_belt(-15.0, -3.0)
        converted = _soft_then_belt(-9.0, -9.0)
        p = self._pipeline(fork_hq_effort_follow_strength=0.0)
        out = p._correct_fork_output(source, converted, SR)
        np.testing.assert_array_equal(out, converted)

    def test_the_chain_follows_effort_by_default(self):
        source = _soft_then_belt(-15.0, -3.0)
        converted = _soft_then_belt(-9.0, -9.0)
        out = self._pipeline()._correct_fork_output(source, converted, SR)
        contrast = _tilt_db(out, 3.75, 5.25) - _tilt_db(out, 0.75, 2.25)
        src_c = _tilt_db(source, 3.75, 5.25) - _tilt_db(source, 0.75, 2.25)
        assert contrast > 0.7 * src_c


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
