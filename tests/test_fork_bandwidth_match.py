"""Bandwidth matching for the fork HQ lane.

The decoder is full-band; a separated stem off a lossy encode usually is not.
Measured on the reference song the render carried +25 dB more 16-22 kHz energy
than the source had - invented content no input evidence supports. These tests
pin the two properties that make the repair safe: it finds a real wall, and it
is a no-op on audio that never had one.
"""
from pathlib import Path

import numpy as np
import pytest

from auto_voice.inference.singing_conversion_pipeline import (
    _detect_bandwidth_hz,
    _lowpass_to,
)

SR = 44100
NYQUIST = SR / 2


def _noise(seconds=2.0, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(SR * seconds)) * 0.1).astype(np.float32)


def _brickwall(audio, cutoff_hz, sr=SR):
    """Hard spectral wall, the shape a lossy codec leaves behind."""
    spec = np.fft.rfft(audio.astype(np.float64))
    freqs = np.fft.rfftfreq(audio.size, 1.0 / sr)
    spec[freqs >= cutoff_hz] = 0.0
    return np.fft.irfft(spec, n=audio.size).astype(np.float32)


class TestDetectBandwidth:
    def test_full_band_noise_reports_nyquist(self):
        assert _detect_bandwidth_hz(_noise(), SR) == pytest.approx(NYQUIST, abs=600)

    @pytest.mark.parametrize("cutoff", [8000.0, 12000.0, 16000.0, 20000.0])
    def test_finds_a_brickwall(self, cutoff):
        got = _detect_bandwidth_hz(_brickwall(_noise(), cutoff), SR)
        assert got == pytest.approx(cutoff, abs=600), f"wall at {cutoff}, detected {got}"

    def test_short_input_is_not_band_limited(self):
        # Too short to analyse: must claim full band so the caller does nothing.
        assert _detect_bandwidth_hz(np.zeros(128, dtype=np.float32), SR) == NYQUIST

    def test_silence_does_not_crash_or_band_limit(self):
        assert _detect_bandwidth_hz(np.zeros(SR, dtype=np.float32), SR) == NYQUIST


class TestLowpassTo:
    def test_preserves_shape_and_dtype(self):
        x = _noise()
        y = _lowpass_to(x, SR, 16000.0)
        assert y.shape == x.shape and y.dtype == np.float32

    def test_cutoff_at_or_above_nyquist_is_identity(self):
        x = _noise()
        assert np.array_equal(_lowpass_to(x, SR, NYQUIST), x)
        assert np.array_equal(_lowpass_to(x, SR, 30000.0), x)

    def test_removes_energy_above_cutoff_and_keeps_it_below(self):
        x = _noise()
        y = _lowpass_to(x, SR, 16000.0)
        freqs = np.fft.rfftfreq(x.size, 1.0 / SR)
        pw = lambda s, m: float(np.mean(np.abs(np.fft.rfft(s))[m] ** 2))
        above = freqs >= 18000
        below = freqs <= 12000
        assert pw(y, above) < pw(x, above) * 1e-3      # gone
        assert pw(y, below) == pytest.approx(pw(x, below), rel=0.05)  # intact

    def test_too_short_for_filtfilt_is_returned_unchanged(self):
        x = _noise(seconds=0.001)
        assert np.array_equal(_lowpass_to(x, SR, 16000.0), x)


class TestRoundTrip:
    def test_matching_a_walled_source_leaves_the_passband_alone(self):
        """The property the fix depends on: only the invented octave moves."""
        source = _brickwall(_noise(seed=1), 16000.0)
        render = _noise(seed=2)                       # full-band, as the model emits
        bw = _detect_bandwidth_hz(source, SR)
        fixed = _lowpass_to(render, SR, bw)
        freqs = np.fft.rfftfreq(render.size, 1.0 / SR)
        pw = lambda s, m: float(np.mean(np.abs(np.fft.rfft(s))[m] ** 2))
        assert pw(fixed, freqs >= 18000) < pw(render, freqs >= 18000) * 1e-2
        assert pw(fixed, freqs <= 12000) == pytest.approx(
            pw(render, freqs <= 12000), rel=0.05)

    def test_full_band_source_means_no_filtering(self):
        source = _noise(seed=3)
        assert _detect_bandwidth_hz(source, SR) >= NYQUIST * 0.95


def test_setting_is_declared_in_the_single_contract():
    """The codebase carries a scar from three drifting copies of this list."""
    from auto_voice.runtime_contract import PIPELINE_SETTING_KEYS
    assert "fork_hq_match_source_bandwidth" in PIPELINE_SETTING_KEYS


class TestMeasureTheSourceNotTheStem:
    """Regression: the fix must measure the ORIGINAL mix, not the vocal stem.

    Separators emit their own energy above the source's wall. Measuring the
    stem therefore reports full-band for a band-limited song and the match
    silently no-ops. Caught end-to-end on "One Last Time": source walls at
    16 kHz, its vocal stem measured 22050, and the render kept ~40 dB of
    invented top octave that the fix existed to remove.
    """

    def test_separator_hf_defeats_stem_based_detection(self):
        """The bug, pinned: a stem with added HF hides the source's wall."""
        # Wall at 12 kHz: far enough from Nyquist that window leakage above it
        # cannot blur the source/stem contrast this test is about. (Right at
        # 16 kHz the detector reads a perfect synthetic wall as ~17 kHz - real
        # files measure exact, but synthetic white noise maximises leakage.)
        wall = 12000.0
        source = _brickwall(_noise(seed=11), wall)
        # A stem the separator has added its own top-octave energy to.
        above = _noise(seed=13) - _brickwall(_noise(seed=13), wall)
        stem = source + above * 0.5

        src_bw = _detect_bandwidth_hz(source, SR)
        stem_bw = _detect_bandwidth_hz(stem, SR)
        assert src_bw < NYQUIST * 0.95, "a walled source must trigger the match"
        assert stem_bw > NYQUIST * 0.95, "stem reads full-band - the bug"
        assert stem_bw - src_bw > 5000.0, (
            f"stem {stem_bw} vs source {src_bw}: measuring the stem loses the wall"
        )

    def test_source_measurement_still_finds_the_wall(self):
        source = _brickwall(_noise(seed=14), 16000.0)
        bw = _detect_bandwidth_hz(source, SR)
        assert bw < NYQUIST * 0.95, "a walled source must trigger the match"
        assert bw == pytest.approx(16000.0, abs=600)


def test_call_site_measures_the_original_mix_not_the_stem():
    """Guard the wiring itself - a unit-correct helper called on the wrong
    signal is exactly the defect this file exists to prevent."""
    src = Path(__file__).resolve().parents[1] / (
        "src/auto_voice/inference/singing_conversion_pipeline.py")
    text = src.read_text()
    call = [ln for ln in text.splitlines() if "_detect_bandwidth_hz(" in ln
            and "def " not in ln]
    assert call, "no call to _detect_bandwidth_hz found"
    joined = "\n".join(call)
    assert "voc_mono" not in joined, (
        "bandwidth is being detected from the separated vocal stem again; "
        "the separator's own HF hides the source's wall"
    )
