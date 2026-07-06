"""Tests for auto_voice.inference.f0_utils (synthetic, fast, CPU-only)."""

import numpy as np

from auto_voice.inference.f0_utils import postprocess_f0, semitone_shift_f0


def test_vibrato_preserved():
    # 440 Hz +/- 30 cents at 6 Hz, 100 fps frame rate
    frames = 400
    t = np.arange(frames) / 100.0
    cents_in = 30.0 * np.sin(2 * np.pi * 6.0 * t)
    f0 = (440.0 * 2.0 ** (cents_in / 1200.0)).astype(np.float32)

    out = postprocess_f0(f0)

    assert out.dtype == np.float32
    assert np.all(out > 0)
    cents_out = 1200.0 * np.log2(out.astype(np.float64) / 440.0)
    rms_err = np.sqrt(np.mean((cents_out - cents_in) ** 2))
    assert rms_err < 5.0, f"vibrato distorted: {rms_err:.2f} cents RMS"


def test_octave_spike_corrected():
    f0 = np.full(100, 200.0, dtype=np.float32)
    f0[40:48] *= 2.0  # 8-frame octave doubling
    out = postprocess_f0(f0)
    assert np.allclose(out, 200.0, rtol=1e-3)


def test_unvoiced_boundaries_preserved():
    f0 = np.zeros(120, dtype=np.float32)
    f0[30:90] = 220.0
    out = postprocess_f0(f0)
    assert np.all(out[:30] == 0)
    assert np.all(out[90:] == 0)
    assert np.all(out[30:90] > 0)


def test_nan_handled_and_single_gap_filled():
    f0 = np.full(60, 220.0, dtype=np.float32)
    f0[10] = np.nan       # single NaN inside voiced run -> interpolated
    f0[30:40] = np.nan    # NaN block -> unvoiced zeros
    out = postprocess_f0(f0)
    assert np.all(np.isfinite(out))
    assert np.isclose(out[10], 220.0, rtol=1e-3)
    assert np.all(out[30:40] == 0)


def test_isolated_island_removed():
    f0 = np.zeros(50, dtype=np.float32)
    f0[25:27] = 300.0  # 2-frame voiced island
    out = postprocess_f0(f0)
    assert np.all(out == 0)


def test_semitone_shift():
    f0 = np.array([0.0, 220.0, np.nan, 440.0], dtype=np.float32)
    out = semitone_shift_f0(f0, 12.0)
    assert out.dtype == np.float32
    assert out[0] == 0.0 and out[2] == 0.0
    assert np.isclose(out[1], 440.0, rtol=1e-5)
    assert np.isclose(out[3], 880.0, rtol=1e-5)
