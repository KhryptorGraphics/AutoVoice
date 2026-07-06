"""Tests for auto_voice.audio.post_mix (synthetic, fast, CPU-only)."""

import numpy as np

from auto_voice.audio.post_mix import (
    transfer_loudness_envelope,
    voicing_gated_passthrough,
)

SR = 16000
HOP = 160  # 100 fps


def _env(x, frame_len=800, hop=200):
    """Independent short-term RMS envelope for assertions."""
    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, frame_len)[::hop]
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))


# ---------------------------------------------------------------- loudness

def test_loudness_transfer_follows_source_envelope():
    t = np.arange(2 * SR) / SR
    am = 0.5 + 0.4 * np.sin(2 * np.pi * 2.0 * t)      # amplitude-modulated source
    source = (am * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    converted = (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)  # flat

    out = transfer_loudness_envelope(source, converted, SR)

    assert out.dtype == np.float32
    assert out.shape == converted.shape
    corr = np.corrcoef(_env(out), _env(source))[0, 1]
    assert corr > 0.9, f"envelope correlation too low: {corr:.3f}"


def test_source_silence_attenuates_converted():
    t = np.arange(SR) / SR
    converted = (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    source = np.zeros(SR, dtype=np.float32)
    out = transfer_loudness_envelope(source, converted, SR)
    assert np.all(np.isfinite(out))
    out_rms = np.sqrt(np.mean(out.astype(np.float64) ** 2))
    cvt_rms = np.sqrt(np.mean(converted.astype(np.float64) ** 2))
    assert out_rms < 0.3 * cvt_rms  # -12 dB clamp -> ~0.25x


def test_max_gain_clamp_respected():
    t = np.arange(SR) / SR
    source = np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    converted = (1e-3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    out = transfer_loudness_envelope(source, converted, SR, max_gain_db=12.0)
    limit = 1e-3 * 10.0 ** (12.0 / 20.0)
    assert np.max(np.abs(out)) <= limit * 1.02


def test_all_zero_inputs_and_length_mismatch():
    source = np.zeros(SR, dtype=np.float32)
    converted = np.zeros(SR + 123, dtype=np.float32)  # converted longer
    out = transfer_loudness_envelope(source, converted, SR)
    assert out.shape == converted.shape
    assert np.all(np.isfinite(out))


# ------------------------------------------------------------- passthrough

def _make_gap_case():
    """Voiced tone with a white-noise 'consonant' burst in an f0==0 gap."""
    n_frames = 100
    n = n_frames * HOP
    t = np.arange(n) / SR
    tone = (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)

    f0 = np.full(n_frames, 220.0, dtype=np.float32)
    f0[40:50] = 0.0

    rng = np.random.default_rng(1234)
    source = tone.copy()
    source[40 * HOP:50 * HOP] = rng.uniform(-0.3, 0.3, 10 * HOP).astype(np.float32)
    converted = tone.copy()  # tone only, no consonant
    return source, converted, f0


def test_passthrough_injects_consonant_energy():
    source, converted, f0 = _make_gap_case()
    out = voicing_gated_passthrough(source, converted, SR, f0, HOP)

    assert out.dtype == np.float32
    assert out.shape == converted.shape

    def hf(x):  # first-difference energy ~ high-frequency content
        return np.mean(np.diff(x.astype(np.float64)) ** 2)

    inner = slice(41 * HOP, 49 * HOP)
    assert hf(out[inner]) > 3.0 * hf(converted[inner])

    # far from the gap the output stays the converted signal
    assert np.allclose(out[:35 * HOP], converted[:35 * HOP], atol=1e-4)
    assert np.allclose(out[55 * HOP:], converted[55 * HOP:], atol=1e-4)


def test_passthrough_no_clicks():
    source, converted, f0 = _make_gap_case()
    out = voicing_gated_passthrough(source, converted, SR, f0, HOP)
    assert np.all(np.isfinite(out))
    # bounded sample-to-sample delta (a hard mask switch / indexing bug would
    # produce ~full-scale steps; the blended noise stays well under this)
    assert np.max(np.abs(np.diff(out))) < 0.9


def test_mix_zero_returns_converted_unchanged():
    source, converted, f0 = _make_gap_case()
    out = voicing_gated_passthrough(source, converted, SR, f0, HOP, mix=0.0)
    assert np.array_equal(out, converted)
