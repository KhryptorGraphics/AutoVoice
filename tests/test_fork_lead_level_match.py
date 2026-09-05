"""The converted lead must come back at the source vocal's level.

Only the backing lines were ever level-matched; the lead was mixed at whatever the
checkpoint emitted. Conor's landed within 0.2 dB by luck, Brandy's was +4.2 dB on
every Hero render ("loud"). Same active-span RMS match as the lines, bounded.
"""
import numpy as np
import pytest

from auto_voice.inference.singing_conversion_pipeline import (
    SingingConversionPipeline, _LEAD_GAIN_MAX, _LEAD_GAIN_MIN)


def _pipe(**cfg):
    p = SingingConversionPipeline.__new__(SingingConversionPipeline)
    p.config = dict(cfg)
    return p


def _voice(sr, secs=4.0, amp=0.3, f=220.0):
    t = np.arange(int(sr * secs)) / sr
    x = amp * np.sin(2 * np.pi * f * t).astype(np.float32)
    x[: int(0.5 * sr)] = 0.0          # a silent lead-in the active-span measure must ignore
    return x


def test_hot_lead_is_brought_down_to_source_level():
    sr = 44100
    src = _voice(sr, amp=0.10)
    conv = _voice(sr, amp=0.40)        # +12 dB hot, like the Brandy renders
    p = _pipe()
    out = p._match_lead_level(conv, src, sr)
    active = slice(int(0.5 * sr), None)
    ratio = np.sqrt(np.mean(out[active] ** 2)) / np.sqrt(np.mean(src[active] ** 2))
    assert abs(20 * np.log10(ratio)) < 0.5
    assert p._last_lead_gain_db == pytest.approx(-12.0, abs=0.5)


def test_gain_is_bounded_and_reported():
    sr = 44100
    src = _voice(sr, amp=0.5)
    conv = _voice(sr, amp=0.001)       # +54 dB wanted; must stop at the ceiling
    p = _pipe()
    out = p._match_lead_level(conv, src, sr)
    got = np.abs(out).max() / np.abs(conv).max()
    assert got == pytest.approx(_LEAD_GAIN_MAX, rel=1e-3)
    assert _LEAD_GAIN_MIN < 1.0 < _LEAD_GAIN_MAX


def test_setting_off_restores_unmatched_behaviour():
    sr = 44100
    src, conv = _voice(sr, amp=0.1), _voice(sr, amp=0.4)
    out = _pipe(fork_hq_match_lead_level=False)._match_lead_level(conv, src, sr)
    assert np.array_equal(out, conv)
