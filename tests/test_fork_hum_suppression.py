"""The between-word hum must go; the singing must not.

Synthetic: the source sings a tone, stops for a second, sings again. The
"conversion" reproduces the tones but adds a 300 Hz hum during the silence -
exactly the decoder defect (periodic content where the source is unvoiced).
"""
import numpy as np
import pytest

from auto_voice.inference.fork_hum_suppression import suppress_hum

SR = 16000


def _tone(f, sec, amp=0.3):
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def _rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)


@pytest.fixture
def clips():
    silence = np.zeros(SR, dtype=np.float32)
    source = np.concatenate([_tone(220, 1.0), silence, _tone(220, 1.0)])
    hum = _tone(300, 1.0, amp=0.03)                      # -20 dB hum in the gap
    converted = np.concatenate([_tone(220, 1.0), hum, _tone(220, 1.0)])
    return source, converted


def test_hum_in_the_gap_is_suppressed(clips):
    source, converted = clips
    out = suppress_hum(source, converted, SR)
    assert out.shape == converted.shape and out.dtype == np.float32
    gap = slice(SR + SR // 5, 2 * SR - SR // 5)          # inner 600 ms, clear of the fades
    before, after = _rms_db(converted[gap]), _rms_db(out[gap])
    assert before - after >= 10.0, f"hum only dropped {before - after:.1f} dB"


def test_singing_is_untouched(clips):
    source, converted = clips
    out = suppress_hum(source, converted, SR)
    sung = slice(SR // 5, SR - SR // 5)                   # inner part of the first note
    assert abs(_rms_db(out[sung]) - _rms_db(converted[sung])) < 1.0


def test_empty_input_passes_through():
    assert suppress_hum(np.zeros(0, np.float32), np.zeros(0, np.float32), SR).shape == (0,)
