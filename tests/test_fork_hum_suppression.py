"""The between-word hum must go; the singing must not.

Synthetic: the source sings a tone, stops for a second, sings again. The
"conversion" reproduces the tones but adds a 300 Hz hum during the silence -
exactly the decoder defect (periodic content where the source is unvoiced).
"""
import numpy as np
import pytest

import auto_voice.inference.fork_hum_suppression as fh
from auto_voice.inference.fork_hum_suppression import suppress_hum
from auto_voice.inference.fork_inharmonic_notch import source_pitch

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


def test_voicing_frames_stay_on_the_time_axis():
    """Frame j of the voicing must be the frame at j * 10 ms on a long take.

    crepe runs in 20 s chunks and, with pad=True, a full chunk yields one
    frame more than its duration. Stitched as-is, every later frame sat
    10 ms late per chunk (12 frames by the end of Hero), so the duck meant
    for a gap landed on the note after it.
    """
    y = np.zeros(65 * SR, dtype=np.float32)
    y[62 * SR:63 * SR] = _tone(220, 1.0)     # chunk 3: 3 frames of drift
    per = fh._voicing(y, SR)
    assert len(per) == 1 + len(y) // 160
    voiced = np.flatnonzero(per > fh._VOICED)
    assert abs(voiced.mean() - 6250) <= 1.5, (
        f"burst centre at frame {voiced.mean():.1f}, expected 6250")


def test_duck_spares_a_note_pyin_hears_sung(monkeypatch):
    """crepe's periodicity collapses on some high belts (Hero, ~990 Hz: 0.13
    while pyin's voicing prob is 0.95+); the duck must not take such a note
    down. Control: without the pyin guard the same note IS ducked."""
    pad = np.zeros(SR // 2, dtype=np.float32)
    source = np.concatenate([pad, _tone(880, 1.5), pad])
    converted = source.copy()
    frames = 1 + len(source) // 160
    monkeypatch.setattr(fh, '_voicing', lambda y, sr: np.full(
        frames, 0.1 if y is source else 0.9, dtype=np.float32))
    note = slice(SR // 2 + SR // 5, 2 * SR - SR // 5)
    pitch = source_pitch(source, SR)
    guarded = suppress_hum(source, converted, SR, pitch=pitch)
    unguarded = suppress_hum(source, converted, SR, pitch=pitch._replace(
        prob=np.zeros_like(pitch.prob)))
    assert abs(_rms_db(guarded[note]) - _rms_db(converted[note])) < 1.0
    assert _rms_db(converted[note]) - _rms_db(unguarded[note]) > 10.0
