"""Unit checks for build_speaker_track (per-speaker crossfade track builder)."""
import numpy as np

from auto_voice.audio.multi_artist_separator import build_speaker_track


def test_places_span_and_silences_the_rest():
    sr = 1000
    vocals = np.ones(sr, dtype=np.float32)  # 1s of constant 1.0
    # One 0.4s span in the middle, 50ms fade.
    track = build_speaker_track(vocals, sr, [(0.3, 0.7)], fade_ms=50.0)

    assert track.shape == vocals.shape
    assert track.dtype == np.float32
    # Everything outside the span is exactly zero.
    assert np.all(track[:300] == 0.0)
    assert np.all(track[700:] == 0.0)
    # Interior (past both 50-sample fades) is untouched -> equals the source.
    assert np.allclose(track[360:640], 1.0)
    # Fade edges are attenuated (start near 0, ramps up).
    assert track[300] < 0.05
    assert track[300] < track[349]


def test_preserves_negative_samples():
    # Regression: a max()-against-zero placement would half-wave rectify this to
    # zero. Additive placement must keep the negatives.
    sr = 1000
    vocals = np.full(sr, -0.5, dtype=np.float32)
    track = build_speaker_track(vocals, sr, [(0.2, 0.8)], fade_ms=10.0)
    assert track.min() < -0.4  # interior stays negative


def test_overlapping_spans_are_unioned():
    sr = 1000
    vocals = np.ones(sr, dtype=np.float32)
    # Overlapping spans union into one continuous span: no 2x double-add in the
    # overlap, no mid-phrase fade dip at the join.
    track = build_speaker_track(vocals, sr, [(0.1, 0.5), (0.4, 0.9)], fade_ms=0.0)
    assert np.allclose(track[100:900], 1.0)
    assert np.all(track[:100] == 0.0)
    assert np.all(track[900:] == 0.0)

    # Same with fades: interior of the union (past the outer edge fades) is
    # continuous — the old per-span behaviour dipped to ~0 around sample 450.
    track = build_speaker_track(vocals, sr, [(0.1, 0.5), (0.4, 0.9)], fade_ms=50.0)
    assert np.allclose(track[160:840], 1.0)


def test_touching_spans_have_no_interior_dip():
    sr = 1000
    vocals = np.ones(sr, dtype=np.float32)
    # Abutting diarization windows (end == next start) must render continuously.
    track = build_speaker_track(vocals, sr, [(0.1, 0.5), (0.5, 0.9)], fade_ms=50.0)
    assert np.allclose(track[160:840], 1.0)


if __name__ == "__main__":
    test_places_span_and_silences_the_rest()
    test_preserves_negative_samples()
    test_overlapping_spans_are_unioned()
    test_touching_spans_have_no_interior_dip()
    print("build_speaker_track: all checks passed")
