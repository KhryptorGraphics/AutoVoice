"""Kept-backing gain and pitch-shift symmetry in _convert_multi_speaker.

When _convert_backing_stack's gates decline every line, it returns the raw
separation residue unmatched in level (no _finish_backing RMS-restore) - that
residue was being summed directly under a level-matched converted lead with
no gain-matching at all, verified against a real conversion's metadata (47.8s
of raw "doubles", backing_energy_ratio 0.1972). These tests pin the fix: two
independent attenuation points gated strictly on mode == 'kept' (never
'partial', which already went through _finish_backing; never content that
was deliberately preserved or never attempted), plus pitch-shift now reaching
backing/simul_backing the same way it already reaches primary_track.

Heavy internals (diarizer, karaoke split, ModelManager) are mocked - this
targets the mixing arithmetic in _convert_multi_speaker itself, not
diarization or model inference.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

N = 1600  # 0.1s @ 16kHz - shape only matters for length-matching, not content


def _bare_pipeline(config):
    """A SingingConversionPipeline instance with no __init__ side effects.

    _convert_multi_speaker only reads self.config and calls self._get_diarizer,
    self._select_speaker_spans, self._convert_backing_stack, and (for the
    karaoke-model separator) self._split_lead_backing_karaoke - everything
    else here is instance-attribute mocks shadowing the real methods.
    """
    obj = object.__new__(SingingConversionPipeline)
    obj.config = config
    obj._get_diarizer = MagicMock(return_value=MagicMock(diarize=MagicMock(return_value=object())))
    obj._select_speaker_spans = MagicMock()
    obj._convert_backing_stack = MagicMock()
    obj._split_lead_backing_karaoke = MagicMock()
    return obj


def _mm(infer_side_effect):
    mm = MagicMock()
    mm.infer = MagicMock(side_effect=infer_side_effect)
    return mm


class TestKeptBackingGain:
    def test_kept_mode_is_attenuated(self):
        """mode='kept' (gates declined every line): gain applies."""
        pipeline = _bare_pipeline({
            'multi_speaker_convert_backing': True,
            'multi_speaker_kept_backing_gain': 0.5,
            'multi_speaker_min_coverage': 0.0,
        })
        pipeline._select_speaker_spans.return_value = (
            [(0.0, 1.0)], [(0.0, 1.0)], {},
        )
        primary_track = np.full(N, 0.3, dtype=np.float32)
        backing = np.full(N, 0.8, dtype=np.float32)
        pipeline._convert_backing_stack.return_value = (
            backing.copy(), {'mode': 'kept', 'lines_detected': 2, 'lines_converted': 0},
        )
        mm = _mm(lambda track, *a, **k: track)  # identity: converted_primary == primary_track

        with patch(
            "auto_voice.audio.multi_artist_separator.build_speaker_track",
            side_effect=[primary_track, backing],
        ):
            result = pipeline._convert_multi_speaker(
                np.zeros(N, dtype=np.float32), 16000, "profile-1", mm, pitch_shift=0,
            )

        assert result is not None, "conversion returned None - a mock likely mismatched a call signature"
        combined, info = result
        # backing attenuated (0.8 * 0.5 = 0.4) + untouched primary (0.3) = 0.7
        assert np.allclose(combined, 0.7, atol=1e-5)
        assert info['backing_mode'] == 'kept'
        assert info['harmony_lines'] == {'detected': 2, 'converted': 0}

    def test_partial_mode_is_not_double_attenuated(self):
        """mode='partial' already passed through _finish_backing - no second gain."""
        pipeline = _bare_pipeline({
            'multi_speaker_convert_backing': True,
            'multi_speaker_kept_backing_gain': 0.5,
            'multi_speaker_min_coverage': 0.0,
        })
        pipeline._select_speaker_spans.return_value = (
            [(0.0, 1.0)], [(0.0, 1.0)], {},
        )
        primary_track = np.full(N, 0.3, dtype=np.float32)
        backing = np.full(N, 0.8, dtype=np.float32)
        pipeline._convert_backing_stack.return_value = (
            backing.copy(), {'mode': 'partial', 'lines_detected': 2, 'lines_converted': 1},
        )
        mm = _mm(lambda track, *a, **k: track)

        with patch(
            "auto_voice.audio.multi_artist_separator.build_speaker_track",
            side_effect=[primary_track, backing],
        ):
            result = pipeline._convert_multi_speaker(
                np.zeros(N, dtype=np.float32), 16000, "profile-1", mm, pitch_shift=0,
            )

        assert result is not None
        combined, info = result
        # backing untouched (0.8) + primary (0.3) = 1.1 - NOT 0.3 + 0.8*0.5=0.7
        assert np.allclose(combined, 1.1, atol=1e-5)
        assert info['backing_mode'] == 'partial'

    def test_never_attempted_backing_is_not_attenuated(self):
        """multi_speaker_convert_backing off: backing never reaches
        _convert_backing_stack at all, so it must never be scaled either -
        both this case and 'kept' otherwise collapse to the same
        info['backing_mode'] string, so the fix must key off the local
        harmony result, not that string."""
        pipeline = _bare_pipeline({
            'multi_speaker_convert_backing': False,
            'multi_speaker_kept_backing_gain': 0.5,
            'multi_speaker_min_coverage': 0.0,
        })
        pipeline._select_speaker_spans.return_value = (
            [(0.0, 1.0)], [(0.0, 1.0)], {},
        )
        primary_track = np.full(N, 0.3, dtype=np.float32)
        backing = np.full(N, 0.8, dtype=np.float32)
        mm = _mm(lambda track, *a, **k: track)

        with patch(
            "auto_voice.audio.multi_artist_separator.build_speaker_track",
            side_effect=[primary_track, backing],
        ):
            result = pipeline._convert_multi_speaker(
                np.zeros(N, dtype=np.float32), 16000, "profile-1", mm, pitch_shift=0,
            )

        assert result is not None
        combined, info = result
        pipeline._convert_backing_stack.assert_not_called()
        assert np.allclose(combined, 1.1, atol=1e-5)
        assert info['backing_mode'] == 'kept'

    def test_preserved_speaker_backing_is_never_attenuated(self):
        """preserved_active=True: the diarization-span backing carries a
        deliberately-preserved co-singer verbatim and must never be scaled -
        only a declined simul_backing (karaoke doubles) may be."""
        pipeline = _bare_pipeline({
            'multi_speaker_convert_backing': True,
            'multi_speaker_kept_backing_gain': 0.5,
            'multi_speaker_min_coverage': 0.0,
            'multi_speaker_separator': 'karaoke_model',
        })
        voc_for_spans = np.zeros(N, dtype=np.float32)
        simul_backing = np.full(N, 0.6, dtype=np.float32)
        pipeline._split_lead_backing_karaoke.return_value = (
            voc_for_spans, simul_backing.copy(),
            {'backing_s': 5.0, 'backing_energy_ratio': 0.2},
        )
        pipeline._select_speaker_spans.return_value = (
            [(0.0, 1.0)], [(0.0, 1.0)], {'preserved_speakers': ['SPEAKER_01']},
        )
        primary_track = np.full(N, 0.3, dtype=np.float32)
        preserved_backing = np.full(N, 0.9, dtype=np.float32)  # verbatim co-singer, must survive untouched
        pipeline._convert_backing_stack.return_value = (
            simul_backing.copy(), {'mode': 'kept', 'lines_detected': 1, 'lines_converted': 0},
        )
        mm = _mm(lambda track, *a, **k: track)

        with patch(
            "auto_voice.audio.multi_artist_separator.build_speaker_track",
            side_effect=[primary_track, preserved_backing],
        ):
            result = pipeline._convert_multi_speaker(
                np.zeros(N, dtype=np.float32), 16000, "profile-1", mm, pitch_shift=0,
            )

        assert result is not None
        combined, info = result
        pipeline._convert_backing_stack.assert_called_once()
        # preserved backing (0.9, untouched) + attenuated simul_backing
        # (0.6 * 0.5 = 0.3) + primary (0.3) = 1.5. A naive single gain applied
        # to the already-merged backing would instead give (0.9+0.6)*0.5 + 0.3
        # = 1.05 - the wrong number this test rules out.
        assert np.allclose(combined, 1.5, atol=1e-5)


class TestPitchShiftSymmetry:
    def test_pitch_shift_reaches_backing_when_set(self):
        """Kept backing must transpose with the lead, or a shifted lead and
        an unshifted backing beat/clash against each other."""
        pipeline = _bare_pipeline({
            'multi_speaker_convert_backing': False,
            'multi_speaker_min_coverage': 0.0,
        })
        pipeline._select_speaker_spans.return_value = (
            [(0.0, 1.0)], [(0.0, 1.0)], {},
        )
        primary_track = np.full(N, 0.3, dtype=np.float32)
        backing = np.full(N, 0.8, dtype=np.float32)
        mm = _mm(lambda track, *a, **k: track)  # identity, so shift-order is the only variable

        with patch(
            "auto_voice.audio.multi_artist_separator.build_speaker_track",
            side_effect=[primary_track, backing],
        ), patch(
            "librosa.effects.pitch_shift",
            side_effect=lambda arr, sr, n_steps: arr * 2,  # distinguishable, order-preserving transform
        ) as mock_shift:
            result = pipeline._convert_multi_speaker(
                np.zeros(N, dtype=np.float32), 16000, "profile-1", mm, pitch_shift=2.0,
            )

        assert result is not None
        combined, info = result
        # backing shifted (0.8*2=1.6) + primary shifted-then-"converted" (0.3*2=0.6) = 2.2
        assert np.allclose(combined, 2.2, atol=1e-5)
        calls_with_n_steps_2 = [c for c in mock_shift.call_args_list if c.kwargs.get('n_steps') == 2.0]
        assert len(calls_with_n_steps_2) == 2  # backing + primary_track

    def test_no_pitch_shift_call_when_unset(self):
        pipeline = _bare_pipeline({
            'multi_speaker_convert_backing': False,
            'multi_speaker_min_coverage': 0.0,
        })
        pipeline._select_speaker_spans.return_value = (
            [(0.0, 1.0)], [(0.0, 1.0)], {},
        )
        primary_track = np.full(N, 0.3, dtype=np.float32)
        backing = np.full(N, 0.8, dtype=np.float32)
        mm = _mm(lambda track, *a, **k: track)

        with patch(
            "auto_voice.audio.multi_artist_separator.build_speaker_track",
            side_effect=[primary_track, backing],
        ), patch("librosa.effects.pitch_shift") as mock_shift:
            result = pipeline._convert_multi_speaker(
                np.zeros(N, dtype=np.float32), 16000, "profile-1", mm, pitch_shift=0,
            )

        assert result is not None
        mock_shift.assert_not_called()
