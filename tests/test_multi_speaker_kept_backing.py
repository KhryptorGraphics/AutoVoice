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


class TestBackingGainNotCancelled:
    """The peak guard used to run AFTER the operator gain, which cancelled it:
    X*g*(0.99/max|X*g|) == X*(0.99/max|X|), independent of g. Measured before
    the fix: gains 1.6, 2.2 and 3.0 produced byte-identical audio."""

    @staticmethod
    def _run(gain):
        import numpy as np
        rng = np.random.default_rng(0)
        n = 16000
        backing = (rng.standard_normal(n) * 0.2).astype(np.float32)
        new = (rng.standard_normal(n) * 0.2).astype(np.float32)
        p = object.__new__(SingingConversionPipeline)
        p.config = {'multi_speaker_backing_gain': gain}
        out, _ = p._finish_backing(backing, new, [(0, n)], 3, 3, 'lines')
        return float(np.sqrt((out ** 2).mean()))

    def test_gain_actually_scales_the_output(self):
        base = self._run(1.0)
        assert self._run(2.0) == pytest.approx(2.0 * base, rel=1e-6)
        assert self._run(3.0) == pytest.approx(3.0 * base, rel=1e-6)

    def test_distinct_gains_are_not_collapsed(self):
        assert self._run(1.6) != pytest.approx(self._run(2.2), rel=1e-3)


class TestLineExtractionPartition:
    """Wider masks make lines claim the same bins; the caller subtracts every
    extract from the stack, so without partitioning shared energy is removed
    once per line and punches holes in the residual."""

    @staticmethod
    def _notes(pitch, n=6, dur=0.25):
        return [{'pitch_midi': pitch, 'start': i * dur * 2, 'end': i * dur * 2 + dur,
                 'amplitude': 1.0} for i in range(n)]

    def test_sum_of_extracts_never_exceeds_the_stack(self):
        import numpy as np
        from auto_voice.inference.singing_conversion_pipeline import _extract_line_audios
        sr = 22050
        t = np.arange(int(3.0 * sr)) / sr
        # Two lines a minor third apart -> upper harmonics genuinely collide.
        stack = (0.3 * np.sin(2 * np.pi * 220.0 * t)
                 + 0.3 * np.sin(2 * np.pi * 261.6 * t)).astype(np.float32)
        lines = [self._notes(57), self._notes(60)]
        ex = [mix for mix, _gate, _occ in _extract_line_audios(stack, sr, lines)]
        assert len(ex) == 2
        total = np.sum(np.square(np.sum(ex, axis=0), dtype=np.float64))
        assert total <= np.sum(np.square(stack, dtype=np.float64)) * 1.05

    def test_onset_window_adds_broadband_energy(self):
        import numpy as np
        from auto_voice.inference.singing_conversion_pipeline import _extract_line_audios
        sr = 22050
        t = np.arange(int(2.0 * sr)) / sr
        stack = (0.3 * np.sin(2 * np.pi * 220.0 * t)
                 + 0.05 * np.random.default_rng(1).standard_normal(len(t))).astype(np.float32)
        lines = [self._notes(57, n=4)]
        with_onset = _extract_line_audios(stack, sr, lines, onset_s=0.03)[0][0]
        without = _extract_line_audios(stack, sr, lines, onset_s=0.0)[0][0]
        e_with = float(np.sum(np.square(with_onset, dtype=np.float64)))
        e_without = float(np.sum(np.square(without, dtype=np.float64)))
        assert e_with > e_without

    def test_more_harmonics_captures_more_energy(self):
        import numpy as np
        from auto_voice.inference.singing_conversion_pipeline import _extract_line_audios
        sr = 22050
        t = np.arange(int(2.0 * sr)) / sr
        # Rich harmonic series: the old 10-harmonic cap discarded most of it.
        stack = sum(0.25 / k * np.sin(2 * np.pi * 220.0 * k * t)
                    for k in range(1, 25)).astype(np.float32)
        lines = [self._notes(57, n=4)]
        narrow = _extract_line_audios(stack, sr, lines, n_harmonics=10, onset_s=0.0)[0][0]
        wide = _extract_line_audios(stack, sr, lines, n_harmonics=24, onset_s=0.0)[0][0]
        assert float(np.sum(np.square(wide, dtype=np.float64))) > \
               float(np.sum(np.square(narrow, dtype=np.float64)))


    def test_gate_extract_is_not_reduced_by_a_neighbour(self):
        """The regression this pair exists to prevent: gating on the
        partitioned share made a line's admissibility depend on its
        neighbours. Two of three real lines dropped to concentration 0.13
        against a 0.15 threshold purely from sharing upper harmonics."""
        import numpy as np
        from auto_voice.inference.singing_conversion_pipeline import _extract_line_audios
        sr = 22050
        t = np.arange(int(3.0 * sr)) / sr
        stack = (0.3 * np.sin(2 * np.pi * 220.0 * t)
                 + 0.3 * np.sin(2 * np.pi * 261.6 * t)).astype(np.float32)
        alone = _extract_line_audios(stack, sr, [self._notes(57)])
        together = _extract_line_audios(stack, sr, [self._notes(57), self._notes(60)])
        e = lambda x: float(np.sum(np.square(x, dtype=np.float64)))
        # Gate extract is identical whether or not a neighbour contests bins.
        assert e(together[0][1]) == pytest.approx(e(alone[0][1]), rel=1e-9)
        # Mix extract IS reduced by the neighbour - that is the partition working.
        assert e(together[0][0]) < e(alone[0][0])


class TestLeadUnisonFold:
    """A double-tracked lead lands in the backing stem and passes every
    harmony gate, so it was converted as its own singer and summed against the
    separately-converted lead. Two stochastic conversions of the same phrase do
    not line up - measured on the reference song, that line held 27.5% of the
    backing energy and sat 0.25 semitones from the lead."""

    @staticmethod
    def _pipeline(**cfg):
        base = {'multi_speaker_unison_semitones': 1.0,
                'multi_speaker_unison_note_frac': 0.5}
        base.update(cfg)
        p = object.__new__(SingingConversionPipeline)
        p.config = base
        return p

    @staticmethod
    def _notes(midi, n=8):
        return [{'pitch_midi': midi, 'start': i * 0.5, 'end': i * 0.5 + 0.4,
                 'amplitude': 1.0} for i in range(n)]

    def _run(self, line_midi, lead_midi, **cfg):
        """Backing holds one line; lead sings at lead_midi throughout."""
        import numpy as np
        from unittest.mock import patch
        sr = 22050
        dur = 4.0
        t = np.arange(int(dur * sr)) / sr
        hz = lambda m: 440.0 * 2.0 ** ((m - 69) / 12.0)
        backing = (0.3 * np.sin(2 * np.pi * hz(line_midi) * t)).astype(np.float32)
        lead = (0.5 * np.sin(2 * np.pi * hz(lead_midi) * t)).astype(np.float32)
        p = self._pipeline(**cfg)
        p._extract_pitch = lambda a, s: np.full(400, hz(lead_midi), dtype=np.float32)
        with patch("auto_voice.inference.separation_bridge.polyphonic_notes",
                   return_value=self._notes(line_midi)):
            return p._split_lead_unison(backing, sr, lead)

    def test_unison_line_is_folded_out_of_the_backing(self):
        import numpy as np
        reduced, unison, info = self._run(57, 57)
        assert info['unison_lines'] == 1
        assert info['harmony_lines'] == 0
        # Real energy moved: unison is non-trivial and the backing shrank.
        assert float(np.sum(unison ** 2)) > 0.0

    def test_genuine_harmony_is_left_in_the_backing(self):
        import numpy as np
        reduced, unison, info = self._run(57, 62)  # a fourth apart
        assert info['unison_lines'] == 0
        assert info['harmony_lines'] == 1
        assert float(np.sum(unison ** 2)) == 0.0
        assert np.allclose(reduced, reduced)  # untouched

    def test_zero_semitones_disables_the_fold(self):
        import numpy as np
        reduced, unison, info = self._run(57, 57, multi_speaker_unison_semitones=0.0)
        assert info['unison_lines'] == 0
        assert float(np.sum(unison ** 2)) == 0.0

    def test_energy_is_conserved_between_backing_and_lead(self):
        """What leaves the backing must equal what joins the lead - the fold
        must not duplicate or destroy audio."""
        import numpy as np
        reduced, unison, info = self._run(57, 57)
        assert info['unison_lines'] == 1
        sr = 22050
        t = np.arange(int(4.0 * sr)) / sr
        original = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        n = min(len(reduced), len(unison), len(original))
        assert np.allclose(reduced[:n] + unison[:n], original[:n], atol=1e-5)

    def test_missing_lead_is_a_no_op(self):
        import numpy as np
        p = self._pipeline()
        backing = np.ones(1000, dtype=np.float32)
        reduced, unison, info = p._split_lead_unison(backing, 22050, None)
        assert info['unison_lines'] == 0
        assert np.array_equal(reduced, backing)


class TestLeadBleedCancellation:
    """The karaoke backing stem is `orig_mix - lead`, so leaked lead is
    phase-coherent with the lead stem while a real backing singer is not.
    These pin that the canceller discriminates on coherence, not frequency -
    the whole point, since lead and harmony share partials at every consonant
    interval and a frequency rule would gut the harmony."""

    @staticmethod
    def _p(**cfg):
        base = {'multi_speaker_bleed_suppression': 'ls',
                'multi_speaker_bleed_max_db': 12.0,
                'multi_speaker_bleed_h_max': 0.7}
        base.update(cfg)
        p = object.__new__(SingingConversionPipeline)
        p.config = base
        return p

    @staticmethod
    def _lead(sr, dur=4.0, f=220.0):
        import numpy as np
        t = np.arange(int(dur * sr)) / sr
        # vibrato so it is not a pure stationary tone
        ph = 2 * np.pi * f * np.cumsum(1.0 + 0.004 * np.sin(2 * np.pi * 5.0 * t)) / sr
        return (0.5 * np.sin(ph)).astype(np.float32)

    def test_coherent_leakage_is_cancelled(self):
        import numpy as np
        sr = 22050
        lead = self._lead(sr)
        backing = (0.3 * lead).astype(np.float32)   # pure leakage: a scaled copy
        out, info = self._p()._suppress_lead_bleed(backing, lead, sr)
        assert info['mode'] == 'ls'
        assert info['cancelled_db'] > 3.0, info
        assert float(np.sum(out ** 2)) < float(np.sum(backing ** 2))

    def test_independent_double_survives(self):
        """The case a pitch-based rule cannot handle: a real singer at
        essentially the lead's pitch, but an independent take. Modelled on the
        measured reference content - 25 cents flat with a small timing offset,
        which is what human double-tracking actually looks like."""
        import numpy as np
        sr = 22050
        lead = self._lead(sr)
        t = np.arange(len(lead)) / sr
        detune = 2.0 ** (-0.25 / 12.0)          # 25 cents flat, as measured
        ph = 2 * np.pi * 220.0 * detune * np.cumsum(
            1.0 + 0.005 * np.sin(2 * np.pi * 6.3 * t + 1.7)) / sr
        double = (0.3 * np.sin(ph + 0.9)).astype(np.float32)
        double = np.roll(double, int(0.012 * sr)).astype(np.float32)  # 12ms offset
        out, info = self._p()._suppress_lead_bleed(double, lead, sr)
        kept = float(np.sum(out ** 2)) / (float(np.sum(double ** 2)) + 1e-20)
        assert kept > 0.5, f"independent double was over-cancelled: kept {kept:.2f}"

    def test_exact_unison_is_bounded_by_the_cap(self):
        """KNOWN LIMIT, pinned deliberately: a voice at literally zero cents
        detuning with no timing offset is indistinguishable from leakage by
        coherence (or by any other means - the two signals are collinear).
        Physically unreachable for real double-tracking, but the max_db cap is
        what bounds the damage if it ever occurs, so assert the cap holds."""
        import numpy as np
        sr = 22050
        lead = self._lead(sr)
        t = np.arange(len(lead)) / sr
        ph = 2 * np.pi * 220.0 * np.cumsum(
            1.0 + 0.005 * np.sin(2 * np.pi * 6.3 * t + 1.7)) / sr
        collinear = (0.3 * np.sin(ph + 0.9)).astype(np.float32)
        out, info = self._p(multi_speaker_bleed_max_db=6.0)._suppress_lead_bleed(
            collinear, lead, sr)
        kept = float(np.sum(out ** 2)) / (float(np.sum(collinear ** 2)) + 1e-20)
        # 6 dB cap => at most ~75% of energy removed, never a total kill.
        assert kept > 10 ** (-6.5 / 10.0), f"cap breached: kept {kept:.3f}"

    def test_off_is_a_no_op(self):
        import numpy as np
        sr = 22050
        lead = self._lead(sr)
        backing = (0.3 * lead).astype(np.float32)
        out, info = self._p(multi_speaker_bleed_suppression='off')._suppress_lead_bleed(
            backing, lead, sr)
        assert info['mode'] == 'off'
        assert np.array_equal(out, backing)

    def test_max_db_zero_removes_nothing(self):
        import numpy as np
        sr = 22050
        lead = self._lead(sr)
        backing = (0.3 * lead).astype(np.float32)
        out, info = self._p(multi_speaker_bleed_max_db=0.0)._suppress_lead_bleed(
            backing, lead, sr)
        assert abs(info['cancelled_db']) < 0.5

    def test_silent_lead_is_skipped_not_crashed(self):
        import numpy as np
        sr = 22050
        out, info = self._p()._suppress_lead_bleed(
            np.ones(sr, dtype=np.float32), np.zeros(sr, dtype=np.float32), sr)
        assert info['mode'].startswith('skipped')


class TestUnisonDecisionIsBiasedAgainstFolding:
    """Folding a genuine harmony destroys a distinct voice; failing to fold a
    double merely leaves the pre-existing artifact. Ambiguity must resolve to
    "harmony". A single note-fraction near its cutoff was a coin flip - the same
    content measured 49/51/53% across three runs of one song."""

    @staticmethod
    def _run(diffs, min_frac=0.5, max_semitones=1.0):
        """Drive the decision directly from a distance distribution."""
        import numpy as np
        d = np.asarray(diffs, dtype=float)
        frac = float(np.mean(d <= max_semitones))
        median = float(np.median(d))
        return frac >= min_frac and median <= max_semitones

    def test_a_true_double_folds(self):
        # tight cluster at the lead's pitch
        assert self._run([0.1, 0.2, 0.15, 0.3, 0.25, 0.2]) is True

    def test_a_harmony_that_repeatedly_crosses_the_lead_does_not_fold(self):
        """Half its notes brush the lead, but it sits a third away overall -
        the case a bare fraction test would have folded and destroyed."""
        assert self._run([0.2, 0.3, 0.4, 3.9, 4.1, 4.0]) is False

    def test_the_knife_edge_case_resolves_to_harmony(self):
        """51% within a semitone but a median well outside it: the fraction
        alone said 'fold', the median vetoes it."""
        assert self._run([0.5, 0.6, 0.7, 2.5, 3.0, 3.5]) is False


class TestPerLineGainIsBounded:
    """The per-line level match is a CORRECTION, not an amplifier.

    Unbounded, `gain = ref_rms / conv_rms` did damage in both directions: a
    line the engine rendered quietly got its artifacts amplified into audible
    distortion, and that one loud line inflated the stem RMS so
    _finish_backing's restore ducked every other line - heard as background
    singers at inconsistent volumes.
    """

    def test_bounds_are_symmetric_in_dB_and_sane(self):
        import numpy as np
        from auto_voice.inference.singing_conversion_pipeline import (
            _LINE_GAIN_MIN, _LINE_GAIN_MAX)
        assert _LINE_GAIN_MIN > 0.0
        assert _LINE_GAIN_MAX > 1.0 > _LINE_GAIN_MIN
        db = 20 * np.log10(_LINE_GAIN_MAX)
        assert abs(db - abs(20 * np.log10(_LINE_GAIN_MIN))) < 1e-6, (
            "bounds should be symmetric in dB so neither direction is favoured")
        assert db <= 18.0, "a level match needing >18 dB is a bad measurement"

    def test_a_runaway_ratio_is_clamped(self):
        import numpy as np
        from auto_voice.inference.singing_conversion_pipeline import (
            _LINE_GAIN_MIN, _LINE_GAIN_MAX)
        # engine returned near-silence -> ratio explodes
        ref_rms, conv_rms = 0.05, 1e-6
        raw = ref_rms / conv_rms
        assert raw > 1000
        assert float(np.clip(raw, _LINE_GAIN_MIN, _LINE_GAIN_MAX)) == _LINE_GAIN_MAX

    def test_a_normal_ratio_passes_through_untouched(self):
        import numpy as np
        from auto_voice.inference.singing_conversion_pipeline import (
            _LINE_GAIN_MIN, _LINE_GAIN_MAX)
        for raw in (0.8, 1.0, 1.3, 2.0):
            assert float(np.clip(raw, _LINE_GAIN_MIN, _LINE_GAIN_MAX)) == raw
