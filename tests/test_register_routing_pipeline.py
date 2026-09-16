"""Register routing wired into the per-speaker path.

Covers the three things the routing layer adds on top of the (untouched,
calibrated) foreign-voice detectors: per-span preserve reasons, the routing
decision reaching the right model, and the output-correction chain running once
per converted track instead of once on the mix.
"""
import numpy as np
import pytest

import auto_voice.inference.singing_conversion_pipeline as scp
import auto_voice.inference.vocal_register as vr
from auto_voice.inference.singing_conversion_pipeline import ConversionError

from tests.test_multi_speaker_routing import (
    VOC,
    VOC_SR,
    FakeDiarizer,
    make_pipeline,
    make_result,
    patch_voiced,
)


def fake_centroids(monkeypatch, cents):
    """Patch librosa's feature extraction so the mix-centroid gate is
    deterministic. ``cents`` is consumed in call order: primary, then each
    non-primary cluster."""
    import librosa as lr
    idx = {'c': 0}

    class FakeFeat:
        @staticmethod
        def spectral_centroid(y=None, sr=None):
            i = idx['c']
            idx['c'] += 1
            n = max(int(1.0 * sr / 512), 1)
            return np.full((1, n), cents[min(i, len(cents) - 1)])

        @staticmethod
        def rms(y=None, sr=None, **kw):
            n = max(len(y) // 512, 1)
            return (0.5 + 0.4 * np.cos(np.arange(n)))[np.newaxis, :]

    monkeypatch.setattr(lr, 'feature', FakeFeat())


def duet_pipeline(monkeypatch, cents, **config):
    """A two-cluster duet: SPEAKER_00 leads 0-30s, SPEAKER_01 sings 30-60s."""
    patch_voiced(monkeypatch, 0.9)      # both are clean monophonic melody
    p = make_pipeline(**config)
    fake_centroids(monkeypatch, cents)
    monkeypatch.setattr(p, '_load_song_mono',
                        lambda path, sr: np.ones(60 * sr, np.float32) * 0.1)
    result = make_result([(0.0, 30.0, 'SPEAKER_00'),
                          (30.0, 60.0, 'SPEAKER_01')])
    return p, result


class TestSpanReasons:
    """info['span_reasons'] must say WHY, not just THAT, a span is preserved."""

    def test_voice_category_reason(self, monkeypatch):
        p, result = duet_pipeline(monkeypatch, [3000.0, 750.0])
        _primary, backing, info = p._select_speaker_spans(
            result, VOC, VOC_SR, song_path='fake.wav')
        assert info['span_reasons'] == [[30.0, 60.0, 'voice_category',
                                         'SPEAKER_01']]
        # The flat span list stays exactly the union it always was.
        assert backing == [(30.0, 60.0)]

    def test_operator_preserve_reason_is_not_voice_category(self, monkeypatch):
        """Both land in `preserved` and both get roles['preserved'] — only the
        reason tells them apart, and only one of them is routable."""
        p, result = duet_pipeline(monkeypatch, [3000.0, 3000.0])
        _primary, _backing, info = p._select_speaker_spans(
            result, VOC, VOC_SR, preserve=['SPEAKER_01'], song_path='fake.wav')
        reasons = {r for _s, _e, r, _g in info['span_reasons']}
        assert reasons == {'operator'}
        assert info['roles'] == {'SPEAKER_01': 'preserved'}

    def test_low_voiced_backing_reason(self, monkeypatch):
        """A harmony stack is backing for a different reason, and stays put."""
        patch_voiced(monkeypatch, 0.2)   # below the 0.65 merge gate
        p = make_pipeline()
        result = make_result([(0.0, 30.0, 'SPEAKER_00'),
                              (30.0, 60.0, 'SPEAKER_01')])
        _primary, _backing, info = p._select_speaker_spans(
            result, VOC, VOC_SR)
        assert info['span_reasons'] == [[30.0, 60.0, 'low_voiced',
                                         'low_voiced']]

    def test_time_range_reason(self, monkeypatch):
        patch_voiced(monkeypatch, 0.9)
        p = make_pipeline()
        result = make_result([(0.0, 30.0, 'SPEAKER_00'),
                              (30.0, 60.0, 'SPEAKER_01')])
        _primary, _backing, info = p._select_speaker_spans(
            result, VOC, VOC_SR, preserve=['5-15'])
        reasons = {r for _s, _e, r, _g in info['span_reasons']}
        assert reasons == {'operator_range'}

    def test_reason_survives_clipping_against_the_lead(self, monkeypatch):
        """Clipping rewrites boundaries; the reason must ride along onto every
        surviving piece, which is why tagging happens before the clip."""
        patch_voiced(monkeypatch, 0.9)
        p = make_pipeline()
        result = make_result([(0.0, 30.0, 'SPEAKER_00'),
                              (30.0, 60.0, 'SPEAKER_01')])
        # Carve a range that straddles the lead/other boundary.
        _primary, backing, info = p._select_speaker_spans(
            result, VOC, VOC_SR, preserve=['10-20'])
        assert info['span_reasons'], "range must produce preserved spans"
        for start, end, reason, _g in info['span_reasons']:
            assert reason == 'operator_range'
            assert end > start
        assert [(s, e) for s, e, _r, _g in info['span_reasons']] == backing


class TestSymmetricVoiceCategoryGate:
    """'if converting either female or male' — the gate fires both ways."""

    def test_darker_cluster_preserved(self, monkeypatch):
        p, result = duet_pipeline(monkeypatch, [3000.0, 750.0])
        _primary, _backing, info = p._select_speaker_spans(
            result, VOC, VOC_SR, song_path='fake.wav')
        assert info['preserved_speakers'] == ['SPEAKER_01']

    def test_brighter_cluster_preserved(self, monkeypatch):
        """A female co-vocal over a male lead: 4x BRIGHTER, not darker. The
        one-directional form never fired here."""
        p, result = duet_pipeline(monkeypatch, [750.0, 3000.0])
        _primary, _backing, info = p._select_speaker_spans(
            result, VOC, VOC_SR, song_path='fake.wav')
        assert info['preserved_speakers'] == ['SPEAKER_01']
        assert info['span_reasons'] == [[30.0, 60.0, 'voice_category',
                                         'SPEAKER_01']]

    def test_calibration_duet_ratio_does_not_newly_fire(self, monkeypatch):
        """Beauty and the Beast: female 2607 Hz vs male primary 2376-2532 Hz.
        Ratio <= 1.10, so neither direction fires and the embedding pass stays
        the discriminator there — the symmetry must not change that."""
        p, result = duet_pipeline(monkeypatch, [2376.0, 2607.0])
        out = p._select_speaker_spans(result, VOC, VOC_SR, song_path='fake.wav')
        # Nothing preserved -> the clean co-vocal merges into the lead and the
        # multi-speaker path declines, exactly as before the change.
        assert out is None


class RecordingMM:
    """Identity inference that records which profile each call used."""

    def __init__(self):
        self.calls = []

    def infer(self, audio, profile_id, *args, **kwargs):
        self.calls.append((profile_id, float(np.abs(audio).sum())))
        return np.asarray(audio, dtype=np.float32)


def patch_register(monkeypatch, by_group):
    """Fake the DIAGNOSTIC measurement: ('male'|'female'|None, median_f0).

    Routing does not read this — the register comes from the operator's slots
    by elimination. Faked only so the tests do not run pyin over silence.
    """
    def fake(audio, sr, split_hz=None, margin_hz=None, max_s=20.0):
        return by_group
    monkeypatch.setattr(vr, 'classify_register', fake)


class TestPlanRegisterRouting:
    def _plan(self, monkeypatch, span_reasons, profiles, lead='lead-id',
              measured=(None, 120.0)):
        patch_register(monkeypatch, measured)
        p = make_pipeline()
        voc = np.ones(60 * VOC_SR, dtype=np.float32) * 0.1
        return p._plan_register_routing(voc, VOC_SR, span_reasons, profiles,
                                        target_profile_id=lead)

    def test_co_vocal_routes_to_the_non_lead_slot(self, monkeypatch):
        """The lead holds the female slot, so the male slot is the co-vocal's."""
        routed, kept, ledger = self._plan(
            monkeypatch, [[30.0, 60.0, 'voice_category', 'SPEAKER_01']],
            {'female': 'lead-id', 'male': 'male-id'})
        assert routed == [('male-id', [(30.0, 60.0)])]
        assert kept == []
        assert ledger[0]['action'] == 'convert'
        assert ledger[0]['register'] == 'male'
        assert ledger[0]['register_source'] == 'operator_elimination'
        assert ledger[0]['median_f0_hz'] == 120.0
        assert ledger[0]['spans_s'] == 30.0

    def test_works_in_both_directions(self, monkeypatch):
        """A male lead with a female co-vocal, the mirror of the case above."""
        routed, _kept, ledger = self._plan(
            monkeypatch, [[30.0, 60.0, 'voice_category', 'SPEAKER_01']],
            {'female': 'fem-id', 'male': 'lead-id'})
        assert routed == [('fem-id', [(30.0, 60.0)])]
        assert ledger[0]['register'] == 'female'

    def test_single_slot_needs_no_lead_slot(self, monkeypatch):
        """'or just one or the other' — one model chosen, nothing else."""
        routed, _kept, _l = self._plan(
            monkeypatch, [[30.0, 60.0, 'embedding', '_embedding']],
            {'male': 'male-id'})
        assert routed == [('male-id', [(30.0, 60.0)])]

    def test_only_the_leads_own_model_keeps_original(self, monkeypatch):
        routed, kept, ledger = self._plan(
            monkeypatch, [[30.0, 60.0, 'voice_category', 'SPEAKER_01']],
            {'female': 'lead-id'})
        assert routed == []
        assert kept == [(30.0, 60.0)]
        assert ledger[0]['action'] == 'keep'
        assert ledger[0]['register'] is None

    def test_measured_register_is_recorded_but_never_routes(self, monkeypatch):
        """The disproved f0 verdict rides in the ledger and decides nothing."""
        routed, kept, ledger = self._plan(
            monkeypatch, [[30.0, 60.0, 'voice_category', 'SPEAKER_01']],
            {'female': 'lead-id'}, measured=('male', 120.0))
        assert ledger[0]['measured_register'] == 'male'
        assert routed == [] and kept == [(30.0, 60.0)], \
            "a male measurement must not route when the operator chose nothing"

    def test_harmony_stack_is_never_measured_or_routed(self, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("low_voiced spans must not be classified")
        monkeypatch.setattr(vr, 'classify_register', boom)
        p = make_pipeline()
        voc = np.ones(60 * VOC_SR, dtype=np.float32) * 0.1
        routed, kept, ledger = p._plan_register_routing(
            voc, VOC_SR, [[30.0, 60.0, 'low_voiced', 'low_voiced']],
            {'male': 'male-id', 'female': 'lead-id'},
            target_profile_id='lead-id')
        assert routed == [] and kept == [(30.0, 60.0)] and ledger == []

    def test_operator_preserve_is_never_routed(self, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("operator preserve must not be classified")
        monkeypatch.setattr(vr, 'classify_register', boom)
        p = make_pipeline()
        voc = np.ones(60 * VOC_SR, dtype=np.float32) * 0.1
        routed, kept, _ = p._plan_register_routing(
            voc, VOC_SR, [[30.0, 60.0, 'operator', 'SPEAKER_01']],
            {'male': 'male-id'}, target_profile_id='lead-id')
        assert routed == [] and kept == [(30.0, 60.0)]

    def test_split_and_margin_come_from_config(self, monkeypatch):
        """The disproved gate stays wired as opt-in diagnostics."""
        seen = {}

        def fake(audio, sr, split_hz=None, margin_hz=None, max_s=20.0):
            seen.update(split=split_hz, margin=margin_hz)
            return ('male', 120.0)

        monkeypatch.setattr(vr, 'classify_register', fake)
        p = make_pipeline(register_split_f0_hz=200.0, register_margin_hz=40.0)
        voc = np.ones(60 * VOC_SR, dtype=np.float32) * 0.1
        p._plan_register_routing(
            voc, VOC_SR, [[30.0, 60.0, 'voice_category', 'S1']],
            {'male': 'male-id'}, target_profile_id='lead-id')
        assert seen == {'split': 200.0, 'margin': 40.0}

    def test_split_defaults_to_disabled(self, monkeypatch):
        seen = {}

        def fake(audio, sr, split_hz=None, margin_hz=None, max_s=20.0):
            seen.update(split=split_hz)
            return (None, 120.0)

        monkeypatch.setattr(vr, 'classify_register', fake)
        p = make_pipeline()
        voc = np.ones(60 * VOC_SR, dtype=np.float32) * 0.1
        p._plan_register_routing(
            voc, VOC_SR, [[30.0, 60.0, 'voice_category', 'S1']],
            {'male': 'male-id'}, target_profile_id='lead-id')
        assert seen == {'split': None}


class TestRoutingInConvertMultiSpeaker:
    def _run(self, monkeypatch, profiles, register=(None, 120.0)):
        patch_voiced(monkeypatch, 0.9)
        patch_register(monkeypatch, register)
        p = make_pipeline(multi_speaker_min_coverage=0.0,
                          fork_hq_hum_suppress=False,
                          fork_hq_subharmonic_suppress=False,
                          fork_hq_voice_polish=False)
        fake_centroids(monkeypatch, [3000.0, 750.0])
        monkeypatch.setattr(p, '_load_song_mono',
                            lambda path, sr: np.ones(60 * sr, np.float32) * 0.1)
        monkeypatch.setattr(
            p, '_get_diarizer',
            lambda: FakeDiarizer(make_result([(0.0, 30.0, 'SPEAKER_00'),
                                              (30.0, 60.0, 'SPEAKER_01')])))
        sr = 4000
        voc = (0.2 * np.sin(np.arange(60 * sr) * 0.05)).astype(np.float32)
        mm = RecordingMM()
        out = p._convert_multi_speaker(voc, sr, 'lead-id', mm, 0.0,
                                       song_path='fake.wav',
                                       register_profiles=profiles)
        return p, mm, out

    def test_routed_part_uses_the_chosen_model(self, monkeypatch):
        _p, mm, out = self._run(monkeypatch, {'male': 'male-id'})
        assert out is not None
        _combined, info = out
        assert sorted(pid for pid, _e in mm.calls) == ['lead-id', 'male-id']
        entry, = info['register_routing']
        assert (entry['action'], entry['profile_id']) == ('convert', 'male-id')

    def test_routed_spans_leave_the_kept_backing(self, monkeypatch):
        """Otherwise that voice plays twice — once original, once converted."""
        _p, _mm, out = self._run(monkeypatch, {'male': 'male-id'})
        _combined, info = out
        backing = info['_backing_audio']
        assert float(np.abs(backing).max()) == pytest.approx(0.0, abs=1e-6)

    def test_no_model_leaves_the_part_original(self, monkeypatch):
        """Only the lead's own model is named, so the co-vocal stays original —
        the user's 'leaving the original voice of whatever voice a model has
        not been chosen for'."""
        _p, mm, out = self._run(monkeypatch, {'female': 'lead-id'})
        _combined, info = out
        assert [pid for pid, _e in mm.calls] == ['lead-id']
        assert float(np.abs(info['_backing_audio']).max()) > 0.0
        assert info['register_routing'][0]['action'] == 'keep'

    def test_female_covocal_over_a_male_lead(self, monkeypatch):
        """Symmetry at the routing level, not just the detector level."""
        _p, mm, out = self._run(monkeypatch,
                                {'male': 'lead-id', 'female': 'fem-id'})
        _combined, info = out
        assert sorted(pid for pid, _e in mm.calls) == ['fem-id', 'lead-id']
        assert info['register_routing'][0]['register'] == 'female'

    def test_no_register_profiles_is_todays_behaviour(self, monkeypatch):
        _p, mm, out = self._run(monkeypatch, None)
        _combined, info = out
        assert [pid for pid, _e in mm.calls] == ['lead-id']
        assert 'register_routing' not in info
        assert float(np.abs(info['_backing_audio']).max()) > 0.0


class TestPerTrackOutputCorrection:
    def test_chain_runs_once_per_converted_track(self, monkeypatch):
        """Two converted tracks -> two chain runs, each on its OWN source.
        One f0 contour cannot describe two singers."""
        keys = []

        def spy(source, converted, sr):
            keys.append(np.asarray(source, dtype=np.float32).copy())
            return converted

        patch_voiced(monkeypatch, 0.9)
        patch_register(monkeypatch, (None, 120.0))
        p = make_pipeline(multi_speaker_min_coverage=0.0)
        monkeypatch.setattr(p, '_correct_fork_output', spy)
        fake_centroids(monkeypatch, [3000.0, 750.0])
        monkeypatch.setattr(p, '_load_song_mono',
                            lambda path, sr: np.ones(60 * sr, np.float32) * 0.1)
        monkeypatch.setattr(
            p, '_get_diarizer',
            lambda: FakeDiarizer(make_result([(0.0, 30.0, 'SPEAKER_00'),
                                              (30.0, 60.0, 'SPEAKER_01')])))
        sr = 4000
        voc = (0.2 * np.sin(np.arange(60 * sr) * 0.05)).astype(np.float32)
        out = p._convert_multi_speaker(voc, sr, 'lead-id', RecordingMM(), 0.0,
                                       song_path='fake.wav',
                                       register_profiles={'male': 'male-id'})
        _combined, info = out
        assert len(keys) == 2, "one chain run per converted track"
        # Each key is that track alone: the lead's key is silent where the
        # co-vocal sings, and vice versa.
        lead_key, routed_key = keys
        half = 30 * sr
        assert float(np.abs(lead_key[half:]).max()) == pytest.approx(0.0,
                                                                     abs=1e-6)
        assert float(np.abs(routed_key[:half]).max()) == pytest.approx(0.0,
                                                                       abs=1e-6)
        assert info['_output_corrected'] is True

    # Every stage of the chain, in order. `make_pipeline` defaults these OFF so
    # unit tests never run torchcrepe on the CPU, so a test that wants the
    # chain has to ask for it by name - which is also what keeps this list
    # honest when a stage is added.
    CHAIN = (
        ('fork_hq_hum_suppress',
         'auto_voice.inference.fork_hum_suppression', 'suppress_hum'),
        ('fork_hq_subharmonic_suppress',
         'auto_voice.inference.fork_subharmonics', 'suppress_subharmonics'),
        ('fork_hq_voice_polish',
         'auto_voice.inference.fork_voice_polish', 'polish_voice'),
        ('fork_hq_hf_detonalize',
         'auto_voice.inference.fork_voice_polish', 'detonalize_hf'),
    )

    def _run(self, pipeline, seen):
        """Patch every stage to record itself, then run the chain."""
        import importlib
        src = np.zeros(1000, dtype=np.float32)
        conv = np.ones(1000, dtype=np.float32)
        with pytest.MonkeyPatch.context() as mp:
            for _flag, mod, fn in self.CHAIN:
                m = importlib.import_module(mod)
                if fn == 'detonalize_hf':      # takes (converted, sr), no source
                    mp.setattr(m, fn,
                               lambda c, sr, _f=fn: (seen.append(_f) or c))
                else:
                    mp.setattr(m, fn,
                               lambda s, c, sr, _f=fn: (seen.append(_f) or c))
            pipeline._correct_fork_output(src, conv, 16000)

    def test_correct_fork_output_honours_every_flag(self):
        """The flags that guard the chain that was silently lost once."""
        on = {flag: True for flag, _m, _f in self.CHAIN}
        seen = []
        self._run(make_pipeline(**on), seen)
        assert seen == ['suppress_hum', 'suppress_subharmonics',
                        'polish_voice', 'detonalize_hf'], "order is load-bearing"

        for flag, _mod, fn in self.CHAIN:
            seen.clear()
            self._run(make_pipeline(**{**on, flag: False}), seen)
            assert fn not in seen, f"{flag}=False must skip {fn}"
            assert seen == [f for _f, _m, f in self.CHAIN if f != fn], (
                f"disabling {flag} must not disturb the other stages")


class TestRegisterProfileValidation:
    """No silent fallback: an unservable model must not become 'the lead sings
    that part too', which is what _convert_multi_speaker's broad except would
    have turned it into."""

    def test_unservable_profile_raises(self, monkeypatch, tmp_path):
        from auto_voice.inference import svc_fork_bridge
        monkeypatch.setattr(svc_fork_bridge, 'is_available',
                            lambda pid, data_dir: pid == 'good-id')
        p = make_pipeline()
        song = tmp_path / 'song.wav'
        song.write_bytes(b'')
        with pytest.raises(ConversionError) as exc:
            p._convert_song_fork_hq(
                str(song), 'lead-id', 1.0, 0.9, False, 'studio', 0.0,
                register_profiles={'male': 'missing-id'})
        assert 'missing-id' in str(exc.value)

    def test_servable_profiles_pass_validation(self, monkeypatch, tmp_path):
        from auto_voice.inference import svc_fork_bridge
        monkeypatch.setattr(svc_fork_bridge, 'is_available',
                            lambda pid, data_dir: True)
        p = make_pipeline()
        song = tmp_path / 'song.wav'
        song.write_bytes(b'')
        # Validation passes; the empty file then fails at load, which is a
        # different error and proves we got past the register check.
        with pytest.raises(ConversionError) as exc:
            p._convert_song_fork_hq(
                str(song), 'lead-id', 1.0, 0.9, False, 'studio', 0.0,
                register_profiles={'male': 'good-id'})
        assert 'good-id' not in str(exc.value)

    def test_ambiguous_both_slots_is_refused(self, monkeypatch, tmp_path):
        """Both slots filled and neither is the lead's: nothing can say which
        model the co-vocal gets, and the disproved f0 gate cannot break the
        tie. Refuse before any work rather than guess."""
        from auto_voice.inference import svc_fork_bridge
        monkeypatch.setattr(svc_fork_bridge, 'is_available',
                            lambda pid, data_dir: True)
        p = make_pipeline()
        song = tmp_path / 'song.wav'
        song.write_bytes(b'')
        with pytest.raises(ConversionError) as exc:
            p._convert_song_fork_hq(
                str(song), 'lead-id', 1.0, 0.9, False, 'studio', 0.0,
                register_profiles={'male': 'a-id', 'female': 'b-id'})
        assert 'lead-id' in str(exc.value)

    def test_lead_in_one_slot_is_not_ambiguous(self, monkeypatch, tmp_path):
        from auto_voice.inference import svc_fork_bridge
        monkeypatch.setattr(svc_fork_bridge, 'is_available',
                            lambda pid, data_dir: True)
        p = make_pipeline()
        song = tmp_path / 'song.wav'
        song.write_bytes(b'')
        with pytest.raises(ConversionError) as exc:
            p._convert_song_fork_hq(
                str(song), 'lead-id', 1.0, 0.9, False, 'studio', 0.0,
                register_profiles={'female': 'lead-id', 'male': 'male-id'})
        assert 'Failed to load audio' in str(exc.value), \
            "validation must pass and fail later, on the empty file"

    def test_declined_multi_speaker_path_warns(self, monkeypatch, caplog,
                                               tmp_path):
        """The operator chose a model and the per-speaker path never ran, so
        the render ignores it. That must not be silent — it is the case both
        calibration duets actually hit."""
        import logging

        import soundfile as sf
        from auto_voice.inference import svc_fork_bridge
        monkeypatch.setattr(svc_fork_bridge, 'is_available',
                            lambda pid, data_dir: True)
        p = make_pipeline()
        sr = 16000
        song = tmp_path / 'song.wav'
        sf.write(str(song), np.zeros((sr, 2), dtype=np.float32), sr)
        monkeypatch.setattr(p, '_output_sample_rate', sr, raising=False)
        monkeypatch.setattr(
            p, '_get_separator',
            lambda: type('S', (), {'separate': staticmethod(
                lambda a, s, mono=False: {
                    'vocals': np.zeros((2, a.shape[-1]), np.float32),
                    'instrumental': np.zeros((2, a.shape[-1]), np.float32)})})())
        monkeypatch.setattr(p, '_get_model_manager', lambda: RecordingMM())
        # The per-speaker path declines, exactly as it does on a dense mix.
        monkeypatch.setattr(p, '_convert_multi_speaker',
                            lambda *a, **k: None)
        with caplog.at_level(logging.WARNING):
            try:
                p._convert_song_fork_hq(
                    str(song), 'lead-id', 1.0, 0.9, False, 'studio', 0.0,
                    enable_multi_speaker=True,
                    register_profiles={'male': 'male-id'})
            except Exception:
                pass  # downstream stages are not what this pins
        assert 'chosen co-vocal model is unused' in caplog.text

    def test_none_assignment_is_not_validated(self, monkeypatch, tmp_path):
        from auto_voice.inference import svc_fork_bridge
        monkeypatch.setattr(svc_fork_bridge, 'is_available',
                            lambda pid, data_dir: False)
        p = make_pipeline()
        song = tmp_path / 'song.wav'
        song.write_bytes(b'')
        with pytest.raises(ConversionError) as exc:
            p._convert_song_fork_hq(
                str(song), 'lead-id', 1.0, 0.9, False, 'studio', 0.0,
                register_profiles={'male': None, 'female': None})
        assert 'register_profiles' not in str(exc.value)
