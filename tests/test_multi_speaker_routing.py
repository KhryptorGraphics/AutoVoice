"""Routing gates for the per-speaker conversion path (no models needed)."""
import numpy as np
import pytest

import auto_voice.inference.singing_conversion_pipeline as scp
from auto_voice.audio.speaker_diarization import DiarizationResult, SpeakerSegment
from auto_voice.inference.singing_conversion_pipeline import (
    SingingConversionPipeline,
    _voiced_fraction,
)

VOC_SR = 1000
VOC = np.zeros(120 * VOC_SR, dtype=np.float32)  # silent canvas spans index into


def make_pipeline(**config):
    config.setdefault('enable_multi_speaker_conversion', True)
    return SingingConversionPipeline(device='cpu', config=config)


def make_result(segments):
    segs = [SpeakerSegment(start=s, end=e, speaker_id=spk) for s, e, spk in segments]
    return DiarizationResult(
        segments=segs,
        num_speakers=len({spk for _, _, spk in segments}),
        audio_duration=max((e for _, e, _ in segments), default=0.0),
    )


def patch_voiced(monkeypatch, value=0.2):
    """Replace the pyin measurement with a constant (or callable) fake."""
    fake = value if callable(value) else (lambda audio, sr, max_s=20.0: value)
    monkeypatch.setattr(scp, '_voiced_fraction', fake)


class IdentityMM:
    def infer(self, audio, *args, **kwargs):
        return np.asarray(audio, dtype=np.float32)


class FakeDiarizer:
    def __init__(self, result):
        self._result = result

    def diarize(self, path, **kwargs):
        return self._result


class TestVoicedFraction:
    def test_sine_is_voiced(self):
        sr = 8000
        t = np.arange(3 * sr) / sr
        sine = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        assert _voiced_fraction(sine, sr) > 0.8

    def test_noise_is_unvoiced(self):
        sr = 8000
        noise = (np.random.default_rng(0).standard_normal(3 * sr) * 0.1).astype(np.float32)
        assert _voiced_fraction(noise, sr) < 0.2

    def test_empty_audio(self):
        assert _voiced_fraction(np.zeros(0, dtype=np.float32), 8000) == 0.0


class TestSelectSpeakerSpans:
    def test_single_speaker_returns_none(self, monkeypatch):
        patch_voiced(monkeypatch)
        p = make_pipeline()
        result = make_result([(0.0, 10.0, 'SPEAKER_00')])
        assert p._select_speaker_spans(result, VOC, VOC_SR) is None

    def test_short_backing_reassigned_to_lead(self, monkeypatch):
        patch_voiced(monkeypatch)
        p = make_pipeline()  # default min segment 2.0s
        result = make_result([
            (0.0, 10.0, 'SPEAKER_00'),   # lead
            (10.0, 11.5, 'SPEAKER_01'),  # 1.5s blip -> lead
            (11.5, 30.0, 'SPEAKER_00'),
            (30.0, 35.0, 'SPEAKER_01'),  # 5s genuine backing
            (35.0, 60.0, 'SPEAKER_00'),
        ])
        primary_spans, backing_spans, info = p._select_speaker_spans(result, VOC, VOC_SR)
        assert (10.0, 11.5) in primary_spans
        assert backing_spans == [(30.0, 35.0)]
        assert info['reassigned_blips'] == 1
        assert info['primary_speaker'] == 'SPEAKER_00'
        assert info['backing_s'] == 5.0
        assert info['roles'] == {'SPEAKER_01': 'backing'}

    def test_only_blip_backing_returns_none(self, monkeypatch):
        patch_voiced(monkeypatch)
        p = make_pipeline()
        result = make_result([
            (0.0, 60.0, 'SPEAKER_00'),
            (10.0, 11.0, 'SPEAKER_01'),  # all backing < 2s -> no backing left
        ])
        assert p._select_speaker_spans(result, VOC, VOC_SR) is None

    def test_min_segment_configurable(self, monkeypatch):
        patch_voiced(monkeypatch)
        p = make_pipeline(multi_speaker_min_segment_s=0.5)
        result = make_result([
            (0.0, 10.0, 'SPEAKER_00'),
            (10.0, 11.0, 'SPEAKER_01'),  # 1s >= 0.5s -> kept as backing
            (11.0, 60.0, 'SPEAKER_00'),
        ])
        _, backing_spans, _ = p._select_speaker_spans(result, VOC, VOC_SR)
        assert backing_spans == [(10.0, 11.0)]

    def test_backing_overlap_with_lead_is_clipped(self, monkeypatch):
        patch_voiced(monkeypatch)
        p = make_pipeline()
        result = make_result([
            (0.0, 10.0, 'SPEAKER_00'),   # lead (20s total)
            (20.0, 30.0, 'SPEAKER_00'),
            (5.0, 15.0, 'SPEAKER_01'),   # overlaps lead 5s -> clipped to (10, 15)
        ])
        primary_spans, backing_spans, info = p._select_speaker_spans(result, VOC, VOC_SR)
        assert backing_spans == [(10.0, 15.0)]
        assert info['backing_s'] == 5.0

    def test_backing_inside_lead_is_removed(self, monkeypatch):
        patch_voiced(monkeypatch)
        p = make_pipeline()
        result = make_result([
            (0.0, 60.0, 'SPEAKER_00'),
            (10.0, 20.0, 'SPEAKER_01'),  # fully lead-covered -> nothing left
        ])
        assert p._select_speaker_spans(result, VOC, VOC_SR) is None

    def test_clip_sliver_reassigned_to_lead(self, monkeypatch):
        patch_voiced(monkeypatch)
        p = make_pipeline()
        result = make_result([
            (10.0, 60.0, 'SPEAKER_00'),
            (9.0, 12.5, 'SPEAKER_01'),   # clip leaves 1s sliver (9, 10) -> lead
            (70.0, 75.0, 'SPEAKER_01'),  # genuine backing keeps the path alive
        ])
        primary_spans, backing_spans, info = p._select_speaker_spans(result, VOC, VOC_SR)
        assert backing_spans == [(70.0, 75.0)]
        assert (9.0, 10.0) in primary_spans
        assert info['reassigned_blips'] == 1

    def test_lead_segments_never_dropped(self, monkeypatch):
        patch_voiced(monkeypatch)
        p = make_pipeline()
        result = make_result([
            (0.0, 0.5, 'SPEAKER_00'),    # short lead segment stays lead
            (1.0, 30.0, 'SPEAKER_00'),
            (30.0, 35.0, 'SPEAKER_01'),
            (35.0, 60.0, 'SPEAKER_00'),
        ])
        primary_spans, _, _ = p._select_speaker_spans(result, VOC, VOC_SR)
        assert (0.0, 0.5) in primary_spans


class TestConvertibilityRoleGate:
    """Clean melody merges into the lead; textural clusters stay backing."""

    def test_high_voiced_cluster_merged_low_kept(self, monkeypatch):
        # Fake keyed on active-audio length: 10s cluster -> 0.9, 5s -> 0.2.
        patch_voiced(monkeypatch,
                     lambda audio, sr, max_s=20.0: 0.9 if len(audio) >= 8 * sr else 0.2)
        p = make_pipeline()
        result = make_result([
            (0.0, 30.0, 'SPEAKER_00'),   # lead
            (30.0, 40.0, 'SPEAKER_01'),  # 10s clean melody -> merge into lead
            (50.0, 55.0, 'SPEAKER_02'),  # 5s texture -> keep as backing
        ])
        primary_spans, backing_spans, info = p._select_speaker_spans(result, VOC, VOC_SR)
        assert (30.0, 40.0) in primary_spans
        assert backing_spans == [(50.0, 55.0)]
        assert info['roles'] == {'SPEAKER_01': 'lead_merge', 'SPEAKER_02': 'backing'}
        assert info['voiced']['SPEAKER_01'] == 0.9
        assert info['voiced']['SPEAKER_02'] == 0.2

    def test_all_merged_falls_back_to_single_stem(self, monkeypatch):
        patch_voiced(monkeypatch, 0.9)  # everything is clean melody
        p = make_pipeline()
        result = make_result([
            (0.0, 30.0, 'SPEAKER_00'),
            (30.0, 40.0, 'SPEAKER_01'),
        ])
        assert p._select_speaker_spans(result, VOC, VOC_SR) is None

    def test_merge_threshold_configurable(self, monkeypatch):
        patch_voiced(monkeypatch, 0.9)
        p = make_pipeline(multi_speaker_merge_voiced_min=0.95)
        result = make_result([
            (0.0, 30.0, 'SPEAKER_00'),
            (30.0, 40.0, 'SPEAKER_01'),  # 0.9 < 0.95 -> stays backing
        ])
        _, backing_spans, info = p._select_speaker_spans(result, VOC, VOC_SR)
        assert backing_spans == [(30.0, 40.0)]
        assert info['roles'] == {'SPEAKER_01': 'backing'}

    def test_tiny_cluster_skips_measurement(self, monkeypatch):
        def boom(audio, sr, max_s=20.0):
            raise AssertionError("must not measure <1.5s clusters")
        patch_voiced(monkeypatch, boom)
        p = make_pipeline()
        result = make_result([
            (0.0, 30.0, 'SPEAKER_00'),
            (30.0, 31.0, 'SPEAKER_01'),  # 1s active -> skipped, blip policy
        ])
        assert p._select_speaker_spans(result, VOC, VOC_SR) is None  # blip -> no backing


class TestConvertMultiSpeaker:
    def test_coverage_gate_falls_back(self, monkeypatch):
        # Backing survives selection (>=2s, low-voiced) but spans cover only
        # 6s of a 60s vocal with energy everywhere -> ~10% << 0.9 -> fallback.
        patch_voiced(monkeypatch)
        sr = 8000
        voc = np.random.default_rng(0).standard_normal(60 * sr).astype(np.float32) * 0.1
        result = make_result([
            (0.0, 3.0, 'SPEAKER_00'),
            (3.0, 6.0, 'SPEAKER_01'),
        ])

        p = make_pipeline()
        p._diarizer = FakeDiarizer(result)
        assert p._convert_multi_speaker(voc, sr, 'pid', IdentityMM(), 0.0) is None

        # Same fixture with the gate disabled converts -> proves it was the
        # coverage gate (not span selection) that fell back above.
        p2 = make_pipeline(multi_speaker_min_coverage=0.0)
        p2._diarizer = FakeDiarizer(result)
        assert p2._convert_multi_speaker(voc, sr, 'pid', IdentityMM(), 0.0) is not None

    def test_full_coverage_converts_and_reports(self, monkeypatch):
        patch_voiced(monkeypatch)
        p = make_pipeline()
        sr = 8000
        # Energy only inside the spans -> coverage ~1 (minus fades).
        voc = np.zeros(10 * sr, dtype=np.float32)
        voc[0:6 * sr] = 0.5
        voc[7 * sr:9 * sr] = 0.3
        p._diarizer = FakeDiarizer(make_result([
            (0.0, 6.0, 'SPEAKER_00'),
            (7.0, 9.0, 'SPEAKER_01'),
        ]))
        out = p._convert_multi_speaker(voc, sr, 'pid', IdentityMM(), 0.0)
        assert out is not None
        combined, info = out
        assert combined.dtype == np.float32
        assert len(combined) == len(voc)
        assert info['num_speakers'] == 2
        assert info['primary_speaker'] == 'SPEAKER_00'
        assert info['backing_s'] == 2.0
        assert info['coverage'] > 0.9
        assert info['roles'] == {'SPEAKER_01': 'backing'}
        # Identity conversion -> combined reconstructs the vocal (minus fades).
        assert np.abs(combined - voc).mean() < 0.01

    def test_diarizer_failure_falls_back(self):
        p = make_pipeline()

        class BoomDiarizer:
            def diarize(self, path, **kwargs):
                raise RuntimeError("boom")

        p._diarizer = BoomDiarizer()
        voc = np.ones(8000, dtype=np.float32)
        assert p._convert_multi_speaker(voc, 8000, 'pid', IdentityMM(), 0.0) is None


class TestMultiSpeakerEnabledOverride:
    """The request/job-settings override wins over config-then-env resolution."""

    def test_override_false_beats_config_true(self):
        p = make_pipeline(enable_multi_speaker_conversion=True)
        assert p._multi_speaker_enabled(override=False) is False

    def test_override_true_beats_config_false(self):
        p = make_pipeline(enable_multi_speaker_conversion=False)
        assert p._multi_speaker_enabled(override=True) is True

    def test_override_none_falls_back_to_config(self):
        assert make_pipeline(enable_multi_speaker_conversion=True)._multi_speaker_enabled(
            override=None) is True
        assert make_pipeline(enable_multi_speaker_conversion=False)._multi_speaker_enabled(
            override=None) is False

    def test_override_none_falls_back_to_env_when_config_unset(self, monkeypatch):
        # config unset -> env decides (existing ops behaviour, unchanged);
        # an explicit override still wins over the env.
        p = SingingConversionPipeline(device='cpu', config={})
        monkeypatch.setenv('ENABLE_MULTI_SPEAKER_CONVERSION', '1')
        assert p._multi_speaker_enabled(override=None) is True
        assert p._multi_speaker_enabled(override=False) is False


def test_convert_multi_speaker_unaffected_by_override(monkeypatch):
    """_convert_multi_speaker never sees the override; its behaviour is the
    single-stem-fallback contract regardless of how the path was enabled."""
    patch_voiced(monkeypatch)
    p = make_pipeline(enable_multi_speaker_conversion=False)  # override is upstream
    result = make_result([(0.0, 10.0, 'SPEAKER_00')])  # 1 speaker -> None
    p._diarizer = FakeDiarizer(result)
    assert p._convert_multi_speaker(VOC, VOC_SR, 'pid', IdentityMM(), 0.0) is None


def test_get_diarizer_is_cached():
    p = make_pipeline()
    d1 = p._get_diarizer()
    assert p._get_diarizer() is d1


# ---------------------------------------------------------------------------
# Karaoke-model separator routing (Mel-RoFormer lead/backing via uvr bridge)
# ---------------------------------------------------------------------------

from auto_voice.inference.singing_conversion_pipeline import (  # noqa: E402
    _extract_line_audio,
    _group_notes_into_lines,
)


def _note(start, end, midi, amp=1.0):
    return {'start': start, 'end': end, 'pitch_midi': float(midi), 'amplitude': amp}


class TestKaraokeSeparatorRouting:
    def _patch_bridge(self, monkeypatch, lead, backing, available=True):
        import auto_voice.inference.separation_bridge as bridge
        calls = {'n': 0}

        def fake_sep(voc, sr, data_dir='data', model=None):
            calls['n'] += 1
            return lead, backing
        monkeypatch.setattr(bridge, 'is_available', lambda: available)
        monkeypatch.setattr(bridge, 'separate_lead_backing', fake_sep)
        return calls

    def test_karaoke_hybrid_path_used(self, monkeypatch):
        sr = 8000
        voc = np.zeros(10 * sr, dtype=np.float32)
        voc[:8 * sr] = 0.5
        lead = voc * 0.8       # de-doubled lead: spans run on this
        simul = voc * 0.2      # simultaneous doubles stem
        calls = self._patch_bridge(monkeypatch, lead, simul)
        patch_voiced(monkeypatch, 0.2)
        p = make_pipeline(multi_speaker_separator='karaoke_model')
        p._diarizer = FakeDiarizer(make_result([
            (0.0, 5.0, 'SPEAKER_00'), (5.0, 8.0, 'SPEAKER_01'),
        ]))
        out = p._convert_multi_speaker(voc, sr, 'pid', IdentityMM(), 0.0)
        assert out is not None
        combined, info = out
        assert calls['n'] == 1
        assert info['separator'] == 'karaoke_model+diarization'
        assert info['backing_mode'] == 'kept'
        assert info['simul_backing_s'] > 0
        # identity conversion: span tracks (from lead) + doubles == voc
        assert np.abs(combined - voc[:len(combined)]).mean() < 0.01

    def test_karaoke_solo_lead_with_doubles_converts_whole_lead(self, monkeypatch):
        sr = 8000
        voc = np.zeros(10 * sr, dtype=np.float32)
        voc[:8 * sr] = 0.5
        self._patch_bridge(monkeypatch, voc * 0.8, voc * 0.2)
        patch_voiced(monkeypatch, 0.2)
        p = make_pipeline(multi_speaker_separator='karaoke_model')
        # De-doubled lead diarizes to a single voice -> whole-lead branch
        p._diarizer = FakeDiarizer(make_result([(0.0, 8.0, 'SPEAKER_00')]))
        out = p._convert_multi_speaker(voc, sr, 'pid', IdentityMM(), 0.0)
        assert out is not None
        combined, info = out
        assert info['num_speakers'] == 1
        assert info['separator'] == 'karaoke_model+diarization'
        assert np.abs(combined - voc[:len(combined)]).mean() < 0.01

    def test_negligible_backing_falls_back_to_spans(self, monkeypatch):
        sr = 8000
        voc = np.ones(10 * sr, dtype=np.float32) * 0.5
        self._patch_bridge(monkeypatch, voc.copy(), np.zeros_like(voc))
        patch_voiced(monkeypatch, 0.2)
        p = make_pipeline(multi_speaker_separator='karaoke_model')
        p._diarizer = FakeDiarizer(make_result([(0.0, 10.0, 'SPEAKER_00')]))
        # span path engaged -> single speaker -> None (single-stem)
        assert p._convert_multi_speaker(voc, sr, 'pid', IdentityMM(), 0.0) is None

    def test_leaky_backing_falls_back_to_spans(self, monkeypatch):
        sr = 8000
        voc = np.ones(10 * sr, dtype=np.float32) * 0.5
        self._patch_bridge(monkeypatch, voc * 0.5, voc * 0.5)  # backing = clean voice
        patch_voiced(monkeypatch, 0.9)  # reads as clean lead -> unreliable split
        p = make_pipeline(multi_speaker_separator='karaoke_model')
        p._diarizer = FakeDiarizer(make_result([(0.0, 10.0, 'SPEAKER_00')]))
        assert p._convert_multi_speaker(voc, sr, 'pid', IdentityMM(), 0.0) is None

    def test_leak_guard_knob_accepts_voiced_backing(self, monkeypatch):
        # Raising multi_speaker_karaoke_leak_voiced_min above the measured
        # backing voicing accepts the split (solo self-doubles case) instead
        # of falling back to spans.
        sr = 8000
        voc = np.ones(10 * sr, dtype=np.float32) * 0.5
        self._patch_bridge(monkeypatch, voc * 0.5, voc * 0.5)
        patch_voiced(monkeypatch, 0.76)  # the real measured leak case
        p = make_pipeline(multi_speaker_separator='karaoke_model',
                          multi_speaker_karaoke_leak_voiced_min=0.85)
        p._diarizer = FakeDiarizer(make_result([(0.0, 10.0, 'SPEAKER_00')]))
        out = p._convert_multi_speaker(voc, sr, 'pid', IdentityMM(), 0.0)
        assert out is not None
        _, info = out
        assert info['separator'] == 'karaoke_model+diarization'

    def test_bridge_unavailable_falls_back(self, monkeypatch):
        sr = 8000
        voc = np.ones(10 * sr, dtype=np.float32) * 0.5
        self._patch_bridge(monkeypatch, voc, voc, available=False)
        patch_voiced(monkeypatch, 0.2)
        p = make_pipeline(multi_speaker_separator='karaoke_model')
        p._diarizer = FakeDiarizer(make_result([(0.0, 10.0, 'SPEAKER_00')]))
        assert p._convert_multi_speaker(voc, sr, 'pid', IdentityMM(), 0.0) is None

    def test_default_config_never_calls_bridge(self, monkeypatch):
        sr = 8000
        voc = np.zeros(10 * sr, dtype=np.float32)
        voc[:6 * sr] = 0.5
        calls = self._patch_bridge(monkeypatch, voc, voc)
        patch_voiced(monkeypatch, 0.2)
        p = make_pipeline()  # separator defaults to diarization
        p._diarizer = FakeDiarizer(make_result([
            (0.0, 4.0, 'SPEAKER_00'), (4.0, 6.0, 'SPEAKER_01'),
        ]))
        out = p._convert_multi_speaker(voc, sr, 'pid', IdentityMM(), 0.0)
        assert calls['n'] == 0
        assert out is not None and out[1]['separator'] == 'diarization'


# ---------------------------------------------------------------------------
# Harmony-line grouping and comb-mask extraction (backing conversion)
# ---------------------------------------------------------------------------

class TestNoteGrouping:
    def test_simultaneous_notes_go_to_different_lines(self):
        notes = [_note(0, 4, 57), _note(0, 4, 64)]
        lines = _group_notes_into_lines(notes)
        assert len(lines) == 2

    def test_sequential_notes_share_a_line(self):
        notes = [_note(0, 1, 57), _note(1.2, 2, 59), _note(2.2, 3, 60)]
        lines = _group_notes_into_lines(notes)
        assert len(lines) == 1
        assert len(lines[0]) == 3

    def test_harmonic_duplicate_dropped(self):
        # midi 76 (~660Hz) is the 3rd harmonic of midi 57 (220Hz): 660/220=3
        notes = [_note(0, 4, 57, amp=0.9), _note(0, 4, 76, amp=0.4)]
        lines = _group_notes_into_lines(notes)
        assert len(lines) == 1
        assert round(lines[0][0]['pitch_midi']) == 57

    def test_weak_and_short_notes_dropped(self):
        notes = [_note(0, 4, 57, amp=0.9),
                 _note(0, 0.05, 60, amp=0.9),   # too short
                 _note(1, 3, 80, amp=0.05)]     # too quiet
        lines = _group_notes_into_lines(notes)
        assert len(lines) == 1

    def test_max_lines_respected(self):
        notes = [_note(0, 4, 50), _note(0, 4, 55), _note(0, 4, 60), _note(0, 4, 66)]
        lines = _group_notes_into_lines(notes, max_lines=3)
        assert len(lines) == 3

    def test_empty(self):
        assert _group_notes_into_lines([]) == []


class TestLineExtraction:
    def test_comb_mask_recovers_two_voices(self):
        sr = 8000
        t = np.arange(4 * sr) / sr

        def voice(f0, amp):
            return amp * sum((1.0 / k) * np.sin(2 * np.pi * k * f0 * t)
                             for k in range(1, 5)) / 2.0
        low, high = voice(220.0, 0.5), voice(330.0, 0.5)
        stack = (low + high).astype(np.float32)
        ext_low = _extract_line_audio(stack, sr, [_note(0, 4, 57)])
        ext_high = _extract_line_audio(stack, sr, [_note(0, 4, 64)])
        # each extract correlates with its voice, not the other
        def corr(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        assert corr(ext_low, low.astype(np.float32)) > 0.8
        assert corr(ext_high, high.astype(np.float32)) > 0.8
        assert corr(ext_low, high.astype(np.float32)) < 0.4

    def test_extract_outside_note_span_is_silent(self):
        sr = 8000
        t = np.arange(4 * sr) / sr
        stack = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        ext = _extract_line_audio(stack, sr, [_note(0, 1.0, 57)])
        assert np.abs(ext[2 * sr:]).max() < 0.05


class TestConvertBackingStack:
    def _pipeline_with_notes(self, monkeypatch, notes):
        import auto_voice.inference.separation_bridge as bridge
        monkeypatch.setattr(bridge, 'polyphonic_notes', lambda a, sr: notes)
        return make_pipeline()

    def test_no_notes_keeps_original(self, monkeypatch):
        p = self._pipeline_with_notes(monkeypatch, [])
        backing = np.ones(8000, dtype=np.float32) * 0.3
        out, info = p._convert_backing_stack(backing, 8000, 'pid', IdentityMM())
        assert info['mode'] == 'kept'
        assert np.array_equal(out, backing)

    def test_bridge_failure_keeps_original(self, monkeypatch):
        import auto_voice.inference.separation_bridge as bridge

        def boom(a, sr):
            raise RuntimeError('nope')
        monkeypatch.setattr(bridge, 'polyphonic_notes', boom)
        p = make_pipeline()
        backing = np.ones(8000, dtype=np.float32) * 0.3
        out, info = p._convert_backing_stack(backing, 8000, 'pid', IdentityMM())
        assert info['mode'] == 'kept'

    def test_clean_lines_converted(self, monkeypatch):
        sr = 8000
        t = np.arange(4 * sr) / sr

        def voice(f0, amp):
            return amp * sum((1.0 / k) * np.sin(2 * np.pi * k * f0 * t)
                             for k in range(1, 5)) / 2.0
        stack = (voice(220.0, 0.5) + voice(330.0, 0.4)).astype(np.float32)
        p = self._pipeline_with_notes(
            monkeypatch, [_note(0, 4, 57), _note(0, 4, 64)])
        # Pin the per-line path (a clean synthetic stack measures fully voiced,
        # which would otherwise route to whole-stem conversion).
        p.config['multi_speaker_backing_whole_voiced_min'] = 2.0
        out, info = p._convert_backing_stack(stack, sr, 'pid', IdentityMM())
        assert info['mode'] == 'converted'
        assert info['lines_detected'] == 2 and info['lines_converted'] == 2
        assert info['method'] == 'lines'
        # identity engine + RMS matching: output stays close to the stack
        assert np.sqrt(((out - stack) ** 2).mean()) < 0.2

    def test_near_monophonic_backing_converted_whole(self, monkeypatch):
        # A doubled voice (not a stack) routes to whole-stem conversion —
        # no basic-pitch call, no per-line residual.
        sr = 8000
        t = np.arange(4 * sr) / sr
        stem = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

        import auto_voice.inference.separation_bridge as bridge

        def boom(a, sr_):
            raise AssertionError('polyphonic_notes must not be called')
        monkeypatch.setattr(bridge, 'polyphonic_notes', boom)
        patch_voiced(monkeypatch, 0.8)
        p = make_pipeline()
        out, info = p._convert_backing_stack(stem, sr, 'pid', IdentityMM())
        assert info['mode'] == 'converted'
        assert info['method'] == 'whole_stem'
        assert info['lines_detected'] == 1 and info['lines_converted'] == 1
        # identity engine + level match: output stays close to the stem
        assert np.sqrt(((out - stem) ** 2).mean()) < 0.2

    def test_late_start_clean_lines_converted(self, monkeypatch):
        # Regression: voicing must be measured on span-active audio, not the
        # first 20s of the full-length extract — a backing part that first
        # enters after 0:20 would otherwise always gate to 'kept'.
        sr = 8000
        t = np.arange(4 * sr) / sr

        def voice(f0, amp):
            return amp * sum((1.0 / k) * np.sin(2 * np.pi * k * f0 * t)
                             for k in range(1, 5)) / 2.0
        stack = np.zeros(30 * sr, dtype=np.float32)
        stack[24 * sr:28 * sr] = (voice(220.0, 0.5) + voice(330.0, 0.4)).astype(np.float32)
        p = self._pipeline_with_notes(
            monkeypatch, [_note(24, 28, 57), _note(24, 28, 64)])
        p.config['multi_speaker_backing_whole_voiced_min'] = 2.0
        out, info = p._convert_backing_stack(stack, sr, 'pid', IdentityMM())
        assert info['mode'] == 'converted'
        assert info['lines_detected'] == 2 and info['lines_converted'] == 2

    def test_backing_voiced_min_knob_overrides_merge_gate(self, monkeypatch):
        # A dedicated backing-line gate: with the knob above the (patched)
        # line voicing, lines are kept; lowering it converts them — without
        # touching the cluster merge threshold.
        sr = 8000
        t = np.arange(4 * sr) / sr

        def voice(f0, amp):
            return amp * sum((1.0 / k) * np.sin(2 * np.pi * k * f0 * t)
                             for k in range(1, 5)) / 2.0
        stack = (voice(220.0, 0.5) + voice(330.0, 0.4)).astype(np.float32)
        notes = [_note(0, 4, 57), _note(0, 4, 64)]

        import auto_voice.inference.separation_bridge as bridge
        monkeypatch.setattr(bridge, 'polyphonic_notes', lambda a, sr: notes)
        # Real lines here measure ~1.0 voiced; a 0.95 gate still passes them,
        # so patch the measurement to a mid value and bracket it with the knob.
        import auto_voice.inference.singing_conversion_pipeline as scp
        monkeypatch.setattr(scp, '_voiced_fraction', lambda a, s, max_s=20.0: 0.5)

        strict = make_pipeline(multi_speaker_backing_voiced_min=0.6)
        out, info = strict._convert_backing_stack(stack, sr, 'pid', IdentityMM())
        assert info['mode'] == 'kept'

        loose = make_pipeline(multi_speaker_backing_voiced_min=0.4)
        out, info = loose._convert_backing_stack(stack, sr, 'pid', IdentityMM())
        assert info['mode'] == 'converted'

    def test_backing_gain_knob_scales_output(self, monkeypatch):
        # multi_speaker_backing_gain is the operator taste knob for converted
        # harmony level; identity engine + level match means the knob alone
        # sets the output/input ratio.
        sr = 8000
        t = np.arange(4 * sr) / sr

        def voice(f0, amp):
            return amp * sum((1.0 / k) * np.sin(2 * np.pi * k * f0 * t)
                             for k in range(1, 5)) / 2.0
        stack = (voice(220.0, 0.5) + voice(330.0, 0.4)).astype(np.float32)
        notes = [_note(0, 4, 57), _note(0, 4, 64)]

        import auto_voice.inference.separation_bridge as bridge
        monkeypatch.setattr(bridge, 'polyphonic_notes', lambda a, sr: notes)
        base = make_pipeline()
        boosted = make_pipeline(multi_speaker_backing_gain=1.5)
        out1, info1 = base._convert_backing_stack(stack, sr, 'pid', IdentityMM())
        out2, info2 = boosted._convert_backing_stack(stack, sr, 'pid', IdentityMM())
        assert info1['mode'] == info2['mode'] == 'converted'
        rms1 = float(np.sqrt((out1 ** 2).mean()))
        rms2 = float(np.sqrt((out2 ** 2).mean()))
        assert 1.3 < rms2 / rms1 < 1.7

    def test_unvoiced_lines_kept(self, monkeypatch):
        sr = 8000
        noise = (np.random.default_rng(0).standard_normal(4 * sr) * 0.2).astype(np.float32)
        p = self._pipeline_with_notes(monkeypatch, [_note(0, 4, 57)])
        out, info = p._convert_backing_stack(noise, sr, 'pid', IdentityMM())
        assert info['mode'] == 'kept'
        assert np.array_equal(out, noise)


class TestPreserveSpeakers:
    def _pipeline(self, monkeypatch, segments):
        patch_voiced(monkeypatch, 0.9)  # clean voice: would normally lead_merge
        p = make_pipeline()
        p._diarizer = FakeDiarizer(make_result(segments))
        return p

    def test_preserved_cluster_kept_and_excluded_from_primary(self, monkeypatch):
        sr = 8000
        voc = np.zeros(12 * sr, dtype=np.float32)
        voc[:int(11.5 * sr)] = 0.5
        # Preserved speaker has MORE total time — without preserve it would be
        # primary; with preserve the other cluster must lead. Includes a
        # preserved blip (<2s) that must NOT reassign into the converted lead.
        p = self._pipeline(monkeypatch, [
            (0.0, 6.0, 'SPEAKER_00'),
            (6.0, 10.0, 'SPEAKER_01'),
            (10.2, 11.4, 'SPEAKER_00'),
        ])
        out = p._convert_multi_speaker(voc, sr, 'pid', IdentityMM(), 0.0,
                                       preserve_speakers=['SPEAKER_00'])
        assert out is not None
        combined, info = out
        assert info['primary_speaker'] == 'SPEAKER_01'
        assert info['roles']['SPEAKER_00'] == 'preserved'
        assert info['preserved_speakers'] == ['SPEAKER_00']
        assert info['preserved_s'] == 7.2
        assert info['reassigned_blips'] == 0
        # preserved audio rides along in the output
        assert float(np.abs(combined[int(1 * sr):int(5 * sr)]).max()) > 0.1

    def test_preserve_skips_span_backing_conversion(self, monkeypatch):
        sr = 8000
        voc = np.zeros(12 * sr, dtype=np.float32)
        voc[:10 * sr] = 0.5
        p = self._pipeline(monkeypatch, [
            (0.0, 6.0, 'SPEAKER_00'),
            (6.0, 10.0, 'SPEAKER_01'),
        ])
        called = []

        def fake_stack(backing, sr_, pid, mm):
            called.append(1)
            return backing, {'mode': 'converted', 'lines_detected': 1,
                             'lines_converted': 1}
        monkeypatch.setattr(p, '_convert_backing_stack', fake_stack)
        out = p._convert_multi_speaker(voc, sr, 'pid', IdentityMM(), 0.0,
                                       convert_backing=True,
                                       preserve_speakers=['SPEAKER_00'])
        assert out is not None
        _, info = out
        assert not called, "span backing with preserved voices must not be re-voiced"
        assert info['backing_mode'] == 'kept'

    def test_preserve_by_time_range_resolves_cluster(self, monkeypatch):
        # Cluster labels are not stable run-to-run; a time range where the
        # already-target singer performs must resolve to whichever cluster
        # owns that range in this run's diarization.
        sr = 8000
        voc = np.zeros(12 * sr, dtype=np.float32)
        voc[:int(11 * sr)] = 0.5
        p = self._pipeline(monkeypatch, [
            (0.0, 6.0, 'SPEAKER_00'),
            (6.0, 11.0, 'SPEAKER_01'),
        ])
        out = p._convert_multi_speaker(voc, sr, 'pid', IdentityMM(), 0.0,
                                       preserve_speakers=['0:07-0:10'])
        assert out is not None
        _, info = out
        assert info['preserved_speakers'] == ['SPEAKER_01']
        assert info['primary_speaker'] == 'SPEAKER_00'
        assert info['roles']['SPEAKER_01'] == 'preserved'

    def test_all_clusters_preserved_falls_back(self, monkeypatch):
        sr = 8000
        voc = np.zeros(10 * sr, dtype=np.float32)
        voc[:8 * sr] = 0.5
        p = self._pipeline(monkeypatch, [
            (0.0, 4.0, 'SPEAKER_00'),
            (4.0, 8.0, 'SPEAKER_01'),
        ])
        out = p._convert_multi_speaker(
            voc, sr, 'pid', IdentityMM(), 0.0,
            preserve_speakers=['SPEAKER_00', 'SPEAKER_01'])
        assert out is None


class TestConvertBackingWiring:
    def test_flag_off_by_default(self, monkeypatch):
        sr = 8000
        voc = np.zeros(10 * sr, dtype=np.float32)
        voc[:6 * sr] = 0.5
        patch_voiced(monkeypatch, 0.2)
        p = make_pipeline()
        p._diarizer = FakeDiarizer(make_result([
            (0.0, 4.0, 'SPEAKER_00'), (4.0, 6.0, 'SPEAKER_01'),
        ]))
        out = p._convert_multi_speaker(voc, sr, 'pid', IdentityMM(), 0.0)
        assert out is not None and out[1]['backing_mode'] == 'kept'
        assert 'harmony_lines' not in out[1]

    def test_per_request_override_engages(self, monkeypatch):
        sr = 8000
        voc = np.zeros(10 * sr, dtype=np.float32)
        voc[:6 * sr] = 0.5
        patch_voiced(monkeypatch, 0.2)
        p = make_pipeline()  # config flag off; per-request True wins
        p._diarizer = FakeDiarizer(make_result([
            (0.0, 4.0, 'SPEAKER_00'), (4.0, 6.0, 'SPEAKER_01'),
        ]))
        seen = {'called': False}

        def fake_convert(backing, sr_, pid, mm):
            seen['called'] = True
            return backing, {'mode': 'converted', 'lines_detected': 2,
                             'lines_converted': 2}
        monkeypatch.setattr(p, '_convert_backing_stack', fake_convert)
        out = p._convert_multi_speaker(voc, sr, 'pid', IdentityMM(), 0.0,
                                       convert_backing=True)
        assert seen['called'] is True
        assert out[1]['backing_mode'] == 'converted'
        assert out[1]['harmony_lines'] == {'detected': 2, 'converted': 2}
