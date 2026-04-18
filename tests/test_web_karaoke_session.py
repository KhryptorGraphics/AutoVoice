"""Focused unit tests for karaoke_session.py."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


def make_session(**overrides):
    from auto_voice.web.karaoke_session import KaraokeSession

    params = {
        'session_id': 'session-1',
        'song_id': 'song-1',
        'vocals_path': '',
        'instrumental_path': '',
        'sample_rate': 24000,
        'device': torch.device('cpu'),
    }
    params.update(overrides)
    return KaraokeSession(**params)


def make_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def test_audio_mixer_load_instrumental_resamples_and_flattens_stereo():
    from auto_voice.web.karaoke_session import AudioMixer

    mixer = AudioMixer(sample_rate=24000)
    stereo = torch.tensor([[1.0, 3.0], [3.0, 1.0]], dtype=torch.float32)
    resampled = torch.tensor([[2.0, 4.0], [4.0, 2.0]], dtype=torch.float32)

    resampler = MagicMock(return_value=resampled)
    with patch('auto_voice.web.karaoke_session.load_audio', return_value=(stereo, 44100)):
        with patch('auto_voice.web.karaoke_session.torchaudio.transforms.Resample', return_value=resampler):
            mixer.load_instrumental('/tmp/instrumental.wav')

    assert torch.allclose(mixer._instrumental_buffer, torch.tensor([3.0, 3.0]))
    assert mixer._playback_position == 0


def test_audio_mixer_load_instrumental_failure_clears_buffer():
    from auto_voice.web.karaoke_session import AudioMixer

    mixer = AudioMixer()
    mixer._instrumental_buffer = torch.ones(2)

    with patch('auto_voice.web.karaoke_session.load_audio', side_effect=RuntimeError('boom')):
        mixer.load_instrumental('/tmp/missing.wav')

    assert mixer._instrumental_buffer is None


def test_audio_mixer_mix_without_instrumental_returns_voice_only():
    from auto_voice.web.karaoke_session import AudioMixer

    mixer = AudioMixer(voice_gain=1.5)
    voice = torch.tensor([0.25, -0.25], dtype=torch.float32)

    mixed, voice_only = mixer.mix(voice)

    assert torch.allclose(mixed, voice * 1.5)
    assert torch.allclose(voice_only, voice * 1.5)


def test_audio_mixer_mix_pads_normalizes_and_advances_position():
    from auto_voice.web.karaoke_session import AudioMixer

    mixer = AudioMixer(voice_gain=1.0, instrumental_gain=1.0)
    mixer._instrumental_buffer = torch.tensor([0.8], dtype=torch.float32)
    voice = torch.tensor([0.8, 0.8], dtype=torch.float32)

    mixed, voice_only = mixer.mix(voice)

    assert mixed.shape == voice.shape
    assert torch.isclose(mixed.abs().max(), torch.tensor(1.0))
    assert torch.allclose(voice_only, voice)
    assert mixer._playback_position == 2


def test_audio_mixer_mix_with_explicit_position_past_end_returns_voice_only():
    from auto_voice.web.karaoke_session import AudioMixer

    mixer = AudioMixer()
    mixer._instrumental_buffer = torch.tensor([0.1, 0.2], dtype=torch.float32)
    mixer._playback_position = 1
    voice = torch.tensor([0.4, 0.5], dtype=torch.float32)

    mixed, voice_only = mixer.mix(voice, position_samples=5)

    assert torch.allclose(mixed, voice)
    assert torch.allclose(voice_only, voice)
    assert mixer._playback_position == 1


def test_audio_mixer_position_and_gain_helpers_clamp_values():
    from auto_voice.web.karaoke_session import AudioMixer

    mixer = AudioMixer()
    mixer.set_position(-5)
    mixer.set_gains(-1.0, 3.5)
    mixer.reset_position()

    assert mixer._playback_position == 0
    assert mixer.voice_gain == 0.0
    assert mixer.instrumental_gain == 2.0


def test_set_speaker_embedding_updates_existing_pipeline_target_voice():
    session = make_session()
    pipeline = MagicMock()
    session._streaming_pipeline = pipeline

    session.set_speaker_embedding(torch.ones(256))

    pipeline.set_target_voice.assert_called_once()
    assert tuple(session.speaker_embedding.shape) == (1, 256)


def test_set_speaker_embedding_swallows_live_pipeline_update_failures():
    session = make_session()
    pipeline = MagicMock()
    pipeline.set_target_voice.side_effect = RuntimeError('update failed')
    session._streaming_pipeline = pipeline

    session.set_speaker_embedding(torch.ones(256))

    assert session.speaker_embedding is not None


def test_load_voice_model_raises_for_missing_model():
    session = make_session()
    registry = MagicMock()
    registry.get_model.return_value = None

    with pytest.raises(ValueError, match='Voice model not found'):
        session.load_voice_model(registry, 'missing')


def test_load_voice_model_raises_for_missing_embedding():
    session = make_session()
    registry = MagicMock()
    registry.get_model.return_value = {'id': 'model-1'}
    registry.get_embedding.return_value = None

    with pytest.raises(ValueError, match='Could not load embedding'):
        session.load_voice_model(registry, 'model-1')


def test_check_trt_available_handles_import_error():
    session = make_session()
    original_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'auto_voice.inference.trt_streaming_pipeline':
            raise ImportError('missing trt')
        return original_import(name, globals, locals, fromlist, level)

    with patch('builtins.__import__', side_effect=fake_import):
        assert session._check_trt_available() is False
        assert session._check_trt_available() is False


def test_get_pipeline_uses_full_model_pipeline_when_available():
    realtime_pipeline = MagicMock()
    realtime_cls = MagicMock(return_value=realtime_pipeline)
    fake_module = make_module(
        'auto_voice.inference.realtime_voice_conversion_pipeline',
        RealtimeVoiceConversionPipeline=realtime_cls,
    )
    session = make_session()
    session._target_model_type = 'full_model'
    session._target_profile_id = 'profile-123'
    session._full_model_path = '/tmp/full-model.pt'
    session.set_speaker_embedding(torch.ones(256))

    with patch.dict(sys.modules, {fake_module.__name__: fake_module}):
        with patch('auto_voice.web.karaoke_session.os.path.exists', return_value=True):
            pipeline = session._get_pipeline()

    assert pipeline is realtime_pipeline
    realtime_cls.assert_called_once()
    realtime_pipeline.set_target_voice.assert_called_once()
    realtime_pipeline.start.assert_called_once()
    assert session.pipeline_type == 'pytorch_full_model'


def test_get_pipeline_raises_when_trt_required_but_unavailable():
    session = make_session(use_trt=True)

    with patch.object(session, '_check_trt_available', return_value=False):
        with pytest.raises(RuntimeError, match='TRT requested but engines not found'):
            session._get_pipeline()


def test_get_pipeline_uses_tensorrt_when_available():
    trt_pipeline = MagicMock()
    trt_cls = MagicMock(return_value=trt_pipeline)
    fake_module = make_module(
        'auto_voice.inference.trt_streaming_pipeline',
        TRTStreamingPipeline=trt_cls,
    )
    session = make_session(use_trt=True)

    with patch.dict(sys.modules, {fake_module.__name__: fake_module}):
        with patch.object(session, '_check_trt_available', return_value=True):
            pipeline = session._get_pipeline()

    assert pipeline is trt_pipeline
    trt_pipeline.load_engines.assert_called_once()
    assert session.pipeline_type == 'tensorrt'


def test_get_pipeline_uses_target_profile_in_streaming_pipeline():
    stream_pipeline = MagicMock()
    stream_cls = MagicMock(return_value=stream_pipeline)
    fake_module = make_module(
        'auto_voice.inference.streaming_pipeline',
        StreamingConversionPipeline=stream_cls,
    )
    session = make_session(use_trt=False)
    session._target_profile_id = 'profile-42'

    with patch.dict(sys.modules, {fake_module.__name__: fake_module}):
        pipeline = session._get_pipeline()

    assert pipeline is stream_pipeline
    stream_pipeline.set_speaker.assert_called_once_with('profile-42')
    stream_pipeline.start_session.assert_not_called()
    assert session.pipeline_type == 'pytorch'


def test_get_pipeline_falls_back_to_start_session_when_set_speaker_fails():
    stream_pipeline = MagicMock()
    stream_pipeline.set_speaker.side_effect = RuntimeError('speaker load failed')
    stream_cls = MagicMock(return_value=stream_pipeline)
    fake_module = make_module(
        'auto_voice.inference.streaming_pipeline',
        StreamingConversionPipeline=stream_cls,
    )
    session = make_session(use_trt=False)
    session._target_profile_id = 'profile-42'
    session.set_speaker_embedding(torch.ones(256))

    with patch.dict(sys.modules, {fake_module.__name__: fake_module}):
        session._get_pipeline()

    stream_pipeline.start_session.assert_called_once()


def test_get_pipeline_starts_session_from_embedding_when_no_profile():
    stream_pipeline = MagicMock()
    stream_cls = MagicMock(return_value=stream_pipeline)
    fake_module = make_module(
        'auto_voice.inference.streaming_pipeline',
        StreamingConversionPipeline=stream_cls,
    )
    session = make_session(use_trt=False)
    session.set_speaker_embedding(torch.ones(256))

    with patch.dict(sys.modules, {fake_module.__name__: fake_module}):
        session._get_pipeline()

    stream_pipeline.start_session.assert_called_once()


def test_get_pipeline_wraps_initialization_failures():
    stream_cls = MagicMock(side_effect=RuntimeError('init failed'))
    fake_module = make_module(
        'auto_voice.inference.streaming_pipeline',
        StreamingConversionPipeline=stream_cls,
    )
    session = make_session(use_trt=False)

    with patch.dict(sys.modules, {fake_module.__name__: fake_module}):
        with pytest.raises(RuntimeError, match='Pipeline initialization failed'):
            session._get_pipeline()


def test_start_is_idempotent_and_swallows_pipeline_errors():
    session = make_session()

    with patch.object(session, '_get_pipeline', side_effect=RuntimeError('not ready')) as mock_get:
        session.start()
        session.start()

    assert session.is_active is True
    assert mock_get.call_count == 1


def test_stop_handles_stop_session_and_stop_methods():
    stop_session_pipeline = MagicMock()
    session = make_session()
    session.is_active = True
    session._streaming_pipeline = stop_session_pipeline
    session.stop()
    stop_session_pipeline.stop_session.assert_called_once()

    stop_pipeline = MagicMock()
    del stop_pipeline.stop_session
    session = make_session()
    session.is_active = True
    session._streaming_pipeline = stop_pipeline
    session.stop()
    stop_pipeline.stop.assert_called_once()


def test_stop_swallows_pipeline_stop_errors():
    pipeline = MagicMock()
    pipeline.stop_session.side_effect = RuntimeError('stop failed')
    session = make_session()
    session.is_active = True
    session._streaming_pipeline = pipeline

    session.stop()

    assert session.is_active is False


def test_process_chunk_validates_active_session_and_embedding():
    session = make_session()

    with pytest.raises(RuntimeError, match='Session is not active'):
        session.process_chunk(torch.ones(4))

    session.is_active = True
    with pytest.raises(RuntimeError, match='Speaker embedding not set'):
        session.process_chunk(torch.ones(4))


def test_process_chunk_converts_numpy_input_and_trims_latency_history():
    pipeline = MagicMock()
    pipeline.process_chunk.return_value = torch.tensor([0.5, -0.5], dtype=torch.float32)
    session = make_session()
    session.set_speaker_embedding(torch.ones(256))
    session.is_active = True
    session._max_latency_history = 1

    with patch.object(session, '_get_pipeline', return_value=pipeline):
        first = session.process_chunk(np.array([0.1, 0.2], dtype=np.float32))
        second = session.process_chunk(np.array([0.3, 0.4], dtype=np.float32))

    assert torch.allclose(first, torch.tensor([0.5, -0.5]))
    assert torch.allclose(second, torch.tensor([0.5, -0.5]))
    assert len(session._latency_history) == 1
    assert session._chunks_processed == 2


def test_process_chunk_handles_full_model_pipeline_outputs_and_runtime_fallback():
    full_pipeline = MagicMock()
    full_pipeline.process_chunk.return_value = np.array([0.2, 0.4], dtype=np.float32)
    session = make_session()
    session._target_model_type = 'full_model'
    session.set_speaker_embedding(torch.ones(256))
    session.is_active = True

    with patch.object(session, '_get_pipeline', return_value=full_pipeline):
        output = session.process_chunk(torch.tensor([0.1, 0.2], dtype=torch.float32))

    full_pipeline.set_target_voice.assert_called_once()
    assert torch.allclose(output, torch.tensor([0.2, 0.4]))

    fallback = make_session()
    fallback.set_speaker_embedding(torch.ones(256))
    fallback.is_active = True
    original = torch.tensor([0.8, -0.8], dtype=torch.float32)

    with patch.object(fallback, '_get_pipeline', side_effect=RuntimeError('test fallback')):
        output = fallback.process_chunk(original.clone())

    assert torch.allclose(output, original)


def test_get_latency_and_stats_cover_empty_and_trt_paths():
    session = make_session()
    assert session.get_latency_ms() == 0.0

    trt_pipeline = MagicMock()
    trt_pipeline.get_engine_memory_usage.return_value = 5 * 1024 * 1024
    session._streaming_pipeline = trt_pipeline
    session._pipeline_type = 'tensorrt'
    session._latency_history = [10.0, 20.0]
    session._started_at = 1.0

    with patch('auto_voice.web.karaoke_session.time.time', return_value=6.0):
        stats = session.get_stats()

    assert stats['avg_latency_ms'] == 15.0
    assert stats['min_latency_ms'] == 10.0
    assert stats['max_latency_ms'] == 20.0
    assert stats['duration_s'] == 5.0
    assert stats['trt_memory_mb'] == 5.0

    trt_pipeline.get_engine_memory_usage.side_effect = RuntimeError('missing stats')
    session.get_stats()


def test_process_chunk_with_mix_and_mixer_helpers_delegate():
    session = make_session()
    converted = torch.tensor([0.3, 0.1], dtype=torch.float32)

    with patch.object(session, 'process_chunk', return_value=converted):
        with patch.object(session._mixer, 'mix', return_value=(torch.tensor([0.9, 0.7]), converted)) as mock_mix:
            speaker, headphone = session.process_chunk_with_mix(torch.tensor([0.1, 0.2]))

    assert torch.allclose(speaker, torch.tensor([0.9, 0.7]))
    assert torch.allclose(headphone, converted)
    mock_mix.assert_called_once_with(converted)

    with patch.object(session._mixer, 'set_gains') as mock_set_gains:
        session.set_mixer_gains(1.1, 0.9)
        mock_set_gains.assert_called_once_with(1.1, 0.9)

    with patch.object(session._mixer, 'reset_position') as mock_reset:
        session.reset_playback()
        mock_reset.assert_called_once_with()

    with patch.object(session._mixer, 'set_position') as mock_set_position:
        session.seek_playback(1.5)
        mock_set_position.assert_called_once_with(36000)
