"""Context-aware tests for karaoke_events.py handlers."""

from __future__ import annotations

import base64
import sys
import time
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from flask import Flask, request


@pytest.fixture
def ws_app():
    app = Flask(__name__)
    app.config['TESTING'] = True
    return app


def test_on_connect_emits_connected(ws_app):
    from auto_voice.web.karaoke_events import KaraokeNamespace

    ns = KaraokeNamespace()

    with ws_app.test_request_context('/socket.io'):
        request.sid = 'client-123'
        with patch('auto_voice.web.karaoke_events.emit') as mock_emit:
            ns.on_connect()

    assert ns._client_connect_time['client-123'] > 0
    mock_emit.assert_called_once()
    assert mock_emit.call_args[0][0] == 'connected'
    assert mock_emit.call_args[0][1]['client_id'] == 'client-123'


def test_on_disconnect_cleans_up_session_and_collector(ws_app):
    from auto_voice.web.karaoke_events import KaraokeNamespace

    ns = KaraokeNamespace()
    mock_session = MagicMock()
    mock_session.get_stats.return_value = {'duration_s': 12.0, 'chunks_processed': 34}
    mock_collector = MagicMock()
    mock_collector.stop_recording.return_value = ['sample-a']
    ns._sessions['session-1'] = mock_session
    ns._client_sessions['client-123'] = 'session-1'
    ns._client_connect_time['client-123'] = time.time()
    ns._sample_collectors['session-1'] = mock_collector

    with ws_app.test_request_context('/socket.io'):
        request.sid = 'client-123'
        with patch('auto_voice.web.karaoke_events.cleanup_session') as mock_cleanup:
            ns.on_disconnect()

    mock_session.stop.assert_called_once()
    mock_collector.stop_recording.assert_called_once()
    mock_cleanup.assert_called_once_with('session-1', reason='client_disconnect')
    assert 'session-1' not in ns._sessions
    assert 'client-123' not in ns._client_sessions


def test_on_start_session_missing_params_emits_error(ws_app):
    from auto_voice.web.karaoke_events import KaraokeNamespace

    ns = KaraokeNamespace()

    with ws_app.test_request_context('/socket.io'):
        request.sid = 'client-123'
        with patch('auto_voice.web.karaoke_events.emit') as mock_emit:
            ns.on_start_session({'session_id': 'session-1'})

    error_calls = [c for c in mock_emit.call_args_list if c[0][0] == 'error']
    assert len(error_calls) == 1


def test_on_start_session_creates_session_and_registers(ws_app):
    from auto_voice.web.karaoke_events import KaraokeNamespace

    ns = KaraokeNamespace()
    mock_session = MagicMock()
    mock_session._target_model_type = None

    with ws_app.test_request_context('/socket.io'):
        request.sid = 'client-123'
        with patch('auto_voice.web.karaoke_events.KaraokeSession', return_value=mock_session) as mock_cls:
            with patch('auto_voice.web.karaoke_events.register_session') as mock_register:
                with patch('auto_voice.web.karaoke_events.emit') as mock_emit:
                    ns.on_start_session({'session_id': 'session-1', 'song_id': 'song-1'})

    mock_cls.assert_called_once()
    mock_session.start.assert_called_once()
    mock_register.assert_called_once_with('session-1', 'song-1', 'client-123')
    assert ns._sessions['session-1'] is mock_session
    assert ns._client_sessions['client-123'] == 'session-1'
    started_calls = [c for c in mock_emit.call_args_list if c[0][0] == 'session_started']
    assert len(started_calls) == 1
    assert started_calls[0][0][1]['session_id'] == 'session-1'


def test_on_start_session_with_embedding_sets_speaker_embedding(ws_app):
    from auto_voice.web.karaoke_events import KaraokeNamespace

    ns = KaraokeNamespace()
    mock_session = MagicMock()
    embedding = np.random.randn(256).astype(np.float32)
    embedding_b64 = base64.b64encode(embedding.tobytes()).decode('utf-8')

    with ws_app.test_request_context('/socket.io'):
        request.sid = 'client-123'
        with patch('auto_voice.web.karaoke_events.KaraokeSession', return_value=mock_session):
            with patch('auto_voice.web.karaoke_events.register_session'):
                with patch('auto_voice.web.karaoke_events.emit'):
                    ns.on_start_session({
                        'session_id': 'session-2',
                        'song_id': 'song-2',
                        'speaker_embedding': embedding_b64,
                    })

    mock_session.set_speaker_embedding.assert_called_once()


def test_on_start_session_with_profile_and_sample_collection(ws_app):
    from auto_voice.web.karaoke_events import KaraokeNamespace

    ns = KaraokeNamespace()
    mock_session = MagicMock()
    store = MagicMock()
    store.load.return_value = {
        'profile_id': 'profile-1',
        'active_model_type': 'full_model',
    }
    store.load_speaker_embedding.return_value = np.ones(256, dtype=np.float32)
    store.profiles_dir = '/tmp/profiles'
    store.trained_models_dir = '/tmp/trained'
    ws_app.voice_cloner = types.SimpleNamespace(store=store)

    collector = MagicMock()
    sample_collector_module = types.SimpleNamespace(SampleCollector=MagicMock(return_value=collector))

    with ws_app.test_request_context('/socket.io'):
        request.sid = 'client-123'
        with patch.dict(sys.modules, {'auto_voice.profiles.sample_collector': sample_collector_module}):
            with patch('auto_voice.web.karaoke_events.os.path.exists', return_value=True):
                with patch('auto_voice.web.karaoke_events.KaraokeSession', return_value=mock_session):
                    with patch('auto_voice.web.karaoke_events.register_session'):
                        with patch('auto_voice.web.karaoke_events.emit') as mock_emit:
                            ns.on_start_session({
                                'session_id': 'session-3',
                                'song_id': 'song-3',
                                'profile_id': 'profile-1',
                                'collect_samples': True,
                            })

    mock_session.set_speaker_embedding.assert_called_once()
    collector.start_recording.assert_called_once()
    started_calls = [c for c in mock_emit.call_args_list if c[0][0] == 'session_started']
    payload = started_calls[0][0][1]
    assert payload['sample_collection_enabled'] is True
    assert payload['target_profile_id'] == 'profile-1'
    assert payload['active_model_type'] == 'full_model'


def test_on_start_session_persists_snapshot_and_uses_recovery_defaults(ws_app, tmp_path):
    from auto_voice.web.karaoke_events import KaraokeNamespace
    from auto_voice.web.persistence import AppStateStore

    ws_app.state_store = AppStateStore(str(tmp_path))
    ws_app.state_store.save_karaoke_session({
        'session_id': 'session-restore',
        'song_id': 'song-restore',
        'vocals_path': '/tmp/vocals.wav',
        'instrumental_path': '/tmp/instrumental.wav',
        'requested_pipeline': 'realtime',
        'speaker_embedding': np.ones(256, dtype=np.float32).tolist(),
    })

    ns = KaraokeNamespace()
    mock_session = MagicMock()
    mock_session._target_model_type = None
    mock_session.get_recovery_snapshot.return_value = {
        'session_id': 'session-restore',
        'song_id': 'song-restore',
        'vocals_path': '/tmp/vocals.wav',
        'instrumental_path': '/tmp/instrumental.wav',
        'requested_pipeline': 'realtime',
        'resolved_pipeline': 'realtime',
        'runtime_backend': 'pytorch',
        'speaker_embedding': np.ones(256, dtype=np.float32).tolist(),
        'stats': {'chunks_processed': 0},
    }

    with ws_app.test_request_context('/socket.io'):
        request.sid = 'client-restore'
        with patch('auto_voice.web.karaoke_events.KaraokeSession', return_value=mock_session) as mock_cls:
            with patch('auto_voice.web.karaoke_events.register_session'):
                with patch('auto_voice.web.karaoke_events.emit'):
                    ns.on_start_session({'session_id': 'session-restore', 'song_id': 'song-restore'})

    assert mock_cls.call_args.kwargs['vocals_path'] == '/tmp/vocals.wav'
    assert mock_cls.call_args.kwargs['instrumental_path'] == '/tmp/instrumental.wav'
    mock_session.set_speaker_embedding.assert_called_once()
    snapshot = ws_app.state_store.get_karaoke_session('session-restore')
    assert snapshot is not None
    assert snapshot['session_id'] == 'session-restore'
    assert snapshot['audio_router_targets']['speaker_device'] is None


def test_on_stop_session_persists_inactive_snapshot(ws_app, tmp_path):
    from auto_voice.web.karaoke_events import KaraokeNamespace
    from auto_voice.web.persistence import AppStateStore

    ws_app.state_store = AppStateStore(str(tmp_path))
    ns = KaraokeNamespace()
    mock_session = MagicMock()
    mock_session.get_stats.return_value = {'duration_s': 3.0, 'chunks_processed': 4}
    mock_session.get_recovery_snapshot.return_value = {
        'session_id': 'session-stop',
        'song_id': 'song-stop',
        'vocals_path': '/tmp/vocals.wav',
        'instrumental_path': '/tmp/instrumental.wav',
        'requested_pipeline': 'realtime',
        'resolved_pipeline': 'realtime',
        'runtime_backend': 'pytorch',
        'is_active': False,
        'stats': {'duration_s': 3.0, 'chunks_processed': 4},
    }
    ns._sessions['session-stop'] = mock_session
    ns._client_sessions['client-stop'] = 'session-stop'

    with ws_app.test_request_context('/socket.io'):
        request.sid = 'client-stop'
        with patch('auto_voice.web.karaoke_events.emit'):
            with patch('auto_voice.web.karaoke_events.cleanup_session') as mock_cleanup:
                ns.on_stop_session({'session_id': 'session-stop'})

    snapshot = ws_app.state_store.get_karaoke_session('session-stop')
    assert snapshot is not None
    assert snapshot['stats']['chunks_processed'] == 4
    assert 'client-stop' not in ns._client_sessions
    mock_cleanup.assert_called_once_with('session-stop', reason='stopped')


def test_on_audio_chunk_without_session_emits_error(ws_app):
    from auto_voice.web.karaoke_events import KaraokeNamespace

    ns = KaraokeNamespace()

    with ws_app.test_request_context('/socket.io'):
        request.sid = 'client-123'
        with patch('auto_voice.web.karaoke_events.emit') as mock_emit:
            ns.on_audio_chunk({'audio': ''})

    assert mock_emit.call_args[0][0] == 'error'


def test_on_audio_chunk_processes_audio_and_emits_result(ws_app):
    from auto_voice.web.karaoke_events import KaraokeNamespace

    ns = KaraokeNamespace()
    mock_session = MagicMock()
    mock_session.is_active = True
    mock_session.process_chunk.return_value = torch.ones(8, dtype=torch.float32)
    mock_session.get_latency_ms.return_value = 12.5
    ns._sessions['session-1'] = mock_session
    ns._client_sessions['client-123'] = 'session-1'
    collector = MagicMock()
    ns._sample_collectors['session-1'] = collector
    audio = np.linspace(-1.0, 1.0, num=8, dtype=np.float32)
    audio_b64 = base64.b64encode(audio.tobytes()).decode('utf-8')

    with ws_app.test_request_context('/socket.io'):
        request.sid = 'client-123'
        with patch('auto_voice.web.karaoke_events.update_session_activity') as mock_update:
            with patch('auto_voice.web.karaoke_events.emit') as mock_emit:
                ns.on_audio_chunk({'audio': audio_b64, 'timestamp': 123})

    mock_update.assert_called_once_with('session-1')
    mock_session.process_chunk.assert_called_once()
    collector.add_chunk.assert_called_once()
    converted_calls = [c for c in mock_emit.call_args_list if c[0][0] == 'converted_audio']
    assert len(converted_calls) == 1
    assert converted_calls[0][0][1]['timestamp'] == 123


def test_on_audio_chunk_processing_error_emits_error(ws_app):
    from auto_voice.web.karaoke_events import KaraokeNamespace

    ns = KaraokeNamespace()
    mock_session = MagicMock()
    mock_session.is_active = True
    mock_session.process_chunk.side_effect = RuntimeError('broken pipeline')
    ns._sessions['session-1'] = mock_session
    audio = np.zeros(4, dtype=np.float32)
    audio_b64 = base64.b64encode(audio.tobytes()).decode('utf-8')

    with ws_app.test_request_context('/socket.io'):
        request.sid = 'client-123'
        with patch('auto_voice.web.karaoke_events.emit') as mock_emit:
            ns.on_audio_chunk({'session_id': 'session-1', 'audio': audio_b64})

    error_calls = [c for c in mock_emit.call_args_list if c[0][0] == 'error']
    assert len(error_calls) == 1
