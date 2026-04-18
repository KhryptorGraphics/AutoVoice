"""Additional branch coverage for karaoke_api.py."""

from __future__ import annotations

import io
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
from flask import Flask


def test_health_check_reports_degraded_components(client, monkeypatch):
    import auto_voice.web.karaoke_api as karaoke_api

    monkeypatch.setattr(client.application, 'karaoke_manager', None, raising=False)

    with patch.object(karaoke_api, '_get_voice_model_registry', side_effect=RuntimeError('registry down')):
        with patch('auto_voice.web.karaoke_api.os.makedirs'):
            with patch('builtins.open', side_effect=OSError('disk full')):
                response = client.get('/api/v1/karaoke/health')

    assert response.status_code == 503
    data = response.get_json()
    assert data['components']['karaoke_manager']['status'] == 'unavailable'
    assert data['components']['voice_model_registry']['status'] == 'unhealthy'
    assert data['components']['temp_storage']['status'] == 'unhealthy'


def test_rate_limit_resets_after_expired_window():
    from auto_voice.web.karaoke_api import _rate_limit_store, rate_limit

    app = Flask(__name__)

    @app.route('/reset-test')
    @rate_limit(max_requests=1, window_seconds=60)
    def limited():
        return 'OK'

    _rate_limit_store.clear()
    with app.test_client() as web_client:
        with patch('auto_voice.web.karaoke_api.time.time', return_value=100.0):
            first = web_client.get('/reset-test')
        with patch('auto_voice.web.karaoke_api.time.time', return_value=161.0):
            second = web_client.get('/reset-test')

    assert first.status_code == 200
    assert second.status_code == 200


def test_cleanup_old_songs_cleans_jobs_and_ignores_unlink_errors(monkeypatch):
    import auto_voice.web.karaoke_api as karaoke_api

    song_id = 'old-song'
    job_id = 'old-job'
    song_path = '/tmp/old-song.wav'
    vocals_path = '/tmp/old-vocals.wav'
    instrumental_path = '/tmp/old-inst.wav'

    karaoke_api._uploaded_songs[song_id] = {
        'id': song_id,
        'path': song_path,
        'uploaded_at': 0.0,
        'separation_job_id': job_id,
    }
    karaoke_api._separation_jobs[job_id] = {
        'job_id': job_id,
        'song_id': song_id,
        'vocals_path': vocals_path,
        'instrumental_path': instrumental_path,
    }

    deleted = []

    def fake_exists(path):
        return True

    def fake_unlink(path):
        deleted.append(path)
        if path in {song_path, vocals_path}:
            raise OSError('cannot unlink')

    monkeypatch.setattr(karaoke_api.os.path, 'exists', fake_exists)
    monkeypatch.setattr(karaoke_api.os, 'unlink', fake_unlink)

    try:
        count = karaoke_api.cleanup_old_songs(max_age_seconds=1)
        assert count >= 1
        assert song_id not in karaoke_api._uploaded_songs
        assert job_id not in karaoke_api._separation_jobs
        assert song_path in deleted
        assert vocals_path in deleted
        assert instrumental_path in deleted
    finally:
        karaoke_api._uploaded_songs.pop(song_id, None)
        karaoke_api._separation_jobs.pop(job_id, None)


def test_cleanup_on_shutdown_removes_sessions_and_temp_files(monkeypatch):
    import auto_voice.web.karaoke_api as karaoke_api

    karaoke_api._active_sessions['session-a'] = {
        'session_id': 'session-a',
        'song_id': 'song-a',
        'client_id': 'client-a',
        'started_at': 0.0,
        'last_activity': 0.0,
    }
    karaoke_api._uploaded_songs['song-a'] = {'path': '/tmp/song-a.wav'}
    karaoke_api._separation_jobs['job-a'] = {
        'vocals_path': '/tmp/song-a-vocals.wav',
        'instrumental_path': '/tmp/song-a-inst.wav',
    }

    cleaned = []
    unlinked = []

    monkeypatch.setattr(karaoke_api.os.path, 'exists', lambda path: True)
    monkeypatch.setattr(karaoke_api.os, 'unlink', lambda path: unlinked.append(path))
    monkeypatch.setattr(karaoke_api, 'cleanup_session', lambda session_id, reason='unknown': cleaned.append((session_id, reason)))

    try:
        karaoke_api._cleanup_on_shutdown()
        assert cleaned == [('session-a', 'shutdown')]
        assert unlinked == [
            '/tmp/song-a.wav',
            '/tmp/song-a-vocals.wav',
            '/tmp/song-a-inst.wav',
        ]
    finally:
        karaoke_api._active_sessions.clear()
        karaoke_api._uploaded_songs.clear()
        karaoke_api._separation_jobs.clear()


def test_get_audio_duration_supports_torchaudio_and_soundfile_fallbacks():
    from auto_voice.web.karaoke_api import _get_audio_duration

    fake_torchaudio = types.ModuleType('torchaudio')
    fake_torchaudio.info = lambda path: types.SimpleNamespace(num_frames=48000, sample_rate=24000)

    with patch.dict(sys.modules, {'torchaudio': fake_torchaudio}):
        assert _get_audio_duration('/tmp/audio.wav') == 2.0

    broken_torchaudio = types.ModuleType('torchaudio')
    broken_torchaudio.info = lambda path: (_ for _ in ()).throw(RuntimeError('bad torchaudio'))
    broken_librosa = types.ModuleType('librosa')
    broken_librosa.get_duration = lambda path=None: (_ for _ in ()).throw(RuntimeError('bad librosa'))
    fake_soundfile = types.ModuleType('soundfile')
    fake_soundfile.info = lambda path: types.SimpleNamespace(duration=3.5)

    with patch.dict(sys.modules, {
        'torchaudio': broken_torchaudio,
        'librosa': broken_librosa,
        'soundfile': fake_soundfile,
    }):
        assert _get_audio_duration('/tmp/audio.wav') == 3.5


def test_get_song_info_success_returns_public_fields(client):
    import auto_voice.web.karaoke_api as karaoke_api

    karaoke_api._uploaded_songs['song-public'] = {
        'id': 'song-public',
        'path': '/tmp/private.wav',
        'duration': 12.5,
        'sample_rate': 44100,
        'format': 'wav',
        'status': 'uploaded',
        'uploaded_at': 123.0,
    }

    try:
        response = client.get('/api/v1/karaoke/songs/song-public')
        assert response.status_code == 200
        data = response.get_json()
        assert data == {
            'song_id': 'song-public',
            'duration': 12.5,
            'sample_rate': 44100,
            'format': 'wav',
            'status': 'uploaded',
            'uploaded_at': 123.0,
        }
    finally:
        karaoke_api._uploaded_songs.pop('song-public', None)


def test_start_separation_handles_manager_failure(client, monkeypatch):
    import auto_voice.web.karaoke_api as karaoke_api

    karaoke_api._uploaded_songs['song-separate'] = {
        'id': 'song-separate',
        'path': '/tmp/song.wav',
        'duration': 60.0,
        'sample_rate': 44100,
        'format': 'wav',
        'file_size': 1000,
        'uploaded_at': 123.0,
        'status': 'uploaded',
    }
    failing_manager = MagicMock()
    failing_manager.start_separation.side_effect = RuntimeError('boom')
    monkeypatch.setattr(client.application, 'karaoke_manager', failing_manager, raising=False)
    karaoke_api._rate_limit_store.clear()

    try:
        response = client.post('/api/v1/karaoke/separate', json={'song_id': 'song-separate'})
        assert response.status_code == 202
        job_id = response.get_json()['job_id']
        assert karaoke_api._separation_jobs[job_id]['status'] == 'failed'
        assert karaoke_api._separation_jobs[job_id]['error'] == 'boom'
    finally:
        karaoke_api._uploaded_songs.pop('song-separate', None)
        karaoke_api._separation_jobs.clear()


def test_get_separation_status_covers_processing_completed_and_failed(client, monkeypatch):
    import auto_voice.web.karaoke_api as karaoke_api

    karaoke_api._separation_jobs['job-processing'] = {
        'job_id': 'job-processing',
        'song_id': 'song-1',
        'status': 'processing',
        'progress': 50,
        'created_at': 0.0,
        'estimated_time': 30,
        'vocals_path': None,
        'instrumental_path': None,
        'error': None,
    }
    karaoke_api._separation_jobs['job-complete'] = {
        'job_id': 'job-complete',
        'song_id': 'song-2',
        'status': 'queued',
        'progress': 0,
        'created_at': 0.0,
        'estimated_time': 30,
        'vocals_path': None,
        'instrumental_path': None,
        'error': None,
    }
    karaoke_api._separation_jobs['job-failed'] = {
        'job_id': 'job-failed',
        'song_id': 'song-3',
        'status': 'failed',
        'progress': 5,
        'created_at': 0.0,
        'estimated_time': 30,
        'vocals_path': None,
        'instrumental_path': None,
        'error': 'bad separation',
    }

    manager = MagicMock()
    manager.get_job_status.side_effect = [
        None,
        {'status': 'completed', 'progress': 100, 'vocals_path': '/tmp/v.wav', 'instrumental_path': '/tmp/i.wav'},
        {'status': 'failed', 'progress': 12, 'error': 'manager failure'},
    ]
    monkeypatch.setattr(client.application, 'karaoke_manager', manager, raising=False)

    with patch('auto_voice.web.karaoke_api.time.time', return_value=10.0):
        processing = client.get('/api/v1/karaoke/separate/job-processing')
    completed = client.get('/api/v1/karaoke/separate/job-complete')
    failed = client.get('/api/v1/karaoke/separate/job-failed')

    try:
        assert processing.get_json()['estimated_remaining'] == 20
        assert completed.get_json()['vocals_ready'] is True
        assert completed.get_json()['instrumental_ready'] is True
        assert failed.get_json()['error'] == 'manager failure'
    finally:
        karaoke_api._separation_jobs.clear()


def test_set_output_device_invalid_headphone_returns_400(client):
    with patch('auto_voice.web.audio_router.list_audio_devices', return_value=[{'index': 0, 'name': 'Default'}]):
        response = client.post(
            '/api/v1/karaoke/devices/output',
            json={'headphone_device': 99},
            content_type='application/json',
        )

    assert response.status_code == 400
    assert 'error' in response.get_json()


def test_get_voice_model_success(client):
    registry = MagicMock()
    registry.get_model.return_value = {'id': 'model-1', 'name': 'Model One', 'type': 'pretrained'}

    with patch('auto_voice.web.karaoke_api._get_voice_model_registry', return_value=registry):
        response = client.get('/api/v1/karaoke/voice-models/model-1')

    assert response.status_code == 200
    assert response.get_json()['id'] == 'model-1'


def test_extract_voice_model_validates_separation_state(client):
    import auto_voice.web.karaoke_api as karaoke_api

    karaoke_api._rate_limit_store.clear()
    karaoke_api._uploaded_songs['song-no-job'] = {'id': 'song-no-job'}
    karaoke_api._uploaded_songs['song-processing'] = {'id': 'song-processing', 'separation_job_id': 'job-processing'}
    karaoke_api._uploaded_songs['song-no-vocals'] = {'id': 'song-no-vocals', 'separation_job_id': 'job-no-vocals'}
    karaoke_api._separation_jobs['job-processing'] = {'status': 'processing'}
    karaoke_api._separation_jobs['job-no-vocals'] = {'status': 'completed', 'vocals_path': None}

    try:
        missing_job = client.post('/api/v1/karaoke/voice-models/extract', json={'song_id': 'song-no-job', 'name': 'Demo'})
        processing = client.post('/api/v1/karaoke/voice-models/extract', json={'song_id': 'song-processing', 'name': 'Demo'})
        no_vocals = client.post('/api/v1/karaoke/voice-models/extract', json={'song_id': 'song-no-vocals', 'name': 'Demo'})

        assert missing_job.status_code == 400
        assert processing.status_code == 400
        assert no_vocals.status_code == 400
    finally:
        karaoke_api._uploaded_songs.clear()
        karaoke_api._separation_jobs.clear()
        karaoke_api._rate_limit_store.clear()


def test_extract_voice_model_success_and_failure_paths(client):
    import auto_voice.web.karaoke_api as karaoke_api

    karaoke_api._rate_limit_store.clear()
    karaoke_api._uploaded_songs['song-extract'] = {
        'id': 'song-extract',
        'separation_job_id': 'job-extract',
    }
    karaoke_api._separation_jobs['job-extract'] = {
        'status': 'completed',
        'vocals_path': '/tmp/vocals.wav',
    }

    registry = MagicMock()
    registry.register_extracted_model.return_value = 'model-new'

    stereo_audio = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    with patch('auto_voice.web.karaoke_api._get_voice_model_registry', return_value=registry):
        with patch('soundfile.read', return_value=(stereo_audio, 24000)):
            with patch('auto_voice.web.voice_model_registry.extract_speaker_embedding', return_value=np.ones(256, dtype=np.float32)):
                success = client.post('/api/v1/karaoke/voice-models/extract', json={'song_id': 'song-extract', 'name': 'Demo'})

    assert success.status_code == 201
    assert success.get_json()['model_id'] == 'model-new'

    karaoke_api._rate_limit_store.clear()
    with patch('soundfile.read', side_effect=RuntimeError('bad audio')):
        failure = client.post('/api/v1/karaoke/voice-models/extract', json={'song_id': 'song-extract', 'name': 'Demo'})

    try:
        assert failure.status_code == 503
    finally:
        karaoke_api._uploaded_songs.clear()
        karaoke_api._separation_jobs.clear()
        karaoke_api._rate_limit_store.clear()


def test_upload_song_rejects_too_long_audio(client):
    import auto_voice.web.karaoke_api as karaoke_api

    karaoke_api._rate_limit_store.clear()

    with patch('auto_voice.web.karaoke_api._get_audio_duration', return_value=601.0):
        response = client.post(
            '/api/v1/karaoke/upload',
            data={'song': (io.BytesIO(b'RIFF' + b'\x00' * 64), 'too_long.wav')},
            content_type='multipart/form-data',
        )

    assert response.status_code == 400
    assert 'Song too long' in response.get_json()['error']
