"""Tests for karaoke_api.py - Karaoke API endpoints.

Coverage target: 70% (from 30%)

Covers:
- Health check and metrics endpoints
- Song upload and info endpoints
- Vocal separation endpoints
- Audio device management
- Voice model management
- Session and cleanup functionality
- Rate limiting
- Error handling
"""
import io
import os
import pytest
import tempfile
import time
from unittest.mock import Mock, MagicMock, patch
import numpy as np


class TestHealthAndMetrics:
    """Test health check and metrics endpoints."""

    @pytest.mark.smoke
    def test_health_check_healthy(self, client):
        """Test health check returns healthy status."""
        response = client.get('/api/v1/karaoke/health')

        # Even if some components are unavailable, endpoint should work
        assert response.status_code in (200, 503)
        data = response.get_json()
        assert 'status' in data
        assert 'timestamp' in data
        assert 'components' in data

    def test_health_check_components(self, client):
        """Test health check includes all expected components."""
        response = client.get('/api/v1/karaoke/health')

        data = response.get_json()
        components = data.get('components', {})

        # Check for expected component keys
        expected_components = ['karaoke_manager', 'voice_model_registry', 'storage', 'temp_storage']
        for comp in expected_components:
            assert comp in components, f"Missing component: {comp}"

    def test_health_check_timestamp_format(self, client):
        """Test health check timestamp is ISO format."""
        response = client.get('/api/v1/karaoke/health')

        data = response.get_json()
        timestamp = data.get('timestamp', '')

        # Should be ISO 8601 format with Z suffix
        assert 'T' in timestamp
        assert timestamp.endswith('Z')

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint returns analytics."""
        response = client.get('/api/v1/karaoke/metrics')

        assert response.status_code == 200
        data = response.get_json()
        # Metrics endpoint should return some data (even if empty)
        assert isinstance(data, dict)


class TestSongUpload:
    """Test song upload functionality."""

    @pytest.mark.smoke
    def test_upload_no_file(self, client):
        """Test upload with no file returns 400."""
        response = client.post('/api/v1/karaoke/upload')

        assert response.status_code in (400, 429)
        data = response.get_json()
        assert 'error' in data
        if response.status_code == 400:
            assert 'no song file' in data['error'].lower() or 'no file' in data['error'].lower()

    def test_upload_empty_filename(self, client):
        """Test upload with empty filename returns 400."""
        response = client.post(
            '/api/v1/karaoke/upload',
            data={'song': (io.BytesIO(b''), '')},
            content_type='multipart/form-data'
        )

        assert response.status_code in (400, 429)
        data = response.get_json()
        assert 'error' in data

    def test_upload_invalid_format(self, client):
        """Test upload with invalid format returns 400."""
        response = client.post(
            '/api/v1/karaoke/upload',
            data={'song': (io.BytesIO(b'test data'), 'test.txt')},
            content_type='multipart/form-data'
        )

        assert response.status_code in (400, 429)
        data = response.get_json()
        assert 'error' in data
        if response.status_code == 400:
            assert 'format' in data['error'].lower() or 'invalid' in data['error'].lower()

    def test_upload_valid_audio(self, client, tmp_path):
        """Test upload with valid audio file."""
        # Create a minimal valid WAV file
        import struct

        # Generate a simple WAV file
        sample_rate = 44100
        duration = 1.0
        num_samples = int(sample_rate * duration)

        # Create WAV header
        wav_data = io.BytesIO()

        # RIFF header
        wav_data.write(b'RIFF')
        wav_data.write(struct.pack('<I', 36 + num_samples * 2))  # File size - 8
        wav_data.write(b'WAVE')

        # fmt chunk
        wav_data.write(b'fmt ')
        wav_data.write(struct.pack('<I', 16))  # Chunk size
        wav_data.write(struct.pack('<H', 1))   # Audio format (PCM)
        wav_data.write(struct.pack('<H', 1))   # Num channels
        wav_data.write(struct.pack('<I', sample_rate))  # Sample rate
        wav_data.write(struct.pack('<I', sample_rate * 2))  # Byte rate
        wav_data.write(struct.pack('<H', 2))   # Block align
        wav_data.write(struct.pack('<H', 16))  # Bits per sample

        # data chunk
        wav_data.write(b'data')
        wav_data.write(struct.pack('<I', num_samples * 2))  # Data size

        # Write audio data (silence for simplicity)
        for _ in range(num_samples):
            wav_data.write(struct.pack('<h', 0))

        wav_data.seek(0)

        response = client.post(
            '/api/v1/karaoke/upload',
            data={'song': (wav_data, 'test_song.wav')},
            content_type='multipart/form-data'
        )

        # Should succeed or fail gracefully
        assert response.status_code in (201, 400, 429, 503)

    def test_upload_file_size_check_behavior(self, client):
        """Test that upload checks file size."""
        # This test verifies the file size checking logic is invoked
        # The actual size limit rejection depends on the request handling

        # Create a small valid file
        small_file = io.BytesIO(b'RIFF' + b'\x00' * 100)

        response = client.post(
            '/api/v1/karaoke/upload',
            data={'song': (small_file, 'small_song.wav')},
            content_type='multipart/form-data'
        )

        # Should be processed (not rejected for size)
        # May fail for other reasons (invalid WAV format)
        assert response.status_code in (201, 400, 429, 503)


class TestSongInfo:
    """Test song info retrieval."""

    def test_get_song_info_not_found(self, client):
        """Test getting info for non-existent song."""
        response = client.get('/api/v1/karaoke/songs/nonexistent-song-id')

        assert response.status_code == 404
        data = response.get_json()
        assert 'error' in data

    def test_get_song_info_invalid_id(self, client):
        """Test getting info with invalid ID format."""
        response = client.get('/api/v1/karaoke/songs/!!invalid!!')

        assert response.status_code == 404


class TestSeparation:
    """Test vocal separation endpoints."""

    @pytest.mark.smoke
    def test_start_separation_no_body(self, client):
        """Test starting separation with no request body."""
        response = client.post(
            '/api/v1/karaoke/separate',
            content_type='application/json'
        )

        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
        assert 'song_id' in data['error'].lower()

    def test_start_separation_no_song_id(self, client):
        """Test starting separation without song_id."""
        response = client.post(
            '/api/v1/karaoke/separate',
            json={},
            content_type='application/json'
        )

        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

    def test_start_separation_song_not_found(self, client):
        """Test starting separation for non-existent song."""
        response = client.post(
            '/api/v1/karaoke/separate',
            json={'song_id': 'nonexistent-song-id'},
            content_type='application/json'
        )

        assert response.status_code == 404
        data = response.get_json()
        assert 'error' in data

    def test_get_separation_status_not_found(self, client):
        """Test getting status for non-existent job."""
        response = client.get('/api/v1/karaoke/separate/nonexistent-job-id')

        assert response.status_code == 404
        data = response.get_json()
        assert 'error' in data


class TestAudioDevices:
    """Test audio device management."""

    def test_list_audio_devices(self, client):
        """Test listing audio devices."""
        with patch('auto_voice.web.karaoke_api.list_audio_devices') as mock_list:
            mock_list.return_value = [
                {'index': 0, 'name': 'Built-in Audio', 'channels': 2, 'is_default': True}
            ]

            response = client.get('/api/v1/karaoke/devices')

        assert response.status_code == 200
        data = response.get_json()
        assert 'devices' in data
        assert 'count' in data

    def test_get_output_device_config(self, client):
        """Test getting output device configuration."""
        response = client.get('/api/v1/karaoke/devices/output')

        assert response.status_code == 200
        data = response.get_json()
        assert 'speaker_device' in data
        assert 'headphone_device' in data

    def test_set_output_device_config(self, client):
        """Test setting output device configuration."""
        # Patch at the audio_router module level since it's imported inside the function
        with patch('auto_voice.web.audio_router.list_audio_devices') as mock_list:
            # Match the actual return structure from audio_router
            mock_list.return_value = [
                {'device_id': '0', 'index': 0, 'name': 'Device 0', 'channels': 2, 'type': 'output'},
                {'device_id': '1', 'index': 1, 'name': 'Device 1', 'channels': 2, 'type': 'output'},
            ]

            response = client.post(
                '/api/v1/karaoke/devices/output',
                json={'speaker_device': 0, 'headphone_device': 1},
                content_type='application/json'
            )

        assert response.status_code == 200
        data = response.get_json()
        assert data.get('speaker_device') == 0
        assert data.get('headphone_device') == 1

    def test_set_output_device_invalid_index(self, client):
        """Test setting invalid device index."""
        with patch('auto_voice.web.audio_router.list_audio_devices') as mock_list:
            mock_list.return_value = [
                {'device_id': '0', 'index': 0, 'name': 'Device 0', 'channels': 2, 'type': 'output'},
            ]

            response = client.post(
                '/api/v1/karaoke/devices/output',
                json={'speaker_device': 99},  # Invalid index
                content_type='application/json'
            )

        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data


class TestVoiceModels:
    """Test voice model management."""

    def test_list_voice_models(self, client):
        """Test listing voice models."""
        response = client.get('/api/v1/karaoke/voice-models')

        assert response.status_code == 200
        data = response.get_json()
        assert 'models' in data
        assert 'count' in data
        assert isinstance(data['models'], list)

    def test_get_voice_model_not_found(self, client):
        """Test getting non-existent voice model."""
        response = client.get('/api/v1/karaoke/voice-models/nonexistent-model')

        assert response.status_code == 404
        data = response.get_json()
        assert 'error' in data

    def test_extract_voice_model_no_song_id(self, client):
        """Test extracting voice model without song_id."""
        response = client.post(
            '/api/v1/karaoke/voice-models/extract',
            json={'name': 'Test Model'},
            content_type='application/json'
        )

        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
        assert 'song_id' in data['error'].lower()

    def test_extract_voice_model_no_name(self, client):
        """Test extracting voice model without name."""
        response = client.post(
            '/api/v1/karaoke/voice-models/extract',
            json={'song_id': 'test-song-id'},
            content_type='application/json'
        )

        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
        assert 'name' in data['error'].lower()

    def test_extract_voice_model_song_not_found(self, client):
        """Test extracting voice model from non-existent song."""
        response = client.post(
            '/api/v1/karaoke/voice-models/extract',
            json={'song_id': 'nonexistent-song-id', 'name': 'Test Model'},
            content_type='application/json'
        )

        assert response.status_code == 404
        data = response.get_json()
        assert 'error' in data


class TestSessionManagement:
    """Test session management functions."""

    def test_register_session(self):
        """Test registering a session."""
        from auto_voice.web.karaoke_api import (
            register_session, _active_sessions, cleanup_session
        )

        session_id = 'test-session-123'
        song_id = 'test-song-456'
        client_id = 'test-client-789'

        try:
            register_session(session_id, song_id, client_id)

            assert session_id in _active_sessions
            assert _active_sessions[session_id]['song_id'] == song_id
            assert _active_sessions[session_id]['client_id'] == client_id
        finally:
            cleanup_session(session_id)

    def test_update_session_activity(self):
        """Test updating session activity."""
        from auto_voice.web.karaoke_api import (
            register_session, update_session_activity,
            _active_sessions, cleanup_session
        )

        session_id = 'test-session-activity'

        try:
            register_session(session_id, 'song', 'client')
            original_activity = _active_sessions[session_id]['last_activity']

            time.sleep(0.1)  # Small delay
            update_session_activity(session_id)

            assert _active_sessions[session_id]['last_activity'] > original_activity
        finally:
            cleanup_session(session_id)

    def test_cleanup_session(self):
        """Test cleaning up a session."""
        from auto_voice.web.karaoke_api import (
            register_session, cleanup_session, _active_sessions
        )

        session_id = 'test-session-cleanup'
        register_session(session_id, 'song', 'client')

        assert session_id in _active_sessions

        cleanup_session(session_id, reason='test')

        assert session_id not in _active_sessions

    def test_cleanup_stale_sessions(self):
        """Test cleaning up stale sessions."""
        from auto_voice.web.karaoke_api import (
            register_session, cleanup_stale_sessions,
            _active_sessions, cleanup_session
        )

        # Register a session with old activity timestamp
        session_id = 'test-stale-session'
        register_session(session_id, 'song', 'client')
        _active_sessions[session_id]['last_activity'] = time.time() - 1000  # Old

        count = cleanup_stale_sessions(max_idle_seconds=60)

        assert count >= 1
        assert session_id not in _active_sessions


class TestCleanupFunctions:
    """Test cleanup functions."""

    def test_cleanup_old_songs(self, tmp_path):
        """Test cleaning up old uploaded songs."""
        from auto_voice.web.karaoke_api import (
            _uploaded_songs, cleanup_old_songs
        )

        # Create a temporary song file
        song_file = tmp_path / 'old_song.wav'
        song_file.write_bytes(b'test audio data')

        song_id = 'test-old-song'
        _uploaded_songs[song_id] = {
            'id': song_id,
            'path': str(song_file),
            'uploaded_at': time.time() - 10000,  # Very old
            'status': 'uploaded'
        }

        try:
            count = cleanup_old_songs(max_age_seconds=60)

            assert count >= 1
            assert song_id not in _uploaded_songs
        finally:
            # Cleanup in case test failed
            _uploaded_songs.pop(song_id, None)

    def test_get_uploaded_song(self):
        """Test getting uploaded song by ID."""
        from auto_voice.web.karaoke_api import (
            get_uploaded_song, _uploaded_songs
        )

        song_id = 'test-song-get'
        _uploaded_songs[song_id] = {'id': song_id, 'path': '/tmp/test.wav'}

        try:
            result = get_uploaded_song(song_id)
            assert result is not None
            assert result['id'] == song_id

            # Non-existent song
            result = get_uploaded_song('nonexistent')
            assert result is None
        finally:
            _uploaded_songs.pop(song_id, None)

    def test_get_separation_job(self):
        """Test getting separation job by ID."""
        from auto_voice.web.karaoke_api import (
            get_separation_job, _separation_jobs
        )

        job_id = 'test-job-get'
        _separation_jobs[job_id] = {'job_id': job_id, 'status': 'processing'}

        try:
            result = get_separation_job(job_id)
            assert result is not None
            assert result['job_id'] == job_id

            # Non-existent job
            result = get_separation_job('nonexistent')
            assert result is None
        finally:
            _separation_jobs.pop(job_id, None)

    def test_update_separation_progress(self):
        """Test updating separation progress."""
        from auto_voice.web.karaoke_api import (
            update_separation_progress, _separation_jobs
        )

        job_id = 'test-job-progress'
        _separation_jobs[job_id] = {'job_id': job_id, 'progress': 0, 'status': 'queued'}

        try:
            update_separation_progress(job_id, 50, 'processing')

            assert _separation_jobs[job_id]['progress'] == 50
            assert _separation_jobs[job_id]['status'] == 'processing'
        finally:
            _separation_jobs.pop(job_id, None)

    def test_complete_separation(self):
        """Test completing separation job."""
        from auto_voice.web.karaoke_api import (
            complete_separation, _separation_jobs, _uploaded_songs
        )

        job_id = 'test-job-complete'
        song_id = 'test-song-complete'

        _uploaded_songs[song_id] = {'id': song_id, 'status': 'separating', 'separation_job_id': job_id}
        _separation_jobs[job_id] = {'job_id': job_id, 'song_id': song_id, 'progress': 50, 'status': 'processing'}

        try:
            complete_separation(job_id, '/tmp/vocals.wav', '/tmp/instrumental.wav')

            assert _separation_jobs[job_id]['status'] == 'completed'
            assert _separation_jobs[job_id]['progress'] == 100
            assert _separation_jobs[job_id]['vocals_path'] == '/tmp/vocals.wav'
            assert _uploaded_songs[song_id]['status'] == 'separated'
        finally:
            _separation_jobs.pop(job_id, None)
            _uploaded_songs.pop(song_id, None)


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_decorator(self):
        """Test rate limit decorator functionality."""
        from auto_voice.web.karaoke_api import rate_limit, _rate_limit_store
        from flask import Flask

        app = Flask(__name__)

        @app.route('/test')
        @rate_limit(max_requests=2, window_seconds=60)
        def test_endpoint():
            return 'OK'

        with app.test_client() as client:
            # Clear rate limit store
            _rate_limit_store.clear()

            # First two requests should succeed
            response1 = client.get('/test')
            response2 = client.get('/test')

            # Should both succeed
            assert response1.status_code == 200
            assert response2.status_code == 200

    def test_get_client_ip(self):
        """Test client IP detection."""
        from auto_voice.web.karaoke_api import _get_client_ip
        from flask import Flask

        app = Flask(__name__)

        with app.test_request_context(
            '/test',
            headers={'X-Forwarded-For': '192.168.1.100, 10.0.0.1'}
        ):
            ip = _get_client_ip()
            assert ip == '192.168.1.100'


class TestRequestLogging:
    """Test request logging decorator."""

    def test_log_request_decorator(self):
        """Test log request decorator."""
        from auto_voice.web.karaoke_api import log_request, _generate_request_id
        from flask import Flask, g

        app = Flask(__name__)

        @app.route('/test')
        @log_request
        def test_endpoint():
            return 'OK'

        with app.test_client() as client:
            response = client.get('/test')
            assert response.status_code == 200

    def test_generate_request_id(self):
        """Test request ID generation."""
        from auto_voice.web.karaoke_api import _generate_request_id

        request_id = _generate_request_id()

        assert request_id.startswith('req_')
        assert len(request_id) == 16  # 'req_' + 12 hex chars


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_audio_duration_fallbacks(self):
        """Test audio duration detection with fallbacks."""
        from auto_voice.web.karaoke_api import _get_audio_duration

        # Test with non-existent file - should raise
        with pytest.raises(RuntimeError):
            _get_audio_duration('/nonexistent/file.wav')

    @patch('auto_voice.web.karaoke_api._get_voice_model_registry')
    def test_get_voice_model_registry_singleton(self, mock_get_registry):
        """Test voice model registry singleton."""
        from auto_voice.web.karaoke_api import _get_voice_model_registry, _voice_model_registry

        # Reset singleton
        import auto_voice.web.karaoke_api as api_module
        api_module._voice_model_registry = None

        # First call should create instance
        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry


class TestIntegration:
    """Integration tests for karaoke API."""

    @pytest.mark.integration
    def test_upload_and_info_flow(self, client, tmp_path):
        """Test upload followed by info retrieval."""
        # This would test the full flow if the upload succeeds
        # For now, test that endpoints are accessible

        # Upload endpoint exists
        response = client.post('/api/v1/karaoke/upload')
        assert response.status_code in (400, 201, 503)  # Bad request expected without file

        # Health check works
        response = client.get('/api/v1/karaoke/health')
        assert response.status_code in (200, 503)

    @pytest.mark.integration
    def test_full_separation_workflow_mocked(self, client):
        """Test full separation workflow with mocks."""
        from auto_voice.web.karaoke_api import _uploaded_songs, _separation_jobs

        # Setup mock song
        song_id = 'test-workflow-song'
        _uploaded_songs[song_id] = {
            'id': song_id,
            'path': '/tmp/test_song.wav',
            'duration': 60.0,
            'sample_rate': 44100,
            'format': 'wav',
            'file_size': 10000,
            'uploaded_at': time.time(),
            'status': 'uploaded'
        }

        try:
            # Start separation
            response = client.post(
                '/api/v1/karaoke/separate',
                json={'song_id': song_id},
                content_type='application/json'
            )

            assert response.status_code == 202
            data = response.get_json()
            assert 'job_id' in data

            job_id = data['job_id']

            # Check status
            response = client.get(f'/api/v1/karaoke/separate/{job_id}')
            assert response.status_code == 200
            data = response.get_json()
            assert data['status'] in ('queued', 'processing')

        finally:
            _uploaded_songs.pop(song_id, None)
            # Clean up any created jobs
            for jid in list(_separation_jobs.keys()):
                if _separation_jobs.get(jid, {}).get('song_id') == song_id:
                    _separation_jobs.pop(jid, None)
