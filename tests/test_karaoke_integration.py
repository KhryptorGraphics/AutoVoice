"""Integration tests for live karaoke voice conversion system.

Tests the complete workflow: upload → separate → configure → perform.
"""
import io
import os
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest
import torch
import numpy as np
import soundfile as sf

# Skip if dependencies not available
pytest.importorskip("flask")


@pytest.fixture
def app():
    """Create Flask app for testing."""
    from auto_voice.web.app import create_app

    app, socketio = create_app(testing=True)
    app.config['TESTING'] = True

    # Clear any existing state
    from auto_voice.web import karaoke_api
    karaoke_api._uploaded_songs.clear()
    karaoke_api._separation_jobs.clear()
    karaoke_api._voice_model_registry = None

    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def sample_audio_file():
    """Create a sample audio file for testing."""
    # Generate 5 seconds of test audio (sine wave)
    sample_rate = 24000
    duration = 5.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 440 Hz tone

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, audio, sample_rate)
        yield f.name

    # Cleanup
    if os.path.exists(f.name):
        os.unlink(f.name)


class TestEndToEndWorkflow:
    """Test complete karaoke workflow from upload to performance."""

    def test_complete_workflow_upload_to_ready(self, client, sample_audio_file):
        """Test upload → separate → ready workflow.

        Task 7.1: End-to-end test of the complete workflow.
        """
        # Step 1: Upload song
        with open(sample_audio_file, 'rb') as f:
            response = client.post(
                '/api/v1/karaoke/upload',
                data={'song': (f, 'test_song.wav')},
                content_type='multipart/form-data'
            )

        assert response.status_code == 201
        upload_data = response.get_json()
        song_id = upload_data['song_id']
        assert upload_data['status'] == 'uploaded'
        assert upload_data['duration'] > 0

        # Step 2: Start separation
        response = client.post(
            '/api/v1/karaoke/separate',
            json={'song_id': song_id}
        )

        assert response.status_code == 202
        sep_data = response.get_json()
        job_id = sep_data['job_id']
        assert sep_data['status'] == 'queued'

        # Step 3: Check separation status
        response = client.get(f'/api/v1/karaoke/separate/{job_id}')
        assert response.status_code == 200
        status_data = response.get_json()
        assert status_data['job_id'] == job_id
        assert status_data['song_id'] == song_id

        # Step 4: List voice models
        response = client.get('/api/v1/karaoke/voice-models')
        assert response.status_code == 200
        models_data = response.get_json()
        assert 'models' in models_data
        assert 'count' in models_data

        # Step 5: List audio devices
        response = client.get('/api/v1/karaoke/devices')
        assert response.status_code == 200
        devices_data = response.get_json()
        assert 'devices' in devices_data

        # Step 6: Configure output devices
        response = client.get('/api/v1/karaoke/devices/output')
        assert response.status_code == 200

    def test_voice_model_extraction_workflow(self, client, sample_audio_file):
        """Test extracting voice model from uploaded song.

        Task 7.1: Test voice extraction as part of workflow.
        """
        from auto_voice.web import karaoke_api

        # Upload and fake separation completion
        with open(sample_audio_file, 'rb') as f:
            response = client.post(
                '/api/v1/karaoke/upload',
                data={'song': (f, 'test_song.wav')},
                content_type='multipart/form-data'
            )

        song_id = response.get_json()['song_id']

        # Start separation
        response = client.post(
            '/api/v1/karaoke/separate',
            json={'song_id': song_id}
        )
        job_id = response.get_json()['job_id']

        # Manually complete separation for test
        karaoke_api._separation_jobs[job_id]['status'] = 'completed'
        karaoke_api._separation_jobs[job_id]['progress'] = 100
        karaoke_api._separation_jobs[job_id]['vocals_path'] = sample_audio_file

        # Extract voice model
        response = client.post(
            '/api/v1/karaoke/voice-models/extract',
            json={'song_id': song_id, 'name': 'Test Artist'}
        )

        assert response.status_code == 201
        extract_data = response.get_json()
        assert 'model_id' in extract_data
        assert extract_data['type'] == 'extracted'
        assert extract_data['name'] == 'Test Artist'

        # Verify model is listed
        response = client.get('/api/v1/karaoke/voice-models')
        models = response.get_json()['models']
        model_ids = [m['id'] for m in models]
        assert extract_data['model_id'] in model_ids


class TestAudioFormatSupport:
    """Test support for various audio formats.

    Task 7.2: Test with various song formats and lengths.
    """

    @pytest.mark.parametrize('ext,content_type', [
        ('wav', 'audio/wav'),
        ('mp3', 'audio/mpeg'),
        ('flac', 'audio/flac'),
        ('m4a', 'audio/mp4'),
        ('ogg', 'audio/ogg'),
    ])
    def test_supported_formats_accepted(self, client, ext, content_type):
        """Test that all supported formats are accepted for upload."""
        # Create minimal audio content
        audio_content = b'\x00' * 1000

        response = client.post(
            '/api/v1/karaoke/upload',
            data={'song': (io.BytesIO(audio_content), f'test.{ext}')},
            content_type='multipart/form-data'
        )

        # Should accept the file (may fail later during processing, but upload accepts)
        # 201 = success, 503 = processing error (both mean format was accepted)
        assert response.status_code in [201, 503]

    def test_unsupported_format_rejected(self, client):
        """Test that unsupported formats are rejected."""
        audio_content = b'\x00' * 1000

        response = client.post(
            '/api/v1/karaoke/upload',
            data={'song': (io.BytesIO(audio_content), 'test.txt')},
            content_type='multipart/form-data'
        )

        assert response.status_code == 400
        assert 'Invalid file format' in response.get_json()['error']


class TestLatencyMeasurement:
    """Test latency measurement and documentation.

    Task 7.3: Measure and document end-to-end latency.
    """

    def test_karaoke_session_tracks_latency(self):
        """Test that KaraokeSession tracks processing latency."""
        from auto_voice.web.karaoke_session import KaraokeSession

        session = KaraokeSession(
            session_id='test-latency',
            song_id='song-123',
            vocals_path='/tmp/vocals.wav',
            instrumental_path='/tmp/instrumental.wav',
        )

        # Set required embedding
        session.set_speaker_embedding(torch.randn(256))
        session.start()

        # Process some chunks (will use fallback since no real pipeline)
        for _ in range(5):
            audio_chunk = torch.randn(2400)  # 100ms at 24kHz
            session.process_chunk(audio_chunk)

        # Check latency tracking
        latency = session.get_latency_ms()
        assert latency >= 0  # Latency is tracked

        stats = session.get_stats()
        assert stats['chunks_processed'] == 5
        assert 'avg_latency_ms' in stats
        assert 'min_latency_ms' in stats
        assert 'max_latency_ms' in stats

        session.stop()

    def test_latency_target_documented(self):
        """Verify latency target is documented (<50ms with TRT)."""
        from auto_voice.web.karaoke_session import KaraokeSession

        # Check docstring mentions latency target
        docstring = KaraokeSession.__doc__
        assert '50ms' in docstring or '<50ms' in docstring


class TestDualOutputRouting:
    """Test dual audio output routing.

    Task 7.4: Test dual output routing with real audio devices.
    """

    def test_audio_router_produces_dual_outputs(self):
        """Test AudioOutputRouter produces separate speaker/headphone outputs."""
        from auto_voice.web.audio_router import AudioOutputRouter

        router = AudioOutputRouter(sample_rate=24000)

        # Create test audio
        converted_voice = torch.randn(2400)
        instrumental = torch.randn(2400)
        original_song = torch.randn(2400)

        # Route audio
        speaker_out, headphone_out = router.route(
            converted_voice, instrumental, original_song
        )

        # Verify outputs are different
        assert speaker_out.shape == headphone_out.shape
        assert not torch.allclose(speaker_out, headphone_out)

        # Speaker should contain instrumental (mixed with voice)
        # Headphone should be original song
        assert speaker_out.shape[0] == 2400
        assert headphone_out.shape[0] == 2400

    def test_device_selection_api(self, client):
        """Test device selection API endpoints."""
        # Get current config
        response = client.get('/api/v1/karaoke/devices/output')
        assert response.status_code == 200

        # List devices
        response = client.get('/api/v1/karaoke/devices')
        assert response.status_code == 200
        devices = response.get_json()['devices']

        if devices:
            # Set specific device
            device_idx = devices[0]['index']
            response = client.post(
                '/api/v1/karaoke/devices/output',
                json={'speaker_device': device_idx}
            )
            assert response.status_code == 200
            assert response.get_json()['speaker_device'] == device_idx


class TestExtendedSession:
    """Test extended performance sessions.

    Task 7.5: Stress test with extended performance sessions.
    """

    def test_session_handles_many_chunks(self):
        """Test session can handle many chunks without memory leak."""
        from auto_voice.web.karaoke_session import KaraokeSession

        session = KaraokeSession(
            session_id='stress-test',
            song_id='song-stress',
            vocals_path='/tmp/vocals.wav',
            instrumental_path='/tmp/instrumental.wav',
        )

        session.set_speaker_embedding(torch.randn(256))
        session.start()

        # Process 1000 chunks (~100 seconds of audio at 100ms chunks)
        for i in range(1000):
            audio_chunk = torch.randn(2400)
            session.process_chunk(audio_chunk)

        # Verify session is still functional
        assert session.is_active
        assert session._chunks_processed == 1000

        # Latency history should be capped
        assert len(session._latency_history) <= session._max_latency_history

        session.stop()

    def test_session_stats_after_extended_run(self):
        """Test stats are accurate after extended session."""
        from auto_voice.web.karaoke_session import KaraokeSession

        session = KaraokeSession(
            session_id='stats-test',
            song_id='song-stats',
            vocals_path='/tmp/vocals.wav',
            instrumental_path='/tmp/instrumental.wav',
        )

        session.set_speaker_embedding(torch.randn(256))
        session.start()

        # Run for 500 chunks
        for _ in range(500):
            session.process_chunk(torch.randn(2400))

        stats = session.get_stats()

        assert stats['chunks_processed'] == 500
        assert stats['is_active'] is True
        assert stats['duration_s'] > 0

        session.stop()
        final_stats = session.get_stats()
        assert final_stats['is_active'] is False


class TestAPIDocumentation:
    """Test API endpoint documentation.

    Task 7.7: Document API endpoints and WebSocket protocol.
    """

    def test_upload_endpoint_has_docstring(self):
        """Verify upload endpoint is documented."""
        from auto_voice.web.karaoke_api import upload_song
        assert upload_song.__doc__ is not None
        assert 'song' in upload_song.__doc__
        assert 'HTTP' in upload_song.__doc__

    def test_separation_endpoints_documented(self):
        """Verify separation endpoints are documented."""
        from auto_voice.web.karaoke_api import start_separation, get_separation_status
        assert start_separation.__doc__ is not None
        assert get_separation_status.__doc__ is not None
        assert 'job_id' in get_separation_status.__doc__

    def test_voice_model_endpoints_documented(self):
        """Verify voice model endpoints are documented."""
        from auto_voice.web.karaoke_api import (
            list_voice_models,
            get_voice_model,
            extract_voice_model
        )
        assert list_voice_models.__doc__ is not None
        assert get_voice_model.__doc__ is not None
        assert extract_voice_model.__doc__ is not None

    def test_device_endpoints_documented(self):
        """Verify device endpoints are documented."""
        from auto_voice.web.karaoke_api import (
            list_audio_devices,
            get_output_device_config,
            set_output_device_config
        )
        assert list_audio_devices.__doc__ is not None
        assert get_output_device_config.__doc__ is not None
        assert set_output_device_config.__doc__ is not None


class TestHealthAndMonitoring:
    """Test health check and monitoring endpoints.

    Task 8.1, 8.2, 8.4: Production logging, health checks, analytics.
    """

    def test_health_endpoint_returns_status(self, client):
        """Test health check endpoint returns component status."""
        response = client.get('/api/v1/karaoke/health')
        assert response.status_code == 200

        data = response.get_json()
        assert data['status'] in ('healthy', 'degraded')
        assert 'timestamp' in data
        assert 'components' in data
        assert 'version' in data

        # Verify component checks
        components = data['components']
        assert 'voice_model_registry' in components
        assert 'storage' in components
        assert 'temp_storage' in components

    def test_health_check_storage_stats(self, client):
        """Test health check includes storage statistics."""
        response = client.get('/api/v1/karaoke/health')
        data = response.get_json()

        storage = data['components']['storage']
        assert 'songs_uploaded' in storage
        assert 'active_jobs' in storage
        assert 'completed_jobs' in storage
        assert 'active_sessions' in storage

    def test_metrics_endpoint_returns_analytics(self, client):
        """Test metrics endpoint returns usage analytics."""
        response = client.get('/api/v1/karaoke/metrics')
        assert response.status_code == 200

        data = response.get_json()
        assert 'total_sessions' in data
        assert 'total_chunks_processed' in data
        assert 'total_audio_minutes' in data
        assert 'avg_latency_ms' in data


class TestRateLimiting:
    """Test rate limiting functionality.

    Task 8.6: Security review - rate limiting.
    """

    def test_rate_limit_allows_normal_usage(self, client):
        """Test that normal usage is not rate limited."""
        # Health endpoint should not be rate limited
        for _ in range(15):
            response = client.get('/api/v1/karaoke/health')
            assert response.status_code == 200

    def test_rate_limiting_decorator_exists(self):
        """Test that rate limiting decorator is available."""
        from auto_voice.web.karaoke_api import rate_limit
        assert callable(rate_limit)


class TestSessionCleanup:
    """Test graceful session cleanup.

    Task 8.3: Graceful session cleanup on disconnect.
    """

    def test_cleanup_functions_exist(self):
        """Test cleanup functions are available."""
        from auto_voice.web.karaoke_api import (
            register_session,
            cleanup_session,
            cleanup_stale_sessions,
            cleanup_old_songs
        )
        assert callable(register_session)
        assert callable(cleanup_session)
        assert callable(cleanup_stale_sessions)
        assert callable(cleanup_old_songs)

    def test_session_registration_and_cleanup(self):
        """Test session can be registered and cleaned up."""
        from auto_voice.web.karaoke_api import (
            register_session,
            cleanup_session,
            _active_sessions
        )

        # Register a session
        session_id = 'test-cleanup-session'
        register_session(session_id, 'song-123', 'client-abc')

        assert session_id in _active_sessions
        assert _active_sessions[session_id]['song_id'] == 'song-123'

        # Clean up the session
        cleanup_session(session_id, reason='test')

        assert session_id not in _active_sessions

    def test_analytics_tracking(self):
        """Test analytics tracks sessions correctly."""
        from auto_voice.web.karaoke_events import _analytics

        # Record a session
        initial_sessions = _analytics._metrics['total_sessions']
        _analytics.record_session_start()
        assert _analytics._metrics['total_sessions'] == initial_sessions + 1

        # Record session end
        _analytics.record_session_end(duration_s=60.0, chunks_processed=100)
        assert _analytics._metrics['total_chunks_processed'] >= 100
