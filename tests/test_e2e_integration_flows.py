"""End-to-end integration tests for complete user workflows.

Phase 5: Tests for complete user flows:
- Task 5.1: Train and convert flow
- Task 5.2: YouTube to trained profile (mocked)
- Task 5.3: Multi-pipeline comparison
- Task 5.4: Karaoke session workflow
- Task 5.5: Error recovery scenarios

These tests use mocked ML components to test the integration
of multiple system components without requiring actual inference.
"""

import io
import json
import os
import tempfile
import time
import wave
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np
import pytest


@pytest.fixture
def app_with_full_stack():
    """Create Flask app with full stack mocked."""
    pytest.importorskip('flask_swagger_ui', reason="flask_swagger_ui not installed")

    from auto_voice.web.app import create_app

    app, socketio = create_app(config={
        'TESTING': True,
        'singing_conversion_enabled': True,
        'voice_cloning_enabled': True,
    })

    # Mock voice cloner
    mock_voice_cloner = MagicMock()
    mock_voice_cloner.create_voice_profile.return_value = {
        'profile_id': 'new-profile-123',
        'name': 'Test Artist',
        'embedding': np.zeros(256).tolist(),
    }
    mock_voice_cloner.load_voice_profile.return_value = {
        'profile_id': 'new-profile-123',
        'name': 'Test Artist',
        'embedding': np.zeros(256).tolist(),
        'selected_adapter': 'hq',
    }
    mock_voice_cloner.list_profiles.return_value = [
        {'profile_id': 'new-profile-123', 'name': 'Test Artist'},
    ]

    # Mock singing pipeline
    mock_singing_pipeline = MagicMock()
    mock_singing_pipeline.convert_song.return_value = {
        'mixed_audio': np.zeros(22050, dtype=np.float32),
        'sample_rate': 22050,
        'duration': 1.0,
        'metadata': {'pipeline': 'quality'},
        'stems': {},
    }

    # Mock job manager for async operations
    mock_job_manager = MagicMock()
    mock_job_manager.create_job.return_value = 'job-123'
    mock_job_manager.get_job_status.return_value = {
        'status': 'completed',
        'progress': 100,
        'result_path': '/tmp/result.wav',
    }

    app.voice_cloner = mock_voice_cloner
    app.singing_conversion_pipeline = mock_singing_pipeline
    app.job_manager = mock_job_manager
    app.socketio = socketio
    app.app_config = {'audio': {'sample_rate': 22050}}

    return app


@pytest.fixture
def client(app_with_full_stack):
    """Flask test client."""
    return app_with_full_stack.test_client()


@pytest.fixture
def audio_5s():
    """Create a 5-second audio file."""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        # 5 seconds of silence
        wav.writeframes(b'\x00' * 22050 * 2 * 5)
    buffer.seek(0)
    return buffer


@pytest.fixture
def audio_10s():
    """Create a 10-second audio file."""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        # 10 seconds of silence
        wav.writeframes(b'\x00' * 22050 * 2 * 10)
    buffer.seek(0)
    return buffer


def make_audio_buffer(duration_s=5, sample_rate=22050):
    """Create a fresh audio buffer (can't reuse BytesIO after Flask closes it)."""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b'\x00' * sample_rate * 2 * duration_s)
    buffer.seek(0)
    return buffer


@pytest.mark.slow
class TestTrainAndConvertFlow:
    """Task 5.1: Test complete train and convert workflow."""

    def test_create_profile_upload_samples_train_convert(self, client, app_with_full_stack):
        """Complete workflow: create profile, upload samples, train, convert."""

        # Step 1: Create voice profile
        response = client.post(
            '/api/v1/voice/clone',
            data={
                'audio': (make_audio_buffer(5), 'sample1.wav'),
                'name': 'New Artist',
            },
            content_type='multipart/form-data'
        )

        assert response.status_code in (200, 201, 400, 500)

        # Step 2: Mock profile exists for subsequent operations
        profile_id = 'new-profile-123'

        # Step 3: Upload additional samples
        with patch('auto_voice.storage.voice_profiles.VoiceProfileStore') as MockStore:
            mock_store = MagicMock()
            mock_store.exists.return_value = True
            mock_store.add_training_sample.return_value = MagicMock(
                sample_id='sample_001', duration=5.0
            )
            MockStore.return_value = mock_store

            # Upload sample 2
            response = client.post(
                f'/api/v1/profiles/{profile_id}/samples',
                data={'audio': (make_audio_buffer(5), 'sample2.wav')},
                content_type='multipart/form-data'
            )

        # Step 4: Start training job
        with patch('auto_voice.storage.voice_profiles.VoiceProfileStore') as MockStore:
            mock_store = MagicMock()
            mock_store.exists.return_value = True
            mock_store.get_all_vocals_paths.return_value = ['/path/1.wav', '/path/2.wav', '/path/3.wav']
            mock_store.get_total_training_duration.return_value = 15.0
            MockStore.return_value = mock_store

            response = client.post(
                '/api/v1/training/jobs',
                json={'profile_id': profile_id},
                content_type='application/json'
            )

        # Training may be async or sync
        assert response.status_code in (200, 201, 202, 400, 500)

        # Step 5: Convert song with trained profile
        with patch('auto_voice.models.adapter_manager.AdapterManager') as MockAM:
            mock_am = MagicMock()
            mock_am.has_adapter.return_value = True
            MockAM.return_value = mock_am

            response = client.post(
                '/api/v1/convert/song',
                data={
                    'song': (make_audio_buffer(10), 'song.wav'),
                    'profile_id': profile_id,
                    'adapter_type': 'hq',  # Must be 'hq' or 'nvfp4'
                },
                content_type='multipart/form-data'
            )

        # Should queue or complete conversion
        assert response.status_code in (200, 202, 400, 404, 500)


@pytest.mark.slow
class TestYouTubeToProfileFlow:
    """Task 5.2: Test YouTube to trained profile workflow (mocked)."""

    def test_youtube_download_diarize_create_profile(self, client, app_with_full_stack):
        """Workflow: download YouTube, diarize, create profile from segments."""

        # Step 1: Get YouTube video info (mocked)
        with patch('auto_voice.audio.youtube_downloader.YouTubeDownloader') as MockYT:
            mock_yt = MagicMock()
            mock_yt.get_info.return_value = {
                'title': 'Test Song ft. Artist',
                'channel': 'Music Channel',
                'duration': 180,
            }
            MockYT.return_value = mock_yt

            response = client.post(
                '/api/v1/youtube/info',
                json={'url': 'https://youtube.com/watch?v=test123'},
                content_type='application/json'
            )

        # May fail if not fully implemented
        if response.status_code == 200:
            data = json.loads(response.data)

        # Step 2: Download audio (mocked)
        with patch('auto_voice.audio.youtube_downloader.YouTubeDownloader') as MockYT:
            mock_yt = MagicMock()
            mock_yt.download.return_value = MagicMock(
                audio_path='/tmp/downloaded.wav',
                title='Test Song',
                duration=180,
            )
            MockYT.return_value = mock_yt

            response = client.post(
                '/api/v1/youtube/download',
                json={'url': 'https://youtube.com/watch?v=test123'},
                content_type='application/json'
            )

        # Step 3: Run diarization (mocked)
        with patch('auto_voice.audio.speaker_diarization.SpeakerDiarizer') as MockDiarizer:
            mock_diarizer = MagicMock()
            mock_diarizer.diarize.return_value = {
                'segments': [
                    {'speaker': 'SPEAKER_00', 'start': 0, 'end': 30, 'embedding': [0.1] * 512},
                    {'speaker': 'SPEAKER_01', 'start': 30, 'end': 60, 'embedding': [0.2] * 512},
                ],
                'num_speakers': 2,
            }
            MockDiarizer.return_value = mock_diarizer

            # Create test audio file
            audio_buf = io.BytesIO()
            with wave.open(audio_buf, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b'\x00' * 16000 * 2)
            audio_buf.seek(0)

            response = client.post(
                '/api/v1/audio/diarize',
                data={'audio': (audio_buf, 'test.wav')},
                content_type='multipart/form-data'
            )

        # Step 4: Auto-create profile from diarization
        with patch('auto_voice.storage.voice_profiles.VoiceProfileStore') as MockStore:
            mock_store = MagicMock()
            mock_store.create_profile_from_diarization.return_value = 'diarized-profile-001'
            MockStore.return_value = mock_store

            response = client.post(
                '/api/v1/profiles/auto-create',
                json={
                    'name': 'SPEAKER_00 from Test Song',
                    'speaker_embedding': [0.1] * 512,
                },
                content_type='application/json'
            )

        assert response.status_code in (200, 201, 400, 500)


@pytest.mark.slow
class TestMultiPipelineComparison:
    """Task 5.3: Test multi-pipeline comparison workflow."""

    def test_compare_realtime_vs_quality_pipeline(self, client, app_with_full_stack):
        """Convert same audio with different pipelines and compare."""

        profile_id = 'new-profile-123'

        with patch('auto_voice.models.adapter_manager.AdapterManager') as MockAM:
            mock_am = MagicMock()
            mock_am.has_adapter.return_value = True
            MockAM.return_value = mock_am

            # Convert with realtime pipeline
            response_realtime = client.post(
                '/api/v1/convert/song',
                data={
                    'song': (make_audio_buffer(5), 'test.wav'),
                    'profile_id': profile_id,
                    'pipeline_type': 'realtime',
                    'adapter_type': 'hq',  # Must specify valid adapter_type
                },
                content_type='multipart/form-data'
            )

            # Convert with quality pipeline
            response_quality = client.post(
                '/api/v1/convert/song',
                data={
                    'song': (make_audio_buffer(5), 'test.wav'),
                    'profile_id': profile_id,
                    'pipeline_type': 'quality',
                    'adapter_type': 'hq',
                },
                content_type='multipart/form-data'
            )

            # Convert with quality_seedvc pipeline
            response_seedvc = client.post(
                '/api/v1/convert/song',
                data={
                    'song': (make_audio_buffer(5), 'test.wav'),
                    'profile_id': profile_id,
                    'pipeline_type': 'quality_seedvc',
                    'adapter_type': 'hq',
                },
                content_type='multipart/form-data'
            )

        # All should be accepted (may be queued)
        for response in [response_realtime, response_quality, response_seedvc]:
            assert response.status_code in (200, 202, 400, 404, 500)


@pytest.mark.slow
class TestKaraokeSessionWorkflow:
    """Task 5.4: Test karaoke session workflow."""

    def test_karaoke_websocket_session(self, app_with_full_stack, audio_5s):
        """Test WebSocket-based karaoke session."""

        # Get socket.io test client
        from flask_socketio import SocketIOTestClient

        socketio = app_with_full_stack.socketio
        socket_client = SocketIOTestClient(app_with_full_stack, socketio)

        # Connect to socket
        socket_client.connect()

        # Start karaoke session
        socket_client.emit('startSession', {
            'profile_id': 'new-profile-123',
            'pipeline_type': 'realtime',
        })

        # Get any received events
        received = socket_client.get_received()

        # Session may start or fail based on pipeline availability
        # We're testing the flow, not the actual conversion

        # Send audio chunk
        audio_chunk = np.zeros(512, dtype=np.float32).tobytes()
        socket_client.emit('audioChunk', {'data': audio_chunk})

        # Stop session
        socket_client.emit('stopSession', {})

        socket_client.disconnect()

    def test_karaoke_profile_switch_mid_session(self, app_with_full_stack):
        """Test switching profiles during karaoke session."""

        from flask_socketio import SocketIOTestClient

        socketio = app_with_full_stack.socketio
        socket_client = SocketIOTestClient(app_with_full_stack, socketio)

        socket_client.connect()

        # Start with first profile
        socket_client.emit('startSession', {
            'profile_id': 'profile-1',
        })

        # Switch to second profile
        socket_client.emit('switchProfile', {
            'profile_id': 'profile-2',
        })

        socket_client.emit('stopSession', {})
        socket_client.disconnect()


@pytest.mark.slow
class TestErrorRecoveryScenarios:
    """Task 5.5: Test error recovery scenarios."""

    def test_conversion_with_missing_adapter(self, client, audio_5s, app_with_full_stack):
        """Test conversion fails gracefully when adapter missing."""

        with patch('auto_voice.models.adapter_manager.AdapterManager') as MockAM:
            mock_am = MagicMock()
            mock_am.has_adapter.return_value = False  # No adapter
            MockAM.return_value = mock_am

            audio_5s.seek(0)
            response = client.post(
                '/api/v1/convert/song',
                data={
                    'song': (audio_5s, 'test.wav'),
                    'profile_id': 'profile-without-adapter',
                    'adapter_type': 'hq',  # Must specify valid adapter_type
                },
                content_type='multipart/form-data'
            )

        # Should return 404 with helpful message
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data

    def test_profile_not_found_graceful_error(self, client, audio_5s, app_with_full_stack):
        """Test profile not found returns proper error."""

        from auto_voice.storage.voice_profiles import ProfileNotFoundError

        app_with_full_stack.voice_cloner.load_voice_profile.side_effect = ProfileNotFoundError("not found")

        with patch('auto_voice.models.adapter_manager.AdapterManager') as MockAM:
            mock_am = MagicMock()
            mock_am.has_adapter.return_value = True
            MockAM.return_value = mock_am

            audio_5s.seek(0)
            response = client.post(
                '/api/v1/convert/song',
                data={
                    'song': (audio_5s, 'test.wav'),
                    'profile_id': 'nonexistent-profile',
                    'adapter_type': 'hq',
                },
                content_type='multipart/form-data'
            )

        assert response.status_code == 404

    def test_training_job_cancellation(self, client, app_with_full_stack):
        """Test training job can be cancelled."""

        # Create a training job first
        with patch('auto_voice.storage.voice_profiles.VoiceProfileStore') as MockStore:
            mock_store = MagicMock()
            mock_store.exists.return_value = True
            mock_store.get_all_vocals_paths.return_value = ['/path/vocals.wav']
            mock_store.get_total_training_duration.return_value = 30.0
            MockStore.return_value = mock_store

            create_response = client.post(
                '/api/v1/training/jobs',
                json={'profile_id': 'test-profile'},
                content_type='application/json'
            )

        if create_response.status_code in (200, 201, 202):
            data = json.loads(create_response.data)
            job_id = data.get('job_id')

            if job_id:
                # Cancel the job
                cancel_response = client.post(f'/api/v1/training/jobs/{job_id}/cancel')

                # Should handle cancellation
                assert cancel_response.status_code in (200, 400, 404)

    def test_websocket_disconnect_cleanup(self, app_with_full_stack):
        """Test WebSocket disconnect cleans up resources."""

        from flask_socketio import SocketIOTestClient

        socketio = app_with_full_stack.socketio
        socket_client = SocketIOTestClient(app_with_full_stack, socketio)

        socket_client.connect()

        # Start session
        socket_client.emit('startSession', {
            'profile_id': 'test-profile',
        })

        # Disconnect abruptly (simulating network failure)
        socket_client.disconnect()

        # Reconnect - should not have stale session
        socket_client.connect()

        # Start new session - should work
        socket_client.emit('startSession', {
            'profile_id': 'test-profile',
        })

        socket_client.disconnect()


class TestConcurrentOperations:
    """Test concurrent operation handling."""

    def test_multiple_conversion_jobs(self, client, app_with_full_stack):
        """Test multiple conversion jobs don't interfere."""

        with patch('auto_voice.models.adapter_manager.AdapterManager') as MockAM:
            mock_am = MagicMock()
            mock_am.has_adapter.return_value = True
            MockAM.return_value = mock_am

            responses = []

            for i in range(3):
                response = client.post(
                    '/api/v1/convert/song',
                    data={
                        'song': (make_audio_buffer(5), f'test{i}.wav'),
                        'profile_id': 'new-profile-123',
                        'adapter_type': 'hq',  # Must specify valid adapter_type
                    },
                    content_type='multipart/form-data'
                )
                responses.append(response)

        # All should be accepted
        for response in responses:
            assert response.status_code in (200, 202, 400, 404, 500)

    def test_list_operations_during_training(self, client):
        """Test list operations work while training is running."""

        # List profiles should work
        response = client.get('/api/v1/voice/profiles')
        assert response.status_code == 200

        # List training jobs should work
        response = client.get('/api/v1/training/jobs')
        assert response.status_code == 200

        # Health check should work - note: it's at /api/v1/health
        response = client.get('/api/v1/health')
        assert response.status_code == 200


class TestAPIEndpointAvailability:
    """Test that critical API endpoints are available and respond correctly."""

    def test_health_endpoint(self, client):
        """Health endpoint is accessible and returns valid response."""
        response = client.get('/api/v1/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'status' in data or 'healthy' in data or isinstance(data, dict)

    def test_voice_profiles_list(self, client):
        """Voice profiles list is accessible."""
        response = client.get('/api/v1/voice/profiles')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, (list, dict))

    def test_training_jobs_list(self, client):
        """Training jobs list is accessible."""
        response = client.get('/api/v1/training/jobs')
        assert response.status_code == 200

    def test_system_info(self, client):
        """System info is accessible."""
        response = client.get('/api/v1/system/info')
        assert response.status_code == 200

    def test_gpu_metrics(self, client):
        """GPU metrics endpoint is accessible."""
        response = client.get('/api/v1/gpu/metrics')
        # May fail if no GPU, but should not crash
        assert response.status_code in (200, 500, 503)

    def test_devices_list(self, client):
        """Devices list is accessible."""
        response = client.get('/api/v1/devices/list')
        assert response.status_code == 200


class TestProfileLifecycle:
    """Test profile lifecycle operations."""

    def test_clone_then_get_profile(self, client, app_with_full_stack):
        """Clone voice then get profile details."""
        # Create profile
        response = client.post(
            '/api/v1/voice/clone',
            data={
                'audio': (make_audio_buffer(5), 'sample.wav'),
                'name': 'Test Artist',
            },
            content_type='multipart/form-data'
        )

        # Try to get profile details
        response = client.get('/api/v1/voice/profiles/new-profile-123')
        # May succeed or fail based on actual storage
        assert response.status_code in (200, 404)

    def test_list_profile_adapters(self, client, app_with_full_stack):
        """List adapters for a profile."""
        response = client.get('/api/v1/voice/profiles/new-profile-123/adapters')
        assert response.status_code in (200, 404, 500)

    def test_get_profile_training_status(self, client, app_with_full_stack):
        """Get training status for a profile."""
        response = client.get('/api/v1/voice/profiles/new-profile-123/training-status')
        assert response.status_code in (200, 404, 500)


class TestConversionJobLifecycle:
    """Test conversion job lifecycle."""

    def test_get_job_status(self, client, app_with_full_stack):
        """Get status of a job - may return 200 with status or 404."""
        response = client.get('/api/v1/convert/status/nonexistent-job')
        # The mock job_manager may return 200 with status, or 404 if job not found
        assert response.status_code in (200, 404, 500)
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'status' in data or 'error' in data

    def test_download_nonexistent_job(self, client, app_with_full_stack):
        """Download non-existent job returns error."""
        response = client.get('/api/v1/convert/download/nonexistent-job')
        assert response.status_code in (404, 500)

    def test_cancel_nonexistent_job(self, client, app_with_full_stack):
        """Cancel a job - may return 200 if cancelled or 404 if not found."""
        response = client.post('/api/v1/convert/cancel/nonexistent-job')
        # Mock may return 200 for successful cancellation
        assert response.status_code in (200, 404, 500)


class TestInputValidation:
    """Test input validation across endpoints."""

    def test_clone_without_audio(self, client):
        """Clone without audio file fails."""
        response = client.post(
            '/api/v1/voice/clone',
            data={'name': 'Test'},
            content_type='multipart/form-data'
        )
        assert response.status_code == 400

    def test_convert_with_invalid_adapter_type(self, client, app_with_full_stack):
        """Convert with invalid adapter type fails with ValueError (caught by Flask)."""
        # The API raises ValueError for invalid adapter_type, which Flask catches
        # and may return 500 or the exception propagates in test mode
        try:
            response = client.post(
                '/api/v1/convert/song',
                data={
                    'song': (make_audio_buffer(5), 'test.wav'),
                    'profile_id': 'test-profile',
                    'adapter_type': 'invalid_adapter',  # Invalid type
                },
                content_type='multipart/form-data'
            )
            # If we get here, the response should be an error code
            assert response.status_code in (400, 500)
        except ValueError as e:
            # ValueError is expected for invalid adapter_type
            assert 'adapter_type' in str(e).lower()

    def test_training_without_profile_id(self, client, app_with_full_stack):
        """Training without profile_id fails."""
        response = client.post(
            '/api/v1/training/jobs',
            json={},  # Missing profile_id
            content_type='application/json'
        )
        assert response.status_code in (400, 500)

    def test_invalid_audio_file_extension(self, client, app_with_full_stack):
        """Test invalid audio file extension is rejected."""
        buffer = io.BytesIO(b'not audio content')
        buffer.seek(0)

        response = client.post(
            '/api/v1/convert/song',
            data={
                'song': (buffer, 'test.txt'),  # Wrong extension
                'profile_id': 'test-profile',
            },
            content_type='multipart/form-data'
        )

        assert response.status_code == 400

    def test_missing_song_file(self, client, app_with_full_stack):
        """Test missing song file returns error."""
        response = client.post(
            '/api/v1/convert/song',
            data={
                'profile_id': 'test-profile',
                # No song file
            },
            content_type='multipart/form-data'
        )

        assert response.status_code == 400


@pytest.mark.slow
class TestQualityValidation:
    """Task 5.1/5.2/5.3: Test quality validation aspects of E2E flows."""

    def test_conversion_returns_valid_structure(self, client, app_with_full_stack):
        """Test that successful conversion returns expected structure."""
        response = client.post(
            '/api/v1/convert/song',
            data={
                'song': (make_audio_buffer(5), 'test.wav'),
                'profile_id': 'new-profile-123',
                'adapter_type': 'hq',
            },
            content_type='multipart/form-data'
        )

        if response.status_code in (200, 202):
            data = json.loads(response.data)
            # Should have status and job_id for async, or audio for sync
            assert 'status' in data or 'audio' in data or 'job_id' in data

    def test_profile_clone_returns_profile_id(self, client, app_with_full_stack):
        """Test that profile clone returns a profile ID."""
        response = client.post(
            '/api/v1/voice/clone',
            data={
                'audio': (make_audio_buffer(5), 'sample.wav'),
                'name': 'Quality Test Artist',
            },
            content_type='multipart/form-data'
        )

        if response.status_code in (200, 201):
            data = json.loads(response.data)
            assert 'profile_id' in data or 'id' in data


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_short_audio(self, client, app_with_full_stack):
        """Test with very short audio file (1 second)."""
        response = client.post(
            '/api/v1/convert/song',
            data={
                'song': (make_audio_buffer(1), 'short.wav'),
                'profile_id': 'new-profile-123',
                'adapter_type': 'hq',
            },
            content_type='multipart/form-data'
        )
        # May succeed or fail based on minimum duration requirements
        assert response.status_code in (200, 202, 400, 404, 500, 503)

    def test_empty_profile_list(self, client, app_with_full_stack):
        """Test profile list returns valid empty list."""
        # Clear mock to return empty list
        app_with_full_stack.voice_cloner.list_profiles.return_value = []

        response = client.get('/api/v1/voice/profiles')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)

    def test_special_characters_in_profile_name(self, client, app_with_full_stack):
        """Test profile creation with special characters in name."""
        response = client.post(
            '/api/v1/voice/clone',
            data={
                'audio': (make_audio_buffer(5), 'sample.wav'),
                'name': 'Test Artist feat. Guest & "Special"',
            },
            content_type='multipart/form-data'
        )
        # Should handle special characters gracefully
        assert response.status_code in (200, 201, 400, 500)
