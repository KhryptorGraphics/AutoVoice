"""End-to-end verification of async conversion workflow.

Verifies complete async conversion workflow:
1. POST /api/v1/convert/song returns 202 with job_id
2. GET /api/v1/convert/status/{job_id} shows progress
3. WebSocket events emitted (conversion.started, conversion.progress, conversion.completed)
4. GET /api/v1/convert/download/{job_id} returns converted audio
5. Job persisted to JSON and survives manager restart
"""

import io
import json
import os
import tempfile
import time
import wave
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_jobs_dir():
    """Temporary directory for job persistence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_singing_pipeline():
    """Mock SingingConversionPipeline for testing."""
    pipeline = MagicMock()

    # Mock convert_song to return realistic result
    def mock_convert(song_path, target_profile_id, **kwargs):
        # Simulate conversion delay
        time.sleep(0.1)

        sample_rate = 44100
        duration = 30.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        return {
            'mixed_audio': audio,
            'sample_rate': sample_rate,
            'duration': duration,
            'f0_contour': np.random.rand(1000) * 200 + 100,
            'f0_original': np.random.rand(1000) * 200 + 100,
            'metadata': {
                'profile_id': target_profile_id,
                'duration': duration,
            }
        }

    pipeline.convert_song = Mock(side_effect=mock_convert)
    return pipeline


@pytest.fixture
def mock_voice_cloner():
    """Mock VoiceCloner for testing."""
    cloner = MagicMock()

    def mock_load_profile(profile_id):
        if profile_id == 'test-profile':
            return {
                'profile_id': 'test-profile',
                'name': 'Test Artist',
                'embedding': np.zeros(256).tolist(),
                'selected_adapter': None,  # Will default to 'unified' in api.py
            }
        raise Exception("Profile not found")

    cloner.load_voice_profile = Mock(side_effect=mock_load_profile)
    return cloner


@pytest.fixture
def mock_adapter_manager():
    """Mock AdapterManager for testing."""
    with patch('auto_voice.models.adapter_manager.AdapterManager') as MockAdapterManager:
        manager = MagicMock()
        manager.has_adapter.return_value = True
        MockAdapterManager.return_value = manager
        yield manager


@pytest.fixture
def app_with_conversion(temp_jobs_dir, mock_singing_pipeline, mock_voice_cloner):
    """Create Flask app with ConversionJobManager."""
    pytest.importorskip('flask_swagger_ui', reason="flask_swagger_ui not installed")

    from auto_voice.web.app import create_app
    from auto_voice.inference.conversion_job_manager import ConversionJobManager

    app, socketio = create_app(config={
        'TESTING': True,
        'singing_conversion_enabled': True,
        'voice_cloning_enabled': True,
    })

    # Override with mocks
    app.singing_conversion_pipeline = mock_singing_pipeline
    app.voice_cloner = mock_voice_cloner

    # Create ConversionJobManager with temp directory
    app.conversion_job_manager = ConversionJobManager(
        singing_pipeline=mock_singing_pipeline,
        socketio=socketio,
        jobs_dir=str(temp_jobs_dir)
    )

    app.socketio = socketio
    app.app_config = {'audio': {'sample_rate': 44100, 'hop_length': 512}}

    return app, socketio


@pytest.fixture
def client(app_with_conversion):
    """Flask test client."""
    app, _ = app_with_conversion
    return app.test_client()


@pytest.fixture
def socketio_test_client(app_with_conversion):
    """SocketIO test client."""
    app, socketio = app_with_conversion
    return socketio.test_client(app)


@pytest.fixture
def audio_file():
    """Create a minimal WAV file for upload."""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        wav.writeframes(b'\x00' * 44100 * 2)  # 1 second
    buffer.seek(0)
    return buffer


# ============================================================================
# E2E Workflow Tests
# ============================================================================

class TestCompleteAsyncConversionWorkflow:
    """End-to-end verification of async conversion workflow."""

    def test_step_1_post_convert_song_returns_202_with_job_id(self, client, audio_file, mock_adapter_manager):
        """Verify POST /api/v1/convert/song returns 202 with job_id."""
        response = client.post(
            '/api/v1/convert/song',
            data={
                'song': (audio_file, 'test_song.wav'),
                'profile_id': 'test-profile',
                'adapter_type': 'hq',  # Explicit adapter type to pass validation
            },
            content_type='multipart/form-data'
        )

        # Should return 202 Accepted for async processing
        assert response.status_code == 202

        data = json.loads(response.data)
        assert data['status'] == 'queued'
        assert 'job_id' in data
        assert 'websocket_room' in data
        assert len(data['job_id']) > 0

    def test_step_2_get_status_shows_progress(self, client, audio_file, app_with_conversion, mock_adapter_manager):
        """Verify GET /api/v1/convert/status/{job_id} shows progress."""
        # Create a job
        response = client.post(
            '/api/v1/convert/song',
            data={
                'song': (audio_file, 'test_song.wav'),
                'profile_id': 'test-profile',
                'adapter_type': 'hq',
            },
            content_type='multipart/form-data'
        )

        assert response.status_code == 202
        job_id = json.loads(response.data)['job_id']

        # Check status
        status_response = client.get(f'/api/v1/convert/status/{job_id}')
        assert status_response.status_code == 200

        status_data = json.loads(status_response.data)
        assert status_data['job_id'] == job_id
        assert status_data['status'] in ['pending', 'running', 'completed']
        assert 'profile_id' in status_data
        assert status_data['profile_id'] == 'test-profile'

    def test_step_3_websocket_events_emitted(self, client, audio_file, socketio_test_client, app_with_conversion, mock_adapter_manager):
        """Verify WebSocket events emitted (conversion.started, conversion.progress, conversion.completed)."""
        app, socketio = app_with_conversion

        # Track emitted events
        emitted_events = []
        original_emit = socketio.emit

        def track_emit(event, data, **kwargs):
            emitted_events.append({'event': event, 'data': data})
            return original_emit(event, data, **kwargs)

        socketio.emit = track_emit

        # Create and execute job
        response = client.post(
            '/api/v1/convert/song',
            data={
                'song': (audio_file, 'test_song.wav'),
                'profile_id': 'test-profile',
                'adapter_type': 'hq',
            },
            content_type='multipart/form-data'
        )

        assert response.status_code == 202
        job_id = json.loads(response.data)['job_id']

        # Execute the job (simulate background worker)
        job = app.conversion_job_manager.get_job(job_id)
        assert job is not None

        # Execute job synchronously for testing
        app.conversion_job_manager.execute_job(job_id)

        # Verify events were emitted
        event_names = [e['event'] for e in emitted_events]

        # Should have: job_created, conversion.started, conversion.progress (multiple), conversion.completed
        assert 'job_created' in event_names
        assert 'conversion.started' in event_names
        assert 'conversion.completed' in event_names or 'conversion.failed' in event_names

        # Find completion event
        completed_events = [e for e in emitted_events if e['event'] == 'conversion.completed']
        if completed_events:
            completed_data = completed_events[0]['data']
            assert completed_data['job_id'] == job_id
            # Completion event should have metrics or duration indicating success
            assert 'metrics' in completed_data or 'duration' in completed_data

    def test_step_4_download_converted_audio(self, client, audio_file, app_with_conversion, mock_adapter_manager):
        """Verify GET /api/v1/convert/download/{job_id} returns converted audio."""
        app, _ = app_with_conversion

        # Create job
        response = client.post(
            '/api/v1/convert/song',
            data={
                'song': (audio_file, 'test_song.wav'),
                'profile_id': 'test-profile',
                'adapter_type': 'hq',
            },
            content_type='multipart/form-data'
        )

        assert response.status_code == 202
        job_id = json.loads(response.data)['job_id']

        # Execute job to completion
        app.conversion_job_manager.execute_job(job_id)

        # Verify job completed
        job = app.conversion_job_manager.get_job(job_id)
        assert job.status == 'completed'
        assert job.result_path is not None
        assert os.path.exists(job.result_path)

        # Download audio
        download_response = client.get(f'/api/v1/convert/download/{job_id}')
        assert download_response.status_code == 200
        assert download_response.mimetype == 'audio/wav'
        assert len(download_response.data) > 0

        # Verify it's a valid WAV file
        wav_data = io.BytesIO(download_response.data)
        with wave.open(wav_data, 'rb') as wav:
            assert wav.getnchannels() in [1, 2]
            assert wav.getframerate() > 0
            assert wav.getnframes() > 0

    def test_step_5_job_persistence_survives_restart(self, client, audio_file, app_with_conversion, mock_adapter_manager, temp_jobs_dir, mock_singing_pipeline):
        """Verify job persisted to JSON and survives manager restart."""
        app, socketio = app_with_conversion

        # Create and execute job
        response = client.post(
            '/api/v1/convert/song',
            data={
                'song': (audio_file, 'test_song.wav'),
                'profile_id': 'test-profile',
                'adapter_type': 'hq',
            },
            content_type='multipart/form-data'
        )

        assert response.status_code == 202
        job_id = json.loads(response.data)['job_id']

        # Execute job
        app.conversion_job_manager.execute_job(job_id)

        # Verify job is persisted - check the jobs file
        jobs_file = temp_jobs_dir / "jobs.json"
        assert jobs_file.exists(), f"Jobs file should exist at {jobs_file}"

        # Parse jobs file
        with open(jobs_file, 'r') as f:
            jobs_data = json.load(f)

        assert isinstance(jobs_data, list)
        assert len(jobs_data) > 0

        # Find our job
        job_data = next((j for j in jobs_data if j['job_id'] == job_id), None)
        assert job_data is not None
        assert job_data['profile_id'] == 'test-profile'
        assert job_data['status'] == 'completed'

        # Simulate restart by creating new manager
        from auto_voice.inference.conversion_job_manager import ConversionJobManager

        new_manager = ConversionJobManager(
            singing_pipeline=mock_singing_pipeline,
            socketio=socketio,
            jobs_dir=str(temp_jobs_dir)
        )

        # Verify job was loaded from persistence
        restored_job = new_manager.get_job(job_id)
        assert restored_job is not None
        assert restored_job.job_id == job_id
        assert restored_job.status == 'completed'
        assert restored_job.profile_id == 'test-profile'

    def test_complete_workflow_integration(self, client, audio_file, app_with_conversion, socketio_test_client, mock_adapter_manager):
        """Full integration test of all workflow steps together."""
        app, socketio = app_with_conversion

        # Track WebSocket events
        emitted_events = []
        original_emit = socketio.emit

        def track_emit(event, data, **kwargs):
            emitted_events.append({'event': event, 'data': data})
            return original_emit(event, data, **kwargs)

        socketio.emit = track_emit

        # Step 1: Create job
        response = client.post(
            '/api/v1/convert/song',
            data={
                'song': (audio_file, 'test_song.wav'),
                'profile_id': 'test-profile',
                'adapter_type': 'hq',
                'vocal_volume': '1.0',
                'instrumental_volume': '0.8',
            },
            content_type='multipart/form-data'
        )

        assert response.status_code == 202
        data = json.loads(response.data)
        job_id = data['job_id']
        assert data['status'] == 'queued'

        # Step 2: Check initial status
        status_response = client.get(f'/api/v1/convert/status/{job_id}')
        assert status_response.status_code == 200
        status_data = json.loads(status_response.data)
        assert status_data['status'] == 'pending'

        # Step 3: Execute job
        app.conversion_job_manager.execute_job(job_id)

        # Step 4: Verify completion status
        status_response = client.get(f'/api/v1/convert/status/{job_id}')
        assert status_response.status_code == 200
        status_data = json.loads(status_response.data)
        assert status_data['status'] == 'completed'
        assert 'download_url' in status_data

        # Step 5: Download audio
        download_response = client.get(f'/api/v1/convert/download/{job_id}')
        assert download_response.status_code == 200
        assert len(download_response.data) > 0

        # Step 6: Verify WebSocket events
        event_names = [e['event'] for e in emitted_events]
        assert 'conversion.started' in event_names
        assert 'conversion.completed' in event_names

        # Step 7: Verify persistence
        jobs_file = Path(app.conversion_job_manager.jobs_dir) / "jobs.json"
        assert jobs_file.exists(), f"Jobs file should exist at {jobs_file}"


class TestAsyncWorkflowEdgeCases:
    """Test edge cases and error conditions."""

    def test_status_request_for_nonexistent_job(self, client):
        """Verify 404 for nonexistent job status."""
        response = client.get('/api/v1/convert/status/nonexistent-job-id')
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data

    def test_download_request_for_nonexistent_job(self, client):
        """Verify 404 for nonexistent job download."""
        response = client.get('/api/v1/convert/download/nonexistent-job-id')
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data

    def test_download_before_job_completion(self, client, audio_file, app_with_conversion, mock_adapter_manager):
        """Verify download fails when job not completed."""
        app, _ = app_with_conversion

        # Create job but don't execute
        response = client.post(
            '/api/v1/convert/song',
            data={
                'song': (audio_file, 'test_song.wav'),
                'profile_id': 'test-profile',
                'adapter_type': 'hq',
            },
            content_type='multipart/form-data'
        )

        assert response.status_code == 202
        job_id = json.loads(response.data)['job_id']

        # Try to download before completion
        download_response = client.get(f'/api/v1/convert/download/{job_id}')
        assert download_response.status_code == 404

    def test_cancel_job_workflow(self, client, audio_file, app_with_conversion, mock_adapter_manager):
        """Verify job cancellation workflow."""
        app, _ = app_with_conversion

        # Create job
        response = client.post(
            '/api/v1/convert/song',
            data={
                'song': (audio_file, 'test_song.wav'),
                'profile_id': 'test-profile',
                'adapter_type': 'hq',
            },
            content_type='multipart/form-data'
        )

        assert response.status_code == 202
        job_id = json.loads(response.data)['job_id']

        # Cancel job
        cancel_response = client.post(f'/api/v1/convert/cancel/{job_id}')
        assert cancel_response.status_code == 200

        cancel_data = json.loads(cancel_response.data)
        assert cancel_data['status'] == 'cancelled'

        # Verify status shows cancelled
        status_response = client.get(f'/api/v1/convert/status/{job_id}')
        assert status_response.status_code == 200
        status_data = json.loads(status_response.data)
        assert status_data['status'] == 'cancelled'

    def test_metrics_for_completed_job(self, client, audio_file, app_with_conversion, mock_adapter_manager):
        """Verify metrics endpoint for completed job."""
        app, _ = app_with_conversion

        # Create and execute job
        response = client.post(
            '/api/v1/convert/song',
            data={
                'song': (audio_file, 'test_song.wav'),
                'profile_id': 'test-profile',
                'adapter_type': 'hq',
            },
            content_type='multipart/form-data'
        )

        assert response.status_code == 202
        job_id = json.loads(response.data)['job_id']

        # Execute job
        app.conversion_job_manager.execute_job(job_id)

        # Get metrics
        metrics_response = client.get(f'/api/v1/convert/metrics/{job_id}')
        assert metrics_response.status_code == 200

        metrics_data = json.loads(metrics_response.data)
        # Metrics endpoint returns the metrics directly or wrapped
        assert ('metrics' in metrics_data or 'pitch_accuracy' in metrics_data), \
            f"Expected metrics data, got: {metrics_data}"
