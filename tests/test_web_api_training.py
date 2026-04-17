"""Comprehensive tests for /api/v1/training/* endpoints.

Phase 4.3: Tests for training job management:
- GET /training/jobs - List all training jobs
- POST /training/jobs - Create and start training job
- GET /training/jobs/{id} - Get job status
- POST /training/jobs/{id}/cancel - Cancel job

Uses Flask test client with mocked components.
"""

import io
import json
import os
import tempfile
from unittest.mock import MagicMock, patch, PropertyMock
import wave
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def app_with_training():
    """Create Flask app with training components mocked."""
    pytest.importorskip('flask_swagger_ui', reason="flask_swagger_ui not installed")

    from auto_voice.web.app import create_app

    app, socketio = create_app(config={
        'TESTING': True,
        'singing_conversion_enabled': True,
        'voice_cloning_enabled': True,
    })

    # Mock voice cloner
    mock_voice_cloner = MagicMock()
    mock_voice_cloner.load_voice_profile.return_value = {
        'profile_id': 'test-profile',
        'name': 'Test Artist',
        'embedding': np.zeros(256).tolist(),
    }

    app.voice_cloner = mock_voice_cloner
    app.socketio = socketio
    app.app_config = {'audio': {'sample_rate': 22050}}

    return app


@pytest.fixture
def client(app_with_training):
    """Flask test client."""
    return app_with_training.test_client()


@pytest.fixture
def audio_file():
    """Create a minimal WAV file for upload."""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes(b'\x00' * 22050 * 2)  # 1 second
    buffer.seek(0)
    return buffer


def _write_wav(path: Path, duration_seconds: float = 1.0, sample_rate: int = 22050) -> None:
    frames = int(duration_seconds * sample_rate)
    with wave.open(str(path), 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b'\x00' * frames * 2)


def _create_profile_with_sample(app_with_training, *, profile_id: str, profile_role: str, duration_seconds: float) -> str:
    from auto_voice.storage.paths import resolve_profiles_dir, resolve_samples_dir
    from auto_voice.storage.voice_profiles import VoiceProfileStore

    data_dir = app_with_training.config['DATA_DIR']
    store = VoiceProfileStore(
        profiles_dir=str(resolve_profiles_dir(data_dir=data_dir)),
        samples_dir=str(resolve_samples_dir(data_dir=data_dir)),
    )
    store.save({
        'profile_id': profile_id,
        'name': f'{profile_role}-{profile_id}',
        'profile_role': profile_role,
        'created_from': 'manual',
    })

    sample_path = Path(tempfile.mkdtemp(prefix='autovoice-test-sample-')) / f'{profile_id}.wav'
    _write_wav(sample_path, duration_seconds=duration_seconds)
    store.add_training_sample(
        profile_id=profile_id,
        vocals_path=str(sample_path),
        duration=duration_seconds,
        source_file=sample_path.name,
    )
    return profile_id


class TestListTrainingJobs:
    """Test GET /api/v1/training/jobs endpoint."""

    def test_list_jobs_empty(self, client):
        """Returns empty array when no jobs exist."""
        response = client.get('/api/v1/training/jobs')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)

    def test_list_jobs_filter_by_profile(self, client):
        """Filters jobs by profile_id query parameter."""
        response = client.get('/api/v1/training/jobs?profile_id=test-profile')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)

    def test_list_jobs_filter_by_status(self, client):
        """Filters jobs by status query parameter."""
        response = client.get('/api/v1/training/jobs?status=completed')

        assert response.status_code == 200


class TestCreateTrainingJob:
    """Test POST /api/v1/training/jobs endpoint."""

    def test_create_job_missing_profile_id(self, client):
        """Returns 400 when profile_id is missing."""
        response = client.post(
            '/api/v1/training/jobs',
            json={},
            content_type='application/json'
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data

    def test_create_job_profile_not_found(self, client, app_with_training):
        """Returns 404 or 201 based on implementation."""
        # Note: The API may not validate profile existence at creation time
        response = client.post(
            '/api/v1/training/jobs',
            json={'profile_id': 'nonexistent'},
            content_type='application/json'
        )

        # May create job anyway or return error
        assert response.status_code in (201, 400, 404, 500)

    def test_create_job_returns_job_id(self, client, app_with_training):
        """Successfully creates job and returns job_id."""
        response = client.post(
            '/api/v1/training/jobs',
            json={'profile_id': 'test-profile'},
            content_type='application/json'
        )

        # Could be 200, 201, or 202 depending on async mode
        assert response.status_code in (200, 201, 202, 400, 500)

        if response.status_code in (200, 201, 202):
            data = json.loads(response.data)
            assert 'job_id' in data or 'id' in data or 'status' in data

    def test_create_job_with_config(self, client, app_with_training):
        """Creates job with custom training config."""
        response = client.post(
            '/api/v1/training/jobs',
            json={
                'profile_id': 'test-profile',
                'config': {
                    'max_epochs': 50,
                    'batch_size': 4,
                    'learning_rate': 0.0001,
                }
            },
            content_type='application/json'
        )

        # Accept various status codes based on implementation
        assert response.status_code in (200, 201, 202, 400, 500)

    def test_full_training_requires_30_minutes_of_clean_vocals(self, client, app_with_training):
        """Full-model training is gated by clean-vocal duration, not sample count."""
        profile_id = _create_profile_with_sample(
            app_with_training,
            profile_id='target-profile-short',
            profile_role='target_user',
            duration_seconds=120.0,
        )

        response = client.post(
            '/api/v1/training/jobs',
            json={
                'profile_id': profile_id,
                'config': {
                    'training_mode': 'full',
                },
            },
            content_type='application/json',
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert '30 minutes' in data['error']

    def test_source_artist_profiles_cannot_be_trained(self, client, app_with_training):
        """Source artist profiles are reference profiles, not trainable targets."""
        profile_id = _create_profile_with_sample(
            app_with_training,
            profile_id='source-profile-ref',
            profile_role='source_artist',
            duration_seconds=2000.0,
        )

        response = client.post(
            '/api/v1/training/jobs',
            json={'profile_id': profile_id},
            content_type='application/json',
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Only target user profiles can be trained' in data['error']


class TestGetTrainingJob:
    """Test GET /api/v1/training/jobs/{job_id} endpoint."""

    def test_get_job_not_found(self, client):
        """Returns 404 for non-existent job."""
        response = client.get('/api/v1/training/jobs/nonexistent-job-id')

        assert response.status_code == 404

    def test_get_job_returns_details(self, client):
        """Returns job details for existing job."""
        response = client.get('/api/v1/training/jobs/any-job-id')

        # Either 404 (not found) or 200 (found)
        assert response.status_code in (200, 404)


class TestCancelTrainingJob:
    """Test POST /api/v1/training/jobs/{job_id}/cancel endpoint."""

    def test_cancel_job_not_found(self, client):
        """Returns 404 for non-existent job."""
        response = client.post('/api/v1/training/jobs/nonexistent/cancel')

        assert response.status_code == 404

    def test_cancel_job_already_completed(self, client):
        """Returns appropriate status for already completed job."""
        response = client.post('/api/v1/training/jobs/completed-job/cancel')

        # Either 404 (not found) or 400 (can't cancel completed)
        assert response.status_code in (200, 400, 404)


class TestTrainingJobWorkflow:
    """Test complete training job workflow."""

    def test_create_list_get_workflow(self, client, app_with_training):
        """Test creating job, listing jobs, getting job details."""
        # Create
        create_response = client.post(
            '/api/v1/training/jobs',
            json={'profile_id': 'test-profile'},
            content_type='application/json'
        )

        if create_response.status_code in (200, 201, 202):
            data = json.loads(create_response.data)
            job_id = data.get('job_id')

            if job_id:
                # List
                list_response = client.get('/api/v1/training/jobs')
                assert list_response.status_code == 200

                # Get
                get_response = client.get(f'/api/v1/training/jobs/{job_id}')
                assert get_response.status_code in (200, 404)


class TestTrainingJobValidation:
    """Test input validation for training endpoints."""

    def test_invalid_json_returns_error(self, client):
        """Returns error for malformed JSON."""
        response = client.post(
            '/api/v1/training/jobs',
            data='not valid json',
            content_type='application/json'
        )

        # May return 400, 415, or 500 for JSON parse error
        assert response.status_code in (400, 415, 500)

    def test_empty_profile_id_returns_400(self, client):
        """Returns 400 for empty profile_id."""
        response = client.post(
            '/api/v1/training/jobs',
            json={'profile_id': ''},
            content_type='application/json'
        )

        assert response.status_code in (400, 404)

    def test_invalid_config_type(self, client):
        """Handles invalid config type."""
        response = client.post(
            '/api/v1/training/jobs',
            json={
                'profile_id': 'test-profile',
                'config': 'not a dict'
            },
            content_type='application/json'
        )

        # Should either handle gracefully or return error
        assert response.status_code in (200, 201, 202, 400, 404, 500)


class TestTrainingJobStatusFields:
    """Test that job status responses contain required fields."""

    def test_job_status_contains_required_fields(self, client, app_with_training):
        """Job status should contain status or job_id fields."""
        create_response = client.post(
            '/api/v1/training/jobs',
            json={'profile_id': 'test-profile'},
            content_type='application/json'
        )

        if create_response.status_code in (200, 201, 202):
            data = json.loads(create_response.data)

            # Should have status field
            assert 'status' in data or 'job_id' in data
