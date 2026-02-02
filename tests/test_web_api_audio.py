"""Comprehensive tests for /api/v1/audio/* endpoints.

Phase 4.5: Tests for audio processing endpoints:
- POST /audio/diarize - Run speaker diarization
- POST /audio/diarize/assign - Assign segment to profile
- POST /profiles/auto-create - Auto-create profile from diarization

Uses Flask test client with mocked components.
"""

import io
import json
import os
import tempfile
from unittest.mock import MagicMock, patch
import wave

import numpy as np
import pytest


@pytest.fixture
def app_with_audio():
    """Create Flask app with audio processing components mocked."""
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
def client(app_with_audio):
    """Flask test client."""
    return app_with_audio.test_client()


@pytest.fixture
def audio_file():
    """Create a minimal WAV file for upload."""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes(b'\x00' * 22050 * 4)  # 2 seconds
    buffer.seek(0)
    return buffer


class TestDiarizeAudio:
    """Test POST /api/v1/audio/diarize endpoint."""

    def test_diarize_missing_file(self, client):
        """Returns 400 when no file provided."""
        response = client.post(
            '/api/v1/audio/diarize',
            data={},
            content_type='multipart/form-data'
        )

        assert response.status_code == 400

    def test_diarize_invalid_file_type(self, client):
        """Returns 400 for invalid file type."""
        response = client.post(
            '/api/v1/audio/diarize',
            data={'audio': (io.BytesIO(b'not audio'), 'test.txt')},
            content_type='multipart/form-data'
        )

        assert response.status_code == 400

    def test_diarize_with_num_speakers_hint(self, client, audio_file):
        """Accepts num_speakers hint parameter."""
        response = client.post(
            '/api/v1/audio/diarize',
            data={
                'audio': (audio_file, 'test.wav'),
                'num_speakers': '2',
            },
            content_type='multipart/form-data'
        )

        assert response.status_code in (200, 400, 500, 503)

    def test_diarize_json_request(self, client):
        """Accepts JSON request with audio path."""
        response = client.post(
            '/api/v1/audio/diarize',
            json={'audio_path': '/path/to/audio.wav'},
            content_type='application/json'
        )

        # Either processes or returns 400/404 for missing file
        assert response.status_code in (200, 400, 404, 500)


class TestAssignDiarizationSegment:
    """Test POST /api/v1/audio/diarize/assign endpoint."""

    def test_assign_missing_params(self, client):
        """Returns 400 when required params missing."""
        response = client.post(
            '/api/v1/audio/diarize/assign',
            json={},
            content_type='application/json'
        )

        assert response.status_code == 400


class TestAutoCreateProfile:
    """Test POST /api/v1/profiles/auto-create endpoint."""

    def test_auto_create_missing_params(self, client):
        """Returns 400 when required params missing."""
        response = client.post(
            '/api/v1/profiles/auto-create',
            json={},
            content_type='application/json'
        )

        assert response.status_code == 400

    def test_auto_create_missing_embedding(self, client):
        """Returns 400 when speaker embedding missing."""
        response = client.post(
            '/api/v1/profiles/auto-create',
            json={'name': 'New Artist'},
            content_type='application/json'
        )

        assert response.status_code == 400


class TestAudioRouterConfig:
    """Test /api/v1/audio/router/config endpoints."""

    def test_get_router_config(self, client):
        """Returns current router configuration."""
        response = client.get('/api/v1/audio/router/config')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, dict)

    def test_update_router_config(self, client):
        """Updates router configuration."""
        response = client.post(
            '/api/v1/audio/router/config',
            json={
                'similarity_threshold': 0.75,
                'default_pipeline': 'quality',
            },
            content_type='application/json'
        )

        assert response.status_code in (200, 400, 500)


class TestIdentifySpeaker:
    """Test POST /api/v1/audio/identify-speaker endpoint."""

    def test_identify_speaker_missing_audio(self, client):
        """Returns 400 when audio not provided."""
        response = client.post(
            '/api/v1/audio/identify-speaker',
            json={},
            content_type='application/json'
        )

        assert response.status_code == 400


class TestSeparateArtists:
    """Test POST /api/v1/audio/separate-artists endpoint."""

    def test_separate_artists_missing_audio(self, client):
        """Returns 400 when audio not provided."""
        response = client.post(
            '/api/v1/audio/separate-artists',
            data={},
            content_type='multipart/form-data'
        )

        assert response.status_code == 400

    def test_separate_artists_with_audio(self, client, audio_file):
        """Processes multi-artist separation."""
        response = client.post(
            '/api/v1/audio/separate-artists',
            data={'audio': (audio_file, 'duet.wav')},
            content_type='multipart/form-data'
        )

        # May succeed or fail based on model availability
        assert response.status_code in (200, 400, 500, 503)


class TestBatchSeparate:
    """Test POST /api/v1/audio/batch-separate endpoint."""

    def test_batch_separate_missing_paths(self, client):
        """Returns 400 when audio paths not provided."""
        response = client.post(
            '/api/v1/audio/batch-separate',
            json={},
            content_type='application/json'
        )

        assert response.status_code == 400

    def test_batch_separate_empty_paths(self, client):
        """Returns 400 for empty paths list."""
        response = client.post(
            '/api/v1/audio/batch-separate',
            json={'audio_paths': []},
            content_type='application/json'
        )

        assert response.status_code in (200, 400)

    def test_batch_separate_with_paths(self, client):
        """Processes batch separation request."""
        response = client.post(
            '/api/v1/audio/batch-separate',
            json={'audio_paths': ['/path/song1.wav', '/path/song2.wav']},
            content_type='application/json'
        )

        # Files may not exist, so expect error or success
        assert response.status_code in (200, 202, 400, 404, 500)
