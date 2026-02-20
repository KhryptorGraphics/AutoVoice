"""Integration tests for diarization API endpoints.

Tests the full diarization workflow:
- Run diarization on audio
- Assign segments to profiles
- Get profile segments
- Auto-create profiles from diarization
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.io import wavfile


@pytest.fixture
def app():
    """Create Flask test app."""
    from auto_voice.web.app import create_app

    app, socketio = create_app(testing=True)
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def test_audio_file():
    """Create a test audio file."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        # Create 3 seconds of audio at 16kHz
        sample_rate = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        # Create audio with some variation (not just silence)
        audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(f.name, sample_rate, audio_int16)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def mock_diarization_result():
    """Mock diarization result for testing."""
    from auto_voice.audio.speaker_diarization import DiarizationResult, SpeakerSegment

    segments = [
        SpeakerSegment(
            start=0.0,
            end=1.5,
            speaker_id='speaker_0',
            confidence=0.95,
            embedding=np.random.randn(512).astype(np.float32),
        ),
        SpeakerSegment(
            start=1.5,
            end=3.0,
            speaker_id='speaker_1',
            confidence=0.90,
            embedding=np.random.randn(512).astype(np.float32),
        ),
    ]
    return DiarizationResult(
        segments=segments,
        audio_duration=3.0,
        num_speakers=2,
    )


class TestDiarizeAudioEndpoint:
    """Tests for POST /api/v1/audio/diarize."""

    def test_diarize_with_audio_path(self, client, test_audio_file, mock_diarization_result):
        """Test diarization with audio_path parameter."""
        with patch('auto_voice.audio.speaker_diarization.SpeakerDiarizer') as MockDiarizer:
            mock_instance = MagicMock()
            mock_instance.diarize.return_value = mock_diarization_result
            MockDiarizer.return_value = mock_instance

            response = client.post(
                '/api/v1/audio/diarize',
                json={'audio_path': test_audio_file},
                content_type='application/json',
            )

            assert response.status_code == 200
            data = response.get_json()

            assert 'diarization_id' in data
            assert data['num_speakers'] == 2
            assert len(data['segments']) == 2
            assert data['audio_duration'] == 3.0

    def test_diarize_with_file_upload(self, client, test_audio_file, mock_diarization_result):
        """Test diarization with file upload."""
        with patch('auto_voice.audio.speaker_diarization.SpeakerDiarizer') as MockDiarizer:
            mock_instance = MagicMock()
            mock_instance.diarize.return_value = mock_diarization_result
            MockDiarizer.return_value = mock_instance

            with open(test_audio_file, 'rb') as f:
                response = client.post(
                    '/api/v1/audio/diarize',
                    data={'file': (f, 'test.wav')},
                    content_type='multipart/form-data',
                )

            assert response.status_code == 200
            data = response.get_json()
            assert 'diarization_id' in data

    def test_diarize_missing_audio(self, client):
        """Test error when no audio provided."""
        response = client.post(
            '/api/v1/audio/diarize',
            json={'audio_path': '/nonexistent/path.wav'},
            content_type='application/json',
        )

        assert response.status_code == 400

    def test_diarize_no_input(self, client):
        """Test error when no input provided."""
        response = client.post('/api/v1/audio/diarize')

        assert response.status_code == 400


class TestAssignSegmentEndpoint:
    """Tests for POST /api/v1/audio/diarize/assign."""

    def test_assign_segment_to_profile(self, client, test_audio_file):
        """Test assigning a segment to a profile."""
        from auto_voice.web import api

        # Create a mock diarization result
        diarization_id = 'test-diarization-123'
        api._diarization_results[diarization_id] = {
            'audio_path': test_audio_file,
            'audio_duration': 3.0,
            'sample_rate': 16000,
            'num_speakers': 2,
            'segments': [
                {'start': 0.0, 'end': 1.5, 'speaker_id': 'speaker_0', 'confidence': 0.95},
                {'start': 1.5, 'end': 3.0, 'speaker_id': 'speaker_1', 'confidence': 0.90},
            ],
        }

        with patch('auto_voice.storage.voice_profiles.VoiceProfileStore') as MockStore:
            mock_store = MagicMock()
            mock_store.exists.return_value = True
            mock_store.add_training_sample.return_value = MagicMock()
            MockStore.return_value = mock_store

            with patch('auto_voice.audio.speaker_diarization.SpeakerDiarizer') as MockDiarizer:
                mock_diarizer = MagicMock()
                mock_diarizer.extract_speaker_audio.return_value = Path(test_audio_file)
                MockDiarizer.return_value = mock_diarizer

                response = client.post(
                    '/api/v1/audio/diarize/assign',
                    json={
                        'diarization_id': diarization_id,
                        'segment_index': 0,
                        'profile_id': 'test-profile',
                    },
                )

                assert response.status_code == 200
                data = response.get_json()
                assert data['status'] == 'success'
                assert data['profile_id'] == 'test-profile'
                assert data['segment_index'] == 0

        # Cleanup
        del api._diarization_results[diarization_id]

    def test_assign_segment_invalid_diarization(self, client):
        """Test error when diarization not found."""
        response = client.post(
            '/api/v1/audio/diarize/assign',
            json={
                'diarization_id': 'nonexistent',
                'segment_index': 0,
                'profile_id': 'test-profile',
            },
        )

        assert response.status_code == 404

    def test_assign_segment_missing_params(self, client):
        """Test error when required params missing."""
        response = client.post(
            '/api/v1/audio/diarize/assign',
            json={'diarization_id': 'test'},
        )

        assert response.status_code == 400


class TestProfileSegmentsEndpoint:
    """Tests for GET /api/v1/profiles/{id}/segments."""

    def test_get_profile_segments(self, client):
        """Test getting segments for a profile."""
        from auto_voice.web import api

        # Clear any state from previous tests
        api._segment_assignments.clear()

        with patch('auto_voice.storage.voice_profiles.VoiceProfileStore') as MockStore:
            mock_store = MagicMock()
            mock_store.exists.return_value = True

            # Mock training sample
            mock_sample = MagicMock()
            mock_sample.sample_id = 'sample_001'
            mock_sample.vocals_path = '/path/to/vocals.wav'
            mock_sample.duration = 30.0
            mock_sample.source_file = 'test.mp3'
            mock_sample.created_at = '2026-01-30T12:00:00Z'
            mock_store.list_training_samples.return_value = [mock_sample]

            MockStore.return_value = mock_store

            response = client.get('/api/v1/profiles/test-profile-segments/segments')

            assert response.status_code == 200
            data = response.get_json()
            assert data['profile_id'] == 'test-profile-segments'
            assert data['total_segments'] == 1
            assert data['total_duration'] == 30.0
            assert len(data['training_samples']) == 1

    def test_get_profile_segments_not_found(self, client):
        """Test error when profile not found."""
        with patch('auto_voice.storage.voice_profiles.VoiceProfileStore') as MockStore:
            mock_store = MagicMock()
            mock_store.exists.return_value = False
            MockStore.return_value = mock_store

            response = client.get('/api/v1/profiles/nonexistent/segments')

            assert response.status_code == 404


class TestAutoCreateProfileEndpoint:
    """Tests for POST /api/v1/profiles/auto-create."""

    def test_auto_create_profile(self, client, test_audio_file):
        """Test auto-creating profile from diarization."""
        from auto_voice.web import api

        # Create a mock diarization result
        diarization_id = 'test-diarization-456'
        api._diarization_results[diarization_id] = {
            'audio_path': test_audio_file,
            'audio_duration': 3.0,
            'sample_rate': 16000,
            'num_speakers': 2,
            'segments': [
                {'start': 0.0, 'end': 1.5, 'speaker_id': 'speaker_0', 'confidence': 0.95},
                {'start': 1.5, 'end': 3.0, 'speaker_id': 'speaker_1', 'confidence': 0.90},
            ],
        }

        with patch('auto_voice.audio.speaker_diarization.SpeakerDiarizer') as MockDiarizer:
            mock_diarizer = MagicMock()
            mock_diarizer.extract_speaker_embedding.return_value = np.random.randn(512).astype(np.float32)
            mock_diarizer.extract_speaker_audio.return_value = Path(test_audio_file)
            MockDiarizer.return_value = mock_diarizer

            with patch('auto_voice.storage.voice_profiles.VoiceProfileStore') as MockStore:
                mock_store = MagicMock()
                mock_store.create_profile_from_diarization.return_value = 'new-profile-id'
                MockStore.return_value = mock_store

                response = client.post(
                    '/api/v1/profiles/auto-create',
                    json={
                        'diarization_id': diarization_id,
                        'speaker_id': 'speaker_0',
                        'name': 'Test Artist',
                    },
                )

                assert response.status_code == 201
                data = response.get_json()
                assert data['status'] == 'success'
                assert data['profile_id'] == 'new-profile-id'
                assert data['name'] == 'Test Artist'
                assert data['num_segments'] == 1
                assert data['embedding_dim'] == 512

        # Cleanup
        del api._diarization_results[diarization_id]

    def test_auto_create_profile_invalid_diarization(self, client):
        """Test error when diarization not found."""
        response = client.post(
            '/api/v1/profiles/auto-create',
            json={
                'diarization_id': 'nonexistent',
                'speaker_id': 'speaker_0',
                'name': 'Test',
            },
        )

        assert response.status_code == 404

    def test_auto_create_profile_invalid_speaker(self, client, test_audio_file):
        """Test error when speaker not found in diarization."""
        from auto_voice.web import api

        diarization_id = 'test-diarization-789'
        api._diarization_results[diarization_id] = {
            'audio_path': test_audio_file,
            'audio_duration': 3.0,
            'sample_rate': 16000,
            'num_speakers': 1,
            'segments': [
                {'start': 0.0, 'end': 3.0, 'speaker_id': 'speaker_0', 'confidence': 0.95},
            ],
        }

        response = client.post(
            '/api/v1/profiles/auto-create',
            json={
                'diarization_id': diarization_id,
                'speaker_id': 'speaker_99',  # Invalid speaker
                'name': 'Test',
            },
        )

        assert response.status_code == 400

        # Cleanup
        del api._diarization_results[diarization_id]

    def test_auto_create_profile_missing_params(self, client):
        """Test error when required params missing."""
        response = client.post(
            '/api/v1/profiles/auto-create',
            json={'diarization_id': 'test'},
        )

        assert response.status_code == 400


class TestSpeakerEmbeddingEndpoints:
    """Tests for speaker embedding endpoints."""

    def test_set_speaker_embedding(self, client, test_audio_file):
        """Test setting speaker embedding."""
        with patch('auto_voice.storage.voice_profiles.VoiceProfileStore') as MockStore:
            mock_store = MagicMock()
            mock_store.exists.return_value = True
            MockStore.return_value = mock_store

            with patch('auto_voice.audio.speaker_diarization.SpeakerDiarizer') as MockDiarizer:
                mock_diarizer = MagicMock()
                mock_diarizer.extract_speaker_embedding.return_value = np.random.randn(512).astype(np.float32)
                MockDiarizer.return_value = mock_diarizer

                response = client.post(
                    '/api/v1/profiles/test-profile/speaker-embedding',
                    json={'audio_path': test_audio_file},
                )

                assert response.status_code == 200
                data = response.get_json()
                assert data['status'] == 'success'
                assert data['embedding_dim'] == 512

    def test_get_speaker_embedding(self, client):
        """Test checking speaker embedding status."""
        with patch('auto_voice.storage.voice_profiles.VoiceProfileStore') as MockStore:
            mock_store = MagicMock()
            mock_store.exists.return_value = True
            mock_store.load_speaker_embedding.return_value = np.random.randn(512).astype(np.float32)
            MockStore.return_value = mock_store

            response = client.get('/api/v1/profiles/test-profile/speaker-embedding')

            assert response.status_code == 200
            data = response.get_json()
            assert data['has_embedding'] is True
            assert data['embedding_dim'] == 512

    def test_get_speaker_embedding_none(self, client):
        """Test when profile has no embedding."""
        with patch('auto_voice.storage.voice_profiles.VoiceProfileStore') as MockStore:
            mock_store = MagicMock()
            mock_store.exists.return_value = True
            mock_store.load_speaker_embedding.return_value = None
            MockStore.return_value = mock_store

            response = client.get('/api/v1/profiles/test-profile/speaker-embedding')

            assert response.status_code == 200
            data = response.get_json()
            assert data['has_embedding'] is False
            assert data['embedding_dim'] is None


class TestFilterSampleEndpoint:
    """Tests for POST /api/v1/profiles/{id}/samples/{id}/filter."""

    def test_filter_sample(self, client, test_audio_file):
        """Test filtering a sample."""
        from auto_voice.web import api

        # Setup mock sample
        api._profile_samples['test-profile'] = {
            'sample_001': {
                'sample_id': 'sample_001',
                'file_path': test_audio_file,
            }
        }

        with patch('auto_voice.storage.voice_profiles.VoiceProfileStore') as MockStore:
            mock_store = MagicMock()
            mock_store.load_speaker_embedding.return_value = np.random.randn(512).astype(np.float32)
            MockStore.return_value = mock_store

            with patch('auto_voice.audio.training_filter.TrainingDataFilter') as MockFilter:
                mock_filter = MagicMock()
                mock_filter.filter_training_audio.return_value = (
                    Path(test_audio_file),
                    {
                        'original_duration': 3.0,
                        'filtered_duration': 2.5,
                        'num_segments': 2,
                        'purity': 0.83,
                        'status': 'success',
                    },
                )
                MockFilter.return_value = mock_filter

                response = client.post(
                    '/api/v1/profiles/test-profile/samples/sample_001/filter',
                    json={'similarity_threshold': 0.7},
                )

                assert response.status_code == 200
                data = response.get_json()
                assert data['status'] == 'success'
                assert data['filtered_duration'] == 2.5
                assert data['purity'] == 0.83

        # Cleanup
        del api._profile_samples['test-profile']

    def test_filter_sample_no_embedding(self, client, test_audio_file):
        """Test error when profile has no embedding."""
        from auto_voice.web import api

        api._profile_samples['test-profile'] = {
            'sample_001': {
                'sample_id': 'sample_001',
                'file_path': test_audio_file,
            }
        }

        with patch('auto_voice.storage.voice_profiles.VoiceProfileStore') as MockStore:
            mock_store = MagicMock()
            mock_store.load_speaker_embedding.return_value = None
            MockStore.return_value = mock_store

            response = client.post(
                '/api/v1/profiles/test-profile/samples/sample_001/filter',
            )

            assert response.status_code == 400
            data = response.get_json()
            assert 'no speaker embedding' in data['error'].lower()

        # Cleanup
        del api._profile_samples['test-profile']
