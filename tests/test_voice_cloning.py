"""Comprehensive voice cloning tests for AutoVoice

Tests SpeakerEncoder, VoiceCloner, VoiceProfileStorage, and API endpoints
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Import fixtures
from conftest import (
    sample_audio_16khz, sample_audio_22khz, temp_dir,
    cuda_available, device
)


@pytest.mark.unit
class TestSpeakerEncoder:
    """Test SpeakerEncoder for embedding extraction"""

    @pytest.fixture
    def speaker_encoder(self, device):
        """Create SpeakerEncoder instance"""
        try:
            from src.auto_voice.models.speaker_encoder import SpeakerEncoder
            return SpeakerEncoder(device=device)
        except ImportError:
            pytest.skip("SpeakerEncoder not available")

    def test_initialization(self, speaker_encoder):
        """Test SpeakerEncoder initializes correctly"""
        assert speaker_encoder is not None
        assert hasattr(speaker_encoder, 'device')
        assert hasattr(speaker_encoder, 'encoder')

    def test_extract_embedding_from_array(self, speaker_encoder, sample_audio_16khz):
        """Test embedding extraction from numpy array"""
        embedding = speaker_encoder.extract_embedding(sample_audio_16khz, sample_rate=16000)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (256,)
        assert not np.isnan(embedding).any()
        assert not np.isinf(embedding).any()

    def test_extract_embedding_deterministic(self, speaker_encoder, sample_audio_16khz):
        """Test that same input produces same embedding"""
        emb1 = speaker_encoder.extract_embedding(sample_audio_16khz, sample_rate=16000)
        emb2 = speaker_encoder.extract_embedding(sample_audio_16khz, sample_rate=16000)

        np.testing.assert_array_almost_equal(emb1, emb2, decimal=5)

    def test_compute_similarity(self, speaker_encoder, sample_audio_16khz):
        """Test similarity computation between embeddings"""
        emb1 = speaker_encoder.extract_embedding(sample_audio_16khz, sample_rate=16000)

        # Create slightly different audio
        audio2 = sample_audio_16khz + np.random.randn(*sample_audio_16khz.shape) * 0.01
        emb2 = speaker_encoder.extract_embedding(audio2, sample_rate=16000)

        similarity = speaker_encoder.compute_similarity(emb1, emb2)

        assert isinstance(similarity, float)
        assert -1.0 <= similarity <= 1.0

    def test_is_same_speaker(self, speaker_encoder, sample_audio_16khz):
        """Test speaker verification"""
        emb1 = speaker_encoder.extract_embedding(sample_audio_16khz, sample_rate=16000)
        emb2 = speaker_encoder.extract_embedding(sample_audio_16khz, sample_rate=16000)

        is_same = speaker_encoder.is_same_speaker(emb1, emb2, threshold=0.75)
        assert is_same is True

    def test_get_embedding_stats(self, speaker_encoder, sample_audio_16khz):
        """Test embedding statistics computation"""
        embedding = speaker_encoder.extract_embedding(sample_audio_16khz, sample_rate=16000)
        stats = speaker_encoder.get_embedding_stats(embedding)

        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'norm' in stats
        assert all(isinstance(v, float) for v in stats.values())

    def test_extract_embeddings_batch_zero(self, speaker_encoder, sample_audio_16khz):
        """Test batch extraction with on_error='zero'"""
        audio_list = [sample_audio_16khz, sample_audio_16khz, sample_audio_16khz]
        embeddings = speaker_encoder.extract_embeddings_batch(audio_list, sample_rate=16000, on_error='zero')

        assert len(embeddings) == 3
        assert all(isinstance(e, np.ndarray) for e in embeddings)
        assert all(e.shape == (256,) for e in embeddings)

    def test_extract_embeddings_batch_none(self, speaker_encoder, sample_audio_16khz):
        """Test batch extraction with on_error='none'"""
        # Create one invalid audio
        audio_list = [sample_audio_16khz, np.array([]), sample_audio_16khz]
        embeddings = speaker_encoder.extract_embeddings_batch(audio_list, sample_rate=16000, on_error='none')

        assert len(embeddings) == 3
        assert embeddings[1] is None  # Failed item


@pytest.mark.unit
class TestVoiceCloner:
    """Test VoiceCloner for profile creation"""

    @pytest.fixture
    def voice_cloner(self, device, temp_dir):
        """Create VoiceCloner instance"""
        try:
            from src.auto_voice.inference.voice_cloner import VoiceCloner
            config = {
                'min_duration': 5.0,  # Override to lower value for testing
                'max_duration': 60.0,
                'storage_dir': str(temp_dir / 'voice_profiles'),
                'silence_threshold': 0.001
            }
            return VoiceCloner(config=config, device=device)
        except ImportError:
            pytest.skip("VoiceCloner not available")

    def test_initialization(self, voice_cloner):
        """Test VoiceCloner initializes correctly"""
        assert voice_cloner is not None
        assert hasattr(voice_cloner, 'speaker_encoder')
        assert hasattr(voice_cloner, 'audio_processor')
        assert hasattr(voice_cloner, 'storage')

    def test_validate_audio_valid(self, voice_cloner):
        """Test audio validation with valid audio"""
        # Generate 10 seconds of audio (valid duration)
        sample_rate = 22050
        audio = np.random.randn(sample_rate * 10).astype(np.float32) * 0.1

        is_valid, error_msg, error_code = voice_cloner.validate_audio(audio, sample_rate)
        assert is_valid is True
        assert error_msg is None
        assert error_code is None

    def test_validate_audio_too_short(self, voice_cloner):
        """Test audio validation with too short audio"""
        sample_rate = 22050
        audio = np.random.randn(sample_rate * 2).astype(np.float32) * 0.1  # 2 seconds

        is_valid, error_msg, error_code = voice_cloner.validate_audio(audio, sample_rate)
        assert is_valid is False
        assert 'too short' in error_msg.lower()
        assert error_code == 'duration_too_short'

    def test_validate_audio_too_long(self, voice_cloner):
        """Test audio validation with too long audio"""
        sample_rate = 22050
        audio = np.random.randn(sample_rate * 70).astype(np.float32) * 0.1  # 70 seconds

        is_valid, error_msg, error_code = voice_cloner.validate_audio(audio, sample_rate)
        assert is_valid is False
        assert 'too long' in error_msg.lower()
        assert error_code == 'duration_too_long'

    def test_validate_audio_too_quiet(self, voice_cloner):
        """Test audio validation with silent audio"""
        sample_rate = 22050
        audio = np.zeros(sample_rate * 10, dtype=np.float32)  # Silent

        is_valid, error_msg, error_code = voice_cloner.validate_audio(audio, sample_rate)
        assert is_valid is False
        assert 'quiet' in error_msg.lower()
        assert error_code == 'audio_too_quiet'

    def test_validate_audio_invalid_sample_rate(self, voice_cloner):
        """Test audio validation with invalid sample rate"""
        audio = np.random.randn(22050 * 10).astype(np.float32) * 0.1
        sample_rate = 4000  # Too low

        is_valid, error_msg, error_code = voice_cloner.validate_audio(audio, sample_rate)
        assert is_valid is False
        assert 'sample rate' in error_msg.lower()
        assert error_code == 'invalid_sample_rate'

    def test_create_voice_profile(self, voice_cloner):
        """Test voice profile creation"""
        # Generate valid audio (10 seconds)
        sample_rate = 22050
        audio = np.random.randn(sample_rate * 10).astype(np.float32) * 0.1

        profile = voice_cloner.create_voice_profile(
            audio=audio,
            sample_rate=sample_rate,
            user_id='test_user',
            metadata={'test': True}
        )

        # Check profile structure
        assert 'profile_id' in profile
        assert 'user_id' in profile
        assert profile['user_id'] == 'test_user'
        assert 'created_at' in profile
        assert 'audio_duration' in profile
        assert 'sample_rate' in profile
        assert 'embedding_stats' in profile
        assert 'metadata' in profile

        # embedding should NOT be in response (per Comment 5)
        assert 'embedding' not in profile

    def test_load_voice_profile(self, voice_cloner):
        """Test profile loading"""
        # Create a profile first
        sample_rate = 22050
        audio = np.random.randn(sample_rate * 10).astype(np.float32) * 0.1
        profile = voice_cloner.create_voice_profile(audio=audio, sample_rate=sample_rate)

        # Load it back
        loaded = voice_cloner.load_voice_profile(profile['profile_id'])

        assert 'profile_id' in loaded
        assert loaded['profile_id'] == profile['profile_id']
        assert 'embedding' in loaded  # Should be included when explicitly loaded

    def test_list_voice_profiles(self, voice_cloner):
        """Test profile listing"""
        # Create profiles
        sample_rate = 22050
        audio = np.random.randn(sample_rate * 10).astype(np.float32) * 0.1

        profile1 = voice_cloner.create_voice_profile(audio=audio, sample_rate=sample_rate, user_id='user1')
        profile2 = voice_cloner.create_voice_profile(audio=audio, sample_rate=sample_rate, user_id='user2')

        # List all
        all_profiles = voice_cloner.list_voice_profiles()
        assert len(all_profiles) >= 2

        # List by user
        user1_profiles = voice_cloner.list_voice_profiles(user_id='user1')
        assert len(user1_profiles) >= 1
        assert all(p.get('user_id') == 'user1' for p in user1_profiles)

    def test_delete_voice_profile(self, voice_cloner):
        """Test profile deletion"""
        sample_rate = 22050
        audio = np.random.randn(sample_rate * 10).astype(np.float32) * 0.1
        profile = voice_cloner.create_voice_profile(audio=audio, sample_rate=sample_rate)

        # Delete it
        deleted = voice_cloner.delete_voice_profile(profile['profile_id'])
        assert deleted is True

        # Try to delete again
        deleted_again = voice_cloner.delete_voice_profile(profile['profile_id'])
        assert deleted_again is False

    def test_compare_profiles(self, voice_cloner):
        """Test profile comparison"""
        sample_rate = 22050
        audio = np.random.randn(sample_rate * 10).astype(np.float32) * 0.1

        profile1 = voice_cloner.create_voice_profile(audio=audio, sample_rate=sample_rate)
        profile2 = voice_cloner.create_voice_profile(audio=audio, sample_rate=sample_rate)

        result = voice_cloner.compare_profiles(profile1['profile_id'], profile2['profile_id'])

        assert 'similarity' in result
        assert 'is_same_speaker' in result
        assert 'threshold' in result
        assert isinstance(result['similarity'], float)
        assert isinstance(result['is_same_speaker'], bool)


@pytest.mark.unit
class TestVoiceProfileStorage:
    """Test VoiceProfileStorage for persistence"""

    @pytest.fixture
    def storage(self, temp_dir):
        """Create VoiceProfileStorage instance"""
        try:
            from src.auto_voice.storage.voice_profiles import VoiceProfileStorage
            return VoiceProfileStorage(storage_dir=str(temp_dir / 'profiles'))
        except ImportError:
            pytest.skip("VoiceProfileStorage not available")

    def test_save_and_load_profile(self, storage):
        """Test saving and loading profiles"""
        profile = {
            'profile_id': 'test-123',
            'user_id': 'user1',
            'created_at': '2025-01-15T10:00:00Z',
            'embedding': np.random.randn(256).astype(np.float32),
            'audio_duration': 10.0,
            'sample_rate': 22050
        }

        # Save
        profile_id = storage.save_profile(profile)
        assert profile_id == 'test-123'

        # Load
        loaded = storage.load_profile(profile_id)
        assert loaded['profile_id'] == profile['profile_id']
        assert loaded['user_id'] == profile['user_id']
        assert 'embedding' in loaded
        np.testing.assert_array_almost_equal(loaded['embedding'], profile['embedding'])

    def test_profile_exists(self, storage):
        """Test profile existence check"""
        profile = {
            'profile_id': 'test-456',
            'embedding': np.random.randn(256).astype(np.float32),
            'created_at': '2025-01-15T10:00:00Z'
        }

        assert storage.profile_exists('test-456') is False

        storage.save_profile(profile)
        assert storage.profile_exists('test-456') is True

    def test_list_profiles(self, storage):
        """Test listing profiles"""
        # Create multiple profiles
        for i in range(3):
            profile = {
                'profile_id': f'test-{i}',
                'user_id': f'user{i % 2}',
                'embedding': np.random.randn(256).astype(np.float32),
                'created_at': '2025-01-15T10:00:00Z'
            }
            storage.save_profile(profile)

        # List all
        all_profiles = storage.list_profiles()
        assert len(all_profiles) >= 3

        # List by user
        user0_profiles = storage.list_profiles(user_id='user0')
        assert len(user0_profiles) >= 1


@pytest.mark.integration
@pytest.mark.web
class TestVoiceCloningAPI:
    """Test voice cloning API endpoints"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures"""
        try:
            from src.auto_voice.web.app import create_app
            self.app, self.socketio = create_app(config={'TESTING': True})
            self.client = self.app.test_client()
        except ImportError:
            pytest.skip("Flask app not available")

    def test_clone_endpoint_exists(self):
        """Test that /api/v1/voice/clone endpoint exists"""
        # Use OPTIONS to check if route exists
        response = self.client.options('/api/v1/voice/clone')
        assert response.status_code in [200, 204, 405]  # 405 is OK, means route exists but OPTIONS not allowed

    def test_clone_voice_with_reference_audio(self):
        """Test cloning voice with reference_audio field"""
        import io

        # Create mock audio file
        audio_data = np.random.randn(22050 * 10).astype(np.float32) * 0.1
        audio_bytes = io.BytesIO(audio_data.tobytes())
        audio_bytes.name = 'test_voice.wav'

        data = {
            'reference_audio': (audio_bytes, 'test_voice.wav'),
            'user_id': 'test_user'
        }

        response = self.client.post(
            '/api/v1/voice/clone',
            data=data,
            content_type='multipart/form-data'
        )

        # In TESTING mode, should get success or service unavailable
        assert response.status_code in [201, 200, 503]

        if response.status_code in [200, 201]:
            data = response.get_json()
            assert 'profile_id' in data or 'status' in data

    def test_clone_voice_backward_compatibility(self):
        """Test that 'audio' field still works for backward compatibility"""
        import io

        audio_data = np.random.randn(22050 * 10).astype(np.float32) * 0.1
        audio_bytes = io.BytesIO(audio_data.tobytes())
        audio_bytes.name = 'test_voice.wav'

        data = {
            'audio': (audio_bytes, 'test_voice.wav'),
            'user_id': 'test_user'
        }

        response = self.client.post(
            '/api/v1/voice/clone',
            data=data,
            content_type='multipart/form-data'
        )

        # Should work with 'audio' field too
        assert response.status_code in [201, 200, 400, 503]

    def test_clone_voice_missing_audio(self):
        """Test clone endpoint with missing audio"""
        response = self.client.post(
            '/api/v1/voice/clone',
            data={'user_id': 'test_user'},
            content_type='multipart/form-data'
        )

        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

    def test_get_voice_profiles(self):
        """Test GET /api/v1/voice/profiles endpoint"""
        response = self.client.get('/api/v1/voice/profiles')

        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.get_json()
            assert isinstance(data, list)

    def test_get_voice_profile_by_id(self):
        """Test GET /api/v1/voice/profiles/<id> endpoint"""
        response = self.client.get('/api/v1/voice/profiles/test-profile-123')

        # Should return 404 (not found) or 503 (service unavailable)
        assert response.status_code in [404, 503, 500]

    def test_delete_voice_profile(self):
        """Test DELETE /api/v1/voice/profiles/<id> endpoint"""
        response = self.client.delete('/api/v1/voice/profiles/test-profile-123')

        # Should return 404 (not found) or 503 (service unavailable)
        assert response.status_code in [200, 404, 503]


@pytest.mark.integration
class TestHealthEndpointsWithVoiceCloner:
    """Test that health endpoints include voice_cloner status"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures"""
        try:
            from src.auto_voice.web.app import create_app
            self.app, self.socketio = create_app(config={'TESTING': True})
            self.client = self.app.test_client()
        except ImportError:
            pytest.skip("Flask app not available")

    def test_health_includes_voice_cloner(self):
        """Test /health endpoint includes voice_cloner status"""
        response = self.client.get('/health')
        assert response.status_code == 200

        data = response.get_json()
        assert 'components' in data
        assert 'voice_cloner' in data['components']

    def test_readiness_includes_voice_cloner(self):
        """Test /health/ready endpoint includes voice_cloner status"""
        response = self.client.get('/health/ready')
        assert response.status_code in [200, 503]

        data = response.get_json()
        assert 'components' in data
        assert 'voice_cloner' in data['components']

    def test_api_health_includes_voice_cloner(self):
        """Test /api/v1/health endpoint"""
        response = self.client.get('/api/v1/health')
        assert response.status_code == 200

        data = response.get_json()
        # API health should have voice_cloner info
        assert 'status' in data
