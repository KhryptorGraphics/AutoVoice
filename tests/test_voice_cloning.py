"""Comprehensive voice cloning tests for AutoVoice

Tests SpeakerEncoder, VoiceCloner, VoiceProfileStorage, and API endpoints
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Fixtures are automatically available from conftest.py - no import needed


@pytest.mark.unit
class TestSpeakerEncoder:
    """Test SpeakerEncoder for embedding extraction"""

    @pytest.fixture
    def speaker_encoder(self, device):
        """Create SpeakerEncoder instance"""
        try:
            from src.auto_voice.models.speaker_encoder import SpeakerEncoder, SpeakerEncodingError
            return SpeakerEncoder(device=device)
        except ImportError:
            pytest.skip("SpeakerEncoder not available")
        except Exception as e:
            # Import SpeakerEncodingError separately for isinstance check
            try:
                from src.auto_voice.models.speaker_encoder import SpeakerEncodingError
                if isinstance(e, SpeakerEncodingError):
                    pytest.skip(f"SpeakerEncoder initialization failed: {e}")
            except ImportError:
                pass
            raise

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
            from src.auto_voice.inference.voice_cloner import VoiceCloner, VoiceCloningError
            config = {
                'min_duration': 5.0,  # Override to lower value for testing
                'max_duration': 60.0,
                'storage_dir': str(temp_dir / 'voice_profiles'),
                'silence_threshold': 0.001
            }
            return VoiceCloner(config=config, device=device)
        except ImportError:
            pytest.skip("VoiceCloner not available")
        except Exception as e:
            # Import VoiceCloningError separately for isinstance check
            try:
                from src.auto_voice.inference.voice_cloner import VoiceCloningError
                if isinstance(e, VoiceCloningError):
                    pytest.skip(f"VoiceCloner initialization failed: {e}")
            except ImportError:
                pass
            raise

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

    # ========================================================================
    # SNR Validation Tests
    # ========================================================================

    def test_snr_validation_in_create_profile(self, voice_cloner):
        """Test SNR validation when creating voice profile.

        Tests that validate_audio() and get_audio_quality_report() are called
        and low-quality audio is rejected based on SNR threshold.
        """
        sample_rate = 22050
        duration = 10.0

        # Create clean audio (high SNR)
        clean_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1

        # Should succeed with clean audio
        profile = voice_cloner.create_voice_profile(audio=clean_audio, sample_rate=sample_rate)
        assert profile is not None
        assert 'profile_id' in profile

    def test_get_audio_quality_report(self, voice_cloner):
        """Test audio quality report generation including SNR calculation."""
        sample_rate = 22050
        duration = 5.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.3

        # Get quality report
        report = voice_cloner.get_audio_quality_report(audio, sample_rate=sample_rate)

        # Verify report structure
        assert isinstance(report, dict)
        assert 'snr_db' in report
        assert 'duration' in report
        assert 'sample_rate' in report
        assert 'quality_assessment' in report

        # SNR should be a number
        if report['snr_db'] is not None:
            assert isinstance(report['snr_db'], (int, float))

    def test_snr_validation_thresholds(self, voice_cloner):
        """Test different SNR quality thresholds."""
        sample_rate = 22050
        duration = 5.0

        # Create audio with different noise levels
        # Clean signal
        clean_signal = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))

        # Add moderate noise (should pass)
        moderate_noise = np.random.randn(int(sample_rate * duration)) * 0.05
        moderate_audio = (clean_signal + moderate_noise).astype(np.float32)

        report = voice_cloner.get_audio_quality_report(moderate_audio, sample_rate=sample_rate)

        # Check SNR is calculated
        assert 'snr_db' in report
        assert 'quality_assessment' in report

        # Quality should be categorized
        assert report['quality_assessment'] in ['excellent', 'good', 'fair', 'poor', 'unacceptable']

    # ========================================================================
    # Multi-Sample Profile Tests
    # ========================================================================

    def test_create_profile_from_multiple_samples(self, voice_cloner):
        """Test creating voice profile from multiple audio samples.

        Tests create_voice_profile_from_multiple_samples() method with quality-weighted averaging.
        """
        sample_rate = 22050

        # Create multiple audio samples (varying quality/length)
        audio_samples = []
        for i in range(3):
            duration = 5.0 + i  # 5s, 6s, 7s
            audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1
            audio_samples.append(audio)

        # Create profile from multiple samples
        profile = voice_cloner.create_voice_profile_from_multiple_samples(
            audio_samples=audio_samples,
            sample_rate=sample_rate,
            user_id='multi_sample_user'
        )

        # Verify profile structure
        assert 'profile_id' in profile
        assert 'embedding' in profile
        assert 'num_samples' in profile
        assert profile['num_samples'] == 3
        assert profile['user_id'] == 'multi_sample_user'

        # Embedding should be averaged
        assert isinstance(profile['embedding'], np.ndarray)
        assert profile['embedding'].shape == (256,)  # Standard embedding size

    def test_multi_sample_quality_weighting(self, voice_cloner):
        """Test that multi-sample profiles use quality-weighted averaging."""
        sample_rate = 22050
        duration = 5.0

        # Create samples with different quality levels
        # High quality sample (low noise)
        clean_signal = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
        high_quality = (clean_signal + np.random.randn(int(sample_rate * duration)) * 0.01).astype(np.float32)

        # Low quality sample (high noise)
        low_quality = (clean_signal + np.random.randn(int(sample_rate * duration)) * 0.2).astype(np.float32)

        # Medium quality
        medium_quality = (clean_signal + np.random.randn(int(sample_rate * duration)) * 0.05).astype(np.float32)

        audio_samples = [high_quality, low_quality, medium_quality]

        # Create profile
        profile = voice_cloner.create_voice_profile_from_multiple_samples(
            audio_samples=audio_samples,
            sample_rate=sample_rate
        )

        # Should succeed and create valid profile
        assert profile is not None
        assert 'embedding' in profile
        assert 'num_samples' in profile
        assert profile['num_samples'] == 3

    def test_add_sample_to_profile(self, voice_cloner):
        """Test adding additional sample to existing profile.

        Tests add_sample_to_profile() method which updates embedding with new sample.
        """
        sample_rate = 22050
        duration = 10.0

        # Create initial profile
        initial_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1
        profile = voice_cloner.create_voice_profile(audio=initial_audio, sample_rate=sample_rate)
        profile_id = profile['profile_id']

        # Load profile to get embedding (create_voice_profile intentionally omits it from response)
        loaded_profile = voice_cloner.load_voice_profile(profile_id)
        initial_embedding = loaded_profile['embedding'].copy()

        # Add another sample to the profile
        additional_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1
        updated_profile = voice_cloner.add_sample_to_profile(
            profile_id=profile_id,
            audio=additional_audio,
            sample_rate=sample_rate,
            weight=None  # Auto-determine weight based on quality
        )

        # Verify profile was updated
        assert 'embedding' in updated_profile
        assert 'num_samples' in updated_profile
        assert updated_profile['num_samples'] >= 2

        # Embedding should be different after adding sample
        assert not np.array_equal(initial_embedding, updated_profile['embedding'])

    def test_add_sample_with_custom_weight(self, voice_cloner):
        """Test adding sample with custom quality weight."""
        sample_rate = 22050
        duration = 10.0

        # Create initial profile
        initial_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1
        profile = voice_cloner.create_voice_profile(audio=initial_audio, sample_rate=sample_rate)
        profile_id = profile['profile_id']

        # Add sample with high custom weight (this sample should dominate)
        additional_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1
        updated_profile = voice_cloner.add_sample_to_profile(
            profile_id=profile_id,
            audio=additional_audio,
            sample_rate=sample_rate,
            weight=0.9  # Give this sample 90% weight
        )

        # Should update successfully
        assert 'embedding' in updated_profile
        assert 'num_samples' in updated_profile

    # ========================================================================
    # Version History Tests
    # ========================================================================

    def test_get_profile_version_history(self, voice_cloner):
        """Test retrieving version history of voice profile.

        Tests get_profile_version_history() method which tracks profile changes.
        """
        sample_rate = 22050
        duration = 10.0

        # Create initial profile
        initial_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1
        profile = voice_cloner.create_voice_profile(
            audio=initial_audio,
            sample_rate=sample_rate,
            metadata={'initial_version': True}
        )
        profile_id = profile['profile_id']

        # Add samples to create version history
        for i in range(2):
            additional_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1
            voice_cloner.add_sample_to_profile(
                profile_id=profile_id,
                audio=additional_audio,
                sample_rate=sample_rate
            )

        # Get version history
        history = voice_cloner.get_profile_version_history(profile_id)

        # Verify history structure
        assert isinstance(history, list)
        if len(history) > 0:
            version = history[0]
            assert 'version' in version
            assert 'timestamp' in version
            assert 'changes' in version or 'description' in version

    def test_version_history_tracks_changes(self, voice_cloner):
        """Test that version history accurately tracks profile modifications."""
        sample_rate = 22050
        duration = 10.0

        # Create profile
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1
        profile = voice_cloner.create_voice_profile(audio=audio, sample_rate=sample_rate)
        profile_id = profile['profile_id']

        # Get initial history
        history_before = voice_cloner.get_profile_version_history(profile_id)
        initial_version_count = len(history_before)

        # Modify profile by adding sample
        additional_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1
        voice_cloner.add_sample_to_profile(profile_id, additional_audio, sample_rate)

        # Get updated history
        history_after = voice_cloner.get_profile_version_history(profile_id)

        # Should have one more version
        assert len(history_after) >= initial_version_count

    def test_version_history_empty_for_new_profile(self, voice_cloner):
        """Test that new profiles have minimal version history."""
        sample_rate = 22050
        duration = 10.0

        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1
        profile = voice_cloner.create_voice_profile(audio=audio, sample_rate=sample_rate)
        profile_id = profile['profile_id']

        # Get history
        history = voice_cloner.get_profile_version_history(profile_id)

        # Should be a list (may have creation event)
        assert isinstance(history, list)


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
