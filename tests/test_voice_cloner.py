"""Tests for VoiceCloner."""
import numpy as np
import pytest

from auto_voice.inference.voice_cloner import (
    VoiceCloner, InvalidAudioError, InsufficientQualityError, InconsistentSamplesError
)
from auto_voice.storage.voice_profiles import ProfileNotFoundError


class TestExceptions:
    """Voice cloner exception tests."""

    @pytest.mark.smoke
    def test_invalid_audio_error(self):
        with pytest.raises(InvalidAudioError):
            raise InvalidAudioError("bad audio")

    def test_insufficient_quality_error(self):
        err = InsufficientQualityError("low quality", details={'snr': -5})
        assert err.error_code == 'insufficient_quality'
        assert err.details == {'snr': -5}

    def test_inconsistent_samples_error(self):
        err = InconsistentSamplesError("different speakers", details={'similarity': 0.2})
        assert err.error_code == 'inconsistent_samples'
        assert err.details == {'similarity': 0.2}


class TestVoiceCloner:
    """VoiceCloner integration tests."""

    @pytest.mark.smoke
    def test_init(self, voice_cloner):
        assert voice_cloner is not None
        assert voice_cloner.store is not None

    def test_create_profile(self, voice_cloner, sample_audio_file):
        result = voice_cloner.create_voice_profile(audio=sample_audio_file, user_id='test-user')
        assert 'profile_id' in result
        assert result['user_id'] == 'test-user'
        assert result['audio_duration'] > 0
        assert 'vocal_range' in result
        assert 'created_at' in result

    def test_create_profile_nonexistent_file(self, voice_cloner):
        with pytest.raises(InvalidAudioError):
            voice_cloner.create_voice_profile(audio='/nonexistent/path.wav')

    def test_create_profile_too_short(self, voice_cloner, short_audio_file):
        with pytest.raises(InvalidAudioError, match="too short"):
            voice_cloner.create_voice_profile(audio=short_audio_file)

    def test_load_profile(self, voice_cloner, sample_audio_file):
        created = voice_cloner.create_voice_profile(audio=sample_audio_file)
        loaded = voice_cloner.load_voice_profile(created['profile_id'])
        assert loaded['profile_id'] == created['profile_id']

    def test_load_nonexistent_profile(self, voice_cloner):
        with pytest.raises(ProfileNotFoundError):
            voice_cloner.load_voice_profile('nonexistent-id')

    def test_list_profiles(self, voice_cloner, sample_audio_file):
        voice_cloner.create_voice_profile(audio=sample_audio_file, user_id='u1')
        profiles = voice_cloner.list_voice_profiles()
        assert len(profiles) >= 1

    def test_list_profiles_by_user(self, voice_cloner, sample_audio_file):
        voice_cloner.create_voice_profile(audio=sample_audio_file, user_id='alice')
        voice_cloner.create_voice_profile(audio=sample_audio_file, user_id='bob')
        alice = voice_cloner.list_voice_profiles(user_id='alice')
        assert all(p['user_id'] == 'alice' for p in alice)

    def test_delete_profile(self, voice_cloner, sample_audio_file):
        created = voice_cloner.create_voice_profile(audio=sample_audio_file)
        assert voice_cloner.delete_voice_profile(created['profile_id']) is True
        with pytest.raises(ProfileNotFoundError):
            voice_cloner.load_voice_profile(created['profile_id'])

    def test_delete_nonexistent(self, voice_cloner):
        assert voice_cloner.delete_voice_profile('nope') is False

    def test_compare_embeddings(self, voice_cloner):
        a = np.random.randn(256).astype(np.float32)
        b = a.copy()  # identical
        similarity = voice_cloner.compare_embeddings(a, b)
        assert abs(similarity - 1.0) < 0.01

    def test_compare_orthogonal_embeddings(self, voice_cloner):
        a = np.zeros(256, dtype=np.float32)
        a[0] = 1.0
        b = np.zeros(256, dtype=np.float32)
        b[1] = 1.0
        similarity = voice_cloner.compare_embeddings(a, b)
        assert abs(similarity) < 0.01
