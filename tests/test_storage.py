"""Tests for voice profile storage."""
import numpy as np
import pytest

from auto_voice.storage.voice_profiles import VoiceProfileStore, ProfileNotFoundError


class TestProfileNotFoundError:
    """ProfileNotFoundError exception tests."""

    @pytest.mark.smoke
    def test_is_exception(self):
        with pytest.raises(ProfileNotFoundError):
            raise ProfileNotFoundError("test")

    def test_message(self):
        try:
            raise ProfileNotFoundError("profile-123 not found")
        except ProfileNotFoundError as e:
            assert "profile-123" in str(e)


class TestVoiceProfileStore:
    """VoiceProfileStore CRUD tests."""

    @pytest.mark.smoke
    def test_save_and_load(self, profile_store):
        profile = {
            'profile_id': 'test-001',
            'user_id': 'user-1',
            'audio_duration': 10.5,
            'vocal_range': {'min_hz': 100, 'max_hz': 500},
            'embedding': np.random.randn(256).astype(np.float32),
        }
        profile_store.save(profile)
        loaded = profile_store.load('test-001')
        assert loaded['profile_id'] == 'test-001'
        assert loaded['user_id'] == 'user-1'
        assert loaded['audio_duration'] == 10.5

    def test_load_nonexistent_raises(self, profile_store):
        with pytest.raises(ProfileNotFoundError):
            profile_store.load('nonexistent-id')

    def test_save_generates_id(self, profile_store):
        profile = {'user_id': 'user-1', 'embedding': np.zeros(256)}
        profile_id = profile_store.save(profile)
        assert profile_id is not None
        assert len(profile_id) > 0

    def test_embedding_preserved(self, profile_store):
        emb = np.random.randn(256).astype(np.float32)
        profile_store.save({'profile_id': 'emb-test', 'embedding': emb})
        loaded = profile_store.load('emb-test')
        np.testing.assert_array_almost_equal(loaded['embedding'], emb, decimal=5)

    def test_list_empty(self, profile_store):
        profiles = profile_store.list_profiles()
        assert profiles == []

    def test_list_all(self, profile_store):
        profile_store.save({'profile_id': 'p1', 'user_id': 'u1'})
        profile_store.save({'profile_id': 'p2', 'user_id': 'u2'})
        profiles = profile_store.list_profiles()
        assert len(profiles) == 2

    def test_list_by_user_id(self, profile_store):
        profile_store.save({'profile_id': 'p1', 'user_id': 'alice'})
        profile_store.save({'profile_id': 'p2', 'user_id': 'bob'})
        profile_store.save({'profile_id': 'p3', 'user_id': 'alice'})
        alice_profiles = profile_store.list_profiles(user_id='alice')
        assert len(alice_profiles) == 2

    def test_delete_existing(self, profile_store):
        profile_store.save({'profile_id': 'del-test'})
        assert profile_store.delete('del-test') is True
        with pytest.raises(ProfileNotFoundError):
            profile_store.load('del-test')

    def test_delete_nonexistent(self, profile_store):
        assert profile_store.delete('nonexistent') is False

    def test_exists(self, profile_store):
        profile_store.save({'profile_id': 'exists-test'})
        assert profile_store.exists('exists-test') is True
        assert profile_store.exists('nope') is False

    def test_created_at_auto_set(self, profile_store):
        profile_store.save({'profile_id': 'time-test'})
        loaded = profile_store.load('time-test')
        assert 'created_at' in loaded
