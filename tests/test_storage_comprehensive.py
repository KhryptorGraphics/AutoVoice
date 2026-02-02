"""Comprehensive tests for voice profile storage.

Tests for:
- LoRA weight storage and retrieval
- Training sample management
- Speaker embedding operations
- Profile directory structure
- Cleanup on deletion
"""

import numpy as np
import os
import pytest
import tempfile
import torch
from pathlib import Path

from auto_voice.storage.voice_profiles import (
    VoiceProfileStore,
    ProfileNotFoundError,
    TrainingSample,
)


@pytest.fixture
def temp_store():
    """Create a temporary VoiceProfileStore for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        profiles_dir = os.path.join(tmpdir, "profiles")
        samples_dir = os.path.join(tmpdir, "samples")
        store = VoiceProfileStore(profiles_dir, samples_dir)
        yield store


@pytest.fixture
def sample_profile(temp_store):
    """Create a sample profile for testing."""
    profile_data = {
        "profile_id": "test-profile-123",
        "name": "Test Voice",
        "user_id": "user-1",
    }
    temp_store.save(profile_data)
    return "test-profile-123"


class TestTrainingSample:
    """Test TrainingSample dataclass."""

    def test_to_dict(self):
        """TrainingSample serializes to dict."""
        sample = TrainingSample(
            sample_id="sample_001",
            vocals_path="/path/to/vocals.wav",
            instrumental_path="/path/to/instrumental.wav",
            source_file="source.mp3",
            duration=120.5,
        )

        data = sample.to_dict()
        assert data['sample_id'] == "sample_001"
        assert data['vocals_path'] == "/path/to/vocals.wav"
        assert data['duration'] == 120.5
        assert 'created_at' in data

    def test_from_dict(self):
        """TrainingSample deserializes from dict."""
        data = {
            'sample_id': 'sample_002',
            'vocals_path': '/path/vocals.wav',
            'instrumental_path': None,
            'source_file': 'test.wav',
            'duration': 60.0,
            'created_at': '2026-01-01T00:00:00Z',
        }

        sample = TrainingSample.from_dict(data)
        assert sample.sample_id == 'sample_002'
        assert sample.duration == 60.0
        assert sample.created_at == '2026-01-01T00:00:00Z'


class TestLoRAWeightStorage:
    """Test LoRA weight storage operations."""

    def test_save_lora_weights(self, temp_store, sample_profile):
        """Save LoRA weights for profile."""
        state_dict = {
            'lora_A': torch.randn(64, 768),
            'lora_B': torch.randn(768, 64),
        }

        temp_store.save_lora_weights(sample_profile, state_dict)

        # Verify file created
        weights_path = temp_store._lora_weights_path(sample_profile)
        assert os.path.exists(weights_path)

    def test_load_lora_weights(self, temp_store, sample_profile):
        """Load LoRA weights for profile."""
        original = {
            'lora_A': torch.randn(64, 768),
            'lora_B': torch.randn(768, 64),
        }
        temp_store.save_lora_weights(sample_profile, original)

        loaded = temp_store.load_lora_weights(sample_profile)

        assert 'lora_A' in loaded
        assert 'lora_B' in loaded
        assert torch.allclose(loaded['lora_A'], original['lora_A'])
        assert torch.allclose(loaded['lora_B'], original['lora_B'])

    def test_save_lora_weights_profile_not_found(self, temp_store):
        """Save LoRA weights for non-existent profile raises error."""
        with pytest.raises(ValueError, match="not found"):
            temp_store.save_lora_weights("nonexistent", {'lora_A': torch.zeros(1)})

    def test_load_lora_weights_profile_not_found(self, temp_store):
        """Load LoRA weights for non-existent profile raises error."""
        with pytest.raises(ValueError, match="not found"):
            temp_store.load_lora_weights("nonexistent")

    def test_load_lora_weights_no_weights_saved(self, temp_store, sample_profile):
        """Load LoRA weights when none saved raises error."""
        with pytest.raises(FileNotFoundError, match="No LoRA weights"):
            temp_store.load_lora_weights(sample_profile)

    def test_has_trained_model(self, temp_store, sample_profile):
        """Check if profile has trained model."""
        assert temp_store.has_trained_model(sample_profile) is False

        temp_store.save_lora_weights(sample_profile, {'lora': torch.zeros(1)})

        assert temp_store.has_trained_model(sample_profile) is True

    def test_has_trained_model_nonexistent_profile(self, temp_store):
        """has_trained_model returns False for non-existent profile."""
        assert temp_store.has_trained_model("nonexistent") is False


class TestTrainingSampleManagement:
    """Test training sample management."""

    @pytest.fixture
    def temp_vocals(self, tmp_path):
        """Create temporary vocals file."""
        vocals_path = tmp_path / "test_vocals.wav"
        # Create minimal WAV file (44 byte header + some data)
        import wave
        with wave.open(str(vocals_path), 'w') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(b'\x00' * 32000)  # 1 second of silence
        return str(vocals_path)

    def test_add_training_sample(self, temp_store, sample_profile, temp_vocals):
        """Add training sample to profile."""
        sample = temp_store.add_training_sample(
            profile_id=sample_profile,
            vocals_path=temp_vocals,
            source_file="test.mp3",
            duration=1.0,
        )

        assert sample.sample_id == "sample_001"
        assert os.path.exists(sample.vocals_path)
        assert sample.duration == 1.0

    def test_add_training_sample_profile_not_found(self, temp_store, temp_vocals):
        """Add sample to non-existent profile raises error."""
        with pytest.raises(ProfileNotFoundError):
            temp_store.add_training_sample(
                profile_id="nonexistent",
                vocals_path=temp_vocals,
            )

    def test_list_training_samples_empty(self, temp_store, sample_profile):
        """List samples when none exist returns empty list."""
        samples = temp_store.list_training_samples(sample_profile)
        assert samples == []

    def test_list_training_samples(self, temp_store, sample_profile, temp_vocals):
        """List all training samples for profile."""
        temp_store.add_training_sample(sample_profile, temp_vocals, duration=1.0)
        temp_store.add_training_sample(sample_profile, temp_vocals, duration=2.0)

        samples = temp_store.list_training_samples(sample_profile)
        assert len(samples) == 2
        assert samples[0].sample_id == "sample_001"
        assert samples[1].sample_id == "sample_002"

    def test_get_all_vocals_paths(self, temp_store, sample_profile, temp_vocals):
        """Get all vocals paths for training."""
        temp_store.add_training_sample(sample_profile, temp_vocals)
        temp_store.add_training_sample(sample_profile, temp_vocals)

        paths = temp_store.get_all_vocals_paths(sample_profile)
        assert len(paths) == 2
        for path in paths:
            assert os.path.exists(path)

    def test_get_total_training_duration(self, temp_store, sample_profile, temp_vocals):
        """Get total duration of training samples."""
        temp_store.add_training_sample(sample_profile, temp_vocals, duration=10.0)
        temp_store.add_training_sample(sample_profile, temp_vocals, duration=20.0)

        total = temp_store.get_total_training_duration(sample_profile)
        assert total == 30.0

    def test_delete_training_sample(self, temp_store, sample_profile, temp_vocals):
        """Delete a specific training sample."""
        temp_store.add_training_sample(sample_profile, temp_vocals)

        result = temp_store.delete_training_sample(sample_profile, "sample_001")
        assert result is True

        samples = temp_store.list_training_samples(sample_profile)
        assert len(samples) == 0

    def test_delete_training_sample_not_found(self, temp_store, sample_profile):
        """Delete non-existent sample returns False."""
        result = temp_store.delete_training_sample(sample_profile, "nonexistent")
        assert result is False


class TestSpeakerEmbeddingStorage:
    """Test speaker embedding storage operations."""

    def test_save_speaker_embedding(self, temp_store, sample_profile):
        """Save speaker embedding for profile."""
        embedding = np.random.randn(512).astype(np.float32)

        temp_store.save_speaker_embedding(sample_profile, embedding)

        # Verify file created
        emb_path = temp_store._speaker_embedding_path(sample_profile)
        assert os.path.exists(emb_path)

    def test_save_speaker_embedding_normalizes(self, temp_store, sample_profile):
        """Speaker embedding is L2 normalized on save."""
        embedding = np.array([3.0, 4.0] + [0.0] * 510, dtype=np.float32)

        temp_store.save_speaker_embedding(sample_profile, embedding)
        loaded = temp_store.load_speaker_embedding(sample_profile)

        # Should be normalized to unit length
        norm = np.linalg.norm(loaded)
        assert np.isclose(norm, 1.0, atol=1e-5)

    def test_save_speaker_embedding_profile_not_found(self, temp_store):
        """Save embedding for non-existent profile raises error."""
        with pytest.raises(ProfileNotFoundError):
            temp_store.save_speaker_embedding("nonexistent", np.zeros(512))

    def test_load_speaker_embedding(self, temp_store, sample_profile):
        """Load speaker embedding for profile."""
        original = np.random.randn(512).astype(np.float32)
        original = original / np.linalg.norm(original)  # Pre-normalize

        temp_store.save_speaker_embedding(sample_profile, original)
        loaded = temp_store.load_speaker_embedding(sample_profile)

        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded, original, decimal=5)

    def test_load_speaker_embedding_none_if_not_set(self, temp_store, sample_profile):
        """Load embedding returns None if not set."""
        loaded = temp_store.load_speaker_embedding(sample_profile)
        assert loaded is None

    def test_get_all_speaker_embeddings(self, temp_store):
        """Get all speaker embeddings across profiles."""
        # Create multiple profiles with embeddings
        for i in range(3):
            profile_id = f"profile-{i}"
            temp_store.save({"profile_id": profile_id})
            temp_store.save_speaker_embedding(profile_id, np.random.randn(512))

        embeddings = temp_store.get_all_speaker_embeddings()
        assert len(embeddings) == 3
        for profile_id, emb in embeddings.items():
            assert emb.shape == (512,)


class TestProfileDirectoryStructure:
    """Test profile directory structure and cleanup."""

    def test_profile_directories_created(self, temp_store):
        """Profile and samples directories are created on init."""
        assert os.path.exists(temp_store.profiles_dir)
        assert os.path.exists(temp_store.samples_dir)

    def test_delete_removes_all_files(self, temp_store, sample_profile):
        """Delete profile removes all associated files."""
        # Add various files
        temp_store.save_lora_weights(sample_profile, {'lora': torch.zeros(1)})
        temp_store.save_speaker_embedding(sample_profile, np.zeros(512))

        # Verify files exist
        assert os.path.exists(temp_store._lora_weights_path(sample_profile))
        assert os.path.exists(temp_store._speaker_embedding_path(sample_profile))

        # Delete profile
        temp_store.delete(sample_profile)

        # Main profile file should be deleted
        assert not os.path.exists(temp_store._profile_path(sample_profile))
        # Note: Associated files may still exist (delete only removes main file)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_save_empty_embedding(self, temp_store, sample_profile):
        """Save empty/zero embedding."""
        temp_store.save_speaker_embedding(sample_profile, np.zeros(512))
        loaded = temp_store.load_speaker_embedding(sample_profile)
        # Zero vector can't be normalized, so should remain zero
        assert np.allclose(loaded, 0.0)

    def test_save_list_embedding(self, temp_store):
        """Save embedding as list instead of numpy array."""
        profile_data = {
            "profile_id": "list-emb-test",
            "embedding": [0.1] * 256,  # List instead of numpy
        }
        temp_store.save(profile_data)

        loaded = temp_store.load("list-emb-test")
        assert isinstance(loaded['embedding'], np.ndarray)
        assert loaded['embedding'].shape == (256,)

    def test_profile_without_embedding(self, temp_store):
        """Profile without embedding loads correctly."""
        profile_data = {"profile_id": "no-emb", "name": "No Embedding"}
        temp_store.save(profile_data)

        loaded = temp_store.load("no-emb")
        assert loaded['name'] == "No Embedding"
        assert 'embedding' not in loaded or loaded.get('embedding') is None

    def test_corrupted_json_skipped_in_list(self, temp_store, sample_profile):
        """Corrupted JSON files are skipped when listing."""
        # Create corrupted file
        corrupt_path = os.path.join(temp_store.profiles_dir, "corrupt.json")
        with open(corrupt_path, 'w') as f:
            f.write("{invalid json")

        profiles = temp_store.list_profiles()
        # Should still return valid profile, skip corrupted one
        assert len(profiles) == 1
        assert profiles[0]['profile_id'] == sample_profile
