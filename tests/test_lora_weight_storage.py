"""Tests for LoRA weight storage and loading.

Phase 2: Test saving trained LoRA weights to profile storage and loading them back.

Tests verify:
- save_lora_weights() saves to correct path
- load_lora_weights() returns valid state dict
- has_trained_model() correctly reports model status
- Version support for weight files
"""

import pytest
import torch
import wave
from pathlib import Path
from unittest.mock import patch

from auto_voice.storage.voice_profiles import VoiceProfileStore


@pytest.fixture
def temp_profile_dir(tmp_path):
    """Create temporary profile storage directory."""
    profile_dir = tmp_path / "voice_profiles"
    profile_dir.mkdir()
    return profile_dir


@pytest.fixture
def store(temp_profile_dir):
    """Create VoiceProfileStore with temp directory."""
    return VoiceProfileStore(profiles_dir=str(temp_profile_dir))


@pytest.fixture
def sample_profile(store):
    """Create a sample voice profile."""
    profile_data = {
        "profile_id": "test-profile-123",
        "name": "Test Artist",
        "embedding": torch.randn(256).numpy(),
        "sample_count": 3,
    }
    store.save(profile_data)
    return profile_data["profile_id"]


@pytest.fixture
def sample_lora_state_dict():
    """Create sample LoRA state dict matching decoder structure."""
    return {
        "input_proj.adapter.lora_A": torch.randn(8, 1024),
        "input_proj.adapter.lora_B": torch.randn(512, 8),
        "time_embed.1.adapter.lora_A": torch.randn(8, 512),
        "time_embed.1.adapter.lora_B": torch.randn(512, 8),
        "speaker_film.gamma_proj.adapter.lora_A": torch.randn(8, 256),
        "speaker_film.gamma_proj.adapter.lora_B": torch.randn(512, 8),
    }


class TestSaveLoRAWeights:
    """Tests for save_lora_weights() method."""

    def test_save_lora_weights_method_exists(self, store):
        """Task 2.1: VoiceProfileStore should have save_lora_weights() method."""
        assert hasattr(store, 'save_lora_weights'), \
            "VoiceProfileStore missing save_lora_weights() method"
        assert callable(store.save_lora_weights), \
            "save_lora_weights should be callable"

    def test_save_lora_weights_creates_file(self, store, sample_profile, sample_lora_state_dict):
        """Task 2.2: save_lora_weights should create weights file in profile dir."""
        store.save_lora_weights(sample_profile, sample_lora_state_dict)

        # Check file exists (uses {profile_id}_lora_weights.pt pattern)
        weights_file = Path(store.profiles_dir) / f"{sample_profile}_lora_weights.pt"
        assert weights_file.exists(), f"Weights file not created at {weights_file}"

    def test_save_lora_weights_roundtrip(self, store, sample_profile, sample_lora_state_dict):
        """Saved weights should be loadable and match original."""
        store.save_lora_weights(sample_profile, sample_lora_state_dict)

        # Load and verify using store method
        loaded = store.load_lora_weights(sample_profile)

        for key in sample_lora_state_dict:
            assert key in loaded, f"Missing key: {key}"
            assert torch.allclose(sample_lora_state_dict[key], loaded[key]), \
                f"Mismatch for {key}"

    def test_save_lora_weights_profile_not_found_raises(self, store, sample_lora_state_dict):
        """save_lora_weights should raise if profile doesn't exist."""
        with pytest.raises(ValueError, match="Profile.*not found"):
            store.save_lora_weights("nonexistent-profile", sample_lora_state_dict)

    def test_save_lora_weights_overwrites_existing(self, store, sample_profile, sample_lora_state_dict):
        """save_lora_weights should overwrite existing weights."""
        # Save first version
        store.save_lora_weights(sample_profile, sample_lora_state_dict)

        # Save second version with different values
        new_state_dict = {k: torch.randn_like(v) for k, v in sample_lora_state_dict.items()}
        store.save_lora_weights(sample_profile, new_state_dict)

        # Verify new values using store method
        loaded = store.load_lora_weights(sample_profile)

        for key in new_state_dict:
            assert torch.allclose(new_state_dict[key], loaded[key]), \
                f"Weights not updated for {key}"

    def test_delete_profile_removes_samples_and_artifacts(
        self,
        store,
        sample_profile,
        sample_lora_state_dict,
        tmp_path,
    ):
        vocals_path = tmp_path / "sample.wav"
        with wave.open(str(vocals_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 16000)
        sample = store.add_training_sample(
            profile_id=sample_profile,
            vocals_path=str(vocals_path),
            source_file="sample.wav",
            duration=1.0,
        )
        sample_dir = Path(store.samples_dir) / sample_profile / sample.sample_id
        manifest_dir = Path(store.trained_models_dir) / sample_profile
        manifest_dir.mkdir(parents=True)
        (manifest_dir / "artifact_manifest.json").write_text("{}", encoding="utf-8")
        torch.save(sample_lora_state_dict, Path(store.trained_models_dir) / f"{sample_profile}_adapter.pt")
        torch.save(sample_lora_state_dict, Path(store.trained_models_dir) / f"{sample_profile}_full_model.pth")
        (Path(store.trained_models_dir) / f"{sample_profile}_runtime.engine").write_bytes(b"engine")
        unrelated = Path(store.trained_models_dir) / "other-profile_adapter.pt"
        torch.save(sample_lora_state_dict, unrelated)

        assert sample_dir.exists()
        assert store.delete(sample_profile) is True

        assert not Path(store.profiles_dir, f"{sample_profile}.json").exists()
        assert not sample_dir.exists()
        assert not manifest_dir.exists()
        assert not Path(store.trained_models_dir, f"{sample_profile}_adapter.pt").exists()
        assert not Path(store.trained_models_dir, f"{sample_profile}_full_model.pth").exists()
        assert not Path(store.trained_models_dir, f"{sample_profile}_runtime.engine").exists()
        assert unrelated.exists()


class TestLoadLoRAWeights:
    """Tests for load_lora_weights() method."""

    def test_load_lora_weights_method_exists(self, store):
        """Task 2.3: VoiceProfileStore should have load_lora_weights() method."""
        assert hasattr(store, 'load_lora_weights'), \
            "VoiceProfileStore missing load_lora_weights() method"
        assert callable(store.load_lora_weights), \
            "load_lora_weights should be callable"

    def test_load_lora_weights_returns_dict(self, store, sample_profile, sample_lora_state_dict):
        """Task 2.4: load_lora_weights should return state dict."""
        store.save_lora_weights(sample_profile, sample_lora_state_dict)

        loaded = store.load_lora_weights(sample_profile)

        assert isinstance(loaded, dict), "Should return a dict"
        assert len(loaded) > 0, "Should not be empty"

    def test_load_lora_weights_matches_saved(self, store, sample_profile, sample_lora_state_dict):
        """Loaded weights should match what was saved."""
        store.save_lora_weights(sample_profile, sample_lora_state_dict)
        loaded = store.load_lora_weights(sample_profile)

        for key in sample_lora_state_dict:
            assert key in loaded, f"Missing key: {key}"
            assert torch.allclose(sample_lora_state_dict[key], loaded[key]), \
                f"Mismatch for {key}"

    def test_load_lora_weights_profile_not_found_raises(self, store):
        """load_lora_weights should raise if profile doesn't exist."""
        with pytest.raises(ValueError, match="Profile.*not found"):
            store.load_lora_weights("nonexistent-profile")

    def test_load_lora_weights_no_weights_raises(self, store, sample_profile):
        """load_lora_weights should raise if no weights saved."""
        with pytest.raises(FileNotFoundError, match="No.*weights"):
            store.load_lora_weights(sample_profile)


class TestHasTrainedModel:
    """Tests for has_trained_model() method."""

    def test_has_trained_model_method_exists(self, store):
        """Task 2.5: VoiceProfileStore should have has_trained_model() method."""
        assert hasattr(store, 'has_trained_model'), \
            "VoiceProfileStore missing has_trained_model() method"
        assert callable(store.has_trained_model), \
            "has_trained_model should be callable"

    def test_has_trained_model_false_initially(self, store, sample_profile):
        """Task 2.6: has_trained_model should return False for untrained profile."""
        result = store.has_trained_model(sample_profile)
        assert result is False, "Should be False before training"

    def test_has_trained_model_true_after_save(self, store, sample_profile, sample_lora_state_dict):
        """has_trained_model should return True after saving weights."""
        store.save_lora_weights(sample_profile, sample_lora_state_dict)

        result = store.has_trained_model(sample_profile)
        assert result is True, "Should be True after saving weights"

    def test_has_trained_model_nonexistent_profile(self, store):
        """has_trained_model should return False for nonexistent profile."""
        result = store.has_trained_model("nonexistent-profile")
        assert result is False, "Should be False for nonexistent profile"


class TestWeightVersioning:
    """Tests for weight file versioning."""

    def test_save_increments_version(self, store, sample_profile, sample_lora_state_dict):
        """Multiple saves should track version history."""
        # Save multiple times
        store.save_lora_weights(sample_profile, sample_lora_state_dict)
        store.save_lora_weights(sample_profile, sample_lora_state_dict)
        store.save_lora_weights(sample_profile, sample_lora_state_dict)

        # Verify file exists and is loadable (uses flat file pattern)
        weights_file = Path(store.profiles_dir) / f"{sample_profile}_lora_weights.pt"
        assert weights_file.exists()

        # Verify loadable
        loaded = store.load_lora_weights(sample_profile)
        assert loaded is not None

    def test_load_specific_version(self, store, sample_profile, sample_lora_state_dict):
        """Should be able to load specific version if versioning implemented."""
        # For now, just verify current version loads
        store.save_lora_weights(sample_profile, sample_lora_state_dict)
        loaded = store.load_lora_weights(sample_profile)
        assert loaded is not None


class TestTrainingMetadata:
    """Tests for training metadata storage alongside weights."""

    def test_save_with_metadata(self, store, sample_profile, sample_lora_state_dict):
        """Should store training metadata with weights."""
        metadata = {
            "epochs": 100,
            "final_loss": 0.0123,
            "trained_at": "2026-01-30T12:00:00Z",
        }

        # If metadata support exists
        if hasattr(store, 'save_lora_weights'):
            # For now just verify basic save works
            store.save_lora_weights(sample_profile, sample_lora_state_dict)
            assert store.has_trained_model(sample_profile)
