"""Tests for AdapterBridge - LoRA to Seed-VC integration.

The AdapterBridge serves as the integration layer between:
1. Trained LoRA adapters (from our MLP-based decoder)
2. Seed-VC's in-context learning approach (reference audio)

Tests cover:
- Loading and caching voice references
- Loading and caching LoRA weights
- Profile mapping and fuzzy matching
- Error handling for missing/corrupt files
"""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from auto_voice.storage.voice_profiles import VoiceProfileStore


def _write_wav(path: Path, frames: int = 1600) -> None:
    import wave

    with wave.open(str(path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00" * frames)


class TestAdapterBridgeInit:
    """Test AdapterBridge initialization and configuration."""

    def test_import_succeeds(self):
        """AdapterBridge can be imported."""
        from auto_voice.inference.adapter_bridge import AdapterBridge
        assert AdapterBridge is not None

    def test_init_creates_instance(self, tmp_path):
        """AdapterBridge initializes with custom directories."""
        from auto_voice.inference.adapter_bridge import AdapterBridge

        profiles_dir = tmp_path / "profiles"
        training_dir = tmp_path / "training"
        lora_dir = tmp_path / "loras"

        profiles_dir.mkdir()
        training_dir.mkdir()
        lora_dir.mkdir()

        bridge = AdapterBridge(
            profiles_dir=str(profiles_dir),
            training_audio_dir=str(training_dir),
            lora_dir=str(lora_dir),
            device="cpu"
        )

        assert bridge is not None
        assert bridge.profiles_dir == profiles_dir
        assert bridge.training_audio_dir == training_dir
        assert bridge.lora_dir == lora_dir
        assert bridge.device == torch.device("cpu")

    def test_init_loads_profile_mappings(self, tmp_path):
        """AdapterBridge loads profile JSON files on initialization."""
        from auto_voice.inference.adapter_bridge import AdapterBridge

        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        # Create test profile
        profile_data = {
            "profile_id": "test-profile-123",
            "name": "John Doe"
        }
        with open(profiles_dir / "test-profile-123.json", "w") as f:
            json.dump(profile_data, f)

        bridge = AdapterBridge(
            profiles_dir=str(profiles_dir),
            training_audio_dir=str(tmp_path / "training"),
            lora_dir=str(tmp_path / "loras"),
            device="cpu"
        )

        assert "test-profile-123" in bridge._profile_to_artist
        assert bridge._profile_to_artist["test-profile-123"] == "John Doe"

    def test_init_handles_corrupt_profile_json(self, tmp_path):
        """AdapterBridge gracefully handles corrupt profile JSON."""
        from auto_voice.inference.adapter_bridge import AdapterBridge

        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        # Create corrupt JSON file
        with open(profiles_dir / "corrupt.json", "w") as f:
            f.write("{ invalid json }")

        # Should not raise - just logs warning
        bridge = AdapterBridge(
            profiles_dir=str(profiles_dir),
            training_audio_dir=str(tmp_path / "training"),
            lora_dir=str(tmp_path / "loras"),
            device="cpu"
        )

        assert bridge is not None
        assert len(bridge._profile_to_artist) == 0

    def test_init_default_directories(self):
        """AdapterBridge uses default directories when not specified."""
        from auto_voice.inference.adapter_bridge import AdapterBridge

        with patch.object(VoiceProfileStore, "list_profiles", return_value=[]):
            bridge = AdapterBridge(device="cpu")

        assert "data/voice_profiles" in str(bridge.profiles_dir)
        assert "data/separated_youtube" in str(bridge.training_audio_dir)
        assert "data/trained_models" in str(bridge.lora_dir)


class TestVoiceReferenceLoading:
    """Test loading voice references for Seed-VC pipeline."""

    @pytest.fixture
    def bridge_setup(self, tmp_path):
        """Create AdapterBridge with test data."""
        from auto_voice.inference.adapter_bridge import AdapterBridge

        profiles_dir = tmp_path / "profiles"
        samples_dir = tmp_path / "samples"
        training_dir = tmp_path / "training"
        lora_dir = tmp_path / "loras"

        training_dir.mkdir()
        store = VoiceProfileStore(
            profiles_dir=str(profiles_dir),
            samples_dir=str(samples_dir),
            trained_models_dir=str(lora_dir),
        )
        profile_id = store.save(
            {
                "profile_id": "profile-abc",
                "name": "John Artist",
                "embedding": np.random.randn(256).astype(np.float32),
            }
        )

        vocal_file = tmp_path / "source_vocals.wav"
        _write_wav(vocal_file, frames=16000 * 12)
        training_sample = store.add_training_sample(
            profile_id=profile_id,
            vocals_path=str(vocal_file),
            duration=12.0,
        )

        lora_state = {"layer1.weight": torch.randn(64, 64)}
        torch.save({"lora_state": lora_state}, lora_dir / "profile-abc_adapter.pt")

        bridge = AdapterBridge(
            profiles_dir=str(profiles_dir),
            training_audio_dir=str(training_dir),
            lora_dir=str(lora_dir),
            device="cpu"
        )

        return bridge, profiles_dir, training_dir, lora_dir, training_sample

    def test_get_voice_reference_returns_dataclass(self, bridge_setup):
        """get_voice_reference returns VoiceReference dataclass."""
        from auto_voice.inference.adapter_bridge import VoiceReference

        bridge, _, _, _, _ = bridge_setup
        ref = bridge.get_voice_reference("profile-abc")

        assert isinstance(ref, VoiceReference)
        assert ref.profile_id == "profile-abc"
        assert ref.profile_name == "John Artist"

    def test_get_voice_reference_finds_audio_files(self, bridge_setup):
        """get_voice_reference finds reference audio files."""
        bridge, _, _, _, training_sample = bridge_setup
        ref = bridge.get_voice_reference("profile-abc")

        assert ref.reference_paths == [Path(training_sample.vocals_path)]
        assert all(p.suffix == ".wav" for p in ref.reference_paths)
        assert all("vocals" in p.name for p in ref.reference_paths)

    def test_get_voice_reference_loads_embedding(self, bridge_setup):
        """get_voice_reference loads pre-computed speaker embedding."""
        bridge, _, _, _, _ = bridge_setup
        ref = bridge.get_voice_reference("profile-abc")

        assert ref.speaker_embedding is not None
        assert ref.speaker_embedding.shape == (256,)

    def test_get_voice_reference_finds_lora_path(self, bridge_setup):
        """get_voice_reference finds LoRA checkpoint path."""
        bridge, _, _, lora_dir, _ = bridge_setup
        ref = bridge.get_voice_reference("profile-abc")

        assert ref.lora_path is not None
        assert ref.lora_path.exists()
        assert ref.lora_path == lora_dir / "profile-abc_adapter.pt"

    def test_get_voice_reference_estimates_duration(self, bridge_setup):
        """get_voice_reference estimates total audio duration."""
        bridge, _, _, _, _ = bridge_setup
        ref = bridge.get_voice_reference("profile-abc")

        assert ref.total_duration == pytest.approx(12.0)

    def test_get_voice_reference_max_references(self, bridge_setup):
        """get_voice_reference respects max_references parameter."""
        bridge, _, _, _, _ = bridge_setup
        ref = bridge.get_voice_reference("profile-abc", max_references=2)

        assert len(ref.reference_paths) <= 2

    def test_get_voice_reference_caching(self, bridge_setup):
        """get_voice_reference caches results."""
        bridge, _, _, _, _ = bridge_setup

        ref1 = bridge.get_voice_reference("profile-abc")
        ref2 = bridge.get_voice_reference("profile-abc")

        assert ref1 is ref2  # Same object (cached)

    def test_get_voice_reference_profile_not_found(self, bridge_setup):
        """get_voice_reference raises ValueError for missing profile."""
        bridge, _, _, _, _ = bridge_setup

        with pytest.raises(ValueError, match="Profile not found"):
            bridge.get_voice_reference("nonexistent-profile")

    def test_get_voice_reference_no_audio_returns_empty_list(self, tmp_path):
        """get_voice_reference returns empty list when no audio available."""
        from auto_voice.inference.adapter_bridge import AdapterBridge

        profiles_dir = tmp_path / "profiles"
        training_dir = tmp_path / "training"
        lora_dir = tmp_path / "loras"
        profiles_dir.mkdir()
        training_dir.mkdir()  # Create the directory, but with no matching artist
        lora_dir.mkdir()

        # Profile exists but no audio directory for this artist
        profile_data = {"profile_id": "no-audio", "name": "No Audio Artist"}
        with open(profiles_dir / "no-audio.json", "w") as f:
            json.dump(profile_data, f)

        bridge = AdapterBridge(
            profiles_dir=str(profiles_dir),
            training_audio_dir=str(training_dir),
            lora_dir=str(lora_dir),
            device="cpu"
        )

        ref = bridge.get_voice_reference("no-audio")
        assert len(ref.reference_paths) == 0


class TestFuzzyMatching:
    """Test fuzzy string matching for artist name variations."""

    def test_fuzzy_match_exact(self):
        """Exact strings match."""
        from auto_voice.inference.adapter_bridge import AdapterBridge

        assert AdapterBridge._fuzzy_match("connor", "connor") is True

    def test_fuzzy_match_one_char_diff(self):
        """Strings with one character difference match."""
        from auto_voice.inference.adapter_bridge import AdapterBridge

        assert AdapterBridge._fuzzy_match("connor", "conor") is True  # Missing 'n'
        assert AdapterBridge._fuzzy_match("john", "jonn") is True  # Wrong char

    def test_fuzzy_match_two_char_diff(self):
        """Strings with two character differences match."""
        from auto_voice.inference.adapter_bridge import AdapterBridge

        assert AdapterBridge._fuzzy_match("steven", "stevan") is True

    def test_fuzzy_match_too_different(self):
        """Strings too different don't match."""
        from auto_voice.inference.adapter_bridge import AdapterBridge

        assert AdapterBridge._fuzzy_match("alice", "bob") is False
        assert AdapterBridge._fuzzy_match("john", "jonathan") is False

    def test_fuzzy_match_empty_strings(self):
        """Empty strings don't match."""
        from auto_voice.inference.adapter_bridge import AdapterBridge

        assert AdapterBridge._fuzzy_match("", "test") is False
        assert AdapterBridge._fuzzy_match("test", "") is False
        assert AdapterBridge._fuzzy_match("", "") is False


class TestLoRALoading:
    """Test LoRA weight loading functionality."""

    @pytest.fixture
    def lora_bridge(self, tmp_path):
        """Create AdapterBridge with LoRA checkpoint."""
        from auto_voice.inference.adapter_bridge import AdapterBridge

        profiles_dir = tmp_path / "profiles"
        samples_dir = tmp_path / "samples"
        lora_dir = tmp_path / "loras"
        store = VoiceProfileStore(
            profiles_dir=str(profiles_dir),
            samples_dir=str(samples_dir),
            trained_models_dir=str(lora_dir),
        )
        store.save({"profile_id": "lora-test", "name": "LoRA Test"})

        # Create LoRA checkpoint with state dict
        lora_state = {
            "encoder.lora_A": torch.randn(32, 64),
            "encoder.lora_B": torch.randn(64, 32),
            "decoder.lora_A": torch.randn(16, 32),
            "decoder.lora_B": torch.randn(32, 16),
        }
        checkpoint = {
            "lora_state": lora_state,
            "artist": "Test Artist",
            "epoch": 100,
            "loss": 0.0123,
            "precision": "fp16",
            "status": "completed",
            "config": {"lr": 1e-4, "rank": 32},
        }
        torch.save(checkpoint, lora_dir / "lora-test_adapter.pt")

        bridge = AdapterBridge(
            profiles_dir=str(profiles_dir),
            training_audio_dir=str(tmp_path / "training"),
            lora_dir=str(lora_dir),
            device="cpu"
        )

        return bridge

    def test_load_lora_returns_state_dict(self, lora_bridge):
        """load_lora returns LoRA state dictionary."""
        lora_state = lora_bridge.load_lora("lora-test")

        assert isinstance(lora_state, dict)
        assert "encoder.lora_A" in lora_state
        assert "encoder.lora_B" in lora_state
        assert "decoder.lora_A" in lora_state
        assert "decoder.lora_B" in lora_state

    def test_load_lora_tensors_on_device(self, lora_bridge):
        """load_lora moves tensors to specified device."""
        lora_state = lora_bridge.load_lora("lora-test")

        for key, tensor in lora_state.items():
            assert tensor.device == torch.device("cpu")

    def test_load_lora_caching(self, lora_bridge):
        """load_lora caches loaded weights."""
        lora1 = lora_bridge.load_lora("lora-test")
        lora2 = lora_bridge.load_lora("lora-test")

        # Tensors should be same objects (cached)
        for key in lora1:
            assert lora1[key] is lora2[key]

    def test_load_lora_no_cache(self, lora_bridge):
        """load_lora can skip caching."""
        lora1 = lora_bridge.load_lora("lora-test", use_cache=False)
        lora2 = lora_bridge.load_lora("lora-test", use_cache=False)

        # Tensors should be different objects
        for key in lora1:
            assert lora1[key] is not lora2[key]

    def test_load_lora_not_found(self, lora_bridge):
        """load_lora raises FileNotFoundError for missing LoRA."""
        with pytest.raises(FileNotFoundError, match="No LoRA found"):
            lora_bridge.load_lora("nonexistent-profile")

    def test_get_lora_metadata(self, lora_bridge):
        """get_lora_metadata returns training metadata."""
        metadata = lora_bridge.get_lora_metadata("lora-test")

        assert metadata["artist"] == "Test Artist"
        assert metadata["epoch"] == 100
        assert metadata["loss"] == 0.0123
        assert metadata["precision"] == "fp16"
        assert metadata["status"] == "completed"
        assert metadata["config"]["lr"] == 1e-4

    def test_get_lora_metadata_missing_returns_empty(self, lora_bridge):
        """get_lora_metadata returns empty dict for missing LoRA."""
        metadata = lora_bridge.get_lora_metadata("nonexistent")
        assert metadata == {}


class TestProfileListing:
    """Test listing available profiles."""

    @pytest.fixture
    def populated_bridge(self, tmp_path):
        """Create AdapterBridge with multiple profiles."""
        from auto_voice.inference.adapter_bridge import AdapterBridge

        profiles_dir = tmp_path / "profiles"
        samples_dir = tmp_path / "samples"
        training_dir = tmp_path / "training"
        lora_dir = tmp_path / "loras"

        training_dir.mkdir()
        store = VoiceProfileStore(
            profiles_dir=str(profiles_dir),
            samples_dir=str(samples_dir),
            trained_models_dir=str(lora_dir),
        )

        store.save({"profile_id": "profile-1", "name": "Alpha Singer"})
        sample_one = tmp_path / "alpha.wav"
        _write_wav(sample_one)
        store.add_training_sample("profile-1", str(sample_one), duration=1.0)
        torch.save({}, lora_dir / "profile-1_adapter.pt")

        store.save({"profile_id": "profile-2", "name": "Zeta Performer"})
        torch.save({}, lora_dir / "profile-2_adapter.pt")

        store.save({"profile_id": "profile-3", "name": "Gamma Vocalist"})
        sample_three = tmp_path / "gamma.wav"
        _write_wav(sample_three)
        store.add_training_sample("profile-3", str(sample_three), duration=1.0)

        return AdapterBridge(
            profiles_dir=str(profiles_dir),
            training_audio_dir=str(training_dir),
            lora_dir=str(lora_dir),
            device="cpu"
        )

    def test_list_available_profiles(self, populated_bridge):
        """list_available_profiles returns all profiles."""
        profiles = populated_bridge.list_available_profiles()

        assert len(profiles) == 3
        profile_ids = [p[0] for p in profiles]
        assert "profile-1" in profile_ids
        assert "profile-2" in profile_ids
        assert "profile-3" in profile_ids

    def test_list_profiles_shows_lora_status(self, populated_bridge):
        """list_available_profiles shows LoRA availability."""
        profiles = populated_bridge.list_available_profiles()
        profiles_dict = {p[0]: p for p in profiles}

        # Profile 1 and 2 have LoRA
        assert profiles_dict["profile-1"][2] is True
        assert profiles_dict["profile-2"][2] is True
        # Profile 3 has no LoRA
        assert profiles_dict["profile-3"][2] is False

    def test_list_profiles_shows_reference_status(self, populated_bridge):
        """list_available_profiles shows reference audio availability."""
        profiles = populated_bridge.list_available_profiles()
        profiles_dict = {p[0]: p for p in profiles}

        # Profile 1 and 3 have reference audio
        assert profiles_dict["profile-1"][3] is True
        assert profiles_dict["profile-3"][3] is True
        # Profile 2 has no reference audio
        assert profiles_dict["profile-2"][3] is False


class TestCacheManagement:
    """Test cache clearing functionality."""

    def test_clear_cache(self, tmp_path):
        """clear_cache empties both caches."""
        from auto_voice.inference.adapter_bridge import AdapterBridge

        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        # Create test profile
        profile_data = {"profile_id": "cache-test", "name": "Cache Test"}
        with open(profiles_dir / "cache-test.json", "w") as f:
            json.dump(profile_data, f)

        bridge = AdapterBridge(
            profiles_dir=str(profiles_dir),
            training_audio_dir=str(tmp_path / "training"),
            lora_dir=str(tmp_path / "loras"),
            device="cpu"
        )

        # Manually populate cache
        bridge._reference_cache["test"] = object()
        bridge._lora_cache["test"] = {"weight": torch.randn(10)}

        assert len(bridge._reference_cache) == 1
        assert len(bridge._lora_cache) == 1

        bridge.clear_cache()

        assert len(bridge._reference_cache) == 0
        assert len(bridge._lora_cache) == 0


class TestSingletonBehavior:
    """Test global singleton instance."""

    def test_get_adapter_bridge_singleton(self):
        """get_adapter_bridge returns singleton instance."""
        from auto_voice.inference import adapter_bridge

        # Reset singleton for test
        adapter_bridge._bridge_instance = None

        with patch.object(Path, 'glob', return_value=[]):
            bridge1 = adapter_bridge.get_adapter_bridge()
            bridge2 = adapter_bridge.get_adapter_bridge()

        assert bridge1 is bridge2

        # Clean up
        adapter_bridge._bridge_instance = None
