"""Unit tests for AdapterManager.

Tests the adapter loading, caching, validation, and application functionality
for voice conversion adapters used across both REALTIME and QUALITY pipelines.
"""
import json
import numpy as np
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import Mock, patch

from auto_voice.models.adapter_manager import (
    AdapterManager,
    AdapterManagerConfig,
    AdapterCache,
    AdapterInfo,
    load_adapter_for_profile,
    get_trained_profiles,
    get_adapter_manager,
)


@pytest.fixture
def temp_dirs(tmp_path):
    """Create temporary directories for adapters and profiles."""
    adapters_dir = tmp_path / "trained_models"
    profiles_dir = tmp_path / "voice_profiles"
    adapters_dir.mkdir()
    profiles_dir.mkdir()
    return {
        "adapters_dir": adapters_dir,
        "profiles_dir": profiles_dir,
        "tmp_path": tmp_path,
    }


@pytest.fixture
def adapter_config(temp_dirs):
    """Create AdapterManagerConfig with temp directories."""
    return AdapterManagerConfig(
        adapters_dir=temp_dirs["adapters_dir"],
        profiles_dir=temp_dirs["profiles_dir"],
        cache_size=3,
        device="cpu",  # Use CPU for tests
        auto_validate=True,
    )


@pytest.fixture
def mock_adapter_state():
    """Create a mock adapter state dict with LoRA structure."""
    return {
        "lora_adapters.content_proj.lora_A": torch.randn(8, 256),
        "lora_adapters.content_proj.lora_B": torch.randn(256, 8),
        "lora_adapters.output.lora_A": torch.randn(8, 512),
        "lora_adapters.output.lora_B": torch.randn(512, 8),
    }


@pytest.fixture
def mock_profile_metadata():
    """Create mock profile metadata."""
    return {
        "name": "Test Voice",
        "created_at": "2026-01-30T12:00:00Z",
        "sample_count": 10,
        "adapter_version": "1.0",
        "adapter_target_modules": ["content_proj", "output"],
        "adapter_rank": 8,
        "adapter_alpha": 16,
        "training_epochs": 50,
        "loss_final": 0.023,
    }


@pytest.fixture
def create_test_adapter(temp_dirs, mock_adapter_state, mock_profile_metadata):
    """Helper to create test adapter files."""
    def _create(profile_id: str):
        # Save adapter weights
        adapter_path = temp_dirs["adapters_dir"] / f"{profile_id}_adapter.pt"
        torch.save(mock_adapter_state, adapter_path)

        # Save profile metadata
        profile_path = temp_dirs["profiles_dir"] / f"{profile_id}.json"
        with open(profile_path, "w") as f:
            json.dump(mock_profile_metadata, f)

        return adapter_path, profile_path

    return _create


@pytest.fixture
def create_test_full_model(temp_dirs):
    """Helper to create full-model checkpoints."""
    def _create(profile_id: str, suffix: str = ".pt"):
        checkpoint_path = temp_dirs["adapters_dir"] / f"{profile_id}_full_model{suffix}"
        torch.save({"weights": torch.ones(2, 2)}, checkpoint_path)
        return checkpoint_path

    return _create


@pytest.fixture
def create_test_engine(temp_dirs):
    """Helper to create TensorRT engine files."""
    def _create(profile_id: str, filename: str | None = None):
        engines_dir = temp_dirs["tmp_path"] / "engines"
        engines_dir.mkdir(exist_ok=True)
        engine_path = engines_dir / (filename or f"{profile_id}_nvfp4.engine")
        engine_path.write_bytes(b"serialized-engine")
        return engine_path, engines_dir

    return _create


@pytest.fixture
def create_test_embedding(temp_dirs):
    """Helper to create profile embeddings."""
    def _create(profile_id: str, embedding: np.ndarray | None = None):
        embedding_path = temp_dirs["profiles_dir"] / f"{profile_id}.npy"
        np.save(
            embedding_path,
            np.asarray(
                embedding if embedding is not None else np.ones(256, dtype=np.float32),
                dtype=np.float32,
            ),
        )
        return embedding_path

    return _create


class TestAdapterCache:
    """Test LRU cache implementation."""

    @pytest.mark.smoke
    def test_cache_init(self):
        cache = AdapterCache(max_size=3)
        assert len(cache) == 0
        assert cache.max_size == 3

    @pytest.mark.smoke
    def test_cache_put_and_get(self):
        cache = AdapterCache(max_size=3)
        state = {"key": torch.tensor([1.0])}

        cache.put("profile1", state)
        assert len(cache) == 1
        assert "profile1" in cache

        retrieved = cache.get("profile1")
        assert retrieved is not None
        assert "key" in retrieved

    def test_cache_lru_eviction(self):
        cache = AdapterCache(max_size=2)

        cache.put("p1", {"data": torch.tensor([1.0])})
        cache.put("p2", {"data": torch.tensor([2.0])})
        cache.put("p3", {"data": torch.tensor([3.0])})  # Should evict p1

        assert len(cache) == 2
        assert "p1" not in cache
        assert "p2" in cache
        assert "p3" in cache

    def test_cache_lru_order_update(self):
        cache = AdapterCache(max_size=2)

        cache.put("p1", {"data": torch.tensor([1.0])})
        cache.put("p2", {"data": torch.tensor([2.0])})

        # Access p1, making it most recent
        _ = cache.get("p1")

        # Add p3, should evict p2 (least recent)
        cache.put("p3", {"data": torch.tensor([3.0])})

        assert "p1" in cache
        assert "p2" not in cache
        assert "p3" in cache

    def test_cache_clear(self):
        cache = AdapterCache(max_size=3)
        cache.put("p1", {"data": torch.tensor([1.0])})
        cache.put("p2", {"data": torch.tensor([2.0])})

        assert len(cache) == 2

        cache.clear()

        assert len(cache) == 0
        assert "p1" not in cache


class TestAdapterManagerInit:
    """Test AdapterManager initialization."""

    @pytest.mark.smoke
    def test_init_default_config(self, temp_dirs):
        manager = AdapterManager()
        assert manager.config is not None
        assert manager.device is not None
        assert isinstance(manager._cache, AdapterCache)

    @pytest.mark.smoke
    def test_init_custom_config(self, adapter_config):
        manager = AdapterManager(adapter_config)
        assert manager.config.cache_size == 3
        assert manager.config.device == "cpu"
        assert manager.device == torch.device("cpu")

    def test_init_creates_directories(self, adapter_config, temp_dirs):
        # Remove adapters_dir to test creation
        temp_dirs["adapters_dir"].rmdir()

        manager = AdapterManager(adapter_config)

        assert temp_dirs["adapters_dir"].exists()


class TestAdapterManagerListing:
    """Test adapter listing functionality."""

    @pytest.mark.smoke
    def test_list_available_adapters_empty(self, adapter_config):
        manager = AdapterManager(adapter_config)
        adapters = manager.list_available_adapters()
        assert adapters == []

    def test_list_available_adapters(self, adapter_config, create_test_adapter):
        create_test_adapter("profile1")
        create_test_adapter("profile2")

        manager = AdapterManager(adapter_config)
        adapters = manager.list_available_adapters()

        assert len(adapters) == 2
        assert "profile1" in adapters
        assert "profile2" in adapters

    @pytest.mark.smoke
    def test_has_adapter_false(self, adapter_config):
        manager = AdapterManager(adapter_config)
        assert manager.has_adapter("nonexistent") is False

    def test_has_adapter_true(self, adapter_config, create_test_adapter):
        create_test_adapter("profile1")

        manager = AdapterManager(adapter_config)
        assert manager.has_adapter("profile1") is True

    def test_get_adapter_path_none(self, adapter_config):
        manager = AdapterManager(adapter_config)
        path = manager.get_adapter_path("nonexistent")
        assert path is None

    def test_get_adapter_path_exists(self, adapter_config, create_test_adapter):
        adapter_path, _ = create_test_adapter("profile1")

        manager = AdapterManager(adapter_config)
        path = manager.get_adapter_path("profile1")

        assert path == adapter_path

    def test_get_available_artifact_types_priority(
        self,
        adapter_config,
        create_test_adapter,
        create_test_full_model,
        create_test_engine,
    ):
        create_test_adapter("profile1")
        create_test_full_model("profile1")
        _, engines_dir = create_test_engine("profile1")
        adapter_config.tensorrt_dir = engines_dir

        manager = AdapterManager(adapter_config)

        assert manager.get_available_artifact_types("profile1") == [
            "tensorrt",
            "full_model",
            "adapter",
        ]

    @pytest.mark.parametrize("suffix", [".pt", ".pth"])
    def test_get_full_model_path_supports_checkpoint_suffixes(
        self,
        adapter_config,
        create_test_full_model,
        suffix,
    ):
        checkpoint_path = create_test_full_model("profile1", suffix=suffix)
        manager = AdapterManager(adapter_config)

        assert manager.get_full_model_path("profile1") == checkpoint_path


class TestAdapterManagerLoading:
    """Test adapter loading functionality."""

    def test_load_adapter_success(self, adapter_config, create_test_adapter):
        create_test_adapter("profile1")

        manager = AdapterManager(adapter_config)
        state_dict = manager.load_adapter("profile1")

        assert state_dict is not None
        assert "lora_adapters.content_proj.lora_A" in state_dict
        assert "lora_adapters.content_proj.lora_B" in state_dict
        assert isinstance(state_dict["lora_adapters.content_proj.lora_A"], torch.Tensor)

    def test_load_adapter_not_found(self, adapter_config):
        manager = AdapterManager(adapter_config)

        with pytest.raises(FileNotFoundError, match="No adapter found for profile"):
            manager.load_adapter("nonexistent")

    def test_load_adapter_caching(self, adapter_config, create_test_adapter):
        create_test_adapter("profile1")

        manager = AdapterManager(adapter_config)

        # First load
        state1 = manager.load_adapter("profile1", use_cache=True)
        assert "profile1" in manager._cache

        # Second load (should hit cache)
        state2 = manager.load_adapter("profile1", use_cache=True)

        # Should be same object (from cache)
        assert state1 is state2

    def test_load_adapter_no_cache(self, adapter_config, create_test_adapter):
        create_test_adapter("profile1")

        manager = AdapterManager(adapter_config)

        state1 = manager.load_adapter("profile1", use_cache=False)
        assert "profile1" not in manager._cache

        state2 = manager.load_adapter("profile1", use_cache=False)

        # Should be different objects (not cached)
        assert state1 is not state2

    def test_load_full_model_artifact(self, adapter_config, create_test_full_model):
        checkpoint_path = create_test_full_model("profile1", suffix=".pth")

        manager = AdapterManager(adapter_config)
        artifact = manager.load_artifact("profile1", artifact_type="full_model")

        assert artifact.profile_id == "profile1"
        assert artifact.artifact_type == "full_model"
        assert artifact.path == checkpoint_path
        assert torch.equal(artifact.handle["weights"], torch.ones(2, 2))

    def test_load_tensorrt_artifact(self, adapter_config, create_test_engine):
        engine_path, engines_dir = create_test_engine("profile1")
        adapter_config.tensorrt_dir = engines_dir

        manager = AdapterManager(adapter_config)
        with patch.object(manager, "_load_tensorrt_engine", return_value=b"engine") as loader:
            artifact = manager.load_artifact("profile1", artifact_type="tensorrt")

        loader.assert_called_once_with(engine_path)
        assert artifact.artifact_type == "tensorrt"
        assert artifact.handle == b"engine"

    def test_load_speaker_embedding_normalizes_and_returns_tensor(
        self,
        adapter_config,
        create_test_embedding,
    ):
        create_test_embedding("profile1", np.full(256, 2.0, dtype=np.float32))

        manager = AdapterManager(adapter_config)
        embedding = manager.load_speaker_embedding("profile1")
        tensor = manager.load_speaker_embedding("profile1", as_tensor=True)

        assert embedding.shape == (256,)
        assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (256,)
        assert torch.isclose(torch.norm(tensor), torch.tensor(1.0), atol=1e-5)

    def test_load_speaker_embedding_rejects_invalid_shape(
        self,
        adapter_config,
        create_test_embedding,
    ):
        create_test_embedding("profile1", np.ones((32,), dtype=np.float32))

        manager = AdapterManager(adapter_config)

        with pytest.raises(ValueError, match="Invalid embedding shape"):
            manager.load_speaker_embedding("profile1")

    def test_swap_artifact_tracks_active_state(
        self,
        adapter_config,
        create_test_adapter,
    ):
        create_test_adapter("profile1")
        manager = AdapterManager(adapter_config)

        artifact = manager.swap_artifact("profile1", artifact_type="adapter")
        stats = manager.get_cache_stats()

        assert artifact.profile_id == "profile1"
        assert stats["active_profile_id"] == "profile1"
        assert stats["active_artifact_type"] == "adapter"
        assert "profile1" in manager._cache

        same_artifact = manager.swap_artifact("profile1", artifact_type="adapter")
        assert same_artifact is artifact

        manager.release_active_artifact()
        released_stats = manager.get_cache_stats()
        assert released_stats["active_profile_id"] is None
        assert released_stats["active_artifact_type"] is None
        assert "profile1" not in manager._cache


class TestAdapterValidation:
    """Test adapter validation functionality."""

    def test_validate_adapter_valid(self, adapter_config, mock_adapter_state):
        manager = AdapterManager(adapter_config)

        # Should not raise
        manager._validate_adapter(mock_adapter_state, "profile1")

    def test_validate_adapter_empty(self, adapter_config):
        manager = AdapterManager(adapter_config)

        with pytest.raises(ValueError, match="Empty adapter state dict"):
            manager._validate_adapter({}, "profile1")

    def test_validate_adapter_missing_lora_structure(self, adapter_config):
        manager = AdapterManager(adapter_config)
        invalid_state = {"some_weight": torch.randn(10, 10)}

        # Should log warning but not raise
        manager._validate_adapter(invalid_state, "profile1")

    def test_validate_adapter_disabled(self, adapter_config):
        adapter_config.auto_validate = False
        manager = AdapterManager(adapter_config)

        # Non-empty state without LoRA structure should not raise when validation disabled
        # Note: empty state always raises regardless of auto_validate
        manager._validate_adapter({"some_weight": torch.randn(10, 10)}, "profile1")


class TestAdapterInfo:
    """Test adapter info retrieval."""

    def test_get_adapter_info_not_found(self, adapter_config):
        manager = AdapterManager(adapter_config)
        info = manager.get_adapter_info("nonexistent")
        assert info is None

    def test_get_adapter_info_success(self, adapter_config, create_test_adapter):
        create_test_adapter("profile1")

        manager = AdapterManager(adapter_config)
        info = manager.get_adapter_info("profile1")

        assert info is not None
        assert isinstance(info, AdapterInfo)
        assert info.profile_id == "profile1"
        assert info.profile_name == "Test Voice"
        assert info.rank == 8
        assert info.alpha == 16
        assert info.sample_count == 10
        assert info.training_epochs == 50
        assert info.loss_final == 0.023

    def test_get_adapter_info_caching(self, adapter_config, create_test_adapter):
        create_test_adapter("profile1")

        manager = AdapterManager(adapter_config)

        info1 = manager.get_adapter_info("profile1")
        info2 = manager.get_adapter_info("profile1")

        # Should return same cached object
        assert info1 is info2


class TestAdapterApplication:
    """Test applying adapters to models."""

    @pytest.fixture
    def mock_model(self, mock_adapter_state):
        """Create a mock model with LoRA structure matching mock_adapter_state."""
        class MockLoRALayer(nn.Module):
            def __init__(self, lora_a_shape, lora_b_shape):
                super().__init__()
                self.lora_A = nn.Parameter(torch.randn(*lora_a_shape))
                self.lora_B = nn.Parameter(torch.randn(*lora_b_shape))

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Match shapes from mock_adapter_state
                self.lora_adapters = nn.ModuleDict({
                    "content_proj": MockLoRALayer((8, 256), (256, 8)),
                    "output": MockLoRALayer((8, 512), (512, 8)),
                })

        return MockModel()

    def test_apply_adapter_success(self, adapter_config, mock_adapter_state, mock_model):
        manager = AdapterManager(adapter_config)

        # Apply adapter
        manager.apply_adapter(mock_model, mock_adapter_state)

        # Verify parameters were updated
        applied_param = mock_model.lora_adapters.content_proj.lora_A
        expected_param = mock_adapter_state["lora_adapters.content_proj.lora_A"]

        assert torch.allclose(applied_param, expected_param)

    def test_apply_adapter_no_matching_params(self, adapter_config, mock_adapter_state):
        manager = AdapterManager(adapter_config)
        empty_model = nn.Module()

        # Should log warning but not raise
        manager.apply_adapter(empty_model, mock_adapter_state)

    def test_remove_adapter(self, adapter_config, mock_model):
        manager = AdapterManager(adapter_config)

        # Store original values
        original_b = mock_model.lora_adapters.content_proj.lora_B.data.clone()

        # Remove adapter (zeros out lora_B)
        manager.remove_adapter(mock_model)

        # Verify lora_B is zeroed
        assert torch.allclose(
            mock_model.lora_adapters.content_proj.lora_B.data,
            torch.zeros_like(original_b)
        )


class TestAdapterSaving:
    """Test saving adapters from models."""

    @pytest.fixture
    def trained_model(self):
        """Create a model with trained LoRA adapters."""
        class MockLoRALayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = nn.Parameter(torch.randn(8, 256))
                self.lora_B = nn.Parameter(torch.randn(256, 8))

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_adapters = nn.ModuleDict({
                    "content_proj": MockLoRALayer(),
                    "output": MockLoRALayer(),
                })

        return MockModel()

    def test_save_adapter_success(self, adapter_config, trained_model, temp_dirs):
        manager = AdapterManager(adapter_config)

        path = manager.save_adapter("profile1", trained_model)

        assert path.exists()
        assert path.name == "profile1_adapter.pt"

        # Verify saved state can be loaded
        saved_state = torch.load(path, map_location="cpu", weights_only=False)
        assert "lora_adapters.content_proj.lora_A" in saved_state
        assert "lora_adapters.content_proj.lora_B" in saved_state

    def test_save_adapter_no_lora_params(self, adapter_config):
        manager = AdapterManager(adapter_config)
        empty_model = nn.Module()

        with pytest.raises(ValueError, match="No adapter parameters found"):
            manager.save_adapter("profile1", empty_model)

    def test_save_adapter_with_metadata(
        self, adapter_config, trained_model, temp_dirs, mock_profile_metadata
    ):
        # Create existing profile
        profile_path = temp_dirs["profiles_dir"] / "profile1.json"
        with open(profile_path, "w") as f:
            json.dump({"name": "Original"}, f)

        manager = AdapterManager(adapter_config)

        # Save with metadata update
        manager.save_adapter("profile1", trained_model, metadata={"name": "Updated"})

        # Verify metadata was updated
        with open(profile_path) as f:
            data = json.load(f)
        assert data["name"] == "Updated"


class TestCacheManagement:
    """Test cache management functionality."""

    def test_clear_cache(self, adapter_config, create_test_adapter):
        create_test_adapter("profile1")

        manager = AdapterManager(adapter_config)

        # Load to populate cache
        manager.load_adapter("profile1")
        manager.get_adapter_info("profile1")

        assert len(manager._cache) > 0
        assert len(manager._adapter_info) > 0

        # Clear cache
        manager.clear_cache()

        assert len(manager._cache) == 0
        assert len(manager._adapter_info) == 0

    def test_get_cache_stats(self, adapter_config, create_test_adapter):
        create_test_adapter("profile1")
        create_test_adapter("profile2")

        manager = AdapterManager(adapter_config)

        manager.load_adapter("profile1")
        manager.load_adapter("profile2")
        manager.get_adapter_info("profile1")

        stats = manager.get_cache_stats()

        assert stats["cached_adapters"] == 2
        assert stats["max_cache_size"] == 3
        assert stats["cached_info"] == 1


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_load_adapter_for_profile(self, adapter_config, create_test_adapter, monkeypatch):
        create_test_adapter("profile1")

        # Mock AdapterManager to use our test config
        def mock_init(self, config=None):
            self.config = adapter_config
            self.device = torch.device("cpu")
            self._cache = AdapterCache(max_size=adapter_config.cache_size)
            self._adapter_info = {}
            self.config.adapters_dir.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(AdapterManager, "__init__", mock_init)

        state_dict = load_adapter_for_profile("profile1", device="cpu")

        assert state_dict is not None
        assert "lora_adapters.content_proj.lora_A" in state_dict

    def test_get_trained_profiles(self, adapter_config, create_test_adapter, monkeypatch):
        create_test_adapter("profile1")
        create_test_adapter("profile2")

        # Mock AdapterManager to use our test config
        def mock_init(self, config=None):
            self.config = adapter_config
            self.device = torch.device("cpu")
            self._cache = AdapterCache(max_size=adapter_config.cache_size)
            self._adapter_info = {}
            self.config.adapters_dir.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(AdapterManager, "__init__", mock_init)

        profiles = get_trained_profiles()

        assert len(profiles) == 2
        profile_ids = [p[0] for p in profiles]
        assert "profile1" in profile_ids
        assert "profile2" in profile_ids

    def test_get_adapter_manager_singleton(self):
        # Import and reset global
        import auto_voice.models.adapter_manager as am
        am._global_manager = None

        manager1 = get_adapter_manager()
        manager2 = get_adapter_manager()

        # Should return same instance
        assert manager1 is manager2


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_load_corrupted_adapter(self, adapter_config, temp_dirs):
        # Create corrupted adapter file
        adapter_path = temp_dirs["adapters_dir"] / "corrupt_adapter.pt"
        with open(adapter_path, "w") as f:
            f.write("not a valid pytorch file")

        manager = AdapterManager(adapter_config)

        with pytest.raises(Exception):  # torch.load will raise
            manager.load_adapter("corrupt")

    def test_profile_metadata_missing_fields(self, adapter_config, temp_dirs, mock_adapter_state):
        # Save adapter
        adapter_path = temp_dirs["adapters_dir"] / "profile1_adapter.pt"
        torch.save(mock_adapter_state, adapter_path)

        # Save incomplete profile metadata
        profile_path = temp_dirs["profiles_dir"] / "profile1.json"
        with open(profile_path, "w") as f:
            json.dump({"name": "Test"}, f)  # Missing most fields

        manager = AdapterManager(adapter_config)
        info = manager.get_adapter_info("profile1")

        # Should use defaults for missing fields
        assert info is not None
        assert info.profile_name == "Test"
        assert info.rank == 8  # Default
        assert info.sample_count == 0  # Default

    def test_device_placement(self, temp_dirs, create_test_adapter):
        create_test_adapter("profile1")

        config = AdapterManagerConfig(
            adapters_dir=temp_dirs["adapters_dir"],
            profiles_dir=temp_dirs["profiles_dir"],
            device="cpu",
        )

        manager = AdapterManager(config)
        state_dict = manager.load_adapter("profile1")

        # Verify all tensors are on correct device
        for tensor in state_dict.values():
            assert tensor.device == torch.device("cpu")


class TestTensorShapes:
    """Test that adapter tensors have correct shapes."""

    def test_adapter_shapes_valid(self, adapter_config, mock_adapter_state):
        manager = AdapterManager(adapter_config)

        # Validate shapes
        lora_a = mock_adapter_state["lora_adapters.content_proj.lora_A"]
        lora_b = mock_adapter_state["lora_adapters.content_proj.lora_B"]

        # LoRA A should be (rank, in_features)
        assert lora_a.shape[0] == 8  # rank

        # LoRA B should be (out_features, rank)
        assert lora_b.shape[1] == 8  # rank

        # A and B should be compatible
        assert lora_a.shape[0] == lora_b.shape[1]

    def test_adapter_weights_not_nan(self, adapter_config, create_test_adapter):
        create_test_adapter("profile1")

        manager = AdapterManager(adapter_config)
        state_dict = manager.load_adapter("profile1")

        # Verify no NaN values
        for name, tensor in state_dict.items():
            assert not torch.isnan(tensor).any(), f"NaN found in {name}"

    def test_adapter_weights_finite(self, adapter_config, create_test_adapter):
        create_test_adapter("profile1")

        manager = AdapterManager(adapter_config)
        state_dict = manager.load_adapter("profile1")

        # Verify all values are finite
        for name, tensor in state_dict.items():
            assert torch.isfinite(tensor).all(), f"Non-finite values in {name}"
