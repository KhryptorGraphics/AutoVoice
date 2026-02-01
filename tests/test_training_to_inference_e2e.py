"""End-to-End tests for training-to-inference integration (Phase 6).

Tests complete flow from training a voice profile to using it for inference.
"""
import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from auto_voice.models.adapter_manager import AdapterManager, AdapterManagerConfig
from auto_voice.inference.sota_pipeline import SOTAConversionPipeline
from auto_voice.inference.realtime_voice_conversion_pipeline import RealtimeVoiceConversionPipeline


@pytest.fixture
def test_audio_5s():
    """Generate 5 seconds of test audio at 24kHz."""
    sample_rate = 24000
    duration = 5.0
    num_samples = int(sample_rate * duration)
    t = torch.linspace(0, duration, num_samples)
    # Generate complex waveform (fundamental + harmonics)
    audio = 0.5 * torch.sin(2 * torch.pi * 440 * t)
    audio += 0.3 * torch.sin(2 * torch.pi * 880 * t)
    audio += 0.2 * torch.sin(2 * torch.pi * 1320 * t)
    return audio, sample_rate


@pytest.fixture
def mock_trained_profile(tmp_path):
    """Create a mock trained profile with adapter and embedding."""
    profile_id = "test-profile-e2e-12345678-1234-1234"

    # Create directories
    trained_models_dir = tmp_path / "trained_models"
    profiles_dir = tmp_path / "voice_profiles"
    trained_models_dir.mkdir(parents=True)
    profiles_dir.mkdir(parents=True)

    # Create mock adapter (LoRA weights)
    adapter_state = {
        "lora_adapters.input_proj.lora_A": torch.randn(8, 1024),
        "lora_adapters.input_proj.lora_B": torch.randn(1024, 8),
        "lora_adapters.speaker_film.gamma_proj.lora_A": torch.randn(8, 256),
        "lora_adapters.speaker_film.gamma_proj.lora_B": torch.randn(256, 8),
        "lora_adapters.speaker_film.beta_proj.lora_A": torch.randn(8, 256),
        "lora_adapters.speaker_film.beta_proj.lora_B": torch.randn(256, 8),
    }
    adapter_path = trained_models_dir / f"{profile_id}_adapter.pt"
    torch.save(adapter_state, adapter_path)

    # Create mock speaker embedding (256-dim, L2-normalized)
    embedding = np.random.randn(256).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)  # L2 normalize

    # Save to both locations
    embedding_path = profiles_dir / f"{profile_id}.npy"
    np.save(embedding_path, embedding)

    # Create profile metadata
    import json
    profile_data = {
        "profile_id": profile_id,
        "name": "Test E2E Profile",
        "created_at": "2026-01-31T12:00:00Z",
        "sample_count": 10,
        "adapter_version": "1.0",
        "adapter_rank": 8,
        "adapter_alpha": 16,
    }
    profile_json = profiles_dir / f"{profile_id}.json"
    with open(profile_json, "w") as f:
        json.dump(profile_data, f)

    return {
        "profile_id": profile_id,
        "adapter_path": adapter_path,
        "embedding_path": embedding_path,
        "trained_models_dir": trained_models_dir,
        "profiles_dir": profiles_dir,
    }


class TestTrainingToInferenceE2E:
    """End-to-end tests for complete training-to-inference flow."""

    @pytest.mark.integration
    def test_adapter_manager_loads_trained_model(self, mock_trained_profile):
        """Task 6.1: Verify AdapterManager can load a trained model."""
        adapter_manager = AdapterManager(AdapterManagerConfig(
            adapters_dir=mock_trained_profile["trained_models_dir"],
            profiles_dir=mock_trained_profile["profiles_dir"],
        ))

        profile_id = mock_trained_profile["profile_id"]

        # Verify adapter exists
        assert adapter_manager.has_adapter(profile_id)

        # Load adapter
        adapter_state = adapter_manager.load_adapter(profile_id)
        assert adapter_state is not None
        assert len(adapter_state) > 0

        # Verify adapter info
        info = adapter_manager.get_adapter_info(profile_id)
        assert info is not None
        assert info.profile_id == profile_id
        assert info.rank == 8
        assert info.alpha == 16

    @pytest.mark.integration
    @pytest.mark.cuda
    def test_sota_pipeline_loads_and_uses_adapter(self, mock_trained_profile, test_audio_5s):
        """Task 6.1: Verify SOTA pipeline can load and use trained adapter."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        profile_id = mock_trained_profile["profile_id"]

        # Create pipeline
        pipeline = SOTAConversionPipeline(
            device=torch.device('cuda'),
            n_steps=1,  # Fast inference
        )

        # Override adapter manager to use test directories
        from auto_voice.models.adapter_manager import AdapterManagerConfig
        pipeline._adapter_manager = AdapterManager(AdapterManagerConfig(
            adapters_dir=mock_trained_profile["trained_models_dir"],
            profiles_dir=mock_trained_profile["profiles_dir"],
        ))

        # Set speaker (loads adapter + embedding)
        pipeline.set_speaker(profile_id)

        # Verify speaker loaded
        assert pipeline.get_current_speaker() == profile_id

        # Verify embedding loaded
        embedding = pipeline.get_speaker_embedding()
        assert embedding is not None
        assert embedding.shape == (256,)
        assert torch.isfinite(embedding).all()

        # Verify L2 normalized
        norm = torch.norm(embedding).item()
        assert abs(norm - 1.0) < 0.01

        # Run conversion
        audio, sample_rate = test_audio_5s
        result = pipeline.convert(audio, sample_rate, embedding)

        # Verify output
        assert 'audio' in result
        assert 'sample_rate' in result
        output_audio = result['audio']
        assert isinstance(output_audio, torch.Tensor)
        assert output_audio.numel() > 0
        assert torch.isfinite(output_audio).all()

    @pytest.mark.integration
    def test_realtime_pipeline_loads_and_uses_adapter(self, mock_trained_profile):
        """Task 6.1: Verify realtime pipeline can load adapter."""
        profile_id = mock_trained_profile["profile_id"]

        # Create pipeline
        pipeline = RealtimeVoiceConversionPipeline(
            device=torch.device('cpu'),  # CPU for speed
        )

        # Override adapter manager
        pipeline._adapter_manager = AdapterManager(AdapterManagerConfig(
            adapters_dir=mock_trained_profile["trained_models_dir"],
            profiles_dir=mock_trained_profile["profiles_dir"],
        ))

        # Set speaker
        pipeline.set_speaker(
            profile_id,
            profiles_dir=str(mock_trained_profile["profiles_dir"])
        )

        # Verify speaker loaded
        assert pipeline.get_current_speaker() == profile_id

        # Verify embedding loaded (stored as _target_embedding in realtime pipeline)
        assert pipeline._target_embedding is not None
        assert pipeline._target_embedding.shape == (256,)


class TestErrorHandling:
    """Task 6.2: Test error handling for missing and corrupt adapters."""

    @pytest.mark.integration
    def test_missing_adapter_error(self, tmp_path):
        """Verify clear error when adapter doesn't exist."""
        adapter_manager = AdapterManager(AdapterManagerConfig(
            adapters_dir=tmp_path / "trained_models",
            profiles_dir=tmp_path / "voice_profiles",
        ))

        # Create directories (AdapterManager might create them first)
        (tmp_path / "trained_models").mkdir(exist_ok=True)
        (tmp_path / "voice_profiles").mkdir(exist_ok=True)

        profile_id = "nonexistent-profile-12345678"

        # Verify adapter doesn't exist
        assert not adapter_manager.has_adapter(profile_id)

        # Attempt to load should raise clear error
        with pytest.raises(FileNotFoundError, match="No adapter found for profile"):
            adapter_manager.load_adapter(profile_id)

    @pytest.mark.integration
    @pytest.mark.cuda
    def test_pipeline_error_on_missing_adapter(self, tmp_path):
        """Verify pipeline raises error when adapter missing."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        pipeline = SOTAConversionPipeline(device=torch.device('cuda'), n_steps=1)

        # Override adapter manager with empty directory
        pipeline._adapter_manager = AdapterManager(AdapterManagerConfig(
            adapters_dir=tmp_path / "empty",
            profiles_dir=tmp_path / "empty",
        ))
        (tmp_path / "empty").mkdir(exist_ok=True)

        # Attempt to set speaker with nonexistent adapter
        with pytest.raises(FileNotFoundError, match="No trained adapter found"):
            pipeline.set_speaker("nonexistent-profile-12345678")

    @pytest.mark.integration
    def test_corrupt_adapter_error(self, tmp_path):
        """Task 6.2: Verify error handling for corrupt adapter file."""
        trained_models_dir = tmp_path / "trained_models"
        profiles_dir = tmp_path / "voice_profiles"
        trained_models_dir.mkdir()
        profiles_dir.mkdir()

        profile_id = "corrupt-profile-12345678"

        # Create corrupt adapter file (not valid torch file)
        adapter_path = trained_models_dir / f"{profile_id}_adapter.pt"
        with open(adapter_path, "w") as f:
            f.write("This is not a valid PyTorch file!")

        # Create valid embedding (so adapter is detected)
        embedding = np.random.randn(256).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        np.save(profiles_dir / f"{profile_id}.npy", embedding)

        adapter_manager = AdapterManager(AdapterManagerConfig(
            adapters_dir=trained_models_dir,
            profiles_dir=profiles_dir,
        ))

        # Adapter should be detected
        assert adapter_manager.has_adapter(profile_id)

        # But loading should fail with clear error
        with pytest.raises(Exception):  # torch.load will raise
            adapter_manager.load_adapter(profile_id)


class TestBothPipelines:
    """Task 6.3: Test both pipelines with the same profile."""

    @pytest.mark.integration
    @pytest.mark.cuda
    @pytest.mark.slow
    def test_both_pipelines_use_same_adapter(self, mock_trained_profile, test_audio_5s):
        """Verify both pipelines can use the same trained adapter."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        profile_id = mock_trained_profile["profile_id"]
        audio, sample_rate = test_audio_5s

        # Test SOTA (quality) pipeline
        sota_pipeline = SOTAConversionPipeline(device=torch.device('cuda'), n_steps=1)
        sota_pipeline._adapter_manager = AdapterManager(AdapterManagerConfig(
            adapters_dir=mock_trained_profile["trained_models_dir"],
            profiles_dir=mock_trained_profile["profiles_dir"],
        ))

        sota_pipeline.set_speaker(profile_id)
        sota_embedding = sota_pipeline.get_speaker_embedding()

        # Verify SOTA loaded successfully
        assert sota_embedding is not None
        assert torch.norm(sota_embedding).item() - 1.0 < 0.01

        # Test Realtime pipeline
        realtime_pipeline = RealtimeVoiceConversionPipeline(device=torch.device('cpu'))
        realtime_pipeline._adapter_manager = AdapterManager(AdapterManagerConfig(
            adapters_dir=mock_trained_profile["trained_models_dir"],
            profiles_dir=mock_trained_profile["profiles_dir"],
        ))

        realtime_pipeline.set_speaker(
            profile_id,
            profiles_dir=str(mock_trained_profile["profiles_dir"])
        )

        # Verify realtime loaded successfully
        assert realtime_pipeline.get_current_speaker() == profile_id
        assert realtime_pipeline._target_embedding is not None

        # Both should have loaded the same embedding
        realtime_embedding = torch.from_numpy(realtime_pipeline._target_embedding)
        assert torch.allclose(sota_embedding.cpu(), realtime_embedding, atol=1e-5)

        # Run conversions (verify outputs exist, not comparing quality)
        sota_result = sota_pipeline.convert(audio, sample_rate, sota_embedding)
        assert 'audio' in sota_result
        assert sota_result['audio'].numel() > 0

        # Realtime conversion would need actual audio processing
        # Just verify the embedding is set correctly
        assert realtime_pipeline._target_embedding.shape == (256,)


class TestMemoryCleanup:
    """Task 6.4: Verify memory cleanup after conversion."""

    @pytest.mark.integration
    @pytest.mark.cuda
    def test_adapter_cache_bounded(self, mock_trained_profile):
        """Verify adapter cache doesn't grow unbounded."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        adapter_manager = AdapterManager(AdapterManagerConfig(
            adapters_dir=mock_trained_profile["trained_models_dir"],
            profiles_dir=mock_trained_profile["profiles_dir"],
            cache_size=2,  # Small cache for testing
        ))

        profile_id = mock_trained_profile["profile_id"]

        # Load multiple times
        for i in range(5):
            adapter_manager.load_adapter(profile_id, use_cache=True)

        # Cache should not exceed max size
        stats = adapter_manager.get_cache_stats()
        assert stats['cached_adapters'] <= 2

    @pytest.mark.integration
    @pytest.mark.cuda
    def test_clear_speaker_frees_resources(self, mock_trained_profile):
        """Verify clear_speaker() releases adapter and embedding."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        profile_id = mock_trained_profile["profile_id"]

        pipeline = SOTAConversionPipeline(device=torch.device('cuda'), n_steps=1)
        pipeline._adapter_manager = AdapterManager(AdapterManagerConfig(
            adapters_dir=mock_trained_profile["trained_models_dir"],
            profiles_dir=mock_trained_profile["profiles_dir"],
        ))

        # Load speaker
        pipeline.set_speaker(profile_id)
        assert pipeline.get_current_speaker() == profile_id
        assert pipeline.get_speaker_embedding() is not None

        # Clear speaker
        pipeline.clear_speaker()

        # Verify cleared
        assert pipeline.get_current_speaker() is None
        assert pipeline.get_speaker_embedding() is None

    @pytest.mark.integration
    @pytest.mark.cuda
    @pytest.mark.slow
    def test_gpu_memory_cleanup(self, mock_trained_profile, test_audio_5s):
        """Verify GPU memory is managed properly during conversion."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        profile_id = mock_trained_profile["profile_id"]
        audio, sample_rate = test_audio_5s

        # Get initial GPU memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()

        pipeline = SOTAConversionPipeline(device=torch.device('cuda'), n_steps=1)
        pipeline._adapter_manager = AdapterManager(AdapterManagerConfig(
            adapters_dir=mock_trained_profile["trained_models_dir"],
            profiles_dir=mock_trained_profile["profiles_dir"],
        ))

        # Load and convert
        pipeline.set_speaker(profile_id)
        embedding = pipeline.get_speaker_embedding()
        result = pipeline.convert(audio, sample_rate, embedding)

        # Clear and force garbage collection
        del result
        pipeline.clear_speaker()
        torch.cuda.empty_cache()

        # Memory should not have grown significantly
        final_memory = torch.cuda.memory_allocated()
        memory_growth = final_memory - initial_memory

        # Allow some growth for model components, but not excessive
        # (This is a soft check - exact threshold depends on model size)
        assert memory_growth < 1e9  # Less than 1GB growth


class TestAdapterValidation:
    """Additional validation tests for adapter format and compatibility."""

    @pytest.mark.integration
    def test_adapter_format_validation(self, mock_trained_profile):
        """Verify adapter format is validated during load."""
        adapter_manager = AdapterManager(AdapterManagerConfig(
            adapters_dir=mock_trained_profile["trained_models_dir"],
            profiles_dir=mock_trained_profile["profiles_dir"],
            auto_validate=True,
        ))

        profile_id = mock_trained_profile["profile_id"]

        # Should load without errors (validation enabled)
        adapter_state = adapter_manager.load_adapter(profile_id)
        assert adapter_state is not None

    @pytest.mark.integration
    def test_embedding_format_validation(self, mock_trained_profile):
        """Verify embedding format is validated."""
        profiles_dir = mock_trained_profile["profiles_dir"]
        profile_id = mock_trained_profile["profile_id"]

        # Load embedding
        embedding_path = profiles_dir / f"{profile_id}.npy"
        embedding = np.load(embedding_path)

        # Verify format
        assert embedding.shape == (256,)
        assert embedding.dtype == np.float32

        # Verify L2 normalized
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01

    @pytest.mark.integration
    def test_adapter_info_accuracy(self, mock_trained_profile):
        """Verify adapter info reflects actual adapter content."""
        adapter_manager = AdapterManager(AdapterManagerConfig(
            adapters_dir=mock_trained_profile["trained_models_dir"],
            profiles_dir=mock_trained_profile["profiles_dir"],
        ))

        profile_id = mock_trained_profile["profile_id"]
        info = adapter_manager.get_adapter_info(profile_id)

        # Verify info matches mock profile
        assert info.profile_name == "Test E2E Profile"
        assert info.rank == 8
        assert info.alpha == 16
        assert info.sample_count == 10
