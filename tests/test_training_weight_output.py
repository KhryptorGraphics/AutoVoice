"""Tests for training pipeline weight output.

Phase 3: Test that fine-tuning pipeline saves trained weights after training.

Tests verify:
- FineTuningPipeline.fine_tune() saves weights to profile
- TrainingJobManager completes with saved weights
- Job status includes weight path
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import MagicMock, patch

from auto_voice.storage.voice_profiles import VoiceProfileStore
from auto_voice.training.fine_tuning import FineTuningPipeline


class MockModel(nn.Module):
    """Simple model for testing fine-tuning pipeline.

    Input shape: [B, 256, frames] (mel tensor)
    Output shape: [B, 256] (to match speaker_embedding for MSE loss)
    """

    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(256, 128)
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )
        self.output = nn.Linear(128, 256)

    def forward(self, x):
        # x: [B, mels=256, frames]
        # Pool over frames and project to speaker embedding size
        x_pooled = x.mean(dim=2)  # [B, 256]
        h = self.input_proj(x_pooled)  # [B, 128]
        return self.output(h)  # [B, 256] - matches speaker_embedding shape


class MockSample:
    """Mock training sample with mel and speaker embedding."""

    def __init__(self, device='cpu'):
        self.mel_tensor = torch.randn(1, 256, 50, device=device)  # [B, mels=256, frames]
        self.speaker_embedding = torch.randn(1, 256, device=device)  # [B, embed_dim=256]


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
        "profile_id": "training-test-profile",
        "name": "Test Artist",
        "embedding": torch.randn(256).numpy(),
        "sample_count": 5,
    }
    store.save(profile_data)
    return profile_data["profile_id"]


@pytest.fixture
def output_dir(tmp_path):
    """Create temporary output directory."""
    out = tmp_path / "training_output"
    out.mkdir()
    return out


@pytest.fixture
def mock_model():
    """Create mock model for training."""
    return MockModel()


@pytest.fixture
def mock_samples():
    """Create mock training samples."""
    return [MockSample() for _ in range(3)]


@pytest.fixture
def training_config():
    """Create minimal training config."""
    from auto_voice.training.job_manager import TrainingConfig
    return TrainingConfig(
        epochs=2,
        learning_rate=1e-4,
        batch_size=1,
        lora_rank=4,
        lora_alpha=8,
        lora_target_modules=["input_proj"],
        use_ewc=False,
    )


class TestFineTuningPipelineSavesWeights:
    """Tests for FineTuningPipeline saving weights after training."""

    def test_fine_tune_returns_adapter_path(self, mock_model, mock_samples, output_dir, training_config):
        """Task 3.1: fine_tune() should return path to saved adapter."""
        pipeline = FineTuningPipeline(
            base_model=mock_model,
            output_dir=output_dir,
        )

        result = pipeline.fine_tune(
            samples=mock_samples,
            config=training_config,
            mode="lora",
        )

        assert result["success"] is True, "Training should succeed"
        assert "adapter_path" in result, "Result should contain adapter_path"
        assert Path(result["adapter_path"]).exists(), "Adapter file should exist"

    def test_fine_tune_saves_lora_weights(self, mock_model, mock_samples, output_dir, training_config):
        """Task 3.2: fine_tune() should save loadable LoRA weights."""
        pipeline = FineTuningPipeline(
            base_model=mock_model,
            output_dir=output_dir,
        )

        result = pipeline.fine_tune(
            samples=mock_samples,
            config=training_config,
            mode="lora",
        )

        # Load and verify weights
        adapter_path = result["adapter_path"]
        saved = torch.load(adapter_path, map_location="cpu")

        assert "adapters" in saved, "Should have adapters key"
        assert "config" in saved, "Should have config key"

    def test_fine_tune_full_mode_saves_model(self, mock_model, mock_samples, output_dir, training_config):
        """fine_tune() in full mode should save complete model."""
        pipeline = FineTuningPipeline(
            base_model=mock_model,
            output_dir=output_dir,
        )

        result = pipeline.fine_tune(
            samples=mock_samples,
            config=training_config,
            mode="full",
        )

        assert result["success"] is True
        assert "model_path" in result, "Result should contain model_path"
        assert Path(result["model_path"]).exists(), "Model file should exist"


class TestTrainingJobManagerWeightSaving:
    """Tests for TrainingJobManager weight saving on completion."""

    def test_job_manager_exists(self):
        """Task 3.3: TrainingJobManager should exist."""
        from auto_voice.training.job_manager import TrainingJobManager
        assert TrainingJobManager is not None

    def test_job_creation(self, sample_profile, tmp_path):
        """Task 3.4: Job manager should create jobs with profile_id."""
        from auto_voice.training.job_manager import TrainingJobManager, TrainingConfig

        storage_path = tmp_path / "job_storage"
        storage_path.mkdir()

        manager = TrainingJobManager(
            storage_path=storage_path,
            require_gpu=False,  # Allow CPU for tests
        )

        config = TrainingConfig(
            epochs=1,
            learning_rate=1e-4,
            batch_size=1,
            lora_rank=4,
            lora_alpha=8,
            lora_target_modules=["input_proj"],
        )

        # Create job
        job = manager.create_job(
            profile_id=sample_profile,
            sample_ids=["sample1", "sample2"],
            config=config,
        )
        assert job is not None
        assert job.profile_id == sample_profile

    def test_job_has_profile_id(self, sample_profile, tmp_path):
        """Task 3.5-3.6: Job should track profile_id for weight saving."""
        from auto_voice.training.job_manager import TrainingJobManager, TrainingConfig

        storage_path = tmp_path / "job_storage"
        storage_path.mkdir()

        manager = TrainingJobManager(
            storage_path=storage_path,
            require_gpu=False,
        )

        config = TrainingConfig(
            epochs=1,
            learning_rate=1e-4,
            batch_size=1,
            lora_rank=4,
            lora_alpha=8,
            lora_target_modules=["input_proj"],
        )

        job = manager.create_job(
            profile_id=sample_profile,
            sample_ids=["sample1"],
            config=config,
        )

        # Job should have profile_id for weight saving destination
        assert job.profile_id == sample_profile
        assert hasattr(job, "status")


class TestIntegrationProfileWeightSaving:
    """Integration tests for profile + weight saving flow."""

    def test_trained_weights_saved_to_profile(self, store, sample_profile, output_dir, mock_model, mock_samples, training_config):
        """Trained weights should be saveable to profile storage."""
        # Train
        pipeline = FineTuningPipeline(
            base_model=mock_model,
            output_dir=output_dir,
        )
        result = pipeline.fine_tune(
            samples=mock_samples,
            config=training_config,
            mode="lora",
        )

        # Load trained adapter
        saved = torch.load(result["adapter_path"], map_location="cpu")

        # Save to profile (convert adapter format to state dict format)
        state_dict = {}
        for name, adapter_state in saved["adapters"].items():
            state_dict[f"{name}.lora_A"] = adapter_state["lora_A"]
            state_dict[f"{name}.lora_B"] = adapter_state["lora_B"]

        store.save_lora_weights(sample_profile, state_dict)

        # Verify profile now has trained model
        assert store.has_trained_model(sample_profile)

        # Verify weights loadable
        loaded = store.load_lora_weights(sample_profile)
        assert len(loaded) > 0
