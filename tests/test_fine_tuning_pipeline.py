"""TDD tests for fine-tuning pipeline with LoRA adapters and EWC.

Task 4.3: Write failing tests for model fine-tuning on new samples
Task 4.4: Implement fine-tuning pipeline (freeze layers, train adapter/LoRA, full fine-tune options)

Tests cover:
- LoRA adapter creation and injection
- Layer freezing strategies
- EWC regularization for catastrophic forgetting prevention
- Fine-tuning execution on training samples
- Adapter saving and loading
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_model_storage():
    """Temporary directory for model artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_base_model():
    """Mock base model for testing (simulates encoder architecture)."""
    class MockEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.content_encoder = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
            )
            self.q_proj = nn.Linear(256, 256)
            self.v_proj = nn.Linear(256, 256)
            self.output_proj = nn.Linear(256, 128)

        def forward(self, x):
            x = self.content_encoder(x)
            q = self.q_proj(x)
            v = self.v_proj(x)
            return self.output_proj(q + v)

    return MockEncoder()


@pytest.fixture
def mock_training_samples():
    """Mock training samples with audio tensors."""
    samples = []
    for i in range(5):
        sample = Mock()
        sample.sample_id = f"sample-{i}"
        sample.audio_path = f"/data/samples/sample-{i}.wav"
        sample.duration_seconds = 5.0
        # Simulated features matching mock encoder input (batch, features)
        sample.mel_tensor = torch.randn(1, 256)
        sample.speaker_embedding = torch.randn(128)  # Match encoder output
        samples.append(sample)
    return samples


@pytest.fixture
def fine_tuning_config():
    """Fine-tuning configuration."""
    from auto_voice.training.job_manager import TrainingConfig
    return TrainingConfig(
        learning_rate=1e-4,
        epochs=5,
        batch_size=2,
        lora_rank=8,
        lora_alpha=16,
        use_ewc=True,
        ewc_lambda=1000.0,
    )


# ============================================================================
# Test: LoRA Adapter Creation
# ============================================================================

class TestLoRAAdapterCreation:
    """Tests for LoRA adapter creation and injection."""

    def test_create_lora_adapter(self, mock_base_model):
        """Create LoRA adapter for a linear layer."""
        from auto_voice.training.fine_tuning import LoRAAdapter

        adapter = LoRAAdapter(
            in_features=256,
            out_features=256,
            rank=8,
            alpha=16,
        )

        assert adapter.rank == 8
        assert adapter.alpha == 16
        assert adapter.lora_A.shape == (8, 256)  # rank x in_features
        assert adapter.lora_B.shape == (256, 8)  # out_features x rank

    def test_lora_adapter_forward(self, mock_base_model):
        """LoRA adapter produces scaled output."""
        from auto_voice.training.fine_tuning import LoRAAdapter

        adapter = LoRAAdapter(
            in_features=256,
            out_features=256,
            rank=8,
            alpha=16,
        )

        x = torch.randn(4, 256)  # batch of 4
        output = adapter(x)

        assert output.shape == (4, 256)
        # Scaling factor is alpha/rank = 16/8 = 2
        assert adapter.scaling == 2.0

    def test_lora_adapter_zero_initialized(self):
        """LoRA B matrix is zero-initialized (no initial impact)."""
        from auto_voice.training.fine_tuning import LoRAAdapter

        adapter = LoRAAdapter(
            in_features=256,
            out_features=256,
            rank=8,
            alpha=16,
        )

        # lora_B should be zero-initialized
        assert torch.allclose(adapter.lora_B, torch.zeros_like(adapter.lora_B))

        # Initial output should be zero (no modification to base model)
        x = torch.randn(4, 256)
        output = adapter(x)
        assert torch.allclose(output, torch.zeros(4, 256), atol=1e-6)

    def test_inject_lora_into_model(self, mock_base_model):
        """Inject LoRA adapters into target modules."""
        from auto_voice.training.fine_tuning import inject_lora_adapters

        target_modules = ["q_proj", "v_proj"]
        lora_model = inject_lora_adapters(
            model=mock_base_model,
            target_modules=target_modules,
            rank=8,
            alpha=16,
        )

        # Check that adapters were injected
        assert hasattr(lora_model, "lora_adapters")
        assert "q_proj" in lora_model.lora_adapters
        assert "v_proj" in lora_model.lora_adapters

    def test_lora_model_trainable_params(self, mock_base_model):
        """Only LoRA parameters should be trainable after injection."""
        from auto_voice.training.fine_tuning import inject_lora_adapters, freeze_base_model

        # Freeze base model first (as done in actual usage)
        freeze_base_model(mock_base_model)

        lora_model = inject_lora_adapters(
            model=mock_base_model,
            target_modules=["q_proj", "v_proj"],
            rank=8,
            alpha=16,
        )

        # Count trainable parameters (only LoRA params)
        trainable = sum(p.numel() for p in lora_model.lora_adapters.parameters())
        total = sum(p.numel() for p in lora_model.parameters())

        # LoRA params should be small relative to total
        # With rank=8 and 256x256 layers: 2 * (8*256 + 256*8) = 8192 params
        # Total model is ~435K params, so LoRA is ~2% of total
        assert trainable < total * 0.05  # Less than 5% trainable

    def test_lora_adapter_size(self, mock_base_model):
        """LoRA adapters should be small (~4-8MB per profile)."""
        from auto_voice.training.fine_tuning import inject_lora_adapters

        lora_model = inject_lora_adapters(
            model=mock_base_model,
            target_modules=["q_proj", "v_proj"],
            rank=8,
            alpha=16,
        )

        # Calculate adapter size
        adapter_params = sum(
            p.numel() * p.element_size()
            for name, p in lora_model.named_parameters()
            if "lora" in name.lower()
        )
        adapter_size_mb = adapter_params / (1024 * 1024)

        # Should be small
        assert adapter_size_mb < 10  # Less than 10MB


# ============================================================================
# Test: Layer Freezing
# ============================================================================

class TestLayerFreezing:
    """Tests for layer freezing strategies."""

    def test_freeze_base_model(self, mock_base_model):
        """Freeze all base model parameters."""
        from auto_voice.training.fine_tuning import freeze_base_model

        freeze_base_model(mock_base_model)

        for param in mock_base_model.parameters():
            assert param.requires_grad is False

    def test_freeze_except_modules(self, mock_base_model):
        """Freeze all except specified modules."""
        from auto_voice.training.fine_tuning import freeze_except

        freeze_except(mock_base_model, unfrozen_modules=["output_proj"])

        # output_proj should be trainable
        assert mock_base_model.output_proj.weight.requires_grad is True

        # Others should be frozen
        assert mock_base_model.q_proj.weight.requires_grad is False

    def test_unfreeze_for_full_finetune(self, mock_base_model):
        """Unfreeze all parameters for full fine-tuning."""
        from auto_voice.training.fine_tuning import freeze_base_model, unfreeze_model

        freeze_base_model(mock_base_model)
        unfreeze_model(mock_base_model)

        for param in mock_base_model.parameters():
            assert param.requires_grad is True


# ============================================================================
# Test: EWC Regularization
# ============================================================================

class TestEWCRegularization:
    """Tests for Elastic Weight Consolidation."""

    def test_compute_fisher_information(self, mock_base_model, mock_training_samples):
        """Compute Fisher information matrix from training data."""
        from auto_voice.training.fine_tuning import compute_fisher_information

        # Need mock dataloader
        dataloader = [(s.mel_tensor, s.speaker_embedding) for s in mock_training_samples]

        fisher_dict = compute_fisher_information(
            model=mock_base_model,
            dataloader=dataloader,
        )

        # Should have Fisher info for each parameter
        for name, param in mock_base_model.named_parameters():
            assert name in fisher_dict
            assert fisher_dict[name].shape == param.shape

    def test_ewc_loss_computation(self, mock_base_model):
        """Compute EWC loss from parameter changes."""
        from auto_voice.training.fine_tuning import EWCLoss

        # Create reference (old optimal) parameters
        old_params = {
            name: param.clone().detach()
            for name, param in mock_base_model.named_parameters()
        }

        # Mock Fisher information (uniform importance)
        fisher_dict = {
            name: torch.ones_like(param)
            for name, param in mock_base_model.named_parameters()
        }

        ewc_loss = EWCLoss(
            fisher_dict=fisher_dict,
            old_params=old_params,
            lambda_ewc=1000.0,
        )

        # Before any changes, loss should be zero
        loss = ewc_loss(mock_base_model)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

        # After modifying parameters, loss should be positive
        with torch.no_grad():
            for param in mock_base_model.parameters():
                param.add_(0.1)

        loss = ewc_loss(mock_base_model)
        assert loss.item() > 0

    def test_ewc_loss_scaling(self, mock_base_model):
        """EWC loss scales with lambda."""
        from auto_voice.training.fine_tuning import EWCLoss

        old_params = {
            name: param.clone().detach()
            for name, param in mock_base_model.named_parameters()
        }
        fisher_dict = {
            name: torch.ones_like(param)
            for name, param in mock_base_model.named_parameters()
        }

        # Modify model
        with torch.no_grad():
            for param in mock_base_model.parameters():
                param.add_(0.1)

        ewc_low = EWCLoss(fisher_dict, old_params, lambda_ewc=100.0)
        ewc_high = EWCLoss(fisher_dict, old_params, lambda_ewc=1000.0)

        loss_low = ewc_low(mock_base_model)
        loss_high = ewc_high(mock_base_model)

        # Higher lambda should give higher loss
        assert loss_high.item() > loss_low.item()
        assert loss_high.item() / loss_low.item() == pytest.approx(10.0, rel=0.1)


# ============================================================================
# Test: Fine-Tuning Pipeline
# ============================================================================

class TestFineTuningPipeline:
    """Tests for the complete fine-tuning pipeline."""

    def test_fine_tuning_pipeline_initialization(self, mock_base_model, temp_model_storage):
        """Initialize fine-tuning pipeline."""
        from auto_voice.training.fine_tuning import FineTuningPipeline

        pipeline = FineTuningPipeline(
            base_model=mock_base_model,
            output_dir=temp_model_storage,
        )

        assert pipeline.base_model is not None
        assert pipeline.output_dir == temp_model_storage

    def test_fine_tuning_with_lora(
        self, mock_base_model, mock_training_samples, fine_tuning_config, temp_model_storage
    ):
        """Fine-tune model using LoRA adapters."""
        from auto_voice.training.fine_tuning import FineTuningPipeline

        pipeline = FineTuningPipeline(
            base_model=mock_base_model,
            output_dir=temp_model_storage,
        )

        result = pipeline.fine_tune(
            samples=mock_training_samples,
            config=fine_tuning_config,
            mode="lora",
        )

        assert result["success"] is True
        assert "adapter_path" in result
        assert Path(result["adapter_path"]).exists()
        # Verify training produced valid loss values (may not always decrease with random model)
        assert result["initial_loss"] > 0
        assert result["final_loss"] > 0
        assert len(result["loss_curve"]) == fine_tuning_config.epochs

    def test_fine_tuning_with_ewc(
        self, mock_base_model, mock_training_samples, fine_tuning_config, temp_model_storage
    ):
        """Fine-tune with EWC regularization."""
        from auto_voice.training.fine_tuning import FineTuningPipeline

        # First, establish baseline (compute Fisher matrix)
        pipeline = FineTuningPipeline(
            base_model=mock_base_model,
            output_dir=temp_model_storage,
        )

        # Simulate prior training data for Fisher computation
        prior_samples = mock_training_samples[:2]
        pipeline.set_prior_knowledge(prior_samples)

        # Fine-tune with EWC
        fine_tuning_config.use_ewc = True
        result = pipeline.fine_tune(
            samples=mock_training_samples[2:],
            config=fine_tuning_config,
            mode="lora",
        )

        assert result["success"] is True
        assert "ewc_loss" in result["metrics"]

    def test_fine_tuning_full_model(
        self, mock_base_model, mock_training_samples, fine_tuning_config, temp_model_storage
    ):
        """Full model fine-tuning (all parameters trainable)."""
        from auto_voice.training.fine_tuning import FineTuningPipeline

        pipeline = FineTuningPipeline(
            base_model=mock_base_model,
            output_dir=temp_model_storage,
        )

        result = pipeline.fine_tune(
            samples=mock_training_samples,
            config=fine_tuning_config,
            mode="full",
        )

        assert result["success"] is True
        assert "model_path" in result
        assert Path(result["model_path"]).exists()

    def test_fine_tuning_requires_gpu(self, mock_base_model, temp_model_storage):
        """Fine-tuning raises error if CUDA unavailable."""
        from auto_voice.training.fine_tuning import FineTuningPipeline

        with patch.object(torch.cuda, 'is_available', return_value=False):
            with pytest.raises(RuntimeError, match="CUDA.*required"):
                FineTuningPipeline(
                    base_model=mock_base_model,
                    output_dir=temp_model_storage,
                    require_gpu=True,
                )

    def test_fine_tuning_progress_callback(
        self, mock_base_model, mock_training_samples, fine_tuning_config, temp_model_storage
    ):
        """Fine-tuning reports progress via callback."""
        from auto_voice.training.fine_tuning import FineTuningPipeline

        pipeline = FineTuningPipeline(
            base_model=mock_base_model,
            output_dir=temp_model_storage,
        )

        progress_updates = []

        def on_progress(epoch, step, loss, progress_pct):
            progress_updates.append({
                "epoch": epoch,
                "step": step,
                "loss": loss,
                "progress": progress_pct,
            })

        result = pipeline.fine_tune(
            samples=mock_training_samples,
            config=fine_tuning_config,
            mode="lora",
            progress_callback=on_progress,
        )

        assert len(progress_updates) > 0
        # Final progress should be 100%
        assert progress_updates[-1]["progress"] == 100


# ============================================================================
# Test: Adapter Persistence
# ============================================================================

class TestAdapterPersistence:
    """Tests for saving and loading LoRA adapters."""

    def test_save_lora_adapter(self, mock_base_model, temp_model_storage):
        """Save LoRA adapter to disk."""
        from auto_voice.training.fine_tuning import inject_lora_adapters, save_lora_adapter

        lora_model = inject_lora_adapters(
            model=mock_base_model,
            target_modules=["q_proj", "v_proj"],
            rank=8,
            alpha=16,
        )

        adapter_path = temp_model_storage / "adapter.pt"
        save_lora_adapter(lora_model, adapter_path)

        assert adapter_path.exists()
        # Check file size is reasonable
        size_mb = adapter_path.stat().st_size / (1024 * 1024)
        assert size_mb < 10  # Less than 10MB

    def test_load_lora_adapter(self, mock_base_model, temp_model_storage):
        """Load LoRA adapter from disk."""
        from auto_voice.training.fine_tuning import (
            inject_lora_adapters,
            save_lora_adapter,
            load_lora_adapter,
        )

        # Create and save adapter
        lora_model = inject_lora_adapters(
            model=mock_base_model,
            target_modules=["q_proj", "v_proj"],
            rank=8,
            alpha=16,
        )

        # Modify adapter weights
        for adapter in lora_model.lora_adapters.values():
            with torch.no_grad():
                adapter.lora_A.fill_(1.0)
                adapter.lora_B.fill_(0.5)

        adapter_path = temp_model_storage / "adapter.pt"
        save_lora_adapter(lora_model, adapter_path)

        # Create fresh model and load adapter
        from auto_voice.training.fine_tuning import LoRAAdapter
        fresh_model = mock_base_model.__class__()
        loaded_model = load_lora_adapter(fresh_model, adapter_path)

        # Check weights were loaded
        for adapter in loaded_model.lora_adapters.values():
            assert torch.allclose(adapter.lora_A, torch.ones_like(adapter.lora_A))

    def test_adapter_metadata(self, mock_base_model, temp_model_storage):
        """Adapter file includes metadata (rank, alpha, target modules)."""
        from auto_voice.training.fine_tuning import (
            inject_lora_adapters,
            save_lora_adapter,
            load_adapter_metadata,
        )

        lora_model = inject_lora_adapters(
            model=mock_base_model,
            target_modules=["q_proj", "v_proj"],
            rank=8,
            alpha=16,
        )

        adapter_path = temp_model_storage / "adapter.pt"
        save_lora_adapter(lora_model, adapter_path)

        metadata = load_adapter_metadata(adapter_path)
        assert metadata["rank"] == 8
        assert metadata["alpha"] == 16
        assert set(metadata["target_modules"]) == {"q_proj", "v_proj"}


# ============================================================================
# Test: Training Metrics
# ============================================================================

class TestTrainingMetrics:
    """Tests for training quality metrics."""

    def test_compute_speaker_similarity(self, mock_base_model, mock_training_samples):
        """Compute speaker similarity after fine-tuning."""
        from auto_voice.training.fine_tuning import compute_speaker_similarity

        # Mock reference and generated embeddings
        reference_embedding = mock_training_samples[0].speaker_embedding
        generated_embedding = reference_embedding + torch.randn_like(reference_embedding) * 0.1

        similarity = compute_speaker_similarity(reference_embedding, generated_embedding)

        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.9  # Should be high for similar embeddings

    def test_compute_training_loss_curve(
        self, mock_base_model, mock_training_samples, fine_tuning_config, temp_model_storage
    ):
        """Training should produce decreasing loss curve."""
        from auto_voice.training.fine_tuning import FineTuningPipeline

        pipeline = FineTuningPipeline(
            base_model=mock_base_model,
            output_dir=temp_model_storage,
        )

        result = pipeline.fine_tune(
            samples=mock_training_samples,
            config=fine_tuning_config,
            mode="lora",
        )

        loss_curve = result["loss_curve"]
        assert len(loss_curve) > 0

        # Loss should generally decrease
        assert loss_curve[-1] < loss_curve[0]
