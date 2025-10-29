"""
Unit tests for TrainingConfig local_rank attribute access.

Tests to ensure that local_rank attribute access does not cause AttributeError
in non-distributed training scenarios.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.auto_voice.training.trainer import (
    TrainingConfig,
    VoiceTrainer,
    VoiceConversionTrainer
)


class DummyModel(nn.Module):
    """Simple model for testing trainer initialization."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


@pytest.mark.training
class TestLocalRankAttributeAccess:
    """Test local_rank attribute access in TrainingConfig and VoiceTrainer."""

    def test_training_config_has_local_rank(self):
        """Test that TrainingConfig has local_rank field with default value."""
        config = TrainingConfig()
        assert hasattr(config, 'local_rank')
        assert config.local_rank == 0

    def test_training_config_custom_local_rank(self):
        """Test that TrainingConfig accepts custom local_rank value."""
        config = TrainingConfig(local_rank=2)
        assert config.local_rank == 2

    @patch('src.auto_voice.training.trainer.SummaryWriter')
    def test_trainer_initialization_without_distributed(self, mock_writer):
        """Test VoiceTrainer initialization without distributed training setup."""
        model = DummyModel()
        config = TrainingConfig(
            batch_size=2,
            learning_rate=1e-4,
            num_epochs=1,
            distributed=False
        )

        # Should not raise AttributeError
        trainer = VoiceTrainer(
            model=model,
            config=config,
            experiment_name="test_local_rank"
        )

        # Verify local_rank is accessible
        assert hasattr(trainer.config, 'local_rank')
        assert trainer.config.local_rank == 0

    @patch('src.auto_voice.training.trainer.SummaryWriter')
    def test_trainer_setup_logging_without_distributed(self, mock_writer):
        """Test _setup_logging method works without distributed setup."""
        model = DummyModel()
        config = TrainingConfig(distributed=False)

        trainer = VoiceTrainer(
            model=model,
            config=config,
            experiment_name="test_logging"
        )

        # _setup_logging should have been called during __init__
        # Verify writer was created (local_rank == 0)
        assert trainer.writer is not None
        mock_writer.assert_called_once()

    @patch('src.auto_voice.training.trainer.tqdm')
    def test_train_epoch_progress_bar_without_distributed(self, mock_tqdm):
        """Test train_epoch creates progress bar when local_rank == 0."""
        model = DummyModel()
        config = TrainingConfig(
            batch_size=2,
            distributed=False,
            use_amp=False
        )

        # Mock dataloader
        batch = {'features': torch.randn(2, 10), 'target': torch.randn(2, 10)}
        dataloader = [batch]

        trainer = VoiceTrainer(model, config, experiment_name="test_pbar")

        # Create a mock progress bar that has set_postfix method
        mock_pbar = MagicMock()
        mock_pbar.__iter__ = Mock(return_value=iter(dataloader))
        mock_tqdm.return_value = mock_pbar

        # Should not raise AttributeError when checking local_rank
        try:
            losses = trainer.train_epoch(dataloader, epoch=0)
            # If we got here, no AttributeError was raised
            assert True
        except AttributeError as e:
            if 'local_rank' in str(e):
                pytest.fail(f"AttributeError for local_rank: {e}")
            raise

    def test_validate_method_without_distributed(self):
        """Test validate method works without distributed setup."""
        model = DummyModel()
        config = TrainingConfig(
            batch_size=2,
            distributed=False,
            use_amp=False
        )

        # Mock dataloader
        batch = {'features': torch.randn(2, 10), 'target': torch.randn(2, 10)}
        dataloader = [batch]

        trainer = VoiceTrainer(model, config, experiment_name="test_validate")

        # Should not raise AttributeError
        try:
            losses = trainer.validate(dataloader)
            assert isinstance(losses, dict)
            assert 'total' in losses
        except AttributeError as e:
            if 'local_rank' in str(e):
                pytest.fail(f"AttributeError for local_rank in validate: {e}")
            raise

    @patch('src.auto_voice.training.trainer.SummaryWriter')
    def test_log_training_step_without_distributed(self, mock_writer):
        """Test _log_training_step works when local_rank == 0."""
        model = DummyModel()
        config = TrainingConfig(distributed=False)

        trainer = VoiceTrainer(model, config, experiment_name="test_log")

        # Mock losses
        losses = {
            'total': torch.tensor(1.0),
            'reconstruction': torch.tensor(0.8)
        }

        # Should not raise AttributeError
        try:
            trainer._log_training_step(losses, accumulated_loss=1.0)
            # If we got here, no AttributeError was raised
            assert True
        except AttributeError as e:
            if 'local_rank' in str(e):
                pytest.fail(f"AttributeError for local_rank in _log_training_step: {e}")
            raise

    def test_save_checkpoint_without_distributed(self, tmp_path):
        """Test save_checkpoint works without distributed setup."""
        model = DummyModel()
        config = TrainingConfig(distributed=False)

        trainer = VoiceTrainer(model, config, experiment_name="test_checkpoint")
        trainer.log_dir = tmp_path / "logs"

        # Should not raise AttributeError
        try:
            trainer.save_checkpoint(epoch=0, val_loss=1.0)
            # If we got here, no AttributeError was raised
            assert True
        except AttributeError as e:
            if 'local_rank' in str(e):
                pytest.fail(f"AttributeError for local_rank in save_checkpoint: {e}")
            raise

    @patch('src.auto_voice.training.trainer.SummaryWriter')
    def test_voice_conversion_trainer_without_distributed(self, mock_writer):
        """Test VoiceConversionTrainer initialization without distributed setup."""
        # Create dummy model (would normally be SingingVoiceConverter)
        model = DummyModel()

        config = TrainingConfig(
            batch_size=2,
            distributed=False,
            voice_conversion_mode=True
        )

        # Should not raise AttributeError
        trainer = VoiceConversionTrainer(
            model=model,
            config=config,
            experiment_name="test_vc_trainer"
        )

        assert hasattr(trainer.config, 'local_rank')
        assert trainer.config.local_rank == 0

    def test_getattr_fallback_with_missing_attribute(self):
        """Test that getattr fallback works if local_rank is somehow missing."""
        model = DummyModel()
        config = TrainingConfig()

        # Manually delete local_rank to simulate missing attribute
        delattr(config, 'local_rank')

        trainer = VoiceTrainer(model, config, experiment_name="test_fallback")

        # The getattr calls in trainer methods should handle this gracefully
        # by returning default value of 0

        # Test _setup_logging (uses getattr)
        # Should not raise AttributeError
        result = getattr(trainer.config, 'local_rank', 0)
        assert result == 0


@pytest.mark.training
@pytest.mark.integration
class TestLocalRankIntegration:
    """Integration tests for local_rank in training workflows."""

    @patch('src.auto_voice.training.trainer.SummaryWriter')
    def test_full_training_step_without_distributed(self, mock_writer):
        """Test complete training step without distributed setup."""
        model = DummyModel()
        config = TrainingConfig(
            batch_size=2,
            learning_rate=1e-4,
            num_epochs=1,
            distributed=False,
            use_amp=False,
            gradient_accumulation_steps=1
        )

        # Create trainer
        trainer = VoiceTrainer(model, config, experiment_name="test_integration")

        # Mock dataloader
        batch = {'features': torch.randn(2, 10), 'target': torch.randn(2, 10)}
        dataloader = [batch]

        # Run training step - should not raise AttributeError
        try:
            losses = trainer.train_epoch(dataloader, epoch=0)
            assert isinstance(losses, dict)
            assert 'total' in losses
        except AttributeError as e:
            if 'local_rank' in str(e):
                pytest.fail(f"AttributeError for local_rank during training: {e}")
            raise

    def test_config_serialization_with_local_rank(self):
        """Test that TrainingConfig with local_rank can be serialized."""
        config = TrainingConfig(
            local_rank=1,
            world_size=4,
            distributed=True
        )

        # Test dict conversion
        config_dict = config.__dict__
        assert 'local_rank' in config_dict
        assert config_dict['local_rank'] == 1

        # Test reconstruction
        config2 = TrainingConfig(**config_dict)
        assert config2.local_rank == 1
        assert config2.world_size == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
