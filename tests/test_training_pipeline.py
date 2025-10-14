"""Test training pipeline with sample dataset.

Tests complete training workflow:
1. Load sample dataset
2. Initialize models and optimizers
3. Run training epochs
4. Validate checkpointing
5. Test checkpoint loading
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Dict, Any

# Import AutoVoice components
from auto_voice.training.data_pipeline import (
    VoiceDataset,
    AudioConfig,
    create_voice_dataloader,
    preprocess_batch
)
from auto_voice.training.trainer import VoiceTrainer
from auto_voice.training.checkpoint_manager import CheckpointManager
from auto_voice.models.transformer import VoiceTransformer
from auto_voice.models.hifigan import HiFiGANGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleTestModel(nn.Module):
    """Simple test model for pipeline validation."""

    def __init__(self, input_dim: int = 80, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, mel_spec: torch.Tensor, speaker_id: torch.Tensor = None) -> torch.Tensor:
        """Forward pass."""
        batch_size, seq_len, feat_dim = mel_spec.shape

        # Encode
        x = mel_spec.reshape(-1, feat_dim)
        encoded = self.encoder(x)

        # Decode
        output = self.decoder(encoded)

        return output.reshape(batch_size, seq_len, feat_dim)

    def forward_for_training(self, mel_spec: torch.Tensor, speaker_id: torch.Tensor = None) -> torch.Tensor:
        """Training-compatible forward pass."""
        return self.forward(mel_spec, speaker_id)


def test_dataloader_creation():
    """Test dataset and dataloader creation."""
    logger.info("=" * 60)
    logger.info("TEST 1: DataLoader Creation")
    logger.info("=" * 60)

    # Configure dataset
    data_dir = "data/sample_audio"
    metadata_file = "data/sample_audio/metadata_train.json"

    audio_config = AudioConfig(
        sample_rate=22050,
        n_mels=80,
        max_audio_length=8192,
        min_audio_length=1024
    )

    # Create dataset
    dataset = VoiceDataset(
        data_dir=data_dir,
        metadata_file=metadata_file,
        audio_config=audio_config
    )

    logger.info(f"✓ Dataset created: {len(dataset)} samples")

    # Create dataloader
    dataloader = create_voice_dataloader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        distributed=False
    )

    logger.info(f"✓ DataLoader created: {len(dataloader)} batches")

    # Test batch loading
    batch = next(iter(dataloader))
    logger.info(f"✓ Batch loaded successfully")
    logger.info(f"  - mel_spec shape: {batch['mel_spec'].shape}")
    logger.info(f"  - audio shape: {batch['audio'].shape}")
    logger.info(f"  - speaker_id shape: {batch['speaker_id'].shape}")

    return dataloader


def test_model_initialization():
    """Test model, optimizer, and loss initialization."""
    logger.info("=" * 60)
    logger.info("TEST 2: Model Initialization")
    logger.info("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create simple test model
    model = SimpleTestModel(input_dim=80, hidden_dim=128)
    model = model.to(device)

    logger.info(f"✓ Model initialized")
    logger.info(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    logger.info(f"✓ Optimizer initialized (Adam, lr=1e-4)")

    # Create loss function
    criterion = nn.MSELoss()
    logger.info(f"✓ Loss function initialized (MSELoss)")

    return model, optimizer, criterion, device


def test_training_step(model, dataloader, optimizer, criterion, device):
    """Test single training step."""
    logger.info("=" * 60)
    logger.info("TEST 3: Training Step")
    logger.info("=" * 60)

    model.train()

    # Get batch
    batch = next(iter(dataloader))
    mel_spec = batch['mel_spec'].to(device)
    speaker_id = batch['speaker_id'].to(device)

    # Forward pass
    optimizer.zero_grad()
    output = model.forward_for_training(mel_spec, speaker_id)

    # Compute loss (reconstruction task)
    loss = criterion(output, mel_spec)

    logger.info(f"✓ Forward pass completed")
    logger.info(f"  - Input shape: {mel_spec.shape}")
    logger.info(f"  - Output shape: {output.shape}")
    logger.info(f"  - Loss: {loss.item():.6f}")

    # Backward pass
    loss.backward()
    optimizer.step()

    logger.info(f"✓ Backward pass completed")

    return loss.item()


def test_training_loop(model, dataloader, optimizer, criterion, device, num_epochs: int = 3):
    """Test complete training loop."""
    logger.info("=" * 60)
    logger.info(f"TEST 4: Training Loop ({num_epochs} epochs)")
    logger.info("=" * 60)

    model.train()
    train_losses = []

    for epoch in range(num_epochs):
        epoch_losses = []

        for batch_idx, batch in enumerate(dataloader):
            mel_spec = batch['mel_spec'].to(device)
            speaker_id = batch['speaker_id'].to(device)

            # Training step
            optimizer.zero_grad()
            output = model.forward_for_training(mel_spec, speaker_id)
            loss = criterion(output, mel_spec)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

            if (batch_idx + 1) % 2 == 0:
                logger.info(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.6f}")

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(avg_loss)
        logger.info(f"✓ Epoch {epoch+1} completed, Average Loss: {avg_loss:.6f}")

    logger.info(f"✓ Training completed")
    logger.info(f"  - Initial loss: {train_losses[0]:.6f}")
    logger.info(f"  - Final loss: {train_losses[-1]:.6f}")
    logger.info(f"  - Loss reduction: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.2f}%")

    return train_losses


def test_checkpoint_manager(model, optimizer, device):
    """Test checkpoint save/load functionality."""
    logger.info("=" * 60)
    logger.info("TEST 5: Checkpoint Management")
    logger.info("=" * 60)

    checkpoint_dir = "checkpoints/test"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create checkpoint manager
    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        max_keep=3,
        keep_best=2,
        metric_name='val_loss'
    )

    logger.info(f"✓ CheckpointManager initialized")

    # Save checkpoint
    checkpoint_mgr.save(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=5,
        step=100,
        metrics={'train_loss': 0.5, 'val_loss': 0.55},
        config={'test': 'config'}
    )

    logger.info(f"✓ Checkpoint saved")

    # List checkpoints
    checkpoints = checkpoint_mgr.list_checkpoints()
    logger.info(f"✓ Found {len(checkpoints)} checkpoint(s)")

    # Modify model weights to verify loading
    original_param = list(model.parameters())[0].clone()
    with torch.no_grad():
        list(model.parameters())[0].fill_(0.0)

    modified_param = list(model.parameters())[0].clone()
    logger.info(f"✓ Model weights modified (verification)")

    # Load checkpoint
    checkpoint_path = checkpoints[0]['path']
    loaded_checkpoint = checkpoint_mgr.load(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        device=device
    )

    restored_param = list(model.parameters())[0].clone()

    logger.info(f"✓ Checkpoint loaded")
    logger.info(f"  - Epoch: {loaded_checkpoint['epoch']}")
    logger.info(f"  - Step: {loaded_checkpoint['step']}")
    logger.info(f"  - Metrics: {loaded_checkpoint['metrics']}")

    # Verify weights restored
    weights_match = torch.allclose(original_param, restored_param, atol=1e-6)
    logger.info(f"✓ Weight restoration: {'PASSED' if weights_match else 'FAILED'}")

    # Test resume functionality
    epoch, step, checkpoint = checkpoint_mgr.resume_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        device=device
    )

    logger.info(f"✓ Resume from checkpoint: epoch={epoch}, step={step}")

    return checkpoint_mgr


def test_trainer_integration(dataloader, device):
    """Test VoiceTrainer class integration."""
    logger.info("=" * 60)
    logger.info("TEST 6: VoiceTrainer Integration")
    logger.info("=" * 60)

    # Create model and optimizer
    model = SimpleTestModel(input_dim=80, hidden_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    criterion = nn.MSELoss()

    # Configuration
    config = {
        'training': {
            'batch_size': 4,
            'learning_rate': 1e-4,
            'num_epochs': 2,
            'checkpoint_dir': 'checkpoints/test_trainer'
        },
        'model': {
            'input_dim': 80,
            'hidden_dim': 128
        }
    }

    # Create trainer
    trainer = VoiceTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        config=config
    )

    logger.info(f"✓ VoiceTrainer initialized")

    # Train for one epoch
    train_loss = trainer.train_epoch(dataloader, epoch=0)
    logger.info(f"✓ Training epoch completed: loss={train_loss:.6f}")

    # Save checkpoint via trainer
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    checkpoint_path = os.path.join(config['training']['checkpoint_dir'], 'test_checkpoint.pt')
    trainer.save_checkpoint(checkpoint_path, epoch=0, best_loss=train_loss)
    logger.info(f"✓ Checkpoint saved via trainer")

    # Load checkpoint via trainer
    loaded_checkpoint = trainer.load_checkpoint(checkpoint_path)
    logger.info(f"✓ Checkpoint loaded via trainer")
    logger.info(f"  - Epoch: {loaded_checkpoint.get('epoch', 'N/A')}")
    logger.info(f"  - Best loss: {loaded_checkpoint.get('best_loss', 'N/A'):.6f}")

    return trainer


def run_all_tests():
    """Run all training pipeline tests."""
    logger.info("\n" + "=" * 60)
    logger.info("AUTOVOICE TRAINING PIPELINE TEST SUITE")
    logger.info("=" * 60 + "\n")

    try:
        # Test 1: DataLoader
        dataloader = test_dataloader_creation()

        # Test 2: Model initialization
        model, optimizer, criterion, device = test_model_initialization()

        # Test 3: Single training step
        test_training_step(model, dataloader, optimizer, criterion, device)

        # Test 4: Training loop
        test_training_loop(model, dataloader, optimizer, criterion, device, num_epochs=3)

        # Test 5: Checkpoint management
        test_checkpoint_manager(model, optimizer, device)

        # Test 6: Trainer integration
        test_trainer_integration(dataloader, device)

        logger.info("\n" + "=" * 60)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
