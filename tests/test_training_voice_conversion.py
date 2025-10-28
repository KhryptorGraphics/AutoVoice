"""
Comprehensive test suite for voice conversion training components.

Tests cover:
- PairedVoiceDataset with source-target audio pairs
- SingingAugmentation transforms
- Voice conversion loss functions
- VoiceConversionTrainer
- End-to-end training workflows
"""

import json
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.auto_voice.training import (
    PairedVoiceDataset,
    SingingAugmentation,
    PairedVoiceCollator,
    create_paired_voice_dataloader,
    create_paired_train_val_datasets,
    VoiceConversionTrainer,
    TrainingConfig,
    PitchConsistencyLoss,
    SpeakerSimilarityLoss,
    KLDivergenceLoss,
    FlowLogLikelihoodLoss,
    AudioConfig
)
from src.auto_voice.models import SingingVoiceConverter


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary directory with synthetic audio data."""
    # Create train and val directories
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir()
    val_dir.mkdir()

    # Generate synthetic audio pairs
    sample_rate = 44100
    duration = 1  # 1 second
    num_samples = sample_rate * duration

    train_pairs = []
    for i in range(10):
        # Generate sine waves
        t = np.linspace(0, duration, num_samples)
        source_audio = np.sin(2 * np.pi * (220 + i * 10) * t).astype(np.float32)
        target_audio = np.sin(2 * np.pi * (330 + i * 15) * t).astype(np.float32)

        # Save as numpy files
        source_file = train_dir / f"source_{i}.npy"
        target_file = train_dir / f"target_{i}.npy"
        np.save(source_file, source_audio)
        np.save(target_file, target_audio)

        train_pairs.append({
            'source_file': f"source_{i}.npy",
            'target_file': f"target_{i}.npy",
            'source_speaker_id': f"speaker_{i % 3}",
            'target_speaker_id': f"speaker_{(i + 1) % 3}",
            'duration': duration
        })

    # Generate validation pairs
    val_pairs = []
    for i in range(3):
        t = np.linspace(0, duration, num_samples)
        source_audio = np.sin(2 * np.pi * (440 + i * 20) * t).astype(np.float32)
        target_audio = np.sin(2 * np.pi * (550 + i * 25) * t).astype(np.float32)

        source_file = val_dir / f"source_{i}.npy"
        target_file = val_dir / f"target_{i}.npy"
        np.save(source_file, source_audio)
        np.save(target_file, target_audio)

        val_pairs.append({
            'source_file': f"source_{i}.npy",
            'target_file': f"target_{i}.npy",
            'source_speaker_id': f"speaker_{i}",
            'target_speaker_id': f"speaker_{(i + 1) % 3}",
            'duration': duration
        })

    # Create metadata files
    train_metadata = tmp_path / "train_pairs.json"
    val_metadata = tmp_path / "val_pairs.json"

    with open(train_metadata, 'w') as f:
        json.dump({'pairs': train_pairs}, f)

    with open(val_metadata, 'w') as f:
        json.dump({'pairs': val_pairs}, f)

    return {
        'root_dir': tmp_path,
        'train_dir': train_dir,
        'val_dir': val_dir,
        'train_metadata': train_metadata,
        'val_metadata': val_metadata
    }


@pytest.fixture
def audio_config():
    """Create audio configuration for testing."""
    return AudioConfig(
        sample_rate=44100,
        n_mels=80,
        n_fft=2048,
        hop_length=512,
        win_length=2048
    )


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing."""
    batch_size = 2
    audio_len = 44100  # 1 second
    mel_len = audio_len // 512  # hop_length = 512
    n_mels = 80

    return {
        'source_audio': torch.randn(batch_size, audio_len),
        'target_audio': torch.randn(batch_size, audio_len),
        'source_mel': torch.randn(batch_size, mel_len, n_mels),
        'target_mel': torch.randn(batch_size, mel_len, n_mels),
        'source_f0': torch.rand(batch_size, mel_len) * 500 + 100,  # F0 in 100-600 Hz
        'target_f0': torch.rand(batch_size, mel_len) * 500 + 100,
        'source_speaker_emb': torch.randn(batch_size, 256),
        'target_speaker_emb': torch.randn(batch_size, 256),
        'source_speaker_id': ['speaker_0', 'speaker_1'],
        'target_speaker_id': ['speaker_1', 'speaker_2'],
        'lengths': torch.LongTensor([mel_len, mel_len]),
        'mel_mask': torch.ones(batch_size, 1, mel_len)
    }


class TestPairedVoiceDataset:
    """Unit tests for PairedVoiceDataset."""

    @pytest.mark.training
    @pytest.mark.unit
    def test_dataset_initialization(self, temp_data_dir, audio_config):
        """Test dataset initialization."""
        dataset = PairedVoiceDataset(
            data_dir=temp_data_dir['train_dir'],
            metadata_file=str(temp_data_dir['train_metadata']),
            audio_config=audio_config,
            extract_f0=False,  # Skip F0 extraction for speed
            extract_speaker_emb=False  # Skip speaker embedding for speed
        )

        assert len(dataset) == 10
        assert dataset.audio_config == audio_config

    @pytest.mark.training
    @pytest.mark.unit
    def test_dataset_getitem(self, temp_data_dir, audio_config):
        """Test dataset item retrieval."""
        dataset = PairedVoiceDataset(
            data_dir=temp_data_dir['train_dir'],
            metadata_file=str(temp_data_dir['train_metadata']),
            audio_config=audio_config,
            extract_f0=False,
            extract_speaker_emb=False
        )

        item = dataset[0]

        # Check required keys
        assert 'source_audio' in item
        assert 'target_audio' in item
        assert 'source_mel' in item
        assert 'target_mel' in item
        assert 'source_speaker_id' in item
        assert 'target_speaker_id' in item
        assert 'lengths' in item

        # Check tensor types and shapes
        assert isinstance(item['source_audio'], torch.Tensor)
        assert isinstance(item['target_audio'], torch.Tensor)
        assert isinstance(item['source_mel'], torch.Tensor)
        assert isinstance(item['target_mel'], torch.Tensor)

        # Check for NaN/Inf
        assert not torch.isnan(item['source_audio']).any()
        assert not torch.isinf(item['source_audio']).any()
        assert not torch.isnan(item['target_mel']).any()
        assert not torch.isinf(item['target_mel']).any()

    @pytest.mark.training
    @pytest.mark.unit
    def test_dataset_caching(self, temp_data_dir, audio_config):
        """Test dataset caching mechanism."""
        dataset = PairedVoiceDataset(
            data_dir=temp_data_dir['train_dir'],
            metadata_file=str(temp_data_dir['train_metadata']),
            audio_config=audio_config,
            extract_f0=False,
            extract_speaker_emb=False,
            cache_size=5
        )

        # First access
        item1 = dataset[0]

        # Check if cached
        assert 0 in dataset.cache

        # Second access (should be from cache)
        item2 = dataset[0]

        # Verify data matches
        assert torch.equal(item1['source_audio'], item2['source_audio'])
        assert torch.equal(item1['target_mel'], item2['target_mel'])


class TestSingingAugmentation:
    """Unit tests for SingingAugmentation transforms."""

    @pytest.mark.training
    @pytest.mark.unit
    def test_noise_injection_snr(self):
        """Test noise injection with SNR control."""
        # Create test data
        audio_len = 44100
        data = {
            'source_audio': torch.randn(audio_len),
            'target_audio': torch.randn(audio_len)
        }

        # Apply noise injection
        augmented = SingingAugmentation.noise_injection_snr(
            data,
            snr_db_range=(30, 30)  # Fixed SNR for testing
        )

        # Check that audio is modified
        assert not torch.equal(data['source_audio'], augmented['source_audio'])
        assert not torch.equal(data['target_audio'], augmented['target_audio'])

    @pytest.mark.training
    @pytest.mark.unit
    def test_formant_shift(self):
        """Test formant shift augmentation."""
        mel_len = 100
        n_mels = 80
        data = {
            'source_mel': torch.randn(mel_len, n_mels),
            'target_mel': torch.randn(mel_len, n_mels)
        }

        # Apply formant shift
        augmented = SingingAugmentation.formant_shift(
            data,
            semitone_range=(-1, 1)
        )

        # Check that mel is modified
        assert not torch.equal(data['source_mel'], augmented['source_mel'])
        assert augmented['source_mel'].shape == data['source_mel'].shape


class TestPairedVoiceCollator:
    """Unit tests for PairedVoiceCollator."""

    @pytest.mark.training
    @pytest.mark.unit
    def test_collator_batching(self, audio_config):
        """Test collator batching with variable lengths."""
        # Create samples with different lengths
        samples = []
        for i in range(3):
            audio_len = 44100 + i * 1000
            mel_len = audio_len // 512

            sample = {
                'source_audio': torch.randn(audio_len),
                'target_audio': torch.randn(audio_len),
                'source_mel': torch.randn(mel_len, 80),
                'target_mel': torch.randn(mel_len, 80),
                'source_speaker_id': f'speaker_{i}',
                'target_speaker_id': f'speaker_{i+1}',
                'lengths': torch.LongTensor([mel_len])
            }
            samples.append(sample)

        collator = PairedVoiceCollator()
        batch = collator(samples)

        # Check batch shapes
        assert batch['source_audio'].shape[0] == 3  # batch size
        assert batch['target_mel'].shape[0] == 3
        assert batch['lengths'].shape[0] == 3

        # Check padding (all should be same length)
        max_audio_len = max(s['source_audio'].size(0) for s in samples)
        assert batch['source_audio'].shape[1] == max_audio_len

    @pytest.mark.training
    @pytest.mark.unit
    def test_collator_mask_creation(self):
        """Test mask creation for variable lengths."""
        samples = [
            {
                'source_audio': torch.randn(44100),
                'target_audio': torch.randn(44100),
                'source_mel': torch.randn(86, 80),
                'target_mel': torch.randn(86, 80),
                'source_speaker_id': 'speaker_0',
                'target_speaker_id': 'speaker_1',
                'lengths': torch.LongTensor([86])
            },
            {
                'source_audio': torch.randn(44100),
                'target_audio': torch.randn(44100),
                'source_mel': torch.randn(70, 80),
                'target_mel': torch.randn(70, 80),
                'source_speaker_id': 'speaker_1',
                'target_speaker_id': 'speaker_2',
                'lengths': torch.LongTensor([70])
            }
        ]

        collator = PairedVoiceCollator()
        batch = collator(samples)

        # Check mask shape
        assert 'mel_mask' in batch
        assert batch['mel_mask'].shape == (2, 1, 86)  # [B, 1, max_len]

        # Check mask values
        assert batch['mel_mask'][0, 0, :86].sum() == 86  # First sample, all valid
        assert batch['mel_mask'][1, 0, :70].sum() == 70  # Second sample, 70 valid


class TestVoiceConversionLosses:
    """Unit tests for voice conversion loss functions."""

    @pytest.mark.training
    @pytest.mark.unit
    def test_kl_divergence_loss(self):
        """Test KL divergence loss computation."""
        loss_fn = KLDivergenceLoss()

        # Create test inputs
        batch_size = 2
        latent_dim = 192
        seq_len = 100

        z_mean = torch.randn(batch_size, latent_dim, seq_len)
        z_logvar = torch.randn(batch_size, latent_dim, seq_len)

        # Compute loss
        loss = loss_fn(z_mean, z_logvar)

        # Check loss properties
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss >= 0  # KL divergence is non-negative

        # Test with standard normal (should be near zero)
        z_mean_zero = torch.zeros(batch_size, latent_dim, seq_len)
        z_logvar_zero = torch.zeros(batch_size, latent_dim, seq_len)
        loss_zero = loss_fn(z_mean_zero, z_logvar_zero)
        assert loss_zero.abs() < 1e-3

    @pytest.mark.training
    @pytest.mark.unit
    def test_flow_log_likelihood_loss(self):
        """Test flow log-likelihood loss."""
        loss_fn = FlowLogLikelihoodLoss()

        # Create test inputs
        batch_size = 2
        latent_dim = 192
        seq_len = 100

        logdet = torch.randn(batch_size, seq_len)
        u = torch.randn(batch_size, latent_dim, seq_len)

        # Compute loss
        loss = loss_fn(logdet, u)

        # Check loss properties
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar


class TestVoiceConversionTrainer:
    """Integration tests for VoiceConversionTrainer."""

    @pytest.mark.training
    @pytest.mark.integration
    def test_trainer_initialization(self):
        """Test VoiceConversionTrainer initialization."""
        # Create model
        model = SingingVoiceConverter(
            latent_dim=192,
            mel_channels=80,
            content_encoder_type='cnn_fallback',  # Use fallback for testing
            num_flows=2  # Smaller for testing
        )

        # Create config
        config = TrainingConfig(
            batch_size=2,
            learning_rate=1e-4,
            num_epochs=1,
            voice_conversion_mode=True,
            use_amp=False  # Disable for CPU testing
        )

        # Initialize trainer
        trainer = VoiceConversionTrainer(
            model=model,
            config=config,
            experiment_name='test_vc'
        )

        # Check initialization
        assert trainer.model is not None
        assert trainer.pitch_loss is not None
        assert trainer.speaker_loss is not None
        assert trainer.kl_loss is not None
        assert trainer.flow_loss is not None

    @pytest.mark.training
    @pytest.mark.integration
    def test_forward_pass(self, sample_batch):
        """Test forward pass through trainer."""
        model = SingingVoiceConverter(
            latent_dim=192,
            mel_channels=80,
            content_encoder_type='cnn_fallback',
            num_flows=2
        )

        config = TrainingConfig(
            batch_size=2,
            learning_rate=1e-4,
            voice_conversion_mode=True,
            use_amp=False
        )

        trainer = VoiceConversionTrainer(model=model, config=config)

        # Forward pass
        outputs = trainer._forward_pass(sample_batch)

        # Check outputs
        assert 'pred_mel' in outputs or 'output' in outputs
        assert isinstance(outputs, dict)

    @pytest.mark.training
    @pytest.mark.integration
    def test_loss_computation(self, sample_batch):
        """Test loss computation."""
        model = SingingVoiceConverter(
            latent_dim=192,
            mel_channels=80,
            content_encoder_type='cnn_fallback',
            num_flows=2
        )

        config = TrainingConfig(
            batch_size=2,
            voice_conversion_mode=True,
            use_amp=False
        )

        trainer = VoiceConversionTrainer(model=model, config=config)

        # Create mock predictions
        mel_len = sample_batch['target_mel'].size(1)
        predictions = {
            'pred_mel': torch.randn_like(sample_batch['target_mel']),
            'z_mean': torch.randn(2, 192, mel_len),
            'z_logvar': torch.randn(2, 192, mel_len),
            'logdet': torch.randn(2, mel_len),
            'u': torch.randn(2, 192, mel_len)
        }

        # Compute losses
        losses = trainer._compute_voice_conversion_losses(predictions, sample_batch)

        # Check losses
        assert 'total' in losses
        assert 'mel_reconstruction' in losses
        assert 'kl_divergence' in losses
        assert 'flow_likelihood' in losses
        assert losses['total'] >= 0


class TestVoiceConversionTrainingWorkflow:
    """End-to-end tests for voice conversion training."""

    @pytest.mark.training
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_full_training_loop(self, temp_data_dir, audio_config):
        """Test complete training workflow."""
        # Create dataset
        train_dataset = PairedVoiceDataset(
            data_dir=temp_data_dir['train_dir'],
            metadata_file=str(temp_data_dir['train_metadata']),
            audio_config=audio_config,
            extract_f0=False,
            extract_speaker_emb=False
        )

        # Create dataloader
        train_loader = create_paired_voice_dataloader(
            dataset=train_dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0  # Single-threaded for testing
        )

        # Create model
        model = SingingVoiceConverter(
            latent_dim=192,
            mel_channels=80,
            content_encoder_type='cnn_fallback',
            num_flows=2
        )

        # Create config
        config = TrainingConfig(
            batch_size=2,
            learning_rate=1e-4,
            num_epochs=2,  # Only 2 epochs for testing
            voice_conversion_mode=True,
            use_amp=False,
            log_interval=1
        )

        # Initialize trainer
        trainer = VoiceConversionTrainer(model=model, config=config)

        # Train for 2 epochs
        for epoch in range(2):
            losses = trainer.train_epoch(train_loader, epoch)

            # Check losses are computed
            assert 'total' in losses
            assert losses['total'] > 0

        # Verify training progressed
        assert trainer.global_step > 0
        assert trainer.epoch > 0


@pytest.mark.training
@pytest.mark.integration
def test_create_paired_datasets(temp_data_dir, audio_config):
    """Test helper function for creating train/val datasets."""
    train_dataset, val_dataset = create_paired_train_val_datasets(
        data_dir=temp_data_dir['train_dir'],
        train_metadata=str(temp_data_dir['train_metadata']),
        val_metadata=str(temp_data_dir['val_metadata']),
        audio_config=audio_config,
        extract_f0=False,
        extract_speaker_emb=False
    )

    assert len(train_dataset) == 10
    assert len(val_dataset) == 3


@pytest.mark.training
@pytest.mark.integration
def test_dataloader_creation(temp_data_dir, audio_config):
    """Test dataloader creation."""
    dataset = PairedVoiceDataset(
        data_dir=temp_data_dir['train_dir'],
        metadata_file=str(temp_data_dir['train_metadata']),
        audio_config=audio_config,
        extract_f0=False,
        extract_speaker_emb=False
    )

    dataloader = create_paired_voice_dataloader(
        dataset=dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )

    # Test iteration
    batch = next(iter(dataloader))

    assert 'source_audio' in batch
    assert 'target_mel' in batch
    assert batch['source_audio'].shape[0] == 2  # batch size
