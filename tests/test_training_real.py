"""Tests for training with real encoder features."""
import tempfile
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from auto_voice.models.so_vits_svc import SoVitsSvc
from auto_voice.training.trainer import Trainer, VoiceDataset


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def training_dir(tmp_path):
    """Create a directory with synthetic singing audio files (4s each for embedding)."""
    sr = 22050
    for i in range(4):
        freq = 220 + i * 110  # 220, 330, 440, 550 Hz
        t = np.linspace(0, 4, sr * 4, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        sf.write(str(tmp_path / f"song_{i}.wav"), audio, sr)
    return str(tmp_path)


@pytest.fixture
def model(device):
    return SoVitsSvc().to(device)


@pytest.fixture
def trainer(model, device, tmp_path):
    config = {
        'epochs': 3,
        'batch_size': 2,
        'checkpoint_dir': str(tmp_path / 'ckpt'),
        'log_every': 1,
        'save_every': 10,
        'sample_rate': 22050,
    }
    return Trainer(model, config=config, device=device)


class TestTrainerSetup:
    """Test trainer initialization with real encoders."""

    @pytest.mark.smoke
    def test_trainer_has_encoders(self, trainer):
        assert trainer.content_encoder is not None
        assert trainer.pitch_encoder is not None

    def test_speaker_embedding_not_set(self, trainer):
        assert trainer.speaker_embedding is None

    def test_set_speaker_embedding(self, trainer, training_dir):
        trainer.set_speaker_embedding(training_dir)
        assert trainer.speaker_embedding is not None
        assert trainer.speaker_embedding.shape == (256,)

    def test_set_speaker_embedding_empty_dir_raises(self, trainer, tmp_path):
        empty = str(tmp_path / "empty")
        os.makedirs(empty)
        with pytest.raises(RuntimeError, match="No audio files"):
            trainer.set_speaker_embedding(empty)


class TestTrainingWithRealFeatures:
    """Test training loop uses real encoder outputs."""

    def test_train_raises_without_speaker_embedding(self, trainer, training_dir):
        """Training should fail if speaker embedding not set."""
        with pytest.raises(RuntimeError, match="Speaker embedding not set"):
            trainer.train(training_dir)

    @pytest.mark.slow
    def test_training_loss_recorded(self, trainer, training_dir):
        trainer.set_speaker_embedding(training_dir)
        trainer.train(training_dir)
        assert len(trainer.train_losses) == 3  # 3 epochs

    @pytest.mark.slow
    def test_training_loss_finite(self, trainer, training_dir):
        trainer.set_speaker_embedding(training_dir)
        trainer.train(training_dir)
        for loss in trainer.train_losses:
            assert np.isfinite(loss)
            assert loss >= 0

    @pytest.mark.slow
    def test_training_produces_finite_loss(self, training_dir, device, tmp_path):
        """Training with real features produces finite loss values."""
        model = SoVitsSvc().to(device)
        t = Trainer(model, config={
            'epochs': 3,
            'batch_size': 2,
            'checkpoint_dir': str(tmp_path / 'ckpt2'),
            'learning_rate': 1e-5,  # Lower LR to prevent divergence with random encoders
            'gradient_clip': 0.5,
            'sample_rate': 22050,
            'log_every': 1,
            'save_every': 10,
        }, device=device)
        t.set_speaker_embedding(training_dir)
        t.train(training_dir)
        # All losses should be finite (not NaN/Inf)
        for loss in t.train_losses:
            assert np.isfinite(loss), f"Loss is not finite: {loss}"


class TestSpecComputation:
    """Test spectrogram computation helper."""

    def test_spec_shape(self, trainer, device):
        audio = torch.randn(2, 32768, device=device)
        spec = trainer._compute_spec(audio, target_frames=64)
        assert spec.shape == (2, 513, 64)

    def test_spec_non_negative(self, trainer, device):
        audio = torch.randn(2, 32768, device=device)
        spec = trainer._compute_spec(audio, target_frames=64)
        assert (spec >= 0).all()


class TestCheckpointing:
    """Test checkpoint save/load with real training state."""

    @pytest.mark.slow
    def test_checkpoint_save_load_cycle(self, trainer, training_dir, device, tmp_path):
        trainer.set_speaker_embedding(training_dir)
        trainer.train(training_dir)

        ckpt_path = str(tmp_path / 'test_ckpt.pth')
        trainer.save_checkpoint(ckpt_path)
        assert os.path.exists(ckpt_path)

        # Create new trainer and load
        model2 = SoVitsSvc().to(device)
        trainer2 = Trainer(model2, config=trainer.config, device=device)
        trainer2.load_checkpoint(ckpt_path)
        assert trainer2.global_step == trainer.global_step
        assert trainer2.current_epoch == trainer.current_epoch


class TestVoiceDataset:
    """Test dataset returns correct fields."""

    def test_dataset_item_keys(self, training_dir):
        ds = VoiceDataset(training_dir, sample_rate=22050, segment_length=32768)
        item = ds[0]
        assert 'audio' in item
        assert 'mel' in item
        assert 'f0' in item
        assert 'path' in item

    def test_dataset_audio_shape(self, training_dir):
        ds = VoiceDataset(training_dir, sample_rate=22050, segment_length=32768)
        item = ds[0]
        assert item['audio'].shape == (32768,)

    def test_dataset_mel_shape(self, training_dir):
        ds = VoiceDataset(training_dir, sample_rate=22050, segment_length=32768)
        item = ds[0]
        assert item['mel'].shape[0] == 80  # n_mels

    def test_dataset_f0_shape(self, training_dir):
        ds = VoiceDataset(training_dir, sample_rate=22050, segment_length=32768)
        item = ds[0]
        assert len(item['f0']) > 0
