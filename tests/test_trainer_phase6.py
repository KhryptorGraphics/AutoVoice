"""Additional trainer coverage for orchestration and checkpoint branches."""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn


class FakeContentEncoder(nn.Module):
    """Minimal content encoder for trainer tests."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def to(self, device):
        return self

    def extract_features(self, audio, sr):
        batch_size = audio.shape[0]
        return torch.ones(batch_size, 4, 768, device=audio.device)


class FakePitchEncoder(nn.Module):
    """Minimal pitch encoder for trainer tests."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def to(self, device):
        return self

    def forward(self, f0):
        batch_size = f0.shape[0]
        return torch.ones(batch_size, 4, 256, device=f0.device)


class TrainingTestModel(nn.Module):
    """Simple model compatible with Trainer."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.content_dim = 768
        self.pitch_dim = 256
        self.speaker_dim = 4

    def forward(self, content, pitch, speaker, spec=None):
        batch_size, frames, _ = content.shape
        return self.scale * torch.ones(batch_size, 80, frames, device=content.device)

    def compute_loss(self, outputs, mel):
        loss = ((outputs - mel) ** 2).mean()
        return {
            'total_loss': loss,
            'reconstruction_loss': loss,
        }


def make_batch(batch_size=2, mel_frames=6, audio_length=4096):
    """Build a trainer batch with matching audio/mel/f0 shapes."""
    return {
        'audio': torch.randn(batch_size, audio_length),
        'mel': torch.randn(batch_size, 80, mel_frames),
        'f0': torch.randn(batch_size, mel_frames).abs(),
    }


@pytest.fixture
def trainer_factory(tmp_path):
    """Create trainers with lightweight encoder dependencies."""

    def _build(config=None, model=None):
        from auto_voice.training.trainer import Trainer

        effective_config = {
            'checkpoint_dir': str(tmp_path / 'checkpoints'),
            'epochs': 1,
            'save_every': 1,
            'log_every': 10,
        }
        if config:
            effective_config.update(config)

        with patch('auto_voice.models.encoder.ContentEncoder', FakeContentEncoder), \
             patch('auto_voice.models.encoder.PitchEncoder', FakePitchEncoder):
            return Trainer(model or TrainingTestModel(), config=effective_config, device='cpu')

    return _build


class TestTrainingLoopOrchestration:
    """Exercise train() branches without running full training."""

    def test_trainer_uses_decoder_pitch_feature_contract(self, tmp_path):
        from auto_voice.training.trainer import Trainer

        seen = {}

        class RecordingPitchEncoder(FakePitchEncoder):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                seen["output_size"] = kwargs.get("output_size")

        with patch('auto_voice.models.encoder.ContentEncoder', FakeContentEncoder), \
             patch('auto_voice.models.encoder.PitchEncoder', RecordingPitchEncoder):
            Trainer(
                TrainingTestModel(),
                config={'checkpoint_dir': str(tmp_path / 'checkpoints')},
                device='cpu',
            )

        assert seen["output_size"] == 256

    def test_train_epoch_reports_feature_shape_mismatch(self, trainer_factory):
        trainer = trainer_factory()
        trainer.speaker_embedding = torch.ones(4)
        trainer.pitch_encoder.forward = MagicMock(
            return_value=torch.ones(2, 4, 768)
        )

        with pytest.raises(RuntimeError, match='pitch_dim=768 expected 256'):
            trainer._train_epoch([make_batch()], epoch=0)

    def test_train_resumes_and_adjusts_small_batch_size(self, trainer_factory):
        trainer = trainer_factory({'batch_size': 8, 'epochs': 1, 'save_every': 1})
        train_dataset = MagicMock()
        train_dataset.__len__.return_value = 3
        loaders = []

        def dataloader_spy(dataset, **kwargs):
            loaders.append({'dataset': dataset, 'kwargs': kwargs})
            return f"loader-{len(loaders)}"

        with patch('auto_voice.training.trainer.VoiceDataset', return_value=train_dataset), \
             patch('auto_voice.training.trainer.DataLoader', side_effect=dataloader_spy), \
             patch.object(trainer, 'load_checkpoint') as mock_load_checkpoint, \
             patch.object(trainer, '_train_epoch', return_value=0.25) as mock_train_epoch, \
             patch.object(trainer, 'save_checkpoint') as mock_save_checkpoint, \
             patch.object(trainer.scheduler, 'step') as mock_scheduler_step, \
             patch.object(trainer, 'assess') as mock_assess:
            trainer.train('train_dir', resume_from='resume.pth')

        mock_load_checkpoint.assert_called_once_with('resume.pth')
        assert loaders[0]['kwargs']['batch_size'] == 3
        assert loaders[0]['kwargs']['drop_last'] is False
        mock_train_epoch.assert_called_once_with('loader-1', 0)
        mock_assess.assert_not_called()
        mock_scheduler_step.assert_called_once()
        assert trainer.train_losses == [0.25]
        assert [call.args[0] for call in mock_save_checkpoint.call_args_list] == [
            trainer.checkpoint_dir / 'checkpoint_epoch_1.pth',
            trainer.checkpoint_dir / 'final.pth',
        ]

    def test_train_with_validation_saves_best_checkpoint(self, trainer_factory, tmp_path):
        trainer = trainer_factory({'batch_size': 4, 'epochs': 1, 'save_every': 1})
        train_dataset = MagicMock()
        train_dataset.__len__.return_value = 5
        val_dataset = MagicMock()
        val_dataset.__len__.return_value = 2
        val_dir = tmp_path / 'val'
        val_dir.mkdir()
        loaders = []

        def dataloader_spy(dataset, **kwargs):
            loaders.append({'dataset': dataset, 'kwargs': kwargs})
            return f"loader-{len(loaders)}"

        with patch('auto_voice.training.trainer.VoiceDataset', side_effect=[train_dataset, val_dataset]), \
             patch('auto_voice.training.trainer.DataLoader', side_effect=dataloader_spy), \
             patch.object(trainer, '_train_epoch', return_value=0.5), \
             patch.object(trainer, 'assess', return_value=0.1) as mock_assess, \
             patch.object(trainer, 'save_checkpoint') as mock_save_checkpoint, \
             patch.object(trainer.scheduler, 'step') as mock_scheduler_step:
            trainer.train('train_dir', val_dir=str(val_dir))

        assert loaders[0]['kwargs']['batch_size'] == 4
        assert loaders[0]['kwargs']['drop_last'] is True
        assert loaders[1]['kwargs']['batch_size'] == 4
        assert loaders[1]['kwargs']['shuffle'] is False
        mock_assess.assert_called_once_with('loader-2')
        mock_scheduler_step.assert_called_once()
        assert trainer.best_loss == 0.1
        assert [call.args[0] for call in mock_save_checkpoint.call_args_list] == [
            trainer.checkpoint_dir / 'best.pth',
            trainer.checkpoint_dir / 'checkpoint_epoch_1.pth',
            trainer.checkpoint_dir / 'final.pth',
        ]


class TestTrainEpochAndAssess:
    """Cover batch-level training and evaluation behavior."""

    def test_train_epoch_requires_speaker_embedding(self, trainer_factory):
        trainer = trainer_factory()

        with pytest.raises(RuntimeError, match='Speaker embedding not set'):
            trainer._train_epoch([make_batch()], epoch=0)

    @pytest.mark.parametrize('bad_loss', [float('nan'), float('inf')])
    def test_train_epoch_skips_non_finite_losses(self, trainer_factory, bad_loss):
        trainer = trainer_factory()
        trainer.speaker_embedding = torch.ones(4)
        trainer.model.compute_loss = MagicMock(return_value={
            'total_loss': torch.tensor(bad_loss),
            'reconstruction_loss': torch.tensor(0.0),
        })

        with patch.object(trainer, '_compute_spec', return_value=torch.ones(2, 513, 6)), \
             patch.object(trainer.optimizer, 'step') as mock_step:
            loss = trainer._train_epoch([make_batch()], epoch=0)

        assert loss == 0.0
        assert trainer.global_step == 0
        mock_step.assert_not_called()

    def test_train_epoch_clamps_large_loss_and_clips_gradients(self, trainer_factory):
        trainer = trainer_factory({'log_every': 1, 'gradient_clip': 0.5})
        trainer.speaker_embedding = torch.ones(4)
        trainer.model.compute_loss = MagicMock(return_value={
            'total_loss': torch.tensor(2_000_000.0, requires_grad=True),
            'reconstruction_loss': torch.tensor(5.0),
        })

        with patch.object(trainer, '_compute_spec', return_value=torch.ones(2, 513, 6)), \
             patch('auto_voice.training.trainer.nn.utils.clip_grad_norm_') as mock_clip, \
             patch.object(trainer.optimizer, 'step') as mock_step:
            loss = trainer._train_epoch([make_batch()], epoch=0)

        assert loss == pytest.approx(1_000_000.0)
        assert trainer.global_step == 1
        mock_clip.assert_called_once()
        mock_step.assert_called_once()

    def test_assess_requires_speaker_embedding(self, trainer_factory):
        trainer = trainer_factory()

        with pytest.raises(RuntimeError, match='Speaker embedding not set'):
            trainer.assess([make_batch()])

    def test_assess_returns_average_loss(self, trainer_factory):
        trainer = trainer_factory()
        trainer.speaker_embedding = torch.ones(4)
        trainer.model.compute_loss = MagicMock(side_effect=[
            {
                'total_loss': torch.tensor(2.0),
                'reconstruction_loss': torch.tensor(2.0),
            },
            {
                'total_loss': torch.tensor(4.0),
                'reconstruction_loss': torch.tensor(4.0),
            },
        ])

        with patch.object(trainer, '_compute_spec', return_value=torch.ones(2, 513, 6)):
            loss = trainer.assess([make_batch(), make_batch()])

        assert loss == pytest.approx(3.0)


class TestCheckpointBranches:
    """Cover LoRA and load checkpoint edge cases."""

    def test_save_checkpoint_writes_lora_only_state(self, trainer_factory, tmp_path):
        model = TrainingTestModel()
        model._lora_injected = True
        model.get_lora_state_dict = MagicMock(return_value={'adapter': torch.tensor([1.0])})
        trainer = trainer_factory(model=model)
        checkpoint_path = tmp_path / 'adapter.pth'

        trainer.save_checkpoint(checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        assert checkpoint['is_lora'] is True
        assert 'lora_state' in checkpoint
        assert 'model' not in checkpoint

    def test_load_checkpoint_warns_when_missing(self, trainer_factory, tmp_path):
        trainer = trainer_factory()

        with patch('auto_voice.training.trainer.logger.warning') as mock_warning:
            trainer.load_checkpoint(str(tmp_path / 'missing.pth'))

        mock_warning.assert_called_once()

    def test_load_checkpoint_rejects_lora_for_non_lora_model(self, trainer_factory, tmp_path):
        trainer = trainer_factory()
        checkpoint_path = tmp_path / 'lora_only.pth'
        torch.save({'is_lora': True, 'lora_state': {'adapter': torch.tensor([1.0])}}, checkpoint_path)

        with pytest.raises(RuntimeError, match="doesn't have LoRA injected"):
            trainer.load_checkpoint(str(checkpoint_path))

    def test_load_checkpoint_restores_lora_metadata(self, trainer_factory, tmp_path):
        model = TrainingTestModel()
        model._lora_injected = True
        model.load_lora_state_dict = MagicMock()
        trainer = trainer_factory(model=model)
        checkpoint_path = tmp_path / 'resume_lora.pth'
        torch.save({
            'is_lora': True,
            'lora_state': {'adapter': torch.tensor([1.0])},
            'global_step': 12,
            'current_epoch': 3,
            'best_loss': 0.25,
        }, checkpoint_path)

        trainer.load_checkpoint(str(checkpoint_path))

        trainer.model.load_lora_state_dict.assert_called_once_with({'adapter': torch.tensor([1.0])})
        assert trainer.global_step == 12
        assert trainer.current_epoch == 3
        assert trainer.best_loss == 0.25

    def test_load_checkpoint_restores_full_state_without_optimizer_entries(self, trainer_factory, tmp_path):
        trainer = trainer_factory()
        checkpoint_path = tmp_path / 'resume_full.pth'
        torch.save({
            'model': trainer.model.state_dict(),
            'global_step': 21,
            'current_epoch': 5,
            'best_loss': 0.5,
            'is_lora': False,
        }, checkpoint_path)

        with patch.object(trainer.optimizer, 'load_state_dict') as mock_optimizer_load, \
             patch.object(trainer.scheduler, 'load_state_dict') as mock_scheduler_load:
            trainer.load_checkpoint(str(checkpoint_path))

        mock_optimizer_load.assert_not_called()
        mock_scheduler_load.assert_not_called()
        assert trainer.global_step == 21
        assert trainer.current_epoch == 5
        assert trainer.best_loss == 0.5


class TestSpeakerEmbeddingSetup:
    """Cover embedding discovery and assignment."""

    def test_set_speaker_embedding_requires_audio_files(self, trainer_factory, tmp_path):
        trainer = trainer_factory()

        with pytest.raises(RuntimeError, match='No audio files found'):
            trainer.set_speaker_embedding(str(tmp_path / 'missing_audio'))

    def test_set_speaker_embedding_uses_supported_extensions(self, trainer_factory, tmp_path):
        trainer = trainer_factory()
        audio_dir = tmp_path / 'speaker_audio'
        audio_dir.mkdir()
        wav_file = audio_dir / 'sample.wav'
        flac_file = audio_dir / 'sample.flac'
        mp3_file = audio_dir / 'sample.mp3'
        ignored_file = audio_dir / 'notes.txt'
        for path in [wav_file, flac_file, mp3_file, ignored_file]:
            path.write_bytes(b'test')

        cloner_instance = MagicMock()
        cloner_instance.create_speaker_embedding.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        fake_module = types.SimpleNamespace(VoiceCloner=MagicMock(return_value=cloner_instance))

        with patch.dict(sys.modules, {'auto_voice.inference.voice_cloner': fake_module}):
            trainer.set_speaker_embedding(str(audio_dir))

        cloner_instance.create_speaker_embedding.assert_called_once_with([
            str(flac_file),
            str(mp3_file),
            str(wav_file),
        ])
        assert trainer.speaker_embedding.dtype == torch.float32
        assert trainer.speaker_embedding.device.type == 'cpu'
