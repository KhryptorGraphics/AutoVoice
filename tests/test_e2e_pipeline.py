"""End-to-end integration tests for the full train→infer cycle."""
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from auto_voice.inference.model_manager import ModelManager
from auto_voice.inference.voice_cloner import VoiceCloner
from auto_voice.models.so_vits_svc import SoVitsSvc
from auto_voice.training.trainer import Trainer


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def training_audio_dir(tmp_path):
    """Directory with synthetic singing recordings for training."""
    train_dir = tmp_path / "training_data"
    train_dir.mkdir()
    sr = 22050
    freqs = [220, 330, 440, 550, 660]
    for i, freq in enumerate(freqs):
        t = np.linspace(0, 4, sr * 4, endpoint=False)
        # Add harmonics to simulate voice-like spectrum
        audio = (
            0.5 * np.sin(2 * np.pi * freq * t) +
            0.25 * np.sin(2 * np.pi * freq * 2 * t) +
            0.125 * np.sin(2 * np.pi * freq * 3 * t)
        ).astype(np.float32)
        sf.write(str(train_dir / f"training_{i}.wav"), audio, sr)
    return str(train_dir)


@pytest.fixture
def inference_audio(tmp_path):
    """Source audio for inference (different from training)."""
    sr = 22050
    t = np.linspace(0, 2, sr * 2, endpoint=False)
    # Different frequency pattern than training
    audio = (
        0.4 * np.sin(2 * np.pi * 300 * t) +
        0.3 * np.sin(2 * np.pi * 600 * t)
    ).astype(np.float32)
    path = str(tmp_path / "source_song.wav")
    sf.write(path, audio, sr)
    return path, audio, sr


@pytest.mark.integration
@pytest.mark.slow
class TestTrainThenInfer:
    """Full train→infer cycle."""

    def test_train_then_infer_e2e(self, device, training_audio_dir, inference_audio, tmp_path):
        """Complete pipeline: train model, save, load, infer."""
        source_path, source_audio, sr = inference_audio
        ckpt_dir = str(tmp_path / 'checkpoints')

        # 1. Create speaker embedding from training audio
        cloner = VoiceCloner(device=device)
        audio_files = sorted(str(p) for p in Path(training_audio_dir).glob('*.wav'))
        speaker_embedding = cloner.create_speaker_embedding(audio_files)
        assert speaker_embedding.shape == (256,)
        assert abs(np.linalg.norm(speaker_embedding) - 1.0) < 1e-5

        # 2. Train SoVitsSvc for a few epochs
        model = SoVitsSvc().to(device)
        trainer = Trainer(model, config={
            'epochs': 3,
            'batch_size': 2,
            'checkpoint_dir': ckpt_dir,
            'sample_rate': sr,
            'log_every': 1,
            'save_every': 10,
        }, device=device)
        trainer.set_speaker_embedding(training_audio_dir)
        trainer.train(training_audio_dir)

        # Verify training produced losses
        assert len(trainer.train_losses) == 3
        for loss in trainer.train_losses:
            assert np.isfinite(loss)

        # 3. Save checkpoint
        ckpt_path = os.path.join(ckpt_dir, 'trained.pth')
        trainer.save_checkpoint(ckpt_path)
        assert os.path.exists(ckpt_path)

        # 4. Load into ModelManager and run inference
        mm = ModelManager(device=device, config={'sample_rate': sr})
        mm.load()
        mm.load_voice_model(ckpt_path, 'target_speaker')

        output = mm.infer(source_audio, 'target_speaker', speaker_embedding, sr=sr)

        # 5. Verify output properties
        assert len(output) == len(source_audio), "Output length must match input"
        assert output.dtype == np.float32
        assert not np.any(np.isnan(output)), "No NaN in output"
        assert not np.any(np.isinf(output)), "No Inf in output"
        assert np.abs(output).max() <= 0.96, "Output should be normalized"
        # Model should transform the audio (not passthrough)
        assert not np.allclose(output, source_audio, atol=0.05)

    def test_speaker_embedding_consistency(self, device, training_audio_dir):
        """Speaker embedding is deterministic across calls."""
        cloner = VoiceCloner(device=device)
        audio_files = sorted(str(p) for p in Path(training_audio_dir).glob('*.wav'))

        emb1 = cloner.create_speaker_embedding(audio_files)
        emb2 = cloner.create_speaker_embedding(audio_files)
        np.testing.assert_array_equal(emb1, emb2)

    def test_different_speakers_different_output(self, device, training_audio_dir, tmp_path):
        """Different speaker embeddings produce different conversion output."""
        sr = 22050
        t = np.linspace(0, 2, sr * 2, endpoint=False)
        source = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Two different "speakers"
        emb1 = np.zeros(256, dtype=np.float32)
        emb1[:128] = 1.0
        emb1 /= np.linalg.norm(emb1)

        emb2 = np.zeros(256, dtype=np.float32)
        emb2[128:] = 1.0
        emb2 /= np.linalg.norm(emb2)

        mm = ModelManager(device=device, config={'sample_rate': sr})
        mm.load()
        model = SoVitsSvc()
        model.to(device)
        mm._sovits_models['speaker1'] = model
        mm._sovits_models['speaker2'] = model  # Same model, different embedding

        out1 = mm.infer(source, 'speaker1', emb1, sr=sr)
        out2 = mm.infer(source, 'speaker2', emb2, sr=sr)

        # Different embeddings should produce different outputs
        assert not np.allclose(out1, out2, atol=1e-3)


@pytest.mark.integration
class TestSingingPipelineWithModel:
    """SingingConversionPipeline with model-based conversion."""

    def test_pipeline_converts_audio(self, singing_pipeline, voice_cloner, sample_audio_file):
        """Pipeline produces converted audio that differs from input."""
        profile = voice_cloner.create_voice_profile(audio=sample_audio_file)
        result = singing_pipeline.convert_song(
            song_path=sample_audio_file,
            target_profile_id=profile['profile_id'],
            preset='draft'
        )

        assert 'mixed_audio' in result
        assert isinstance(result['mixed_audio'], np.ndarray)
        assert result['mixed_audio'].size > 0
        assert result['duration'] > 0
        assert not np.any(np.isnan(result['mixed_audio']))

    def test_pipeline_correct_sample_rate(self, singing_pipeline, voice_cloner, sample_audio_file):
        profile = voice_cloner.create_voice_profile(audio=sample_audio_file)
        result = singing_pipeline.convert_song(
            song_path=sample_audio_file,
            target_profile_id=profile['profile_id'],
            preset='draft'
        )
        assert result['sample_rate'] == 22050


@pytest.mark.integration
class TestE2EBackendIntegration:
    """E2E tests verifying pipeline with non-default backends."""

    def test_convert_with_bigvgan_vocoder(self, device):
        """Full conversion with BigVGAN vocoder via config."""
        from auto_voice.models.so_vits_svc import SoVitsSvc

        mm = ModelManager(device=device, config={'sample_rate': 22050})
        mm.load(vocoder_type='bigvgan')

        # BigVGAN uses 100-band mel; SoVitsSvc must output matching mels
        model = SoVitsSvc(config={'n_mels': 100})
        model.to(device)
        mm._sovits_models['default'] = model

        # Synthesize test audio
        sr = 22050
        t = np.linspace(0, 2, sr * 2, endpoint=False)
        audio = (0.4 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)
        embedding /= np.linalg.norm(embedding)

        output = mm.infer(audio, 'default', embedding, sr=sr)

        assert output.dtype == np.float32
        assert len(output) == len(audio)
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))
        assert np.abs(output).max() <= 0.96

    def test_convert_with_contentvec_encoder(self, device):
        """Full conversion with ContentVec encoder backend."""
        from auto_voice.models.so_vits_svc import SoVitsSvc

        mm = ModelManager(device=device, config={'sample_rate': 22050})
        mm.load(encoder_backend='contentvec')

        model = SoVitsSvc()
        model.to(device)
        mm._sovits_models['default'] = model

        sr = 22050
        t = np.linspace(0, 2, sr * 2, endpoint=False)
        audio = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)
        embedding /= np.linalg.norm(embedding)

        output = mm.infer(audio, 'default', embedding, sr=sr)

        assert output.dtype == np.float32
        assert len(output) == len(audio)
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))
        assert np.abs(output).max() <= 0.96

    def test_convert_with_conformer_encoder(self, device):
        """Full conversion with Conformer encoder type."""
        from auto_voice.models.so_vits_svc import SoVitsSvc

        mm = ModelManager(device=device, config={'sample_rate': 22050})
        mm.load(
            encoder_type='conformer',
            conformer_config={
                'n_layers': 2,
                'n_heads': 2,
                'hidden_dim': 256,
                'kernel_size': 3,
            },
        )

        model = SoVitsSvc()
        model.to(device)
        mm._sovits_models['default'] = model

        sr = 22050
        t = np.linspace(0, 2, sr * 2, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)
        embedding /= np.linalg.norm(embedding)

        output = mm.infer(audio, 'default', embedding, sr=sr)

        assert output.dtype == np.float32
        assert len(output) == len(audio)
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))
        assert np.abs(output).max() <= 0.96
