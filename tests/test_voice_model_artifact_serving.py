"""Artifact-aware voice model loading: checkpoints load into the class that produced them."""
import numpy as np
import pytest
import torch

from auto_voice.inference.model_manager import ModelManager
from auto_voice.models.svc_decoder import CoMoSVCDecoder
from auto_voice.models.so_vits_svc import SoVitsSvc


DIMS = dict(content_dim=8, pitch_dim=8, speaker_dim=4, n_mels=5, hidden_dim=16, n_layers=2)


@pytest.fixture
def manager():
    return ModelManager(device=torch.device('cpu'), config={'sample_rate': 22050})


def _small_decoder(inject_lora: bool = False) -> CoMoSVCDecoder:
    torch.manual_seed(0)
    model = CoMoSVCDecoder(device=torch.device('cpu'), **DIMS)
    if inject_lora:
        model.inject_lora(rank=2, alpha=4)
    return model


class TestBuildVoiceModel:
    def test_comosvc_full_state_loads_into_comosvc(self, manager, tmp_path):
        source = _small_decoder()
        path = tmp_path / 'full_model.pt'
        torch.save(source.state_dict(), path)

        manager.load_voice_model(str(path), 'spk')
        model = manager._sovits_models['spk']

        assert isinstance(model, CoMoSVCDecoder)
        assert model.n_mels == DIMS['n_mels']
        assert model.hidden_dim == DIMS['hidden_dim']
        # weights actually match — not a random re-init
        torch.testing.assert_close(
            model.state_dict()['input_proj.weight'],
            source.state_dict()['input_proj.weight'],
        )

    def test_comosvc_infer_contract_after_load(self, manager, tmp_path):
        path = tmp_path / 'full_model.pt'
        torch.save(_small_decoder().state_dict(), path)
        manager.load_voice_model(str(path), 'spk')
        model = manager._sovits_models['spk']

        content = torch.randn(1, 12, DIMS['content_dim'])
        pitch = torch.randn(1, 12, DIMS['pitch_dim'])
        speaker = torch.randn(1, DIMS['speaker_dim'])
        with torch.no_grad():
            mel = model.infer(content, pitch, speaker)
        assert mel.shape == (1, DIMS['n_mels'], 12)
        assert not torch.isnan(mel).any()

    def test_self_contained_adapter_artifact_loads_with_lora(self, manager, tmp_path):
        source = _small_decoder(inject_lora=True)
        payload = {
            'model_state_dict': source.state_dict(),
            'lora_config': {'rank': 2, 'alpha': 4},
        }
        path = tmp_path / 'adapter_model.pt'
        torch.save(payload, path)

        manager.load_voice_model(str(path), 'spk')
        model = manager._sovits_models['spk']

        assert isinstance(model, CoMoSVCDecoder)
        loaded_keys = set(model.state_dict().keys())
        assert any('.adapter.lora_' in key for key in loaded_keys)
        torch.testing.assert_close(
            model.state_dict()['input_proj.adapter.lora_A'],
            source.state_dict()['input_proj.adapter.lora_A'],
        )

    def test_deltas_only_adapter_is_rejected(self, manager, tmp_path):
        source = _small_decoder(inject_lora=True)
        deltas = {k: v for k, v in source.state_dict().items() if '.adapter.lora_' in k}
        path = tmp_path / 'adapter.pt'
        torch.save(deltas, path)

        with pytest.raises(RuntimeError, match='only LoRA deltas'):
            manager.load_voice_model(str(path), 'spk')

    def test_sovits_state_loads_into_sovits(self, manager, tmp_path):
        torch.manual_seed(0)
        source = SoVitsSvc()
        path = tmp_path / 'sovits.pt'
        torch.save(source.state_dict(), path)

        manager.load_voice_model(str(path), 'spk')
        assert isinstance(manager._sovits_models['spk'], SoVitsSvc)

    def test_unknown_architecture_is_rejected(self, manager, tmp_path):
        path = tmp_path / 'mystery.pt'
        torch.save({'some.weight': torch.ones(3)}, path)

        with pytest.raises(RuntimeError, match='known voice model architecture'):
            manager.load_voice_model(str(path), 'spk')

    def test_missing_path_raises_file_not_found(self, manager, tmp_path):
        with pytest.raises(FileNotFoundError):
            manager.load_voice_model(str(tmp_path / 'nope.pt'), 'spk')

    def test_speaker_embedding_is_stored(self, manager, tmp_path):
        path = tmp_path / 'full_model.pt'
        torch.save(_small_decoder().state_dict(), path)
        emb = np.ones(4, dtype=np.float32)
        manager.load_voice_model(str(path), 'spk', speaker_embedding=emb)
        assert 'spk' in manager._speaker_embeddings


@pytest.mark.cuda
@pytest.mark.slow
class TestRealtimeDurationContract:
    def test_process_chunk_preserves_duration(self):
        if not torch.cuda.is_available():
            pytest.skip('CUDA required')
        from auto_voice.inference.realtime_pipeline import RealtimePipeline

        pipe = RealtimePipeline(device='cuda')
        emb = np.random.randn(256).astype(np.float32)
        emb /= np.linalg.norm(emb)
        pipe.set_speaker_embedding(emb)

        for seconds in (1.0, 1.7):
            n = int(16000 * seconds)
            t = np.linspace(0, seconds, n, endpoint=False)
            chunk = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
            out = pipe.process_chunk(chunk)
            expected = int(round(n * 22050 / 16000))
            assert len(out) == expected, (
                f'{seconds}s chunk: got {len(out)} samples, expected {expected}'
            )


@pytest.mark.cuda
@pytest.mark.slow
class TestRealtimeTrainedServing:
    def _pipeline(self, **kwargs):
        if not torch.cuda.is_available():
            pytest.skip('CUDA required')
        from auto_voice.inference.realtime_pipeline import RealtimePipeline
        return RealtimePipeline(device='cuda', **kwargs)

    def test_default_vocoder_checkpoint_loads(self, caplog):
        import logging
        with caplog.at_level(logging.INFO):
            self._pipeline()
        assert 'HiFiGAN vocoder loaded from' in caplog.text

    def test_bad_vocoder_checkpoint_raises(self, tmp_path):
        bogus = tmp_path / 'not-a-checkpoint.pt'
        bogus.write_bytes(b'garbage')
        with pytest.raises(RuntimeError):
            self._pipeline(vocoder_checkpoint=str(bogus))

    def test_trained_voice_model_drives_conversion(self, tmp_path):
        pipe = self._pipeline()
        emb = np.random.randn(256).astype(np.float32)
        emb /= np.linalg.norm(emb)
        pipe.set_speaker_embedding(emb)

        # trained artifact must match runtime feature dims (768/768/256);
        # hidden/layers stay tiny for speed, n_mels matches the vocoder
        from auto_voice.models.svc_decoder import CoMoSVCDecoder
        torch.manual_seed(0)
        decoder = CoMoSVCDecoder(
            content_dim=768, pitch_dim=768, speaker_dim=256,
            n_mels=80, hidden_dim=32, n_layers=2, device=torch.device('cpu'),
        )
        artifact = tmp_path / 'profile_full_model.pt'
        torch.save(decoder.state_dict(), artifact)

        n = 16000
        chunk = (0.3 * np.sin(2 * np.pi * 220 * np.linspace(0, 1, n, endpoint=False))).astype(np.float32)
        baseline = pipe.process_chunk(chunk)

        pipe.load_voice_model(str(artifact))
        assert pipe._voice_model is not None
        converted = pipe.process_chunk(chunk)

        expected = int(round(n * 22050 / 16000))
        assert len(converted) == expected
        assert not np.allclose(converted, baseline), \
            'trained decoder output should differ from SimpleDecoder output'

        pipe.clear_voice_model()
        assert pipe._voice_model is None


class TestActiveModelTypePreference:
    """convert_song must honor the profile's active_model_type when both artifacts exist."""

    def _pipeline_with_artifacts(self, tmp_path):
        from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

        trained_dir = tmp_path / 'trained_models'
        trained_dir.mkdir()
        for name in ('p1_full_model.pt', 'p1_adapter_model.pt'):
            torch.save(_small_decoder().state_dict(), trained_dir / name)

        pipeline = SingingConversionPipeline(device='cpu', config={})
        store = type('S', (), {'trained_models_dir': str(trained_dir)})()
        pipeline._voice_cloner = type('C', (), {'store': store})()
        pipeline._model_manager = ModelManager(device=torch.device('cpu'), config={})
        return pipeline

    def test_adapter_preference_wins_when_active(self, tmp_path):
        pipeline = self._pipeline_with_artifacts(tmp_path)
        _, model_type = pipeline._resolve_target_speaker(
            'p1', np.ones(4, dtype=np.float32), active_model_type='adapter'
        )
        assert model_type == 'adapter'

    def test_full_model_preferred_by_default(self, tmp_path):
        pipeline = self._pipeline_with_artifacts(tmp_path)
        _, model_type = pipeline._resolve_target_speaker(
            'p1', np.ones(4, dtype=np.float32), active_model_type='full_model'
        )
        assert model_type == 'full_model'
