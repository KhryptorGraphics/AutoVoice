"""Tests for ModelManager inference orchestrator."""
import numpy as np
import pytest
import torch
from unittest.mock import patch, MagicMock

from auto_voice.inference.model_manager import ModelManager
from auto_voice.models.so_vits_svc import SoVitsSvc
from auto_voice.models.encoder import ContentEncoder


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def model_manager(device):
    """ModelManager with shared models loaded (random weights)."""
    mm = ModelManager(device=device)
    mm.load()
    return mm


@pytest.fixture
def model_manager_with_voice(model_manager, device):
    """ModelManager with a SoVitsSvc model loaded."""
    model = SoVitsSvc()
    model.to(device)
    model_manager._sovits_models['default'] = model
    return model_manager


@pytest.fixture
def test_audio():
    """2-second sine wave at 440Hz, sr=22050."""
    sr = 22050
    t = np.linspace(0, 2, sr * 2, endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio


@pytest.fixture
def speaker_embedding():
    """Normalized random speaker embedding."""
    emb = np.random.randn(256).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return emb


class TestModelManagerInit:
    """Initialization tests."""

    @pytest.mark.smoke
    def test_init(self, device):
        mm = ModelManager(device=device)
        assert mm.device == device
        assert mm._content_encoder is None
        assert mm._pitch_encoder is None
        assert mm._vocoder is None

    def test_init_with_config(self, device):
        mm = ModelManager(device=device, config={'sample_rate': 16000})
        assert mm.sample_rate == 16000

    def test_load_initializes_models(self, model_manager):
        assert model_manager._content_encoder is not None
        assert model_manager._pitch_encoder is not None
        assert model_manager._vocoder is not None


class TestModelManagerErrors:
    """Error handling tests - no fallback behavior."""

    def test_infer_without_load_raises(self, device, test_audio, speaker_embedding):
        mm = ModelManager(device=device)
        with pytest.raises(RuntimeError, match="ContentEncoder not loaded"):
            mm.infer(test_audio, 'default', speaker_embedding)

    def test_infer_without_voice_model_raises(self, model_manager, test_audio, speaker_embedding):
        with pytest.raises(RuntimeError, match="No trained model"):
            model_manager.infer(test_audio, 'default', speaker_embedding)

    def test_infer_unknown_speaker_raises(self, model_manager_with_voice, test_audio, speaker_embedding):
        with pytest.raises(RuntimeError, match="No trained model for speaker 'unknown'"):
            model_manager_with_voice.infer(test_audio, 'unknown', speaker_embedding)


class TestModelManagerInference:
    """Inference tests with loaded models."""

    def test_infer_output_length_matches_input(self, model_manager_with_voice, test_audio, speaker_embedding):
        output = model_manager_with_voice.infer(test_audio, 'default', speaker_embedding)
        assert len(output) == len(test_audio)

    def test_infer_output_is_float32(self, model_manager_with_voice, test_audio, speaker_embedding):
        output = model_manager_with_voice.infer(test_audio, 'default', speaker_embedding)
        assert output.dtype == np.float32

    def test_infer_output_no_nan(self, model_manager_with_voice, test_audio, speaker_embedding):
        output = model_manager_with_voice.infer(test_audio, 'default', speaker_embedding)
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

    def test_output_normalized(self, model_manager_with_voice, test_audio, speaker_embedding):
        output = model_manager_with_voice.infer(test_audio, 'default', speaker_embedding)
        assert np.abs(output).max() <= 0.96  # Should be normalized to <= 0.95

    def test_infer_output_differs_from_input(self, model_manager_with_voice, test_audio, speaker_embedding):
        output = model_manager_with_voice.infer(test_audio, 'default', speaker_embedding)
        # With random weights, output should differ from input
        assert not np.allclose(output, test_audio, atol=0.1)

    def test_different_embeddings_different_output(self, model_manager_with_voice, test_audio):
        emb1 = np.zeros(256, dtype=np.float32)
        emb1[0] = 1.0
        emb2 = np.zeros(256, dtype=np.float32)
        emb2[1] = 1.0

        out1 = model_manager_with_voice.infer(test_audio, 'default', emb1)
        out2 = model_manager_with_voice.infer(test_audio, 'default', emb2)
        # Different embeddings should produce non-identical outputs
        # (with random weights the difference may be small but non-zero)
        assert not np.array_equal(out1, out2)

    def test_frame_alignment(self, model_manager_with_voice, speaker_embedding):
        """Content and pitch get aligned to same frame count internally."""
        # Shorter audio to make the test faster
        sr = 22050
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float32)
        output = model_manager_with_voice.infer(audio, 'default', speaker_embedding, sr=sr)
        assert len(output) == len(audio)


class TestModelManagerBackendConfig:
    """Tests for encoder_backend, encoder_type, and vocoder_type passthrough."""

    def test_load_default_hubert_linear(self, device):
        """Default load() uses hubert backend with linear projection."""
        mm = ModelManager(device=device)
        mm.load()
        assert mm._content_encoder is not None
        assert mm._content_encoder.encoder_backend == 'hubert'
        assert mm._content_encoder.encoder_type == 'linear'

    def test_load_hubert_conformer(self, device):
        """load() with encoder_type='conformer' creates conformer projection."""
        mm = ModelManager(device=device)
        mm.load(encoder_backend='hubert', encoder_type='conformer')
        assert mm._content_encoder.encoder_backend == 'hubert'
        assert mm._content_encoder.encoder_type == 'conformer'
        from auto_voice.models.conformer import ConformerEncoder
        assert isinstance(mm._content_encoder.projection, ConformerEncoder)

    def test_load_contentvec_linear(self, device):
        """load() with encoder_backend='contentvec' creates ContentVec encoder."""
        mm = ModelManager(device=device)
        mm.load(encoder_backend='contentvec', encoder_type='linear')
        assert mm._content_encoder.encoder_backend == 'contentvec'
        assert mm._content_encoder.encoder_type == 'linear'
        assert mm._content_encoder._contentvec is not None

    def test_load_contentvec_conformer(self, device):
        """load() with contentvec+conformer uses both."""
        mm = ModelManager(device=device)
        mm.load(encoder_backend='contentvec', encoder_type='conformer')
        assert mm._content_encoder.encoder_backend == 'contentvec'
        assert mm._content_encoder.encoder_type == 'conformer'
        assert mm._content_encoder._contentvec is not None
        from auto_voice.models.conformer import ConformerEncoder
        assert isinstance(mm._content_encoder.projection, ConformerEncoder)

    def test_load_conformer_config_passthrough(self, device):
        """conformer_config dict is passed to ConformerEncoder."""
        cfg = {'n_layers': 2, 'hidden_dim': 128}
        mm = ModelManager(device=device)
        mm.load(encoder_type='conformer', conformer_config=cfg)
        proj = mm._content_encoder.projection
        assert proj.n_layers == 2
        assert proj.hidden_dim == 128
        assert len(proj.layers) == 2

    def test_load_vocoder_hifigan(self, device):
        """vocoder_type='hifigan' creates HiFiGANVocoder."""
        mm = ModelManager(device=device)
        mm.load(vocoder_type='hifigan')
        from auto_voice.models.vocoder import HiFiGANVocoder
        assert isinstance(mm._vocoder, HiFiGANVocoder)

    def test_load_vocoder_bigvgan(self, device):
        """vocoder_type='bigvgan' creates BigVGANVocoder."""
        mm = ModelManager(device=device)
        mm.load(vocoder_type='bigvgan')
        from auto_voice.models.vocoder import BigVGANVocoder
        assert isinstance(mm._vocoder, BigVGANVocoder)

    def test_load_unknown_vocoder_raises(self, device):
        """Unknown vocoder_type raises RuntimeError."""
        mm = ModelManager(device=device)
        with pytest.raises(RuntimeError, match="Unknown vocoder_type"):
            mm.load(vocoder_type='unknown_vocoder')

    def test_hubert_path_not_loaded_for_contentvec(self, device):
        """hubert_path is ignored when encoder_backend='contentvec'."""
        mm = ModelManager(device=device)
        mm.load(encoder_backend='contentvec', hubert_path='/fake/path.pt')
        # ContentVec backend - _hubert should not be loaded
        assert mm._content_encoder._hubert is None

    def test_encoder_extract_features_hubert_linear(self, device):
        """HuBERT+linear encoder produces correct output shape."""
        mm = ModelManager(device=device)
        mm.load(encoder_backend='hubert', encoder_type='linear')
        audio = torch.randn(1, 16000).to(device)
        with torch.no_grad():
            features = mm._content_encoder.extract_features(audio, sr=16000)
        assert features.dim() == 3
        assert features.shape[0] == 1
        assert features.shape[2] == 768

    def test_encoder_extract_features_hubert_conformer(self, device):
        """HuBERT+conformer encoder produces correct output shape."""
        mm = ModelManager(device=device)
        mm.load(encoder_backend='hubert', encoder_type='conformer',
                conformer_config={'n_layers': 2})
        audio = torch.randn(1, 16000).to(device)
        with torch.no_grad():
            features = mm._content_encoder.extract_features(audio, sr=16000)
        assert features.dim() == 3
        assert features.shape[0] == 1
        assert features.shape[2] == 768


class TestPipelineConfigPassthrough:
    """Tests for SingingConversionPipeline and RealtimeVoiceConversionPipeline
    passing config keys through to ModelManager.load()."""

    def test_singing_pipeline_passes_vocoder_type(self, device):
        """SingingConversionPipeline passes vocoder_type from config."""
        from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

        config = {'vocoder_type': 'bigvgan'}
        pipeline = SingingConversionPipeline(device=device, config=config)

        with patch('auto_voice.inference.model_manager.ModelManager.load') as mock_load:
            pipeline._get_model_manager()
            mock_load.assert_called_once()
            call_kwargs = mock_load.call_args[1]
            assert call_kwargs['vocoder_type'] == 'bigvgan'

    def test_singing_pipeline_passes_encoder_backend(self, device):
        """SingingConversionPipeline passes encoder_backend from config."""
        from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

        config = {'encoder_backend': 'contentvec'}
        pipeline = SingingConversionPipeline(device=device, config=config)

        with patch('auto_voice.inference.model_manager.ModelManager.load') as mock_load:
            pipeline._get_model_manager()
            call_kwargs = mock_load.call_args[1]
            assert call_kwargs['encoder_backend'] == 'contentvec'

    def test_singing_pipeline_passes_encoder_type(self, device):
        """SingingConversionPipeline passes encoder_type from config."""
        from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

        config = {'encoder_type': 'conformer'}
        pipeline = SingingConversionPipeline(device=device, config=config)

        with patch('auto_voice.inference.model_manager.ModelManager.load') as mock_load:
            pipeline._get_model_manager()
            call_kwargs = mock_load.call_args[1]
            assert call_kwargs['encoder_type'] == 'conformer'

    def test_singing_pipeline_defaults(self, device):
        """SingingConversionPipeline uses correct defaults when config empty."""
        from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

        pipeline = SingingConversionPipeline(device=device, config={})

        with patch('auto_voice.inference.model_manager.ModelManager.load') as mock_load:
            pipeline._get_model_manager()
            call_kwargs = mock_load.call_args[1]
            assert call_kwargs['vocoder_type'] == 'hifigan'
            assert call_kwargs['encoder_backend'] == 'hubert'
            assert call_kwargs['encoder_type'] == 'linear'

    def test_realtime_pipeline_passes_vocoder_type(self, device):
        """RealtimeVoiceConversionPipeline passes vocoder_type from config."""
        from auto_voice.inference.realtime_voice_conversion_pipeline import RealtimeVoiceConversionPipeline

        config = {'vocoder_type': 'bigvgan'}
        pipeline = RealtimeVoiceConversionPipeline(device=device, config=config)

        with patch('auto_voice.inference.model_manager.ModelManager.load') as mock_load:
            pipeline._get_model_manager()
            call_kwargs = mock_load.call_args[1]
            assert call_kwargs['vocoder_type'] == 'bigvgan'

    def test_realtime_pipeline_passes_encoder_backend(self, device):
        """RealtimeVoiceConversionPipeline passes encoder_backend from config."""
        from auto_voice.inference.realtime_voice_conversion_pipeline import RealtimeVoiceConversionPipeline

        config = {'encoder_backend': 'contentvec'}
        pipeline = RealtimeVoiceConversionPipeline(device=device, config=config)

        with patch('auto_voice.inference.model_manager.ModelManager.load') as mock_load:
            pipeline._get_model_manager()
            call_kwargs = mock_load.call_args[1]
            assert call_kwargs['encoder_backend'] == 'contentvec'

    def test_realtime_pipeline_passes_encoder_type(self, device):
        """RealtimeVoiceConversionPipeline passes encoder_type from config."""
        from auto_voice.inference.realtime_voice_conversion_pipeline import RealtimeVoiceConversionPipeline

        config = {'encoder_type': 'conformer'}
        pipeline = RealtimeVoiceConversionPipeline(device=device, config=config)

        with patch('auto_voice.inference.model_manager.ModelManager.load') as mock_load:
            pipeline._get_model_manager()
            call_kwargs = mock_load.call_args[1]
            assert call_kwargs['encoder_type'] == 'conformer'

    def test_realtime_pipeline_defaults(self, device):
        """RealtimeVoiceConversionPipeline uses correct defaults when config empty."""
        from auto_voice.inference.realtime_voice_conversion_pipeline import RealtimeVoiceConversionPipeline

        pipeline = RealtimeVoiceConversionPipeline(device=device, config={})

        with patch('auto_voice.inference.model_manager.ModelManager.load') as mock_load:
            pipeline._get_model_manager()
            call_kwargs = mock_load.call_args[1]
            assert call_kwargs['vocoder_type'] == 'hifigan'
            assert call_kwargs['encoder_backend'] == 'hubert'
            assert call_kwargs['encoder_type'] == 'linear'

    def test_singing_pipeline_passes_conformer_config(self, device):
        """SingingConversionPipeline passes conformer_config from config."""
        from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

        cfg = {'n_layers': 3, 'n_heads': 4}
        config = {'encoder_type': 'conformer', 'conformer_config': cfg}
        pipeline = SingingConversionPipeline(device=device, config=config)

        with patch('auto_voice.inference.model_manager.ModelManager.load') as mock_load:
            pipeline._get_model_manager()
            call_kwargs = mock_load.call_args[1]
            assert call_kwargs['conformer_config'] == cfg

    def test_realtime_pipeline_passes_conformer_config(self, device):
        """RealtimeVoiceConversionPipeline passes conformer_config from config."""
        from auto_voice.inference.realtime_voice_conversion_pipeline import RealtimeVoiceConversionPipeline

        cfg = {'n_layers': 3, 'n_heads': 4}
        config = {'encoder_type': 'conformer', 'conformer_config': cfg}
        pipeline = RealtimeVoiceConversionPipeline(device=device, config=config)

        with patch('auto_voice.inference.model_manager.ModelManager.load') as mock_load:
            pipeline._get_model_manager()
            call_kwargs = mock_load.call_args[1]
            assert call_kwargs['conformer_config'] == cfg


class TestModelManagerIntegrationBackends:
    """Integration tests: real ModelManager with all backend combinations."""

    def test_hubert_linear_produces_output(self, device):
        """hubert/linear full integration produces valid features."""
        mm = ModelManager(device=device)
        mm.load(encoder_backend='hubert', encoder_type='linear')
        audio = torch.randn(1, 16000).to(device)
        with torch.no_grad():
            out = mm._content_encoder.extract_features(audio, sr=16000)
        assert out.shape[0] == 1
        assert out.shape[2] == 768
        assert not torch.any(torch.isnan(out))

    def test_hubert_conformer_produces_output(self, device):
        """hubert/conformer full integration produces valid features."""
        mm = ModelManager(device=device)
        mm.load(encoder_backend='hubert', encoder_type='conformer',
                conformer_config={'n_layers': 2})
        audio = torch.randn(1, 16000).to(device)
        with torch.no_grad():
            out = mm._content_encoder.extract_features(audio, sr=16000)
        assert out.shape[0] == 1
        assert out.shape[2] == 768
        assert not torch.any(torch.isnan(out))

    def test_contentvec_linear_produces_output(self, device):
        """contentvec/linear full integration produces valid features."""
        mm = ModelManager(device=device)
        mm.load(encoder_backend='contentvec', encoder_type='linear')
        audio = torch.randn(1, 16000).to(device)
        with torch.no_grad():
            out = mm._content_encoder.extract_features(audio, sr=16000)
        assert out.shape[0] == 1
        assert out.shape[2] == 768
        assert not torch.any(torch.isnan(out))

    def test_contentvec_conformer_produces_output(self, device):
        """contentvec/conformer full integration produces valid features."""
        mm = ModelManager(device=device)
        mm.load(encoder_backend='contentvec', encoder_type='conformer',
                conformer_config={'n_layers': 2})
        audio = torch.randn(1, 16000).to(device)
        with torch.no_grad():
            out = mm._content_encoder.extract_features(audio, sr=16000)
        assert out.shape[0] == 1
        assert out.shape[2] == 768
        assert not torch.any(torch.isnan(out))

    def test_hifigan_vocoder_synthesizes(self, device):
        """HiFiGAN vocoder synthesizes waveform from mel."""
        mm = ModelManager(device=device)
        mm.load(vocoder_type='hifigan')
        mel = torch.randn(1, 80, 50).to(device)
        with torch.no_grad():
            wav = mm._vocoder.synthesize(mel)
        assert wav.dim() >= 1
        assert not torch.any(torch.isnan(wav))

    def test_bigvgan_vocoder_synthesizes(self, device):
        """BigVGAN vocoder synthesizes waveform from mel."""
        mm = ModelManager(device=device)
        mm.load(vocoder_type='bigvgan')
        mel = torch.randn(1, 100, 50).to(device)
        with torch.no_grad():
            wav = mm._vocoder.synthesize(mel)
        assert wav.dim() >= 1
        assert not torch.any(torch.isnan(wav))


class TestModelManagerConfigValidation:
    """Tests for config validation in ModelManager.__init__()."""

    def test_validation_invalid_vocoder_type_raises(self, device):
        """Invalid vocoder_type in config raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Invalid vocoder_type"):
            ModelManager(device=device, config={'vocoder_type': 'wavenet'})

    def test_validation_invalid_encoder_backend_raises(self, device):
        """Invalid encoder_backend in config raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Invalid encoder_backend"):
            ModelManager(device=device, config={'encoder_backend': 'wav2vec'})

    def test_validation_invalid_encoder_type_raises(self, device):
        """Invalid encoder_type in config raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Invalid encoder_type"):
            ModelManager(device=device, config={'encoder_type': 'transformer'})

    def test_validation_error_lists_valid_options_vocoder(self, device):
        """Error message lists valid vocoder_type options."""
        with pytest.raises(RuntimeError, match="hifigan.*bigvgan|bigvgan.*hifigan"):
            ModelManager(device=device, config={'vocoder_type': 'bad'})

    def test_validation_error_lists_valid_options_encoder_backend(self, device):
        """Error message lists valid encoder_backend options."""
        with pytest.raises(RuntimeError, match="hubert.*contentvec|contentvec.*hubert"):
            ModelManager(device=device, config={'encoder_backend': 'bad'})

    def test_validation_error_lists_valid_options_encoder_type(self, device):
        """Error message lists valid encoder_type options."""
        with pytest.raises(RuntimeError, match="linear.*conformer|conformer.*linear"):
            ModelManager(device=device, config={'encoder_type': 'bad'})

    def test_validation_valid_vocoder_type_hifigan(self, device):
        """Valid vocoder_type='hifigan' does not raise."""
        mm = ModelManager(device=device, config={'vocoder_type': 'hifigan'})
        assert mm.config['vocoder_type'] == 'hifigan'

    def test_validation_valid_vocoder_type_bigvgan(self, device):
        """Valid vocoder_type='bigvgan' does not raise."""
        mm = ModelManager(device=device, config={'vocoder_type': 'bigvgan'})
        assert mm.config['vocoder_type'] == 'bigvgan'

    def test_validation_valid_encoder_backend_hubert(self, device):
        """Valid encoder_backend='hubert' does not raise."""
        mm = ModelManager(device=device, config={'encoder_backend': 'hubert'})
        assert mm.config['encoder_backend'] == 'hubert'

    def test_validation_valid_encoder_backend_contentvec(self, device):
        """Valid encoder_backend='contentvec' does not raise."""
        mm = ModelManager(device=device, config={'encoder_backend': 'contentvec'})
        assert mm.config['encoder_backend'] == 'contentvec'

    def test_validation_valid_encoder_type_linear(self, device):
        """Valid encoder_type='linear' does not raise."""
        mm = ModelManager(device=device, config={'encoder_type': 'linear'})
        assert mm.config['encoder_type'] == 'linear'

    def test_validation_valid_encoder_type_conformer(self, device):
        """Valid encoder_type='conformer' does not raise."""
        mm = ModelManager(device=device, config={'encoder_type': 'conformer'})
        assert mm.config['encoder_type'] == 'conformer'

    def test_validation_no_config_keys_no_error(self, device):
        """Empty config does not trigger validation errors."""
        mm = ModelManager(device=device, config={})
        assert mm.config == {}

    def test_validation_unrelated_config_keys_ignored(self, device):
        """Unknown config keys are not validated (forward-compatible)."""
        mm = ModelManager(device=device, config={'sample_rate': 44100, 'custom_key': True})
        assert mm.sample_rate == 44100

    def test_validation_multiple_invalid_raises_first(self, device):
        """Multiple invalid values - raises on first checked (vocoder_type)."""
        with pytest.raises(RuntimeError, match="Invalid vocoder_type"):
            ModelManager(device=device, config={
                'vocoder_type': 'bad',
                'encoder_backend': 'bad',
            })
