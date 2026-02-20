"""Tests for neural network model architectures."""
import pytest
import torch
import numpy as np


class TestContentEncoder:
    """Tests for ContentEncoder."""

    def test_init_default(self):
        from auto_voice.models.encoder import ContentEncoder
        encoder = ContentEncoder()
        assert encoder.hidden_size == 256
        assert encoder.output_size == 256

    def test_init_custom_sizes(self):
        from auto_voice.models.encoder import ContentEncoder
        encoder = ContentEncoder(hidden_size=128, output_size=64)
        assert encoder.hidden_size == 128
        assert encoder.output_size == 64

    def test_forward_1d_audio(self):
        from auto_voice.models.encoder import ContentEncoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = ContentEncoder(device=device).to(device)
        audio = torch.randn(16000, device=device)  # 1 second at 16kHz
        features = encoder(audio, sr=16000)
        assert features.dim() == 3
        assert features.shape[0] == 1  # batch
        assert features.shape[2] == 256  # output_size

    def test_forward_batched_audio(self):
        from auto_voice.models.encoder import ContentEncoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = ContentEncoder(device=device).to(device)
        audio = torch.randn(2, 16000, device=device)
        features = encoder(audio, sr=16000)
        assert features.shape[0] == 2

    def test_hubert_always_initialized(self):
        from auto_voice.models.encoder import ContentEncoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = ContentEncoder(device=device).to(device)
        # Without pretrained weights, HuBERTSoft is initialized with random weights
        audio = torch.randn(1, 8000, device=device)
        features = encoder.extract_features(audio, sr=16000)
        assert features.shape[2] == 256
        assert encoder._hubert is not None  # Always initialized

    def test_load_pretrained_missing(self):
        from auto_voice.models.encoder import ContentEncoder
        encoder = ContentEncoder.load_pretrained('/nonexistent/path.pt')
        # HuBERTSoft is always initialized (with random weights when path missing)
        assert encoder._hubert is not None
        assert encoder._hubert_loaded is True

    def test_projection_output(self):
        from auto_voice.models.encoder import ContentEncoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = ContentEncoder(output_size=128, device=device).to(device)
        audio = torch.randn(1, 16000, device=device)
        features = encoder(audio, sr=16000)
        assert features.shape[2] == 128


class TestPitchEncoder:
    """Tests for PitchEncoder."""

    def test_init(self):
        from auto_voice.models.encoder import PitchEncoder
        encoder = PitchEncoder()
        assert encoder.output_size == 256

    def test_forward_2d(self):
        from auto_voice.models.encoder import PitchEncoder
        encoder = PitchEncoder()
        f0 = torch.randn(2, 100)  # [batch, time]
        output = encoder(f0)
        assert output.shape == (2, 100, 256)

    def test_forward_3d(self):
        from auto_voice.models.encoder import PitchEncoder
        encoder = PitchEncoder()
        f0 = torch.randn(2, 100, 1)  # [batch, time, 1]
        output = encoder(f0)
        assert output.shape == (2, 100, 256)

    def test_custom_output_size(self):
        from auto_voice.models.encoder import PitchEncoder
        encoder = PitchEncoder(output_size=128)
        f0 = torch.randn(1, 50)
        output = encoder(f0)
        assert output.shape == (1, 50, 128)

    def test_gradient_flow(self):
        from auto_voice.models.encoder import PitchEncoder
        encoder = PitchEncoder()
        f0 = torch.randn(1, 50, requires_grad=True)
        output = encoder(f0)
        loss = output.sum()
        loss.backward()
        assert f0.grad is not None


class TestHuBERTSoft:
    """Tests for HuBERTSoft."""

    def test_init(self):
        from auto_voice.models.encoder import HuBERTSoft
        model = HuBERTSoft()
        assert not model._loaded

    def test_encode_shape(self):
        from auto_voice.models.encoder import HuBERTSoft
        model = HuBERTSoft()
        audio = torch.randn(1, 16000)
        units = model.encode(audio)
        assert units.dim() == 3
        assert units.shape[0] == 1
        assert units.shape[2] == 256

    def test_forward_equals_encode(self):
        from auto_voice.models.encoder import HuBERTSoft
        model = HuBERTSoft()
        audio = torch.randn(1, 16000)
        assert torch.equal(model(audio), model.encode(audio))


class TestContentVecEncoder:
    """Tests for ContentVecEncoder."""

    def test_init_default(self):
        from auto_voice.models.encoder import ContentVecEncoder
        encoder = ContentVecEncoder(pretrained=None)
        assert encoder.output_dim == 768
        assert encoder.layer == 12
        assert encoder._model is None

    def test_init_custom_layer(self):
        from auto_voice.models.encoder import ContentVecEncoder
        encoder = ContentVecEncoder(output_dim=128, layer=6, pretrained=None)
        assert encoder.output_dim == 128
        assert encoder.layer == 6

    def test_encode_1d(self):
        from auto_voice.models.encoder import ContentVecEncoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = ContentVecEncoder(pretrained=None, device=device).to(device)
        audio = torch.randn(16000, device=device)  # 1 second at 16kHz
        features = encoder.encode(audio)
        assert features.dim() == 3
        assert features.shape[0] == 1  # batch
        assert features.shape[2] == 768  # output_dim (Layer 12, no projection)

    def test_encode_batched(self):
        from auto_voice.models.encoder import ContentVecEncoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = ContentVecEncoder(pretrained=None, device=device).to(device)
        audio = torch.randn(2, 16000, device=device)
        features = encoder.encode(audio)
        assert features.shape[0] == 2
        assert features.shape[2] == 768

    def test_frame_resolution_20ms(self):
        from auto_voice.models.encoder import ContentVecEncoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = ContentVecEncoder(pretrained=None, device=device).to(device)
        # 1 second = 50 frames at 20ms per frame
        audio = torch.randn(1, 16000, device=device)
        features = encoder.encode(audio)
        # HuBERT/ContentVec produces ~49-50 frames for 1 second at 16kHz
        assert 40 <= features.shape[1] <= 55

    def test_forward_equals_encode(self):
        from auto_voice.models.encoder import ContentVecEncoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = ContentVecEncoder(pretrained=None, device=device).to(device)
        audio = torch.randn(1, 16000, device=device)
        # Both should produce same result
        enc = encoder.encode(audio)
        fwd = encoder(audio)
        assert torch.equal(enc, fwd)

    def test_output_finite(self):
        from auto_voice.models.encoder import ContentVecEncoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = ContentVecEncoder(pretrained=None, device=device).to(device)
        audio = torch.randn(1, 16000, device=device)
        features = encoder.encode(audio)
        assert torch.isfinite(features).all()

    def test_lazy_loading(self):
        from auto_voice.models.encoder import ContentVecEncoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = ContentVecEncoder(pretrained=None, device=device).to(device)
        assert encoder._model is None
        assert not encoder._loaded
        # Model loads on first encode
        audio = torch.randn(1, 16000, device=device)
        encoder.encode(audio)
        assert encoder._model is not None
        assert encoder._loaded


class TestContentEncoderBackends:
    """Tests for ContentEncoder with different backends."""

    def test_default_backend_is_hubert(self):
        from auto_voice.models.encoder import ContentEncoder
        encoder = ContentEncoder()
        assert encoder.encoder_backend == 'hubert'
        assert encoder._contentvec is None

    def test_contentvec_backend_init(self):
        from auto_voice.models.encoder import ContentEncoder
        encoder = ContentEncoder(encoder_backend='contentvec', contentvec_model=None)
        assert encoder.encoder_backend == 'contentvec'
        assert encoder._contentvec is not None
        assert encoder._hubert is None

    def test_contentvec_backend_forward(self):
        from auto_voice.models.encoder import ContentEncoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = ContentEncoder(
            encoder_backend='contentvec',
            contentvec_model=None,
            device=device,
        ).to(device)
        audio = torch.randn(1, 16000, device=device)
        features = encoder(audio, sr=16000)
        assert features.dim() == 3
        assert features.shape[0] == 1
        assert features.shape[2] == 256  # output_size

    def test_contentvec_custom_output_size(self):
        from auto_voice.models.encoder import ContentEncoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = ContentEncoder(
            output_size=128,
            encoder_backend='contentvec',
            contentvec_model=None,
            device=device,
        ).to(device)
        audio = torch.randn(1, 16000, device=device)
        features = encoder(audio, sr=16000)
        assert features.shape[2] == 128

    def test_contentvec_with_conformer_projection(self):
        from auto_voice.models.encoder import ContentEncoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = ContentEncoder(
            encoder_backend='contentvec',
            contentvec_model=None,
            encoder_type='conformer',
            device=device,
        ).to(device)
        audio = torch.randn(1, 16000, device=device)
        features = encoder(audio, sr=16000)
        assert features.dim() == 3
        assert features.shape[2] == 256

    def test_contentvec_custom_layer(self):
        from auto_voice.models.encoder import ContentEncoder
        encoder = ContentEncoder(
            encoder_backend='contentvec',
            contentvec_model=None,
            contentvec_layer=12,
        )
        assert encoder._contentvec.layer == 12

    def test_hubert_backend_unchanged(self):
        from auto_voice.models.encoder import ContentEncoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = ContentEncoder(device=device).to(device)
        audio = torch.randn(1, 16000, device=device)
        features = encoder(audio, sr=16000)
        assert features.shape[2] == 256
        assert encoder._hubert is not None


class TestHiFiGANVocoder:
    """Tests for HiFiGAN vocoder."""

    def test_init(self):
        from auto_voice.models.vocoder import HiFiGANVocoder
        vocoder = HiFiGANVocoder()
        assert vocoder.sample_rate == 22050

    def test_synthesize_2d(self):
        from auto_voice.models.vocoder import HiFiGANVocoder
        vocoder = HiFiGANVocoder()
        mel = torch.randn(80, 50)  # [mels, time]
        audio = vocoder.synthesize(mel)
        assert audio.dim() == 2  # [batch, time]
        assert audio.shape[1] > 0

    def test_synthesize_3d(self):
        from auto_voice.models.vocoder import HiFiGANVocoder
        vocoder = HiFiGANVocoder()
        mel = torch.randn(2, 80, 50)  # [batch, mels, time]
        audio = vocoder.synthesize(mel)
        assert audio.shape[0] == 2

    def test_output_range(self):
        from auto_voice.models.vocoder import HiFiGANVocoder
        vocoder = HiFiGANVocoder()
        mel = torch.randn(1, 80, 20)
        audio = vocoder.synthesize(mel)
        # tanh output should be in [-1, 1]
        assert audio.max() <= 1.0
        assert audio.min() >= -1.0

    def test_mel_to_audio_numpy(self):
        from auto_voice.models.vocoder import HiFiGANVocoder
        vocoder = HiFiGANVocoder()
        mel = np.random.randn(80, 30).astype(np.float32)
        audio = vocoder.mel_to_audio(mel)
        assert isinstance(audio, np.ndarray)

    def test_load_pretrained_missing(self):
        from auto_voice.models.vocoder import HiFiGANVocoder
        vocoder = HiFiGANVocoder.load_pretrained('/nonexistent.ckpt')
        assert not vocoder._loaded

    def test_generator_upsample_ratio(self):
        from auto_voice.models.vocoder import HiFiGANGenerator
        gen = HiFiGANGenerator()
        mel = torch.randn(1, 80, 10)
        audio = gen(mel)
        # Upsample rates: 8*8*2*2 = 256
        assert audio.shape[-1] == 10 * 256


class TestHiFiGANGenerator:
    """Tests for HiFiGAN generator module."""

    def test_init(self):
        from auto_voice.models.vocoder import HiFiGANGenerator
        gen = HiFiGANGenerator()
        assert gen.num_upsamples == 4
        assert gen.num_kernels == 3

    def test_forward(self):
        from auto_voice.models.vocoder import HiFiGANGenerator
        gen = HiFiGANGenerator()
        mel = torch.randn(1, 80, 20)
        audio = gen(mel)
        assert audio.shape[0] == 1
        assert audio.shape[1] == 1  # mono channel

    def test_remove_weight_norm(self):
        from auto_voice.models.vocoder import HiFiGANGenerator
        gen = HiFiGANGenerator()
        gen.remove_weight_norm()  # Should not raise


class TestSoVitsSvc:
    """Tests for So-VITS-SVC model."""

    def test_init_default(self):
        from auto_voice.models.so_vits_svc import SoVitsSvc
        model = SoVitsSvc()
        assert model.content_dim == 256
        assert model.hidden_dim == 192

    def test_init_custom_config(self):
        from auto_voice.models.so_vits_svc import SoVitsSvc
        model = SoVitsSvc(config={'hidden_dim': 128})
        assert model.hidden_dim == 128

    def test_infer_output_shape(self):
        from auto_voice.models.so_vits_svc import SoVitsSvc
        model = SoVitsSvc()
        content = torch.randn(1, 50, 256)
        pitch = torch.randn(1, 50, 256)
        speaker = torch.randn(1, 256)
        mel = model.infer(content, pitch, speaker)
        assert mel.shape == (1, 80, 50)

    def test_forward_training(self):
        from auto_voice.models.so_vits_svc import SoVitsSvc
        model = SoVitsSvc()
        content = torch.randn(2, 30, 256)
        pitch = torch.randn(2, 30, 256)
        speaker = torch.randn(2, 256)
        spec = torch.randn(2, 513, 30)
        outputs = model(content, pitch, speaker, spec=spec)
        assert 'mel_pred' in outputs
        assert 'mean' in outputs
        assert 'logvar' in outputs
        assert 'z_flow' in outputs

    def test_forward_inference(self):
        from auto_voice.models.so_vits_svc import SoVitsSvc
        model = SoVitsSvc()
        content = torch.randn(1, 40, 256)
        pitch = torch.randn(1, 40, 256)
        speaker = torch.randn(1, 256)
        outputs = model(content, pitch, speaker, spec=None)
        assert 'mel_pred' in outputs
        assert 'z' in outputs
        assert 'mean' not in outputs  # No posterior in inference

    def test_compute_loss(self):
        from auto_voice.models.so_vits_svc import SoVitsSvc
        model = SoVitsSvc()
        content = torch.randn(2, 20, 256)
        pitch = torch.randn(2, 20, 256)
        speaker = torch.randn(2, 256)
        spec = torch.randn(2, 513, 20)
        target_mel = torch.randn(2, 80, 20)

        outputs = model(content, pitch, speaker, spec=spec)
        losses = model.compute_loss(outputs, target_mel)

        assert 'reconstruction_loss' in losses
        assert 'kl_loss' in losses
        assert 'flow_loss' in losses
        assert 'total_loss' in losses
        assert losses['total_loss'].requires_grad

    def test_load_pretrained_missing(self):
        from auto_voice.models.so_vits_svc import SoVitsSvc
        model = SoVitsSvc.load_pretrained('/nonexistent.pth')
        # Should still work, just uninitialized
        content = torch.randn(1, 10, 256)
        pitch = torch.randn(1, 10, 256)
        speaker = torch.randn(1, 256)
        mel = model.infer(content, pitch, speaker)
        assert mel.shape == (1, 80, 10)

    def test_gradient_flow(self):
        from auto_voice.models.so_vits_svc import SoVitsSvc
        model = SoVitsSvc()
        content = torch.randn(1, 10, 256)
        pitch = torch.randn(1, 10, 256)
        speaker = torch.randn(1, 256)
        spec = torch.randn(1, 513, 10)
        target = torch.randn(1, 80, 10)

        outputs = model(content, pitch, speaker, spec=spec)
        losses = model.compute_loss(outputs, target)
        losses['total_loss'].backward()

        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                break


class TestFlowDecoder:
    """Tests for flow normalizing decoder."""

    def test_forward(self):
        from auto_voice.models.so_vits_svc import FlowDecoder
        flow = FlowDecoder(channels=192)
        x = torch.randn(1, 192, 20)
        y = flow(x)
        assert y.shape == x.shape

    def test_reverse(self):
        from auto_voice.models.so_vits_svc import FlowDecoder
        flow = FlowDecoder(channels=192)
        x = torch.randn(1, 192, 20)
        y = flow(x, reverse=True)
        assert y.shape == x.shape

    def test_invertibility(self):
        from auto_voice.models.so_vits_svc import FlowDecoder
        flow = FlowDecoder(channels=192, n_flows=2)
        x = torch.randn(1, 192, 10)
        y = flow(x)
        x_rec = flow(y, reverse=True)
        # Should be approximately invertible
        assert torch.allclose(x, x_rec, atol=1e-4)


# Small config for fast tests (reduces 1536->64 channels)
BIGVGAN_TEST_CONFIG = {
    'num_mels': 100,
    'upsample_rates': [4, 4, 2, 2, 2, 2],
    'upsample_kernel_sizes': [8, 8, 4, 4, 4, 4],
    'upsample_initial_channel': 64,
    'resblock_kernel_sizes': [3, 7, 11],
    'resblock_dilation_sizes': [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    'sample_rate': 24000,
    'hop_size': 256,
    'activation': 'snakebeta',
    'snake_logscale': True,
}


class TestSnakeBeta:
    """Tests for SnakeBeta activation function."""

    def test_init(self):
        from auto_voice.models.vocoder import SnakeBeta
        act = SnakeBeta(channels=32)
        assert act.log_alpha.shape == (1, 32, 1)
        assert act.log_beta.shape == (1, 32, 1)

    def test_forward_shape(self):
        from auto_voice.models.vocoder import SnakeBeta
        act = SnakeBeta(channels=16)
        x = torch.randn(2, 16, 100)
        y = act(x)
        assert y.shape == x.shape

    def test_output_not_equal_input(self):
        from auto_voice.models.vocoder import SnakeBeta
        act = SnakeBeta(channels=8)
        # Set non-zero alpha to produce non-trivial output
        act.log_alpha.data.fill_(1.0)
        act.log_beta.data.fill_(0.5)
        x = torch.randn(1, 8, 50)
        y = act(x)
        assert not torch.equal(x, y)

    def test_output_finite(self):
        from auto_voice.models.vocoder import SnakeBeta
        act = SnakeBeta(channels=16)
        x = torch.randn(1, 16, 200)
        y = act(x)
        assert torch.isfinite(y).all()

    def test_residual_connection(self):
        from auto_voice.models.vocoder import SnakeBeta
        act = SnakeBeta(channels=4)
        x = torch.randn(1, 4, 10)
        y = act(x)
        # Output should be x + periodic_component, so always >= x for positive sin^2
        # The periodic part is always non-negative
        diff = y - x
        assert (diff >= -1e-6).all()


class TestSnake:
    """Tests for Snake activation function."""

    def test_init(self):
        from auto_voice.models.vocoder import Snake
        act = Snake(channels=32)
        assert act.log_alpha.shape == (1, 32, 1)

    def test_forward_shape(self):
        from auto_voice.models.vocoder import Snake
        act = Snake(channels=16)
        x = torch.randn(2, 16, 100)
        y = act(x)
        assert y.shape == x.shape

    def test_output_finite(self):
        from auto_voice.models.vocoder import Snake
        act = Snake(channels=8)
        x = torch.randn(1, 8, 300)
        y = act(x)
        assert torch.isfinite(y).all()


class TestActivation1d:
    """Tests for anti-aliased 1D activation."""

    def test_shape_preserved(self):
        from auto_voice.models.vocoder import Activation1d, SnakeBeta
        act = Activation1d(SnakeBeta(channels=8))
        x = torch.randn(1, 8, 50)
        y = act(x)
        assert y.shape == x.shape

    def test_custom_up_ratio(self):
        from auto_voice.models.vocoder import Activation1d, Snake
        act = Activation1d(Snake(channels=4), up_ratio=4)
        x = torch.randn(2, 4, 30)
        y = act(x)
        assert y.shape == x.shape

    def test_output_finite(self):
        from auto_voice.models.vocoder import Activation1d, SnakeBeta
        act = Activation1d(SnakeBeta(channels=16))
        x = torch.randn(1, 16, 100)
        y = act(x)
        assert torch.isfinite(y).all()


class TestAMPBlock:
    """Tests for Anti-aliased Multi-Periodicity block."""

    def test_forward_shape(self):
        from auto_voice.models.vocoder import AMPBlock
        block = AMPBlock(channels=16, kernel_size=3, dilations=[1, 3, 5])
        x = torch.randn(1, 16, 50)
        y = block(x)
        assert y.shape == x.shape

    def test_residual_connection(self):
        from auto_voice.models.vocoder import AMPBlock
        block = AMPBlock(channels=8, kernel_size=3, dilations=[1, 3, 5])
        x = torch.randn(1, 8, 30)
        y = block(x)
        # Output should differ from input (residuals added)
        assert not torch.equal(x, y)

    def test_snake_activation_variant(self):
        from auto_voice.models.vocoder import AMPBlock
        block = AMPBlock(channels=8, kernel_size=7, dilations=[1, 3, 5],
                         activation='snake')
        x = torch.randn(1, 8, 20)
        y = block(x)
        assert y.shape == x.shape

    def test_unknown_activation_raises(self):
        from auto_voice.models.vocoder import AMPBlock
        with pytest.raises(RuntimeError, match="Unknown activation"):
            AMPBlock(channels=8, kernel_size=3, dilations=[1], activation='invalid')

    def test_gradient_flow(self):
        from auto_voice.models.vocoder import AMPBlock
        block = AMPBlock(channels=8, kernel_size=3, dilations=[1, 3, 5])
        x = torch.randn(1, 8, 20, requires_grad=True)
        y = block(x)
        y.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestBigVGANGenerator:
    """Tests for BigVGAN generator module."""

    def test_init(self):
        from auto_voice.models.vocoder import BigVGANGenerator
        gen = BigVGANGenerator(upsample_initial_channel=64)
        assert gen.num_upsamples == 6
        assert gen.num_kernels == 3

    def test_forward_shape(self):
        from auto_voice.models.vocoder import BigVGANGenerator
        gen = BigVGANGenerator(num_mels=100, upsample_initial_channel=64)
        mel = torch.randn(1, 100, 10)
        audio = gen(mel)
        # Upsample: 4*4*2*2*2*2 = 256
        assert audio.shape == (1, 1, 10 * 256)

    def test_forward_batch(self):
        from auto_voice.models.vocoder import BigVGANGenerator
        gen = BigVGANGenerator(num_mels=100, upsample_initial_channel=64)
        mel = torch.randn(2, 100, 8)
        audio = gen(mel)
        assert audio.shape[0] == 2
        assert audio.shape[1] == 1
        assert audio.shape[2] == 8 * 256

    def test_output_range_tanh(self):
        from auto_voice.models.vocoder import BigVGANGenerator
        gen = BigVGANGenerator(num_mels=100, upsample_initial_channel=64)
        mel = torch.randn(1, 100, 5)
        audio = gen(mel)
        assert audio.max() <= 1.0
        assert audio.min() >= -1.0

    def test_output_finite(self):
        from auto_voice.models.vocoder import BigVGANGenerator
        gen = BigVGANGenerator(num_mels=100, upsample_initial_channel=64)
        mel = torch.randn(1, 100, 10)
        audio = gen(mel)
        assert torch.isfinite(audio).all()

    def test_remove_weight_norm(self):
        from auto_voice.models.vocoder import BigVGANGenerator
        gen = BigVGANGenerator(num_mels=100, upsample_initial_channel=64)
        gen.remove_weight_norm()  # Should not raise
        mel = torch.randn(1, 100, 5)
        audio = gen(mel)
        assert audio.shape[2] == 5 * 256

    def test_snake_activation_variant(self):
        from auto_voice.models.vocoder import BigVGANGenerator
        gen = BigVGANGenerator(num_mels=100, upsample_initial_channel=64,
                               activation='snake')
        mel = torch.randn(1, 100, 5)
        audio = gen(mel)
        assert audio.shape == (1, 1, 5 * 256)

    def test_gradient_flow(self):
        from auto_voice.models.vocoder import BigVGANGenerator
        gen = BigVGANGenerator(num_mels=100, upsample_initial_channel=64)
        mel = torch.randn(1, 100, 5, requires_grad=True)
        audio = gen(mel)
        audio.sum().backward()
        assert mel.grad is not None


class TestBigVGANVocoder:
    """Tests for BigVGAN vocoder high-level interface."""

    def test_init(self):
        from auto_voice.models.vocoder import BigVGANVocoder
        vocoder = BigVGANVocoder(config=BIGVGAN_TEST_CONFIG)
        assert vocoder.sample_rate == 24000

    def test_synthesize_2d(self):
        from auto_voice.models.vocoder import BigVGANVocoder
        vocoder = BigVGANVocoder(config=BIGVGAN_TEST_CONFIG)
        mel = torch.randn(100, 20)  # [mels, time]
        audio = vocoder.synthesize(mel)
        assert audio.dim() == 2  # [batch, time]
        assert audio.shape[0] == 1
        assert audio.shape[1] == 20 * 256

    def test_synthesize_3d(self):
        from auto_voice.models.vocoder import BigVGANVocoder
        vocoder = BigVGANVocoder(config=BIGVGAN_TEST_CONFIG)
        mel = torch.randn(2, 100, 15)  # [batch, mels, time]
        audio = vocoder.synthesize(mel)
        assert audio.shape[0] == 2
        assert audio.shape[1] == 15 * 256

    def test_output_range(self):
        from auto_voice.models.vocoder import BigVGANVocoder
        vocoder = BigVGANVocoder(config=BIGVGAN_TEST_CONFIG)
        mel = torch.randn(1, 100, 10)
        audio = vocoder.synthesize(mel)
        assert audio.max() <= 1.0
        assert audio.min() >= -1.0

    def test_mel_to_audio_numpy(self):
        from auto_voice.models.vocoder import BigVGANVocoder
        vocoder = BigVGANVocoder(config=BIGVGAN_TEST_CONFIG)
        mel = np.random.randn(100, 10).astype(np.float32)
        audio = vocoder.mel_to_audio(mel)
        assert isinstance(audio, np.ndarray)
        assert len(audio) == 10 * 256

    def test_load_checkpoint_missing_raises(self):
        from auto_voice.models.vocoder import BigVGANVocoder
        vocoder = BigVGANVocoder(config=BIGVGAN_TEST_CONFIG)
        with pytest.raises(RuntimeError, match="checkpoint not found"):
            vocoder.load_checkpoint('/nonexistent/bigvgan.pt')

    def test_load_pretrained_missing_raises(self):
        from auto_voice.models.vocoder import BigVGANVocoder
        with pytest.raises(RuntimeError, match="checkpoint not found"):
            BigVGANVocoder.load_pretrained('/nonexistent.pt')

    def test_default_config_is_24khz_100band(self):
        from auto_voice.models.vocoder import BigVGANVocoder, BIGVGAN_24KHZ_100BAND_CONFIG
        vocoder = BigVGANVocoder()
        assert vocoder.config == BIGVGAN_24KHZ_100BAND_CONFIG
        assert vocoder.sample_rate == 24000

    def test_lazy_initialization(self):
        from auto_voice.models.vocoder import BigVGANVocoder
        vocoder = BigVGANVocoder(config=BIGVGAN_TEST_CONFIG)
        assert vocoder._generator is None
        # Synthesize triggers initialization
        mel = torch.randn(100, 5)
        vocoder.synthesize(mel)
        assert vocoder._generator is not None


class TestModelManagerBigVGAN:
    """Tests for ModelManager with BigVGAN vocoder option."""

    def test_load_bigvgan(self):
        from auto_voice.inference.model_manager import ModelManager
        mgr = ModelManager()
        mgr.load(vocoder_type='bigvgan')
        from auto_voice.models.vocoder import BigVGANVocoder
        assert isinstance(mgr._vocoder, BigVGANVocoder)

    def test_load_hifigan_default(self):
        from auto_voice.inference.model_manager import ModelManager
        mgr = ModelManager()
        mgr.load(vocoder_type='hifigan')
        from auto_voice.models.vocoder import HiFiGANVocoder
        assert isinstance(mgr._vocoder, HiFiGANVocoder)

    def test_load_default_is_hifigan(self):
        from auto_voice.inference.model_manager import ModelManager
        mgr = ModelManager()
        mgr.load()
        from auto_voice.models.vocoder import HiFiGANVocoder
        assert isinstance(mgr._vocoder, HiFiGANVocoder)

    def test_load_unknown_vocoder_raises(self):
        from auto_voice.inference.model_manager import ModelManager
        mgr = ModelManager()
        with pytest.raises(RuntimeError, match="Unknown vocoder_type"):
            mgr.load(vocoder_type='wavenet')
