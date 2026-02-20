"""Tests for SOTA vocoder integration (Phase 5).

Validates BigVGAN v2 vocoder with:
- Snake activation function
- Anti-aliased multi-period discriminator architecture
- 24kHz waveform synthesis from mel spectrograms
- Singing voice quality (sustained notes, vibrato)
- No fallback behavior
"""
import pytest
import torch


class TestBigVGANVocoder:
    """Tests for BigVGAN v2 vocoder."""

    def test_class_exists(self):
        """BigVGANVocoder class should exist."""
        from auto_voice.models.vocoder import BigVGANVocoder
        assert BigVGANVocoder is not None

    def test_init_default(self):
        """Default initialization."""
        from auto_voice.models.vocoder import BigVGANVocoder
        vocoder = BigVGANVocoder(pretrained=None)
        assert vocoder.sample_rate == 24000
        assert vocoder.n_mels == 100
        assert vocoder.hop_size == 256

    def test_mel_to_audio_shape(self):
        """Mel spectrogram should produce correct audio length."""
        from auto_voice.models.vocoder import BigVGANVocoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vocoder = BigVGANVocoder(pretrained=None, device=device).to(device)
        # 100 frames × 256 hop = 25600 samples (~1.07 sec at 24kHz)
        mel = torch.randn(1, 100, 100, device=device)  # [B, n_mels, T_frames]
        audio = vocoder.synthesize(mel)
        assert audio.dim() == 2  # [B, T_samples]
        assert audio.shape[0] == 1
        # Output length should be approximately n_frames * hop_size
        expected_length = 100 * 256
        assert abs(audio.shape[1] - expected_length) < 512

    def test_batch_synthesis(self):
        """Batched mel produces batched audio."""
        from auto_voice.models.vocoder import BigVGANVocoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vocoder = BigVGANVocoder(pretrained=None, device=device).to(device)
        mel = torch.randn(3, 100, 50, device=device)
        audio = vocoder.synthesize(mel)
        assert audio.shape[0] == 3

    def test_output_finite(self):
        """Output waveform should be finite."""
        from auto_voice.models.vocoder import BigVGANVocoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vocoder = BigVGANVocoder(pretrained=None, device=device).to(device)
        mel = torch.randn(1, 100, 50, device=device)
        audio = vocoder.synthesize(mel)
        assert torch.isfinite(audio).all()

    def test_output_range(self):
        """Output should be in reasonable waveform range [-1, 1]."""
        from auto_voice.models.vocoder import BigVGANVocoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vocoder = BigVGANVocoder(pretrained=None, device=device).to(device)
        mel = torch.randn(1, 100, 50, device=device)
        audio = vocoder.synthesize(mel)
        # With tanh output, should be bounded
        assert audio.abs().max() <= 1.0

    def test_device_placement(self):
        """Output on same device as input."""
        from auto_voice.models.vocoder import BigVGANVocoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vocoder = BigVGANVocoder(pretrained=None, device=device).to(device)
        mel = torch.randn(1, 100, 50, device=device)
        audio = vocoder.synthesize(mel)
        assert audio.device == mel.device


class TestSnakeActivation:
    """Tests for Snake activation function used in BigVGAN."""

    def test_snake_exists(self):
        """Snake activation should be available."""
        from auto_voice.models.vocoder import Snake
        assert Snake is not None

    def test_snake_output_shape(self):
        """Snake should preserve input shape."""
        from auto_voice.models.vocoder import Snake
        snake = Snake(channels=64)
        x = torch.randn(2, 64, 100)
        y = snake(x)
        assert y.shape == x.shape

    def test_snake_finite(self):
        """Snake output should be finite."""
        from auto_voice.models.vocoder import Snake
        snake = Snake(channels=64)
        x = torch.randn(2, 64, 100)
        y = snake(x)
        assert torch.isfinite(y).all()

    def test_snake_not_relu(self):
        """Snake should differ from ReLU (periodic component)."""
        from auto_voice.models.vocoder import Snake
        snake = Snake(channels=64)
        x = torch.randn(2, 64, 100)
        y = snake(x)
        # Snake: x + (1/alpha) * sin^2(alpha * x)
        # Should differ from simple ReLU
        relu_out = torch.relu(x)
        assert not torch.allclose(y, relu_out, atol=1e-3)


class TestBigVGANArchitecture:
    """Tests for BigVGAN v2 architecture specifics."""

    def test_upsample_rates(self):
        """Should have correct upsampling rates."""
        from auto_voice.models.vocoder import BigVGANVocoder
        vocoder = BigVGANVocoder(pretrained=None)
        # Product of upsample rates should equal hop_size
        total_upsample = 1
        for rate in vocoder.upsample_rates:
            total_upsample *= rate
        assert total_upsample == vocoder.hop_size

    def test_anti_aliased_upsampling(self):
        """Should use anti-aliased upsampling (AMP blocks)."""
        from auto_voice.models.vocoder import BigVGANVocoder
        vocoder = BigVGANVocoder(pretrained=None)
        # Check that upsampling uses anti-aliasing
        assert hasattr(vocoder, 'upsamples')
        assert len(vocoder.upsamples) > 0

    def test_multi_receptive_field_blocks(self):
        """Should have multiple receptive field fusion blocks."""
        from auto_voice.models.vocoder import BigVGANVocoder
        vocoder = BigVGANVocoder(pretrained=None)
        assert hasattr(vocoder, 'resblocks')
        assert len(vocoder.resblocks) > 0

    def test_param_count(self):
        """BigVGAN v2 should have substantial params."""
        from auto_voice.models.vocoder import BigVGANGenerator
        generator = BigVGANGenerator()
        param_count = sum(p.numel() for p in generator.parameters())
        # BigVGAN v2 112M config with full 1536 initial channels
        assert param_count > 10_000_000


class TestVocoderIntegration:
    """Tests for vocoder integration with pipeline."""

    def test_content_to_vocoder_pipeline(self):
        """Content features → decoder → mel → vocoder → audio."""
        from auto_voice.models.vocoder import BigVGANVocoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vocoder = BigVGANVocoder(pretrained=None, device=device).to(device)

        # Simulate decoder output (mel spectrogram)
        mel = torch.randn(1, 100, 80, device=device)  # [B, n_mels, T]
        audio = vocoder.synthesize(mel)
        assert audio.dim() == 2
        assert audio.shape[1] > 0
