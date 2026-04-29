"""Tests for SOTA SVC decoder (Phase 6).

Validates the CoMoSVC consistency model decoder with:
- Single-step inference (consistency distillation)
- BiDilConv decoder architecture
- Content (768-dim) + pitch (768-dim) + speaker (256-dim) conditioning
- Mel spectrogram output for BigVGAN vocoder
- Speaker similarity preservation
- Pitch preservation through conversion
"""
import pytest
import torch

from auto_voice.models.feature_contract import DEFAULT_CONTENT_DIM, DEFAULT_PITCH_DIM, DEFAULT_SPEAKER_DIM


class TestCoMoSVCDecoder:
    """Tests for CoMoSVC consistency model decoder."""

    def test_class_exists(self):
        """CoMoSVCDecoder class should exist."""
        from auto_voice.models.svc_decoder import CoMoSVCDecoder
        assert CoMoSVCDecoder is not None

    def test_init_default(self):
        """Default initialization."""
        from auto_voice.models.svc_decoder import CoMoSVCDecoder
        decoder = CoMoSVCDecoder()
        assert decoder.content_dim == DEFAULT_CONTENT_DIM  # ContentVec Layer 12
        assert decoder.pitch_dim == DEFAULT_PITCH_DIM  # Canonical training pitch contract
        assert decoder.speaker_dim == DEFAULT_SPEAKER_DIM  # mel-statistics embedding
        assert decoder.n_mels == 100  # BigVGAN input

    def test_forward_shape(self):
        """Decoder produces correct mel spectrogram shape."""
        from auto_voice.models.svc_decoder import CoMoSVCDecoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = CoMoSVCDecoder(device=device).to(device)

        T = 50  # 50 frames
        content = torch.randn(1, T, DEFAULT_CONTENT_DIM, device=device)
        pitch = torch.randn(1, T, DEFAULT_PITCH_DIM, device=device)
        speaker = torch.randn(1, DEFAULT_SPEAKER_DIM, device=device)  # Global speaker embedding

        mel = decoder(content, pitch, speaker)
        assert mel.shape == (1, 100, T)  # [B, n_mels, T]

    def test_single_step_inference(self):
        """Consistency model should produce output in single step."""
        from auto_voice.models.svc_decoder import CoMoSVCDecoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = CoMoSVCDecoder(device=device).to(device)

        T = 30
        content = torch.randn(1, T, DEFAULT_CONTENT_DIM, device=device)
        pitch = torch.randn(1, T, DEFAULT_PITCH_DIM, device=device)
        speaker = torch.randn(1, DEFAULT_SPEAKER_DIM, device=device)

        # Single-step (default) should work
        mel = decoder.infer(content, pitch, speaker, n_steps=1)
        assert mel.shape == (1, 100, T)
        assert torch.isfinite(mel).all()

    def test_multi_step_inference(self):
        """Multi-step should also work (higher quality)."""
        from auto_voice.models.svc_decoder import CoMoSVCDecoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = CoMoSVCDecoder(device=device).to(device)

        T = 30
        content = torch.randn(1, T, DEFAULT_CONTENT_DIM, device=device)
        pitch = torch.randn(1, T, DEFAULT_PITCH_DIM, device=device)
        speaker = torch.randn(1, DEFAULT_SPEAKER_DIM, device=device)

        mel_1step = decoder.infer(content, pitch, speaker, n_steps=1)
        mel_4step = decoder.infer(content, pitch, speaker, n_steps=4)

        # Both should produce valid output
        assert mel_1step.shape == mel_4step.shape
        assert torch.isfinite(mel_4step).all()

    def test_batch_processing(self):
        """Batched input produces batched output."""
        from auto_voice.models.svc_decoder import CoMoSVCDecoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = CoMoSVCDecoder(device=device).to(device)

        T = 40
        content = torch.randn(2, T, DEFAULT_CONTENT_DIM, device=device)
        pitch = torch.randn(2, T, DEFAULT_PITCH_DIM, device=device)
        speaker = torch.randn(2, DEFAULT_SPEAKER_DIM, device=device)

        mel = decoder(content, pitch, speaker)
        assert mel.shape == (2, 100, T)

    def test_output_finite(self):
        """All outputs should be finite."""
        from auto_voice.models.svc_decoder import CoMoSVCDecoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = CoMoSVCDecoder(device=device).to(device)

        T = 50
        content = torch.randn(1, T, DEFAULT_CONTENT_DIM, device=device)
        pitch = torch.randn(1, T, DEFAULT_PITCH_DIM, device=device)
        speaker = torch.randn(1, DEFAULT_SPEAKER_DIM, device=device)

        mel = decoder(content, pitch, speaker)
        assert torch.isfinite(mel).all()

    def test_device_placement(self):
        """Output on same device as input."""
        from auto_voice.models.svc_decoder import CoMoSVCDecoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = CoMoSVCDecoder(device=device).to(device)

        T = 30
        content = torch.randn(1, T, DEFAULT_CONTENT_DIM, device=device)
        pitch = torch.randn(1, T, DEFAULT_PITCH_DIM, device=device)
        speaker = torch.randn(1, DEFAULT_SPEAKER_DIM, device=device)

        mel = decoder(content, pitch, speaker)
        assert mel.device == content.device


class TestBiDilConv:
    """Tests for BiDilConv (Bidirectional Dilated Convolution) network."""

    def test_bidilconv_exists(self):
        """BiDilConv class should exist."""
        from auto_voice.models.svc_decoder import BiDilConv
        assert BiDilConv is not None

    def test_bidilconv_output_shape(self):
        """BiDilConv preserves temporal dimension."""
        from auto_voice.models.svc_decoder import BiDilConv
        block = BiDilConv(channels=256, kernel_size=3, n_layers=4)
        x = torch.randn(1, 256, 50)
        y = block(x)
        assert y.shape == x.shape

    def test_bidilconv_dilated(self):
        """Should use dilated convolutions with increasing dilation."""
        from auto_voice.models.svc_decoder import BiDilConv
        block = BiDilConv(channels=256, kernel_size=3, n_layers=4)
        # Should have layers with dilation 1, 2, 4, 8 (or similar pattern)
        assert len(block.layers) >= 4


class TestSpeakerConditioning:
    """Tests for speaker embedding conditioning."""

    def test_speaker_embedding_shape(self):
        """Speaker embedding should be 256-dim (mel-statistics)."""
        from auto_voice.models.svc_decoder import CoMoSVCDecoder
        decoder = CoMoSVCDecoder()
        assert decoder.speaker_dim == DEFAULT_SPEAKER_DIM

    def test_different_speakers_different_output(self):
        """Different speaker embeddings should produce different mels."""
        from auto_voice.models.svc_decoder import CoMoSVCDecoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = CoMoSVCDecoder(device=device).to(device)

        T = 30
        content = torch.randn(1, T, DEFAULT_CONTENT_DIM, device=device)
        pitch = torch.randn(1, T, DEFAULT_PITCH_DIM, device=device)
        speaker_a = torch.randn(1, DEFAULT_SPEAKER_DIM, device=device)
        speaker_b = torch.randn(1, DEFAULT_SPEAKER_DIM, device=device)

        mel_a = decoder(content, pitch, speaker_a)
        mel_b = decoder(content, pitch, speaker_b)

        assert not torch.allclose(mel_a, mel_b, atol=1e-3)

    def test_same_speaker_consistent(self):
        """Same speaker + same content with same seed produces same output."""
        from auto_voice.models.svc_decoder import CoMoSVCDecoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = CoMoSVCDecoder(device=device).to(device)
        decoder.eval()

        T = 30
        content = torch.randn(1, T, DEFAULT_CONTENT_DIM, device=device)
        pitch = torch.randn(1, T, DEFAULT_PITCH_DIM, device=device)
        speaker = torch.randn(1, DEFAULT_SPEAKER_DIM, device=device)

        with torch.no_grad():
            torch.manual_seed(42)
            mel_1 = decoder(content, pitch, speaker)
            torch.manual_seed(42)
            mel_2 = decoder(content, pitch, speaker)

        assert torch.allclose(mel_1, mel_2, atol=1e-5)


class TestPitchPreservation:
    """Tests for pitch preservation through decoder."""

    def test_pitch_affects_output(self):
        """Different pitch should produce different mels."""
        from auto_voice.models.svc_decoder import CoMoSVCDecoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = CoMoSVCDecoder(device=device).to(device)

        T = 30
        content = torch.randn(1, T, DEFAULT_CONTENT_DIM, device=device)
        pitch_low = torch.randn(1, T, DEFAULT_PITCH_DIM, device=device) * 0.5
        pitch_high = torch.randn(1, T, DEFAULT_PITCH_DIM, device=device) * 2.0
        speaker = torch.randn(1, DEFAULT_SPEAKER_DIM, device=device)

        mel_low = decoder(content, pitch_low, speaker)
        mel_high = decoder(content, pitch_high, speaker)

        assert not torch.allclose(mel_low, mel_high, atol=1e-3)
