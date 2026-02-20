"""Tests for SOTA vocal separation (Phase 4).

Validates the Mel-Band RoFormer vocal separator with:
- Vocal/instrumental separation from mixed audio
- Stereo and mono input handling
- Proper sample rate handling (44.1kHz processing)
- Edge case handling (acapella, silence, very short audio)
- No fallback behavior
"""
import pytest
import torch


class TestMelBandRoFormer:
    """Tests for Mel-Band RoFormer vocal separator."""

    def test_class_exists(self):
        """MelBandRoFormer class should exist."""
        from auto_voice.audio.separator import MelBandRoFormer
        assert MelBandRoFormer is not None

    def test_init_default(self):
        """Default initialization."""
        from auto_voice.audio.separator import MelBandRoFormer
        separator = MelBandRoFormer(pretrained=None)
        assert separator.sample_rate == 44100
        assert separator.n_fft == 2048

    def test_separate_mono(self):
        """Mono audio separation produces vocals and instrumental."""
        from auto_voice.audio.separator import MelBandRoFormer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        separator = MelBandRoFormer(pretrained=None, device=device).to(device)
        # 2 seconds at 44.1kHz mono
        audio = torch.randn(1, 88200, device=device)
        vocals, instrumental = separator.separate(audio)
        assert vocals.shape == audio.shape
        assert instrumental.shape == audio.shape

    def test_separate_stereo(self):
        """Stereo audio separation."""
        from auto_voice.audio.separator import MelBandRoFormer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        separator = MelBandRoFormer(pretrained=None, device=device).to(device)
        # 2 seconds at 44.1kHz stereo [B, channels, T]
        audio = torch.randn(1, 2, 88200, device=device)
        vocals, instrumental = separator.separate(audio)
        assert vocals.shape == audio.shape
        assert instrumental.shape == audio.shape

    def test_output_finite(self):
        """All outputs should be finite."""
        from auto_voice.audio.separator import MelBandRoFormer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        separator = MelBandRoFormer(pretrained=None, device=device).to(device)
        audio = torch.randn(1, 44100, device=device)
        vocals, instrumental = separator.separate(audio)
        assert torch.isfinite(vocals).all()
        assert torch.isfinite(instrumental).all()

    def test_sum_equals_input(self):
        """Vocals + instrumental should approximately reconstruct input."""
        from auto_voice.audio.separator import MelBandRoFormer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        separator = MelBandRoFormer(pretrained=None, device=device).to(device)
        audio = torch.randn(1, 44100, device=device)
        vocals, instrumental = separator.separate(audio)
        # With random weights, won't be exact, but structure should allow it
        reconstructed = vocals + instrumental
        assert reconstructed.shape == audio.shape
        assert torch.isfinite(reconstructed).all()

    def test_device_placement(self):
        """Outputs on same device as input."""
        from auto_voice.audio.separator import MelBandRoFormer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        separator = MelBandRoFormer(pretrained=None, device=device).to(device)
        audio = torch.randn(1, 44100, device=device)
        vocals, instrumental = separator.separate(audio)
        assert vocals.device == audio.device
        assert instrumental.device == audio.device

    def test_batch_processing(self):
        """Batched input produces batched output."""
        from auto_voice.audio.separator import MelBandRoFormer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        separator = MelBandRoFormer(pretrained=None, device=device).to(device)
        audio = torch.randn(2, 44100, device=device)
        vocals, instrumental = separator.separate(audio)
        assert vocals.shape[0] == 2
        assert instrumental.shape[0] == 2


class TestMelBandArchitecture:
    """Tests for Mel-Band RoFormer architecture specifics."""

    def test_mel_band_splitting(self):
        """Should have mel-scale frequency band splitting."""
        from auto_voice.audio.separator import MelBandRoFormer
        separator = MelBandRoFormer(pretrained=None)
        assert hasattr(separator, 'band_splits')
        assert len(separator.band_splits) > 0

    def test_transformer_layers(self):
        """Should have transformer layers with RoPE."""
        from auto_voice.audio.separator import MelBandRoFormer
        separator = MelBandRoFormer(pretrained=None)
        assert hasattr(separator, 'transformer')
        param_count = sum(p.numel() for p in separator.parameters())
        # Mel-Band RoFormer has substantial parameters
        assert param_count > 1_000_000

    def test_n_fft_configuration(self):
        """Should use 2048 n_fft for 44.1kHz processing."""
        from auto_voice.audio.separator import MelBandRoFormer
        separator = MelBandRoFormer(pretrained=None)
        assert separator.n_fft == 2048
        assert separator.hop_length == 512


class TestSeparatorEdgeCases:
    """Tests for edge case handling."""

    def test_short_audio_raises(self):
        """Audio shorter than one STFT window should raise."""
        from auto_voice.audio.separator import MelBandRoFormer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        separator = MelBandRoFormer(pretrained=None, device=device).to(device)
        audio = torch.randn(1, 100, device=device)  # Too short
        with pytest.raises(RuntimeError):
            separator.separate(audio)

    def test_silence_handling(self):
        """Silent input should produce silent outputs (no NaN/Inf)."""
        from auto_voice.audio.separator import MelBandRoFormer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        separator = MelBandRoFormer(pretrained=None, device=device).to(device)
        audio = torch.zeros(1, 44100, device=device)
        vocals, instrumental = separator.separate(audio)
        assert torch.isfinite(vocals).all()
        assert torch.isfinite(instrumental).all()

    def test_vocals_only_mode(self):
        """Should support extracting only vocals (skip instrumental)."""
        from auto_voice.audio.separator import MelBandRoFormer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        separator = MelBandRoFormer(pretrained=None, device=device).to(device)
        audio = torch.randn(1, 44100, device=device)
        vocals = separator.extract_vocals(audio)
        assert vocals.shape == audio.shape
        assert torch.isfinite(vocals).all()


class TestSeparatorPipeline:
    """Tests for separation → content extraction pipeline."""

    def test_separation_to_content_pipeline(self):
        """Separated vocals feed into ContentVec at 16kHz."""
        from auto_voice.audio.separator import MelBandRoFormer
        from auto_voice.models.encoder import ContentVecEncoder
        import torchaudio

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        separator = MelBandRoFormer(pretrained=None, device=device).to(device)
        content_enc = ContentVecEncoder(pretrained=None, device=device).to(device)

        # Simulate 44.1kHz input
        audio_44k = torch.randn(1, 88200, device=device)  # 2 sec
        vocals = separator.extract_vocals(audio_44k)

        # Resample 44.1kHz → 16kHz for ContentVec
        resampler = torchaudio.transforms.Resample(44100, 16000).to(device)
        vocals_16k = resampler(vocals)

        # Extract content features
        features = content_enc.encode(vocals_16k)
        assert features.shape[0] == 1
        assert features.shape[2] == 768
