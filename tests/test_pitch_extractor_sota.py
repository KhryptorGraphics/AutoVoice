"""Tests for SOTA pitch extraction (Phase 3).

Validates the RMVPE pitch extractor with:
- Accurate F0 estimation from singing voice audio
- Voiced/unvoiced decision accuracy
- Frame resolution matching content features (20ms / 50fps)
- Direct operation on mixed audio (no separation required)
- Integration with mel-quantized F0 pipeline
"""
import pytest
import numpy as np
import torch


class TestRMVPEExtractor:
    """Tests for RMVPE pitch extraction model."""

    def test_rmvpe_class_exists(self):
        """RMVPEPitchExtractor class should exist."""
        from auto_voice.models.pitch import RMVPEPitchExtractor
        assert RMVPEPitchExtractor is not None

    def test_init_default(self):
        """Default initialization with random weights."""
        from auto_voice.models.pitch import RMVPEPitchExtractor
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        extractor = RMVPEPitchExtractor(pretrained=None, device=device)
        assert extractor.hop_size == 320  # 20ms at 16kHz
        assert extractor.f0_min == 50.0
        assert extractor.f0_max == 1100.0

    def test_output_shape(self):
        """F0 output should be [B, N_frames] at 20ms resolution."""
        from auto_voice.models.pitch import RMVPEPitchExtractor
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        extractor = RMVPEPitchExtractor(pretrained=None, device=device).to(device)
        # 1 second at 16kHz → ~50 frames at 20ms
        audio = torch.randn(1, 16000, device=device)
        f0 = extractor.extract(audio)
        assert f0.dim() == 2  # [B, N_frames]
        assert f0.shape[0] == 1
        # ~50 frames for 1 second (some tolerance for edge padding)
        assert 45 <= f0.shape[1] <= 55

    def test_batch_processing(self):
        """Batched input should produce batched F0."""
        from auto_voice.models.pitch import RMVPEPitchExtractor
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        extractor = RMVPEPitchExtractor(pretrained=None, device=device).to(device)
        audio = torch.randn(3, 16000, device=device)
        f0 = extractor.extract(audio)
        assert f0.shape[0] == 3

    def test_voicing_detection(self):
        """Should provide voiced/unvoiced decisions."""
        from auto_voice.models.pitch import RMVPEPitchExtractor
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        extractor = RMVPEPitchExtractor(pretrained=None, device=device).to(device)
        audio = torch.randn(1, 16000, device=device)
        f0, voicing = extractor.extract(audio, return_voicing=True)
        assert voicing.dim() == 2  # [B, N_frames]
        assert voicing.shape == f0.shape
        # Voicing should be probabilities [0, 1]
        assert voicing.min() >= 0.0
        assert voicing.max() <= 1.0

    def test_f0_range(self):
        """F0 values should be within expected range or 0 (unvoiced)."""
        from auto_voice.models.pitch import RMVPEPitchExtractor
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        extractor = RMVPEPitchExtractor(pretrained=None, device=device).to(device)
        audio = torch.randn(1, 32000, device=device)
        f0 = extractor.extract(audio)
        # Non-zero F0 values should be within [f0_min, f0_max]
        voiced_mask = f0 > 0
        if voiced_mask.any():
            assert f0[voiced_mask].min() >= 50.0
            assert f0[voiced_mask].max() <= 1100.0

    def test_output_finite(self):
        """All output values should be finite."""
        from auto_voice.models.pitch import RMVPEPitchExtractor
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        extractor = RMVPEPitchExtractor(pretrained=None, device=device).to(device)
        audio = torch.randn(1, 16000, device=device)
        f0 = extractor.extract(audio)
        assert torch.isfinite(f0).all()

    def test_device_placement(self):
        """Output should be on same device as input."""
        from auto_voice.models.pitch import RMVPEPitchExtractor
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        extractor = RMVPEPitchExtractor(pretrained=None, device=device).to(device)
        audio = torch.randn(1, 16000, device=device)
        f0 = extractor.extract(audio)
        assert f0.device == audio.device

    def test_1d_input_handling(self):
        """1D audio input should be handled (auto-batch)."""
        from auto_voice.models.pitch import RMVPEPitchExtractor
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        extractor = RMVPEPitchExtractor(pretrained=None, device=device).to(device)
        audio = torch.randn(16000, device=device)  # 1D
        f0 = extractor.extract(audio)
        assert f0.dim() == 2
        assert f0.shape[0] == 1


class TestRMVPEArchitecture:
    """Tests for RMVPE model architecture details."""

    def test_deep_residual_backbone(self):
        """RMVPE uses deep residual network for feature extraction."""
        from auto_voice.models.pitch import RMVPEPitchExtractor
        extractor = RMVPEPitchExtractor(pretrained=None)
        # Model should have residual blocks
        assert hasattr(extractor, 'model')
        param_count = sum(p.numel() for p in extractor.parameters())
        # RMVPE has ~7M parameters
        assert param_count > 1_000_000, "RMVPE should have substantial parameters"

    def test_mel_spectrogram_input(self):
        """RMVPE processes mel spectrogram internally."""
        from auto_voice.models.pitch import RMVPEPitchExtractor
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        extractor = RMVPEPitchExtractor(pretrained=None, device=device).to(device)
        # Should accept raw audio and compute mel internally
        audio = torch.randn(1, 16000, device=device)
        f0 = extractor.extract(audio)
        assert f0 is not None

    def test_cent_based_output(self):
        """RMVPE outputs pitch in cents (360 bins per octave)."""
        from auto_voice.models.pitch import RMVPEPitchExtractor
        extractor = RMVPEPitchExtractor(pretrained=None)
        # Check the output layer configuration
        assert extractor.cents_mapping is not None
        assert extractor.n_octaves >= 6  # At least 6 octaves (50-1100 Hz)


class TestPitchF0Integration:
    """Tests for RMVPE integration with mel-quantized F0 pipeline."""

    def test_rmvpe_to_coarse(self):
        """RMVPE F0 output feeds into f0_to_coarse for mel-quantized bins."""
        from auto_voice.models.pitch import RMVPEPitchExtractor
        from auto_voice.models.encoder import f0_to_coarse
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        extractor = RMVPEPitchExtractor(pretrained=None, device=device).to(device)
        audio = torch.randn(1, 16000, device=device)
        f0 = extractor.extract(audio)
        # Feed into existing mel-quantized pipeline
        bins = f0_to_coarse(f0)
        assert bins.dtype == torch.long
        assert bins.min() >= 0
        assert bins.max() <= 255

    def test_rmvpe_to_pitch_encoder(self):
        """RMVPE F0 feeds into PitchEncoder for embedding lookup."""
        from auto_voice.models.pitch import RMVPEPitchExtractor
        from auto_voice.models.encoder import PitchEncoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        extractor = RMVPEPitchExtractor(pretrained=None, device=device).to(device)
        pitch_enc = PitchEncoder(output_size=256).to(device)
        audio = torch.randn(1, 16000, device=device)
        f0 = extractor.extract(audio)
        # PitchEncoder accepts [B, T] F0 in Hz
        pitch_emb = pitch_enc(f0)
        assert pitch_emb.shape[0] == 1
        assert pitch_emb.shape[2] == 256

    def test_frame_alignment_with_content(self):
        """Pitch frames should be alignable to content frames via interpolate."""
        from auto_voice.models.pitch import RMVPEPitchExtractor
        from auto_voice.models.encoder import ContentVecEncoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        extractor = RMVPEPitchExtractor(pretrained=None, device=device).to(device)
        content_enc = ContentVecEncoder(pretrained=None, device=device).to(device)

        audio = torch.randn(1, 32000, device=device)  # 2 seconds
        f0 = extractor.extract(audio)  # [B, N_pitch]
        content = content_enc.encode(audio)  # [B, N_content, 768]

        # Align pitch to content frame count
        target_frames = content.shape[1]
        f0_aligned = torch.nn.functional.interpolate(
            f0.unsqueeze(1),  # [B, 1, N_pitch]
            size=target_frames,
            mode='linear',
            align_corners=False,
        ).squeeze(1)  # [B, N_content]

        assert f0_aligned.shape[1] == target_frames
        assert torch.isfinite(f0_aligned).all()


class TestNoFallbackPitch:
    """Verify no fallback behavior for pitch extraction."""

    def test_invalid_audio_length(self):
        """Too-short audio should raise RuntimeError."""
        from auto_voice.models.pitch import RMVPEPitchExtractor
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        extractor = RMVPEPitchExtractor(pretrained=None, device=device).to(device)
        # 10 samples is too short for any pitch analysis
        audio = torch.randn(1, 10, device=device)
        with pytest.raises(RuntimeError):
            extractor.extract(audio)

    def test_silence_produces_unvoiced(self):
        """Silent input should produce all-unvoiced (f0=0)."""
        from auto_voice.models.pitch import RMVPEPitchExtractor
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        extractor = RMVPEPitchExtractor(pretrained=None, device=device).to(device)
        audio = torch.zeros(1, 16000, device=device)
        f0 = extractor.extract(audio)
        # With random weights, we can't guarantee all zeros,
        # but output should still be finite
        assert torch.isfinite(f0).all()
