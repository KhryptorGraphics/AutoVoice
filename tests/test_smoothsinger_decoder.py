"""Tests for SmoothSinger decoder (CUTTING_EDGE_PIPELINE component).

Validates the SmoothSinger-inspired decoder with:
- Multi-resolution non-sequential U-Net processing
- Reference-guided dual-branch architecture
- Vocoder-free codec diffusion output
- Integration with HQ-SVC super-resolution concepts

Reference: SmoothSinger (Jun 2025), HQ-SVC (AAAI 2026)
"""
import pytest
import numpy as np
import torch


class TestSmoothSingerDecoder:
    """Tests for SmoothSinger multi-resolution decoder."""

    def test_class_exists(self):
        """SmoothSingerDecoder class should exist."""
        from auto_voice.models.smoothsinger_decoder import SmoothSingerDecoder
        assert SmoothSingerDecoder is not None

    def test_init_default(self):
        """Default initialization with expected dimensions."""
        from auto_voice.models.smoothsinger_decoder import SmoothSingerDecoder
        decoder = SmoothSingerDecoder()
        assert decoder.content_dim == 768  # Whisper/ContentVec
        assert decoder.style_dim == 192  # CAMPPlus style encoder
        assert decoder.f0_bins == 256  # RMVPE pitch bins
        assert decoder.hidden_dim == 512
        assert decoder.n_resolutions == 3  # Multi-resolution levels

    def test_forward_shape(self):
        """Decoder produces correct output shape."""
        from auto_voice.models.smoothsinger_decoder import SmoothSingerDecoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = SmoothSingerDecoder(device=device).to(device)

        B, T = 1, 50
        content = torch.randn(B, T, 768, device=device)
        f0 = torch.randn(B, T, 256, device=device)
        style = torch.randn(B, 192, device=device)

        output = decoder(content, f0, style)
        # Output should be mel spectrogram for vocoder (or codec tokens)
        assert output.shape[0] == B
        assert output.shape[2] == T  # Temporal dimension preserved

    def test_batch_processing(self):
        """Batched input produces batched output."""
        from auto_voice.models.smoothsinger_decoder import SmoothSingerDecoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = SmoothSingerDecoder(device=device).to(device)

        B, T = 4, 40
        content = torch.randn(B, T, 768, device=device)
        f0 = torch.randn(B, T, 256, device=device)
        style = torch.randn(B, 192, device=device)

        output = decoder(content, f0, style)
        assert output.shape[0] == B

    def test_output_finite(self):
        """All outputs should be finite (no NaN/Inf)."""
        from auto_voice.models.smoothsinger_decoder import SmoothSingerDecoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = SmoothSingerDecoder(device=device).to(device)

        B, T = 1, 30
        content = torch.randn(B, T, 768, device=device)
        f0 = torch.randn(B, T, 256, device=device)
        style = torch.randn(B, 192, device=device)

        output = decoder(content, f0, style)
        assert torch.isfinite(output).all(), "Output contains NaN or Inf"

    def test_device_placement(self):
        """Output on same device as input."""
        from auto_voice.models.smoothsinger_decoder import SmoothSingerDecoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = SmoothSingerDecoder(device=device).to(device)

        B, T = 1, 30
        content = torch.randn(B, T, 768, device=device)
        f0 = torch.randn(B, T, 256, device=device)
        style = torch.randn(B, 192, device=device)

        output = decoder(content, f0, style)
        assert output.device == content.device


class TestMultiResolutionUNet:
    """Tests for multi-resolution non-sequential U-Net processing."""

    def test_multi_res_block_exists(self):
        """MultiResolutionBlock class should exist."""
        from auto_voice.models.smoothsinger_decoder import MultiResolutionBlock
        assert MultiResolutionBlock is not None

    def test_multi_res_output_shape(self):
        """MultiResolutionBlock preserves shape."""
        from auto_voice.models.smoothsinger_decoder import MultiResolutionBlock
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        block = MultiResolutionBlock(channels=256, n_resolutions=3).to(device)

        x = torch.randn(1, 256, 50, device=device)
        y = block(x)
        assert y.shape == x.shape

    def test_multi_res_different_resolutions(self):
        """Block should process at multiple temporal resolutions."""
        from auto_voice.models.smoothsinger_decoder import MultiResolutionBlock
        block = MultiResolutionBlock(channels=256, n_resolutions=3)
        # Should have conv layers for different resolutions
        assert hasattr(block, 'resolution_convs') or hasattr(block, 'branches')
        # Should have at least 3 resolution branches
        if hasattr(block, 'resolution_convs'):
            assert len(block.resolution_convs) >= 3
        elif hasattr(block, 'branches'):
            assert len(block.branches) >= 3

    def test_non_sequential_processing(self):
        """Verify non-sequential (parallel) resolution processing."""
        from auto_voice.models.smoothsinger_decoder import MultiResolutionBlock
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        block = MultiResolutionBlock(channels=256, n_resolutions=3).to(device)

        x = torch.randn(1, 256, 50, device=device)
        # Forward should work without errors (parallel processing)
        y = block(x)
        assert torch.isfinite(y).all()


class TestDualBranchProcessing:
    """Tests for reference-guided dual-branch architecture."""

    def test_dual_branch_exists(self):
        """DualBranchFusion class should exist."""
        from auto_voice.models.smoothsinger_decoder import DualBranchFusion
        assert DualBranchFusion is not None

    def test_dual_branch_fusion_shape(self):
        """DualBranchFusion combines content and style branches."""
        from auto_voice.models.smoothsinger_decoder import DualBranchFusion
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        fusion = DualBranchFusion(content_dim=512, style_dim=192).to(device)

        content_features = torch.randn(1, 512, 50, device=device)
        style_embedding = torch.randn(1, 192, device=device)

        fused = fusion(content_features, style_embedding)
        # Fused output should have same temporal dim as content
        assert fused.shape[0] == 1
        assert fused.shape[2] == 50

    def test_style_affects_output(self):
        """Different style embeddings should produce different outputs."""
        from auto_voice.models.smoothsinger_decoder import DualBranchFusion
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        fusion = DualBranchFusion(content_dim=512, style_dim=192).to(device)

        content = torch.randn(1, 512, 50, device=device)
        style_a = torch.randn(1, 192, device=device)
        style_b = torch.randn(1, 192, device=device)

        out_a = fusion(content, style_a)
        out_b = fusion(content, style_b)

        assert not torch.allclose(out_a, out_b, atol=1e-3)

    def test_reference_guided_conditioning(self):
        """Decoder should accept optional reference audio features."""
        from auto_voice.models.smoothsinger_decoder import SmoothSingerDecoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = SmoothSingerDecoder(device=device).to(device)

        B, T = 1, 50
        content = torch.randn(B, T, 768, device=device)
        f0 = torch.randn(B, T, 256, device=device)
        style = torch.randn(B, 192, device=device)
        # Optional reference mel for dual-branch guidance
        reference_mel = torch.randn(B, 128, 100, device=device)

        # Should accept reference_mel as optional parameter
        output = decoder(content, f0, style, reference_mel=reference_mel)
        assert torch.isfinite(output).all()


class TestVocoderFreeCodec:
    """Tests for vocoder-free codec diffusion output."""

    def test_codec_output_mode(self):
        """Decoder should support codec token output mode."""
        from auto_voice.models.smoothsinger_decoder import SmoothSingerDecoder
        decoder = SmoothSingerDecoder(output_mode='codec')
        assert decoder.output_mode == 'codec'

    def test_mel_output_mode(self):
        """Decoder should support mel spectrogram output mode."""
        from auto_voice.models.smoothsinger_decoder import SmoothSingerDecoder
        decoder = SmoothSingerDecoder(output_mode='mel')
        assert decoder.output_mode == 'mel'

    def test_codec_output_shape(self):
        """Codec output should have appropriate shape."""
        from auto_voice.models.smoothsinger_decoder import SmoothSingerDecoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = SmoothSingerDecoder(output_mode='codec', device=device).to(device)

        B, T = 1, 50
        content = torch.randn(B, T, 768, device=device)
        f0 = torch.randn(B, T, 256, device=device)
        style = torch.randn(B, 192, device=device)

        output = decoder(content, f0, style)
        # Codec output: [B, n_codebooks, T] or similar structure
        assert output.dim() == 3
        assert output.shape[0] == B

    def test_super_resolution_support(self):
        """Decoder should support HQ-SVC style super-resolution (16->44.1kHz)."""
        from auto_voice.models.smoothsinger_decoder import SmoothSingerDecoder
        decoder = SmoothSingerDecoder(enable_super_resolution=True)
        assert decoder.enable_super_resolution is True


class TestDiffusionSampling:
    """Tests for diffusion-based mel/codec generation."""

    def test_single_step_inference(self):
        """Should support single-step inference (consistency-like)."""
        from auto_voice.models.smoothsinger_decoder import SmoothSingerDecoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = SmoothSingerDecoder(device=device).to(device)

        B, T = 1, 30
        content = torch.randn(B, T, 768, device=device)
        f0 = torch.randn(B, T, 256, device=device)
        style = torch.randn(B, 192, device=device)

        output = decoder.infer(content, f0, style, n_steps=1)
        assert torch.isfinite(output).all()

    def test_multi_step_inference(self):
        """Should support multi-step diffusion for higher quality."""
        from auto_voice.models.smoothsinger_decoder import SmoothSingerDecoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = SmoothSingerDecoder(device=device).to(device)

        B, T = 1, 30
        content = torch.randn(B, T, 768, device=device)
        f0 = torch.randn(B, T, 256, device=device)
        style = torch.randn(B, 192, device=device)

        output_1 = decoder.infer(content, f0, style, n_steps=1)
        output_4 = decoder.infer(content, f0, style, n_steps=4)

        # Both should be valid
        assert output_1.shape == output_4.shape
        assert torch.isfinite(output_4).all()


class TestSmoothSingerPostProcessor:
    """Tests for the lightweight offline SmoothSinger controls."""

    def test_smooth_f0_contour_preserves_shape(self):
        from auto_voice.models.smoothsinger_decoder import SmoothSingerPostProcessor

        processor = SmoothSingerPostProcessor(smoothness_strength=0.5, smoothing_kernel=7)
        base = np.full(128, 220.0, dtype=np.float32)
        base[::8] += 15.0

        smoothed = processor.smooth_f0_contour(base)

        assert smoothed.shape == base.shape
        assert np.isfinite(smoothed).all()

    def test_transfer_dynamics_preserves_bounds(self):
        from auto_voice.models.smoothsinger_decoder import SmoothSingerPostProcessor

        processor = SmoothSingerPostProcessor(dynamics_mix=0.5)
        audio = np.linspace(-0.2, 0.2, 32000, dtype=np.float32)
        reference = np.sin(np.linspace(0, np.pi * 8, 32000, dtype=np.float32)).astype(np.float32) * 0.6

        transferred = processor.transfer_dynamics(audio, reference)

        assert transferred.shape == audio.shape
        assert np.max(np.abs(transferred)) <= 0.98

    def test_cfg_rate_parameter(self):
        """Should support classifier-free guidance rate."""
        from auto_voice.models.smoothsinger_decoder import SmoothSingerDecoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = SmoothSingerDecoder(device=device).to(device)

        B, T = 1, 30
        content = torch.randn(B, T, 768, device=device)
        f0 = torch.randn(B, T, 256, device=device)
        style = torch.randn(B, 192, device=device)

        # Different CFG rates should produce different outputs
        output_low_cfg = decoder.infer(content, f0, style, cfg_rate=0.3)
        output_high_cfg = decoder.infer(content, f0, style, cfg_rate=0.9)

        # CFG affects output
        assert not torch.allclose(output_low_cfg, output_high_cfg, atol=1e-3)


class TestStyleConditioning:
    """Tests for style/speaker conditioning."""

    def test_style_dimension(self):
        """Style embedding should be 192-dim (CAMPPlus)."""
        from auto_voice.models.smoothsinger_decoder import SmoothSingerDecoder
        decoder = SmoothSingerDecoder()
        assert decoder.style_dim == 192

    def test_different_styles_different_output(self):
        """Different style embeddings should produce different outputs."""
        from auto_voice.models.smoothsinger_decoder import SmoothSingerDecoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = SmoothSingerDecoder(device=device).to(device)

        B, T = 1, 30
        content = torch.randn(B, T, 768, device=device)
        f0 = torch.randn(B, T, 256, device=device)
        style_a = torch.randn(B, 192, device=device)
        style_b = torch.randn(B, 192, device=device)

        out_a = decoder(content, f0, style_a)
        out_b = decoder(content, f0, style_b)

        assert not torch.allclose(out_a, out_b, atol=1e-3)


class TestF0Conditioning:
    """Tests for F0 (pitch) conditioning."""

    def test_f0_dimension(self):
        """F0 should use 256 bins (RMVPE standard)."""
        from auto_voice.models.smoothsinger_decoder import SmoothSingerDecoder
        decoder = SmoothSingerDecoder()
        assert decoder.f0_bins == 256

    def test_f0_affects_output(self):
        """Different F0 contours should produce different outputs."""
        from auto_voice.models.smoothsinger_decoder import SmoothSingerDecoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = SmoothSingerDecoder(device=device).to(device)

        B, T = 1, 30
        content = torch.randn(B, T, 768, device=device)
        f0_low = torch.randn(B, T, 256, device=device) * 0.5
        f0_high = torch.randn(B, T, 256, device=device) * 2.0
        style = torch.randn(B, 192, device=device)

        out_low = decoder(content, f0_low, style)
        out_high = decoder(content, f0_high, style)

        assert not torch.allclose(out_low, out_high, atol=1e-3)


class TestTrainingInterface:
    """Tests for training-compatible interface."""

    def test_compute_loss_exists(self):
        """Decoder should have compute_loss method."""
        from auto_voice.models.smoothsinger_decoder import SmoothSingerDecoder
        decoder = SmoothSingerDecoder()
        assert hasattr(decoder, 'compute_loss')
        assert callable(decoder.compute_loss)

    def test_compute_loss_returns_dict(self):
        """compute_loss should return dict with loss components."""
        from auto_voice.models.smoothsinger_decoder import SmoothSingerDecoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = SmoothSingerDecoder(device=device).to(device)

        B, T = 1, 30
        outputs = torch.randn(B, 128, T, device=device)
        targets = torch.randn(B, 128, T, device=device)

        loss_dict = decoder.compute_loss(outputs, targets)
        assert isinstance(loss_dict, dict)
        assert 'total_loss' in loss_dict
        assert torch.is_tensor(loss_dict['total_loss'])

    def test_lora_injection(self):
        """Should support LoRA injection for fine-tuning."""
        from auto_voice.models.smoothsinger_decoder import SmoothSingerDecoder
        decoder = SmoothSingerDecoder()
        assert hasattr(decoder, 'inject_lora')
        assert callable(decoder.inject_lora)

    def test_lora_state_dict(self):
        """Should support getting/loading LoRA state dict."""
        from auto_voice.models.smoothsinger_decoder import SmoothSingerDecoder
        decoder = SmoothSingerDecoder()
        decoder.inject_lora(rank=8, alpha=16)

        # Should be able to get LoRA parameters
        lora_state = decoder.get_lora_state_dict()
        assert isinstance(lora_state, dict)
        assert len(lora_state) > 0


class TestIntegrationWithSeedVC:
    """Tests for integration with Seed-VC architecture."""

    def test_compatible_content_dim(self):
        """Content dim should match Whisper encoder output (768)."""
        from auto_voice.models.smoothsinger_decoder import SmoothSingerDecoder
        decoder = SmoothSingerDecoder()
        assert decoder.content_dim == 768

    def test_compatible_style_dim(self):
        """Style dim should match CAMPPlus output (192)."""
        from auto_voice.models.smoothsinger_decoder import SmoothSingerDecoder
        decoder = SmoothSingerDecoder()
        assert decoder.style_dim == 192

    def test_output_compatible_with_bigvgan(self):
        """Mel output should be compatible with BigVGAN (128 bands, 44.1kHz)."""
        from auto_voice.models.smoothsinger_decoder import SmoothSingerDecoder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        decoder = SmoothSingerDecoder(output_mode='mel', n_mels=128, device=device).to(device)

        B, T = 1, 50
        content = torch.randn(B, T, 768, device=device)
        f0 = torch.randn(B, T, 256, device=device)
        style = torch.randn(B, 192, device=device)

        mel = decoder(content, f0, style)
        # BigVGAN expects 128-band mel
        assert mel.shape[1] == 128
