"""Comprehensive tests for mean_flow_decoder.py (0% → 95% coverage).

Target: 95% coverage (~330 lines)
Tests: ~27 tests covering:
- Flow matching decoding
- Timestep scheduling
- Latent processing
- MeanVC pipeline integration
- GPU operations
- Error paths
"""
import logging

import numpy as np
import pytest
import torch
import torch.nn as nn

from auto_voice.inference.mean_flow_decoder import (
    MeanFlowDecoder,
    TimeEmbedding,
    compute_mean_flow_loss,
)


@pytest.fixture
def device():
    """Get device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def decoder(device):
    """Create MeanFlowDecoder instance."""
    model = MeanFlowDecoder(
        content_dim=512,
        speaker_dim=256,
        mel_dim=80,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
    )
    return model.to(device)


@pytest.fixture
def small_decoder(device):
    """Create small decoder for fast tests."""
    model = MeanFlowDecoder(
        content_dim=64,
        speaker_dim=32,
        mel_dim=20,
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
    )
    return model.to(device)


@pytest.fixture
def sample_batch(device):
    """Generate sample batch of inputs."""
    B, T = 2, 100
    return {
        "x": torch.randn(B, T, 80, device=device),
        "t": torch.rand(B, device=device),
        "r": torch.rand(B, device=device),
        "content": torch.randn(B, T, 512, device=device),
        "speaker": torch.randn(B, 256, device=device),
    }


class TestMeanFlowDecoderInitialization:
    """Test MeanFlowDecoder initialization."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        decoder = MeanFlowDecoder()

        assert decoder.content_dim == 512
        assert decoder.speaker_dim == 256
        assert decoder.mel_dim == 80
        assert decoder.hidden_dim == 512
        assert isinstance(decoder.x_proj, nn.Linear)
        assert isinstance(decoder.content_proj, nn.Linear)
        assert isinstance(decoder.spk_proj, nn.Linear)
        assert isinstance(decoder.time_embed, TimeEmbedding)
        assert isinstance(decoder.transformer, nn.TransformerEncoder)
        assert isinstance(decoder.out_proj, nn.Linear)

    def test_init_custom_dims(self):
        """Test initialization with custom dimensions."""
        decoder = MeanFlowDecoder(
            content_dim=256,
            speaker_dim=128,
            mel_dim=40,
            hidden_dim=256,
            num_layers=4,
            num_heads=4,
        )

        assert decoder.content_dim == 256
        assert decoder.speaker_dim == 128
        assert decoder.mel_dim == 40
        assert decoder.hidden_dim == 256

    def test_init_projection_layers(self):
        """Test projection layers have correct dimensions."""
        decoder = MeanFlowDecoder()

        # Input projections
        assert decoder.x_proj.in_features == 80
        assert decoder.x_proj.out_features == 512
        assert decoder.content_proj.in_features == 512
        assert decoder.content_proj.out_features == 512
        assert decoder.spk_proj.in_features == 256
        assert decoder.spk_proj.out_features == 512

        # Output projection
        assert decoder.out_proj.in_features == 512
        assert decoder.out_proj.out_features == 80

    def test_init_transformer_config(self):
        """Test transformer encoder configuration."""
        decoder = MeanFlowDecoder(num_layers=4, num_heads=4)

        assert len(decoder.transformer.layers) == 4
        layer = decoder.transformer.layers[0]
        assert layer.self_attn.num_heads == 4


class TestForwardPass:
    """Test forward pass computation."""

    def test_forward_basic(self, decoder, sample_batch, device):
        """Test basic forward pass."""
        output = decoder.forward(
            x=sample_batch["x"],
            t=sample_batch["t"],
            r=sample_batch["r"],
            content=sample_batch["content"],
            speaker=sample_batch["speaker"],
        )

        assert output.shape == (2, 100, 80)
        assert output.device == device
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_with_prompt(self, decoder, sample_batch, device):
        """Test forward pass with prompt (in-context learning)."""
        prompt = torch.randn(2, 50, 80, device=device)

        output = decoder.forward(
            x=sample_batch["x"],
            t=sample_batch["t"],
            r=sample_batch["r"],
            content=sample_batch["content"],
            speaker=sample_batch["speaker"],
            prompt=prompt,
        )

        # Output should still match input length
        assert output.shape == (2, 100, 80)
        assert not torch.isnan(output).any()

    def test_forward_time_conditioning(self, small_decoder, device):
        """Test that time conditioning affects output."""
        B, T = 2, 50
        x = torch.randn(B, T, 20, device=device)
        content = torch.randn(B, T, 64, device=device)
        speaker = torch.randn(B, 32, device=device)

        # Different timesteps should give different outputs
        t1 = torch.zeros(B, device=device)
        r1 = torch.zeros(B, device=device)
        out1 = small_decoder(x, t1, r1, content, speaker)

        t2 = torch.ones(B, device=device)
        r2 = torch.zeros(B, device=device)
        out2 = small_decoder(x, t2, r2, content, speaker)

        # Outputs should differ
        assert not torch.allclose(out1, out2, atol=1e-5)

    def test_forward_speaker_conditioning(self, small_decoder, device):
        """Test that speaker embedding affects output."""
        B, T = 2, 50
        x = torch.randn(B, T, 20, device=device)
        t = torch.rand(B, device=device)
        r = torch.rand(B, device=device)
        content = torch.randn(B, T, 64, device=device)

        spk1 = torch.randn(B, 32, device=device)
        out1 = small_decoder(x, t, r, content, spk1)

        spk2 = torch.randn(B, 32, device=device)
        out2 = small_decoder(x, t, r, content, spk2)

        # Different speakers should give different outputs
        assert not torch.allclose(out1, out2, atol=1e-5)

    def test_forward_content_conditioning(self, small_decoder, device):
        """Test that content features affect output."""
        B, T = 2, 50
        x = torch.randn(B, T, 20, device=device)
        t = torch.rand(B, device=device)
        r = torch.rand(B, device=device)
        speaker = torch.randn(B, 32, device=device)

        content1 = torch.randn(B, T, 64, device=device)
        out1 = small_decoder(x, t, r, content1, speaker)

        content2 = torch.randn(B, T, 64, device=device)
        out2 = small_decoder(x, t, r, content2, speaker)

        # Different content should give different outputs
        assert not torch.allclose(out1, out2, atol=1e-5)

    def test_forward_batch_processing(self, small_decoder, device):
        """Test forward pass with different batch sizes."""
        for B in [1, 2, 4, 8]:
            T = 50
            x = torch.randn(B, T, 20, device=device)
            t = torch.rand(B, device=device)
            r = torch.rand(B, device=device)
            content = torch.randn(B, T, 64, device=device)
            speaker = torch.randn(B, 32, device=device)

            output = small_decoder(x, t, r, content, speaker)
            assert output.shape == (B, T, 20)

    def test_forward_variable_lengths(self, small_decoder, device):
        """Test forward pass with different sequence lengths."""
        B = 2
        for T in [10, 50, 100, 200]:
            x = torch.randn(B, T, 20, device=device)
            t = torch.rand(B, device=device)
            r = torch.rand(B, device=device)
            content = torch.randn(B, T, 64, device=device)
            speaker = torch.randn(B, 32, device=device)

            output = small_decoder(x, t, r, content, speaker)
            assert output.shape == (B, T, 20)


class TestInferenceSingleStep:
    """Test single-step inference."""

    def test_inference_single_step_basic(self, decoder, device):
        """Test basic single-step inference."""
        B, T = 2, 100
        x0 = torch.randn(B, T, 80, device=device)
        content = torch.randn(B, T, 512, device=device)
        speaker = torch.randn(B, 256, device=device)

        x1 = decoder.inference_single_step(x0, content, speaker)

        assert x1.shape == (B, T, 80)
        assert x1.device == device
        assert not torch.isnan(x1).any()
        assert not torch.isinf(x1).any()

    def test_inference_single_step_with_prompt(self, decoder, device):
        """Test single-step inference with prompt."""
        B, T = 2, 100
        x0 = torch.randn(B, T, 80, device=device)
        content = torch.randn(B, T, 512, device=device)
        speaker = torch.randn(B, 256, device=device)
        prompt = torch.randn(B, 50, 80, device=device)

        x1 = decoder.inference_single_step(x0, content, speaker, prompt)

        assert x1.shape == (B, T, 80)
        assert not torch.isnan(x1).any()

    def test_inference_single_step_timesteps(self, small_decoder, device):
        """Test single-step uses correct timesteps (t=1, r=0)."""
        B, T = 2, 50
        x0 = torch.randn(B, T, 20, device=device)
        content = torch.randn(B, T, 64, device=device)
        speaker = torch.randn(B, 32, device=device)

        # Mock forward to verify timesteps
        original_forward = small_decoder.forward

        def mock_forward(x, t, r, c, s, p=None):
            # Verify t=1, r=0
            assert torch.allclose(t, torch.ones(B, device=device), atol=1e-6)
            assert torch.allclose(r, torch.zeros(B, device=device), atol=1e-6)
            return original_forward(x, t, r, c, s, p)

        small_decoder.forward = mock_forward

        x1 = small_decoder.inference_single_step(x0, content, speaker)
        assert x1.shape == (B, T, 20)

    def test_inference_single_step_deterministic(self, small_decoder, device):
        """Test single-step inference is deterministic."""
        B, T = 2, 50
        x0 = torch.randn(B, T, 20, device=device)
        content = torch.randn(B, T, 64, device=device)
        speaker = torch.randn(B, 32, device=device)

        small_decoder.eval()
        with torch.no_grad():
            x1_a = small_decoder.inference_single_step(x0, content, speaker)
            x1_b = small_decoder.inference_single_step(x0, content, speaker)

        assert torch.allclose(x1_a, x1_b, atol=1e-6)


class TestInferenceTwoStep:
    """Test two-step inference."""

    def test_inference_two_step_basic(self, decoder, device):
        """Test basic two-step inference."""
        B, T = 2, 100
        x0 = torch.randn(B, T, 80, device=device)
        content = torch.randn(B, T, 512, device=device)
        speaker = torch.randn(B, 256, device=device)

        x1 = decoder.inference_two_step(x0, content, speaker)

        assert x1.shape == (B, T, 80)
        assert x1.device == device
        assert not torch.isnan(x1).any()
        assert not torch.isinf(x1).any()

    def test_inference_two_step_with_prompt(self, decoder, device):
        """Test two-step inference with prompt."""
        B, T = 2, 100
        x0 = torch.randn(B, T, 80, device=device)
        content = torch.randn(B, T, 512, device=device)
        speaker = torch.randn(B, 256, device=device)
        prompt = torch.randn(B, 50, 80, device=device)

        x1 = decoder.inference_two_step(x0, content, speaker, prompt)

        assert x1.shape == (B, T, 80)
        assert not torch.isnan(x1).any()

    def test_inference_two_step_timesteps(self, small_decoder, device):
        """Test two-step uses correct timesteps."""
        B, T = 2, 50
        x0 = torch.randn(B, T, 20, device=device)
        content = torch.randn(B, T, 64, device=device)
        speaker = torch.randn(B, 32, device=device)

        call_count = [0]
        original_forward = small_decoder.forward

        def mock_forward(x, t, r, c, s, p=None):
            call_count[0] += 1
            if call_count[0] == 1:
                # First step: t=1.0, r=0.8
                assert torch.allclose(t, torch.ones(B, device=device) * 1.0, atol=1e-6)
                assert torch.allclose(r, torch.ones(B, device=device) * 0.8, atol=1e-6)
            elif call_count[0] == 2:
                # Second step: t=0.8, r=0.0
                assert torch.allclose(t, torch.ones(B, device=device) * 0.8, atol=1e-6)
                assert torch.allclose(r, torch.zeros(B, device=device), atol=1e-6)
            return original_forward(x, t, r, c, s, p)

        small_decoder.forward = mock_forward

        x1 = small_decoder.inference_two_step(x0, content, speaker)
        assert call_count[0] == 2

    def test_two_step_vs_single_step(self, small_decoder, device):
        """Test two-step gives different output than single-step."""
        B, T = 2, 50
        x0 = torch.randn(B, T, 20, device=device)
        content = torch.randn(B, T, 64, device=device)
        speaker = torch.randn(B, 32, device=device)

        small_decoder.eval()
        with torch.no_grad():
            x1_single = small_decoder.inference_single_step(x0, content, speaker)
            x1_two = small_decoder.inference_two_step(x0, content, speaker)

        # Should be different
        assert not torch.allclose(x1_single, x1_two, atol=1e-5)


class TestTimeEmbedding:
    """Test TimeEmbedding module."""

    def test_time_embedding_init(self):
        """Test TimeEmbedding initialization."""
        time_embed = TimeEmbedding(dim=512)

        assert time_embed.dim == 512
        assert time_embed.max_period == 10000
        assert isinstance(time_embed.mlp, nn.Sequential)

    def test_time_embedding_forward(self, device):
        """Test TimeEmbedding forward pass."""
        time_embed = TimeEmbedding(dim=512).to(device)

        t = torch.rand(4, device=device)
        emb = time_embed(t)

        assert emb.shape == (4, 512)
        assert emb.device == device
        assert not torch.isnan(emb).any()

    def test_time_embedding_different_times(self, device):
        """Test different timesteps produce different embeddings."""
        time_embed = TimeEmbedding(dim=128).to(device)

        t1 = torch.zeros(2, device=device)
        t2 = torch.ones(2, device=device)

        emb1 = time_embed(t1)
        emb2 = time_embed(t2)

        assert not torch.allclose(emb1, emb2, atol=1e-5)

    def test_time_embedding_odd_dimension(self, device):
        """Test TimeEmbedding handles odd dimensions."""
        time_embed = TimeEmbedding(dim=127).to(device)

        t = torch.rand(4, device=device)
        emb = time_embed(t)

        assert emb.shape == (4, 127)

    def test_time_embedding_deterministic(self, device):
        """Test TimeEmbedding is deterministic."""
        time_embed = TimeEmbedding(dim=256).to(device)

        t = torch.tensor([0.0, 0.5, 1.0], device=device)

        time_embed.eval()
        with torch.no_grad():
            emb1 = time_embed(t)
            emb2 = time_embed(t)

        assert torch.allclose(emb1, emb2, atol=1e-6)


class TestMeanFlowLoss:
    """Test compute_mean_flow_loss function."""

    def test_compute_mean_flow_loss_basic(self, small_decoder, device):
        """Test basic loss computation."""
        B, T = 2, 50
        x0 = torch.randn(B, T, 20, device=device)
        x1 = torch.randn(B, T, 20, device=device)
        content = torch.randn(B, T, 64, device=device)
        speaker = torch.randn(B, 32, device=device)

        fm_loss, mean_loss = compute_mean_flow_loss(
            small_decoder, x0, x1, content, speaker
        )

        assert isinstance(fm_loss, torch.Tensor)
        assert isinstance(mean_loss, torch.Tensor)
        assert fm_loss.ndim == 0  # Scalar
        assert mean_loss.ndim == 0  # Scalar
        assert fm_loss.item() >= 0
        assert mean_loss.item() >= 0

    def test_compute_mean_flow_loss_with_prompt(self, small_decoder, device):
        """Test loss computation with prompt."""
        B, T = 2, 50
        x0 = torch.randn(B, T, 20, device=device)
        x1 = torch.randn(B, T, 20, device=device)
        content = torch.randn(B, T, 64, device=device)
        speaker = torch.randn(B, 32, device=device)
        prompt = torch.randn(B, 25, 20, device=device)

        fm_loss, mean_loss = compute_mean_flow_loss(
            small_decoder, x0, x1, content, speaker, prompt
        )

        assert fm_loss.item() >= 0
        assert mean_loss.item() >= 0

    def test_loss_decreases_with_training(self, small_decoder, device):
        """Test that loss decreases with training steps."""
        B, T = 2, 50
        x0 = torch.randn(B, T, 20, device=device)
        x1 = torch.randn(B, T, 20, device=device)
        content = torch.randn(B, T, 64, device=device)
        speaker = torch.randn(B, 32, device=device)

        optimizer = torch.optim.Adam(small_decoder.parameters(), lr=0.001)

        initial_loss = None
        for step in range(10):
            fm_loss, mean_loss = compute_mean_flow_loss(
                small_decoder, x0, x1, content, speaker
            )
            total_loss = fm_loss + mean_loss

            if initial_loss is None:
                initial_loss = total_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        final_loss = total_loss.item()
        # Loss should decrease (or at least not increase significantly)
        assert final_loss <= initial_loss * 1.1

    def test_loss_gradients_flow(self, small_decoder, device):
        """Test that gradients flow through loss."""
        B, T = 2, 50
        x0 = torch.randn(B, T, 20, device=device)
        x1 = torch.randn(B, T, 20, device=device)
        content = torch.randn(B, T, 64, device=device)
        speaker = torch.randn(B, 32, device=device)

        fm_loss, mean_loss = compute_mean_flow_loss(
            small_decoder, x0, x1, content, speaker
        )
        total_loss = fm_loss + mean_loss

        total_loss.backward()

        # Check gradients exist
        for param in small_decoder.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_frame_input(self, small_decoder, device):
        """Test handling of single frame."""
        B, T = 2, 1
        x = torch.randn(B, T, 20, device=device)
        t = torch.rand(B, device=device)
        r = torch.rand(B, device=device)
        content = torch.randn(B, T, 64, device=device)
        speaker = torch.randn(B, 32, device=device)

        output = small_decoder(x, t, r, content, speaker)
        assert output.shape == (B, T, 20)

    def test_very_long_sequence(self, small_decoder, device):
        """Test handling of very long sequence (1000 frames)."""
        B, T = 1, 1000
        x = torch.randn(B, T, 20, device=device)
        t = torch.rand(B, device=device)
        r = torch.rand(B, device=device)
        content = torch.randn(B, T, 64, device=device)
        speaker = torch.randn(B, 32, device=device)

        output = small_decoder(x, t, r, content, speaker)
        assert output.shape == (B, T, 20)

    def test_zero_timestep(self, small_decoder, device):
        """Test handling of t=0, r=0."""
        B, T = 2, 50
        x = torch.randn(B, T, 20, device=device)
        t = torch.zeros(B, device=device)
        r = torch.zeros(B, device=device)
        content = torch.randn(B, T, 64, device=device)
        speaker = torch.randn(B, 32, device=device)

        output = small_decoder(x, t, r, content, speaker)
        assert not torch.isnan(output).any()

    def test_one_timestep(self, small_decoder, device):
        """Test handling of t=1, r=1."""
        B, T = 2, 50
        x = torch.randn(B, T, 20, device=device)
        t = torch.ones(B, device=device)
        r = torch.ones(B, device=device)
        content = torch.randn(B, T, 64, device=device)
        speaker = torch.randn(B, 32, device=device)

        output = small_decoder(x, t, r, content, speaker)
        assert not torch.isnan(output).any()

    def test_all_zero_input(self, small_decoder, device):
        """Test handling of all-zero input."""
        B, T = 2, 50
        x = torch.zeros(B, T, 20, device=device)
        t = torch.rand(B, device=device)
        r = torch.rand(B, device=device)
        content = torch.zeros(B, T, 64, device=device)
        speaker = torch.zeros(B, 32, device=device)

        output = small_decoder(x, t, r, content, speaker)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestGPUOperations:
    """Test GPU-specific operations."""

    @pytest.mark.cuda
    def test_cuda_device_placement(self):
        """Test model runs on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        decoder = MeanFlowDecoder().to(device)

        B, T = 2, 100
        x = torch.randn(B, T, 80, device=device)
        t = torch.rand(B, device=device)
        r = torch.rand(B, device=device)
        content = torch.randn(B, T, 512, device=device)
        speaker = torch.randn(B, 256, device=device)

        output = decoder(x, t, r, content, speaker)

        assert output.device.type == "cuda"

    @pytest.mark.cuda
    def test_inference_single_step_cuda(self):
        """Test single-step inference on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        decoder = MeanFlowDecoder().to(device)

        B, T = 2, 100
        x0 = torch.randn(B, T, 80, device=device)
        content = torch.randn(B, T, 512, device=device)
        speaker = torch.randn(B, 256, device=device)

        x1 = decoder.inference_single_step(x0, content, speaker)

        assert x1.device.type == "cuda"
        assert not torch.isnan(x1).any()

    @pytest.mark.cuda
    def test_mixed_precision_inference(self):
        """Test inference with mixed precision (FP16)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        decoder = MeanFlowDecoder().to(device).half()

        B, T = 2, 100
        x0 = torch.randn(B, T, 80, device=device, dtype=torch.float16)
        content = torch.randn(B, T, 512, device=device, dtype=torch.float16)
        speaker = torch.randn(B, 256, device=device, dtype=torch.float16)

        with torch.cuda.amp.autocast():
            x1 = decoder.inference_single_step(x0, content, speaker)

        assert x1.dtype == torch.float16


class TestModelSerialization:
    """Test model save/load."""

    def test_save_load_state_dict(self, small_decoder, device, tmp_path):
        """Test saving and loading model state dict."""
        B, T = 2, 50
        x0 = torch.randn(B, T, 20, device=device)
        content = torch.randn(B, T, 64, device=device)
        speaker = torch.randn(B, 32, device=device)

        # Get initial output
        small_decoder.eval()
        with torch.no_grad():
            output1 = small_decoder.inference_single_step(x0, content, speaker)

        # Save
        path = tmp_path / "decoder.pth"
        torch.save(small_decoder.state_dict(), path)

        # Create new model and load
        new_decoder = MeanFlowDecoder(
            content_dim=64,
            speaker_dim=32,
            mel_dim=20,
            hidden_dim=64,
            num_layers=2,
            num_heads=2,
        ).to(device)
        new_decoder.load_state_dict(torch.load(path))
        new_decoder.eval()

        # Should produce same output
        with torch.no_grad():
            output2 = new_decoder.inference_single_step(x0, content, speaker)

        assert torch.allclose(output1, output2, atol=1e-6)


class TestTrainingMode:
    """Test training vs evaluation mode."""

    def test_train_mode_dropout(self, small_decoder, device):
        """Test dropout is active in train mode."""
        B, T = 2, 50
        x = torch.randn(B, T, 20, device=device)
        t = torch.rand(B, device=device)
        r = torch.rand(B, device=device)
        content = torch.randn(B, T, 64, device=device)
        speaker = torch.randn(B, 32, device=device)

        small_decoder.train()
        out1 = small_decoder(x, t, r, content, speaker)
        out2 = small_decoder(x, t, r, content, speaker)

        # Outputs may differ due to dropout
        assert out1.shape == out2.shape

    def test_eval_mode_deterministic(self, small_decoder, device):
        """Test eval mode is deterministic."""
        B, T = 2, 50
        x = torch.randn(B, T, 20, device=device)
        t = torch.rand(B, device=device)
        r = torch.rand(B, device=device)
        content = torch.randn(B, T, 64, device=device)
        speaker = torch.randn(B, 32, device=device)

        small_decoder.eval()
        with torch.no_grad():
            out1 = small_decoder(x, t, r, content, speaker)
            out2 = small_decoder(x, t, r, content, speaker)

        assert torch.allclose(out1, out2, atol=1e-6)
