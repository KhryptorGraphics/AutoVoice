"""Comprehensive tests for mean_flow_decoder.py module.

Tests cover:
- MeanFlowDecoder initialization and configuration
- Forward pass with various input shapes
- Single-step and two-step inference
- TimeEmbedding module
- compute_mean_flow_loss utility function
- Edge cases and error handling

Target: 70%+ coverage
"""
import pytest
import torch
import torch.nn as nn
import logging
from unittest.mock import patch, MagicMock

from auto_voice.inference.mean_flow_decoder import (
    MeanFlowDecoder,
    TimeEmbedding,
    compute_mean_flow_loss,
)


class TestMeanFlowDecoderInitialization:
    """Test MeanFlowDecoder initialization and configuration."""

    def test_default_initialization(self):
        """Test decoder initialization with default parameters."""
        model = MeanFlowDecoder()

        assert model.content_dim == 512
        assert model.speaker_dim == 256
        assert model.mel_dim == 80
        assert model.hidden_dim == 512

        # Check module existence
        assert isinstance(model.x_proj, nn.Linear)
        assert isinstance(model.content_proj, nn.Linear)
        assert isinstance(model.spk_proj, nn.Linear)
        assert isinstance(model.time_embed, TimeEmbedding)
        assert isinstance(model.transformer, nn.TransformerEncoder)
        assert isinstance(model.out_proj, nn.Linear)

    def test_custom_dimensions(self):
        """Test initialization with custom dimensions."""
        model = MeanFlowDecoder(
            content_dim=768,
            speaker_dim=512,
            mel_dim=128,
            hidden_dim=1024,
            num_layers=12,
            num_heads=16,
        )

        assert model.content_dim == 768
        assert model.speaker_dim == 512
        assert model.mel_dim == 128
        assert model.hidden_dim == 1024

        # Verify projection layers match dimensions
        assert model.x_proj.in_features == 128
        assert model.x_proj.out_features == 1024
        assert model.content_proj.in_features == 768
        assert model.content_proj.out_features == 1024
        assert model.spk_proj.in_features == 512
        assert model.spk_proj.out_features == 1024

    def test_initialization_logging(self, caplog):
        """Test that initialization logs model configuration."""
        with caplog.at_level(logging.INFO):
            model = MeanFlowDecoder(
                content_dim=512,
                hidden_dim=512,
                mel_dim=80,
                num_layers=6,
                num_heads=8,
            )

        # Check that log message contains key information
        assert "MeanFlowDecoder initialized" in caplog.text
        assert "512D content" in caplog.text
        assert "80D mel" in caplog.text
        assert "6 layers" in caplog.text
        assert "8 heads" in caplog.text

    def test_parameter_count(self):
        """Test that model has reasonable parameter count."""
        model = MeanFlowDecoder()
        param_count = sum(p.numel() for p in model.parameters())

        # Should have millions of parameters (typical for transformer)
        assert param_count > 1_000_000
        assert param_count < 100_000_000  # But not excessive


class TestMeanFlowDecoderForward:
    """Test forward pass functionality."""

    @pytest.fixture
    def model(self):
        """Create model instance for testing."""
        return MeanFlowDecoder(
            content_dim=512,
            speaker_dim=256,
            mel_dim=80,
            hidden_dim=512,
            num_layers=4,
            num_heads=8,
        )

    @pytest.fixture
    def batch_inputs(self):
        """Create batch of input tensors."""
        B, T = 2, 100
        return {
            'x': torch.randn(B, T, 80),
            't': torch.rand(B),
            'r': torch.rand(B),
            'content': torch.randn(B, T, 512),
            'speaker': torch.randn(B, 256),
        }

    def test_forward_basic(self, model, batch_inputs):
        """Test basic forward pass."""
        v = model.forward(**batch_inputs)

        assert v.shape == (2, 100, 80)
        assert not torch.isnan(v).any()
        assert not torch.isinf(v).any()

    def test_forward_with_prompt(self, model, batch_inputs):
        """Test forward pass with prompt conditioning."""
        batch_inputs['prompt'] = torch.randn(2, 50, 80)
        v = model.forward(**batch_inputs)

        # Output should still match original sequence length (not include prompt)
        assert v.shape == (2, 100, 80)
        assert not torch.isnan(v).any()

    def test_forward_different_batch_sizes(self, model):
        """Test forward with various batch sizes."""
        for B in [1, 4, 8]:
            T = 100
            inputs = {
                'x': torch.randn(B, T, 80),
                't': torch.rand(B),
                'r': torch.rand(B),
                'content': torch.randn(B, T, 512),
                'speaker': torch.randn(B, 256),
            }
            v = model.forward(**inputs)
            assert v.shape == (B, T, 80)

    def test_forward_different_sequence_lengths(self, model):
        """Test forward with various sequence lengths."""
        B = 2
        for T in [50, 100, 200, 400]:
            inputs = {
                'x': torch.randn(B, T, 80),
                't': torch.rand(B),
                'r': torch.rand(B),
                'content': torch.randn(B, T, 512),
                'speaker': torch.randn(B, 256),
            }
            v = model.forward(**inputs)
            assert v.shape == (B, T, 80)

    def test_forward_time_conditioning(self, model, batch_inputs):
        """Test that different time values produce different outputs."""
        # Forward with t=0
        batch_inputs['t'] = torch.zeros(2)
        batch_inputs['r'] = torch.zeros(2)
        v1 = model.forward(**batch_inputs)

        # Forward with t=1
        batch_inputs['t'] = torch.ones(2)
        batch_inputs['r'] = torch.zeros(2)
        v2 = model.forward(**batch_inputs)

        # Outputs should be different
        assert not torch.allclose(v1, v2, atol=1e-5)

    def test_forward_speaker_conditioning(self, model, batch_inputs):
        """Test that different speakers produce different outputs."""
        v1 = model.forward(**batch_inputs)

        # Change speaker
        batch_inputs['speaker'] = torch.randn(2, 256)
        v2 = model.forward(**batch_inputs)

        # Outputs should be different
        assert not torch.allclose(v1, v2, atol=1e-5)

    def test_forward_gradient_flow(self, model, batch_inputs):
        """Test that gradients flow through the model."""
        # Enable gradient tracking
        for param in model.parameters():
            param.requires_grad = True

        v = model.forward(**batch_inputs)
        loss = v.sum()
        loss.backward()

        # Check that gradients exist for key modules
        assert model.x_proj.weight.grad is not None
        assert model.content_proj.weight.grad is not None
        assert model.out_proj.weight.grad is not None
        assert not torch.isnan(model.x_proj.weight.grad).any()


class TestMeanFlowDecoderInference:
    """Test inference methods."""

    @pytest.fixture
    def model(self):
        """Create model instance."""
        model = MeanFlowDecoder()
        model.eval()
        return model

    @pytest.fixture
    def inference_inputs(self):
        """Create inference input tensors."""
        B, T = 2, 100
        return {
            'x0': torch.randn(B, T, 80),
            'content': torch.randn(B, T, 512),
            'speaker': torch.randn(B, 256),
        }

    def test_inference_single_step(self, model, inference_inputs):
        """Test single-step inference."""
        with torch.no_grad():
            x1 = model.inference_single_step(**inference_inputs)

        assert x1.shape == (2, 100, 80)
        assert not torch.isnan(x1).any()
        assert not torch.isinf(x1).any()

    def test_inference_single_step_with_prompt(self, model, inference_inputs):
        """Test single-step inference with prompt."""
        inference_inputs['prompt'] = torch.randn(2, 50, 80)

        with torch.no_grad():
            x1 = model.inference_single_step(**inference_inputs)

        assert x1.shape == (2, 100, 80)
        assert not torch.isnan(x1).any()

    def test_inference_two_step(self, model, inference_inputs):
        """Test two-step inference."""
        with torch.no_grad():
            x1 = model.inference_two_step(**inference_inputs)

        assert x1.shape == (2, 100, 80)
        assert not torch.isnan(x1).any()
        assert not torch.isinf(x1).any()

    def test_inference_two_step_vs_single_step(self, model, inference_inputs):
        """Test that two-step produces different output than single-step."""
        with torch.no_grad():
            x1_single = model.inference_single_step(**inference_inputs)
            x1_two = model.inference_two_step(**inference_inputs)

        # Should be different (two-step is more refined)
        assert not torch.allclose(x1_single, x1_two, atol=1e-3)

    def test_inference_deterministic(self, model, inference_inputs):
        """Test that inference is deterministic (no dropout in eval mode)."""
        model.eval()

        with torch.no_grad():
            x1_first = model.inference_single_step(**inference_inputs)
            x1_second = model.inference_single_step(**inference_inputs)

        assert torch.allclose(x1_first, x1_second, atol=1e-6)

    def test_inference_different_sequence_lengths(self, model):
        """Test inference with various sequence lengths."""
        B = 2
        for T in [50, 100, 200]:
            inputs = {
                'x0': torch.randn(B, T, 80),
                'content': torch.randn(B, T, 512),
                'speaker': torch.randn(B, 256),
            }

            with torch.no_grad():
                x1 = model.inference_single_step(**inputs)

            assert x1.shape == (B, T, 80)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_inference_cuda(self, model, inference_inputs):
        """Test inference on GPU."""
        model = model.cuda()
        inference_inputs = {k: v.cuda() for k, v in inference_inputs.items()}

        with torch.no_grad():
            x1 = model.inference_single_step(**inference_inputs)

        assert x1.is_cuda
        assert x1.shape == (2, 100, 80)


class TestTimeEmbedding:
    """Test TimeEmbedding module."""

    def test_time_embedding_initialization(self):
        """Test TimeEmbedding initialization."""
        embed = TimeEmbedding(dim=512)

        assert embed.dim == 512
        assert embed.max_period == 10000
        assert isinstance(embed.mlp, nn.Sequential)

    def test_time_embedding_forward(self):
        """Test TimeEmbedding forward pass."""
        embed = TimeEmbedding(dim=512)
        t = torch.rand(4)  # Batch of 4 time values

        emb = embed(t)

        assert emb.shape == (4, 512)
        assert not torch.isnan(emb).any()

    def test_time_embedding_range(self):
        """Test TimeEmbedding with various time ranges."""
        embed = TimeEmbedding(dim=512)

        # Test edge values
        t_zero = torch.zeros(2)
        t_one = torch.ones(2)
        t_mid = torch.ones(2) * 0.5

        emb_zero = embed(t_zero)
        emb_one = embed(t_one)
        emb_mid = embed(t_mid)

        assert emb_zero.shape == (2, 512)
        assert emb_one.shape == (2, 512)
        assert emb_mid.shape == (2, 512)

        # Different times should produce different embeddings
        assert not torch.allclose(emb_zero, emb_one, atol=1e-3)
        assert not torch.allclose(emb_zero, emb_mid, atol=1e-3)

    def test_time_embedding_odd_dimension(self):
        """Test TimeEmbedding with odd dimension."""
        embed = TimeEmbedding(dim=513)  # Odd dimension
        t = torch.rand(4)

        emb = embed(t)

        # Should handle odd dimension correctly (padding with zeros)
        assert emb.shape == (4, 513)
        assert not torch.isnan(emb).any()

    def test_time_embedding_gradient_flow(self):
        """Test gradient flow through TimeEmbedding."""
        embed = TimeEmbedding(dim=512)
        t = torch.rand(4, requires_grad=True)

        emb = embed(t)
        loss = emb.sum()
        loss.backward()

        # Gradients should flow back to MLP parameters
        for param in embed.mlp.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_time_embedding_custom_max_period(self):
        """Test TimeEmbedding with custom max_period."""
        embed = TimeEmbedding(dim=256, max_period=5000)
        t = torch.rand(4)

        emb = embed(t)

        assert emb.shape == (4, 256)
        assert embed.max_period == 5000


class TestComputeMeanFlowLoss:
    """Test compute_mean_flow_loss utility function."""

    @pytest.fixture
    def model(self):
        """Create model for loss testing."""
        return MeanFlowDecoder(
            content_dim=512,
            speaker_dim=256,
            mel_dim=80,
            hidden_dim=256,
            num_layers=2,
        )

    @pytest.fixture
    def loss_inputs(self):
        """Create inputs for loss computation."""
        B, T = 4, 100
        return {
            'x0': torch.randn(B, T, 80),
            'x1': torch.randn(B, T, 80),
            'content': torch.randn(B, T, 512),
            'speaker': torch.randn(B, 256),
        }

    def test_compute_mean_flow_loss_basic(self, model, loss_inputs):
        """Test basic loss computation."""
        fm_loss, mean_loss = compute_mean_flow_loss(model, **loss_inputs)

        assert isinstance(fm_loss, torch.Tensor)
        assert isinstance(mean_loss, torch.Tensor)
        assert fm_loss.ndim == 0  # Scalar
        assert mean_loss.ndim == 0  # Scalar
        assert fm_loss.item() >= 0
        assert mean_loss.item() >= 0

    def test_compute_mean_flow_loss_with_prompt(self, model, loss_inputs):
        """Test loss computation with prompt."""
        loss_inputs['prompt'] = torch.randn(4, 50, 80)

        fm_loss, mean_loss = compute_mean_flow_loss(model, **loss_inputs)

        assert fm_loss.item() >= 0
        assert mean_loss.item() >= 0

    def test_compute_mean_flow_loss_gradient_flow(self, model, loss_inputs):
        """Test that gradients flow through loss computation."""
        fm_loss, mean_loss = compute_mean_flow_loss(model, **loss_inputs)

        total_loss = fm_loss + mean_loss
        total_loss.backward()

        # Check gradients exist
        assert model.out_proj.weight.grad is not None
        assert not torch.isnan(model.out_proj.weight.grad).any()

    def test_compute_mean_flow_loss_different_targets(self, model, loss_inputs):
        """Test that different targets produce different losses."""
        fm_loss1, mean_loss1 = compute_mean_flow_loss(model, **loss_inputs)

        # Change target
        loss_inputs['x1'] = torch.randn(4, 100, 80)
        fm_loss2, mean_loss2 = compute_mean_flow_loss(model, **loss_inputs)

        # Losses should be different
        assert not torch.allclose(fm_loss1, fm_loss2, atol=1e-5)
        assert not torch.allclose(mean_loss1, mean_loss2, atol=1e-5)

    def test_compute_mean_flow_loss_zero_velocity(self, model):
        """Test loss when x0 and x1 are identical (zero velocity)."""
        B, T = 4, 100
        x = torch.randn(B, T, 80)

        inputs = {
            'x0': x,
            'x1': x,  # Same as x0
            'content': torch.randn(B, T, 512),
            'speaker': torch.randn(B, 256),
        }

        fm_loss, mean_loss = compute_mean_flow_loss(model, **inputs)

        # Losses should be relatively small (ideally zero if model predicts zero)
        assert fm_loss.item() >= 0
        assert mean_loss.item() >= 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_frame_sequence(self):
        """Test with minimum sequence length (1 frame)."""
        model = MeanFlowDecoder()
        B, T = 2, 1

        inputs = {
            'x': torch.randn(B, T, 80),
            't': torch.rand(B),
            'r': torch.rand(B),
            'content': torch.randn(B, T, 512),
            'speaker': torch.randn(B, 256),
        }

        v = model.forward(**inputs)
        assert v.shape == (B, T, 80)

    def test_large_batch_size(self):
        """Test with large batch size."""
        model = MeanFlowDecoder()
        B, T = 32, 100

        inputs = {
            'x': torch.randn(B, T, 80),
            't': torch.rand(B),
            'r': torch.rand(B),
            'content': torch.randn(B, T, 512),
            'speaker': torch.randn(B, 256),
        }

        with torch.no_grad():
            v = model.forward(**inputs)

        assert v.shape == (B, T, 80)

    def test_long_prompt(self):
        """Test with prompt longer than main sequence."""
        model = MeanFlowDecoder()
        B, T = 2, 50

        inputs = {
            'x': torch.randn(B, T, 80),
            't': torch.rand(B),
            'r': torch.rand(B),
            'content': torch.randn(B, T, 512),
            'speaker': torch.randn(B, 256),
            'prompt': torch.randn(B, 200, 80),  # Longer than main sequence
        }

        v = model.forward(**inputs)
        assert v.shape == (B, T, 80)  # Should still match original length

    def test_model_mode_switching(self):
        """Test switching between train and eval modes."""
        model = MeanFlowDecoder()

        model.train()
        assert model.training

        model.eval()
        assert not model.training

    def test_determinism_in_eval_mode(self):
        """Test that eval mode produces deterministic results."""
        model = MeanFlowDecoder()
        model.eval()

        B, T = 2, 100
        inputs = {
            'x': torch.randn(B, T, 80),
            't': torch.rand(B),
            'r': torch.rand(B),
            'content': torch.randn(B, T, 512),
            'speaker': torch.randn(B, 256),
        }

        # Multiple forward passes should be identical in eval mode
        with torch.no_grad():
            v1 = model.forward(**inputs)
            v2 = model.forward(**inputs)

        assert torch.allclose(v1, v2, atol=1e-6)


class TestMainScriptExecution:
    """Test the __main__ script sanity check."""

    @patch('auto_voice.inference.mean_flow_decoder.torch.cuda.is_available')
    def test_main_cpu_mode(self, mock_cuda, capsys):
        """Test main script execution in CPU mode."""
        mock_cuda.return_value = False

        # Import and run the main block
        import auto_voice.inference.mean_flow_decoder as mfd

        # Manually execute main block logic
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = mfd.MeanFlowDecoder().to(device)

        B, T = 2, 100
        x0 = torch.randn(B, T, 80, device=device)
        content = torch.randn(B, T, 512, device=device)
        speaker = torch.randn(B, 256, device=device)

        x1 = model.inference_single_step(x0, content, speaker)
        assert x1.shape == (B, T, 80)

        x1_2step = model.inference_two_step(x0, content, speaker)
        assert x1_2step.shape == (B, T, 80)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_main_cuda_mode(self):
        """Test main script execution in CUDA mode."""
        import auto_voice.inference.mean_flow_decoder as mfd

        device = torch.device('cuda')
        model = mfd.MeanFlowDecoder().to(device)

        B, T = 2, 100
        x0 = torch.randn(B, T, 80, device=device)
        content = torch.randn(B, T, 512, device=device)
        speaker = torch.randn(B, 256, device=device)

        x1 = model.inference_single_step(x0, content, speaker)
        assert x1.is_cuda
        assert x1.shape == (B, T, 80)


class TestModelPersistence:
    """Test model saving and loading."""

    def test_state_dict_save_load(self, tmp_path):
        """Test saving and loading model state dict."""
        model1 = MeanFlowDecoder()

        # Save state dict
        checkpoint_path = tmp_path / "model.pt"
        torch.save(model1.state_dict(), checkpoint_path)

        # Create new model and load state
        model2 = MeanFlowDecoder()
        model2.load_state_dict(torch.load(checkpoint_path, weights_only=True))

        # Use same random seed for both forward passes
        torch.manual_seed(42)
        B, T = 2, 100
        inputs = {
            'x': torch.randn(B, T, 80),
            't': torch.rand(B),
            'r': torch.rand(B),
            'content': torch.randn(B, T, 512),
            'speaker': torch.randn(B, 256),
        }

        model1.eval()
        model2.eval()

        with torch.no_grad():
            v1 = model1.forward(**inputs)

        # Reset seed and regenerate inputs
        torch.manual_seed(42)
        inputs = {
            'x': torch.randn(B, T, 80),
            't': torch.rand(B),
            'r': torch.rand(B),
            'content': torch.randn(B, T, 512),
            'speaker': torch.randn(B, 256),
        }

        with torch.no_grad():
            v2 = model2.forward(**inputs)

        assert torch.allclose(v1, v2, atol=1e-5)

    def test_full_model_save_load(self, tmp_path):
        """Test saving and loading full model."""
        model1 = MeanFlowDecoder()

        # Save full model
        checkpoint_path = tmp_path / "full_model.pt"
        torch.save(model1, checkpoint_path)

        # Load model with weights_only=False (safe for testing)
        model2 = torch.load(checkpoint_path, weights_only=False)

        # Verify parameters match
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-6)
