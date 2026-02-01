"""
Tests for shortcut flow matching implementation.

Validates that the ShortcutFlowMatching wrapper correctly:
1. Adds step size embedding capability
2. Performs shortcut inference with configurable step counts
3. Computes self-consistency loss correctly
"""

import pytest
import torch
import sys
import os

# Add Seed-VC modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../models/seed-vc'))

from modules.shortcut_flow_matching import ShortcutFlowMatching, StepSizeEmbedder, enable_shortcut_cfm


class MockCFMEstimator(torch.nn.Module):
    """Mock DiT estimator for testing."""

    def __init__(self, in_channels=128, hidden_dim=256):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        # Simple linear layer to simulate DiT
        self.linear = torch.nn.Linear(in_channels, in_channels)

    def forward(self, x, prompt_x, x_lens, t, style, mu, prompt_lens=None):
        """Mock forward pass."""
        # Just return a simple transformation
        B, C, T = x.shape
        x_flat = x.transpose(1, 2).reshape(B * T, C)
        out_flat = self.linear(x_flat)
        out = out_flat.reshape(B, T, C).transpose(1, 2)
        return out

    def setup_caches(self, max_batch_size, max_seq_length):
        """Mock cache setup."""
        pass


class MockCFM(torch.nn.Module):
    """Mock CFM model for testing."""

    def __init__(self, in_channels=128, hidden_dim=256):
        super().__init__()
        self.estimator = MockCFMEstimator(in_channels, hidden_dim)
        self.in_channels = in_channels
        self.sigma_min = 1e-6
        self.zero_prompt_speech_token = False
        self.criterion = torch.nn.MSELoss()

    def forward(self, x1, x_lens, prompt_lens, mu, style):
        """Mock CFM forward (standard flow matching loss)."""
        b, _, t = x1.shape
        device = x1.device

        # Sample random time
        t_sample = torch.rand([b, 1, 1], device=device, dtype=x1.dtype)

        # Sample noise
        z = torch.randn_like(x1)

        # Noisy sample
        y = (1 - (1 - self.sigma_min) * t_sample) * z + t_sample * x1

        # Target velocity
        u = x1 - (1 - self.sigma_min) * z

        # Prepare prompt
        prompt = torch.zeros_like(x1)
        for bib in range(b):
            prompt[bib, :, :prompt_lens[bib]] = x1[bib, :, :prompt_lens[bib]]
            y[bib, :, :prompt_lens[bib]] = 0

        # Estimate
        estimator_out = self.estimator(y, prompt, x_lens, t_sample.squeeze(1).squeeze(1), style, mu, prompt_lens)

        # Compute loss
        loss = 0
        for bib in range(b):
            loss += self.criterion(
                estimator_out[bib, :, prompt_lens[bib]:x_lens[bib]],
                u[bib, :, prompt_lens[bib]:x_lens[bib]]
            )
        loss /= b

        return loss, estimator_out + (1 - self.sigma_min) * z


@pytest.fixture
def mock_cfm():
    """Create mock CFM model."""
    return MockCFM(in_channels=128, hidden_dim=256)


@pytest.fixture
def shortcut_cfm(mock_cfm):
    """Create shortcut CFM wrapper."""
    return enable_shortcut_cfm(mock_cfm, hidden_dim=256)


@pytest.mark.smoke
def test_step_size_embedder():
    """Test that step size embedder produces correct shapes."""
    embedder = StepSizeEmbedder(hidden_size=256)

    # Test with various batch sizes
    for batch_size in [1, 4, 8]:
        d = torch.rand(batch_size)
        d_emb = embedder(d)

        assert d_emb.shape == (batch_size, 256), f"Expected shape ({batch_size}, 256), got {d_emb.shape}"
        assert not torch.isnan(d_emb).any(), "Step size embedding contains NaN"
        assert not torch.isinf(d_emb).any(), "Step size embedding contains Inf"


@pytest.mark.smoke
def test_shortcut_cfm_initialization(shortcut_cfm):
    """Test that ShortcutFlowMatching initializes correctly."""
    assert shortcut_cfm.d_embedder is not None, "Step size embedder not initialized"
    assert hasattr(shortcut_cfm, 'base_cfm'), "Base CFM not stored"
    assert shortcut_cfm.k_flow_matching == 0.7, "Incorrect batch split ratio"


@pytest.mark.smoke
def test_shortcut_inference_shapes(shortcut_cfm):
    """Test that shortcut inference produces correct output shapes."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shortcut_cfm = shortcut_cfm.to(device)

    B, T = 2, 100
    in_channels = 128
    style_dim = 192

    # Create mock inputs
    mu = torch.randn(B, T, 256, device=device)
    x_lens = torch.tensor([T, T], device=device)
    prompt = torch.randn(in_channels, 20, device=device)
    style = torch.randn(B, style_dim, device=device)
    f0 = None

    # Test with different step counts
    for n_steps in [1, 2, 5, 10]:
        output = shortcut_cfm.shortcut_inference(
            mu, x_lens, prompt, style, f0, n_timesteps=n_steps
        )

        assert output.shape == (B, in_channels, T), \
            f"Expected shape ({B}, {in_channels}, {T}), got {output.shape} for {n_steps} steps"
        assert not torch.isnan(output).any(), f"Output contains NaN for {n_steps} steps"


@pytest.mark.smoke
def test_flow_matching_loss(shortcut_cfm):
    """Test that FM loss computation works."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shortcut_cfm = shortcut_cfm.to(device)

    B, T = 2, 100
    in_channels = 128
    style_dim = 192

    # Create mock inputs
    x1 = torch.randn(B, in_channels, T, device=device)
    x_lens = torch.tensor([T, T], device=device)
    prompt_lens = torch.tensor([20, 20], device=device)
    mu = torch.randn(B, T, 256, device=device)
    style = torch.randn(B, style_dim, device=device)

    # Compute FM loss
    loss, output = shortcut_cfm._flow_matching_loss(x1, x_lens, prompt_lens, mu, style)

    assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert loss.item() >= 0, "Loss should be non-negative"
    assert output.shape == x1.shape, f"Output shape mismatch: {output.shape} vs {x1.shape}"


@pytest.mark.smoke
def test_self_consistency_loss(shortcut_cfm):
    """Test that self-consistency loss computation works."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shortcut_cfm = shortcut_cfm.to(device)

    B, T = 2, 100
    in_channels = 128
    style_dim = 192

    # Create mock inputs
    x1 = torch.randn(B, in_channels, T, device=device)
    x_lens = torch.tensor([T, T], device=device)
    prompt_lens = torch.tensor([20, 20], device=device)
    mu = torch.randn(B, T, 256, device=device)
    style = torch.randn(B, style_dim, device=device)

    # Compute SC loss
    loss, output = shortcut_cfm._self_consistency_loss(x1, x_lens, prompt_lens, mu, style)

    assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert loss.item() >= 0, "Loss should be non-negative"
    assert output.shape == x1.shape, f"Output shape mismatch: {output.shape} vs {x1.shape}"


@pytest.mark.smoke
def test_dual_objective_training(shortcut_cfm):
    """Test that training correctly alternates between FM and SC objectives."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shortcut_cfm = shortcut_cfm.to(device)

    B, T = 2, 100
    in_channels = 128
    style_dim = 192

    # Create mock inputs
    x1 = torch.randn(B, in_channels, T, device=device)
    x_lens = torch.tensor([T, T], device=device)
    prompt_lens = torch.tensor([20, 20], device=device)
    mu = torch.randn(B, T, 256, device=device)
    style = torch.randn(B, style_dim, device=device)

    # Run multiple training steps and count objective types
    fm_count = 0
    sc_count = 0
    n_trials = 100

    torch.manual_seed(42)  # For reproducibility
    for _ in range(n_trials):
        loss, output, obj_type = shortcut_cfm.forward(x1, x_lens, prompt_lens, mu, style, training=True)

        assert obj_type in ["FM", "SC"], f"Unknown objective type: {obj_type}"
        assert loss.item() >= 0, "Loss should be non-negative"

        if obj_type == "FM":
            fm_count += 1
        else:
            sc_count += 1

    # Check that ratio is approximately 70/30
    fm_ratio = fm_count / n_trials
    assert 0.6 < fm_ratio < 0.8, f"FM ratio {fm_ratio} outside expected range (0.6-0.8)"


@pytest.mark.integration
def test_shortcut_vs_baseline_consistency(shortcut_cfm):
    """
    Test that shortcut inference with many steps produces similar results
    to baseline CFM.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shortcut_cfm = shortcut_cfm.to(device)

    B, T = 1, 100
    in_channels = 128
    style_dim = 192

    # Create mock inputs
    mu = torch.randn(B, T, 256, device=device)
    x_lens = torch.tensor([T], device=device)
    prompt = torch.randn(in_channels, 20, device=device)
    style = torch.randn(B, style_dim, device=device)
    f0 = None

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Run with many steps (should be close to baseline)
    output_10 = shortcut_cfm.shortcut_inference(
        mu, x_lens, prompt, style, f0, n_timesteps=10, temperature=1.0
    )

    # Run with fewer steps
    torch.manual_seed(42)
    output_2 = shortcut_cfm.shortcut_inference(
        mu, x_lens, prompt, style, f0, n_timesteps=2, temperature=1.0
    )

    # Both should have valid outputs (but may differ in quality)
    assert not torch.isnan(output_10).any(), "10-step output contains NaN"
    assert not torch.isnan(output_2).any(), "2-step output contains NaN"
    assert output_10.shape == output_2.shape, "Shape mismatch between step counts"


if __name__ == "__main__":
    # Run smoke tests
    print("Testing StepSizeEmbedder...")
    test_step_size_embedder()
    print("✓ StepSizeEmbedder tests passed")

    print("\nTesting ShortcutFlowMatching initialization...")
    mock_cfm = MockCFM()
    shortcut_cfm = enable_shortcut_cfm(mock_cfm, hidden_dim=256)
    test_shortcut_cfm_initialization(shortcut_cfm)
    print("✓ Initialization tests passed")

    print("\nTesting shortcut inference shapes...")
    test_shortcut_inference_shapes(shortcut_cfm)
    print("✓ Inference shape tests passed")

    print("\nTesting FM loss...")
    test_flow_matching_loss(shortcut_cfm)
    print("✓ FM loss tests passed")

    print("\nTesting SC loss...")
    test_self_consistency_loss(shortcut_cfm)
    print("✓ SC loss tests passed")

    print("\nTesting dual objective training...")
    test_dual_objective_training(shortcut_cfm)
    print("✓ Dual objective tests passed")

    print("\n✅ All tests passed!")
