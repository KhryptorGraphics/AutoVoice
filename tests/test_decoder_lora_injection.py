"""Tests for LoRA injection in CoMoSVCDecoder.

Task 1.1-1.6: Test LoRA adapter injection and removal in the decoder.

Tests verify:
- inject_lora() wraps Linear layers with LoRALinear
- remove_lora() restores original layers
- Output shape unchanged after LoRA operations
"""

import pytest
import torch
import torch.nn as nn

from auto_voice.models.svc_decoder import CoMoSVCDecoder
from auto_voice.models.feature_contract import DEFAULT_CONTENT_DIM, DEFAULT_PITCH_DIM, DEFAULT_SPEAKER_DIM
from auto_voice.training.fine_tuning import LoRALinear


@pytest.fixture
def device():
    """Get available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def decoder(device):
    """Create decoder instance."""
    return CoMoSVCDecoder(device=device).to(device)


@pytest.fixture
def sample_inputs(device):
    """Create sample inputs for decoder."""
    batch_size = 2
    seq_len = 100
    content = torch.randn(batch_size, seq_len, DEFAULT_CONTENT_DIM, device=device)
    pitch = torch.randn(batch_size, seq_len, DEFAULT_PITCH_DIM, device=device)
    speaker = torch.randn(batch_size, DEFAULT_SPEAKER_DIM, device=device)
    return content, pitch, speaker


class TestLoRAInjection:
    """Tests for inject_lora() method."""

    def test_inject_lora_method_exists(self, decoder):
        """Task 1.1: Decoder should have inject_lora() method."""
        assert hasattr(decoder, 'inject_lora'), "Decoder missing inject_lora() method"
        assert callable(decoder.inject_lora), "inject_lora should be callable"

    def test_inject_lora_wraps_input_proj(self, decoder):
        """Task 1.2: inject_lora should wrap input_proj Linear with LoRALinear."""
        # Before injection - should be nn.Linear
        assert isinstance(decoder.input_proj, nn.Linear), "input_proj should start as nn.Linear"

        # Inject LoRA
        decoder.inject_lora(rank=8, alpha=16)

        # After injection - should be LoRALinear
        assert isinstance(decoder.input_proj, LoRALinear), \
            "input_proj should be wrapped with LoRALinear after injection"

    def test_inject_lora_with_custom_rank(self, decoder):
        """inject_lora should respect custom rank parameter."""
        decoder.inject_lora(rank=16, alpha=32)

        assert isinstance(decoder.input_proj, LoRALinear)
        assert decoder.input_proj.adapter.rank == 16, "LoRA rank should be 16"
        assert decoder.input_proj.adapter.alpha == 32, "LoRA alpha should be 32"

    def test_inject_lora_idempotent(self, decoder):
        """Calling inject_lora twice should not double-wrap."""
        decoder.inject_lora(rank=8)
        decoder.inject_lora(rank=8)  # Second call

        # Should still be LoRALinear, not nested
        assert isinstance(decoder.input_proj, LoRALinear)
        assert not isinstance(decoder.input_proj.original, LoRALinear), \
            "Should not double-wrap LoRA"

    def test_lora_injected_flag(self, decoder):
        """Decoder should track whether LoRA is injected."""
        assert not getattr(decoder, '_lora_injected', False), \
            "Should not have LoRA injected initially"

        decoder.inject_lora(rank=8)

        assert decoder._lora_injected is True, \
            "_lora_injected should be True after injection"


class TestLoRARemoval:
    """Tests for remove_lora() method."""

    def test_remove_lora_method_exists(self, decoder):
        """Task 1.3: Decoder should have remove_lora() method."""
        assert hasattr(decoder, 'remove_lora'), "Decoder missing remove_lora() method"
        assert callable(decoder.remove_lora), "remove_lora should be callable"

    def test_remove_lora_restores_linear(self, decoder):
        """Task 1.4: remove_lora should restore original nn.Linear."""
        decoder.inject_lora(rank=8)
        assert isinstance(decoder.input_proj, LoRALinear)

        decoder.remove_lora()

        assert isinstance(decoder.input_proj, nn.Linear), \
            "input_proj should be restored to nn.Linear"
        assert not isinstance(decoder.input_proj, LoRALinear), \
            "Should not be LoRALinear after removal"

    def test_remove_lora_clears_flag(self, decoder):
        """remove_lora should clear _lora_injected flag."""
        decoder.inject_lora(rank=8)
        assert decoder._lora_injected is True

        decoder.remove_lora()

        assert decoder._lora_injected is False, \
            "_lora_injected should be False after removal"

    def test_remove_lora_idempotent(self, decoder):
        """Calling remove_lora without injection should be safe."""
        # Should not raise
        decoder.remove_lora()
        decoder.remove_lora()

        assert isinstance(decoder.input_proj, nn.Linear)


class TestLoRAOutputShape:
    """Tests for decoder output shape with LoRA."""

    def test_output_shape_without_lora(self, decoder, sample_inputs):
        """Task 1.5: Verify baseline output shape."""
        content, pitch, speaker = sample_inputs

        with torch.no_grad():
            output = decoder.infer(content, pitch, speaker, n_steps=1)

        # Output should be [batch, n_mels, seq_len]
        assert output.shape == (2, 100, 100), \
            f"Expected shape (2, 100, 100), got {output.shape}"

    def test_output_shape_with_lora(self, decoder, sample_inputs):
        """Task 1.6: Output shape should be unchanged after LoRA injection."""
        content, pitch, speaker = sample_inputs

        # Inject LoRA
        decoder.inject_lora(rank=8, alpha=16)

        with torch.no_grad():
            output = decoder.infer(content, pitch, speaker, n_steps=1)

        # Shape should be same as without LoRA
        assert output.shape == (2, 100, 100), \
            f"LoRA should not change output shape. Got {output.shape}"

    def test_output_shape_after_lora_removal(self, decoder, sample_inputs):
        """Output shape should remain correct after LoRA removal."""
        content, pitch, speaker = sample_inputs

        decoder.inject_lora(rank=8)
        decoder.remove_lora()

        with torch.no_grad():
            output = decoder.infer(content, pitch, speaker, n_steps=1)

        assert output.shape == (2, 100, 100), \
            f"Shape should be preserved after LoRA removal. Got {output.shape}"


class TestLoRAStateDict:
    """Tests for LoRA state dict operations."""

    def test_get_lora_state_dict(self, decoder):
        """Should be able to extract LoRA parameters."""
        decoder.inject_lora(rank=8)

        assert hasattr(decoder, 'get_lora_state_dict'), \
            "Decoder should have get_lora_state_dict() method"

        state_dict = decoder.get_lora_state_dict()

        assert isinstance(state_dict, dict), "Should return a dict"
        assert len(state_dict) > 0, "LoRA state dict should not be empty"
        # Should contain lora_A and lora_B parameters
        assert any('lora_A' in k for k in state_dict.keys()), \
            "Should contain lora_A parameters"
        assert any('lora_B' in k for k in state_dict.keys()), \
            "Should contain lora_B parameters"

    def test_load_lora_state_dict(self, decoder, device):
        """Should be able to load LoRA parameters."""
        decoder.inject_lora(rank=8)

        # Save state
        state_dict = decoder.get_lora_state_dict()

        # Create new decoder and inject LoRA
        decoder2 = CoMoSVCDecoder(device=device).to(device)
        decoder2.inject_lora(rank=8)

        assert hasattr(decoder2, 'load_lora_state_dict'), \
            "Decoder should have load_lora_state_dict() method"

        # Load state into new decoder
        decoder2.load_lora_state_dict(state_dict)

        # Parameters should match
        state_dict2 = decoder2.get_lora_state_dict()
        for key in state_dict:
            assert torch.allclose(state_dict[key], state_dict2[key]), \
                f"Parameter {key} should match after loading"
