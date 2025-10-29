"""
Unit tests for VTLP (Vocal Tract Length Perturbation) augmentation.

Tests verify:
1. VTLP preserves mel-spectrogram shape
2. VTLP maintains temporal alignment
3. VTLP applies warping correctly
"""

import numpy as np
import pytest
import torch

from src.auto_voice.training.dataset import SingingAugmentation


class TestVTLPAugmentation:
    """Test suite for Vocal Tract Length Perturbation augmentation."""

    def test_vtlp_preserves_shape(self):
        """Test that VTLP preserves mel-spectrogram shape."""
        # Create synthetic mel-spectrogram
        mel_len = 150
        n_mels = 80
        source_mel = torch.randn(mel_len, n_mels)
        target_mel = torch.randn(mel_len, n_mels)

        # Create sample data dict
        data = {
            'source_mel': source_mel,
            'target_mel': target_mel,
            'lengths': torch.LongTensor([mel_len]),
            'source_audio': torch.randn(44100 * 2),  # 2 seconds
            'target_audio': torch.randn(44100 * 2),
            'sample_rate': 44100
        }

        # Apply VTLP augmentation
        augmented_data = SingingAugmentation.vocal_tract_length_perturbation(
            data,
            alpha_range=(0.95, 1.05)
        )

        # Verify shapes are preserved
        assert augmented_data['source_mel'].shape == source_mel.shape, \
            f"Source mel shape changed: {augmented_data['source_mel'].shape} != {source_mel.shape}"
        assert augmented_data['target_mel'].shape == target_mel.shape, \
            f"Target mel shape changed: {augmented_data['target_mel'].shape} != {target_mel.shape}"

        # Verify lengths are preserved
        assert augmented_data['lengths'].item() == mel_len, \
            f"Mel length changed: {augmented_data['lengths'].item()} != {mel_len}"

    def test_vtlp_maintains_alignment(self):
        """Test that VTLP maintains temporal alignment between source and target."""
        mel_len = 200
        n_mels = 80

        # Create aligned mel-spectrograms (similar patterns)
        time_axis = torch.linspace(0, 10, mel_len).unsqueeze(1)
        freq_axis = torch.linspace(0, 1, n_mels).unsqueeze(0)

        # Create pattern with temporal structure
        source_mel = torch.sin(time_axis) * torch.cos(freq_axis * 10)
        target_mel = source_mel * 1.1  # Slightly different amplitude but same structure

        data = {
            'source_mel': source_mel,
            'target_mel': target_mel,
            'lengths': torch.LongTensor([mel_len]),
            'source_audio': torch.randn(44100 * 3),
            'target_audio': torch.randn(44100 * 3),
            'sample_rate': 44100
        }

        # Apply VTLP
        augmented_data = SingingAugmentation.vocal_tract_length_perturbation(
            data,
            alpha_range=(1.0, 1.0)  # No warping to test alignment preservation
        )

        # With alpha=1.0, mels should be identical (no warping)
        assert torch.allclose(augmented_data['source_mel'], source_mel, atol=1e-6), \
            "VTLP with alpha=1.0 should not modify mel-spectrogram"
        assert torch.allclose(augmented_data['target_mel'], target_mel, atol=1e-6), \
            "VTLP with alpha=1.0 should not modify target mel-spectrogram"

    def test_vtlp_applies_warping(self):
        """Test that VTLP actually applies frequency warping."""
        mel_len = 100
        n_mels = 80

        # Create mel with distinct frequency content
        source_mel = torch.zeros(mel_len, n_mels)
        # Add energy at specific frequency bin
        source_mel[:, 40] = 1.0  # Middle frequency

        data = {
            'source_mel': source_mel.clone(),
            'target_mel': source_mel.clone(),
            'lengths': torch.LongTensor([mel_len]),
            'source_audio': torch.randn(44100),
            'target_audio': torch.randn(44100),
            'sample_rate': 44100
        }

        # Apply VTLP with significant warping
        augmented_data = SingingAugmentation.vocal_tract_length_perturbation(
            data,
            alpha_range=(1.2, 1.2)  # 20% upward warping
        )

        # With alpha=1.2, energy should shift to higher frequencies
        # Original peak at bin 40, with alpha=1.2 should shift upward
        warped_mel = augmented_data['source_mel']

        # Check that the mel has changed
        assert not torch.allclose(warped_mel, source_mel, atol=1e-6), \
            "VTLP with alpha=1.2 should modify mel-spectrogram"

        # Check that energy has shifted (not exactly at bin 40 anymore)
        original_peak = source_mel[:, 40].sum().item()
        warped_peak = warped_mel[:, 40].sum().item()
        assert warped_peak < original_peak * 0.9, \
            f"Energy should shift from bin 40 after warping: {warped_peak} vs {original_peak}"

    def test_vtlp_skips_small_alpha(self):
        """Test that VTLP skips augmentation when alpha is very close to 1.0."""
        mel_len = 100
        n_mels = 80
        source_mel = torch.randn(mel_len, n_mels)
        target_mel = torch.randn(mel_len, n_mels)

        data = {
            'source_mel': source_mel.clone(),
            'target_mel': target_mel.clone(),
            'lengths': torch.LongTensor([mel_len]),
            'source_audio': torch.randn(44100),
            'target_audio': torch.randn(44100),
            'sample_rate': 44100
        }

        # Apply VTLP with alpha very close to 1.0 (should skip)
        augmented_data = SingingAugmentation.vocal_tract_length_perturbation(
            data,
            alpha_range=(1.005, 1.005)  # Within skip threshold
        )

        # Should return original data unchanged
        assert torch.allclose(augmented_data['source_mel'], source_mel), \
            "VTLP should skip when alpha is very close to 1.0"
        assert torch.allclose(augmented_data['target_mel'], target_mel), \
            "VTLP should skip when alpha is very close to 1.0"

    def test_vtlp_handles_variable_length(self):
        """Test that VTLP correctly handles variable-length mel-spectrograms."""
        mel_len = 150
        actual_len = 100  # Actual content length
        n_mels = 80

        # Create mel with padded zeros
        source_mel = torch.randn(mel_len, n_mels)
        source_mel[actual_len:] = 0  # Zero padding

        target_mel = torch.randn(mel_len, n_mels)
        target_mel[actual_len:] = 0

        data = {
            'source_mel': source_mel.clone(),
            'target_mel': target_mel.clone(),
            'lengths': torch.LongTensor([actual_len]),  # Actual length
            'source_audio': torch.randn(44100),
            'target_audio': torch.randn(44100),
            'sample_rate': 44100
        }

        # Apply VTLP
        augmented_data = SingingAugmentation.vocal_tract_length_perturbation(
            data,
            alpha_range=(0.9, 1.1)
        )

        # Verify length is preserved
        assert augmented_data['lengths'].item() == actual_len, \
            f"Actual length should be preserved: {augmented_data['lengths'].item()} != {actual_len}"

        # Verify shape is preserved
        assert augmented_data['source_mel'].shape == (mel_len, n_mels), \
            "Mel shape should be preserved"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
