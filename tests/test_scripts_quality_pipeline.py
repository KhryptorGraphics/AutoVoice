#!/usr/bin/env python3
"""Unit tests for SOTA quality pipeline (scripts/quality_pipeline.py).

Tests the Seed-VC + BigVGAN pipeline for high-quality conversion.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'models' / 'seed-vc'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

import pytest
import torch
import numpy as np

from quality_pipeline import QualityVoiceConverter, QualityConfig


@pytest.fixture
def converter():
    """Create a quality converter instance."""
    config = QualityConfig(
        sample_rate=44100,
        diffusion_steps=10,  # Use fewer steps for testing
        fp16=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    return QualityVoiceConverter(config)


@pytest.fixture
def sample_audio():
    """Generate synthetic audio for testing."""
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sr


class TestQualityConfig:
    """Test configuration."""

    def test_default_config(self):
        """Test default values."""
        config = QualityConfig()
        assert config.sample_rate == 44100
        assert config.diffusion_steps == 30
        assert config.fp16 is True


class TestQualityConverter:
    """Test converter."""

    def test_initialization(self, converter):
        """Test init."""
        assert converter is not None
        assert converter.config.sample_rate == 44100

    def test_unload(self, converter):
        """Test unload."""
        converter.unload()
        # Models are lazily loaded, so unload should succeed even if not loaded

    @pytest.mark.cuda
    @pytest.mark.slow
    def test_convert(self, converter, sample_audio):
        """Test conversion."""
        audio, sr = sample_audio
        # Need reference audio for Seed-VC
        reference = audio.copy()
        
        converted, out_sr = converter.convert(
            source_audio=audio,
            source_sr=sr,
            reference_audio=reference,
            reference_sr=sr,
            pitch_shift=0
        )
        
        assert len(converted) > 0
        assert out_sr == 44100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
