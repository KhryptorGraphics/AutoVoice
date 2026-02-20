#!/usr/bin/env python3
"""Unit tests for SOTA realtime pipeline (scripts/realtime_pipeline.py).

Tests the ContentVec + Simple Decoder + HiFiGAN pipeline for low-latency conversion.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'models' / 'seed-vc'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

import pytest
import torch
import numpy as np

from realtime_pipeline import RealtimeVoiceConverter, RealtimeConfig


@pytest.fixture
def converter():
    """Create a realtime converter instance."""
    config = RealtimeConfig(
        sample_rate=22050,
        chunk_size_ms=100,
        overlap_ms=20,
        fp16=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    return RealtimeVoiceConverter(config)


@pytest.fixture
def sample_audio():
    """Generate synthetic audio for testing."""
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sr


@pytest.fixture
def speaker_embedding():
    """Generate synthetic speaker embedding."""
    return np.random.randn(256).astype(np.float32)


class TestRealtimeConfig:
    """Test configuration."""

    def test_default_config(self):
        """Test default values."""
        config = RealtimeConfig()
        assert config.sample_rate == 22050
        assert config.chunk_size_ms == 100
        assert config.fp16 is True


class TestRealtimeConverter:
    """Test converter."""

    def test_initialization(self, converter):
        """Test init."""
        assert converter is not None
        assert converter.chunk_samples == 2205

    def test_unload(self, converter):
        """Test unload."""
        converter.unload()
        assert converter._contentvec is None

    @pytest.mark.cuda
    @pytest.mark.slow
    def test_convert_full(self, converter, sample_audio, speaker_embedding):
        """Test conversion."""
        audio, sr = sample_audio
        converted, out_sr = converter.convert_full(audio, sr, speaker_embedding)
        assert len(converted) > 0
        assert out_sr == 22050


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
