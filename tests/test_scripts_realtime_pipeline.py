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

from realtime_pipeline import RealtimeVoiceConverter, RealtimeConfig, load_speaker_embedding


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


def test_load_speaker_embedding_uses_data_dir_env(monkeypatch, tmp_path):
    profile_id = "profile-123"
    data_dir = tmp_path / "runtime-data"
    profiles_dir = data_dir / "voice_profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    expected = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    np.save(profiles_dir / f"{profile_id}.npy", expected)

    monkeypatch.setenv("DATA_DIR", str(data_dir))

    loaded = load_speaker_embedding(profile_id)

    assert np.array_equal(loaded, expected)


def test_load_speaker_embedding_supports_explicit_data_dir(tmp_path):
    profile_id = "profile-456"
    data_dir = tmp_path / "custom-data"
    profiles_dir = data_dir / "voice_profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.save(profiles_dir / f"{profile_id}.npy", expected)

    loaded = load_speaker_embedding(profile_id, data_dir=str(data_dir))

    assert np.array_equal(loaded, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
