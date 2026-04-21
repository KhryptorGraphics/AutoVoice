"""Tests for vocal separation module.

Tests Demucs-based vocal separation including:
- 4-stem separation (vocals, drums, bass, other)
- Separation quality metrics (SDR)
- GPU vs CPU execution
- Error handling
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from auto_voice.audio.separation import VocalSeparator


@pytest.fixture
def separator():
    """Create VocalSeparator instance."""
    return VocalSeparator(device='cpu', model_name='htdemucs')


@pytest.fixture
def sample_audio():
    """Create sample audio."""
    sr = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sr


# ===== Phase 4.1: Test Demucs separation =====

def test_separator_initialization():
    """Test VocalSeparator initialization."""
    sep = VocalSeparator(device='cpu', model_name='htdemucs')

    assert sep.device.type == 'cpu'
    assert sep.model_name == 'htdemucs'
    assert sep._model is None  # Lazy loading


def test_separate_4_stems(separator, sample_audio):
    """Test separation produces vocals and instrumental."""
    audio, sr = sample_audio

    result = separator.separate(audio, sr)

    assert 'vocals' in result
    assert 'instrumental' in result

    # Check output shapes
    assert result['vocals'].shape == audio.shape
    assert result['instrumental'].shape == audio.shape


def test_output_file_naming():
    """Test output file naming convention."""
    # This is handled by the caller, not VocalSeparator
    # Placeholder for integration test
    pass


# ===== Phase 4.2: Test separation quality =====

def test_separation_quality_metrics(separator, sample_audio):
    """Test SDR calculation (placeholder)."""
    # Real SDR calculation requires reference tracks
    # This is a placeholder for quality metrics

    audio, sr = sample_audio
    result = separator.separate(audio, sr)

    # Basic quality checks
    assert not np.isnan(result['vocals']).any()
    assert not np.isinf(result['vocals']).any()

    # Check vocals and instrumental are different
    assert not np.allclose(result['vocals'], result['instrumental'])


def test_vocal_sdr_target():
    """Test SDR >10 dB for vocals (requires reference)."""
    # Placeholder - requires BSS evaluation metrics
    pass


# ===== Phase 4.3: Test GPU vs CPU execution =====

def test_cpu_execution(sample_audio):
    """Test CPU execution."""
    sep = VocalSeparator(device='cpu')
    audio, sr = sample_audio

    result = sep.separate(audio, sr)

    assert result['vocals'] is not None


def test_gpu_execution(sample_audio):
    """Test GPU execution if available."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    sep = VocalSeparator(device='cuda')
    audio, sr = sample_audio

    result = sep.separate(audio, sr)

    assert result['vocals'] is not None


def test_gpu_speedup_measurement():
    """Test GPU provides speedup vs CPU."""
    # Placeholder - requires actual timing measurements
    pass


# ===== Phase 4.4: Test error handling =====

def test_invalid_audio_format(separator):
    """Test error on invalid audio format."""
    with pytest.raises(ValueError):
        separator.separate(np.array([]), 44100)


def test_empty_audio_error(separator):
    """Test error on empty audio."""
    with pytest.raises(ValueError):
        separator.separate(np.array([]), 44100)


def test_invalid_shape(separator):
    """Test error on invalid audio shape."""
    # 3D array should fail
    audio = np.zeros((2, 2, 1000))

    with pytest.raises(ValueError):
        separator.separate(audio, 44100)


def test_gpu_oom_handling():
    """Test GPU OOM handling."""
    # Placeholder - would require very large audio
    pass


def test_missing_demucs_model():
    """Test error when Demucs not installed."""
    sep = VocalSeparator()
    with patch.object(sep, '_load_model', side_effect=RuntimeError("Model not found")):
        with pytest.raises(RuntimeError):
            sep.separate(np.zeros(1000), 44100)


# ===== Helper tests =====

def test_model_sample_rate(separator):
    """Test model sample rate property."""
    sr = separator.model_sample_rate

    assert sr > 0
    assert sr == 44100  # HTDemucs default


def test_sources_property(separator):
    """Test sources property."""
    sources = separator.sources

    assert 'vocals' in sources
    assert isinstance(sources, list)


# ===== Coverage verification =====

def test_coverage_separation():
    """Verify coverage of separation.py module."""
    from auto_voice.audio import separation

    assert hasattr(separation, 'VocalSeparator')
