"""Standalone tests for VoiceConversionPipeline (minimal dependencies).

These tests avoid importing the full src package to prevent scipy/dependency issues.
"""

import sys
from pathlib import Path

# Add src to path without importing the package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import numpy as np
import torch


def test_pipeline_config_import():
    """Test that PipelineConfig can be imported."""
    from src.auto_voice.inference.voice_conversion_pipeline import PipelineConfig

    config = PipelineConfig()
    assert config.sample_rate == 22050
    assert config.use_cuda is True


def test_pipeline_basic_initialization():
    """Test basic pipeline initialization."""
    from src.auto_voice.inference.voice_conversion_pipeline import (
        VoiceConversionPipeline,
        PipelineConfig
    )

    config = PipelineConfig(use_cuda=False)  # Force CPU to avoid GPU dependency
    pipeline = VoiceConversionPipeline(config)

    assert pipeline is not None
    assert pipeline.device.type == 'cpu'


def test_pipeline_convert_basic():
    """Test basic conversion (will use fallback)."""
    from src.auto_voice.inference.voice_conversion_pipeline import (
        VoiceConversionPipeline,
        PipelineConfig
    )

    config = PipelineConfig(use_cuda=False, fallback_on_error=True)
    pipeline = VoiceConversionPipeline(config)

    # Create test data
    sample_rate = 22050
    duration = 1.0
    audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
    embedding = np.random.randn(256).astype(np.float32)

    # This should succeed with fallback
    result = pipeline.convert(audio, embedding)

    assert result is not None
    assert isinstance(result, np.ndarray)


def test_pipeline_stats():
    """Test statistics collection."""
    from src.auto_voice.inference.voice_conversion_pipeline import (
        VoiceConversionPipeline,
        PipelineConfig
    )

    config = PipelineConfig(use_cuda=False, fallback_on_error=True)
    pipeline = VoiceConversionPipeline(config)

    audio = np.random.randn(22050).astype(np.float32)
    embedding = np.random.randn(256).astype(np.float32)

    # Run conversion
    pipeline.convert(audio, embedding)

    # Check stats
    stats = pipeline.get_stats()
    assert stats['total_conversions'] >= 1
    assert stats['device'] == 'cpu'


def test_pipeline_profile_conversion():
    """Test profiling functionality."""
    from src.auto_voice.inference.voice_conversion_pipeline import (
        VoiceConversionPipeline,
        PipelineConfig
    )

    config = PipelineConfig(use_cuda=False, fallback_on_error=True)
    pipeline = VoiceConversionPipeline(config)

    audio = np.random.randn(22050).astype(np.float32)
    embedding = np.random.randn(256).astype(np.float32)

    # Profile conversion
    metrics = pipeline.profile_conversion(audio, embedding)

    assert metrics is not None
    assert 'total_ms' in metrics
    assert 'audio_duration_s' in metrics
    assert 'rtf' in metrics
    assert metrics['total_ms'] > 0
    assert metrics['audio_duration_s'] > 0


def test_pipeline_batch_convert():
    """Test batch conversion."""
    from src.auto_voice.inference.voice_conversion_pipeline import (
        VoiceConversionPipeline,
        PipelineConfig
    )

    config = PipelineConfig(use_cuda=False, fallback_on_error=True)
    pipeline = VoiceConversionPipeline(config)

    # Create batch data
    audio_list = [
        np.random.randn(22050).astype(np.float32),
        np.random.randn(22050).astype(np.float32)
    ]
    embeddings = [
        np.random.randn(256).astype(np.float32),
        np.random.randn(256).astype(np.float32)
    ]

    results = pipeline.batch_convert(audio_list, embeddings)

    assert len(results) == 2
    for result in results:
        assert isinstance(result, np.ndarray)


def test_pipeline_warmup():
    """Test pipeline warmup."""
    from src.auto_voice.inference.voice_conversion_pipeline import (
        VoiceConversionPipeline,
        PipelineConfig
    )

    config = PipelineConfig(use_cuda=False, fallback_on_error=True)
    pipeline = VoiceConversionPipeline(config)

    # Warmup should not crash
    pipeline.warmup(num_iterations=2)

    # Stats should reflect warmup runs
    assert pipeline.stats['total_conversions'] >= 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pipeline_cuda_initialization():
    """Test CUDA initialization when available."""
    from src.auto_voice.inference.voice_conversion_pipeline import (
        VoiceConversionPipeline,
        PipelineConfig
    )

    config = PipelineConfig(use_cuda=True)
    pipeline = VoiceConversionPipeline(config)

    assert pipeline.device.type == 'cuda'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
