#!/usr/bin/env python3
"""Direct test of voice_conversion_pipeline module (bypassing pytest).

This script directly tests the pipeline without pytest to avoid scipy issues.
"""

import sys
from pathlib import Path

# Direct import of the module
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import numpy as np
import torch

# Import directly from the module file
from auto_voice.inference.voice_conversion_pipeline import (
    VoiceConversionPipeline,
    PipelineConfig,
    VoiceConversionError
)


def test_config():
    """Test PipelineConfig."""
    print("Testing PipelineConfig...")
    config = PipelineConfig()
    assert config.sample_rate == 22050
    assert config.use_cuda is True
    print("✓ PipelineConfig works")


def test_initialization():
    """Test pipeline initialization."""
    print("\nTesting pipeline initialization...")
    config = PipelineConfig(use_cuda=False)  # Force CPU
    pipeline = VoiceConversionPipeline(config)
    assert pipeline is not None
    assert pipeline.device.type == 'cpu'
    print("✓ Pipeline initialization works")


def test_conversion():
    """Test basic conversion."""
    print("\nTesting conversion...")
    config = PipelineConfig(use_cuda=False, fallback_on_error=True)
    pipeline = VoiceConversionPipeline(config)

    # Create test data
    audio = np.random.randn(22050).astype(np.float32)
    embedding = np.random.randn(256).astype(np.float32)

    result = pipeline.convert(audio, embedding)
    assert result is not None
    assert isinstance(result, np.ndarray)
    print(f"✓ Conversion works (output shape: {result.shape})")


def test_profiling():
    """Test profiling."""
    print("\nTesting profiling...")
    config = PipelineConfig(use_cuda=False, fallback_on_error=True)
    pipeline = VoiceConversionPipeline(config)

    audio = np.random.randn(22050).astype(np.float32)
    embedding = np.random.randn(256).astype(np.float32)

    metrics = pipeline.profile_conversion(audio, embedding)
    assert metrics is not None
    assert 'total_ms' in metrics
    assert 'rtf' in metrics
    print(f"✓ Profiling works (RTF: {metrics['rtf']:.3f}x, Total: {metrics['total_ms']:.1f}ms)")


def test_batch():
    """Test batch conversion."""
    print("\nTesting batch conversion...")
    config = PipelineConfig(use_cuda=False, fallback_on_error=True)
    pipeline = VoiceConversionPipeline(config)

    audio_list = [np.random.randn(11025).astype(np.float32) for _ in range(3)]
    embeddings = [np.random.randn(256).astype(np.float32) for _ in range(3)]

    results = pipeline.batch_convert(audio_list, embeddings)
    assert len(results) == 3
    print(f"✓ Batch conversion works (processed {len(results)} items)")


def test_stats():
    """Test statistics."""
    print("\nTesting statistics...")
    config = PipelineConfig(use_cuda=False, fallback_on_error=True)
    pipeline = VoiceConversionPipeline(config)

    audio = np.random.randn(11025).astype(np.float32)
    embedding = np.random.randn(256).astype(np.float32)

    pipeline.convert(audio, embedding)
    stats = pipeline.get_stats()

    assert stats['total_conversions'] >= 1
    assert 'device' in stats
    print(f"✓ Statistics work (conversions: {stats['total_conversions']}, device: {stats['device']})")


def test_warmup():
    """Test warmup."""
    print("\nTesting warmup...")
    config = PipelineConfig(use_cuda=False, fallback_on_error=True)
    pipeline = VoiceConversionPipeline(config)

    pipeline.warmup(num_iterations=2)
    assert pipeline.stats['total_conversions'] >= 2
    print(f"✓ Warmup works (ran {pipeline.stats['total_conversions']} warmup iterations)")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Direct Testing of VoiceConversionPipeline")
    print("=" * 60)

    tests = [
        test_config,
        test_initialization,
        test_conversion,
        test_profiling,
        test_batch,
        test_stats,
        test_warmup
    ]

    failed = []
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            failed.append((test_func.__name__, e))
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {len(tests) - len(failed)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed tests:")
        for name, error in failed:
            print(f"  - {name}: {error}")
        return 1
    else:
        print("\n✓ All tests passed!")
        return 0


if __name__ == '__main__':
    sys.exit(main())
