"""
Tests for MeanVC streaming pipeline.

Validates:
1. Pipeline initialization and model loading
2. Reference audio setting
3. Chunk processing latency (<100ms target)
4. Full audio conversion
5. Streaming session management
"""

import pytest
import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from auto_voice.inference.meanvc_pipeline import MeanVCPipeline
from auto_voice.inference.pipeline_factory import PipelineFactory


@pytest.fixture
def sample_audio():
    """Generate sample audio for testing."""
    # 1 second of audio at 16kHz
    duration = 1.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Simple sine wave
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sample_rate


@pytest.fixture
def reference_audio():
    """Generate reference audio for voice cloning."""
    # 3 seconds of reference at 16kHz
    duration = 3.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Different frequency for reference
    audio = np.sin(2 * np.pi * 330 * t).astype(np.float32) * 0.8
    return audio, sample_rate


@pytest.mark.smoke
def test_meanvc_initialization():
    """Test MeanVC pipeline initializes correctly."""
    pipeline = MeanVCPipeline(device=torch.device('cpu'), steps=2)

    assert pipeline.device.type == 'cpu', "Should use CPU device"
    assert pipeline.steps == 2, "Should use 2 steps"
    assert pipeline.sample_rate == 16000, "Sample rate should be 16kHz"
    assert pipeline.chunk_size == 3200, "Chunk size should be 200ms (3200 samples)"
    assert not pipeline._initialized, "Models should not be loaded yet (lazy loading)"


@pytest.mark.smoke
def test_pipeline_factory_registration():
    """Test that MeanVC is registered in PipelineFactory."""
    factory = PipelineFactory.get_instance()

    # Check it's in the valid pipeline types
    status = factory.get_status()
    assert 'realtime_meanvc' in status, "MeanVC should be in factory status"

    meanvc_status = status['realtime_meanvc']
    assert meanvc_status['sample_rate'] == 16000
    assert meanvc_status['latency_target_ms'] == 80
    assert 'single-step' in meanvc_status['description'].lower()


@pytest.mark.smoke
def test_chunk_size_calculation():
    """Test chunk size calculation for streaming."""
    pipeline = MeanVCPipeline(device=torch.device('cpu'))

    # Verify chunk parameters
    assert pipeline._decoding_chunk_size == 5, "Decoding chunk should be 5 frames"
    assert pipeline._subsampling == 4, "Subsampling should be 4"
    assert pipeline._stride == 20, "Stride should be 20 (5 * 4)"

    # Chunk size: 160 samples/frame * 20 frames = 3200 samples
    assert pipeline.chunk_size == 3200, "Chunk size should be 3200 samples"

    # Duration: 3200 / 16000 = 0.2s = 200ms
    chunk_duration_ms = (pipeline.chunk_size / pipeline.sample_rate) * 1000
    assert abs(chunk_duration_ms - 200) < 1, "Chunk duration should be ~200ms"


@pytest.mark.integration
@pytest.mark.skipif(
    not Path(__file__).parent.parent.joinpath("models/meanvc/src/ckpt").exists(),
    reason="MeanVC models not downloaded"
)
def test_reference_audio_setting(reference_audio):
    """Test setting reference audio."""
    pipeline = MeanVCPipeline(device=torch.device('cpu'), steps=2)
    audio, sr = reference_audio

    # Set reference
    pipeline.set_reference_audio(audio, sr)

    # Verify reference was set
    assert pipeline._spk_emb is not None, "Speaker embedding should be extracted"
    assert pipeline._prompt_mel is not None, "Prompt mel should be extracted"
    assert pipeline._initialized, "Models should be loaded after setting reference"

    # Check embedding shapes
    assert pipeline._spk_emb.shape == (1, 256), f"Speaker embedding shape: {pipeline._spk_emb.shape}"
    assert pipeline._prompt_mel.shape[0] == 1, "Prompt mel batch dimension"
    assert pipeline._prompt_mel.shape[2] == 80, "Prompt mel should have 80 mel bins"


@pytest.mark.integration
@pytest.mark.skipif(
    not Path(__file__).parent.parent.joinpath("models/meanvc/src/ckpt").exists(),
    reason="MeanVC models not downloaded"
)
def test_chunk_processing_latency(reference_audio, sample_audio):
    """Test chunk processing latency (<100ms target)."""
    import time

    pipeline = MeanVCPipeline(device=torch.device('cpu'), steps=2)
    ref_audio, ref_sr = reference_audio
    pipeline.set_reference_audio(ref_audio, ref_sr)

    # Generate a chunk
    chunk = np.random.randn(pipeline.chunk_size).astype(np.float32) * 0.1

    # Process and measure latency
    start = time.perf_counter()
    output = pipeline.process_chunk(chunk)
    latency = (time.perf_counter() - start) * 1000  # ms

    # Verify output
    assert output is not None, "Should return output"
    assert isinstance(output, np.ndarray), "Output should be numpy array"
    assert output.dtype == np.float32, "Output should be float32"

    # Check latency (relaxed for CI/CPU)
    print(f"Chunk latency: {latency:.1f}ms")
    assert latency < 500, f"Latency {latency:.1f}ms too high (target <100ms on GPU, <500ms on CPU)"

    # Get metrics
    metrics = pipeline.get_latency_metrics()
    print(f"Latency breakdown: {metrics}")
    assert 'total_ms' in metrics
    assert metrics['total_ms'] > 0


@pytest.mark.integration
@pytest.mark.skipif(
    not Path(__file__).parent.parent.joinpath("models/meanvc/src/ckpt").exists(),
    reason="MeanVC models not downloaded"
)
def test_full_audio_conversion(reference_audio, sample_audio):
    """Test full audio conversion (non-streaming)."""
    pipeline = MeanVCPipeline(device=torch.device('cpu'), steps=2)
    ref_audio, ref_sr = reference_audio
    pipeline.set_reference_audio(ref_audio, ref_sr)

    audio, sr = sample_audio

    # Convert
    result = pipeline.convert(audio, sr)

    # Verify result structure
    assert 'audio' in result
    assert 'sample_rate' in result
    assert 'metadata' in result

    # Check audio output
    output_audio = result['audio']
    assert isinstance(output_audio, torch.Tensor)
    assert output_audio.dim() == 1, "Output should be 1D (mono)"
    assert result['sample_rate'] == 16000

    # Check metadata
    metadata = result['metadata']
    assert 'processing_time' in metadata
    assert 'steps' in metadata
    assert metadata['steps'] == 2
    assert metadata['pipeline'] == 'meanvc'


@pytest.mark.integration
@pytest.mark.skipif(
    not Path(__file__).parent.parent.joinpath("models/meanvc/src/ckpt").exists(),
    reason="MeanVC models not downloaded"
)
def test_streaming_session_reset(reference_audio):
    """Test streaming session can be reset."""
    pipeline = MeanVCPipeline(device=torch.device('cpu'), steps=2)
    ref_audio, ref_sr = reference_audio
    pipeline.set_reference_audio(ref_audio, ref_sr)

    # Process some chunks
    chunk = np.random.randn(pipeline.chunk_size).astype(np.float32) * 0.1
    output1 = pipeline.process_chunk(chunk)

    # Reset session
    pipeline.reset_session()

    # Process again (should work)
    output2 = pipeline.process_chunk(chunk)

    assert output1 is not None
    assert output2 is not None
    assert output1.shape == output2.shape


@pytest.mark.integration
@pytest.mark.skipif(
    not Path(__file__).parent.parent.joinpath("models/meanvc/src/ckpt").exists(),
    reason="MeanVC models not downloaded"
)
def test_pipeline_factory_creates_meanvc():
    """Test PipelineFactory can create MeanVC pipeline."""
    factory = PipelineFactory.get_instance()

    # Get MeanVC pipeline
    pipeline = factory.get_pipeline('realtime_meanvc')

    assert isinstance(pipeline, MeanVCPipeline)
    assert pipeline.device.type == 'cpu', "MeanVC should use CPU"
    assert pipeline.steps == 2

    # Check it's cached
    assert factory.is_loaded('realtime_meanvc')

    # Get again (should return cached)
    pipeline2 = factory.get_pipeline('realtime_meanvc')
    assert pipeline2 is pipeline, "Should return same instance"

    # Cleanup
    factory.unload_pipeline('realtime_meanvc')


@pytest.mark.smoke
def test_metrics_collection():
    """Test pipeline collects latency metrics."""
    pipeline = MeanVCPipeline(device=torch.device('cpu'), steps=2)

    # Get metrics before any processing
    metrics = pipeline.get_metrics()

    assert 'device' in metrics
    assert 'sample_rate' in metrics
    assert 'steps' in metrics
    assert 'chunk_size_ms' in metrics
    assert 'has_reference' in metrics

    assert metrics['device'] == 'cpu'
    assert metrics['sample_rate'] == 16000
    assert metrics['steps'] == 2
    assert abs(metrics['chunk_size_ms'] - 200) < 1  # 200ms chunks
    assert not metrics['has_reference']  # No reference set yet


@pytest.mark.smoke
@pytest.mark.skipif(
    True,  # Skip until S3PRL torchaudio compatibility is fixed
    reason="S3PRL uses deprecated torchaudio.set_audio_backend API"
)
def test_resampling_reference():
    """Test reference audio resampling."""
    pipeline = MeanVCPipeline(device=torch.device('cpu'))

    # Create 22kHz audio (needs resampling to 16kHz)
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # This should automatically resample
    # Will only work if models are available
    try:
        pipeline.set_reference_audio(audio, sr)
        assert pipeline._reference_audio.shape[-1] == int(16000 * duration)
    except FileNotFoundError:
        pytest.skip("MeanVC models not available")


if __name__ == "__main__":
    # Run smoke tests
    print("Testing MeanVC initialization...")
    test_meanvc_initialization()
    print("✓ Initialization test passed")

    print("\nTesting pipeline factory registration...")
    test_pipeline_factory_registration()
    print("✓ Factory registration test passed")

    print("\nTesting chunk size calculation...")
    test_chunk_size_calculation()
    print("✓ Chunk size test passed")

    print("\nTesting metrics collection...")
    test_metrics_collection()
    print("✓ Metrics test passed")

    print("\n✅ All smoke tests passed!")
    print("\nRun integration tests with: pytest tests/test_meanvc_streaming.py -v -m integration")
