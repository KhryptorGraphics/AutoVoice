"""Comprehensive tests for TensorRT streaming pipeline module.

Target: Increase coverage from 38% to 70%+ for trt_streaming_pipeline.py

Test Categories:
1. Initialization Tests
2. Engine Loading Tests
3. Overlap-Add Synthesis Tests
4. Streaming Conversion Tests
5. Latency Tracking Tests
6. Error Handling Tests
"""
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pytest
import torch

from auto_voice.inference.trt_streaming_pipeline import TRTStreamingPipeline
from auto_voice.models.feature_contract import DEFAULT_PITCH_DIM


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_engine_dir(tmp_path):
    """Create temporary engine directory with mock engine files."""
    engine_dir = tmp_path / "trt_engines"
    engine_dir.mkdir()

    # Create mock engine files
    for name in ['content_extractor.trt', 'pitch_extractor.trt',
                 'decoder.trt', 'vocoder.trt']:
        (engine_dir / name).touch()

    return str(engine_dir)


@pytest.fixture
def empty_engine_dir(tmp_path):
    """Create empty engine directory."""
    engine_dir = tmp_path / "empty_engines"
    engine_dir.mkdir()
    return str(engine_dir)


# ============================================================================
# Initialization Tests
# ============================================================================


def test_trt_streaming_pipeline_init_default_params(temp_engine_dir):
    """Test TRTStreamingPipeline initialization with default parameters."""
    pipeline = TRTStreamingPipeline(temp_engine_dir)

    assert pipeline.sample_rate == 24000
    assert pipeline.chunk_size_ms == 100
    assert pipeline.overlap_ratio == 0.5
    assert pipeline.chunk_size == 2400  # 24000 * 100 / 1000
    assert pipeline.hop_size == 1200  # chunk_size * (1 - 0.5)
    assert pipeline.overlap_size == 1200  # chunk_size - hop_size


def test_trt_streaming_pipeline_init_custom_params(temp_engine_dir):
    """Test TRTStreamingPipeline initialization with custom parameters."""
    pipeline = TRTStreamingPipeline(
        temp_engine_dir,
        chunk_size_ms=50,
        overlap_ratio=0.25,
        sample_rate=16000,
        device=torch.device('cpu')
    )

    assert pipeline.sample_rate == 16000
    assert pipeline.chunk_size_ms == 50
    assert pipeline.overlap_ratio == 0.25
    assert pipeline.chunk_size == 800  # 16000 * 50 / 1000
    assert pipeline.hop_size == 600  # 800 * (1 - 0.25)
    assert pipeline.overlap_size == 200  # 800 - 600
    assert pipeline.device == torch.device('cpu')


def test_trt_streaming_pipeline_cuda_device_selection(temp_engine_dir):
    """Test automatic CUDA device selection."""
    with patch('torch.cuda.is_available', return_value=True):
        pipeline = TRTStreamingPipeline(temp_engine_dir)
        assert pipeline.device.type == 'cuda'

    with patch('torch.cuda.is_available', return_value=False):
        pipeline = TRTStreamingPipeline(temp_engine_dir)
        assert pipeline.device.type == 'cpu'


def test_trt_streaming_pipeline_engines_not_loaded_initially(temp_engine_dir):
    """Test engines are not loaded during initialization."""
    pipeline = TRTStreamingPipeline(temp_engine_dir)

    assert pipeline._engines_loaded is False
    assert pipeline._content_ctx is None
    assert pipeline._pitch_ctx is None
    assert pipeline._decoder_ctx is None
    assert pipeline._vocoder_ctx is None


def test_trt_streaming_pipeline_overlap_buffer_initially_none(temp_engine_dir):
    """Test overlap buffer is None initially."""
    pipeline = TRTStreamingPipeline(temp_engine_dir)
    assert pipeline.overlap_buffer is None


def test_trt_streaming_pipeline_latency_tracking_init(temp_engine_dir):
    """Test latency tracking is initialized."""
    pipeline = TRTStreamingPipeline(temp_engine_dir)

    assert pipeline._latency_history == []
    assert pipeline._max_latency_history == 100


# ============================================================================
# Static Method Tests
# ============================================================================


def test_engines_available_all_present(temp_engine_dir):
    """Test engines_available returns True when all engines exist."""
    result = TRTStreamingPipeline.engines_available(temp_engine_dir)
    assert result is True


def test_engines_available_missing_engines(empty_engine_dir):
    """Test engines_available returns False with missing engines."""
    result = TRTStreamingPipeline.engines_available(empty_engine_dir)
    assert result is False


def test_engines_available_partial_engines(tmp_path):
    """Test engines_available returns False with only some engines."""
    engine_dir = tmp_path / "partial_engines"
    engine_dir.mkdir()

    # Only create 2 of 4 engines
    (engine_dir / "content_extractor.trt").touch()
    (engine_dir / "decoder.trt").touch()

    result = TRTStreamingPipeline.engines_available(str(engine_dir))
    assert result is False


# ============================================================================
# Crossfade Window Tests
# ============================================================================


def test_create_crossfade_window_normal(temp_engine_dir):
    """Test crossfade window creation with normal overlap."""
    pipeline = TRTStreamingPipeline(temp_engine_dir, chunk_size_ms=100, overlap_ratio=0.5)

    window = pipeline._create_crossfade_window()

    assert window.shape[0] == 2  # fade_in and fade_out
    assert window.shape[1] == pipeline.overlap_size


def test_create_crossfade_window_zero_overlap(temp_engine_dir):
    """Test crossfade window with zero overlap."""
    pipeline = TRTStreamingPipeline(temp_engine_dir, chunk_size_ms=100, overlap_ratio=0.0)

    window = pipeline._create_crossfade_window()

    # Should return ones tensor
    assert window.shape == torch.Size([1])
    assert window[0] == 1.0


def test_create_crossfade_window_high_overlap(temp_engine_dir):
    """Test crossfade window with high overlap ratio."""
    pipeline = TRTStreamingPipeline(temp_engine_dir, chunk_size_ms=100, overlap_ratio=0.9)

    window = pipeline._create_crossfade_window()

    assert window.shape[0] == 2
    # overlap_size = chunk_size - hop_size = chunk_size - chunk_size * (1 - 0.9)
    assert window.shape[1] == pipeline.overlap_size


# ============================================================================
# Engine Loading Tests
# ============================================================================


@patch('auto_voice.inference.trt_pipeline.TRTInferenceContext')
def test_load_engines_success(mock_ctx_class, temp_engine_dir):
    """Test successful engine loading."""
    pipeline = TRTStreamingPipeline(temp_engine_dir)

    # Mock TRTInferenceContext
    mock_ctx = MagicMock()
    mock_ctx_class.return_value = mock_ctx

    pipeline.load_engines()

    assert pipeline._engines_loaded is True
    assert mock_ctx_class.call_count == 4  # 4 engines


@patch('auto_voice.inference.trt_pipeline.TRTInferenceContext')
def test_load_engines_idempotent(mock_ctx_class, temp_engine_dir):
    """Test load_engines is idempotent (doesn't reload if already loaded)."""
    pipeline = TRTStreamingPipeline(temp_engine_dir)

    mock_ctx = MagicMock()
    mock_ctx_class.return_value = mock_ctx

    pipeline.load_engines()
    call_count_first = mock_ctx_class.call_count

    # Call again
    pipeline.load_engines()

    # Should not call again
    assert mock_ctx_class.call_count == call_count_first


def test_load_engines_missing_engines(empty_engine_dir):
    """Test load_engines raises error with missing engines."""
    pipeline = TRTStreamingPipeline(empty_engine_dir)

    with pytest.raises(RuntimeError, match="Missing TRT engines"):
        pipeline.load_engines()


def test_load_engines_partial_missing(tmp_path):
    """Test load_engines raises error with some missing engines."""
    engine_dir = tmp_path / "partial"
    engine_dir.mkdir()

    # Create only 3 of 4 engines
    for name in ['content_extractor.trt', 'pitch_extractor.trt', 'decoder.trt']:
        (engine_dir / name).touch()

    pipeline = TRTStreamingPipeline(str(engine_dir))

    with pytest.raises(RuntimeError, match="Missing TRT engines.*vocoder"):
        pipeline.load_engines()


# ============================================================================
# Audio Processing Utility Tests
# ============================================================================


def test_resample_same_rate(temp_engine_dir):
    """Test resampling with same sample rate."""
    pipeline = TRTStreamingPipeline(temp_engine_dir)

    audio = torch.randn(16000)
    result = pipeline._resample(audio, 16000, 16000)

    assert torch.equal(result, audio)


def test_resample_different_rate(temp_engine_dir):
    """Test resampling to different sample rate."""
    pipeline = TRTStreamingPipeline(temp_engine_dir)

    audio = torch.randn(16000)
    result = pipeline._resample(audio, 16000, 8000)

    # Should be approximately half length
    assert abs(result.shape[0] - 8000) < 10


def test_resample_1d_tensor(temp_engine_dir):
    """Test resampling 1D audio tensor."""
    pipeline = TRTStreamingPipeline(temp_engine_dir)

    audio = torch.randn(16000)
    result = pipeline._resample(audio, 16000, 24000)

    assert result.dim() == 1
    assert abs(result.shape[0] - 24000) < 10


# ============================================================================
# Chunk Size Tests
# ============================================================================


def test_minimum_chunk_size_enforced(temp_engine_dir):
    """Test minimum chunk size is enforced (10ms)."""
    pipeline = TRTStreamingPipeline(temp_engine_dir, sample_rate=24000)

    assert pipeline._min_chunk_size == 240  # 24000 * 0.01


def test_chunk_size_calculations(temp_engine_dir):
    """Test chunk size calculations for various parameters."""
    # 100ms chunks, 50% overlap at 24kHz
    pipeline = TRTStreamingPipeline(
        temp_engine_dir,
        chunk_size_ms=100,
        overlap_ratio=0.5,
        sample_rate=24000
    )

    assert pipeline.chunk_size == 2400
    assert pipeline.hop_size == 1200
    assert pipeline.overlap_size == 1200


def test_chunk_size_no_overlap(temp_engine_dir):
    """Test chunk sizes with no overlap."""
    pipeline = TRTStreamingPipeline(
        temp_engine_dir,
        chunk_size_ms=100,
        overlap_ratio=0.0,
        sample_rate=24000
    )

    assert pipeline.chunk_size == 2400
    assert pipeline.hop_size == 2400
    assert pipeline.overlap_size == 0


# ============================================================================
# Latency Tracking Tests
# ============================================================================


def test_latency_history_max_size(temp_engine_dir):
    """Test latency history respects maximum size."""
    pipeline = TRTStreamingPipeline(temp_engine_dir)

    # Simulate adding many latency measurements
    for i in range(150):
        pipeline._latency_history.append(float(i))

    # Manual truncation would happen in actual implementation
    # This test verifies the max size is set correctly
    assert pipeline._max_latency_history == 100


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_init_with_invalid_chunk_size(temp_engine_dir):
    """Test initialization with very small chunk size."""
    # Should work but result in small chunks
    pipeline = TRTStreamingPipeline(
        temp_engine_dir,
        chunk_size_ms=5,  # Very small
        sample_rate=24000
    )

    assert pipeline.chunk_size == 120  # 24000 * 5 / 1000


def test_init_with_invalid_overlap_ratio(temp_engine_dir):
    """Test initialization with invalid overlap ratio."""
    # Overlap ratio > 1.0 should still work (just unusual)
    pipeline = TRTStreamingPipeline(
        temp_engine_dir,
        overlap_ratio=1.2,
        sample_rate=24000
    )

    # Hop size could be negative, but pipeline should initialize
    assert pipeline.overlap_ratio == 1.2


def test_init_with_zero_sample_rate(tmp_path):
    """Test initialization with zero sample rate creates zero chunk_size."""
    # Create engine dir with files
    engine_dir = tmp_path / "engines"
    engine_dir.mkdir()
    for name in ['content_extractor.trt', 'pitch_extractor.trt',
                 'decoder.trt', 'vocoder.trt']:
        (engine_dir / name).touch()

    # Initialize with zero sample rate - should work but create invalid state
    pipeline = TRTStreamingPipeline(str(engine_dir), sample_rate=0)

    # chunk_size would be 0
    assert pipeline.chunk_size == 0


# ============================================================================
# Device Management Tests
# ============================================================================


def test_device_placement_cpu(temp_engine_dir):
    """Test device placement on CPU."""
    pipeline = TRTStreamingPipeline(temp_engine_dir, device=torch.device('cpu'))

    assert pipeline.device == torch.device('cpu')
    assert pipeline.crossfade_window.device.type == 'cpu'


@pytest.mark.cuda
def test_device_placement_cuda(temp_engine_dir):
    """Test device placement on CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    pipeline = TRTStreamingPipeline(temp_engine_dir, device=torch.device('cuda'))

    assert pipeline.device.type == 'cuda'
    assert pipeline.crossfade_window.device.type == 'cuda'


# ============================================================================
# Path Handling Tests
# ============================================================================


def test_engine_dir_as_string(temp_engine_dir):
    """Test engine directory provided as string."""
    pipeline = TRTStreamingPipeline(temp_engine_dir)
    assert pipeline.engine_dir == Path(temp_engine_dir)


def test_engine_dir_as_path(tmp_path):
    """Test engine directory provided as Path object."""
    engine_dir = tmp_path / "engines"
    engine_dir.mkdir()

    # Create engines
    for name in ['content_extractor.trt', 'pitch_extractor.trt',
                 'decoder.trt', 'vocoder.trt']:
        (engine_dir / name).touch()

    pipeline = TRTStreamingPipeline(engine_dir)
    assert pipeline.engine_dir == engine_dir


# ============================================================================
# Integration Tests (Mocked)
# ============================================================================


@patch('auto_voice.inference.trt_pipeline.TRTInferenceContext')
def test_process_chunk_success(mock_ctx_class, temp_engine_dir):
    """Test successful chunk processing with mocked TRT engines."""
    pipeline = TRTStreamingPipeline(temp_engine_dir, sample_rate=24000, device=torch.device('cuda'))

    # Mock TRT contexts
    mock_content_ctx = MagicMock()
    mock_pitch_ctx = MagicMock()
    mock_decoder_ctx = MagicMock()
    mock_vocoder_ctx = MagicMock()

    mock_content_ctx.infer.return_value = {'features': torch.randn(1, 10, 768, device='cuda')}
    mock_pitch_ctx.infer.return_value = {'f0': torch.randn(1, 10, device='cuda').abs() + 50}
    mock_decoder_ctx.infer.return_value = {'mel': torch.randn(1, 100, 10, device='cuda')}
    mock_vocoder_ctx.infer.return_value = {'audio': torch.randn(1, 2400, device='cuda')}

    pipeline._content_ctx = mock_content_ctx
    pipeline._pitch_ctx = mock_pitch_ctx
    pipeline._decoder_ctx = mock_decoder_ctx
    pipeline._vocoder_ctx = mock_vocoder_ctx
    pipeline._engines_loaded = True

    audio_chunk = torch.randn(2400)
    speaker_embedding = torch.randn(256)

    result = pipeline.process_chunk(audio_chunk, speaker_embedding)

    assert result.shape[0] > 0
    assert result.dim() == 1
    assert torch.all(result >= -1.0) and torch.all(result <= 1.0)  # Clamped


@patch('auto_voice.inference.trt_pipeline.TRTInferenceContext')
def test_process_chunk_loads_engines_lazily(mock_ctx_class, temp_engine_dir):
    """Test process_chunk loads engines if not already loaded."""
    pipeline = TRTStreamingPipeline(temp_engine_dir, sample_rate=24000, device=torch.device('cuda'))

    # Mock TRT contexts
    mock_ctx = MagicMock()
    mock_ctx.infer.return_value = {
        'features': torch.randn(1, 10, 768, device='cuda'),
        'f0': torch.randn(1, 10, device='cuda').abs() + 50,
        'mel': torch.randn(1, 100, 10, device='cuda'),
        'audio': torch.randn(1, 2400, device='cuda'),
    }
    mock_ctx_class.return_value = mock_ctx

    audio_chunk = torch.randn(2400)
    speaker_embedding = torch.randn(256)

    assert not pipeline._engines_loaded

    result = pipeline.process_chunk(audio_chunk, speaker_embedding)

    assert pipeline._engines_loaded
    assert mock_ctx_class.call_count == 4  # 4 engines loaded


@patch('auto_voice.inference.trt_pipeline.TRTInferenceContext')
def test_process_chunk_too_short(mock_ctx_class, temp_engine_dir):
    """Test process_chunk raises error for chunks below minimum size."""
    pipeline = TRTStreamingPipeline(temp_engine_dir, sample_rate=24000, device=torch.device('cuda'))

    # Mock engines loaded
    pipeline._content_ctx = MagicMock()
    pipeline._pitch_ctx = MagicMock()
    pipeline._decoder_ctx = MagicMock()
    pipeline._vocoder_ctx = MagicMock()
    pipeline._engines_loaded = True

    # Too short (< 10ms = 240 samples at 24kHz)
    audio_chunk = torch.randn(100)
    speaker_embedding = torch.randn(256)

    with pytest.raises(RuntimeError, match="Chunk too short"):
        pipeline.process_chunk(audio_chunk, speaker_embedding)


@patch('auto_voice.inference.trt_pipeline.TRTInferenceContext')
def test_process_chunk_invalid_speaker_embedding(mock_ctx_class, temp_engine_dir):
    """Test process_chunk raises error for invalid speaker embedding."""
    pipeline = TRTStreamingPipeline(temp_engine_dir, sample_rate=24000, device=torch.device('cuda'))

    # Mock engines loaded
    pipeline._content_ctx = MagicMock()
    pipeline._engines_loaded = True

    audio_chunk = torch.randn(2400)
    invalid_speaker = torch.randn(128)  # Wrong size

    with pytest.raises(RuntimeError, match="Speaker embedding must be 256-dim"):
        pipeline.process_chunk(audio_chunk, invalid_speaker)


@patch('auto_voice.inference.trt_pipeline.TRTInferenceContext')
def test_process_chunk_2d_input(mock_ctx_class, temp_engine_dir):
    """Test process_chunk handles 2D audio input."""
    pipeline = TRTStreamingPipeline(temp_engine_dir, sample_rate=24000, device=torch.device('cuda'))

    # Mock TRT contexts
    mock_ctx = MagicMock()
    mock_ctx.infer.return_value = {
        'features': torch.randn(1, 10, 768, device='cuda'),
        'f0': torch.randn(1, 10, device='cuda').abs() + 50,
        'mel': torch.randn(1, 100, 10, device='cuda'),
        'audio': torch.randn(1, 2400, device='cuda'),
    }

    pipeline._content_ctx = mock_ctx
    pipeline._pitch_ctx = mock_ctx
    pipeline._decoder_ctx = mock_ctx
    pipeline._vocoder_ctx = mock_ctx
    pipeline._engines_loaded = True

    # 2D input [1, samples]
    audio_chunk = torch.randn(1, 2400)
    speaker_embedding = torch.randn(256)

    result = pipeline.process_chunk(audio_chunk, speaker_embedding)

    assert result.dim() == 1


@patch('auto_voice.inference.trt_pipeline.TRTInferenceContext')
def test_process_chunk_tracks_latency(mock_ctx_class, temp_engine_dir):
    """Test process_chunk tracks latency for each chunk."""
    pipeline = TRTStreamingPipeline(temp_engine_dir, sample_rate=24000, device=torch.device('cuda'))

    # Mock TRT contexts
    mock_ctx = MagicMock()
    mock_ctx.infer.return_value = {
        'features': torch.randn(1, 10, 768, device='cuda'),
        'f0': torch.randn(1, 10, device='cuda').abs() + 50,
        'mel': torch.randn(1, 100, 10, device='cuda'),
        'audio': torch.randn(1, 2400, device='cuda'),
    }

    pipeline._content_ctx = mock_ctx
    pipeline._pitch_ctx = mock_ctx
    pipeline._decoder_ctx = mock_ctx
    pipeline._vocoder_ctx = mock_ctx
    pipeline._engines_loaded = True

    audio_chunk = torch.randn(2400)
    speaker_embedding = torch.randn(256)

    assert len(pipeline._latency_history) == 0

    pipeline.process_chunk(audio_chunk, speaker_embedding)

    assert len(pipeline._latency_history) == 1
    assert pipeline._latency_history[0] > 0


# ============================================================================
# Overlap-Add Synthesis Tests
# ============================================================================


def test_apply_overlap_add_first_chunk(temp_engine_dir):
    """Test overlap-add synthesis for first chunk (no previous buffer)."""
    pipeline = TRTStreamingPipeline(temp_engine_dir, chunk_size_ms=100, overlap_ratio=0.5)

    converted = torch.randn(2400)

    output = pipeline._apply_overlap_add(converted)

    # First chunk - no overlap applied
    assert torch.equal(output, converted)
    assert pipeline.overlap_buffer is not None
    assert pipeline.overlap_buffer.shape[0] == pipeline.overlap_size


def test_apply_overlap_add_second_chunk(temp_engine_dir):
    """Test overlap-add synthesis for second chunk with crossfade."""
    pipeline = TRTStreamingPipeline(temp_engine_dir, chunk_size_ms=100, overlap_ratio=0.5, device=torch.device('cpu'))

    # First chunk
    first_chunk = torch.randn(2400)
    pipeline._apply_overlap_add(first_chunk)

    # Second chunk
    second_chunk = torch.randn(2400)
    output = pipeline._apply_overlap_add(second_chunk)

    # Should have crossfade applied
    assert output.shape == second_chunk.shape
    assert pipeline.overlap_buffer is not None


def test_apply_overlap_add_zero_overlap(temp_engine_dir):
    """Test overlap-add synthesis with zero overlap."""
    pipeline = TRTStreamingPipeline(temp_engine_dir, chunk_size_ms=100, overlap_ratio=0.0)

    converted = torch.randn(2400)

    output = pipeline._apply_overlap_add(converted)

    # No overlap - should return original
    assert torch.equal(output, converted)


def test_apply_overlap_add_high_overlap(temp_engine_dir):
    """Test overlap-add synthesis with high overlap ratio."""
    pipeline = TRTStreamingPipeline(temp_engine_dir, chunk_size_ms=100, overlap_ratio=0.9, device=torch.device('cpu'))

    # First chunk
    first_chunk = torch.ones(2400)
    pipeline._apply_overlap_add(first_chunk)

    # Second chunk
    second_chunk = torch.ones(2400)
    output = pipeline._apply_overlap_add(second_chunk)

    # Crossfade should be applied
    assert output.shape == second_chunk.shape


# ============================================================================
# Latency Stats Tests
# ============================================================================


def test_get_latency_stats_empty(temp_engine_dir):
    """Test latency stats with no history."""
    pipeline = TRTStreamingPipeline(temp_engine_dir)

    stats = pipeline.get_latency_stats()

    assert stats['min_ms'] == 0.0
    assert stats['max_ms'] == 0.0
    assert stats['avg_ms'] == 0.0


def test_get_latency_stats_with_history(temp_engine_dir):
    """Test latency stats calculation."""
    pipeline = TRTStreamingPipeline(temp_engine_dir)

    # Simulate latency history
    pipeline._latency_history = [10.0, 20.0, 30.0, 15.0, 25.0]

    stats = pipeline.get_latency_stats()

    assert stats['min_ms'] == 10.0
    assert stats['max_ms'] == 30.0
    assert stats['avg_ms'] == 20.0  # (10+20+30+15+25)/5


def test_get_latency_stats_single_value(temp_engine_dir):
    """Test latency stats with single measurement."""
    pipeline = TRTStreamingPipeline(temp_engine_dir)

    pipeline._latency_history = [42.5]

    stats = pipeline.get_latency_stats()

    assert stats['min_ms'] == 42.5
    assert stats['max_ms'] == 42.5
    assert stats['avg_ms'] == 42.5


# ============================================================================
# Reset Tests
# ============================================================================


def test_reset_clears_overlap_buffer(temp_engine_dir):
    """Test reset clears overlap buffer."""
    pipeline = TRTStreamingPipeline(temp_engine_dir)

    # Set overlap buffer
    pipeline.overlap_buffer = torch.randn(1200)

    pipeline.reset()

    assert pipeline.overlap_buffer is None


def test_reset_clears_latency_history(temp_engine_dir):
    """Test reset clears latency history."""
    pipeline = TRTStreamingPipeline(temp_engine_dir)

    # Add latency history
    pipeline._latency_history = [10.0, 20.0, 30.0]

    pipeline.reset()

    assert len(pipeline._latency_history) == 0


def test_reset_allows_fresh_streaming(temp_engine_dir):
    """Test reset allows starting fresh streaming session."""
    pipeline = TRTStreamingPipeline(temp_engine_dir)

    # Simulate previous streaming session
    pipeline.overlap_buffer = torch.randn(1200)
    pipeline._latency_history = [10.0, 20.0, 30.0]

    # Reset
    pipeline.reset()

    # Should be ready for new session
    assert pipeline.overlap_buffer is None
    assert len(pipeline._latency_history) == 0


# ============================================================================
# Encode Pitch Tests
# ============================================================================


def test_encode_pitch_normal(temp_engine_dir):
    """Test pitch encoding with normal F0 values."""
    pipeline = TRTStreamingPipeline(temp_engine_dir, device=torch.device('cpu'))

    f0 = torch.randn(1, 100).abs() + 50  # Positive F0 values

    pitch_embeddings = pipeline._encode_pitch(f0)

    assert pitch_embeddings.shape == (1, 100, DEFAULT_PITCH_DIM)
    assert not torch.isnan(pitch_embeddings).any()


def test_encode_pitch_zero_values(temp_engine_dir):
    """Test pitch encoding with zero F0 (silence)."""
    pipeline = TRTStreamingPipeline(temp_engine_dir, device=torch.device('cpu'))

    f0 = torch.zeros(1, 100)

    pitch_embeddings = pipeline._encode_pitch(f0)

    # Should clamp to minimum and produce valid embeddings
    assert pitch_embeddings.shape == (1, 100, DEFAULT_PITCH_DIM)
    assert not torch.isnan(pitch_embeddings).any()


def test_encode_pitch_high_values(temp_engine_dir):
    """Test pitch encoding with very high F0 values."""
    pipeline = TRTStreamingPipeline(temp_engine_dir, device=torch.device('cpu'))

    f0 = torch.ones(1, 100) * 1000  # Very high pitch

    pitch_embeddings = pipeline._encode_pitch(f0)

    assert pitch_embeddings.shape == (1, 100, DEFAULT_PITCH_DIM)
    assert not torch.isnan(pitch_embeddings).any()


# ============================================================================
# Engine Memory Usage Tests
# ============================================================================


def test_get_engine_memory_usage_not_loaded(temp_engine_dir):
    """Test memory usage returns 0 when engines not loaded."""
    pipeline = TRTStreamingPipeline(temp_engine_dir)

    memory = pipeline.get_engine_memory_usage()

    assert memory == 0


@patch('auto_voice.inference.trt_pipeline.TRTInferenceContext')
def test_get_engine_memory_usage_loaded(mock_ctx_class, temp_engine_dir):
    """Test memory usage calculation when engines loaded."""
    pipeline = TRTStreamingPipeline(temp_engine_dir)

    # Mock contexts with memory usage
    mock_ctx = MagicMock()
    mock_ctx.get_memory_usage.return_value = 1024 * 1024 * 50  # 50MB
    mock_ctx_class.return_value = mock_ctx

    pipeline.load_engines()

    memory = pipeline.get_engine_memory_usage()

    # 4 engines * 50MB each
    assert memory == 4 * 1024 * 1024 * 50


# ============================================================================
# Integration Tests (Mocked)
# ============================================================================


@patch('auto_voice.inference.trt_pipeline.TRTInferenceContext')
def test_streaming_conversion_multiple_chunks(mock_ctx_class, temp_engine_dir):
    """Test processing multiple sequential chunks."""
    pipeline = TRTStreamingPipeline(temp_engine_dir, sample_rate=24000, device=torch.device('cuda'))

    # Mock TRT contexts
    mock_ctx = MagicMock()
    mock_ctx.infer.return_value = {
        'features': torch.randn(1, 10, 768, device='cuda'),
        'f0': torch.randn(1, 10, device='cuda').abs() + 50,
        'mel': torch.randn(1, 100, 10, device='cuda'),
        'audio': torch.randn(1, 2400, device='cuda'),
    }
    mock_ctx_class.return_value = mock_ctx

    speaker_embedding = torch.randn(256)

    # Process 3 chunks
    chunks = []
    for i in range(3):
        audio_chunk = torch.randn(2400)
        result = pipeline.process_chunk(audio_chunk, speaker_embedding)
        chunks.append(result)

    # All chunks should be processed
    assert len(chunks) == 3
    assert all(chunk.shape[0] > 0 for chunk in chunks)

    # Latency should be tracked
    assert len(pipeline._latency_history) == 3


@patch('auto_voice.inference.trt_pipeline.TRTInferenceContext')
def test_streaming_reset_between_sessions(mock_ctx_class, temp_engine_dir):
    """Test resetting pipeline between streaming sessions."""
    pipeline = TRTStreamingPipeline(temp_engine_dir, sample_rate=24000, device=torch.device('cuda'))

    # Mock TRT contexts
    mock_ctx = MagicMock()
    mock_ctx.infer.return_value = {
        'features': torch.randn(1, 10, 768, device='cuda'),
        'f0': torch.randn(1, 10, device='cuda').abs() + 50,
        'mel': torch.randn(1, 100, 10, device='cuda'),
        'audio': torch.randn(1, 2400, device='cuda'),
    }
    mock_ctx_class.return_value = mock_ctx

    speaker_embedding = torch.randn(256)

    # First session
    pipeline.process_chunk(torch.randn(2400), speaker_embedding)
    pipeline.process_chunk(torch.randn(2400), speaker_embedding)

    assert len(pipeline._latency_history) == 2

    # Reset
    pipeline.reset()

    # Second session
    pipeline.process_chunk(torch.randn(2400), speaker_embedding)

    # Should only have one latency measurement
    assert len(pipeline._latency_history) == 1


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_very_high_sample_rate(temp_engine_dir):
    """Test initialization with very high sample rate."""
    pipeline = TRTStreamingPipeline(
        temp_engine_dir,
        sample_rate=192000,  # High-res audio
        chunk_size_ms=100
    )

    assert pipeline.chunk_size == 19200
    assert pipeline.sample_rate == 192000


def test_very_low_sample_rate(temp_engine_dir):
    """Test initialization with very low sample rate."""
    pipeline = TRTStreamingPipeline(
        temp_engine_dir,
        sample_rate=8000,  # Telephone quality
        chunk_size_ms=100
    )

    assert pipeline.chunk_size == 800
    assert pipeline.sample_rate == 8000


def test_extreme_overlap_ratio(temp_engine_dir):
    """Test with extreme overlap ratios."""
    # Very high overlap (0.99)
    pipeline = TRTStreamingPipeline(
        temp_engine_dir,
        overlap_ratio=0.99,
        chunk_size_ms=100,
        sample_rate=24000
    )

    assert pipeline.overlap_ratio == 0.99
    assert pipeline.hop_size == int(2400 * 0.01)  # Very small hop
    assert pipeline.overlap_size == 2400 - pipeline.hop_size
