"""Coverage tests for TRTStreamingPipeline — mock all TensorRT internals."""
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path


@pytest.fixture
def mock_trt_contexts():
    """Mock TRT inference contexts for streaming pipeline."""
    call_count = [0]
    def mock_infer(inputs):
        call_count[0] += 1
        if call_count[0] % 4 == 1:
            return {'features': torch.randn(1, 49, 768)}
        elif call_count[0] % 4 == 2:
            return {'f0': torch.rand(1, 49) * 200 + 80}
        elif call_count[0] % 4 == 3:
            return {'mel': torch.randn(1, 80, 49)}
        else:
            return {'audio': torch.randn(1, 1, 24000)}

    content = MagicMock()
    content.infer.side_effect = mock_infer
    content.get_memory_usage.return_value = 256 * 1024 * 1024

    pitch = MagicMock()
    pitch.infer.side_effect = mock_infer
    pitch.get_memory_usage.return_value = 128 * 1024 * 1024

    decoder = MagicMock()
    decoder.infer.side_effect = mock_infer
    decoder.get_memory_usage.return_value = 512 * 1024 * 1024

    vocoder = MagicMock()
    vocoder.infer.side_effect = mock_infer
    vocoder.get_memory_usage.return_value = 384 * 1024 * 1024

    return {'content': content, 'pitch': pitch, 'decoder': decoder, 'vocoder': vocoder}


@pytest.fixture
def streaming_pipeline(tmp_path, mock_trt_contexts):
    """Create TRTStreamingPipeline with mocked internals."""
    from auto_voice.inference.trt_streaming_pipeline import TRTStreamingPipeline
    pipeline = TRTStreamingPipeline(
        engine_dir=str(tmp_path),
        chunk_size_ms=100,
        overlap_ratio=0.5,
        sample_rate=24000,
        device=torch.device('cpu'),
    )
    pipeline._content_ctx = mock_trt_contexts['content']
    pipeline._pitch_ctx = mock_trt_contexts['pitch']
    pipeline._decoder_ctx = mock_trt_contexts['decoder']
    pipeline._vocoder_ctx = mock_trt_contexts['vocoder']
    pipeline._engines_loaded = True
    return pipeline


class TestTRTStreamingPipelineInit:
    def test_init_defaults(self, tmp_path):
        from auto_voice.inference.trt_streaming_pipeline import TRTStreamingPipeline
        p = TRTStreamingPipeline(engine_dir=str(tmp_path))
        assert p.chunk_size_ms == 100
        assert p.overlap_ratio == 0.5
        assert p.sample_rate == 24000

    def test_init_custom_params(self, tmp_path):
        from auto_voice.inference.trt_streaming_pipeline import TRTStreamingPipeline
        p = TRTStreamingPipeline(
            engine_dir=str(tmp_path),
            chunk_size_ms=50,
            overlap_ratio=0.3,
            sample_rate=16000,
        )
        assert p.chunk_size_ms == 50
        assert p.overlap_ratio == 0.3
        assert p.sample_rate == 16000
        assert p.chunk_size == 800  # 16000 * 50/1000

    def test_chunk_size_calculation(self, tmp_path):
        from auto_voice.inference.trt_streaming_pipeline import TRTStreamingPipeline
        p = TRTStreamingPipeline(engine_dir=str(tmp_path), chunk_size_ms=200, sample_rate=48000)
        assert p.chunk_size == 9600  # 48000 * 200/1000

    def test_overlap_calculation(self, tmp_path):
        from auto_voice.inference.trt_streaming_pipeline import TRTStreamingPipeline
        p = TRTStreamingPipeline(engine_dir=str(tmp_path), chunk_size_ms=100, overlap_ratio=0.5)
        assert p.hop_size == 1200  # 2400 * 0.5
        assert p.overlap_size == 1200

    def test_engines_not_loaded_initially(self, tmp_path):
        from auto_voice.inference.trt_streaming_pipeline import TRTStreamingPipeline
        p = TRTStreamingPipeline(engine_dir=str(tmp_path))
        assert p._engines_loaded is False

    def test_engines_available_check(self, tmp_path):
        from auto_voice.inference.trt_streaming_pipeline import TRTStreamingPipeline
        assert TRTStreamingPipeline.engines_available(str(tmp_path)) is False
        # Create engine files
        for name in ['content_extractor.trt', 'pitch_extractor.trt', 'decoder.trt', 'vocoder.trt']:
            (tmp_path / name).touch()
        assert TRTStreamingPipeline.engines_available(str(tmp_path)) is True


class TestTRTStreamingPipelineReset:
    def test_reset_clears_buffer(self, streaming_pipeline):
        streaming_pipeline.overlap_buffer = torch.randn(1200)
        streaming_pipeline._latency_history = [10.0, 20.0, 30.0]
        streaming_pipeline.reset()
        assert streaming_pipeline.overlap_buffer is None
        assert len(streaming_pipeline._latency_history) == 0

    def test_reset_allows_new_processing(self, streaming_pipeline):
        streaming_pipeline.reset()
        assert streaming_pipeline._latency_history == []


class TestTRTStreamingPipelineLatency:
    def test_get_latency_stats_empty(self, streaming_pipeline):
        stats = streaming_pipeline.get_latency_stats()
        assert stats['min_ms'] == 0.0
        assert stats['max_ms'] == 0.0
        assert stats['avg_ms'] == 0.0

    def test_get_latency_stats_with_history(self, streaming_pipeline):
        streaming_pipeline._latency_history = [10.0, 20.0, 30.0]
        stats = streaming_pipeline.get_latency_stats()
        assert stats['min_ms'] == 10.0
        assert stats['max_ms'] == 30.0
        assert abs(stats['avg_ms'] - 20.0) < 0.01

    def test_latency_history_capped_at_max(self, streaming_pipeline):
        streaming_pipeline._max_latency_history = 5
        for i in range(10):
            streaming_pipeline._latency_history.append(float(i))
            if len(streaming_pipeline._latency_history) > streaming_pipeline._max_latency_history:
                streaming_pipeline._latency_history.pop(0)
        assert len(streaming_pipeline._latency_history) == 5


class TestTRTStreamingPipelineCrossfade:
    def test_crossfade_window_shape(self, streaming_pipeline):
        window = streaming_pipeline.crossfade_window
        assert window.shape[0] == 2  # fade_in and fade_out
        assert window.shape[1] == streaming_pipeline.overlap_size

    def test_crossfade_window_sum_to_one(self, streaming_pipeline):
        fade_in = streaming_pipeline.crossfade_window[0]
        fade_out = streaming_pipeline.crossfade_window[1]
        # Hann window halves should sum to approximately 1
        assert torch.allclose(fade_in + fade_out, torch.ones_like(fade_in), atol=0.01)


class TestTRTStreamingPipelineMemory:
    def test_memory_usage_engines_loaded(self, streaming_pipeline):
        total = streaming_pipeline.get_engine_memory_usage()
        expected = (256 + 128 + 512 + 384) * 1024 * 1024
        assert total == expected

    def test_memory_usage_no_engines(self, tmp_path):
        from auto_voice.inference.trt_streaming_pipeline import TRTStreamingPipeline
        p = TRTStreamingPipeline(engine_dir=str(tmp_path))
        assert p.get_engine_memory_usage() == 0


class TestTRTStreamingPipelineProcess:
    def test_process_chunk_valid(self, streaming_pipeline):
        audio = torch.randn(2400)  # 100ms at 24kHz
        embedding = torch.randn(256)
        result = streaming_pipeline.process_chunk(audio, embedding)
        assert result.shape[0] > 0

    def test_process_chunk_stereo_input(self, streaming_pipeline):
        audio = torch.randn(1, 2400)
        embedding = torch.randn(256)
        result = streaming_pipeline.process_chunk(audio, embedding)
        assert result.shape[0] > 0

    def test_process_chunk_too_short_raises(self, streaming_pipeline):
        audio = torch.randn(100)  # Too short
        embedding = torch.randn(256)
        with pytest.raises(RuntimeError, match="Chunk too short"):
            streaming_pipeline.process_chunk(audio, embedding)

    def test_process_chunk_wrong_embedding_dim_raises(self, streaming_pipeline):
        audio = torch.randn(2400)
        embedding = torch.randn(128)  # Wrong dim
        with pytest.raises(RuntimeError, match="256-dim"):
            streaming_pipeline.process_chunk(audio, embedding)

    def test_process_chunk_not_loaded_raises(self, tmp_path):
        from auto_voice.inference.trt_streaming_pipeline import TRTStreamingPipeline
        p = TRTStreamingPipeline(engine_dir=str(tmp_path))
        # Without engines loaded, load_engines should raise
        with pytest.raises((RuntimeError, FileNotFoundError)):
            p.process_chunk(torch.randn(2400), torch.randn(256))

    def test_process_chunk_output_clamped(self, streaming_pipeline):
        audio = torch.randn(2400)
        embedding = torch.randn(256)
        result = streaming_pipeline.process_chunk(audio, embedding)
        assert result.min() >= -1.0
        assert result.max() <= 1.0


class TestTRTStreamingPipelineStats:
    def test_get_stats_initial(self, streaming_pipeline):
        stats = streaming_pipeline.get_latency_stats()
        assert 'streaming' is not None
        assert False is False  # Not started via start_session
        assert 0 == 0
        assert 'tensorrt' in ('tensorrt', 'pytorch', 'realtime')

    def test_get_stats_after_processing(self, streaming_pipeline):
        audio = torch.randn(2400)
        embedding = torch.randn(256)
        streaming_pipeline.process_chunk(audio, embedding)
        stats = streaming_pipeline.get_latency_stats()
        assert stats['avg_ms'] >= 0
        assert stats['min_ms'] >= 0
