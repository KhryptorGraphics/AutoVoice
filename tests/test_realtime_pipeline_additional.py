"""Targeted branch coverage for inference.realtime_pipeline."""

from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from auto_voice.inference.realtime_pipeline import RealtimePipeline


def _bare_pipeline(device="cpu"):
    """Build a lightweight pipeline shell without running full initialization."""
    pipeline = object.__new__(RealtimePipeline)
    pipeline.device = torch.device(device)
    pipeline.sample_rate = 16000
    pipeline.output_sample_rate = 22050
    pipeline._latency_history = {
        "content_encoder": deque(maxlen=100),
        "pitch_extractor": deque(maxlen=100),
        "decoder": deque(maxlen=100),
        "vocoder": deque(maxlen=100),
        "total": deque(maxlen=100),
    }
    pipeline._speaker_embedding = None
    pipeline._log_gpu_memory = MagicMock()
    return pipeline


def test_init_content_encoder_handles_oom_and_unexpected_errors():
    """Content encoder init should map GPU and generic errors to RuntimeError."""
    pipeline = _bare_pipeline()

    with patch("auto_voice.models.encoder.ContentVecEncoder", side_effect=torch.cuda.OutOfMemoryError("oom")):
        with patch("torch.cuda.empty_cache") as empty_cache:
            with pytest.raises(RuntimeError, match="Insufficient GPU memory"):
                RealtimePipeline._init_content_encoder(pipeline, None)
    empty_cache.assert_called_once()
    pipeline._log_gpu_memory.assert_called_once()

    with patch("auto_voice.models.encoder.ContentVecEncoder", side_effect=ValueError("bad model")):
        with pytest.raises(RuntimeError, match="Failed to initialize ContentVec: bad model"):
            RealtimePipeline._init_content_encoder(pipeline, None)


def test_init_pitch_extractor_handles_file_missing_oom_and_unexpected_errors():
    """Pitch extractor init should translate its failure modes consistently."""
    pipeline = _bare_pipeline()

    with patch("auto_voice.models.pitch.RMVPEPitchExtractor", side_effect=FileNotFoundError("missing")):
        with pytest.raises(RuntimeError, match="model file missing"):
            RealtimePipeline._init_pitch_extractor(pipeline)

    with patch("auto_voice.models.pitch.RMVPEPitchExtractor", side_effect=torch.cuda.OutOfMemoryError("oom")):
        with patch("torch.cuda.empty_cache") as empty_cache:
            with pytest.raises(RuntimeError, match="Insufficient GPU memory"):
                RealtimePipeline._init_pitch_extractor(pipeline)
    empty_cache.assert_called_once()
    assert pipeline._log_gpu_memory.call_count >= 1

    with patch("auto_voice.models.pitch.RMVPEPitchExtractor", side_effect=ValueError("broken")):
        with pytest.raises(RuntimeError, match="Failed to initialize RMVPE: broken"):
            RealtimePipeline._init_pitch_extractor(pipeline)


def test_init_decoder_and_vocoder_handle_runtime_failures():
    """Decoder and vocoder init should wrap OOM and unexpected exceptions."""
    pipeline = _bare_pipeline()

    with patch("auto_voice.inference.realtime_pipeline.SimpleDecoder", side_effect=torch.cuda.OutOfMemoryError("oom")):
        with patch("torch.cuda.empty_cache") as empty_cache:
            with pytest.raises(RuntimeError, match="Insufficient GPU memory for SimpleDecoder"):
                RealtimePipeline._init_decoder(pipeline)
    empty_cache.assert_called_once()

    with patch("auto_voice.inference.realtime_pipeline.SimpleDecoder", side_effect=ValueError("bad decoder")):
        with pytest.raises(RuntimeError, match="Failed to initialize SimpleDecoder: bad decoder"):
            RealtimePipeline._init_decoder(pipeline)

    with patch("auto_voice.models.vocoder.HiFiGANVocoder", side_effect=torch.cuda.OutOfMemoryError("oom")):
        with patch("torch.cuda.empty_cache") as empty_cache:
            with pytest.raises(RuntimeError, match="Insufficient GPU memory for HiFiGAN vocoder"):
                RealtimePipeline._init_vocoder(pipeline, checkpoint=None)
    empty_cache.assert_called_once()

    with patch("auto_voice.models.vocoder.HiFiGANVocoder", side_effect=ValueError("bad vocoder")):
        with pytest.raises(RuntimeError, match="Failed to initialize HiFiGAN: bad vocoder"):
            RealtimePipeline._init_vocoder(pipeline, checkpoint=None)


def test_clear_speaker_and_zero_frame_processing_paths():
    """Clearing the speaker and zero-frame model output should both behave safely."""
    pipeline = _bare_pipeline()
    pipeline._speaker_embedding = torch.randn(1, 256)
    RealtimePipeline.clear_speaker(pipeline)
    assert pipeline._speaker_embedding is None

    pipeline._speaker_embedding = torch.randn(1, 256)
    pipeline._content_encoder = MagicMock()
    pipeline._content_encoder.encode.return_value = torch.randn(1, 0, 768)
    pipeline._pitch_extractor = MagicMock()
    pipeline._pitch_extractor.extract.return_value = torch.randn(1, 10)
    pipeline._pitch_encoder = MagicMock(return_value=torch.randn(1, 10, 256))

    output = RealtimePipeline.process_chunk(pipeline, np.ones(1600, dtype=np.float32))

    assert output.shape == (2205,)
    assert np.allclose(output, 0.0)


def test_process_chunk_normalizes_non_peak_outputs_and_handles_generic_failures():
    """Normal outputs should scale to 0.9 peak; unexpected errors should passthrough."""
    pipeline = _bare_pipeline()
    pipeline._speaker_embedding = torch.randn(1, 256)
    pipeline._content_encoder = MagicMock()
    pipeline._content_encoder.encode.return_value = torch.randn(1, 5, 768)
    pipeline._pitch_extractor = MagicMock()
    pipeline._pitch_extractor.extract.return_value = torch.randn(1, 5)
    pipeline._pitch_encoder = MagicMock(return_value=torch.randn(1, 5, 256))
    pipeline._decoder = MagicMock(return_value=torch.randn(1, 80, 5))
    pipeline._vocoder = MagicMock()
    pipeline._vocoder.synthesize.return_value = torch.tensor([[0.25, -0.5]], dtype=torch.float32)

    normalized = RealtimePipeline.process_chunk(pipeline, np.ones(1600, dtype=np.float32))
    assert normalized.dtype == np.float32
    assert np.isclose(np.abs(normalized).max(), 0.9, atol=1e-5)

    failing = _bare_pipeline()
    failing._speaker_embedding = torch.randn(1, 256)
    failing._content_encoder = MagicMock()
    failing._content_encoder.encode.return_value = torch.randn(1, 5, 768)
    failing._pitch_extractor = MagicMock()
    failing._pitch_extractor.extract.side_effect = ValueError("bad pitch")

    audio = np.arange(16, dtype=np.float32)
    output = RealtimePipeline.process_chunk(failing, audio)
    assert np.array_equal(output, audio)


def test_log_gpu_memory_warns_when_query_fails(caplog):
    """GPU memory logging should warn instead of raising on telemetry failures."""
    pipeline = _bare_pipeline()

    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.memory_allocated", side_effect=RuntimeError("no telemetry")):
            RealtimePipeline._log_gpu_memory(pipeline)

    assert "Could not log GPU memory" in caplog.text
