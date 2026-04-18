"""Targeted branch coverage for inference.streaming_pipeline."""

import builtins
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from auto_voice.inference.streaming_pipeline import (
    AudioInputStream,
    AudioOutputStream,
    StreamingConversionPipeline,
)


def _make_pipeline(device=None, overlap_ratio=0.5):
    """Create a streaming pipeline with a mocked conversion backend."""
    with patch("auto_voice.inference.streaming_pipeline.SOTAConversionPipeline") as mock_pipeline_cls:
        backend = MagicMock()
        backend.convert.return_value = {"audio": torch.tensor([1.2, -0.8, 0.4], dtype=torch.float32)}
        backend.get_speaker_embedding.return_value = torch.tensor([0.1] * 256, dtype=torch.float32)
        mock_pipeline_cls.return_value = backend
        pipeline = StreamingConversionPipeline(
            chunk_size_ms=100,
            overlap_ratio=overlap_ratio,
            sample_rate=24000,
            device=device,
        )
    pipeline._pipeline = backend
    return pipeline, backend


def test_streaming_pipeline_uses_explicit_device_and_zero_overlap_window():
    """Explicit devices and zero-overlap windows should hit their specialized branches."""
    pipeline, _ = _make_pipeline(device=torch.device("cpu"), overlap_ratio=0.0)

    assert pipeline.device == torch.device("cpu")
    assert torch.equal(pipeline.crossfade_window, torch.ones(1))


def test_process_chunk_squeezes_2d_input_and_trims_latency_history():
    """2D chunks should be squeezed and latency history capped at max size."""
    pipeline, backend = _make_pipeline(device=torch.device("cpu"))
    pipeline._latency_history = [float(i) for i in range(pipeline._max_latency_history)]
    speaker = torch.randn(256)
    chunk = torch.randn(1, 2400)

    output = pipeline.process_chunk(chunk, speaker)

    assert output.dim() == 1
    assert len(pipeline._latency_history) == pipeline._max_latency_history


def test_get_latency_stats_empty_returns_zeros():
    """Latency stats should be zeroed before any chunks are processed."""
    pipeline, _ = _make_pipeline(device=torch.device("cpu"))
    pipeline._latency_history = []

    assert pipeline.get_latency_stats() == {"min_ms": 0.0, "max_ms": 0.0, "avg_ms": 0.0}


def test_set_speaker_tracks_current_profile_and_missing_embedding_fails():
    """Speaker loading should store the active profile ID and reject missing embeddings."""
    pipeline, backend = _make_pipeline(device=torch.device("cpu"))

    pipeline.set_speaker("profile-1")
    assert pipeline.get_current_speaker() == "profile-1"
    assert pipeline._speaker_embedding.shape == (256,)

    backend.get_speaker_embedding.return_value = None
    with pytest.raises(RuntimeError, match="did not expose an embedding"):
        pipeline.set_speaker("profile-2")


def test_audio_input_stream_start_covers_missing_callback_import_and_runtime_error(monkeypatch):
    """Input stream startup should handle missing callbacks, ImportError, and runtime failures."""
    stream = AudioInputStream()
    with pytest.raises(RuntimeError, match="No callback set"):
        stream.start()

    original_import = builtins.__import__

    def missing_sounddevice(name, *args, **kwargs):
        if name == "sounddevice":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_sounddevice)
    stream.set_callback(lambda audio: None)
    with pytest.raises(RuntimeError, match="sounddevice package required"):
        stream.start()

    monkeypatch.setattr(builtins, "__import__", original_import)
    fake_sounddevice = types.SimpleNamespace(InputStream=MagicMock(side_effect=RuntimeError("device busy")))
    with patch.dict(sys.modules, {"sounddevice": fake_sounddevice}):
        with pytest.raises(RuntimeError, match="Failed to start audio capture: device busy"):
            stream.start()


def test_audio_input_stream_start_invokes_callback_and_stop_cleans_up():
    """Successful input startup should forward microphone frames and stop cleanly."""
    received = []
    stream = AudioInputStream(buffer_size=4)
    stream.set_callback(received.append)

    stream_holder = {}

    class FakeInputStream:
        def __init__(self, **kwargs):
            stream_holder["callback"] = kwargs["callback"]

        def start(self):
            return None

        def stop(self):
            stream_holder["stopped"] = True

        def close(self):
            stream_holder["closed"] = True

    with patch.dict(sys.modules, {"sounddevice": types.SimpleNamespace(InputStream=FakeInputStream)}):
        stream.start()
        stream_holder["callback"](np.ones((4, 1), dtype=np.float32), 4, {}, "status")
        stream.stop()

    assert stream.is_running is False
    assert len(received) == 1
    assert torch.equal(received[0], torch.ones(4))
    assert stream_holder["stopped"] is True
    assert stream_holder["closed"] is True


def test_audio_output_stream_write_flush_start_stop_and_import_fallback(monkeypatch):
    """Output stream should flush when running, ignore missing sounddevice during flush, and error on start."""
    stream = AudioOutputStream()
    stream._is_running = True
    stream._stream = object()
    stream._flush_buffer = MagicMock()
    stream.write(torch.tensor([1.0]))
    stream._flush_buffer.assert_called_once()

    stream = AudioOutputStream(sample_rate=16000)
    stream.write(torch.tensor([0.5, -0.5]))
    original_import = builtins.__import__

    def missing_sounddevice(name, *args, **kwargs):
        if name == "sounddevice":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_sounddevice)
    stream._flush_buffer()
    assert len(stream._buffer) == 1
    with pytest.raises(RuntimeError, match="sounddevice package required for audio output"):
        stream.start()

    monkeypatch.setattr(builtins, "__import__", original_import)
    played = {}
    fake_sd = types.SimpleNamespace(play=lambda audio, samplerate, device=None: played.update({
        "audio": audio,
        "samplerate": samplerate,
        "device": device,
    }))

    with patch.dict(sys.modules, {"sounddevice": fake_sd}):
        stream.start()
        assert stream.is_running is True
        stream._flush_buffer()
        stream.stop()

    assert np.array_equal(played["audio"], np.array([0.5, -0.5], dtype=np.float32))
    assert played["samplerate"] == 16000
    assert stream.is_running is False
    assert stream._buffer == []
