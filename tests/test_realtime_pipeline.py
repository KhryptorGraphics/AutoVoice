"""Tests for real-time voice conversion pipeline."""
import time
import pytest
import numpy as np


class TestRealtimePipeline:
    """Tests for RealtimeVoiceConversionPipeline."""

    def test_init(self):
        from auto_voice.inference.realtime_voice_conversion_pipeline import (
            RealtimeVoiceConversionPipeline
        )
        pipe = RealtimeVoiceConversionPipeline()
        assert pipe.sample_rate == 22050
        assert pipe.chunk_size == 4096
        assert not pipe.is_running

    def test_init_custom_config(self):
        from auto_voice.inference.realtime_voice_conversion_pipeline import (
            RealtimeVoiceConversionPipeline
        )
        pipe = RealtimeVoiceConversionPipeline(config={
            'sample_rate': 16000,
            'chunk_size': 2048,
        })
        assert pipe.sample_rate == 16000
        assert pipe.chunk_size == 2048

    def test_set_target_voice(self):
        from auto_voice.inference.realtime_voice_conversion_pipeline import (
            RealtimeVoiceConversionPipeline
        )
        pipe = RealtimeVoiceConversionPipeline()
        embedding = np.random.randn(256).astype(np.float32)
        pipe.set_target_voice(embedding)
        assert pipe._target_embedding is not None

    def test_start_stop(self):
        from auto_voice.inference.realtime_voice_conversion_pipeline import (
            RealtimeVoiceConversionPipeline
        )
        pipe = RealtimeVoiceConversionPipeline()
        pipe.start()
        assert pipe.is_running
        time.sleep(0.05)
        pipe.stop()
        assert not pipe.is_running

    def test_passthrough_no_target(self):
        from auto_voice.inference.realtime_voice_conversion_pipeline import (
            RealtimeVoiceConversionPipeline
        )
        pipe = RealtimeVoiceConversionPipeline()
        pipe._running = True
        chunk = np.random.randn(4096).astype(np.float32)
        # No target set, should pass through
        result = pipe.process_chunk(chunk)
        assert result is not None
        np.testing.assert_array_equal(result, chunk)

    def test_process_with_target(self):
        from auto_voice.inference.realtime_voice_conversion_pipeline import (
            RealtimeVoiceConversionPipeline
        )
        pipe = RealtimeVoiceConversionPipeline()
        pipe._running = True
        pipe.set_target_voice(np.random.randn(256))
        chunk = np.random.randn(4096).astype(np.float32)
        result = pipe.process_chunk(chunk)
        assert result is not None
        assert len(result) == 4096

    def test_output_range(self):
        from auto_voice.inference.realtime_voice_conversion_pipeline import (
            RealtimeVoiceConversionPipeline
        )
        pipe = RealtimeVoiceConversionPipeline()
        pipe._running = True
        pipe.set_target_voice(np.random.randn(256))
        chunk = np.sin(2 * np.pi * 440 * np.linspace(0, 0.186, 4096)).astype(np.float32)
        result = pipe.process_chunk(chunk)
        assert np.abs(result).max() <= 1.0

    def test_get_metrics(self):
        from auto_voice.inference.realtime_voice_conversion_pipeline import (
            RealtimeVoiceConversionPipeline
        )
        pipe = RealtimeVoiceConversionPipeline()
        metrics = pipe.get_metrics()
        assert 'is_running' in metrics
        assert 'chunks_processed' in metrics
        assert 'avg_latency_ms' in metrics
        assert 'buffer_latency_ms' in metrics

    def test_buffer_latency(self):
        from auto_voice.inference.realtime_voice_conversion_pipeline import (
            RealtimeVoiceConversionPipeline
        )
        pipe = RealtimeVoiceConversionPipeline(config={'chunk_size': 4096, 'sample_rate': 22050})
        expected_ms = (4096 / 22050) * 1000
        assert abs(pipe.buffer_latency_ms - expected_ms) < 0.01

    def test_push_mode_callback(self):
        from auto_voice.inference.realtime_voice_conversion_pipeline import (
            RealtimeVoiceConversionPipeline
        )
        results = []
        pipe = RealtimeVoiceConversionPipeline()
        pipe.set_target_voice(np.random.randn(256))
        pipe.start(on_output=lambda x: results.append(x))
        time.sleep(0.05)
        chunk = np.random.randn(4096).astype(np.float32)
        pipe.process_chunk(chunk)
        time.sleep(0.2)  # Wait for processing
        pipe.stop()
        assert len(results) >= 1

    def test_not_running_returns_none(self):
        from auto_voice.inference.realtime_voice_conversion_pipeline import (
            RealtimeVoiceConversionPipeline
        )
        pipe = RealtimeVoiceConversionPipeline()
        chunk = np.random.randn(4096).astype(np.float32)
        result = pipe.process_chunk(chunk)
        assert result is None

    def test_short_chunk_padding(self):
        from auto_voice.inference.realtime_voice_conversion_pipeline import (
            RealtimeVoiceConversionPipeline
        )
        pipe = RealtimeVoiceConversionPipeline()
        pipe._running = True
        pipe.set_target_voice(np.random.randn(256))
        chunk = np.random.randn(1000).astype(np.float32)
        result = pipe.process_chunk(chunk)
        assert result is not None
        assert len(result) == 1000

    def test_conversion_with_model(self):
        """Test conversion with properly loaded ModelManager."""
        import torch
        from auto_voice.inference.realtime_voice_conversion_pipeline import (
            RealtimeVoiceConversionPipeline
        )
        from auto_voice.inference.model_manager import ModelManager
        from auto_voice.models.so_vits_svc import SoVitsSvc

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pipe = RealtimeVoiceConversionPipeline(device=device)
        pipe._running = True

        # Set up ModelManager with random-weight model
        mm = ModelManager(device=device)
        mm.load()
        model = SoVitsSvc()
        model.to(device)
        mm._sovits_models['default'] = model
        pipe._model_manager = mm

        embedding = np.random.randn(256).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        pipe.set_target_voice(embedding)

        chunk = np.sin(2 * np.pi * 440 * np.linspace(0, 0.186, 4096)).astype(np.float32)
        result = pipe.process_chunk(chunk)
        assert result is not None
        assert len(result) == 4096
        assert not np.any(np.isnan(result))

    def test_realtime_with_consistency_student(self):
        """Test realtime pipeline with ConsistencyStudent loaded for 1-step inference."""
        import torch
        from auto_voice.inference.realtime_voice_conversion_pipeline import (
            RealtimeVoiceConversionPipeline
        )
        from auto_voice.inference.model_manager import ModelManager
        from auto_voice.models.consistency import ConsistencyStudent

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pipe = RealtimeVoiceConversionPipeline(device=device)
        pipe._running = True

        # Load ModelManager with vocoder (needed by consistency path)
        mm = ModelManager(device=device)
        mm.load()
        pipe._model_manager = mm

        # Create and load a ConsistencyStudent (random weights for test)
        student = ConsistencyStudent(
            n_mels=80, hidden_dim=128, n_blocks=4, cond_dim=256,
        ).to(device)
        pipe.load_consistency_student(student)

        assert pipe._use_consistency is True
        assert pipe._consistency_student is not None

        embedding = np.random.randn(256).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        pipe.set_target_voice(embedding)

        chunk = np.sin(2 * np.pi * 440 * np.linspace(0, 0.186, 4096)).astype(np.float32)
        result = pipe.process_chunk(chunk)

        # Consistency path goes through mel-space (content encoder downsamples,
        # vocoder upsamples) so output length depends on mel frame count
        assert result is not None
        assert len(result) > 0
        assert result.dtype == np.float32
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
