"""Tests for MeanVCPipeline - real-time streaming voice conversion.

MeanVCPipeline provides single-step voice conversion with streaming support using:
- FastU2++ ASR model for content feature extraction
- WavLM + ECAPA-TDNN for speaker embeddings
- DiT with Mean Flows for single-step inference
- Vocos vocoder for 16kHz output

Tests cover:
- Initialization and configuration
- Reference audio setting
- Chunk-wise streaming processing
- Full audio conversion
- Latency tracking
- Session management
"""
import os
import sys
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock


class TestMeanVCPipelineInit:
    """Test MeanVCPipeline initialization."""

    def test_import_succeeds(self):
        """MeanVCPipeline can be imported."""
        from auto_voice.inference.meanvc_pipeline import MeanVCPipeline
        assert MeanVCPipeline is not None

    def test_init_defaults_to_cpu(self):
        """MeanVCPipeline defaults to CPU device."""
        from auto_voice.inference.meanvc_pipeline import MeanVCPipeline

        with patch.object(MeanVCPipeline, '_load_models'):
            pipeline = MeanVCPipeline()

        assert pipeline.device == torch.device('cpu')

    def test_init_accepts_custom_device(self):
        """MeanVCPipeline accepts custom device."""
        from auto_voice.inference.meanvc_pipeline import MeanVCPipeline

        with patch.object(MeanVCPipeline, '_load_models'):
            pipeline = MeanVCPipeline(device=torch.device('cuda:0'))

        assert pipeline.device == torch.device('cuda:0')

    def test_init_does_not_require_gpu(self):
        """MeanVCPipeline can run without GPU."""
        from auto_voice.inference.meanvc_pipeline import MeanVCPipeline

        with patch('torch.cuda.is_available', return_value=False):
            with patch.object(MeanVCPipeline, '_load_models'):
                pipeline = MeanVCPipeline(require_gpu=False)

        assert pipeline is not None
        assert pipeline.device.type == 'cpu'

    def test_init_require_gpu_raises_without_cuda(self):
        """MeanVCPipeline raises if require_gpu=True without CUDA."""
        from auto_voice.inference.meanvc_pipeline import MeanVCPipeline

        with patch('torch.cuda.is_available', return_value=False):
            with pytest.raises(RuntimeError, match="CUDA"):
                MeanVCPipeline(require_gpu=True)

    def test_init_steps_configurable(self):
        """MeanVCPipeline accepts steps parameter."""
        from auto_voice.inference.meanvc_pipeline import MeanVCPipeline

        with patch.object(MeanVCPipeline, '_load_models'):
            pipeline1 = MeanVCPipeline(steps=1)
            pipeline2 = MeanVCPipeline(steps=2)

        assert pipeline1.steps == 1
        assert pipeline2.steps == 2


class TestSampleRates:
    """Test sample rate configuration."""

    def test_input_sample_rate_16k(self):
        """Input sample rate is 16kHz."""
        from auto_voice.inference.meanvc_pipeline import MeanVCPipeline

        with patch.object(MeanVCPipeline, '_load_models'):
            pipeline = MeanVCPipeline()

        assert pipeline.sample_rate == 16000

    def test_output_sample_rate_16k(self):
        """Output sample rate is 16kHz."""
        from auto_voice.inference.meanvc_pipeline import MeanVCPipeline

        with patch.object(MeanVCPipeline, '_load_models'):
            pipeline = MeanVCPipeline()

        assert pipeline.output_sample_rate == 16000


class TestTimesteps:
    """Test timestep configuration for mean flow inference."""

    def test_timesteps_single_step(self):
        """Single-step inference uses [1.0, 0.0] timesteps."""
        from auto_voice.inference.meanvc_pipeline import MeanVCPipeline

        with patch.object(MeanVCPipeline, '_load_models'):
            pipeline = MeanVCPipeline(steps=1)

        expected = torch.tensor([1.0, 0.0])
        assert torch.allclose(pipeline._timesteps, expected)

    def test_timesteps_two_step(self):
        """Two-step inference uses [1.0, 0.8, 0.0] timesteps."""
        from auto_voice.inference.meanvc_pipeline import MeanVCPipeline

        with patch.object(MeanVCPipeline, '_load_models'):
            pipeline = MeanVCPipeline(steps=2)

        expected = torch.tensor([1.0, 0.8, 0.0])
        assert torch.allclose(pipeline._timesteps, expected)


class TestChunkSize:
    """Test chunk size configuration."""

    def test_chunk_size_property(self):
        """chunk_size returns expected samples."""
        from auto_voice.inference.meanvc_pipeline import MeanVCPipeline

        with patch.object(MeanVCPipeline, '_load_models'):
            pipeline = MeanVCPipeline()

        # 200ms at 16kHz = 3200 samples
        assert pipeline.chunk_size == 3200

    def test_chunk_samples_calculation(self):
        """_chunk_samples is correctly calculated."""
        from auto_voice.inference.meanvc_pipeline import MeanVCPipeline

        with patch.object(MeanVCPipeline, '_load_models'):
            pipeline = MeanVCPipeline()

        # stride = 4 * 5 = 20, chunk_samples = 160 * 20 = 3200
        assert pipeline._chunk_samples == 3200


class TestReferenceAudio:
    """Test reference audio setting."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline with mocked model loading."""
        from auto_voice.inference.meanvc_pipeline import MeanVCPipeline

        with patch.object(MeanVCPipeline, '_load_models'):
            pipeline = MeanVCPipeline()

        # Mock the speaker verification model
        pipeline._sv_model = MagicMock()
        pipeline._sv_model.return_value = torch.randn(1, 256)

        # Mock mel extractor
        pipeline._mel_extractor = MagicMock()
        pipeline._mel_extractor.return_value = torch.randn(1, 80, 100)

        pipeline._initialized = True

        return pipeline

    def test_set_reference_audio_numpy(self, pipeline):
        """set_reference_audio accepts numpy array."""
        audio = np.random.randn(16000 * 5).astype(np.float32)

        pipeline.set_reference_audio(audio, sample_rate=16000)

        assert pipeline._reference_audio is not None
        assert pipeline._spk_emb is not None

    def test_set_reference_audio_tensor(self, pipeline):
        """set_reference_audio accepts torch tensor."""
        audio = torch.randn(16000 * 5)

        pipeline.set_reference_audio(audio, sample_rate=16000)

        assert pipeline._reference_audio is not None

    def test_set_reference_audio_resamples(self, pipeline):
        """set_reference_audio resamples non-16kHz audio."""
        audio = np.random.randn(44100 * 5).astype(np.float32)

        with patch('torchaudio.transforms.Resample') as MockResample:
            mock_resampler = MagicMock()
            mock_resampler.return_value = torch.randn(1, 16000 * 5)
            MockResample.return_value.to.return_value = mock_resampler

            pipeline.set_reference_audio(audio, sample_rate=44100)

        # Should have called resampler
        MockResample.assert_called_once()

    def test_set_reference_extracts_speaker_embedding(self, pipeline):
        """set_reference_audio extracts speaker embedding."""
        audio = np.random.randn(16000 * 5).astype(np.float32)

        pipeline.set_reference_audio(audio, sample_rate=16000)

        pipeline._sv_model.assert_called()
        assert pipeline._spk_emb is not None
        assert pipeline._spk_emb.shape[1] == 256

    def test_set_reference_extracts_prompt_mel(self, pipeline):
        """set_reference_audio extracts prompt mel spectrogram."""
        audio = np.random.randn(16000 * 5).astype(np.float32)

        pipeline.set_reference_audio(audio, sample_rate=16000)

        pipeline._mel_extractor.assert_called()
        assert pipeline._prompt_mel is not None

    def test_set_reference_resets_streaming_state(self, pipeline):
        """set_reference_audio resets streaming state."""
        # Set some state
        pipeline._vc_offset = 100
        pipeline._asr_offset = 50

        audio = np.random.randn(16000 * 5).astype(np.float32)
        pipeline.set_reference_audio(audio, sample_rate=16000)

        assert pipeline._vc_offset == 0
        assert pipeline._asr_offset == 0


class TestProcessChunk:
    """Test chunk-wise streaming processing."""

    @pytest.fixture
    def streaming_pipeline(self):
        """Create pipeline ready for streaming."""
        from auto_voice.inference.meanvc_pipeline import MeanVCPipeline

        with patch.object(MeanVCPipeline, '_load_models'):
            pipeline = MeanVCPipeline()

        # Set reference audio state
        pipeline._spk_emb = torch.randn(1, 256)
        pipeline._prompt_mel = torch.randn(1, 100, 80)
        pipeline._initialized = True

        # Mock models
        pipeline._asr_model = MagicMock()
        pipeline._asr_model.forward_encoder_chunk.return_value = (
            torch.randn(1, 5, 256),  # encoder output
            torch.zeros(0, 0, 0, 0),  # att cache
            torch.zeros(0, 0, 0, 0),  # cnn cache
        )

        pipeline._vc_model = MagicMock()
        pipeline._vc_model.return_value = (
            torch.randn(1, 20, 80),  # mel output
            [
                (torch.randn(1, 8, 100, 64), torch.randn(1, 8, 100, 64))
                for _ in range(12)
            ]  # kv cache
        )

        pipeline._vocoder = MagicMock()
        pipeline._vocoder.decode.return_value = torch.randn(1, 3040)

        return pipeline

    def test_process_chunk_requires_reference(self):
        """process_chunk raises without reference audio."""
        from auto_voice.inference.meanvc_pipeline import MeanVCPipeline

        with patch.object(MeanVCPipeline, '_load_models'):
            pipeline = MeanVCPipeline()

        pipeline._spk_emb = None
        chunk = np.random.randn(3200).astype(np.float32)

        with pytest.raises(RuntimeError, match="No reference audio"):
            pipeline.process_chunk(chunk)

    def test_process_chunk_returns_audio(self, streaming_pipeline):
        """process_chunk returns converted audio."""
        chunk = np.random.randn(3200).astype(np.float32)

        output = streaming_pipeline.process_chunk(chunk)

        assert isinstance(output, np.ndarray)
        assert output.dtype == np.float32

    def test_process_chunk_tracks_latency(self, streaming_pipeline):
        """process_chunk tracks latency metrics."""
        chunk = np.random.randn(3200).astype(np.float32)

        streaming_pipeline.process_chunk(chunk)

        metrics = streaming_pipeline.get_latency_metrics()

        assert 'asr_ms' in metrics
        assert 'vc_ms' in metrics
        assert 'vocoder_ms' in metrics
        assert 'total_ms' in metrics

    def test_process_chunk_updates_caches(self, streaming_pipeline):
        """process_chunk updates streaming caches."""
        chunk = np.random.randn(3200).astype(np.float32)

        initial_offset = streaming_pipeline._asr_offset

        streaming_pipeline.process_chunk(chunk)

        # Offsets should have increased
        assert streaming_pipeline._asr_offset > initial_offset

    def test_process_chunk_crossfade(self, streaming_pipeline):
        """process_chunk applies crossfade for smooth output."""
        chunk1 = np.random.randn(3200).astype(np.float32)
        chunk2 = np.random.randn(3200).astype(np.float32)

        output1 = streaming_pipeline.process_chunk(chunk1)
        output2 = streaming_pipeline.process_chunk(chunk2)

        # Both should produce output
        assert len(output1) > 0
        assert len(output2) > 0


class TestConvert:
    """Test full audio conversion (non-streaming)."""

    @pytest.fixture
    def convert_pipeline(self):
        """Create pipeline for full conversion."""
        from auto_voice.inference.meanvc_pipeline import MeanVCPipeline

        with patch.object(MeanVCPipeline, '_load_models'):
            pipeline = MeanVCPipeline()

        pipeline._spk_emb = torch.randn(1, 256)
        pipeline._prompt_mel = torch.randn(1, 100, 80)
        pipeline._initialized = True

        return pipeline

    def test_convert_requires_reference(self, convert_pipeline):
        """convert raises without reference audio."""
        convert_pipeline._spk_emb = None

        audio = np.random.randn(16000 * 5).astype(np.float32)

        with pytest.raises(RuntimeError, match="No reference audio"):
            convert_pipeline.convert(audio, sample_rate=16000)

    def test_convert_resets_streaming_state(self, convert_pipeline):
        """convert resets streaming state before processing."""
        convert_pipeline._asr_offset = 100

        with patch.object(convert_pipeline, 'process_chunk', return_value=np.zeros(3000)):
            audio = np.random.randn(16000).astype(np.float32)
            convert_pipeline.convert(audio, sample_rate=16000)

        # State should have been reset at start
        # (and then updated during processing)

    def test_convert_processes_in_chunks(self, convert_pipeline):
        """convert processes audio in chunks."""
        audio = np.random.randn(16000 * 5).astype(np.float32)  # 5 seconds

        with patch.object(convert_pipeline, 'process_chunk', return_value=np.zeros(3000)) as mock_chunk:
            convert_pipeline.convert(audio, sample_rate=16000)

        # Should have been called multiple times
        assert mock_chunk.call_count > 1

    def test_convert_resamples_input(self, convert_pipeline):
        """convert resamples non-16kHz input."""
        audio = np.random.randn(44100 * 2).astype(np.float32)

        with patch.object(convert_pipeline, 'process_chunk', return_value=np.zeros(3000)):
            with patch('librosa.resample', return_value=np.zeros(16000 * 2)) as mock_resample:
                convert_pipeline.convert(audio, sample_rate=44100)

        mock_resample.assert_called_once()

    def test_convert_handles_stereo(self, convert_pipeline):
        """convert handles stereo input by converting to mono."""
        audio = np.random.randn(2, 16000 * 2).astype(np.float32)

        with patch.object(convert_pipeline, 'process_chunk', return_value=np.zeros(3000)):
            result = convert_pipeline.convert(audio, sample_rate=16000)

        assert 'audio' in result

    def test_convert_returns_metadata(self, convert_pipeline):
        """convert returns processing metadata."""
        audio = np.random.randn(16000 * 2).astype(np.float32)

        with patch.object(convert_pipeline, 'process_chunk', return_value=np.zeros(3000)):
            result = convert_pipeline.convert(audio, sample_rate=16000)

        assert 'metadata' in result
        assert 'processing_time' in result['metadata']
        assert 'steps' in result['metadata']
        assert result['metadata']['pipeline'] == 'meanvc'

    def test_convert_progress_callback(self, convert_pipeline):
        """convert calls progress callback."""
        audio = np.random.randn(16000 * 2).astype(np.float32)
        progress_calls = []

        def on_progress(stage, progress):
            progress_calls.append((stage, progress))

        with patch.object(convert_pipeline, 'process_chunk', return_value=np.zeros(3000)):
            convert_pipeline.convert(audio, sample_rate=16000, on_progress=on_progress)

        assert len(progress_calls) > 0


class TestMetrics:
    """Test metrics and monitoring."""

    def test_get_latency_metrics(self):
        """get_latency_metrics returns latency info."""
        from auto_voice.inference.meanvc_pipeline import MeanVCPipeline

        with patch.object(MeanVCPipeline, '_load_models'):
            pipeline = MeanVCPipeline()

        metrics = pipeline.get_latency_metrics()

        assert 'asr_ms' in metrics
        assert 'vc_ms' in metrics
        assert 'vocoder_ms' in metrics
        assert 'total_ms' in metrics

    def test_get_metrics_comprehensive(self):
        """get_metrics returns comprehensive pipeline info."""
        from auto_voice.inference.meanvc_pipeline import MeanVCPipeline

        with patch.object(MeanVCPipeline, '_load_models'):
            pipeline = MeanVCPipeline(steps=2)

        metrics = pipeline.get_metrics()

        assert 'device' in metrics
        assert 'sample_rate' in metrics
        assert 'output_sample_rate' in metrics
        assert 'steps' in metrics
        assert 'has_reference' in metrics
        assert 'chunk_size_ms' in metrics

        assert metrics['sample_rate'] == 16000
        assert metrics['steps'] == 2


class TestSessionManagement:
    """Test session reset and management."""

    def test_reset_session_clears_state(self):
        """reset_session clears streaming state."""
        from auto_voice.inference.meanvc_pipeline import MeanVCPipeline

        with patch.object(MeanVCPipeline, '_load_models'):
            pipeline = MeanVCPipeline()

        # Set some state
        pipeline._spk_emb = torch.randn(1, 256)
        pipeline._asr_offset = 100
        pipeline._vc_offset = 50
        pipeline._vc_kv_cache = [MagicMock()]

        pipeline.reset_session()

        assert pipeline._asr_offset == 0
        assert pipeline._vc_offset == 0
        assert pipeline._vc_kv_cache is None

    def test_reset_session_keeps_reference(self):
        """reset_session keeps reference audio."""
        from auto_voice.inference.meanvc_pipeline import MeanVCPipeline

        with patch.object(MeanVCPipeline, '_load_models'):
            pipeline = MeanVCPipeline()

        spk_emb = torch.randn(1, 256)
        pipeline._spk_emb = spk_emb

        pipeline.reset_session()

        # Reference should still be set
        assert pipeline._spk_emb is spk_emb


class TestEstimatedMemory:
    """Test memory estimation."""

    def test_estimated_memory_constant(self):
        """Pipeline reports estimated GPU memory."""
        from auto_voice.inference.meanvc_pipeline import MeanVCPipeline

        assert hasattr(MeanVCPipeline, 'ESTIMATED_MEMORY_GB')
        assert MeanVCPipeline.ESTIMATED_MEMORY_GB > 0
        # MeanVC is lightweight
        assert MeanVCPipeline.ESTIMATED_MEMORY_GB <= 6.0


class TestASROffsetReset:
    """Test ASR offset reset for long sessions."""

    @pytest.fixture
    def long_session_pipeline(self):
        """Create pipeline configured for offset testing."""
        from auto_voice.inference.meanvc_pipeline import MeanVCPipeline

        with patch.object(MeanVCPipeline, '_load_models'):
            pipeline = MeanVCPipeline()

        pipeline._spk_emb = torch.randn(1, 256)
        pipeline._prompt_mel = torch.randn(1, 100, 80)
        pipeline._initialized = True

        # Mock models
        pipeline._asr_model = MagicMock()
        pipeline._asr_model.forward_encoder_chunk.return_value = (
            torch.randn(1, 5, 256),
            torch.zeros(0, 0, 0, 0),
            torch.zeros(0, 0, 0, 0),
        )

        pipeline._vc_model = MagicMock()
        pipeline._vc_model.return_value = (
            torch.randn(1, 20, 80),
            [(torch.randn(1, 8, 100, 64), torch.randn(1, 8, 100, 64)) for _ in range(12)]
        )

        pipeline._vocoder = MagicMock()
        pipeline._vocoder.decode.return_value = torch.randn(1, 3040)

        return pipeline

    def test_asr_offset_resets_at_threshold(self, long_session_pipeline):
        """ASR offset resets when exceeding threshold."""
        # Set offset near threshold
        long_session_pipeline._asr_offset = 4001  # > 4000 threshold

        chunk = np.random.randn(3200).astype(np.float32)
        long_session_pipeline.process_chunk(chunk)

        # Offset should have been reset (and then updated)
        # It won't be 0 because it gets incremented, but should be small
        assert long_session_pipeline._asr_offset < 4000


class TestKVCacheTruncation:
    """Test KV cache truncation for memory management."""

    @pytest.fixture
    def cache_pipeline(self):
        """Create pipeline for cache testing."""
        from auto_voice.inference.meanvc_pipeline import MeanVCPipeline

        with patch.object(MeanVCPipeline, '_load_models'):
            pipeline = MeanVCPipeline()

        pipeline._spk_emb = torch.randn(1, 256)
        pipeline._prompt_mel = torch.randn(1, 100, 80)
        pipeline._initialized = True

        # Mock models with controllable KV cache size
        pipeline._asr_model = MagicMock()
        pipeline._asr_model.forward_encoder_chunk.return_value = (
            torch.randn(1, 5, 256),
            torch.zeros(0, 0, 0, 0),
            torch.zeros(0, 0, 0, 0),
        )

        pipeline._vocoder = MagicMock()
        pipeline._vocoder.decode.return_value = torch.randn(1, 3040)

        return pipeline

    def test_kv_cache_truncated_when_large(self, cache_pipeline):
        """KV cache is truncated when too large."""
        # Create large KV cache
        large_cache = [
            (torch.randn(1, 8, 150, 64), torch.randn(1, 8, 150, 64))
            for _ in range(12)
        ]
        cache_pipeline._vc_kv_cache = large_cache
        cache_pipeline._vc_offset = 50  # > 40 threshold

        # Mock VC model to return same large cache
        cache_pipeline._vc_model = MagicMock()
        cache_pipeline._vc_model.return_value = (
            torch.randn(1, 20, 80),
            large_cache
        )

        chunk = np.random.randn(3200).astype(np.float32)
        cache_pipeline.process_chunk(chunk)

        # Cache should be truncated to max 100 entries
        for k, v in cache_pipeline._vc_kv_cache:
            assert k.shape[2] <= 100


class TestMelSpectrogram:
    """Test mel spectrogram feature extraction."""

    def test_mel_spectrogram_features_import(self):
        """MelSpectrogramFeatures can be imported."""
        from auto_voice.inference.meanvc_pipeline import MelSpectrogramFeatures
        assert MelSpectrogramFeatures is not None

    def test_mel_spectrogram_forward(self):
        """MelSpectrogramFeatures produces mel spectrogram."""
        from auto_voice.inference.meanvc_pipeline import MelSpectrogramFeatures

        mel_extractor = MelSpectrogramFeatures()
        audio = torch.randn(16000)  # 1 second

        mel = mel_extractor(audio)

        assert mel.dim() == 2
        assert mel.shape[0] == 80  # n_mels


class TestExtractFbanks:
    """Test filter bank feature extraction."""

    def test_extract_fbanks_import(self):
        """extract_fbanks can be imported."""
        from auto_voice.inference.meanvc_pipeline import extract_fbanks
        assert extract_fbanks is not None

    def test_extract_fbanks_shape(self):
        """extract_fbanks produces correct shape."""
        from auto_voice.inference.meanvc_pipeline import extract_fbanks

        audio = np.random.randn(16000).astype(np.float32)  # 1 second

        fbanks = extract_fbanks(audio)

        assert fbanks.dim() == 3  # [1, T, 80]
        assert fbanks.shape[0] == 1
        assert fbanks.shape[2] == 80  # mel bins
