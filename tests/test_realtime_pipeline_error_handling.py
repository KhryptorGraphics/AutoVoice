"""Tests for realtime pipeline error handling.

Tests the P0 error handling requirements:
- Invalid speaker embedding validation
- Empty audio handling
- GPU OOM graceful fallback
- NaN/Inf input validation

Track: realtime-error-handling_20260201
"""
import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from auto_voice.inference.realtime_pipeline import RealtimePipeline, SimpleDecoder


class TestSimpleDecoderValidation:
    """Test SimpleDecoder input validation."""

    def test_decoder_forward_shape(self):
        """Test decoder forward pass produces correct output shape."""
        decoder = SimpleDecoder(
            content_dim=768,
            pitch_dim=256,
            speaker_dim=256,
            n_mels=80,
            hidden_dim=256,
        )

        batch_size = 2
        n_frames = 100
        content = torch.randn(batch_size, n_frames, 768)
        pitch = torch.randn(batch_size, n_frames, 256)
        speaker = torch.randn(batch_size, 256)

        mel = decoder(content, pitch, speaker)

        assert mel.shape == (batch_size, 80, n_frames)
        assert torch.isfinite(mel).all()


class TestRealtimePipelineSpeakerValidation:
    """Test speaker embedding validation."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create pipeline with mocked components."""
        with patch.object(RealtimePipeline, '_init_content_encoder'):
            with patch.object(RealtimePipeline, '_init_pitch_extractor'):
                with patch.object(RealtimePipeline, '_init_decoder'):
                    with patch.object(RealtimePipeline, '_init_vocoder'):
                        pipeline = RealtimePipeline()
                        return pipeline

    def test_invalid_speaker_embedding_shape(self, mock_pipeline):
        """Test that wrong embedding dimension raises ValueError."""
        with pytest.raises(ValueError, match="must be 256-dimensional"):
            mock_pipeline.set_speaker_embedding(np.random.randn(128))

    def test_speaker_embedding_wrong_shape_512(self, mock_pipeline):
        """Test that 512-dim embedding raises ValueError."""
        with pytest.raises(ValueError, match="must be 256-dimensional"):
            mock_pipeline.set_speaker_embedding(np.random.randn(512))

    def test_speaker_embedding_with_nan(self, mock_pipeline):
        """Test that NaN in embedding raises ValueError."""
        embedding = np.random.randn(256)
        embedding[0] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            mock_pipeline.set_speaker_embedding(embedding)

    def test_speaker_embedding_with_inf(self, mock_pipeline):
        """Test that Inf in embedding raises ValueError."""
        embedding = np.random.randn(256)
        embedding[100] = np.inf
        with pytest.raises(ValueError, match="NaN or Inf"):
            mock_pipeline.set_speaker_embedding(embedding)

    def test_speaker_embedding_zero_norm(self, mock_pipeline):
        """Test that zero norm embedding raises ValueError."""
        embedding = np.zeros(256)
        with pytest.raises(ValueError, match="zero norm"):
            mock_pipeline.set_speaker_embedding(embedding)

    def test_speaker_embedding_auto_normalize(self, mock_pipeline):
        """Test that unnormalized embedding is auto-normalized."""
        embedding = np.random.randn(256) * 5.0  # Not normalized
        mock_pipeline.set_speaker_embedding(embedding)
        # Should succeed without error (auto-normalized)
        assert mock_pipeline._speaker_embedding is not None
        # Check it's normalized
        norm = torch.linalg.norm(mock_pipeline._speaker_embedding).item()
        assert abs(norm - 1.0) < 0.01

    def test_speaker_embedding_already_normalized(self, mock_pipeline):
        """Test that normalized embedding works correctly."""
        embedding = np.random.randn(256)
        embedding = embedding / np.linalg.norm(embedding)
        mock_pipeline.set_speaker_embedding(embedding)
        assert mock_pipeline._speaker_embedding is not None

    def test_speaker_embedding_2d_input(self, mock_pipeline):
        """Test that 2D embedding is flattened correctly."""
        embedding = np.random.randn(1, 256)
        embedding = embedding / np.linalg.norm(embedding)
        mock_pipeline.set_speaker_embedding(embedding)
        assert mock_pipeline._speaker_embedding.shape == (1, 256)


class TestRealtimePipelineInputValidation:
    """Test audio input validation."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create pipeline with mocked components."""
        with patch.object(RealtimePipeline, '_init_content_encoder'):
            with patch.object(RealtimePipeline, '_init_pitch_extractor'):
                with patch.object(RealtimePipeline, '_init_decoder'):
                    with patch.object(RealtimePipeline, '_init_vocoder'):
                        pipeline = RealtimePipeline()
                        return pipeline

    def test_process_empty_audio(self, mock_pipeline):
        """Test that empty audio returns silence."""
        # Set mock speaker embedding to trigger processing path
        mock_pipeline._speaker_embedding = torch.randn(1, 256)

        output = mock_pipeline.process_chunk(np.array([]))

        assert output.size > 0
        assert np.allclose(output, 0.0)
        assert output.dtype == np.float32

    def test_process_audio_with_nan(self, mock_pipeline):
        """Test that audio with NaN raises ValueError."""
        mock_pipeline._speaker_embedding = torch.randn(1, 256)

        audio = np.random.randn(16000).astype(np.float32)
        audio[100] = np.nan

        with pytest.raises(ValueError, match="NaN or Inf"):
            mock_pipeline.process_chunk(audio)

    def test_process_audio_with_inf(self, mock_pipeline):
        """Test that audio with Inf raises ValueError."""
        mock_pipeline._speaker_embedding = torch.randn(1, 256)

        audio = np.random.randn(16000).astype(np.float32)
        audio[100] = np.inf

        with pytest.raises(ValueError, match="NaN or Inf"):
            mock_pipeline.process_chunk(audio)

    def test_passthrough_when_no_speaker(self, mock_pipeline):
        """Test that audio passes through when no speaker is set."""
        audio = np.random.randn(16000).astype(np.float32)
        output = mock_pipeline.process_chunk(audio)

        assert output.shape == audio.shape
        assert np.allclose(output, audio)


class TestRealtimePipelineGPUErrorHandling:
    """Test GPU error graceful fallback."""

    @pytest.fixture
    def mock_pipeline_with_oom(self):
        """Create pipeline that raises OOM during processing."""
        with patch.object(RealtimePipeline, '_init_content_encoder'):
            with patch.object(RealtimePipeline, '_init_pitch_extractor'):
                with patch.object(RealtimePipeline, '_init_decoder'):
                    with patch.object(RealtimePipeline, '_init_vocoder'):
                        pipeline = RealtimePipeline()
                        pipeline._speaker_embedding = torch.randn(1, 256)

                        # Mock content encoder to raise OOM
                        mock_encoder = MagicMock()
                        mock_encoder.encode.side_effect = torch.cuda.OutOfMemoryError("Simulated OOM")
                        pipeline._content_encoder = mock_encoder

                        # Mock _log_gpu_memory
                        pipeline._log_gpu_memory = MagicMock()

                        return pipeline

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_gpu_oom_returns_passthrough(self, mock_pipeline_with_oom):
        """Test graceful fallback when GPU OOM occurs."""
        audio = np.random.randn(16000).astype(np.float32)

        # Should not crash, should return passthrough audio
        output = mock_pipeline_with_oom.process_chunk(audio)

        assert output.shape == audio.shape
        assert np.allclose(output, audio)

    @pytest.fixture
    def mock_pipeline_with_cuda_error(self):
        """Create pipeline that raises CUDA error during processing."""
        with patch.object(RealtimePipeline, '_init_content_encoder'):
            with patch.object(RealtimePipeline, '_init_pitch_extractor'):
                with patch.object(RealtimePipeline, '_init_decoder'):
                    with patch.object(RealtimePipeline, '_init_vocoder'):
                        pipeline = RealtimePipeline()
                        pipeline._speaker_embedding = torch.randn(1, 256)

                        # Mock content encoder to raise CUDA error
                        mock_encoder = MagicMock()
                        mock_encoder.encode.side_effect = RuntimeError("CUDA error: device-side assert triggered")
                        pipeline._content_encoder = mock_encoder

                        return pipeline

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_cuda_error_returns_passthrough(self, mock_pipeline_with_cuda_error):
        """Test graceful fallback when CUDA error occurs."""
        audio = np.random.randn(16000).astype(np.float32)

        # Should not crash, should return passthrough audio
        output = mock_pipeline_with_cuda_error.process_chunk(audio)

        assert output.shape == audio.shape
        assert np.allclose(output, audio)


class TestRealtimePipelineInitializationErrors:
    """Test initialization error handling."""

    def test_init_content_encoder_missing_model(self):
        """Test that missing model file raises RuntimeError."""
        with patch('auto_voice.models.encoder.ContentVecEncoder') as MockEncoder:
            MockEncoder.side_effect = FileNotFoundError("Model not found")

            with pytest.raises(RuntimeError, match="model file missing"):
                RealtimePipeline(contentvec_model="/nonexistent/model")

    def test_init_vocoder_missing_checkpoint(self):
        """Test that missing vocoder checkpoint raises RuntimeError."""
        with patch.object(RealtimePipeline, '_init_content_encoder'):
            with patch.object(RealtimePipeline, '_init_pitch_extractor'):
                with patch.object(RealtimePipeline, '_init_decoder'):
                    with patch('auto_voice.models.vocoder.HiFiGANVocoder') as MockVocoder:
                        mock_instance = MagicMock()
                        mock_instance.load_checkpoint.side_effect = FileNotFoundError("Checkpoint not found")
                        MockVocoder.return_value = mock_instance

                        with pytest.raises(RuntimeError, match="checkpoint missing"):
                            RealtimePipeline(vocoder_checkpoint="/nonexistent/checkpoint")


class TestRealtimePipelineMetrics:
    """Test metrics and monitoring."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create pipeline with mocked components."""
        with patch.object(RealtimePipeline, '_init_content_encoder'):
            with patch.object(RealtimePipeline, '_init_pitch_extractor'):
                with patch.object(RealtimePipeline, '_init_decoder'):
                    with patch.object(RealtimePipeline, '_init_vocoder'):
                        pipeline = RealtimePipeline()
                        return pipeline

    def test_get_latency_metrics_empty(self, mock_pipeline):
        """Test latency metrics returns zeros when no processing done."""
        metrics = mock_pipeline.get_latency_metrics()

        assert 'content_encoder_ms' in metrics
        assert 'pitch_extractor_ms' in metrics
        assert 'decoder_ms' in metrics
        assert 'vocoder_ms' in metrics
        assert 'total_ms' in metrics
        assert all(v == 0.0 for v in metrics.values())

    def test_get_metrics(self, mock_pipeline):
        """Test comprehensive metrics includes all fields."""
        metrics = mock_pipeline.get_metrics()

        assert 'device' in metrics
        assert 'sample_rate' in metrics
        assert 'output_sample_rate' in metrics
        assert 'has_speaker' in metrics


class TestSimpleDecoderFilm:
    """Test FiLM conditioning in SimpleDecoder."""

    def test_speaker_conditioning_changes_output(self):
        """Test that different speakers produce different outputs."""
        decoder = SimpleDecoder()

        content = torch.randn(1, 50, 768)
        pitch = torch.randn(1, 50, 256)

        speaker1 = torch.randn(1, 256)
        speaker2 = torch.randn(1, 256)

        mel1 = decoder(content, pitch, speaker1)
        mel2 = decoder(content, pitch, speaker2)

        # Different speakers should produce different mel spectrograms
        assert not torch.allclose(mel1, mel2)

    def test_same_speaker_same_output(self):
        """Test that same speaker produces same output."""
        decoder = SimpleDecoder()

        content = torch.randn(1, 50, 768)
        pitch = torch.randn(1, 50, 256)
        speaker = torch.randn(1, 256)

        mel1 = decoder(content, pitch, speaker)
        mel2 = decoder(content, pitch, speaker)

        # Same inputs should produce identical outputs
        assert torch.allclose(mel1, mel2)
