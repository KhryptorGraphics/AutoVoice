"""Tests for VoiceConversionPipeline.

This module tests the production-ready voice conversion pipeline,
including GPU acceleration, error handling, and profiling capabilities.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.auto_voice.inference.voice_conversion_pipeline import (
    VoiceConversionPipeline,
    PipelineConfig,
    VoiceConversionError
)


@pytest.fixture
def pipeline_config():
    """Create test pipeline configuration."""
    return PipelineConfig(
        sample_rate=22050,
        use_cuda=torch.cuda.is_available(),
        use_half_precision=False,
        batch_size=1,
        enable_profiling=True
    )


@pytest.fixture
def pipeline(pipeline_config):
    """Create test pipeline instance."""
    return VoiceConversionPipeline(pipeline_config)


@pytest.fixture
def test_audio():
    """Create test audio data (2 seconds at 22050 Hz)."""
    sample_rate = 22050
    duration = 2.0
    num_samples = int(sample_rate * duration)

    # Generate simple sine wave
    t = np.linspace(0, duration, num_samples, dtype=np.float32)
    frequency = 440.0  # A4 note
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)

    return audio


@pytest.fixture
def test_embedding():
    """Create test speaker embedding."""
    return np.random.randn(256).astype(np.float32)


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig()

        assert config.sample_rate == 22050
        assert config.n_fft == 2048
        assert config.hop_length == 512
        assert config.n_mels == 80
        assert config.speaker_embedding_dim == 256
        assert config.use_cuda is True
        assert config.fallback_on_error is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PipelineConfig(
            sample_rate=16000,
            n_fft=1024,
            use_cuda=False,
            batch_size=8
        )

        assert config.sample_rate == 16000
        assert config.n_fft == 1024
        assert config.use_cuda is False
        assert config.batch_size == 8


class TestVoiceConversionPipeline:
    """Tests for VoiceConversionPipeline."""

    def test_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline is not None
        assert pipeline.config is not None
        assert pipeline.device is not None
        assert isinstance(pipeline.stats, dict)

    def test_device_selection_cuda(self):
        """Test CUDA device selection when available."""
        if torch.cuda.is_available():
            config = PipelineConfig(use_cuda=True)
            pipeline = VoiceConversionPipeline(config)
            assert pipeline.device.type == 'cuda'
        else:
            pytest.skip("CUDA not available")

    def test_device_selection_cpu(self):
        """Test CPU device selection."""
        config = PipelineConfig(use_cuda=False)
        pipeline = VoiceConversionPipeline(config)
        assert pipeline.device.type == 'cpu'

    def test_convert_basic(self, pipeline, test_audio, test_embedding):
        """Test basic voice conversion."""
        result = pipeline.convert(
            source_audio=test_audio,
            target_embedding=test_embedding,
            source_sample_rate=22050
        )

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32 or result.dtype == np.float64
        # Allow some tolerance for audio processing
        assert abs(len(result) - len(test_audio)) < 1000

    def test_convert_with_progress_callback(self, pipeline, test_audio, test_embedding):
        """Test conversion with progress callback."""
        progress_updates = []

        def progress_callback(progress: float, stage: str):
            progress_updates.append((progress, stage))

        result = pipeline.convert(
            source_audio=test_audio,
            target_embedding=test_embedding,
            progress_callback=progress_callback
        )

        assert result is not None
        assert len(progress_updates) > 0
        # Should have start and end markers
        assert any('preprocessing' in stage for _, stage in progress_updates)

    def test_convert_with_pitch_shift(self, pipeline, test_audio, test_embedding):
        """Test conversion with pitch shift."""
        result = pipeline.convert(
            source_audio=test_audio,
            target_embedding=test_embedding,
            pitch_shift_semitones=2.0
        )

        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_convert_different_sample_rates(self, pipeline, test_audio, test_embedding):
        """Test conversion with different input/output sample rates."""
        # Resample to 16kHz
        result = pipeline.convert(
            source_audio=test_audio,
            target_embedding=test_embedding,
            source_sample_rate=22050,
            output_sample_rate=16000
        )

        assert result is not None
        # Output should be resampled
        expected_length = int(len(test_audio) * (16000 / 22050))
        assert abs(len(result) - expected_length) < 1000

    def test_batch_convert(self, pipeline, test_audio, test_embedding):
        """Test batch conversion."""
        audio_list = [test_audio, test_audio * 0.5, test_audio * 0.8]
        embeddings = [test_embedding, test_embedding, test_embedding]

        results = pipeline.batch_convert(audio_list, embeddings)

        assert len(results) == len(audio_list)
        for result in results:
            assert isinstance(result, np.ndarray)

    def test_batch_convert_mismatched_lengths(self, pipeline, test_audio, test_embedding):
        """Test batch conversion with mismatched input lengths."""
        audio_list = [test_audio, test_audio]
        embeddings = [test_embedding]

        with pytest.raises(VoiceConversionError):
            pipeline.batch_convert(audio_list, embeddings)

    def test_warmup(self, pipeline):
        """Test pipeline warmup."""
        # Should not raise any exceptions
        pipeline.warmup(num_iterations=2)

        # Stats should show warmup conversions
        assert pipeline.stats['total_conversions'] >= 2

    def test_get_stats(self, pipeline, test_audio, test_embedding):
        """Test statistics retrieval."""
        # Run some conversions
        pipeline.convert(test_audio, test_embedding)
        pipeline.convert(test_audio, test_embedding)

        stats = pipeline.get_stats()

        assert stats is not None
        assert 'total_conversions' in stats
        assert 'successful_conversions' in stats
        assert 'success_rate' in stats
        assert 'device' in stats
        assert stats['total_conversions'] >= 2

    def test_profile_conversion(self, pipeline, test_audio, test_embedding):
        """Test profiling conversion with timing metrics."""
        # Warmup first for stable measurements
        pipeline.warmup(num_iterations=1)

        metrics = pipeline.profile_conversion(
            source_audio=test_audio,
            target_embedding=test_embedding
        )

        assert metrics is not None
        assert 'total_ms' in metrics
        assert 'audio_duration_s' in metrics
        assert 'rtf' in metrics
        assert 'throughput_samples_per_sec' in metrics
        assert 'device' in metrics
        assert 'stages' in metrics

        # Verify metrics are reasonable
        assert metrics['total_ms'] > 0
        assert metrics['audio_duration_s'] > 0
        assert metrics['rtf'] > 0
        assert metrics['throughput_samples_per_sec'] > 0

    def test_error_handling_invalid_audio(self, pipeline, test_embedding):
        """Test error handling with invalid audio."""
        invalid_audio = None

        with pytest.raises((VoiceConversionError, TypeError, AttributeError)):
            pipeline.convert(invalid_audio, test_embedding)

    def test_error_handling_invalid_embedding(self, pipeline, test_audio):
        """Test error handling with invalid embedding."""
        invalid_embedding = None

        with pytest.raises((VoiceConversionError, TypeError, AttributeError)):
            pipeline.convert(test_audio, invalid_embedding)

    def test_fallback_on_error(self, test_audio):
        """Test fallback conversion when main pipeline fails."""
        config = PipelineConfig(fallback_on_error=True)
        pipeline = VoiceConversionPipeline(config)

        # Use mock to force an error in main conversion
        with patch.object(pipeline, '_extract_features', side_effect=Exception("Mock error")):
            result = pipeline.convert(test_audio, np.random.randn(256).astype(np.float32))

            # Should return fallback result (normalized audio)
            assert result is not None
            assert isinstance(result, np.ndarray)

    def test_no_fallback_on_error(self, test_audio):
        """Test that conversion fails without fallback."""
        config = PipelineConfig(fallback_on_error=False)
        pipeline = VoiceConversionPipeline(config)

        # Use mock to force an error
        with patch.object(pipeline, '_extract_features', side_effect=Exception("Mock error")):
            with pytest.raises(VoiceConversionError):
                pipeline.convert(test_audio, np.random.randn(256).astype(np.float32))

    def test_preprocess_audio(self, pipeline, test_audio):
        """Test audio preprocessing."""
        result = pipeline._preprocess_audio(test_audio, sample_rate=22050)

        assert isinstance(result, torch.Tensor)
        assert result.device == pipeline.device
        # Should be normalized
        assert torch.max(torch.abs(result)) <= 1.0

    def test_preprocess_audio_resampling(self, pipeline):
        """Test audio preprocessing with resampling."""
        # Create audio at different sample rate
        audio_16k = np.random.randn(16000 * 2).astype(np.float32)

        result = pipeline._preprocess_audio(audio_16k, sample_rate=16000)

        assert isinstance(result, torch.Tensor)
        # Should be resampled to config sample rate
        expected_length = int(len(audio_16k) * (pipeline.config.sample_rate / 16000))
        assert abs(len(result) - expected_length) < 100

    def test_extract_features(self, pipeline, test_audio):
        """Test feature extraction."""
        audio_tensor = torch.from_numpy(test_audio).to(pipeline.device)

        features = pipeline._extract_features(audio_tensor)

        assert features is not None
        assert 'mel_spec' in features
        assert 'f0' in features
        assert 'speaker_embedding' in features

        # Verify tensor shapes
        assert isinstance(features['mel_spec'], torch.Tensor)
        assert isinstance(features['f0'], torch.Tensor)

    def test_extract_features_with_provided_f0(self, pipeline, test_audio):
        """Test feature extraction with pre-computed F0."""
        audio_tensor = torch.from_numpy(test_audio).to(pipeline.device)

        # Provide dummy F0
        f0 = np.ones(100, dtype=np.float32) * 220.0  # A3 note

        features = pipeline._extract_features(audio_tensor, f0=f0)

        assert features is not None
        assert 'f0' in features
        # F0 should be the provided one
        assert torch.allclose(features['f0'], torch.from_numpy(f0).to(pipeline.device))

    def test_encode_speaker(self, pipeline, test_embedding):
        """Test speaker encoding."""
        embedding_tensor = torch.from_numpy(test_embedding).to(pipeline.device)

        encoded = pipeline._encode_speaker(embedding_tensor)

        assert isinstance(encoded, torch.Tensor)
        assert encoded.device == pipeline.device
        # Should be normalized
        assert torch.allclose(torch.norm(encoded, p=2, dim=-1), torch.ones(1).to(pipeline.device), atol=1e-5)

    def test_synthesize_voice(self, pipeline, test_audio):
        """Test voice synthesis."""
        # Create dummy features
        audio_tensor = torch.from_numpy(test_audio).to(pipeline.device)
        features = pipeline._extract_features(audio_tensor)

        # Create dummy speaker features
        speaker_features = torch.randn(1, 256).to(pipeline.device)

        waveform = pipeline._synthesize_voice(features, speaker_features)

        assert isinstance(waveform, torch.Tensor)
        assert waveform.ndim == 1

    def test_synthesize_voice_with_pitch_shift(self, pipeline, test_audio):
        """Test voice synthesis with pitch shift."""
        audio_tensor = torch.from_numpy(test_audio).to(pipeline.device)
        features = pipeline._extract_features(audio_tensor)
        speaker_features = torch.randn(1, 256).to(pipeline.device)

        waveform = pipeline._synthesize_voice(
            features, speaker_features, pitch_shift=5.0
        )

        assert isinstance(waveform, torch.Tensor)

    def test_postprocess_audio(self, pipeline):
        """Test audio postprocessing."""
        # Create dummy audio tensor
        audio_tensor = torch.randn(22050 * 2).to(pipeline.device)

        result = pipeline._postprocess_audio(audio_tensor, target_sample_rate=22050)

        assert isinstance(result, np.ndarray)
        # Should be normalized and clipped
        assert np.max(np.abs(result)) <= 1.0

    def test_postprocess_audio_resampling(self, pipeline):
        """Test audio postprocessing with resampling."""
        audio_tensor = torch.randn(22050 * 2).to(pipeline.device)

        result = pipeline._postprocess_audio(audio_tensor, target_sample_rate=16000)

        assert isinstance(result, np.ndarray)
        # Should be resampled
        expected_length = int(len(audio_tensor) * (16000 / pipeline.config.sample_rate))
        assert abs(len(result) - expected_length) < 100


class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""

    def test_end_to_end_conversion(self):
        """Test complete end-to-end conversion workflow."""
        # Create pipeline
        config = PipelineConfig(use_cuda=torch.cuda.is_available())
        pipeline = VoiceConversionPipeline(config)

        # Create test data
        sample_rate = 22050
        duration = 2.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        # Warmup
        pipeline.warmup(num_iterations=1)

        # Convert
        result = pipeline.convert(audio, embedding)

        # Verify result
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_multiple_sequential_conversions(self):
        """Test multiple sequential conversions."""
        config = PipelineConfig(use_cuda=torch.cuda.is_available())
        pipeline = VoiceConversionPipeline(config)

        audio = np.random.randn(22050 * 2).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        # Run multiple conversions
        results = []
        for _ in range(3):
            result = pipeline.convert(audio, embedding)
            results.append(result)

        # All should succeed
        assert len(results) == 3
        for result in results:
            assert isinstance(result, np.ndarray)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_management(self):
        """Test GPU memory management during conversion."""
        config = PipelineConfig(use_cuda=True)
        pipeline = VoiceConversionPipeline(config)

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()

        # Run conversion
        audio = np.random.randn(22050 * 2).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        pipeline.convert(audio, embedding)

        # Check memory was allocated and freed properly
        peak_memory = torch.cuda.max_memory_allocated()
        current_memory = torch.cuda.memory_allocated()

        assert peak_memory > 0
        # Memory should be mostly freed (allow some caching)
        assert current_memory < peak_memory * 0.5

    def test_cpu_fallback(self):
        """Test fallback to CPU when CUDA fails."""
        # Force CPU usage
        config = PipelineConfig(use_cuda=False)
        pipeline = VoiceConversionPipeline(config)

        audio = np.random.randn(22050 * 2).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        result = pipeline.convert(audio, embedding)

        assert result is not None
        assert pipeline.device.type == 'cpu'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
