"""Extended tests for VoiceConversionPipeline - achieving 80% coverage.

This module provides comprehensive tests covering:
- Error handling and edge cases
- GPU/CPU fallback scenarios
- Configuration validation
- Batch processing edge cases
- Warmup functionality
- Statistics collection
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
from src.auto_voice.gpu.cuda_kernels import CUDAKernelError


@pytest.fixture
def cpu_pipeline():
    """Create CPU-only pipeline for fallback tests."""
    config = PipelineConfig(use_cuda=False, fallback_on_error=True)
    return VoiceConversionPipeline(config)


@pytest.fixture
def no_fallback_pipeline():
    """Create pipeline without fallback for error testing."""
    config = PipelineConfig(use_cuda=False, fallback_on_error=False)
    return VoiceConversionPipeline(config)


class TestErrorHandling:
    """Test error handling paths."""

    def test_invalid_audio_none(self, cpu_pipeline):
        """Test handling of None audio input."""
        embedding = np.random.randn(256).astype(np.float32)

        with pytest.raises((VoiceConversionError, TypeError, AttributeError)):
            cpu_pipeline.convert(None, embedding)

    def test_invalid_audio_empty(self, cpu_pipeline):
        """Test handling of empty audio array."""
        audio = np.array([])
        embedding = np.random.randn(256).astype(np.float32)

        with pytest.raises((VoiceConversionError, RuntimeError, ValueError)):
            cpu_pipeline.convert(audio, embedding)

    def test_invalid_audio_wrong_type(self, cpu_pipeline):
        """Test handling of wrong audio type."""
        audio = "not an array"
        embedding = np.random.randn(256).astype(np.float32)

        with pytest.raises((VoiceConversionError, TypeError, AttributeError)):
            cpu_pipeline.convert(audio, embedding)

    def test_invalid_embedding_none(self, cpu_pipeline):
        """Test handling of None embedding input."""
        audio = np.random.randn(22050).astype(np.float32)

        with pytest.raises((VoiceConversionError, TypeError, AttributeError)):
            cpu_pipeline.convert(audio, None)

    def test_invalid_embedding_wrong_shape(self, cpu_pipeline):
        """Test handling of wrong embedding shape."""
        audio = np.random.randn(22050).astype(np.float32)
        embedding = np.random.randn(10, 10).astype(np.float32)  # Should be 1D

        # Should handle gracefully or raise error
        result = cpu_pipeline.convert(audio, embedding)
        assert result is not None

    def test_negative_pitch_shift(self, cpu_pipeline):
        """Test handling of extreme pitch shifts."""
        audio = np.random.randn(22050).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        # Should handle negative pitch shift
        result = cpu_pipeline.convert(audio, embedding, pitch_shift_semitones=-12.0)
        assert result is not None

    def test_extreme_pitch_shift(self, cpu_pipeline):
        """Test handling of extreme pitch shifts."""
        audio = np.random.randn(22050).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        # Should handle extreme pitch shift
        result = cpu_pipeline.convert(audio, embedding, pitch_shift_semitones=24.0)
        assert result is not None

    def test_invalid_sample_rate_zero(self, cpu_pipeline):
        """Test handling of zero sample rate."""
        audio = np.random.randn(22050).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        with pytest.raises((VoiceConversionError, ValueError, RuntimeError)):
            cpu_pipeline.convert(audio, embedding, source_sample_rate=0)

    def test_invalid_sample_rate_negative(self, cpu_pipeline):
        """Test handling of negative sample rate."""
        audio = np.random.randn(22050).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        with pytest.raises((VoiceConversionError, ValueError, RuntimeError)):
            cpu_pipeline.convert(audio, embedding, source_sample_rate=-1)

    def test_preprocessing_error_handling(self, no_fallback_pipeline):
        """Test preprocessing error without fallback."""
        audio = np.random.randn(22050).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        with patch.object(no_fallback_pipeline, '_preprocess_audio', side_effect=Exception("Mock error")):
            with pytest.raises(VoiceConversionError):
                no_fallback_pipeline.convert(audio, embedding)

    def test_feature_extraction_error_handling(self, no_fallback_pipeline):
        """Test feature extraction error without fallback."""
        audio = np.random.randn(22050).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        with patch.object(no_fallback_pipeline, '_extract_features', side_effect=Exception("Mock error")):
            with pytest.raises(VoiceConversionError):
                no_fallback_pipeline.convert(audio, embedding)

    def test_synthesis_error_handling(self, no_fallback_pipeline):
        """Test synthesis error without fallback."""
        audio = np.random.randn(22050).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        with patch.object(no_fallback_pipeline, '_synthesize_voice', side_effect=Exception("Mock error")):
            with pytest.raises(VoiceConversionError):
                no_fallback_pipeline.convert(audio, embedding)

    def test_postprocessing_error_handling(self, no_fallback_pipeline):
        """Test postprocessing error without fallback."""
        audio = np.random.randn(22050).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        with patch.object(no_fallback_pipeline, '_postprocess_audio', side_effect=Exception("Mock error")):
            with pytest.raises(VoiceConversionError):
                no_fallback_pipeline.convert(audio, embedding)

    def test_fallback_conversion_activation(self, cpu_pipeline):
        """Test that fallback conversion is activated on errors."""
        audio = np.random.randn(22050).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        with patch.object(cpu_pipeline, '_extract_features', side_effect=Exception("Mock error")):
            result = cpu_pipeline.convert(audio, embedding)

            # Should return normalized audio as fallback
            assert result is not None
            assert np.max(np.abs(result)) <= 1.0

    def test_fallback_conversion_direct(self, cpu_pipeline):
        """Test fallback conversion method directly."""
        audio = np.random.randn(22050).astype(np.float32) * 2.0  # Unnormalized

        result = cpu_pipeline._fallback_conversion(audio)

        assert result is not None
        assert np.max(np.abs(result)) <= 1.0
        assert len(result) == len(audio)


class TestGPUCPUFallback:
    """Test GPU/CPU fallback logic."""

    def test_cpu_device_forced(self):
        """Test CPU device selection when forced."""
        config = PipelineConfig(use_cuda=False)
        pipeline = VoiceConversionPipeline(config)

        assert pipeline.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_device_when_available(self):
        """Test GPU device selection when available."""
        config = PipelineConfig(use_cuda=True)
        pipeline = VoiceConversionPipeline(config)

        assert pipeline.device.type == 'cuda'

    def test_resample_fallback_to_interpolation(self, cpu_pipeline):
        """Test resampling fallback to linear interpolation."""
        audio = torch.randn(22050)

        # Mock torchaudio failure
        with patch('torchaudio.transforms.Resample', side_effect=Exception("Mock error")):
            result = cpu_pipeline._resample(audio, 22050, 16000)

            assert result is not None
            expected_length = int(len(audio) * (16000 / 22050))
            assert abs(len(result) - expected_length) < 100

    def test_resample_with_batch_dimension(self, cpu_pipeline):
        """Test resampling with batch dimension handling."""
        audio = torch.randn(22050)

        result = cpu_pipeline._resample(audio, 22050, 16000)

        assert result.ndim == 1  # Should remove batch dimension

    def test_resample_preserves_batch(self, cpu_pipeline):
        """Test resampling preserves existing batch dimension."""
        audio = torch.randn(2, 22050)

        result = cpu_pipeline._resample(audio, 22050, 16000)

        assert result.shape[0] == 2  # Batch preserved


class TestConfigurationValidation:
    """Test configuration validation."""

    def test_config_default_values(self):
        """Test all default configuration values."""
        config = PipelineConfig()

        assert config.sample_rate == 22050
        assert config.n_fft == 2048
        assert config.hop_length == 512
        assert config.win_length == 2048
        assert config.n_mels == 80
        assert config.f_min == 0.0
        assert config.f_max == 8000.0
        assert config.f0_min == 80.0
        assert config.f0_max == 800.0
        assert config.frame_length == 2048
        assert config.chunk_size == 8192
        assert config.batch_size == 4
        assert config.use_cuda is True
        assert config.use_half_precision is False
        assert config.speaker_embedding_dim == 256
        assert config.content_embedding_dim == 512
        assert config.max_retries == 3
        assert config.fallback_on_error is True
        assert config.enable_profiling is False
        assert config.cache_enabled is True
        assert config.cache_dir is None

    def test_config_custom_sample_rate(self):
        """Test custom sample rate configuration."""
        config = PipelineConfig(sample_rate=16000)
        pipeline = VoiceConversionPipeline(config)

        assert pipeline.config.sample_rate == 16000

    def test_config_custom_batch_size(self):
        """Test custom batch size configuration."""
        config = PipelineConfig(batch_size=16)
        pipeline = VoiceConversionPipeline(config)

        assert pipeline.config.batch_size == 16

    def test_config_half_precision(self):
        """Test half precision configuration."""
        config = PipelineConfig(use_half_precision=True)
        pipeline = VoiceConversionPipeline(config)

        assert pipeline.config.use_half_precision is True

    def test_config_profiling_enabled(self):
        """Test profiling enabled configuration."""
        config = PipelineConfig(enable_profiling=True)
        pipeline = VoiceConversionPipeline(config)

        assert pipeline.config.enable_profiling is True


class TestBatchProcessing:
    """Test batch processing edge cases."""

    def test_batch_convert_single_item(self, cpu_pipeline):
        """Test batch conversion with single item."""
        audio = np.random.randn(22050).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        results = cpu_pipeline.batch_convert([audio], [embedding])

        assert len(results) == 1
        assert isinstance(results[0], np.ndarray)

    def test_batch_convert_multiple_items(self, cpu_pipeline):
        """Test batch conversion with multiple items."""
        audio_list = [np.random.randn(22050).astype(np.float32) for _ in range(5)]
        embedding_list = [np.random.randn(256).astype(np.float32) for _ in range(5)]

        results = cpu_pipeline.batch_convert(audio_list, embedding_list)

        assert len(results) == 5
        for result in results:
            assert isinstance(result, np.ndarray)

    def test_batch_convert_length_mismatch(self, cpu_pipeline):
        """Test batch conversion with mismatched lengths."""
        audio_list = [np.random.randn(22050).astype(np.float32) for _ in range(3)]
        embedding_list = [np.random.randn(256).astype(np.float32) for _ in range(5)]

        with pytest.raises(VoiceConversionError):
            cpu_pipeline.batch_convert(audio_list, embedding_list)

    def test_batch_convert_empty_lists(self, cpu_pipeline):
        """Test batch conversion with empty lists."""
        results = cpu_pipeline.batch_convert([], [])

        assert len(results) == 0

    def test_batch_convert_with_error_fallback(self, cpu_pipeline):
        """Test batch conversion with errors and fallback."""
        audio_list = [np.random.randn(22050).astype(np.float32) for _ in range(3)]
        embedding_list = [np.random.randn(256).astype(np.float32) for _ in range(3)]

        # Mock error on second item
        original_convert = cpu_pipeline.convert
        call_count = [0]

        def mock_convert(audio, embedding, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Mock error on second item")
            return original_convert(audio, embedding, **kwargs)

        with patch.object(cpu_pipeline, 'convert', side_effect=mock_convert):
            results = cpu_pipeline.batch_convert(audio_list, embedding_list)

            # Should have 3 results (second one is fallback)
            assert len(results) == 3

    def test_batch_convert_no_fallback_on_error(self):
        """Test batch conversion raises error without fallback."""
        config = PipelineConfig(use_cuda=False, fallback_on_error=False)
        pipeline = VoiceConversionPipeline(config)

        audio_list = [np.random.randn(22050).astype(np.float32) for _ in range(2)]
        embedding_list = [np.random.randn(256).astype(np.float32) for _ in range(2)]

        with patch.object(pipeline, '_extract_features', side_effect=Exception("Mock error")):
            with pytest.raises(VoiceConversionError):
                pipeline.batch_convert(audio_list, embedding_list)


class TestWarmupFunctionality:
    """Test warmup functionality."""

    def test_warmup_default_iterations(self, cpu_pipeline):
        """Test warmup with default iterations."""
        initial_count = cpu_pipeline.stats['total_conversions']

        cpu_pipeline.warmup()

        assert cpu_pipeline.stats['total_conversions'] >= initial_count + 3

    def test_warmup_custom_iterations(self, cpu_pipeline):
        """Test warmup with custom iterations."""
        initial_count = cpu_pipeline.stats['total_conversions']
        num_iterations = 5

        cpu_pipeline.warmup(num_iterations=num_iterations)

        assert cpu_pipeline.stats['total_conversions'] >= initial_count + num_iterations

    def test_warmup_single_iteration(self, cpu_pipeline):
        """Test warmup with single iteration."""
        initial_count = cpu_pipeline.stats['total_conversions']

        cpu_pipeline.warmup(num_iterations=1)

        assert cpu_pipeline.stats['total_conversions'] >= initial_count + 1

    def test_warmup_handles_errors(self, cpu_pipeline):
        """Test warmup handles errors gracefully."""
        # Mock error on warmup
        with patch.object(cpu_pipeline, 'convert', side_effect=Exception("Mock error")):
            # Should not raise exception
            cpu_pipeline.warmup(num_iterations=2)


class TestStatisticsCollection:
    """Test statistics collection."""

    def test_stats_initialization(self, cpu_pipeline):
        """Test initial statistics values."""
        assert cpu_pipeline.stats['total_conversions'] >= 0
        assert cpu_pipeline.stats['successful_conversions'] >= 0
        assert cpu_pipeline.stats['failed_conversions'] >= 0
        assert cpu_pipeline.stats['average_processing_time'] >= 0.0

    def test_stats_update_on_success(self, cpu_pipeline):
        """Test statistics update on successful conversion."""
        initial_total = cpu_pipeline.stats['total_conversions']
        initial_success = cpu_pipeline.stats['successful_conversions']

        audio = np.random.randn(22050).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        cpu_pipeline.convert(audio, embedding)

        assert cpu_pipeline.stats['total_conversions'] == initial_total + 1
        assert cpu_pipeline.stats['successful_conversions'] == initial_success + 1

    def test_stats_update_on_failure_with_fallback(self, cpu_pipeline):
        """Test statistics update on failed conversion with fallback."""
        initial_total = cpu_pipeline.stats['total_conversions']
        initial_failed = cpu_pipeline.stats['failed_conversions']

        audio = np.random.randn(22050).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        with patch.object(cpu_pipeline, '_extract_features', side_effect=Exception("Mock error")):
            cpu_pipeline.convert(audio, embedding)

            assert cpu_pipeline.stats['total_conversions'] == initial_total + 1
            assert cpu_pipeline.stats['failed_conversions'] == initial_failed + 1

    def test_stats_average_processing_time(self, cpu_pipeline):
        """Test average processing time calculation."""
        audio = np.random.randn(22050).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        # Run multiple conversions
        for _ in range(3):
            cpu_pipeline.convert(audio, embedding)

        stats = cpu_pipeline.get_stats()
        assert stats['average_processing_time'] > 0

    def test_stats_success_rate(self, cpu_pipeline):
        """Test success rate calculation."""
        audio = np.random.randn(22050).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        # Successful conversion
        cpu_pipeline.convert(audio, embedding)

        stats = cpu_pipeline.get_stats()
        assert 'success_rate' in stats
        assert stats['success_rate'] >= 0.0
        assert stats['success_rate'] <= 1.0

    def test_stats_device_info(self, cpu_pipeline):
        """Test device information in statistics."""
        stats = cpu_pipeline.get_stats()

        assert 'device' in stats
        assert 'cuda_available' in stats
        assert stats['device'] == 'cpu'

    def test_stats_empty_initial_state(self):
        """Test statistics with no conversions."""
        config = PipelineConfig(use_cuda=False)
        pipeline = VoiceConversionPipeline(config)

        stats = pipeline.get_stats()
        assert stats['total_conversions'] == 0
        assert stats['success_rate'] == 0.0


class TestProfilingFeatures:
    """Test profiling functionality."""

    def test_profile_conversion_basic(self, cpu_pipeline):
        """Test basic profiling conversion."""
        audio = np.random.randn(22050).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        metrics = cpu_pipeline.profile_conversion(audio, embedding)

        assert 'total_ms' in metrics
        assert 'audio_duration_s' in metrics
        assert 'rtf' in metrics
        assert 'throughput_samples_per_sec' in metrics
        assert 'device' in metrics
        assert 'num_samples' in metrics
        assert 'sample_rate' in metrics

    def test_profile_conversion_rtf_calculation(self, cpu_pipeline):
        """Test RTF (Real-Time Factor) calculation."""
        audio = np.random.randn(22050 * 2).astype(np.float32)  # 2 seconds
        embedding = np.random.randn(256).astype(np.float32)

        metrics = cpu_pipeline.profile_conversion(audio, embedding, source_sample_rate=22050)

        assert metrics['audio_duration_s'] > 0
        assert metrics['rtf'] > 0
        assert metrics['total_ms'] > 0

    def test_profile_conversion_throughput(self, cpu_pipeline):
        """Test throughput calculation in profiling."""
        audio = np.random.randn(22050).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        metrics = cpu_pipeline.profile_conversion(audio, embedding)

        assert metrics['throughput_samples_per_sec'] > 0

    def test_profile_conversion_with_warmup(self, cpu_pipeline):
        """Test profiling after warmup for stable measurements."""
        cpu_pipeline.warmup(num_iterations=2)

        audio = np.random.randn(22050).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        metrics = cpu_pipeline.profile_conversion(audio, embedding)

        assert metrics['total_ms'] > 0
        assert metrics['rtf'] > 0


class TestAudioProcessingMethods:
    """Test audio processing helper methods."""

    def test_preprocess_audio_normalization(self, cpu_pipeline):
        """Test audio normalization in preprocessing."""
        audio = np.random.randn(22050).astype(np.float32) * 10.0  # Unnormalized

        result = cpu_pipeline._preprocess_audio(audio, 22050)

        assert torch.max(torch.abs(result)) <= 1.0

    def test_preprocess_audio_device_placement(self, cpu_pipeline):
        """Test audio is moved to correct device."""
        audio = np.random.randn(22050).astype(np.float32)

        result = cpu_pipeline._preprocess_audio(audio, 22050)

        assert result.device == cpu_pipeline.device

    def test_encode_speaker_normalization(self, cpu_pipeline):
        """Test speaker embedding normalization."""
        embedding = torch.randn(256).to(cpu_pipeline.device)

        result = cpu_pipeline._encode_speaker(embedding)

        # Should be L2 normalized
        norm = torch.norm(result, p=2, dim=-1)
        assert torch.allclose(norm, torch.ones(1).to(cpu_pipeline.device), atol=1e-5)

    def test_encode_speaker_dimension_expansion(self, cpu_pipeline):
        """Test speaker embedding dimension handling."""
        # 1D embedding
        embedding = torch.randn(256).to(cpu_pipeline.device)

        result = cpu_pipeline._encode_speaker(embedding)

        # Should have batch dimension
        assert result.ndim == 2

    def test_postprocess_audio_clipping(self, cpu_pipeline):
        """Test audio clipping in postprocessing."""
        audio = torch.randn(22050).to(cpu_pipeline.device) * 10.0  # Large values

        result = cpu_pipeline._postprocess_audio(audio, 22050)

        assert np.max(np.abs(result)) <= 1.0

    def test_postprocess_audio_device_transfer(self, cpu_pipeline):
        """Test audio is transferred to CPU in postprocessing."""
        audio = torch.randn(22050).to(cpu_pipeline.device)

        result = cpu_pipeline._postprocess_audio(audio, 22050)

        assert isinstance(result, np.ndarray)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_short_audio(self, cpu_pipeline):
        """Test conversion with very short audio (< 1 second)."""
        audio = np.random.randn(100).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        result = cpu_pipeline.convert(audio, embedding)
        assert result is not None

    def test_very_long_audio(self, cpu_pipeline):
        """Test conversion with long audio (> 10 seconds)."""
        audio = np.random.randn(220500).astype(np.float32)  # ~10 seconds
        embedding = np.random.randn(256).astype(np.float32)

        result = cpu_pipeline.convert(audio, embedding)
        assert result is not None

    def test_zero_audio(self, cpu_pipeline):
        """Test conversion with all-zero audio."""
        audio = np.zeros(22050).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        result = cpu_pipeline.convert(audio, embedding)
        assert result is not None

    def test_constant_audio(self, cpu_pipeline):
        """Test conversion with constant audio."""
        audio = np.ones(22050).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        result = cpu_pipeline.convert(audio, embedding)
        assert result is not None

    def test_zero_embedding(self, cpu_pipeline):
        """Test conversion with all-zero embedding."""
        audio = np.random.randn(22050).astype(np.float32)
        embedding = np.zeros(256).astype(np.float32)

        result = cpu_pipeline.convert(audio, embedding)
        assert result is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
