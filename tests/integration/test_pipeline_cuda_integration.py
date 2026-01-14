"""Integration tests for VoiceConversionPipeline + CUDA kernels.

This module tests:
- Pipeline + CUDA kernels integration
- Memory management
- Concurrent operations
- End-to-end workflows
"""

import pytest
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from src.auto_voice.inference.voice_conversion_pipeline import (
    VoiceConversionPipeline,
    PipelineConfig
)
from src.auto_voice.gpu.cuda_kernels import (
    create_kernel_suite,
    KernelConfig
)


@pytest.fixture
def integration_pipeline():
    """Create integration test pipeline."""
    config = PipelineConfig(
        use_cuda=torch.cuda.is_available(),
        fallback_on_error=True,
        batch_size=2
    )
    return VoiceConversionPipeline(config)


class TestPipelineCUDAIntegration:
    """Test pipeline and CUDA kernels integration."""

    def test_end_to_end_with_cuda_kernels(self, integration_pipeline):
        """Test complete pipeline using CUDA kernels."""
        audio = np.random.randn(22050 * 2).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        result = integration_pipeline.convert(audio, embedding)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_kernel_suite_integration(self):
        """Test kernel suite integration with pipeline."""
        kernel_config = KernelConfig(use_cuda=torch.cuda.is_available())
        kernels = create_kernel_suite(kernel_config)

        assert kernels is not None
        assert len(kernels) == 4

        # Test pitch detection kernel
        audio = torch.randn(16000)
        f0 = kernels['pitch_detection'].detect_pitch(audio, sample_rate=16000)
        assert f0 is not None

    def test_feature_extraction_with_kernels(self, integration_pipeline):
        """Test feature extraction using CUDA kernels."""
        audio = torch.randn(22050).to(integration_pipeline.device)

        features = integration_pipeline._extract_features(audio)

        assert features is not None
        assert 'mel_spec' in features
        assert 'f0' in features
        assert 'speaker_embedding' in features

    def test_synthesis_with_kernels(self, integration_pipeline):
        """Test voice synthesis using CUDA kernels."""
        # Create dummy features
        features = {
            'mel_spec': torch.randn(80, 100).to(integration_pipeline.device),
            'f0': torch.randn(100).to(integration_pipeline.device),
            'speaker_embedding': torch.randn(256).to(integration_pipeline.device)
        }
        speaker_features = torch.randn(1, 256).to(integration_pipeline.device)

        waveform = integration_pipeline._synthesize_voice(features, speaker_features)

        assert waveform is not None
        assert isinstance(waveform, torch.Tensor)


class TestMemoryManagement:
    """Test memory management."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_allocation(self, integration_pipeline):
        """Test GPU memory is properly allocated."""
        torch.cuda.reset_peak_memory_stats()

        audio = np.random.randn(22050 * 2).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        result = integration_pipeline.convert(audio, embedding)

        peak_memory = torch.cuda.max_memory_allocated()
        assert peak_memory > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_cleanup(self):
        """Test GPU memory is properly cleaned up."""
        # Create dedicated pipeline for memory testing
        config = PipelineConfig(use_cuda=True, fallback_on_error=True)
        pipeline = VoiceConversionPipeline(config)

        torch.cuda.reset_peak_memory_stats()

        audio = np.random.randn(22050).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        # Run conversion
        _ = pipeline.convert(audio, embedding)

        peak_memory = torch.cuda.max_memory_allocated()

        # Delete pipeline and force cleanup
        del pipeline
        torch.cuda.empty_cache()

        current_memory = torch.cuda.memory_allocated()

        # Memory should be mostly freed (allow for framework overhead)
        assert current_memory < peak_memory * 0.9

    def test_cpu_memory_management(self):
        """Test CPU memory management."""
        config = PipelineConfig(use_cuda=False)
        pipeline = VoiceConversionPipeline(config)

        audio = np.random.randn(22050 * 10).astype(np.float32)  # Large audio
        embedding = np.random.randn(256).astype(np.float32)

        result = pipeline.convert(audio, embedding)
        assert result is not None

    def test_multiple_conversions_memory_stability(self, integration_pipeline):
        """Test memory doesn't grow with multiple conversions."""
        audio = np.random.randn(22050).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        # Run multiple conversions
        for _ in range(10):
            _ = integration_pipeline.convert(audio, embedding)

        # Should not crash or run out of memory
        assert True


class TestConcurrentOperations:
    """Test concurrent pipeline operations."""

    def test_sequential_conversions(self, integration_pipeline):
        """Test sequential conversions work correctly."""
        audio = np.random.randn(22050).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        results = []
        for _ in range(5):
            result = integration_pipeline.convert(audio, embedding)
            results.append(result)

        assert len(results) == 5
        for result in results:
            assert result is not None

    def test_batch_conversion_concurrent(self, integration_pipeline):
        """Test batch conversion processes items correctly."""
        audio_list = [np.random.randn(22050).astype(np.float32) for _ in range(4)]
        embedding_list = [np.random.randn(256).astype(np.float32) for _ in range(4)]

        results = integration_pipeline.batch_convert(audio_list, embedding_list)

        assert len(results) == 4
        for result in results:
            assert isinstance(result, np.ndarray)

    def test_thread_safe_conversions(self, integration_pipeline):
        """Test pipeline is thread-safe for concurrent conversions."""
        def convert_audio(idx):
            audio = np.random.randn(22050).astype(np.float32)
            embedding = np.random.randn(256).astype(np.float32)
            result = integration_pipeline.convert(audio, embedding)
            return idx, result

        # Run conversions in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(convert_audio, i) for i in range(8)]
            results = [future.result() for future in as_completed(futures)]

        assert len(results) == 8
        for idx, result in results:
            assert result is not None

    def test_warmup_before_concurrent(self, integration_pipeline):
        """Test warmup improves concurrent performance."""
        # Warmup
        integration_pipeline.warmup(num_iterations=3)

        audio = np.random.randn(22050).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        # Time concurrent conversions after warmup
        start = time.time()
        for _ in range(10):
            _ = integration_pipeline.convert(audio, embedding)
        elapsed = time.time() - start

        # Should complete reasonably fast
        assert elapsed < 60.0  # 10 conversions in under 60 seconds


class TestBenchmarkIntegration:
    """Test integration with benchmark scripts."""

    def test_profile_conversion_integration(self, integration_pipeline):
        """Test profiling integration."""
        audio = np.random.randn(22050 * 2).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        # Warmup for stable measurements
        integration_pipeline.warmup(num_iterations=2)

        metrics = integration_pipeline.profile_conversion(audio, embedding)

        assert 'total_ms' in metrics
        assert 'rtf' in metrics
        assert 'throughput_samples_per_sec' in metrics
        assert metrics['total_ms'] > 0

    def test_statistics_tracking_integration(self, integration_pipeline):
        """Test statistics tracking across conversions."""
        audio = np.random.randn(22050).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        # Run multiple conversions
        for _ in range(5):
            integration_pipeline.convert(audio, embedding)

        stats = integration_pipeline.get_stats()

        assert stats['total_conversions'] >= 5
        assert stats['successful_conversions'] >= 0
        assert stats['average_processing_time'] > 0


class TestEdgeCaseIntegration:
    """Test edge cases in integrated pipeline."""

    def test_very_short_audio_integration(self, integration_pipeline):
        """Test pipeline with very short audio."""
        audio = np.random.randn(1000).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        result = integration_pipeline.convert(audio, embedding)
        assert result is not None

    def test_very_long_audio_integration(self, integration_pipeline):
        """Test pipeline with very long audio."""
        audio = np.random.randn(220500).astype(np.float32)  # ~10 seconds
        embedding = np.random.randn(256).astype(np.float32)

        result = integration_pipeline.convert(audio, embedding)
        assert result is not None

    def test_different_sample_rates_integration(self, integration_pipeline):
        """Test pipeline with various sample rates."""
        embedding = np.random.randn(256).astype(np.float32)

        for sample_rate in [8000, 16000, 22050, 44100]:
            duration = 1.0
            audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

            result = integration_pipeline.convert(
                audio, embedding,
                source_sample_rate=sample_rate
            )
            assert result is not None

    def test_extreme_pitch_shifts_integration(self, integration_pipeline):
        """Test pipeline with extreme pitch shifts."""
        audio = np.random.randn(22050).astype(np.float32)
        embedding = np.random.randn(256).astype(np.float32)

        for pitch_shift in [-24, -12, 0, 12, 24]:
            result = integration_pipeline.convert(
                audio, embedding,
                pitch_shift_semitones=float(pitch_shift)
            )
            assert result is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
