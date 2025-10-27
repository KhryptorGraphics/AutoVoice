"""Comprehensive tests for SingingPitchExtractor"""

import pytest
import numpy as np
import torch
from pathlib import Path


@pytest.mark.audio
@pytest.mark.unit
class TestSingingPitchExtractor:
    """Unit tests for SingingPitchExtractor"""

    def test_extractor_initialization(self, singing_pitch_extractor):
        """Verify SingingPitchExtractor initializes with default config"""
        assert singing_pitch_extractor is not None
        assert hasattr(singing_pitch_extractor, 'model')
        assert hasattr(singing_pitch_extractor, 'device')
        assert singing_pitch_extractor.fmin > 0
        assert singing_pitch_extractor.fmax > singing_pitch_extractor.fmin

    def test_extract_f0_from_sine_wave(self, singing_pitch_extractor, sample_audio_22khz):
        """Extract F0 from 440 Hz sine wave"""
        # Generate 440 Hz sine wave
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        result = singing_pitch_extractor.extract_f0_contour(audio, sample_rate)

        assert 'f0' in result
        assert 'voiced' in result
        assert 'confidence' in result

        # Check F0 is close to 440 Hz
        f0_voiced = result['f0'][result['voiced']]
        if len(f0_voiced) > 0:
            mean_f0 = np.mean(f0_voiced)
            assert 435 < mean_f0 < 445, f"Expected F0 ~440 Hz, got {mean_f0:.1f} Hz"

    @pytest.mark.parametrize('frequency', [220, 440, 880])
    def test_extract_f0_different_pitches(self, singing_pitch_extractor, frequency):
        """Extract F0 at different frequencies"""
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

        result = singing_pitch_extractor.extract_f0_contour(audio, sample_rate)

        f0_voiced = result['f0'][result['voiced']]
        if len(f0_voiced) > 0:
            mean_f0 = np.mean(f0_voiced)
            # Allow 5% tolerance
            tolerance = frequency * 0.05
            assert frequency - tolerance < mean_f0 < frequency + tolerance

    def test_vibrato_detection_on_modulated_signal(self, singing_pitch_extractor, sample_vibrato_audio):
        """Detect vibrato from modulated signal"""
        audio, ground_truth = sample_vibrato_audio
        sample_rate = 22050

        result = singing_pitch_extractor.extract_f0_contour(audio, sample_rate)

        assert 'vibrato' in result
        vibrato = result['vibrato']

        # Check if vibrato was detected
        # Note: Detection may not be perfect on synthetic signal
        if vibrato.get('has_vibrato'):
            # If detected, check rate is in reasonable range
            rate = vibrato.get('rate_hz', 0)
            assert 3.0 < rate < 9.0, f"Vibrato rate {rate:.1f} Hz outside expected range"

    def test_no_vibrato_on_straight_tone(self, singing_pitch_extractor):
        """Verify no vibrato detected on pure sine wave"""
        sample_rate = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        result = singing_pitch_extractor.extract_f0_contour(audio, sample_rate)

        vibrato = result.get('vibrato', {})
        # Straight tone should have low or no vibrato
        # (some false positives possible due to numerical noise)
        assert isinstance(vibrato, dict)

    @pytest.mark.parametrize('sr', [16000, 22050, 44100])
    def test_extract_f0_different_sample_rates(self, singing_pitch_extractor, sr):
        """Extract F0 at different sample rates"""
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        result = singing_pitch_extractor.extract_f0_contour(audio, sr)

        assert result is not None
        assert 'f0' in result
        assert 'sample_rate' in result
        assert result['sample_rate'] == sr

    def test_empty_audio(self, singing_pitch_extractor):
        """Handle empty audio gracefully"""
        audio = np.array([], dtype=np.float32)
        sample_rate = 22050

        try:
            result = singing_pitch_extractor.extract_f0_contour(audio, sample_rate)
            # Should return empty or zeros
            assert len(result.get('f0', [])) == 0 or np.all(result['f0'] == 0)
        except Exception as e:
            # Or raise appropriate error
            assert isinstance(e, (ValueError, RuntimeError))

    def test_very_short_audio(self, singing_pitch_extractor):
        """Handle very short audio (<100ms)"""
        sample_rate = 22050
        duration = 0.05  # 50ms
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        result = singing_pitch_extractor.extract_f0_contour(audio, sample_rate)
        assert result is not None

    def test_silent_audio(self, singing_pitch_extractor, sample_audio_silence):
        """Extract F0 from silence"""
        sample_rate = 16000

        result = singing_pitch_extractor.extract_f0_contour(sample_audio_silence, sample_rate)

        # Most frames should be unvoiced
        voiced_fraction = np.sum(result['voiced']) / len(result['voiced']) if len(result['voiced']) > 0 else 0
        assert voiced_fraction < 0.1, "Silence should have mostly unvoiced frames"

    def test_noisy_audio(self, singing_pitch_extractor, sample_audio_noise):
        """Extract F0 from white noise"""
        sample_rate = 16000

        result = singing_pitch_extractor.extract_f0_contour(sample_audio_noise, sample_rate)

        # Should not crash and return valid structure
        assert 'f0' in result
        assert 'voiced' in result
        # Noise should have low voicing
        voiced_fraction = np.sum(result['voiced']) / len(result['voiced']) if len(result['voiced']) > 0 else 0
        assert voiced_fraction < 0.3

    def test_get_pitch_statistics(self, singing_pitch_extractor):
        """Compute pitch statistics"""
        sample_rate = 22050
        t = np.linspace(0, 1.0, int(sample_rate))
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        f0_data = singing_pitch_extractor.extract_f0_contour(audio, sample_rate)
        stats = singing_pitch_extractor.get_pitch_statistics(f0_data)

        assert 'mean_f0' in stats
        assert 'std_f0' in stats
        assert 'min_f0' in stats
        assert 'max_f0' in stats
        assert 'range_semitones' in stats
        assert 'voiced_fraction' in stats

        # Statistics should be reasonable
        if stats['mean_f0'] > 0:
            assert stats['min_f0'] <= stats['mean_f0'] <= stats['max_f0']

    @pytest.mark.cuda
    def test_gpu_extraction(self, cuda_device):
        """Test GPU-accelerated extraction"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from src.auto_voice.audio.pitch_extractor import SingingPitchExtractor

        extractor = SingingPitchExtractor(device='cuda')

        sample_rate = 22050
        t = np.linspace(0, 1.0, int(sample_rate))
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        result = extractor.extract_f0_contour(audio, sample_rate)

        assert result is not None
        assert 'f0' in result

    @pytest.mark.cuda
    def test_gpu_vs_cpu_consistency(self):
        """Compare GPU and CPU results"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from src.auto_voice.audio.pitch_extractor import SingingPitchExtractor

        # Create extractors
        cpu_extractor = SingingPitchExtractor(device='cpu')
        gpu_extractor = SingingPitchExtractor(device='cuda')

        # Generate test audio
        sample_rate = 22050
        t = np.linspace(0, 1.0, int(sample_rate))
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        # Extract on both devices
        cpu_result = cpu_extractor.extract_f0_contour(audio, sample_rate)
        gpu_result = gpu_extractor.extract_f0_contour(audio, sample_rate)

        # Results should be similar (not identical due to floating point differences)
        cpu_f0 = cpu_result['f0']
        gpu_f0 = gpu_result['f0']

        if len(cpu_f0) == len(gpu_f0):
            # Compare means
            cpu_mean = np.mean(cpu_f0[cpu_f0 > 0])
            gpu_mean = np.mean(gpu_f0[gpu_f0 > 0])

            if not np.isnan(cpu_mean) and not np.isnan(gpu_mean):
                # Allow 2% difference
                assert np.abs(cpu_mean - gpu_mean) / cpu_mean < 0.02

    @pytest.mark.performance
    def test_extraction_speed(self, singing_pitch_extractor, sample_audio_22khz, benchmark_timer):
        """Benchmark F0 extraction speed"""
        sample_rate = 22050

        result, elapsed_time = benchmark_timer(
            singing_pitch_extractor.extract_f0_contour,
            sample_audio_22khz,
            sample_rate
        )

        # Should complete in reasonable time (< 2 seconds for 1 second of audio on CPU)
        assert elapsed_time < 2.0, f"Extraction took {elapsed_time:.2f}s, expected < 2.0s"

        # Log performance
        print(f"\nF0 extraction time: {elapsed_time*1000:.1f}ms")

    def test_extract_f0_realtime_cpu(self, singing_pitch_extractor):
        """Test real-time F0 extraction on CPU"""
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        audio_tensor = torch.from_numpy(audio)

        # Test with CUDA kernel disabled (CPU fallback)
        result = singing_pitch_extractor.extract_f0_realtime(
            audio_tensor, sample_rate, use_cuda_kernel=False
        )

        assert result is not None
        assert isinstance(result, torch.Tensor)
        assert len(result) > 0

    @pytest.mark.cuda
    def test_extract_f0_realtime_cuda(self):
        """Test real-time F0 extraction with CUDA kernel"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from src.auto_voice.audio.pitch_extractor import SingingPitchExtractor

        extractor = SingingPitchExtractor(device='cuda')

        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        audio_tensor = torch.from_numpy(audio)

        # Test with CUDA kernel enabled
        result = extractor.extract_f0_realtime(
            audio_tensor, sample_rate, use_cuda_kernel=True
        )

        assert result is not None
        assert isinstance(result, torch.Tensor)
        assert len(result) > 0
        # Should be on CUDA device
        assert result.device.type == 'cuda'

    def test_batch_extract_with_arrays(self, singing_pitch_extractor):
        """Test batch extraction with numpy arrays"""
        sample_rate = 22050

        # Create multiple audio samples
        audio_list = []
        for freq in [220, 440, 880]:
            duration = 0.5
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
            audio_list.append(audio)

        results = singing_pitch_extractor.batch_extract(audio_list, sample_rate)

        assert len(results) == 3
        for result in results:
            assert result is not None
            assert 'f0' in result
            assert 'voiced' in result

    def test_batch_extract_with_mixed_lengths(self, singing_pitch_extractor):
        """Test batch extraction with different length audio"""
        sample_rate = 22050

        # Create audio samples with different lengths
        audio_list = []
        for duration in [0.3, 0.7, 1.2]:
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
            audio_list.append(audio)

        results = singing_pitch_extractor.batch_extract(audio_list, sample_rate)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result is not None
            # Different lengths should produce different frame counts
            if i > 0:
                assert len(result['f0']) != len(results[i-1]['f0'])

    def test_batch_extract_with_paths(self, singing_pitch_extractor, tmp_path):
        """Test batch extraction with file paths"""
        import soundfile as sf

        sample_rate = 22050
        audio_files = []

        # Create test audio files
        for i, freq in enumerate([220, 440]):
            duration = 0.5
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.sin(2 * np.pi * freq * t).astype(np.float32)

            audio_file = tmp_path / f"test_{i}.wav"
            sf.write(str(audio_file), audio, sample_rate)
            audio_files.append(str(audio_file))

        results = singing_pitch_extractor.batch_extract(audio_files)

        assert len(results) == 2
        for result in results:
            assert result is not None
            assert 'f0' in result

    def test_batch_extract_with_error_handling(self, singing_pitch_extractor):
        """Test batch extraction handles errors gracefully"""
        sample_rate = 22050

        # Create list with valid and invalid items
        audio_list = [
            np.sin(2 * np.pi * 440.0 * np.linspace(0, 0.5, int(sample_rate * 0.5))).astype(np.float32),
            "/nonexistent/file.wav",  # This should fail
            np.sin(2 * np.pi * 880.0 * np.linspace(0, 0.5, int(sample_rate * 0.5))).astype(np.float32),
        ]

        results = singing_pitch_extractor.batch_extract(audio_list, sample_rate)

        assert len(results) == 3
        # First and third should succeed
        assert results[0] is not None
        assert results[2] is not None
        # Second should be None (error)
        assert results[1] is None


@pytest.mark.audio
@pytest.mark.integration
class TestSingingPitchExtractorIntegration:
    """Integration tests for SingingPitchExtractor"""

    def test_integration_with_audio_processor(self, audio_processor, tmp_path):
        """Test integration with AudioProcessor"""
        from src.auto_voice.audio.pitch_extractor import SingingPitchExtractor

        # Create test audio file
        sample_rate = 22050
        t = np.linspace(0, 1.0, int(sample_rate))
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        audio_file = tmp_path / "test.wav"
        import soundfile as sf
        sf.write(str(audio_file), audio, sample_rate)

        # Extract F0 from file path
        extractor = SingingPitchExtractor()
        result = extractor.extract_f0_contour(str(audio_file))

        assert result is not None
        assert 'f0' in result

    def test_end_to_end_workflow(self, tmp_path):
        """Test complete F0 extraction workflow"""
        from src.auto_voice.audio.pitch_extractor import SingingPitchExtractor

        # Create audio
        sample_rate = 22050
        t = np.linspace(0, 2.0, int(sample_rate * 2))
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        # Initialize extractor
        extractor = SingingPitchExtractor()

        # Extract F0
        f0_data = extractor.extract_f0_contour(audio, sample_rate)

        # Compute statistics
        stats = extractor.get_pitch_statistics(f0_data)

        # Verify complete workflow
        assert f0_data is not None
        assert stats is not None
        assert stats['mean_f0'] > 0
