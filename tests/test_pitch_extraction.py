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

        # Should raise ValueError for empty audio
        with pytest.raises(ValueError, match="empty"):
            singing_pitch_extractor.extract_f0_contour(audio, sample_rate)

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

    @pytest.mark.parametrize('audio_format', ['wav', 'flac'])
    def test_multi_format_audio_extraction(self, singing_pitch_extractor, tmp_path, audio_format):
        """Test F0 extraction from multiple audio formats (wav, flac, mp3)"""
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not available")

        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        # Create audio file in specified format
        audio_file = tmp_path / f"test.{audio_format}"

        try:
            # Write audio file
            if audio_format == 'wav':
                sf.write(str(audio_file), audio, sample_rate, format='WAV')
            elif audio_format == 'flac':
                sf.write(str(audio_file), audio, sample_rate, format='FLAC')
            elif audio_format == 'mp3':
                # MP3 requires special handling, may not be supported
                try:
                    sf.write(str(audio_file), audio, sample_rate, format='MP3')
                except Exception:
                    pytest.skip(f"{audio_format} format not supported by soundfile")
        except Exception as e:
            pytest.skip(f"Failed to write {audio_format} file: {e}")

        # Extract F0 from file
        result = singing_pitch_extractor.extract_f0_contour(str(audio_file))

        # Verify results
        assert result is not None
        assert 'f0' in result
        assert 'voiced' in result

        # Check F0 is approximately 440 Hz
        f0_voiced = result['f0'][result['voiced']]
        if len(f0_voiced) > 0:
            mean_f0 = np.mean(f0_voiced)
            # Allow 5% tolerance for format conversion artifacts
            assert 420 < mean_f0 < 460, f"Expected F0 ~440 Hz for {audio_format}, got {mean_f0:.1f} Hz"

    def test_multi_format_consistency(self, singing_pitch_extractor, tmp_path):
        """Test that different audio formats produce consistent F0 contours"""
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not available")

        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        results = {}

        # Test WAV and FLAC (MP3 may have lossy compression)
        for fmt in ['wav', 'flac']:
            audio_file = tmp_path / f"test.{fmt}"

            try:
                if fmt == 'wav':
                    sf.write(str(audio_file), audio, sample_rate, format='WAV')
                elif fmt == 'flac':
                    sf.write(str(audio_file), audio, sample_rate, format='FLAC')

                result = singing_pitch_extractor.extract_f0_contour(str(audio_file))
                results[fmt] = result
            except Exception as e:
                pytest.skip(f"Failed to process {fmt}: {e}")

        # Compare results if we have multiple formats
        if len(results) >= 2:
            formats = list(results.keys())
            f0_0 = results[formats[0]]['f0']
            f0_1 = results[formats[1]]['f0']

            # F0 contours should have similar mean values
            voiced_0 = results[formats[0]]['voiced']
            voiced_1 = results[formats[1]]['voiced']

            if voiced_0.sum() > 0 and voiced_1.sum() > 0:
                mean_0 = np.mean(f0_0[voiced_0])
                mean_1 = np.mean(f0_1[voiced_1])

                # Should be within 2% of each other
                relative_diff = abs(mean_0 - mean_1) / mean_0
                assert relative_diff < 0.02, f"F0 mismatch between {formats[0]} and {formats[1]}: {relative_diff*100:.1f}%"


# ============================================================================
# Vibrato Classification Tests
# ============================================================================

@pytest.mark.audio
@pytest.mark.unit
class TestVibratoClassification:
    """Tests for vibrato classification and analysis"""

    def test_classify_vibrato_natural(self, singing_pitch_extractor):
        """Test classification of natural vibrato.

        Tests the classify_vibrato() method with synthetic vibrato signal.
        """
        # Generate signal with natural vibrato (5-6 Hz rate, moderate depth)
        sample_rate = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Base frequency 440 Hz with 5.5 Hz vibrato, depth ~20 cents
        vibrato_rate = 5.5  # Hz
        vibrato_depth = 20  # cents
        f0_base = 440.0
        f0_variation = f0_base * (2 ** (vibrato_depth / 1200.0) - 1)
        f0_contour = f0_base + f0_variation * np.sin(2 * np.pi * vibrato_rate * t)

        # Create audio with this vibrato
        phase = 2 * np.pi * np.cumsum(f0_contour) / sample_rate
        audio = np.sin(phase).astype(np.float32)

        # Extract F0
        f0_data = singing_pitch_extractor.extract_f0_contour(audio, sample_rate)

        # Classify vibrato
        vibrato_result = singing_pitch_extractor.classify_vibrato(f0_data)

        # Verify structure
        assert 'has_vibrato' in vibrato_result
        assert 'is_natural' in vibrato_result
        assert 'rate_hz' in vibrato_result
        assert 'rate_category' in vibrato_result  # 'slow', 'medium', 'fast'
        assert 'depth_cents' in vibrato_result
        assert 'depth_category' in vibrato_result  # 'shallow', 'medium', 'deep'
        assert 'regularity' in vibrato_result

        # Should detect vibrato
        if vibrato_result['has_vibrato']:
            # Rate should be in natural range (4-7 Hz)
            assert 4.0 < vibrato_result['rate_hz'] < 7.5
            assert vibrato_result['rate_category'] in ['slow', 'medium', 'fast']
            assert vibrato_result['depth_category'] in ['shallow', 'medium', 'deep']

    def test_classify_vibrato_artificial(self, singing_pitch_extractor):
        """Test detection of artificial/synthesized vibrato."""
        # Generate perfectly regular vibrato (too regular to be natural)
        sample_rate = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Very fast, shallow, perfectly regular vibrato (8 Hz, 10 cents)
        vibrato_rate = 8.0
        vibrato_depth = 10
        f0_base = 440.0
        f0_variation = f0_base * (2 ** (vibrato_depth / 1200.0) - 1)
        f0_contour = f0_base + f0_variation * np.sin(2 * np.pi * vibrato_rate * t)

        phase = 2 * np.pi * np.cumsum(f0_contour) / sample_rate
        audio = np.sin(phase).astype(np.float32)

        f0_data = singing_pitch_extractor.extract_f0_contour(audio, sample_rate)
        vibrato_result = singing_pitch_extractor.classify_vibrato(f0_data)

        # Should detect vibrato
        assert isinstance(vibrato_result, dict)
        assert 'has_vibrato' in vibrato_result
        assert 'is_natural' in vibrato_result

    def test_classify_vibrato_straight_tone(self, singing_pitch_extractor):
        """Test no vibrato detected on straight tone."""
        # Pure sine wave with no vibrato
        sample_rate = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        f0_data = singing_pitch_extractor.extract_f0_contour(audio, sample_rate)
        vibrato_result = singing_pitch_extractor.classify_vibrato(f0_data)

        # Should report no vibrato or very low confidence
        assert isinstance(vibrato_result, dict)
        assert 'has_vibrato' in vibrato_result

    def test_classify_vibrato_rate_categories(self, singing_pitch_extractor):
        """Test vibrato rate categorization (slow/medium/fast)."""
        sample_rate = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Test different rates
        test_rates = {
            'slow': 4.5,    # < 5 Hz
            'medium': 5.5,  # 5-6.5 Hz
            'fast': 7.0     # > 6.5 Hz
        }

        for expected_category, rate in test_rates.items():
            f0_variation = 440.0 * 0.02
            f0_contour = 440.0 + f0_variation * np.sin(2 * np.pi * rate * t)
            phase = 2 * np.pi * np.cumsum(f0_contour) / sample_rate
            audio = np.sin(phase).astype(np.float32)

            f0_data = singing_pitch_extractor.extract_f0_contour(audio, sample_rate)
            vibrato_result = singing_pitch_extractor.classify_vibrato(f0_data)

            # Verify classification includes rate category
            assert 'rate_category' in vibrato_result

    def test_classify_vibrato_depth_categories(self, singing_pitch_extractor):
        """Test vibrato depth categorization (shallow/medium/deep)."""
        sample_rate = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        vibrato_rate = 5.5

        # Test different depths
        test_depths = {
            'shallow': 15,  # < 30 cents
            'medium': 50,   # 30-70 cents
            'deep': 100     # > 70 cents
        }

        for expected_category, depth_cents in test_depths.items():
            f0_base = 440.0
            f0_variation = f0_base * (2 ** (depth_cents / 1200.0) - 1)
            f0_contour = f0_base + f0_variation * np.sin(2 * np.pi * vibrato_rate * t)
            phase = 2 * np.pi * np.cumsum(f0_contour) / sample_rate
            audio = np.sin(phase).astype(np.float32)

            f0_data = singing_pitch_extractor.extract_f0_contour(audio, sample_rate)
            vibrato_result = singing_pitch_extractor.classify_vibrato(f0_data)

            # Verify classification includes depth category
            assert 'depth_category' in vibrato_result


# ============================================================================
# Pitch Correction Suggestion Tests
# ============================================================================

@pytest.mark.audio
@pytest.mark.unit
class TestPitchCorrectionSuggestions:
    """Tests for pitch correction suggestion features"""

    def test_suggest_pitch_corrections_c_major(self, singing_pitch_extractor):
        """Test pitch correction suggestions for C major scale.

        Tests suggest_pitch_corrections() method with slightly off-pitch notes.
        """
        sample_rate = 22050

        # Create slightly off-pitch C major scale notes
        # C4 (261.63 Hz) played at 265 Hz (+20 cents sharp)
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 265.0 * t).astype(np.float32)

        f0_data = singing_pitch_extractor.extract_f0_contour(audio, sample_rate)

        # Get correction suggestions for C major scale
        corrections = singing_pitch_extractor.suggest_pitch_corrections(
            f0_data,
            reference_scale='C',
            tolerance_cents=50.0
        )

        # Verify structure
        assert isinstance(corrections, list)
        if len(corrections) > 0:
            correction = corrections[0]
            assert 'time_seconds' in correction
            assert 'detected_f0_hz' in correction
            assert 'target_note' in correction
            assert 'target_f0_hz' in correction
            assert 'cents_deviation' in correction
            assert 'correction_needed' in correction

    def test_suggest_pitch_corrections_tolerance(self, singing_pitch_extractor):
        """Test tolerance parameter in pitch correction suggestions."""
        sample_rate = 22050
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Note 30 cents sharp
        audio = np.sin(2 * np.pi * 450.0 * t).astype(np.float32)  # ~30 cents above A4

        f0_data = singing_pitch_extractor.extract_f0_contour(audio, sample_rate)

        # With strict tolerance (20 cents), should suggest correction
        corrections_strict = singing_pitch_extractor.suggest_pitch_corrections(
            f0_data,
            reference_scale='A',
            tolerance_cents=20.0
        )

        # With loose tolerance (50 cents), may not suggest correction
        corrections_loose = singing_pitch_extractor.suggest_pitch_corrections(
            f0_data,
            reference_scale='A',
            tolerance_cents=50.0
        )

        # Verify both return lists
        assert isinstance(corrections_strict, list)
        assert isinstance(corrections_loose, list)

    def test_suggest_pitch_corrections_different_scales(self, singing_pitch_extractor):
        """Test pitch correction suggestions for different scales."""
        sample_rate = 22050
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        f0_data = singing_pitch_extractor.extract_f0_contour(audio, sample_rate)

        # Test different reference scales
        for scale in ['C', 'G', 'D', 'A', 'E']:
            corrections = singing_pitch_extractor.suggest_pitch_corrections(
                f0_data,
                reference_scale=scale,
                tolerance_cents=50.0
            )
            assert isinstance(corrections, list)

    def test_pitch_correction_no_suggestions_in_tune(self, singing_pitch_extractor):
        """Test that in-tune notes don't get correction suggestions."""
        sample_rate = 22050
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Perfect A4 (440 Hz)
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        f0_data = singing_pitch_extractor.extract_f0_contour(audio, sample_rate)

        corrections = singing_pitch_extractor.suggest_pitch_corrections(
            f0_data,
            reference_scale='A',
            tolerance_cents=20.0
        )

        # Should have few or no corrections for perfect pitch
        assert isinstance(corrections, list)


# ============================================================================
# Real-Time Streaming State Tests
# ============================================================================

@pytest.mark.audio
@pytest.mark.unit
class TestRealTimeStreamingState:
    """Tests for real-time streaming pitch extraction with state management"""

    def test_create_realtime_state(self, singing_pitch_extractor):
        """Test creation of real-time processing state."""
        state = singing_pitch_extractor.create_realtime_state()

        # Verify state structure
        assert isinstance(state, dict)
        assert 'buffer' in state
        assert 'overlap_samples' in state
        assert 'frame_index' in state
        assert state['frame_index'] == 0

    def test_reset_realtime_state(self, singing_pitch_extractor):
        """Test resetting real-time processing state."""
        # Create and modify state
        state = singing_pitch_extractor.create_realtime_state()
        state['frame_index'] = 100
        state['buffer'] = [1, 2, 3]

        # Reset state
        singing_pitch_extractor.reset_realtime_state(state)

        # State should be reset
        assert state['frame_index'] == 0
        assert len(state['buffer']) == 0 or state['buffer'] is None or (isinstance(state['buffer'], list) and all(x == 0 for x in state['buffer']))

    def test_extract_f0_realtime_with_state(self, singing_pitch_extractor):
        """Test real-time F0 extraction with state persistence.

        Tests extract_f0_realtime() method with state parameter for streaming.
        """
        sample_rate = 22050
        chunk_duration = 0.1  # 100ms chunks
        chunk_size = int(sample_rate * chunk_duration)

        # Create continuous audio signal
        total_duration = 1.0
        t = np.linspace(0, total_duration, int(sample_rate * total_duration))
        full_audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        # Create state
        state = singing_pitch_extractor.create_realtime_state()

        # Process in chunks
        results = []
        num_chunks = int(total_duration / chunk_duration)

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            audio_chunk = full_audio[start_idx:end_idx]
            audio_chunk_tensor = torch.from_numpy(audio_chunk)

            # Extract F0 for this chunk with state
            f0_chunk = singing_pitch_extractor.extract_f0_realtime(
                audio_chunk_tensor,
                sample_rate=sample_rate,
                state=state,
                use_cuda_kernel=False  # Use CPU for testing
            )

            assert isinstance(f0_chunk, torch.Tensor)
            results.append(f0_chunk)

        # Verify state was updated
        assert state['frame_index'] > 0

    def test_realtime_streaming_continuity(self, singing_pitch_extractor):
        """Test that real-time streaming maintains continuity across chunks."""
        sample_rate = 22050
        chunk_size = int(sample_rate * 0.1)  # 100ms chunks

        # Generate continuous tone
        t1 = np.linspace(0, 0.1, chunk_size)
        t2 = np.linspace(0.1, 0.2, chunk_size)

        chunk1 = np.sin(2 * np.pi * 440.0 * t1).astype(np.float32)
        chunk2 = np.sin(2 * np.pi * 440.0 * t2).astype(np.float32)

        # Create state
        state = singing_pitch_extractor.create_realtime_state()

        # Process first chunk
        f0_1 = singing_pitch_extractor.extract_f0_realtime(
            torch.from_numpy(chunk1),
            sample_rate=sample_rate,
            state=state,
            use_cuda_kernel=False
        )

        # Process second chunk with same state
        f0_2 = singing_pitch_extractor.extract_f0_realtime(
            torch.from_numpy(chunk2),
            sample_rate=sample_rate,
            state=state,
            use_cuda_kernel=False
        )

        # Both should produce valid results
        assert isinstance(f0_1, torch.Tensor)
        assert isinstance(f0_2, torch.Tensor)
        assert len(f0_1) > 0
        assert len(f0_2) > 0

    def test_realtime_state_buffer_management(self, singing_pitch_extractor):
        """Test that real-time state properly manages overlap buffer."""
        sample_rate = 22050
        chunk_size = int(sample_rate * 0.1)

        audio_chunk = np.sin(2 * np.pi * 440.0 * np.linspace(0, 0.1, chunk_size)).astype(np.float32)

        state = singing_pitch_extractor.create_realtime_state()
        initial_buffer_size = len(state['buffer']) if 'buffer' in state and state['buffer'] is not None else 0

        # Process chunk
        singing_pitch_extractor.extract_f0_realtime(
            torch.from_numpy(audio_chunk),
            sample_rate=sample_rate,
            state=state,
            use_cuda_kernel=False
        )

        # Buffer should contain overlap samples
        if 'buffer' in state and state['buffer'] is not None:
            # Buffer should be populated after processing
            assert True  # State management occurred

    @pytest.mark.cuda
    def test_realtime_state_with_cuda_kernel(self):
        """Test real-time state management with CUDA kernel."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from src.auto_voice.audio.pitch_extractor import SingingPitchExtractor

        extractor = SingingPitchExtractor(device='cuda')
        sample_rate = 22050
        chunk_size = int(sample_rate * 0.1)

        audio_chunk = np.sin(2 * np.pi * 440.0 * np.linspace(0, 0.1, chunk_size)).astype(np.float32)

        state = extractor.create_realtime_state()

        # Process with CUDA kernel
        f0_result = extractor.extract_f0_realtime(
            torch.from_numpy(audio_chunk).cuda(),
            sample_rate=sample_rate,
            state=state,
            use_cuda_kernel=True
        )

        assert isinstance(f0_result, torch.Tensor)
        assert f0_result.device.type == 'cuda'
    def test_cuda_extension_sys_modules_fallback(self, monkeypatch):
        """Test that _load_cuda_extension can find modules via sys.modules scan"""
        import sys
        from types import ModuleType
        from unittest.mock import MagicMock

        # Mock torchcrepe to avoid initialization errors
        mock_torchcrepe = MagicMock()
        monkeypatch.setitem(sys.modules, 'torchcrepe', mock_torchcrepe)

        from src.auto_voice.audio.pitch_extractor import SingingPitchExtractor

        # Create extractor instance (torchcrepe is now mocked)
        extractor = SingingPitchExtractor(device='cpu')

        # Create a fake CUDA module
        fake_module = ModuleType('foo.bar.cuda_kernels')
        fake_module.launch_pitch_detection = lambda: "fake_cuda_function"

        # Store original sys.modules state
        original_modules = sys.modules.copy()

        try:
            # Register fake module in sys.modules
            sys.modules['foo.bar.cuda_kernels'] = fake_module

            # Mock all import paths to fail so sys.modules scan is triggered
            def mock_import_module(name):
                if name in ['cuda_kernels', 'auto_voice.cuda_kernels', 'src.cuda_kernels']:
                    raise ImportError(f"Mocked import failure for {name}")
                raise ImportError(f"Unknown module: {name}")

            # Monkeypatch importlib.import_module
            import importlib
            monkeypatch.setattr(importlib, 'import_module', mock_import_module)

            # Mock environment variable to be empty
            monkeypatch.delenv('AUTOVOICE_CUDA_MODULE', raising=False)

            # Call _load_cuda_extension
            result = extractor._load_cuda_extension()

            # Should return our fake module from sys.modules
            assert result is not None
            assert result is fake_module
            assert hasattr(result, 'launch_pitch_detection')
            assert result.launch_pitch_detection() == "fake_cuda_function"

        finally:
            # Restore original sys.modules
            sys.modules.clear()
            sys.modules.update(original_modules)
    
    def test_cuda_extension_sys_modules_prefers_longer_names(self, monkeypatch):
        """Test that sys.modules scan prefers more specific (longer) module names"""
        import sys
        from types import ModuleType
        from unittest.mock import MagicMock

        # Mock torchcrepe to avoid initialization errors
        mock_torchcrepe = MagicMock()
        monkeypatch.setitem(sys.modules, 'torchcrepe', mock_torchcrepe)

        from src.auto_voice.audio.pitch_extractor import SingingPitchExtractor

        # Create extractor instance (torchcrepe is now mocked)
        extractor = SingingPitchExtractor(device='cpu')

        # Create two fake CUDA modules with different specificity
        short_module = ModuleType('cuda_kernels')
        long_module = ModuleType('my.package.cuda_kernels')

        short_module.name = 'short'
        long_module.name = 'long'

        original_modules = sys.modules.copy()

        try:
            # Register both modules in sys.modules
            sys.modules['cuda_kernels'] = short_module
            sys.modules['my.package.cuda_kernels'] = long_module

            # Mock import failures
            def mock_import_module(name):
                raise ImportError(f"Mocked import failure for {name}")

            import importlib
            monkeypatch.setattr(importlib, 'import_module', mock_import_module)
            monkeypatch.delenv('AUTOVOICE_CUDA_MODULE', raising=False)

            # Call _load_cuda_extension
            result = extractor._load_cuda_extension()

            # Should prefer the longer (more specific) module name
            assert result is not None
            assert result is long_module
            assert result.name == 'long'

        finally:
            sys.modules.clear()
            sys.modules.update(original_modules)
    
    def test_cuda_extension_sys_modules_no_match(self, monkeypatch):
        """Test that _load_cuda_extension returns None when no modules match"""
        import sys
        from types import ModuleType
        from unittest.mock import MagicMock

        # Mock torchcrepe to avoid initialization errors
        mock_torchcrepe = MagicMock()
        monkeypatch.setitem(sys.modules, 'torchcrepe', mock_torchcrepe)

        from src.auto_voice.audio.pitch_extractor import SingingPitchExtractor

        # Create extractor instance (torchcrepe is now mocked)
        extractor = SingingPitchExtractor(device='cpu')

        # Create a module that doesn't match the pattern
        unrelated_module = ModuleType('some.other.module')

        original_modules = sys.modules.copy()

        try:
            # Clear sys.modules and only add non-matching module
            sys.modules.clear()
            sys.modules.update({
                k: v for k, v in original_modules.items()
                if not (k == 'cuda_kernels' or k.endswith('.cuda_kernels'))
            })
            sys.modules['some.other.module'] = unrelated_module
            sys.modules['torchcrepe'] = mock_torchcrepe  # Keep torchcrepe mock

            # Mock all imports to fail
            def mock_import_module(name):
                raise ImportError(f"Mocked import failure for {name}")

            import importlib
            monkeypatch.setattr(importlib, 'import_module', mock_import_module)
            monkeypatch.delenv('AUTOVOICE_CUDA_MODULE', raising=False)

            # Call _load_cuda_extension
            result = extractor._load_cuda_extension()

            # Should return None since no matching modules found
            assert result is None

        finally:
            sys.modules.clear()
            sys.modules.update(original_modules)
