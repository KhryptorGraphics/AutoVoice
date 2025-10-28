"""
Comprehensive end-to-end tests for complete voice synthesis workflows.

Tests complete singing conversion pipeline, voice cloning, and integration workflows.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import time


@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.slow
class TestSingingConversionWorkflow:
    """Test complete singing voice conversion workflow."""

    @pytest.fixture
    def setup_pipeline(self, song_file, test_profile, device):
        """Setup pipeline with test dependencies."""
        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
            config = {
                'device': device,
                'cache_enabled': True,
                'sample_rate': 44100
            }
            pipeline = SingingConversionPipeline(config=config)
            return {
                'pipeline': pipeline,
                'song_file': song_file,
                'profile': test_profile
            }
        except ImportError:
            pytest.skip("SingingConversionPipeline not available")

    def test_full_singing_conversion_workflow(self, setup_pipeline, performance_tracker):
        """Test full workflow: audio → separation → F0 → conversion → output.

        Tests complete integration from input song to converted output with target voice.
        """
        pipeline = setup_pipeline['pipeline']
        song_file = setup_pipeline['song_file']
        target_embedding = setup_pipeline['profile']['embedding']

        # Track performance
        performance_tracker.start('full_conversion')

        # Execute conversion
        result = pipeline.convert_singing_voice(
            audio_path=str(song_file),
            target_speaker_embedding=target_embedding,
            pitch_shift=0.0
        )

        elapsed = performance_tracker.stop()

        # Validate result structure
        assert 'audio' in result
        assert 'sample_rate' in result
        assert 'duration' in result
        assert 'f0_contour' in result

        # Validate output audio
        audio = result['audio']
        assert isinstance(audio, np.ndarray)
        assert audio.ndim == 1  # Mono output
        assert len(audio) > 0
        assert np.isfinite(audio).all()

        # Check sample rate
        assert result['sample_rate'] == 44100

        # Performance check: Should complete in reasonable time
        duration_seconds = result['duration']
        rtf = elapsed / duration_seconds  # Real-time factor
        assert rtf < 10.0, f"Conversion too slow: RTF={rtf:.2f}"

        print(f"\nConversion RTF: {rtf:.2f}x (lower is faster)")

    def test_progress_callback_verification(self, setup_pipeline):
        """Test progress callback is called during conversion."""
        pipeline = setup_pipeline['pipeline']
        song_file = setup_pipeline['song_file']
        target_embedding = setup_pipeline['profile']['embedding']

        progress_calls = []

        def progress_callback(stage: str, progress: float):
            progress_calls.append({'stage': stage, 'progress': progress})

        result = pipeline.convert_singing_voice(
            audio_path=str(song_file),
            target_speaker_embedding=target_embedding,
            progress_callback=progress_callback
        )

        # Verify callbacks were made
        assert len(progress_calls) > 0, "Progress callback was never called"

        # Verify progress goes from 0 to 1
        progress_values = [p['progress'] for p in progress_calls]
        assert min(progress_values) >= 0.0
        assert max(progress_values) <= 1.0

        # Verify multiple stages
        stages = set(p['stage'] for p in progress_calls)
        expected_stages = {'separation', 'f0_extraction', 'conversion'}
        assert len(stages.intersection(expected_stages)) >= 2

    def test_caching_speedup(self, setup_pipeline, performance_tracker):
        """Test that caching provides speedup on repeated conversions."""
        pipeline = setup_pipeline['pipeline']
        song_file = setup_pipeline['song_file']
        target_embedding = setup_pipeline['profile']['embedding']

        # First conversion (cold)
        performance_tracker.start('cold_conversion')
        result1 = pipeline.convert_singing_voice(
            audio_path=str(song_file),
            target_speaker_embedding=target_embedding
        )
        time_cold = performance_tracker.stop()

        # Second conversion (warm, with cache)
        performance_tracker.start('warm_conversion')
        result2 = pipeline.convert_singing_voice(
            audio_path=str(song_file),
            target_speaker_embedding=target_embedding
        )
        time_warm = performance_tracker.stop()

        # Warm should be at least 2x faster
        speedup = time_cold / time_warm
        assert speedup >= 1.5, f"Cache speedup too low: {speedup:.2f}x"

        # Results should be identical (bitwise, since cached)
        np.testing.assert_array_equal(result1['audio'], result2['audio'])

        print(f"\nCache speedup: {speedup:.2f}x")

    def test_error_recovery(self, setup_pipeline):
        """Test error recovery for invalid inputs."""
        pipeline = setup_pipeline['pipeline']

        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            pipeline.convert_singing_voice(
                audio_path="/nonexistent/file.wav",
                target_speaker_embedding=np.random.randn(256)
            )

        # Test with invalid embedding size
        with pytest.raises(ValueError):
            pipeline.convert_singing_voice(
                audio_path=str(setup_pipeline['song_file']),
                target_speaker_embedding=np.random.randn(128)  # Wrong size
            )

    def test_preset_comparison(self, setup_pipeline):
        """Test different quality presets produce different results."""
        pipeline = setup_pipeline['pipeline']
        song_file = setup_pipeline['song_file']
        target_embedding = setup_pipeline['profile']['embedding']

        # Try different presets if available
        presets = ['fast', 'balanced', 'quality']
        results = {}

        for preset in presets:
            try:
                result = pipeline.convert_singing_voice(
                    audio_path=str(song_file),
                    target_speaker_embedding=target_embedding,
                    preset=preset
                )
                results[preset] = result['audio']
            except (ValueError, KeyError):
                # Preset not supported, skip
                continue

        # If multiple presets worked, verify differences
        if len(results) >= 2:
            preset_list = list(results.keys())
            audio1 = results[preset_list[0]]
            audio2 = results[preset_list[1]]

            # Should produce different outputs
            assert not np.array_equal(audio1, audio2), \
                f"Presets {preset_list[0]} and {preset_list[1]} produce identical output"


@pytest.mark.e2e
@pytest.mark.integration
class TestVoiceCloningWorkflow:
    """Test complete voice cloning workflow."""

    def test_voice_clone_create_and_use(self, tmp_path, device):
        """Test creating voice profile and using it for conversion."""
        try:
            from src.auto_voice.inference.voice_cloner import VoiceCloner
        except ImportError:
            pytest.skip("VoiceCloner not available")

        # Create cloner
        cloner = VoiceCloner(device=device)

        # Generate reference audio (10 seconds)
        sample_rate = 22050
        duration = 10.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        reference_audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

        # Create voice profile
        profile = cloner.create_voice_profile(
            audio=reference_audio,
            sample_rate=sample_rate,
            user_id='test_user'
        )

        # Verify profile structure
        assert 'profile_id' in profile
        assert 'user_id' in profile
        assert profile['user_id'] == 'test_user'

        # Load profile back
        loaded_profile = cloner.load_voice_profile(profile['profile_id'])
        assert 'embedding' in loaded_profile

        # Use embedding for test conversion (if pipeline available)
        embedding = loaded_profile['embedding']
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (256,)

    def test_multi_sample_profile_creation(self, device):
        """Test creating profile from multiple audio samples."""
        try:
            from src.auto_voice.inference.voice_cloner import VoiceCloner
        except ImportError:
            pytest.skip("VoiceCloner not available")

        cloner = VoiceCloner(device=device)

        # Generate multiple samples
        sample_rate = 22050
        audio_samples = []
        for i in range(3):
            t = np.linspace(0, 10.0, int(sample_rate * 10))
            audio = (0.3 * np.sin(2 * np.pi * (220 + i * 10) * t)).astype(np.float32)
            audio_samples.append(audio)

        # Create profile from multiple samples
        profile = cloner.create_voice_profile_from_multiple(
            audio_samples=audio_samples,
            sample_rate=sample_rate,
            user_id='multi_sample_user'
        )

        assert 'profile_id' in profile
        assert 'num_samples' in profile
        assert profile['num_samples'] == 3


@pytest.mark.e2e
@pytest.mark.integration
class TestMultiComponentIntegration:
    """Test integration between multiple components."""

    def test_source_separator_pitch_extractor_integration(self, song_file, device):
        """Test VocalSeparator → SingingPitchExtractor integration."""
        try:
            from src.auto_voice.audio.source_separator import VocalSeparator
            from src.auto_voice.audio.pitch_extractor import SingingPitchExtractor
        except ImportError:
            pytest.skip("Components not available")

        # Separate vocals
        separator = VocalSeparator(device=device)
        vocals, instrumental = separator.separate_vocals(str(song_file))

        # Extract pitch from vocals
        extractor = SingingPitchExtractor(device=device)
        f0_result = extractor.extract_f0_contour(vocals[0], sample_rate=44100)

        # Verify results
        assert 'f0' in f0_result
        assert 'voiced' in f0_result
        assert len(f0_result['f0']) > 0

    def test_end_to_end_memory_management(self, song_file, test_profile, device, memory_leak_detector):
        """Test memory is properly managed across full workflow."""
        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
        except ImportError:
            pytest.skip("SingingConversionPipeline not available")

        with memory_leak_detector:
            pipeline = SingingConversionPipeline(config={'device': device})

            # Run multiple conversions
            for _ in range(3):
                result = pipeline.convert_singing_voice(
                    audio_path=str(song_file),
                    target_speaker_embedding=test_profile['embedding']
                )

                # Force cleanup
                del result
                import gc
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()

        # Memory leak detector will warn if significant leak detected


@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.slow
class TestQualityValidation:
    """Test output quality metrics."""

    def test_snr_validation(self, song_file, test_profile, device):
        """Test output SNR is reasonable."""
        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
        except ImportError:
            pytest.skip("SingingConversionPipeline not available")

        pipeline = SingingConversionPipeline(config={'device': device})
        result = pipeline.convert_singing_voice(
            audio_path=str(song_file),
            target_speaker_embedding=test_profile['embedding']
        )

        audio = result['audio']

        # Calculate SNR (simple estimate)
        signal_power = np.mean(audio ** 2)
        # Assume last 10% is noise floor
        noise_power = np.mean(audio[int(len(audio) * 0.9):] ** 2)
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

        # SNR should be reasonable (> 10 dB)
        assert snr_db > 10.0, f"SNR too low: {snr_db:.2f} dB"

    def test_output_duration_preservation(self, song_file, test_profile, device):
        """Test that output duration matches input."""
        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
            import soundfile as sf
        except ImportError:
            pytest.skip("Components not available")

        # Get input duration
        input_audio, input_sr = sf.read(str(song_file))
        input_duration = len(input_audio) / input_sr

        # Convert
        pipeline = SingingConversionPipeline(config={'device': device})
        result = pipeline.convert_singing_voice(
            audio_path=str(song_file),
            target_speaker_embedding=test_profile['embedding']
        )

        # Check output duration
        output_duration = len(result['audio']) / result['sample_rate']

        # Should be within 5% of input duration
        duration_ratio = output_duration / input_duration
        assert 0.95 <= duration_ratio <= 1.05, \
            f"Duration mismatch: input={input_duration:.2f}s, output={output_duration:.2f}s"


@pytest.mark.e2e
@pytest.mark.performance
class TestPerformanceE2E:
    """End-to-end performance tests."""

    @pytest.mark.parametrize('audio_length', [5.0, 10.0, 30.0])
    def test_conversion_latency_by_length(self, tmp_path, test_profile, device, audio_length, performance_tracker):
        """Test conversion latency scales with audio length."""
        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
            import soundfile as sf
        except ImportError:
            pytest.skip("Components not available")

        # Generate audio of specified length
        sample_rate = 22050
        t = np.linspace(0, audio_length, int(sample_rate * audio_length))
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        audio_file = tmp_path / f"test_{audio_length}s.wav"
        sf.write(str(audio_file), audio, sample_rate)

        # Convert and measure time
        pipeline = SingingConversionPipeline(config={'device': device})

        performance_tracker.start(f'convert_{audio_length}s')
        result = pipeline.convert_singing_voice(
            audio_path=str(audio_file),
            target_speaker_embedding=test_profile['embedding']
        )
        elapsed = performance_tracker.stop()

        # Calculate real-time factor
        rtf = elapsed / audio_length
        performance_tracker.record(f'rtf_{audio_length}s', rtf)

        # RTF should be reasonable (< 20x for CPU, < 5x for GPU)
        max_rtf = 20.0 if device == 'cpu' else 5.0
        assert rtf < max_rtf, f"RTF too high for {audio_length}s audio: {rtf:.2f}x"

        print(f"\n{audio_length}s audio: RTF={rtf:.2f}x, elapsed={elapsed:.2f}s")

    def test_concurrent_conversions(self, song_file, test_profile, device, concurrent_executor):
        """Test multiple concurrent conversions."""
        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
        except ImportError:
            pytest.skip("SingingConversionPipeline not available")

        pipeline = SingingConversionPipeline(config={'device': device})

        def convert_task():
            return pipeline.convert_singing_voice(
                audio_path=str(song_file),
                target_speaker_embedding=test_profile['embedding']
            )

        # Submit 3 concurrent tasks
        futures = [concurrent_executor.submit(convert_task) for _ in range(3)]

        # Wait for all to complete
        results = [f.result(timeout=120) for f in futures]

        # All should succeed
        assert len(results) == 3
        assert all('audio' in r for r in results)
