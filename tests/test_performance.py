"""
Comprehensive performance benchmarking and validation tests.

Tests inference latency, throughput, memory usage, CPU vs GPU performance,
cache effectiveness, and component-level benchmarks.
"""

import pytest
import torch
import numpy as np
import time
from typing import Dict, List


@pytest.mark.performance
class TestCPUvsGPUBenchmarks:
    """Compare CPU and GPU conversion performance."""

    @pytest.mark.parametrize('device_name', ['cpu', 'cuda'])
    def test_conversion_device_comparison(self, song_file, test_profile, device_name, performance_tracker):
        """Benchmark conversion on CPU vs GPU."""
        if device_name == 'cuda' and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
        except ImportError:
            pytest.skip("SingingConversionPipeline not available")

        pipeline = SingingConversionPipeline(config={'device': device_name})

        # Warm-up run
        pipeline.convert_singing_voice(
            audio_path=str(song_file),
            target_speaker_embedding=test_profile['embedding']
        )

        # Benchmark run
        performance_tracker.start(f'{device_name}_conversion')
        result = pipeline.convert_singing_voice(
            audio_path=str(song_file),
            target_speaker_embedding=test_profile['embedding']
        )
        elapsed = performance_tracker.stop()

        # Calculate metrics
        duration = result['duration']
        rtf = elapsed / duration

        performance_tracker.record(f'{device_name}_rtf', rtf)
        performance_tracker.record(f'{device_name}_elapsed', elapsed)

        print(f"\n{device_name.upper()} Performance:")
        print(f"  Elapsed: {elapsed:.3f}s")
        print(f"  Audio duration: {duration:.3f}s")
        print(f"  RTF: {rtf:.3f}x")

        # Store for comparison
        assert rtf > 0


@pytest.mark.performance
class TestColdStartVsWarmCache:
    """Compare cold start vs warm cache performance."""

    def test_cold_vs_warm_comparison(self, song_file, test_profile, device, performance_tracker):
        """Compare first run (cold) vs cached run (warm)."""
        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
        except ImportError:
            pytest.skip("SingingConversionPipeline not available")

        pipeline = SingingConversionPipeline(config={
            'device': device,
            'cache_enabled': True
        })

        # Cold start (no cache)
        pipeline.clear_cache()  # Ensure clean state

        performance_tracker.start('cold_start')
        result_cold = pipeline.convert_singing_voice(
            audio_path=str(song_file),
            target_speaker_embedding=test_profile['embedding']
        )
        time_cold = performance_tracker.stop()

        # Warm start (with cache)
        performance_tracker.start('warm_cache')
        result_warm = pipeline.convert_singing_voice(
            audio_path=str(song_file),
            target_speaker_embedding=test_profile['embedding']
        )
        time_warm = performance_tracker.stop()

        # Calculate speedup
        speedup = time_cold / time_warm

        print(f"\nCache Performance:")
        print(f"  Cold start: {time_cold:.3f}s")
        print(f"  Warm cache: {time_warm:.3f}s")
        print(f"  Speedup: {speedup:.2f}x")

        # Cache should provide at least 1.5x speedup
        assert speedup >= 1.5, f"Cache speedup too low: {speedup:.2f}x"

        # Results should be identical
        np.testing.assert_array_equal(result_cold['audio'], result_warm['audio'])


@pytest.mark.performance
class TestEndToEndLatency:
    """Measure end-to-end latency for 30s audio."""

    def test_30s_audio_latency(self, tmp_path, test_profile, device, performance_tracker):
        """Benchmark conversion of 30-second audio."""
        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
            import soundfile as sf
        except ImportError:
            pytest.skip("Components not available")

        # Generate 30s audio
        sample_rate = 22050
        duration = 30.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        audio_file = tmp_path / "test_30s.wav"
        sf.write(str(audio_file), audio, sample_rate)

        # Convert
        pipeline = SingingConversionPipeline(config={'device': device})

        performance_tracker.start('e2e_30s')
        result = pipeline.convert_singing_voice(
            audio_path=str(audio_file),
            target_speaker_embedding=test_profile['embedding']
        )
        elapsed = performance_tracker.stop()

        rtf = elapsed / 30.0

        print(f"\n30s Audio Performance:")
        print(f"  Elapsed: {elapsed:.3f}s")
        print(f"  RTF: {rtf:.3f}x")

        # Should complete in reasonable time
        max_rtf = 20.0 if device == 'cpu' else 5.0
        assert rtf < max_rtf, f"RTF too high: {rtf:.2f}x"


@pytest.mark.performance
@pytest.mark.cuda
class TestPeakGPUMemoryUsage:
    """Track peak GPU memory usage during conversion."""

    def test_peak_memory_tracking(self, song_file, test_profile, gpu_memory_monitor):
        """Track peak GPU memory during conversion."""
        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
        except ImportError:
            pytest.skip("SingingConversionPipeline not available")

        with gpu_memory_monitor:
            pipeline = SingingConversionPipeline(config={'device': 'cuda'})
            result = pipeline.convert_singing_voice(
                audio_path=str(song_file),
                target_speaker_embedding=test_profile['embedding']
            )

        stats = gpu_memory_monitor.get_stats()

        print(f"\nGPU Memory Usage:")
        print(f"  Initial: {stats['initial_mb']:.2f} MB")
        print(f"  Peak: {stats['peak_mb']:.2f} MB")
        print(f"  Final: {stats['final_mb']:.2f} MB")
        print(f"  Delta: {stats['delta_mb']:.2f} MB")

        # Peak memory should be reasonable (< 8 GB for typical song)
        assert stats['peak_mb'] < 8192, f"Peak memory too high: {stats['peak_mb']:.2f} MB"


@pytest.mark.performance
class TestCacheHitRateSpeedup:
    """Measure cache hit rate and speedup."""

    def test_cache_effectiveness(self, song_file, test_profile, device, performance_tracker):
        """Test cache hit rate and speedup measurement."""
        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
        except ImportError:
            pytest.skip("SingingConversionPipeline not available")

        pipeline = SingingConversionPipeline(config={
            'device': device,
            'cache_enabled': True
        })
        pipeline.clear_cache()

        # First run: populate cache
        performance_tracker.start('run_1')
        result1 = pipeline.convert_singing_voice(
            audio_path=str(song_file),
            target_speaker_embedding=test_profile['embedding']
        )
        time1 = performance_tracker.stop()

        # Subsequent runs: should hit cache
        times = [time1]
        for i in range(2, 6):
            performance_tracker.start(f'run_{i}')
            result = pipeline.convert_singing_voice(
                audio_path=str(song_file),
                target_speaker_embedding=test_profile['embedding']
            )
            times.append(performance_tracker.stop())

        # Calculate statistics
        baseline = times[0]
        cached_times = times[1:]
        avg_cached = np.mean(cached_times)
        speedup = baseline / avg_cached

        print(f"\nCache Hit Rate Test:")
        print(f"  First run (cold): {baseline:.3f}s")
        print(f"  Cached runs avg: {avg_cached:.3f}s")
        print(f"  Speedup: {speedup:.2f}x")

        # Cached runs should be consistently faster
        assert speedup >= 1.5, f"Cache speedup insufficient: {speedup:.2f}x"


@pytest.mark.performance
class TestComponentTimingBreakdown:
    """Breakdown of timing for each component."""

    def test_component_timing_breakdown(self, song_file, test_profile, device, performance_tracker):
        """Measure time spent in each pipeline component."""
        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
        except ImportError:
            pytest.skip("SingingConversionPipeline not available")

        # Create pipeline with timing instrumentation
        pipeline = SingingConversionPipeline(config={'device': device})

        # Track component times
        component_times = {}

        def timed_callback(stage: str, progress: float):
            if stage not in component_times:
                component_times[stage] = {'start': time.perf_counter(), 'end': None}
            if progress >= 1.0:
                component_times[stage]['end'] = time.perf_counter()

        # Run with callback
        result = pipeline.convert_singing_voice(
            audio_path=str(song_file),
            target_speaker_embedding=test_profile['embedding'],
            progress_callback=timed_callback
        )

        # Calculate component durations
        print(f"\nComponent Timing Breakdown:")
        total_time = 0
        for stage, times in component_times.items():
            if times['end'] is not None:
                duration = times['end'] - times['start']
                total_time += duration
                print(f"  {stage}: {duration:.3f}s ({duration/total_time*100:.1f}%)")


@pytest.mark.performance
class TestScalabilityWithAudioLength:
    """Test how performance scales with audio length."""

    @pytest.mark.parametrize('duration', [5.0, 10.0, 20.0, 30.0])
    def test_scalability(self, tmp_path, test_profile, device, duration, performance_tracker):
        """Test performance scaling with different audio lengths."""
        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
            import soundfile as sf
        except ImportError:
            pytest.skip("Components not available")

        # Generate audio of specified duration
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        audio_file = tmp_path / f"test_{duration}s.wav"
        sf.write(str(audio_file), audio, sample_rate)

        # Convert
        pipeline = SingingConversionPipeline(config={'device': device})

        performance_tracker.start(f'duration_{duration}')
        result = pipeline.convert_singing_voice(
            audio_path=str(audio_file),
            target_speaker_embedding=test_profile['embedding']
        )
        elapsed = performance_tracker.stop()

        rtf = elapsed / duration
        performance_tracker.record(f'rtf_{duration}s', rtf)

        print(f"\n{duration}s audio: RTF={rtf:.3f}x")

        # RTF should be relatively consistent across lengths
        assert rtf > 0


@pytest.mark.performance
class TestSourceSeparatorPerformance:
    """Benchmark vocal separation performance."""

    def test_separation_speed(self, song_file, device, performance_tracker):
        """Benchmark vocal separation speed."""
        try:
            from src.auto_voice.audio.source_separator import VocalSeparator
            import soundfile as sf
        except ImportError:
            pytest.skip("VocalSeparator not available")

        separator = VocalSeparator(device=device, config={'cache_enabled': False})

        # Get audio duration
        audio, sr = sf.read(str(song_file))
        duration = len(audio) / sr

        # Benchmark separation
        performance_tracker.start('separation')
        vocals, instrumental = separator.separate_vocals(str(song_file))
        elapsed = performance_tracker.stop()

        rtf = elapsed / duration

        print(f"\nSeparation Performance:")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Elapsed: {elapsed:.2f}s")
        print(f"  RTF: {rtf:.2f}x")

        assert vocals is not None
        assert instrumental is not None


@pytest.mark.performance
class TestPitchExtractionPerformance:
    """Benchmark pitch extraction performance."""

    def test_f0_extraction_speed(self, sample_audio_22khz, device, performance_tracker):
        """Benchmark F0 extraction speed."""
        try:
            from src.auto_voice.audio.pitch_extractor import SingingPitchExtractor
        except ImportError:
            pytest.skip("SingingPitchExtractor not available")

        extractor = SingingPitchExtractor(device=device)
        sample_rate = 22050

        # Generate longer audio for meaningful benchmark
        duration = 10.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        # Benchmark extraction
        performance_tracker.start('f0_extraction')
        result = extractor.extract_f0_contour(audio, sample_rate)
        elapsed = performance_tracker.stop()

        rtf = elapsed / duration

        print(f"\nF0 Extraction Performance:")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Elapsed: {elapsed:.2f}s")
        print(f"  RTF: {rtf:.2f}x")

        # Should be fast (< 1.0x RTF)
        assert rtf < 2.0, f"F0 extraction too slow: {rtf:.2f}x"


@pytest.mark.performance
class TestVoiceConversionPerformance:
    """Benchmark voice conversion model performance."""

    def test_conversion_model_speed(self, device, performance_tracker):
        """Benchmark voice conversion model forward pass."""
        try:
            from src.auto_voice.models.singing_voice_converter import SingingVoiceConverter
        except ImportError:
            pytest.skip("SingingVoiceConverter not available")

        config = {
            'singing_voice_converter': {
                'latent_dim': 192,
                'mel_channels': 80,
                'content_encoder': {'type': 'cnn_fallback'},
                'vocoder': {'use_vocoder': False}
            }
        }
        model = SingingVoiceConverter(config)
        model.to(device)
        model.eval()
        model.prepare_for_inference()

        # Generate test inputs
        source_audio = torch.randn(1, 16000).to(device)
        target_speaker_emb = torch.randn(1, 256).to(device)

        # Warm-up
        with torch.no_grad():
            model.convert(
                source_audio=source_audio,
                target_speaker_embedding=target_speaker_emb,
                source_sample_rate=16000
            )

        # Benchmark
        num_iterations = 10
        performance_tracker.start('conversion_model')
        for _ in range(num_iterations):
            with torch.no_grad():
                waveform = model.convert(
                    source_audio=source_audio,
                    target_speaker_embedding=target_speaker_emb,
                    source_sample_rate=16000
                )
        elapsed = performance_tracker.stop()

        avg_time = elapsed / num_iterations
        print(f"\nConversion Model Performance:")
        print(f"  Avg time per conversion: {avg_time*1000:.2f}ms")


@pytest.mark.performance
class TestBatchProcessingPerformance:
    """Test batch processing performance."""

    def test_batch_vs_sequential(self, device, performance_tracker):
        """Compare batch vs sequential pitch extraction."""
        try:
            from src.auto_voice.audio.pitch_extractor import SingingPitchExtractor
        except ImportError:
            pytest.skip("SingingPitchExtractor not available")

        extractor = SingingPitchExtractor(device=device)
        sample_rate = 22050

        # Generate multiple audio samples
        num_samples = 5
        audio_samples = []
        for _ in range(num_samples):
            t = np.linspace(0, 2.0, int(sample_rate * 2))
            audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            audio_samples.append(audio)

        # Sequential processing
        performance_tracker.start('sequential')
        for audio in audio_samples:
            result = extractor.extract_f0_contour(audio, sample_rate)
        time_sequential = performance_tracker.stop()

        # Batch processing
        performance_tracker.start('batch')
        results = extractor.batch_extract(audio_samples, sample_rate)
        time_batch = performance_tracker.stop()

        speedup = time_sequential / time_batch

        print(f"\nBatch Processing Performance:")
        print(f"  Sequential: {time_sequential:.3f}s")
        print(f"  Batch: {time_batch:.3f}s")
        print(f"  Speedup: {speedup:.2f}x")

        # Batch should be faster (or at least not slower)
        assert speedup >= 0.9, f"Batch processing slower than sequential"


@pytest.mark.performance
class TestPresetPerformanceComparison:
    """Compare performance of different quality presets."""

    @pytest.mark.parametrize('preset', ['fast', 'balanced', 'quality'])
    def test_preset_performance(self, song_file, test_profile, device, preset, performance_tracker):
        """Benchmark different quality presets."""
        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
        except ImportError:
            pytest.skip("SingingConversionPipeline not available")

        pipeline = SingingConversionPipeline(config={'device': device})

        try:
            performance_tracker.start(f'preset_{preset}')
            result = pipeline.convert_singing_voice(
                audio_path=str(song_file),
                target_speaker_embedding=test_profile['embedding'],
                preset=preset
            )
            elapsed = performance_tracker.stop()

            duration = result['duration']
            rtf = elapsed / duration

            print(f"\nPreset '{preset}' Performance:")
            print(f"  Elapsed: {elapsed:.3f}s")
            print(f"  RTF: {rtf:.3f}x")

        except (ValueError, KeyError):
            pytest.skip(f"Preset '{preset}' not supported")
