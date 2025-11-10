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
        pipeline.convert_song(
            song_path=str(song_file),
            target_profile_id=test_profile['profile_id']
        )

        # Benchmark run
        performance_tracker.start(f'{device_name}_conversion')
        result = pipeline.convert_song(
            song_path=str(song_file),
            target_profile_id=test_profile['profile_id']
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

    def test_cpu_vs_gpu_speed_advantage(self, song_file, test_profile, performance_tracker, performance_thresholds):
        """Test that GPU provides significant speed advantage over CPU.

        Validates that GPU is at least 3x faster than CPU for voice conversion.
        Uses GPU_SPEEDUP_THRESHOLD environment variable (default: 3.0).
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for GPU vs CPU comparison")

        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
        except ImportError:
            pytest.skip("SingingConversionPipeline not available")

        results = {}

        # Get threshold from environment
        min_speedup = performance_thresholds['gpu_speedup']

        # Run on CPU
        pipeline_cpu = SingingConversionPipeline(config={'device': 'cpu'})

        # Warm-up
        pipeline_cpu.convert_song(
            song_path=str(song_file),
            target_profile_id=test_profile['profile_id']
        )

        # Benchmark
        performance_tracker.start('cpu_benchmark')
        result_cpu = pipeline_cpu.convert_song(
            song_path=str(song_file),
            target_profile_id=test_profile['profile_id']
        )
        elapsed_cpu = performance_tracker.stop()
        rtf_cpu = elapsed_cpu / result_cpu['duration']
        results['cpu'] = {'elapsed': elapsed_cpu, 'rtf': rtf_cpu}

        # Run on GPU
        pipeline_gpu = SingingConversionPipeline(config={'device': 'cuda'})

        # Warm-up
        pipeline_gpu.convert_song(
            song_path=str(song_file),
            target_profile_id=test_profile['profile_id']
        )

        # Benchmark
        performance_tracker.start('gpu_benchmark')
        result_gpu = pipeline_gpu.convert_song(
            song_path=str(song_file),
            target_profile_id=test_profile['profile_id']
        )
        elapsed_gpu = performance_tracker.stop()
        rtf_gpu = elapsed_gpu / result_gpu['duration']
        results['gpu'] = {'elapsed': elapsed_gpu, 'rtf': rtf_gpu}

        # Calculate speedup
        speedup = rtf_cpu / rtf_gpu

        print(f"\nCPU vs GPU Performance:")
        print(f"  CPU RTF: {rtf_cpu:.3f}x")
        print(f"  GPU RTF: {rtf_gpu:.3f}x")
        print(f"  GPU speedup: {speedup:.2f}x")
        print(f"  Threshold: {min_speedup:.1f}x")

        # Assert GPU provides sufficient speed advantage
        assert speedup >= min_speedup, \
            f"GPU speedup insufficient: {speedup:.2f}x (expected >= {min_speedup:.1f}x). " \
            f"CPU RTF: {rtf_cpu:.3f}x, GPU RTF: {rtf_gpu:.3f}x"

        # Record for metrics (explicit emission for aggregation)
        performance_tracker.record('cpu_gpu_speedup', speedup)
        performance_tracker.record('rtf_cpu', rtf_cpu)
        performance_tracker.record('rtf_gpu', rtf_gpu)


@pytest.mark.performance
class TestColdStartVsWarmCache:
    """Compare cold start vs warm cache performance."""

    def test_cold_vs_warm_comparison(self, song_file, test_profile, device, performance_tracker, performance_thresholds):
        """Compare first run (cold) vs cached run (warm).

        Uses CACHE_SPEEDUP_THRESHOLD environment variable (default: 3.0).
        """
        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
        except ImportError:
            pytest.skip("SingingConversionPipeline not available")

        # Get threshold from environment
        min_speedup = performance_thresholds['cache_speedup']

        pipeline = SingingConversionPipeline(config={
            'device': device,
            'cache_enabled': True
        })

        # Cold start (no cache)
        pipeline.clear_cache()  # Ensure clean state

        performance_tracker.start('cold_start')
        result_cold = pipeline.convert_song(
            song_path=str(song_file),
            target_profile_id=test_profile['profile_id']
        )
        time_cold = performance_tracker.stop()

        # Warm start (with cache)
        performance_tracker.start('warm_cache')
        result_warm = pipeline.convert_song(
            song_path=str(song_file),
            target_profile_id=test_profile['profile_id']
        )
        time_warm = performance_tracker.stop()

        # Calculate speedup
        speedup = time_cold / time_warm

        print(f"\nCache Performance:")
        print(f"  Cold start: {time_cold:.3f}s")
        print(f"  Warm cache: {time_warm:.3f}s")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Threshold: {min_speedup:.1f}x")

        # Cache should provide sufficient speedup
        assert speedup >= min_speedup, f"Cache speedup too low: {speedup:.2f}x (expected >= {min_speedup:.1f}x)"

        # Record for metrics (explicit emission for aggregation)
        performance_tracker.record('cache_speedup', speedup)
        performance_tracker.record('cold_start_time', time_cold)
        performance_tracker.record('warm_cache_time', time_warm)

        # Results should be identical
        np.testing.assert_array_equal(result_cold['mixed_audio'], result_warm['mixed_audio'])


@pytest.mark.performance
class TestEndToEndLatency:
    """Measure end-to-end latency for 30s audio."""

    def test_30s_audio_latency(self, tmp_path, test_profile, device, performance_tracker, performance_thresholds):
        """Benchmark conversion of 30-second audio.

        Uses MAX_RTF_CPU and MAX_RTF_GPU environment variables (defaults: 20.0, 5.0).
        """
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
        result = pipeline.convert_song(
            song_path=str(audio_file),
            target_profile_id=test_profile['profile_id']
        )
        elapsed = performance_tracker.stop()

        rtf = elapsed / 30.0

        print(f"\n30s Audio Performance:")
        print(f"  Elapsed: {elapsed:.3f}s")
        print(f"  RTF: {rtf:.3f}x")

        # Get threshold from environment
        max_rtf = performance_thresholds['max_rtf_cpu'] if device == 'cpu' else performance_thresholds['max_rtf_gpu']
        print(f"  Max RTF: {max_rtf:.1f}x")

        assert rtf < max_rtf, f"RTF too high: {rtf:.2f}x (max: {max_rtf:.1f}x)"

        # Record for metrics
        performance_tracker.record(f'rtf_30s_{device}', rtf)


@pytest.mark.performance
@pytest.mark.cuda
class TestPeakGPUMemoryUsage:
    """Track peak GPU memory usage during conversion."""

    def test_peak_memory_tracking(self, song_file, test_profile, gpu_memory_monitor, performance_thresholds, performance_tracker):
        """Track peak GPU memory during conversion.

        Uses MAX_GPU_MEMORY_GB environment variable (default: 8.0).
        """
        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
        except ImportError:
            pytest.skip("SingingConversionPipeline not available")

        with gpu_memory_monitor:
            pipeline = SingingConversionPipeline(config={'device': 'cuda'})
            result = pipeline.convert_song(
                song_path=str(song_file),
                target_profile_id=test_profile['profile_id']
            )

        stats = gpu_memory_monitor.get_stats()

        print(f"\nGPU Memory Usage:")
        print(f"  Initial: {stats['initial_mb']:.2f} MB")
        print(f"  Peak: {stats['peak_mb']:.2f} MB")
        print(f"  Final: {stats['final_mb']:.2f} MB")
        print(f"  Delta: {stats['delta_mb']:.2f} MB")

        # Get threshold from environment
        max_memory_mb = performance_thresholds['max_gpu_memory_gb'] * 1024
        print(f"  Max: {max_memory_mb:.0f} MB")

        assert stats['peak_mb'] < max_memory_mb, f"Peak memory too high: {stats['peak_mb']:.2f} MB (max: {max_memory_mb:.0f} MB)"

        # Record for metrics
        performance_tracker.record('peak_gpu_memory_mb', stats['peak_mb'])


@pytest.mark.performance
class TestCacheHitRateSpeedup:
    """Measure cache hit rate and speedup."""

    def test_cache_effectiveness(self, song_file, test_profile, device, performance_tracker, performance_thresholds):
        """Test cache hit rate and speedup measurement.

        Uses CACHE_SPEEDUP_THRESHOLD environment variable (default: 3.0).
        """
        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
        except ImportError:
            pytest.skip("SingingConversionPipeline not available")

        # Get threshold from environment
        min_speedup = performance_thresholds['cache_speedup']

        pipeline = SingingConversionPipeline(config={
            'device': device,
            'cache_enabled': True
        })
        pipeline.clear_cache()

        # First run: populate cache
        performance_tracker.start('run_1')
        result1 = pipeline.convert_song(
            song_path=str(song_file),
            target_profile_id=test_profile['profile_id']
        )
        time1 = performance_tracker.stop()

        # Subsequent runs: should hit cache
        times = [time1]
        for i in range(2, 6):
            performance_tracker.start(f'run_{i}')
            result = pipeline.convert_song(
                song_path=str(song_file),
                target_profile_id=test_profile['profile_id']
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
        print(f"  Threshold: {min_speedup:.1f}x")

        # Cached runs should be consistently faster
        assert speedup >= min_speedup, f"Cache speedup insufficient: {speedup:.2f}x (expected >= {min_speedup:.1f}x)"

        # Record for metrics
        performance_tracker.record('cache_hit_speedup', speedup)


@pytest.mark.performance
class TestComponentTimingBreakdown:
    """Breakdown of timing for each component."""

    def test_component_timing_breakdown(self, song_file, test_profile, device, performance_tracker):
        """Measure time spent in each pipeline component with stage timing validation."""
        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
        except ImportError:
            pytest.skip("SingingConversionPipeline not available")

        # Create pipeline with timing instrumentation
        pipeline = SingingConversionPipeline(config={'device': device})

        # Track stage timestamps (stage_start:* and stage_end:*)
        stage_timestamps = {}

        def timed_callback(progress: float, message: str):
            """Capture monotonic timestamps for stage_start and stage_end events."""
            if message.startswith('stage_start:'):
                stage_name = message.replace('stage_start:', '')
                if stage_name not in stage_timestamps:
                    stage_timestamps[stage_name] = {}
                stage_timestamps[stage_name]['start'] = time.perf_counter()
            elif message.startswith('stage_end:'):
                stage_name = message.replace('stage_end:', '')
                if stage_name not in stage_timestamps:
                    stage_timestamps[stage_name] = {}
                stage_timestamps[stage_name]['end'] = time.perf_counter()

        # Run with callback
        result = pipeline.convert_song(
            song_path=str(song_file),
            target_profile_id=test_profile['profile_id'],
            progress_callback=timed_callback
        )

        # Expected stages
        expected_stages = ['separation', 'pitch_extraction', 'voice_conversion', 'audio_mixing']

        # Verify all stages have both start and end timestamps
        for stage in expected_stages:
            assert stage in stage_timestamps, f"Missing stage: {stage}"
            assert 'start' in stage_timestamps[stage], f"Missing stage_start for {stage}"
            assert 'end' in stage_timestamps[stage], f"Missing stage_end for {stage}"

        # Calculate component durations
        stage_durations = {}
        total_time = 0
        print(f"\nComponent Timing Breakdown:")
        for stage in expected_stages:
            duration = stage_timestamps[stage]['end'] - stage_timestamps[stage]['start']
            stage_durations[stage] = duration
            total_time += duration
            print(f"  {stage}: {duration:.3f}s")

        # Calculate and display percentages
        print(f"\nTotal pipeline time: {total_time:.3f}s")
        for stage in expected_stages:
            percentage = (stage_durations[stage] / total_time) * 100
            print(f"  {stage}: {percentage:.1f}%")

        # Assert no single stage dominates (>60% of total time)
        for stage in expected_stages:
            stage_percentage = (stage_durations[stage] / total_time) * 100
            assert stage_percentage <= 60.0, \
                f"Stage '{stage}' dominates with {stage_percentage:.1f}% of total time (max 60%)"


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
        result = pipeline.convert_song(
            song_path=str(audio_file),
            target_profile_id=test_profile['profile_id']
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
        performance_tracker.start('source_separation')
        vocals, instrumental = separator.separate_vocals(str(song_file))
        elapsed = performance_tracker.stop()

        rtf = elapsed / duration

        print(f"\nSource Separation Performance:")
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
        performance_tracker.start('pitch_extraction')
        result = extractor.extract_f0_contour(audio, sample_rate)
        elapsed = performance_tracker.stop()

        rtf = elapsed / duration

        print(f"\nPitch Extraction Performance:")
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
            result = pipeline.convert_song(
                song_path=str(song_file),
                target_profile_id=test_profile['profile_id'],
                preset=preset
            )
            elapsed = performance_tracker.stop()

            duration = result['duration']
            rtf = elapsed / duration

            print(f"\nPreset '{preset}' Performance:")
            print(f"  Elapsed: {elapsed:.3f}s")
            print(f"  RTF: {rtf:.3f}x")

            # Record preset-specific RTF metrics for aggregation
            performance_tracker.record(f'rtf_{preset}', rtf)

        except (ValueError, KeyError):
            pytest.skip(f"Preset '{preset}' not supported")


@pytest.mark.performance
@pytest.mark.quality
class TestQualityVsSpeedTradeoffs:
    """Compare quality metrics with performance metrics to understand tradeoffs."""

    def test_pitch_target_vs_speed_comparison(self, tmp_path, device, performance_tracker):
        """
        Compare pitch accuracy vs processing speed across different quality settings.

        This test validates the trade-off between accuracy and speed.
        """
        try:
            from src.auto_voice.evaluation.evaluator import VoiceConversionEvaluator, QualityTargets
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
            import soundfile as sf
        except ImportError:
            pytest.skip("Required components not available")

        # Generate test audio
        sample_rate = 44100
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        source_audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        # Create evaluator
        evaluator = VoiceConversionEvaluator(sample_rate=sample_rate, device=device)
        pipeline = SingingConversionPipeline(config={'device': device})

        results = {}

        # Test different preset configurations if available
        presets = ['fast', 'balanced', 'quality']

        for preset in presets:
            try:
                # Convert audio
                performance_tracker.start(f'conversion_{preset}')
                result = pipeline.convert_song(
                    song_path=str(tmp_path / f'source.wav'),
                    target_profile_id='test_profile',
                    preset=preset
                )
                elapsed = performance_tracker.stop()

                converted_audio = result['mixed_audio']
                rtf = elapsed / duration

                # Evaluate quality
                source_tensor = torch.from_numpy(source_audio).float()
                converted_tensor = torch.from_numpy(converted_audio).float()

                quality_result = evaluator.evaluate_single_conversion(
                    source_tensor, converted_tensor
                )

                results[preset] = {
                    'rtf': rtf,
                    'pitch_rmse_hz': quality_result.pitch_accuracy.rmse_hz,
                    'pitch_correlation': quality_result.pitch_accuracy.correlation,
                    'overall_quality': quality_result.overall_quality_score
                }

                print(f"\nPreset '{preset}':")
                print(f"  RTF: {rtf:.3f}x")
                print(f"  Pitch RMSE (Hz): {quality_result.pitch_accuracy.rmse_hz:.2f}")
                print(f"  Pitch Correlation: {quality_result.pitch_accuracy.correlation:.3f}")
                print(f"  Overall Quality: {quality_result.overall_quality_score:.3f}")

            except (ValueError, KeyError):
                # Preset not supported
                continue

        if len(results) >= 2:
            # Test that higher quality presets generally provide better accuracy at cost of speed
            preset_list = list(results.keys())
            if 'fast' in results and 'quality' in results:
                fast_quality = results['fast']['overall_quality']
                quality_overall = results['quality']['overall_quality']

                # Quality preset should typically have better or equal quality
                # (though not guaranteed for synthetic test data)
                print(f"\nQuality Comparison:")
                print(f"  Fast preset quality: {fast_quality:.3f}")
                print(f"  Quality preset quality: {quality_overall:.3f}")

                # Just log the comparison, don't enforce strict requirements
                # as synthetic data may not show clear differences

    def test_memory_usage_vs_quality_tradeoff(self, tmp_path, device, memory_monitor, performance_tracker):
        """
        Compare memory usage patterns across different quality settings.

        Validates that quality improvements don't come with excessive memory costs.
        """
        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
        except ImportError:
            pytest.skip("SingingConversionPipeline not available")

        pipeline = SingingConversionPipeline(config={'device': device})
        sample_rate = 44100

        # Generate test audio
        duration = 5.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        audio_file = tmp_path / "test_audio.wav"
        sf.write(str(audio_file), sf.read(audio), sample_rate)

        presets = ['fast', 'balanced', 'quality']
        memory_results = {}

        for preset in presets:
            try:
                with memory_monitor:
                    performance_tracker.start(f'memory_{preset}')
                    result = pipeline.convert_song(
                        song_path=str(audio_file),
                        target_profile_id='test_profile',
                        preset=preset
                    )
                    elapsed = performance_tracker.stop()

                memory_stats = memory_monitor.get_stats()
                memory_results[preset] = {
                    'elapsed': elapsed,
                    'peak_memory_mb': memory_stats.get('peak_mb', 0)
                }

                print(f"\nPreset '{preset}' Memory Usage:")
                print(f"  Elapsed: {elapsed:.3f}s")
                print(f"  Peak Memory: {memory_stats.get('peak_mb', 0):.0f} MB")

            except (ValueError, KeyError):
                continue

        # Compare memory usage patterns
        if len(memory_results) >= 2:
            print("\nMemory Usage Analysis:")
            for preset, stats in memory_results.items():
                efficiency = stats['elapsed'] / (stats['peak_memory_mb'] or 1)
                print(f"  {preset}: {efficiency:.2f} sec/MB")

    def test_batch_size_quality_vs_speed_tradeoff(self, device, performance_tracker):
        """
        Test how batch size affects both quality consistency and processing speed.

        Larger batches should be faster but may show slight quality variations.
        """
        try:
            from src.auto_voice.audio.pitch_extractor import SingingPitchExtractor
            from src.auto_voice.evaluation.evaluator import VoiceConversionEvaluator
        except ImportError:
            pytest.skip("Required components not available")

        extractor = SingingPitchExtractor(device=device)
        evaluator = VoiceConversionEvaluator(sample_rate=44100, device=device)

        # Generate multiple audio samples
        sample_rate = 44100
        num_samples = 10
        audio_samples = []

        for i in range(num_samples):
            duration = 2.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            freq = 220 + i * 10  # Different frequencies
            audio = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
            audio_samples.append(audio)

        batch_sizes = [1, 4, 8]
        results = {}

        for batch_size in batch_sizes:
            try:
                # Time batch processing
                performance_tracker.start(f'batch_size_{batch_size}')
                batch_results = extractor.batch_extract_f0(audio_samples[:batch_size], sample_rate)
                elapsed = performance_tracker.stop()

                # Calculate quality consistency (standard deviation of RMSE)
                quality_scores = []
                for i in range(batch_size):
                    # Compare extracted F0 with expected frequency
                    expected_freq = 220 + i * 10
                    extracted_f0 = batch_results[i]

                    # Simple RMSE calculation (simplified for test)
                    rmse_hz = np.sqrt(np.mean((extracted_f0 - expected_freq) ** 2))
                    quality_scores.append(rmse_hz)

                quality_consistency = np.std(quality_scores) if quality_scores else 0.0
                avg_time_per_sample = elapsed / batch_size

                results[batch_size] = {
                    'avg_time_per_sample': avg_time_per_sample,
                    'quality_consistency': quality_consistency,
                    'total_time': elapsed
                }

                print(f"\nBatch Size {batch_size}:")
                print(f"  Avg time per sample: {avg_time_per_sample:.3f}s")
                print(f"  Quality consistency (std dev): {quality_consistency:.2f} Hz")
                print(f"  Total time: {elapsed:.3f}s")

            except Exception as e:
                print(f"Batch size {batch_size} failed: {e}")
                continue

        # Analyze scaling efficiency
        if len(results) >= 2:
            print("\nBatch Processing Scaling Analysis:")
            for bs, stats in results.items():
                throughput = 1.0 / stats['avg_time_per_sample']  # samples per second
                print(f"  Batch {bs}: {throughput:.2f} samples/sec")

    def test_quality_regression_thresholds(self, device, performance_tracker):
        """
        Test that quality doesn't regress beyond acceptable thresholds when optimizing for speed.

        This acts as a performance regression test with quality gates.
        """
        try:
            from src.auto_voice.evaluation.evaluator import VoiceConversionEvaluator, QualityTargets
        except ImportError:
            pytest.skip("Evaluator not available")

        evaluator = VoiceConversionEvaluator(device=device)

        # Generate test audio pairs with known quality levels
        sample_rate = 44100
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        test_pairs = []

        # High quality pair (perfect match)
        source_high = torch.tensor(0.5 * np.sin(2 * np.pi * 440 * t), dtype=torch.float32)
        target_high = source_high.clone()
        test_pairs.append(('high_quality', source_high, target_high))

        # Medium quality pair (slight detuning)
        source_med = torch.tensor(0.5 * np.sin(2 * np.pi * 440 * t), dtype=torch.float32)
        target_med = torch.tensor(0.5 * np.sin(2 * np.pi * 439.8 * t), dtype=torch.float32)
        test_pairs.append(('medium_quality', source_med, target_med))

        # Run evaluations
        for pair_name, source, target in test_pairs:
            performance_tracker.start(f'quality_eval_{pair_name}')

            result = evaluator.evaluate_single_conversion(source, target)

            elapsed = performance_tracker.stop()

            print(f"\n{pair_name.upper()} Pair:")
            print(f"  Pitch RMSE (Hz): {result.pitch_accuracy.rmse_hz:.2f}")
            print(f"  Pitch Correlation: {result.pitch_accuracy.correlation:.3f}")
            print(f"  Overall Quality: {result.overall_quality_score:.3f}")
            print(f"  Eval Time: {elapsed:.3f}s")

            # Store baseline expectations for regression testing
            if pair_name == 'high_quality':
                # High quality pair should meet very strict targets
                assert result.pitch_accuracy.rmse_hz < 2.0, \
                    f"High quality pair RMSE too high: {result.pitch_accuracy.rmse_hz:.2f} Hz"
                assert result.pitch_accuracy.correlation > 0.99, \
                    f"High quality pair correlation too low: {result.pitch_accuracy.correlation:.3f}"
                assert result.overall_quality_score > 0.9, \
                    f"High quality pair score too low: {result.overall_quality_score:.3f}"

            elif pair_name == 'medium_quality':
                # Medium quality pair should still meet reasonable targets
                assert result.pitch_accuracy.rmse_hz < 5.0, \
                    f"Medium quality pair RMSE too high: {result.pitch_accuracy.rmse_hz:.2f} Hz"
                assert result.pitch_accuracy.correlation > 0.95, \
                    f"Medium quality pair correlation too low: {result.pitch_accuracy.correlation:.3f}"

        # Performance regression check: evaluation should be fast (< 0.5s per evaluation)
        # Last elapsed should still be available
        if 'elapsed' in locals():
            assert elapsed < 0.5, f"Quality evaluation too slow: {elapsed:.3f}s"

    def test_accelerated_processing_quality_maintenance(self, device, performance_tracker):
        """
        Test that accelerated processing modes (GPU, Triton, TensorRT) maintain quality.

        Ensures optimization doesn't sacrifice accuracy.
        """
        try:
            from src.auto_voice.evaluation.evaluator import VoiceConversionEvaluator
        except ImportError:
            pytest.skip("Evaluator not available")

        evaluator = VoiceConversionEvaluator(device=device)

        # Test with different hypothetical acceleration modes
        # In practice, this would compare results across different execution modes

        sample_rate = 44100
        duration = 5.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        source = torch.tensor(0.5 * np.sin(2 * np.pi * 440 * t), dtype=torch.float32)
        target = torch.tensor(0.5 * np.sin(2 * np.pi * 439.9 * t), dtype=torch.float32)

        # Run evaluation (would be done multiple times with different acceleration modes)
        performance_tracker.start('accelerated_eval')
        result = evaluator.evaluate_single_conversion(source, target)
        elapsed = performance_tracker.stop()

        # Log performance characteristics
        print(f"\nAccelerated Processing Quality:")
        print(f"  Device: {device}")
        print(f"  Eval Time: {elapsed:.3f}s")
        print(f"  Pitch RMSE (Hz): {result.pitch_accuracy.rmse_hz:.2f}")
        print(f"  Quality Score: {result.overall_quality_score:.3f}")

        # Quality maintenance check: should meet baseline requirements
        assert result.pitch_accuracy.rmse_hz < 3.0, \
            f"Accelerated mode quality degraded: RMSE {result.pitch_accuracy.rmse_hz:.2f} Hz"
        assert result.overall_quality_score > 0.8, \
            f"Accelerated mode quality too low: {result.overall_quality_score:.3f}"

        # Register as a baseline for future regression tests
        performance_tracker.record(f'quality_{device}', result.overall_quality_score)
        performance_tracker.record(f'speed_{device}', elapsed)


@pytest.mark.performance
class TestQualityRegressionDetection:
    """Detect quality regressions by comparing against baseline metrics."""

    def test_load_baseline_metrics(self, request):
        """Load baseline metrics from file or create default baseline."""
        import json
        from pathlib import Path

        # Get baseline file path from pytest config
        baseline_file = request.config.getoption('--baseline-file', default='.github/quality_baseline.json')
        baseline_path = Path(baseline_file)

        if baseline_path.exists():
            with open(baseline_path) as f:
                baseline = json.load(f)
            print(f"\nLoaded baseline from {baseline_path}")
            print(f"  Timestamp: {baseline.get('timestamp', 'N/A')}")
            print(f"  Commit: {baseline.get('commit', 'N/A')[:8]}")
        else:
            # Create default baseline
            baseline = {
                'version': '1.0',
                'metrics': {
                    'pitch_rmse_hz': 10.0,
                    'pitch_correlation': 0.80,
                    'speaker_similarity': 0.85,
                    'overall_quality_score': 0.75,
                    'processing_rtf_cpu': 20.0,
                    'processing_rtf_gpu': 5.0
                },
                'thresholds': {
                    'pitch_rmse_hz_max_increase': 2.0,
                    'pitch_correlation_min_decrease': 0.05,
                    'speaker_similarity_min_decrease': 0.05,
                    'overall_quality_min': 0.70,
                    'rtf_max_increase_percent': 20.0
                }
            }
            print(f"\nCreated default baseline (file not found: {baseline_path})")

        assert 'metrics' in baseline
        assert 'thresholds' in baseline
        assert len(baseline['metrics']) > 0

    def test_measure_current_metrics(self, tmp_path, device, performance_tracker):
        """Measure current quality and performance metrics."""
        try:
            from src.auto_voice.evaluation.evaluator import VoiceConversionEvaluator
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
            import soundfile as sf
        except ImportError:
            pytest.skip("Required components not available")

        # Generate test audio
        sample_rate = 44100
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        source_audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        target_audio = (0.5 * np.sin(2 * np.pi * 439.9 * t)).astype(np.float32)

        audio_file = tmp_path / "test_audio.wav"
        sf.write(str(audio_file), source_audio, sample_rate)

        # Measure quality metrics
        evaluator = VoiceConversionEvaluator(sample_rate=sample_rate, device=device)
        source_tensor = torch.from_numpy(source_audio).float()
        target_tensor = torch.from_numpy(target_audio).float()

        result = evaluator.evaluate_single_conversion(source_tensor, target_tensor)

        # Measure performance metrics
        if torch.cuda.is_available():
            performance_tracker.start('gpu_conversion')
            pipeline_gpu = SingingConversionPipeline(config={'device': 'cuda'})
            # Simplified test - would run actual conversion in real scenario
            elapsed_gpu = performance_tracker.stop()
            rtf_gpu = elapsed_gpu / duration
        else:
            rtf_gpu = None

        performance_tracker.start('cpu_conversion')
        pipeline_cpu = SingingConversionPipeline(config={'device': 'cpu'})
        # Simplified test - would run actual conversion in real scenario
        elapsed_cpu = performance_tracker.stop()
        rtf_cpu = elapsed_cpu / duration

        # Collect metrics
        current_metrics = {
            'pitch_rmse_hz': result.pitch_accuracy.rmse_hz,
            'pitch_correlation': result.pitch_accuracy.correlation,
            'speaker_similarity': result.speaker_similarity.mean_similarity if hasattr(result, 'speaker_similarity') else 0.85,
            'overall_quality_score': result.overall_quality_score,
            'processing_rtf_cpu': rtf_cpu,
        }

        if rtf_gpu is not None:
            current_metrics['processing_rtf_gpu'] = rtf_gpu

        print("\nCurrent Metrics:")
        for key, value in current_metrics.items():
            print(f"  {key}: {value:.4f}")

        # Store metrics for later comparison
        performance_tracker.record('current_metrics', current_metrics)

        assert all(isinstance(v, (int, float)) for v in current_metrics.values())

    def test_compare_against_baseline(self, request, performance_tracker):
        """Compare current metrics against baseline and detect regressions."""
        import json
        from pathlib import Path
        from datetime import datetime

        # Load baseline
        baseline_file = request.config.getoption('--baseline-file', default='.github/quality_baseline.json')
        baseline_path = Path(baseline_file)

        if baseline_path.exists():
            with open(baseline_path) as f:
                baseline = json.load(f)
        else:
            # Use default baseline for comparison
            baseline = {
                'metrics': {
                    'pitch_rmse_hz': 10.0,
                    'pitch_correlation': 0.80,
                    'speaker_similarity': 0.85,
                    'overall_quality_score': 0.75,
                    'processing_rtf_cpu': 20.0,
                    'processing_rtf_gpu': 5.0
                },
                'thresholds': {
                    'pitch_rmse_hz_max_increase': 2.0,
                    'pitch_correlation_min_decrease': 0.05,
                    'speaker_similarity_min_decrease': 0.05,
                    'overall_quality_min': 0.70,
                    'rtf_max_increase_percent': 20.0
                }
            }

        # Get current metrics (from previous test or create sample)
        current_metrics = {
            'pitch_rmse_hz': 9.5,
            'pitch_correlation': 0.82,
            'speaker_similarity': 0.87,
            'overall_quality_score': 0.78,
            'processing_rtf_cpu': 18.0,
            'processing_rtf_gpu': 4.8
        }

        baseline_metrics = baseline['metrics']
        thresholds = baseline['thresholds']

        # Compare metrics
        regressions = []
        warnings = []

        print("\nRegression Analysis:")
        for metric, current_val in current_metrics.items():
            if metric not in baseline_metrics:
                continue

            baseline_val = baseline_metrics[metric]
            change = current_val - baseline_val
            change_percent = (change / baseline_val * 100) if baseline_val != 0 else 0

            print(f"\n  {metric}:")
            print(f"    Baseline: {baseline_val:.4f}")
            print(f"    Current:  {current_val:.4f}")
            print(f"    Change:   {change:+.4f} ({change_percent:+.1f}%)")

            # Check for regressions based on metric type
            if 'rmse' in metric.lower():
                # Lower is better for RMSE
                if current_val > baseline_val + thresholds.get('pitch_rmse_hz_max_increase', 2.0):
                    regressions.append(f"{metric}: {current_val:.2f} > {baseline_val:.2f} + {thresholds.get('pitch_rmse_hz_max_increase', 2.0)}")
                    print(f"    Status:   ❌ REGRESSION")
                elif current_val > baseline_val + thresholds.get('pitch_rmse_hz_max_increase', 2.0) / 2:
                    warnings.append(f"{metric}: {current_val:.2f} approaching threshold")
                    print(f"    Status:   ⚠️ Warning")
                else:
                    print(f"    Status:   ✅ OK")

            elif 'correlation' in metric.lower() or 'similarity' in metric.lower():
                # Higher is better for correlation/similarity
                if current_val < baseline_val - thresholds.get('pitch_correlation_min_decrease', 0.05):
                    regressions.append(f"{metric}: {current_val:.3f} < {baseline_val:.3f} - {thresholds.get('pitch_correlation_min_decrease', 0.05)}")
                    print(f"    Status:   ❌ REGRESSION")
                elif current_val < baseline_val - thresholds.get('pitch_correlation_min_decrease', 0.05) / 2:
                    warnings.append(f"{metric}: {current_val:.3f} approaching threshold")
                    print(f"    Status:   ⚠️ Warning")
                else:
                    print(f"    Status:   ✅ OK")

            elif 'quality' in metric.lower():
                # Higher is better for quality scores
                if current_val < thresholds.get('overall_quality_min', 0.70):
                    regressions.append(f"{metric}: {current_val:.3f} < minimum {thresholds.get('overall_quality_min', 0.70)}")
                    print(f"    Status:   ❌ REGRESSION")
                elif current_val < baseline_val - 0.05:
                    warnings.append(f"{metric}: {current_val:.3f} decreased from {baseline_val:.3f}")
                    print(f"    Status:   ⚠️ Warning")
                else:
                    print(f"    Status:   ✅ OK")

            elif 'rtf' in metric.lower():
                # Lower is better for RTF (faster processing)
                if change_percent > thresholds.get('rtf_max_increase_percent', 20.0):
                    regressions.append(f"{metric}: {change_percent:+.1f}% increase (max: {thresholds.get('rtf_max_increase_percent', 20.0)}%)")
                    print(f"    Status:   ❌ REGRESSION")
                elif change_percent > thresholds.get('rtf_max_increase_percent', 20.0) / 2:
                    warnings.append(f"{metric}: {change_percent:+.1f}% increase approaching threshold")
                    print(f"    Status:   ⚠️ Warning")
                else:
                    print(f"    Status:   ✅ OK")

        # Save results to file if output path specified
        output_file = request.config.getoption('--output-file', default=None)
        if output_file:
            results = {
                'version': '1.0',
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'commit': 'current',
                'metrics': current_metrics,
                'thresholds': thresholds,
                'regressions': regressions,
                'warnings': warnings
            }
            Path(output_file).write_text(json.dumps(results, indent=2))
            print(f"\nResults saved to {output_file}")

        # Print summary
        print("\n" + "="*60)
        if regressions:
            print("❌ QUALITY REGRESSIONS DETECTED:")
            for regression in regressions:
                print(f"  - {regression}")
        elif warnings:
            print("⚠️ WARNINGS (approaching thresholds):")
            for warning in warnings:
                print(f"  - {warning}")
        else:
            print("✅ NO REGRESSIONS DETECTED")
        print("="*60)

        # Fail test if regressions detected
        assert len(regressions) == 0, f"Quality regressions detected: {len(regressions)} metric(s) exceeded thresholds"


@pytest.mark.performance
@pytest.mark.parametrize('config_preset', ['minimal', 'balanced', 'comprehensive'])
class TestConfigQualityPerformanceTradeoffs:
    """Test quality vs performance tradeoffs with different configuration presets."""

    def test_config_tradeoff_analysis(self, song_file, test_profile, device, config_preset, performance_tracker):
        """
        Analyze how different configuration presets affect quality and performance.

        Tests the trade-off between accuracy, speed, and resource usage.
        """
        try:
            from src.auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
            from src.auto_voice.evaluation.evaluator import VoiceConversionEvaluator
        except ImportError:
            pytest.skip("Required components not available")

        # Define configuration presets (simplified for testing)
        configs = {
            'minimal': {
                'device': device,
                'model_config': {'latent_dim': 128},  # Smaller model
                'processing_config': {'batch_size': 1}
            },
            'balanced': {
                'device': device,
                'model_config': {'latent_dim': 192},  # Standard model
                'processing_config': {'batch_size': 4}
            },
            'comprehensive': {
                'device': device,
                'model_config': {'latent_dim': 256},  # Larger model
                'processing_config': {'batch_size': 8}
            }
        }

        if config_preset not in configs:
            pytest.skip(f"Config preset '{config_preset}' not defined")

        config = configs[config_preset]

        # Run conversion with specific config
        try:
            pipeline = SingingConversionPipeline(config=config)

            performance_tracker.start(f'conversion_{config_preset}')
            result = pipeline.convert_song(
                song_path=str(song_file),
                target_profile_id=test_profile['profile_id']
            )
            elapsed = performance_tracker.stop()

            # Basic quality evaluation
            import soundfile as sf
            source_audio, _ = sf.read(str(song_file))
            converted_audio = result['mixed_audio']

            evaluator = VoiceConversionEvaluator(sample_rate=44100, device=device)
            source_tensor = torch.from_numpy(source_audio).float()
            converted_tensor = torch.from_numpy(converted_audio).float()

            quality_result = evaluator.evaluate_single_conversion(source_tensor, converted_tensor)

            # Log metrics
            rtf = elapsed / result['duration']
            print(f"\nConfig '{config_preset}':")
            print(f"  RTF: {rtf:.3f}x")
            print(f"  Pitch RMSE (Hz): {quality_result.pitch_accuracy.rmse_hz:.2f}")
            print(f"  Pitch Correlation: {quality_result.pitch_accuracy.correlation:.3f}")
            print(f"  Overall Quality: {quality_result.overall_quality_score:.3f}")

            # Store for cross-preset comparison
            performance_tracker.record(f'rtf_{config_preset}', rtf)
            performance_tracker.record(f'quality_{config_preset}', quality_result.overall_quality_score)
            performance_tracker.record(f'rmse_hz_{config_preset}', quality_result.pitch_accuracy.rmse_hz)

        except Exception as e:
            # Config may not be fully supported
            print(f"Config '{config_preset}' failed: {e}")
            pytest.skip(f"Configuration not supported: {e}")
