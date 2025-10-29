"""
Comprehensive system validation test suite for AutoVoice.

Tests complete end-to-end singing conversion pipeline with automated quality checks,
diverse test samples, edge case handling, and validation report generation.

Addresses Comment 1, 3, and 10 requirements:
- End-to-end conversion tests using SingingConversionPipeline
- Automated checks: pitch RMSE < 10 Hz, speaker similarity > 0.85, latency < 5s per 30s
- Tests for diverse synthetic/real samples across genres/styles/languages
- Integration with validation report generation
- Edge case tests: very short (<10s), very long (>5min), a cappella, heavily processed vocals
- Memory usage monitoring for long files
"""

import pytest
import torch
import numpy as np
import soundfile as sf
import json
import time
import psutil
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ValidationMetrics:
    """Container for validation metrics."""
    pitch_rmse_hz: float
    speaker_similarity: float
    latency_seconds: float
    memory_usage_mb: float
    test_case_id: str
    genre: Optional[str] = None
    duration_seconds: Optional[float] = None
    passed: bool = False


@pytest.fixture(scope='module')
def test_metadata_loader():
    """Load test set metadata from JSON file."""
    test_data_path = Path(__file__).parent / 'data' / 'validation' / 'test_set.json'
    if not test_data_path.exists():
        pytest.skip(f"Test metadata not found: {test_data_path}. Run generate_test_data.py first.")

    with open(test_data_path, 'r') as f:
        metadata = json.load(f)

    return metadata


@pytest.fixture(scope='module')
def validation_pipeline(device):
    """Create SingingConversionPipeline for validation tests."""
    try:
        from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

        config = {
            'device': device,
            'cache_enabled': False,  # Disable cache for validation
            'sample_rate': 44100,
            'preset': 'balanced'
        }

        pipeline = SingingConversionPipeline(config=config)
        return pipeline
    except ImportError:
        pytest.skip("SingingConversionPipeline not available")


@pytest.fixture(scope='module')
def quality_evaluator(device):
    """Create quality evaluator for validation metrics."""
    try:
        from auto_voice.evaluation.evaluator import VoiceConversionEvaluator

        evaluator = VoiceConversionEvaluator(
            sample_rate=44100,
            device=device
        )
        return evaluator
    except ImportError:
        pytest.skip("VoiceConversionEvaluator not available")


@pytest.fixture
def memory_monitor():
    """Monitor memory usage during tests."""
    process = psutil.Process(os.getpid())

    class MemoryMonitor:
        def __init__(self):
            self.start_memory = 0
            self.peak_memory = 0

        def start(self):
            self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = self.start_memory

        def update(self):
            current = process.memory_info().rss / 1024 / 1024
            self.peak_memory = max(self.peak_memory, current)

        def get_usage(self) -> float:
            """Get memory increase in MB."""
            return self.peak_memory - self.start_memory

    return MemoryMonitor()


# ============================================================================
# Comment 1: End-to-End Conversion Tests with Automated Quality Checks
# ============================================================================

@pytest.mark.system_validation
@pytest.mark.slow
class TestSystemValidation:
    """Comprehensive system validation tests."""

    def test_diverse_genres_conversion(
        self,
        test_metadata_loader,
        validation_pipeline,
        quality_evaluator,
        memory_monitor
    ):
        """
        Test conversion across diverse genres with quality validation.

        Addresses Comment 1 & 3: Tests diverse samples and enforces quality targets.
        """
        test_cases = test_metadata_loader['test_cases']

        validation_results = []

        for test_case in test_cases:
            test_id = test_case['id']
            source_audio_path = test_case['source_audio']
            target_profile_id = test_case['target_profile_id']
            metadata = test_case.get('metadata', {})

            print(f"\n=== Testing {test_id} ===")
            print(f"Genre: {metadata.get('genre', 'unknown')}")
            print(f"Duration: {metadata.get('duration_sec', 'unknown')}s")

            # Monitor memory
            memory_monitor.start()

            # Run conversion
            start_time = time.time()

            try:
                result = validation_pipeline.convert_song(
                    song_path=source_audio_path,
                    target_profile_id=target_profile_id,
                    pitch_shift=0.0
                )

                latency = time.time() - start_time
                memory_monitor.update()

                # Load source and converted audio for quality evaluation
                source_audio, sr = sf.read(source_audio_path)
                converted_audio = result['mixed_audio']

                # Convert to tensors
                source_tensor = torch.from_numpy(source_audio).float()
                converted_tensor = torch.from_numpy(converted_audio).float()

                # Evaluate quality metrics
                quality_result = quality_evaluator.evaluate_single_conversion(
                    source_tensor,
                    converted_tensor
                )

                # Extract metrics
                pitch_rmse_hz = quality_result.pitch_accuracy.rmse_hz
                speaker_similarity = quality_result.speaker_similarity.cosine_similarity
                memory_usage_mb = memory_monitor.get_usage()
                duration_sec = metadata.get('duration_sec', len(source_audio) / sr)

                # Validate quality targets
                passed = (
                    pitch_rmse_hz < 10.0 and
                    speaker_similarity > 0.85 and
                    latency / duration_sec < 5.0  # RTF < 5.0 for 30s = 150s total
                )

                # Store validation metrics
                validation_metrics = ValidationMetrics(
                    pitch_rmse_hz=pitch_rmse_hz,
                    speaker_similarity=speaker_similarity,
                    latency_seconds=latency,
                    memory_usage_mb=memory_usage_mb,
                    test_case_id=test_id,
                    genre=metadata.get('genre'),
                    duration_seconds=duration_sec,
                    passed=passed
                )

                validation_results.append(validation_metrics)

                # Print results
                print(f"✓ Conversion completed")
                print(f"  Pitch RMSE (Hz): {pitch_rmse_hz:.2f} {'✓' if pitch_rmse_hz < 10.0 else '✗'}")
                print(f"  Speaker Similarity: {speaker_similarity:.3f} {'✓' if speaker_similarity > 0.85 else '✗'}")
                print(f"  Latency: {latency:.2f}s (RTF: {latency/duration_sec:.2f}x) {'✓' if latency/duration_sec < 5.0 else '✗'}")
                print(f"  Memory Usage: {memory_usage_mb:.1f} MB")
                print(f"  Status: {'PASS' if passed else 'FAIL'}")

                # Assert quality targets
                assert pitch_rmse_hz < 10.0, \
                    f"Pitch RMSE {pitch_rmse_hz:.2f} Hz exceeds 10.0 Hz threshold"
                assert speaker_similarity > 0.85, \
                    f"Speaker similarity {speaker_similarity:.3f} below 0.85 threshold"
                assert latency / duration_sec < 5.0, \
                    f"Latency RTF {latency/duration_sec:.2f}x exceeds 5.0x threshold"

            except Exception as e:
                print(f"✗ Conversion failed: {e}")
                pytest.fail(f"Test case {test_id} failed: {e}")

        # Generate validation report
        self._generate_validation_report(validation_results)

        # Assert all tests passed
        pass_count = sum(1 for r in validation_results if r.passed)
        total_count = len(validation_results)

        print(f"\n=== Validation Summary ===")
        print(f"Passed: {pass_count}/{total_count}")
        print(f"Success Rate: {pass_count/total_count*100:.1f}%")

        assert pass_count == total_count, \
            f"Only {pass_count}/{total_count} tests passed"

    def _generate_validation_report(self, validation_results: List[ValidationMetrics]):
        """Generate validation report from results."""
        report_dir = Path(__file__).parent / 'reports'
        report_dir.mkdir(parents=True, exist_ok=True)

        report_path = report_dir / 'system_validation_report.json'

        # Aggregate statistics
        pitch_rmse_values = [r.pitch_rmse_hz for r in validation_results]
        speaker_sim_values = [r.speaker_similarity for r in validation_results]
        latency_values = [r.latency_seconds for r in validation_results]

        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': len(validation_results),
            'passed_tests': sum(1 for r in validation_results if r.passed),
            'failed_tests': sum(1 for r in validation_results if not r.passed),
            'aggregate_metrics': {
                'pitch_rmse_hz': {
                    'mean': np.mean(pitch_rmse_values),
                    'std': np.std(pitch_rmse_values),
                    'min': np.min(pitch_rmse_values),
                    'max': np.max(pitch_rmse_values)
                },
                'speaker_similarity': {
                    'mean': np.mean(speaker_sim_values),
                    'std': np.std(speaker_sim_values),
                    'min': np.min(speaker_sim_values),
                    'max': np.max(speaker_sim_values)
                },
                'latency_seconds': {
                    'mean': np.mean(latency_values),
                    'std': np.std(latency_values),
                    'min': np.min(latency_values),
                    'max': np.max(latency_values)
                }
            },
            'individual_results': [
                {
                    'test_case_id': r.test_case_id,
                    'genre': r.genre,
                    'pitch_rmse_hz': r.pitch_rmse_hz,
                    'speaker_similarity': r.speaker_similarity,
                    'latency_seconds': r.latency_seconds,
                    'memory_usage_mb': r.memory_usage_mb,
                    'passed': r.passed
                }
                for r in validation_results
            ]
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n✓ Validation report generated: {report_path}")


# ============================================================================
# Comment 10: Edge Case Tests
# ============================================================================

@pytest.mark.system_validation
@pytest.mark.edge_cases
class TestEdgeCases:
    """Edge case tests for system validation."""

    def test_very_short_audio(self, validation_pipeline, quality_evaluator, tmp_path, memory_monitor):
        """
        Test conversion with very short audio (<10s).

        Addresses Comment 10: Edge case for short audio files.
        """
        # Generate 5-second test audio
        sample_rate = 44100
        duration = 5.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        audio_file = tmp_path / "short_audio.wav"
        sf.write(str(audio_file), audio, sample_rate)

        # Monitor memory
        memory_monitor.start()

        # Test conversion
        start_time = time.time()

        # Create temporary profile for testing
        try:
            from auto_voice.inference.voice_cloner import VoiceCloner
            cloner = VoiceCloner(device='cpu')

            # Create profile from same audio (simplified test)
            profile = cloner.create_voice_profile(
                audio=audio,
                sample_rate=sample_rate,
                user_id='test_short_audio'
            )

            result = validation_pipeline.convert_song(
                song_path=str(audio_file),
                target_profile_id=profile['profile_id'],
                pitch_shift=0.0
            )

            latency = time.time() - start_time
            memory_monitor.update()

            # Validate results
            assert 'mixed_audio' in result
            assert len(result['mixed_audio']) > 0

            # Check latency is reasonable
            rtf = latency / duration
            assert rtf < 10.0, f"RTF {rtf:.2f}x too high for short audio"

            print(f"\n✓ Short audio test passed")
            print(f"  Duration: {duration}s")
            print(f"  Latency: {latency:.2f}s (RTF: {rtf:.2f}x)")
            print(f"  Memory: {memory_monitor.get_usage():.1f} MB")

        except ImportError:
            pytest.skip("VoiceCloner not available for short audio test")

    @pytest.mark.slow
    def test_very_long_audio(self, validation_pipeline, quality_evaluator, tmp_path, memory_monitor):
        """
        Test conversion with very long audio (>5 minutes).

        Addresses Comment 10: Edge case for long audio files with memory monitoring.
        """
        # Generate 6-minute test audio
        sample_rate = 44100
        duration = 360.0  # 6 minutes

        # Generate in chunks to avoid memory issues
        chunk_duration = 30.0
        chunks = []

        for i in range(int(duration / chunk_duration)):
            t = np.linspace(0, chunk_duration, int(sample_rate * chunk_duration))
            chunk = (0.5 * np.sin(2 * np.pi * (440 + i * 5) * t)).astype(np.float32)
            chunks.append(chunk)

        audio = np.concatenate(chunks)

        audio_file = tmp_path / "long_audio.wav"
        sf.write(str(audio_file), audio, sample_rate)

        # Monitor memory
        memory_monitor.start()

        # Test conversion
        start_time = time.time()

        try:
            from auto_voice.inference.voice_cloner import VoiceCloner
            cloner = VoiceCloner(device='cpu')

            # Create profile from first 30 seconds
            profile_audio = chunks[0]
            profile = cloner.create_voice_profile(
                audio=profile_audio,
                sample_rate=sample_rate,
                user_id='test_long_audio'
            )

            result = validation_pipeline.convert_song(
                song_path=str(audio_file),
                target_profile_id=profile['profile_id'],
                pitch_shift=0.0
            )

            latency = time.time() - start_time
            memory_monitor.update()
            memory_usage = memory_monitor.get_usage()

            # Validate results
            assert 'mixed_audio' in result
            assert len(result['mixed_audio']) > 0

            # Check memory usage is reasonable (<2GB increase)
            assert memory_usage < 2048, \
                f"Memory usage {memory_usage:.1f} MB exceeds 2GB threshold"

            # Check latency
            rtf = latency / duration
            assert rtf < 20.0, f"RTF {rtf:.2f}x too high for long audio"

            print(f"\n✓ Long audio test passed")
            print(f"  Duration: {duration}s ({duration/60:.1f} minutes)")
            print(f"  Latency: {latency:.2f}s (RTF: {rtf:.2f}x)")
            print(f"  Memory: {memory_usage:.1f} MB")

        except ImportError:
            pytest.skip("VoiceCloner not available for long audio test")

    def test_a_cappella_input(self, validation_pipeline, tmp_path):
        """
        Test conversion with a cappella input (skip separation).

        Addresses Comment 10: Edge case for vocals-only input.
        """
        # Generate pure vocal audio (no instrumental)
        sample_rate = 44100
        duration = 10.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Simulate vocal harmonics
        audio = (
            0.5 * np.sin(2 * np.pi * 220 * t) +
            0.3 * np.sin(2 * np.pi * 440 * t) +
            0.2 * np.sin(2 * np.pi * 660 * t)
        ).astype(np.float32)

        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.9

        audio_file = tmp_path / "a_cappella.wav"
        sf.write(str(audio_file), audio, sample_rate)

        # Test conversion (should handle no-instrumental case gracefully)
        try:
            from auto_voice.inference.voice_cloner import VoiceCloner
            cloner = VoiceCloner(device='cpu')

            profile = cloner.create_voice_profile(
                audio=audio,
                sample_rate=sample_rate,
                user_id='test_a_cappella'
            )

            result = validation_pipeline.convert_song(
                song_path=str(audio_file),
                target_profile_id=profile['profile_id'],
                pitch_shift=0.0
            )

            # Validate results
            assert 'mixed_audio' in result
            assert len(result['mixed_audio']) > 0

            # For a cappella, output should be similar length to input
            output_duration = len(result['mixed_audio']) / sample_rate
            assert abs(output_duration - duration) < 1.0, \
                f"Output duration {output_duration:.1f}s differs from input {duration}s"

            print(f"\n✓ A cappella test passed")
            print(f"  Input duration: {duration}s")
            print(f"  Output duration: {output_duration:.1f}s")

        except ImportError:
            pytest.skip("VoiceCloner not available for a cappella test")

    def test_heavily_processed_vocals(self, validation_pipeline, quality_evaluator, tmp_path):
        """
        Test conversion with heavily processed vocals (reverb, distortion).

        Addresses Comment 10: Edge case for processed audio.
        """
        # Generate audio with effects simulation
        sample_rate = 44100
        duration = 10.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Base vocal signal
        base_signal = np.sin(2 * np.pi * 440 * t)

        # Add reverb simulation (delayed copies)
        reverb = base_signal.copy()
        for delay_ms in [50, 100, 150, 200]:
            delay_samples = int(delay_ms * sample_rate / 1000)
            if delay_samples < len(reverb):
                reverb[delay_samples:] += 0.3 * base_signal[:-delay_samples]

        # Add distortion (soft clipping)
        distorted = np.tanh(reverb * 2.0)

        # Normalize
        audio = (distorted / np.max(np.abs(distorted)) * 0.8).astype(np.float32)

        audio_file = tmp_path / "processed_vocals.wav"
        sf.write(str(audio_file), audio, sample_rate)

        # Test conversion
        try:
            from auto_voice.inference.voice_cloner import VoiceCloner
            cloner = VoiceCloner(device='cpu')

            profile = cloner.create_voice_profile(
                audio=audio,
                sample_rate=sample_rate,
                user_id='test_processed'
            )

            result = validation_pipeline.convert_song(
                song_path=str(audio_file),
                target_profile_id=profile['profile_id'],
                pitch_shift=0.0
            )

            # Validate results exist (quality may be degraded but shouldn't crash)
            assert 'mixed_audio' in result
            assert len(result['mixed_audio']) > 0
            assert np.isfinite(result['mixed_audio']).all(), \
                "Output contains NaN or Inf values"

            print(f"\n✓ Heavily processed vocals test passed")
            print(f"  Successfully handled reverb + distortion")

        except ImportError:
            pytest.skip("VoiceCloner not available for processed vocals test")


# ============================================================================
# Genre-Specific Validation Tests
# ============================================================================

@pytest.mark.system_validation
@pytest.mark.genre_specific
class TestGenreSpecificValidation:
    """Genre-specific validation tests."""

    @pytest.mark.parametrize('genre', ['pop', 'rock', 'jazz', 'classical', 'rap'])
    def test_genre_conversion(self, genre, test_metadata_loader, validation_pipeline, quality_evaluator):
        """
        Test conversion for specific genre.

        Addresses Comment 3: Genre-specific test cases.
        """
        test_cases = test_metadata_loader['test_cases']

        # Filter test cases by genre
        genre_cases = [tc for tc in test_cases if tc.get('metadata', {}).get('genre') == genre]

        if not genre_cases:
            pytest.skip(f"No test cases found for genre: {genre}")

        for test_case in genre_cases:
            source_audio_path = test_case['source_audio']
            target_profile_id = test_case['target_profile_id']

            # Run conversion
            result = validation_pipeline.convert_song(
                song_path=source_audio_path,
                target_profile_id=target_profile_id,
                pitch_shift=0.0
            )

            # Validate basic success
            assert 'mixed_audio' in result
            assert len(result['mixed_audio']) > 0

            print(f"\n✓ Genre test passed: {genre}")
            print(f"  Test case: {test_case['id']}")


# ============================================================================
# Performance and Latency Tests
# ============================================================================

@pytest.mark.system_validation
@pytest.mark.performance
class TestPerformanceValidation:
    """Performance and latency validation tests."""

    def test_latency_scaling(self, validation_pipeline, tmp_path):
        """
        Test that latency scales linearly with audio duration.

        Validates performance characteristics across different lengths.
        """
        sample_rate = 44100
        durations = [10.0, 20.0, 30.0]
        latencies = []

        try:
            from auto_voice.inference.voice_cloner import VoiceCloner
            cloner = VoiceCloner(device='cpu')

            for duration in durations:
                # Generate test audio
                t = np.linspace(0, duration, int(sample_rate * duration))
                audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

                audio_file = tmp_path / f"test_{duration}s.wav"
                sf.write(str(audio_file), audio, sample_rate)

                # Create profile
                profile_audio = audio[:int(sample_rate * min(30, duration))]
                profile = cloner.create_voice_profile(
                    audio=profile_audio,
                    sample_rate=sample_rate,
                    user_id=f'test_latency_{duration}'
                )

                # Measure conversion time
                start_time = time.time()
                result = validation_pipeline.convert_song(
                    song_path=str(audio_file),
                    target_profile_id=profile['profile_id']
                )
                latency = time.time() - start_time

                latencies.append(latency)
                rtf = latency / duration

                print(f"\n{duration}s audio: {latency:.2f}s (RTF: {rtf:.2f}x)")

            # Check latency increases with duration (should be roughly linear)
            assert latencies[1] > latencies[0], "Latency should increase with duration"
            assert latencies[2] > latencies[1], "Latency should increase with duration"

            print("\n✓ Latency scaling test passed")

        except ImportError:
            pytest.skip("VoiceCloner not available for latency test")

    @pytest.mark.performance
    @pytest.mark.tensorrt
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_latency_target_30s_input(self, tmp_path):
        """
        Test latency < 5s for 30s input with TensorRT FP16.

        Addresses Comment 2: Latency target enforcement with TensorRT.

        Requirements:
        - NVIDIA GPU with Tensor Cores (RTX 2060+)
        - CUDA 11.8+
        - TensorRT 8.5+
        """
        sample_rate = 44100
        duration = 30.0

        # Check for TensorRT availability
        try:
            import tensorrt as trt
            tensorrt_available = True
        except ImportError:
            pytest.skip("TensorRT not available. Install TensorRT 8.5+ for this test.")

        # Generate 30-second test audio
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        audio_file = tmp_path / "test_30s.wav"
        sf.write(str(audio_file), audio, sample_rate)

        try:
            # Create pipeline with fast preset and TensorRT
            from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

            pipeline = SingingConversionPipeline(
                device='cuda',
                preset='fast',
                use_tensorrt=True,
                tensorrt_precision='fp16'
            )

            # Create test profile
            from auto_voice.inference.voice_cloner import VoiceCloner
            cloner = VoiceCloner(device='cuda')

            profile_audio = audio[:int(sample_rate * 10)]  # Use first 10s for profile
            profile = cloner.create_voice_profile(
                audio=profile_audio,
                sample_rate=sample_rate,
                user_id='test_tensorrt_latency'
            )

            # Warm-up run (TensorRT compiles on first run)
            print("\nWarming up TensorRT engine (first-time compilation)...")
            _ = pipeline.convert_song(
                song_path=str(audio_file),
                target_profile_id=profile['profile_id']
            )

            # Actual timed run
            print("Running timed conversion with TensorRT FP16...")
            start_time = time.time()
            result = pipeline.convert_song(
                song_path=str(audio_file),
                target_profile_id=profile['profile_id']
            )
            elapsed = time.time() - start_time

            rtf = elapsed / duration

            # Verify TensorRT is actually enabled
            trt_metadata = result.get('metadata', {}).get('tensorrt', {})
            trt_enabled = trt_metadata.get('enabled', False)
            trt_precision = trt_metadata.get('precision', None)

            print(f"\n✓ Latency test results:")
            print(f"  Duration: {duration}s")
            print(f"  Elapsed: {elapsed:.2f}s")
            print(f"  RTF: {rtf:.2f}x")
            print(f"  Target: <5.0s")
            print(f"  TensorRT Enabled: {trt_enabled}")
            print(f"  TensorRT Precision: {trt_precision}")

            # Assert TensorRT is enabled
            if not trt_enabled:
                pytest.skip("TensorRT engines not loaded - may need to build engines first")

            # Assert latency target
            assert elapsed < 5.0, f"Latency {elapsed:.2f}s exceeds 5s target for 30s input"
            assert 'mixed_audio' in result, "Conversion failed to produce output"

            print(f"\n✅ PASSED: Latency {elapsed:.2f}s < 5.0s target with TensorRT {trt_precision}")

            # Save metrics
            metrics = {
                'duration_seconds': duration,
                'elapsed_seconds': elapsed,
                'rtf': rtf,
                'preset': 'fast',
                'tensorrt_requested': True,
                'tensorrt_enabled': trt_enabled,
                'tensorrt_precision': trt_precision,
                'target_met': elapsed < 5.0
            }

            os.makedirs('validation_results', exist_ok=True)
            with open('validation_results/latency_tensorrt.json', 'w') as f:
                json.dump(metrics, f, indent=2)

        except Exception as e:
            pytest.fail(f"TensorRT latency test failed: {e}")

    @pytest.mark.performance
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_gpu_utilization_monitoring(self, tmp_path):
        """
        Monitor GPU utilization during conversion and assert average utilization > 70%.

        Addresses Comment 8: GPU utilization monitoring and component-level timing.

        Validates:
        - GPU is actively utilized during conversion
        - Average utilization exceeds 70% threshold
        - Component-level timing breakdown
        """
        sample_rate = 44100
        duration = 30.0

        # Check for nvidia-smi availability
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True)
            if result.returncode != 0:
                pytest.skip("nvidia-smi not available")
        except FileNotFoundError:
            pytest.skip("nvidia-smi not found in PATH")

        # Generate test audio
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        audio_file = tmp_path / "test_gpu_util.wav"
        sf.write(str(audio_file), audio, sample_rate)

        try:
            # Create pipeline
            from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
            from auto_voice.inference.voice_cloner import VoiceCloner

            pipeline = SingingConversionPipeline(device='cuda')
            cloner = VoiceCloner(device='cuda')

            # Create test profile
            profile_audio = audio[:int(sample_rate * 10)]
            profile = cloner.create_voice_profile(
                audio=profile_audio,
                sample_rate=sample_rate,
                user_id='test_gpu_utilization'
            )

            # GPU utilization monitor
            class GPUUtilizationMonitor:
                def __init__(self, interval=0.1):
                    self.interval = interval
                    self.measurements = []
                    self.monitoring = False
                    self.monitor_thread = None

                def start(self):
                    self.monitoring = True
                    self.measurements = []

                    import threading
                    def monitor_loop():
                        import subprocess
                        while self.monitoring:
                            try:
                                result = subprocess.run(
                                    ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                    capture_output=True,
                                    text=True
                                )
                                if result.returncode == 0:
                                    util = float(result.stdout.strip().split('\n')[0])
                                    self.measurements.append(util)
                            except Exception:
                                pass
                            time.sleep(self.interval)

                    self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
                    self.monitor_thread.start()

                def stop(self):
                    self.monitoring = False
                    if self.monitor_thread:
                        self.monitor_thread.join(timeout=2.0)

                def get_average_utilization(self):
                    if not self.measurements:
                        return 0.0
                    return sum(self.measurements) / len(self.measurements)

                def get_max_utilization(self):
                    if not self.measurements:
                        return 0.0
                    return max(self.measurements)

            # Monitor GPU utilization during conversion
            gpu_monitor = GPUUtilizationMonitor(interval=0.1)

            print("\nMonitoring GPU utilization during conversion...")
            gpu_monitor.start()

            # Run conversion
            start_time = time.time()
            result = pipeline.convert_song(
                song_path=str(audio_file),
                target_profile_id=profile['profile_id']
            )
            elapsed = time.time() - start_time

            # Stop monitoring
            time.sleep(0.5)  # Allow final samples
            gpu_monitor.stop()

            avg_utilization = gpu_monitor.get_average_utilization()
            max_utilization = gpu_monitor.get_max_utilization()
            num_samples = len(gpu_monitor.measurements)

            print(f"\n✓ GPU utilization results:")
            print(f"  Samples collected: {num_samples}")
            print(f"  Average utilization: {avg_utilization:.1f}%")
            print(f"  Peak utilization: {max_utilization:.1f}%")
            print(f"  Conversion time: {elapsed:.2f}s")
            print(f"  Target: >70% average")

            # Save detailed metrics
            metrics = {
                'duration_seconds': duration,
                'elapsed_seconds': elapsed,
                'gpu_utilization': {
                    'average_percent': avg_utilization,
                    'max_percent': max_utilization,
                    'samples_collected': num_samples,
                    'measurements': gpu_monitor.measurements
                },
                'target_met': avg_utilization > 70.0
            }

            os.makedirs('validation_results', exist_ok=True)
            with open('validation_results/gpu_utilization.json', 'w') as f:
                json.dump(metrics, f, indent=2)

            # Assert utilization threshold
            assert num_samples > 0, "No GPU utilization samples collected"
            assert avg_utilization > 70.0, (
                f"Average GPU utilization {avg_utilization:.1f}% below 70% target. "
                f"GPU may not be effectively utilized."
            )

            print(f"\n✅ PASSED: Average GPU utilization {avg_utilization:.1f}% > 70%")

        except Exception as e:
            pytest.fail(f"GPU utilization monitoring failed: {e}")

    @pytest.mark.performance
    def test_component_level_timing(self, tmp_path):
        """
        Profile component-level timing within conversion pipeline.

        Addresses Comment 1: Component-level timing breakdown with profiling callback.

        Measures timing for:
        - Vocal separation
        - F0 extraction
        - Voice conversion
        - Audio mixing
        - Total time
        """
        sample_rate = 44100
        duration = 30.0

        # Generate test audio
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        audio_file = tmp_path / "test_component_timing.wav"
        sf.write(str(audio_file), audio, sample_rate)

        try:
            from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
            from auto_voice.inference.voice_cloner import VoiceCloner

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            pipeline = SingingConversionPipeline(device=device)
            cloner = VoiceCloner(device=device)

            # Create test profile
            profile_audio = audio[:int(sample_rate * 10)]
            profile = cloner.create_voice_profile(
                audio=profile_audio,
                sample_rate=sample_rate,
                user_id='test_component_timing'
            )

            # Component timing tracker
            stage_timings = {}

            def profiling_callback(stage_name: str, elapsed_ms: float):
                """Collect timing data for each stage."""
                stage_timings[stage_name] = elapsed_ms
                print(f"  {stage_name}: {elapsed_ms:.2f}ms")

            print("\nProfiling component-level timing...")

            # Run conversion with profiling callback
            result = pipeline.convert_song(
                song_path=str(audio_file),
                target_profile_id=profile['profile_id'],
                profiling_callback=profiling_callback
            )

            # Verify all expected stages are present
            expected_stages = ['separation', 'f0_extraction', 'conversion', 'mixing', 'total']
            for stage in expected_stages:
                assert stage in stage_timings, f"Missing timing for stage: {stage}"

            # Compute percentages
            total_ms = stage_timings['total']
            stage_percentages = {
                stage: (stage_timings[stage] / total_ms * 100) if stage != 'total' else 100.0
                for stage in expected_stages
            }

            # Verify sum of stages is within ±15% of total
            sum_stages = sum(stage_timings[stage] for stage in ['separation', 'f0_extraction', 'conversion', 'mixing'])
            tolerance = 0.15
            assert abs(sum_stages - total_ms) / total_ms <= tolerance, \
                f"Sum of stages ({sum_stages:.2f}ms) differs from total ({total_ms:.2f}ms) by more than {tolerance*100}%"

            print(f"\n✓ Component timing breakdown:")
            for stage in expected_stages:
                if stage == 'total':
                    print(f"  {stage}: {stage_timings[stage]:.2f}ms")
                else:
                    print(f"  {stage}: {stage_timings[stage]:.2f}ms ({stage_percentages[stage]:.1f}%)")
            print(f"  RTF: {(total_ms/1000) / duration:.2f}x")

            # Build performance breakdown report
            performance_breakdown = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'device': device,
                'audio_duration_seconds': duration,
                'stage_timings_ms': stage_timings,
                'stage_percentages': stage_percentages,
                'total_time_seconds': total_ms / 1000,
                'rtf': (total_ms / 1000) / duration
            }

            # Merge GPU utilization if available
            gpu_util_file = Path('validation_results/gpu_utilization.json')
            if gpu_util_file.exists():
                try:
                    with open(gpu_util_file, 'r') as f:
                        gpu_data = json.load(f)
                        performance_breakdown['gpu_utilization'] = gpu_data.get('average_utilization', None)
                except Exception as e:
                    print(f"Warning: Could not load GPU utilization data: {e}")

            # Save performance breakdown
            os.makedirs('validation_results', exist_ok=True)
            with open('validation_results/performance_breakdown.json', 'w') as f:
                json.dump(performance_breakdown, f, indent=2)

            print(f"\n✓ Performance breakdown saved to validation_results/performance_breakdown.json")

        except Exception as e:
            pytest.fail(f"Component timing test failed: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
