"""Tests for inference latency profiling.

Task 7.3: Profile inference latency with continuous training models

Tests cover:
- Stage-by-stage latency breakdown
- Latency tracking with trained vs base models
- Real-time factor (RTF) calculation
- Latency regression detection
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Get CUDA device, skip test if unavailable."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device('cuda')


@pytest.fixture
def temp_storage():
    """Temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_rate():
    """Standard sample rate for tests."""
    return 22050


@pytest.fixture
def sample_audio(sample_rate):
    """Generate test audio (1 second sine wave)."""
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


# ============================================================================
# Test: Inference Latency Profiler
# ============================================================================

@pytest.mark.cuda
class TestInferenceLatencyProfiler:
    """Tests for InferenceLatencyProfiler class."""

    def test_profiler_creation(self, device):
        """Profiler should initialize with device."""
        from auto_voice.gpu.latency_profiler import InferenceLatencyProfiler

        profiler = InferenceLatencyProfiler(device='cuda:0')

        assert profiler.device == 'cuda:0'
        assert profiler.measurements == {}

    def test_measure_stage_records_time(self, device):
        """measure_stage should record execution time."""
        from auto_voice.gpu.latency_profiler import InferenceLatencyProfiler

        profiler = InferenceLatencyProfiler(device='cuda:0')

        with profiler.measure_stage('test_stage'):
            time.sleep(0.1)  # 100ms

        assert 'test_stage' in profiler.measurements
        assert len(profiler.measurements['test_stage']) == 1
        # Should be roughly 100ms (allow 50ms tolerance)
        assert 0.05 < profiler.measurements['test_stage'][0] < 0.2

    def test_measure_multiple_stages(self, device):
        """measure_stage should track multiple distinct stages."""
        from auto_voice.gpu.latency_profiler import InferenceLatencyProfiler

        profiler = InferenceLatencyProfiler(device='cuda:0')

        with profiler.measure_stage('stage_a'):
            time.sleep(0.05)
        with profiler.measure_stage('stage_b'):
            time.sleep(0.05)
        with profiler.measure_stage('stage_a'):  # Second measurement of stage_a
            time.sleep(0.05)

        assert len(profiler.measurements['stage_a']) == 2
        assert len(profiler.measurements['stage_b']) == 1

    def test_get_stats_returns_summary(self, device):
        """get_stats should return mean, std, min, max for each stage."""
        from auto_voice.gpu.latency_profiler import InferenceLatencyProfiler

        profiler = InferenceLatencyProfiler(device='cuda:0')

        # Add multiple measurements
        for _ in range(5):
            with profiler.measure_stage('test_stage'):
                time.sleep(0.05)

        stats = profiler.get_stats()

        assert 'test_stage' in stats
        assert 'mean_ms' in stats['test_stage']
        assert 'std_ms' in stats['test_stage']
        assert 'min_ms' in stats['test_stage']
        assert 'max_ms' in stats['test_stage']
        assert stats['test_stage']['count'] == 5
        # Mean should be roughly 50ms
        assert 30 < stats['test_stage']['mean_ms'] < 100

    def test_sync_cuda_before_timing(self, device):
        """Profiler should sync CUDA before timing GPU operations."""
        from auto_voice.gpu.latency_profiler import InferenceLatencyProfiler

        profiler = InferenceLatencyProfiler(device='cuda:0', sync_cuda=True)

        # GPU operation
        with profiler.measure_stage('gpu_op'):
            x = torch.randn(1000, 1000, device=device)
            y = x @ x  # Matrix multiply
            del x, y

        assert 'gpu_op' in profiler.measurements

    def test_calculate_rtf(self, device, sample_audio, sample_rate):
        """calculate_rtf should compute real-time factor correctly."""
        from auto_voice.gpu.latency_profiler import InferenceLatencyProfiler

        profiler = InferenceLatencyProfiler(device='cuda:0')

        # Simulate processing 1 second of audio in 0.5 seconds
        with profiler.measure_stage('inference'):
            time.sleep(0.5)

        audio_duration = len(sample_audio) / sample_rate  # 1.0 seconds
        rtf = profiler.calculate_rtf('inference', audio_duration)

        # RTF should be roughly 0.5 (processing in half the audio duration)
        assert 0.3 < rtf < 0.7

    def test_clear_measurements(self, device):
        """clear should reset all measurements."""
        from auto_voice.gpu.latency_profiler import InferenceLatencyProfiler

        profiler = InferenceLatencyProfiler(device='cuda:0')

        with profiler.measure_stage('test_stage'):
            time.sleep(0.01)

        assert len(profiler.measurements['test_stage']) == 1

        profiler.clear()

        assert profiler.measurements == {}


@pytest.mark.cuda
class TestPipelineLatencyIntegration:
    """Tests for latency profiling integrated with inference pipeline."""

    def test_profile_model_inference(self, device, sample_audio, sample_rate):
        """Profile model inference latency."""
        from auto_voice.gpu.latency_profiler import InferenceLatencyProfiler
        from auto_voice.models.so_vits_svc import SoVitsSvc

        profiler = InferenceLatencyProfiler(device='cuda:0', sync_cuda=True)
        model = SoVitsSvc().to(device)
        model.train(False)  # Set to inference mode

        # Profile multiple inferences
        # SoVitsSvc expects: content [B, T, content_dim], pitch [B, T, pitch_dim], speaker [B, speaker_dim]
        # Default dims from model config: content_dim=256, pitch_dim=256, speaker_dim=256
        batch_size = 1
        seq_len = 100
        content_dim = 256
        pitch_dim = 256
        speaker_dim = 256

        for _ in range(3):
            with profiler.measure_stage('model_forward'):
                content = torch.randn(batch_size, seq_len, content_dim, device=device)
                pitch = torch.randn(batch_size, seq_len, pitch_dim, device=device)
                speaker = torch.randn(batch_size, speaker_dim, device=device)

                with torch.no_grad():
                    _ = model(content, pitch, speaker)

        stats = profiler.get_stats()
        assert 'model_forward' in stats
        assert stats['model_forward']['count'] == 3

    def test_compare_base_vs_trained_latency(self, device, temp_storage):
        """Compare latency between base model and continuously trained model."""
        from auto_voice.gpu.latency_profiler import InferenceLatencyProfiler, compare_model_latency
        from auto_voice.models.so_vits_svc import SoVitsSvc

        profiler = InferenceLatencyProfiler(device='cuda:0', sync_cuda=True)

        # Create two models (simulating base vs trained)
        base_model = SoVitsSvc().to(device)
        trained_model = SoVitsSvc().to(device)  # In practice, this would be fine-tuned

        # SoVitsSvc expects: content [B, T, content_dim], pitch [B, T, pitch_dim], speaker [B, speaker_dim]
        # Default dims from model config: content_dim=256, pitch_dim=256, speaker_dim=256
        batch_size = 1
        seq_len = 100
        content_dim = 256
        pitch_dim = 256
        speaker_dim = 256

        def create_inputs():
            return {
                'content': torch.randn(batch_size, seq_len, content_dim, device=device),
                'pitch': torch.randn(batch_size, seq_len, pitch_dim, device=device),
                'speaker': torch.randn(batch_size, speaker_dim, device=device),
            }

        comparison = compare_model_latency(
            base_model=base_model,
            trained_model=trained_model,
            input_fn=create_inputs,
            device=device,
            num_runs=5,
        )

        assert 'base_mean_ms' in comparison
        assert 'trained_mean_ms' in comparison
        assert 'latency_diff_ms' in comparison
        assert 'latency_diff_pct' in comparison

    def test_detect_latency_regression(self, device):
        """Detect latency regression when model becomes slower."""
        from auto_voice.gpu.latency_profiler import detect_latency_regression

        # Baseline stats (fast model)
        baseline_stats = {
            'model_forward': {
                'mean_ms': 10.0,
                'std_ms': 1.0,
                'min_ms': 8.0,
                'max_ms': 12.0,
                'count': 100,
            }
        }

        # Current stats (slower model - regression)
        current_stats = {
            'model_forward': {
                'mean_ms': 15.0,  # 50% slower
                'std_ms': 2.0,
                'min_ms': 12.0,
                'max_ms': 20.0,
                'count': 10,
            }
        }

        regression = detect_latency_regression(
            baseline_stats=baseline_stats,
            current_stats=current_stats,
            threshold_pct=20.0,  # 20% threshold
        )

        assert regression['has_regression'] is True
        assert 'model_forward' in regression['stages_with_regression']

    def test_no_regression_within_threshold(self, device):
        """No regression when latency within threshold."""
        from auto_voice.gpu.latency_profiler import detect_latency_regression

        baseline_stats = {
            'model_forward': {
                'mean_ms': 10.0,
                'std_ms': 1.0,
            }
        }

        current_stats = {
            'model_forward': {
                'mean_ms': 11.0,  # 10% slower
                'std_ms': 1.0,
            }
        }

        regression = detect_latency_regression(
            baseline_stats=baseline_stats,
            current_stats=current_stats,
            threshold_pct=20.0,  # 20% threshold
        )

        assert regression['has_regression'] is False


@pytest.mark.cuda
class TestPipelineStageBreakdown:
    """Tests for detailed stage-by-stage latency breakdown."""

    def test_profile_full_pipeline_stages(self, device, sample_audio, sample_rate):
        """Profile all stages of the inference pipeline."""
        from auto_voice.gpu.latency_profiler import (
            InferenceLatencyProfiler,
            PIPELINE_STAGES,
        )

        profiler = InferenceLatencyProfiler(device='cuda:0')

        # Verify standard pipeline stages are defined
        expected_stages = [
            'load_audio',
            'vocal_separation',
            'pitch_extraction',
            'technique_detection',
            'voice_conversion',
            'mixing',
        ]

        for stage in expected_stages:
            assert stage in PIPELINE_STAGES, f"Missing standard stage: {stage}"

    def test_export_profile_results(self, device, temp_storage):
        """Export profile results to JSON."""
        from auto_voice.gpu.latency_profiler import InferenceLatencyProfiler

        profiler = InferenceLatencyProfiler(device='cuda:0')

        with profiler.measure_stage('test_stage'):
            time.sleep(0.05)

        output_path = temp_storage / 'profile_results.json'
        profiler.export_json(str(output_path))

        assert output_path.exists()

        import json
        with open(output_path) as f:
            data = json.load(f)

        assert 'device' in data
        assert 'stats' in data
        assert 'test_stage' in data['stats']

    def test_profile_with_warmup(self, device):
        """Profile should support warmup iterations."""
        from auto_voice.gpu.latency_profiler import InferenceLatencyProfiler

        profiler = InferenceLatencyProfiler(device='cuda:0', warmup_runs=2)

        # Run 5 iterations (first 2 are warmup)
        for _ in range(5):
            with profiler.measure_stage('test_stage'):
                time.sleep(0.01)

        # Should only have 3 measurements (excluding warmup)
        assert profiler.measurements.get('test_stage') is not None
        # With warmup_runs=2, the profiler skips first 2 measurements
        assert len(profiler.measurements['test_stage']) == 3


@pytest.mark.cuda
class TestLatencyBenchmarkSuite:
    """Tests for the latency benchmarking suite."""

    def test_benchmark_runner_executes(self, device, temp_storage):
        """Benchmark runner should execute and produce results."""
        from auto_voice.gpu.latency_profiler import LatencyBenchmarkRunner

        runner = LatencyBenchmarkRunner(
            device='cuda:0',
            output_dir=str(temp_storage),
        )

        # Run simple benchmark
        results = runner.run_benchmark(
            model_name='test_model',
            input_sizes=[(1, 256, 100)],  # (batch, dim, seq)
            num_runs=3,
        )

        assert 'model_name' in results
        assert 'input_sizes' in results
        assert 'latencies' in results

    def test_benchmark_generates_report(self, device, temp_storage):
        """Benchmark should generate a report file."""
        from auto_voice.gpu.latency_profiler import LatencyBenchmarkRunner

        runner = LatencyBenchmarkRunner(
            device='cuda:0',
            output_dir=str(temp_storage),
        )

        runner.run_benchmark(
            model_name='test_model',
            input_sizes=[(1, 256, 100)],
            num_runs=3,
        )

        report_path = temp_storage / 'latency_report_test_model.json'
        runner.generate_report(str(report_path))

        assert report_path.exists()
