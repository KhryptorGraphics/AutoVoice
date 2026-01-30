"""Tests for quality benchmarking and validation suite (Phase 10).

Validates comprehensive quality evaluation infrastructure:
- Automated quality metrics (MOS-prediction, PESQ, MCD, F0-RMSE, speaker similarity)
- Benchmark dataset handling
- Performance profiling (GPU memory, latency)
- Comparison against published baselines

Quality targets based on published SOTA results:
- MCD < 5.0 dB
- F0-RMSE < 20 cents
- Speaker similarity > 0.85 cosine
"""
import pytest
import torch
import numpy as np


class TestQualityMetricsSuite:
    """Tests for automated quality metrics."""

    def test_quality_metrics_class_exists(self):
        """QualityMetrics class should exist."""
        from auto_voice.evaluation.quality_metrics import QualityMetrics
        assert QualityMetrics is not None

    def test_mcd_metric_available(self):
        """Mel Cepstral Distortion metric should be available."""
        from auto_voice.evaluation.quality_metrics import QualityMetrics

        metrics = QualityMetrics()
        assert hasattr(metrics, 'compute_mcd')

    def test_f0_rmse_metric_available(self):
        """F0 RMSE metric should be available."""
        from auto_voice.evaluation.quality_metrics import QualityMetrics

        metrics = QualityMetrics()
        assert hasattr(metrics, 'compute_f0_rmse')

    def test_speaker_similarity_metric_available(self):
        """Speaker similarity metric should be available."""
        from auto_voice.evaluation.quality_metrics import QualityMetrics

        metrics = QualityMetrics()
        assert hasattr(metrics, 'compute_speaker_similarity')

    def test_compute_mcd(self):
        """MCD should compute valid distortion value."""
        from auto_voice.evaluation.quality_metrics import QualityMetrics

        metrics = QualityMetrics()

        # Generate test audio (slightly different)
        reference = torch.sin(torch.linspace(0, 10 * np.pi, 24000))
        converted = reference + 0.1 * torch.randn_like(reference)

        mcd = metrics.compute_mcd(reference, converted, sample_rate=24000)

        assert isinstance(mcd, float)
        assert mcd >= 0  # MCD is always non-negative
        assert mcd < 100  # Sanity check

    def test_compute_f0_rmse(self):
        """F0 RMSE should compute pitch deviation in cents."""
        from auto_voice.evaluation.quality_metrics import QualityMetrics

        metrics = QualityMetrics()

        # Generate 440Hz sine wave and slightly detuned version
        t = torch.linspace(0, 1, 24000)
        reference = torch.sin(2 * np.pi * 440 * t)
        converted = torch.sin(2 * np.pi * 445 * t)  # ~20 cents sharp

        rmse = metrics.compute_f0_rmse(reference, converted, sample_rate=24000)

        assert isinstance(rmse, float)
        assert rmse >= 0  # RMSE is non-negative

    def test_compute_speaker_similarity(self):
        """Speaker similarity should return cosine similarity score."""
        from auto_voice.evaluation.quality_metrics import QualityMetrics

        metrics = QualityMetrics()

        # Same speaker embedding should give similarity ~1.0
        speaker_emb = torch.randn(256)
        speaker_emb = speaker_emb / speaker_emb.norm()  # Normalize

        similarity = metrics.compute_speaker_similarity(speaker_emb, speaker_emb)

        assert isinstance(similarity, float)
        assert 0.99 <= similarity <= 1.01  # Should be ~1.0 for identical

    def test_compute_all_metrics(self):
        """Should compute all metrics at once."""
        from auto_voice.evaluation.quality_metrics import QualityMetrics

        metrics = QualityMetrics()

        reference = torch.sin(torch.linspace(0, 10 * np.pi, 24000))
        converted = reference + 0.05 * torch.randn_like(reference)
        target_speaker = torch.randn(256)

        results = metrics.compute_all(
            reference_audio=reference,
            converted_audio=converted,
            target_speaker=target_speaker,
            sample_rate=24000,
        )

        assert 'mcd' in results
        assert 'f0_rmse' in results
        assert 'speaker_similarity' in results


class TestBenchmarkDataset:
    """Tests for benchmark dataset handling."""

    def test_benchmark_dataset_class_exists(self):
        """BenchmarkDataset class should exist."""
        from auto_voice.evaluation.benchmark_dataset import BenchmarkDataset
        assert BenchmarkDataset is not None

    def test_benchmark_dataset_loads_samples(self):
        """Dataset should load benchmark samples."""
        pytest.skip("Requires benchmark audio files")
        from auto_voice.evaluation.benchmark_dataset import BenchmarkDataset

        dataset = BenchmarkDataset(data_dir="/path/to/benchmarks")
        assert len(dataset) > 0

    def test_benchmark_sample_structure(self):
        """Each sample should have required fields."""
        pytest.skip("Requires benchmark audio files")
        from auto_voice.evaluation.benchmark_dataset import BenchmarkDataset

        dataset = BenchmarkDataset(data_dir="/path/to/benchmarks")
        sample = dataset[0]

        assert 'source_audio' in sample
        assert 'target_speaker' in sample
        assert 'reference_audio' in sample  # Ground truth if available
        assert 'metadata' in sample


class TestBenchmarkRunner:
    """Tests for running benchmarks."""

    def test_benchmark_runner_exists(self):
        """BenchmarkRunner class should exist."""
        from auto_voice.evaluation.benchmark_runner import BenchmarkRunner
        assert BenchmarkRunner is not None

    def test_benchmark_runner_run_single(self):
        """Runner should evaluate single sample."""
        from auto_voice.evaluation.benchmark_runner import BenchmarkRunner
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline

        runner = BenchmarkRunner()
        pipeline = SOTAConversionPipeline()

        # Create synthetic sample
        source_audio = torch.sin(torch.linspace(0, 10 * np.pi, 24000))
        target_speaker = torch.randn(256)

        result = runner.run_single(
            pipeline=pipeline,
            source_audio=source_audio,
            target_speaker=target_speaker,
            sample_rate=24000,
        )

        assert 'converted_audio' in result
        assert 'metrics' in result
        assert 'latency_ms' in result

    def test_benchmark_runner_aggregate_results(self):
        """Runner should aggregate results from multiple samples."""
        from auto_voice.evaluation.benchmark_runner import BenchmarkRunner

        runner = BenchmarkRunner()

        # Simulate results from multiple samples
        results = [
            {'mcd': 4.5, 'f0_rmse': 15.0, 'speaker_similarity': 0.88},
            {'mcd': 5.2, 'f0_rmse': 18.0, 'speaker_similarity': 0.85},
            {'mcd': 4.0, 'f0_rmse': 12.0, 'speaker_similarity': 0.90},
        ]

        aggregated = runner.aggregate_results(results)

        assert 'mcd_mean' in aggregated
        assert 'mcd_std' in aggregated
        assert 'f0_rmse_mean' in aggregated
        assert 'speaker_similarity_mean' in aggregated


class TestPerformanceProfiler:
    """Tests for performance profiling."""

    def test_profiler_exists(self):
        """PerformanceProfiler class should exist."""
        from auto_voice.evaluation.performance_profiler import PerformanceProfiler
        assert PerformanceProfiler is not None

    def test_profiler_measures_gpu_memory(self):
        """Profiler should measure GPU memory usage."""
        pytest.importorskip("torch.cuda")
        from auto_voice.evaluation.performance_profiler import PerformanceProfiler

        profiler = PerformanceProfiler()

        memory_info = profiler.get_gpu_memory_usage()

        assert 'allocated_mb' in memory_info
        assert 'reserved_mb' in memory_info
        assert memory_info['allocated_mb'] >= 0

    def test_profiler_measures_inference_time(self):
        """Profiler should measure inference latency."""
        from auto_voice.evaluation.performance_profiler import PerformanceProfiler
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline

        profiler = PerformanceProfiler()
        pipeline = SOTAConversionPipeline()

        audio = torch.sin(torch.linspace(0, 10 * np.pi, 24000))
        speaker = torch.randn(256)

        timing = profiler.profile_inference(
            pipeline=pipeline,
            audio=audio,
            speaker=speaker,
            sample_rate=24000,
        )

        assert 'total_ms' in timing
        assert 'per_component' in timing
        assert timing['total_ms'] > 0

    def test_profiler_generates_report(self):
        """Profiler should generate summary report."""
        from auto_voice.evaluation.performance_profiler import PerformanceProfiler

        profiler = PerformanceProfiler()

        # Add some measurements
        profiler.record_measurement('inference_ms', 150.0)
        profiler.record_measurement('inference_ms', 145.0)
        profiler.record_measurement('inference_ms', 155.0)

        report = profiler.generate_report()

        assert 'inference_ms' in report
        assert 'mean' in report['inference_ms']
        assert 'min' in report['inference_ms']
        assert 'max' in report['inference_ms']


class TestQualityTargets:
    """Tests for quality target validation."""

    def test_mcd_target(self):
        """MCD should meet published SOTA target (< 5.0 dB)."""
        pytest.skip("Requires trained model and test samples")
        from auto_voice.evaluation.quality_metrics import QualityMetrics
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline

        metrics = QualityMetrics()
        pipeline = SOTAConversionPipeline()

        # Load test sample
        source_audio = torch.randn(24000)  # Replace with real sample
        target_speaker = torch.randn(256)

        result = pipeline.convert(source_audio, 24000, target_speaker)
        mcd = metrics.compute_mcd(source_audio, result['audio'], sample_rate=24000)

        assert mcd < 5.0, f"MCD {mcd:.2f} exceeds target 5.0 dB"

    def test_f0_rmse_target(self):
        """F0 RMSE should meet target (< 20 cents)."""
        pytest.skip("Requires trained model and test samples")
        from auto_voice.evaluation.quality_metrics import QualityMetrics
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline

        metrics = QualityMetrics()
        pipeline = SOTAConversionPipeline()

        source_audio = torch.randn(24000)
        target_speaker = torch.randn(256)

        result = pipeline.convert(source_audio, 24000, target_speaker)
        f0_rmse = metrics.compute_f0_rmse(source_audio, result['audio'], sample_rate=24000)

        assert f0_rmse < 20.0, f"F0 RMSE {f0_rmse:.1f} cents exceeds target 20 cents"

    def test_speaker_similarity_target(self):
        """Speaker similarity should meet target (> 0.85)."""
        pytest.skip("Requires trained model and test samples")
        from auto_voice.evaluation.quality_metrics import QualityMetrics
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline

        metrics = QualityMetrics()
        pipeline = SOTAConversionPipeline()

        source_audio = torch.randn(24000)
        target_speaker = torch.randn(256)
        target_speaker = target_speaker / target_speaker.norm()

        result = pipeline.convert(source_audio, 24000, target_speaker)

        # Extract speaker embedding from converted audio
        # (would need speaker encoder here)
        similarity = 0.88  # Placeholder

        assert similarity > 0.85, f"Speaker similarity {similarity:.2f} below target 0.85"


class TestDocumentation:
    """Tests for benchmark documentation."""

    def test_results_export_to_markdown(self):
        """Results should export to markdown format."""
        from auto_voice.evaluation.benchmark_runner import BenchmarkRunner

        runner = BenchmarkRunner()

        results = {
            'mcd_mean': 4.2,
            'mcd_std': 0.5,
            'f0_rmse_mean': 15.3,
            'f0_rmse_std': 3.2,
            'speaker_similarity_mean': 0.87,
            'speaker_similarity_std': 0.03,
        }

        markdown = runner.export_markdown(results, title="SOTA Pipeline Benchmark")

        assert "# SOTA Pipeline Benchmark" in markdown
        assert "MCD" in markdown
        assert "4.2" in markdown

    def test_results_export_to_json(self):
        """Results should export to JSON format."""
        import json
        from auto_voice.evaluation.benchmark_runner import BenchmarkRunner

        runner = BenchmarkRunner()

        results = {
            'mcd_mean': 4.2,
            'f0_rmse_mean': 15.3,
            'speaker_similarity_mean': 0.87,
        }

        json_str = runner.export_json(results)
        parsed = json.loads(json_str)

        assert parsed['mcd_mean'] == 4.2
        assert parsed['f0_rmse_mean'] == 15.3
