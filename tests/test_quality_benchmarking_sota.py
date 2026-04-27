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
import json

import pytest
import torch
import numpy as np
import soundfile as sf


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

    @pytest.fixture
    def benchmark_fixture_dir(self, tmp_path):
        data_dir = tmp_path / "benchmark-fixture"
        audio_dir = data_dir / "audio"
        speakers_dir = data_dir / "speakers"
        audio_dir.mkdir(parents=True)
        speakers_dir.mkdir(parents=True)

        sample_rate = 24000
        time_axis = np.linspace(0, 1.0, sample_rate, endpoint=False)
        source = 0.5 * np.sin(2 * np.pi * 220 * time_axis).astype(np.float32)
        reference = 0.5 * np.sin(2 * np.pi * 330 * time_axis).astype(np.float32)
        sf.write(audio_dir / "source.wav", source, sample_rate)
        sf.write(audio_dir / "reference.wav", reference, sample_rate)
        np.save(speakers_dir / "target.npy", np.ones(256, dtype=np.float32))

        manifest = {
            "dataset_name": "production-confidence-fixture",
            "defaults": {"sample_rate": sample_rate},
            "samples": [
                {
                    "sample_id": "fixture-001",
                    "split": "holdout",
                    "source_audio": "audio/source.wav",
                    "reference_audio": "audio/reference.wav",
                    "target_speaker_embedding": "speakers/target.npy",
                    "metadata": {"source_artist": "source", "target_artist": "target"},
                }
            ],
        }
        (data_dir / "metadata.json").write_text(json.dumps(manifest), encoding="utf-8")
        return data_dir

    def test_benchmark_dataset_class_exists(self):
        """BenchmarkDataset class should exist."""
        from auto_voice.evaluation.benchmark_dataset import BenchmarkDataset
        assert BenchmarkDataset is not None

    def test_benchmark_dataset_loads_samples(self, benchmark_fixture_dir):
        """Dataset should load benchmark samples."""
        from auto_voice.evaluation.benchmark_dataset import BenchmarkDataset

        dataset = BenchmarkDataset(data_dir=str(benchmark_fixture_dir), split="holdout")
        assert len(dataset) == 1

    def test_benchmark_sample_structure(self, benchmark_fixture_dir):
        """Each sample should have required fields."""
        from auto_voice.evaluation.benchmark_dataset import BenchmarkDataset

        dataset = BenchmarkDataset(data_dir=str(benchmark_fixture_dir), split="holdout")
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
        """Canonical benchmark evidence should enforce MCD target (< 5.0 dB)."""
        from auto_voice.evaluation.benchmark_reporting import build_benchmark_dashboard

        dashboard = build_benchmark_dashboard({
            "quality_seedvc": {"summary": {"sample_count": 1, "mcd_mean": 4.2}},
            "realtime": {"summary": {"sample_count": 1, "mcd_mean": 4.5}},
        })
        assert dashboard["pipelines"]["quality_seedvc"]["summary"]["mcd_mean"]["target_status"] == "pass"

    def test_f0_rmse_target(self):
        """Canonical benchmark evidence should enforce F0 target (< 20 cents)."""
        from auto_voice.evaluation.benchmark_reporting import build_benchmark_dashboard

        dashboard = build_benchmark_dashboard({
            "quality_seedvc": {"summary": {"sample_count": 1, "f0_rmse_mean": 15.0}},
            "realtime": {"summary": {"sample_count": 1, "f0_rmse_mean": 18.0}},
        })
        assert dashboard["pipelines"]["quality_seedvc"]["summary"]["f0_rmse_mean"]["target_status"] == "pass"

    def test_speaker_similarity_target(self):
        """Canonical benchmark evidence should enforce speaker similarity target (> 0.85)."""
        from auto_voice.evaluation.benchmark_reporting import build_benchmark_dashboard

        dashboard = build_benchmark_dashboard({
            "quality_seedvc": {"summary": {"sample_count": 1, "speaker_similarity_mean": 0.88}},
            "realtime": {"summary": {"sample_count": 1, "speaker_similarity_mean": 0.87}},
        })
        assert (
            dashboard["pipelines"]["quality_seedvc"]["summary"]["speaker_similarity_mean"]["target_status"]
            == "pass"
        )


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
