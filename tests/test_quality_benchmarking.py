"""Tests for quality improvement metrics benchmarking.

Task 7.7: Benchmark quality improvement metrics (MOS, speaker similarity)

Tests cover:
- MOS (Mean Opinion Score) prediction
- Speaker similarity metrics
- Pitch accuracy metrics
- Quality benchmark runner
"""

import tempfile
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
    return torch.device("cuda")


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
    """Generate test audio (1 second)."""
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


# ============================================================================
# Test: MOS Prediction
# ============================================================================


@pytest.mark.cuda
class TestMOSPrediction:
    """Tests for Mean Opinion Score prediction."""

    def test_mos_predictor_creation(self, device):
        """MOS predictor should initialize on GPU."""
        from auto_voice.evaluation.quality_metrics import MOSPredictor

        predictor = MOSPredictor(device=str(device))
        assert predictor.device == str(device)

    def test_mos_returns_valid_score(self, device, sample_audio, sample_rate):
        """MOS should return score between 1 and 5."""
        from auto_voice.evaluation.quality_metrics import MOSPredictor

        predictor = MOSPredictor(device=str(device))
        mos = predictor.predict(sample_audio, sample_rate)

        assert 1.0 <= mos <= 5.0

    def test_mos_batch_prediction(self, device, sample_audio, sample_rate):
        """MOS should support batch prediction."""
        from auto_voice.evaluation.quality_metrics import MOSPredictor

        predictor = MOSPredictor(device=str(device))
        audios = [sample_audio, sample_audio * 0.5, sample_audio * 0.8]
        scores = predictor.predict_batch(audios, sample_rate)

        assert len(scores) == 3
        assert all(1.0 <= s <= 5.0 for s in scores)

    def test_mos_different_for_different_quality(self, device, sample_rate):
        """MOS should differentiate between quality levels."""
        from auto_voice.evaluation.quality_metrics import MOSPredictor

        predictor = MOSPredictor(device=str(device))

        # High quality: clean sine wave
        t = np.linspace(0, 1.0, sample_rate, endpoint=False)
        clean_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Low quality: noisy sine wave
        noisy_audio = clean_audio + np.random.randn(sample_rate).astype(np.float32) * 0.5

        clean_mos = predictor.predict(clean_audio, sample_rate)
        noisy_mos = predictor.predict(noisy_audio, sample_rate)

        # Clean should score higher than noisy
        assert clean_mos > noisy_mos


# ============================================================================
# Test: Speaker Similarity
# ============================================================================


@pytest.mark.cuda
class TestSpeakerSimilarity:
    """Tests for speaker similarity metrics."""

    def test_speaker_similarity_creation(self, device):
        """Speaker similarity metric should initialize on GPU."""
        from auto_voice.evaluation.quality_metrics import SpeakerSimilarity

        similarity = SpeakerSimilarity(device=str(device))
        assert similarity.device == str(device)

    def test_speaker_similarity_same_audio(self, device, sample_audio, sample_rate):
        """Same audio should have high similarity."""
        from auto_voice.evaluation.quality_metrics import SpeakerSimilarity

        similarity = SpeakerSimilarity(device=str(device))
        score = similarity.compute(sample_audio, sample_audio, sample_rate)

        # Self-similarity should be very high (close to 1.0)
        assert score > 0.95

    def test_speaker_similarity_different_audio(self, device, sample_rate):
        """Different audio should have lower similarity."""
        from auto_voice.evaluation.quality_metrics import SpeakerSimilarity

        t = np.linspace(0, 1.0, sample_rate, endpoint=False)
        audio1 = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 440 Hz
        audio2 = np.sin(2 * np.pi * 880 * t).astype(np.float32)  # 880 Hz

        similarity = SpeakerSimilarity(device=str(device))
        score = similarity.compute(audio1, audio2, sample_rate)

        # Different frequencies should have some similarity but less than same
        assert 0.0 <= score <= 1.0

    def test_extract_speaker_embedding(self, device, sample_audio, sample_rate):
        """Should extract speaker embedding from audio."""
        from auto_voice.evaluation.quality_metrics import SpeakerSimilarity

        similarity = SpeakerSimilarity(device=str(device))
        embedding = similarity.extract_embedding(sample_audio, sample_rate)

        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1
        assert len(embedding) > 0


# ============================================================================
# Test: Pitch Accuracy
# ============================================================================


@pytest.mark.cuda
class TestPitchAccuracy:
    """Tests for pitch accuracy metrics."""

    def test_pitch_accuracy_creation(self, device):
        """Pitch accuracy metric should initialize."""
        from auto_voice.evaluation.quality_metrics import PitchAccuracy

        accuracy = PitchAccuracy(device=str(device))
        assert accuracy is not None

    def test_pitch_rmse_same_audio(self, device, sample_audio, sample_rate):
        """Same audio should have zero pitch RMSE."""
        from auto_voice.evaluation.quality_metrics import PitchAccuracy

        accuracy = PitchAccuracy(device=str(device))
        rmse = accuracy.compute_rmse(sample_audio, sample_audio, sample_rate)

        # Self-comparison should have very low RMSE
        assert rmse < 1.0  # Less than 1 Hz error

    def test_pitch_rmse_different_audio(self, device, sample_rate):
        """Different pitches should have higher RMSE."""
        from auto_voice.evaluation.quality_metrics import PitchAccuracy

        t = np.linspace(0, 1.0, sample_rate, endpoint=False)
        audio_440 = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        audio_450 = np.sin(2 * np.pi * 450 * t).astype(np.float32)

        accuracy = PitchAccuracy(device=str(device))
        rmse = accuracy.compute_rmse(audio_440, audio_450, sample_rate)

        # 10 Hz difference should show up
        assert rmse > 5.0  # At least some pitch difference

    def test_pitch_correlation(self, device, sample_audio, sample_rate):
        """Pitch correlation should be high for same audio."""
        from auto_voice.evaluation.quality_metrics import PitchAccuracy

        accuracy = PitchAccuracy(device=str(device))
        correlation = accuracy.compute_correlation(sample_audio, sample_audio, sample_rate)

        # Self-correlation should be 1.0
        assert correlation > 0.99


# ============================================================================
# Test: Quality Benchmark Runner
# ============================================================================


@pytest.mark.cuda
class TestQualityBenchmarkRunner:
    """Tests for the quality benchmarking runner."""

    def test_benchmark_runner_creation(self, device, temp_storage):
        """Benchmark runner should initialize."""
        from auto_voice.evaluation.quality_metrics import QualityBenchmarkRunner

        runner = QualityBenchmarkRunner(
            device=str(device),
            output_dir=str(temp_storage),
        )
        assert runner.device == str(device)

    def test_benchmark_single_pair(self, device, temp_storage, sample_audio, sample_rate):
        """Benchmark should run on a single audio pair."""
        from auto_voice.evaluation.quality_metrics import QualityBenchmarkRunner

        runner = QualityBenchmarkRunner(
            device=str(device),
            output_dir=str(temp_storage),
        )

        results = runner.benchmark_pair(
            reference=sample_audio,
            converted=sample_audio * 0.9,
            sample_rate=sample_rate,
        )

        assert "mos" in results
        assert "speaker_similarity" in results
        assert "pitch_rmse" in results

    def test_benchmark_batch(self, device, temp_storage, sample_audio, sample_rate):
        """Benchmark should run on batch of audio pairs."""
        from auto_voice.evaluation.quality_metrics import QualityBenchmarkRunner

        runner = QualityBenchmarkRunner(
            device=str(device),
            output_dir=str(temp_storage),
        )

        pairs = [
            (sample_audio, sample_audio * 0.9),
            (sample_audio, sample_audio * 0.8),
            (sample_audio, sample_audio * 0.95),
        ]

        results = runner.benchmark_batch(pairs, sample_rate)

        assert len(results["results"]) == 3
        assert results["count"] == 3
        assert "mean_mos" in results
        assert "mean_similarity" in results

    def test_benchmark_generates_report(self, device, temp_storage, sample_audio, sample_rate):
        """Benchmark should generate a report file."""
        from auto_voice.evaluation.quality_metrics import QualityBenchmarkRunner

        runner = QualityBenchmarkRunner(
            device=str(device),
            output_dir=str(temp_storage),
        )

        runner.benchmark_pair(
            reference=sample_audio,
            converted=sample_audio,
            sample_rate=sample_rate,
        )

        report_path = temp_storage / "quality_report.json"
        runner.generate_report(str(report_path))

        assert report_path.exists()


# ============================================================================
# Test: Quality Comparison
# ============================================================================


@pytest.mark.cuda
class TestQualityComparison:
    """Tests for comparing quality between model versions."""

    def test_compare_model_quality(self, device, temp_storage, sample_audio, sample_rate):
        """Should compare quality between base and trained model outputs."""
        from auto_voice.evaluation.quality_metrics import compare_model_quality

        # Simulate different quality outputs
        base_output = sample_audio + np.random.randn(len(sample_audio)).astype(np.float32) * 0.1
        trained_output = sample_audio + np.random.randn(len(sample_audio)).astype(np.float32) * 0.05

        comparison = compare_model_quality(
            reference=sample_audio,
            base_output=base_output,
            trained_output=trained_output,
            sample_rate=sample_rate,
            device=str(device),
        )

        assert "base_metrics" in comparison
        assert "trained_metrics" in comparison
        assert "improvement" in comparison

    def test_quality_improvement_detected(self, device, sample_audio, sample_rate):
        """Should detect quality improvement."""
        from auto_voice.evaluation.quality_metrics import compare_model_quality

        # Base: more noise
        base_output = sample_audio + np.random.randn(len(sample_audio)).astype(np.float32) * 0.2
        # Trained: less noise (better)
        trained_output = sample_audio + np.random.randn(len(sample_audio)).astype(np.float32) * 0.02

        comparison = compare_model_quality(
            reference=sample_audio,
            base_output=base_output,
            trained_output=trained_output,
            sample_rate=sample_rate,
            device=str(device),
        )

        # MOS should improve (trained closer to reference)
        assert comparison["improvement"]["mos_delta"] >= 0


# ============================================================================
# Test: Aggregate Quality Stats
# ============================================================================


@pytest.mark.cuda
class TestAggregateQualityStats:
    """Tests for aggregating quality statistics."""

    def test_aggregate_metrics(self, device):
        """Should aggregate metrics from multiple evaluations."""
        from auto_voice.evaluation.quality_metrics import aggregate_quality_stats

        metrics_list = [
            {"mos": 3.5, "speaker_similarity": 0.85, "pitch_rmse": 10.0},
            {"mos": 4.0, "speaker_similarity": 0.90, "pitch_rmse": 8.0},
            {"mos": 3.8, "speaker_similarity": 0.88, "pitch_rmse": 9.0},
        ]

        stats = aggregate_quality_stats(metrics_list)

        assert "mean_mos" in stats
        assert "std_mos" in stats
        assert "mean_speaker_similarity" in stats
        assert abs(stats["mean_mos"] - 3.77) < 0.1  # Average of 3.5, 4.0, 3.8

    def test_percentile_calculation(self, device):
        """Should calculate quality percentiles."""
        from auto_voice.evaluation.quality_metrics import aggregate_quality_stats

        metrics_list = [
            {"mos": m, "speaker_similarity": 0.9, "pitch_rmse": 10.0}
            for m in np.linspace(2.0, 5.0, 100)
        ]

        stats = aggregate_quality_stats(metrics_list)

        assert "p50_mos" in stats
        assert "p95_mos" in stats
        assert stats["p50_mos"] < stats["p95_mos"]
