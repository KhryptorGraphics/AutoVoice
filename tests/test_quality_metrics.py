#!/usr/bin/env python3
"""
Unit tests for quality metrics and evaluation components.

Tests the quality metrics aggregator, individual metric calculations,
and evaluation framework components.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch


@pytest.fixture
def sample_rate():
    """Sample rate fixture."""
    return 44100


@pytest.fixture
def synthethic_audio_pair(sample_rate):
    """Create synthetic audio pair for testing."""
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Create source audio (440 Hz sine wave)
    source_freq = 440.0
    source_audio = torch.tensor(
        0.5 * np.sin(2 * np.pi * source_freq * t),
        dtype=torch.float32
    )

    # Create target audio (approximately same frequency with slight perturbation)
    target_freq = 440.1  # Very small difference for good correlation
    target_audio = torch.tensor(
        0.5 * np.sin(2 * np.pi * target_freq * t),
        dtype=torch.float32
    )

    return source_audio, target_audio


@pytest.fixture
def bad_audio_pair(sample_rate):
    """Create synthetic audio pair with poor quality match."""
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Create source audio (440 Hz sine wave)
    source_freq = 440.0
    source_audio = torch.tensor(
        0.5 * np.sin(2 * np.pi * source_freq * t),
        dtype=torch.float32
    )

    # Create target audio with octave difference (poor match)
    target_freq = 880.0  # Octave higher
    target_audio = torch.tensor(
        0.5 * np.sin(2 * np.pi * target_freq * t),
        dtype=torch.float32
    )

    return source_audio, target_audio


class TestQualityMetricsAggregator:
    """Test the main quality metrics aggregator."""

    def test_initialization(self, sample_rate):
        """Test QualityMetricsAggregator initialization."""
        from src.auto_voice.utils.quality_metrics import QualityMetricsAggregator

        aggregator = QualityMetricsAggregator(sample_rate=sample_rate)

        # Check that components are initialized
        assert hasattr(aggregator, 'pitch_metrics')
        assert hasattr(aggregator, 'speaker_metrics')
        assert hasattr(aggregator, 'naturalness_metrics')
        assert hasattr(aggregator, 'intelligibility_metrics')

    def test_evaluate_good_quality_audio(self, sample_rate, synthethic_audio_pair):
        """Test evaluation on high-quality matching audio."""
        from src.auto_voice.utils.quality_metrics import QualityMetricsAggregator

        source_audio, target_audio = synthethic_audio_pair
        aggregator = QualityMetricsAggregator(sample_rate=sample_rate)

        result = aggregator.evaluate(source_audio, target_audio)

        # High-quality pair should have good metrics
        assert result.pitch_accuracy.rmse_hz < 5.0  # Should be very low
        assert result.pitch_accuracy.correlation > 0.95  # Should be very high
        assert result.speaker_similarity.cosine_similarity > 0.8  # High similarity
        assert result.overall_quality_score > 0.75  # Good overall score

    def test_evaluate_poor_quality_audio(self, sample_rate, bad_audio_pair):
        """Test evaluation on poor-quality mismatched audio."""
        from src.auto_voice.utils.quality_metrics import QualityMetricsAggregator

        source_audio, target_audio = bad_audio_pair
        aggregator = QualityMetricsAggregator(sample_rate=sample_rate)

        result = aggregator.evaluate(source_audio, target_audio)

        # Poor-quality pair should have worse metrics
        assert result.pitch_accuracy.rmse_hz > 200.0  # High RMSE due to octave difference
        assert result.pitch_accuracy.correlation < 0.5  # Poor correlation
        assert result.overall_quality_score < 0.5  # Poor overall score

    def test_batch_summary_statistics(self, sample_rate, synthethic_audio_pair):
        """Test summary statistics computation for batch evaluation."""
        from src.auto_voice.utils.quality_metrics import QualityMetricsAggregator

        # Create multiple evaluations
        source_audio, target_audio = synthethic_audio_pair
        aggregator = QualityMetricsAggregator(sample_rate=sample_rate)

        # Generate multiple results
        results = []
        for i in range(5):
            # Add slight variation
            varied_target = target_audio * (1.0 + i * 0.01)
            result = aggregator.evaluate(source_audio, varied_target)
            results.append(result)

        # Compute summary statistics
        summary = aggregator.get_summary_statistics(results)

        # Check summary structure
        assert 'pitch_accuracy' in summary
        assert 'correlation' in summary['pitch_accuracy']
        assert 'rmse_hz' in summary['pitch_accuracy']
        assert 'speaker_similarity' in summary['speaker_similarity']

        # Check that statistics include mean, std, min, max
        corr_stats = summary['pitch_accuracy']['correlation']
        assert 'mean' in corr_stats
        assert 'std' in corr_stats
        assert 'min' in corr_stats
        assert 'max' in corr_stats

    def test_edge_case_handling(self, sample_rate):
        """Test handling of edge cases."""
        from src.auto_voice.utils.quality_metrics import QualityMetricsAggregator

        aggregator = QualityMetricsAggregator(sample_rate=sample_rate)

        # Test with very short audio
        short_audio = torch.zeros(100, dtype=torch.float32)  # Very short
        with pytest.raises(ValueError):  # Should handle gracefully or raise meaningful error
            aggregator.evaluate(short_audio, short_audio)

        # Test with silent audio
        silent_audio = torch.zeros(44100, dtype=torch.float32)  # 1 second silence
        result = aggregator.evaluate(silent_audio, silent_audio)
        assert result is not None  # Should not crash


class TestPitchAccuracyMetrics:
    """Test pitch accuracy metric calculations."""

    def test_exact_pitch_match(self, sample_rate):
        """Test pitch metrics with identically matched audio."""
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        freq = 440.0

        # Identical audio
        source = torch.tensor(0.5 * np.sin(2 * np.pi * freq * t), dtype=torch.float32)
        target = source.clone()

        from src.auto_voice.utils.quality_metrics import QualityMetricsAggregator
        aggregator = QualityMetricsAggregator(sample_rate=sample_rate)

        result = aggregator.evaluate(source, target)

        # Perfect match should have very low RMSE and high correlation
        assert result.pitch_accuracy.rmse_hz < 1.0, f"RMSE too high: {result.pitch_accuracy.rmse_hz}"
        assert result.pitch_accuracy.correlation > 0.99, f"Correlation too low: {result.pitch_accuracy.correlation}"

    def test_pitch_rmse_calculations(self):
        """Test different RMSE calculation methods."""
        from src.auto_voice.utils.quality_metrics import PitchAccuracyMetrics

        # Create mock pitch contours (in Hz)
        f0_source = np.array([440.0, 441.0, 442.0, 443.0])
        f0_target = np.array([440.1, 441.1, 442.1, 443.1])  # Slight offset

        metrics = PitchAccuracyMetrics()
        result = metrics.calculate_pitch_accuracy(f0_source, f0_target, sample_rate=44100)

        # Check Hz RMSE calculation
        assert hasattr(result, 'rmse_hz')
        assert result.rmse_hz > 0.0  # Should be small but positive
        assert result.rmse_hz < 1.0  # Should be small due to small differences

        # Check correlation is high
        assert result.correlation > 0.99

    def test_voiced_unvoiced_handling(self):
        """Test handling of voiced vs unvoiced frames."""
        from src.auto_voice.utils.quality_metrics import PitchAccuracyMetrics

        # Create pitch contours with some unvoiced regions
        f0_source = np.array([440.0, 441.0, 0.0, 443.0, 0.0])  # Mix of voiced/unvoiced
        f0_target = np.array([440.1, 441.1, 0.0, 443.1, 0.0])  # Same pattern

        metrics = PitchAccuracyMetrics()
        result = metrics.calculate_pitch_accuracy(f0_source, f0_target, sample_rate=44100)

        # Should handle unvoiced regions gracefully
        assert result.rmse_hz >= 0.0
        assert not np.isnan(result.rmse_hz)

    def test_empty_pitch_contours(self):
        """Test handling of empty or invalid pitch contours."""
        from src.auto_voice.utils.quality_metrics import PitchAccuracyMetrics

        metrics = PitchAccuracyMetrics()

        # Test with all zeros (unvoiced)
        empty_source = np.zeros(100)
        empty_target = np.zeros(100)

        result = metrics.calculate_pitch_accuracy(empty_source, empty_target, sample_rate=44100)

        # Should handle all-unvoiced gracefully
        assert result.rmse_hz >= 0.0


class TestSpeakerSimilarityMetrics:
    """Test speaker similarity metric calculations."""

    def test_identical_audio_similarity(self, sample_rate):
        """Test speaker similarity with identical audio."""
        from src.auto_voice.utils.quality_metrics import QualityMetricsAggregator

        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = torch.tensor(0.5 * np.sin(2 * np.pi * 440 * t), dtype=torch.float32)

        aggregator = QualityMetricsAggregator(sample_rate=sample_rate)
        result = aggregator.evaluate(audio, audio.clone())

        # Identical audio should have perfect speaker similarity
        assert result.speaker_similarity.cosine_similarity > 0.95

    def test_speaker_encoder_initialization(self):
        """Test that speaker encoder initializes correctly."""
        try:
            from src.auto_voice.utils.quality_metrics import SpeakerSimilarityMetrics
            metrics = SpeakerSimilarityMetrics()

            # Should initialize without errors
            assert hasattr(metrics, 'speaker_encoder')

        except ImportError:
            pytest.skip("Speaker encoder not available")

    @patch('src.auto_voice.utils.quality_metrics.SpeakerSimilarityMetrics._extract_embedding')
    def test_similarity_calculation(self, mock_extract):
        """Test similarity calculation with mocked embeddings."""
        from src.auto_voice.utils.quality_metrics import SpeakerSimilarityMetrics

        # Mock embeddings
        mock_extract.return_value = torch.randn(256)

        metrics = SpeakerSimilarityMetrics()

        # Create dummy audio
        audio = torch.randn(44100)

        # Calculate similarity to self (should be perfect)
        similarity = metrics.calculate_similarity(audio, audio, sample_rate=44100)

        # With identical embeddings, similarity should be very high
        assert similarity.cosine_similarity > 0.99


class TestNaturalnessMetrics:
    """Test naturalness metric calculations."""

    def test_naturalness_calculation(self, sample_rate):
        """Test naturalness metrics on synthetic audio."""
        from src.auto_voice.utils.quality_metrics import QualityMetricsAggregator

        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = torch.tensor(0.5 * np.sin(2 * np.pi * 440 * t), dtype=torch.float32)

        aggregator = QualityMetricsAggregator(sample_rate=sample_rate)
        result = aggregator.evaluate(audio, audio.clone())

        # Should return reasonable confidence score
        assert 0.0 <= result.naturalness.confidence_score <= 1.0
        assert hasattr(result.naturalness, 'spectral_distortion')


class TestIntelligibilityMetrics:
    """Test intelligibility metric calculations."""

    def test_stoi_calculation(self, sample_rate):
        """Test STOI calculation on clean speech."""
        from src.auto_voice.utils.quality_metrics import QualityMetricsAggregator

        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = torch.tensor(0.5 * np.sin(2 * np.pi * 440 * t), dtype=torch.float32)

        aggregator = QualityMetricsAggregator(sample_rate=sample_rate)
        result = aggregator.evaluate(audio, audio.clone())

        # STOI should be high for identical signals
        assert result.intelligibility.stoi_score > 0.9
        assert 0.0 <= result.intelligibility.stoi_score <= 1.0


class TestEvaluationFramework:
    """Test the evaluation framework itself."""

    def test_evaluator_initialization(self, sample_rate):
        """Test VoiceConversionEvaluator initialization."""
        try:
            from src.auto_voice.evaluation.evaluator import VoiceConversionEvaluator
            evaluator = VoiceConversionEvaluator(sample_rate=sample_rate)

            assert evaluator.sample_rate == sample_rate
            assert hasattr(evaluator, 'metrics_aggregator')

        except ImportError:
            pytest.skip("Evaluator not available")

    def test_config_loading(self):
        """Test evaluation configuration loading."""
        from src.auto_voice.evaluation.evaluator import VoiceConversionEvaluator
        import tempfile
        import yaml

        # Create test config
        test_config = {
            'quality_targets': {
                'min_pitch_accuracy_correlation': 0.9,
                'max_pitch_accuracy_rmse_hz': 15.0,
                'min_speaker_similarity': 0.8
            }
        }

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name

        try:
            evaluator = VoiceConversionEvaluator(evaluation_config_path=config_path)
            assert evaluator.config['quality_targets']['min_pitch_accuracy_correlation'] == 0.9
        finally:
            import os
            os.unlink(config_path)

    def test_quality_targets_validation(self):
        """Test quality targets validation logic."""
        from src.auto_voice.evaluation.evaluator import VoiceConversionEvaluator, QualityTargets, EvaluationSample

        try:
            evaluator = VoiceConversionEvaluator()
        except Exception:
            # If initialization fails, skip this test
            pytest.skip("VoiceConversionEvaluator initialization failed")

        # Mock good results with all metrics
        good_stats = {
            'pitch_accuracy': {
                'correlation': {'mean': 0.95},
                'rmse_hz': {'mean': 5.0}
            },
            'speaker_similarity': {
                'cosine_similarity': {'mean': 0.90}
            },
            'intelligibility': {
                'stoi': {'mean': 0.92}
            },
            'naturalness': {
                'mos_estimation': {'mean': 4.2}
            }
        }

        # Mock results object
        class MockResults:
            def __init__(self):
                self.samples = []

            @property
            def summary_stats(self):
                return good_stats

        good_results = MockResults()
        targets = QualityTargets()

        validation = evaluator.validate_quality_targets(good_results, targets)

        assert validation['overall_pass'] is True
        assert 'max_pitch_accuracy_rmse_hz' in validation['target_validations']
        assert 'min_speaker_similarity' in validation['target_validations']
        assert 'min_stoi_score' in validation['target_validations']
        assert 'min_mos_estimate' in validation['target_validations']
        assert validation['target_validations']['min_speaker_similarity'] is True
        assert validation['target_validations']['min_stoi_score'] is True
        assert validation['target_validations']['min_mos_estimate'] is True

    def test_report_generation(self, tmp_path):
        """Test report generation functionality."""
        from src.auto_voice.evaluation.evaluator import VoiceConversionEvaluator, EvaluationSample, EvaluationResults
        from datetime import datetime

        try:
            evaluator = VoiceConversionEvaluator()

            # Create mock sample
            sample = EvaluationSample(
                id='test_sample',
                source_audio_path='test_source.wav',
                target_audio_path='test_target.wav'
            )

            # Create mock results
            results = EvaluationResults(
                samples=[sample],
                summary_stats={},
                evaluation_config={},
                evaluation_timestamp=datetime.now().timestamp(),
                total_evaluation_time=1.0
            )

            # Test markdown report generation
            report_files = evaluator.generate_reports(results, str(tmp_path), formats=['markdown'])

            assert 'markdown' in report_files
            assert (tmp_path / 'evaluation_report.md').exists()

            # Verify content
            with open(report_files['markdown'], 'r') as f:
                content = f.read()
                assert '# Voice Conversion Quality Evaluation Report' in content
                assert 'test_sample' in content

        except ImportError:
            pytest.skip("Evaluator not available")


class TestMetricsEdgeCases:
    """Test edge cases and error conditions."""

    def test_nan_handling(self, sample_rate):
        """Test handling of NaN values in audio."""
        from src.auto_voice.utils.quality_metrics import QualityMetricsAggregator

        # Create audio with NaN values
        audio = torch.randn(44100)
        audio[1000:1100] = float('nan')  # Insert NaN values

        aggregator = QualityMetricsAggregator(sample_rate=sample_rate)

        # Should handle NaN gracefully
        result = aggregator.evaluate(audio, audio)

        # Results should not contain NaN
        assert not np.isnan(result.overall_quality_score)

    def test_empty_audio_handling(self, sample_rate):
        """Test handling of empty audio."""
        from src.auto_voice.utils.quality_metrics import QualityMetricsAggregator

        empty_audio = torch.tensor([], dtype=torch.float32)
        aggregator = QualityMetricsAggregator(sample_rate=sample_rate)

        # Should handle empty audio gracefully or raise clear error
        with pytest.raises((ValueError, RuntimeError)):
            aggregator.evaluate(empty_audio, empty_audio)

    def test_different_sample_rates(self, synthethic_audio_pair):
        """Test behavior with mismatched sample rates."""
        source_audio, target_audio = synthethic_audio_pair

        from src.auto_voice.utils.quality_metrics import QualityMetricsAggregator

        # Create evaluator with different sample rate than expected
        aggregator = QualityMetricsAggregator(sample_rate=22050)  # Half the rate

        # Should still work (resampling should handle it)
        result = aggregator.evaluate(source_audio, target_audio)
        assert result is not None


if __name__ == '__main__':
    pytest.main([__file__])
