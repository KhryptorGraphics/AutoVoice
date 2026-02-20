"""Final targeted tests to reach 90% coverage for ConversionQualityAnalyzer.

Focuses on remaining uncovered lines without requiring transformers/problematic imports.
Target: Achieve 90%+ coverage (currently 86%).
"""

import numpy as np
import pytest
import soundfile as sf
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from auto_voice.evaluation.conversion_quality_analyzer import (
    ConversionQualityAnalyzer,
    QualityMetrics,
    ConversionAnalysis,
)


@pytest.fixture
def analyzer():
    """Create analyzer instance."""
    return ConversionQualityAnalyzer(device="cpu")


@pytest.fixture
def mock_audio_pair(tmp_path):
    """Create pair of audio files for testing."""
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    source_path = tmp_path / "source.wav"
    sf.write(str(source_path), audio, sr)

    converted_path = tmp_path / "converted.wav"
    sf.write(str(converted_path), audio, sr)

    return str(source_path), str(converted_path), sr


# ============================================================================
# Test Quality Score Edge Cases
# ============================================================================

def test_quality_score_with_negative_values():
    """Test quality score handles negative metric values correctly."""
    analyzer = ConversionQualityAnalyzer()

    metrics = QualityMetrics(
        speaker_similarity=0.0,  # Minimum
        mcd=15.0,  # Very high
        f0_correlation=-0.5,  # Negative
        snr=0.0,  # Minimum
        pesq=1.0,  # Minimum
        stoi=0.0,  # Minimum
    )

    score = analyzer._compute_quality_score(metrics)

    # Score should be non-negative even with negative inputs
    assert score >= 0
    # But should be very low
    assert score < 30


def test_quality_score_with_extreme_mcd():
    """Test quality score handles extreme MCD values."""
    analyzer = ConversionQualityAnalyzer()

    # Test very high MCD (>10 dB)
    metrics = QualityMetrics(
        speaker_similarity=0.8,
        mcd=25.0,  # Extremely high
        f0_correlation=0.9,
        snr=20.0,
        pesq=3.0,
        stoi=0.8,
    )

    score = analyzer._compute_quality_score(metrics)
    assert isinstance(score, float)
    # Should be penalized for high MCD
    assert score < 70


def test_quality_score_normalization():
    """Test that quality score stays in 0-100 range."""
    analyzer = ConversionQualityAnalyzer()

    # Test various extreme combinations
    test_cases = [
        # All zeros
        QualityMetrics(speaker_similarity=0, mcd=0, f0_correlation=0, snr=0, pesq=1.0, stoi=0),
        # All max
        QualityMetrics(speaker_similarity=1, mcd=0, f0_correlation=1, snr=50, pesq=4.5, stoi=1),
        # Mixed extreme
        QualityMetrics(speaker_similarity=1, mcd=20, f0_correlation=0, snr=100, pesq=1, stoi=1),
    ]

    for metrics in test_cases:
        score = analyzer._compute_quality_score(metrics)
        assert 0 <= score <= 100, f"Score {score} out of range for {metrics}"


# ============================================================================
# Test Analysis Edge Cases Without Real Audio Loading
# ============================================================================

def test_analyze_with_mocked_load_audio(analyzer, tmp_path):
    """Test full analyze flow with mocked audio loading."""
    # Create dummy paths
    source_path = tmp_path / "source.wav"
    converted_path = tmp_path / "converted.wav"

    # Mock audio loading and all metric computations
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    with patch.object(analyzer, "_load_audio", return_value=(audio, sr)), \
         patch.object(analyzer, "_compute_mcd", return_value=3.5), \
         patch.object(analyzer, "_compute_f0_metrics", return_value=(0.92, 15.0)), \
         patch.object(analyzer, "_compute_snr", return_value=22.0), \
         patch.object(analyzer, "_compute_pesq", return_value=3.6), \
         patch.object(analyzer, "_compute_stoi", return_value=0.87), \
         patch.object(analyzer, "_extract_speaker_embedding", return_value=np.random.randn(768)):

        embedding = np.random.randn(768).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        analysis = analyzer.analyze(
            source_audio=str(source_path),
            converted_audio=str(converted_path),
            target_speaker_embedding=embedding,
            methodology="test_method",
            processing_time_ms=1500.0,
        )

    assert isinstance(analysis, ConversionAnalysis)
    assert analysis.methodology == "test_method"
    assert analysis.metrics.mcd == 3.5
    assert analysis.metrics.f0_correlation == 0.92
    assert analysis.metrics.rtf > 0  # Should compute RTF


def test_threshold_checking_comprehensive(analyzer, tmp_path):
    """Test all threshold types are checked."""
    source_path = tmp_path / "source.wav"
    converted_path = tmp_path / "converted.wav"

    sr = 22050
    audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 2, int(sr * 2))).astype(np.float32)

    with patch.object(analyzer, "_load_audio", return_value=(audio, sr)), \
         patch.object(analyzer, "_compute_mcd", return_value=8.0), \
         patch.object(analyzer, "_compute_f0_metrics", return_value=(0.70, 35.0)), \
         patch.object(analyzer, "_compute_snr", return_value=15.0), \
         patch.object(analyzer, "_compute_pesq", return_value=2.5), \
         patch.object(analyzer, "_compute_stoi", return_value=0.70):

        # Very poor embedding similarity
        poor_embedding = np.zeros(768)

        analysis = analyzer.analyze(
            source_audio=str(source_path),
            converted_audio=str(converted_path),
            target_speaker_embedding=poor_embedding,
            methodology="realtime_test",
            processing_time_ms=2000.0,  # High RTF for realtime
        )

    # Should fail multiple thresholds
    assert not analysis.passes_thresholds
    assert len(analysis.threshold_failures) >= 3

    # Check specific failures
    failures_str = " ".join(analysis.threshold_failures)
    assert "Speaker similarity" in failures_str
    assert "MCD" in failures_str
    assert "F0 correlation" in failures_str


def test_recommendations_all_types(analyzer, tmp_path):
    """Test all recommendation types are generated."""
    source_path = tmp_path / "source.wav"
    converted_path = tmp_path / "converted.wav"

    sr = 22050
    audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 2, int(sr * 2))).astype(np.float32)

    with patch.object(analyzer, "_load_audio", return_value=(audio, sr)), \
         patch.object(analyzer, "_compute_mcd", return_value=6.0), \
         patch.object(analyzer, "_compute_f0_metrics", return_value=(0.75, 25.0)), \
         patch.object(analyzer, "_compute_snr", return_value=15.0):

        low_embedding = np.random.randn(768) * 0.1

        analysis = analyzer.analyze(
            source_audio=str(source_path),
            converted_audio=str(converted_path),
            target_speaker_embedding=low_embedding,
            methodology="test",
        )

    # Should have recommendations
    assert len(analysis.recommendations) >= 3

    recommendations = " ".join(analysis.recommendations).lower()
    assert any(word in recommendations for word in ["training", "epochs", "samples"])
    assert any(word in recommendations for word in ["vocoder", "decoder", "capacity"])
    assert any(word in recommendations for word in ["pitch", "f0", "accuracy"])
    assert any(word in recommendations for word in ["noise", "reduction", "quality"])


def test_compare_methodologies_empty_analyses(analyzer, tmp_path):
    """Test methodology comparison with no valid outputs."""
    source_path = tmp_path / "source.wav"

    sr = 22050
    audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 2, int(sr * 2))).astype(np.float32)
    sf.write(str(source_path), audio, sr)

    comparison = analyzer.compare_methodologies(
        source_audio=str(source_path),
        target_profile_id="test_profile",
        methodologies=["method_a", "method_b"],
        converted_outputs={},  # No outputs
    )

    assert comparison.best_methodology == "none"
    assert len(comparison.analyses) == 0
    assert "No methodologies analyzed" in comparison.summary


def test_compare_methodologies_voice_identifier_missing_profile(analyzer, tmp_path):
    """Test methodology comparison when profile not in voice identifier."""
    source_path = tmp_path / "source.wav"
    converted_path = tmp_path / "converted.wav"

    sr = 22050
    audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 2, int(sr * 2))).astype(np.float32)
    sf.write(str(source_path), audio, sr)
    sf.write(str(converted_path), audio, sr)

    # Mock voice identifier without the requested profile
    mock_identifier = MagicMock()
    mock_identifier._embeddings = {"other_profile": np.random.randn(768)}

    with patch("auto_voice.inference.voice_identifier.get_voice_identifier", return_value=mock_identifier):
        comparison = analyzer.compare_methodologies(
            source_audio=str(source_path),
            target_profile_id="missing_profile",  # Not in embeddings
            methodologies=["test_method"],
            converted_outputs={"test_method": str(converted_path)},
        )

    # Should complete without target embedding
    assert isinstance(comparison, MethodologyComparison)


def test_save_analysis_with_complex_data(analyzer, tmp_path):
    """Test saving analysis with all fields populated."""
    metrics = QualityMetrics(
        speaker_similarity=0.88,
        mcd=3.8,
        log_f0_rmse=0.12,
        f0_correlation=0.91,
        f0_rmse=18.5,
        rtf=0.28,
        processing_time_ms=1400.0,
        snr=24.5,
        pesq=3.7,
        stoi=0.86,
        artifact_score=0.15,
        quality_score=86.3,
    )

    analysis = ConversionAnalysis(
        methodology="comprehensive_test",
        source_audio="/path/to/source.wav",
        converted_audio="/path/to/converted.wav",
        target_profile_id="profile_abc123",
        metrics=metrics,
        timestamp="2026-02-02T14:30:00",
        passes_thresholds=True,
        threshold_failures=[],
        recommendations=["Good quality"],
    )

    output_path = tmp_path / "comprehensive_analysis.json"
    analyzer.save_analysis(analysis, str(output_path))

    # Verify file exists
    assert output_path.exists()

    # Load and verify all fields
    import json
    with open(output_path) as f:
        data = json.load(f)

    assert data["methodology"] == "comprehensive_test"
    assert data["target_profile_id"] == "profile_abc123"
    assert data["metrics"]["speaker_similarity"] == 0.88
    assert data["metrics"]["log_f0_rmse"] == 0.12
    assert data["metrics"]["artifact_score"] == 0.15
    assert data["passes_thresholds"] is True


def test_quality_metrics_all_fields():
    """Test QualityMetrics dataclass has all expected fields."""
    metrics = QualityMetrics(
        speaker_similarity=0.9,
        mcd=3.0,
        log_f0_rmse=0.1,
        f0_correlation=0.95,
        f0_rmse=10.0,
        rtf=0.25,
        processing_time_ms=1000.0,
        snr=25.0,
        pesq=3.8,
        stoi=0.88,
        artifact_score=0.1,
        quality_score=90.0,
    )

    # Convert to dict
    data = metrics.to_dict()

    # Verify all fields present
    expected_fields = [
        "speaker_similarity", "mcd", "log_f0_rmse", "f0_correlation",
        "f0_rmse", "rtf", "processing_time_ms", "snr", "pesq", "stoi",
        "artifact_score", "quality_score"
    ]

    for field in expected_fields:
        assert field in data, f"Missing field: {field}"


def test_methodology_comparison_dataclass():
    """Test MethodologyComparison has all expected fields."""
    from auto_voice.evaluation.conversion_quality_analyzer import MethodologyComparison

    metrics = QualityMetrics(quality_score=85.0)
    analysis = ConversionAnalysis(
        methodology="test",
        source_audio="source.wav",
        converted_audio="converted.wav",
        target_profile_id="profile",
        metrics=metrics,
        timestamp="2026-01-01",
        passes_thresholds=True,
        threshold_failures=[],
        recommendations=[],
    )

    comparison = MethodologyComparison(
        source_audio="source.wav",
        target_profile_id="profile",
        analyses={"test": analysis},
        best_methodology="test",
        rankings={"test": 1},
        summary="Test summary",
    )

    assert comparison.source_audio == "source.wav"
    assert comparison.target_profile_id == "profile"
    assert len(comparison.analyses) == 1
    assert comparison.best_methodology == "test"
    assert comparison.rankings["test"] == 1


def test_analyzer_constants():
    """Verify all analyzer constants are properly defined."""
    analyzer = ConversionQualityAnalyzer()

    # Check thresholds
    assert hasattr(analyzer, "SPEAKER_SIMILARITY_MIN")
    assert hasattr(analyzer, "MCD_MAX")
    assert hasattr(analyzer, "F0_CORRELATION_MIN")
    assert hasattr(analyzer, "F0_RMSE_MAX")
    assert hasattr(analyzer, "RTF_MAX_REALTIME")
    assert hasattr(analyzer, "SNR_MIN")
    assert hasattr(analyzer, "PESQ_MIN")
    assert hasattr(analyzer, "STOI_MIN")

    # Check weights
    assert hasattr(analyzer, "WEIGHTS")
    assert isinstance(analyzer.WEIGHTS, dict)
    assert "speaker_similarity" in analyzer.WEIGHTS
    assert "mcd" in analyzer.WEIGHTS
    assert "f0_correlation" in analyzer.WEIGHTS


# ============================================================================
# Import the MethodologyComparison for testing
# ============================================================================
from auto_voice.evaluation.conversion_quality_analyzer import MethodologyComparison


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
