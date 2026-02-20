"""Enhanced comprehensive tests for ConversionQualityAnalyzer to achieve 90% coverage.

This test file adds additional coverage for:
1. Speaker embedding extraction with real model loading paths
2. Methodology comparison with voice identifier integration
3. Additional edge cases and error paths
4. Log F0 RMSE calculations
5. Artifact detection and scoring

Target Coverage: 90%+ (from current 84%)
Missing Lines to Cover: 171-178, 182-184, 190-198, 216-230, 250-269, 295-296, 347-352, 521, 529-531
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import numpy as np
import pytest
import soundfile as sf
import torch

from auto_voice.evaluation.conversion_quality_analyzer import (
    ConversionQualityAnalyzer,
    QualityMetrics,
    ConversionAnalysis,
    MethodologyComparison,
    analyze_conversion,
)


# ============================================================================
# Additional Fixtures
# ============================================================================

@pytest.fixture
def real_audio_16khz(tmp_path):
    """Create 16kHz audio for embedding extraction."""
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    path = tmp_path / "audio_16k.wav"
    sf.write(str(path), audio, sr)
    return str(path), audio, sr


@pytest.fixture
def real_audio_22khz(tmp_path):
    """Create 22kHz audio for resampling tests."""
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    path = tmp_path / "audio_22k.wav"
    sf.write(str(path), audio, sr)
    return str(path), audio, sr


# ============================================================================
# Test Speaker Embedding Extraction (Lines 171-198)
# ============================================================================

def test_extract_speaker_embedding_model_loading():
    """Test speaker embedding extraction with model loading."""
    analyzer = ConversionQualityAnalyzer(device="cpu")
    sr = 16000
    duration = 2.0
    audio = np.random.randn(int(sr * duration)).astype(np.float32)

    # Mock the transformers imports and model
    with patch("transformers.Wav2Vec2FeatureExtractor") as MockProcessor, \
         patch("transformers.WavLMModel") as MockModel:

        # Setup processor mock
        mock_processor = MagicMock()
        MockProcessor.from_pretrained.return_value = mock_processor
        mock_input_values = torch.randn(1, int(sr * duration))
        mock_processor.return_value = MagicMock(input_values=mock_input_values)

        # Setup model mock
        mock_model = MagicMock()
        MockModel.from_pretrained.return_value = mock_model

        # Mock parameters() to return proper iterator
        mock_param = torch.tensor([0.0])
        mock_model.parameters = MagicMock(return_value=iter([mock_param]))

        # Mock model output
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 100, 768)
        mock_model.return_value = mock_output

        # Extract embedding
        embedding = analyzer._extract_speaker_embedding(audio, sr)

        # Verify model was loaded
        MockProcessor.from_pretrained.assert_called_once_with("microsoft/wavlm-base-plus")
        MockModel.from_pretrained.assert_called_once_with("microsoft/wavlm-base-plus")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == 768
        # Should be normalized
        norm = np.linalg.norm(embedding)
        assert 0.9 < norm < 1.1


def test_extract_speaker_embedding_cuda_device():
    """Test speaker embedding extraction with CUDA device."""
    analyzer = ConversionQualityAnalyzer(device="cuda")
    sr = 16000
    audio = np.random.randn(int(sr * 2)).astype(np.float32)

    with patch("transformers.Wav2Vec2FeatureExtractor") as MockProcessor, \
         patch("transformers.WavLMModel") as MockModel, \
         patch("torch.cuda.is_available", return_value=True):

        mock_processor = MagicMock()
        MockProcessor.from_pretrained.return_value = mock_processor
        mock_input_values = torch.randn(1, int(sr * 2))
        mock_processor.return_value = MagicMock(input_values=mock_input_values)

        mock_model = MagicMock()
        MockModel.from_pretrained.return_value = mock_model
        mock_param = torch.tensor([0.0])
        mock_model.parameters = MagicMock(return_value=iter([mock_param]))
        mock_model.cuda = MagicMock(return_value=mock_model)

        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 100, 768)
        mock_model.return_value = mock_output

        embedding = analyzer._extract_speaker_embedding(audio, sr)

        # Verify CUDA was called
        mock_model.cuda.assert_called_once()
        assert isinstance(embedding, np.ndarray)


def test_extract_speaker_embedding_resampling_path():
    """Test speaker embedding extraction triggers resampling for non-16kHz audio."""
    analyzer = ConversionQualityAnalyzer(device="cpu")
    sr = 22050  # Non-standard rate
    duration = 2.0
    audio = np.random.randn(int(sr * duration)).astype(np.float32)

    with patch("transformers.Wav2Vec2FeatureExtractor") as MockProcessor, \
         patch("transformers.WavLMModel") as MockModel, \
         patch("torchaudio.transforms.Resample") as MockResample:

        mock_processor = MagicMock()
        MockProcessor.from_pretrained.return_value = mock_processor
        mock_input_values = torch.randn(1, int(16000 * duration))
        mock_processor.return_value = MagicMock(input_values=mock_input_values)

        mock_model = MagicMock()
        MockModel.from_pretrained.return_value = mock_model
        mock_param = torch.tensor([0.0])
        mock_model.parameters = MagicMock(return_value=iter([mock_param]))

        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 100, 768)
        mock_model.return_value = mock_output

        # Mock resampler
        mock_resampler = MagicMock()
        MockResample.return_value = mock_resampler
        mock_resampler.return_value = torch.randn(int(16000 * duration))

        embedding = analyzer._extract_speaker_embedding(audio, sr)

        # Verify resampling was called
        MockResample.assert_called_once_with(sr, 16000)
        mock_resampler.assert_called_once()
        assert isinstance(embedding, np.ndarray)


def test_extract_speaker_embedding_model_cached():
    """Test that speaker model is cached after first use."""
    analyzer = ConversionQualityAnalyzer(device="cpu")
    sr = 16000
    audio = np.random.randn(int(sr * 2)).astype(np.float32)

    with patch("transformers.Wav2Vec2FeatureExtractor") as MockProcessor, \
         patch("transformers.WavLMModel") as MockModel:

        mock_processor = MagicMock()
        MockProcessor.from_pretrained.return_value = mock_processor
        mock_input_values = torch.randn(1, int(sr * 2))
        mock_processor.return_value = MagicMock(input_values=mock_input_values)

        mock_model = MagicMock()
        MockModel.from_pretrained.return_value = mock_model
        mock_param = torch.tensor([0.0])
        mock_model.parameters = MagicMock(return_value=iter([mock_param]))

        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 100, 768)
        mock_model.return_value = mock_output

        # First call - should load model
        embedding1 = analyzer._extract_speaker_embedding(audio, sr)

        # Second call - should use cached model
        mock_model.parameters = MagicMock(return_value=iter([mock_param]))  # Reset iterator
        embedding2 = analyzer._extract_speaker_embedding(audio, sr)

        # Model should be loaded only once
        assert MockModel.from_pretrained.call_count == 1
        assert analyzer._speaker_model is not None


# ============================================================================
# Test MCD Computation (Lines 216-230)
# ============================================================================

def test_compute_mcd_with_librosa():
    """Test MCD computation using librosa."""
    analyzer = ConversionQualityAnalyzer(device="cpu")
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))

    # Create similar but not identical signals
    source = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    # Add small noise to make it different
    converted = source + 0.01 * np.random.randn(len(source)).astype(np.float32)

    with patch("librosa.feature.mfcc") as mock_mfcc:
        # Return realistic MFCC features
        mock_mfcc.side_effect = [
            np.random.randn(13, 100).astype(np.float32),  # Source MFCCs
            np.random.randn(13, 100).astype(np.float32),  # Converted MFCCs
        ]

        mcd = analyzer._compute_mcd(source, converted, sr)

        # Verify mfcc was called twice
        assert mock_mfcc.call_count == 2
        assert isinstance(mcd, float)
        assert mcd > 0


def test_compute_mcd_length_alignment():
    """Test MCD handles different length MFCCs."""
    analyzer = ConversionQualityAnalyzer(device="cpu")
    sr = 16000
    source = np.random.randn(int(sr * 2)).astype(np.float32)
    converted = np.random.randn(int(sr * 3)).astype(np.float32)

    with patch("librosa.feature.mfcc") as mock_mfcc:
        # Return different lengths
        mock_mfcc.side_effect = [
            np.random.randn(13, 100).astype(np.float32),  # Source: 100 frames
            np.random.randn(13, 150).astype(np.float32),  # Converted: 150 frames
        ]

        mcd = analyzer._compute_mcd(source, converted, sr)

        assert isinstance(mcd, float)
        # Should align to shorter length internally


# ============================================================================
# Test F0 Metrics (Lines 250-269)
# ============================================================================

def test_compute_f0_metrics_with_librosa():
    """Test F0 metrics using librosa.pyin."""
    analyzer = ConversionQualityAnalyzer(device="cpu")
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))

    source = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    converted = 0.5 * np.sin(2 * np.pi * 442 * t).astype(np.float32)

    with patch("librosa.pyin") as mock_pyin:
        # Return realistic F0 contours
        f0_source = np.full(200, 440.0)  # Constant 440Hz
        f0_converted = np.full(200, 442.0)  # Constant 442Hz

        mock_pyin.side_effect = [
            (f0_source, None, None),
            (f0_converted, None, None),
        ]

        correlation, rmse = analyzer._compute_f0_metrics(source, converted, sr)

        # Should have very high correlation (both constant)
        assert correlation > 0.9
        # RMSE should be ~2Hz
        assert 1.0 < rmse < 3.0


def test_compute_f0_metrics_with_nan_values():
    """Test F0 metrics handling of NaN values."""
    analyzer = ConversionQualityAnalyzer(device="cpu")
    sr = 16000
    audio = np.random.randn(int(sr * 2)).astype(np.float32)

    with patch("librosa.pyin") as mock_pyin:
        # Return F0 with many NaN values
        f0_with_nans = np.full(100, np.nan)
        f0_with_nans[:20] = 440.0  # Only 20 valid frames

        mock_pyin.side_effect = [
            (f0_with_nans.copy(), None, None),
            (f0_with_nans.copy(), None, None),
        ]

        correlation, rmse = analyzer._compute_f0_metrics(audio, audio, sr)

        # Should handle NaN values and compute on valid frames
        assert isinstance(correlation, float)
        assert isinstance(rmse, float)


def test_compute_f0_metrics_insufficient_valid_frames():
    """Test F0 metrics with too few valid frames."""
    analyzer = ConversionQualityAnalyzer(device="cpu")
    sr = 16000
    audio = np.random.randn(int(sr * 2)).astype(np.float32)

    with patch("librosa.pyin") as mock_pyin:
        # Return F0 with < 10 valid frames
        f0_mostly_nan = np.full(100, np.nan)
        f0_mostly_nan[:5] = 440.0  # Only 5 valid frames

        mock_pyin.side_effect = [
            (f0_mostly_nan.copy(), None, None),
            (f0_mostly_nan.copy(), None, None),
        ]

        correlation, rmse = analyzer._compute_f0_metrics(audio, audio, sr)

        # Should return defaults for insufficient data
        assert correlation == 0.0
        assert rmse == 100.0


# ============================================================================
# Test SNR Computation (Lines 295-296)
# ============================================================================

def test_compute_snr_zero_division_handling():
    """Test SNR handles zero power gracefully."""
    analyzer = ConversionQualityAnalyzer(device="cpu")

    # Very quiet audio
    audio = np.zeros(16000, dtype=np.float32)
    audio[1000] = 0.001  # Single tiny spike

    snr = analyzer._compute_snr(audio)

    # Should not crash and return a value
    assert isinstance(snr, float)
    assert not np.isnan(snr)
    assert not np.isinf(snr)


def test_compute_snr_realistic_signal():
    """Test SNR with realistic signal/noise ratio."""
    analyzer = ConversionQualityAnalyzer(device="cpu")
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))

    # Signal with known SNR
    signal = 1.0 * np.sin(2 * np.pi * 440 * t)  # Strong signal
    noise = 0.1 * np.random.randn(len(signal))   # Weak noise
    audio = (signal + noise).astype(np.float32)

    snr = analyzer._compute_snr(audio)

    # Should be positive and reasonable
    assert snr > 10  # At least 10 dB for this signal/noise ratio


# ============================================================================
# Test PESQ/STOI Error Paths (Lines 347-352)
# ============================================================================

def test_compute_stoi_import_not_available():
    """Test STOI when pystoi module is not available."""
    analyzer = ConversionQualityAnalyzer(device="cpu")
    sr = 16000
    audio = np.random.randn(int(sr * 2)).astype(np.float32) * 0.1

    # Simulate ImportError
    with patch.dict("sys.modules", {"pystoi": None}):
        # Force re-import attempt
        score = analyzer._compute_stoi(audio, audio, sr)

    # Should return None when module unavailable
    assert score is None


def test_compute_stoi_computation_exception():
    """Test STOI handles computation exceptions."""
    analyzer = ConversionQualityAnalyzer(device="cpu")
    sr = 16000
    audio = np.random.randn(int(sr * 2)).astype(np.float32) * 0.1

    # Create a mock that's importable but raises on call
    mock_stoi_func = MagicMock(side_effect=ValueError("STOI computation failed"))

    with patch.dict("sys.modules", {"pystoi": MagicMock(stoi=mock_stoi_func)}):
        with patch("pystoi.stoi", mock_stoi_func):
            score = analyzer._compute_stoi(audio, audio, sr)

    assert score is None


# ============================================================================
# Test Methodology Comparison (Lines 521, 529-531)
# ============================================================================

def test_compare_methodologies_with_voice_identifier():
    """Test methodology comparison loading target embedding from voice identifier."""
    analyzer = ConversionQualityAnalyzer(device="cpu")

    # Create mock audio files
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create test audio
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        source_path = tmp_path / "source.wav"
        sf.write(str(source_path), audio, sr)

        converted_path = tmp_path / "converted.wav"
        sf.write(str(converted_path), audio, sr)

        converted_outputs = {"test_method": str(converted_path)}

        # Mock voice identifier with embedding
        mock_identifier = MagicMock()
        mock_identifier._embeddings = {
            "test_profile": np.random.randn(768).astype(np.float32)
        }

        with patch("auto_voice.evaluation.conversion_quality_analyzer.get_voice_identifier",
                   return_value=mock_identifier):

            comparison = analyzer.compare_methodologies(
                source_audio=str(source_path),
                target_profile_id="test_profile",
                methodologies=["test_method"],
                converted_outputs=converted_outputs,
            )

        assert isinstance(comparison, MethodologyComparison)
        assert len(comparison.analyses) == 1
        assert "test_method" in comparison.analyses


def test_compare_methodologies_identifier_error():
    """Test methodology comparison when voice identifier fails to load."""
    analyzer = ConversionQualityAnalyzer(device="cpu")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        source_path = tmp_path / "source.wav"
        sf.write(str(source_path), audio, sr)

        converted_path = tmp_path / "converted.wav"
        sf.write(str(converted_path), audio, sr)

        converted_outputs = {"test_method": str(converted_path)}

        # Mock voice identifier to raise exception
        with patch("auto_voice.evaluation.conversion_quality_analyzer.get_voice_identifier",
                   side_effect=ImportError("Module not found")):

            comparison = analyzer.compare_methodologies(
                source_audio=str(source_path),
                target_profile_id="test_profile",
                methodologies=["test_method"],
                converted_outputs=converted_outputs,
            )

        # Should still complete without embedding
        assert isinstance(comparison, MethodologyComparison)
        assert len(comparison.analyses) == 1


def test_compare_methodologies_default_methodologies():
    """Test methodology comparison uses default list when not specified."""
    analyzer = ConversionQualityAnalyzer(device="cpu")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        sr = 22050
        audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 2, int(sr * 2))).astype(np.float32)

        source_path = tmp_path / "source.wav"
        sf.write(str(source_path), audio, sr)

        # Don't provide any converted outputs
        comparison = analyzer.compare_methodologies(
            source_audio=str(source_path),
            target_profile_id="test_profile",
            methodologies=None,  # Should use defaults
            converted_outputs={},
        )

        # Should handle gracefully with no outputs
        assert isinstance(comparison, MethodologyComparison)
        assert comparison.best_methodology == "none"


# ============================================================================
# Test Additional Edge Cases
# ============================================================================

def test_analyze_with_log_f0_rmse():
    """Test that log_f0_rmse is computed (currently not tested)."""
    analyzer = ConversionQualityAnalyzer(device="cpu")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        source_path = tmp_path / "source.wav"
        sf.write(str(source_path), audio, sr)

        converted_path = tmp_path / "converted.wav"
        sf.write(str(converted_path), audio, sr)

        analysis = analyzer.analyze(
            source_audio=str(source_path),
            converted_audio=str(converted_path),
            methodology="test",
        )

        # Check that log_f0_rmse exists in metrics
        assert hasattr(analysis.metrics, 'log_f0_rmse')
        assert isinstance(analysis.metrics.log_f0_rmse, float)


def test_quality_score_weights_sum_to_one():
    """Verify that quality score weights are properly normalized."""
    analyzer = ConversionQualityAnalyzer()

    total_weight = sum(analyzer.WEIGHTS.values())
    assert 0.99 < total_weight < 1.01  # Allow for floating point precision


def test_analyzer_thresholds_are_sensible():
    """Verify analyzer thresholds are set to documented values."""
    analyzer = ConversionQualityAnalyzer()

    assert analyzer.SPEAKER_SIMILARITY_MIN == 0.85
    assert analyzer.MCD_MAX == 4.5
    assert analyzer.F0_CORRELATION_MIN == 0.90
    assert analyzer.F0_RMSE_MAX == 20.0
    assert analyzer.RTF_MAX_REALTIME == 0.30
    assert analyzer.SNR_MIN == 20.0
    assert analyzer.PESQ_MIN == 3.5
    assert analyzer.STOI_MIN == 0.85


def test_analyze_with_all_metrics_passing():
    """Test analysis where all metrics pass thresholds."""
    analyzer = ConversionQualityAnalyzer(device="cpu")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        source_path = tmp_path / "source.wav"
        sf.write(str(source_path), audio, sr)

        converted_path = tmp_path / "converted.wav"
        sf.write(str(converted_path), audio, sr)

        # Mock all metrics to pass
        with patch.object(analyzer, "_compute_mcd", return_value=3.0), \
             patch.object(analyzer, "_compute_f0_metrics", return_value=(0.95, 10.0)), \
             patch.object(analyzer, "_compute_snr", return_value=25.0), \
             patch.object(analyzer, "_compute_pesq", return_value=4.0), \
             patch.object(analyzer, "_compute_stoi", return_value=0.90):

            mock_embedding = np.random.randn(768).astype(np.float32)
            mock_embedding = mock_embedding / np.linalg.norm(mock_embedding)

            analysis = analyzer.analyze(
                source_audio=str(source_path),
                converted_audio=str(converted_path),
                target_speaker_embedding=mock_embedding,
                methodology="perfect_test",
            )

        # Should pass all thresholds
        assert analysis.passes_thresholds
        assert len(analysis.threshold_failures) == 0
        assert len(analysis.recommendations) == 0


def test_analyze_generates_all_recommendation_types():
    """Test that all recommendation types can be generated."""
    analyzer = ConversionQualityAnalyzer(device="cpu")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        sr = 22050
        audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 2, int(sr * 2))).astype(np.float32)

        source_path = tmp_path / "source.wav"
        sf.write(str(source_path), audio, sr)

        converted_path = tmp_path / "converted.wav"
        sf.write(str(converted_path), audio, sr)

        # Set all metrics to fail
        with patch.object(analyzer, "_compute_mcd", return_value=10.0), \
             patch.object(analyzer, "_compute_f0_metrics", return_value=(0.5, 50.0)), \
             patch.object(analyzer, "_compute_snr", return_value=10.0):

            # Low speaker similarity
            poor_embedding = np.random.randn(768).astype(np.float32) * 0.01

            analysis = analyzer.analyze(
                source_audio=str(source_path),
                converted_audio=str(converted_path),
                target_speaker_embedding=poor_embedding,
                methodology="poor_test",
            )

        # Should have recommendations for all failing metrics
        recommendations = "\n".join(analysis.recommendations)
        assert "training epochs" in recommendations or "training samples" in recommendations
        assert "vocoder" in recommendations or "decoder" in recommendations
        assert "pitch" in recommendations or "F0" in recommendations
        assert "noise reduction" in recommendations


def test_save_analysis_creates_valid_json(tmp_path):
    """Test that saved analysis is valid JSON with all fields."""
    analyzer = ConversionQualityAnalyzer(device="cpu")

    metrics = QualityMetrics(
        speaker_similarity=0.92,
        mcd=3.2,
        f0_correlation=0.93,
        f0_rmse=15.5,
        rtf=0.25,
        processing_time_ms=1200.0,
        snr=28.0,
        pesq=3.8,
        stoi=0.88,
        quality_score=88.5,
    )

    analysis = ConversionAnalysis(
        methodology="test_methodology",
        source_audio="/path/to/source.wav",
        converted_audio="/path/to/converted.wav",
        target_profile_id="profile_123",
        metrics=metrics,
        timestamp="2026-02-02T12:00:00",
        passes_thresholds=True,
        threshold_failures=[],
        recommendations=["Excellent quality"],
    )

    output_path = tmp_path / "test_analysis.json"
    analyzer.save_analysis(analysis, str(output_path))

    # Load and verify structure
    with open(output_path) as f:
        data = json.load(f)

    assert data["methodology"] == "test_methodology"
    assert data["target_profile_id"] == "profile_123"
    assert data["passes_thresholds"] is True

    # Verify all metrics are present
    assert "speaker_similarity" in data["metrics"]
    assert "mcd" in data["metrics"]
    assert "f0_correlation" in data["metrics"]
    assert "f0_rmse" in data["metrics"]
    assert "rtf" in data["metrics"]
    assert "snr" in data["metrics"]
    assert "pesq" in data["metrics"]
    assert "stoi" in data["metrics"]
    assert "quality_score" in data["metrics"]


def test_methodology_comparison_summary_format():
    """Test methodology comparison summary is well-formatted."""
    analyzer = ConversionQualityAnalyzer(device="cpu")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        sr = 22050
        audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 2, int(sr * 2))).astype(np.float32)

        source_path = tmp_path / "source.wav"
        sf.write(str(source_path), audio, sr)

        converted_a = tmp_path / "converted_a.wav"
        sf.write(str(converted_a), audio, sr)

        converted_b = tmp_path / "converted_b.wav"
        sf.write(str(converted_b), audio, sr)

        outputs = {
            "method_a": str(converted_a),
            "method_b": str(converted_b),
        }

        comparison = analyzer.compare_methodologies(
            source_audio=str(source_path),
            target_profile_id="test",
            methodologies=["method_a", "method_b"],
            converted_outputs=outputs,
        )

        # Verify summary format
        assert "Best methodology:" in comparison.summary
        assert "score:" in comparison.summary
        assert "similarity:" in comparison.summary
        assert "MCD:" in comparison.summary


# ============================================================================
# Test Performance and Edge Cases
# ============================================================================

def test_analyze_very_long_audio(tmp_path):
    """Test analysis with very long audio file."""
    analyzer = ConversionQualityAnalyzer(device="cpu")

    sr = 16000
    duration = 10.0  # 10 seconds
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    source_path = tmp_path / "long_source.wav"
    sf.write(str(source_path), audio, sr)

    converted_path = tmp_path / "long_converted.wav"
    sf.write(str(converted_path), audio, sr)

    # Should handle without issues
    analysis = analyzer.analyze(
        source_audio=str(source_path),
        converted_audio=str(converted_path),
        methodology="long_audio_test",
    )

    assert isinstance(analysis, ConversionAnalysis)
    assert analysis.metrics.quality_score > 0


def test_analyze_extreme_sample_rate(tmp_path):
    """Test analysis with extreme sample rate."""
    analyzer = ConversionQualityAnalyzer(device="cpu")

    sr = 48000  # High sample rate
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    source_path = tmp_path / "high_sr_source.wav"
    sf.write(str(source_path), audio, sr)

    converted_path = tmp_path / "high_sr_converted.wav"
    sf.write(str(converted_path), audio, sr)

    analysis = analyzer.analyze(
        source_audio=str(source_path),
        converted_audio=str(converted_path),
        methodology="high_sr_test",
    )

    assert isinstance(analysis, ConversionAnalysis)


def test_compute_quality_score_edge_values():
    """Test quality score computation with edge case values."""
    analyzer = ConversionQualityAnalyzer()

    # Test with extremely poor metrics
    poor_metrics = QualityMetrics(
        speaker_similarity=0.0,
        mcd=20.0,  # Very high
        f0_correlation=-1.0,  # Negative correlation
        snr=-10.0,  # Negative SNR
        pesq=1.0,  # Minimum
        stoi=0.0,  # Minimum
    )

    score = analyzer._compute_quality_score(poor_metrics)
    assert 0 <= score <= 100
    assert score < 20  # Should be very low

    # Test with boundary metrics
    boundary_metrics = QualityMetrics(
        speaker_similarity=1.0,
        mcd=0.0,
        f0_correlation=1.0,
        snr=50.0,  # Very high
        pesq=4.5,  # Maximum
        stoi=1.0,  # Maximum
    )

    score = analyzer._compute_quality_score(boundary_metrics)
    assert 0 <= score <= 100
    assert score > 95  # Should be near perfect


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
