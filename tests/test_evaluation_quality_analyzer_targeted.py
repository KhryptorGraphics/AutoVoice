"""Targeted tests to cover specific uncovered lines and reach 90% coverage.

This file specifically targets lines that are hard to test due to external dependencies.
Uses deep mocking to bypass library import issues.

Missing lines to cover (from 84% report):
- 171-178: WavLM model loading
- 182-184: Audio resampling
- 190-198: Model inference
- 216-230: MCD computation with librosa
- 250-269: F0 metrics with librosa.pyin
- 295-296: SNR edge case
- 347-352: STOI import error handling
- 521, 529-531: Voice identifier integration
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import sys

from auto_voice.evaluation.conversion_quality_analyzer import (
    ConversionQualityAnalyzer,
    QualityMetrics,
)


def test_extract_speaker_embedding_full_path():
    """Test speaker embedding extraction through the actual code path."""
    analyzer = ConversionQualityAnalyzer(device="cpu")
    sr = 16000
    audio = np.random.randn(int(sr * 2)).astype(np.float32)

    # Create comprehensive mocks that match the actual import structure
    mock_processor_instance = MagicMock()
    mock_processor_class = MagicMock(return_value=None)
    mock_processor_class.from_pretrained = MagicMock(return_value=mock_processor_instance)

    mock_model_instance = MagicMock()
    mock_model_class = MagicMock(return_value=None)
    mock_model_class.from_pretrained = MagicMock(return_value=mock_model_instance)

    # Setup processor behavior
    import torch
    mock_inputs = MagicMock()
    mock_inputs.input_values = torch.randn(1, int(sr * 2))
    mock_processor_instance.return_value = mock_inputs

    # Setup model behavior
    mock_model_instance.parameters.return_value = iter([torch.tensor([0.0], device='cpu')])
    mock_output = MagicMock()
    mock_output.last_hidden_state = torch.randn(1, 100, 768)
    mock_model_instance.return_value = mock_output

    # Patch at the import location
    with patch.dict(sys.modules, {
        'transformers': MagicMock(
            Wav2Vec2FeatureExtractor=mock_processor_class,
            WavLMModel=mock_model_class
        ),
        'transformers.models.wavlm': MagicMock(),
        'transformers.models.wavlm.modeling_wavlm': MagicMock(WavLMModel=mock_model_class),
    }):
        # Trigger lazy imports inside the method
        with patch('torch.cuda.is_available', return_value=False):
            embedding = analyzer._extract_speaker_embedding(audio, sr)

    # Verify result
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] == 768

    # Verify the model was loaded (tests lines 171-178)
    # Note: Due to mocking complexity, we verify the error path was NOT taken
    assert not np.allclose(embedding, 0.0)  # Should not be all zeros (error case)


def test_extract_speaker_embedding_with_resampling():
    """Test embedding extraction that requires resampling (lines 182-184)."""
    analyzer = ConversionQualityAnalyzer(device="cpu")
    sr = 22050  # Non-16kHz
    audio = np.random.randn(int(sr * 2)).astype(np.float32)

    import torch

    # Create mock resampler
    mock_resampler = MagicMock()
    resampled_audio = torch.randn(int(16000 * 2))
    mock_resampler.return_value = resampled_audio

    mock_resample_class = MagicMock(return_value=mock_resampler)

    # Mock transformers components
    mock_processor = MagicMock()
    mock_inputs = MagicMock()
    mock_inputs.input_values = torch.randn(1, int(16000 * 2))
    mock_processor.return_value = mock_inputs

    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([torch.tensor([0.0])])
    mock_output = MagicMock()
    mock_output.last_hidden_state = torch.randn(1, 100, 768)
    mock_model.return_value = mock_output

    with patch.dict(sys.modules, {
        'transformers': MagicMock(
            Wav2Vec2FeatureExtractor=MagicMock(from_pretrained=MagicMock(return_value=mock_processor)),
            WavLMModel=MagicMock(from_pretrained=MagicMock(return_value=mock_model))
        ),
        'torchaudio.transforms': MagicMock(Resample=mock_resample_class),
    }):
        with patch('torchaudio.transforms.Resample', mock_resample_class):
            embedding = analyzer._extract_speaker_embedding(audio, sr)

    # Verify resampling was triggered
    mock_resample_class.assert_called_once_with(sr, 16000)
    assert isinstance(embedding, np.ndarray)


def test_compute_mcd_full_computation():
    """Test MCD computation through actual librosa path (lines 216-230)."""
    analyzer = ConversionQualityAnalyzer()
    sr = 22050
    source = np.random.randn(int(sr * 2)).astype(np.float32)
    converted = np.random.randn(int(sr * 2)).astype(np.float32)

    # Mock librosa.feature.mfcc to return realistic MFCC features
    mock_mfcc_source = np.random.randn(13, 150).astype(np.float32)
    mock_mfcc_converted = np.random.randn(13, 150).astype(np.float32)

    call_count = [0]

    def mock_mfcc_func(y, sr, n_mfcc):
        call_count[0] += 1
        if call_count[0] == 1:
            return mock_mfcc_source
        else:
            return mock_mfcc_converted

    with patch('librosa.feature.mfcc', side_effect=mock_mfcc_func):
        mcd = analyzer._compute_mcd(source, converted, sr)

    # Verify computation was performed
    assert isinstance(mcd, float)
    assert mcd >= 0  # MCD should be non-negative
    assert not np.isnan(mcd)


def test_compute_mcd_length_alignment_path():
    """Test MCD with mismatched MFCC lengths (tests alignment logic)."""
    analyzer = ConversionQualityAnalyzer()
    sr = 16000
    source = np.random.randn(int(sr * 2)).astype(np.float32)
    converted = np.random.randn(int(sr * 3)).astype(np.float32)

    # Return different lengths to trigger alignment
    mock_mfcc_source = np.random.randn(13, 100).astype(np.float32)
    mock_mfcc_converted = np.random.randn(13, 150).astype(np.float32)

    call_count = [0]

    def mock_mfcc_func(y, sr, n_mfcc):
        call_count[0] += 1
        if call_count[0] == 1:
            return mock_mfcc_source
        else:
            return mock_mfcc_converted

    with patch('librosa.feature.mfcc', side_effect=mock_mfcc_func):
        mcd = analyzer._compute_mcd(source, converted, sr)

    assert isinstance(mcd, float)


def test_compute_f0_metrics_full_computation():
    """Test F0 metrics computation through librosa.pyin (lines 250-269)."""
    analyzer = ConversionQualityAnalyzer()
    sr = 22050
    source = np.random.randn(int(sr * 2)).astype(np.float32)
    converted = np.random.randn(int(sr * 2)).astype(np.float32)

    # Create realistic F0 contours with some valid and some NaN values
    f0_source = np.full(200, 440.0)
    f0_source[::5] = np.nan  # 20% NaN values

    f0_converted = np.full(200, 442.0)
    f0_converted[::5] = np.nan  # 20% NaN values

    call_count = [0]

    def mock_pyin_func(y, fmin, fmax, sr):
        call_count[0] += 1
        if call_count[0] == 1:
            return (f0_source, None, None)
        else:
            return (f0_converted, None, None)

    with patch('librosa.pyin', side_effect=mock_pyin_func):
        correlation, rmse = analyzer._compute_f0_metrics(source, converted, sr)

    # Verify computation completed
    assert isinstance(correlation, float)
    assert isinstance(rmse, float)
    assert -1 <= correlation <= 1
    assert rmse >= 0


def test_compute_f0_metrics_valid_mask_path():
    """Test F0 metrics with insufficient valid frames (line 257-258)."""
    analyzer = ConversionQualityAnalyzer()
    sr = 16000
    audio = np.random.randn(int(sr * 2)).astype(np.float32)

    # Return F0 with < 10 valid frames
    f0_mostly_nan = np.full(100, np.nan)
    f0_mostly_nan[:5] = 440.0  # Only 5 valid frames

    call_count = [0]

    def mock_pyin_func(y, fmin, fmax, sr):
        call_count[0] += 1
        return (f0_mostly_nan.copy(), None, None)

    with patch('librosa.pyin', side_effect=mock_pyin_func):
        correlation, rmse = analyzer._compute_f0_metrics(audio, audio, sr)

    # Should return defaults (0.0, 100.0)
    assert correlation == 0.0
    assert rmse == 100.0


def test_compute_snr_edge_case_divide_by_zero():
    """Test SNR computation with zero/near-zero power (line 295-296)."""
    analyzer = ConversionQualityAnalyzer()

    # Audio with zero signal power
    audio_zeros = np.zeros(16000, dtype=np.float32)
    snr_zeros = analyzer._compute_snr(audio_zeros)

    # Should handle gracefully (either -inf or very low value)
    assert isinstance(snr_zeros, float)
    assert snr_zeros <= 0 or np.isinf(snr_zeros)

    # Audio with very low noise floor
    audio_quiet = np.ones(16000, dtype=np.float32) * 1e-10
    snr_quiet = analyzer._compute_snr(audio_quiet)

    assert isinstance(snr_quiet, float)


def test_compute_stoi_import_none_in_sys_modules():
    """Test STOI when module exists but returns None (line 349)."""
    analyzer = ConversionQualityAnalyzer()
    sr = 16000
    audio = np.random.randn(int(sr * 2)).astype(np.float32) * 0.1

    # Simulate ImportError by having None in sys.modules
    with patch.dict(sys.modules, {'pystoi': None}):
        # Try to import will fail
        try:
            from pystoi import stoi as stoi_func
            assert False, "Should have raised ImportError"
        except (ImportError, AttributeError):
            pass

        # Now test the actual method
        score = analyzer._compute_stoi(audio, audio, sr)

    # Should return None when import fails
    assert score is None


def test_compute_stoi_exception_during_computation():
    """Test STOI handles exceptions during computation (line 356-358)."""
    analyzer = ConversionQualityAnalyzer()
    sr = 16000
    audio = np.random.randn(int(sr * 2)).astype(np.float32) * 0.1

    # Mock stoi to raise exception
    mock_stoi_func = MagicMock(side_effect=RuntimeError("STOI computation error"))

    with patch.dict(sys.modules, {'pystoi': MagicMock(stoi=mock_stoi_func)}):
        with patch('pystoi.stoi', mock_stoi_func):
            score = analyzer._compute_stoi(audio, audio, sr)

    assert score is None


def test_compare_methodologies_voice_identifier_import_path():
    """Test methodology comparison voice identifier import (lines 526, 529-531)."""
    analyzer = ConversionQualityAnalyzer(device="cpu")

    import tempfile
    from pathlib import Path
    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        sr = 22050
        audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 2, int(sr * 2))).astype(np.float32)

        source_path = tmp_path / "source.wav"
        sf.write(str(source_path), audio, sr)

        converted_path = tmp_path / "converted.wav"
        sf.write(str(converted_path), audio, sr)

        # Mock get_voice_identifier function
        mock_identifier = MagicMock()
        mock_identifier._embeddings = {
            "test_profile": np.random.randn(768).astype(np.float32)
        }

        # Patch at the module level where it's imported
        with patch('auto_voice.evaluation.conversion_quality_analyzer.get_voice_identifier',
                   return_value=mock_identifier) as mock_get:

            comparison = analyzer.compare_methodologies(
                source_audio=str(source_path),
                target_profile_id="test_profile",
                methodologies=["test_method"],
                converted_outputs={"test_method": str(converted_path)},
            )

        # Verify get_voice_identifier was called (tests line 527)
        mock_get.assert_called_once()

        # Verify comparison was created
        assert comparison is not None


def test_compare_methodologies_voice_identifier_exception():
    """Test methodology comparison when voice identifier raises exception (line 530-531)."""
    analyzer = ConversionQualityAnalyzer(device="cpu")

    import tempfile
    from pathlib import Path
    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        sr = 22050
        audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 2, int(sr * 2))).astype(np.float32)

        source_path = tmp_path / "source.wav"
        sf.write(str(source_path), audio, sr)

        converted_path = tmp_path / "converted.wav"
        sf.write(str(converted_path), audio, sr)

        # Mock to raise exception
        with patch('auto_voice.evaluation.conversion_quality_analyzer.get_voice_identifier',
                   side_effect=Exception("Voice identifier failed")):

            comparison = analyzer.compare_methodologies(
                source_audio=str(source_path),
                target_profile_id="test_profile",
                methodologies=["test_method"],
                converted_outputs={"test_method": str(converted_path)},
            )

        # Should complete without embedding (tests exception handling)
        assert comparison is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
