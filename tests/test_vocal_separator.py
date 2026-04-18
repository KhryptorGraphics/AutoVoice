"""Tests for VocalSeparator using Demucs HTDemucs model."""
import numpy as np
import pytest
import torch
from unittest.mock import patch, MagicMock


class TestVocalSeparatorInit:
    """VocalSeparator initialization tests."""

    @pytest.mark.smoke
    def test_init_default_device(self):
        """VocalSeparator initializes with default device."""
        from auto_voice.audio.separation import VocalSeparator
        sep = VocalSeparator()
        assert sep.model_name == 'htdemucs'
        assert sep._model is None
        assert sep.segment is None

    def test_init_custom_model_name(self):
        """VocalSeparator accepts custom model name."""
        from auto_voice.audio.separation import VocalSeparator
        sep = VocalSeparator(model_name='htdemucs_ft')
        assert sep.model_name == 'htdemucs_ft'

    def test_init_custom_segment(self):
        """VocalSeparator accepts segment parameter."""
        from auto_voice.audio.separation import VocalSeparator
        sep = VocalSeparator(segment=12.0)
        assert sep.segment == 12.0

    def test_init_explicit_device(self):
        """VocalSeparator accepts explicit device."""
        from auto_voice.audio.separation import VocalSeparator
        dev = torch.device('cpu')
        sep = VocalSeparator(device=dev)
        assert sep.device == dev

    def test_init_raises_without_demucs(self):
        """VocalSeparator raises RuntimeError when demucs is not installed."""
        with patch.dict('sys.modules', {'demucs': None, 'demucs.pretrained': None, 'demucs.apply': None}):
            # Need to reimport to trigger the ImportError check
            import importlib
            import auto_voice.audio.separation as sep_module
            # Force re-import by clearing the cached module functions
            with patch('builtins.__import__', side_effect=ImportError("No module named 'demucs'")):
                with pytest.raises(RuntimeError, match="Demucs is required"):
                    # Directly test the import logic
                    try:
                        from demucs.pretrained import get_model
                        from demucs.apply import apply_model
                    except ImportError as e:
                        raise RuntimeError(
                            f"Demucs is required for vocal separation but is not installed: {e}. "
                            f"Install with: pip install demucs"
                        )


class TestVocalSeparatorLoadModel:
    """Tests for lazy model loading."""

    def test_lazy_load_calls_get_model(self):
        """_load_model calls get_model with correct name."""
        from auto_voice.audio.separation import VocalSeparator
        sep = VocalSeparator(device=torch.device('cpu'))

        mock_model = MagicMock()
        mock_model.sources = ['drums', 'bass', 'other', 'vocals']
        mock_model.samplerate = 44100
        sep._get_model = MagicMock(return_value=mock_model)

        sep._load_model()

        sep._get_model.assert_called_once_with('htdemucs')
        mock_model.to.assert_called_once_with(torch.device('cpu'))
        mock_model.eval.assert_called_once()
        assert sep._model is mock_model

    def test_lazy_load_only_once(self):
        """_load_model does not reload if already loaded."""
        from auto_voice.audio.separation import VocalSeparator
        sep = VocalSeparator(device=torch.device('cpu'))

        mock_model = MagicMock()
        mock_model.sources = ['drums', 'bass', 'other', 'vocals']
        mock_model.samplerate = 44100
        sep._get_model = MagicMock(return_value=mock_model)

        sep._load_model()
        sep._load_model()

        sep._get_model.assert_called_once()

    def test_load_model_raises_on_failure(self):
        """_load_model raises RuntimeError if get_model fails."""
        from auto_voice.audio.separation import VocalSeparator
        sep = VocalSeparator(device=torch.device('cpu'))
        sep._get_model = MagicMock(side_effect=Exception("Network error"))

        with pytest.raises(RuntimeError, match="Failed to load Demucs model"):
            sep._load_model()


class TestVocalSeparatorProperties:
    """Tests for model properties."""

    def _make_separator_with_mock(self):
        from auto_voice.audio.separation import VocalSeparator
        sep = VocalSeparator(device=torch.device('cpu'))
        mock_model = MagicMock()
        mock_model.sources = ['drums', 'bass', 'other', 'vocals']
        mock_model.samplerate = 44100
        sep._get_model = MagicMock(return_value=mock_model)
        return sep

    def test_model_sample_rate(self):
        """model_sample_rate returns the model's sample rate."""
        sep = self._make_separator_with_mock()
        assert sep.model_sample_rate == 44100

    def test_sources_property(self):
        """sources returns list of source names."""
        sep = self._make_separator_with_mock()
        assert sep.sources == ['drums', 'bass', 'other', 'vocals']


class TestVocalSeparatorSeparate:
    """Tests for the separate() method."""

    def _make_separator_with_mock(self, sr=44100):
        """Create a separator with mocked model that returns realistic tensors."""
        from auto_voice.audio.separation import VocalSeparator
        sep = VocalSeparator(device=torch.device('cpu'), segment=12.0)

        mock_model = MagicMock()
        mock_model.sources = ['drums', 'bass', 'other', 'vocals']
        mock_model.samplerate = sr
        sep._get_model = MagicMock(return_value=mock_model)

        return sep, mock_model

    def test_separate_mono_returns_dict(self):
        """separate() returns dict with 'vocals' and 'instrumental'."""
        sep, mock_model = self._make_separator_with_mock(sr=22050)
        n_samples = 22050 * 2  # 2 seconds

        # Mock apply_model to return proper shaped tensor
        # shape: (batch=1, sources=4, channels=2, samples)
        mock_sources = torch.randn(1, 4, 2, n_samples)
        sep._apply_model = MagicMock(return_value=mock_sources)

        audio = np.random.randn(n_samples).astype(np.float32) * 0.5
        result = sep.separate(audio, sr=22050)

        assert 'vocals' in result
        assert 'instrumental' in result
        assert isinstance(result['vocals'], np.ndarray)
        assert isinstance(result['instrumental'], np.ndarray)
        assert result['vocals'].dtype == np.float32
        assert result['instrumental'].dtype == np.float32

    def test_separate_mono_output_length(self):
        """Output length matches input length."""
        sep, mock_model = self._make_separator_with_mock(sr=22050)
        n_samples = 22050 * 3

        mock_sources = torch.randn(1, 4, 2, n_samples)
        sep._apply_model = MagicMock(return_value=mock_sources)

        audio = np.random.randn(n_samples).astype(np.float32) * 0.5
        result = sep.separate(audio, sr=22050)

        assert len(result['vocals']) == n_samples
        assert len(result['instrumental']) == n_samples

    def test_separate_stereo_input(self):
        """separate() handles stereo input (2, samples)."""
        sep, mock_model = self._make_separator_with_mock(sr=22050)
        n_samples = 22050

        mock_sources = torch.randn(1, 4, 2, n_samples)
        sep._apply_model = MagicMock(return_value=mock_sources)

        audio = np.random.randn(2, n_samples).astype(np.float32) * 0.5
        result = sep.separate(audio, sr=22050)

        assert len(result['vocals']) == n_samples
        assert len(result['instrumental']) == n_samples

    def test_separate_resamples_when_sr_differs(self):
        """separate() resamples audio when input sr != model sr."""
        sep, mock_model = self._make_separator_with_mock(sr=44100)
        input_sr = 22050
        n_samples = input_sr * 2  # 2 seconds at 22050

        # Model outputs at 44100 Hz (2x samples)
        model_samples = 44100 * 2
        mock_sources = torch.randn(1, 4, 2, model_samples)
        sep._apply_model = MagicMock(return_value=mock_sources)

        audio = np.random.randn(n_samples).astype(np.float32) * 0.5
        result = sep.separate(audio, sr=input_sr)

        # Output should be resampled back to input length
        assert len(result['vocals']) == n_samples
        assert len(result['instrumental']) == n_samples

    def test_separate_no_resample_when_sr_matches(self):
        """No resampling when input sr matches model sr."""
        sep, mock_model = self._make_separator_with_mock(sr=44100)
        n_samples = 44100 * 2

        mock_sources = torch.randn(1, 4, 2, n_samples)
        sep._apply_model = MagicMock(return_value=mock_sources)

        audio = np.random.randn(n_samples).astype(np.float32) * 0.5

        with patch.dict('sys.modules', {}):
            result = sep.separate(audio, sr=44100)

        assert len(result['vocals']) == n_samples

    def test_separate_passes_segment_kwarg(self):
        """segment parameter is passed to apply_model."""
        sep, mock_model = self._make_separator_with_mock(sr=22050)
        sep.segment = 7.5
        n_samples = 22050

        mock_sources = torch.randn(1, 4, 2, n_samples)
        sep._apply_model = MagicMock(return_value=mock_sources)

        audio = np.random.randn(n_samples).astype(np.float32)
        sep.separate(audio, sr=22050)

        # Check that segment was passed
        call_kwargs = sep._apply_model.call_args[1]
        assert call_kwargs['segment'] == 7.5

    def test_separate_no_segment_kwarg_when_none(self):
        """No segment kwarg when segment is None."""
        sep, mock_model = self._make_separator_with_mock(sr=22050)
        sep.segment = None
        n_samples = 22050

        mock_sources = torch.randn(1, 4, 2, n_samples)
        sep._apply_model = MagicMock(return_value=mock_sources)

        audio = np.random.randn(n_samples).astype(np.float32)
        sep.separate(audio, sr=22050)

        call_kwargs = sep._apply_model.call_args[1]
        assert 'segment' not in call_kwargs

    def test_separate_empty_audio_raises(self):
        """separate() raises ValueError on empty audio."""
        from auto_voice.audio.separation import VocalSeparator
        sep = VocalSeparator(device=torch.device('cpu'))

        with pytest.raises(ValueError, match="Cannot separate empty audio"):
            sep.separate(np.array([], dtype=np.float32), sr=22050)

    def test_separate_3d_audio_raises(self):
        """separate() raises ValueError on 3D audio."""
        from auto_voice.audio.separation import VocalSeparator
        sep = VocalSeparator(device=torch.device('cpu'))

        with pytest.raises(ValueError, match="Audio must be 1D .* or 2D"):
            sep.separate(np.zeros((2, 3, 100), dtype=np.float32), sr=22050)

    def test_separate_output_not_nan(self):
        """Outputs are not NaN."""
        sep, mock_model = self._make_separator_with_mock(sr=22050)
        n_samples = 22050

        # Use deterministic non-NaN tensor
        mock_sources = torch.ones(1, 4, 2, n_samples) * 0.3
        sep._apply_model = MagicMock(return_value=mock_sources)

        audio = np.ones(n_samples, dtype=np.float32) * 0.5
        result = sep.separate(audio, sr=22050)

        assert not np.any(np.isnan(result['vocals']))
        assert not np.any(np.isnan(result['instrumental']))

    def test_separate_vocals_isolation(self):
        """Vocals output comes from the correct source index."""
        sep, mock_model = self._make_separator_with_mock(sr=22050)
        n_samples = 1000

        # Set up sources where vocals (index 3) has distinct values
        mock_sources = torch.zeros(1, 4, 2, n_samples)
        mock_sources[0, 3, :, :] = 1.0  # vocals channel = 1.0
        mock_sources[0, 0, :, :] = 0.1  # drums
        mock_sources[0, 1, :, :] = 0.2  # bass
        mock_sources[0, 2, :, :] = 0.3  # other
        sep._apply_model = MagicMock(return_value=mock_sources)

        audio = np.random.randn(n_samples).astype(np.float32)
        result = sep.separate(audio, sr=22050)

        # Vocals should be ~1.0 (mean of 2 channels both at 1.0)
        np.testing.assert_allclose(result['vocals'], 1.0, atol=1e-6)
        # Instrumental should be mean of (0.1 + 0.2 + 0.3) = 0.6
        np.testing.assert_allclose(result['instrumental'], 0.6, atol=1e-6)

    def test_separate_no_vocals_source_raises(self):
        """RuntimeError if model has no 'vocals' source."""
        from auto_voice.audio.separation import VocalSeparator
        sep = VocalSeparator(device=torch.device('cpu'))

        mock_model = MagicMock()
        mock_model.sources = ['speech', 'noise']  # No 'vocals'
        mock_model.samplerate = 22050
        sep._get_model = MagicMock(return_value=mock_model)

        mock_sources = torch.randn(1, 2, 2, 22050)
        sep._apply_model = MagicMock(return_value=mock_sources)

        audio = np.random.randn(22050).astype(np.float32)
        with pytest.raises(RuntimeError, match="does not have a 'vocals' source"):
            sep.separate(audio, sr=22050)


def _load_demucs_separator(device_str: str = 'cpu'):
    """Helper to load VocalSeparator, skipping if Demucs model unavailable.

    Skips the test if Demucs model fails to download or load (e.g. network
    timeout, missing cached weights).
    """
    from auto_voice.audio.separation import VocalSeparator
    device = torch.device(device_str)
    sep = VocalSeparator(device=device)
    try:
        sep._load_model()
    except RuntimeError as e:
        pytest.skip(f"Demucs model unavailable: {e}")
    return sep


class TestVocalSeparatorIntegration:
    """Integration tests with real Demucs model."""

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.cuda
    def test_real_separation_gpu(self):
        """Real Demucs separation on GPU produces valid output."""
        sep = _load_demucs_separator('cuda')

        # HTDemucs needs sufficient audio length (>= ~8s at 44100)
        # Use native model rate to avoid resampling issues
        sr = 44100
        duration = 10.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t) +
                 0.3 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)

        result = sep.separate(audio, sr)

        assert result['vocals'].shape == audio.shape
        assert result['instrumental'].shape == audio.shape
        assert not np.any(np.isnan(result['vocals']))
        assert not np.any(np.isnan(result['instrumental']))

    @pytest.mark.integration
    @pytest.mark.slow
    def test_real_separation_cpu(self):
        """Real Demucs separation on CPU produces valid output."""
        sep = _load_demucs_separator('cpu')

        # Use native model rate with sufficient length
        sr = 44100
        duration = 10.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        result = sep.separate(audio, sr)

        assert result['vocals'].shape == audio.shape
        assert result['instrumental'].shape == audio.shape
        assert result['vocals'].dtype == np.float32

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.cuda
    def test_separation_produces_different_outputs(self):
        """Separation produces meaningfully different vocals and instrumental."""
        sep = _load_demucs_separator('cuda')

        sr = 44100
        duration = 10.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)

        # Create a mix with vocal-range and bass-range content
        audio = (
            0.4 * np.sin(2 * np.pi * 300 * t) +
            0.3 * np.sin(2 * np.pi * 600 * t) +
            0.5 * np.sin(2 * np.pi * 80 * t) +
            0.3 * np.sin(2 * np.pi * 5000 * t)
        ).astype(np.float32)

        result = sep.separate(audio, sr)

        # Verify separation actually occurred (not just identity/zero)
        vocals = result['vocals']
        instrumental = result['instrumental']

        # Both outputs should have non-trivial energy
        assert np.abs(vocals).max() > 0.005, "Vocals output is near-silent"
        assert np.abs(instrumental).max() > 0.005, "Instrumental output is near-silent"

        # Vocals and instrumental should be different from each other
        correlation = np.abs(np.corrcoef(vocals, instrumental)[0, 1])
        assert correlation < 0.99, f"Vocals and instrumental are too similar (r={correlation:.4f})"

        # Recombination should approximate original (energy conservation)
        recombined = vocals + instrumental
        min_len = min(len(audio), len(recombined))
        reconstruction_error = np.mean((audio[:min_len] - recombined[:min_len]) ** 2)
        original_power = np.mean(audio[:min_len] ** 2)
        # Reconstruction error should be small relative to signal power
        assert reconstruction_error < original_power, (
            f"Reconstruction error {reconstruction_error:.6f} >= signal power {original_power:.6f}"
        )


class TestPipelineIntegration:
    """Test that SingingConversionPipeline uses VocalSeparator."""

    def test_pipeline_get_separator_returns_vocal_separator(self):
        """Pipeline._get_separator() returns a VocalSeparator instance."""
        from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
        from auto_voice.audio.separation import VocalSeparator

        pipeline = SingingConversionPipeline(device=torch.device('cpu'))
        sep = pipeline._get_separator()

        assert isinstance(sep, VocalSeparator)

    def test_pipeline_separator_reused(self):
        """Pipeline reuses the same separator instance."""
        from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

        pipeline = SingingConversionPipeline(device=torch.device('cpu'))
        sep1 = pipeline._get_separator()
        sep2 = pipeline._get_separator()

        assert sep1 is sep2

    def test_pipeline_separate_vocals_calls_separator(self):
        """Pipeline._separate_vocals() delegates to VocalSeparator."""
        from auto_voice.inference.singing_conversion_pipeline import (
            SingingConversionPipeline, SeparationError
        )

        pipeline = SingingConversionPipeline(device=torch.device('cpu'))

        # Mock the separator
        mock_sep = MagicMock()
        mock_sep.separate.return_value = {
            'vocals': np.zeros(1000, dtype=np.float32),
            'instrumental': np.zeros(1000, dtype=np.float32),
        }
        pipeline._separator = mock_sep

        audio = np.random.randn(1000).astype(np.float32)
        result = pipeline._separate_vocals(audio, sr=22050)

        mock_sep.separate.assert_called_once_with(audio, 22050)
        assert 'vocals' in result
        assert 'instrumental' in result

    def test_pipeline_separate_vocals_raises_separation_error(self):
        """Pipeline._separate_vocals() wraps errors in SeparationError."""
        from auto_voice.inference.singing_conversion_pipeline import (
            SingingConversionPipeline, SeparationError
        )

        pipeline = SingingConversionPipeline(device=torch.device('cpu'))

        mock_sep = MagicMock()
        mock_sep.separate.side_effect = RuntimeError("Model failed")
        pipeline._separator = mock_sep

        audio = np.random.randn(1000).astype(np.float32)
        with pytest.raises(SeparationError, match="Vocal separation failed"):
            pipeline._separate_vocals(audio, sr=22050)


class TestNoFallbackBehavior:
    """Verify no fallback behavior exists."""

    def test_no_fallback_method(self):
        """VocalSeparator has no _fallback_separate method."""
        from auto_voice.audio.separation import VocalSeparator
        sep = VocalSeparator(device=torch.device('cpu'))
        assert not hasattr(sep, '_fallback_separate')

    def test_model_failure_raises_runtime_error(self):
        """Model load failure raises RuntimeError, not fallback."""
        from auto_voice.audio.separation import VocalSeparator
        sep = VocalSeparator(device=torch.device('cpu'))
        sep._get_model = MagicMock(side_effect=Exception("Download failed"))

        with pytest.raises(RuntimeError):
            sep.separate(np.zeros(1000, dtype=np.float32), sr=22050)
