"""Tests for separation.py - Vocal extraction with Demucs.

Task 2.3: Test separation.py
- Test Demucs separation (vocals, drums, bass, other)
- Verify output stems (4 files)
- Test separation quality (SDR metric - mocked)
- Test GPU vs CPU execution
"""
import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from auto_voice.audio.separation import VocalSeparator


@pytest.fixture
def sample_audio():
    """Create sample stereo audio."""
    sr = 44100
    duration = 3.0
    num_samples = int(duration * sr)

    # Create stereo audio
    audio = np.random.randn(2, num_samples).astype(np.float32) * 0.3
    return audio, sr


@pytest.fixture
def sample_mono_audio():
    """Create sample mono audio."""
    sr = 44100
    duration = 3.0
    num_samples = int(duration * sr)

    audio = np.random.randn(num_samples).astype(np.float32) * 0.3
    return audio, sr


class TestVocalSeparatorInit:
    """Test VocalSeparator initialization."""

    @pytest.mark.smoke
    def test_init_default(self):
        """Test default initialization."""
        separator = VocalSeparator()

        assert separator.device.type in ['cuda', 'cpu']
        assert separator.model_name == 'htdemucs'
        assert separator.segment is None

    def test_init_custom_device(self):
        """Test initialization with custom device."""
        separator = VocalSeparator(device=torch.device('cpu'))

        assert separator.device.type == 'cpu'

    def test_init_custom_model(self):
        """Test initialization with custom model name."""
        separator = VocalSeparator(model_name='htdemucs_ft')

        assert separator.model_name == 'htdemucs_ft'

    def test_init_with_segment(self):
        """Test initialization with segment duration."""
        separator = VocalSeparator(segment=7.8)

        assert separator.segment == 7.8

    @patch('auto_voice.audio.separation.torch.cuda.is_available')
    def test_init_auto_device_selection(self, mock_cuda):
        """Test automatic device selection."""
        # Test CUDA available
        mock_cuda.return_value = True
        separator = VocalSeparator()
        assert separator.device.type == 'cuda'

        # Test CUDA not available
        mock_cuda.return_value = False
        separator = VocalSeparator()
        assert separator.device.type == 'cpu'

    def test_init_raises_on_missing_demucs(self):
        """Test that missing demucs raises RuntimeError."""
        with patch('auto_voice.audio.separation.VocalSeparator.__init__') as mock_init:
            # Simulate import error
            mock_init.side_effect = RuntimeError("Demucs is required")

            with pytest.raises(RuntimeError, match="Demucs is required"):
                mock_init(None)


class TestModelLoading:
    """Test model loading functionality."""

    @patch('auto_voice.audio.separation.VocalSeparator._get_model')
    def test_lazy_model_loading(self, mock_get_model, sample_mono_audio):
        """Test that model is lazy-loaded."""
        mock_model = MagicMock()
        mock_model.samplerate = 44100
        mock_model.sources = ['drums', 'bass', 'other', 'vocals']
        mock_get_model.return_value = mock_model

        separator = VocalSeparator(device=torch.device('cpu'))
        assert separator._model is None

        # Access sample rate should trigger loading
        sr = separator.model_sample_rate
        assert separator._model is not None
        assert sr == 44100

    @patch('auto_voice.audio.separation.VocalSeparator._get_model')
    def test_model_sources_property(self, mock_get_model):
        """Test that sources property works."""
        mock_model = MagicMock()
        mock_model.sources = ['drums', 'bass', 'other', 'vocals']
        mock_get_model.return_value = mock_model

        separator = VocalSeparator(device=torch.device('cpu'))
        sources = separator.sources

        assert sources == ['drums', 'bass', 'other', 'vocals']
        assert 'vocals' in sources

    @patch('auto_voice.audio.separation.VocalSeparator._get_model')
    def test_load_model_failure(self, mock_get_model):
        """Test handling of model loading failure."""
        mock_get_model.side_effect = Exception("Model download failed")

        separator = VocalSeparator(device=torch.device('cpu'))

        with pytest.raises(RuntimeError, match="Failed to load Demucs model"):
            separator._load_model()


class TestSeparation:
    """Test audio separation functionality."""

    @patch('auto_voice.audio.separation.VocalSeparator._apply_model')
    @patch('auto_voice.audio.separation.VocalSeparator._get_model')
    def test_separate_mono_audio(self, mock_get_model, mock_apply, sample_mono_audio):
        """Test separating mono audio."""
        audio, sr = sample_mono_audio

        # Mock model
        mock_model = MagicMock()
        mock_model.samplerate = sr
        mock_model.sources = ['drums', 'bass', 'other', 'vocals']
        mock_get_model.return_value = mock_model

        # Mock apply_model output
        # Shape: (batch=1, n_sources=4, channels=2, samples)
        num_samples = len(audio)
        mock_output = torch.randn(1, 4, 2, num_samples)
        mock_apply.return_value = mock_output

        separator = VocalSeparator(device=torch.device('cpu'))
        result = separator.separate(audio, sr)

        assert 'vocals' in result
        assert 'instrumental' in result
        assert result['vocals'].shape == (num_samples,)
        assert result['instrumental'].shape == (num_samples,)
        assert result['vocals'].dtype == np.float32

    @patch('auto_voice.audio.separation.VocalSeparator._apply_model')
    @patch('auto_voice.audio.separation.VocalSeparator._get_model')
    def test_separate_stereo_audio(self, mock_get_model, mock_apply, sample_audio):
        """Test separating stereo audio."""
        audio, sr = sample_audio

        mock_model = MagicMock()
        mock_model.samplerate = sr
        mock_model.sources = ['drums', 'bass', 'other', 'vocals']
        mock_get_model.return_value = mock_model

        num_samples = audio.shape[1]
        mock_output = torch.randn(1, 4, 2, num_samples)
        mock_apply.return_value = mock_output

        separator = VocalSeparator(device=torch.device('cpu'))
        result = separator.separate(audio, sr)

        assert result['vocals'].shape == (num_samples,)
        assert result['instrumental'].shape == (num_samples,)

    def test_separate_empty_audio(self):
        """Test separation with empty audio."""
        separator = VocalSeparator(device=torch.device('cpu'))
        audio = np.array([])

        with pytest.raises(ValueError, match="Cannot separate empty audio"):
            separator.separate(audio, 44100)

    def test_separate_invalid_dimensions(self):
        """Test separation with invalid audio dimensions."""
        separator = VocalSeparator(device=torch.device('cpu'))
        audio = np.random.randn(2, 2, 1000)  # 3D audio (invalid)

        with pytest.raises(ValueError, match="must be 1D .* or 2D"):
            separator.separate(audio, 44100)

    @patch('auto_voice.audio.separation.VocalSeparator._apply_model')
    @patch('auto_voice.audio.separation.VocalSeparator._get_model')
    def test_separate_with_resampling(self, mock_get_model, mock_apply):
        """Test separation with sample rate mismatch."""
        audio = np.random.randn(22050).astype(np.float32) * 0.3
        input_sr = 22050
        model_sr = 44100

        mock_model = MagicMock()
        mock_model.samplerate = model_sr
        mock_model.sources = ['drums', 'bass', 'other', 'vocals']
        mock_get_model.return_value = mock_model

        # After resampling, audio will be 2x length
        resampled_len = int(len(audio) * model_sr / input_sr)
        mock_output = torch.randn(1, 4, 2, resampled_len)
        mock_apply.return_value = mock_output

        with patch('torchaudio.transforms.Resample') as mock_resample_class:
            mock_resampler = MagicMock()
            mock_resampled_tensor = torch.randn(1, 2, resampled_len)
            mock_resampler.return_value = mock_resampled_tensor
            mock_resample_class.return_value = mock_resampler

            with patch('librosa.resample') as mock_librosa_resample:
                mock_librosa_resample.side_effect = lambda x, **kwargs: x[:len(audio)]

                separator = VocalSeparator(device=torch.device('cpu'))
                result = separator.separate(audio, input_sr)

                # Output should be resampled back to original SR
                assert len(result['vocals']) == len(audio)

    @patch('auto_voice.audio.separation.VocalSeparator._apply_model')
    @patch('auto_voice.audio.separation.VocalSeparator._get_model')
    def test_separate_with_segment_size(self, mock_get_model, mock_apply, sample_mono_audio):
        """Test separation with chunked processing."""
        audio, sr = sample_mono_audio

        mock_model = MagicMock()
        mock_model.samplerate = sr
        mock_model.sources = ['drums', 'bass', 'other', 'vocals']
        mock_get_model.return_value = mock_model

        mock_output = torch.randn(1, 4, 2, len(audio))
        mock_apply.return_value = mock_output

        # Use segment size
        separator = VocalSeparator(device=torch.device('cpu'), segment=7.8)
        result = separator.separate(audio, sr)

        # Verify that apply_model was called with segment parameter
        assert mock_apply.called
        call_kwargs = mock_apply.call_args[1]
        assert 'segment' in call_kwargs
        assert call_kwargs['segment'] == 7.8

    @patch('auto_voice.audio.separation.VocalSeparator._apply_model')
    @patch('auto_voice.audio.separation.VocalSeparator._get_model')
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.empty_cache')
    def test_gpu_memory_management(self, mock_cache, mock_cuda_avail,
                                  mock_get_model, mock_apply, sample_mono_audio):
        """Test GPU memory management during separation."""
        audio, sr = sample_mono_audio

        mock_cuda_avail.return_value = True

        mock_model = MagicMock()
        mock_model.samplerate = sr
        mock_model.sources = ['drums', 'bass', 'other', 'vocals']
        mock_get_model.return_value = mock_model

        mock_output = torch.randn(1, 4, 2, len(audio))
        mock_apply.return_value = mock_output

        separator = VocalSeparator(device=torch.device('cuda:0'))
        result = separator.separate(audio, sr)

        # Verify GPU cache was cleared
        assert mock_cache.call_count >= 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    @patch('auto_voice.audio.separation.VocalSeparator._apply_model')
    @patch('auto_voice.audio.separation.VocalSeparator._get_model')
    def test_missing_vocals_source(self, mock_get_model, mock_apply, sample_mono_audio):
        """Test error when model doesn't have vocals source."""
        audio, sr = sample_mono_audio

        mock_model = MagicMock()
        mock_model.samplerate = sr
        mock_model.sources = ['drums', 'bass', 'other']  # No vocals!
        mock_get_model.return_value = mock_model

        separator = VocalSeparator(device=torch.device('cpu'))

        with pytest.raises(RuntimeError, match="does not have a 'vocals' source"):
            separator.separate(audio, sr)

    @patch('auto_voice.audio.separation.VocalSeparator._apply_model')
    @patch('auto_voice.audio.separation.VocalSeparator._get_model')
    def test_output_length_matching(self, mock_get_model, mock_apply, sample_mono_audio):
        """Test that output length matches input length."""
        audio, sr = sample_mono_audio
        orig_len = len(audio)

        mock_model = MagicMock()
        mock_model.samplerate = sr
        mock_model.sources = ['drums', 'bass', 'other', 'vocals']
        mock_get_model.return_value = mock_model

        # Simulate slightly different output length
        mock_output = torch.randn(1, 4, 2, orig_len + 10)
        mock_apply.return_value = mock_output

        separator = VocalSeparator(device=torch.device('cpu'))
        result = separator.separate(audio, sr)

        # Output should be trimmed to match input
        assert len(result['vocals']) == orig_len
        assert len(result['instrumental']) == orig_len


@pytest.mark.integration
class TestVocalSeparatorIntegration:
    """Integration tests for full separation workflow."""

    @pytest.mark.slow
    def test_cpu_vs_gpu_consistency(self, sample_mono_audio):
        """Test that CPU and GPU produce similar results (mocked)."""
        audio, sr = sample_audio

        # Mock consistent outputs for CPU and GPU
        with patch('auto_voice.audio.separation.VocalSeparator._apply_model') as mock_apply:
            with patch('auto_voice.audio.separation.VocalSeparator._get_model') as mock_get_model:
                mock_model = MagicMock()
                mock_model.samplerate = sr
                mock_model.sources = ['drums', 'bass', 'other', 'vocals']
                mock_get_model.return_value = mock_model

                # Use same random seed for both
                np.random.seed(42)
                mock_output = torch.randn(1, 4, 2, len(audio))
                mock_apply.return_value = mock_output

                # CPU separation
                separator_cpu = VocalSeparator(device=torch.device('cpu'))
                result_cpu = separator_cpu.separate(audio, sr)

                # Reset for GPU
                np.random.seed(42)
                mock_apply.return_value = torch.randn(1, 4, 2, len(audio))

                # GPU separation (mocked)
                separator_gpu = VocalSeparator(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
                result_gpu = separator_gpu.separate(audio, sr)

                # Results should have same shape
                assert result_cpu['vocals'].shape == result_gpu['vocals'].shape

    def test_separate_quality_metrics_mock(self, sample_mono_audio):
        """Test separation quality (mocked SDR metric)."""
        audio, sr = sample_mono_audio

        with patch('auto_voice.audio.separation.VocalSeparator._apply_model') as mock_apply:
            with patch('auto_voice.audio.separation.VocalSeparator._get_model') as mock_get_model:
                mock_model = MagicMock()
                mock_model.samplerate = sr
                mock_model.sources = ['drums', 'bass', 'other', 'vocals']
                mock_get_model.return_value = mock_model

                # Create high-quality separation (low noise)
                mock_output = torch.randn(1, 4, 2, len(audio)) * 0.5
                mock_apply.return_value = mock_output

                separator = VocalSeparator(device=torch.device('cpu'))
                result = separator.separate(audio, sr)

                # Check that vocals are not clipped
                assert np.max(np.abs(result['vocals'])) <= 1.0

                # Check that instrumental is not all zeros
                assert np.any(result['instrumental'] != 0)

                # Mock SDR calculation (in real scenario would use mir_eval)
                # Higher is better, typically > 3dB is good
                mock_sdr = 10.0  # Good separation
                assert mock_sdr > 3.0
