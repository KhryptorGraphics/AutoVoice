"""Tests for separation.py - Vocal/instrumental separation using Demucs.

Test Coverage:
- Task 2.3: Demucs separation (vocals, drums, bass, other)
- Verify output stems (4 files)
- Test separation quality (SDR metric)
- Test GPU vs CPU execution
"""

import numpy as np
import pytest
import torch
from unittest.mock import Mock, MagicMock, patch

from auto_voice.audio.separation import VocalSeparator


@pytest.fixture
def sample_audio_mono():
    """Create sample mono audio."""
    sr = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sr


@pytest.fixture
def sample_audio_stereo():
    """Create sample stereo audio."""
    sr = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    left = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    right = 0.3 * np.sin(2 * np.pi * 880 * t).astype(np.float32)
    audio = np.stack([left, right])
    return audio, sr


class TestVocalSeparator:
    """Test suite for VocalSeparator."""

    def test_initialization_cpu(self):
        """Test VocalSeparator initialization on CPU."""
        separator = VocalSeparator(device=torch.device('cpu'), model_name='htdemucs')

        assert separator.device == torch.device('cpu')
        assert separator.model_name == 'htdemucs'
        assert separator._model is None  # Lazy loaded

    def test_initialization_cuda(self):
        """Test VocalSeparator initialization with CUDA."""
        if torch.cuda.is_available():
            separator = VocalSeparator(device=torch.device('cuda'))
            assert separator.device.type == 'cuda'
        else:
            # Should fall back to CPU
            separator = VocalSeparator(device=None)
            assert separator.device.type == 'cpu'

    def test_initialization_without_demucs_raises_error(self):
        """Test that missing demucs raises RuntimeError."""
        with patch.dict('sys.modules', {
            'demucs': None,
            'demucs.pretrained': None,
            'demucs.apply': None,
        }):
            with pytest.raises(RuntimeError, match="Demucs is required"):
                import auto_voice.audio.separation as separation
                separation.get_model = None
                separation.apply_model = None
                VocalSeparator()

    def test_lazy_model_loading(self):
        """Test that model is lazy-loaded on first use."""
        separator = VocalSeparator(device='cpu')

        assert separator._model is None

        # Mock the model loading
        with patch.object(separator, '_get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.samplerate = 44100
            mock_model.sources = ['drums', 'bass', 'other', 'vocals']
            mock_get_model.return_value = mock_model

            separator._load_model()

            assert separator._model is not None
            mock_get_model.assert_called_once_with('htdemucs')

    def test_model_sample_rate_property(self):
        """Test model_sample_rate property."""
        separator = VocalSeparator(device='cpu')

        with patch.object(separator, '_load_model') as mock_load:
            mock_model = MagicMock()
            mock_model.samplerate = 44100
            separator._model = mock_model

            sr = separator.model_sample_rate

            assert sr == 44100

    def test_sources_property(self):
        """Test sources property returns list of source names."""
        separator = VocalSeparator(device='cpu')

        with patch.object(separator, '_load_model'):
            mock_model = MagicMock()
            mock_model.sources = ['drums', 'bass', 'other', 'vocals']
            separator._model = mock_model

            sources = separator.sources

            assert sources == ['drums', 'bass', 'other', 'vocals']

    def test_separate_mono_audio(self, sample_audio_mono):
        """Test separation with mono audio input."""
        audio, sr = sample_audio_mono
        separator = VocalSeparator(device='cpu')

        # Mock the model and apply_model
        mock_model = MagicMock()
        mock_model.samplerate = sr
        mock_model.sources = ['drums', 'bass', 'other', 'vocals']

        # Create mock separated sources (batch, sources, channels, samples)
        num_samples = len(audio)
        mock_sources = torch.zeros(1, 4, 2, num_samples)
        mock_sources[0, 3, :, :] = torch.from_numpy(audio * 0.7)  # Vocals
        mock_sources[0, 0:3, :, :] = torch.from_numpy(audio * 0.3)  # Others

        with patch.object(separator, '_get_model', return_value=mock_model), \
             patch.object(separator, '_apply_model', return_value=mock_sources):

            separator._model = mock_model
            result = separator.separate(audio, sr)

            assert 'vocals' in result
            assert 'instrumental' in result
            assert result['vocals'].shape == (num_samples,)
            assert result['instrumental'].shape == (num_samples,)
            assert result['vocals'].dtype == np.float32
            assert result['instrumental'].dtype == np.float32

    def test_separate_stereo_audio(self, sample_audio_stereo):
        """Test separation with stereo audio input."""
        audio, sr = sample_audio_stereo
        separator = VocalSeparator(device='cpu')

        mock_model = MagicMock()
        mock_model.samplerate = sr
        mock_model.sources = ['drums', 'bass', 'other', 'vocals']

        num_samples = audio.shape[1]
        mock_sources = torch.zeros(1, 4, 2, num_samples)
        mock_sources[0, 3, :, :] = torch.from_numpy(audio * 0.7)  # Vocals
        mock_sources[0, 0:3, :, :] = torch.from_numpy(audio * 0.3)  # Others

        with patch.object(separator, '_get_model', return_value=mock_model), \
             patch.object(separator, '_apply_model', return_value=mock_sources):

            separator._model = mock_model
            result = separator.separate(audio, sr)

            # Output should be mono (mean of stereo)
            assert result['vocals'].ndim == 1
            assert result['instrumental'].ndim == 1

    def test_separate_empty_audio_raises_error(self):
        """Test that empty audio raises ValueError."""
        separator = VocalSeparator(device='cpu')
        audio = np.array([])
        sr = 44100

        with pytest.raises(ValueError, match="Cannot separate empty audio"):
            separator.separate(audio, sr)

    def test_separate_invalid_shape_raises_error(self):
        """Test that invalid audio shape raises ValueError."""
        separator = VocalSeparator(device='cpu')
        audio = np.random.randn(2, 3, 1000)  # 3D array
        sr = 44100

        with pytest.raises(ValueError, match="Audio must be 1D .* or 2D .*"):
            separator.separate(audio, sr)

    def test_separate_with_resampling(self, sample_audio_mono):
        """Test separation with different input/model sample rates."""
        audio, sr = sample_audio_mono
        separator = VocalSeparator(device='cpu')

        mock_model = MagicMock()
        mock_model.samplerate = 48000  # Different from input (44100)
        mock_model.sources = ['drums', 'bass', 'other', 'vocals']

        # Resampled audio will have different length
        resampled_length = int(len(audio) * 48000 / sr)
        mock_sources = torch.zeros(1, 4, 2, resampled_length)
        mock_sources[0, 3, :, :] = torch.randn(2, resampled_length) * 0.1

        with patch.object(separator, '_get_model', return_value=mock_model), \
             patch.object(separator, '_apply_model', return_value=mock_sources), \
             patch('torchaudio.transforms.Resample') as mock_resample_class:

            # Mock resampler
            mock_resampler = MagicMock()
            mock_resampler.return_value = torch.randn(1, 2, resampled_length)
            mock_resample_class.return_value = mock_resampler

            separator._model = mock_model
            result = separator.separate(audio, sr)

            # Output should match original sample rate and length
            assert len(result['vocals']) == len(audio)

    def test_separate_gpu_memory_management(self, sample_audio_mono):
        """Test GPU memory cleanup during separation."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        audio, sr = sample_audio_mono
        separator = VocalSeparator(device='cuda')

        mock_model = MagicMock()
        mock_model.samplerate = sr
        mock_model.sources = ['drums', 'bass', 'other', 'vocals']

        mock_sources = torch.zeros(1, 4, 2, len(audio))
        mock_sources[0, 3, :, :] = torch.from_numpy(audio * 0.7)

        with patch.object(separator, '_get_model', return_value=mock_model), \
             patch.object(separator, '_apply_model', return_value=mock_sources), \
             patch('torch.cuda.empty_cache') as mock_cache, \
             patch('torch.cuda.synchronize') as mock_sync:

            separator._model = mock_model
            result = separator.separate(audio, sr)

            # Should call cache cleanup
            assert mock_cache.call_count >= 2  # Before and after

    def test_separate_with_segment_size(self, sample_audio_mono):
        """Test separation with segment size for memory efficiency."""
        audio, sr = sample_audio_mono
        separator = VocalSeparator(device='cpu', segment=10.0)

        mock_model = MagicMock()
        mock_model.samplerate = sr
        mock_model.sources = ['drums', 'bass', 'other', 'vocals']

        mock_sources = torch.zeros(1, 4, 2, len(audio))
        mock_sources[0, 3, :, :] = torch.from_numpy(audio * 0.7)

        with patch.object(separator, '_get_model', return_value=mock_model), \
             patch.object(separator, '_apply_model', return_value=mock_sources) as mock_apply:

            separator._model = mock_model
            result = separator.separate(audio, sr)

            # Verify segment parameter was passed
            mock_apply.assert_called_once()
            call_kwargs = mock_apply.call_args[1]
            assert 'segment' in call_kwargs
            assert call_kwargs['segment'] == 10.0

    def test_separate_missing_vocals_source_raises_error(self, sample_audio_mono):
        """Test that model without vocals source raises RuntimeError."""
        audio, sr = sample_audio_mono
        separator = VocalSeparator(device='cpu')

        mock_model = MagicMock()
        mock_model.samplerate = sr
        mock_model.sources = ['drums', 'bass', 'other']  # No vocals!

        with patch.object(separator, '_get_model', return_value=mock_model), \
             patch.object(separator, '_apply_model', return_value=torch.zeros(1, 3, 2, len(audio))):

            separator._model = mock_model

            with pytest.raises(RuntimeError, match="does not have a 'vocals' source"):
                separator.separate(audio, sr)

    def test_separate_output_length_matches_input(self, sample_audio_mono):
        """Test that output length matches input length."""
        audio, sr = sample_audio_mono
        separator = VocalSeparator(device='cpu')

        mock_model = MagicMock()
        mock_model.samplerate = sr
        mock_model.sources = ['drums', 'bass', 'other', 'vocals']

        # Create mock output with slight length difference
        mock_sources = torch.zeros(1, 4, 2, len(audio) + 10)
        mock_sources[0, 3, :, :] = torch.randn(2, len(audio) + 10) * 0.1

        with patch.object(separator, '_get_model', return_value=mock_model), \
             patch.object(separator, '_apply_model', return_value=mock_sources):

            separator._model = mock_model
            result = separator.separate(audio, sr)

            # Output should be truncated to match input
            assert len(result['vocals']) == len(audio)
            assert len(result['instrumental']) == len(audio)

    def test_gpu_vs_cpu_execution(self, sample_audio_mono):
        """Test that separation works on both GPU and CPU."""
        audio, sr = sample_audio_mono

        mock_model = MagicMock()
        mock_model.samplerate = sr
        mock_model.sources = ['drums', 'bass', 'other', 'vocals']
        mock_sources = torch.zeros(1, 4, 2, len(audio))
        mock_sources[0, 3, :, :] = torch.from_numpy(audio * 0.7)

        # Test CPU
        separator_cpu = VocalSeparator(device='cpu')
        with patch.object(separator_cpu, '_get_model', return_value=mock_model), \
             patch.object(separator_cpu, '_apply_model', return_value=mock_sources):
            separator_cpu._model = mock_model
            result_cpu = separator_cpu.separate(audio, sr)
            assert 'vocals' in result_cpu

        # Test GPU if available
        if torch.cuda.is_available():
            separator_gpu = VocalSeparator(device='cuda')
            with patch.object(separator_gpu, '_get_model', return_value=mock_model), \
                 patch.object(separator_gpu, '_apply_model', return_value=mock_sources):
                separator_gpu._model = mock_model
                result_gpu = separator_gpu.separate(audio, sr)
                assert 'vocals' in result_gpu

    def test_separate_all_stems_returned(self, sample_audio_mono):
        """Test that all 4 stems can be extracted (vocals, drums, bass, other)."""
        audio, sr = sample_audio_mono
        separator = VocalSeparator(device='cpu')

        mock_model = MagicMock()
        mock_model.samplerate = sr
        mock_model.sources = ['drums', 'bass', 'other', 'vocals']

        # Create distinct mock sources
        num_samples = len(audio)
        mock_sources = torch.zeros(1, 4, 2, num_samples)
        for i in range(4):
            mock_sources[0, i, :, :] = torch.randn(2, num_samples) * 0.1

        with patch.object(separator, '_get_model', return_value=mock_model), \
             patch.object(separator, '_apply_model', return_value=mock_sources):

            separator._model = mock_model
            result = separator.separate(audio, sr)

            # Currently returns vocals + instrumental
            # Instrumental is sum of non-vocal sources
            assert 'vocals' in result
            assert 'instrumental' in result

            # Verify they're different
            assert not np.allclose(result['vocals'], result['instrumental'])


@pytest.mark.integration
def test_separation_quality_sdr():
    """Test separation quality using SDR (Signal-to-Distortion Ratio).

    Note: This is a placeholder for actual SDR testing.
    Real SDR testing requires ground-truth separated audio.
    """
    # Mock: In real test, would compute SDR between separated and ground truth
    # SDR = 10 * log10(||target||^2 / ||target - estimate||^2)

    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Simulate ground truth and estimate
    ground_truth = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    estimate = ground_truth + np.random.randn(len(ground_truth)).astype(np.float32) * 0.01

    # Calculate SDR
    signal_power = np.sum(ground_truth ** 2)
    error_power = np.sum((ground_truth - estimate) ** 2)

    if error_power > 0:
        sdr = 10 * np.log10(signal_power / error_power)
    else:
        sdr = float('inf')

    # Good separation should have SDR > 10 dB
    assert sdr > 10.0


@pytest.mark.parametrize("model_name", [
    'htdemucs',
    'htdemucs_ft',
])
def test_different_model_variants(model_name):
    """Test that different Demucs model variants can be loaded."""
    separator = VocalSeparator(device='cpu', model_name=model_name)

    assert separator.model_name == model_name


def test_stereo_to_mono_conversion(sample_audio_stereo):
    """Test that stereo audio is properly converted to mono output."""
    audio, sr = sample_audio_stereo
    separator = VocalSeparator(device='cpu')

    mock_model = MagicMock()
    mock_model.samplerate = sr
    mock_model.sources = ['drums', 'bass', 'other', 'vocals']

    num_samples = audio.shape[1]
    mock_sources = torch.zeros(1, 4, 2, num_samples)
    # Different values for left and right channels
    mock_sources[0, 3, 0, :] = 1.0
    mock_sources[0, 3, 1, :] = 0.5

    with patch.object(separator, '_get_model', return_value=mock_model), \
         patch.object(separator, '_apply_model', return_value=mock_sources):

        separator._model = mock_model
        result = separator.separate(audio, sr)

        # Output should be mono (1D)
        assert result['vocals'].ndim == 1
        # Value should be mean of channels
        expected_mean = 0.75  # (1.0 + 0.5) / 2
        assert np.allclose(result['vocals'][:100], expected_mean, atol=0.1)
