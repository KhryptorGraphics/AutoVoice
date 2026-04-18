"""Current-contract tests for separation.py."""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

from auto_voice.audio.separation import VocalSeparator


@pytest.fixture
def sample_audio():
    """Create stereo audio."""
    sr = 44100
    duration = 3.0
    samples = int(duration * sr)
    audio = np.random.randn(2, samples).astype(np.float32) * 0.3
    return audio, sr


@pytest.fixture
def sample_mono_audio():
    """Create mono audio."""
    sr = 44100
    duration = 3.0
    samples = int(duration * sr)
    audio = np.random.randn(samples).astype(np.float32) * 0.3
    return audio, sr


def _build_separator(*, model_sr: int = 44100, sources=None, device=None, segment=None):
    """Create a separator with mocked demucs hooks."""
    separator = VocalSeparator(device=device or torch.device('cpu'), segment=segment)
    model = MagicMock()
    model.samplerate = model_sr
    model.sources = sources or ['drums', 'bass', 'other', 'vocals']
    separator._get_model = MagicMock(return_value=model)
    separator._apply_model = MagicMock()
    return separator, model


class TestVocalSeparatorInit:
    """Initialization tests."""

    @pytest.mark.smoke
    def test_init_default(self):
        separator = VocalSeparator()
        assert separator.device.type in ['cuda', 'cpu']
        assert separator.model_name == 'htdemucs'
        assert separator.segment is None

    def test_init_custom_device(self):
        separator = VocalSeparator(device=torch.device('cpu'))
        assert separator.device.type == 'cpu'

    def test_init_custom_model(self):
        separator = VocalSeparator(model_name='htdemucs_ft', device=torch.device('cpu'))
        assert separator.model_name == 'htdemucs_ft'

    def test_init_with_segment(self):
        separator = VocalSeparator(segment=7.8, device=torch.device('cpu'))
        assert separator.segment == 7.8

    @patch('auto_voice.audio.separation.torch.cuda.is_available')
    def test_init_auto_device_selection(self, mock_cuda):
        mock_cuda.return_value = False
        assert VocalSeparator().device.type == 'cpu'

        mock_cuda.return_value = True
        assert VocalSeparator().device.type == 'cuda'


class TestModelLoading:
    """Model loading tests."""

    def test_lazy_model_loading(self):
        separator, model = _build_separator()
        assert separator._model is None
        assert separator.model_sample_rate == 44100
        assert separator._model is model
        separator._get_model.assert_called_once()

    def test_model_sources_property(self):
        separator, _ = _build_separator(sources=['drums', 'bass', 'other', 'vocals'])
        assert separator.sources == ['drums', 'bass', 'other', 'vocals']

    def test_load_model_failure(self):
        separator = VocalSeparator(device=torch.device('cpu'))
        separator._get_model = MagicMock(side_effect=Exception("Model download failed"))

        with pytest.raises(RuntimeError, match="Failed to load Demucs model"):
            separator._load_model()


class TestSeparation:
    """Core separation behavior."""

    def test_separate_mono_audio(self, sample_mono_audio):
        audio, sr = sample_mono_audio
        separator, _ = _build_separator(model_sr=sr)
        separator._apply_model.return_value = torch.randn(1, 4, 2, len(audio))

        result = separator.separate(audio, sr)

        assert result['vocals'].shape == (len(audio),)
        assert result['instrumental'].shape == (len(audio),)
        assert result['vocals'].dtype == np.float32
        assert result['instrumental'].dtype == np.float32

    def test_separate_stereo_audio(self, sample_audio):
        audio, sr = sample_audio
        separator, _ = _build_separator(model_sr=sr)
        separator._apply_model.return_value = torch.randn(1, 4, 2, audio.shape[1])

        result = separator.separate(audio, sr)

        assert result['vocals'].shape == (audio.shape[1],)
        assert result['instrumental'].shape == (audio.shape[1],)

    def test_separate_empty_audio(self):
        separator = VocalSeparator(device=torch.device('cpu'))
        with pytest.raises(ValueError, match="Cannot separate empty audio"):
            separator.separate(np.array([]), 44100)

    def test_separate_invalid_dimensions(self):
        separator = VocalSeparator(device=torch.device('cpu'))
        with pytest.raises(ValueError, match="must be 1D .* or 2D"):
            separator.separate(np.random.randn(2, 2, 1000).astype(np.float32), 44100)

    def test_separate_with_resampling(self):
        input_sr = 22050
        audio = np.random.randn(input_sr).astype(np.float32) * 0.3
        resampled_len = input_sr * 2
        separator, _ = _build_separator(model_sr=44100)
        separator._apply_model.return_value = torch.randn(1, 4, 2, resampled_len)

        with patch('torchaudio.transforms.Resample') as mock_resample_class, \
             patch('librosa.resample') as mock_librosa_resample:
            mock_resampler = MagicMock()
            mock_resampled_tensor = torch.randn(1, 2, resampled_len)
            mock_resampler.return_value = mock_resampled_tensor
            mock_resample_class.return_value = mock_resampler
            mock_librosa_resample.side_effect = lambda x, **kwargs: x[:len(audio)]

            result = separator.separate(audio, input_sr)

        assert len(result['vocals']) == len(audio)
        assert len(result['instrumental']) == len(audio)

    def test_separate_with_segment_size(self, sample_mono_audio):
        audio, sr = sample_mono_audio
        separator, _ = _build_separator(model_sr=sr, segment=7.8)
        separator._apply_model.return_value = torch.randn(1, 4, 2, len(audio))

        separator.separate(audio, sr)

        assert separator._apply_model.called
        assert separator._apply_model.call_args.kwargs['segment'] == 7.8

    def test_missing_vocals_source_raises(self, sample_mono_audio):
        audio, sr = sample_mono_audio
        separator, _ = _build_separator(model_sr=sr, sources=['drums', 'bass', 'other'])
        separator._apply_model.return_value = torch.randn(1, 3, 2, len(audio))

        with pytest.raises(RuntimeError, match="does not have a 'vocals' source"):
            separator.separate(audio, sr)

    def test_output_length_matching_padding(self, sample_mono_audio):
        audio, sr = sample_mono_audio
        separator, _ = _build_separator(model_sr=sr)
        separator._apply_model.return_value = torch.randn(1, 4, 2, len(audio) - 25)

        result = separator.separate(audio, sr)

        assert len(result['vocals']) == len(audio)
        assert len(result['instrumental']) == len(audio)

    def test_output_length_matching_trimming(self, sample_mono_audio):
        audio, sr = sample_mono_audio
        separator, _ = _build_separator(model_sr=sr)
        separator._apply_model.return_value = torch.randn(1, 4, 2, len(audio) + 25)

        result = separator.separate(audio, sr)

        assert len(result['vocals']) == len(audio)
        assert len(result['instrumental']) == len(audio)
