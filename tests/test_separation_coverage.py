"""Coverage tests for VocalSeparator — mock Demucs entirely."""
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_demucs():
    """Mock demucs pretrained and apply modules."""
    mock_model = MagicMock()
    mock_model.sources = ['drums', 'bass', 'other', 'vocals']
    mock_model.samplerate = 44100
    mock_get = MagicMock(return_value=mock_model)
    mock_apply = MagicMock(return_value=torch.randn(1, 4, 2, 44100))

    # Patch at module level
    import auto_voice.audio.separation as sep_mod
    original_get = sep_mod.get_model
    original_apply = sep_mod.apply_model
    sep_mod.get_model = mock_get
    sep_mod.apply_model = mock_apply
    yield {'get_model': mock_get, 'apply_model': mock_apply, 'model': mock_model}
    sep_mod.get_model = original_get
    sep_mod.apply_model = original_apply


@pytest.fixture
def separator(mock_demucs):
    """Create VocalSeparator with mocked demucs."""
    from auto_voice.audio.separation import VocalSeparator
    return VocalSeparator(device=torch.device('cpu'))


class TestVocalSeparatorInit:
    def test_init_with_mock(self, mock_demucs):
        from auto_voice.audio.separation import VocalSeparator
        sep = VocalSeparator(device=torch.device('cpu'))
        assert sep.device == torch.device('cpu')
        assert sep.model_name == 'htdemucs'

    def test_init_custom_model(self, mock_demucs):
        from auto_voice.audio.separation import VocalSeparator
        sep = VocalSeparator(device=torch.device('cpu'), model_name='htdemucs_ft')
        assert sep.model_name == 'htdemucs_ft'

    def test_init_with_segment(self, mock_demucs):
        from auto_voice.audio.separation import VocalSeparator
        sep = VocalSeparator(device=torch.device('cpu'), segment=10.0)
        assert sep.segment == 10.0

    def test_init_no_demucs_raises(self):
        """Without demucs installed, raises RuntimeError."""
        import auto_voice.audio.separation as sep_mod
        # Save and clear
        orig_get = sep_mod.get_model
        orig_apply = sep_mod.apply_model
        # Clear to trigger the import path
        sep_mod.get_model = None
        sep_mod.apply_model = None
        try:
            # Need to make the import fail
            with patch.dict('sys.modules', {'demucs': None, 'demucs.pretrained': None, 'demucs.apply': None}):
                from auto_voice.audio.separation import VocalSeparator
                with pytest.raises(RuntimeError, match="Demucs is required"):
                    VocalSeparator(device=torch.device('cpu'))
        finally:
            sep_mod.get_model = orig_get
            sep_mod.apply_model = orig_apply

    def test_default_device_selection(self, mock_demucs):
        from auto_voice.audio.separation import VocalSeparator
        sep = VocalSeparator()
        assert sep.device.type in ('cpu', 'cuda')

    def test_string_device(self, mock_demucs):
        from auto_voice.audio.separation import VocalSeparator
        sep = VocalSeparator(device='cpu')
        assert sep.device == torch.device('cpu')


class TestVocalSeparatorModel:
    def test_model_sample_rate(self, separator, mock_demucs):
        sr = separator.model_sample_rate
        assert sr == 44100

    def test_model_sources(self, separator, mock_demucs):
        sources = separator.sources
        assert 'vocals' in sources

    def test_lazy_load_only_once(self, separator, mock_demucs):
        """Model loads lazily and only once."""
        separator._load_model()
        first_model = separator._model
        separator._load_model()
        assert separator._model is first_model

    def test_model_load_failure_raises(self, mock_demucs):
        from auto_voice.audio.separation import VocalSeparator
        mock_demucs['get_model'].side_effect = Exception("Model not found")
        sep = VocalSeparator(device=torch.device('cpu'))
        with pytest.raises(RuntimeError, match="Failed to load"):
            _ = sep.model_sample_rate


class TestVocalSeparatorSeparate:
    def test_separate_mono(self, separator, mock_demucs):
        audio = np.random.randn(44100).astype(np.float32)
        result = separator.separate(audio, 44100)
        assert 'vocals' in result
        assert 'instrumental' in result

    def test_separate_stereo(self, separator, mock_demucs):
        audio = np.random.randn(2, 44100).astype(np.float32)
        result = separator.separate(audio, 44100)
        assert 'vocals' in result

    def test_separate_empty_raises(self, separator):
        audio = np.array([], dtype=np.float32)
        with pytest.raises(ValueError, match="empty"):
            separator.separate(audio, 44100)

    def test_separate_invalid_dims_raises(self, separator):
        audio = np.random.randn(1, 2, 44100).astype(np.float32)
        with pytest.raises(ValueError, match="mono.*or.*2D.*stereo"):
            separator.separate(audio, 44100)

    def test_separate_with_resample(self, separator, mock_demucs):
        """Separation resamples if input sr != model sr."""
        audio = np.random.randn(22050).astype(np.float32)
        result = separator.separate(audio, 22050)
        assert 'vocals' in result

    def test_separate_with_segment(self, separator, mock_demucs):
        """Segment parameter passed to apply_model."""
        sep_with_segment = type(separator)(
            device=torch.device('cpu'),
            model_name='htdemucs',
            segment=5.0
        )
        sep_with_segment._load_model()
        audio = np.random.randn(44100).astype(np.float32)
        result = sep_with_segment.separate(audio, 44100)
        assert 'vocals' in result

    def test_separate_output_types(self, separator, mock_demucs):
        audio = np.random.randn(44100).astype(np.float32)
        result = separator.separate(audio, 44100)
        assert isinstance(result['vocals'], np.ndarray)
        assert isinstance(result['instrumental'], np.ndarray)

    def test_separate_output_mono(self, separator, mock_demucs):
        """Output is always mono (1D)."""
        audio = np.random.randn(44100).astype(np.float32)
        result = separator.separate(audio, 44100)
        assert result['vocals'].ndim == 1
        assert result['instrumental'].ndim == 1
