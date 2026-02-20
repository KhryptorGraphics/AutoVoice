"""Comprehensive tests for audio separation module - targeting 90% coverage.

Tests VocalSeparator class with full coverage of:
- Initialization and error handling
- Model loading (lazy and explicit)
- Audio separation with various formats
- Resampling and format conversion
- Memory management and GPU cleanup
- Edge cases and error conditions
"""
import pytest
import numpy as np
import torch
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from pathlib import Path


# Test suite for VocalSeparator initialization and setup
class TestVocalSeparatorInitialization:
    """Test VocalSeparator initialization and configuration."""

    def test_init_missing_demucs_raises_error(self):
        """Test that missing demucs package raises RuntimeError with clear message."""
        with patch.dict('sys.modules', {'demucs': None}):
            with pytest.raises(RuntimeError, match="Demucs is required"):
                from auto_voice.audio.separation import VocalSeparator
                VocalSeparator()

    def test_init_demucs_import_error_includes_install_hint(self):
        """Test error message includes installation instructions."""
        # Mock the demucs import to fail
        import sys
        original_import = __builtins__.__import__

        def mock_import(name, *args, **kwargs):
            if 'demucs' in name:
                raise ImportError("No module named 'demucs'")
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            with pytest.raises(RuntimeError, match="pip install demucs"):
                # Clear cached module if exists
                if 'auto_voice.audio.separation' in sys.modules:
                    del sys.modules['auto_voice.audio.separation']
                from auto_voice.audio.separation import VocalSeparator
                VocalSeparator()

    @patch('auto_voice.audio.separation.torch.cuda.is_available', return_value=True)
    @patch('demucs.pretrained.get_model')
    @patch('demucs.apply.apply_model')
    def test_init_with_default_device_cuda(self, mock_apply, mock_get, mock_cuda):
        """Test default device selection prefers CUDA when available."""
        from auto_voice.audio.separation import VocalSeparator
        separator = VocalSeparator()
        assert separator.device.type == 'cuda'

    @patch('auto_voice.audio.separation.torch.cuda.is_available', return_value=False)
    @patch('demucs.pretrained.get_model')
    @patch('demucs.apply.apply_model')
    def test_init_with_default_device_cpu(self, mock_apply, mock_get, mock_cuda):
        """Test default device selection falls back to CPU."""
        from auto_voice.audio.separation import VocalSeparator
        separator = VocalSeparator()
        assert separator.device.type == 'cpu'

    @patch('demucs.pretrained.get_model')
    @patch('demucs.apply.apply_model')
    def test_init_with_explicit_device(self, mock_apply, mock_get):
        """Test initialization with explicit device selection."""
        from auto_voice.audio.separation import VocalSeparator
        device = torch.device('cpu')
        separator = VocalSeparator(device=device)
        assert separator.device == device

    @patch('demucs.pretrained.get_model')
    @patch('demucs.apply.apply_model')
    def test_init_with_custom_model_name(self, mock_apply, mock_get):
        """Test initialization with custom Demucs model."""
        from auto_voice.audio.separation import VocalSeparator
        separator = VocalSeparator(model_name='htdemucs_ft')
        assert separator.model_name == 'htdemucs_ft'

    @patch('demucs.pretrained.get_model')
    @patch('demucs.apply.apply_model')
    def test_init_with_segment_size(self, mock_apply, mock_get):
        """Test initialization with custom segment size for chunked processing."""
        from auto_voice.audio.separation import VocalSeparator
        separator = VocalSeparator(segment=10.0)
        assert separator.segment == 10.0

    @patch('demucs.pretrained.get_model')
    @patch('demucs.apply.apply_model')
    def test_init_model_not_loaded_immediately(self, mock_apply, mock_get):
        """Test that model is not loaded during __init__ (lazy loading)."""
        from auto_voice.audio.separation import VocalSeparator
        separator = VocalSeparator()
        mock_get.assert_not_called()
        assert separator._model is None


class TestVocalSeparatorModelLoading:
    """Test model loading behavior and error handling."""

    @patch('demucs.pretrained.get_model')
    @patch('demucs.apply.apply_model')
    def test_load_model_success(self, mock_apply, mock_get_model):
        """Test successful model loading."""
        # Setup mock model
        mock_model = MagicMock()
        mock_model.sources = ['vocals', 'drums', 'bass', 'other']
        mock_model.samplerate = 44100
        mock_get_model.return_value = mock_model

        from auto_voice.audio.separation import VocalSeparator
        separator = VocalSeparator()
        separator._load_model()

        mock_get_model.assert_called_once_with('htdemucs')
        mock_model.to.assert_called_once()
        mock_model.eval.assert_called_once()
        assert separator._model == mock_model

    @patch('demucs.pretrained.get_model')
    @patch('demucs.apply.apply_model')
    def test_load_model_only_once(self, mock_apply, mock_get_model):
        """Test model is loaded only once (caching)."""
        mock_model = MagicMock()
        mock_model.sources = ['vocals', 'drums', 'bass', 'other']
        mock_model.samplerate = 44100
        mock_get_model.return_value = mock_model

        from auto_voice.audio.separation import VocalSeparator
        separator = VocalSeparator()
        separator._load_model()
        separator._load_model()
        separator._load_model()

        # Should be called only once despite multiple _load_model calls
        assert mock_get_model.call_count == 1

    @patch('demucs.pretrained.get_model')
    @patch('demucs.apply.apply_model')
    def test_load_model_failure_raises_runtime_error(self, mock_apply, mock_get_model):
        """Test that model loading failure raises RuntimeError."""
        mock_get_model.side_effect = Exception("Model download failed")

        from auto_voice.audio.separation import VocalSeparator
        separator = VocalSeparator()

        with pytest.raises(RuntimeError, match="Failed to load Demucs model"):
            separator._load_model()

    @patch('demucs.pretrained.get_model')
    @patch('demucs.apply.apply_model')
    def test_model_sample_rate_property(self, mock_apply, mock_get_model):
        """Test model_sample_rate property triggers lazy loading."""
        mock_model = MagicMock()
        mock_model.sources = ['vocals', 'drums', 'bass', 'other']
        mock_model.samplerate = 44100
        mock_get_model.return_value = mock_model

        from auto_voice.audio.separation import VocalSeparator
        separator = VocalSeparator()
        sr = separator.model_sample_rate

        assert sr == 44100
        mock_get_model.assert_called_once()

    @patch('demucs.pretrained.get_model')
    @patch('demucs.apply.apply_model')
    def test_sources_property(self, mock_apply, mock_get_model):
        """Test sources property returns list of source names."""
        mock_model = MagicMock()
        mock_model.sources = ['vocals', 'drums', 'bass', 'other']
        mock_model.samplerate = 44100
        mock_get_model.return_value = mock_model

        from auto_voice.audio.separation import VocalSeparator
        separator = VocalSeparator()
        sources = separator.sources

        assert sources == ['vocals', 'drums', 'bass', 'other']
        mock_get_model.assert_called_once()


class TestVocalSeparatorSeparation:
    """Test audio separation functionality."""

    @pytest.fixture
    def mock_separator(self):
        """Create a VocalSeparator with mocked demucs backend."""
        with patch('auto_voice.audio.separation.get_model') as mock_get_model, \
             patch('auto_voice.audio.separation.apply_model') as mock_apply:
            # Setup mock model
            mock_model = MagicMock()
            mock_model.sources = ['vocals', 'drums', 'bass', 'other']
            mock_model.samplerate = 44100
            mock_get_model.return_value = mock_model

            # Setup mock separation output
            def mock_apply_fn(model, audio_tensor, **kwargs):
                batch, channels, samples = audio_tensor.shape
                n_sources = 4
                # Return shape: (batch, n_sources, channels, samples)
                sources = torch.randn(batch, n_sources, channels, samples)
                return sources

            mock_apply.side_effect = mock_apply_fn

            from auto_voice.audio.separation import VocalSeparator
            separator = VocalSeparator()
            yield separator, mock_apply

    def test_separate_mono_audio(self, mock_separator):
        """Test separation of mono audio."""
        separator, mock_apply = mock_separator

        # Create mono audio (1D array)
        sr = 44100
        duration = 1.0
        audio = np.random.randn(int(sr * duration)).astype(np.float32)

        result = separator.separate(audio, sr)

        assert 'vocals' in result
        assert 'instrumental' in result
        assert result['vocals'].dtype == np.float32
        assert result['instrumental'].dtype == np.float32
        assert len(result['vocals']) == len(audio)
        assert len(result['instrumental']) == len(audio)

    def test_separate_stereo_audio(self, mock_separator):
        """Test separation of stereo audio."""
        separator, mock_apply = mock_separator

        # Create stereo audio (2D array)
        sr = 44100
        duration = 1.0
        audio = np.random.randn(2, int(sr * duration)).astype(np.float32)

        result = separator.separate(audio, sr)

        assert 'vocals' in result
        assert 'instrumental' in result
        # Output should be mono (averaged across channels)
        assert result['vocals'].ndim == 1
        assert result['instrumental'].ndim == 1

    def test_separate_empty_audio_raises_error(self, mock_separator):
        """Test that empty audio raises ValueError."""
        separator, _ = mock_separator

        empty_audio = np.array([])

        with pytest.raises(ValueError, match="Cannot separate empty audio"):
            separator.separate(empty_audio, 44100)

    def test_separate_invalid_dimensions_raises_error(self, mock_separator):
        """Test that audio with >2 dimensions raises ValueError."""
        separator, _ = mock_separator

        # Create 3D audio (invalid)
        audio_3d = np.random.randn(2, 2, 1000).astype(np.float32)

        with pytest.raises(ValueError, match="must be 1D .* or 2D"):
            separator.separate(audio_3d, 44100)

    def test_separate_with_resampling(self):
        """Test separation with different input/model sample rates."""
        with patch('auto_voice.audio.separation.get_model') as mock_get_model, \
             patch('auto_voice.audio.separation.apply_model') as mock_apply, \
             patch('auto_voice.audio.separation.torchaudio.transforms.Resample') as mock_resample:

            # Setup mock model with different sample rate
            mock_model = MagicMock()
            mock_model.sources = ['vocals', 'drums', 'bass', 'other']
            mock_model.samplerate = 44100
            mock_get_model.return_value = mock_model

            # Setup resampler mock
            mock_resampler = MagicMock()
            mock_resampler.return_value = torch.randn(1, 2, 44100)  # Resampled audio
            mock_resample.return_value = mock_resampler

            # Setup separation output
            def mock_apply_fn(model, audio_tensor, **kwargs):
                batch, channels, samples = audio_tensor.shape
                sources = torch.randn(batch, 4, channels, samples)
                return sources
            mock_apply.side_effect = mock_apply_fn

            from auto_voice.audio.separation import VocalSeparator
            separator = VocalSeparator()

            # Input audio at different sample rate
            audio = np.random.randn(16000).astype(np.float32)  # 16kHz
            result = separator.separate(audio, 16000)

            # Verify resampling was called
            mock_resample.assert_called_once_with(16000, 44100)
            assert 'vocals' in result
            assert 'instrumental' in result

    def test_separate_with_segment_processing(self, mock_separator):
        """Test separation with segmented processing for memory efficiency."""
        separator, mock_apply = mock_separator
        separator.segment = 5.0  # 5-second segments

        # Create long audio
        sr = 44100
        duration = 10.0
        audio = np.random.randn(int(sr * duration)).astype(np.float32)

        result = separator.separate(audio, sr)

        # Verify segment parameter was passed
        call_args = mock_apply.call_args
        assert 'segment' in call_args.kwargs
        assert call_args.kwargs['segment'] == 5.0

    @patch('auto_voice.audio.separation.torch.cuda.is_available', return_value=True)
    @patch('auto_voice.audio.separation.torch.cuda.empty_cache')
    @patch('auto_voice.audio.separation.torch.cuda.synchronize')
    def test_separate_gpu_memory_cleanup(self, mock_sync, mock_cache, mock_cuda, mock_separator):
        """Test GPU memory is cleaned up before and after separation."""
        separator, _ = mock_separator
        separator.device = torch.device('cuda')

        audio = np.random.randn(44100).astype(np.float32)
        separator.separate(audio, 44100)

        # Should call empty_cache twice (before and after)
        assert mock_cache.call_count >= 2
        assert mock_sync.call_count >= 1

    def test_separate_model_without_vocals_raises_error(self):
        """Test error when model doesn't have vocals source."""
        with patch('auto_voice.audio.separation.get_model') as mock_get_model, \
             patch('auto_voice.audio.separation.apply_model') as mock_apply:

            # Model without vocals source
            mock_model = MagicMock()
            mock_model.sources = ['drums', 'bass', 'other']  # No vocals!
            mock_model.samplerate = 44100
            mock_get_model.return_value = mock_model

            mock_apply.return_value = torch.randn(1, 3, 2, 44100)

            from auto_voice.audio.separation import VocalSeparator
            separator = VocalSeparator()

            audio = np.random.randn(44100).astype(np.float32)

            with pytest.raises(RuntimeError, match="does not have a 'vocals' source"):
                separator.separate(audio, 44100)


class TestVocalSeparatorOutputFormatting:
    """Test output audio formatting and length matching."""

    @pytest.fixture
    def separator_with_resampling(self):
        """Create separator that requires resampling."""
        with patch('auto_voice.audio.separation.get_model') as mock_get_model, \
             patch('auto_voice.audio.separation.apply_model') as mock_apply, \
             patch('auto_voice.audio.separation.librosa.resample') as mock_resample:

            mock_model = MagicMock()
            mock_model.sources = ['vocals', 'drums', 'bass', 'other']
            mock_model.samplerate = 44100
            mock_get_model.return_value = mock_model

            def mock_apply_fn(model, audio_tensor, **kwargs):
                batch, channels, samples = audio_tensor.shape
                sources = torch.randn(batch, 4, channels, samples)
                return sources
            mock_apply.side_effect = mock_apply_fn

            # Mock resampling to return slightly different length
            def mock_resample_fn(audio, orig_sr, target_sr):
                new_len = int(len(audio) * target_sr / orig_sr)
                return np.random.randn(new_len).astype(np.float32)
            mock_resample.side_effect = mock_resample_fn

            from auto_voice.audio.separation import VocalSeparator
            separator = VocalSeparator()
            yield separator, mock_resample

    def test_output_length_matches_input_after_resampling(self, separator_with_resampling):
        """Test output is trimmed/padded to match input length."""
        separator, mock_resample = separator_with_resampling

        input_len = 16000
        audio = np.random.randn(input_len).astype(np.float32)

        result = separator.separate(audio, 16000)

        # Output should match input length exactly
        assert len(result['vocals']) == input_len
        assert len(result['instrumental']) == input_len

    def test_output_padding_when_too_short(self, separator_with_resampling):
        """Test zero-padding is applied when output is shorter than input."""
        separator, mock_resample = separator_with_resampling

        # Force resampling to return shorter audio
        def short_resample(audio, orig_sr, target_sr):
            return np.random.randn(len(audio) - 100).astype(np.float32)
        mock_resample.side_effect = short_resample

        input_len = 16000
        audio = np.random.randn(input_len).astype(np.float32)

        result = separator.separate(audio, 16000)

        # Should be padded to match input
        assert len(result['vocals']) == input_len
        assert len(result['instrumental']) == input_len

    def test_output_trimming_when_too_long(self, separator_with_resampling):
        """Test trimming is applied when output is longer than input."""
        separator, mock_resample = separator_with_resampling

        # Force resampling to return longer audio
        def long_resample(audio, orig_sr, target_sr):
            return np.random.randn(len(audio) + 100).astype(np.float32)
        mock_resample.side_effect = long_resample

        input_len = 16000
        audio = np.random.randn(input_len).astype(np.float32)

        result = separator.separate(audio, 16000)

        # Should be trimmed to match input
        assert len(result['vocals']) == input_len
        assert len(result['instrumental']) == input_len

    def test_output_dtype_is_float32(self, separator_with_resampling):
        """Test output is always float32 regardless of processing."""
        separator, _ = separator_with_resampling

        audio = np.random.randn(16000).astype(np.float64)  # Input as float64

        result = separator.separate(audio, 16000)

        assert result['vocals'].dtype == np.float32
        assert result['instrumental'].dtype == np.float32


class TestVocalSeparatorEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_short_audio(self):
        """Test separation of very short audio (< 1 second)."""
        with patch('auto_voice.audio.separation.get_model') as mock_get_model, \
             patch('auto_voice.audio.separation.apply_model') as mock_apply:

            mock_model = MagicMock()
            mock_model.sources = ['vocals', 'drums', 'bass', 'other']
            mock_model.samplerate = 44100
            mock_get_model.return_value = mock_model

            def mock_apply_fn(model, audio_tensor, **kwargs):
                batch, channels, samples = audio_tensor.shape
                sources = torch.randn(batch, 4, channels, samples)
                return sources
            mock_apply.side_effect = mock_apply_fn

            from auto_voice.audio.separation import VocalSeparator
            separator = VocalSeparator()

            # Very short audio (100ms)
            audio = np.random.randn(4410).astype(np.float32)
            result = separator.separate(audio, 44100)

            assert len(result['vocals']) == 4410
            assert len(result['instrumental']) == 4410

    def test_very_long_audio_with_segments(self):
        """Test separation of long audio with segment processing."""
        with patch('auto_voice.audio.separation.get_model') as mock_get_model, \
             patch('auto_voice.audio.separation.apply_model') as mock_apply:

            mock_model = MagicMock()
            mock_model.sources = ['vocals', 'drums', 'bass', 'other']
            mock_model.samplerate = 44100
            mock_get_model.return_value = mock_model

            def mock_apply_fn(model, audio_tensor, **kwargs):
                batch, channels, samples = audio_tensor.shape
                sources = torch.randn(batch, 4, channels, samples)
                return sources
            mock_apply.side_effect = mock_apply_fn

            from auto_voice.audio.separation import VocalSeparator
            separator = VocalSeparator(segment=10.0)

            # 60 second audio
            audio = np.random.randn(44100 * 60).astype(np.float32)
            result = separator.separate(audio, 44100)

            assert len(result['vocals']) == 44100 * 60
            assert len(result['instrumental']) == 44100 * 60

    def test_single_sample_audio(self):
        """Test separation of single sample (extreme edge case)."""
        with patch('auto_voice.audio.separation.get_model') as mock_get_model, \
             patch('auto_voice.audio.separation.apply_model') as mock_apply:

            mock_model = MagicMock()
            mock_model.sources = ['vocals', 'drums', 'bass', 'other']
            mock_model.samplerate = 44100
            mock_get_model.return_value = mock_model

            def mock_apply_fn(model, audio_tensor, **kwargs):
                batch, channels, samples = audio_tensor.shape
                sources = torch.randn(batch, 4, channels, samples)
                return sources
            mock_apply.side_effect = mock_apply_fn

            from auto_voice.audio.separation import VocalSeparator
            separator = VocalSeparator()

            audio = np.array([0.5], dtype=np.float32)
            result = separator.separate(audio, 44100)

            assert len(result['vocals']) == 1
            assert len(result['instrumental']) == 1

    def test_silence_audio(self):
        """Test separation of silence (all zeros)."""
        with patch('auto_voice.audio.separation.get_model') as mock_get_model, \
             patch('auto_voice.audio.separation.apply_model') as mock_apply:

            mock_model = MagicMock()
            mock_model.sources = ['vocals', 'drums', 'bass', 'other']
            mock_model.samplerate = 44100
            mock_get_model.return_value = mock_model

            def mock_apply_fn(model, audio_tensor, **kwargs):
                batch, channels, samples = audio_tensor.shape
                sources = torch.zeros(batch, 4, channels, samples)
                return sources
            mock_apply.side_effect = mock_apply_fn

            from auto_voice.audio.separation import VocalSeparator
            separator = VocalSeparator()

            audio = np.zeros(44100, dtype=np.float32)
            result = separator.separate(audio, 44100)

            assert not np.isnan(result['vocals']).any()
            assert not np.isnan(result['instrumental']).any()
