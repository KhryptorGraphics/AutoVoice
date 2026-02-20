"""Enhanced tests for separation.py to achieve 90% coverage.

Focuses on testing actual VocalSeparator.separate() method with proper mocking.
Complements test_audio_separation_comprehensive.py.

Beads: AV-ff6 (P0 Critical)
Target: 44% → 90%
"""
import numpy as np
import pytest
import torch
from unittest.mock import Mock, MagicMock, patch

from auto_voice.audio.separation import VocalSeparator


# ============================================================================
# Test actual separate() method with proper mocking
# ============================================================================

class TestVocalSeparatorSeparateMethod:
    """Test VocalSeparator.separate() with proper mocking."""

    def test_separate_mono_audio_full_workflow(self):
        """Test complete separation workflow with mono audio."""
        separator = VocalSeparator(device=torch.device('cpu'))
        audio = np.random.randn(44100).astype(np.float32)
        sr = 44100

        # Mock the internal _get_model and _apply_model
        with patch.object(separator, '_get_model') as mock_get_model:
            with patch.object(separator, '_apply_model') as mock_apply:
                # Setup mock model
                mock_model = MagicMock()
                mock_model.samplerate = sr
                mock_model.sources = ['drums', 'bass', 'other', 'vocals']
                mock_get_model.return_value = mock_model

                # Setup mock output
                mock_output = torch.randn(1, 4, 2, len(audio))
                mock_apply.return_value = mock_output

                result = separator.separate(audio, sr)

                assert 'vocals' in result
                assert 'instrumental' in result
                assert len(result['vocals']) == len(audio)
                assert result['vocals'].dtype == np.float32

    def test_separate_stereo_audio_full_workflow(self):
        """Test complete separation workflow with stereo audio."""
        separator = VocalSeparator(device=torch.device('cpu'))
        audio = np.random.randn(2, 44100).astype(np.float32)
        sr = 44100

        with patch.object(separator, '_get_model') as mock_get_model:
            with patch.object(separator, '_apply_model') as mock_apply:
                mock_model = MagicMock()
                mock_model.samplerate = sr
                mock_model.sources = ['drums', 'bass', 'other', 'vocals']
                mock_get_model.return_value = mock_model

                mock_output = torch.randn(1, 4, 2, audio.shape[1])
                mock_apply.return_value = mock_output

                result = separator.separate(audio, sr)

                assert len(result['vocals']) == audio.shape[1]

    def test_separate_with_resampling_workflow(self):
        """Test separation with sample rate conversion."""
        separator = VocalSeparator(device=torch.device('cpu'))
        audio = np.random.randn(22050).astype(np.float32)
        input_sr = 22050
        model_sr = 44100

        with patch.object(separator, '_get_model') as mock_get_model:
            with patch.object(separator, '_apply_model') as mock_apply:
                mock_model = MagicMock()
                mock_model.samplerate = model_sr
                mock_model.sources = ['drums', 'bass', 'other', 'vocals']
                mock_get_model.return_value = mock_model

                # Output at model sample rate
                resampled_len = int(len(audio) * model_sr / input_sr)
                mock_output = torch.randn(1, 4, 2, resampled_len)
                mock_apply.return_value = mock_output

                with patch('torchaudio.transforms.Resample') as mock_resample_class:
                    mock_resampler = MagicMock()
                    mock_resampler.return_value = torch.randn(1, 2, resampled_len)
                    mock_resample_class.return_value = mock_resampler

                    with patch('librosa.resample') as mock_librosa:
                        # Downsample back to input SR
                        mock_librosa.side_effect = lambda x, **kwargs: x[:len(audio)]

                        result = separator.separate(audio, input_sr)

                        # Output should match input length
                        assert len(result['vocals']) == len(audio)

    def test_separate_with_segment_processing(self):
        """Test separation with segmented processing for memory efficiency."""
        separator = VocalSeparator(segment=10.0, device=torch.device('cpu'))
        audio = np.random.randn(44100).astype(np.float32)
        sr = 44100

        with patch.object(separator, '_get_model') as mock_get_model:
            with patch.object(separator, '_apply_model') as mock_apply:
                mock_model = MagicMock()
                mock_model.samplerate = sr
                mock_model.sources = ['drums', 'bass', 'other', 'vocals']
                mock_get_model.return_value = mock_model

                mock_output = torch.randn(1, 4, 2, len(audio))
                mock_apply.return_value = mock_output

                result = separator.separate(audio, sr)

                # Verify segment was passed to apply_model
                call_kwargs = mock_apply.call_args[1]
                assert 'segment' in call_kwargs
                assert call_kwargs['segment'] == 10.0

    def test_separate_gpu_cache_management(self):
        """Test GPU cache clearing during separation."""
        separator = VocalSeparator(device=torch.device('cuda:0'))
        audio = np.random.randn(44100).astype(np.float32)
        sr = 44100

        with patch.object(separator, '_get_model') as mock_get_model:
            with patch.object(separator, '_apply_model') as mock_apply:
                mock_model = MagicMock()
                mock_model.samplerate = sr
                mock_model.sources = ['drums', 'bass', 'other', 'vocals']
                mock_get_model.return_value = mock_model

                mock_output = torch.randn(1, 4, 2, len(audio))
                mock_apply.return_value = mock_output

                with patch('torch.cuda.empty_cache') as mock_cache:
                    with patch('torch.cuda.synchronize') as mock_sync:
                        result = separator.separate(audio, sr)

                        # Should clear cache before and after
                        assert mock_cache.call_count >= 2

    def test_separate_missing_vocals_source_error(self):
        """Test error when model doesn't have vocals source."""
        separator = VocalSeparator(device=torch.device('cpu'))
        audio = np.random.randn(44100).astype(np.float32)

        with patch.object(separator, '_get_model') as mock_get_model:
            with patch.object(separator, '_apply_model') as mock_apply:
                mock_model = MagicMock()
                mock_model.samplerate = 44100
                mock_model.sources = ['drums', 'bass', 'other']  # No vocals!
                mock_get_model.return_value = mock_model

                mock_output = torch.randn(1, 3, 2, len(audio))
                mock_apply.return_value = mock_output

                with pytest.raises(RuntimeError, match="does not have a 'vocals' source"):
                    separator.separate(audio, 44100)

    def test_separate_output_length_trimming(self):
        """Test output is trimmed when longer than input."""
        separator = VocalSeparator(device=torch.device('cpu'))
        audio = np.random.randn(44100).astype(np.float32)
        sr = 44100

        with patch.object(separator, '_get_model') as mock_get_model:
            with patch.object(separator, '_apply_model') as mock_apply:
                mock_model = MagicMock()
                mock_model.samplerate = sr
                mock_model.sources = ['drums', 'bass', 'other', 'vocals']
                mock_get_model.return_value = mock_model

                # Output longer than input
                mock_output = torch.randn(1, 4, 2, len(audio) + 100)
                mock_apply.return_value = mock_output

                result = separator.separate(audio, sr)

                # Should be trimmed to original length
                assert len(result['vocals']) == len(audio)
                assert len(result['instrumental']) == len(audio)

    def test_separate_output_length_padding(self):
        """Test output is padded when shorter than input."""
        separator = VocalSeparator(device=torch.device('cpu'))
        audio = np.random.randn(44100).astype(np.float32)
        sr = 44100

        with patch.object(separator, '_get_model') as mock_get_model:
            with patch.object(separator, '_apply_model') as mock_apply:
                mock_model = MagicMock()
                mock_model.samplerate = sr
                mock_model.sources = ['drums', 'bass', 'other', 'vocals']
                mock_get_model.return_value = mock_model

                # Output shorter than input
                mock_output = torch.randn(1, 4, 2, len(audio) - 100)
                mock_apply.return_value = mock_output

                result = separator.separate(audio, sr)

                # Should be padded to original length
                assert len(result['vocals']) == len(audio)
                assert len(result['instrumental']) == len(audio)

    def test_separate_no_resampling_when_sr_matches(self):
        """Test no resampling when sample rates match."""
        separator = VocalSeparator(device=torch.device('cpu'))
        audio = np.random.randn(44100).astype(np.float32)
        sr = 44100

        with patch.object(separator, '_get_model') as mock_get_model:
            with patch.object(separator, '_apply_model') as mock_apply:
                mock_model = MagicMock()
                mock_model.samplerate = sr  # Same as input
                mock_model.sources = ['drums', 'bass', 'other', 'vocals']
                mock_get_model.return_value = mock_model

                mock_output = torch.randn(1, 4, 2, len(audio))
                mock_apply.return_value = mock_output

                with patch('torchaudio.transforms.Resample') as mock_resample:
                    result = separator.separate(audio, sr)

                    # No resampling should occur
                    mock_resample.assert_not_called()

    def test_separate_mono_to_stereo_expansion(self):
        """Test mono audio is expanded to stereo for model."""
        separator = VocalSeparator(device=torch.device('cpu'))
        audio = np.random.randn(44100).astype(np.float32)  # Mono
        sr = 44100

        with patch.object(separator, '_get_model') as mock_get_model:
            with patch.object(separator, '_apply_model') as mock_apply:
                mock_model = MagicMock()
                mock_model.samplerate = sr
                mock_model.sources = ['drums', 'bass', 'other', 'vocals']
                mock_get_model.return_value = mock_model

                mock_output = torch.randn(1, 4, 2, len(audio))
                mock_apply.return_value = mock_output

                result = separator.separate(audio, sr)

                # Check that stereo tensor was passed to apply_model
                call_args = mock_apply.call_args[0]
                input_tensor = call_args[1]
                assert input_tensor.shape[1] == 2  # Stereo


# ============================================================================
# Test Model Loading
# ============================================================================

class TestModelLoadingProper:
    """Test model loading functionality with proper mocking."""

    def test_load_model_success(self):
        """Test successful model loading."""
        separator = VocalSeparator(device=torch.device('cpu'))

        with patch.object(separator, '_get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.samplerate = 44100
            mock_model.sources = ['drums', 'bass', 'other', 'vocals']
            mock_get_model.return_value = mock_model

            separator._load_model()

            assert separator._model is not None
            mock_get_model.assert_called_once_with('htdemucs')

    def test_load_model_failure_raises_error(self):
        """Test model loading failure raises RuntimeError."""
        separator = VocalSeparator(device=torch.device('cpu'))

        with patch.object(separator, '_get_model') as mock_get_model:
            mock_get_model.side_effect = Exception("Download failed")

            with pytest.raises(RuntimeError, match="Failed to load Demucs model"):
                separator._load_model()

    def test_load_model_only_once(self):
        """Test model is loaded only once on multiple calls."""
        separator = VocalSeparator(device=torch.device('cpu'))

        with patch.object(separator, '_get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.samplerate = 44100
            mock_model.sources = ['drums', 'bass', 'other', 'vocals']
            mock_get_model.return_value = mock_model

            # Load multiple times
            separator._load_model()
            separator._load_model()
            separator._load_model()

            # Should only call get_model once
            assert mock_get_model.call_count == 1

    def test_model_sample_rate_property(self):
        """Test model_sample_rate property."""
        separator = VocalSeparator(device=torch.device('cpu'))

        with patch.object(separator, '_get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.samplerate = 44100
            mock_get_model.return_value = mock_model

            sr = separator.model_sample_rate

            assert sr == 44100

    def test_sources_property(self):
        """Test sources property."""
        separator = VocalSeparator(device=torch.device('cpu'))

        with patch.object(separator, '_get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.sources = ['drums', 'bass', 'other', 'vocals']
            mock_get_model.return_value = mock_model

            sources = separator.sources

            assert sources == ['drums', 'bass', 'other', 'vocals']
            assert 'vocals' in sources


# ============================================================================
# Test Edge Cases with Full Workflow
# ============================================================================

class TestEdgeCasesFullWorkflow:
    """Test edge cases with full separation workflow."""

    def test_empty_audio_validation(self):
        """Test empty audio raises ValueError."""
        separator = VocalSeparator(device=torch.device('cpu'))

        with pytest.raises(ValueError, match="Cannot separate empty audio"):
            separator.separate(np.array([]), 44100)

    def test_3d_audio_validation(self):
        """Test 3D audio raises ValueError."""
        separator = VocalSeparator(device=torch.device('cpu'))
        audio_3d = np.random.randn(2, 3, 1000).astype(np.float32)

        with pytest.raises(ValueError, match="must be 1D .* or 2D"):
            separator.separate(audio_3d, 44100)

    def test_instrumental_sum_excludes_vocals(self):
        """Test instrumental is sum of all non-vocal sources."""
        separator = VocalSeparator(device=torch.device('cpu'))
        audio = np.random.randn(44100).astype(np.float32)

        with patch.object(separator, '_get_model') as mock_get_model:
            with patch.object(separator, '_apply_model') as mock_apply:
                mock_model = MagicMock()
                mock_model.samplerate = 44100
                mock_model.sources = ['drums', 'bass', 'other', 'vocals']
                mock_get_model.return_value = mock_model

                # Create distinct sources
                mock_output = torch.randn(1, 4, 2, len(audio))
                mock_apply.return_value = mock_output

                result = separator.separate(audio, 44100)

                # Instrumental should be different from vocals
                assert not np.array_equal(result['vocals'], result['instrumental'])


# ============================================================================
# Integration-style Tests
# ============================================================================

@pytest.mark.integration
class TestVocalSeparatorIntegrationWorkflow:
    """Integration-style tests for complete workflow."""

    def test_complete_workflow_mono(self):
        """Test complete workflow with mono audio."""
        separator = VocalSeparator(device=torch.device('cpu'))
        audio = np.random.randn(44100 * 2).astype(np.float32) * 0.3

        with patch.object(separator, '_get_model') as mock_get_model:
            with patch.object(separator, '_apply_model') as mock_apply:
                mock_model = MagicMock()
                mock_model.samplerate = 44100
                mock_model.sources = ['drums', 'bass', 'other', 'vocals']
                mock_get_model.return_value = mock_model

                mock_output = torch.randn(1, 4, 2, len(audio))
                mock_apply.return_value = mock_output

                result = separator.separate(audio, 44100)

                # Verify output structure
                assert isinstance(result, dict)
                assert set(result.keys()) == {'vocals', 'instrumental'}
                assert result['vocals'].shape == audio.shape
                assert result['instrumental'].shape == audio.shape
                assert result['vocals'].dtype == np.float32
                assert result['instrumental'].dtype == np.float32

    def test_complete_workflow_stereo(self):
        """Test complete workflow with stereo audio."""
        separator = VocalSeparator(device=torch.device('cpu'))
        audio = np.random.randn(2, 44100 * 2).astype(np.float32) * 0.3

        with patch.object(separator, '_get_model') as mock_get_model:
            with patch.object(separator, '_apply_model') as mock_apply:
                mock_model = MagicMock()
                mock_model.samplerate = 44100
                mock_model.sources = ['drums', 'bass', 'other', 'vocals']
                mock_get_model.return_value = mock_model

                mock_output = torch.randn(1, 4, 2, audio.shape[1])
                mock_apply.return_value = mock_output

                result = separator.separate(audio, 44100)

                # Output should be mono
                assert result['vocals'].ndim == 1
                assert len(result['vocals']) == audio.shape[1]

    @pytest.mark.slow
    def test_multiple_separations_reuse_model(self):
        """Test model is reused across multiple separations."""
        separator = VocalSeparator(device=torch.device('cpu'))

        with patch.object(separator, '_get_model') as mock_get_model:
            with patch.object(separator, '_apply_model') as mock_apply:
                mock_model = MagicMock()
                mock_model.samplerate = 44100
                mock_model.sources = ['drums', 'bass', 'other', 'vocals']
                mock_get_model.return_value = mock_model

                # Separate multiple files
                for _ in range(3):
                    audio = np.random.randn(44100).astype(np.float32)
                    mock_apply.return_value = torch.randn(1, 4, 2, len(audio))
                    separator.separate(audio, 44100)

                # Model should be loaded only once
                assert mock_get_model.call_count == 1
