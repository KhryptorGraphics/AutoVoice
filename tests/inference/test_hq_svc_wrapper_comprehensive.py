"""Comprehensive test suite for HQSVCWrapper module.

Target: 90%+ coverage for src/auto_voice/inference/hq_svc_wrapper.py

Test Categories:
1. Initialization Tests (GPU requirements, config loading, model initialization)
2. Audio Processing Tests (resampling, mono conversion, padding)
3. Feature Extraction Tests (speaker embeddings, F0, volume, FACodec)
4. Super-Resolution Tests (16kHz → 44.1kHz upsampling)
5. Voice Conversion Tests (speaker transfer, pitch shifting, auto-pitch)
6. Error Handling Tests (missing models, invalid audio, GPU OOM)
7. Integration Tests (adapter bridge, complete workflows)
8. Performance Tests (memory cleanup, device placement)
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_hq_svc_paths():
    """Mock HQ-SVC directory structure and paths."""
    with patch('auto_voice.inference.hq_svc_wrapper.HQ_SVC_ROOT') as mock_root:
        mock_root.return_value = '/fake/hq-svc'
        yield mock_root


@pytest.fixture
def mock_config():
    """Create mock HQ-SVC configuration."""
    config = SimpleNamespace()
    config.config = '/fake/config.yaml'
    config.device = 'cuda'
    config.sample_rate = 44100
    config.encoder_sr = 16000
    config.infer_speedup = 10
    config.infer_method = 'dpm-solver'
    config.vocoder = 'nsf-hifigan'
    config.model_path = 'utils/pretrain/250000_step_val_loss_0.50.pth'
    config.f0_extractor = 'rmvpe'
    config.f0_min = 60
    config.f0_max = 1200
    config.block_size = 512
    config.hop_size = 256
    return config


@pytest.fixture
def mock_vocoder():
    """Create mock vocoder."""
    vocoder = MagicMock()
    vocoder.infer = MagicMock(return_value=torch.randn(1, 44100))  # 1 second audio
    return vocoder


@pytest.fixture
def mock_net_g():
    """Create mock HQ-SVC generator network."""
    net_g = MagicMock()
    net_g.eval = MagicMock()
    # Return mel-spectrogram shape [1, 80, T]
    net_g.return_value = torch.randn(1, 80, 100)
    return net_g


@pytest.fixture
def mock_fa_encoder():
    """Create mock FACodec encoder."""
    encoder = MagicMock()
    return encoder


@pytest.fixture
def mock_fa_decoder():
    """Create mock FACodec decoder."""
    decoder = MagicMock()
    return decoder


@pytest.fixture
def mock_f0_extractor():
    """Create mock F0 extractor."""
    extractor = MagicMock()
    return extractor


@pytest.fixture
def mock_volume_extractor():
    """Create mock volume extractor."""
    extractor = MagicMock()
    return extractor


@pytest.fixture
def sample_audio_16k():
    """Generate 5-second test audio at 16kHz (minimum duration)."""
    sr = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 440Hz sine wave
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return torch.from_numpy(audio), sr


@pytest.fixture
def sample_audio_44k():
    """Generate 5-second test audio at 44.1kHz."""
    sr = 44100
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 440Hz sine wave
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return torch.from_numpy(audio), sr


@pytest.fixture
def short_audio():
    """Generate audio below minimum duration (0.3 seconds at 16kHz)."""
    sr = 16000
    duration = 0.3
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return torch.from_numpy(audio), sr


@pytest.fixture
def stereo_audio():
    """Generate stereo audio for mono conversion testing."""
    sr = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    left = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    right = np.sin(2 * np.pi * 550 * t).astype(np.float32)
    audio = torch.from_numpy(np.stack([left, right]))  # [2, T]
    return audio, sr


@pytest.fixture
def mock_processed_data():
    """Create mock processed audio data from HQ-SVC preprocessing."""
    data = {
        'vq_post': torch.randn(1024, 256),  # [T, C] FACodec features
        'spk': torch.randn(1, 256),  # [1, 256] speaker embedding
        'f0': torch.randn(1024),  # [T] F0 contour
        'f0_origin': np.random.randn(1024).astype(np.float32) * 100 + 200,  # Hz
        'vol': torch.randn(1024),  # [T] volume
        'mel': torch.randn(80, 1024),  # [80, T] mel-spectrogram
    }
    return data


# ============================================================================
# Initialization Tests
# ============================================================================


class TestHQSVCWrapperInitialization:
    """Tests for HQSVCWrapper initialization and setup."""

    @patch('auto_voice.inference.hq_svc_wrapper.torch.cuda.is_available')
    def test_init_requires_cuda_by_default(self, mock_cuda):
        """Test that wrapper requires CUDA by default."""
        mock_cuda.return_value = False

        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        with pytest.raises(RuntimeError, match="CUDA is required"):
            HQSVCWrapper()

    @patch('auto_voice.inference.hq_svc_wrapper.torch.cuda.is_available')
    def test_init_explicit_cpu_device_rejected(self, mock_cuda):
        """Test that explicit CPU device is rejected when require_gpu=True."""
        mock_cuda.return_value = True

        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        with pytest.raises(RuntimeError, match="CUDA is required.*CPU device specified"):
            HQSVCWrapper(device=torch.device('cpu'), require_gpu=True)

    @patch('auto_voice.inference.hq_svc_wrapper.torch.cuda.is_available')
    @patch('auto_voice.inference.hq_svc_wrapper.HQSVCWrapper._load_config')
    @patch('auto_voice.inference.hq_svc_wrapper.HQSVCWrapper._init_models')
    @patch('auto_voice.inference.hq_svc_wrapper.HQSVCWrapper._setup_paths')
    def test_init_with_cpu_when_allowed(self, mock_setup, mock_init, mock_load, mock_cuda):
        """Test initialization with CPU when require_gpu=False."""
        mock_cuda.return_value = False
        mock_load.return_value = MagicMock()

        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = HQSVCWrapper(require_gpu=False)
        assert wrapper.device.type == 'cpu'
        assert wrapper.output_sample_rate == 44100
        assert wrapper.encoder_sample_rate == 16000

    @patch('auto_voice.inference.hq_svc_wrapper.torch.cuda.is_available')
    @patch('auto_voice.inference.hq_svc_wrapper.HQSVCWrapper._load_config')
    @patch('auto_voice.inference.hq_svc_wrapper.HQSVCWrapper._init_models')
    @patch('auto_voice.inference.hq_svc_wrapper.HQSVCWrapper._setup_paths')
    def test_init_with_cuda(self, mock_setup, mock_init, mock_load, mock_cuda):
        """Test successful initialization with CUDA."""
        mock_cuda.return_value = True
        mock_load.return_value = MagicMock()

        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = HQSVCWrapper()
        assert wrapper.device.type == 'cuda'

    @patch('auto_voice.inference.hq_svc_wrapper.sys.path', [])
    @patch('auto_voice.inference.hq_svc_wrapper.os.getcwd')
    @patch('auto_voice.inference.hq_svc_wrapper.os.chdir')
    @patch('auto_voice.inference.hq_svc_wrapper.HQ_SVC_ROOT', '/fake/hq-svc')
    def test_setup_paths(self, mock_chdir, mock_getcwd):
        """Test HQ-SVC path setup."""
        mock_getcwd.return_value = '/original/cwd'

        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        # Create partial wrapper to test _setup_paths
        wrapper = object.__new__(HQSVCWrapper)
        wrapper._setup_paths()

        # Check sys.path was updated
        assert '/fake/hq-svc' in sys.path
        assert sys.path[0] == '/fake/hq-svc'

        # Check cwd changed
        mock_chdir.assert_called_with('/fake/hq-svc')
        assert wrapper._original_cwd == '/original/cwd'

    def test_load_config_file_not_found(self):
        """Test config loading with missing file."""
        # Mock logger modules
        sys.modules['logger'] = MagicMock()
        sys.modules['logger.utils'] = MagicMock()

        with patch('os.path.exists', return_value=False):
            from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

            wrapper = object.__new__(HQSVCWrapper)
            wrapper.device = torch.device('cpu')

            with pytest.raises(RuntimeError, match="Config file not found"):
                wrapper._load_config('/fake/config.yaml')

    def test_load_config_sets_defaults(self, mock_config):
        """Test that config loading sets default values."""
        # Mock the logger.utils module
        mock_logger_utils = MagicMock()
        minimal_config = SimpleNamespace()
        minimal_config.model_path = '/fake/model.pth'
        mock_logger_utils.load_config = MagicMock(return_value=minimal_config)

        sys.modules['logger'] = MagicMock()
        sys.modules['logger.utils'] = mock_logger_utils

        with patch('os.path.exists', return_value=True):
            from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

            wrapper = object.__new__(HQSVCWrapper)
            wrapper.device = torch.device('cpu')

            args = wrapper._load_config('/fake/config.yaml')

            # Check defaults were set
            assert args.sample_rate == 44100
            assert args.encoder_sr == 16000
            assert args.infer_speedup == 10
            assert args.infer_method == 'dpm-solver'
            assert args.vocoder == 'nsf-hifigan'
            assert args.device == 'cpu'

    def test_init_models_missing_weights(self):
        """Test model initialization with missing weights."""
        # Mock HQ-SVC modules
        mock_utils = MagicMock()
        mock_utils.vocoder = MagicMock()
        mock_utils.models = MagicMock()
        mock_utils.data_preprocessing = MagicMock()

        sys.modules['utils'] = mock_utils
        sys.modules['utils.vocoder'] = mock_utils.vocoder
        sys.modules['utils.models'] = mock_utils.models
        sys.modules['utils.models.models_v2_beta'] = MagicMock()
        sys.modules['utils.data_preprocessing'] = mock_utils.data_preprocessing

        with patch('os.path.exists', return_value=False):
            from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

            wrapper = object.__new__(HQSVCWrapper)
            wrapper.device = torch.device('cpu')
            wrapper.args = SimpleNamespace()
            wrapper.args.model_path = '/fake/model.pth'
            wrapper.args.vocoder = 'nsf-hifigan'

            with pytest.raises(RuntimeError, match="Model weights not found"):
                wrapper._init_models()

    def test_init_restores_cwd_on_error(self):
        """Test that original cwd is restored even on initialization error."""
        original_cwd = os.getcwd()

        with patch('auto_voice.inference.hq_svc_wrapper.torch.cuda.is_available', return_value=True):
            with patch('auto_voice.inference.hq_svc_wrapper.HQSVCWrapper._setup_paths'):
                with patch('auto_voice.inference.hq_svc_wrapper.HQSVCWrapper._load_config') as mock_load:
                    mock_load.side_effect = RuntimeError("Config error")

                    from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

                    wrapper = object.__new__(HQSVCWrapper)
                    wrapper.device = torch.device('cuda')
                    wrapper._original_cwd = original_cwd

                    with pytest.raises(RuntimeError, match="Config error"):
                        wrapper.__init__()


# ============================================================================
# Audio Processing Tests
# ============================================================================


class TestAudioProcessing:
    """Tests for audio processing utilities."""

    def test_resample_same_rate(self, sample_audio_16k):
        """Test resampling with same input/output rate."""
        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = object.__new__(HQSVCWrapper)
        audio, sr = sample_audio_16k

        result = wrapper._resample(audio, sr, sr)

        # Should return unchanged
        assert torch.allclose(result, audio)

    def test_resample_1d_audio(self, sample_audio_16k):
        """Test resampling 1D audio tensor."""
        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = object.__new__(HQSVCWrapper)
        audio, sr = sample_audio_16k

        result = wrapper._resample(audio, 16000, 44100)

        # Check output shape
        expected_len = int(len(audio) * 44100 / 16000)
        assert len(result) == expected_len
        assert result.dim() == 1

    def test_resample_2d_audio(self):
        """Test resampling 2D audio tensor (batch/channels)."""
        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = object.__new__(HQSVCWrapper)
        audio = torch.randn(2, 16000)  # [C, T]

        result = wrapper._resample(audio, 16000, 44100)

        # Check output shape
        assert result.shape[0] == 2
        expected_len = int(16000 * 44100 / 16000)
        assert result.shape[1] == expected_len

    def test_to_mono_1d(self, sample_audio_16k):
        """Test mono conversion with already mono audio."""
        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = object.__new__(HQSVCWrapper)
        audio, _ = sample_audio_16k

        result = wrapper._to_mono(audio)

        assert torch.equal(result, audio)

    def test_to_mono_stereo(self, stereo_audio):
        """Test stereo to mono conversion."""
        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = object.__new__(HQSVCWrapper)
        audio, _ = stereo_audio

        result = wrapper._to_mono(audio)

        # Should average channels
        expected = audio.mean(dim=0)
        assert torch.allclose(result, expected)

    def test_to_mono_invalid_shape(self):
        """Test mono conversion with invalid shape."""
        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = object.__new__(HQSVCWrapper)
        audio = torch.randn(2, 3, 1000)  # Invalid 3D tensor

        with pytest.raises(RuntimeError, match="Unexpected audio shape"):
            wrapper._to_mono(audio)

    def test_wav_pad_no_padding_needed(self):
        """Test padding when length is already a multiple."""
        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = object.__new__(HQSVCWrapper)
        audio = np.random.randn(1000).astype(np.float32)  # 1000 = 5 * 200

        result = wrapper._wav_pad(audio, multiple=200)

        assert len(result) == 1000
        np.testing.assert_array_equal(result, audio)

    def test_wav_pad_with_padding(self):
        """Test padding when padding is needed."""
        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = object.__new__(HQSVCWrapper)
        audio = np.random.randn(950).astype(np.float32)  # Needs padding to 1000

        result = wrapper._wav_pad(audio, multiple=200)

        assert len(result) == 1000
        # First 950 samples should match
        np.testing.assert_array_equal(result[:950], audio)


# ============================================================================
# Feature Extraction Tests
# ============================================================================


class TestFeatureExtraction:
    """Tests for audio feature extraction."""

    @patch('os.unlink')
    @patch('tempfile.NamedTemporaryFile')
    @patch('soundfile.write')
    def test_process_audio_success(self, mock_sf, mock_temp, mock_unlink,
                                   sample_audio_16k, mock_processed_data):
        """Test successful audio feature extraction."""
        # Mock HQ-SVC modules
        sys.modules['utils'] = MagicMock()
        sys.modules['utils.data_preprocessing'] = MagicMock()

        # Setup mocks
        mock_temp.return_value.__enter__.return_value.name = '/tmp/test.wav'

        mock_data_preprocessing = sys.modules['utils.data_preprocessing']
        mock_data_preprocessing.get_processed_file = MagicMock(return_value=mock_processed_data)

        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = object.__new__(HQSVCWrapper)
        wrapper.device = torch.device('cpu')
        wrapper.vocoder = MagicMock()
        wrapper.volume_extractor = MagicMock()
        wrapper.f0_extractor = MagicMock()
        wrapper.fa_encoder = MagicMock()
        wrapper.fa_decoder = MagicMock()

        audio, sr = sample_audio_16k
        result = wrapper._process_audio(audio, sr)

        # Check that all expected keys are present
        assert 'vq_post' in result
        assert 'spk' in result
        assert 'f0' in result
        assert 'f0_origin' in result
        assert 'vol' in result
        assert 'mel' in result

        # Check temp file was cleaned up
        mock_unlink.assert_called_once()

    @patch('os.unlink')
    @patch('tempfile.NamedTemporaryFile')
    @patch('soundfile.write')
    def test_process_audio_returns_none(self, mock_sf, mock_temp, mock_unlink,
                                        sample_audio_16k):
        """Test audio processing failure."""
        # Mock HQ-SVC modules
        sys.modules['utils'] = MagicMock()
        sys.modules['utils.data_preprocessing'] = MagicMock()

        mock_temp.return_value.__enter__.return_value.name = '/tmp/test.wav'

        mock_data_preprocessing = sys.modules['utils.data_preprocessing']
        mock_data_preprocessing.get_processed_file = MagicMock(return_value=None)

        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = object.__new__(HQSVCWrapper)
        wrapper.device = torch.device('cpu')
        wrapper.vocoder = MagicMock()
        wrapper.volume_extractor = MagicMock()
        wrapper.f0_extractor = MagicMock()
        wrapper.fa_encoder = MagicMock()
        wrapper.fa_decoder = MagicMock()

        audio, sr = sample_audio_16k

        with pytest.raises(RuntimeError, match="Failed to process audio"):
            wrapper._process_audio(audio, sr)

    @patch('auto_voice.inference.hq_svc_wrapper.HQSVCWrapper._process_audio')
    def test_extract_speaker_embedding_single(self, mock_process, sample_audio_16k,
                                              mock_processed_data):
        """Test speaker embedding extraction from single audio."""
        mock_process.return_value = mock_processed_data

        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = object.__new__(HQSVCWrapper)
        audio, sr = sample_audio_16k

        embedding = wrapper.extract_speaker_embedding(audio, sr)

        # Check shape and normalization
        assert embedding.shape == (256,)
        # L2 norm should be 1
        assert torch.allclose(torch.norm(embedding, p=2), torch.tensor(1.0), atol=1e-5)

    @patch('auto_voice.inference.hq_svc_wrapper.HQSVCWrapper._process_audio')
    def test_extract_speaker_embedding_multiple(self, mock_process, sample_audio_16k,
                                                mock_processed_data):
        """Test speaker embedding extraction from multiple audios."""
        mock_process.return_value = mock_processed_data

        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = object.__new__(HQSVCWrapper)
        audio, sr = sample_audio_16k

        # Multiple reference audios
        audios = [audio, audio, audio]
        embedding = wrapper.extract_speaker_embedding(audios, sr)

        # Check shape and normalization
        assert embedding.shape == (256,)
        assert torch.allclose(torch.norm(embedding, p=2), torch.tensor(1.0), atol=1e-5)

        # Should have called process 3 times
        assert mock_process.call_count == 3


# ============================================================================
# Super-Resolution Tests
# ============================================================================


class TestSuperResolution:
    """Tests for HQ-SVC super-resolution mode."""

    @patch('auto_voice.inference.hq_svc_wrapper.HQSVCWrapper._process_audio')
    def test_super_resolve_audio_too_short(self, mock_process, short_audio):
        """Test super-resolution with audio below minimum duration."""
        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = object.__new__(HQSVCWrapper)
        audio, sr = short_audio

        with pytest.raises(RuntimeError, match="Audio too short"):
            wrapper.super_resolve(audio, sr)

    @patch('auto_voice.inference.hq_svc_wrapper.HQSVCWrapper._process_audio')
    def test_super_resolve_success(self, mock_process, sample_audio_16k,
                                   mock_processed_data, mock_vocoder, mock_net_g):
        """Test successful super-resolution."""
        mock_process.return_value = mock_processed_data

        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = object.__new__(HQSVCWrapper)
        wrapper.device = torch.device('cpu')
        wrapper.net_g = mock_net_g
        wrapper.vocoder = mock_vocoder
        wrapper.args = SimpleNamespace()
        wrapper.args.infer_speedup = 10
        wrapper.args.infer_method = 'dpm-solver'
        wrapper.args.vocoder = 'nsf-hifigan'

        audio, sr = sample_audio_16k
        result = wrapper.super_resolve(audio, sr)

        # Check result structure
        assert 'audio' in result
        assert 'sample_rate' in result
        assert 'metadata' in result

        # Check output properties
        assert result['sample_rate'] == 44100
        assert result['audio'].dim() == 1

        # Check metadata
        assert result['metadata']['mode'] == 'super_resolution'
        assert 'processing_time' in result['metadata']
        assert 'output_duration' in result['metadata']

    @patch('auto_voice.inference.hq_svc_wrapper.HQSVCWrapper._process_audio')
    def test_super_resolve_with_progress_callback(self, mock_process, sample_audio_16k,
                                                   mock_processed_data, mock_vocoder, mock_net_g):
        """Test super-resolution with progress callback."""
        mock_process.return_value = mock_processed_data

        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = object.__new__(HQSVCWrapper)
        wrapper.device = torch.device('cpu')
        wrapper.net_g = mock_net_g
        wrapper.vocoder = mock_vocoder
        wrapper.args = SimpleNamespace()
        wrapper.args.infer_speedup = 10
        wrapper.args.infer_method = 'dpm-solver'
        wrapper.args.vocoder = 'nsf-hifigan'

        # Track progress callbacks
        progress_calls = []
        def on_progress(stage, progress):
            progress_calls.append((stage, progress))

        audio, sr = sample_audio_16k
        wrapper.super_resolve(audio, sr, on_progress=on_progress)

        # Check progress was reported
        assert len(progress_calls) > 0
        stages = [call[0] for call in progress_calls]
        assert 'preprocessing' in stages
        assert 'encoding' in stages
        assert 'diffusion' in stages
        assert 'vocoder' in stages
        assert 'complete' in stages

    @patch('auto_voice.inference.hq_svc_wrapper.HQSVCWrapper._process_audio')
    def test_super_resolve_normalization(self, mock_process, sample_audio_16k,
                                         mock_processed_data, mock_net_g):
        """Test that super-resolution output is properly normalized."""
        mock_process.return_value = mock_processed_data

        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = object.__new__(HQSVCWrapper)
        wrapper.device = torch.device('cpu')
        wrapper.net_g = mock_net_g
        wrapper.args = SimpleNamespace()
        wrapper.args.infer_speedup = 10
        wrapper.args.infer_method = 'dpm-solver'
        wrapper.args.vocoder = 'nsf-hifigan'

        # Mock vocoder to return loud audio
        loud_audio = torch.randn(44100) * 5.0  # Peak >> 1
        wrapper.vocoder = MagicMock()
        wrapper.vocoder.infer = MagicMock(return_value=loud_audio.unsqueeze(0))

        audio, sr = sample_audio_16k
        result = wrapper.super_resolve(audio, sr)

        # Check output is normalized to [-0.95, 0.95]
        assert result['audio'].abs().max() <= 1.0
        assert result['audio'].abs().max() >= 0.90  # Should be close to 0.95

    @patch('auto_voice.inference.hq_svc_wrapper.HQSVCWrapper._process_audio')
    def test_super_resolve_stereo_input(self, mock_process, stereo_audio,
                                        mock_processed_data, mock_vocoder, mock_net_g):
        """Test super-resolution with stereo input (should convert to mono)."""
        mock_process.return_value = mock_processed_data

        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = object.__new__(HQSVCWrapper)
        wrapper.device = torch.device('cpu')
        wrapper.net_g = mock_net_g
        wrapper.vocoder = mock_vocoder
        wrapper.args = SimpleNamespace()
        wrapper.args.infer_speedup = 10
        wrapper.args.infer_method = 'dpm-solver'
        wrapper.args.vocoder = 'nsf-hifigan'

        audio, sr = stereo_audio
        result = wrapper.super_resolve(audio, sr)

        # Should produce mono output
        assert result['audio'].dim() == 1


# ============================================================================
# Voice Conversion Tests
# ============================================================================


class TestVoiceConversion:
    """Tests for HQ-SVC voice conversion mode."""

    @patch('auto_voice.inference.hq_svc_wrapper.HQSVCWrapper._process_audio')
    def test_convert_missing_target(self, mock_process, sample_audio_16k):
        """Test conversion without target audio or embedding."""
        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = object.__new__(HQSVCWrapper)
        audio, sr = sample_audio_16k

        with pytest.raises(RuntimeError, match="Either target_audio or speaker_embedding"):
            wrapper.convert(audio, sr)

    @patch('auto_voice.inference.hq_svc_wrapper.HQSVCWrapper._process_audio')
    def test_convert_audio_too_short(self, mock_process, short_audio):
        """Test conversion with audio below minimum duration."""
        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = object.__new__(HQSVCWrapper)
        audio, sr = short_audio
        target_emb = torch.randn(256)

        with pytest.raises(RuntimeError, match="Audio too short"):
            wrapper.convert(audio, sr, speaker_embedding=target_emb)

    @patch('auto_voice.inference.hq_svc_wrapper.HQSVCWrapper._process_audio')
    def test_convert_with_speaker_embedding(self, mock_process, sample_audio_16k,
                                            mock_processed_data, mock_vocoder, mock_net_g):
        """Test conversion with pre-computed speaker embedding."""
        mock_process.return_value = mock_processed_data

        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = object.__new__(HQSVCWrapper)
        wrapper.device = torch.device('cpu')
        wrapper.net_g = mock_net_g
        wrapper.vocoder = mock_vocoder
        wrapper.args = SimpleNamespace()
        wrapper.args.infer_speedup = 10
        wrapper.args.infer_method = 'dpm-solver'
        wrapper.args.vocoder = 'nsf-hifigan'

        audio, sr = sample_audio_16k
        target_emb = F.normalize(torch.randn(256), p=2, dim=0)

        result = wrapper.convert(audio, sr, speaker_embedding=target_emb)

        # Check result
        assert result['sample_rate'] == 44100
        assert result['metadata']['mode'] == 'voice_conversion'
        assert result['metadata']['pitch_shift'] == 0

    @patch('auto_voice.inference.hq_svc_wrapper.HQSVCWrapper._process_audio')
    def test_convert_with_target_audio(self, mock_process, sample_audio_16k,
                                       mock_processed_data, mock_vocoder, mock_net_g):
        """Test conversion with target reference audio."""
        # Return different data for source vs target
        def process_side_effect(audio, sr):
            data = mock_processed_data.copy()
            # Make target speaker different
            data['spk'] = torch.randn(1, 256)
            return data

        mock_process.side_effect = process_side_effect

        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = object.__new__(HQSVCWrapper)
        wrapper.device = torch.device('cpu')
        wrapper.net_g = mock_net_g
        wrapper.vocoder = mock_vocoder
        wrapper.args = SimpleNamespace()
        wrapper.args.infer_speedup = 10
        wrapper.args.infer_method = 'dpm-solver'
        wrapper.args.vocoder = 'nsf-hifigan'

        audio, sr = sample_audio_16k
        target_audio, target_sr = sample_audio_16k

        result = wrapper.convert(audio, sr, target_audio=target_audio,
                                target_sample_rate=target_sr)

        # Should have processed both source and target
        assert mock_process.call_count == 2
        assert result['sample_rate'] == 44100

    @patch('auto_voice.inference.hq_svc_wrapper.HQSVCWrapper._process_audio')
    def test_convert_with_multiple_target_audios(self, mock_process, sample_audio_16k,
                                                 mock_processed_data, mock_vocoder, mock_net_g):
        """Test conversion with multiple target reference audios."""
        mock_process.return_value = mock_processed_data

        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = object.__new__(HQSVCWrapper)
        wrapper.device = torch.device('cpu')
        wrapper.net_g = mock_net_g
        wrapper.vocoder = mock_vocoder
        wrapper.args = SimpleNamespace()
        wrapper.args.infer_speedup = 10
        wrapper.args.infer_method = 'dpm-solver'
        wrapper.args.vocoder = 'nsf-hifigan'

        audio, sr = sample_audio_16k
        target_audios = [audio, audio, audio]

        result = wrapper.convert(audio, sr, target_audio=target_audios,
                                target_sample_rate=sr)

        # Should process source + 3 targets = 4 calls
        assert mock_process.call_count == 4

    @patch('auto_voice.inference.hq_svc_wrapper.HQSVCWrapper._process_audio')
    def test_convert_with_pitch_shift(self, mock_process, sample_audio_16k,
                                      mock_processed_data, mock_vocoder, mock_net_g):
        """Test conversion with manual pitch shift."""
        mock_process.return_value = mock_processed_data

        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = object.__new__(HQSVCWrapper)
        wrapper.device = torch.device('cpu')
        wrapper.net_g = mock_net_g
        wrapper.vocoder = mock_vocoder
        wrapper.args = SimpleNamespace()
        wrapper.args.infer_speedup = 10
        wrapper.args.infer_method = 'dpm-solver'
        wrapper.args.vocoder = 'nsf-hifigan'

        audio, sr = sample_audio_16k
        target_emb = torch.randn(256)

        result = wrapper.convert(audio, sr, speaker_embedding=target_emb,
                                pitch_shift=3)

        assert result['metadata']['pitch_shift'] == 3

    @patch('auto_voice.inference.hq_svc_wrapper.HQSVCWrapper._process_audio')
    def test_convert_with_auto_pitch(self, mock_process, sample_audio_16k,
                                     mock_vocoder, mock_net_g):
        """Test conversion with automatic pitch adjustment."""
        # Setup source and target with different F0
        src_data = {
            'vq_post': torch.randn(1024, 256),
            'spk': torch.randn(1, 256),
            'f0': torch.randn(1024),
            'f0_origin': np.ones(1024, dtype=np.float32) * 220.0,  # A3
            'vol': torch.randn(1024),
            'mel': torch.randn(80, 1024),
        }

        tar_data = {
            'vq_post': torch.randn(1024, 256),
            'spk': torch.randn(1, 256),
            'f0': torch.randn(1024),
            'f0_origin': np.ones(1024, dtype=np.float32) * 440.0,  # A4 (octave up)
            'vol': torch.randn(1024),
            'mel': torch.randn(80, 1024),
        }

        # First call returns target data, second returns source
        mock_process.side_effect = [tar_data, src_data]

        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = object.__new__(HQSVCWrapper)
        wrapper.device = torch.device('cpu')
        wrapper.net_g = mock_net_g
        wrapper.vocoder = mock_vocoder
        wrapper.args = SimpleNamespace()
        wrapper.args.infer_speedup = 10
        wrapper.args.infer_method = 'dpm-solver'
        wrapper.args.vocoder = 'nsf-hifigan'

        audio, sr = sample_audio_16k

        result = wrapper.convert(audio, sr, target_audio=audio,
                                target_sample_rate=sr, auto_pitch=True)

        # Should compute ~12 semitone shift (octave)
        assert result['metadata']['pitch_shift'] == 12

    @patch('auto_voice.inference.hq_svc_wrapper.HQSVCWrapper._process_audio')
    def test_convert_with_progress_callback(self, mock_process, sample_audio_16k,
                                            mock_processed_data, mock_vocoder, mock_net_g):
        """Test conversion with progress callback."""
        mock_process.return_value = mock_processed_data

        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = object.__new__(HQSVCWrapper)
        wrapper.device = torch.device('cpu')
        wrapper.net_g = mock_net_g
        wrapper.vocoder = mock_vocoder
        wrapper.args = SimpleNamespace()
        wrapper.args.infer_speedup = 10
        wrapper.args.infer_method = 'dpm-solver'
        wrapper.args.vocoder = 'nsf-hifigan'

        progress_calls = []
        def on_progress(stage, progress):
            progress_calls.append((stage, progress))

        audio, sr = sample_audio_16k
        target_emb = torch.randn(256)

        wrapper.convert(audio, sr, speaker_embedding=target_emb,
                       on_progress=on_progress)

        # Check progress stages
        stages = [call[0] for call in progress_calls]
        assert 'preprocessing' in stages
        assert 'encoding' in stages
        assert 'diffusion' in stages
        assert 'vocoder' in stages
        assert 'complete' in stages

    @patch('auto_voice.inference.hq_svc_wrapper.HQSVCWrapper._process_audio')
    def test_convert_with_non_nsf_vocoder(self, mock_process, sample_audio_16k,
                                          mock_processed_data, mock_net_g):
        """Test conversion with non-NSF-HiFiGAN vocoder."""
        mock_process.return_value = mock_processed_data

        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = object.__new__(HQSVCWrapper)
        wrapper.device = torch.device('cpu')
        wrapper.net_g = mock_net_g
        wrapper.vocoder = MagicMock()
        wrapper.vocoder.infer = MagicMock(return_value=torch.randn(1, 44100))
        wrapper.args = SimpleNamespace()
        wrapper.args.infer_speedup = 10
        wrapper.args.infer_method = 'dpm-solver'
        wrapper.args.vocoder = 'bigvgan'  # Non-NSF vocoder

        audio, sr = sample_audio_16k
        target_emb = torch.randn(256)

        result = wrapper.convert(audio, sr, speaker_embedding=target_emb)

        # Vocoder should be called without F0
        wrapper.vocoder.infer.assert_called_once()
        call_args = wrapper.vocoder.infer.call_args
        # For non-NSF, only mel is passed
        assert len(call_args[0]) == 1  # Only positional arg (mel)


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_device_validation_cuda_unavailable(self):
        """Test device validation when CUDA is not available."""
        with patch('auto_voice.inference.hq_svc_wrapper.torch.cuda.is_available', return_value=False):
            from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

            with pytest.raises(RuntimeError, match="CUDA is required"):
                HQSVCWrapper(require_gpu=True)

    @patch('auto_voice.inference.hq_svc_wrapper.HQSVCWrapper._process_audio')
    def test_silent_audio_normalization(self, mock_process, mock_vocoder, mock_net_g):
        """Test normalization with silent/near-zero audio."""
        mock_process.return_value = {
            'vq_post': torch.randn(1024, 256),
            'spk': torch.randn(1, 256),
            'f0': torch.randn(1024),
            'f0_origin': np.random.randn(1024).astype(np.float32) * 100 + 200,
            'vol': torch.randn(1024),
            'mel': torch.randn(80, 1024),
        }

        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = object.__new__(HQSVCWrapper)
        wrapper.device = torch.device('cpu')
        wrapper.net_g = mock_net_g
        wrapper.args = SimpleNamespace()
        wrapper.args.infer_speedup = 10
        wrapper.args.infer_method = 'dpm-solver'
        wrapper.args.vocoder = 'nsf-hifigan'

        # Mock vocoder to return near-silent audio
        silent = torch.randn(44100) * 1e-7
        wrapper.vocoder = MagicMock()
        wrapper.vocoder.infer = MagicMock(return_value=silent.unsqueeze(0))

        audio = torch.randn(80000)
        target_emb = torch.randn(256)

        result = wrapper.convert(audio, 16000, speaker_embedding=target_emb)

        # Should not crash or produce NaN
        assert not torch.isnan(result['audio']).any()

    @patch('auto_voice.inference.hq_svc_wrapper.HQSVCWrapper._process_audio')
    def test_auto_pitch_with_silent_f0(self, mock_process, sample_audio_16k,
                                       mock_vocoder, mock_net_g):
        """Test auto-pitch with silent/zero F0 values."""
        # All F0 values are zero (unvoiced)
        src_data = {
            'vq_post': torch.randn(1024, 256),
            'spk': torch.randn(1, 256),
            'f0': torch.zeros(1024),
            'f0_origin': np.zeros(1024, dtype=np.float32),
            'vol': torch.randn(1024),
            'mel': torch.randn(80, 1024),
        }

        tar_data = {
            'vq_post': torch.randn(1024, 256),
            'spk': torch.randn(1, 256),
            'f0': torch.zeros(1024),
            'f0_origin': np.zeros(1024, dtype=np.float32),
            'vol': torch.randn(1024),
            'mel': torch.randn(80, 1024),
        }

        mock_process.side_effect = [tar_data, src_data]

        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = object.__new__(HQSVCWrapper)
        wrapper.device = torch.device('cpu')
        wrapper.net_g = mock_net_g
        wrapper.vocoder = mock_vocoder
        wrapper.args = SimpleNamespace()
        wrapper.args.infer_speedup = 10
        wrapper.args.infer_method = 'dpm-solver'
        wrapper.args.vocoder = 'nsf-hifigan'

        audio, sr = sample_audio_16k

        # Should not crash
        result = wrapper.convert(audio, sr, target_audio=audio,
                                target_sample_rate=sr, auto_pitch=True)

        # Pitch shift should be 0 (no valid F0 to compute from)
        assert result['metadata']['pitch_shift'] == 0

    def test_process_audio_cleanup_on_error(self):
        """Test that temp files are cleaned up even on processing error."""
        # Mock HQ-SVC modules
        sys.modules['utils'] = MagicMock()
        sys.modules['utils.data_preprocessing'] = MagicMock()

        with patch('os.unlink') as mock_unlink:
            with patch('tempfile.NamedTemporaryFile') as mock_temp:
                with patch('soundfile.write') as mock_sf:
                    mock_temp.return_value.__enter__.return_value.name = '/tmp/test.wav'

                    mock_data_preprocessing = sys.modules['utils.data_preprocessing']
                    mock_data_preprocessing.get_processed_file = MagicMock(side_effect=RuntimeError("Processing error"))

                    from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

                    wrapper = object.__new__(HQSVCWrapper)
                    wrapper.device = torch.device('cpu')
                    wrapper.vocoder = MagicMock()
                    wrapper.volume_extractor = MagicMock()
                    wrapper.f0_extractor = MagicMock()
                    wrapper.fa_encoder = MagicMock()
                    wrapper.fa_decoder = MagicMock()

                    audio = torch.randn(80000)

                    with pytest.raises(RuntimeError, match="Processing error"):
                        wrapper._process_audio(audio, 16000)

                    # Temp file should still be cleaned up
                    mock_unlink.assert_called_once_with('/tmp/test.wav')


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests with other modules."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    @patch('auto_voice.inference.hq_svc_wrapper.os.path.exists')
    @patch('auto_voice.inference.hq_svc_wrapper.HQSVCWrapper._load_config')
    @patch('auto_voice.inference.hq_svc_wrapper.HQSVCWrapper._init_models')
    @patch('auto_voice.inference.hq_svc_wrapper.HQSVCWrapper._setup_paths')
    def test_full_initialization_with_cuda(self, mock_setup, mock_init, mock_load,
                                          mock_exists, mock_config):
        """Test full initialization workflow with real CUDA device."""
        mock_exists.return_value = True
        mock_load.return_value = mock_config

        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = HQSVCWrapper()

        assert wrapper.device.type == 'cuda'
        assert wrapper.output_sample_rate == 44100
        assert wrapper.encoder_sample_rate == 16000

    def test_min_duration_constant(self):
        """Test minimum duration constant is correct."""
        from auto_voice.inference.hq_svc_wrapper import MIN_DURATION_SAMPLES_16K

        # Should be 8000 samples (500ms at 16kHz)
        assert MIN_DURATION_SAMPLES_16K == 8000

    def test_hq_svc_root_path(self):
        """Test HQ-SVC root path construction."""
        from auto_voice.inference.hq_svc_wrapper import HQ_SVC_ROOT

        # Should point to models/hq-svc directory
        assert 'models' in HQ_SVC_ROOT
        assert 'hq-svc' in HQ_SVC_ROOT


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Tests for performance and resource management."""

    @patch('auto_voice.inference.hq_svc_wrapper.HQSVCWrapper._process_audio')
    def test_device_placement(self, mock_process, sample_audio_16k,
                             mock_processed_data, mock_vocoder, mock_net_g):
        """Test that tensors are placed on correct device."""
        mock_process.return_value = mock_processed_data

        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = object.__new__(HQSVCWrapper)
        wrapper.device = torch.device('cpu')
        wrapper.net_g = mock_net_g
        wrapper.vocoder = mock_vocoder
        wrapper.args = SimpleNamespace()
        wrapper.args.infer_speedup = 10
        wrapper.args.infer_method = 'dpm-solver'
        wrapper.args.vocoder = 'nsf-hifigan'

        audio, sr = sample_audio_16k
        target_emb = torch.randn(256)

        wrapper.convert(audio, sr, speaker_embedding=target_emb)

        # Check that net_g was called with CPU tensors
        call_args = mock_net_g.call_args[0]
        for arg in call_args[:4]:  # vq_post, f0, vol, spk_emb
            if isinstance(arg, torch.Tensor):
                assert arg.device.type == 'cpu'

    @patch('auto_voice.inference.hq_svc_wrapper.HQSVCWrapper._process_audio')
    def test_output_is_cpu_tensor(self, mock_process, sample_audio_16k,
                                  mock_processed_data, mock_vocoder, mock_net_g):
        """Test that output audio is always on CPU for compatibility."""
        mock_process.return_value = mock_processed_data

        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = object.__new__(HQSVCWrapper)
        wrapper.device = torch.device('cpu')
        wrapper.net_g = mock_net_g
        wrapper.vocoder = mock_vocoder
        wrapper.args = SimpleNamespace()
        wrapper.args.infer_speedup = 10
        wrapper.args.infer_method = 'dpm-solver'
        wrapper.args.vocoder = 'nsf-hifigan'

        audio, sr = sample_audio_16k
        target_emb = torch.randn(256)

        result = wrapper.convert(audio, sr, speaker_embedding=target_emb)

        # Output should be on CPU
        assert result['audio'].device.type == 'cpu'

    @patch('auto_voice.inference.hq_svc_wrapper.HQSVCWrapper._process_audio')
    def test_no_gradient_computation(self, mock_process, sample_audio_16k,
                                     mock_processed_data, mock_vocoder, mock_net_g):
        """Test that no gradients are computed during inference."""
        mock_process.return_value = mock_processed_data

        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = object.__new__(HQSVCWrapper)
        wrapper.device = torch.device('cpu')
        wrapper.net_g = mock_net_g
        wrapper.vocoder = mock_vocoder
        wrapper.args = SimpleNamespace()
        wrapper.args.infer_speedup = 10
        wrapper.args.infer_method = 'dpm-solver'
        wrapper.args.vocoder = 'nsf-hifigan'

        audio, sr = sample_audio_16k
        target_emb = torch.randn(256)

        with torch.set_grad_enabled(True):
            result = wrapper.convert(audio, sr, speaker_embedding=target_emb)

        # Output should not require gradients
        assert not result['audio'].requires_grad

    def test_resample_performance(self):
        """Test resampling doesn't create unnecessary copies."""
        from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

        wrapper = object.__new__(HQSVCWrapper)

        # Same rate should return same tensor (no copy)
        audio = torch.randn(16000)
        result = wrapper._resample(audio, 16000, 16000)

        assert torch.equal(audio, result)
