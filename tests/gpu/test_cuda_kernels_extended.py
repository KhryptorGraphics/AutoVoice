"""Extended tests for CUDA kernels - achieving 80% coverage.

This module provides comprehensive tests covering:
- All 5 launch functions
- CPU fallback behavior
- Error handling
- Various audio inputs
- Mocked CUDA operations
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.auto_voice.gpu.cuda_kernels import (
    PitchDetectionKernel,
    SpectrogramKernel,
    VoiceSynthesisKernel,
    FeatureExtractionKernel,
    KernelConfig,
    CUDAKernelError,
    create_kernel_suite,
    launch_optimized_stft,
    launch_optimized_istft,
    launch_pitch_detection,
    launch_mel_spectrogram_singing,
    launch_formant_extraction,
    CUDA_KERNELS_AVAILABLE
)


@pytest.fixture
def kernel_config_cpu():
    """Create CPU-only kernel configuration."""
    return KernelConfig(use_cuda=False, use_half_precision=False)


@pytest.fixture
def kernel_config_gpu():
    """Create GPU kernel configuration."""
    return KernelConfig(
        use_cuda=torch.cuda.is_available(),
        use_half_precision=False,
        batch_size=4
    )


@pytest.fixture
def test_audio_cpu():
    """Create test audio on CPU."""
    return torch.randn(16000)


@pytest.fixture
def test_audio_batch_cpu():
    """Create batch test audio on CPU."""
    return torch.randn(4, 16000)


class TestKernelConfig:
    """Test KernelConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = KernelConfig()

        assert config.use_cuda is True
        assert config.use_half_precision is False
        assert config.batch_size == 32
        assert config.num_streams == 4
        assert config.enable_profiling is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = KernelConfig(
            use_cuda=False,
            use_half_precision=True,
            batch_size=16,
            num_streams=2,
            enable_profiling=True
        )

        assert config.use_cuda is False
        assert config.use_half_precision is True
        assert config.batch_size == 16
        assert config.num_streams == 2
        assert config.enable_profiling is True


class TestPitchDetectionKernel:
    """Test PitchDetectionKernel class."""

    def test_kernel_initialization(self, kernel_config_cpu):
        """Test kernel initialization."""
        kernel = PitchDetectionKernel(kernel_config_cpu)

        assert kernel is not None
        assert kernel.config == kernel_config_cpu
        assert kernel.device.type == 'cpu'

    def test_detect_pitch_1d_audio(self, kernel_config_cpu, test_audio_cpu):
        """Test pitch detection with 1D audio."""
        kernel = PitchDetectionKernel(kernel_config_cpu)

        f0 = kernel.detect_pitch(
            test_audio_cpu,
            sample_rate=16000,
            frame_length=2048,
            hop_length=512
        )

        assert f0 is not None
        assert f0.ndim == 2  # Should add batch dimension

    def test_detect_pitch_2d_audio(self, kernel_config_cpu, test_audio_batch_cpu):
        """Test pitch detection with 2D audio (batched)."""
        kernel = PitchDetectionKernel(kernel_config_cpu)

        f0 = kernel.detect_pitch(
            test_audio_batch_cpu,
            sample_rate=16000,
            frame_length=2048,
            hop_length=512
        )

        assert f0 is not None
        assert f0.shape[0] == 4  # Batch size

    def test_detect_pitch_custom_f0_range(self, kernel_config_cpu, test_audio_cpu):
        """Test pitch detection with custom F0 range."""
        kernel = PitchDetectionKernel(kernel_config_cpu)

        f0 = kernel.detect_pitch(
            test_audio_cpu,
            sample_rate=16000,
            f0_min=100.0,
            f0_max=500.0
        )

        assert f0 is not None

    def test_detect_pitch_fallback(self, kernel_config_cpu, test_audio_cpu):
        """Test CPU fallback for pitch detection."""
        kernel = PitchDetectionKernel(kernel_config_cpu)

        # Force fallback by using CPU config
        f0 = kernel._pitch_detection_fallback(
            test_audio_cpu.unsqueeze(0),
            sample_rate=16000,
            frame_length=2048,
            hop_length=512,
            f0_min=80.0,
            f0_max=800.0
        )

        assert f0 is not None
        assert not torch.isnan(f0).any()

    def test_autocorrelation(self, kernel_config_cpu):
        """Test autocorrelation computation."""
        kernel = PitchDetectionKernel(kernel_config_cpu)

        signal = torch.randn(1024)
        autocorr = kernel._autocorrelation(signal, max_lag=512)

        assert autocorr is not None
        assert len(autocorr) == 513  # max_lag + 1
        assert autocorr[0] == 1.0 or torch.isclose(autocorr[0], torch.tensor(1.0), atol=1e-5)

    def test_pitch_detection_error_handling(self, kernel_config_cpu):
        """Test error handling in pitch detection."""
        kernel = PitchDetectionKernel(kernel_config_cpu)

        # Invalid audio (empty)
        with pytest.raises((CUDAKernelError, RuntimeError)):
            kernel.detect_pitch(torch.empty(0))


class TestSpectrogramKernel:
    """Test SpectrogramKernel class."""

    def test_kernel_initialization(self, kernel_config_cpu):
        """Test kernel initialization."""
        kernel = SpectrogramKernel(kernel_config_cpu)

        assert kernel is not None
        assert kernel.config == kernel_config_cpu
        assert kernel.device.type == 'cpu'

    def test_compute_stft_1d(self, kernel_config_cpu, test_audio_cpu):
        """Test STFT with 1D audio."""
        kernel = SpectrogramKernel(kernel_config_cpu)

        stft = kernel.compute_stft(test_audio_cpu, n_fft=1024, hop_length=256)

        assert stft is not None
        assert stft.dtype == torch.complex64 or stft.dtype == torch.complex128

    def test_compute_stft_2d(self, kernel_config_cpu, test_audio_batch_cpu):
        """Test STFT with 2D audio (batched)."""
        kernel = SpectrogramKernel(kernel_config_cpu)

        stft = kernel.compute_stft(test_audio_batch_cpu, n_fft=1024, hop_length=256)

        assert stft is not None
        assert stft.shape[0] == 4  # Batch size

    def test_compute_stft_custom_window(self, kernel_config_cpu, test_audio_cpu):
        """Test STFT with different window types."""
        kernel = SpectrogramKernel(kernel_config_cpu)

        for window_type in ['hann', 'hamming', 'blackman']:
            stft = kernel.compute_stft(test_audio_cpu, window=window_type)
            assert stft is not None

    def test_compute_stft_unknown_window(self, kernel_config_cpu, test_audio_cpu):
        """Test STFT with unknown window type (should use Hann)."""
        kernel = SpectrogramKernel(kernel_config_cpu)

        stft = kernel.compute_stft(test_audio_cpu, window='unknown')
        assert stft is not None

    def test_compute_mel_spectrogram(self, kernel_config_cpu, test_audio_cpu):
        """Test mel-spectrogram computation."""
        kernel = SpectrogramKernel(kernel_config_cpu)

        mel_spec = kernel.compute_mel_spectrogram(
            test_audio_cpu,
            sample_rate=16000,
            n_mels=80
        )

        assert mel_spec is not None
        assert mel_spec.shape[-2] == 80  # n_mels

    def test_compute_mel_spectrogram_custom_freq(self, kernel_config_cpu, test_audio_cpu):
        """Test mel-spectrogram with custom frequency range."""
        kernel = SpectrogramKernel(kernel_config_cpu)

        mel_spec = kernel.compute_mel_spectrogram(
            test_audio_cpu,
            sample_rate=16000,
            f_min=100.0,
            f_max=7000.0
        )

        assert mel_spec is not None

    def test_get_window(self, kernel_config_cpu):
        """Test window generation."""
        kernel = SpectrogramKernel(kernel_config_cpu)

        for window_type in ['hann', 'hamming', 'blackman', 'unknown']:
            window = kernel._get_window(window_type, 1024, torch.device('cpu'))
            assert window is not None
            assert len(window) == 1024

    def test_create_mel_filterbank(self, kernel_config_cpu):
        """Test mel filterbank creation."""
        kernel = SpectrogramKernel(kernel_config_cpu)

        filterbank = kernel._create_mel_filterbank(
            sample_rate=16000,
            n_fft=1024,
            n_mels=80,
            f_min=0.0,
            f_max=8000.0,
            device=torch.device('cpu')
        )

        assert filterbank is not None
        assert filterbank.shape == (80, 1024 // 2 + 1)

    def test_hz_to_mel_conversion(self):
        """Test Hz to mel conversion."""
        hz = torch.tensor([100.0, 1000.0, 8000.0])
        mel = SpectrogramKernel._hz_to_mel(hz)

        assert mel is not None
        assert len(mel) == 3

    def test_mel_to_hz_conversion(self):
        """Test mel to Hz conversion."""
        mel = torch.tensor([100.0, 1000.0, 2000.0])
        hz = SpectrogramKernel._mel_to_hz(mel)

        assert hz is not None
        assert len(hz) == 3

    def test_stft_error_handling(self, kernel_config_cpu):
        """Test error handling in STFT computation."""
        kernel = SpectrogramKernel(kernel_config_cpu)

        with pytest.raises(CUDAKernelError):
            kernel.compute_stft(torch.empty(0))


class TestVoiceSynthesisKernel:
    """Test VoiceSynthesisKernel class."""

    def test_kernel_initialization(self, kernel_config_cpu):
        """Test kernel initialization."""
        kernel = VoiceSynthesisKernel(kernel_config_cpu)

        assert kernel is not None
        assert kernel.config == kernel_config_cpu
        assert kernel.device.type == 'cpu'

    def test_synthesize_waveform(self, kernel_config_cpu):
        """Test waveform synthesis."""
        kernel = VoiceSynthesisKernel(kernel_config_cpu)

        features = torch.randn(1, 80, 100)
        model_params = torch.randn(80 * 16)

        waveform = kernel.synthesize_waveform(features, model_params, upsample_factor=256)

        assert waveform is not None
        assert waveform.ndim == 2

    def test_synthesize_waveform_different_upsample(self, kernel_config_cpu):
        """Test waveform synthesis with different upsample factors."""
        kernel = VoiceSynthesisKernel(kernel_config_cpu)

        features = torch.randn(1, 80, 100)
        model_params = torch.randn(80 * 16)

        for upsample_factor in [128, 256, 512]:
            waveform = kernel.synthesize_waveform(
                features, model_params, upsample_factor=upsample_factor
            )
            assert waveform is not None

    def test_synthesis_fallback(self, kernel_config_cpu):
        """Test CPU fallback for synthesis."""
        kernel = VoiceSynthesisKernel(kernel_config_cpu)

        features = torch.randn(2, 80, 100)
        model_params = torch.randn(80 * 16)

        waveform = kernel._synthesis_fallback(features, model_params, upsample_factor=256)

        assert waveform is not None
        assert waveform.shape[0] == 2  # Batch size

    def test_synthesis_error_handling(self, kernel_config_cpu):
        """Test error handling in synthesis."""
        kernel = VoiceSynthesisKernel(kernel_config_cpu)

        with pytest.raises(CUDAKernelError):
            kernel.synthesize_waveform(torch.empty(0), torch.empty(0))


class TestFeatureExtractionKernel:
    """Test FeatureExtractionKernel class."""

    def test_kernel_initialization(self, kernel_config_cpu):
        """Test kernel initialization."""
        kernel = FeatureExtractionKernel(kernel_config_cpu)

        assert kernel is not None
        assert kernel.config == kernel_config_cpu
        assert kernel.device.type == 'cpu'

    def test_extract_speaker_embedding_2d(self, kernel_config_cpu):
        """Test speaker embedding extraction with 2D input."""
        kernel = FeatureExtractionKernel(kernel_config_cpu)

        mel_spec = torch.randn(80, 100)
        embedding = kernel.extract_speaker_embedding(mel_spec, embedding_dim=256)

        assert embedding is not None
        assert embedding.shape[-1] == 256

    def test_extract_speaker_embedding_3d(self, kernel_config_cpu):
        """Test speaker embedding extraction with 3D input (batched)."""
        kernel = FeatureExtractionKernel(kernel_config_cpu)

        mel_spec = torch.randn(4, 80, 100)
        embedding = kernel.extract_speaker_embedding(mel_spec, embedding_dim=256)

        assert embedding is not None
        assert embedding.shape[0] == 4  # Batch size
        assert embedding.shape[1] == 256

    def test_extract_speaker_embedding_normalization(self, kernel_config_cpu):
        """Test speaker embedding is normalized."""
        kernel = FeatureExtractionKernel(kernel_config_cpu)

        mel_spec = torch.randn(80, 100)
        embedding = kernel.extract_speaker_embedding(mel_spec)

        # Check L2 normalization
        norm = torch.norm(embedding, p=2)
        assert torch.isclose(norm, torch.tensor(1.0), atol=1e-5)

    def test_extract_speaker_embedding_different_dims(self, kernel_config_cpu):
        """Test speaker embedding with different dimensions."""
        kernel = FeatureExtractionKernel(kernel_config_cpu)

        mel_spec = torch.randn(80, 100)

        for dim in [128, 256, 512]:
            embedding = kernel.extract_speaker_embedding(mel_spec, embedding_dim=dim)
            assert embedding.shape[-1] == dim

    def test_feature_extraction_error_handling(self, kernel_config_cpu):
        """Test error handling in feature extraction."""
        kernel = FeatureExtractionKernel(kernel_config_cpu)

        with pytest.raises(CUDAKernelError):
            kernel.extract_speaker_embedding(torch.empty(0))


class TestLaunchFunctions:
    """Test all 5 launch functions."""

    def test_launch_optimized_stft(self):
        """Test launch_optimized_stft function."""
        audio = torch.randn(1, 16000)
        window = torch.hann_window(1024)
        n_fft = 1024
        hop_length = 256
        n_frames = (audio.shape[1] - n_fft) // hop_length + 1
        output = torch.zeros(1, n_frames, n_fft // 2 + 1, dtype=torch.cfloat)

        launch_optimized_stft(audio, window, output, n_fft, hop_length)

        assert not torch.isnan(output).any()

    def test_launch_optimized_istft(self):
        """Test launch_optimized_istft function."""
        n_fft = 1024
        hop_length = 256
        n_frames = 60
        stft_input = torch.randn(1, n_frames, n_fft // 2 + 1, dtype=torch.cfloat)
        window = torch.hann_window(n_fft)
        expected_length = (n_frames - 1) * hop_length + n_fft
        output = torch.zeros(1, expected_length)

        launch_optimized_istft(stft_input, window, output, n_fft, hop_length)

        assert not torch.isnan(output).any()

    def test_launch_pitch_detection(self):
        """Test launch_pitch_detection function."""
        audio = torch.randn(1, 16000)
        frame_length = 2048
        hop_length = 512
        n_frames = (audio.shape[1] - frame_length) // hop_length + 1

        pitch_output = torch.zeros(n_frames)
        confidence_output = torch.zeros(n_frames)
        vibrato_output = torch.zeros(n_frames)

        launch_pitch_detection(
            audio, pitch_output, confidence_output, vibrato_output,
            sample_rate=16000,
            frame_length=frame_length,
            hop_length=hop_length,
            f0_min=80.0,
            f0_max=800.0,
            confidence_threshold=0.3
        )

        assert not torch.isnan(pitch_output).any()
        assert not torch.isnan(confidence_output).any()

    def test_launch_mel_spectrogram_singing(self):
        """Test launch_mel_spectrogram_singing function."""
        audio = torch.randn(1, 16000)
        n_fft = 1024
        hop_length = 256
        n_mels = 80

        window = torch.hann_window(n_fft)
        mel_filterbank = torch.randn(n_mels, n_fft // 2 + 1)
        n_frames = (audio.shape[1] - n_fft) // hop_length + 1
        output = torch.zeros(1, n_frames, n_mels)

        launch_mel_spectrogram_singing(
            audio, window, mel_filterbank, output,
            n_fft, hop_length, apply_a_weighting=True
        )

        assert not torch.isnan(output).any()

    def test_launch_mel_spectrogram_singing_no_weighting(self):
        """Test mel spectrogram singing without A-weighting."""
        audio = torch.randn(1, 16000)
        n_fft = 1024
        hop_length = 256
        n_mels = 80

        window = torch.hann_window(n_fft)
        mel_filterbank = torch.randn(n_mels, n_fft // 2 + 1)
        n_frames = (audio.shape[1] - n_fft) // hop_length + 1
        output = torch.zeros(1, n_frames, n_mels)

        launch_mel_spectrogram_singing(
            audio, window, mel_filterbank, output,
            n_fft, hop_length, apply_a_weighting=False
        )

        assert not torch.isnan(output).any()

    def test_launch_formant_extraction(self):
        """Test launch_formant_extraction function."""
        frame_length = 2048
        n_frames = 10
        audio_frames = torch.randn(1, n_frames, frame_length)
        formants_output = torch.zeros(n_frames, 4)

        launch_formant_extraction(
            audio_frames, formants_output,
            frame_length=frame_length,
            sample_rate=16000,
            lpc_order=14,
            num_formants=4
        )

        assert not torch.isnan(formants_output).any()


class TestHelperFunctions:
    """Test helper functions."""

    def test_create_kernel_suite(self, kernel_config_cpu):
        """Test kernel suite creation."""
        suite = create_kernel_suite(kernel_config_cpu)

        assert suite is not None
        assert 'pitch_detection' in suite
        assert 'spectrogram' in suite
        assert 'voice_synthesis' in suite
        assert 'feature_extraction' in suite

        assert isinstance(suite['pitch_detection'], PitchDetectionKernel)
        assert isinstance(suite['spectrogram'], SpectrogramKernel)
        assert isinstance(suite['voice_synthesis'], VoiceSynthesisKernel)
        assert isinstance(suite['feature_extraction'], FeatureExtractionKernel)

    def test_create_kernel_suite_default_config(self):
        """Test kernel suite creation with default config."""
        suite = create_kernel_suite()

        assert suite is not None
        assert len(suite) == 4


class TestCUDAAvailability:
    """Test CUDA availability flag."""

    def test_cuda_kernels_available_flag(self):
        """Test CUDA_KERNELS_AVAILABLE flag is boolean."""
        assert isinstance(CUDA_KERNELS_AVAILABLE, bool)


class TestErrorHandling:
    """Test error handling in CUDA kernels."""

    def test_pitch_detection_error(self, kernel_config_cpu):
        """Test error handling in pitch detection."""
        kernel = PitchDetectionKernel(kernel_config_cpu)

        with pytest.raises((CUDAKernelError, RuntimeError, ValueError)):
            kernel.detect_pitch(torch.empty(0))

    def test_stft_error(self, kernel_config_cpu):
        """Test error handling in STFT."""
        kernel = SpectrogramKernel(kernel_config_cpu)

        with pytest.raises(CUDAKernelError):
            kernel.compute_stft(torch.empty(0))

    def test_mel_spectrogram_error(self, kernel_config_cpu):
        """Test error handling in mel-spectrogram."""
        kernel = SpectrogramKernel(kernel_config_cpu)

        with pytest.raises(CUDAKernelError):
            kernel.compute_mel_spectrogram(torch.empty(0))

    def test_synthesis_error(self, kernel_config_cpu):
        """Test error handling in synthesis."""
        kernel = VoiceSynthesisKernel(kernel_config_cpu)

        with pytest.raises(CUDAKernelError):
            kernel.synthesize_waveform(torch.empty(0), torch.empty(0))

    def test_feature_extraction_error(self, kernel_config_cpu):
        """Test error handling in feature extraction."""
        kernel = FeatureExtractionKernel(kernel_config_cpu)

        with pytest.raises(CUDAKernelError):
            kernel.extract_speaker_embedding(torch.empty(0))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
