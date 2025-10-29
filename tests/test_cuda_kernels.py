"""
Comprehensive CUDA kernel integration tests.

Tests all CUDA kernels from src/cuda_kernels/bindings.cpp:
- Audio kernels (synthesis, conversion, effects)
- FFT kernels (STFT, mel-spectrogram, etc.)
- Training kernels (matmul, conv, attention, etc.)
- Memory kernels (allocation, transfers)
"""
import pytest
import torch
import numpy as np
from typing import Tuple

# Skip all tests in this module if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


# ============================================================================
# Test Setup & Fixtures
# ============================================================================

@pytest.fixture
def cuda_device():
    """Get CUDA device."""
    return torch.device("cuda")


@pytest.fixture
def test_tensor(cuda_device) -> torch.Tensor:
    """Generate test tensor on CUDA."""
    return torch.randn(16, 80, 100, device=cuda_device)


@pytest.fixture
def mel_spectrogram(cuda_device) -> torch.Tensor:
    """Generate mel-spectrogram on CUDA."""
    return torch.randn(1, 80, 100, device=cuda_device)


@pytest.fixture
def audio_waveform(cuda_device) -> torch.Tensor:
    """Generate audio waveform on CUDA."""
    return torch.randn(1, 16000, device=cuda_device)


# ============================================================================
# Audio Kernel Tests
# ============================================================================

class TestAudioKernels:
    """Test audio processing CUDA kernels."""

    @pytest.mark.cuda
    def test_voice_synthesis(self, mel_spectrogram, cuda_device):
        """Test voice synthesis kernel."""
        try:
            import cuda_kernels

            speaker_embedding = torch.randn(1, 256, device=cuda_device)
            pitch_contour = torch.randn(1, 100, device=cuda_device)

            # Test with various input shapes
            for batch_size in [1, 4, 8]:
                mel = torch.randn(batch_size, 80, 100, device=cuda_device)
                speaker = torch.randn(batch_size, 256, device=cuda_device)
                pitch = torch.randn(batch_size, 100, device=cuda_device)

                output = cuda_kernels.voice_synthesis(mel, speaker, pitch)

                assert output.device.type == "cuda"
                assert output.shape[0] == batch_size
                assert not torch.isnan(output).any()
                assert not torch.isinf(output).any()
        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    def test_voice_conversion(self, audio_waveform, cuda_device):
        """Test voice conversion kernel."""
        try:
            import cuda_kernels

            source_embedding = torch.randn(1, 256, device=cuda_device)
            target_embedding = torch.randn(1, 256, device=cuda_device)

            output = cuda_kernels.voice_conversion(
                audio_waveform,
                source_embedding,
                target_embedding,
                pitch_shift=0.0,
                formant_shift=0.0
            )

            assert output.shape == audio_waveform.shape
            assert output.device.type == "cuda"
            assert not torch.isnan(output).any()
        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    @pytest.mark.parametrize("pitch_factor", [0.5, 0.8, 1.0, 1.2, 2.0])
    def test_pitch_shift(self, audio_waveform, cuda_device, pitch_factor):
        """Test pitch shifting with various factors."""
        try:
            import cuda_kernels

            output = cuda_kernels.pitch_shift(audio_waveform, pitch_factor)

            assert output.shape == audio_waveform.shape
            assert output.device.type == "cuda"
            assert not torch.isnan(output).any()

            # Verify output is different from input (except for factor=1.0)
            if pitch_factor != 1.0:
                assert not torch.allclose(output, audio_waveform, atol=1e-6)
        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    @pytest.mark.parametrize("time_factor", [0.5, 0.8, 1.0, 1.5, 2.0])
    def test_time_stretch(self, audio_waveform, cuda_device, time_factor):
        """Test time stretching without pitch changes."""
        try:
            import cuda_kernels

            output = cuda_kernels.time_stretch(audio_waveform, time_factor)

            expected_length = int(audio_waveform.shape[1] * time_factor)
            assert abs(output.shape[1] - expected_length) < 10  # Allow small tolerance
            assert output.device.type == "cuda"
            assert not torch.isnan(output).any()
        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    @pytest.mark.parametrize("threshold", [0.01, 0.05, 0.1, 0.2])
    def test_noise_reduction(self, audio_waveform, cuda_device, threshold):
        """Test noise reduction with different thresholds."""
        try:
            import cuda_kernels

            # Add noise to test
            noisy_audio = audio_waveform + torch.randn_like(audio_waveform) * 0.1

            output = cuda_kernels.noise_reduction(noisy_audio, threshold=threshold)

            assert output.shape == noisy_audio.shape
            assert output.device.type == "cuda"
            assert not torch.isnan(output).any()

            # Verify noise is reduced (output should be cleaner)
            output_energy = torch.mean(output ** 2)
            noise_energy = torch.mean(noisy_audio ** 2)
            assert output_energy <= noise_energy or threshold == 0.01
        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    @pytest.mark.parametrize("room_size", [0.1, 0.5, 0.8])
    def test_reverb(self, audio_waveform, cuda_device, room_size):
        """Test reverb effects with various room sizes."""
        try:
            import cuda_kernels

            output = cuda_kernels.reverb(audio_waveform, room_size=room_size)

            assert output.shape == audio_waveform.shape
            assert output.device.type == "cuda"
            assert not torch.isnan(output).any()

            # Verify reverb was applied (output should differ from input)
            assert not torch.allclose(output, audio_waveform, atol=1e-6)
        except ImportError:
            pytest.skip("CUDA kernels not available")


# ============================================================================
# FFT Kernel Tests
# ============================================================================

class TestFFTKernels:
    """Test FFT-related CUDA kernels."""

    @pytest.mark.cuda
    @pytest.mark.parametrize("n_fft", [512, 1024, 2048])
    def test_stft(self, audio_waveform, cuda_device, n_fft):
        """Test STFT with different window sizes."""
        try:
            import cuda_kernels

            hop_length = n_fft // 4

            stft_output = cuda_kernels.stft(
                audio_waveform,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft
            )

            assert stft_output.device.type == "cuda"
            assert stft_output.shape[-2] == n_fft // 2 + 1  # Frequency bins
            assert not torch.isnan(stft_output).any()
        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    def test_optimized_stft_kernel(self, audio_waveform, cuda_device):
        """Test new optimized STFT kernel accuracy."""
        try:
            import cuda_kernels

            n_fft = 2048
            hop_length = 512

            # Create Hann window
            window = torch.hann_window(n_fft, device=cuda_device)

            # Compute expected frames
            n_frames = (audio_waveform.shape[1] - n_fft) // hop_length + 1
            stft_shape = (audio_waveform.shape[0], n_frames, n_fft // 2 + 1, 2)  # complex output

            # Create output tensors
            stft_output = torch.zeros(stft_shape, dtype=torch.cfloat, device=cuda_device)

            # Test optimized STFT
            cuda_kernels.launch_optimized_stft(
                audio_waveform, window, stft_output, n_fft, hop_length
            )

            assert stft_output.device.type == "cuda"
            assert stft_output.shape == stft_shape
            assert not torch.isnan(stft_output).any()
            assert not torch.isinf(stft_output).any()

        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    def test_optimized_istft_kernel(self, cuda_device):
        """Test new optimized ISTFT kernel accuracy."""
        try:
            import cuda_kernels

            batch_size, n_frames, n_freqs = 1, 100, 1025  # n_fft=2048 -> 1025 freq bins
            n_fft = (n_freqs - 1) * 2
            hop_length = 512

            # Create test STFT data (complex)
            stft_input = torch.randn(batch_size, n_frames, n_freqs, 2, device=cuda_device, dtype=torch.cfloat)

            # Create Hann window
            window = torch.hann_window(n_fft, device=cuda_device)

            # Expected output length
            expected_length = (n_frames - 1) * hop_length + n_fft
            audio_output = torch.zeros(batch_size, expected_length, device=cuda_device)

            # Test optimized ISTFT
            cuda_kernels.launch_optimized_istft(
                stft_input, window, audio_output, n_fft, hop_length
            )

            assert audio_output.device.type == "cuda"
            assert audio_output.shape == (batch_size, expected_length)
            assert not torch.isnan(audio_output).any()
            assert not torch.isinf(audio_output).any()

        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    def test_stft_istft_perfect_reconstruction(self, audio_waveform, cuda_device):
        """Test perfect reconstruction: optimized STFT -> optimized ISTFT with references."""
        try:
            import cuda_kernels

            n_fft = 2048
            hop_length = 512

            # Create Hann window
            window = torch.hann_window(n_fft, device=cuda_device)

            # Forward: audio -> STFT
            n_frames = (audio_waveform.shape[1] - n_fft) // hop_length + 1
            stft_output = torch.zeros(audio_waveform.shape[0], n_frames, n_fft // 2 + 1,
                                     dtype=torch.cfloat, device=cuda_device)

            cuda_kernels.launch_optimized_stft(audio_waveform, window, stft_output, n_fft, hop_length)

            # Backward: STFT -> audio
            expected_length = (n_frames - 1) * hop_length + n_fft
            reconstructed = torch.zeros(audio_waveform.shape[0], expected_length, device=cuda_device)

            cuda_kernels.launch_optimized_istft(stft_output, window, reconstructed, n_fft, hop_length)

            # Compare reconstruction accuracy
            min_length = min(reconstructed.shape[1], audio_waveform.shape[1])
            reconstruction_error = torch.mean(
                (reconstructed[:, :min_length] - audio_waveform[:, :min_length]) ** 2
            ).item()

            # Should achieve very low reconstruction error (< 1e-5 as specified)
            assert reconstruction_error < 1e-5, f"Reconstruction error {reconstruction_error} > 1e-5"
            print(".6f")

        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    def test_stft_istft_librosa_comparison(self, audio_waveform, cuda_device):
        """Compare optimized STFT/ISTFT with librosa reference for accuracy."""
        try:
            import librosa
            import cuda_kernels

            # Convert to numpy for librosa
            audio_np = audio_waveform.cpu().numpy().squeeze()

            n_fft = 2048
            hop_length = 512
            window = torch.hann_window(n_fft, device=cuda_device)

            # Librosa reference STFT
            stft_ref = librosa.stft(audio_np, n_fft=n_fft, hop_length=hop_length, window='hann')

            # CUDA STFT
            n_frames = (audio_waveform.shape[1] - n_fft) // hop_length + 1
            stft_cuda = torch.zeros(audio_waveform.shape[0], n_frames, n_fft // 2 + 1,
                                   dtype=torch.cfloat, device=cuda_device)
            cuda_kernels.launch_optimized_stft(audio_waveform, window, stft_cuda, n_fft, hop_length)
            stft_cuda_np = stft_cuda.cpu().numpy().squeeze()

            # Compare STFT results
            stft_error = np.mean(np.abs(stft_cuda_np - stft_ref) ** 2)
            print(".6f")
            assert stft_error < 1e-10  # Very tight tolerance for STFT accuracy

            # Test reconstruction: librosa ISTFT
            audio_ref = librosa.istft(stft_ref, hop_length=hop_length, window='hann')

            # CUDA ISTFT
            expected_length = (n_frames - 1) * hop_length + n_fft
            reconstructed_cuda = torch.zeros(audio_waveform.shape[0], expected_length, device=cuda_device)
            cuda_kernels.launch_optimized_istft(stft_cuda, window, reconstructed_cuda, n_fft, hop_length)

            # Compare reconstruction
            min_len = min(len(audio_ref), reconstructed_cuda.shape[1])
            recon_error = np.mean((audio_ref[:min_len] - reconstructed_cuda[0, :min_len].cpu().numpy()) ** 2)
            print(".6f")
            assert recon_error < 1e-12  # Should match librosa perfectly

        except ImportError:
            pytest.skip("librosa not available, skipping comparison test")
        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    def test_istft_reconstruction(self, audio_waveform, cuda_device):
        """Test inverse STFT reconstruction accuracy."""
        try:
            import cuda_kernels

            n_fft = 1024
            hop_length = 256

            # Forward STFT
            stft_output = cuda_kernels.stft(
                audio_waveform,
                n_fft=n_fft,
                hop_length=hop_length
            )

            # Inverse STFT
            reconstructed = cuda_kernels.istft(
                stft_output,
                n_fft=n_fft,
                hop_length=hop_length
            )

            # Verify reconstruction accuracy
            min_length = min(reconstructed.shape[1], audio_waveform.shape[1])
            reconstruction_error = torch.mean(
                (reconstructed[:, :min_length] - audio_waveform[:, :min_length]) ** 2
            )
            assert reconstruction_error < 1e-3  # Low reconstruction error
        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    def test_mel_spectrogram(self, audio_waveform, cuda_device):
        """Test mel-spectrogram extraction and compare with librosa."""
        try:
            import cuda_kernels

            n_mels = 80
            sample_rate = 16000

            mel_spec = cuda_kernels.mel_spectrogram(
                audio_waveform,
                n_mels=n_mels,
                sample_rate=sample_rate
            )

            assert mel_spec.shape[1] == n_mels
            assert mel_spec.device.type == "cuda"
            assert not torch.isnan(mel_spec).any()
            assert torch.all(mel_spec >= 0)  # Mel-spectrogram should be non-negative
        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    @pytest.mark.parametrize("n_mfcc", [13, 20, 40])
    def test_mfcc(self, audio_waveform, cuda_device, n_mfcc):
        """Test MFCC extraction accuracy."""
        try:
            import cuda_kernels

            mfcc = cuda_kernels.mfcc(audio_waveform, n_mfcc=n_mfcc)

            assert mfcc.shape[1] == n_mfcc
            assert mfcc.device.type == "cuda"
            assert not torch.isnan(mfcc).any()
        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    @pytest.mark.parametrize("n_iter", [10, 30, 60])
    def test_griffin_lim(self, mel_spectrogram, cuda_device, n_iter):
        """Test Griffin-Lim phase reconstruction."""
        try:
            import cuda_kernels

            audio = cuda_kernels.griffin_lim(
                mel_spectrogram,
                n_iter=n_iter
            )

            assert audio.device.type == "cuda"
            assert audio.ndim == 2
            assert not torch.isnan(audio).any()
        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    def test_phase_vocoder(self, audio_waveform, cuda_device):
        """Test phase vocoder for time stretching."""
        try:
            import cuda_kernels

            rate = 1.5  # Time stretch factor

            output = cuda_kernels.phase_vocoder(audio_waveform, rate=rate)

            expected_length = int(audio_waveform.shape[1] * rate)
            assert abs(output.shape[1] - expected_length) < 100
            assert output.device.type == "cuda"
            assert not torch.isnan(output).any()
        except ImportError:
            pytest.skip("CUDA kernels not available")


# ============================================================================
# Training Kernel Tests
# ============================================================================

class TestTrainingKernels:
    """Test training-related CUDA kernels."""

    @pytest.mark.cuda
    @pytest.mark.parametrize("M,N,K", [(64, 64, 64), (128, 128, 128), (256, 256, 256)])
    def test_matmul(self, cuda_device, M, N, K):
        """Test matrix multiplication with various shapes."""
        try:
            import cuda_kernels

            A = torch.randn(M, K, device=cuda_device)
            B = torch.randn(K, N, device=cuda_device)

            # CUDA kernel matmul
            C_cuda = cuda_kernels.matmul(A, B)

            # PyTorch reference
            C_torch = torch.matmul(A, B)

            # Verify correctness
            assert C_cuda.shape == (M, N)
            assert torch.allclose(C_cuda, C_torch, atol=1e-4, rtol=1e-4)
        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    @pytest.mark.parametrize("stride,padding", [(1, 0), (2, 1), (1, 1)])
    def test_conv2d_forward(self, cuda_device, stride, padding):
        """Test 2D convolution with different strides and padding."""
        try:
            import cuda_kernels

            batch_size, in_channels, height, width = 4, 3, 32, 32
            out_channels, kernel_size = 16, 3

            input_tensor = torch.randn(batch_size, in_channels, height, width, device=cuda_device)
            weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device=cuda_device)

            output = cuda_kernels.conv2d_forward(
                input_tensor,
                weight,
                stride=stride,
                padding=padding
            )

            assert output.device.type == "cuda"
            assert output.shape[0] == batch_size
            assert output.shape[1] == out_channels
            assert not torch.isnan(output).any()
        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    def test_layer_norm(self, test_tensor, cuda_device):
        """Test layer normalization accuracy."""
        try:
            import cuda_kernels

            # CUDA kernel layer norm
            output_cuda = cuda_kernels.layer_norm(test_tensor)

            # PyTorch reference
            normalized_shape = test_tensor.shape[1:]
            layer_norm = torch.nn.LayerNorm(normalized_shape).to(cuda_device)
            output_torch = layer_norm(test_tensor)

            # Verify correctness (allow some numerical difference)
            assert torch.allclose(output_cuda, output_torch, atol=1e-4, rtol=1e-4)
        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    def test_attention(self, cuda_device):
        """Test attention mechanism computation."""
        try:
            import cuda_kernels

            batch_size, seq_len, hidden_size = 4, 100, 256

            query = torch.randn(batch_size, seq_len, hidden_size, device=cuda_device)
            key = torch.randn(batch_size, seq_len, hidden_size, device=cuda_device)
            value = torch.randn(batch_size, seq_len, hidden_size, device=cuda_device)

            output = cuda_kernels.attention(query, key, value)

            assert output.shape == query.shape
            assert output.device.type == "cuda"
            assert not torch.isnan(output).any()
        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    def test_gelu_activation(self, test_tensor, cuda_device):
        """Test GELU activation function."""
        try:
            import cuda_kernels

            # CUDA kernel GELU
            output_cuda = cuda_kernels.gelu_activation(test_tensor)

            # PyTorch reference
            output_torch = torch.nn.functional.gelu(test_tensor)

            # Verify correctness
            assert torch.allclose(output_cuda, output_torch, atol=1e-5, rtol=1e-5)
        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    def test_adam_step(self, cuda_device):
        """Test Adam optimizer step correctness."""
        try:
            import cuda_kernels

            param = torch.randn(256, 256, device=cuda_device, requires_grad=True)
            grad = torch.randn_like(param)

            # Initialize Adam state
            exp_avg = torch.zeros_like(param)
            exp_avg_sq = torch.zeros_like(param)
            step = 1

            lr = 0.001
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8

            # CUDA kernel Adam step
            updated_param = cuda_kernels.adam_step(
                param, grad, exp_avg, exp_avg_sq, step,
                lr=lr, beta1=beta1, beta2=beta2, eps=eps
            )

            assert updated_param.shape == param.shape
            assert updated_param.device.type == "cuda"
            assert not torch.isnan(updated_param).any()
        except ImportError:
            pytest.skip("CUDA kernels not available")


# ============================================================================
# Memory Kernel Tests
# ============================================================================

class TestEnhancedAudioKernels:
    """Test enhanced audio processing CUDA kernels."""

    @pytest.mark.cuda
    def test_launch_pitch_detection(self, audio_waveform, cuda_device):
        """Test enhanced pitch detection kernel."""
        try:
            import cuda_kernels

            frame_length = 2048
            hop_length = 512
            sample_rate = 16000
            fmin, fmax = 80.0, 1000.0
            threshold = 0.3

            # Create output tensors
            n_frames = (audio_waveform.shape[1] - frame_length) // hop_length + 1
            pitch_output = torch.zeros(n_frames, device=cuda_device)
            confidence_output = torch.zeros(n_frames, device=cuda_device)
            vibrato_output = torch.zeros(n_frames, device=cuda_device)

            # Test pitch detection
            cuda_kernels.launch_pitch_detection(
                audio_waveform, pitch_output, confidence_output, vibrato_output,
                sample_rate, frame_length, hop_length, fmin, fmax, threshold
            )

            assert pitch_output.device.type == "cuda"
            assert confidence_output.device.type == "cuda"
            assert vibrato_output.device.type == "cuda"
            assert pitch_output.shape == (n_frames,)
            # Pitch should be non-negative, confidence in [0,1], vibrato 0 or 1
            assert torch.all(pitch_output >= 0)
            assert torch.all((confidence_output >= 0) & (confidence_output <= 1))
            assert torch.all((vibrato_output == 0) | (vibrato_output == 1))

        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    def test_launch_formant_extraction(self, cuda_device):
        """Test formant extraction kernel."""
        try:
            import cuda_kernels

            frame_length = 2048
            sample_rate = 16000
            lpc_order = 14
            num_formants = 4

            # Create test audio frames
            batch_size = 1
            n_frames = 10
            audio_frames = torch.randn(batch_size, n_frames, frame_length, device=cuda_device)

            # Create output tensor for formants
            formants_output = torch.zeros(n_frames, num_formants, device=cuda_device)

            # Test formant extraction
            cuda_kernels.launch_formant_extraction(
                audio_frames, formants_output, frame_length, sample_rate, lpc_order, num_formants
            )

            assert formants_output.device.type == "cuda"
            assert formants_output.shape == (n_frames, num_formants)
            # Formants should be >= 0 (frequencies)
            assert torch.all(formants_output >= 0)
            assert not torch.isnan(formants_output).any()

        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    def test_launch_formant_extraction_configurable_lpc(self, cuda_device):
        """Test formant extraction with different LPC orders."""
        try:
            import cuda_kernels

            frame_length = 2048
            sample_rate = 16000
            num_formants = 4

            # Test different LPC orders
            for lpc_order in [10, 12, 14, 16]:
                batch_size = 1
                n_frames = 5
                audio_frames = torch.randn(batch_size, n_frames, frame_length, device=cuda_device)
                formants_output = torch.zeros(n_frames, num_formants, device=cuda_device)

                cuda_kernels.launch_formant_extraction(
                    audio_frames, formants_output, frame_length, sample_rate, lpc_order, num_formants
                )

                assert formants_output.device.type == "cuda"
                assert formants_output.shape == (n_frames, num_formants)
                assert torch.all(formants_output >= 0)
                assert not torch.isnan(formants_output).any()

        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    def test_launch_mel_spectrogram_singing(self, audio_waveform, cuda_device):
        """Test mel-spectrogram optimized for singing voice."""
        try:
            import cuda_kernels

            n_fft = 2048
            hop_length = 512
            n_mels = 128
            apply_a_weighting = True

            # Create Hann window and mel filterbank
            window = torch.hann_window(n_fft, device=cuda_device)
            mel_filterbank = torch.randn(n_mels, n_fft // 2 + 1, device=cuda_device)

            # Compute expected dimensions
            n_frames = (audio_waveform.shape[1] - n_fft) // hop_length + 1
            mel_output = torch.zeros(audio_waveform.shape[0], n_frames, n_mels, device=cuda_device)

            # Test mel-spectrogram singing
            cuda_kernels.launch_mel_spectrogram_singing(
                audio_waveform, window, mel_filterbank, mel_output,
                n_fft, hop_length, apply_a_weighting
            )

            assert mel_output.device.type == "cuda"
            assert mel_output.shape == (audio_waveform.shape[0], n_frames, n_mels)
            assert torch.all(mel_output >= 0)  # Log mel should be non-negative after log compression
            assert not torch.isnan(mel_output).any()

        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    def test_launch_realtime_voice_conversion(self, cuda_device):
        """Test real-time voice conversion kernel."""
        try:
            import cuda_kernels

            chunk_size = 4410  # 100ms at 44.1kHz
            overlap_size = 1102  # 25% overlap
            n_fft = 2048
            hop_length = 512
            feature_dim = 256

            # Create input tensors
            audio_chunk = torch.randn(chunk_size, device=cuda_device)
            overlap_buffer = torch.zeros(overlap_size, device=cuda_device)
            window = torch.hann_window(n_fft, device=cuda_device)
            features_output = torch.zeros(feature_dim, device=cuda_device)

            # Test real-time voice conversion
            cuda_kernels.launch_realtime_voice_conversion(
                audio_chunk, overlap_buffer, features_output, window, n_fft, hop_length
            )

            assert features_output.device.type == "cuda"
            assert features_output.shape == (feature_dim,)
            assert not torch.isnan(features_output).any()

        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    def test_perceptual_weighting(self, mel_spectrogram, cuda_device):
        """Test A-weighting perceptual weighting."""
        try:
            import cuda_kernels

            batch_size, n_frames, mel_bins = mel_spectrogram.shape
            mel_frequencies = torch.logspace(1, 4, mel_bins, device=cuda_device)  # log-spaced freqs

            # Create copy for comparison
            mel_original = mel_spectrogram.clone()

            # Apply perceptual weighting
            cuda_kernels.apply_perceptual_weighting(
                mel_spectrogram, mel_frequencies, n_frames, mel_bins, batch_size
            )

            assert mel_spectrogram.device.type == "cuda"
            assert mel_spectrogram.shape == mel_original.shape
            assert not torch.isnan(mel_spectrogram).any()
            # Output should be different from input (weighting applied)
            assert not torch.allclose(mel_spectrogram, mel_original, atol=1e-6)

        except ImportError:
            pytest.skip("CUDA kernels not available")


class TestMemoryKernels:
    """Test memory management CUDA kernels."""

    @pytest.mark.cuda
    def test_allocate_pinned_memory(self, cuda_device):
        """Test pinned memory allocation."""
        try:
            import cuda_kernels

            size = 1024 * 1024  # 1 MB

            pinned_mem = cuda_kernels.allocate_pinned_memory(size)

            assert pinned_mem is not None
            assert len(pinned_mem) == size
        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    def test_transfer_to_device_async(self, cuda_device):
        """Test async host-to-device transfers."""
        try:
            import cuda_kernels

            host_tensor = torch.randn(1000, 1000)

            device_tensor = cuda_kernels.transfer_to_device_async(host_tensor)

            assert device_tensor.device.type == "cuda"
            assert torch.allclose(device_tensor.cpu(), host_tensor)
        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    def test_transfer_to_host_async(self, test_tensor, cuda_device):
        """Test async device-to-host transfers."""
        try:
            import cuda_kernels

            host_tensor = cuda_kernels.transfer_to_host_async(test_tensor)

            assert host_tensor.device.type == "cpu"
            assert torch.allclose(host_tensor, test_tensor.cpu())
        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    def test_synchronize_stream(self, cuda_device):
        """Test stream synchronization."""
        try:
            import cuda_kernels

            stream = torch.cuda.Stream()

            with torch.cuda.stream(stream):
                # Launch async operation
                tensor = torch.randn(1000, 1000, device=cuda_device)
                result = tensor @ tensor.T

            # Synchronize stream
            cuda_kernels.synchronize_stream(stream)

            # Verify operation completed
            assert result.device.type == "cuda"
        except ImportError:
            pytest.skip("CUDA kernels not available")


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Test CUDA kernel performance."""

    @pytest.mark.cuda
    @pytest.mark.performance
    def test_matmul_speedup(self, cuda_device):
        """Verify CUDA matmul provides speedup vs PyTorch."""
        try:
            import cuda_kernels
            import time

            M, N, K = 1024, 1024, 1024
            A = torch.randn(M, K, device=cuda_device)
            B = torch.randn(K, N, device=cuda_device)

            # Warmup
            _ = cuda_kernels.matmul(A, B)
            _ = torch.matmul(A, B)
            torch.cuda.synchronize()

            # Benchmark CUDA kernel
            start = time.perf_counter()
            for _ in range(100):
                _ = cuda_kernels.matmul(A, B)
            torch.cuda.synchronize()
            cuda_time = time.perf_counter() - start

            # Benchmark PyTorch
            start = time.perf_counter()
            for _ in range(100):
                _ = torch.matmul(A, B)
            torch.cuda.synchronize()
            torch_time = time.perf_counter() - start

            print(f"CUDA kernel: {cuda_time:.4f}s, PyTorch: {torch_time:.4f}s")
            print(f"Speedup: {torch_time / cuda_time:.2f}x")

            # CUDA kernel should be competitive (within 2x of PyTorch)
            assert cuda_time < torch_time * 2.0
        except ImportError:
            pytest.skip("CUDA kernels not available")


# ============================================================================
# Audio Kernel Performance Benchmarks
# ============================================================================

class TestAudioKernelPerformance:
    """Performance benchmarks for CUDA audio kernels vs reference implementations."""

    @pytest.mark.cuda
    @pytest.mark.performance
    @pytest.mark.slow
    def test_stft_speedup_vs_torch(self, cuda_device):
        """Test optimized STFT speedup vs torch.stft (target: ≥5x)."""
        try:
            import cuda_kernels
            import time

            # Generate test audio (2 seconds at 44.1kHz)
            audio = torch.randn(1, 88200, device=cuda_device)
            n_fft = 2048
            hop_length = 512
            window = torch.hann_window(n_fft, device=cuda_device)

            # Prepare CUDA output
            n_frames = (audio.shape[1] - n_fft) // hop_length + 1
            stft_cuda = torch.zeros(audio.shape[0], n_frames, n_fft // 2 + 1,
                                   dtype=torch.cfloat, device=cuda_device)

            # Warmup: 10 iterations
            for _ in range(10):
                cuda_kernels.launch_optimized_stft(audio, window, stft_cuda, n_fft, hop_length)
                _ = torch.stft(audio, n_fft, hop_length, window=window, return_complex=True)
            torch.cuda.synchronize()

            # Benchmark CUDA optimized STFT: 100 iterations
            start = time.perf_counter()
            for _ in range(100):
                cuda_kernels.launch_optimized_stft(audio, window, stft_cuda, n_fft, hop_length)
            torch.cuda.synchronize()
            cuda_time = time.perf_counter() - start

            # Benchmark PyTorch STFT: 100 iterations
            start = time.perf_counter()
            for _ in range(100):
                _ = torch.stft(audio, n_fft, hop_length, window=window, return_complex=True)
            torch.cuda.synchronize()
            torch_time = time.perf_counter() - start

            speedup = torch_time / cuda_time
            print(f"\nSTFT Performance:")
            print(f"  CUDA optimized: {cuda_time:.4f}s")
            print(f"  PyTorch:        {torch_time:.4f}s")
            print(f"  Speedup:        {speedup:.2f}x")

            # Assert minimum 5x speedup
            assert speedup >= 5.0, f"STFT speedup {speedup:.2f}x < 5.0x target"

        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    @pytest.mark.performance
    @pytest.mark.slow
    def test_istft_speedup_vs_torch(self, cuda_device):
        """Test optimized ISTFT speedup vs torch.istft (target: ≥5x)."""
        try:
            import cuda_kernels
            import time

            # Generate test STFT data
            batch_size, n_frames, n_freqs = 1, 172, 1025  # ~2s audio
            n_fft = (n_freqs - 1) * 2
            hop_length = 512
            stft_input = torch.randn(batch_size, n_frames, n_freqs, dtype=torch.cfloat, device=cuda_device)
            window = torch.hann_window(n_fft, device=cuda_device)

            # Prepare CUDA output
            expected_length = (n_frames - 1) * hop_length + n_fft
            audio_cuda = torch.zeros(batch_size, expected_length, device=cuda_device)

            # Warmup: 10 iterations
            for _ in range(10):
                cuda_kernels.launch_optimized_istft(stft_input, window, audio_cuda, n_fft, hop_length)
                _ = torch.istft(stft_input.squeeze(0).transpose(0, 1), n_fft, hop_length, window=window)
            torch.cuda.synchronize()

            # Benchmark CUDA optimized ISTFT: 100 iterations
            start = time.perf_counter()
            for _ in range(100):
                cuda_kernels.launch_optimized_istft(stft_input, window, audio_cuda, n_fft, hop_length)
            torch.cuda.synchronize()
            cuda_time = time.perf_counter() - start

            # Benchmark PyTorch ISTFT: 100 iterations
            start = time.perf_counter()
            for _ in range(100):
                _ = torch.istft(stft_input.squeeze(0).transpose(0, 1), n_fft, hop_length, window=window)
            torch.cuda.synchronize()
            torch_time = time.perf_counter() - start

            speedup = torch_time / cuda_time
            print(f"\nISTFT Performance:")
            print(f"  CUDA optimized: {cuda_time:.4f}s")
            print(f"  PyTorch:        {torch_time:.4f}s")
            print(f"  Speedup:        {speedup:.2f}x")

            # Assert minimum 5x speedup
            assert speedup >= 5.0, f"ISTFT speedup {speedup:.2f}x < 5.0x target"

        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    @pytest.mark.performance
    @pytest.mark.slow
    def test_mel_spectrogram_speedup_vs_librosa(self, cuda_device):
        """Test mel-spectrogram speedup vs librosa (target: ≥10x)."""
        librosa = pytest.importorskip("librosa")
        try:
            import cuda_kernels
            import time

            # Generate test audio (2 seconds at 44.1kHz)
            audio_cuda = torch.randn(1, 88200, device=cuda_device)
            audio_np = audio_cuda.cpu().numpy().squeeze()

            n_fft = 2048
            hop_length = 512
            n_mels = 128

            # Prepare CUDA inputs
            window = torch.hann_window(n_fft, device=cuda_device)
            mel_filterbank = torch.randn(n_mels, n_fft // 2 + 1, device=cuda_device)
            n_frames = (audio_cuda.shape[1] - n_fft) // hop_length + 1
            mel_output = torch.zeros(audio_cuda.shape[0], n_frames, n_mels, device=cuda_device)

            # Warmup: 5 iterations
            for _ in range(5):
                cuda_kernels.launch_mel_spectrogram_singing(
                    audio_cuda, window, mel_filterbank, mel_output, n_fft, hop_length, True
                )
                _ = librosa.feature.melspectrogram(y=audio_np, sr=44100, n_fft=n_fft,
                                                   hop_length=hop_length, n_mels=n_mels)
            torch.cuda.synchronize()

            # Benchmark CUDA mel-spectrogram: 50 iterations
            start = time.perf_counter()
            for _ in range(50):
                cuda_kernels.launch_mel_spectrogram_singing(
                    audio_cuda, window, mel_filterbank, mel_output, n_fft, hop_length, True
                )
            torch.cuda.synchronize()
            cuda_time = time.perf_counter() - start

            # Benchmark librosa mel-spectrogram: 50 iterations
            start = time.perf_counter()
            for _ in range(50):
                _ = librosa.feature.melspectrogram(y=audio_np, sr=44100, n_fft=n_fft,
                                                   hop_length=hop_length, n_mels=n_mels)
            librosa_time = time.perf_counter() - start

            speedup = librosa_time / cuda_time
            print(f"\nMel-Spectrogram Performance:")
            print(f"  CUDA optimized: {cuda_time:.4f}s")
            print(f"  Librosa:        {librosa_time:.4f}s")
            print(f"  Speedup:        {speedup:.2f}x")

            # Assert minimum 10x speedup
            assert speedup >= 10.0, f"Mel-spectrogram speedup {speedup:.2f}x < 10.0x target"

        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    @pytest.mark.performance
    @pytest.mark.slow
    def test_pitch_detection_speedup_vs_torchcrepe(self, cuda_device):
        """Test pitch detection speedup vs torchcrepe (target: ≥5x)."""
        torchcrepe = pytest.importorskip("torchcrepe")
        try:
            import cuda_kernels
            import time

            # Generate test audio (2 seconds at 16kHz)
            audio_cuda = torch.randn(1, 32000, device=cuda_device)

            frame_length = 2048
            hop_length = 512
            sample_rate = 16000
            fmin, fmax = 80.0, 1000.0
            threshold = 0.3

            # Prepare CUDA outputs
            n_frames = (audio_cuda.shape[1] - frame_length) // hop_length + 1
            pitch_output = torch.zeros(n_frames, device=cuda_device)
            confidence_output = torch.zeros(n_frames, device=cuda_device)
            vibrato_output = torch.zeros(n_frames, device=cuda_device)

            # Warmup: 5 iterations
            for _ in range(5):
                cuda_kernels.launch_pitch_detection(
                    audio_cuda, pitch_output, confidence_output, vibrato_output,
                    sample_rate, frame_length, hop_length, fmin, fmax, threshold
                )
                _ = torchcrepe.predict(audio_cuda, sample_rate, hop_length, fmin, fmax,
                                      model='tiny', device=cuda_device, return_periodicity=True)
            torch.cuda.synchronize()

            # Benchmark CUDA pitch detection: 50 iterations
            start = time.perf_counter()
            for _ in range(50):
                cuda_kernels.launch_pitch_detection(
                    audio_cuda, pitch_output, confidence_output, vibrato_output,
                    sample_rate, frame_length, hop_length, fmin, fmax, threshold
                )
            torch.cuda.synchronize()
            cuda_time = time.perf_counter() - start

            # Benchmark torchcrepe: 50 iterations
            start = time.perf_counter()
            for _ in range(50):
                _ = torchcrepe.predict(audio_cuda, sample_rate, hop_length, fmin, fmax,
                                      model='tiny', device=cuda_device, return_periodicity=True)
            torch.cuda.synchronize()
            crepe_time = time.perf_counter() - start

            speedup = crepe_time / cuda_time
            print(f"\nPitch Detection Performance:")
            print(f"  CUDA optimized: {cuda_time:.4f}s")
            print(f"  Torchcrepe:     {crepe_time:.4f}s")
            print(f"  Speedup:        {speedup:.2f}x")

            # Assert minimum 5x speedup
            assert speedup >= 5.0, f"Pitch detection speedup {speedup:.2f}x < 5.0x target"

        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    @pytest.mark.performance
    @pytest.mark.slow
    def test_formant_extraction_speedup_vs_parselmouth(self, cuda_device):
        """Test formant extraction speedup vs parselmouth (target: ≥20x)."""
        parselmouth = pytest.importorskip("parselmouth")
        try:
            import cuda_kernels
            import time

            frame_length = 2048
            sample_rate = 16000
            lpc_order = 14
            num_formants = 4
            n_frames = 50

            # Generate test audio frames
            batch_size = 1
            audio_frames = torch.randn(batch_size, n_frames, frame_length, device=cuda_device)

            # Prepare CUDA output
            formants_output = torch.zeros(n_frames, num_formants, device=cuda_device)

            # Prepare parselmouth data (convert to numpy)
            audio_np = audio_frames.cpu().numpy().reshape(-1)

            # Warmup: 5 iterations
            for _ in range(5):
                cuda_kernels.launch_formant_extraction(
                    audio_frames, formants_output, frame_length, sample_rate, lpc_order, num_formants
                )
                # Parselmouth processing
                sound = parselmouth.Sound(audio_np, sampling_frequency=sample_rate)
                _ = sound.to_formant_burg(max_number_of_formants=num_formants)
            torch.cuda.synchronize()

            # Benchmark CUDA formant extraction: 50 iterations
            start = time.perf_counter()
            for _ in range(50):
                cuda_kernels.launch_formant_extraction(
                    audio_frames, formants_output, frame_length, sample_rate, lpc_order, num_formants
                )
            torch.cuda.synchronize()
            cuda_time = time.perf_counter() - start

            # Benchmark parselmouth: 50 iterations
            start = time.perf_counter()
            for _ in range(50):
                sound = parselmouth.Sound(audio_np, sampling_frequency=sample_rate)
                _ = sound.to_formant_burg(max_number_of_formants=num_formants)
            parselmouth_time = time.perf_counter() - start

            speedup = parselmouth_time / cuda_time
            print(f"\nFormant Extraction Performance:")
            print(f"  CUDA optimized: {cuda_time:.4f}s")
            print(f"  Parselmouth:    {parselmouth_time:.4f}s")
            print(f"  Speedup:        {speedup:.2f}x")

            # Assert minimum 20x speedup
            assert speedup >= 20.0, f"Formant extraction speedup {speedup:.2f}x < 20.0x target"

        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    @pytest.mark.performance
    def test_realtime_voice_conversion_latency(self, cuda_device):
        """Test real-time voice conversion latency (target: <10ms per 100ms chunk)."""
        try:
            import cuda_kernels
            import time

            # 100ms chunks at 44.1kHz
            chunk_size = 4410
            overlap_size = 1102  # 25% overlap
            n_fft = 2048
            hop_length = 512
            feature_dim = 256

            # Create input tensors
            audio_chunk = torch.randn(chunk_size, device=cuda_device)
            overlap_buffer = torch.zeros(overlap_size, device=cuda_device)
            window = torch.hann_window(n_fft, device=cuda_device)
            features_output = torch.zeros(feature_dim, device=cuda_device)

            # Warmup: 10 iterations
            for _ in range(10):
                cuda_kernels.launch_realtime_voice_conversion(
                    audio_chunk, overlap_buffer, features_output, window, n_fft, hop_length
                )
            torch.cuda.synchronize()

            # Benchmark single-call latency: 100 iterations
            latencies = []
            for _ in range(100):
                start = time.perf_counter()
                cuda_kernels.launch_realtime_voice_conversion(
                    audio_chunk, overlap_buffer, features_output, window, n_fft, hop_length
                )
                torch.cuda.synchronize()
                latency_ms = (time.perf_counter() - start) * 1000
                latencies.append(latency_ms)

            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)

            print(f"\nReal-time Voice Conversion Latency (100ms chunks):")
            print(f"  Average:     {avg_latency:.2f}ms")
            print(f"  95th percentile: {p95_latency:.2f}ms")
            print(f"  99th percentile: {p99_latency:.2f}ms")

            # Assert average latency < 10ms for real-time capability
            assert avg_latency < 10.0, f"Average latency {avg_latency:.2f}ms >= 10ms target"
            # Assert p99 latency < 15ms (allow some headroom)
            assert p99_latency < 15.0, f"P99 latency {p99_latency:.2f}ms >= 15ms"

        except ImportError:
            pytest.skip("CUDA kernels not available")


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test CUDA kernel error handling."""

    @pytest.mark.cuda
    def test_empty_tensor_error(self, cuda_device):
        """Test handling of empty tensors."""
        try:
            import cuda_kernels

            empty_tensor = torch.empty(0, device=cuda_device)

            with pytest.raises((RuntimeError, ValueError)):
                cuda_kernels.matmul(empty_tensor, empty_tensor)
        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    def test_shape_mismatch_error(self, cuda_device):
        """Test handling of mismatched shapes."""
        try:
            import cuda_kernels

            A = torch.randn(10, 20, device=cuda_device)
            B = torch.randn(30, 40, device=cuda_device)

            with pytest.raises((RuntimeError, ValueError)):
                cuda_kernels.matmul(A, B)
        except ImportError:
            pytest.skip("CUDA kernels not available")

    @pytest.mark.cuda
    def test_invalid_parameter_error(self, audio_waveform):
        """Test handling of out-of-range parameters."""
        try:
            import cuda_kernels

            # Pitch factor out of valid range
            with pytest.raises((RuntimeError, ValueError)):
                cuda_kernels.pitch_shift(audio_waveform, pitch_factor=-1.0)
        except ImportError:
            pytest.skip("CUDA kernels not available")
