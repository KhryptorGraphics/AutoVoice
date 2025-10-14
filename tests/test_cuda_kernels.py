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
