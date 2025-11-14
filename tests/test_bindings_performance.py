"""Performance benchmarking tests for CUDA bindings

These tests measure and compare CUDA vs CPU performance for pitch detection.
"""

import pytest
import torch
import numpy as np
import time
from typing import Dict, List, Tuple


@pytest.mark.performance
@pytest.mark.cuda
class TestCUDABindingsPerformance:
    """Performance tests for CUDA bindings"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        try:
            import cuda_kernels
            self.cuda_kernels = cuda_kernels
        except ImportError:
            try:
                from auto_voice import cuda_kernels
                self.cuda_kernels = cuda_kernels
            except ImportError:
                pytest.skip("cuda_kernels module not available")

        # Default parameters for pitch detection
        self.f0_min = 50.0
        self.f0_max = 1000.0
        self.confidence_threshold = 0.7

        # Warm up GPU
        dummy = torch.randn(1000, device='cuda')
        _ = dummy * 2
        torch.cuda.synchronize()

    def benchmark_cuda_pitch_detection(
        self, audio_length: int, sample_rate: float, iterations: int = 10
    ) -> Dict[str, float]:
        """Benchmark CUDA pitch detection"""
        # Generate test audio
        t = np.linspace(0, audio_length / sample_rate, audio_length, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        audio_tensor = torch.from_numpy(audio).cuda()

        frame_length = 2048
        hop_length = 512
        n_frames = max(0, (audio_length - frame_length) // hop_length + 1)

        output_pitch = torch.zeros(n_frames, device='cuda')
        output_confidence = torch.zeros(n_frames, device='cuda')
        output_vibrato = torch.zeros(n_frames, device='cuda')

        # Warm up
        self.cuda_kernels.launch_pitch_detection(
            audio_tensor, output_pitch, output_confidence, output_vibrato,
            sample_rate, frame_length, hop_length,
            self.f0_min, self.f0_max, self.confidence_threshold
        )
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()

            self.cuda_kernels.launch_pitch_detection(
                audio_tensor, output_pitch, output_confidence, output_vibrato,
                sample_rate, frame_length, hop_length,
                self.f0_min, self.f0_max, self.confidence_threshold
            )

            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'throughput': audio_length / sample_rate / np.mean(times),  # Real-time factor
        }

    def benchmark_cpu_reference(
        self, audio_length: int, sample_rate: float, iterations: int = 10
    ) -> Dict[str, float]:
        """Benchmark CPU reference implementation (if available)"""
        try:
            import librosa
        except ImportError:
            pytest.skip("librosa not available for CPU reference")

        # Generate test audio
        t = np.linspace(0, audio_length / sample_rate, audio_length, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        hop_length = 512

        # Warm up
        _ = librosa.yin(audio, fmin=80, fmax=800, sr=sample_rate, hop_length=hop_length)

        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = librosa.yin(audio, fmin=80, fmax=800, sr=sample_rate, hop_length=hop_length)
            end = time.perf_counter()
            times.append(end - start)

        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'throughput': audio_length / sample_rate / np.mean(times),  # Real-time factor
        }

    def test_performance_short_audio(self):
        """Benchmark short audio (1 second)"""
        sample_rate = 22050.0
        audio_length = int(sample_rate * 1.0)

        cuda_results = self.benchmark_cuda_pitch_detection(audio_length, sample_rate)

        print(f"\n=== Short Audio (1s) ===")
        print(f"CUDA Mean Time: {cuda_results['mean_time'] * 1000:.2f} ms")
        print(f"CUDA Throughput: {cuda_results['throughput']:.2f}x real-time")

        # Should be faster than real-time
        assert cuda_results['throughput'] > 1.0, "Should process faster than real-time"

    def test_performance_medium_audio(self):
        """Benchmark medium audio (10 seconds)"""
        sample_rate = 22050.0
        audio_length = int(sample_rate * 10.0)

        cuda_results = self.benchmark_cuda_pitch_detection(audio_length, sample_rate)

        print(f"\n=== Medium Audio (10s) ===")
        print(f"CUDA Mean Time: {cuda_results['mean_time'] * 1000:.2f} ms")
        print(f"CUDA Throughput: {cuda_results['throughput']:.2f}x real-time")

        # Should be faster than real-time
        assert cuda_results['throughput'] > 1.0, "Should process faster than real-time"

    @pytest.mark.slow
    def test_performance_long_audio(self):
        """Benchmark long audio (60 seconds)"""
        sample_rate = 22050.0
        audio_length = int(sample_rate * 60.0)

        cuda_results = self.benchmark_cuda_pitch_detection(audio_length, sample_rate, iterations=5)

        print(f"\n=== Long Audio (60s) ===")
        print(f"CUDA Mean Time: {cuda_results['mean_time']:.2f} s")
        print(f"CUDA Throughput: {cuda_results['throughput']:.2f}x real-time")

        # Should be faster than real-time even for long audio
        assert cuda_results['throughput'] > 1.0, "Should process faster than real-time"

    def test_performance_cuda_vs_cpu(self):
        """Compare CUDA vs CPU performance"""
        sample_rate = 22050.0
        audio_length = int(sample_rate * 5.0)

        cuda_results = self.benchmark_cuda_pitch_detection(audio_length, sample_rate)

        try:
            cpu_results = self.benchmark_cpu_reference(audio_length, sample_rate)

            speedup = cpu_results['mean_time'] / cuda_results['mean_time']

            print(f"\n=== CUDA vs CPU (5s audio) ===")
            print(f"CUDA Time: {cuda_results['mean_time'] * 1000:.2f} ms")
            print(f"CPU Time: {cpu_results['mean_time'] * 1000:.2f} ms")
            print(f"Speedup: {speedup:.2f}x")
            print(f"CUDA Throughput: {cuda_results['throughput']:.2f}x real-time")
            print(f"CPU Throughput: {cpu_results['throughput']:.2f}x real-time")

            # CUDA should be faster than CPU
            assert cuda_results['mean_time'] < cpu_results['mean_time'], "CUDA should be faster"

            # Expect at least 2x speedup
            assert speedup >= 2.0, f"Expected at least 2x speedup, got {speedup:.2f}x"
        except pytest.skip.Exception:
            print("\n=== CUDA Performance (5s audio, no CPU reference) ===")
            print(f"CUDA Time: {cuda_results['mean_time'] * 1000:.2f} ms")
            print(f"CUDA Throughput: {cuda_results['throughput']:.2f}x real-time")

    def test_performance_various_batch_sizes(self):
        """Test performance with various audio lengths"""
        sample_rate = 22050.0
        test_durations = [0.5, 1.0, 2.0, 5.0, 10.0]

        print("\n=== Performance vs Audio Length ===")
        print(f"{'Duration (s)':<15} {'Time (ms)':<15} {'Throughput (x)':<15}")
        print("-" * 45)

        for duration in test_durations:
            audio_length = int(sample_rate * duration)
            results = self.benchmark_cuda_pitch_detection(audio_length, sample_rate, iterations=5)

            print(f"{duration:<15.1f} {results['mean_time'] * 1000:<15.2f} {results['throughput']:<15.2f}")

            # All should be faster than real-time
            assert results['throughput'] > 1.0, f"{duration}s audio should process faster than real-time"

    def test_performance_vibrato_analysis(self):
        """Benchmark vibrato analysis performance"""
        sample_rate = 22050.0
        duration = 5.0
        audio_length = int(sample_rate * duration)

        # Generate audio and pitch contour first
        t = np.linspace(0, duration, audio_length, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        audio_tensor = torch.from_numpy(audio).cuda()

        frame_length = 2048
        hop_length = 256
        n_frames = max(0, (audio_length - frame_length) // hop_length + 1)

        pitch_contour = torch.zeros(n_frames, device='cuda')
        output_confidence = torch.zeros(n_frames, device='cuda')
        output_vibrato = torch.zeros(n_frames, device='cuda')

        # Get pitch contour
        self.cuda_kernels.launch_pitch_detection(
            audio_tensor, pitch_contour, output_confidence, output_vibrato,
            sample_rate, frame_length, hop_length,
            self.f0_min, self.f0_max, self.confidence_threshold
        )
        torch.cuda.synchronize()

        # Benchmark vibrato analysis
        vibrato_rate = torch.zeros(n_frames, device='cuda')
        vibrato_depth = torch.zeros(n_frames, device='cuda')

        # Warm up
        self.cuda_kernels.launch_vibrato_analysis(
            pitch_contour, vibrato_rate, vibrato_depth,
            hop_length, int(sample_rate)
        )
        torch.cuda.synchronize()

        # Benchmark
        iterations = 20
        times = []
        for _ in range(iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()

            self.cuda_kernels.launch_vibrato_analysis(
                pitch_contour, vibrato_rate, vibrato_depth,
                hop_length, int(sample_rate)
            )

            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

        mean_time = np.mean(times)
        throughput = duration / mean_time

        print(f"\n=== Vibrato Analysis Performance ({duration}s audio) ===")
        print(f"Mean Time: {mean_time * 1000:.2f} ms")
        print(f"Throughput: {throughput:.2f}x real-time")

        # Should be much faster than real-time
        assert throughput > 10.0, "Vibrato analysis should be very fast"

    def test_memory_usage_scaling(self):
        """Test memory usage with increasing audio length"""
        sample_rate = 22050.0
        test_durations = [1.0, 5.0, 10.0, 30.0]

        print("\n=== Memory Usage vs Audio Length ===")
        print(f"{'Duration (s)':<15} {'Memory (MB)':<15}")
        print("-" * 30)

        for duration in test_durations:
            # Clear cache
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            audio_length = int(sample_rate * duration)
            t = np.linspace(0, duration, audio_length, dtype=np.float32)
            audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

            initial_memory = torch.cuda.memory_allocated()

            audio_tensor = torch.from_numpy(audio).cuda()

            frame_length = 2048
            hop_length = 512
            n_frames = max(0, (audio_length - frame_length) // hop_length + 1)

            output_pitch = torch.zeros(n_frames, device='cuda')
            output_confidence = torch.zeros(n_frames, device='cuda')
            self.cuda_kernels.launch_pitch_detection(
                audio_tensor, output_pitch, output_confidence, output_vibrato,
                sample_rate, frame_length, hop_length,
                self.f0_min, self.f0_max, self.confidence_threshold
            )
            torch.cuda.synchronize()

            peak_memory = torch.cuda.max_memory_allocated()
            memory_used = (peak_memory - initial_memory) / (1024 * 1024)

            print(f"{duration:<15.1f} {memory_used:<15.2f}")

            # Memory should scale roughly linearly with audio length
            expected_memory = duration * 0.5  # Rough estimate: 0.5 MB per second
            assert memory_used < expected_memory * 3, f"Memory usage too high for {duration}s"

    def test_latency_measurement(self):
        """Measure kernel launch latency"""
        sample_rate = 22050.0
        audio_length = int(sample_rate * 0.1)  # Very short audio

        # Generate test audio
        t = np.linspace(0, 0.1, audio_length, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        audio_tensor = torch.from_numpy(audio).cuda()

        frame_length = 2048
        hop_length = 512
        n_frames = max(0, (audio_length - frame_length) // hop_length + 1)

        output_pitch = torch.zeros(n_frames, device='cuda')
        output_confidence = torch.zeros(n_frames, device='cuda')
        output_vibrato = torch.zeros(n_frames, device='cuda')

        # Warm up
        for _ in range(10):
            self.cuda_kernels.launch_pitch_detection(
                audio_tensor, output_pitch, output_confidence, output_vibrato,
                sample_rate, frame_length, hop_length,
                self.f0_min, self.f0_max, self.confidence_threshold
            )
        torch.cuda.synchronize()

        # Measure latency
        iterations = 100
        times = []
        for _ in range(iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()

            self.cuda_kernels.launch_pitch_detection(
                audio_tensor, output_pitch, output_confidence, output_vibrato,
                sample_rate, frame_length, hop_length,
                self.f0_min, self.f0_max, self.confidence_threshold
            )

            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

        mean_latency = np.mean(times) * 1000  # Convert to ms
        std_latency = np.std(times) * 1000

        print(f"\n=== Kernel Launch Latency ===")
        print(f"Mean Latency: {mean_latency:.2f} ms")
        print(f"Std Latency: {std_latency:.2f} ms")
        print(f"Min Latency: {np.min(times) * 1000:.2f} ms")
        print(f"Max Latency: {np.max(times) * 1000:.2f} ms")

        # Latency should be reasonable for real-time use
        assert mean_latency < 50.0, f"Mean latency too high: {mean_latency:.2f} ms"

    def test_throughput_sustained(self):
        """Test sustained throughput with continuous processing"""
        sample_rate = 22050.0
        duration = 2.0
        audio_length = int(sample_rate * duration)

        # Generate test audio
        t = np.linspace(0, duration, audio_length, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        audio_tensor = torch.from_numpy(audio).cuda()

        frame_length = 2048
        hop_length = 512
        n_frames = max(0, (audio_length - frame_length) // hop_length + 1)

        output_pitch = torch.zeros(n_frames, device='cuda')
        output_confidence = torch.zeros(n_frames, device='cuda')
        output_vibrato = torch.zeros(n_frames, device='cuda')

        # Process continuously for 5 seconds
        iterations = 50
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        for _ in range(iterations):
            self.cuda_kernels.launch_pitch_detection(
                audio_tensor, output_pitch, output_confidence, output_vibrato,
                sample_rate, frame_length, hop_length,
                self.f0_min, self.f0_max, self.confidence_threshold
            )

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        total_time = end_time - start_time
        total_audio_processed = duration * iterations
        sustained_throughput = total_audio_processed / total_time

        print(f"\n=== Sustained Throughput ===")
        print(f"Total Audio Processed: {total_audio_processed:.1f} s")
        print(f"Total Time: {total_time:.2f} s")
        print(f"Sustained Throughput: {sustained_throughput:.2f}x real-time")

        # Should maintain real-time processing under sustained load
        assert sustained_throughput > 1.0, "Should maintain real-time processing"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
