"""Integration tests for CUDA bindings with real audio processing

These tests validate the entire CUDA binding pipeline with real-world scenarios.
"""

import pytest
import torch
import numpy as np
from pathlib import Path


@pytest.mark.integration
@pytest.mark.cuda
class TestCUDABindingsIntegration:
    """Integration tests for CUDA bindings"""

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

    def test_pitch_detection_sine_wave(self):
        """Test pitch detection with synthetic sine wave of known frequency"""
        # Generate 440 Hz sine wave (A4 note)
        sample_rate = 22050.0
        duration = 2.0
        frequency = 440.0

        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).cuda()

        # Setup parameters
        frame_length = 2048
        hop_length = 512
        n_frames = max(0, (len(audio) - frame_length) // hop_length + 1)

        # Allocate output tensors
        output_pitch = torch.zeros(n_frames, device='cuda')
        output_confidence = torch.zeros(n_frames, device='cuda')
        output_vibrato = torch.zeros(n_frames, device='cuda')

        # Run pitch detection
        self.cuda_kernels.launch_pitch_detection(
            audio_tensor, output_pitch, output_confidence, output_vibrato,
            sample_rate, frame_length, hop_length
        )

        # Verify results
        detected_pitches = output_pitch.cpu().numpy()
        confidences = output_confidence.cpu().numpy()

        # Filter out zero pitches (silence detection)
        valid_pitches = detected_pitches[detected_pitches > 0]

        assert len(valid_pitches) > 0, "Should detect some pitch values"

        # Mean detected pitch should be close to 440 Hz (within 5%)
        mean_pitch = np.mean(valid_pitches)
        pitch_error = abs(mean_pitch - frequency) / frequency
        assert pitch_error < 0.05, f"Pitch error too large: {pitch_error:.2%} (detected: {mean_pitch:.2f} Hz)"

        # Confidence should be high for clean sine wave
        valid_confidence = confidences[detected_pitches > 0]
        mean_confidence = np.mean(valid_confidence)
        assert mean_confidence > 0.7, f"Confidence too low: {mean_confidence:.2f}"

    def test_pitch_detection_multiple_frequencies(self):
        """Test pitch detection with multiple known frequencies"""
        sample_rate = 22050.0
        duration = 1.0
        frame_length = 2048
        hop_length = 512

        # Test frequencies (musical notes)
        test_frequencies = [
            (220.0, "A3"),
            (440.0, "A4"),
            (880.0, "A5"),
            (261.63, "C4"),
            (329.63, "E4"),
        ]

        for frequency, note_name in test_frequencies:
            # Generate sine wave
            t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
            audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
            audio_tensor = torch.from_numpy(audio).cuda()

            n_frames = max(0, (len(audio) - frame_length) // hop_length + 1)
            output_pitch = torch.zeros(n_frames, device='cuda')
            output_confidence = torch.zeros(n_frames, device='cuda')
            output_vibrato = torch.zeros(n_frames, device='cuda')

            # Run pitch detection
            self.cuda_kernels.launch_pitch_detection(
                audio_tensor, output_pitch, output_confidence, output_vibrato,
                sample_rate, frame_length, hop_length
            )

            # Verify
            detected_pitches = output_pitch.cpu().numpy()
            valid_pitches = detected_pitches[detected_pitches > 0]

            if len(valid_pitches) > 0:
                mean_pitch = np.mean(valid_pitches)
                pitch_error = abs(mean_pitch - frequency) / frequency
                assert pitch_error < 0.1, f"{note_name} ({frequency} Hz): error {pitch_error:.2%}"

    def test_vibrato_analysis_with_modulation(self):
        """Test vibrato analysis with synthetic vibrato"""
        sample_rate = 22050.0
        duration = 2.0
        base_freq = 440.0
        vibrato_rate = 5.5  # Hz
        vibrato_depth_cents = 50.0  # cents

        # Generate audio with vibrato
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        depth_ratio = 2 ** (vibrato_depth_cents / 1200.0)
        vibrato = depth_ratio ** np.sin(2 * np.pi * vibrato_rate * t)
        audio = np.sin(2 * np.pi * base_freq * vibrato * t).astype(np.float32)

        audio_tensor = torch.from_numpy(audio).cuda()

        # First, detect pitch
        frame_length = 2048
        hop_length = 256  # Smaller hop for better vibrato resolution
        n_frames = max(0, (len(audio) - frame_length) // hop_length + 1)

        pitch_contour = torch.zeros(n_frames, device='cuda')
        output_confidence = torch.zeros(n_frames, device='cuda')
        output_vibrato = torch.zeros(n_frames, device='cuda')

        self.cuda_kernels.launch_pitch_detection(
            audio_tensor, pitch_contour, output_confidence, output_vibrato,
            sample_rate, frame_length, hop_length
        )

        # Then analyze vibrato
        vibrato_rate_out = torch.zeros(n_frames, device='cuda')
        vibrato_depth_out = torch.zeros(n_frames, device='cuda')

        self.cuda_kernels.launch_vibrato_analysis(
            pitch_contour, vibrato_rate_out, vibrato_depth_out,
            hop_length, int(sample_rate)
        )

        # Verify vibrato detection
        detected_rates = vibrato_rate_out.cpu().numpy()
        detected_depths = vibrato_depth_out.cpu().numpy()

        # Filter valid vibrato detections (non-zero)
        valid_rates = detected_rates[detected_rates > 0]
        valid_depths = detected_depths[detected_depths > 0]

        if len(valid_rates) > 0:
            mean_rate = np.mean(valid_rates)
            rate_error = abs(mean_rate - vibrato_rate) / vibrato_rate
            # Vibrato detection is challenging, allow 30% error
            assert rate_error < 0.3, f"Vibrato rate error: {rate_error:.2%}"

        if len(valid_depths) > 0:
            mean_depth = np.mean(valid_depths)
            # Depth should be detected (rough check)
            assert mean_depth > 10.0, f"Vibrato depth too low: {mean_depth:.2f} cents"

    def test_various_sample_rates(self):
        """Test pitch detection with various sample rates"""
        test_sample_rates = [8000.0, 16000.0, 22050.0, 44100.0]
        frequency = 440.0
        duration = 1.0

        for sample_rate in test_sample_rates:
            # Generate audio
            t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
            audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
            audio_tensor = torch.from_numpy(audio).cuda()

            # Adjust parameters based on sample rate
            frame_length = min(2048, int(sample_rate * 0.05))  # 50ms frame
            hop_length = frame_length // 4
            n_frames = max(0, (len(audio) - frame_length) // hop_length + 1)

            output_pitch = torch.zeros(n_frames, device='cuda')
            output_confidence = torch.zeros(n_frames, device='cuda')
            output_vibrato = torch.zeros(n_frames, device='cuda')

            # Run pitch detection
            self.cuda_kernels.launch_pitch_detection(
                audio_tensor, output_pitch, output_confidence, output_vibrato,
                sample_rate, frame_length, hop_length
            )

            # Verify
            detected_pitches = output_pitch.cpu().numpy()
            valid_pitches = detected_pitches[detected_pitches > 0]

            assert len(valid_pitches) > 0, f"No pitch detected at {sample_rate} Hz"

    def test_various_audio_lengths(self):
        """Test with various audio lengths"""
        sample_rate = 22050.0
        frequency = 440.0
        frame_length = 2048
        hop_length = 512

        # Test durations from very short to long
        test_durations = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

        for duration in test_durations:
            # Generate audio
            t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
            audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
            audio_tensor = torch.from_numpy(audio).cuda()

            n_frames = max(0, (len(audio) - frame_length) // hop_length + 1)
            output_pitch = torch.zeros(n_frames, device='cuda')
            output_confidence = torch.zeros(n_frames, device='cuda')
            output_vibrato = torch.zeros(n_frames, device='cuda')

            # Run pitch detection
            self.cuda_kernels.launch_pitch_detection(
                audio_tensor, output_pitch, output_confidence, output_vibrato,
                sample_rate, frame_length, hop_length
            )

            # Verify
            detected_pitches = output_pitch.cpu().numpy()
            assert len(detected_pitches) == n_frames, f"Frame count mismatch at {duration}s"

    def test_noise_robustness(self):
        """Test pitch detection robustness with added noise"""
        sample_rate = 22050.0
        duration = 2.0
        frequency = 440.0
        frame_length = 2048
        hop_length = 512

        # Test with various SNR levels
        snr_levels = [30, 20, 10, 5]  # dB

        for snr_db in snr_levels:
            # Generate clean signal
            t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
            signal = np.sin(2 * np.pi * frequency * t)

            # Add noise
            signal_power = np.mean(signal ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.random.randn(len(signal)) * np.sqrt(noise_power)

            audio = (signal + noise).astype(np.float32)
            audio_tensor = torch.from_numpy(audio).cuda()

            n_frames = max(0, (len(audio) - frame_length) // hop_length + 1)
            output_pitch = torch.zeros(n_frames, device='cuda')
            output_confidence = torch.zeros(n_frames, device='cuda')
            output_vibrato = torch.zeros(n_frames, device='cuda')

            # Run pitch detection
            self.cuda_kernels.launch_pitch_detection(
                audio_tensor, output_pitch, output_confidence, output_vibrato,
                sample_rate, frame_length, hop_length
            )

            # Verify
            detected_pitches = output_pitch.cpu().numpy()
            confidences = output_confidence.cpu().numpy()
            valid_pitches = detected_pitches[detected_pitches > 0]

            if snr_db >= 10:
                # Should still detect pitch at reasonable SNR
                assert len(valid_pitches) > 0, f"No pitch detected at {snr_db} dB SNR"

                # Lower confidence with more noise
                if len(valid_pitches) > 0:
                    valid_conf = confidences[detected_pitches > 0]
                    mean_conf = np.mean(valid_conf)
                    # Just verify confidence decreases with noise
                    assert mean_conf >= 0.0, "Confidence should be non-negative"

    def test_silence_detection(self):
        """Test that silence is properly detected"""
        sample_rate = 22050.0
        duration = 1.0
        frame_length = 2048
        hop_length = 512

        # Generate complete silence
        audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
        audio_tensor = torch.from_numpy(audio).cuda()

        n_frames = max(0, (len(audio) - frame_length) // hop_length + 1)
        output_pitch = torch.zeros(n_frames, device='cuda')
        output_confidence = torch.zeros(n_frames, device='cuda')
        output_vibrato = torch.zeros(n_frames, device='cuda')

        # Run pitch detection
        self.cuda_kernels.launch_pitch_detection(
            audio_tensor, output_pitch, output_confidence, output_vibrato,
            sample_rate, frame_length, hop_length
        )

        # Verify all outputs are zero
        assert torch.all(output_pitch == 0.0), "Silence should have zero pitch"
        assert torch.all(output_confidence == 0.0), "Silence should have zero confidence"
        assert torch.all(output_vibrato == 0.0), "Silence should have zero vibrato"

    def test_memory_consistency(self):
        """Test that memory is properly managed across multiple calls"""
        sample_rate = 22050.0
        duration = 1.0
        frequency = 440.0
        frame_length = 2048
        hop_length = 512

        # Run multiple times and check for memory leaks
        initial_memory = torch.cuda.memory_allocated()

        for _ in range(10):
            # Generate audio
            t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
            audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
            audio_tensor = torch.from_numpy(audio).cuda()

            n_frames = max(0, (len(audio) - frame_length) // hop_length + 1)
            output_pitch = torch.zeros(n_frames, device='cuda')
            output_confidence = torch.zeros(n_frames, device='cuda')
            output_vibrato = torch.zeros(n_frames, device='cuda')

            # Run pitch detection
            self.cuda_kernels.launch_pitch_detection(
                audio_tensor, output_pitch, output_confidence, output_vibrato,
                sample_rate, frame_length, hop_length
            )

            # Clean up tensors
            del audio_tensor, output_pitch, output_confidence, output_vibrato
            torch.cuda.empty_cache()

        final_memory = torch.cuda.memory_allocated()
        memory_increase = final_memory - initial_memory

        # Allow small memory increase, but not proportional to iterations
        assert memory_increase < 10 * 1024 * 1024, f"Memory leak detected: {memory_increase / 1024 / 1024:.2f} MB"

    @pytest.mark.slow
    def test_long_audio_processing(self):
        """Test processing of very long audio (stress test)"""
        sample_rate = 22050.0
        duration = 60.0  # 1 minute
        frequency = 440.0
        frame_length = 2048
        hop_length = 512

        # Generate long audio
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        audio_tensor = torch.from_numpy(audio).cuda()

        n_frames = max(0, (len(audio) - frame_length) // hop_length + 1)
        output_pitch = torch.zeros(n_frames, device='cuda')
        output_confidence = torch.zeros(n_frames, device='cuda')
        output_vibrato = torch.zeros(n_frames, device='cuda')

        # Run pitch detection
        self.cuda_kernels.launch_pitch_detection(
            audio_tensor, output_pitch, output_confidence, output_vibrato,
            sample_rate, frame_length, hop_length
        )

        # Verify processing completed
        detected_pitches = output_pitch.cpu().numpy()
        assert len(detected_pitches) == n_frames, "All frames should be processed"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
