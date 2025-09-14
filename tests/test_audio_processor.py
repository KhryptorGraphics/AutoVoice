"""
Tests for audio processing components
"""
import pytest
import torch
import numpy as np
from src.auto_voice.audio.processor import AudioProcessor


class TestAudioProcessor:
    """Test cases for AudioProcessor"""

    def setup_method(self):
        """Set up test fixtures"""
        self.processor = AudioProcessor(device='cpu')  # Use CPU for testing
        self.sample_rate = 22050
        self.audio_length = 22050  # 1 second of audio
        self.test_audio = torch.randn(self.audio_length)

    def test_pitch_extraction(self):
        """Test pitch extraction"""
        pitch = self.processor.extract_pitch(self.test_audio, self.sample_rate)
        assert pitch is not None
        assert pitch.shape[0] > 0

    def test_voice_activity_detection(self):
        """Test voice activity detection"""
        vad = self.processor.voice_activity_detection(self.test_audio)
        assert vad is not None
        assert vad.shape[0] > 0
        assert torch.all((vad >= 0) & (vad <= 1))  # Should be binary values

    def test_spectrogram_computation(self):
        """Test spectrogram computation"""
        spec = self.processor.compute_spectrogram(self.test_audio)
        assert spec is not None
        assert len(spec.shape) == 2  # Should be 2D (frames, freq_bins)
        assert spec.shape[1] == 513  # n_fft//2 + 1 for 1024 FFT

    def test_empty_audio(self):
        """Test handling of empty audio"""
        empty_audio = torch.zeros(0)
        pitch = self.processor.extract_pitch(empty_audio, self.sample_rate)
        assert pitch.shape[0] == 0  # Should handle empty input gracefully