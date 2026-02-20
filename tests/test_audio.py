"""Tests for audio processing module."""
import numpy as np
import pytest

from auto_voice.audio.processor import AudioProcessor
from auto_voice.audio.effects import pitch_shift, volume_adjust, fade_in, fade_out


class TestAudioProcessor:
    """AudioProcessor tests."""

    @pytest.mark.smoke
    def test_init(self, audio_processor):
        assert audio_processor.sample_rate == 22050

    def test_normalize(self, audio_processor):
        audio = np.array([0.5, -0.3, 0.8, -0.1], dtype=np.float32)
        normalized = audio_processor.normalize(audio, peak=0.95)
        assert abs(np.abs(normalized).max() - 0.95) < 0.001

    def test_normalize_silent(self, audio_processor):
        audio = np.zeros(100, dtype=np.float32)
        normalized = audio_processor.normalize(audio)
        np.testing.assert_array_equal(normalized, audio)

    def test_to_mono_already_mono(self, audio_processor):
        audio = np.random.randn(1000).astype(np.float32)
        mono = audio_processor.to_mono(audio)
        np.testing.assert_array_equal(mono, audio)

    def test_to_mono_stereo(self, audio_processor):
        stereo = np.random.randn(2, 1000).astype(np.float32)
        mono = audio_processor.to_mono(stereo)
        assert mono.ndim == 1
        assert len(mono) == 1000

    def test_save_and_load(self, audio_processor, tmp_path, sample_audio):
        audio, sr = sample_audio
        path = str(tmp_path / "test.wav")
        audio_processor.save(path, audio, sr)
        loaded, loaded_sr = audio_processor.load(path, sr=sr)
        assert loaded_sr == sr
        assert len(loaded) == len(audio)

    def test_resample_same_rate(self, audio_processor, sample_audio):
        audio, sr = sample_audio
        result = audio_processor.resample(audio, sr, sr)
        np.testing.assert_array_equal(result, audio)

    def test_resample_different_rate(self, audio_processor, sample_audio):
        audio, sr = sample_audio
        result = audio_processor.resample(audio, sr, 16000)
        expected_len = int(len(audio) * 16000 / sr)
        assert abs(len(result) - expected_len) < 10


class TestEffects:
    """Audio effects tests."""

    @pytest.mark.smoke
    def test_volume_adjust_unity(self):
        audio = np.array([0.5, -0.3, 0.8], dtype=np.float32)
        result = volume_adjust(audio, 1.0)
        np.testing.assert_array_almost_equal(result, audio)

    def test_volume_adjust_double(self):
        audio = np.array([0.3, -0.2], dtype=np.float32)
        result = volume_adjust(audio, 2.0)
        np.testing.assert_array_almost_equal(result, [0.6, -0.4])

    def test_volume_adjust_clips(self):
        audio = np.array([0.8, -0.9], dtype=np.float32)
        result = volume_adjust(audio, 2.0)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_pitch_shift_zero(self):
        audio = np.random.randn(22050).astype(np.float32)
        result = pitch_shift(audio, 22050, 0.0)
        np.testing.assert_array_equal(result, audio)

    @pytest.mark.slow
    def test_pitch_shift_up(self):
        sr = 22050
        t = np.linspace(0, 1, sr, endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        result = pitch_shift(audio, sr, 2.0)
        assert len(result) == len(audio)

    def test_fade_in(self):
        audio = np.ones(1000, dtype=np.float32)
        result = fade_in(audio, 100)
        assert result[0] == 0.0
        assert abs(result[99] - 1.0) < 0.02
        assert result[500] == 1.0

    def test_fade_out(self):
        audio = np.ones(1000, dtype=np.float32)
        result = fade_out(audio, 100)
        assert result[0] == 1.0
        assert abs(result[-1]) < 0.02
        assert result[500] == 1.0

    def test_fade_in_zero_duration(self):
        audio = np.ones(100, dtype=np.float32)
        result = fade_in(audio, 0)
        np.testing.assert_array_equal(result, audio)
