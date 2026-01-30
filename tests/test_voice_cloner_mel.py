"""Tests for mel-statistics speaker embedding."""
import tempfile
import os

import numpy as np
import pytest
import soundfile as sf

from auto_voice.inference.voice_cloner import VoiceCloner, InvalidAudioError


@pytest.fixture
def cloner():
    return VoiceCloner()


@pytest.fixture
def audio_file_440hz(tmp_path):
    """5-second 440Hz sine wave."""
    sr = 16000
    t = np.linspace(0, 5, sr * 5, endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    path = str(tmp_path / "tone_440.wav")
    sf.write(path, audio, sr)
    return path


@pytest.fixture
def audio_file_220hz(tmp_path):
    """5-second 220Hz sine wave (different 'speaker')."""
    sr = 16000
    t = np.linspace(0, 5, sr * 5, endpoint=False)
    audio = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    path = str(tmp_path / "tone_220.wav")
    sf.write(path, audio, sr)
    return path


@pytest.fixture
def silent_audio_file(tmp_path):
    """5-second silence."""
    sr = 16000
    audio = np.zeros(sr * 5, dtype=np.float32)
    path = str(tmp_path / "silence.wav")
    sf.write(path, audio, sr)
    return path


@pytest.fixture
def short_audio_file(tmp_path):
    """1-second audio (below min duration)."""
    sr = 16000
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float32)
    path = str(tmp_path / "short.wav")
    sf.write(path, audio, sr)
    return path


class TestMelEmbedding:
    """Mel-statistics embedding behavior."""

    @pytest.mark.smoke
    def test_embedding_dimension_256(self, cloner, audio_file_440hz):
        emb = cloner._extract_embedding(audio_file_440hz)
        assert emb.shape == (256,)

    def test_embedding_is_float32(self, cloner, audio_file_440hz):
        emb = cloner._extract_embedding(audio_file_440hz)
        assert emb.dtype == np.float32

    def test_embedding_deterministic(self, cloner, audio_file_440hz):
        emb1 = cloner._extract_embedding(audio_file_440hz)
        emb2 = cloner._extract_embedding(audio_file_440hz)
        np.testing.assert_array_equal(emb1, emb2)

    def test_embedding_normalized(self, cloner, audio_file_440hz):
        emb = cloner._extract_embedding(audio_file_440hz)
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-5

    def test_embedding_different_speakers(self, cloner, audio_file_440hz, audio_file_220hz):
        emb1 = cloner._extract_embedding(audio_file_440hz)
        emb2 = cloner._extract_embedding(audio_file_220hz)
        # Different frequency content -> different embedding
        similarity = np.dot(emb1, emb2)
        assert similarity < 0.99  # Not identical

    def test_silent_audio_raises(self, cloner, silent_audio_file):
        with pytest.raises(InvalidAudioError, match="Silent audio"):
            cloner._extract_embedding(silent_audio_file)

    def test_short_audio_raises(self, cloner, short_audio_file):
        with pytest.raises(InvalidAudioError, match="too short"):
            cloner._extract_embedding(short_audio_file)

    def test_nonexistent_file_raises(self, cloner):
        with pytest.raises(InvalidAudioError, match="Failed to load"):
            cloner._extract_embedding("/nonexistent/file.wav")


class TestMultiFileEmbedding:
    """Multi-file averaging tests."""

    def test_multi_file_averaging(self, cloner, audio_file_440hz, audio_file_220hz):
        avg = cloner.create_speaker_embedding([audio_file_440hz, audio_file_220hz])
        assert avg.shape == (256,)
        norm = np.linalg.norm(avg)
        assert abs(norm - 1.0) < 1e-5

    def test_single_file_same_as_extract(self, cloner, audio_file_440hz):
        single = cloner._extract_embedding(audio_file_440hz)
        multi = cloner.create_speaker_embedding([audio_file_440hz])
        np.testing.assert_allclose(single, multi, atol=1e-5)

    def test_empty_list_raises(self, cloner):
        with pytest.raises(InvalidAudioError, match="No audio files"):
            cloner.create_speaker_embedding([])

    def test_multi_file_deterministic(self, cloner, audio_file_440hz, audio_file_220hz):
        emb1 = cloner.create_speaker_embedding([audio_file_440hz, audio_file_220hz])
        emb2 = cloner.create_speaker_embedding([audio_file_440hz, audio_file_220hz])
        np.testing.assert_array_equal(emb1, emb2)
