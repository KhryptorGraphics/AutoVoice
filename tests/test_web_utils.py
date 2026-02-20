"""Tests for web utilities."""
import pytest

from auto_voice.web.utils import allowed_file, ALLOWED_AUDIO_EXTENSIONS


class TestAllowedFile:
    """allowed_file() tests."""

    @pytest.mark.smoke
    def test_wav_allowed(self):
        assert allowed_file('song.wav') is True

    def test_mp3_allowed(self):
        assert allowed_file('song.mp3') is True

    def test_flac_allowed(self):
        assert allowed_file('song.flac') is True

    def test_ogg_allowed(self):
        assert allowed_file('song.ogg') is True

    def test_txt_not_allowed(self):
        assert allowed_file('notes.txt') is False

    def test_py_not_allowed(self):
        assert allowed_file('script.py') is False

    def test_no_extension(self):
        assert allowed_file('noext') is False

    def test_empty_string(self):
        assert allowed_file('') is False

    def test_case_insensitive(self):
        assert allowed_file('song.WAV') is True
        assert allowed_file('song.Mp3') is True

    def test_multiple_dots(self):
        assert allowed_file('my.song.wav') is True


class TestAllowedExtensions:
    """ALLOWED_AUDIO_EXTENSIONS set tests."""

    @pytest.mark.smoke
    def test_is_set(self):
        assert isinstance(ALLOWED_AUDIO_EXTENSIONS, set)

    def test_common_formats_present(self):
        for ext in ['wav', 'mp3', 'flac', 'ogg']:
            assert ext in ALLOWED_AUDIO_EXTENSIONS

    def test_no_dangerous_extensions(self):
        for ext in ['exe', 'sh', 'py', 'js', 'html']:
            assert ext not in ALLOWED_AUDIO_EXTENSIONS
