"""Tests for audio validation methods."""
import os
import pytest
import numpy as np
import soundfile as sf

from auto_voice.audio.processor import AudioProcessor, ALLOWED_AUDIO_EXTENSIONS


class TestValidateFormat:
    """Tests for validate_format method."""

    @pytest.mark.smoke
    def test_valid_wav_extension(self, audio_processor):
        """Test validation passes for .wav file."""
        assert audio_processor.validate_format("test.wav") is True

    @pytest.mark.smoke
    def test_valid_mp3_extension(self, audio_processor):
        """Test validation passes for .mp3 file."""
        assert audio_processor.validate_format("test.mp3") is True

    def test_valid_flac_extension(self, audio_processor):
        """Test validation passes for .flac file."""
        assert audio_processor.validate_format("audio.flac") is True

    def test_valid_with_path(self, audio_processor):
        """Test validation with full path."""
        assert audio_processor.validate_format("/path/to/audio.ogg") is True

    def test_case_insensitive(self, audio_processor):
        """Test validation is case insensitive."""
        assert audio_processor.validate_format("test.WAV") is True
        assert audio_processor.validate_format("test.Mp3") is True
        assert audio_processor.validate_format("test.FLAC") is True

    def test_invalid_extension(self, audio_processor):
        """Test validation fails for invalid extension."""
        with pytest.raises(ValueError, match="Unsupported audio format: .txt"):
            audio_processor.validate_format("test.txt")

    def test_invalid_extension_video(self, audio_processor):
        """Test validation fails for video format."""
        with pytest.raises(ValueError, match="Unsupported audio format: .mp4"):
            audio_processor.validate_format("video.mp4")

    def test_no_extension(self, audio_processor):
        """Test validation fails when no extension present."""
        with pytest.raises(ValueError, match="Invalid file path: must contain an extension"):
            audio_processor.validate_format("noextension")

    def test_empty_path(self, audio_processor):
        """Test validation fails for empty path."""
        with pytest.raises(ValueError, match="Invalid file path"):
            audio_processor.validate_format("")

    def test_all_allowed_extensions(self, audio_processor):
        """Test all allowed extensions are accepted."""
        for ext in ALLOWED_AUDIO_EXTENSIONS:
            assert audio_processor.validate_format(f"test.{ext}") is True


class TestGetAudioInfo:
    """Tests for get_audio_info method."""

    @pytest.mark.smoke
    def test_get_info_from_valid_file(self, audio_processor, sample_audio_file):
        """Test getting info from valid audio file."""
        info = audio_processor.get_audio_info(sample_audio_file)

        assert 'duration' in info
        assert 'sample_rate' in info
        assert 'channels' in info
        assert 'format' in info
        assert 'frames' in info

        assert info['sample_rate'] == 22050
        assert info['channels'] == 1
        assert info['duration'] > 0

    def test_get_info_duration_calculation(self, audio_processor, sample_audio_file):
        """Test duration is correctly calculated."""
        info = audio_processor.get_audio_info(sample_audio_file)

        # sample_audio fixture creates 5-second audio
        assert abs(info['duration'] - 5.0) < 0.1

    def test_get_info_frame_count(self, audio_processor, sample_audio_file):
        """Test frame count matches expected value."""
        info = audio_processor.get_audio_info(sample_audio_file)

        # Duration * sample_rate = frames
        expected_frames = int(info['duration'] * info['sample_rate'])
        assert abs(info['frames'] - expected_frames) < 10

    def test_get_info_nonexistent_file(self, audio_processor):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            audio_processor.get_audio_info("/nonexistent/file.wav")

    def test_get_info_invalid_file(self, audio_processor, tmp_path):
        """Test error when file is not valid audio."""
        invalid_file = tmp_path / "invalid.wav"
        invalid_file.write_text("not audio data")

        with pytest.raises(RuntimeError, match="Failed to read audio file metadata"):
            audio_processor.get_audio_info(str(invalid_file))

    def test_get_info_stereo_file(self, audio_processor, tmp_path):
        """Test getting info from stereo audio file."""
        # Create stereo audio
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        left = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        right = np.sin(2 * np.pi * 880 * t).astype(np.float32)
        stereo = np.vstack([left, right]).T

        path = tmp_path / "stereo.wav"
        sf.write(str(path), stereo, sr)

        info = audio_processor.get_audio_info(str(path))
        assert info['channels'] == 2
        assert info['sample_rate'] == 22050


class TestValidateDuration:
    """Tests for validate_duration method."""

    @pytest.mark.smoke
    def test_validate_duration_within_range(self, audio_processor, sample_audio_file):
        """Test validation passes when duration is within range."""
        # sample_audio is ~5 seconds
        assert audio_processor.validate_duration(
            sample_audio_file,
            min_duration=1.0,
            max_duration=10.0
        ) is True

    def test_validate_duration_no_limits(self, audio_processor, sample_audio_file):
        """Test validation with no duration limits."""
        assert audio_processor.validate_duration(sample_audio_file) is True

    def test_validate_duration_only_min(self, audio_processor, sample_audio_file):
        """Test validation with only minimum duration."""
        assert audio_processor.validate_duration(
            sample_audio_file,
            min_duration=1.0
        ) is True

    def test_validate_duration_only_max(self, audio_processor, sample_audio_file):
        """Test validation with only maximum duration."""
        assert audio_processor.validate_duration(
            sample_audio_file,
            max_duration=10.0
        ) is True

    def test_validate_duration_below_minimum(self, audio_processor, sample_audio_file):
        """Test validation fails when duration below minimum."""
        with pytest.raises(ValueError, match="below minimum"):
            audio_processor.validate_duration(
                sample_audio_file,
                min_duration=10.0
            )

    def test_validate_duration_above_maximum(self, audio_processor, sample_audio_file):
        """Test validation fails when duration above maximum."""
        with pytest.raises(ValueError, match="exceeds maximum"):
            audio_processor.validate_duration(
                sample_audio_file,
                max_duration=1.0
            )

    def test_validate_duration_exact_minimum(self, audio_processor, tmp_path):
        """Test validation passes when duration equals minimum."""
        # Create 3-second audio
        sr = 22050
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        path = tmp_path / "exact.wav"
        sf.write(str(path), audio, sr)

        assert audio_processor.validate_duration(str(path), min_duration=3.0) is True

    def test_validate_duration_exact_maximum(self, audio_processor, tmp_path):
        """Test validation passes when duration equals maximum."""
        # Create 3-second audio
        sr = 22050
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        path = tmp_path / "exact.wav"
        sf.write(str(path), audio, sr)

        assert audio_processor.validate_duration(str(path), max_duration=3.0) is True

    def test_validate_duration_nonexistent_file(self, audio_processor):
        """Test validation fails for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            audio_processor.validate_duration("/nonexistent/file.wav")


class TestValidateSampleRate:
    """Tests for validate_sample_rate method."""

    @pytest.mark.smoke
    def test_validate_sample_rate_allowed(self, audio_processor, sample_audio_file):
        """Test validation passes when sample rate is allowed."""
        assert audio_processor.validate_sample_rate(
            sample_audio_file,
            allowed_rates=[22050, 44100]
        ) is True

    def test_validate_sample_rate_no_restriction(self, audio_processor, sample_audio_file):
        """Test validation passes with no rate restriction."""
        assert audio_processor.validate_sample_rate(sample_audio_file) is True

    def test_validate_sample_rate_not_allowed(self, audio_processor, sample_audio_file):
        """Test validation fails when sample rate not in allowed list."""
        with pytest.raises(ValueError, match="not allowed"):
            audio_processor.validate_sample_rate(
                sample_audio_file,
                allowed_rates=[16000, 48000]
            )

    def test_validate_sample_rate_single_allowed(self, audio_processor, sample_audio_file):
        """Test validation with single allowed rate."""
        assert audio_processor.validate_sample_rate(
            sample_audio_file,
            allowed_rates=[22050]
        ) is True

    def test_validate_sample_rate_common_rates(self, audio_processor, tmp_path):
        """Test validation for common sample rates."""
        common_rates = [8000, 16000, 22050, 44100, 48000]

        for sr in common_rates:
            # Create audio at specific sample rate
            duration = 1.0
            t = np.linspace(0, duration, int(sr * duration), endpoint=False)
            audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

            path = tmp_path / f"audio_{sr}.wav"
            sf.write(str(path), audio, sr)

            # Should pass when rate is in allowed list
            assert audio_processor.validate_sample_rate(
                str(path),
                allowed_rates=common_rates
            ) is True

    def test_validate_sample_rate_nonexistent_file(self, audio_processor):
        """Test validation fails for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            audio_processor.validate_sample_rate("/nonexistent/file.wav")


class TestValidateAudioFile:
    """Tests for comprehensive validate_audio_file method."""

    @pytest.mark.smoke
    def test_validate_audio_file_no_constraints(self, audio_processor, sample_audio_file):
        """Test validation with no constraints (format + readability only)."""
        info = audio_processor.validate_audio_file(sample_audio_file)

        assert info is not None
        assert 'duration' in info
        assert 'sample_rate' in info
        assert info['duration'] > 0

    def test_validate_audio_file_all_constraints(self, audio_processor, sample_audio_file):
        """Test validation with all constraints."""
        info = audio_processor.validate_audio_file(
            sample_audio_file,
            min_duration=1.0,
            max_duration=10.0,
            allowed_sample_rates=[22050, 44100]
        )

        assert info['duration'] >= 1.0
        assert info['duration'] <= 10.0
        assert info['sample_rate'] in [22050, 44100]

    def test_validate_audio_file_duration_only(self, audio_processor, sample_audio_file):
        """Test validation with only duration constraints."""
        info = audio_processor.validate_audio_file(
            sample_audio_file,
            min_duration=1.0,
            max_duration=10.0
        )

        assert info['duration'] >= 1.0
        assert info['duration'] <= 10.0

    def test_validate_audio_file_sample_rate_only(self, audio_processor, sample_audio_file):
        """Test validation with only sample rate constraint."""
        info = audio_processor.validate_audio_file(
            sample_audio_file,
            allowed_sample_rates=[22050, 44100]
        )

        assert info['sample_rate'] in [22050, 44100]

    def test_validate_audio_file_invalid_format(self, audio_processor, tmp_path):
        """Test validation fails for invalid file format."""
        invalid_file = tmp_path / "test.txt"
        invalid_file.write_text("not audio")

        with pytest.raises(ValueError, match="Unsupported audio format"):
            audio_processor.validate_audio_file(str(invalid_file))

    def test_validate_audio_file_nonexistent(self, audio_processor):
        """Test validation fails for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            audio_processor.validate_audio_file("/nonexistent/file.wav")

    def test_validate_audio_file_duration_too_short(self, audio_processor, short_audio_file):
        """Test validation fails when audio is too short."""
        with pytest.raises(ValueError, match="below minimum"):
            audio_processor.validate_audio_file(
                short_audio_file,
                min_duration=5.0
            )

    def test_validate_audio_file_duration_too_long(self, audio_processor, sample_audio_file):
        """Test validation fails when audio is too long."""
        with pytest.raises(ValueError, match="exceeds maximum"):
            audio_processor.validate_audio_file(
                sample_audio_file,
                max_duration=1.0
            )

    def test_validate_audio_file_invalid_sample_rate(self, audio_processor, sample_audio_file):
        """Test validation fails for invalid sample rate."""
        with pytest.raises(ValueError, match="not allowed"):
            audio_processor.validate_audio_file(
                sample_audio_file,
                allowed_sample_rates=[16000, 48000]
            )

    def test_validate_audio_file_returns_complete_info(self, audio_processor, sample_audio_file):
        """Test that validation returns complete audio metadata."""
        info = audio_processor.validate_audio_file(sample_audio_file)

        required_keys = ['duration', 'sample_rate', 'channels', 'format', 'frames']
        for key in required_keys:
            assert key in info, f"Missing key: {key}"

        # Check types
        assert isinstance(info['duration'], float)
        assert isinstance(info['sample_rate'], int)
        assert isinstance(info['channels'], int)
        assert isinstance(info['format'], str)
        assert isinstance(info['frames'], int)

    def test_validate_audio_file_edge_case_min_duration_zero(self, audio_processor, sample_audio_file):
        """Test validation with min_duration=0."""
        info = audio_processor.validate_audio_file(
            sample_audio_file,
            min_duration=0
        )
        assert info is not None

    def test_validate_audio_file_multiple_formats(self, audio_processor, tmp_path):
        """Test validation works for multiple audio formats."""
        # Create test audio
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Test WAV format
        wav_path = tmp_path / "test.wav"
        sf.write(str(wav_path), audio, sr)
        info = audio_processor.validate_audio_file(str(wav_path))
        assert 'WAV' in info['format'] or 'wav' in info['format'].lower()

        # Test FLAC format
        flac_path = tmp_path / "test.flac"
        sf.write(str(flac_path), audio, sr, format='FLAC')
        info = audio_processor.validate_audio_file(str(flac_path))
        assert 'FLAC' in info['format'] or 'flac' in info['format'].lower()

    def test_validate_audio_file_unreadable_file(self, audio_processor, tmp_path):
        """Test validation fails gracefully for unreadable files."""
        # Create a file with .wav extension but invalid content
        invalid_wav = tmp_path / "invalid.wav"
        invalid_wav.write_bytes(b"not a wav file")

        with pytest.raises(RuntimeError, match="Failed to read audio file metadata"):
            audio_processor.validate_audio_file(str(invalid_wav))
