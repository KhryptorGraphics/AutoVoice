"""Focused coverage for audio processor validation and metadata helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from auto_voice.audio.processor import AudioProcessor


@pytest.fixture
def processor() -> AudioProcessor:
    return AudioProcessor(sample_rate=22050)


@pytest.fixture
def wav_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.wav"
    audio = np.linspace(-0.2, 0.2, 22050, dtype=np.float32)
    sf.write(path, audio, 22050)
    return path


def test_trim_silence_removes_padding(processor: AudioProcessor) -> None:
    audio = np.concatenate(
        [
            np.zeros(2048, dtype=np.float32),
            np.ones(2048, dtype=np.float32) * 0.25,
            np.zeros(2048, dtype=np.float32),
        ]
    )

    trimmed = processor.trim_silence(audio)

    assert len(trimmed) < len(audio)
    assert np.max(np.abs(trimmed)) > 0


def test_validate_format_accepts_supported_extension(processor: AudioProcessor) -> None:
    assert processor.validate_format("track.FLAC") is True


@pytest.mark.parametrize(
    ("file_path", "message"),
    [
        ("", "Invalid file path"),
        ("missing_extension", "Invalid file path"),
        ("track.txt", "Unsupported audio format"),
    ],
)
def test_validate_format_rejects_invalid_paths(
    processor: AudioProcessor,
    file_path: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        processor.validate_format(file_path)


def test_get_audio_info_returns_metadata(processor: AudioProcessor, wav_file: Path) -> None:
    info = processor.get_audio_info(str(wav_file))

    assert info["sample_rate"] == 22050
    assert info["channels"] == 1
    assert info["frames"] > 0
    assert info["duration"] > 0
    assert info["format"].startswith("WAV/")


def test_get_audio_info_missing_file_raises(processor: AudioProcessor, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        processor.get_audio_info(str(tmp_path / "missing.wav"))


def test_get_audio_info_wraps_soundfile_errors(processor: AudioProcessor, wav_file: Path) -> None:
    with patch("soundfile.SoundFile", side_effect=RuntimeError("decode failed")):
        with pytest.raises(RuntimeError, match="Failed to read audio file metadata"):
            processor.get_audio_info(str(wav_file))


def test_validate_duration_checks_min_and_max(processor: AudioProcessor, wav_file: Path) -> None:
    assert processor.validate_duration(str(wav_file), min_duration=0.5, max_duration=2.0) is True

    with pytest.raises(ValueError, match="below minimum"):
        processor.validate_duration(str(wav_file), min_duration=2.0)

    with pytest.raises(ValueError, match="exceeds maximum"):
        processor.validate_duration(str(wav_file), max_duration=0.5)


def test_validate_sample_rate_checks_allowed_values(
    processor: AudioProcessor,
    wav_file: Path,
) -> None:
    assert processor.validate_sample_rate(str(wav_file), allowed_rates=[16000, 22050]) is True

    with pytest.raises(ValueError, match="is not allowed"):
        processor.validate_sample_rate(str(wav_file), allowed_rates=[16000])


def test_validate_audio_file_returns_info_without_optional_constraints(
    processor: AudioProcessor,
    wav_file: Path,
) -> None:
    info = processor.validate_audio_file(str(wav_file))

    assert info["sample_rate"] == 22050
    assert info["duration"] > 0


def test_validate_audio_file_applies_all_constraints(
    processor: AudioProcessor,
    wav_file: Path,
) -> None:
    info = processor.validate_audio_file(
        str(wav_file),
        min_duration=0.5,
        max_duration=2.0,
        allowed_sample_rates=[22050],
    )

    assert info["channels"] == 1


def test_validate_audio_file_reuses_format_validation(processor: AudioProcessor) -> None:
    with pytest.raises(ValueError, match="Unsupported audio format"):
        processor.validate_audio_file("bad.mp4")
