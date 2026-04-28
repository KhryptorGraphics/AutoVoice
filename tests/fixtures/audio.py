"""Deterministic audio fixtures for tests that need valid voiced WAV input."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np


def voiced_audio(
    *,
    duration_seconds: float = 1.0,
    sample_rate: int = 22050,
    frequency_hz: float = 220.0,
) -> np.ndarray:
    """Return a deterministic harmonic tone with enough voiced content for QA."""
    sample_count = max(1, int(duration_seconds * sample_rate))
    time_axis = np.arange(sample_count, dtype=np.float32) / float(sample_rate)
    envelope = np.minimum(1.0, np.linspace(0.0, 12.0, sample_count, dtype=np.float32))
    audio = (
        0.45 * np.sin(2.0 * np.pi * frequency_hz * time_axis)
        + 0.20 * np.sin(2.0 * np.pi * frequency_hz * 2.0 * time_axis)
        + 0.08 * np.sin(2.0 * np.pi * frequency_hz * 3.0 * time_axis)
    )
    return (audio * envelope).astype(np.float32)


def write_voiced_wav(
    path: str | Path,
    *,
    duration_seconds: float = 1.0,
    sample_rate: int = 22050,
    frequency_hz: float = 220.0,
) -> None:
    """Write a mono 16-bit PCM WAV with deterministic voiced content."""
    audio = voiced_audio(
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        frequency_hz=frequency_hz,
    )
    pcm = np.clip(audio, -1.0, 1.0)
    pcm_i16 = (pcm * np.iinfo(np.int16).max).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_i16.tobytes())


def voiced_wav_io(
    *,
    duration_seconds: float = 1.0,
    sample_rate: int = 22050,
    frequency_hz: float = 220.0,
) -> io.BytesIO:
    """Return an in-memory WAV buffer ready for Flask multipart uploads."""
    buffer = io.BytesIO()
    audio = voiced_audio(
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        frequency_hz=frequency_hz,
    )
    pcm = np.clip(audio, -1.0, 1.0)
    pcm_i16 = (pcm * np.iinfo(np.int16).max).astype("<i2")
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_i16.tobytes())
    buffer.seek(0)
    return buffer
