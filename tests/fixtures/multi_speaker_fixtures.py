"""Multi-speaker audio fixtures for E2E testing of speaker diarization.

This module provides utilities to create realistic multi-speaker test audio
by concatenating existing single-speaker samples or generating synthetic audio.
"""

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.io import wavfile


@dataclass
class SpeakerInfo:
    """Information about a speaker segment in test audio."""

    speaker_id: str
    name: str
    start: float  # Start time in seconds
    end: float  # End time in seconds
    source_file: Optional[str] = None  # Original source file if from real audio

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class MultiSpeakerFixture:
    """A multi-speaker audio fixture with ground truth annotations."""

    audio_path: str
    sample_rate: int
    duration: float
    speakers: List[SpeakerInfo]
    num_speakers: int = field(init=False)

    def __post_init__(self):
        self.num_speakers = len(set(s.speaker_id for s in self.speakers))

    def get_speaker_segments(self, speaker_id: str) -> List[SpeakerInfo]:
        """Get all segments for a specific speaker."""
        return [s for s in self.speakers if s.speaker_id == speaker_id]

    def get_speaker_total_duration(self, speaker_id: str) -> float:
        """Get total duration for a speaker."""
        return sum(s.duration for s in self.get_speaker_segments(speaker_id))

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "audio_path": self.audio_path,
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "num_speakers": self.num_speakers,
            "speakers": [
                {
                    "speaker_id": s.speaker_id,
                    "name": s.name,
                    "start": s.start,
                    "end": s.end,
                    "source_file": s.source_file,
                }
                for s in self.speakers
            ],
        }


def create_synthetic_multi_speaker(
    output_path: str,
    durations: List[Tuple[str, float]] = None,
    sample_rate: int = 16000,
) -> MultiSpeakerFixture:
    """Create synthetic multi-speaker audio using different frequency tones.

    Args:
        output_path: Where to save the generated audio
        durations: List of (speaker_id, duration) tuples
        sample_rate: Output sample rate

    Returns:
        MultiSpeakerFixture with ground truth annotations
    """
    if durations is None:
        # Default: 2 speakers alternating
        durations = [
            ("SPEAKER_00", 2.0),
            ("SPEAKER_01", 2.0),
            ("SPEAKER_00", 1.5),
            ("SPEAKER_01", 1.5),
        ]

    # Different base frequencies for different speakers
    speaker_freqs = {
        "SPEAKER_00": 200.0,  # Lower voice
        "SPEAKER_01": 280.0,  # Higher voice
        "SPEAKER_02": 150.0,  # Even lower
        "SPEAKER_03": 320.0,  # Even higher
    }

    segments = []
    current_time = 0.0
    waveforms = []

    for speaker_id, duration in durations:
        freq = speaker_freqs.get(speaker_id, 200.0)
        num_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, num_samples, endpoint=False)

        # Create speech-like signal with harmonics and amplitude modulation
        fundamental = np.sin(2 * np.pi * freq * t)
        harmonic2 = 0.5 * np.sin(2 * np.pi * freq * 2 * t)
        harmonic3 = 0.25 * np.sin(2 * np.pi * freq * 3 * t)

        # Amplitude envelope for speech-like quality
        envelope = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 3 * t))

        waveform = (fundamental + harmonic2 + harmonic3) * envelope * 0.4

        # Add slight noise
        waveform += np.random.randn(num_samples) * 0.02

        waveforms.append(waveform)

        segments.append(SpeakerInfo(
            speaker_id=speaker_id,
            name=f"Synthetic {speaker_id}",
            start=current_time,
            end=current_time + duration,
            source_file=None,
        ))

        current_time += duration

    # Concatenate all segments
    full_waveform = np.concatenate(waveforms)

    # Normalize and convert to int16
    full_waveform = full_waveform / (np.max(np.abs(full_waveform)) + 1e-8)
    waveform_int = (full_waveform * 32767 * 0.9).astype(np.int16)

    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    wavfile.write(output_path, sample_rate, waveform_int)

    return MultiSpeakerFixture(
        audio_path=output_path,
        sample_rate=sample_rate,
        duration=current_time,
        speakers=segments,
    )


def create_multi_speaker_audio(
    speaker_files: List[Tuple[str, str, float, float]],
    output_path: str,
    sample_rate: int = 16000,
    crossfade_ms: int = 50,
) -> MultiSpeakerFixture:
    """Create multi-speaker audio by concatenating segments from real audio files.

    Args:
        speaker_files: List of (speaker_id, audio_path, start_sec, end_sec) tuples
        output_path: Where to save the generated audio
        sample_rate: Output sample rate (files will be resampled)
        crossfade_ms: Crossfade duration in milliseconds between segments

    Returns:
        MultiSpeakerFixture with ground truth annotations
    """
    try:
        import soundfile as sf
        from scipy import signal
    except ImportError:
        raise ImportError("soundfile and scipy are required for real audio concatenation")

    segments = []
    waveforms = []
    current_time = 0.0
    crossfade_samples = int(crossfade_ms * sample_rate / 1000)

    for speaker_id, audio_path, start_sec, end_sec in speaker_files:
        # Load audio with soundfile
        waveform, sr = sf.read(audio_path)

        # Convert to mono if stereo
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)

        # Resample if needed
        if sr != sample_rate:
            num_samples = int(len(waveform) * sample_rate / sr)
            waveform = signal.resample(waveform, num_samples)

        # Extract segment
        start_sample = int(start_sec * sample_rate)
        end_sample = int(end_sec * sample_rate)
        end_sample = min(end_sample, len(waveform))
        segment_waveform = waveform[start_sample:end_sample].astype(np.float32)

        # Apply crossfade if not first segment
        if waveforms and crossfade_samples > 0:
            fade_len = min(crossfade_samples, len(segment_waveform), len(waveforms[-1]))
            if fade_len > 0:
                # Crossfade: fade out previous, fade in current
                fade_out = np.linspace(1, 0, fade_len)
                fade_in = np.linspace(0, 1, fade_len)

                waveforms[-1][-fade_len:] *= fade_out
                segment_waveform[:fade_len] *= fade_in
                waveforms[-1][-fade_len:] += segment_waveform[:fade_len]
                segment_waveform = segment_waveform[fade_len:]
                current_time -= crossfade_ms / 1000

        waveforms.append(segment_waveform)

        duration = len(segment_waveform) / sample_rate
        if waveforms and crossfade_samples > 0 and len(waveforms) > 1:
            duration += crossfade_ms / 1000

        segments.append(SpeakerInfo(
            speaker_id=speaker_id,
            name=speaker_id,
            start=current_time,
            end=current_time + (end_sec - start_sec),
            source_file=audio_path,
        ))

        current_time += (end_sec - start_sec)

    # Concatenate all segments
    full_waveform = np.concatenate(waveforms)

    # Normalize
    max_val = np.max(np.abs(full_waveform))
    if max_val > 0:
        full_waveform = full_waveform / max_val * 0.9

    # Convert to int16
    waveform_int = (full_waveform * 32767).astype(np.int16)

    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    wavfile.write(output_path, sample_rate, waveform_int)

    return MultiSpeakerFixture(
        audio_path=output_path,
        sample_rate=sample_rate,
        duration=current_time,
        speakers=segments,
    )


# Pre-defined fixture generators using existing quality samples
def get_quality_samples_dir() -> Path:
    """Get the path to quality samples directory."""
    return Path(__file__).parent.parent / "quality_samples"


def create_duet_fixture(output_dir: str = None) -> Optional[MultiSpeakerFixture]:
    """Create a duet fixture using Conor Maynard and William Singe samples.

    Returns:
        MultiSpeakerFixture or None if samples not available
    """
    samples_dir = get_quality_samples_dir()
    conor_file = samples_dir / "conor_maynard_pillowtalk.wav"
    william_file = samples_dir / "william_singe_pillowtalk.wav"

    if not conor_file.exists() or not william_file.exists():
        return None

    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    output_path = os.path.join(output_dir, "duet_fixture.wav")

    # Create alternating segments (simulating a duet)
    return create_multi_speaker_audio(
        speaker_files=[
            ("conor", str(conor_file), 0.0, 5.0),
            ("william", str(william_file), 0.0, 5.0),
            ("conor", str(conor_file), 5.0, 10.0),
            ("william", str(william_file), 5.0, 10.0),
        ],
        output_path=output_path,
        sample_rate=16000,
    )


def create_interview_fixture(output_dir: str = None) -> Optional[MultiSpeakerFixture]:
    """Create an interview-style fixture with longer speaker turns.

    Returns:
        MultiSpeakerFixture or None if samples not available
    """
    samples_dir = get_quality_samples_dir()
    conor_file = samples_dir / "conor_maynard_pillowtalk.wav"
    william_file = samples_dir / "william_singe_pillowtalk.wav"

    if not conor_file.exists() or not william_file.exists():
        return None

    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    output_path = os.path.join(output_dir, "interview_fixture.wav")

    # Longer turns like an interview
    return create_multi_speaker_audio(
        speaker_files=[
            ("interviewer", str(conor_file), 0.0, 8.0),
            ("guest", str(william_file), 0.0, 12.0),
            ("interviewer", str(conor_file), 8.0, 12.0),
            ("guest", str(william_file), 12.0, 20.0),
        ],
        output_path=output_path,
        sample_rate=16000,
    )
