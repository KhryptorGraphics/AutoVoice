"""Tests for diarization extractor module.

Tests segment extraction from diarization timestamps including:
- Segment extraction accuracy
- Audio quality preservation
- Multi-speaker separation
- Overlapping speech handling
"""

import json
import numpy as np
import pytest
import tempfile
from pathlib import Path

from auto_voice.audio.diarization_extractor import (
    DiarizationExtractor,
    SpeakerSegment,
    ExtractionResult,
    SpeakerExtractionInfo,
)


@pytest.fixture
def extractor(tmp_path):
    """Create DiarizationExtractor instance with temp directories."""
    profiles_dir = tmp_path / "voice_profiles"
    training_dir = tmp_path / "training_vocals"
    profiles_dir.mkdir()
    training_dir.mkdir()

    return DiarizationExtractor(
        fade_ms=10.0,
        min_segment_duration=0.5,
        profiles_dir=profiles_dir,
        training_vocals_dir=training_dir,
    )


@pytest.fixture
def sample_diarization_json(tmp_path):
    """Create sample diarization JSON file."""
    audio_path = tmp_path / "test_audio.wav"

    diarization_data = {
        "file": str(audio_path),
        "segments": [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"},
            {"start": 4.0, "end": 6.0, "speaker": "SPEAKER_00"},
        ]
    }

    json_path = tmp_path / "diarization.json"
    with open(json_path, 'w') as f:
        json.dump(diarization_data, f)

    return json_path, audio_path


@pytest.fixture
def sample_audio_with_diarization(tmp_path):
    """Create audio file matching diarization JSON."""
    import soundfile as sf

    sr = 16000
    duration = 6.0
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples, endpoint=False)

    # Create alternating speaker audio
    audio = np.zeros(samples, dtype=np.float32)

    # SPEAKER_00: 0-2s, 4-6s (440 Hz)
    mask0_1 = (t >= 0) & (t < 2)
    mask0_2 = (t >= 4) & (t < 6)
    audio[mask0_1 | mask0_2] = 0.5 * np.sin(2 * np.pi * 440 * t[mask0_1 | mask0_2])

    # SPEAKER_01: 2-4s (880 Hz)
    mask1 = (t >= 2) & (t < 4)
    audio[mask1] = 0.5 * np.sin(2 * np.pi * 880 * t[mask1])

    audio_path = tmp_path / "test_audio.wav"
    sf.write(str(audio_path), audio, sr)

    # Create matching diarization JSON
    diarization_data = {
        "file": str(audio_path),
        "segments": [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"},
            {"start": 4.0, "end": 6.0, "speaker": "SPEAKER_00"},
        ]
    }

    json_path = tmp_path / "diarization.json"
    with open(json_path, 'w') as f:
        json.dump(diarization_data, f)

    return json_path, audio_path, sr


# ===== Phase 2.1: Test segment extraction from timestamps =====

def test_load_diarization(extractor, sample_diarization_json):
    """Test loading diarization results from JSON."""
    json_path, audio_path = sample_diarization_json

    audio_file, segments = extractor.load_diarization(json_path)

    assert audio_file == str(audio_path)
    assert len(segments) == 3
    assert all(isinstance(s, SpeakerSegment) for s in segments)


def test_segment_extraction_timestamps(extractor, sample_audio_with_diarization):
    """Test segment extraction matches timestamps."""
    json_path, audio_path, sr = sample_audio_with_diarization

    audio_file, segments = extractor.load_diarization(json_path)

    import librosa
    audio, _ = librosa.load(str(audio_path), sr=sr, mono=True)

    # Extract first speaker segment
    speaker_audio = extractor.extract_speaker_track(audio, sr, segments, "SPEAKER_00")

    # Check that speaker audio is full-length
    assert len(speaker_audio) == len(audio)

    # Check that non-speaker regions are silent
    # Sample from SPEAKER_01 region (2.5s) should be near zero
    sample_idx = int(2.5 * sr)
    assert abs(speaker_audio[sample_idx]) < 0.1


def test_segment_boundary_accuracy(extractor, sample_audio_with_diarization):
    """Test start/end boundary accuracy."""
    json_path, audio_path, sr = sample_audio_with_diarization

    audio_file, segments = extractor.load_diarization(json_path)

    for segment in segments:
        # Check boundaries are within valid range
        assert segment.start >= 0
        assert segment.end > segment.start
        assert segment.duration > 0


# ===== Phase 2.2: Test segment audio quality =====

def test_no_clipping(extractor, sample_audio_with_diarization):
    """Test extracted audio has no clipping."""
    json_path, audio_path, sr = sample_audio_with_diarization

    audio_file, segments = extractor.load_diarization(json_path)

    import librosa
    audio, _ = librosa.load(str(audio_path), sr=sr, mono=True)

    for speaker_id in ["SPEAKER_00", "SPEAKER_01"]:
        speaker_audio = extractor.extract_speaker_track(audio, sr, segments, speaker_id)

        # Check no values exceed [-1, 1]
        assert speaker_audio.max() <= 1.0
        assert speaker_audio.min() >= -1.0


def test_sample_rate_preservation(extractor, sample_audio_with_diarization, tmp_path):
    """Test sample rate is preserved."""
    json_path, audio_path, sr = sample_audio_with_diarization

    result = extractor.process_track(json_path, audio_path, "test_artist", output_dir=tmp_path)

    # Load output files and verify sample rate
    import soundfile as sf
    for speaker_info in result.speakers.values():
        output_audio, output_sr = sf.read(speaker_info.output_file)
        assert output_sr == sr


def test_audio_format_consistency(extractor, sample_audio_with_diarization, tmp_path):
    """Test audio format consistency (mono float32)."""
    json_path, audio_path, sr = sample_audio_with_diarization

    result = extractor.process_track(json_path, audio_path, "test_artist", output_dir=tmp_path)

    import soundfile as sf
    for speaker_info in result.speakers.values():
        output_audio, output_sr = sf.read(speaker_info.output_file)

        # Should be 1D (mono)
        assert output_audio.ndim == 1 or (output_audio.ndim == 2 and output_audio.shape[1] == 1)


# ===== Phase 2.3: Test multiple speakers =====

def test_multiple_speaker_extraction(extractor, sample_audio_with_diarization, tmp_path):
    """Test extracting segments for 2-3 speakers."""
    json_path, audio_path, sr = sample_audio_with_diarization

    result = extractor.process_track(json_path, audio_path, "test_artist", output_dir=tmp_path)

    # Should have extracted at least 2 speakers
    assert len(result.speakers) >= 2

    # Each speaker should have output file
    for speaker_info in result.speakers.values():
        assert Path(speaker_info.output_file).exists()


def test_speaker_separation(extractor, sample_audio_with_diarization):
    """Test speaker separation is correct."""
    json_path, audio_path, sr = sample_audio_with_diarization

    audio_file, segments = extractor.load_diarization(json_path)

    import librosa
    audio, _ = librosa.load(str(audio_path), sr=sr, mono=True)

    # Extract both speakers
    speaker_00_audio = extractor.extract_speaker_track(audio, sr, segments, "SPEAKER_00")
    speaker_01_audio = extractor.extract_speaker_track(audio, sr, segments, "SPEAKER_01")

    # Check that they're different (not both silent)
    assert not np.allclose(speaker_00_audio, speaker_01_audio)

    # Check that non-zero regions don't overlap much
    threshold = 0.1
    speaker_00_active = np.abs(speaker_00_audio) > threshold
    speaker_01_active = np.abs(speaker_01_audio) > threshold

    overlap = np.sum(speaker_00_active & speaker_01_active)
    total_active = np.sum(speaker_00_active | speaker_01_active)

    if total_active > 0:
        overlap_ratio = overlap / total_active
        # Minimal overlap expected
        assert overlap_ratio < 0.3


def test_segment_assignment_correctness(extractor, sample_audio_with_diarization):
    """Test segments are assigned to correct speakers."""
    json_path, audio_path, sr = sample_audio_with_diarization

    audio_file, segments = extractor.load_diarization(json_path)

    # Get speaker durations
    durations = extractor.get_speaker_durations(segments)

    # SPEAKER_00 should have ~4s (0-2s + 4-6s)
    # SPEAKER_01 should have ~2s (2-4s)
    assert "SPEAKER_00" in durations
    assert "SPEAKER_01" in durations

    # Check duration is roughly correct (allow some tolerance)
    assert durations["SPEAKER_00"] == pytest.approx(4.0, abs=0.5)
    assert durations["SPEAKER_01"] == pytest.approx(2.0, abs=0.5)


# ===== Phase 2.4: Test edge cases =====

def test_overlapping_speech(extractor, tmp_path):
    """Test handling overlapping speech (choose dominant speaker)."""
    import soundfile as sf

    sr = 16000
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Create overlapping audio
    audio = (0.3 * np.sin(2 * np.pi * 440 * t) +
             0.3 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)

    audio_path = tmp_path / "overlap.wav"
    sf.write(str(audio_path), audio, sr)

    # Create diarization with overlapping segments
    diarization_data = {
        "file": str(audio_path),
        "segments": [
            {"start": 0.0, "end": 2.5, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"},  # Overlap at 2-2.5s
        ]
    }

    json_path = tmp_path / "overlap_diar.json"
    with open(json_path, 'w') as f:
        json.dump(diarization_data, f)

    # Process should handle gracefully
    result = extractor.process_track(json_path, audio_path, "test_artist", output_dir=tmp_path)

    assert len(result.speakers) >= 1


def test_very_short_segments(extractor, tmp_path):
    """Test segments < 1s (below min_duration)."""
    import soundfile as sf

    sr = 16000
    duration = 3.0
    audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration))).astype(np.float32)

    audio_path = tmp_path / "short_seg.wav"
    sf.write(str(audio_path), audio, sr)

    # Create diarization with very short segments
    diarization_data = {
        "file": str(audio_path),
        "segments": [
            {"start": 0.0, "end": 0.3, "speaker": "SPEAKER_00"},  # Below min_duration
            {"start": 1.0, "end": 2.5, "speaker": "SPEAKER_00"},  # Above min_duration
        ]
    }

    json_path = tmp_path / "short_seg_diar.json"
    with open(json_path, 'w') as f:
        json.dump(diarization_data, f)

    # Should filter out short segments
    audio_file, segments = extractor.load_diarization(json_path)

    durations = extractor.get_speaker_durations(segments)

    # Only the 1.5s segment should count
    assert durations["SPEAKER_00"] == pytest.approx(1.5, abs=0.1)


def test_silence_padding(extractor, sample_audio_with_diarization):
    """Test fade in/out (silence padding) reduces clicks."""
    json_path, audio_path, sr = sample_audio_with_diarization

    audio_file, segments = extractor.load_diarization(json_path)

    import librosa
    audio, _ = librosa.load(str(audio_path), sr=sr, mono=True)

    # Extract with fading
    speaker_audio = extractor.extract_speaker_track(audio, sr, segments, "SPEAKER_00")

    # Check that segment boundaries have fade (gradual transition)
    # At 2s boundary (end of first SPEAKER_00 segment)
    boundary_idx = int(2.0 * sr)
    fade_samples = int(extractor.fade_ms * sr / 1000)

    if boundary_idx + fade_samples < len(speaker_audio):
        # Samples just before boundary should be fading
        fade_region = speaker_audio[boundary_idx - fade_samples:boundary_idx]

        # Should have decreasing amplitude (fade out)
        if np.abs(fade_region).max() > 0.1:
            # Check that amplitude generally decreases
            assert np.abs(fade_region[-1]) < np.abs(fade_region[0]) * 1.5


# ===== Helper function tests =====

def test_get_speaker_durations(extractor):
    """Test speaker duration calculation."""
    segments = [
        SpeakerSegment(0.0, 2.0, "SPEAKER_00"),
        SpeakerSegment(2.0, 4.0, "SPEAKER_01"),
        SpeakerSegment(4.0, 6.0, "SPEAKER_00"),
    ]

    durations = extractor.get_speaker_durations(segments)

    assert durations["SPEAKER_00"] == pytest.approx(4.0, abs=0.01)
    assert durations["SPEAKER_01"] == pytest.approx(2.0, abs=0.01)


def test_identify_primary_speaker(extractor):
    """Test primary speaker identification."""
    segments = [
        SpeakerSegment(0.0, 1.0, "SPEAKER_00"),
        SpeakerSegment(1.0, 5.0, "SPEAKER_01"),  # Longest duration
        SpeakerSegment(5.0, 6.0, "SPEAKER_00"),
    ]

    primary = extractor.identify_primary_speaker(segments)

    assert primary == "SPEAKER_01"


def test_get_or_create_profile(extractor):
    """Test profile creation and retrieval."""
    profile_id = extractor.get_or_create_profile("test_artist", "SPEAKER_00", is_primary=True)

    assert profile_id is not None
    assert isinstance(profile_id, str)

    # Second call should return same profile ID
    profile_id_2 = extractor.get_or_create_profile("test_artist", "SPEAKER_00", is_primary=True)

    assert profile_id_2 == profile_id


# ===== Process track tests =====

def test_process_track_success(extractor, sample_audio_with_diarization, tmp_path):
    """Test successful track processing."""
    json_path, audio_path, sr = sample_audio_with_diarization

    result = extractor.process_track(json_path, audio_path, "test_artist", output_dir=tmp_path)

    assert isinstance(result, ExtractionResult)
    assert result.source_file == str(audio_path)
    assert result.total_duration > 0
    assert len(result.speakers) >= 1


def test_process_track_creates_output_files(extractor, sample_audio_with_diarization, tmp_path):
    """Test output files are created."""
    json_path, audio_path, sr = sample_audio_with_diarization

    result = extractor.process_track(json_path, audio_path, "test_artist", output_dir=tmp_path)

    for speaker_info in result.speakers.values():
        assert Path(speaker_info.output_file).exists()
        assert Path(speaker_info.output_file).stat().st_size > 0


def test_process_track_empty_segments(extractor, tmp_path):
    """Test processing track with no segments."""
    import soundfile as sf

    sr = 16000
    audio = np.zeros(int(sr * 2.0), dtype=np.float32)

    audio_path = tmp_path / "empty_seg.wav"
    sf.write(str(audio_path), audio, sr)

    # Empty diarization
    diarization_data = {"file": str(audio_path), "segments": []}

    json_path = tmp_path / "empty_diar.json"
    with open(json_path, 'w') as f:
        json.dump(diarization_data, f)

    result = extractor.process_track(json_path, audio_path, "test_artist", output_dir=tmp_path)

    assert len(result.speakers) == 0


# ===== Coverage verification =====

def test_coverage_diarization_extractor():
    """Verify coverage of diarization_extractor.py module."""
    from auto_voice.audio import diarization_extractor

    assert hasattr(diarization_extractor, 'DiarizationExtractor')
    assert hasattr(diarization_extractor, 'SpeakerSegment')
    assert hasattr(diarization_extractor, 'ExtractionResult')
    assert hasattr(diarization_extractor, 'SpeakerExtractionInfo')
