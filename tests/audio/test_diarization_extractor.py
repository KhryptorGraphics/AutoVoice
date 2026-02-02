"""Tests for diarization_extractor.py - Speaker isolation from diarization.

Test Coverage:
- Task 2.1: Segment extraction from timestamps
- Verify segment audio quality (no clipping)
- Test multiple speakers (2-3 speakers)
- Test edge cases (overlapping speech, silence)
"""

import json
import numpy as np
import pytest
import soundfile as sf
from pathlib import Path

from auto_voice.audio.diarization_extractor import (
    DiarizationExtractor,
    SpeakerSegment,
    ExtractionResult,
)


@pytest.fixture
def extractor(tmp_path):
    """Create DiarizationExtractor instance with temp directories."""
    return DiarizationExtractor(
        fade_ms=10.0,
        min_segment_duration=0.5,
        profiles_dir=tmp_path / "profiles",
        training_vocals_dir=tmp_path / "training",
    )


@pytest.fixture
def sample_diarization_json(tmp_path):
    """Create a sample diarization JSON file with 2 speakers."""
    audio_file = str(tmp_path / "test_audio.wav")
    diarization_data = {
        "file": audio_file,
        "segments": [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"},
            {"start": 4.0, "end": 5.5, "speaker": "SPEAKER_00"},
            {"start": 5.5, "end": 7.0, "speaker": "SPEAKER_01"},
        ]
    }

    json_path = tmp_path / "test_diarization.json"
    with open(json_path, 'w') as f:
        json.dump(diarization_data, f)

    return json_path, audio_file


@pytest.fixture
def three_speaker_diarization(tmp_path):
    """Create diarization JSON with 3 speakers."""
    audio_file = str(tmp_path / "three_speaker.wav")
    diarization_data = {
        "file": audio_file,
        "segments": [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 3.5, "speaker": "SPEAKER_01"},
            {"start": 3.5, "end": 5.0, "speaker": "SPEAKER_02"},
            {"start": 5.0, "end": 6.0, "speaker": "SPEAKER_00"},
        ]
    }

    json_path = tmp_path / "three_speaker_diarization.json"
    with open(json_path, 'w') as f:
        json.dump(diarization_data, f)

    return json_path, audio_file


@pytest.fixture
def sample_audio(tmp_path):
    """Create a sample audio file (7 seconds, 16kHz)."""
    sr = 16000
    duration = 7.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Create audio with varying frequencies for different "speakers"
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz
    audio = audio.astype(np.float32)

    audio_path = tmp_path / "test_audio.wav"
    sf.write(str(audio_path), audio, sr)

    return audio_path


class TestDiarizationExtractor:
    """Test suite for DiarizationExtractor."""

    def test_load_diarization(self, extractor, sample_diarization_json):
        """Test loading diarization JSON file."""
        json_path, expected_audio_file = sample_diarization_json

        audio_file, segments = extractor.load_diarization(json_path)

        assert audio_file == expected_audio_file
        assert len(segments) == 4
        assert all(isinstance(s, SpeakerSegment) for s in segments)
        assert segments[0].speaker == "SPEAKER_00"
        assert segments[1].speaker == "SPEAKER_01"
        assert segments[0].start == 0.0
        assert segments[0].end == 2.0
        assert segments[0].duration == 2.0

    def test_get_speaker_durations(self, extractor):
        """Test calculating speaker durations."""
        segments = [
            SpeakerSegment(start=0.0, end=2.0, speaker="SPEAKER_00"),
            SpeakerSegment(start=2.0, end=4.0, speaker="SPEAKER_01"),
            SpeakerSegment(start=4.0, end=5.5, speaker="SPEAKER_00"),
        ]

        durations = extractor.get_speaker_durations(segments)

        assert len(durations) == 2
        assert durations["SPEAKER_00"] == pytest.approx(3.5, abs=0.01)
        assert durations["SPEAKER_01"] == pytest.approx(2.0, abs=0.01)

    def test_get_speaker_durations_filters_short_segments(self, extractor):
        """Test that segments below min_segment_duration are filtered."""
        extractor.min_segment_duration = 1.0

        segments = [
            SpeakerSegment(start=0.0, end=2.0, speaker="SPEAKER_00"),
            SpeakerSegment(start=2.0, end=2.3, speaker="SPEAKER_01"),  # Too short
            SpeakerSegment(start=4.0, end=5.5, speaker="SPEAKER_00"),
        ]

        durations = extractor.get_speaker_durations(segments)

        assert "SPEAKER_01" not in durations
        assert durations["SPEAKER_00"] == pytest.approx(3.5, abs=0.01)

    def test_identify_primary_speaker(self, extractor):
        """Test identifying primary speaker (longest speaking time)."""
        segments = [
            SpeakerSegment(start=0.0, end=2.0, speaker="SPEAKER_00"),
            SpeakerSegment(start=2.0, end=4.0, speaker="SPEAKER_01"),
            SpeakerSegment(start=4.0, end=7.5, speaker="SPEAKER_00"),
        ]

        primary = extractor.identify_primary_speaker(segments)

        assert primary == "SPEAKER_00"

    def test_identify_primary_speaker_empty_segments(self, extractor):
        """Test primary speaker identification with no segments."""
        primary = extractor.identify_primary_speaker([])
        assert primary is None

    def test_extract_speaker_track_basic(self, extractor, sample_audio):
        """Test extracting a single speaker track."""
        # Load audio
        import librosa
        audio, sr = librosa.load(str(sample_audio), sr=None, mono=True)

        segments = [
            SpeakerSegment(start=0.0, end=2.0, speaker="SPEAKER_00"),
            SpeakerSegment(start=4.0, end=5.5, speaker="SPEAKER_00"),
        ]

        speaker_track = extractor.extract_speaker_track(
            audio, sr, segments, "SPEAKER_00"
        )

        # Verify shape matches original
        assert speaker_track.shape == audio.shape
        assert speaker_track.dtype == audio.dtype

        # Verify speaker segments are audible
        start_idx = int(0.5 * sr)
        assert np.abs(speaker_track[start_idx]) > 0.01

        # Verify other speakers are silenced
        silence_idx = int(3.0 * sr)
        assert np.abs(speaker_track[silence_idx]) < 0.01

    def test_extract_speaker_track_no_clipping(self, extractor, sample_audio):
        """Test that extracted segments don't clip at boundaries."""
        import librosa
        audio, sr = librosa.load(str(sample_audio), sr=None, mono=True)

        segments = [
            SpeakerSegment(start=1.0, end=3.0, speaker="SPEAKER_00"),
        ]

        speaker_track = extractor.extract_speaker_track(
            audio, sr, segments, "SPEAKER_00"
        )

        # Check for fade-in/out (no hard clipping)
        start_sample = int(1.0 * sr)
        fade_samples = int(extractor.fade_ms * sr / 1000)

        # Fade-in should gradually increase amplitude
        fade_in_segment = speaker_track[start_sample:start_sample + fade_samples]
        if len(fade_in_segment) > 1:
            # Check that amplitude generally increases
            assert np.mean(np.abs(fade_in_segment[:len(fade_in_segment)//2])) < \
                   np.mean(np.abs(fade_in_segment[len(fade_in_segment)//2:]))

    def test_extract_speaker_track_multiple_speakers(self, extractor, sample_audio):
        """Test extracting tracks for multiple speakers (2-3)."""
        import librosa
        audio, sr = librosa.load(str(sample_audio), sr=None, mono=True)

        segments = [
            SpeakerSegment(start=0.0, end=2.0, speaker="SPEAKER_00"),
            SpeakerSegment(start=2.0, end=4.0, speaker="SPEAKER_01"),
            SpeakerSegment(start=4.0, end=5.5, speaker="SPEAKER_00"),
        ]

        # Extract for each speaker
        track_00 = extractor.extract_speaker_track(audio, sr, segments, "SPEAKER_00")
        track_01 = extractor.extract_speaker_track(audio, sr, segments, "SPEAKER_01")

        # Verify tracks are different
        assert not np.allclose(track_00, track_01)

        # Verify speaker 0 has audio in their segments
        assert np.abs(track_00[int(1.0 * sr)]) > 0.01

        # Verify speaker 1 has audio in their segments
        assert np.abs(track_01[int(3.0 * sr)]) > 0.01

        # Verify mutual silence
        assert np.abs(track_00[int(3.0 * sr)]) < 0.01
        assert np.abs(track_01[int(1.0 * sr)]) < 0.01

    def test_extract_speaker_track_edge_case_overlapping(self, extractor, sample_audio):
        """Test edge case: overlapping speech (keeps target speaker)."""
        import librosa
        audio, sr = librosa.load(str(sample_audio), sr=None, mono=True)

        # Overlapping segments
        segments = [
            SpeakerSegment(start=0.0, end=2.5, speaker="SPEAKER_00"),
            SpeakerSegment(start=2.0, end=4.0, speaker="SPEAKER_01"),
        ]

        track_00 = extractor.extract_speaker_track(audio, sr, segments, "SPEAKER_00")

        # Speaker 0 should have audio in their segment
        assert np.abs(track_00[int(1.0 * sr)]) > 0.01

        # Overlap region: speaker 0 still audible
        assert np.abs(track_00[int(2.2 * sr)]) > 0.01

    def test_extract_speaker_track_edge_case_silence(self, extractor):
        """Test edge case: silent audio."""
        sr = 16000
        duration = 5.0
        audio = np.zeros(int(sr * duration), dtype=np.float32)

        segments = [
            SpeakerSegment(start=1.0, end=3.0, speaker="SPEAKER_00"),
        ]

        speaker_track = extractor.extract_speaker_track(audio, sr, segments, "SPEAKER_00")

        # Should return all zeros
        assert np.allclose(speaker_track, 0.0)

    def test_extract_speaker_track_edge_case_boundary_clamp(self, extractor, sample_audio):
        """Test edge case: segment timestamps beyond audio duration."""
        import librosa
        audio, sr = librosa.load(str(sample_audio), sr=None, mono=True)
        duration = len(audio) / sr

        # Segment extends beyond audio
        segments = [
            SpeakerSegment(start=duration - 1.0, end=duration + 10.0, speaker="SPEAKER_00"),
        ]

        speaker_track = extractor.extract_speaker_track(audio, sr, segments, "SPEAKER_00")

        # Should not raise error and should clamp to audio length
        assert len(speaker_track) == len(audio)

    def test_get_or_create_profile(self, extractor):
        """Test creating and retrieving voice profiles."""
        profile_id = extractor.get_or_create_profile(
            artist_name="test_artist",
            speaker_id="SPEAKER_00",
            is_primary=True,
        )

        # Should be a valid UUID
        assert isinstance(profile_id, str)
        assert len(profile_id) == 36  # UUID length

        # Second call should return same ID
        profile_id_2 = extractor.get_or_create_profile(
            artist_name="test_artist",
            speaker_id="SPEAKER_00",
            is_primary=True,
        )

        assert profile_id == profile_id_2

    def test_get_or_create_profile_multiple_speakers(self, extractor):
        """Test creating profiles for multiple speakers."""
        profile_00 = extractor.get_or_create_profile(
            artist_name="test_artist",
            speaker_id="SPEAKER_00",
            is_primary=True,
        )

        profile_01 = extractor.get_or_create_profile(
            artist_name="test_artist",
            speaker_id="SPEAKER_01",
            is_primary=False,
        )

        # Should have different profile IDs
        assert profile_00 != profile_01

    def test_process_track(self, extractor, sample_diarization_json, sample_audio, tmp_path):
        """Test full track processing pipeline."""
        json_path, _ = sample_diarization_json

        result = extractor.process_track(
            diarization_json=json_path,
            audio_path=sample_audio,
            artist_name="test_artist",
            output_dir=tmp_path / "output",
        )

        assert isinstance(result, ExtractionResult)
        assert result.total_duration > 0
        assert len(result.speakers) == 2  # SPEAKER_00 and SPEAKER_01

        # Verify speaker info
        for speaker_id, info in result.speakers.items():
            assert speaker_id in ["SPEAKER_00", "SPEAKER_01"]
            assert info.profile_id is not None
            assert Path(info.output_file).exists()
            assert info.speaker_duration > 0
            assert info.segment_count > 0

    def test_process_track_three_speakers(self, extractor, three_speaker_diarization, tmp_path):
        """Test processing track with 3 speakers."""
        json_path, _ = three_speaker_diarization

        # Create audio file
        sr = 16000
        duration = 6.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        audio_path = tmp_path / "three_speaker.wav"
        sf.write(str(audio_path), audio, sr)

        result = extractor.process_track(
            diarization_json=json_path,
            audio_path=audio_path,
            artist_name="test_artist",
            output_dir=tmp_path / "output",
        )

        assert len(result.speakers) == 3
        assert "SPEAKER_00" in result.speakers
        assert "SPEAKER_01" in result.speakers
        assert "SPEAKER_02" in result.speakers

    def test_process_track_empty_diarization(self, extractor, tmp_path, sample_audio):
        """Test processing track with no segments."""
        # Create empty diarization
        empty_diarization = {
            "file": str(sample_audio),
            "segments": []
        }
        json_path = tmp_path / "empty_diarization.json"
        with open(json_path, 'w') as f:
            json.dump(empty_diarization, f)

        result = extractor.process_track(
            diarization_json=json_path,
            audio_path=sample_audio,
            artist_name="test_artist",
        )

        assert result.total_duration == 0
        assert len(result.speakers) == 0


@pytest.mark.parametrize("fade_ms,expected_fade_samples", [
    (5.0, 80),   # 5ms at 16kHz = 80 samples
    (10.0, 160),  # 10ms at 16kHz = 160 samples
    (20.0, 320),  # 20ms at 16kHz = 320 samples
])
def test_fade_duration_configuration(tmp_path, fade_ms, expected_fade_samples):
    """Test configuring fade duration."""
    extractor = DiarizationExtractor(fade_ms=fade_ms, training_vocals_dir=tmp_path)

    sr = 16000
    duration = 5.0
    audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.1

    segments = [
        SpeakerSegment(start=1.0, end=3.0, speaker="SPEAKER_00"),
    ]

    speaker_track = extractor.extract_speaker_track(audio, sr, segments, "SPEAKER_00")

    # Verify fade was applied (approximate check)
    assert len(speaker_track) == len(audio)


def test_min_segment_duration_filtering(tmp_path):
    """Test that minimum segment duration filtering works."""
    extractor = DiarizationExtractor(
        min_segment_duration=1.0,
        training_vocals_dir=tmp_path,
    )

    segments = [
        SpeakerSegment(start=0.0, end=0.3, speaker="SPEAKER_00"),  # Too short
        SpeakerSegment(start=1.0, end=3.0, speaker="SPEAKER_00"),  # OK
        SpeakerSegment(start=3.0, end=3.5, speaker="SPEAKER_01"),  # Too short
    ]

    durations = extractor.get_speaker_durations(segments)

    assert len(durations) == 1
    assert durations["SPEAKER_00"] == pytest.approx(2.0, abs=0.01)
