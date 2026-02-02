"""Tests for diarization_extractor.py - Speaker isolation from diarized audio.

Task 2.1: Test diarization_extractor.py
- Test segment extraction from timestamps
- Verify segment audio quality (no clipping)
- Test multiple speakers (2-3 speakers)
- Test edge cases (overlapping speech, silence)
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
def sample_diarization_json(tmp_path):
    """Create a sample diarization JSON file."""
    diarization_data = {
        "file": "test_audio.wav",
        "segments": [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 2.5, "end": 4.0, "speaker": "SPEAKER_01"},
            {"start": 4.5, "end": 6.5, "speaker": "SPEAKER_00"},
            {"start": 7.0, "end": 8.0, "speaker": "SPEAKER_02"},
        ]
    }
    json_path = tmp_path / "test_diarization.json"
    with open(json_path, 'w') as f:
        json.dump(diarization_data, f)
    return json_path


@pytest.fixture
def sample_audio():
    """Create sample audio for testing (16kHz, 10 seconds)."""
    sr = 16000
    duration = 10.0
    num_samples = int(duration * sr)

    # Create a simple sine wave as test audio
    t = np.linspace(0, duration, num_samples, endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5

    return audio, sr


@pytest.fixture
def extractor(tmp_path):
    """Create a DiarizationExtractor instance."""
    profiles_dir = tmp_path / "profiles"
    training_dir = tmp_path / "training"
    return DiarizationExtractor(
        profiles_dir=profiles_dir,
        training_vocals_dir=training_dir,
    )


class TestSpeakerSegment:
    """Test SpeakerSegment dataclass."""

    def test_segment_creation(self):
        """Test creating a speaker segment."""
        segment = SpeakerSegment(start=0.0, end=2.5, speaker="SPEAKER_00")
        assert segment.start == 0.0
        assert segment.end == 2.5
        assert segment.speaker == "SPEAKER_00"

    def test_segment_duration(self):
        """Test segment duration calculation."""
        segment = SpeakerSegment(start=1.5, end=4.7, speaker="SPEAKER_00")
        assert abs(segment.duration - 3.2) < 0.001


class TestDiarizationExtractor:
    """Test DiarizationExtractor main functionality."""

    @pytest.mark.smoke
    def test_init(self, extractor):
        """Test extractor initialization."""
        assert extractor.fade_ms == 10.0
        assert extractor.min_segment_duration == 0.5
        assert extractor.profiles_dir.name == "profiles"

    def test_load_diarization(self, extractor, sample_diarization_json):
        """Test loading diarization JSON."""
        audio_file, segments = extractor.load_diarization(sample_diarization_json)

        assert audio_file == "test_audio.wav"
        assert len(segments) == 4
        assert all(isinstance(s, SpeakerSegment) for s in segments)

        # Check first segment
        assert segments[0].start == 0.0
        assert segments[0].end == 2.0
        assert segments[0].speaker == "SPEAKER_00"

    def test_get_speaker_durations(self, extractor):
        """Test speaker duration calculation."""
        segments = [
            SpeakerSegment(0.0, 2.0, "SPEAKER_00"),  # 2.0s
            SpeakerSegment(2.5, 4.0, "SPEAKER_01"),  # 1.5s
            SpeakerSegment(4.5, 6.5, "SPEAKER_00"),  # 2.0s
            SpeakerSegment(7.0, 8.0, "SPEAKER_02"),  # 1.0s
        ]

        durations = extractor.get_speaker_durations(segments)

        assert abs(durations["SPEAKER_00"] - 4.0) < 0.001
        assert abs(durations["SPEAKER_01"] - 1.5) < 0.001
        assert abs(durations["SPEAKER_02"] - 1.0) < 0.001

    def test_get_speaker_durations_min_filter(self, extractor):
        """Test that short segments are filtered out."""
        segments = [
            SpeakerSegment(0.0, 2.0, "SPEAKER_00"),  # 2.0s - kept
            SpeakerSegment(2.0, 2.3, "SPEAKER_01"),  # 0.3s - filtered (< 0.5s)
            SpeakerSegment(3.0, 4.0, "SPEAKER_00"),  # 1.0s - kept
        ]

        durations = extractor.get_speaker_durations(segments)

        assert abs(durations["SPEAKER_00"] - 3.0) < 0.001
        assert "SPEAKER_01" not in durations  # Too short

    def test_identify_primary_speaker(self, extractor):
        """Test primary speaker identification."""
        segments = [
            SpeakerSegment(0.0, 2.0, "SPEAKER_00"),
            SpeakerSegment(2.5, 4.0, "SPEAKER_01"),
            SpeakerSegment(4.5, 6.5, "SPEAKER_00"),
            SpeakerSegment(7.0, 8.0, "SPEAKER_02"),
        ]

        primary = extractor.identify_primary_speaker(segments)
        assert primary == "SPEAKER_00"  # Longest total duration

    def test_identify_primary_speaker_empty(self, extractor):
        """Test primary speaker with no segments."""
        primary = extractor.identify_primary_speaker([])
        assert primary is None

    def test_extract_speaker_track_basic(self, extractor, sample_audio):
        """Test basic speaker track extraction."""
        audio, sr = sample_audio
        segments = [
            SpeakerSegment(0.0, 2.0, "SPEAKER_00"),
            SpeakerSegment(4.0, 6.0, "SPEAKER_01"),
        ]

        # Extract SPEAKER_00
        output = extractor.extract_speaker_track(audio, sr, segments, "SPEAKER_00")

        assert len(output) == len(audio)
        assert output.dtype == audio.dtype

        # Check that SPEAKER_00 segment is non-zero
        start_sample = 0
        end_sample = int(2.0 * sr)
        assert np.any(output[start_sample:end_sample] != 0)

        # Check that SPEAKER_01 segment is zero (silenced)
        start_sample = int(4.0 * sr)
        end_sample = int(6.0 * sr)
        assert np.all(output[start_sample:end_sample] == 0)

    def test_extract_speaker_track_no_clipping(self, extractor, sample_audio):
        """Test that extraction doesn't cause clipping."""
        audio, sr = sample_audio
        segments = [SpeakerSegment(0.0, 5.0, "SPEAKER_00")]

        output = extractor.extract_speaker_track(audio, sr, segments, "SPEAKER_00")

        # Check no clipping (values should stay within [-1, 1])
        assert np.all(output >= -1.0)
        assert np.all(output <= 1.0)

    def test_extract_speaker_track_fade(self, extractor):
        """Test that fade in/out is applied at segment boundaries."""
        sr = 16000
        duration = 10.0
        audio = np.ones(int(duration * sr), dtype=np.float32) * 0.5

        segments = [SpeakerSegment(2.0, 5.0, "SPEAKER_00")]

        output = extractor.extract_speaker_track(audio, sr, segments, "SPEAKER_00")

        fade_samples = int(extractor.fade_ms * sr / 1000)
        start_sample = int(2.0 * sr)

        # Check fade in at start
        assert output[start_sample] < 0.1  # Starts at ~0
        assert output[start_sample + fade_samples] > 0.4  # Ends at full volume

        # Check fade out at end
        end_sample = int(5.0 * sr)
        assert output[end_sample - 1] < 0.1  # Ends at ~0
        assert output[end_sample - fade_samples - 1] > 0.4  # Was at full volume

    def test_extract_speaker_track_multiple_segments(self, extractor, sample_audio):
        """Test extraction with multiple segments for same speaker."""
        audio, sr = sample_audio
        segments = [
            SpeakerSegment(0.0, 2.0, "SPEAKER_00"),
            SpeakerSegment(5.0, 7.0, "SPEAKER_00"),
        ]

        output = extractor.extract_speaker_track(audio, sr, segments, "SPEAKER_00")

        # Both segments should be non-zero
        assert np.any(output[0:int(2.0*sr)] != 0)
        assert np.any(output[int(5.0*sr):int(7.0*sr)] != 0)

        # Gap should be zero
        assert np.all(output[int(2.5*sr):int(4.5*sr)] == 0)

    def test_extract_speaker_track_edge_case_zero_duration(self, extractor, sample_audio):
        """Test extraction with zero-duration segment."""
        audio, sr = sample_audio
        segments = [SpeakerSegment(2.0, 2.0, "SPEAKER_00")]  # Zero duration

        output = extractor.extract_speaker_track(audio, sr, segments, "SPEAKER_00")

        # Should be all zeros (no valid segments)
        assert np.all(output == 0)

    def test_extract_speaker_track_out_of_bounds(self, extractor, sample_audio):
        """Test extraction with out-of-bounds timestamps."""
        audio, sr = sample_audio
        duration = len(audio) / sr

        # Segment extends beyond audio
        segments = [SpeakerSegment(duration - 1.0, duration + 5.0, "SPEAKER_00")]

        # Should not crash, just clamp to valid range
        output = extractor.extract_speaker_track(audio, sr, segments, "SPEAKER_00")

        assert len(output) == len(audio)
        # Last second should be non-zero
        assert np.any(output[int((duration-1.0)*sr):] != 0)

    def test_get_or_create_profile_new(self, extractor):
        """Test creating a new profile."""
        profile_id = extractor.get_or_create_profile(
            artist_name="test_artist",
            speaker_id="SPEAKER_00",
            is_primary=True,
        )

        assert profile_id is not None
        assert len(profile_id) == 36  # UUID format

        # Check mapping file was created
        mapping_file = extractor.profiles_dir / "test_artist" / "speaker_profiles.json"
        assert mapping_file.exists()

        with open(mapping_file) as f:
            mappings = json.load(f)

        assert "SPEAKER_00" in mappings
        assert mappings["SPEAKER_00"]["profile_id"] == profile_id
        assert mappings["SPEAKER_00"]["is_primary"] is True

    def test_get_or_create_profile_existing(self, extractor):
        """Test retrieving an existing profile."""
        # Create first
        profile_id1 = extractor.get_or_create_profile(
            artist_name="test_artist",
            speaker_id="SPEAKER_00",
            is_primary=True,
        )

        # Get again - should return same ID
        profile_id2 = extractor.get_or_create_profile(
            artist_name="test_artist",
            speaker_id="SPEAKER_00",
            is_primary=True,
        )

        assert profile_id1 == profile_id2

    def test_process_track_multiple_speakers(self, extractor, tmp_path, sample_audio):
        """Test processing a track with multiple speakers."""
        import soundfile as sf

        audio, sr = sample_audio

        # Create audio file
        audio_path = tmp_path / "test_vocals.wav"
        sf.write(str(audio_path), audio, sr)

        # Create diarization JSON
        diarization_data = {
            "file": str(audio_path),
            "segments": [
                {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
                {"start": 3.5, "end": 6.0, "speaker": "SPEAKER_01"},
                {"start": 6.5, "end": 9.0, "speaker": "SPEAKER_00"},
            ]
        }
        json_path = tmp_path / "test_diarization.json"
        with open(json_path, 'w') as f:
            json.dump(diarization_data, f)

        # Process
        result = extractor.process_track(
            diarization_json=json_path,
            audio_path=audio_path,
            artist_name="test_artist",
            output_dir=tmp_path / "output",
        )

        assert isinstance(result, ExtractionResult)
        assert len(result.speakers) == 2  # SPEAKER_00 and SPEAKER_01
        assert "SPEAKER_00" in result.speakers
        assert "SPEAKER_01" in result.speakers

        # Check SPEAKER_00 info
        speaker00_info = result.speakers["SPEAKER_00"]
        assert speaker00_info.is_primary is True  # Longest duration
        assert speaker00_info.segment_count == 2
        assert Path(speaker00_info.output_file).exists()

        # Check SPEAKER_01 info
        speaker01_info = result.speakers["SPEAKER_01"]
        assert speaker01_info.is_primary is False
        assert speaker01_info.segment_count == 1

    def test_process_track_empty_segments(self, extractor, tmp_path, sample_audio):
        """Test processing with no valid segments."""
        import soundfile as sf

        audio, sr = sample_audio
        audio_path = tmp_path / "test_vocals.wav"
        sf.write(str(audio_path), audio, sr)

        # Empty segments
        diarization_data = {
            "file": str(audio_path),
            "segments": []
        }
        json_path = tmp_path / "test_diarization.json"
        with open(json_path, 'w') as f:
            json.dump(diarization_data, f)

        result = extractor.process_track(
            diarization_json=json_path,
            audio_path=audio_path,
            artist_name="test_artist",
        )

        assert result.total_duration == 0
        assert len(result.speakers) == 0

    def test_process_track_short_segments_filtered(self, extractor, tmp_path, sample_audio):
        """Test that segments below min_segment_duration are filtered."""
        import soundfile as sf

        audio, sr = sample_audio
        audio_path = tmp_path / "test_vocals.wav"
        sf.write(str(audio_path), audio, sr)

        # Include a very short segment
        diarization_data = {
            "file": str(audio_path),
            "segments": [
                {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},  # 2.0s - kept
                {"start": 2.0, "end": 2.2, "speaker": "SPEAKER_01"},  # 0.2s - filtered
                {"start": 3.0, "end": 5.0, "speaker": "SPEAKER_00"},  # 2.0s - kept
            ]
        }
        json_path = tmp_path / "test_diarization.json"
        with open(json_path, 'w') as f:
            json.dump(diarization_data, f)

        result = extractor.process_track(
            diarization_json=json_path,
            audio_path=audio_path,
            artist_name="test_artist",
        )

        # Should only have SPEAKER_00 (SPEAKER_01 segments too short)
        assert len(result.speakers) == 1
        assert "SPEAKER_00" in result.speakers
        assert "SPEAKER_01" not in result.speakers


@pytest.mark.integration
class TestDiarizationExtractorIntegration:
    """Integration tests for full extraction workflow."""

    def test_full_extraction_workflow(self, tmp_path):
        """Test complete extraction from diarization to isolated tracks."""
        from tests.fixtures.multi_speaker_fixtures import create_synthetic_multi_speaker

        # Create synthetic multi-speaker audio
        audio_path = tmp_path / "multi_speaker.wav"
        fixture = create_synthetic_multi_speaker(
            str(audio_path),
            durations=[
                ("SPEAKER_00", 2.0),
                ("SPEAKER_01", 1.5),
                ("SPEAKER_00", 2.5),
            ],
            sample_rate=16000,
        )

        # Create diarization JSON from fixture
        diarization_data = {
            "file": str(audio_path),
            "segments": [
                {"start": s.start, "end": s.end, "speaker": s.speaker_id}
                for s in fixture.speakers
            ]
        }
        json_path = tmp_path / "diarization.json"
        with open(json_path, 'w') as f:
            json.dump(diarization_data, f)

        # Extract
        extractor = DiarizationExtractor(
            profiles_dir=tmp_path / "profiles",
            training_vocals_dir=tmp_path / "training",
        )

        result = extractor.process_track(
            diarization_json=json_path,
            audio_path=audio_path,
            artist_name="test_artist",
        )

        # Verify results
        assert isinstance(result, ExtractionResult)
        assert len(result.speakers) == 2

        # Check durations match ground truth
        speaker00_info = result.speakers["SPEAKER_00"]
        assert abs(speaker00_info.speaker_duration - 4.5) < 0.1  # 2.0 + 2.5

        speaker01_info = result.speakers["SPEAKER_01"]
        assert abs(speaker01_info.speaker_duration - 1.5) < 0.1

        # Verify output files exist and have correct duration
        import soundfile as sf

        for speaker_info in result.speakers.values():
            output_file = Path(speaker_info.output_file)
            assert output_file.exists()

            # Load and check duration
            audio, sr = sf.read(str(output_file))
            file_duration = len(audio) / sr
            assert abs(file_duration - result.total_duration) < 0.1
