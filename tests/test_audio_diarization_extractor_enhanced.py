"""Enhanced tests for diarization_extractor.py - Comprehensive TDD coverage.

This file adds missing test coverage to reach 90%:
- Multi-speaker scenarios (3, 5, 7+ speakers)
- process_artist batch workflows
- Error handling and edge cases
- CLI interface and statistics
- Performance with large track sets

Coverage Target: 64% → 90% (70 uncovered lines → <30 lines)
"""
import json
import numpy as np
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from auto_voice.audio.diarization_extractor import (
    DiarizationExtractor,
    SpeakerSegment,
    ExtractionResult,
    SpeakerExtractionInfo,
    run_extraction,
)


@pytest.fixture
def extractor_with_temp_dirs(tmp_path):
    """Create extractor with isolated temporary directories."""
    profiles_dir = tmp_path / "profiles"
    training_dir = tmp_path / "training"
    return DiarizationExtractor(
        profiles_dir=profiles_dir,
        training_vocals_dir=training_dir,
        fade_ms=10.0,
        min_segment_duration=0.5,
    )


@pytest.fixture
def create_test_audio_file(tmp_path):
    """Factory fixture to create test audio files."""
    import soundfile as sf

    def _create(filename, duration=10.0, sr=16000, frequency=440):
        """Create a synthetic audio file."""
        num_samples = int(duration * sr)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32) * 0.5

        audio_path = tmp_path / filename
        sf.write(str(audio_path), audio, sr)
        return audio_path

    return _create


@pytest.fixture
def create_diarization_json(tmp_path):
    """Factory fixture to create diarization JSON files."""

    def _create(filename, audio_file, segments):
        """Create a diarization JSON file.

        Args:
            filename: JSON filename
            audio_file: Path to audio file
            segments: List of (start, end, speaker_id) tuples
        """
        diarization_data = {
            "file": str(audio_file),
            "segments": [
                {"start": start, "end": end, "speaker": speaker}
                for start, end, speaker in segments
            ]
        }

        json_path = tmp_path / filename
        with open(json_path, 'w') as f:
            json.dump(diarization_data, f)

        return json_path

    return _create


class TestMultiSpeakerScenarios:
    """Test diarization with 3, 5, and 7+ speakers."""

    def test_three_speakers(self, extractor_with_temp_dirs, create_test_audio_file, create_diarization_json):
        """Test extraction with 3 speakers."""
        # Create audio
        audio_path = create_test_audio_file("three_speakers.wav", duration=12.0)

        # Create diarization with 3 speakers
        segments = [
            (0.0, 3.0, "SPEAKER_00"),  # Primary (4.5s total)
            (3.5, 5.5, "SPEAKER_01"),  # 2.0s
            (6.0, 7.5, "SPEAKER_02"),  # 1.5s
            (8.0, 9.5, "SPEAKER_00"),  # More primary
        ]
        json_path = create_diarization_json("three_speakers.json", audio_path, segments)

        # Process
        result = extractor_with_temp_dirs.process_track(
            json_path, audio_path, "test_artist"
        )

        # Verify
        assert len(result.speakers) == 3
        assert result.speakers["SPEAKER_00"].is_primary is True
        assert result.speakers["SPEAKER_01"].is_primary is False
        assert result.speakers["SPEAKER_02"].is_primary is False

        # Check durations
        assert abs(result.speakers["SPEAKER_00"].speaker_duration - 4.5) < 0.1
        assert abs(result.speakers["SPEAKER_01"].speaker_duration - 2.0) < 0.1
        assert abs(result.speakers["SPEAKER_02"].speaker_duration - 1.5) < 0.1

    def test_five_speakers(self, extractor_with_temp_dirs, create_test_audio_file, create_diarization_json):
        """Test extraction with 5 speakers."""
        audio_path = create_test_audio_file("five_speakers.wav", duration=20.0)

        # 5 speakers with varying durations
        segments = [
            (0.0, 4.0, "SPEAKER_00"),   # Primary: 6.0s total
            (4.5, 6.0, "SPEAKER_01"),   # 1.5s
            (6.5, 8.0, "SPEAKER_02"),   # 1.5s
            (8.5, 9.5, "SPEAKER_03"),   # 1.0s
            (10.0, 11.0, "SPEAKER_04"),  # 1.0s
            (11.5, 13.5, "SPEAKER_00"),  # More primary
        ]
        json_path = create_diarization_json("five_speakers.json", audio_path, segments)

        result = extractor_with_temp_dirs.process_track(
            json_path, audio_path, "test_artist"
        )

        assert len(result.speakers) == 5
        assert result.speakers["SPEAKER_00"].is_primary is True

        # All other speakers should be non-primary
        for i in range(1, 5):
            speaker_id = f"SPEAKER_0{i}"
            assert result.speakers[speaker_id].is_primary is False

    def test_many_speakers_seven_plus(self, extractor_with_temp_dirs, create_test_audio_file, create_diarization_json):
        """Test extraction with 7+ speakers (complex scenario)."""
        audio_path = create_test_audio_file("many_speakers.wav", duration=30.0)

        # 8 speakers with various durations
        segments = []
        for i in range(8):
            # Primary speaker gets more time
            if i == 0:
                segments.extend([
                    (i * 3.5, i * 3.5 + 2.5, f"SPEAKER_0{i}"),
                    (20.0, 22.5, "SPEAKER_00"),  # Additional segment
                ])
            else:
                segments.append((i * 3.5, i * 3.5 + 1.0, f"SPEAKER_0{i}"))

        json_path = create_diarization_json("many_speakers.json", audio_path, segments)

        result = extractor_with_temp_dirs.process_track(
            json_path, audio_path, "test_artist"
        )

        assert len(result.speakers) == 8
        assert result.speakers["SPEAKER_00"].is_primary is True

        # Verify all output files exist
        for speaker_info in result.speakers.values():
            assert Path(speaker_info.output_file).exists()


class TestProcessArtistBatchWorkflow:
    """Test process_artist for multi-track processing."""

    def test_process_artist_single_track(self, extractor_with_temp_dirs, tmp_path, create_test_audio_file):
        """Test processing a single track for an artist."""
        artist_name = "test_artist"

        # Set up directories
        diarized_dir = tmp_path / "diarized" / artist_name
        separated_dir = tmp_path / "separated" / artist_name
        diarized_dir.mkdir(parents=True)
        separated_dir.mkdir(parents=True)

        # Create audio file
        audio_path = separated_dir / "track_001.wav"
        import soundfile as sf
        sr = 16000
        duration = 10.0
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(duration * sr))).astype(np.float32) * 0.5
        sf.write(str(audio_path), audio, sr)

        # Create diarization JSON
        diarization_data = {
            "file": str(audio_path),
            "segments": [
                {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
                {"start": 5.5, "end": 8.0, "speaker": "SPEAKER_01"},
            ]
        }
        json_path = diarized_dir / "track_001_diarization.json"
        with open(json_path, 'w') as f:
            json.dump(diarization_data, f)

        # Process
        stats = extractor_with_temp_dirs.process_artist(
            artist_name,
            diarization_dir=diarized_dir,
            separated_dir=separated_dir,
        )

        assert stats['total_tracks'] == 1
        assert stats['processed_tracks'] == 1
        assert stats['skipped_tracks'] == 0
        assert len(stats['speakers']) == 2

    def test_process_artist_multiple_tracks(self, extractor_with_temp_dirs, tmp_path):
        """Test processing multiple tracks for an artist."""
        artist_name = "multi_track_artist"

        diarized_dir = tmp_path / "diarized" / artist_name
        separated_dir = tmp_path / "separated" / artist_name
        diarized_dir.mkdir(parents=True)
        separated_dir.mkdir(parents=True)

        import soundfile as sf
        sr = 16000

        # Create 3 tracks
        for track_num in range(1, 4):
            # Create audio
            duration = 8.0
            audio = np.random.randn(int(duration * sr)).astype(np.float32) * 0.1
            audio_path = separated_dir / f"track_00{track_num}.wav"
            sf.write(str(audio_path), audio, sr)

            # Create diarization
            diarization_data = {
                "file": str(audio_path),
                "segments": [
                    {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"},
                    {"start": 4.5, "end": 7.0, "speaker": "SPEAKER_01"},
                ]
            }
            json_path = diarized_dir / f"track_00{track_num}_diarization.json"
            with open(json_path, 'w') as f:
                json.dump(diarization_data, f)

        # Process all tracks
        stats = extractor_with_temp_dirs.process_artist(
            artist_name,
            diarization_dir=diarized_dir,
            separated_dir=separated_dir,
        )

        assert stats['total_tracks'] == 3
        assert stats['processed_tracks'] == 3
        assert stats['skipped_tracks'] == 0
        assert stats['total_duration_minutes'] > 0

        # Both speakers should appear across all tracks
        assert 'SPEAKER_00' in stats['speakers']
        assert 'SPEAKER_01' in stats['speakers']
        assert stats['speakers']['SPEAKER_00']['track_count'] == 3
        assert stats['speakers']['SPEAKER_01']['track_count'] == 3

    def test_process_artist_with_skipped_tracks(self, extractor_with_temp_dirs, tmp_path):
        """Test that missing audio files are skipped."""
        artist_name = "artist_with_missing"

        diarized_dir = tmp_path / "diarized" / artist_name
        separated_dir = tmp_path / "separated" / artist_name
        diarized_dir.mkdir(parents=True)
        separated_dir.mkdir(parents=True)

        # Create diarization without corresponding audio
        diarization_data = {
            "file": "nonexistent.wav",
            "segments": [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]
        }
        json_path = diarized_dir / "missing_track_diarization.json"
        with open(json_path, 'w') as f:
            json.dump(diarization_data, f)

        stats = extractor_with_temp_dirs.process_artist(
            artist_name,
            diarization_dir=diarized_dir,
            separated_dir=separated_dir,
        )

        assert stats['total_tracks'] == 1
        assert stats['processed_tracks'] == 0
        assert stats['skipped_tracks'] == 1

    def test_process_artist_nonexistent_directory(self, extractor_with_temp_dirs):
        """Test error handling when diarization directory doesn't exist."""
        with pytest.raises(FileNotFoundError):
            extractor_with_temp_dirs.process_artist(
                "nonexistent_artist",
                diarization_dir=Path("/nonexistent/path"),
            )


class TestErrorHandlingAndEdgeCases:
    """Test error paths and edge cases."""

    def test_load_diarization_missing_file(self, extractor_with_temp_dirs):
        """Test loading non-existent diarization file."""
        with pytest.raises(FileNotFoundError):
            extractor_with_temp_dirs.load_diarization(Path("/nonexistent.json"))

    def test_load_diarization_invalid_json(self, extractor_with_temp_dirs, tmp_path):
        """Test loading malformed JSON."""
        invalid_json = tmp_path / "invalid.json"
        with open(invalid_json, 'w') as f:
            f.write("{ invalid json }")

        with pytest.raises(json.JSONDecodeError):
            extractor_with_temp_dirs.load_diarization(invalid_json)

    def test_extract_speaker_track_very_short_segment(self, extractor_with_temp_dirs):
        """Test extraction with segment shorter than fade duration."""
        sr = 16000
        audio = np.ones(int(10.0 * sr), dtype=np.float32) * 0.5

        # Segment shorter than 2 * fade_samples
        segments = [SpeakerSegment(0.0, 0.001, "SPEAKER_00")]  # 1ms segment

        output = extractor_with_temp_dirs.extract_speaker_track(
            audio, sr, segments, "SPEAKER_00"
        )

        # Should handle gracefully (may skip fade)
        assert len(output) == len(audio)

    def test_extract_speaker_track_negative_timestamps(self, extractor_with_temp_dirs):
        """Test extraction with negative timestamps (should be clamped)."""
        sr = 16000
        audio = np.ones(int(10.0 * sr), dtype=np.float32) * 0.5

        # Negative start time
        segments = [SpeakerSegment(-1.0, 2.0, "SPEAKER_00")]

        output = extractor_with_temp_dirs.extract_speaker_track(
            audio, sr, segments, "SPEAKER_00"
        )

        # Should clamp to [0, len(audio)]
        assert len(output) == len(audio)
        assert np.any(output[:int(2.0*sr)] != 0)  # First 2 seconds non-zero

    def test_process_track_with_processing_error(self, extractor_with_temp_dirs, tmp_path):
        """Test error recovery in process_artist when a track fails."""
        artist_name = "error_artist"

        diarized_dir = tmp_path / "diarized" / artist_name
        separated_dir = tmp_path / "separated" / artist_name
        diarized_dir.mkdir(parents=True)
        separated_dir.mkdir(parents=True)

        # Create invalid diarization JSON (missing required fields)
        invalid_data = {"file": "test.wav"}  # Missing 'segments'
        json_path = diarized_dir / "invalid_diarization.json"
        with open(json_path, 'w') as f:
            json.dump(invalid_data, f)

        # Should handle error gracefully
        stats = extractor_with_temp_dirs.process_artist(
            artist_name,
            diarization_dir=diarized_dir,
            separated_dir=separated_dir,
        )

        # Track should be skipped due to error
        assert stats['skipped_tracks'] >= 1


class TestCLIAndStatistics:
    """Test CLI interface and statistics functions."""

    def test_run_extraction_default_artists(self, tmp_path):
        """Test run_extraction with default artist list."""
        # Mock the artist directories
        for artist in ['conor_maynard', 'william_singe']:
            diarized_dir = tmp_path / f'data/diarized_youtube/{artist}'
            separated_dir = tmp_path / f'data/separated_youtube/{artist}'
            diarized_dir.mkdir(parents=True)
            separated_dir.mkdir(parents=True)

        # Run with empty directories (no tracks to process)
        with patch('auto_voice.audio.diarization_extractor.Path') as mock_path:
            # Redirect to tmp_path
            def path_constructor(path_str):
                if path_str.startswith('data/'):
                    return tmp_path / path_str
                return Path(path_str)

            mock_path.side_effect = path_constructor

            # This will process but find no tracks
            stats = run_extraction(output_dir=tmp_path / "training")

            assert 'conor_maynard' in stats or 'william_singe' in stats

    def test_run_extraction_custom_artists(self, tmp_path):
        """Test run_extraction with custom artist list."""
        artist_name = "custom_artist"

        # Create directories
        diarized_dir = tmp_path / f'data/diarized_youtube/{artist_name}'
        separated_dir = tmp_path / f'data/separated_youtube/{artist_name}'
        diarized_dir.mkdir(parents=True)
        separated_dir.mkdir(parents=True)

        with patch('auto_voice.audio.diarization_extractor.Path') as mock_path:
            def path_constructor(path_str):
                if path_str.startswith('data/'):
                    return tmp_path / path_str
                return Path(path_str)

            mock_path.side_effect = path_constructor

            stats = run_extraction(
                artists=[artist_name],
                output_dir=tmp_path / "training"
            )

            assert artist_name in stats


class TestProfileManagement:
    """Test voice profile creation and management."""

    def test_get_or_create_profile_primary_naming(self, extractor_with_temp_dirs):
        """Test that primary speaker gets clean name."""
        profile_id = extractor_with_temp_dirs.get_or_create_profile(
            artist_name="test_artist",
            speaker_id="SPEAKER_00",
            is_primary=True,
        )

        # Check mapping file
        mapping_file = extractor_with_temp_dirs.profiles_dir / "test_artist" / "speaker_profiles.json"
        with open(mapping_file) as f:
            mappings = json.load(f)

        assert mappings["SPEAKER_00"]["profile_name"] == "Test Artist"
        assert mappings["SPEAKER_00"]["is_primary"] is True

    def test_get_or_create_profile_featured_naming(self, extractor_with_temp_dirs):
        """Test that featured speakers get speaker ID in name."""
        profile_id = extractor_with_temp_dirs.get_or_create_profile(
            artist_name="test_artist",
            speaker_id="SPEAKER_01",
            is_primary=False,
        )

        mapping_file = extractor_with_temp_dirs.profiles_dir / "test_artist" / "speaker_profiles.json"
        with open(mapping_file) as f:
            mappings = json.load(f)

        assert "SPEAKER_01" in mappings["SPEAKER_01"]["profile_name"]
        assert mappings["SPEAKER_01"]["is_primary"] is False

    def test_profile_persistence_across_tracks(self, extractor_with_temp_dirs):
        """Test that same speaker gets same profile across different tracks."""
        # Create profile for track 1
        profile_id1 = extractor_with_temp_dirs.get_or_create_profile(
            "artist", "SPEAKER_00", True
        )

        # Get profile for track 2 (should be same)
        profile_id2 = extractor_with_temp_dirs.get_or_create_profile(
            "artist", "SPEAKER_00", True
        )

        assert profile_id1 == profile_id2


class TestSegmentFiltering:
    """Test segment filtering by minimum duration."""

    def test_min_segment_duration_filtering(self, extractor_with_temp_dirs):
        """Test that segments below minimum are filtered."""
        segments = [
            SpeakerSegment(0.0, 2.0, "SPEAKER_00"),   # 2.0s - kept
            SpeakerSegment(2.0, 2.3, "SPEAKER_01"),   # 0.3s - filtered
            SpeakerSegment(3.0, 3.2, "SPEAKER_01"),   # 0.2s - filtered
            SpeakerSegment(4.0, 6.0, "SPEAKER_01"),   # 2.0s - kept
        ]

        durations = extractor_with_temp_dirs.get_speaker_durations(segments)

        # SPEAKER_00: 2.0s (kept)
        assert abs(durations["SPEAKER_00"] - 2.0) < 0.01

        # SPEAKER_01: only the 2.0s segment counted
        assert abs(durations["SPEAKER_01"] - 2.0) < 0.01

    def test_custom_min_segment_duration(self, tmp_path):
        """Test custom minimum segment duration."""
        extractor = DiarizationExtractor(
            profiles_dir=tmp_path / "profiles",
            training_vocals_dir=tmp_path / "training",
            min_segment_duration=1.0,  # Require 1.0s minimum
        )

        segments = [
            SpeakerSegment(0.0, 0.5, "SPEAKER_00"),   # 0.5s - filtered
            SpeakerSegment(1.0, 2.5, "SPEAKER_00"),   # 1.5s - kept
            SpeakerSegment(3.0, 3.8, "SPEAKER_01"),   # 0.8s - filtered
        ]

        durations = extractor.get_speaker_durations(segments)

        assert "SPEAKER_00" in durations
        assert abs(durations["SPEAKER_00"] - 1.5) < 0.01
        assert "SPEAKER_01" not in durations  # All segments too short


class TestFadeConfiguration:
    """Test fade in/out configuration."""

    def test_custom_fade_duration(self, tmp_path):
        """Test custom fade duration."""
        extractor = DiarizationExtractor(
            profiles_dir=tmp_path / "profiles",
            training_vocals_dir=tmp_path / "training",
            fade_ms=50.0,  # 50ms fade
        )

        sr = 16000
        audio = np.ones(int(10.0 * sr), dtype=np.float32) * 0.5
        segments = [SpeakerSegment(2.0, 5.0, "SPEAKER_00")]

        output = extractor.extract_speaker_track(audio, sr, segments, "SPEAKER_00")

        fade_samples = int(50.0 * sr / 1000)
        start_sample = int(2.0 * sr)

        # Check fade is applied
        assert output[start_sample] < 0.1
        assert output[start_sample + fade_samples] > 0.4

    def test_zero_fade(self, tmp_path):
        """Test with zero fade duration."""
        extractor = DiarizationExtractor(
            profiles_dir=tmp_path / "profiles",
            training_vocals_dir=tmp_path / "training",
            fade_ms=0.0,  # No fade
        )

        assert extractor.fade_ms == 0.0
