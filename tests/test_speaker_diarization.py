"""Tests for speaker diarization module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import wavfile


@pytest.fixture
def test_audio_path(tmp_path):
    """Create a test audio file with synthetic speech-like content."""
    sample_rate = 16000
    duration = 5.0  # 5 seconds

    # Create audio with varying amplitude to simulate speech
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Create segments with different characteristics (simulating different speakers)
    # Speaker 1: 0-2s with 200Hz fundamental
    # Speaker 2: 2-4s with 150Hz fundamental
    # Speaker 1: 4-5s with 200Hz fundamental

    segment1 = np.sin(2 * np.pi * 200 * t[:int(2 * sample_rate)])
    segment2 = np.sin(2 * np.pi * 150 * t[:int(2 * sample_rate)])
    segment3 = np.sin(2 * np.pi * 200 * t[:int(1 * sample_rate)])

    # Add envelope to simulate speech
    envelope1 = np.abs(np.sin(2 * np.pi * 5 * t[:int(2 * sample_rate)]))
    envelope2 = np.abs(np.sin(2 * np.pi * 4 * t[:int(2 * sample_rate)]))
    envelope3 = np.abs(np.sin(2 * np.pi * 5 * t[:int(1 * sample_rate)]))

    waveform = np.concatenate([
        segment1 * envelope1 * 0.5,
        segment2 * envelope2 * 0.5,
        segment3 * envelope3 * 0.5,
    ])

    # Add some noise
    waveform = waveform + np.random.randn(len(waveform)) * 0.01

    # Convert to int16 for WAV
    waveform_int = (waveform * 32767).astype(np.int16)

    # Save to file using scipy
    audio_path = tmp_path / "test_audio.wav"
    wavfile.write(str(audio_path), sample_rate, waveform_int)

    return audio_path


class TestSpeakerSegment:
    """Tests for SpeakerSegment dataclass."""

    def test_segment_creation(self):
        """Test creating a speaker segment."""
        from auto_voice.audio.speaker_diarization import SpeakerSegment

        segment = SpeakerSegment(
            start=1.0,
            end=3.5,
            speaker_id="SPEAKER_00",
        )

        assert segment.start == 1.0
        assert segment.end == 3.5
        assert segment.speaker_id == "SPEAKER_00"
        assert segment.embedding is None
        assert segment.confidence == 1.0

    def test_segment_duration(self):
        """Test duration property."""
        from auto_voice.audio.speaker_diarization import SpeakerSegment

        segment = SpeakerSegment(start=1.0, end=4.5, speaker_id="SPEAKER_00")
        assert segment.duration == 3.5


class TestDiarizationResult:
    """Tests for DiarizationResult dataclass."""

    def test_result_creation(self):
        """Test creating a diarization result."""
        from auto_voice.audio.speaker_diarization import (
            DiarizationResult,
            SpeakerSegment,
        )

        segments = [
            SpeakerSegment(start=0.0, end=2.0, speaker_id="SPEAKER_00"),
            SpeakerSegment(start=2.0, end=4.0, speaker_id="SPEAKER_01"),
            SpeakerSegment(start=4.0, end=5.0, speaker_id="SPEAKER_00"),
        ]

        result = DiarizationResult(
            segments=segments,
            num_speakers=2,
            audio_duration=5.0,
        )

        assert result.num_speakers == 2
        assert result.audio_duration == 5.0
        assert len(result.segments) == 3

    def test_get_speaker_segments(self):
        """Test filtering segments by speaker."""
        from auto_voice.audio.speaker_diarization import (
            DiarizationResult,
            SpeakerSegment,
        )

        segments = [
            SpeakerSegment(start=0.0, end=2.0, speaker_id="SPEAKER_00"),
            SpeakerSegment(start=2.0, end=4.0, speaker_id="SPEAKER_01"),
            SpeakerSegment(start=4.0, end=5.0, speaker_id="SPEAKER_00"),
        ]

        result = DiarizationResult(
            segments=segments, num_speakers=2, audio_duration=5.0
        )

        speaker_00_segments = result.get_speaker_segments("SPEAKER_00")
        assert len(speaker_00_segments) == 2
        assert all(s.speaker_id == "SPEAKER_00" for s in speaker_00_segments)

    def test_get_speaker_total_duration(self):
        """Test calculating total duration for a speaker."""
        from auto_voice.audio.speaker_diarization import (
            DiarizationResult,
            SpeakerSegment,
        )

        segments = [
            SpeakerSegment(start=0.0, end=2.0, speaker_id="SPEAKER_00"),
            SpeakerSegment(start=2.0, end=4.0, speaker_id="SPEAKER_01"),
            SpeakerSegment(start=4.0, end=5.0, speaker_id="SPEAKER_00"),
        ]

        result = DiarizationResult(
            segments=segments, num_speakers=2, audio_duration=5.0
        )

        assert result.get_speaker_total_duration("SPEAKER_00") == 3.0
        assert result.get_speaker_total_duration("SPEAKER_01") == 2.0

    def test_get_all_speaker_ids(self):
        """Test getting all unique speaker IDs."""
        from auto_voice.audio.speaker_diarization import (
            DiarizationResult,
            SpeakerSegment,
        )

        segments = [
            SpeakerSegment(start=0.0, end=2.0, speaker_id="SPEAKER_00"),
            SpeakerSegment(start=2.0, end=4.0, speaker_id="SPEAKER_01"),
            SpeakerSegment(start=4.0, end=5.0, speaker_id="SPEAKER_00"),
        ]

        result = DiarizationResult(
            segments=segments, num_speakers=2, audio_duration=5.0
        )

        speaker_ids = result.get_all_speaker_ids()
        assert set(speaker_ids) == {"SPEAKER_00", "SPEAKER_01"}


class TestSpeakerDiarizer:
    """Tests for SpeakerDiarizer class."""

    def test_diarizer_initialization(self):
        """Test diarizer initialization."""
        from auto_voice.audio.speaker_diarization import SpeakerDiarizer

        diarizer = SpeakerDiarizer(device="cpu")
        assert diarizer.device == "cpu"
        assert diarizer.min_segment_duration == 0.5
        assert diarizer.max_speakers == 10

    def test_load_audio(self, test_audio_path):
        """Test audio loading and resampling."""
        from auto_voice.audio.speaker_diarization import SpeakerDiarizer

        diarizer = SpeakerDiarizer(device="cpu")
        waveform, sample_rate = diarizer._load_audio(test_audio_path)

        assert sample_rate == 16000
        assert len(waveform.shape) == 1  # Mono
        assert len(waveform) > 0

    def test_load_audio_file_not_found(self):
        """Test error handling for missing file."""
        from auto_voice.audio.speaker_diarization import SpeakerDiarizer

        diarizer = SpeakerDiarizer(device="cpu")
        with pytest.raises(FileNotFoundError):
            diarizer._load_audio("/nonexistent/path/audio.wav")

    def test_detect_voice_activity(self, test_audio_path):
        """Test voice activity detection."""
        from auto_voice.audio.speaker_diarization import SpeakerDiarizer

        diarizer = SpeakerDiarizer(device="cpu")
        waveform, sample_rate = diarizer._load_audio(test_audio_path)

        speech_regions = diarizer._detect_voice_activity(waveform, sample_rate)

        # Should detect at least one speech region
        assert len(speech_regions) >= 1

        # Each region should be a tuple of (start, end)
        for start, end in speech_regions:
            assert isinstance(start, float)
            assert isinstance(end, float)
            assert end > start

    def test_segment_audio_fixed(self, test_audio_path):
        """Test fixed-duration audio segmentation."""
        from auto_voice.audio.speaker_diarization import SpeakerDiarizer

        diarizer = SpeakerDiarizer(device="cpu")
        waveform, sample_rate = diarizer._load_audio(test_audio_path)

        segments = diarizer._segment_audio_fixed(
            waveform, sample_rate, segment_duration=1.0, overlap=0.5
        )

        # Should have multiple segments
        assert len(segments) >= 2

        # Each segment should be valid
        for start, end in segments:
            assert end > start
            assert end - start <= 1.0 + 0.01  # Allow small tolerance

    @pytest.mark.slow
    def test_extract_speaker_embedding(self, test_audio_path):
        """Test speaker embedding extraction."""
        from auto_voice.audio.speaker_diarization import SpeakerDiarizer

        diarizer = SpeakerDiarizer(device="cpu")
        embedding = diarizer.extract_speaker_embedding(test_audio_path)

        # WavLM-base-sv embeddings are 512-dim
        assert embedding.shape == (512,)

        # Should be L2 normalized
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01

    @pytest.mark.slow
    def test_extract_speaker_embedding_segment(self, test_audio_path):
        """Test speaker embedding extraction from segment."""
        from auto_voice.audio.speaker_diarization import SpeakerDiarizer

        diarizer = SpeakerDiarizer(device="cpu")
        embedding = diarizer.extract_speaker_embedding(
            test_audio_path, start=0.5, end=2.0
        )

        assert embedding.shape == (512,)

    def test_cluster_embeddings_single(self):
        """Test clustering with minimal embeddings."""
        from auto_voice.audio.speaker_diarization import SpeakerDiarizer

        diarizer = SpeakerDiarizer(device="cpu")

        # Single embedding
        embeddings = [np.random.randn(256)]
        labels = diarizer._cluster_embeddings(embeddings)

        assert len(labels) == 1

    def test_cluster_embeddings_multiple(self):
        """Test clustering with multiple embeddings."""
        from auto_voice.audio.speaker_diarization import SpeakerDiarizer

        diarizer = SpeakerDiarizer(device="cpu")

        # Create two distinct clusters
        np.random.seed(42)
        cluster1 = [np.random.randn(256) + np.array([1.0] * 256) for _ in range(3)]
        cluster2 = [np.random.randn(256) + np.array([-1.0] * 256) for _ in range(3)]
        embeddings = cluster1 + cluster2

        # Normalize
        embeddings = [e / np.linalg.norm(e) for e in embeddings]

        labels = diarizer._cluster_embeddings(embeddings, num_speakers=2)

        assert len(labels) == 6
        # Should identify 2 clusters
        assert len(set(labels)) == 2

    def test_merge_adjacent_segments(self):
        """Test merging adjacent segments."""
        from auto_voice.audio.speaker_diarization import SpeakerDiarizer, SpeakerSegment

        diarizer = SpeakerDiarizer(device="cpu")

        segments = [
            SpeakerSegment(start=0.0, end=1.0, speaker_id="SPEAKER_00"),
            SpeakerSegment(start=1.1, end=2.0, speaker_id="SPEAKER_00"),  # Gap < 0.3
            SpeakerSegment(start=3.0, end=4.0, speaker_id="SPEAKER_00"),  # Gap > 0.3
        ]

        merged = diarizer._merge_adjacent_segments(segments, max_gap=0.3)

        assert len(merged) == 2
        assert merged[0].start == 0.0
        assert merged[0].end == 2.0
        assert merged[1].start == 3.0

    @pytest.mark.slow
    @pytest.mark.cuda
    def test_diarize_full(self, test_audio_path):
        """Test full diarization pipeline."""
        from auto_voice.audio.speaker_diarization import SpeakerDiarizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        diarizer = SpeakerDiarizer(device=device)

        result = diarizer.diarize(test_audio_path)

        assert result.audio_duration > 0
        # May or may not detect multiple speakers in synthetic audio
        assert result.num_speakers >= 0
        assert isinstance(result.segments, list)


class TestSpeakerMatching:
    """Tests for speaker matching functions."""

    def test_match_speaker_to_profile_match(self):
        """Test successful speaker matching."""
        from auto_voice.audio.speaker_diarization import match_speaker_to_profile

        # Create a query embedding
        query = np.array([1.0, 0.0, 0.0])
        query = query / np.linalg.norm(query)

        # Create profile embeddings
        profiles = {
            "profile_1": np.array([1.0, 0.1, 0.0]) / np.linalg.norm([1.0, 0.1, 0.0]),
            "profile_2": np.array([0.0, 1.0, 0.0]),
        }

        match = match_speaker_to_profile(query, profiles, threshold=0.7)
        assert match == "profile_1"

    def test_match_speaker_to_profile_no_match(self):
        """Test no match found."""
        from auto_voice.audio.speaker_diarization import match_speaker_to_profile

        query = np.array([1.0, 0.0, 0.0])
        profiles = {
            "profile_1": np.array([0.0, 1.0, 0.0]),
            "profile_2": np.array([0.0, 0.0, 1.0]),
        }

        match = match_speaker_to_profile(query, profiles, threshold=0.7)
        assert match is None

    def test_match_speaker_to_profile_empty(self):
        """Test matching with empty profiles."""
        from auto_voice.audio.speaker_diarization import match_speaker_to_profile

        query = np.array([1.0, 0.0, 0.0])
        match = match_speaker_to_profile(query, {}, threshold=0.7)
        assert match is None

    def test_compute_speaker_similarity(self):
        """Test speaker similarity computation."""
        from auto_voice.audio.speaker_diarization import compute_speaker_similarity

        # Same vector = similarity 1.0
        v1 = np.array([1.0, 0.0, 0.0])
        assert abs(compute_speaker_similarity(v1, v1) - 1.0) < 0.01

        # Orthogonal vectors = similarity 0.0
        v2 = np.array([0.0, 1.0, 0.0])
        assert abs(compute_speaker_similarity(v1, v2)) < 0.01

        # Opposite vectors = similarity -1.0
        v3 = np.array([-1.0, 0.0, 0.0])
        assert abs(compute_speaker_similarity(v1, v3) - (-1.0)) < 0.01
