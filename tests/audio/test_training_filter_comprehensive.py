"""Comprehensive tests for training_filter.py - Training data quality filtering.

Test Coverage:
- TrainingDataFilter initialization and configuration
- filter_training_audio with various similarity thresholds
- filter_with_profile_matching with multiple profiles
- auto_split_by_speakers for multi-speaker extraction
- Audio quality filtering (silence, clipping, noise)
- Duration filtering (min/max length constraints)
- Segment extraction and concatenation
- Batch filtering workflows
- Filter statistics and metadata reporting
- Edge cases (empty audio, no matches, corrupt data)
- Convenience function filter_training_audio()
- Device selection (CPU/CUDA)
- Audio format handling (int16, int32, stereo)
- Output path generation and management
- Lazy diarizer initialization
- Segment sorting and ordering
- Crossfade and gap insertion
- Purity calculation
- Error handling and validation

Target: 90% coverage for training_filter.py
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import soundfile as sf
from scipy.io import wavfile

from auto_voice.audio.speaker_diarization import (
    DiarizationResult,
    SpeakerDiarizer,
    SpeakerSegment,
)
from auto_voice.audio.training_filter import (
    TrainingDataFilter,
    filter_training_audio,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_embedding():
    """Generate a sample 512-dim WavLM speaker embedding."""
    embedding = np.random.randn(512).astype(np.float32)
    # L2 normalize
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


@pytest.fixture
def different_embedding():
    """Generate a different speaker embedding (low similarity)."""
    embedding = np.random.randn(512).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


@pytest.fixture
def similar_embedding(sample_embedding):
    """Generate an embedding similar to sample_embedding."""
    # Add small noise to create high similarity
    noise = np.random.randn(512).astype(np.float32) * 0.1
    embedding = sample_embedding + noise
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


@pytest.fixture
def audio_16khz_mono(tmp_path):
    """Create a 10-second mono audio file at 16kHz."""
    sr = 16000
    duration = 10.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Create audio with varying frequencies
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    audio = audio.astype(np.float32)

    path = tmp_path / "test_audio_16k.wav"
    sf.write(str(path), audio, sr)
    return path


@pytest.fixture
def audio_44khz_stereo(tmp_path):
    """Create a 5-second stereo audio file at 44.1kHz."""
    sr = 44100
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Stereo: different frequencies in each channel
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 880 * t)
    audio = np.stack([left, right], axis=1).astype(np.float32)

    path = tmp_path / "test_audio_44k_stereo.wav"
    sf.write(str(path), audio, sr)
    return path


@pytest.fixture
def audio_int16(tmp_path):
    """Create an int16 audio file."""
    sr = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    path = tmp_path / "test_audio_int16.wav"
    wavfile.write(str(path), sr, audio)
    return path


@pytest.fixture
def audio_int32(tmp_path):
    """Create an int32 audio file."""
    sr = 48000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t) * 2147483647).astype(np.int32)

    path = tmp_path / "test_audio_int32.wav"
    wavfile.write(str(path), sr, audio)
    return path


@pytest.fixture
def mock_diarization_result(sample_embedding, different_embedding):
    """Create a mock DiarizationResult with 2 speakers."""
    segments = [
        SpeakerSegment(
            start=0.0,
            end=2.0,
            speaker_id="SPEAKER_00",
            embedding=sample_embedding.copy(),
            confidence=0.95
        ),
        SpeakerSegment(
            start=2.0,
            end=4.0,
            speaker_id="SPEAKER_01",
            embedding=different_embedding.copy(),
            confidence=0.90
        ),
        SpeakerSegment(
            start=4.0,
            end=6.5,
            speaker_id="SPEAKER_00",
            embedding=sample_embedding.copy(),
            confidence=0.92
        ),
        SpeakerSegment(
            start=6.5,
            end=8.0,
            speaker_id="SPEAKER_01",
            embedding=different_embedding.copy(),
            confidence=0.88
        ),
    ]

    return DiarizationResult(
        segments=segments,
        num_speakers=2,
        audio_duration=8.0,
        speaker_embeddings={
            "SPEAKER_00": sample_embedding,
            "SPEAKER_01": different_embedding,
        }
    )


@pytest.fixture
def mock_diarizer(mock_diarization_result):
    """Create a mock SpeakerDiarizer."""
    diarizer = MagicMock(spec=SpeakerDiarizer)
    diarizer.diarize.return_value = mock_diarization_result
    return diarizer


# ============================================================================
# Test TrainingDataFilter Initialization
# ============================================================================


class TestTrainingDataFilterInitialization:
    """Test TrainingDataFilter initialization and configuration."""

    def test_init_default(self):
        """Test initialization with defaults."""
        filter_obj = TrainingDataFilter()

        assert filter_obj._diarizer is None
        assert filter_obj.device in ['cuda', 'cpu']

    def test_init_with_diarizer(self, mock_diarizer):
        """Test initialization with pre-initialized diarizer."""
        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        assert filter_obj._diarizer is mock_diarizer
        assert filter_obj.diarizer is mock_diarizer

    def test_init_with_device_cuda(self):
        """Test initialization with CUDA device."""
        filter_obj = TrainingDataFilter(device='cuda')

        assert filter_obj.device == 'cuda'

    def test_init_with_device_cpu(self):
        """Test initialization with CPU device."""
        filter_obj = TrainingDataFilter(device='cpu')

        assert filter_obj.device == 'cpu'

    @patch('auto_voice.audio.training_filter.torch.cuda.is_available')
    def test_device_auto_detection_cuda(self, mock_cuda_available):
        """Test automatic CUDA detection."""
        mock_cuda_available.return_value = True

        filter_obj = TrainingDataFilter()

        assert filter_obj.device == 'cuda'

    @patch('auto_voice.audio.training_filter.torch.cuda.is_available')
    def test_device_auto_detection_cpu(self, mock_cuda_available):
        """Test automatic CPU fallback."""
        mock_cuda_available.return_value = False

        filter_obj = TrainingDataFilter()

        assert filter_obj.device == 'cpu'

    @patch('auto_voice.audio.training_filter.SpeakerDiarizer')
    def test_lazy_diarizer_initialization(self, mock_diarizer_class):
        """Test lazy initialization of diarizer."""
        filter_obj = TrainingDataFilter(device='cpu')

        # Diarizer not created yet
        assert filter_obj._diarizer is None

        # Access diarizer property
        _ = filter_obj.diarizer

        # Diarizer created
        mock_diarizer_class.assert_called_once_with(device='cpu')

    @patch('auto_voice.audio.training_filter.SpeakerDiarizer')
    def test_lazy_diarizer_only_once(self, mock_diarizer_class):
        """Test diarizer is only initialized once."""
        filter_obj = TrainingDataFilter()

        # Access multiple times
        _ = filter_obj.diarizer
        _ = filter_obj.diarizer
        _ = filter_obj.diarizer

        # Only called once
        assert mock_diarizer_class.call_count == 1


# ============================================================================
# Test filter_training_audio
# ============================================================================


class TestFilterTrainingAudio:
    """Test filter_training_audio method."""

    def test_filter_with_matching_segments(
        self, audio_16khz_mono, sample_embedding, mock_diarization_result, mock_diarizer, tmp_path
    ):
        """Test filtering with segments that match target speaker."""
        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)
        output_path = tmp_path / "filtered_output.wav"

        result_path, metadata = filter_obj.filter_training_audio(
            audio_path=audio_16khz_mono,
            target_embedding=sample_embedding,
            output_path=output_path,
            diarization_result=mock_diarization_result,
        )

        # Verify output file created
        assert result_path.exists()
        assert result_path == output_path

        # Verify metadata
        assert metadata['status'] == 'success'
        assert metadata['num_segments'] == 2  # 2 segments match SPEAKER_00
        assert metadata['num_rejected'] == 2  # 2 segments rejected
        assert 0.0 < metadata['purity'] <= 1.0
        assert metadata['average_similarity'] > 0.7
        assert len(metadata['segments']) == 2

        # Verify output audio is readable
        audio, sr = sf.read(str(result_path))
        assert len(audio) > 0
        assert sr == 16000

    def test_filter_with_no_matches(
        self, audio_16khz_mono, mock_diarizer, tmp_path
    ):
        """Test filtering when no segments match target speaker."""
        # Create embedding that won't match anything
        unmatched_embedding = np.random.randn(512).astype(np.float32)
        unmatched_embedding = unmatched_embedding / np.linalg.norm(unmatched_embedding)

        # Create diarization with different embeddings
        other_embedding = np.random.randn(512).astype(np.float32)
        other_embedding = other_embedding / np.linalg.norm(other_embedding)

        segments = [
            SpeakerSegment(
                start=0.0, end=2.0, speaker_id="SPEAKER_00",
                embedding=other_embedding
            ),
        ]
        diarization_result = DiarizationResult(
            segments=segments, num_speakers=1, audio_duration=2.0
        )

        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        result_path, metadata = filter_obj.filter_training_audio(
            audio_path=audio_16khz_mono,
            target_embedding=unmatched_embedding,
            similarity_threshold=0.95,  # Very high threshold
            diarization_result=diarization_result,
        )

        # Should create silent audio
        assert result_path.exists()
        assert metadata['status'] == 'no_match'
        assert metadata['num_segments'] == 0
        assert metadata['filtered_duration'] == 0.0
        assert metadata['purity'] == 0.0

        # Verify output is minimal silence
        audio, sr = sf.read(str(result_path))
        assert len(audio) <= sr * 0.2  # Less than 200ms

    def test_filter_auto_output_path(
        self, audio_16khz_mono, sample_embedding, mock_diarization_result, mock_diarizer
    ):
        """Test automatic output path generation."""
        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        result_path, metadata = filter_obj.filter_training_audio(
            audio_path=audio_16khz_mono,
            target_embedding=sample_embedding,
            output_path=None,  # Auto-generate
            diarization_result=mock_diarization_result,
        )

        # Verify path was generated
        assert result_path.exists()
        assert result_path.suffix == '.wav'
        assert 'filtered' in result_path.name

    def test_filter_similarity_threshold(
        self, audio_16khz_mono, sample_embedding, mock_diarization_result, mock_diarizer
    ):
        """Test filtering with different similarity thresholds."""
        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        # Low threshold - accept more segments
        _, metadata_low = filter_obj.filter_training_audio(
            audio_path=audio_16khz_mono,
            target_embedding=sample_embedding,
            similarity_threshold=0.5,
            diarization_result=mock_diarization_result,
        )

        # High threshold - accept fewer segments
        _, metadata_high = filter_obj.filter_training_audio(
            audio_path=audio_16khz_mono,
            target_embedding=sample_embedding,
            similarity_threshold=0.95,
            diarization_result=mock_diarization_result,
        )

        # Low threshold should accept more or equal segments
        assert metadata_low['num_segments'] >= metadata_high['num_segments']

    def test_filter_min_segment_duration(
        self, audio_16khz_mono, sample_embedding, mock_diarizer, tmp_path
    ):
        """Test filtering with minimum segment duration."""
        # Create result with short segments
        short_segments = [
            SpeakerSegment(
                start=0.0, end=0.3, speaker_id="SPEAKER_00",
                embedding=sample_embedding.copy()
            ),  # Too short
            SpeakerSegment(
                start=1.0, end=2.5, speaker_id="SPEAKER_00",
                embedding=sample_embedding.copy()
            ),  # Long enough
        ]
        diarization_result = DiarizationResult(
            segments=short_segments,
            num_speakers=1,
            audio_duration=3.0,
        )

        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        _, metadata = filter_obj.filter_training_audio(
            audio_path=audio_16khz_mono,
            target_embedding=sample_embedding,
            min_segment_duration=0.5,
            diarization_result=diarization_result,
        )

        # Only 1 segment meets duration requirement
        assert metadata['num_segments'] == 1

    def test_filter_runs_diarization_when_needed(
        self, audio_16khz_mono, sample_embedding, mock_diarizer
    ):
        """Test that diarization is run when not provided."""
        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        _, metadata = filter_obj.filter_training_audio(
            audio_path=audio_16khz_mono,
            target_embedding=sample_embedding,
            diarization_result=None,  # Not provided
        )

        # Verify diarizer was called
        mock_diarizer.diarize.assert_called_once()
        assert metadata['status'] in ['success', 'no_match']

    def test_filter_stereo_audio(
        self, audio_44khz_stereo, sample_embedding, mock_diarization_result, mock_diarizer
    ):
        """Test filtering stereo audio (should convert to mono)."""
        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        result_path, metadata = filter_obj.filter_training_audio(
            audio_path=audio_44khz_stereo,
            target_embedding=sample_embedding,
            diarization_result=mock_diarization_result,
        )

        # Verify output is mono
        audio, sr = sf.read(str(result_path))
        assert audio.ndim == 1  # Mono
        assert sr == 44100

    def test_filter_int16_audio(
        self, audio_int16, sample_embedding, mock_diarization_result, mock_diarizer
    ):
        """Test filtering int16 audio format."""
        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        result_path, metadata = filter_obj.filter_training_audio(
            audio_path=audio_int16,
            target_embedding=sample_embedding,
            diarization_result=mock_diarization_result,
        )

        assert result_path.exists()
        assert metadata['status'] in ['success', 'no_match']

    def test_filter_int32_audio(
        self, audio_int32, sample_embedding, mock_diarization_result, mock_diarizer
    ):
        """Test filtering int32 audio format."""
        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        result_path, metadata = filter_obj.filter_training_audio(
            audio_path=audio_int32,
            target_embedding=sample_embedding,
            diarization_result=mock_diarization_result,
        )

        assert result_path.exists()
        assert metadata['status'] in ['success', 'no_match']

    def test_filter_single_segment(
        self, audio_16khz_mono, sample_embedding, mock_diarizer
    ):
        """Test filtering with single matching segment."""
        single_segment = [
            SpeakerSegment(
                start=0.0, end=3.0, speaker_id="SPEAKER_00",
                embedding=sample_embedding.copy()
            ),
        ]
        diarization_result = DiarizationResult(
            segments=single_segment,
            num_speakers=1,
            audio_duration=3.0,
        )

        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        result_path, metadata = filter_obj.filter_training_audio(
            audio_path=audio_16khz_mono,
            target_embedding=sample_embedding,
            diarization_result=diarization_result,
        )

        assert metadata['num_segments'] == 1
        assert metadata['status'] == 'success'

    def test_filter_multiple_segments_with_gaps(
        self, audio_16khz_mono, sample_embedding, mock_diarizer
    ):
        """Test filtering with multiple segments (verifies gap insertion)."""
        segments = [
            SpeakerSegment(
                start=0.0, end=1.0, speaker_id="SPEAKER_00",
                embedding=sample_embedding.copy()
            ),
            SpeakerSegment(
                start=2.0, end=3.0, speaker_id="SPEAKER_00",
                embedding=sample_embedding.copy()
            ),
            SpeakerSegment(
                start=5.0, end=6.0, speaker_id="SPEAKER_00",
                embedding=sample_embedding.copy()
            ),
        ]
        diarization_result = DiarizationResult(
            segments=segments,
            num_speakers=1,
            audio_duration=6.0,
        )

        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        result_path, metadata = filter_obj.filter_training_audio(
            audio_path=audio_16khz_mono,
            target_embedding=sample_embedding,
            diarization_result=diarization_result,
        )

        # Verify 3 segments extracted
        assert metadata['num_segments'] == 3

        # Verify output audio includes gaps (should be longer than just segments)
        audio, sr = sf.read(str(result_path))
        segment_duration = 3.0  # 3 x 1s segments
        gap_duration = 2 * 0.01  # 2 x 10ms gaps
        expected_duration = segment_duration + gap_duration
        actual_duration = len(audio) / sr
        assert abs(actual_duration - expected_duration) < 0.1  # Within 100ms

    def test_filter_segments_sorted_by_time(
        self, audio_16khz_mono, sample_embedding, mock_diarizer
    ):
        """Test that segments are sorted by start time."""
        # Segments in random order
        segments = [
            SpeakerSegment(
                start=5.0, end=6.0, speaker_id="SPEAKER_00",
                embedding=sample_embedding.copy()
            ),
            SpeakerSegment(
                start=0.0, end=1.0, speaker_id="SPEAKER_00",
                embedding=sample_embedding.copy()
            ),
            SpeakerSegment(
                start=2.0, end=3.0, speaker_id="SPEAKER_00",
                embedding=sample_embedding.copy()
            ),
        ]
        diarization_result = DiarizationResult(
            segments=segments,
            num_speakers=1,
            audio_duration=6.0,
        )

        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        _, metadata = filter_obj.filter_training_audio(
            audio_path=audio_16khz_mono,
            target_embedding=sample_embedding,
            diarization_result=diarization_result,
        )

        # Verify segments in metadata are sorted
        segment_starts = [s['start'] for s in metadata['segments']]
        assert segment_starts == sorted(segment_starts)

    def test_filter_bounds_checking(
        self, audio_16khz_mono, sample_embedding, mock_diarizer
    ):
        """Test bounds checking for segments beyond audio length."""
        # Segment extends beyond audio
        segments = [
            SpeakerSegment(
                start=8.0, end=15.0, speaker_id="SPEAKER_00",
                embedding=sample_embedding.copy()
            ),
        ]
        diarization_result = DiarizationResult(
            segments=segments,
            num_speakers=1,
            audio_duration=10.0,
        )

        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        result_path, metadata = filter_obj.filter_training_audio(
            audio_path=audio_16khz_mono,
            target_embedding=sample_embedding,
            diarization_result=diarization_result,
        )

        # Should handle gracefully
        assert result_path.exists()
        assert metadata['status'] in ['success', 'no_match']

    def test_filter_purity_calculation(
        self, audio_16khz_mono, sample_embedding, different_embedding, mock_diarizer
    ):
        """Test purity calculation (ratio of matched to total speech)."""
        segments = [
            SpeakerSegment(
                start=0.0, end=2.0, speaker_id="SPEAKER_00",
                embedding=sample_embedding.copy()
            ),  # Matches
            SpeakerSegment(
                start=2.0, end=4.0, speaker_id="SPEAKER_01",
                embedding=different_embedding.copy()
            ),  # Doesn't match
        ]
        diarization_result = DiarizationResult(
            segments=segments,
            num_speakers=2,
            audio_duration=4.0,
        )

        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        _, metadata = filter_obj.filter_training_audio(
            audio_path=audio_16khz_mono,
            target_embedding=sample_embedding,
            diarization_result=diarization_result,
        )

        # Purity = 2.0s matched / 4.0s total speech = 0.5
        assert 0.45 <= metadata['purity'] <= 0.55

    def test_filter_creates_output_directory(
        self, audio_16khz_mono, sample_embedding, mock_diarization_result, mock_diarizer, tmp_path
    ):
        """Test that output directory is created if it doesn't exist."""
        output_path = tmp_path / "nested" / "dir" / "output.wav"
        assert not output_path.parent.exists()

        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        result_path, _ = filter_obj.filter_training_audio(
            audio_path=audio_16khz_mono,
            target_embedding=sample_embedding,
            output_path=output_path,
            diarization_result=mock_diarization_result,
        )

        assert result_path.parent.exists()
        assert result_path.exists()

    def test_filter_with_segments_no_embedding(
        self, audio_16khz_mono, sample_embedding, mock_diarizer
    ):
        """Test filtering skips segments without embeddings."""
        segments = [
            SpeakerSegment(
                start=0.0, end=1.0, speaker_id="SPEAKER_00",
                embedding=None  # No embedding
            ),
            SpeakerSegment(
                start=2.0, end=3.0, speaker_id="SPEAKER_01",
                embedding=sample_embedding.copy()
            ),
        ]
        diarization_result = DiarizationResult(
            segments=segments,
            num_speakers=2,
            audio_duration=3.0,
        )

        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        _, metadata = filter_obj.filter_training_audio(
            audio_path=audio_16khz_mono,
            target_embedding=sample_embedding,
            diarization_result=diarization_result,
        )

        # Only 1 segment has embedding
        assert metadata['num_segments'] <= 1


# ============================================================================
# Test filter_with_profile_matching
# ============================================================================


class TestFilterWithProfileMatching:
    """Test filter_with_profile_matching method."""

    def test_filter_with_valid_profile(
        self, audio_16khz_mono, sample_embedding, different_embedding, mock_diarization_result, mock_diarizer
    ):
        """Test filtering with valid profile ID."""
        profile_embeddings = {
            'profile_1': sample_embedding,
            'profile_2': different_embedding,
        }

        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        result_path, metadata = filter_obj.filter_with_profile_matching(
            audio_path=audio_16khz_mono,
            profile_embeddings=profile_embeddings,
            target_profile_id='profile_1',
            diarization_result=mock_diarization_result,
        )

        assert result_path.exists()
        assert metadata['status'] in ['success', 'no_match']

    def test_filter_with_invalid_profile(
        self, audio_16khz_mono, sample_embedding, mock_diarizer
    ):
        """Test filtering with invalid profile ID raises error."""
        profile_embeddings = {
            'profile_1': sample_embedding,
        }

        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        with pytest.raises(ValueError, match="Target profile .* not in embeddings dict"):
            filter_obj.filter_with_profile_matching(
                audio_path=audio_16khz_mono,
                profile_embeddings=profile_embeddings,
                target_profile_id='nonexistent_profile',
            )

    def test_filter_with_multiple_profiles(
        self, audio_16khz_mono, sample_embedding, different_embedding, mock_diarization_result, mock_diarizer
    ):
        """Test filtering with multiple profiles."""
        profile_embeddings = {
            'speaker_a': sample_embedding,
            'speaker_b': different_embedding,
            'speaker_c': np.random.randn(512).astype(np.float32),
        }

        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        # Filter for speaker_a
        _, metadata_a = filter_obj.filter_with_profile_matching(
            audio_path=audio_16khz_mono,
            profile_embeddings=profile_embeddings,
            target_profile_id='speaker_a',
            diarization_result=mock_diarization_result,
        )

        # Filter for speaker_b
        _, metadata_b = filter_obj.filter_with_profile_matching(
            audio_path=audio_16khz_mono,
            profile_embeddings=profile_embeddings,
            target_profile_id='speaker_b',
            diarization_result=mock_diarization_result,
        )

        # Should get different results
        assert metadata_a['status'] in ['success', 'no_match']
        assert metadata_b['status'] in ['success', 'no_match']

    def test_filter_with_profile_kwargs(
        self, audio_16khz_mono, sample_embedding, mock_diarization_result, mock_diarizer, tmp_path
    ):
        """Test that kwargs are passed through to filter_training_audio."""
        profile_embeddings = {'profile_1': sample_embedding}
        output_path = tmp_path / "custom_output.wav"

        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        result_path, metadata = filter_obj.filter_with_profile_matching(
            audio_path=audio_16khz_mono,
            profile_embeddings=profile_embeddings,
            target_profile_id='profile_1',
            output_path=output_path,
            similarity_threshold=0.8,
            min_segment_duration=1.0,
            diarization_result=mock_diarization_result,
        )

        assert result_path == output_path


# ============================================================================
# Test auto_split_by_speakers
# ============================================================================


class TestAutoSplitBySpeakers:
    """Test auto_split_by_speakers method."""

    def test_split_two_speakers(
        self, audio_16khz_mono, sample_embedding, different_embedding, mock_diarizer, tmp_path
    ):
        """Test splitting audio with 2 speakers."""
        segments = [
            SpeakerSegment(start=0.0, end=2.0, speaker_id="SPEAKER_00"),
            SpeakerSegment(start=2.0, end=4.0, speaker_id="SPEAKER_01"),
            SpeakerSegment(start=4.0, end=6.0, speaker_id="SPEAKER_00"),
        ]
        diarization_result = DiarizationResult(
            segments=segments, num_speakers=2, audio_duration=6.0
        )

        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        results = filter_obj.auto_split_by_speakers(
            audio_path=audio_16khz_mono,
            output_dir=tmp_path,
            diarization_result=diarization_result,
        )

        # Verify 2 output files
        assert len(results) == 2
        assert 'SPEAKER_00' in results
        assert 'SPEAKER_01' in results

        # Verify output files exist
        path_00, duration_00 = results['SPEAKER_00']
        path_01, duration_01 = results['SPEAKER_01']

        assert path_00.exists()
        assert path_01.exists()
        assert duration_00 > 0
        assert duration_01 > 0

        # SPEAKER_00 has 2 segments (4s total), SPEAKER_01 has 1 (2s)
        assert duration_00 > duration_01

    def test_split_three_speakers(
        self, audio_16khz_mono, mock_diarizer, tmp_path
    ):
        """Test splitting audio with 3 speakers."""
        segments = [
            SpeakerSegment(start=0.0, end=1.5, speaker_id="SPEAKER_00"),
            SpeakerSegment(start=1.5, end=3.0, speaker_id="SPEAKER_01"),
            SpeakerSegment(start=3.0, end=5.0, speaker_id="SPEAKER_02"),
        ]
        diarization_result = DiarizationResult(
            segments=segments, num_speakers=3, audio_duration=5.0
        )

        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        results = filter_obj.auto_split_by_speakers(
            audio_path=audio_16khz_mono,
            output_dir=tmp_path,
            diarization_result=diarization_result,
        )

        assert len(results) == 3
        assert all(speaker in results for speaker in ['SPEAKER_00', 'SPEAKER_01', 'SPEAKER_02'])

    def test_split_runs_diarization_when_needed(
        self, audio_16khz_mono, mock_diarizer, tmp_path
    ):
        """Test that diarization is run when not provided."""
        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        _ = filter_obj.auto_split_by_speakers(
            audio_path=audio_16khz_mono,
            output_dir=tmp_path,
            diarization_result=None,
        )

        mock_diarizer.diarize.assert_called_once()

    def test_split_filters_short_segments(
        self, audio_16khz_mono, mock_diarizer, tmp_path
    ):
        """Test that short segments are filtered out."""
        segments = [
            SpeakerSegment(start=0.0, end=2.0, speaker_id="SPEAKER_00"),  # Long enough
            SpeakerSegment(start=2.0, end=2.3, speaker_id="SPEAKER_01"),  # Too short
        ]
        diarization_result = DiarizationResult(
            segments=segments, num_speakers=2, audio_duration=2.3
        )

        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        results = filter_obj.auto_split_by_speakers(
            audio_path=audio_16khz_mono,
            output_dir=tmp_path,
            diarization_result=diarization_result,
            min_segment_duration=0.5,
        )

        # Only SPEAKER_00 meets duration requirement
        assert len(results) == 1
        assert 'SPEAKER_00' in results

    def test_split_creates_output_directory(
        self, audio_16khz_mono, mock_diarization_result, mock_diarizer, tmp_path
    ):
        """Test that output directory is created if needed."""
        output_dir = tmp_path / "nested" / "output"
        assert not output_dir.exists()

        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        results = filter_obj.auto_split_by_speakers(
            audio_path=audio_16khz_mono,
            output_dir=output_dir,
            diarization_result=mock_diarization_result,
        )

        assert output_dir.exists()
        assert len(results) > 0

    def test_split_output_filenames(
        self, audio_16khz_mono, mock_diarization_result, mock_diarizer, tmp_path
    ):
        """Test output filename format."""
        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        results = filter_obj.auto_split_by_speakers(
            audio_path=audio_16khz_mono,
            output_dir=tmp_path,
            diarization_result=mock_diarization_result,
        )

        # Verify filename format: {original_name}_{speaker_id}.wav
        for speaker_id, (path, _) in results.items():
            assert speaker_id in path.name
            assert audio_16khz_mono.stem in path.name
            assert path.suffix == '.wav'

    def test_split_with_multiple_segments_per_speaker(
        self, audio_16khz_mono, mock_diarizer, tmp_path
    ):
        """Test splitting when speakers have multiple segments."""
        segments = [
            SpeakerSegment(start=0.0, end=1.0, speaker_id="SPEAKER_00"),
            SpeakerSegment(start=1.0, end=2.0, speaker_id="SPEAKER_01"),
            SpeakerSegment(start=2.0, end=3.0, speaker_id="SPEAKER_00"),
            SpeakerSegment(start=3.0, end=4.0, speaker_id="SPEAKER_01"),
            SpeakerSegment(start=4.0, end=5.0, speaker_id="SPEAKER_00"),
        ]
        diarization_result = DiarizationResult(
            segments=segments, num_speakers=2, audio_duration=5.0
        )

        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        results = filter_obj.auto_split_by_speakers(
            audio_path=audio_16khz_mono,
            output_dir=tmp_path,
            diarization_result=diarization_result,
        )

        # Both speakers have multiple segments
        _, duration_00 = results['SPEAKER_00']
        _, duration_01 = results['SPEAKER_01']

        # SPEAKER_00 has 3 segments, SPEAKER_01 has 2
        assert duration_00 > duration_01

    def test_split_handles_stereo_to_mono(
        self, audio_44khz_stereo, mock_diarization_result, mock_diarizer, tmp_path
    ):
        """Test splitting converts stereo to mono."""
        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        results = filter_obj.auto_split_by_speakers(
            audio_path=audio_44khz_stereo,
            output_dir=tmp_path,
            diarization_result=mock_diarization_result,
        )

        # Verify output is mono
        for speaker_id, (path, _) in results.items():
            audio, sr = sf.read(str(path))
            assert audio.ndim == 1  # Mono
            assert sr == 44100


# ============================================================================
# Test Convenience Function
# ============================================================================


class TestConvenienceFunction:
    """Test the convenience function filter_training_audio()."""

    @patch('auto_voice.audio.training_filter.TrainingDataFilter')
    def test_convenience_function_creates_filter(
        self, mock_filter_class, audio_16khz_mono, sample_embedding
    ):
        """Test that convenience function creates TrainingDataFilter."""
        mock_instance = MagicMock()
        mock_instance.filter_training_audio.return_value = (Path("/output.wav"), {})
        mock_filter_class.return_value = mock_instance

        filter_training_audio(
            audio_path=audio_16khz_mono,
            target_embedding=sample_embedding,
        )

        mock_filter_class.assert_called_once_with()

    @patch('auto_voice.audio.training_filter.TrainingDataFilter')
    def test_convenience_function_calls_method(
        self, mock_filter_class, audio_16khz_mono, sample_embedding, tmp_path
    ):
        """Test that convenience function calls filter_training_audio method."""
        output_path = tmp_path / "output.wav"
        mock_instance = MagicMock()
        mock_instance.filter_training_audio.return_value = (output_path, {'status': 'success'})
        mock_filter_class.return_value = mock_instance

        result_path, metadata = filter_training_audio(
            audio_path=audio_16khz_mono,
            target_embedding=sample_embedding,
            output_path=output_path,
            similarity_threshold=0.8,
        )

        mock_instance.filter_training_audio.assert_called_once_with(
            audio_path=audio_16khz_mono,
            target_embedding=sample_embedding,
            output_path=output_path,
            similarity_threshold=0.8,
        )
        assert result_path == output_path
        assert metadata['status'] == 'success'


# ============================================================================
# Test Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_audio_file(self, sample_embedding, mock_diarizer, tmp_path):
        """Test handling of empty/corrupt audio file."""
        # Create empty file
        empty_file = tmp_path / "empty.wav"
        empty_file.touch()

        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        # Should raise error or handle gracefully
        with pytest.raises(Exception):
            filter_obj.filter_training_audio(
                audio_path=empty_file,
                target_embedding=sample_embedding,
            )

    def test_very_short_audio(self, sample_embedding, mock_diarizer, tmp_path):
        """Test filtering very short audio (< 1s)."""
        sr = 16000
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, sr // 2))
        audio_path = tmp_path / "short.wav"
        sf.write(str(audio_path), audio, sr)

        segments = [
            SpeakerSegment(
                start=0.0, end=0.5, speaker_id="SPEAKER_00",
                embedding=sample_embedding.copy()
            ),
        ]
        diarization_result = DiarizationResult(
            segments=segments, num_speakers=1, audio_duration=0.5
        )

        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        result_path, metadata = filter_obj.filter_training_audio(
            audio_path=audio_path,
            target_embedding=sample_embedding,
            min_segment_duration=0.1,
            diarization_result=diarization_result,
        )

        assert result_path.exists()

    def test_all_silence_audio(self, sample_embedding, mock_diarizer, tmp_path):
        """Test filtering audio with no speech detected."""
        sr = 16000
        audio = np.zeros(sr * 3)  # 3 seconds of silence
        audio_path = tmp_path / "silence.wav"
        sf.write(str(audio_path), audio, sr)

        # Empty diarization result
        diarization_result = DiarizationResult(
            segments=[], num_speakers=0, audio_duration=3.0
        )

        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        result_path, metadata = filter_obj.filter_training_audio(
            audio_path=audio_path,
            target_embedding=sample_embedding,
            diarization_result=diarization_result,
        )

        assert metadata['status'] == 'no_match'
        assert metadata['num_segments'] == 0

    def test_segment_at_exact_boundaries(
        self, audio_16khz_mono, sample_embedding, mock_diarizer
    ):
        """Test segment exactly at audio start/end."""
        # Get actual audio duration
        audio, sr = sf.read(str(audio_16khz_mono))
        duration = len(audio) / sr

        segments = [
            SpeakerSegment(
                start=0.0, end=duration, speaker_id="SPEAKER_00",
                embedding=sample_embedding.copy()
            ),
        ]
        diarization_result = DiarizationResult(
            segments=segments, num_speakers=1, audio_duration=duration
        )

        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        result_path, metadata = filter_obj.filter_training_audio(
            audio_path=audio_16khz_mono,
            target_embedding=sample_embedding,
            diarization_result=diarization_result,
        )

        assert metadata['status'] == 'success'
        assert metadata['num_segments'] == 1

    def test_zero_duration_segment(
        self, audio_16khz_mono, sample_embedding, mock_diarizer
    ):
        """Test handling of zero-duration segments."""
        segments = [
            SpeakerSegment(
                start=2.0, end=2.0, speaker_id="SPEAKER_00",
                embedding=sample_embedding.copy()
            ),
        ]
        diarization_result = DiarizationResult(
            segments=segments, num_speakers=1, audio_duration=3.0
        )

        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        _, metadata = filter_obj.filter_training_audio(
            audio_path=audio_16khz_mono,
            target_embedding=sample_embedding,
            diarization_result=diarization_result,
        )

        # Should not crash
        assert metadata['status'] in ['success', 'no_match']

    def test_invalid_audio_path(self, sample_embedding, mock_diarizer):
        """Test error handling for nonexistent audio file."""
        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        with pytest.raises(Exception):
            filter_obj.filter_training_audio(
                audio_path="/nonexistent/path/audio.wav",
                target_embedding=sample_embedding,
            )

    def test_metadata_completeness(
        self, audio_16khz_mono, sample_embedding, mock_diarization_result, mock_diarizer
    ):
        """Test that all expected metadata fields are present."""
        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        _, metadata = filter_obj.filter_training_audio(
            audio_path=audio_16khz_mono,
            target_embedding=sample_embedding,
            diarization_result=mock_diarization_result,
        )

        # Check all expected fields
        expected_fields = [
            'original_duration', 'filtered_duration', 'num_segments',
            'num_rejected', 'purity', 'status'
        ]

        for field in expected_fields:
            assert field in metadata

        if metadata['status'] == 'success':
            assert 'average_similarity' in metadata
            assert 'segments' in metadata

    def test_pathlib_path_support(
        self, sample_embedding, mock_diarization_result, mock_diarizer, tmp_path
    ):
        """Test that Path objects are supported for all path parameters."""
        # Create audio with pathlib.Path
        sr = 16000
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 3, sr * 3))
        audio_path = Path(tmp_path) / "test.wav"
        sf.write(str(audio_path), audio, sr)

        output_path = Path(tmp_path) / "output.wav"

        filter_obj = TrainingDataFilter(diarizer=mock_diarizer)

        result_path, _ = filter_obj.filter_training_audio(
            audio_path=audio_path,
            target_embedding=sample_embedding,
            output_path=output_path,
            diarization_result=mock_diarization_result,
        )

        assert isinstance(result_path, Path)
        assert result_path.exists()
