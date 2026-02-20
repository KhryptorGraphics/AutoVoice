"""Comprehensive tests for multi_artist_separator.py.

Coverage Target: 0% → 90% (342 lines)
Test Count: ~28 tests

Beads: AV-u94 (P0 Critical)
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open

import numpy as np
import pytest
import torch

from auto_voice.audio.multi_artist_separator import (
    ArtistSegment,
    SeparationResult,
    MultiArtistSeparator,
    get_multi_artist_separator,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_vocals():
    """Generate synthetic vocal audio."""
    sr = 44100
    duration = 10.0
    num_samples = int(sr * duration)

    # Create multi-speaker vocals (different frequencies)
    t = np.linspace(0, duration, num_samples)
    speaker1 = 0.5 * np.sin(2 * np.pi * 220 * t[:num_samples//2])  # First half
    speaker2 = 0.5 * np.sin(2 * np.pi * 330 * t[num_samples//2:])  # Second half
    vocals = np.concatenate([speaker1, speaker2]).astype(np.float32)

    return vocals, sr


@pytest.fixture
def sample_instrumental():
    """Generate synthetic instrumental audio."""
    sr = 44100
    duration = 10.0
    num_samples = int(sr * duration)

    # Create drums/bass pattern
    t = np.linspace(0, duration, num_samples)
    instrumental = 0.3 * np.sin(2 * np.pi * 110 * t).astype(np.float32)

    return instrumental, sr


@pytest.fixture
def sample_mixed_audio():
    """Generate mixed audio (vocals + instrumental)."""
    sr = 44100
    duration = 10.0
    num_samples = int(sr * duration)

    t = np.linspace(0, duration, num_samples)
    vocals = 0.5 * np.sin(2 * np.pi * 440 * t)
    instrumental = 0.3 * np.sin(2 * np.pi * 110 * t)
    mixed = (vocals + instrumental).astype(np.float32)

    return mixed, sr


@pytest.fixture
def mock_diarization_segments():
    """Create mock diarization segments."""
    return [
        {
            'start': 0.0,
            'end': 5.0,
            'speaker_id': 'speaker_0',
            'embedding': np.random.randn(256).astype(np.float32),
            'duration': 5.0,
        },
        {
            'start': 5.0,
            'end': 10.0,
            'speaker_id': 'speaker_1',
            'embedding': np.random.randn(256).astype(np.float32),
            'duration': 5.0,
        },
    ]


@pytest.fixture
def temp_profiles_dir(tmp_path):
    """Create temporary profiles directory."""
    profiles_dir = tmp_path / "voice_profiles"
    profiles_dir.mkdir()
    return profiles_dir


# ============================================================================
# Test ArtistSegment Dataclass
# ============================================================================

class TestArtistSegment:
    """Test ArtistSegment dataclass."""

    @pytest.mark.smoke
    def test_create_segment(self, sample_vocals):
        """Test creating an artist segment."""
        vocals, sr = sample_vocals
        segment_audio = vocals[:sr]  # 1 second

        segment = ArtistSegment(
            profile_id="test_profile",
            profile_name="Test Artist",
            start=0.0,
            end=1.0,
            audio=segment_audio,
            sample_rate=sr,
            embedding=np.random.randn(256),
            similarity=0.92,
            is_new_profile=False,
        )

        assert segment.profile_id == "test_profile"
        assert segment.profile_name == "Test Artist"
        assert segment.start == 0.0
        assert segment.end == 1.0
        assert segment.duration == 1.0
        assert len(segment.audio) == sr
        assert segment.sample_rate == sr
        assert segment.similarity == 0.92
        assert segment.is_new_profile is False

    def test_segment_duration_property(self):
        """Test duration property calculation."""
        segment = ArtistSegment(
            profile_id="test",
            profile_name="Test",
            start=2.5,
            end=7.3,
            audio=np.zeros(100),
            sample_rate=16000,
            embedding=np.zeros(256),
            similarity=0.9,
        )

        assert segment.duration == pytest.approx(4.8)


class TestSeparationResult:
    """Test SeparationResult dataclass."""

    def test_create_result(self, sample_vocals, sample_instrumental):
        """Test creating a separation result."""
        vocals, sr = sample_vocals
        instrumental, _ = sample_instrumental

        artists = {
            "artist1": [
                ArtistSegment(
                    profile_id="artist1",
                    profile_name="Artist One",
                    start=0.0,
                    end=5.0,
                    audio=vocals[:sr*5],
                    sample_rate=sr,
                    embedding=np.zeros(256),
                    similarity=0.9,
                )
            ]
        }

        result = SeparationResult(
            artists=artists,
            vocals=vocals,
            instrumental=instrumental,
            sample_rate=sr,
            total_duration=10.0,
            num_artists=1,
            new_profiles_created=["artist1"],
        )

        assert result.num_artists == 1
        assert result.total_duration == 10.0
        assert "artist1" in result.artists
        assert len(result.new_profiles_created) == 1
        assert result.sample_rate == sr


# ============================================================================
# Test MultiArtistSeparator Initialization
# ============================================================================

class TestMultiArtistSeparatorInit:
    """Test MultiArtistSeparator initialization."""

    @pytest.mark.smoke
    def test_init_default(self, temp_profiles_dir):
        """Test default initialization."""
        separator = MultiArtistSeparator(profiles_dir=temp_profiles_dir)

        assert separator.profiles_dir == temp_profiles_dir
        assert separator.device.type in ['cuda', 'cpu']
        assert separator.auto_create_profiles is True
        assert separator.auto_queue_training is True
        assert separator._separator is None
        assert separator._diarizer is None
        assert separator._identifier is None

    def test_init_cpu_device(self, temp_profiles_dir):
        """Test initialization with CPU device."""
        separator = MultiArtistSeparator(
            profiles_dir=temp_profiles_dir,
            device="cpu"
        )

        assert separator.device.type == "cpu"

    def test_init_no_auto_create(self, temp_profiles_dir):
        """Test initialization without auto profile creation."""
        separator = MultiArtistSeparator(
            profiles_dir=temp_profiles_dir,
            auto_create_profiles=False,
            auto_queue_training=False,
        )

        assert separator.auto_create_profiles is False
        assert separator.auto_queue_training is False

    def test_similarity_threshold_constant(self):
        """Test similarity threshold constant."""
        assert MultiArtistSeparator.SIMILARITY_THRESHOLD == 0.85

    def test_min_segment_duration_constant(self):
        """Test minimum segment duration constant."""
        assert MultiArtistSeparator.MIN_SEGMENT_DURATION == 1.0


# ============================================================================
# Test Lazy Loading
# ============================================================================

class TestLazyLoading:
    """Test lazy loading of components."""

    def test_load_separator(self, temp_profiles_dir):
        """Test lazy loading of vocal separator."""
        separator = MultiArtistSeparator(profiles_dir=temp_profiles_dir, device="cpu")
        assert separator._separator is None

        with patch('auto_voice.audio.separation.VocalSeparator') as mock_sep:
            mock_instance = MagicMock()
            mock_sep.return_value = mock_instance

            separator._load_separator()

            assert separator._separator is not None
            mock_sep.assert_called_once()

    def test_load_diarizer(self, temp_profiles_dir):
        """Test lazy loading of speaker diarizer."""
        separator = MultiArtistSeparator(profiles_dir=temp_profiles_dir, device="cpu")
        assert separator._diarizer is None

        with patch('auto_voice.audio.speaker_diarization.SpeakerDiarizer') as mock_diar:
            mock_instance = MagicMock()
            mock_diar.return_value = mock_instance

            separator._load_diarizer()

            assert separator._diarizer is not None
            mock_diar.assert_called_once()

    def test_load_identifier(self, temp_profiles_dir):
        """Test lazy loading of voice identifier."""
        separator = MultiArtistSeparator(profiles_dir=temp_profiles_dir, device="cpu")
        assert separator._identifier is None

        with patch('auto_voice.inference.voice_identifier.VoiceIdentifier') as mock_id:
            mock_instance = MagicMock()
            mock_instance._embeddings = {'profile1': np.zeros(256)}
            mock_id.return_value = mock_instance

            separator._load_identifier()

            assert separator._identifier is not None
            mock_instance.load_all_embeddings.assert_called_once()

    def test_load_job_manager_success(self, temp_profiles_dir):
        """Test lazy loading of job manager."""
        separator = MultiArtistSeparator(profiles_dir=temp_profiles_dir, device="cpu")

        with patch('auto_voice.training.job_manager.TrainingJobManager') as mock_jm:
            mock_instance = MagicMock()
            mock_jm.return_value = mock_instance

            separator._load_job_manager()

            assert separator._job_manager is not None

    def test_load_job_manager_failure(self, temp_profiles_dir):
        """Test job manager loading failure is handled gracefully."""
        separator = MultiArtistSeparator(profiles_dir=temp_profiles_dir, device="cpu")

        with patch('auto_voice.training.job_manager.TrainingJobManager') as mock_jm:
            mock_jm.side_effect = Exception("JobManager not available")

            separator._load_job_manager()

            assert separator._job_manager is None


# ============================================================================
# Test Vocal Separation
# ============================================================================

class TestVocalSeparation:
    """Test vocal/instrumental separation."""

    def test_separate_vocals(self, temp_profiles_dir, sample_mixed_audio):
        """Test separating vocals from instrumental."""
        audio, sr = sample_mixed_audio
        separator = MultiArtistSeparator(profiles_dir=temp_profiles_dir, device="cpu")

        with patch.object(separator, '_load_separator'):
            mock_sep = MagicMock()
            mock_sep.separate.return_value = {
                'vocals': np.random.randn(len(audio)).astype(np.float32),
                'instrumental': np.random.randn(len(audio)).astype(np.float32),
            }
            separator._separator = mock_sep

            vocals, instrumental = separator.separate_vocals(audio, sr)

            assert len(vocals) == len(audio)
            assert len(instrumental) == len(audio)
            assert vocals.dtype == np.float32
            mock_sep.separate.assert_called_once_with(audio, sr)


# ============================================================================
# Test Speaker Diarization
# ============================================================================

class TestSpeakerDiarization:
    """Test speaker diarization functionality."""

    def test_diarize_vocals(self, temp_profiles_dir, sample_vocals):
        """Test speaker diarization on vocals."""
        vocals, sr = sample_vocals
        separator = MultiArtistSeparator(profiles_dir=temp_profiles_dir, device="cpu")

        # Mock diarizer
        with patch.object(separator, '_load_diarizer'):
            mock_diar = MagicMock()
            mock_result = MagicMock()
            mock_result.segments = [
                MagicMock(
                    start=0.0,
                    end=5.0,
                    speaker_id='speaker_0',
                    embedding=np.random.randn(256),
                    duration=5.0,
                ),
                MagicMock(
                    start=5.0,
                    end=10.0,
                    speaker_id='speaker_1',
                    embedding=np.random.randn(256),
                    duration=5.0,
                ),
            ]
            mock_diar.diarize.return_value = mock_result
            separator._diarizer = mock_diar

            # Mock tempfile
            with patch('tempfile.NamedTemporaryFile') as mock_temp:
                mock_file = MagicMock()
                mock_file.name = '/tmp/test.wav'
                mock_file.__enter__ = MagicMock(return_value=mock_file)
                mock_file.__exit__ = MagicMock(return_value=False)
                mock_temp.return_value = mock_file

                with patch('soundfile.write'):
                    with patch('os.unlink'):
                        segments = separator.diarize_vocals(vocals, sr, num_speakers=2)

            assert len(segments) == 2
            assert segments[0]['start'] == 0.0
            assert segments[0]['end'] == 5.0
            assert segments[0]['speaker_id'] == 'speaker_0'
            assert segments[1]['speaker_id'] == 'speaker_1'

    def test_diarize_temp_file_cleanup(self, temp_profiles_dir, sample_vocals):
        """Test that temporary files are cleaned up."""
        vocals, sr = sample_vocals
        separator = MultiArtistSeparator(profiles_dir=temp_profiles_dir, device="cpu")

        with patch.object(separator, '_load_diarizer'):
            mock_diar = MagicMock()
            mock_result = MagicMock()
            mock_result.segments = []
            mock_diar.diarize.return_value = mock_result
            separator._diarizer = mock_diar

            with patch('tempfile.NamedTemporaryFile') as mock_temp:
                mock_file = MagicMock()
                mock_file.name = '/tmp/test.wav'
                mock_file.__enter__ = MagicMock(return_value=mock_file)
                mock_file.__exit__ = MagicMock(return_value=False)
                mock_temp.return_value = mock_file

                with patch('soundfile.write'):
                    with patch('os.unlink') as mock_unlink:
                        separator.diarize_vocals(vocals, sr)

                        # Verify temp file was deleted
                        mock_unlink.assert_called_once()


# ============================================================================
# Test Profile Matching
# ============================================================================

class TestProfileMatching:
    """Test matching segments to voice profiles."""

    def test_match_segments_to_profiles(self, temp_profiles_dir, sample_vocals, mock_diarization_segments):
        """Test matching diarization segments to profiles."""
        vocals, sr = sample_vocals
        separator = MultiArtistSeparator(
            profiles_dir=temp_profiles_dir,
            device="cpu",
            auto_create_profiles=False
        )

        with patch.object(separator, '_load_identifier'):
            mock_id = MagicMock()

            # Mock identification results
            mock_result1 = MagicMock()
            mock_result1.profile_id = "profile1"
            mock_result1.profile_name = "Artist 1"
            mock_result1.similarity = 0.9

            mock_result2 = MagicMock()
            mock_result2.profile_id = "profile2"
            mock_result2.profile_name = "Artist 2"
            mock_result2.similarity = 0.88

            mock_id.identify.side_effect = [mock_result1, mock_result2]
            separator._identifier = mock_id

            artists, new_profiles = separator.match_segments_to_profiles(
                segments=mock_diarization_segments,
                vocals=vocals,
                sample_rate=sr,
            )

            assert len(artists) == 2
            assert "profile1" in artists
            assert "profile2" in artists
            assert len(new_profiles) == 0
            assert len(artists["profile1"]) == 1
            assert artists["profile1"][0].similarity == 0.9

    def test_match_with_auto_create(self, temp_profiles_dir, sample_vocals, mock_diarization_segments):
        """Test matching with automatic profile creation."""
        vocals, sr = sample_vocals
        separator = MultiArtistSeparator(
            profiles_dir=temp_profiles_dir,
            device="cpu",
            auto_create_profiles=True
        )

        with patch.object(separator, '_load_identifier'):
            mock_id = MagicMock()

            # Mock new profile creation
            mock_result = MagicMock()
            mock_result.profile_id = "new_profile_1"
            mock_result.profile_name = "New Artist"
            mock_result.similarity = 1.0  # Perfect match = new profile

            mock_id.identify_or_create.return_value = mock_result
            separator._identifier = mock_id

            artists, new_profiles = separator.match_segments_to_profiles(
                segments=mock_diarization_segments,
                vocals=vocals,
                sample_rate=sr,
            )

            assert len(new_profiles) > 0
            assert "new_profile_1" in artists

    def test_skip_short_segments(self, temp_profiles_dir, sample_vocals):
        """Test that segments shorter than MIN_SEGMENT_DURATION are skipped."""
        vocals, sr = sample_vocals
        separator = MultiArtistSeparator(profiles_dir=temp_profiles_dir, device="cpu")

        short_segments = [
            {
                'start': 0.0,
                'end': 0.5,  # Too short (< 1.0s)
                'speaker_id': 'speaker_0',
                'embedding': np.random.randn(256),
                'duration': 0.5,
            }
        ]

        with patch.object(separator, '_load_identifier'):
            separator._identifier = MagicMock()

            artists, new_profiles = separator.match_segments_to_profiles(
                segments=short_segments,
                vocals=vocals,
                sample_rate=sr,
            )

            # No segments should be matched
            assert len(artists) == 0

    def test_missing_embedding_extraction(self, temp_profiles_dir, sample_vocals):
        """Test embedding extraction when not provided in segment."""
        vocals, sr = sample_vocals
        separator = MultiArtistSeparator(profiles_dir=temp_profiles_dir, device="cpu")

        segments = [
            {
                'start': 0.0,
                'end': 2.0,
                'speaker_id': 'speaker_0',
                'embedding': None,  # No embedding
                'duration': 2.0,
            }
        ]

        with patch.object(separator, '_load_identifier'):
            mock_id = MagicMock()
            mock_id.extract_embedding.return_value = np.random.randn(256)

            mock_result = MagicMock()
            mock_result.profile_id = "profile1"
            mock_result.profile_name = "Artist 1"
            mock_result.similarity = 0.9

            mock_id.identify_or_create.return_value = mock_result
            separator._identifier = mock_id

            artists, _ = separator.match_segments_to_profiles(
                segments=segments,
                vocals=vocals,
                sample_rate=sr,
            )

            # Should extract embedding
            mock_id.extract_embedding.assert_called_once()


# ============================================================================
# Test Full Pipeline
# ============================================================================

class TestFullPipeline:
    """Test complete separation and routing pipeline."""

    def test_separate_and_route(self, temp_profiles_dir, sample_mixed_audio):
        """Test full separation and routing pipeline."""
        audio, sr = sample_mixed_audio
        separator = MultiArtistSeparator(
            profiles_dir=temp_profiles_dir,
            device="cpu",
            auto_queue_training=False
        )

        # Mock all components
        with patch.object(separator, 'separate_vocals') as mock_sep:
            vocals = np.random.randn(len(audio)).astype(np.float32)
            instrumental = np.random.randn(len(audio)).astype(np.float32)
            mock_sep.return_value = (vocals, instrumental)

            with patch.object(separator, 'diarize_vocals') as mock_diar:
                mock_diar.return_value = [
                    {
                        'start': 0.0,
                        'end': 5.0,
                        'speaker_id': 'speaker_0',
                        'embedding': np.random.randn(256),
                        'duration': 5.0,
                    }
                ]

                with patch.object(separator, 'match_segments_to_profiles') as mock_match:
                    mock_match.return_value = (
                        {
                            'profile1': [
                                ArtistSegment(
                                    profile_id='profile1',
                                    profile_name='Artist 1',
                                    start=0.0,
                                    end=5.0,
                                    audio=vocals[:sr*5],
                                    sample_rate=sr,
                                    embedding=np.zeros(256),
                                    similarity=0.9,
                                )
                            ]
                        },
                        []  # No new profiles
                    )

                    result = separator.separate_and_route(
                        audio=audio,
                        sample_rate=sr,
                        num_speakers=1,
                    )

                    assert isinstance(result, SeparationResult)
                    assert result.num_artists == 1
                    assert len(result.vocals) == len(audio)
                    assert len(result.instrumental) == len(audio)
                    assert result.total_duration == len(audio) / sr

    def test_auto_queue_training(self, temp_profiles_dir, sample_mixed_audio):
        """Test automatic training job queueing."""
        audio, sr = sample_mixed_audio
        separator = MultiArtistSeparator(
            profiles_dir=temp_profiles_dir,
            device="cpu",
            auto_queue_training=True
        )

        with patch.object(separator, 'separate_vocals') as mock_sep:
            mock_sep.return_value = (
                np.random.randn(len(audio)).astype(np.float32),
                np.random.randn(len(audio)).astype(np.float32)
            )

            with patch.object(separator, 'diarize_vocals') as mock_diar:
                mock_diar.return_value = []

                with patch.object(separator, 'match_segments_to_profiles') as mock_match:
                    # Return new profile with long duration
                    long_audio = np.random.randn(sr * 35).astype(np.float32)  # 35 seconds
                    mock_match.return_value = (
                        {
                            'new_profile': [
                                ArtistSegment(
                                    profile_id='new_profile',
                                    profile_name='New Artist',
                                    start=0.0,
                                    end=35.0,
                                    audio=long_audio,
                                    sample_rate=sr,
                                    embedding=np.zeros(256),
                                    similarity=1.0,
                                    is_new_profile=True,
                                )
                            ]
                        },
                        ['new_profile']
                    )

                    with patch.object(separator, '_queue_training_for_profiles') as mock_queue:
                        result = separator.separate_and_route(audio, sr)

                        # Training should be queued for new profile
                        mock_queue.assert_called_once()


# ============================================================================
# Test Training Job Queueing
# ============================================================================

class TestTrainingJobQueue:
    """Test training job queueing functionality."""

    def test_queue_training_sufficient_duration(self, temp_profiles_dir):
        """Test queueing training when duration threshold is met."""
        separator = MultiArtistSeparator(profiles_dir=temp_profiles_dir, device="cpu")

        # Create segments with >30s total duration
        segments = [
            ArtistSegment(
                profile_id='profile1',
                profile_name='Artist 1',
                start=0.0,
                end=35.0,
                audio=np.zeros(44100 * 35),
                sample_rate=44100,
                embedding=np.zeros(256),
                similarity=0.9,
            )
        ]

        artists = {'profile1': segments}

        with patch.object(separator, '_load_job_manager'):
            mock_jm = MagicMock()
            mock_job = MagicMock()
            mock_job.job_id = "job123"
            mock_jm.auto_queue_training.return_value = mock_job
            separator._job_manager = mock_jm

            separator._queue_training_for_profiles(artists)

            mock_jm.auto_queue_training.assert_called_once()

    def test_queue_training_insufficient_duration(self, temp_profiles_dir):
        """Test no queueing when duration is too short."""
        separator = MultiArtistSeparator(profiles_dir=temp_profiles_dir, device="cpu")

        # Create segments with <30s total duration
        segments = [
            ArtistSegment(
                profile_id='profile1',
                profile_name='Artist 1',
                start=0.0,
                end=10.0,  # Only 10 seconds
                audio=np.zeros(44100 * 10),
                sample_rate=44100,
                embedding=np.zeros(256),
                similarity=0.9,
            )
        ]

        artists = {'profile1': segments}

        with patch.object(separator, '_load_job_manager'):
            mock_jm = MagicMock()
            separator._job_manager = mock_jm

            separator._queue_training_for_profiles(artists)

            # Should not queue training
            mock_jm.auto_queue_training.assert_not_called()

    def test_queue_training_no_job_manager(self, temp_profiles_dir):
        """Test graceful handling when job manager unavailable."""
        separator = MultiArtistSeparator(profiles_dir=temp_profiles_dir, device="cpu")

        segments = [
            ArtistSegment(
                profile_id='profile1',
                profile_name='Artist 1',
                start=0.0,
                end=35.0,
                audio=np.zeros(44100 * 35),
                sample_rate=44100,
                embedding=np.zeros(256),
                similarity=0.9,
            )
        ]

        with patch.object(separator, '_load_job_manager'):
            separator._job_manager = None  # Not available

            # Should not raise exception
            separator._queue_training_for_profiles({'profile1': segments})


# ============================================================================
# Test File Processing
# ============================================================================

class TestFileProcessing:
    """Test audio file processing."""

    def test_process_file(self, temp_profiles_dir, tmp_path):
        """Test processing a single audio file."""
        # Create test audio file
        audio_path = tmp_path / "test.wav"
        audio = np.random.randn(44100 * 5).astype(np.float32)

        with patch('torchaudio.load') as mock_load:
            waveform = torch.from_numpy(audio).unsqueeze(0)
            mock_load.return_value = (waveform, 44100)

            separator = MultiArtistSeparator(profiles_dir=temp_profiles_dir, device="cpu")

            with patch.object(separator, 'separate_and_route') as mock_separate:
                mock_result = MagicMock(spec=SeparationResult)
                mock_result.num_artists = 1
                mock_separate.return_value = mock_result

                result = separator.process_file(str(audio_path))

                assert isinstance(result, SeparationResult)
                mock_separate.assert_called_once()

    def test_process_file_stereo_to_mono(self, temp_profiles_dir, tmp_path):
        """Test stereo audio is converted to mono."""
        audio_path = tmp_path / "stereo.wav"

        with patch('torchaudio.load') as mock_load:
            # Stereo waveform
            stereo = torch.randn(2, 44100 * 5)
            mock_load.return_value = (stereo, 44100)

            separator = MultiArtistSeparator(profiles_dir=temp_profiles_dir, device="cpu")

            with patch.object(separator, 'separate_and_route') as mock_separate:
                mock_result = MagicMock(spec=SeparationResult)
                mock_separate.return_value = mock_result

                separator.process_file(str(audio_path))

                # Check that mono audio was passed
                call_args = mock_separate.call_args
                audio_arg = call_args[1]['audio']
                assert audio_arg.ndim == 1


# ============================================================================
# Test Batch Processing
# ============================================================================

class TestBatchProcessing:
    """Test batch file processing."""

    def test_process_batch_success(self, temp_profiles_dir, tmp_path):
        """Test successful batch processing."""
        audio_files = [
            str(tmp_path / "song1.wav"),
            str(tmp_path / "song2.wav"),
        ]

        separator = MultiArtistSeparator(profiles_dir=temp_profiles_dir, device="cpu")

        with patch.object(separator, 'process_file') as mock_process:
            mock_result = MagicMock(spec=SeparationResult)
            mock_result.num_artists = 1
            mock_result.new_profiles_created = []
            mock_result.artists = {
                'artist1': [
                    MagicMock(
                        profile_name='Artist 1',
                        duration=5.0,
                        audio=np.zeros(44100 * 5)
                    )
                ]
            }
            mock_process.return_value = mock_result

            result = separator.process_batch(audio_files)

            assert result['files_processed'] == 2
            assert result['files_successful'] == 2
            assert result['artists_found'] == 1

    def test_process_batch_with_errors(self, temp_profiles_dir, tmp_path):
        """Test batch processing with some files failing."""
        audio_files = [
            str(tmp_path / "song1.wav"),
            str(tmp_path / "song2.wav"),
        ]

        separator = MultiArtistSeparator(profiles_dir=temp_profiles_dir, device="cpu")

        with patch.object(separator, 'process_file') as mock_process:
            # First file succeeds, second fails
            mock_result = MagicMock(spec=SeparationResult)
            mock_result.num_artists = 1
            mock_result.new_profiles_created = []
            mock_result.artists = {}

            mock_process.side_effect = [
                mock_result,
                Exception("Failed to load file")
            ]

            result = separator.process_batch(audio_files)

            assert result['files_processed'] == 2
            assert result['files_successful'] == 1
            assert len(result['file_results']) == 2
            assert result['file_results'][1]['success'] is False

    def test_process_batch_artist_aggregation(self, temp_profiles_dir, tmp_path):
        """Test artist aggregation across files."""
        audio_files = [str(tmp_path / "song1.wav")]

        separator = MultiArtistSeparator(profiles_dir=temp_profiles_dir, device="cpu")

        with patch.object(separator, 'process_file') as mock_process:
            segment = MagicMock()
            segment.profile_name = 'Artist 1'
            segment.duration = 30.0
            segment.audio = MagicMock()
            segment.audio.tobytes = MagicMock(return_value=b'unique_data')

            mock_result = MagicMock(spec=SeparationResult)
            mock_result.num_artists = 1
            mock_result.new_profiles_created = []
            mock_result.artists = {'artist1': [segment]}
            mock_process.return_value = mock_result

            result = separator.process_batch(audio_files, aggregate_by_artist=True)

            assert 'artist1' in result['artist_summary']
            assert result['artist_summary']['artist1']['total_segments'] == 1
            assert result['artist_summary']['artist1']['total_duration'] == 30.0


# ============================================================================
# Test Saving Segments
# ============================================================================

class TestSavingSegments:
    """Test saving separated artist segments."""

    def test_save_artist_segments(self, temp_profiles_dir, tmp_path, sample_vocals, sample_instrumental):
        """Test saving artist segments to files."""
        vocals, sr = sample_vocals
        instrumental, _ = sample_instrumental

        segments = [
            ArtistSegment(
                profile_id='artist1',
                profile_name='Artist 1',
                start=0.0,
                end=5.0,
                audio=vocals[:sr*5],
                sample_rate=sr,
                embedding=np.zeros(256),
                similarity=0.9,
            )
        ]

        result = SeparationResult(
            artists={'artist1': segments},
            vocals=vocals,
            instrumental=instrumental,
            sample_rate=sr,
            total_duration=10.0,
            num_artists=1,
        )

        separator = MultiArtistSeparator(profiles_dir=temp_profiles_dir, device="cpu")
        output_dir = tmp_path / "output"

        with patch('soundfile.write') as mock_write:
            saved_files = separator.save_artist_segments(result, output_dir)

            assert 'artist1' in saved_files
            assert len(saved_files['artist1']) == 1
            # Should save segments + instrumental
            assert mock_write.call_count == 2

    def test_save_creates_directories(self, temp_profiles_dir, tmp_path, sample_vocals, sample_instrumental):
        """Test that output directories are created."""
        vocals, sr = sample_vocals
        instrumental, _ = sample_instrumental

        segments = [
            ArtistSegment(
                profile_id='artist1',
                profile_name='Artist 1',
                start=0.0,
                end=5.0,
                audio=vocals[:sr*5],
                sample_rate=sr,
                embedding=np.zeros(256),
                similarity=0.9,
            )
        ]

        result = SeparationResult(
            artists={'artist1': segments},
            vocals=vocals,
            instrumental=instrumental,
            sample_rate=sr,
            total_duration=10.0,
            num_artists=1,
        )

        separator = MultiArtistSeparator(profiles_dir=temp_profiles_dir, device="cpu")
        output_dir = tmp_path / "new_output"

        with patch('soundfile.write'):
            separator.save_artist_segments(result, output_dir)

            assert output_dir.exists()
            assert (output_dir / 'artist1').exists()


# ============================================================================
# Test Global Instance
# ============================================================================

class TestGlobalInstance:
    """Test global separator instance management."""

    def test_get_multi_artist_separator(self):
        """Test getting global separator instance."""
        # Reset global instance
        import auto_voice.audio.multi_artist_separator as mas
        mas._global_separator = None

        separator1 = get_multi_artist_separator()
        separator2 = get_multi_artist_separator()

        # Should return same instance
        assert separator1 is separator2

    def test_global_instance_initialization(self):
        """Test global instance is properly initialized."""
        import auto_voice.audio.multi_artist_separator as mas
        mas._global_separator = None

        separator = get_multi_artist_separator()

        assert isinstance(separator, MultiArtistSeparator)
        assert separator.auto_create_profiles is True
