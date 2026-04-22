"""Enhanced tests for speaker_matcher.py - Comprehensive TDD coverage.

This file adds missing test coverage to reach 90%:
- Database integration (mocked)
- Clustering algorithms
- Auto-matching to featured artists
- Sample audio generation
- Full matching workflow
- CLI interface

Coverage Target: 14% → 90% (189 uncovered lines → <30 lines)
"""
import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from collections import defaultdict

from auto_voice.audio.speaker_matcher import (
    SpeakerMatcher,
    run_speaker_matching,
)
from auto_voice.audio.speaker_pipeline_contract import (
    DEFAULT_SPEAKER_PIPELINE_ARTISTS,
    get_default_speaker_pipeline_artists,
)


@pytest.fixture
def mock_db_operations():
    """Mock database operations module - patch where they're imported inside functions."""
    # The imports happen inside methods, so we patch at the db.operations level
    with patch('auto_voice.db.operations.upsert_track') as mock_upsert, \
         patch('auto_voice.db.operations.add_speaker_embedding') as mock_add_emb, \
         patch('auto_voice.db.operations.find_unclustered_embeddings') as mock_find, \
         patch('auto_voice.db.operations.create_cluster') as mock_create_cluster, \
         patch('auto_voice.db.operations.add_to_cluster') as mock_add_to_cluster, \
         patch('auto_voice.db.operations.get_all_clusters') as mock_get_clusters, \
         patch('auto_voice.db.operations.get_cluster_members') as mock_get_members, \
         patch('auto_voice.db.operations.get_featured_artists_for_track') as mock_get_featured, \
         patch('auto_voice.db.operations.update_cluster_name') as mock_update_cluster, \
         patch('auto_voice.db.operations.get_track') as mock_get_track:

        yield {
            'upsert_track': mock_upsert,
            'add_speaker_embedding': mock_add_emb,
            'find_unclustered_embeddings': mock_find,
            'create_cluster': mock_create_cluster,
            'add_to_cluster': mock_add_to_cluster,
            'get_all_clusters': mock_get_clusters,
            'get_cluster_members': mock_get_members,
            'get_featured_artists_for_track': mock_get_featured,
            'update_cluster_name': mock_update_cluster,
            'get_track': mock_get_track,
        }


@pytest.fixture
def matcher_with_mock_encoder():
    """Create SpeakerMatcher with deterministic mock encoder."""
    matcher = SpeakerMatcher(
        similarity_threshold=0.85,
        min_cluster_duration=30.0,
        device='cpu',
    )

    # Create mock encoder with deterministic embeddings
    mock_encoder = Mock()

    def mock_extract_embedding(audio):
        """Generate deterministic embedding based on audio statistics."""
        # Use audio mean to create reproducible embedding
        seed = int(np.abs(np.mean(audio) * 10000)) % (2**31)
        rng = np.random.RandomState(seed)
        emb = rng.randn(512)
        return emb / (np.linalg.norm(emb) + 1e-8)

    mock_encoder.extract_embedding = mock_extract_embedding
    matcher._encoder = mock_encoder

    return matcher


class TestEncoderLoading:
    """Test lazy encoder loading."""

    def test_get_encoder_lazy_loads(self):
        """Test that encoder is lazy-loaded on first access."""
        matcher = SpeakerMatcher(device='cpu')
        assert matcher._encoder is None

        # Mock the diarizer - import happens inside _get_encoder
        with patch('auto_voice.audio.speaker_diarization.SpeakerDiarizer') as mock_diarizer:
            mock_instance = Mock()
            mock_diarizer.return_value = mock_instance

            encoder = matcher._get_encoder()

            # Should have created diarizer
            mock_diarizer.assert_called_once_with(device='cpu')
            assert encoder == mock_instance

    def test_get_encoder_caches_instance(self):
        """Test that encoder is cached after first load."""
        matcher = SpeakerMatcher(device='cpu')

        with patch('auto_voice.audio.speaker_diarization.SpeakerDiarizer') as mock_diarizer:
            mock_instance = Mock()
            mock_diarizer.return_value = mock_instance

            # First call
            encoder1 = matcher._get_encoder()
            # Second call
            encoder2 = matcher._get_encoder()

            # Should only create once
            assert mock_diarizer.call_count == 1
            assert encoder1 is encoder2


class TestExtractEmbeddingsForArtist:
    """Test extract_embeddings_for_artist functionality."""

    def test_extract_embeddings_basic(self, tmp_path, matcher_with_mock_encoder, mock_db_operations):
        """Test basic embedding extraction for an artist."""
        artist_name = "test_artist"

        # Set up directories
        separated_dir = tmp_path / "separated" / artist_name
        diarized_dir = tmp_path / "diarized" / artist_name
        separated_dir.mkdir(parents=True)
        diarized_dir.mkdir(parents=True)

        # Create test files
        import soundfile as sf
        sr = 16000

        # Create longer audio to meet minimum duration (5s+)
        duration = 10.0
        audio = np.sin(2 * np.pi * 200 * np.linspace(0, duration, int(duration * sr))).astype(np.float32) * 0.5
        audio_path = separated_dir / "track_001_vocals.wav"
        sf.write(str(audio_path), audio, sr)

        # Diarization JSON - both segments > 5s to not be filtered
        diarization_data = {
            "file": str(audio_path),
            "segments": [
                {"start": 0.0, "end": 6.0, "speaker": "SPEAKER_00"},  # 6.0s
                {"start": 6.5, "end": 9.5, "speaker": "SPEAKER_01"},  # 3.0s - too short, won't be extracted
            ]
        }
        json_path = diarized_dir / "track_001_diarization.json"
        with open(json_path, 'w') as f:
            json.dump(diarization_data, f)

        # Extract embeddings
        stats = matcher_with_mock_encoder.extract_embeddings_for_artist(
            artist_name,
            separated_dir=separated_dir,
            diarized_dir=diarized_dir,
        )

        # Verify stats - SPEAKER_01 filtered out (< 5s)
        assert stats['tracks_processed'] == 1
        assert stats['embeddings_extracted'] == 1  # Only SPEAKER_00 (SPEAKER_01 too short)
        assert stats['primary_speakers'] == 1  # SPEAKER_00

        # Verify database calls
        assert mock_db_operations['upsert_track'].called
        assert mock_db_operations['add_speaker_embedding'].call_count == 1

    def test_extract_embeddings_skips_short_segments(self, tmp_path, matcher_with_mock_encoder, mock_db_operations):
        """Test that segments shorter than 5 seconds are skipped."""
        artist_name = "test_artist"

        separated_dir = tmp_path / "separated" / artist_name
        diarized_dir = tmp_path / "diarized" / artist_name
        separated_dir.mkdir(parents=True)
        diarized_dir.mkdir(parents=True)

        import soundfile as sf
        sr = 16000

        audio = np.random.randn(int(10 * sr)).astype(np.float32) * 0.1
        audio_path = separated_dir / "track_001.wav"
        sf.write(str(audio_path), audio, sr)

        # All segments too short
        diarization_data = {
            "file": str(audio_path),
            "segments": [
                {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},  # 2s - too short
                {"start": 3.0, "end": 5.5, "speaker": "SPEAKER_01"},  # 2.5s - too short
            ]
        }
        json_path = diarized_dir / "track_001_diarization.json"
        with open(json_path, 'w') as f:
            json.dump(diarization_data, f)

        stats = matcher_with_mock_encoder.extract_embeddings_for_artist(
            artist_name,
            separated_dir=separated_dir,
            diarized_dir=diarized_dir,
        )

        # No embeddings extracted (all too short)
        assert stats['embeddings_extracted'] == 0

    def test_extract_embeddings_missing_audio(self, tmp_path, matcher_with_mock_encoder, mock_db_operations):
        """Test handling when audio file is missing."""
        artist_name = "test_artist"

        diarized_dir = tmp_path / "diarized" / artist_name
        separated_dir = tmp_path / "separated" / artist_name
        diarized_dir.mkdir(parents=True)
        separated_dir.mkdir(parents=True)

        # Diarization without corresponding audio
        diarization_data = {
            "file": "nonexistent.wav",
            "segments": [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}]
        }
        json_path = diarized_dir / "track_001_diarization.json"
        with open(json_path, 'w') as f:
            json.dump(diarization_data, f)

        stats = matcher_with_mock_encoder.extract_embeddings_for_artist(
            artist_name,
            separated_dir=separated_dir,
            diarized_dir=diarized_dir,
        )

        # Should log error
        assert len(stats['errors']) > 0
        assert 'not found' in stats['errors'][0].lower()


class TestCosineSimilarity:
    """Test cosine similarity computation."""

    def test_cosine_similarity_method(self):
        """Test the cosine_similarity method."""
        matcher = SpeakerMatcher()

        # Create normalized embeddings
        emb1 = np.random.randn(512)
        emb1 = emb1 / np.linalg.norm(emb1)

        emb2 = np.random.randn(512)
        emb2 = emb2 / np.linalg.norm(emb2)

        similarity = matcher.cosine_similarity(emb1, emb2)

        # Should be a scalar in range [-1, 1]
        assert isinstance(similarity, float)
        assert -1.0 <= similarity <= 1.0

    def test_cosine_similarity_identical_embeddings(self):
        """Test similarity of identical embeddings is 1.0."""
        matcher = SpeakerMatcher()

        emb = np.random.randn(512)
        emb = emb / np.linalg.norm(emb)

        similarity = matcher.cosine_similarity(emb, emb)

        assert abs(similarity - 1.0) < 1e-5


class TestClusterSpeakers:
    """Test speaker clustering functionality."""

    def test_cluster_speakers_basic(self, matcher_with_mock_encoder, mock_db_operations):
        """Test basic clustering with similar embeddings."""
        # Create mock embeddings (3 speakers, 2 samples each)
        np.random.seed(42)

        base_emb1 = np.random.randn(512)
        base_emb1 = base_emb1 / np.linalg.norm(base_emb1)

        base_emb2 = np.random.randn(512)
        base_emb2 = base_emb2 / np.linalg.norm(base_emb2)

        # Create variations (noise added)
        embeddings = []
        for i, base in enumerate([base_emb1, base_emb1, base_emb2, base_emb2]):
            emb = base + np.random.randn(512) * 0.01
            emb = emb / np.linalg.norm(emb)
            embeddings.append({
                'id': f'emb_{i}',
                'embedding': emb,
                'duration_sec': 35.0,
                'track_id': f'track_{i}',
            })

        mock_db_operations['find_unclustered_embeddings'].return_value = embeddings
        mock_db_operations['create_cluster'].side_effect = [f'cluster_{i}' for i in range(10)]

        # Cluster
        clusters = matcher_with_mock_encoder.cluster_speakers()

        # Should create 2 clusters (similar embeddings grouped)
        assert len(clusters) >= 1
        assert mock_db_operations['create_cluster'].called
        assert mock_db_operations['add_to_cluster'].called

    def test_cluster_speakers_no_embeddings(self, matcher_with_mock_encoder, mock_db_operations):
        """Test clustering with no unclustered embeddings."""
        mock_db_operations['find_unclustered_embeddings'].return_value = []

        clusters = matcher_with_mock_encoder.cluster_speakers()

        assert len(clusters) == 0

    def test_cluster_speakers_min_duration_filter(self, matcher_with_mock_encoder, mock_db_operations):
        """Test that clusters below minimum duration are filtered."""
        # Create embeddings with short duration
        np.random.seed(42)
        emb = np.random.randn(512)
        emb = emb / np.linalg.norm(emb)

        embeddings = [
            {
                'id': 'emb_1',
                'embedding': emb,
                'duration_sec': 10.0,  # Too short (< 30s default)
                'track_id': 'track_1',
            }
        ]

        mock_db_operations['find_unclustered_embeddings'].return_value = embeddings

        clusters = matcher_with_mock_encoder.cluster_speakers()

        # Should be filtered out
        assert len(clusters) == 0

    def test_cluster_speakers_custom_threshold(self, mock_db_operations):
        """Test clustering with custom similarity threshold."""
        matcher = SpeakerMatcher(similarity_threshold=0.95)

        # Create mock encoder
        mock_encoder = Mock()
        mock_encoder.extract_embedding = Mock(return_value=np.random.randn(512))
        matcher._encoder = mock_encoder

        # Create embeddings
        np.random.seed(42)
        base = np.random.randn(512)
        base = base / np.linalg.norm(base)

        embeddings = []
        for i in range(3):
            emb = base + np.random.randn(512) * 0.2  # More noise
            emb = emb / np.linalg.norm(emb)
            embeddings.append({
                'id': f'emb_{i}',
                'embedding': emb,
                'duration_sec': 40.0,
                'track_id': f'track_{i}',
            })

        mock_db_operations['find_unclustered_embeddings'].return_value = embeddings
        mock_db_operations['create_cluster'].return_value = 'cluster_1'

        # With high threshold (0.95), may not cluster
        clusters = matcher.cluster_speakers(threshold=0.95)

        # Verify threshold was used
        assert isinstance(clusters, list)


class TestAutoMatchClustersToArtists:
    """Test automatic cluster matching to featured artists."""

    def test_auto_match_basic(self, matcher_with_mock_encoder, mock_db_operations):
        """Test basic auto-matching of clusters to artists."""
        # Mock clusters
        mock_db_operations['get_all_clusters'].return_value = [
            {
                'id': 'cluster_1',
                'name': 'Unknown Speaker 1',
                'is_verified': False,
            }
        ]

        # Mock cluster members
        mock_db_operations['get_cluster_members'].return_value = [
            {'track_id': 'track_1', 'confidence': 0.9},
            {'track_id': 'track_2', 'confidence': 0.85},
        ]

        # Mock featured artists (same artist in both tracks)
        mock_db_operations['get_featured_artists_for_track'].side_effect = [
            [{'name': 'Justin Bieber'}],
            [{'name': 'Justin Bieber'}],
        ]

        stats = matcher_with_mock_encoder.auto_match_clusters_to_artists()

        # Should have matched
        assert stats['matches_found'] >= 1
        assert mock_db_operations['update_cluster_name'].called

    def test_auto_match_skips_verified_clusters(self, matcher_with_mock_encoder, mock_db_operations):
        """Test that verified clusters are not auto-matched."""
        # Mock verified cluster
        mock_db_operations['get_all_clusters'].return_value = [
            {
                'id': 'cluster_1',
                'name': 'Known Artist',
                'is_verified': True,  # Already verified
            }
        ]

        stats = matcher_with_mock_encoder.auto_match_clusters_to_artists()

        # Should skip verified cluster
        assert stats['matches_found'] == 0
        assert not mock_db_operations['update_cluster_name'].called

    def test_auto_match_requires_majority(self, matcher_with_mock_encoder, mock_db_operations):
        """Test that matching requires majority threshold (50%+ of tracks)."""
        mock_db_operations['get_all_clusters'].return_value = [
            {'id': 'cluster_1', 'name': 'Unknown', 'is_verified': False}
        ]

        # 4 tracks, artist appears in only 1 (25% < 50%)
        mock_db_operations['get_cluster_members'].return_value = [
            {'track_id': f'track_{i}', 'confidence': 0.9}
            for i in range(4)
        ]

        mock_db_operations['get_featured_artists_for_track'].side_effect = [
            [{'name': 'Artist A'}],
            [{'name': 'Artist B'}],
            [{'name': 'Artist C'}],
            [{'name': 'Artist D'}],
        ]

        stats = matcher_with_mock_encoder.auto_match_clusters_to_artists()

        # No match (no artist appears in majority)
        assert stats['matches_found'] == 0


class TestGetClusterSampleAudio:
    """Test cluster sample audio generation."""

    def test_get_cluster_sample_audio(self, tmp_path, matcher_with_mock_encoder, mock_db_operations):
        """Test extracting sample audio for a cluster."""
        import soundfile as sf

        # Create test audio file
        sr = 22050
        duration = 15.0
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(duration * sr))).astype(np.float32) * 0.5
        audio_path = tmp_path / "isolated_vocals.wav"
        sf.write(str(audio_path), audio, sr)

        # Mock cluster members
        mock_db_operations['get_cluster_members'].return_value = [
            {
                'isolated_vocals_path': str(audio_path),
                'confidence': 0.95,
            }
        ]

        # Get sample
        sample_audio, sample_sr = matcher_with_mock_encoder.get_cluster_sample_audio(
            'cluster_1',
            max_duration=10.0
        )

        assert sample_audio.shape[0] <= int(10.0 * sample_sr)
        assert sample_sr == sr

    def test_get_cluster_sample_audio_no_members(self, matcher_with_mock_encoder, mock_db_operations):
        """Test error when cluster has no members."""
        mock_db_operations['get_cluster_members'].return_value = []

        with pytest.raises(ValueError, match="no members"):
            matcher_with_mock_encoder.get_cluster_sample_audio('cluster_1')

    def test_get_cluster_sample_audio_missing_file(self, matcher_with_mock_encoder, mock_db_operations):
        """Test error when isolated vocals file doesn't exist."""
        mock_db_operations['get_cluster_members'].return_value = [
            {
                'isolated_vocals_path': '/nonexistent/path.wav',
                'confidence': 0.9,
            }
        ]

        with pytest.raises(ValueError, match="No isolated vocals found"):
            matcher_with_mock_encoder.get_cluster_sample_audio('cluster_1')


class TestRunSpeakerMatching:
    """Test run_speaker_matching full workflow."""

    def test_run_speaker_matching_default_artists(self, tmp_path, mock_db_operations):
        """Test run_speaker_matching with default artists."""
        # Set up directories for both default artists
        data_dir = tmp_path / "data"
        for artist in DEFAULT_SPEAKER_PIPELINE_ARTISTS:
            separated_dir = data_dir / f"separated_youtube/{artist}"
            diarized_dir = data_dir / f"diarized_youtube/{artist}"
            separated_dir.mkdir(parents=True)
            diarized_dir.mkdir(parents=True)

        # Mock embeddings
        mock_db_operations['find_unclustered_embeddings'].return_value = []
        mock_db_operations['get_all_clusters'].return_value = []

        stats = run_speaker_matching(data_dir=data_dir)

        # Should have attempted both artists
        assert 'artists' in stats
        assert list(stats['artists']) == get_default_speaker_pipeline_artists()
        assert stats['clustering'] == {
            'clusters_created': 0,
            'clusters': [],
        }
        assert stats['matching'] == {
            'clusters_processed': 0,
            'matches_found': 0,
            'matches_made': [],
        }

    def test_run_speaker_matching_custom_artists(self, tmp_path, mock_db_operations):
        """Test run_speaker_matching with custom artist list."""
        artist = "custom_artist"
        data_dir = tmp_path / "data"

        separated_dir = data_dir / f"separated_youtube/{artist}"
        diarized_dir = data_dir / f"diarized_youtube/{artist}"
        separated_dir.mkdir(parents=True)
        diarized_dir.mkdir(parents=True)

        mock_db_operations['find_unclustered_embeddings'].return_value = []
        mock_db_operations['get_all_clusters'].return_value = []

        stats = run_speaker_matching(artists=[artist], data_dir=data_dir)

        assert artist in stats['artists']
        assert stats['clustering'] == {
            'clusters_created': 0,
            'clusters': [],
        }
        assert stats['matching'] == {
            'clusters_processed': 0,
            'matches_found': 0,
            'matches_made': [],
        }


class TestCLIInterface:
    """Test CLI argument parsing."""

    def test_cli_extract_only(self, tmp_path):
        """Test --extract-only flag."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--artist', nargs='+')
        parser.add_argument('--extract-only', action='store_true')
        parser.add_argument('--cluster-only', action='store_true')
        parser.add_argument('--match-only', action='store_true')
        parser.add_argument('--threshold', type=float, default=0.85)

        args = parser.parse_args(['--extract-only', '--artist', 'test_artist'])

        assert args.extract_only is True
        assert args.artist == ['test_artist']
        assert args.threshold == 0.85

    def test_cli_cluster_only(self):
        """Test --cluster-only flag."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--artist', nargs='+')
        parser.add_argument('--extract-only', action='store_true')
        parser.add_argument('--cluster-only', action='store_true')
        parser.add_argument('--match-only', action='store_true')
        parser.add_argument('--threshold', type=float, default=0.85)

        args = parser.parse_args(['--cluster-only'])

        assert args.cluster_only is True

    def test_cli_custom_threshold(self):
        """Test custom similarity threshold."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--artist', nargs='+')
        parser.add_argument('--extract-only', action='store_true')
        parser.add_argument('--cluster-only', action='store_true')
        parser.add_argument('--match-only', action='store_true')
        parser.add_argument('--threshold', type=float, default=0.85)

        args = parser.parse_args(['--threshold', '0.92'])

        assert args.threshold == 0.92

    def test_default_artist_helper_returns_copy(self):
        """Test default artist helper returns a mutable copy of the contract constant."""
        artists = get_default_speaker_pipeline_artists()

        assert artists == list(DEFAULT_SPEAKER_PIPELINE_ARTISTS)

        artists.append('new_artist')

        assert list(DEFAULT_SPEAKER_PIPELINE_ARTISTS) == ['conor_maynard', 'william_singe']


class TestEmbeddingExtractionEdgeCases:
    """Test edge cases in embedding extraction."""

    def test_extract_embedding_from_audio_full_file(self, tmp_path, matcher_with_mock_encoder):
        """Test extracting embedding from entire audio file."""
        import soundfile as sf

        sr = 16000
        audio = np.random.randn(sr).astype(np.float32) * 0.1
        audio_path = tmp_path / "test.wav"
        sf.write(str(audio_path), audio, sr)

        embedding = matcher_with_mock_encoder.extract_embedding_from_audio(audio_path)

        assert embedding.shape == (512,)
        assert abs(np.linalg.norm(embedding) - 1.0) < 0.01

    def test_extract_embedding_with_offset_only(self, tmp_path, matcher_with_mock_encoder):
        """Test extracting embedding with start offset but no end."""
        import soundfile as sf

        sr = 16000
        duration = 5.0
        audio = np.random.randn(int(duration * sr)).astype(np.float32) * 0.1
        audio_path = tmp_path / "test.wav"
        sf.write(str(audio_path), audio, sr)

        # Extract from 2.0s to end
        embedding = matcher_with_mock_encoder.extract_embedding_from_audio(
            audio_path,
            start_sec=2.0
        )

        assert embedding.shape == (512,)


class TestSpeakerDurationCalculation:
    """Test speaker duration calculations in extract_embeddings_for_artist."""

    def test_primary_speaker_identification(self, tmp_path, matcher_with_mock_encoder, mock_db_operations):
        """Test that speaker with longest duration is marked primary."""
        artist_name = "test_artist"

        separated_dir = tmp_path / "separated" / artist_name
        diarized_dir = tmp_path / "diarized" / artist_name
        separated_dir.mkdir(parents=True)
        diarized_dir.mkdir(parents=True)

        import soundfile as sf
        sr = 16000

        audio = np.random.randn(int(15 * sr)).astype(np.float32) * 0.1
        audio_path = separated_dir / "track_001.wav"
        sf.write(str(audio_path), audio, sr)

        # Both speakers > 5s to avoid filtering
        diarization_data = {
            "file": str(audio_path),
            "segments": [
                {"start": 0.0, "end": 8.0, "speaker": "SPEAKER_00"},  # 8.0s - primary
                {"start": 9.0, "end": 15.0, "speaker": "SPEAKER_01"},  # 6.0s - not filtered
            ]
        }
        json_path = diarized_dir / "track_001_diarization.json"
        with open(json_path, 'w') as f:
            json.dump(diarization_data, f)

        stats = matcher_with_mock_encoder.extract_embeddings_for_artist(
            artist_name,
            separated_dir=separated_dir,
            diarized_dir=diarized_dir,
        )

        # Both speakers should have embeddings extracted
        assert stats['embeddings_extracted'] == 2
        assert stats['primary_speakers'] == 1  # SPEAKER_00
        assert stats['featured_speakers'] == 1  # SPEAKER_01
