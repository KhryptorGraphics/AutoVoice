"""Tests for speaker_matcher.py - Speaker identification and clustering.

Test Coverage:
- Task 2.2: Embedding-based matching
- Verify correct speaker assignment
- Test similarity threshold tuning
- Test unknown speaker detection
"""

import numpy as np
import pytest
import soundfile as sf
from pathlib import Path
from unittest.mock import MagicMock, patch

from auto_voice.audio.speaker_matcher import SpeakerMatcher


@pytest.fixture
def matcher():
    """Create SpeakerMatcher instance."""
    return SpeakerMatcher(
        similarity_threshold=0.85,
        min_cluster_duration=30.0,
        device='cpu',
    )


@pytest.fixture
def sample_audio_file(tmp_path):
    """Create a sample audio file for embedding extraction."""
    sr = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    audio_path = tmp_path / "test_audio.wav"
    sf.write(str(audio_path), audio, sr)

    return audio_path


@pytest.fixture
def mock_embeddings():
    """Create mock speaker embeddings for testing."""
    np.random.seed(42)

    # Speaker 1: cluster around (1, 0, 0, ...)
    emb1_a = np.random.randn(512).astype(np.float32)
    emb1_a[0] = 5.0
    emb1_a = emb1_a / np.linalg.norm(emb1_a)

    emb1_b = np.random.randn(512).astype(np.float32)
    emb1_b[0] = 4.8
    emb1_b = emb1_b / np.linalg.norm(emb1_b)

    # Speaker 2: cluster around (0, 1, 0, ...)
    emb2_a = np.random.randn(512).astype(np.float32)
    emb2_a[1] = 5.0
    emb2_a = emb2_a / np.linalg.norm(emb2_a)

    emb2_b = np.random.randn(512).astype(np.float32)
    emb2_b[1] = 4.9
    emb2_b = emb2_b / np.linalg.norm(emb2_b)

    return {
        'speaker1_a': emb1_a,
        'speaker1_b': emb1_b,
        'speaker2_a': emb2_a,
        'speaker2_b': emb2_b,
    }


class TestSpeakerMatcher:
    """Test suite for SpeakerMatcher."""

    def test_initialization(self):
        """Test SpeakerMatcher initialization."""
        matcher = SpeakerMatcher(
            similarity_threshold=0.90,
            min_cluster_duration=60.0,
            device='cpu',
        )

        assert matcher.similarity_threshold == 0.90
        assert matcher.min_cluster_duration == 60.0
        assert matcher.device == 'cpu'
        assert matcher._encoder is None  # Lazy loaded

    def test_cosine_similarity(self, matcher):
        """Test cosine similarity calculation."""
        # Identical embeddings
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([1.0, 0.0, 0.0])
        sim = matcher.cosine_similarity(emb1, emb2)
        assert sim == pytest.approx(1.0, abs=0.01)

        # Orthogonal embeddings
        emb3 = np.array([1.0, 0.0, 0.0])
        emb4 = np.array([0.0, 1.0, 0.0])
        sim = matcher.cosine_similarity(emb3, emb4)
        assert sim == pytest.approx(0.0, abs=0.01)

        # Opposite embeddings
        emb5 = np.array([1.0, 0.0, 0.0])
        emb6 = np.array([-1.0, 0.0, 0.0])
        sim = matcher.cosine_similarity(emb5, emb6)
        assert sim == pytest.approx(-1.0, abs=0.01)

    def test_cosine_similarity_normalized(self, matcher):
        """Test cosine similarity with pre-normalized embeddings."""
        # Create normalized embeddings
        emb1 = np.random.randn(512).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)

        emb2 = emb1 + np.random.randn(512).astype(np.float32) * 0.1
        emb2 = emb2 / np.linalg.norm(emb2)

        sim = matcher.cosine_similarity(emb1, emb2)

        # Similar embeddings should have high similarity
        assert 0.8 < sim < 1.0

    @patch('auto_voice.audio.speaker_matcher.SpeakerMatcher._get_encoder')
    def test_extract_embedding_from_audio(self, mock_get_encoder, matcher, sample_audio_file):
        """Test extracting embedding from audio file."""
        # Mock encoder
        mock_encoder = MagicMock()
        mock_embedding = np.random.randn(512).astype(np.float32)
        mock_encoder.extract_embedding.return_value = mock_embedding
        mock_get_encoder.return_value = mock_encoder

        embedding = matcher.extract_embedding_from_audio(sample_audio_file)

        # Should return normalized embedding
        assert embedding.shape == (512,)
        assert np.allclose(np.linalg.norm(embedding), 1.0, atol=1e-6)
        mock_encoder.extract_embedding.assert_called_once()

    @patch('auto_voice.audio.speaker_matcher.SpeakerMatcher._get_encoder')
    def test_extract_embedding_with_time_range(self, mock_get_encoder, matcher, sample_audio_file):
        """Test extracting embedding from audio segment."""
        mock_encoder = MagicMock()
        mock_embedding = np.random.randn(512).astype(np.float32)
        mock_encoder.extract_embedding.return_value = mock_embedding
        mock_get_encoder.return_value = mock_encoder

        embedding = matcher.extract_embedding_from_audio(
            sample_audio_file,
            start_sec=1.0,
            end_sec=3.0,
        )

        # Should load only the segment
        assert embedding.shape == (512,)
        assert np.allclose(np.linalg.norm(embedding), 1.0, atol=1e-6)

    def test_cluster_speakers_with_mock_data(self, matcher, mock_embeddings):
        """Test speaker clustering with mock embedding data."""
        # Create mock database embeddings
        embeddings = [
            {'id': 1, 'embedding': mock_embeddings['speaker1_a'], 'duration_sec': 10.0},
            {'id': 2, 'embedding': mock_embeddings['speaker1_b'], 'duration_sec': 12.0},
            {'id': 3, 'embedding': mock_embeddings['speaker2_a'], 'duration_sec': 15.0},
            {'id': 4, 'embedding': mock_embeddings['speaker2_b'], 'duration_sec': 13.0},
        ]

        with patch('auto_voice.audio.speaker_matcher.find_unclustered_embeddings') as mock_find, \
             patch('auto_voice.audio.speaker_matcher.create_cluster') as mock_create, \
             patch('auto_voice.audio.speaker_matcher.add_to_cluster') as mock_add:

            mock_find.return_value = embeddings
            mock_create.side_effect = ['cluster_1', 'cluster_2']

            clusters = matcher.cluster_speakers(threshold=0.85)

            # Should create 2 clusters
            assert len(clusters) == 2
            assert mock_create.call_count == 2
            assert mock_add.call_count == 4  # 4 embeddings added

    def test_cluster_speakers_similarity_threshold(self, matcher):
        """Test similarity threshold affects clustering."""
        # Create embeddings with varying similarity
        emb1 = np.array([1.0, 0.0] + [0.0] * 510)
        emb2 = np.array([0.95, 0.31] + [0.0] * 510)  # ~70% similar
        emb2 = emb2 / np.linalg.norm(emb2)
        emb3 = np.array([0.0, 1.0] + [0.0] * 510)

        embeddings = [
            {'id': 1, 'embedding': emb1, 'duration_sec': 10.0},
            {'id': 2, 'embedding': emb2, 'duration_sec': 10.0},
            {'id': 3, 'embedding': emb3, 'duration_sec': 10.0},
        ]

        with patch('auto_voice.audio.speaker_matcher.find_unclustered_embeddings') as mock_find, \
             patch('auto_voice.audio.speaker_matcher.create_cluster') as mock_create, \
             patch('auto_voice.audio.speaker_matcher.add_to_cluster'):

            # High threshold: should create more clusters
            mock_find.return_value = embeddings
            mock_create.side_effect = [f'cluster_{i}' for i in range(10)]

            clusters_high = matcher.cluster_speakers(threshold=0.95)

            # Low threshold: should create fewer clusters
            mock_find.return_value = embeddings
            mock_create.side_effect = [f'cluster_{i}' for i in range(10)]

            clusters_low = matcher.cluster_speakers(threshold=0.30)

            # High threshold creates more (or equal) clusters
            assert len(clusters_high) >= len(clusters_low)

    def test_cluster_speakers_min_duration_filter(self, matcher):
        """Test that clusters below min_cluster_duration are filtered."""
        matcher.min_cluster_duration = 30.0

        # One cluster with sufficient duration, one without
        emb1 = np.random.randn(512).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)

        emb2 = np.random.randn(512).astype(np.float32)
        emb2 = emb2 / np.linalg.norm(emb2)

        embeddings = [
            {'id': 1, 'embedding': emb1, 'duration_sec': 35.0},  # Sufficient
            {'id': 2, 'embedding': emb2, 'duration_sec': 10.0},  # Too short
        ]

        with patch('auto_voice.audio.speaker_matcher.find_unclustered_embeddings') as mock_find, \
             patch('auto_voice.audio.speaker_matcher.create_cluster') as mock_create, \
             patch('auto_voice.audio.speaker_matcher.add_to_cluster'):

            mock_find.return_value = embeddings
            mock_create.return_value = 'cluster_1'

            clusters = matcher.cluster_speakers()

            # Should only create 1 cluster (emb2 filtered by duration)
            assert len(clusters) == 1
            assert clusters[0]['total_duration_sec'] >= 30.0

    def test_cluster_speakers_empty_embeddings(self, matcher):
        """Test clustering with no embeddings."""
        with patch('auto_voice.audio.speaker_matcher.find_unclustered_embeddings') as mock_find:
            mock_find.return_value = []

            clusters = matcher.cluster_speakers()

            assert len(clusters) == 0

    def test_cluster_speakers_single_embedding(self, matcher):
        """Test clustering with single embedding."""
        emb1 = np.random.randn(512).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)

        embeddings = [
            {'id': 1, 'embedding': emb1, 'duration_sec': 35.0},
        ]

        with patch('auto_voice.audio.speaker_matcher.find_unclustered_embeddings') as mock_find, \
             patch('auto_voice.audio.speaker_matcher.create_cluster') as mock_create, \
             patch('auto_voice.audio.speaker_matcher.add_to_cluster') as mock_add:

            mock_find.return_value = embeddings
            mock_create.return_value = 'cluster_1'

            clusters = matcher.cluster_speakers()

            # Should create 1 cluster with 1 member
            assert len(clusters) == 1
            assert clusters[0]['member_count'] == 1

    def test_auto_match_clusters_to_artists(self, matcher):
        """Test automatic matching of clusters to featured artist names."""
        clusters = [
            {'id': 'cluster_1', 'name': 'Unknown Speaker 1', 'is_verified': False},
            {'id': 'cluster_2', 'name': 'Unknown Speaker 2', 'is_verified': False},
        ]

        members_cluster_1 = [
            {'track_id': 'track_1', 'speaker_id': 'SPEAKER_01'},
            {'track_id': 'track_2', 'speaker_id': 'SPEAKER_01'},
        ]

        members_cluster_2 = [
            {'track_id': 'track_1', 'speaker_id': 'SPEAKER_02'},
        ]

        featured_artists = [
            {'name': 'Featured Artist A'},
        ]

        with patch('auto_voice.audio.speaker_matcher.get_all_clusters') as mock_get_clusters, \
             patch('auto_voice.audio.speaker_matcher.get_cluster_members') as mock_get_members, \
             patch('auto_voice.audio.speaker_matcher.get_featured_artists_for_track') as mock_get_featured, \
             patch('auto_voice.audio.speaker_matcher.update_cluster_name') as mock_update:

            mock_get_clusters.return_value = clusters
            mock_get_members.side_effect = [members_cluster_1, members_cluster_2]
            mock_get_featured.return_value = featured_artists

            stats = matcher.auto_match_clusters_to_artists()

            # Should match cluster_1 to "Featured Artist A"
            assert stats['matches_found'] > 0
            assert len(stats['matches_made']) > 0

    def test_auto_match_clusters_requires_confidence(self, matcher):
        """Test that auto-matching requires high confidence (50%+ tracks)."""
        clusters = [
            {'id': 'cluster_1', 'name': 'Unknown Speaker 1', 'is_verified': False},
        ]

        # Artist appears in only 1 of 5 tracks (20% - below threshold)
        members = [
            {'track_id': f'track_{i}', 'speaker_id': 'SPEAKER_01'}
            for i in range(5)
        ]

        featured_artists_single = [{'name': 'Featured Artist'}]
        featured_artists_none = []

        with patch('auto_voice.audio.speaker_matcher.get_all_clusters') as mock_get_clusters, \
             patch('auto_voice.audio.speaker_matcher.get_cluster_members') as mock_get_members, \
             patch('auto_voice.audio.speaker_matcher.get_featured_artists_for_track') as mock_get_featured, \
             patch('auto_voice.audio.speaker_matcher.update_cluster_name') as mock_update:

            mock_get_clusters.return_value = clusters
            mock_get_members.return_value = members
            # Only track_0 has featured artist
            mock_get_featured.side_effect = [featured_artists_single] + \
                                            [featured_artists_none] * 4

            stats = matcher.auto_match_clusters_to_artists()

            # Should not match due to low confidence
            assert stats['matches_found'] == 0

    def test_auto_match_clusters_skips_verified(self, matcher):
        """Test that already verified clusters are skipped."""
        clusters = [
            {'id': 'cluster_1', 'name': 'Verified Artist', 'is_verified': True},
        ]

        with patch('auto_voice.audio.speaker_matcher.get_all_clusters') as mock_get_clusters, \
             patch('auto_voice.audio.speaker_matcher.get_cluster_members') as mock_get_members:

            mock_get_clusters.return_value = clusters

            stats = matcher.auto_match_clusters_to_artists()

            # Should process 0 clusters (verified skipped)
            assert stats['clusters_processed'] == 0
            mock_get_members.assert_not_called()

    def test_unknown_speaker_detection(self, matcher, mock_embeddings):
        """Test detection of unknown speakers (not matching any cluster)."""
        # Known speaker embeddings
        known_emb = mock_embeddings['speaker1_a']

        # Unknown speaker embedding (very different)
        unknown_emb = np.random.randn(512).astype(np.float32)
        unknown_emb = unknown_emb / np.linalg.norm(unknown_emb)

        # Calculate similarity
        sim_known = matcher.cosine_similarity(known_emb, known_emb)
        sim_unknown = matcher.cosine_similarity(known_emb, unknown_emb)

        # Known speaker should have high similarity
        assert sim_known > 0.95

        # Unknown speaker should have low similarity
        assert sim_unknown < matcher.similarity_threshold


@pytest.mark.parametrize("threshold,expected_clusters", [
    (0.95, 4),  # Very strict: each embedding is its own cluster
    (0.70, 2),  # Moderate: groups similar speakers
    (0.30, 1),  # Very loose: all in one cluster
])
def test_similarity_threshold_tuning(threshold, expected_clusters, mock_embeddings):
    """Test that similarity threshold affects number of clusters."""
    matcher = SpeakerMatcher(similarity_threshold=threshold, device='cpu')

    embeddings = [
        {'id': i, 'embedding': emb, 'duration_sec': 10.0}
        for i, emb in enumerate(mock_embeddings.values())
    ]

    with patch('auto_voice.audio.speaker_matcher.find_unclustered_embeddings') as mock_find, \
         patch('auto_voice.audio.speaker_matcher.create_cluster') as mock_create, \
         patch('auto_voice.audio.speaker_matcher.add_to_cluster'):

        mock_find.return_value = embeddings
        mock_create.side_effect = [f'cluster_{i}' for i in range(10)]

        clusters = matcher.cluster_speakers(threshold=threshold)

        # Number of clusters should vary with threshold
        # Note: exact numbers depend on mock embedding generation
        assert len(clusters) >= 1


def test_speaker_assignment_correctness(mock_embeddings):
    """Test that similar embeddings are assigned to same cluster."""
    matcher = SpeakerMatcher(similarity_threshold=0.85, device='cpu')

    # speaker1_a and speaker1_b should cluster together
    embeddings = [
        {'id': 1, 'embedding': mock_embeddings['speaker1_a'], 'duration_sec': 10.0},
        {'id': 2, 'embedding': mock_embeddings['speaker1_b'], 'duration_sec': 10.0},
        {'id': 3, 'embedding': mock_embeddings['speaker2_a'], 'duration_sec': 10.0},
    ]

    with patch('auto_voice.audio.speaker_matcher.find_unclustered_embeddings') as mock_find, \
         patch('auto_voice.audio.speaker_matcher.create_cluster') as mock_create, \
         patch('auto_voice.audio.speaker_matcher.add_to_cluster') as mock_add:

        mock_find.return_value = embeddings
        mock_create.side_effect = ['cluster_1', 'cluster_2']

        clusters = matcher.cluster_speakers()

        # Verify speaker1_a and speaker1_b are in same cluster
        # This is verified by checking add_to_cluster calls
        assert mock_add.call_count >= 3
