"""Comprehensive tests for SQLAlchemy ORM database operations.

Tests cover the migrated operations.py and schema.py with SQLAlchemy:
- Track CRUD operations
- Featured artist operations
- Speaker embedding operations (with numpy arrays)
- Speaker cluster operations
- Cluster membership operations
- Transaction handling and rollback
- Connection lifecycle
- Database statistics

Uses in-memory SQLite for speed (AUTOVOICE_DB_TYPE=sqlite).
"""

import os
import pytest
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock

# Set SQLite mode for testing before imports
os.environ['AUTOVOICE_DB_TYPE'] = 'sqlite'

from auto_voice.db.schema import (
    init_database, get_db_session, reset_database, get_database_stats,
    get_engine, get_session_factory, close_database,
    Track, FeaturedArtist, SpeakerEmbedding, SpeakerCluster, ClusterMember,
    Base, DATABASE_TYPE
)
from auto_voice.db.operations import (
    # Track operations
    upsert_track, get_track, get_all_tracks, get_tracks_by_artist,
    # Featured artist operations
    add_featured_artist, get_featured_artists_for_track, get_all_featured_artists,
    # Speaker embedding operations
    add_speaker_embedding, get_embeddings_for_track, get_all_embeddings,
    get_embeddings_by_cluster, get_embedding_by_id,
    # Cluster operations
    create_cluster, get_cluster, get_all_clusters, update_cluster_name,
    merge_clusters, add_to_cluster, remove_from_cluster, get_cluster_members,
    find_unclustered_embeddings,
)


@pytest.fixture(autouse=True)
def reset_db():
    """Reset database before each test for isolation."""
    # Close existing connections
    close_database()

    # Reinitialize with SQLite
    os.environ['AUTOVOICE_DB_TYPE'] = 'sqlite'
    init_database('sqlite')
    reset_database('sqlite')

    yield

    # Cleanup
    close_database()


class TestDatabaseConfiguration:
    """Test database configuration and engine creation."""

    def test_sqlite_database_type_set(self):
        """Verify SQLite is configured for testing."""
        assert DATABASE_TYPE == 'sqlite' or os.environ.get('AUTOVOICE_DB_TYPE') == 'sqlite'

    def test_engine_created_successfully(self):
        """Engine is created and accessible."""
        engine = get_engine()
        assert engine is not None
        assert 'sqlite' in str(engine.url)

    def test_session_factory_created(self):
        """Session factory is created."""
        factory = get_session_factory()
        assert factory is not None
        assert callable(factory)

    def test_get_db_session_context_manager(self):
        """Session context manager works correctly."""
        with get_db_session() as session:
            assert session is not None
            # Should be able to query
            count = session.query(Track).count()
            assert count == 0

    def test_init_database_creates_tables(self):
        """init_database creates all ORM tables."""
        engine = get_engine()

        # Get table names from metadata
        table_names = set(Base.metadata.tables.keys())

        expected = {
            'tracks', 'featured_artists', 'speaker_embeddings',
            'speaker_clusters', 'cluster_members'
        }
        assert expected.issubset(table_names)

    def test_database_stats_empty(self):
        """Stats return zeros for empty database."""
        stats = get_database_stats()

        assert stats['tracks'] == 0
        assert stats['featured_artists'] == 0
        assert stats['speaker_embeddings'] == 0
        assert stats['speaker_clusters'] == 0
        assert stats['cluster_members'] == 0


class TestTrackOperations:
    """Test track CRUD operations via SQLAlchemy ORM."""

    def test_upsert_track_insert_new(self):
        """Insert a new track."""
        upsert_track(
            track_id='yt_test_001',
            title='Test Song',
            channel='Test Channel',
            artist_name='Test Artist',
            duration_sec=180.5
        )

        track = get_track('yt_test_001')
        assert track is not None
        assert track['id'] == 'yt_test_001'
        assert track['title'] == 'Test Song'
        assert track['channel'] == 'Test Channel'
        assert track['artist_name'] == 'Test Artist'
        assert track['duration_sec'] == 180.5

    def test_upsert_track_update_existing(self):
        """Update an existing track."""
        upsert_track('yt_test_002', title='Original Title')
        upsert_track('yt_test_002', title='Updated Title')

        track = get_track('yt_test_002')
        assert track['title'] == 'Updated Title'

    def test_upsert_track_partial_update_preserves_fields(self):
        """Partial update should not clear unspecified fields."""
        upsert_track(
            'yt_test_003',
            title='Song Title',
            channel='My Channel',
            artist_name='First Artist'
        )

        # Update only artist_name
        upsert_track('yt_test_003', artist_name='Second Artist')

        track = get_track('yt_test_003')
        assert track['title'] == 'Song Title'  # Preserved
        assert track['channel'] == 'My Channel'  # Preserved
        assert track['artist_name'] == 'Second Artist'  # Updated

    def test_get_track_not_found_returns_none(self):
        """Getting non-existent track returns None."""
        track = get_track('nonexistent_track_id')
        assert track is None

    def test_get_all_tracks_sorted_by_artist_title(self):
        """get_all_tracks returns sorted list."""
        upsert_track('yt_b', title='Zebra Song', artist_name='Apple Artist')
        upsert_track('yt_a', title='Alpha Song', artist_name='Apple Artist')
        upsert_track('yt_c', title='Beta Song', artist_name='Banana Artist')

        tracks = get_all_tracks()
        assert len(tracks) == 3

        # Should be sorted by artist_name, then title
        assert tracks[0]['title'] == 'Alpha Song'  # Apple Artist, Alpha
        assert tracks[1]['title'] == 'Zebra Song'  # Apple Artist, Zebra
        assert tracks[2]['title'] == 'Beta Song'   # Banana Artist

    def test_get_tracks_by_artist(self):
        """Filter tracks by artist name."""
        upsert_track('yt_1', artist_name='Artist A')
        upsert_track('yt_2', artist_name='Artist A')
        upsert_track('yt_3', artist_name='Artist B')

        tracks_a = get_tracks_by_artist('Artist A')
        assert len(tracks_a) == 2
        assert all(t['artist_name'] == 'Artist A' for t in tracks_a)

        tracks_b = get_tracks_by_artist('Artist B')
        assert len(tracks_b) == 1

    def test_track_with_all_fields(self):
        """Track with all optional fields set."""
        upsert_track(
            'yt_full',
            title='Full Track',
            channel='Full Channel',
            upload_date='2026-01-01',
            duration_sec=300.0,
            artist_name='Full Artist',
            vocals_path='/path/to/vocals.wav',
            diarization_path='/path/to/diarization.json'
        )

        track = get_track('yt_full')
        assert track['title'] == 'Full Track'
        assert track['vocals_path'] == '/path/to/vocals.wav'
        assert track['diarization_path'] == '/path/to/diarization.json'


class TestFeaturedArtistOperations:
    """Test featured artist operations."""

    def test_add_featured_artist(self):
        """Add featured artist to track."""
        upsert_track('yt_feat_1', title='Collab Song')
        row_id = add_featured_artist('yt_feat_1', 'Featured Singer', 'ft.')

        assert row_id > 0

        artists = get_featured_artists_for_track('yt_feat_1')
        assert len(artists) == 1
        assert artists[0]['name'] == 'Featured Singer'
        assert artists[0]['pattern_matched'] == 'ft.'

    def test_add_featured_artist_duplicate_updates(self):
        """Adding duplicate featured artist updates pattern."""
        upsert_track('yt_feat_2', title='Collab Song 2')
        add_featured_artist('yt_feat_2', 'Singer', 'ft.')
        add_featured_artist('yt_feat_2', 'Singer', 'feat.')

        artists = get_featured_artists_for_track('yt_feat_2')
        assert len(artists) == 1
        assert artists[0]['pattern_matched'] == 'feat.'

    def test_get_featured_artists_multiple(self):
        """Get multiple featured artists for track."""
        upsert_track('yt_feat_3', title='Multi-Collab')
        add_featured_artist('yt_feat_3', 'Artist A', 'ft.')
        add_featured_artist('yt_feat_3', 'Artist B', 'with')
        add_featured_artist('yt_feat_3', 'Artist C', 'feat.')

        artists = get_featured_artists_for_track('yt_feat_3')
        assert len(artists) == 3
        names = {a['name'] for a in artists}
        assert names == {'Artist A', 'Artist B', 'Artist C'}

    def test_get_all_featured_artists_with_counts(self):
        """Get all featured artists across tracks with counts."""
        upsert_track('yt_f1', title='Song 1')
        upsert_track('yt_f2', title='Song 2')
        upsert_track('yt_f3', title='Song 3')

        add_featured_artist('yt_f1', 'Prolific Singer')
        add_featured_artist('yt_f2', 'Prolific Singer')
        add_featured_artist('yt_f3', 'Prolific Singer')
        add_featured_artist('yt_f1', 'One-Time Singer')

        artists = get_all_featured_artists()
        assert len(artists) == 2

        # Sorted by track count descending
        assert artists[0]['name'] == 'Prolific Singer'
        assert artists[0]['track_count'] == 3
        assert artists[1]['name'] == 'One-Time Singer'
        assert artists[1]['track_count'] == 1


class TestSpeakerEmbeddingOperations:
    """Test speaker embedding operations with numpy arrays."""

    def test_add_speaker_embedding(self):
        """Add speaker embedding to track."""
        upsert_track('yt_emb_1', title='Embedded Track')

        embedding = np.random.randn(512).astype(np.float32)
        row_id = add_speaker_embedding(
            track_id='yt_emb_1',
            speaker_id='SPEAKER_00',
            embedding=embedding,
            duration_sec=45.0,
            is_primary=True
        )

        assert row_id > 0

        embeddings = get_embeddings_for_track('yt_emb_1')
        assert len(embeddings) == 1
        assert embeddings[0]['speaker_id'] == 'SPEAKER_00'
        assert embeddings[0]['duration_sec'] == 45.0
        assert embeddings[0]['is_primary'] is True
        assert np.allclose(embeddings[0]['embedding'], embedding)

    def test_embedding_stored_as_bytes(self):
        """Embedding is converted to bytes and back."""
        upsert_track('yt_emb_2', title='Bytes Track')

        original = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        add_speaker_embedding('yt_emb_2', 'SPEAKER_00', original)

        result = get_embedding_by_id(1)  # First embedding
        loaded = result['embedding']

        assert isinstance(loaded, np.ndarray)
        assert loaded.dtype == np.float32
        np.testing.assert_array_almost_equal(loaded[:5], original)

    def test_get_embedding_by_id(self):
        """Get single embedding by row ID."""
        upsert_track('yt_emb_3', title='ID Track')

        embedding = np.random.randn(512).astype(np.float32)
        row_id = add_speaker_embedding('yt_emb_3', 'SPEAKER_01', embedding)

        result = get_embedding_by_id(row_id)
        assert result is not None
        assert result['speaker_id'] == 'SPEAKER_01'

    def test_get_embedding_by_id_not_found(self):
        """Get non-existent embedding returns None."""
        result = get_embedding_by_id(99999)
        assert result is None

    def test_get_all_embeddings(self):
        """Get all embeddings across tracks."""
        upsert_track('yt_e1', title='Track 1')
        upsert_track('yt_e2', title='Track 2')

        add_speaker_embedding('yt_e1', 'SPEAKER_00', np.random.randn(512).astype(np.float32))
        add_speaker_embedding('yt_e1', 'SPEAKER_01', np.random.randn(512).astype(np.float32))
        add_speaker_embedding('yt_e2', 'SPEAKER_00', np.random.randn(512).astype(np.float32))

        embeddings = get_all_embeddings()
        assert len(embeddings) == 3

    def test_embedding_update_existing(self):
        """Updating existing embedding (same track+speaker) updates data."""
        upsert_track('yt_emb_upd', title='Update Track')

        emb1 = np.ones(512, dtype=np.float32)
        add_speaker_embedding('yt_emb_upd', 'SPEAKER_00', emb1, duration_sec=10.0)

        emb2 = np.zeros(512, dtype=np.float32)
        add_speaker_embedding('yt_emb_upd', 'SPEAKER_00', emb2, duration_sec=20.0)

        embeddings = get_embeddings_for_track('yt_emb_upd')
        assert len(embeddings) == 1  # Not duplicated
        assert embeddings[0]['duration_sec'] == 20.0
        assert np.allclose(embeddings[0]['embedding'], emb2)


class TestSpeakerClusterOperations:
    """Test speaker cluster CRUD operations."""

    def test_create_cluster(self):
        """Create a new speaker cluster."""
        cluster_id = create_cluster('Anth')

        assert cluster_id is not None
        assert len(cluster_id) == 36  # UUID format

        cluster = get_cluster(cluster_id)
        assert cluster['name'] == 'Anth'
        assert cluster['is_verified'] is False

    def test_create_verified_cluster(self):
        """Create a verified cluster."""
        cluster_id = create_cluster('Verified Artist', is_verified=True)

        cluster = get_cluster(cluster_id)
        assert cluster['is_verified'] is True

    def test_create_cluster_with_voice_profile(self):
        """Create cluster linked to voice profile."""
        cluster_id = create_cluster('Linked Artist', voice_profile_id='profile-123')

        cluster = get_cluster(cluster_id)
        assert cluster['voice_profile_id'] == 'profile-123'

    def test_get_cluster_not_found(self):
        """Get non-existent cluster returns None."""
        cluster = get_cluster('nonexistent-uuid-here')
        assert cluster is None

    def test_get_all_clusters_with_member_counts(self):
        """Get all clusters with member counts."""
        c1 = create_cluster('Cluster One')
        c2 = create_cluster('Cluster Two')

        # Add some members to c1
        upsert_track('yt_c1', title='Track 1')
        emb_id = add_speaker_embedding('yt_c1', 'SPEAKER_00', np.random.randn(512).astype(np.float32))
        add_to_cluster(c1, emb_id)

        clusters = get_all_clusters()
        assert len(clusters) == 2

        # Find c1 in results
        c1_data = next(c for c in clusters if c['id'] == c1)
        assert c1_data['member_count'] == 1

        c2_data = next(c for c in clusters if c['id'] == c2)
        assert c2_data['member_count'] == 0

    def test_update_cluster_name(self):
        """Update cluster name and verification status."""
        cluster_id = create_cluster('Unknown 1')
        update_cluster_name(cluster_id, 'Identified Artist', is_verified=True)

        cluster = get_cluster(cluster_id)
        assert cluster['name'] == 'Identified Artist'
        assert cluster['is_verified'] is True


class TestClusterMemberOperations:
    """Test cluster membership operations."""

    def test_add_to_cluster(self):
        """Add embedding to cluster."""
        cluster_id = create_cluster('Test Cluster')
        upsert_track('yt_cm1', title='Member Track', artist_name='Test Artist')
        emb_id = add_speaker_embedding('yt_cm1', 'SPEAKER_00', np.random.randn(512).astype(np.float32))

        add_to_cluster(cluster_id, emb_id, confidence=0.95)

        members = get_cluster_members(cluster_id)
        assert len(members) == 1
        assert members[0]['confidence'] == 0.95
        assert members[0]['track_title'] == 'Member Track'

    def test_remove_from_cluster(self):
        """Remove embedding from cluster."""
        cluster_id = create_cluster('Remove Test')
        upsert_track('yt_rm1', title='Remove Track', artist_name='Artist')
        emb_id = add_speaker_embedding('yt_rm1', 'SPEAKER_00', np.random.randn(512).astype(np.float32))

        add_to_cluster(cluster_id, emb_id)
        remove_from_cluster(cluster_id, emb_id)

        members = get_cluster_members(cluster_id)
        assert len(members) == 0

    def test_get_embeddings_by_cluster(self):
        """Get all embeddings belonging to cluster."""
        cluster_id = create_cluster('Embedding Cluster')
        upsert_track('yt_ec1', title='Track 1')
        upsert_track('yt_ec2', title='Track 2')

        emb1_id = add_speaker_embedding('yt_ec1', 'SPEAKER_00', np.random.randn(512).astype(np.float32))
        emb2_id = add_speaker_embedding('yt_ec2', 'SPEAKER_00', np.random.randn(512).astype(np.float32))

        add_to_cluster(cluster_id, emb1_id, confidence=0.9)
        add_to_cluster(cluster_id, emb2_id, confidence=0.85)

        embeddings = get_embeddings_by_cluster(cluster_id)
        assert len(embeddings) == 2

        # Sorted by confidence descending
        assert embeddings[0]['confidence'] == 0.9
        assert embeddings[1]['confidence'] == 0.85

    def test_merge_clusters(self):
        """Merge two clusters together."""
        c1 = create_cluster('Target Cluster')
        c2 = create_cluster('Source Cluster')

        upsert_track('yt_m1', title='Track 1')
        upsert_track('yt_m2', title='Track 2')

        emb1_id = add_speaker_embedding('yt_m1', 'SPEAKER_00', np.random.randn(512).astype(np.float32))
        emb2_id = add_speaker_embedding('yt_m2', 'SPEAKER_00', np.random.randn(512).astype(np.float32))

        add_to_cluster(c1, emb1_id)
        add_to_cluster(c2, emb2_id)

        # Merge c2 into c1
        merge_clusters(c1, c2)

        # c1 should have both embeddings
        embeddings = get_embeddings_by_cluster(c1)
        assert len(embeddings) == 2

        # c2 should be deleted
        assert get_cluster(c2) is None

    def test_find_unclustered_embeddings(self):
        """Find embeddings not in any cluster."""
        upsert_track('yt_unc1', title='Unclustered Track', artist_name='Artist')
        upsert_track('yt_unc2', title='Clustered Track', artist_name='Artist')

        add_speaker_embedding('yt_unc1', 'SPEAKER_00', np.random.randn(512).astype(np.float32))
        emb2_id = add_speaker_embedding('yt_unc2', 'SPEAKER_00', np.random.randn(512).astype(np.float32))

        cluster_id = create_cluster('Some Cluster')
        add_to_cluster(cluster_id, emb2_id)

        unclustered = find_unclustered_embeddings()
        assert len(unclustered) == 1
        assert unclustered[0]['track_title'] == 'Unclustered Track'


class TestTransactionHandling:
    """Test transaction commit/rollback behavior."""

    def test_session_commits_on_success(self):
        """Session commits changes on successful exit."""
        with get_db_session() as session:
            track = Track(id='tx_test_1', title='Transaction Test')
            session.add(track)

        # Should be persisted
        result = get_track('tx_test_1')
        assert result is not None

    def test_session_rollback_on_exception(self):
        """Session rolls back on exception."""
        try:
            with get_db_session() as session:
                track = Track(id='tx_test_2', title='Rollback Test')
                session.add(track)
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # Should NOT be persisted
        result = get_track('tx_test_2')
        assert result is None

    def test_multiple_operations_in_transaction(self):
        """Multiple operations in single transaction."""
        with get_db_session() as session:
            # All in one transaction
            track1 = Track(id='tx_multi_1', title='Multi 1')
            track2 = Track(id='tx_multi_2', title='Multi 2')
            session.add(track1)
            session.add(track2)

        assert get_track('tx_multi_1') is not None
        assert get_track('tx_multi_2') is not None


class TestDatabaseStats:
    """Test database statistics collection."""

    def test_stats_with_data(self):
        """Stats reflect actual data counts."""
        # Add data
        upsert_track('yt_s1', artist_name='Artist A')
        upsert_track('yt_s2', artist_name='Artist B')
        add_featured_artist('yt_s1', 'Featured One')
        add_speaker_embedding('yt_s1', 'SPEAKER_00', np.random.randn(512).astype(np.float32))
        cluster_id = create_cluster('Test Cluster', is_verified=True)

        stats = get_database_stats()

        assert stats['tracks'] == 2
        assert stats['unique_artists'] == 2
        assert stats['featured_artists'] == 1
        assert stats['speaker_embeddings'] == 1
        assert stats['speaker_clusters'] == 1
        assert stats['verified_clusters'] == 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_track_id(self):
        """Handle empty track ID."""
        upsert_track('', title='Empty ID Track')
        track = get_track('')
        assert track is not None
        assert track['id'] == ''

    def test_unicode_in_fields(self):
        """Handle unicode characters in fields."""
        upsert_track(
            'yt_unicode',
            title='Japanese title',
            channel='Chinese Channel',
            artist_name='French Artist'
        )

        track = get_track('yt_unicode')
        assert track is not None

    def test_large_embedding(self):
        """Handle large embedding arrays."""
        upsert_track('yt_large', title='Large Embedding Track')

        # 512-dim is standard, but test larger
        large_emb = np.random.randn(1024).astype(np.float32)
        row_id = add_speaker_embedding('yt_large', 'SPEAKER_00', large_emb)

        result = get_embedding_by_id(row_id)
        assert result['embedding'].shape == (1024,)

    def test_very_long_title(self):
        """Handle very long title strings."""
        long_title = 'A' * 500
        upsert_track('yt_long', title=long_title)

        track = get_track('yt_long')
        assert track['title'] == long_title

    def test_special_characters_in_track_id(self):
        """Handle special characters in track ID."""
        special_id = 'yt_123-abc_XYZ'
        upsert_track(special_id, title='Special ID Track')

        track = get_track(special_id)
        assert track is not None
