"""Comprehensive tests for SQLite database schema and operations.

Tests for:
- Schema creation and initialization
- Track CRUD operations
- Featured artist operations
- Speaker embedding operations
- Speaker cluster operations
- Cluster member operations
- Database context manager and transactions
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from auto_voice.db.schema import (
    init_database,
    get_connection,
    get_db_context,
    reset_database,
    get_database_stats,
)
from auto_voice.db.operations import (
    # Track operations
    upsert_track,
    get_track,
    get_all_tracks,
    get_tracks_by_artist,
    # Featured artist operations
    add_featured_artist,
    get_featured_artists_for_track,
    get_all_featured_artists,
    # Speaker embedding operations
    add_speaker_embedding,
    get_embeddings_for_track,
    get_all_embeddings,
    get_embeddings_by_cluster,
    get_embedding_by_id,
    # Cluster operations
    create_cluster,
    get_cluster,
    get_all_clusters,
    update_cluster_name,
    merge_clusters,
    add_to_cluster,
    remove_from_cluster,
    get_cluster_members,
    find_unclustered_embeddings,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    import os
    from auto_voice.db import schema

    # Force SQLite for testing
    old_db_type = os.environ.get('AUTOVOICE_DB_TYPE')
    os.environ['AUTOVOICE_DB_TYPE'] = 'sqlite'

    # Reset engine to pick up new env var
    schema._engine = None
    schema._SessionFactory = None

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        init_database(db_type='sqlite')
        yield db_path

        # Cleanup
        schema.close_database()
        schema._engine = None
        schema._SessionFactory = None

    # Restore original env var
    if old_db_type is not None:
        os.environ['AUTOVOICE_DB_TYPE'] = old_db_type
    elif 'AUTOVOICE_DB_TYPE' in os.environ:
        del os.environ['AUTOVOICE_DB_TYPE']


class TestSchemaCreation:
    """Test database schema initialization."""

    def test_init_database_creates_tables(self, temp_db):
        """Database initialization creates all required tables."""
        session = get_connection(temp_db)

        # Check tables exist using SQLAlchemy inspector
        from sqlalchemy import inspect
        inspector = inspect(session.bind)
        tables = set(inspector.get_table_names())

        expected_tables = {
            'tracks', 'featured_artists', 'speaker_embeddings',
            'speaker_clusters', 'cluster_members'
        }
        assert expected_tables.issubset(tables)
        session.close()

    def test_init_database_creates_indexes(self, temp_db):
        """Database initialization creates required indexes."""
        session = get_connection(temp_db)

        # Check indexes exist using SQLAlchemy inspector
        from sqlalchemy import inspect
        inspector = inspect(session.bind)

        # Get indexes for tracks table
        tracks_indexes = inspector.get_indexes('tracks')
        tracks_index_cols = {idx['column_names'][0] for idx in tracks_indexes if idx['column_names']}
        assert 'artist_name' in tracks_index_cols

        # Get indexes for speaker_embeddings table
        embeddings_indexes = inspector.get_indexes('speaker_embeddings')
        embeddings_index_cols = {idx['column_names'][0] for idx in embeddings_indexes if idx['column_names']}
        assert 'track_id' in embeddings_index_cols

        session.close()

    def test_get_connection_enables_foreign_keys(self, temp_db):
        """Connection enables foreign key support."""
        session = get_connection(temp_db)

        # Check if foreign keys are enabled (SQLite-specific)
        if 'sqlite' in str(session.bind.url):
            result = session.execute("PRAGMA foreign_keys").fetchone()
            assert result[0] == 1

        session.close()

    def test_reset_database_clears_data(self, temp_db):
        """Reset database clears all data."""
        # Insert some data
        upsert_track("test1", title="Test Track", db_path=temp_db)

        # Verify data exists
        assert get_track("test1", db_path=temp_db) is not None

        # Reset
        reset_database(temp_db)

        # Verify data cleared
        assert get_track("test1", db_path=temp_db) is None


class TestTrackOperations:
    """Test track CRUD operations."""

    def test_upsert_track_insert(self, temp_db):
        """Insert new track."""
        upsert_track(
            "yt123",
            title="Test Song",
            channel="Test Channel",
            artist_name="Test Artist",
            duration_sec=180.5,
            db_path=temp_db
        )

        track = get_track("yt123", db_path=temp_db)
        assert track is not None
        assert track['id'] == "yt123"
        assert track['title'] == "Test Song"
        assert track['channel'] == "Test Channel"
        assert track['artist_name'] == "Test Artist"
        assert track['duration_sec'] == 180.5

    def test_upsert_track_update(self, temp_db):
        """Update existing track."""
        upsert_track("yt123", title="Original", db_path=temp_db)
        upsert_track("yt123", title="Updated", db_path=temp_db)

        track = get_track("yt123", db_path=temp_db)
        assert track['title'] == "Updated"

    def test_upsert_track_partial_update(self, temp_db):
        """Partial update doesn't clear existing fields."""
        upsert_track("yt123", title="Test", channel="Channel1", db_path=temp_db)
        upsert_track("yt123", artist_name="Artist1", db_path=temp_db)

        track = get_track("yt123", db_path=temp_db)
        assert track['title'] == "Test"
        assert track['channel'] == "Channel1"
        assert track['artist_name'] == "Artist1"

    def test_get_track_not_found(self, temp_db):
        """Get non-existent track returns None."""
        track = get_track("nonexistent", db_path=temp_db)
        assert track is None

    def test_get_all_tracks(self, temp_db):
        """Get all tracks returns sorted list."""
        upsert_track("yt1", title="Song B", artist_name="Artist A", db_path=temp_db)
        upsert_track("yt2", title="Song A", artist_name="Artist A", db_path=temp_db)
        upsert_track("yt3", title="Song C", artist_name="Artist B", db_path=temp_db)

        tracks = get_all_tracks(db_path=temp_db)
        assert len(tracks) == 3
        # Sorted by artist, then title
        assert tracks[0]['title'] == "Song A"
        assert tracks[1]['title'] == "Song B"
        assert tracks[2]['title'] == "Song C"

    def test_get_tracks_by_artist(self, temp_db):
        """Get tracks filtered by artist."""
        upsert_track("yt1", artist_name="Artist A", db_path=temp_db)
        upsert_track("yt2", artist_name="Artist A", db_path=temp_db)
        upsert_track("yt3", artist_name="Artist B", db_path=temp_db)

        tracks = get_tracks_by_artist("Artist A", db_path=temp_db)
        assert len(tracks) == 2
        for track in tracks:
            assert track['artist_name'] == "Artist A"


class TestFeaturedArtistOperations:
    """Test featured artist operations."""

    def test_add_featured_artist(self, temp_db):
        """Add featured artist to track."""
        upsert_track("yt123", db_path=temp_db)
        row_id = add_featured_artist("yt123", "Anth", "ft.", db_path=temp_db)

        assert row_id > 0
        artists = get_featured_artists_for_track("yt123", db_path=temp_db)
        assert len(artists) == 1
        assert artists[0]['name'] == "Anth"
        assert artists[0]['pattern_matched'] == "ft."

    def test_add_featured_artist_duplicate(self, temp_db):
        """Adding duplicate featured artist updates pattern."""
        upsert_track("yt123", db_path=temp_db)
        add_featured_artist("yt123", "Anth", "ft.", db_path=temp_db)
        add_featured_artist("yt123", "Anth", "feat.", db_path=temp_db)

        artists = get_featured_artists_for_track("yt123", db_path=temp_db)
        assert len(artists) == 1  # Still just one
        assert artists[0]['pattern_matched'] == "feat."

    def test_get_all_featured_artists(self, temp_db):
        """Get all featured artists with track counts."""
        upsert_track("yt1", db_path=temp_db)
        upsert_track("yt2", db_path=temp_db)
        add_featured_artist("yt1", "Anth", db_path=temp_db)
        add_featured_artist("yt2", "Anth", db_path=temp_db)
        add_featured_artist("yt1", "Other", db_path=temp_db)

        artists = get_all_featured_artists(db_path=temp_db)
        assert len(artists) == 2
        # Anth should be first (more tracks)
        assert artists[0]['name'] == "Anth"
        assert artists[0]['track_count'] == 2

    def test_featured_artist_cascade_delete(self, temp_db):
        """Featured artists are deleted when track is deleted."""
        upsert_track("yt123", db_path=temp_db)
        add_featured_artist("yt123", "Anth", db_path=temp_db)

        # Delete track
        with get_db_context(temp_db) as conn:
            conn.execute("DELETE FROM tracks WHERE id = ?", ("yt123",))

        artists = get_featured_artists_for_track("yt123", db_path=temp_db)
        assert len(artists) == 0


class TestSpeakerEmbeddingOperations:
    """Test speaker embedding operations."""

    def test_add_speaker_embedding(self, temp_db):
        """Add speaker embedding to track."""
        upsert_track("yt123", db_path=temp_db)
        embedding = np.random.randn(512).astype(np.float32)

        row_id = add_speaker_embedding(
            "yt123", "SPEAKER_00", embedding,
            duration_sec=45.5,
            is_primary=True,
            db_path=temp_db
        )

        assert row_id > 0
        embeddings = get_embeddings_for_track("yt123", db_path=temp_db)
        assert len(embeddings) == 1
        assert embeddings[0]['speaker_id'] == "SPEAKER_00"
        assert embeddings[0]['duration_sec'] == 45.5
        assert embeddings[0]['is_primary'] == True
        assert np.allclose(embeddings[0]['embedding'], embedding)

    def test_get_embedding_by_id(self, temp_db):
        """Get single embedding by ID."""
        upsert_track("yt123", db_path=temp_db)
        embedding = np.random.randn(512).astype(np.float32)
        row_id = add_speaker_embedding("yt123", "SPEAKER_00", embedding, db_path=temp_db)

        result = get_embedding_by_id(row_id, db_path=temp_db)
        assert result is not None
        assert np.allclose(result['embedding'], embedding)

    def test_get_embedding_by_id_not_found(self, temp_db):
        """Get non-existent embedding returns None."""
        result = get_embedding_by_id(99999, db_path=temp_db)
        assert result is None

    def test_get_all_embeddings(self, temp_db):
        """Get all embeddings across tracks."""
        upsert_track("yt1", db_path=temp_db)
        upsert_track("yt2", db_path=temp_db)

        add_speaker_embedding("yt1", "SPEAKER_00", np.random.randn(512), db_path=temp_db)
        add_speaker_embedding("yt1", "SPEAKER_01", np.random.randn(512), db_path=temp_db)
        add_speaker_embedding("yt2", "SPEAKER_00", np.random.randn(512), db_path=temp_db)

        embeddings = get_all_embeddings(db_path=temp_db)
        assert len(embeddings) == 3

    def test_find_unclustered_embeddings(self, temp_db):
        """Find embeddings not in any cluster."""
        upsert_track("yt123", title="Test", artist_name="Artist", db_path=temp_db)
        emb_id = add_speaker_embedding("yt123", "SPEAKER_00", np.random.randn(512), db_path=temp_db)

        unclustered = find_unclustered_embeddings(db_path=temp_db)
        assert len(unclustered) == 1
        assert unclustered[0]['track_title'] == "Test"


class TestClusterOperations:
    """Test speaker cluster operations."""

    def test_create_cluster(self, temp_db):
        """Create a new cluster."""
        cluster_id = create_cluster("Anth", db_path=temp_db)

        assert cluster_id is not None
        cluster = get_cluster(cluster_id, db_path=temp_db)
        assert cluster['name'] == "Anth"
        assert cluster['is_verified'] == False

    def test_create_verified_cluster(self, temp_db):
        """Create a verified cluster."""
        cluster_id = create_cluster("Anth", is_verified=True, db_path=temp_db)

        cluster = get_cluster(cluster_id, db_path=temp_db)
        assert cluster['is_verified'] == True

    def test_get_cluster_not_found(self, temp_db):
        """Get non-existent cluster returns None."""
        cluster = get_cluster("fake-uuid", db_path=temp_db)
        assert cluster is None

    def test_get_all_clusters(self, temp_db):
        """Get all clusters with member counts."""
        c1_id = create_cluster("Anth", db_path=temp_db)
        c2_id = create_cluster("Other", db_path=temp_db)

        clusters = get_all_clusters(db_path=temp_db)
        assert len(clusters) == 2
        # Sorted by name
        assert clusters[0]['name'] == "Anth"
        assert clusters[0]['member_count'] == 0

    def test_update_cluster_name(self, temp_db):
        """Update cluster name."""
        cluster_id = create_cluster("Unknown 1", db_path=temp_db)
        update_cluster_name(cluster_id, "Anth", is_verified=True, db_path=temp_db)

        cluster = get_cluster(cluster_id, db_path=temp_db)
        assert cluster['name'] == "Anth"
        assert cluster['is_verified'] == True


class TestClusterMemberOperations:
    """Test cluster member operations."""

    def test_add_to_cluster(self, temp_db):
        """Add embedding to cluster."""
        upsert_track("yt123", title="Test Song", artist_name="Test Artist", db_path=temp_db)
        emb_id = add_speaker_embedding("yt123", "SPEAKER_00", np.random.randn(512), db_path=temp_db)
        cluster_id = create_cluster("Anth", db_path=temp_db)

        add_to_cluster(cluster_id, emb_id, confidence=0.95, db_path=temp_db)

        members = get_cluster_members(cluster_id, db_path=temp_db)
        assert len(members) == 1
        assert members[0]['confidence'] == 0.95
        assert members[0]['track_title'] == "Test Song"

        embeddings = get_embeddings_by_cluster(cluster_id, db_path=temp_db)
        assert len(embeddings) == 1
        assert embeddings[0]['confidence'] == 0.95

    def test_remove_from_cluster(self, temp_db):
        """Remove embedding from cluster."""
        upsert_track("yt123", title="Test", artist_name="Artist", db_path=temp_db)
        emb_id = add_speaker_embedding("yt123", "SPEAKER_00", np.random.randn(512), db_path=temp_db)
        cluster_id = create_cluster("Anth", db_path=temp_db)

        add_to_cluster(cluster_id, emb_id, db_path=temp_db)
        remove_from_cluster(cluster_id, emb_id, db_path=temp_db)

        embeddings = get_embeddings_by_cluster(cluster_id, db_path=temp_db)
        assert len(embeddings) == 0

    def test_merge_clusters(self, temp_db):
        """Merge two clusters."""
        upsert_track("yt1", db_path=temp_db)
        upsert_track("yt2", db_path=temp_db)
        emb1_id = add_speaker_embedding("yt1", "SPEAKER_00", np.random.randn(512), db_path=temp_db)
        emb2_id = add_speaker_embedding("yt2", "SPEAKER_00", np.random.randn(512), db_path=temp_db)

        c1_id = create_cluster("Cluster1", db_path=temp_db)
        c2_id = create_cluster("Cluster2", db_path=temp_db)

        add_to_cluster(c1_id, emb1_id, db_path=temp_db)
        add_to_cluster(c2_id, emb2_id, db_path=temp_db)

        # Merge c2 into c1
        merge_clusters(c1_id, c2_id, db_path=temp_db)

        # c1 should have both embeddings
        embeddings = get_embeddings_by_cluster(c1_id, db_path=temp_db)
        assert len(embeddings) == 2

        # c2 should be deleted
        cluster = get_cluster(c2_id, db_path=temp_db)
        assert cluster is None


class TestDatabaseContext:
    """Test database context manager and transactions."""

    def test_context_commits_on_success(self, temp_db):
        """Context manager commits on successful completion."""
        with get_db_context(temp_db) as conn:
            conn.execute("""
                INSERT INTO tracks (id, title) VALUES ('test1', 'Test')
            """)

        # Data should be persisted
        track = get_track("test1", db_path=temp_db)
        assert track is not None

    def test_context_rollback_on_error(self, temp_db):
        """Context manager rolls back on exception."""
        try:
            with get_db_context(temp_db) as conn:
                conn.execute("""
                    INSERT INTO tracks (id, title) VALUES ('test1', 'Test')
                """)
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # Data should NOT be persisted
        track = get_track("test1", db_path=temp_db)
        assert track is None


class TestDatabaseStats:
    """Test database statistics."""

    def test_get_database_stats_empty(self, temp_db):
        """Get stats for empty database."""
        stats = get_database_stats(db_path=temp_db)

        assert stats['tracks'] == 0
        assert stats['featured_artists'] == 0
        assert stats['speaker_embeddings'] == 0
        assert stats['speaker_clusters'] == 0
        assert stats['cluster_members'] == 0

    def test_get_database_stats_with_data(self, temp_db):
        """Get stats for database with data."""
        upsert_track("yt1", artist_name="Artist1", db_path=temp_db)
        upsert_track("yt2", artist_name="Artist2", db_path=temp_db)
        add_featured_artist("yt1", "Anth", db_path=temp_db)
        add_speaker_embedding("yt1", "SPEAKER_00", np.random.randn(512), db_path=temp_db)
        cluster_id = create_cluster("Anth", is_verified=True, db_path=temp_db)

        stats = get_database_stats(db_path=temp_db)

        assert stats['tracks'] == 2
        assert stats['unique_artists'] == 2
        assert stats['featured_artists'] == 1
        assert stats['speaker_embeddings'] == 1
        assert stats['speaker_clusters'] == 1
        assert stats['verified_clusters'] == 1
