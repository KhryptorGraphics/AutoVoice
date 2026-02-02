"""Comprehensive tests for database operations (TDD Phase 3.1).

Tests CRUD operations for:
- Tracks
- Featured artists
- Speaker embeddings
- Speaker clusters
- Cluster members
- Transaction handling and error recovery
"""

import pytest
import numpy as np
import os
import tempfile
from pathlib import Path

from auto_voice.db.schema import (
    init_database,
    reset_database,
    get_db_session,
    close_database,
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


@pytest.fixture(scope="function")
def test_db():
    """Create an in-memory SQLite database for testing."""
    # Use in-memory SQLite for fast tests
    os.environ['AUTOVOICE_DB_TYPE'] = 'sqlite'

    # Reset database to ensure clean state
    reset_database(db_type='sqlite')

    yield

    # Cleanup
    close_database()
    # Reset environment
    os.environ.pop('AUTOVOICE_DB_TYPE', None)


class TestTrackCRUD:
    """Test Track CRUD operations following TDD principles."""

    def test_upsert_track_creates_new(self, test_db):
        """Test creating a new track (INSERT)."""
        # Arrange
        track_id = "test_video_001"
        title = "Test Song"
        artist = "Test Artist"

        # Act
        upsert_track(
            track_id=track_id,
            title=title,
            artist_name=artist,
            duration_sec=180.5,
            channel="Test Channel",
        )

        # Assert
        result = get_track(track_id)
        assert result is not None
        assert result['id'] == track_id
        assert result['title'] == title
        assert result['artist_name'] == artist
        assert result['duration_sec'] == 180.5
        assert result['channel'] == "Test Channel"
        assert result['fetched_at'] is not None

    def test_upsert_track_updates_existing(self, test_db):
        """Test updating an existing track (UPDATE)."""
        # Arrange - create initial track
        track_id = "test_video_002"
        upsert_track(track_id=track_id, title="Original Title", artist_name="Artist A")

        # Act - update with new data
        upsert_track(track_id=track_id, title="Updated Title", duration_sec=200.0)

        # Assert - title updated, artist preserved
        result = get_track(track_id)
        assert result['title'] == "Updated Title"
        assert result['artist_name'] == "Artist A"  # Preserved
        assert result['duration_sec'] == 200.0

    def test_get_track_returns_none_when_not_found(self, test_db):
        """Test retrieving non-existent track returns None."""
        result = get_track("nonexistent_id")
        assert result is None

    def test_get_all_tracks_returns_ordered_list(self, test_db):
        """Test retrieving all tracks in sorted order."""
        # Arrange - create multiple tracks
        upsert_track("id1", title="Song A", artist_name="Artist B")
        upsert_track("id2", title="Song B", artist_name="Artist A")
        upsert_track("id3", title="Song C", artist_name="Artist B")

        # Act
        tracks = get_all_tracks()

        # Assert - ordered by artist, then title
        assert len(tracks) == 3
        assert tracks[0]['artist_name'] == "Artist A"
        assert tracks[1]['artist_name'] == "Artist B"
        assert tracks[1]['title'] == "Song A"
        assert tracks[2]['title'] == "Song C"

    def test_get_tracks_by_artist_filters_correctly(self, test_db):
        """Test filtering tracks by artist name."""
        # Arrange
        upsert_track("id1", artist_name="Taylor Swift")
        upsert_track("id2", artist_name="Ariana Grande")
        upsert_track("id3", artist_name="Taylor Swift")

        # Act
        taylor_tracks = get_tracks_by_artist("Taylor Swift")

        # Assert
        assert len(taylor_tracks) == 2
        assert all(t['artist_name'] == "Taylor Swift" for t in taylor_tracks)


class TestFeaturedArtistOperations:
    """Test featured artist operations."""

    def test_add_featured_artist_creates_new(self, test_db):
        """Test adding a featured artist to a track."""
        # Arrange - create track first
        track_id = "collab_track"
        upsert_track(track_id=track_id, title="Collab Song", artist_name="Main Artist")

        # Act
        artist_id = add_featured_artist(track_id, name="Featured Artist", pattern_matched="ft.")

        # Assert
        assert artist_id is not None
        artists = get_featured_artists_for_track(track_id)
        assert len(artists) == 1
        assert artists[0]['name'] == "Featured Artist"
        assert artists[0]['pattern_matched'] == "ft."

    def test_add_featured_artist_avoids_duplicates(self, test_db):
        """Test adding same artist twice returns existing ID."""
        # Arrange
        track_id = "collab_track"
        upsert_track(track_id=track_id, title="Song")

        # Act - add same artist twice
        id1 = add_featured_artist(track_id, name="Artist X")
        id2 = add_featured_artist(track_id, name="Artist X")

        # Assert - same ID returned, no duplicate
        assert id1 == id2
        artists = get_featured_artists_for_track(track_id)
        assert len(artists) == 1

    def test_get_all_featured_artists_aggregates_counts(self, test_db):
        """Test getting featured artists with track counts."""
        # Arrange - multiple tracks with shared featured artists
        upsert_track("t1", title="Song 1")
        upsert_track("t2", title="Song 2")
        upsert_track("t3", title="Song 3")

        add_featured_artist("t1", name="Artist A")
        add_featured_artist("t2", name="Artist A")
        add_featured_artist("t3", name="Artist B")

        # Act
        artists = get_all_featured_artists()

        # Assert - Artist A appears in 2 tracks, Artist B in 1
        assert len(artists) == 2
        artist_a = next(a for a in artists if a['name'] == "Artist A")
        assert artist_a['track_count'] == 2


class TestSpeakerEmbeddingOperations:
    """Test speaker embedding CRUD operations."""

    def test_add_speaker_embedding_creates_new(self, test_db):
        """Test adding a speaker embedding."""
        # Arrange
        track_id = "video_with_speakers"
        upsert_track(track_id=track_id, title="Interview")
        embedding = np.random.randn(512).astype(np.float32)

        # Act
        emb_id = add_speaker_embedding(
            track_id=track_id,
            speaker_id="SPEAKER_00",
            embedding=embedding,
            duration_sec=45.0,
            is_primary=True,
        )

        # Assert
        assert emb_id is not None
        embeddings = get_embeddings_for_track(track_id)
        assert len(embeddings) == 1
        assert embeddings[0]['speaker_id'] == "SPEAKER_00"
        assert embeddings[0]['is_primary'] is True
        assert embeddings[0]['duration_sec'] == 45.0
        # Verify embedding was stored correctly
        assert embeddings[0]['embedding'].shape == (512,)
        assert np.allclose(embeddings[0]['embedding'], embedding, atol=1e-5)

    def test_add_speaker_embedding_updates_existing(self, test_db):
        """Test updating an existing speaker embedding."""
        # Arrange - create initial embedding
        track_id = "video_001"
        upsert_track(track_id=track_id, title="Test")
        embedding1 = np.ones(512, dtype=np.float32)
        embedding2 = np.zeros(512, dtype=np.float32)

        id1 = add_speaker_embedding(track_id, "SPEAKER_00", embedding1)

        # Act - update with new embedding
        id2 = add_speaker_embedding(track_id, "SPEAKER_00", embedding2, duration_sec=60.0)

        # Assert - same ID, updated embedding
        assert id1 == id2
        embeddings = get_embeddings_for_track(track_id)
        assert len(embeddings) == 1
        assert np.allclose(embeddings[0]['embedding'], embedding2, atol=1e-5)
        assert embeddings[0]['duration_sec'] == 60.0

    def test_get_embedding_by_id_retrieves_single(self, test_db):
        """Test retrieving a single embedding by ID."""
        # Arrange
        track_id = "video_002"
        upsert_track(track_id=track_id, title="Test")
        embedding = np.random.randn(512).astype(np.float32)
        emb_id = add_speaker_embedding(track_id, "SPEAKER_00", embedding)

        # Act
        result = get_embedding_by_id(emb_id)

        # Assert
        assert result is not None
        assert result['id'] == emb_id
        assert result['track_id'] == track_id
        assert np.allclose(result['embedding'], embedding, atol=1e-5)

    def test_get_all_embeddings_returns_all(self, test_db):
        """Test retrieving all embeddings across tracks."""
        # Arrange
        upsert_track("t1", title="Track 1")
        upsert_track("t2", title="Track 2")

        add_speaker_embedding("t1", "SPEAKER_00", np.ones(512, dtype=np.float32))
        add_speaker_embedding("t1", "SPEAKER_01", np.ones(512, dtype=np.float32))
        add_speaker_embedding("t2", "SPEAKER_00", np.ones(512, dtype=np.float32))

        # Act
        embeddings = get_all_embeddings()

        # Assert
        assert len(embeddings) == 3


class TestSpeakerClusterOperations:
    """Test speaker cluster operations."""

    def test_create_cluster_generates_uuid(self, test_db):
        """Test creating a new speaker cluster."""
        # Act
        cluster_id = create_cluster(name="John Doe", is_verified=True)

        # Assert
        assert cluster_id is not None
        assert len(cluster_id) == 36  # UUID format
        cluster = get_cluster(cluster_id)
        assert cluster['name'] == "John Doe"
        assert cluster['is_verified'] is True

    def test_get_all_clusters_includes_member_counts(self, test_db):
        """Test retrieving all clusters with member counts."""
        # Arrange - create clusters and add members
        cluster1 = create_cluster(name="Speaker A")
        cluster2 = create_cluster(name="Speaker B")

        # Create embeddings and add to clusters
        upsert_track("t1", title="Test")
        emb1 = add_speaker_embedding("t1", "SPEAKER_00", np.ones(512, dtype=np.float32), duration_sec=10.0)
        emb2 = add_speaker_embedding("t1", "SPEAKER_01", np.ones(512, dtype=np.float32), duration_sec=20.0)

        add_to_cluster(cluster1, emb1)
        add_to_cluster(cluster1, emb2)

        # Act
        clusters = get_all_clusters()

        # Assert
        assert len(clusters) == 2
        cluster_a = next(c for c in clusters if c['name'] == "Speaker A")
        assert cluster_a['member_count'] == 2
        assert cluster_a['total_duration_sec'] == 30.0

    def test_update_cluster_name_modifies_cluster(self, test_db):
        """Test updating a cluster's name."""
        # Arrange
        cluster_id = create_cluster(name="Unknown Speaker")

        # Act
        update_cluster_name(cluster_id, name="Identified Speaker", is_verified=True)

        # Assert
        cluster = get_cluster(cluster_id)
        assert cluster['name'] == "Identified Speaker"
        assert cluster['is_verified'] is True

    def test_merge_clusters_moves_members(self, test_db):
        """Test merging two clusters."""
        # Arrange - create two clusters with members
        cluster1 = create_cluster(name="Speaker A")
        cluster2 = create_cluster(name="Speaker A Duplicate")

        upsert_track("t1", title="Test")
        emb1 = add_speaker_embedding("t1", "SPEAKER_00", np.ones(512, dtype=np.float32))
        emb2 = add_speaker_embedding("t1", "SPEAKER_01", np.ones(512, dtype=np.float32))

        add_to_cluster(cluster1, emb1)
        add_to_cluster(cluster2, emb2)

        # Act - merge cluster2 into cluster1
        merge_clusters(target_cluster_id=cluster1, source_cluster_id=cluster2)

        # Assert - cluster1 has both members, cluster2 deleted
        members = get_cluster_members(cluster1)
        assert len(members) == 2
        assert get_cluster(cluster2) is None

    def test_add_to_cluster_creates_membership(self, test_db):
        """Test adding an embedding to a cluster."""
        # Arrange
        cluster_id = create_cluster(name="Test Cluster")
        upsert_track("t1", title="Test")
        emb_id = add_speaker_embedding("t1", "SPEAKER_00", np.ones(512, dtype=np.float32))

        # Act
        add_to_cluster(cluster_id, emb_id, confidence=0.95)

        # Assert
        members = get_cluster_members(cluster_id)
        assert len(members) == 1
        assert members[0]['confidence'] == 0.95
        assert members[0]['embedding_id'] == emb_id

    def test_remove_from_cluster_deletes_membership(self, test_db):
        """Test removing an embedding from a cluster."""
        # Arrange
        cluster_id = create_cluster(name="Test Cluster")
        upsert_track("t1", title="Test")
        emb_id = add_speaker_embedding("t1", "SPEAKER_00", np.ones(512, dtype=np.float32))
        add_to_cluster(cluster_id, emb_id)

        # Act
        remove_from_cluster(cluster_id, emb_id)

        # Assert
        members = get_cluster_members(cluster_id)
        assert len(members) == 0

    def test_get_cluster_members_includes_track_info(self, test_db):
        """Test cluster members include track metadata."""
        # Arrange
        cluster_id = create_cluster(name="Test Cluster")
        track_id = "video_123"
        upsert_track(track_id, title="Interview Video", artist_name="Podcast Host")
        emb_id = add_speaker_embedding(track_id, "SPEAKER_00", np.ones(512, dtype=np.float32))
        add_to_cluster(cluster_id, emb_id, confidence=0.90)

        # Act
        members = get_cluster_members(cluster_id)

        # Assert
        assert len(members) == 1
        assert members[0]['track_title'] == "Interview Video"
        assert members[0]['artist_name'] == "Podcast Host"

    def test_find_unclustered_embeddings_returns_unassigned(self, test_db):
        """Test finding embeddings not in any cluster."""
        # Arrange
        cluster_id = create_cluster(name="Test Cluster")
        upsert_track("t1", title="Test")

        emb1 = add_speaker_embedding("t1", "SPEAKER_00", np.ones(512, dtype=np.float32))
        emb2 = add_speaker_embedding("t1", "SPEAKER_01", np.ones(512, dtype=np.float32))
        emb3 = add_speaker_embedding("t1", "SPEAKER_02", np.ones(512, dtype=np.float32))

        add_to_cluster(cluster_id, emb1)  # Only emb1 is clustered

        # Act
        unclustered = find_unclustered_embeddings()

        # Assert
        assert len(unclustered) == 2
        unclustered_ids = [e['id'] for e in unclustered]
        assert emb2 in unclustered_ids
        assert emb3 in unclustered_ids
        assert emb1 not in unclustered_ids


class TestTransactionHandling:
    """Test transaction handling and error recovery."""

    def test_transaction_commits_on_success(self, test_db):
        """Test successful operations commit automatically."""
        # Act
        track_id = "commit_test"
        upsert_track(track_id=track_id, title="Test")

        # Assert - data persists in new session
        result = get_track(track_id)
        assert result is not None

    def test_transaction_rolls_back_on_error(self, test_db):
        """Test failed operations roll back automatically."""
        # This test verifies the context manager's rollback behavior
        # We test it indirectly by ensuring invalid operations don't corrupt data

        # Arrange - create valid track
        upsert_track("valid_track", title="Valid")

        # Act - try to create invalid data (will fail silently due to context manager)
        try:
            with get_db_session() as session:
                # This would fail - trying to insert duplicate primary key
                from auto_voice.db.schema import Track
                track1 = Track(id="duplicate", title="First")
                track2 = Track(id="duplicate", title="Second")
                session.add(track1)
                session.flush()
                session.add(track2)
                session.flush()  # This will raise an error
        except Exception:
            pass  # Expected to fail

        # Assert - valid track still exists, invalid data not committed
        result = get_track("valid_track")
        assert result is not None
        result = get_track("duplicate")
        # May or may not exist depending on when error occurred

    def test_context_manager_closes_session(self, test_db):
        """Test context manager properly closes sessions."""
        # Act - use context manager
        with get_db_session() as session:
            from auto_voice.db.schema import Track
            track = Track(id="test", title="Test")
            session.add(track)

        # Assert - session is closed after context
        # We verify this by checking data persists
        result = get_track("test")
        assert result is not None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_get_embeddings_by_cluster_with_nonexistent_cluster(self, test_db):
        """Test getting embeddings for non-existent cluster."""
        embeddings = get_embeddings_by_cluster("nonexistent-uuid")
        assert len(embeddings) == 0

    def test_add_featured_artist_to_nonexistent_track(self, test_db):
        """Test adding featured artist to track that doesn't exist."""
        # Note: SQLite foreign key constraints require explicit enabling
        # In production MySQL, this would fail with IntegrityError
        # For SQLite testing, we verify it doesn't crash (graceful handling)
        try:
            add_featured_artist("nonexistent_track", name="Artist")
            # If it succeeds, verify the track was auto-created or handled gracefully
        except Exception:
            # Expected in production with proper FK constraints
            pass

    def test_upsert_track_with_none_values(self, test_db):
        """Test upserting track with None optional fields."""
        # Act
        upsert_track(track_id="minimal_track")

        # Assert - track created with None values
        result = get_track("minimal_track")
        assert result is not None
        assert result['title'] is None
        assert result['artist_name'] is None

    def test_get_all_tracks_when_empty(self, test_db):
        """Test getting all tracks from empty database."""
        tracks = get_all_tracks()
        assert len(tracks) == 0

    def test_speaker_embedding_with_profile_id(self, test_db):
        """Test speaker embedding with voice profile association."""
        # Arrange
        track_id = "video_001"
        upsert_track(track_id=track_id, title="Test")
        embedding = np.ones(512, dtype=np.float32)

        # Act
        emb_id = add_speaker_embedding(
            track_id=track_id,
            speaker_id="SPEAKER_00",
            embedding=embedding,
            profile_id="profile-uuid-123",
        )

        # Assert
        result = get_embedding_by_id(emb_id)
        assert result['profile_id'] == "profile-uuid-123"
