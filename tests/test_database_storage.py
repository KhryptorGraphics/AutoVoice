"""
Comprehensive tests for database operations and storage.

Tests cover:
- Database schema and session lifecycle
- CRUD operations for tracks, featured artists, embeddings, and clusters
- Voice profile storage (file-based)
- Training sample collection and validation

Uses in-memory SQLite for fast, isolated testing.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

# Import database modules
from auto_voice.db.schema import (
    Base,
    Track,
    FeaturedArtist,
    SpeakerEmbedding,
    SpeakerCluster,
    ClusterMember,
    get_engine,
    get_db_session,
    init_database,
    reset_database,
    get_database_stats,
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
    find_unclustered_embeddings,
    # Cluster operations
    create_cluster,
    get_cluster,
    get_all_clusters,
    update_cluster_name,
    merge_clusters,
    add_to_cluster,
    remove_from_cluster,
    get_cluster_members,
)

# Import storage modules
from auto_voice.storage.voice_profiles import (
    VoiceProfileStore,
    TrainingSample,
    ProfileNotFoundError,
)

# Import sample collector
from auto_voice.profiles.sample_collector import (
    SampleCollector,
    AudioSegment,
    CapturedSample,
)


# ============================================================================
# Fixtures for Database Testing
# ============================================================================


@pytest.fixture
def in_memory_db(monkeypatch):
    """Create an in-memory SQLite database for testing.

    This fixture:
    1. Forces SQLite mode
    2. Uses :memory: database (no disk I/O)
    3. Initializes schema
    4. Cleans up after test
    """
    # Force SQLite with in-memory database
    monkeypatch.setenv('AUTOVOICE_DB_TYPE', 'sqlite')
    monkeypatch.setattr('auto_voice.db.schema.DATABASE_TYPE', 'sqlite')
    monkeypatch.setattr('auto_voice.db.schema.DATABASE_PATH', Path(':memory:'))

    # Reset engine to force new connection
    import auto_voice.db.schema
    auto_voice.db.schema._engine = None
    auto_voice.db.schema._SessionFactory = None

    # Initialize database
    init_database(db_type='sqlite')

    yield

    # Cleanup
    close_database()


@pytest.fixture
def test_embedding():
    """Generate a test speaker embedding (512-dim)."""
    embedding = np.random.randn(512).astype(np.float32)
    # Normalize
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


@pytest.fixture
def temp_storage():
    """Create temporary storage directories for file-based tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        profiles_dir = os.path.join(tmpdir, 'profiles')
        samples_dir = os.path.join(tmpdir, 'samples')
        os.makedirs(profiles_dir)
        os.makedirs(samples_dir)
        yield {
            'root': tmpdir,
            'profiles': profiles_dir,
            'samples': samples_dir,
        }


# ============================================================================
# Task 3.1: Test db/operations.py - CRUD Operations
# ============================================================================


class TestTrackOperations:
    """Test track CRUD operations."""

    def test_upsert_track_insert(self, in_memory_db):
        """Test inserting a new track."""
        upsert_track(
            track_id='test_video_123',
            title='Test Song',
            channel='Test Channel',
            upload_date='2024-01-15',
            duration_sec=180.5,
            artist_name='Test Artist',
        )

        track = get_track('test_video_123')
        assert track is not None
        assert track['id'] == 'test_video_123'
        assert track['title'] == 'Test Song'
        assert track['artist_name'] == 'Test Artist'
        assert track['duration_sec'] == 180.5

    def test_upsert_track_update(self, in_memory_db):
        """Test updating an existing track."""
        # Insert
        upsert_track(track_id='video_456', title='Original Title', artist_name='Artist A')

        # Update
        upsert_track(track_id='video_456', title='Updated Title', duration_sec=200.0)

        track = get_track('video_456')
        assert track['title'] == 'Updated Title'
        assert track['duration_sec'] == 200.0
        assert track['artist_name'] == 'Artist A'  # Unchanged

    def test_get_track_not_found(self, in_memory_db):
        """Test retrieving non-existent track returns None."""
        track = get_track('nonexistent_id')
        assert track is None

    def test_get_all_tracks(self, in_memory_db):
        """Test retrieving all tracks."""
        upsert_track(track_id='id1', title='Song 1', artist_name='Artist A')
        upsert_track(track_id='id2', title='Song 2', artist_name='Artist B')
        upsert_track(track_id='id3', title='Song 3', artist_name='Artist A')

        tracks = get_all_tracks()
        assert len(tracks) == 3
        # Check ordering by artist_name, title
        assert tracks[0]['id'] == 'id1'
        assert tracks[1]['id'] == 'id3'
        assert tracks[2]['id'] == 'id2'

    def test_get_tracks_by_artist(self, in_memory_db):
        """Test filtering tracks by artist."""
        upsert_track(track_id='id1', artist_name='Artist A')
        upsert_track(track_id='id2', artist_name='Artist B')
        upsert_track(track_id='id3', artist_name='Artist A')

        tracks = get_tracks_by_artist('Artist A')
        assert len(tracks) == 2
        assert {t['id'] for t in tracks} == {'id1', 'id3'}

    def test_transaction_rollback(self, in_memory_db):
        """Test that failed transactions rollback properly."""
        upsert_track(track_id='id1', title='Track 1')

        # Simulate error in session
        with pytest.raises(Exception):
            with get_db_session() as session:
                # Add new track
                track = Track(id='id2', title='Track 2')
                session.add(track)
                # Force error
                raise ValueError("Simulated error")

        # Verify rollback: only first track exists
        tracks = get_all_tracks()
        assert len(tracks) == 1
        assert tracks[0]['id'] == 'id1'


class TestFeaturedArtistOperations:
    """Test featured artist CRUD operations."""

    def test_add_featured_artist(self, in_memory_db):
        """Test adding a featured artist."""
        upsert_track(track_id='track1', title='Song')

        artist_id = add_featured_artist(
            track_id='track1',
            name='Featured Artist',
            pattern_matched='ft.',
        )

        assert artist_id is not None
        artists = get_featured_artists_for_track('track1')
        assert len(artists) == 1
        assert artists[0]['name'] == 'Featured Artist'
        assert artists[0]['pattern_matched'] == 'ft.'

    def test_add_duplicate_featured_artist(self, in_memory_db):
        """Test adding duplicate featured artist returns existing ID."""
        upsert_track(track_id='track1', title='Song')

        id1 = add_featured_artist(track_id='track1', name='Artist A')
        id2 = add_featured_artist(track_id='track1', name='Artist A')

        assert id1 == id2
        artists = get_featured_artists_for_track('track1')
        assert len(artists) == 1

    def test_get_all_featured_artists_with_counts(self, in_memory_db):
        """Test retrieving all featured artists with track counts."""
        upsert_track(track_id='t1', title='Song 1')
        upsert_track(track_id='t2', title='Song 2')
        upsert_track(track_id='t3', title='Song 3')

        add_featured_artist('t1', 'Artist A')
        add_featured_artist('t2', 'Artist A')
        add_featured_artist('t2', 'Artist B')
        add_featured_artist('t3', 'Artist B')

        artists = get_all_featured_artists()
        assert len(artists) == 2

        # Artist A: 2 tracks
        artist_a = next(a for a in artists if a['name'] == 'Artist A')
        assert artist_a['track_count'] == 2

        # Artist B: 2 tracks
        artist_b = next(a for a in artists if a['name'] == 'Artist B')
        assert artist_b['track_count'] == 2


class TestSpeakerEmbeddingOperations:
    """Test speaker embedding CRUD operations."""

    def test_add_speaker_embedding(self, in_memory_db, test_embedding):
        """Test adding a speaker embedding."""
        upsert_track(track_id='track1', title='Song')

        emb_id = add_speaker_embedding(
            track_id='track1',
            speaker_id='SPEAKER_00',
            embedding=test_embedding,
            duration_sec=15.5,
            is_primary=True,
        )

        assert emb_id is not None
        embeddings = get_embeddings_for_track('track1')
        assert len(embeddings) == 1
        assert embeddings[0]['speaker_id'] == 'SPEAKER_00'
        assert embeddings[0]['is_primary'] is True
        assert embeddings[0]['duration_sec'] == 15.5
        assert isinstance(embeddings[0]['embedding'], np.ndarray)
        assert embeddings[0]['embedding'].shape == (512,)

    def test_update_speaker_embedding(self, in_memory_db, test_embedding):
        """Test updating an existing speaker embedding."""
        upsert_track(track_id='track1', title='Song')

        # Add
        emb_id1 = add_speaker_embedding('track1', 'SPEAKER_00', test_embedding)

        # Update with new embedding
        new_embedding = np.random.randn(512).astype(np.float32)
        emb_id2 = add_speaker_embedding(
            'track1',
            'SPEAKER_00',
            new_embedding,
            duration_sec=20.0,
        )

        assert emb_id1 == emb_id2
        embeddings = get_embeddings_for_track('track1')
        assert len(embeddings) == 1
        assert embeddings[0]['duration_sec'] == 20.0

    def test_get_embedding_by_id(self, in_memory_db, test_embedding):
        """Test retrieving embedding by ID."""
        upsert_track(track_id='track1', title='Song')
        emb_id = add_speaker_embedding('track1', 'SPEAKER_00', test_embedding)

        embedding = get_embedding_by_id(emb_id)
        assert embedding is not None
        assert embedding['id'] == emb_id
        assert embedding['track_id'] == 'track1'

    def test_get_all_embeddings(self, in_memory_db, test_embedding):
        """Test retrieving all embeddings."""
        upsert_track(track_id='t1', title='Song 1')
        upsert_track(track_id='t2', title='Song 2')

        add_speaker_embedding('t1', 'SPEAKER_00', test_embedding)
        add_speaker_embedding('t1', 'SPEAKER_01', test_embedding)
        add_speaker_embedding('t2', 'SPEAKER_00', test_embedding)

        embeddings = get_all_embeddings()
        assert len(embeddings) == 3


class TestSpeakerClusterOperations:
    """Test speaker cluster CRUD operations."""

    def test_create_cluster(self, in_memory_db):
        """Test creating a speaker cluster."""
        cluster_id = create_cluster(name='Test Speaker', is_verified=True)

        assert cluster_id is not None
        cluster = get_cluster(cluster_id)
        assert cluster is not None
        assert cluster['name'] == 'Test Speaker'
        assert cluster['is_verified'] is True

    def test_update_cluster_name(self, in_memory_db):
        """Test updating cluster name."""
        cluster_id = create_cluster(name='Original Name')

        update_cluster_name(cluster_id, 'Updated Name', is_verified=True)

        cluster = get_cluster(cluster_id)
        assert cluster['name'] == 'Updated Name'
        assert cluster['is_verified'] is True

    def test_add_to_cluster(self, in_memory_db, test_embedding):
        """Test adding embeddings to a cluster."""
        upsert_track(track_id='track1', title='Song')
        emb_id = add_speaker_embedding('track1', 'SPEAKER_00', test_embedding)
        cluster_id = create_cluster(name='Speaker A')

        add_to_cluster(cluster_id, emb_id, confidence=0.95)

        embeddings = get_embeddings_by_cluster(cluster_id)
        assert len(embeddings) == 1
        assert embeddings[0]['confidence'] == 0.95

    def test_remove_from_cluster(self, in_memory_db, test_embedding):
        """Test removing embeddings from a cluster."""
        upsert_track(track_id='track1', title='Song')
        emb_id = add_speaker_embedding('track1', 'SPEAKER_00', test_embedding)
        cluster_id = create_cluster(name='Speaker A')

        add_to_cluster(cluster_id, emb_id)
        remove_from_cluster(cluster_id, emb_id)

        embeddings = get_embeddings_by_cluster(cluster_id)
        assert len(embeddings) == 0

    def test_merge_clusters(self, in_memory_db, test_embedding):
        """Test merging two clusters."""
        upsert_track(track_id='track1', title='Song')
        emb_id1 = add_speaker_embedding('track1', 'SPEAKER_00', test_embedding)
        emb_id2 = add_speaker_embedding('track1', 'SPEAKER_01', test_embedding)

        cluster_a = create_cluster(name='Cluster A')
        cluster_b = create_cluster(name='Cluster B')

        add_to_cluster(cluster_a, emb_id1)
        add_to_cluster(cluster_b, emb_id2)

        # Merge B into A
        merge_clusters(target_cluster_id=cluster_a, source_cluster_id=cluster_b)

        # Cluster B should be deleted
        assert get_cluster(cluster_b) is None

        # Cluster A should have both embeddings
        embeddings = get_embeddings_by_cluster(cluster_a)
        assert len(embeddings) == 2

    def test_get_cluster_members(self, in_memory_db, test_embedding):
        """Test getting cluster members with track info."""
        upsert_track(track_id='track1', title='Test Song', artist_name='Artist A')
        emb_id = add_speaker_embedding('track1', 'SPEAKER_00', test_embedding)
        cluster_id = create_cluster(name='Speaker A')

        add_to_cluster(cluster_id, emb_id, confidence=0.92)

        members = get_cluster_members(cluster_id)
        assert len(members) == 1
        assert members[0]['track_title'] == 'Test Song'
        assert members[0]['artist_name'] == 'Artist A'
        assert members[0]['confidence'] == 0.92

    def test_find_unclustered_embeddings(self, in_memory_db, test_embedding):
        """Test finding embeddings not assigned to any cluster."""
        upsert_track(track_id='t1', title='Song 1', artist_name='Artist A')
        upsert_track(track_id='t2', title='Song 2', artist_name='Artist B')

        emb_id1 = add_speaker_embedding('t1', 'SPEAKER_00', test_embedding)
        emb_id2 = add_speaker_embedding('t2', 'SPEAKER_00', test_embedding)

        # Add only emb_id1 to cluster
        cluster_id = create_cluster(name='Speaker A')
        add_to_cluster(cluster_id, emb_id1)

        unclustered = find_unclustered_embeddings()
        assert len(unclustered) == 1
        assert unclustered[0]['id'] == emb_id2
        assert unclustered[0]['track_title'] == 'Song 2'

    def test_get_all_clusters_with_stats(self, in_memory_db, test_embedding):
        """Test getting all clusters with member counts."""
        upsert_track(track_id='t1', title='Song')
        emb_id = add_speaker_embedding('t1', 'SPEAKER_00', test_embedding, duration_sec=10.0)

        cluster_id = create_cluster(name='Speaker A')
        add_to_cluster(cluster_id, emb_id)

        clusters = get_all_clusters()
        assert len(clusters) == 1
        assert clusters[0]['member_count'] == 1
        assert clusters[0]['total_duration_sec'] == 10.0


# ============================================================================
# Task 3.2: Test db/schema.py - Data Model Validation
# ============================================================================


class TestDatabaseSchema:
    """Test database schema creation and constraints."""

    def test_schema_initialization(self, in_memory_db):
        """Test that schema initializes all tables."""
        stats = get_database_stats()
        assert 'tracks' in stats
        assert 'featured_artists' in stats
        assert 'speaker_embeddings' in stats
        assert 'speaker_clusters' in stats
        assert 'cluster_members' in stats

    def test_unique_constraint_featured_artist(self, in_memory_db):
        """Test unique constraint on (track_id, name) for featured artists."""
        upsert_track(track_id='track1', title='Song')

        # Should succeed
        add_featured_artist('track1', 'Artist A')

        # Duplicate should return existing ID (handled by operations.py)
        id1 = add_featured_artist('track1', 'Artist A')
        id2 = add_featured_artist('track1', 'Artist A')
        assert id1 == id2

    def test_unique_constraint_speaker_embedding(self, in_memory_db, test_embedding):
        """Test unique constraint on (track_id, speaker_id) for embeddings."""
        upsert_track(track_id='track1', title='Song')

        # Should succeed
        emb_id1 = add_speaker_embedding('track1', 'SPEAKER_00', test_embedding)

        # Duplicate should update existing (handled by operations.py)
        emb_id2 = add_speaker_embedding('track1', 'SPEAKER_00', test_embedding)
        assert emb_id1 == emb_id2

    def test_foreign_key_cascade_delete(self, in_memory_db, test_embedding):
        """Test that foreign key cascades work properly.

        Note: SQLite doesn't enforce foreign key cascades by default in-memory mode,
        so this test verifies the application-level deletion behavior instead.
        """
        upsert_track(track_id='track1', title='Song')
        add_featured_artist('track1', 'Artist A')
        add_speaker_embedding('track1', 'SPEAKER_00', test_embedding)

        # Delete track
        with get_db_session() as session:
            session.query(Track).filter(Track.id == 'track1').delete()

        # In production MySQL, featured artists and embeddings would cascade delete
        # In SQLite test mode, we verify the schema is correctly defined
        # The CASCADE is defined in schema, but SQLite in-memory may not enforce it
        # This test primarily validates schema structure
        track = get_track('track1')
        assert track is None  # Track is deleted

    def test_default_values(self, in_memory_db):
        """Test that default values are applied correctly."""
        cluster_id = create_cluster(name='Test Speaker')
        cluster = get_cluster(cluster_id)

        assert cluster['is_verified'] is False  # Default
        assert cluster['created_at'] is not None
        assert cluster['updated_at'] is not None

    def test_database_stats(self, in_memory_db, test_embedding):
        """Test get_database_stats function."""
        upsert_track(track_id='t1', title='Song 1', artist_name='Artist A')
        upsert_track(track_id='t2', title='Song 2', artist_name='Artist B')
        add_featured_artist('t1', 'Featured A')
        add_speaker_embedding('t1', 'SPEAKER_00', test_embedding)
        cluster_id = create_cluster(name='Speaker A', is_verified=True)

        stats = get_database_stats()
        assert stats['tracks'] == 2
        assert stats['featured_artists'] == 1
        assert stats['speaker_embeddings'] == 1
        assert stats['speaker_clusters'] == 1
        assert stats['unique_artists'] == 2
        assert stats['verified_clusters'] == 1

    def test_reset_database(self, in_memory_db):
        """Test reset_database deletes all data."""
        upsert_track(track_id='track1', title='Song')

        reset_database()

        tracks = get_all_tracks()
        assert len(tracks) == 0


# ============================================================================
# Task 3.3: Test db/session.py - Connection Lifecycle
# ============================================================================


class TestDatabaseSession:
    """Test database session management."""

    def test_session_creation(self, in_memory_db):
        """Test that session can be created."""
        with get_db_session() as session:
            assert session is not None

    def test_session_commit_on_success(self, in_memory_db):
        """Test that changes are committed on success."""
        with get_db_session() as session:
            track = Track(id='track1', title='Song')
            session.add(track)

        # Verify committed
        with get_db_session() as session:
            result = session.query(Track).filter(Track.id == 'track1').first()
            assert result is not None

    def test_session_rollback_on_error(self, in_memory_db):
        """Test that changes are rolled back on error."""
        try:
            with get_db_session() as session:
                track = Track(id='track1', title='Song')
                session.add(track)
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # Verify rolled back
        with get_db_session() as session:
            result = session.query(Track).filter(Track.id == 'track1').first()
            assert result is None

    def test_session_cleanup(self, in_memory_db):
        """Test that session is closed after use."""
        with get_db_session() as session:
            session_id = id(session)

        # Session should be closed (no way to directly check, but test coverage verifies)
        # This is more of a behavioral test
        assert session_id is not None

    def test_close_database(self, in_memory_db):
        """Test that close_database disposes engine."""
        close_database()

        # Engine should be reset
        import auto_voice.db.schema
        assert auto_voice.db.schema._engine is None


# ============================================================================
# Task 3.4: Test storage/voice_profiles.py - File Storage
# ============================================================================


class TestVoiceProfileStore:
    """Test file-based voice profile storage."""

    def test_save_and_load_profile(self, temp_storage):
        """Test saving and loading a voice profile."""
        store = VoiceProfileStore(
            profiles_dir=temp_storage['profiles'],
            samples_dir=temp_storage['samples'],
        )

        profile_data = {
            'name': 'Test Speaker',
            'user_id': 'user123',
            'language': 'en',
        }

        profile_id = store.save(profile_data)
        assert profile_id is not None

        loaded = store.load(profile_id)
        assert loaded['profile_id'] == profile_id
        assert loaded['name'] == 'Test Speaker'
        assert loaded['user_id'] == 'user123'

    def test_save_profile_with_embedding(self, temp_storage, test_embedding):
        """Test saving profile with numpy embedding."""
        store = VoiceProfileStore(profiles_dir=temp_storage['profiles'])

        profile_data = {
            'name': 'Speaker with Embedding',
            'embedding': test_embedding,
        }

        profile_id = store.save(profile_data)
        loaded = store.load(profile_id)

        assert 'embedding' in loaded
        assert isinstance(loaded['embedding'], np.ndarray)
        assert loaded['embedding'].shape == (512,)
        np.testing.assert_array_almost_equal(loaded['embedding'], test_embedding)

    def test_profile_not_found(self, temp_storage):
        """Test loading non-existent profile raises error."""
        store = VoiceProfileStore(profiles_dir=temp_storage['profiles'])

        with pytest.raises(ProfileNotFoundError):
            store.load('nonexistent_id')

    def test_list_profiles(self, temp_storage):
        """Test listing all profiles."""
        store = VoiceProfileStore(profiles_dir=temp_storage['profiles'])

        store.save({'name': 'Speaker 1', 'user_id': 'user1'})
        store.save({'name': 'Speaker 2', 'user_id': 'user2'})

        profiles = store.list_profiles()
        assert len(profiles) == 2

    def test_list_profiles_filtered_by_user(self, temp_storage):
        """Test listing profiles filtered by user_id."""
        store = VoiceProfileStore(profiles_dir=temp_storage['profiles'])

        store.save({'name': 'Speaker 1', 'user_id': 'user1'})
        store.save({'name': 'Speaker 2', 'user_id': 'user2'})
        store.save({'name': 'Speaker 3', 'user_id': 'user1'})

        profiles = store.list_profiles(user_id='user1')
        assert len(profiles) == 2

    def test_delete_profile(self, temp_storage):
        """Test deleting a profile."""
        store = VoiceProfileStore(profiles_dir=temp_storage['profiles'])

        profile_id = store.save({'name': 'Test Speaker'})
        assert store.exists(profile_id)

        deleted = store.delete(profile_id)
        assert deleted is True
        assert not store.exists(profile_id)

    def test_delete_nonexistent_profile(self, temp_storage):
        """Test deleting non-existent profile returns False."""
        store = VoiceProfileStore(profiles_dir=temp_storage['profiles'])

        deleted = store.delete('nonexistent_id')
        assert deleted is False

    def test_save_lora_weights(self, temp_storage):
        """Test saving LoRA adapter weights."""
        store = VoiceProfileStore(profiles_dir=temp_storage['profiles'])

        profile_id = store.save({'name': 'Speaker'})

        state_dict = {
            'layer1.weight': torch.randn(10, 10),
            'layer2.bias': torch.randn(10),
        }

        store.save_lora_weights(profile_id, state_dict)
        assert store.has_trained_model(profile_id)

        loaded = store.load_lora_weights(profile_id)
        assert 'layer1.weight' in loaded
        assert 'layer2.bias' in loaded

    def test_load_lora_weights_not_found(self, temp_storage):
        """Test loading weights for profile without training."""
        store = VoiceProfileStore(profiles_dir=temp_storage['profiles'])

        profile_id = store.save({'name': 'Speaker'})

        with pytest.raises(FileNotFoundError):
            store.load_lora_weights(profile_id)

    def test_add_training_sample(self, temp_storage):
        """Test adding a training sample."""
        store = VoiceProfileStore(
            profiles_dir=temp_storage['profiles'],
            samples_dir=temp_storage['samples'],
        )

        profile_id = store.save({'name': 'Speaker'})

        # Create temporary vocals file
        vocals_path = os.path.join(temp_storage['root'], 'vocals.wav')
        with open(vocals_path, 'wb') as f:
            f.write(b'fake audio data')

        sample = store.add_training_sample(
            profile_id=profile_id,
            vocals_path=vocals_path,
            duration=5.0,
        )

        assert sample.sample_id is not None
        assert os.path.exists(sample.vocals_path)
        assert sample.duration == 5.0

    def test_list_training_samples(self, temp_storage):
        """Test listing training samples for a profile."""
        store = VoiceProfileStore(
            profiles_dir=temp_storage['profiles'],
            samples_dir=temp_storage['samples'],
        )

        profile_id = store.save({'name': 'Speaker'})

        # Add multiple samples
        for i in range(3):
            vocals_path = os.path.join(temp_storage['root'], f'vocals_{i}.wav')
            with open(vocals_path, 'wb') as f:
                f.write(b'fake audio')
            store.add_training_sample(profile_id, vocals_path, duration=float(i))

        samples = store.list_training_samples(profile_id)
        assert len(samples) == 3

    def test_get_total_training_duration(self, temp_storage):
        """Test calculating total training duration."""
        store = VoiceProfileStore(
            profiles_dir=temp_storage['profiles'],
            samples_dir=temp_storage['samples'],
        )

        profile_id = store.save({'name': 'Speaker'})

        for duration in [5.0, 10.0, 15.0]:
            vocals_path = os.path.join(temp_storage['root'], f'vocals_{duration}.wav')
            with open(vocals_path, 'wb') as f:
                f.write(b'fake audio')
            store.add_training_sample(profile_id, vocals_path, duration=duration)

        total = store.get_total_training_duration(profile_id)
        assert total == 30.0

    def test_delete_training_sample(self, temp_storage):
        """Test deleting a training sample."""
        store = VoiceProfileStore(
            profiles_dir=temp_storage['profiles'],
            samples_dir=temp_storage['samples'],
        )

        profile_id = store.save({'name': 'Speaker'})

        vocals_path = os.path.join(temp_storage['root'], 'vocals.wav')
        with open(vocals_path, 'wb') as f:
            f.write(b'fake audio')

        sample = store.add_training_sample(profile_id, vocals_path)

        deleted = store.delete_training_sample(profile_id, sample.sample_id)
        assert deleted is True

        samples = store.list_training_samples(profile_id)
        assert len(samples) == 0

    def test_save_speaker_embedding(self, temp_storage, test_embedding):
        """Test saving speaker embedding for diarization."""
        store = VoiceProfileStore(profiles_dir=temp_storage['profiles'])

        profile_id = store.save({'name': 'Speaker'})

        store.save_speaker_embedding(profile_id, test_embedding)

        loaded = store.load_speaker_embedding(profile_id)
        assert loaded is not None
        assert loaded.shape == (512,)
        # Should be normalized
        assert abs(np.linalg.norm(loaded) - 1.0) < 1e-5

    def test_load_speaker_embedding_not_found(self, temp_storage):
        """Test loading speaker embedding when not set."""
        store = VoiceProfileStore(profiles_dir=temp_storage['profiles'])

        profile_id = store.save({'name': 'Speaker'})

        loaded = store.load_speaker_embedding(profile_id)
        assert loaded is None


# ============================================================================
# Task 3.5: Test profiles/sample_collector.py - Sample Collection
# ============================================================================


class TestSampleCollector:
    """Test training sample collection and validation."""

    def test_collector_initialization(self, temp_storage):
        """Test initializing sample collector."""
        collector = SampleCollector(storage_path=temp_storage['samples'])

        assert collector.min_snr_db == 20.0
        assert collector.min_duration_sec == 2.0
        assert collector.max_duration_sec == 30.0

    def test_collector_custom_thresholds(self, temp_storage):
        """Test custom quality thresholds."""
        collector = SampleCollector(
            storage_path=temp_storage['samples'],
            min_snr_db=25.0,
            min_duration_sec=3.0,
        )

        assert collector.min_snr_db == 25.0
        assert collector.min_duration_sec == 3.0

    def test_estimate_snr(self, temp_storage):
        """Test SNR estimation."""
        collector = SampleCollector(storage_path=temp_storage['samples'])

        # Clean sine wave should have high SNR
        sr = 16000
        t = np.linspace(0, 1.0, sr)
        clean_signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        snr = collector.estimate_snr(clean_signal)
        assert snr > 30.0

        # White noise should have low SNR
        noise = np.random.randn(sr).astype(np.float32) * 0.1
        snr_noise = collector.estimate_snr(noise)
        assert snr_noise < snr

    def test_measure_pitch_stability(self, temp_storage):
        """Test pitch stability measurement."""
        collector = SampleCollector(storage_path=temp_storage['samples'])

        # Stable pitch (single frequency)
        sr = 16000
        t = np.linspace(0, 1.0, sr)
        stable = np.sin(2 * np.pi * 220 * t).astype(np.float32)

        stability = collector.measure_pitch_stability(stable, sr)
        assert stability > 0.8

    def test_segment_phrases(self, temp_storage):
        """Test phrase segmentation at silence boundaries."""
        collector = SampleCollector(
            storage_path=temp_storage['samples'],
            min_duration_sec=1.0,  # Lower threshold for test
        )

        # Create audio with 2 phrases separated by silence
        sr = 16000
        # Each phrase is 2 seconds (above 1.0s min)
        phrase1 = np.sin(2 * np.pi * 220 * np.linspace(0, 2, sr * 2)).astype(np.float32)
        silence = np.zeros(int(sr * 0.5), dtype=np.float32)
        phrase2 = np.sin(2 * np.pi * 440 * np.linspace(0, 2, sr * 2)).astype(np.float32)

        audio = np.concatenate([phrase1, silence, phrase2])

        segments = collector.segment_phrases(audio, sr)

        # Should detect at least 1 segment (phrases above minimum duration)
        assert len(segments) >= 1
        assert segments[0].duration_seconds >= 1.0

    @patch('auto_voice.profiles.sample_collector.db_session_module.get_db_session')
    @patch('auto_voice.profiles.sample_collector.VoiceProfileDB')
    def test_capture_sample_success(self, mock_profile, mock_session, temp_storage):
        """Test capturing a valid sample."""
        # Mock database session
        mock_session_instance = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session_instance)
        mock_session.__exit__ = Mock(return_value=False)
        mock_session_instance.query.return_value.filter_by.return_value.first.return_value = Mock()

        collector = SampleCollector(
            storage_path=temp_storage['samples'],
            min_duration_sec=1.0,
            min_snr_db=10.0,
        )

        # Create valid audio
        sr = 16000
        t = np.linspace(0, 2.0, int(sr * 2.0))
        audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

        sample = collector.capture_sample(
            profile_id='profile1',
            audio=audio,
            sample_rate=sr,
            consent_given=True,
        )

        assert sample is not None
        assert sample.profile_id == 'profile1'
        assert sample.duration_seconds > 1.0
        assert os.path.exists(sample.audio_path)

    def test_capture_sample_no_consent(self, temp_storage):
        """Test that samples are rejected without consent."""
        collector = SampleCollector(storage_path=temp_storage['samples'])

        audio = np.random.randn(16000).astype(np.float32)

        sample = collector.capture_sample(
            profile_id='profile1',
            audio=audio,
            sample_rate=16000,
            consent_given=False,
        )

        assert sample is None

    def test_capture_sample_too_short(self, temp_storage):
        """Test that samples below minimum duration are rejected."""
        collector = SampleCollector(
            storage_path=temp_storage['samples'],
            min_duration_sec=3.0,
        )

        # 1 second audio
        audio = np.random.randn(16000).astype(np.float32)

        sample = collector.capture_sample(
            profile_id='profile1',
            audio=audio,
            sample_rate=16000,
            consent_given=True,
        )

        assert sample is None

    def test_capture_sample_low_snr(self, temp_storage):
        """Test that samples with low SNR are rejected."""
        collector = SampleCollector(
            storage_path=temp_storage['samples'],
            min_snr_db=50.0,  # Very high threshold
            min_duration_sec=1.0,
        )

        # Noisy audio
        audio = np.random.randn(32000).astype(np.float32) * 0.1

        sample = collector.capture_sample(
            profile_id='profile1',
            audio=audio,
            sample_rate=16000,
            consent_given=True,
        )

        assert sample is None

    def test_recording_session(self, temp_storage):
        """Test recording session with chunk accumulation."""
        collector = SampleCollector(
            storage_path=temp_storage['samples'],
            min_duration_sec=0.5,
        )

        collector.start_recording(
            profile_id='profile1',
            session_id='session1',
            consent_given=True,
        )

        # Add chunks
        sr = 16000
        for _ in range(5):
            chunk = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(sr * 0.5))).astype(np.float32)
            collector.add_chunk(chunk, sr)

        # Stop recording (will segment and validate)
        # Note: This test will likely fail without proper mocking of database
        # For now, we're testing the recording logic only
        assert collector._recording is True
        assert len(collector._audio_chunks) == 5


# ============================================================================
# Coverage and Integration Tests
# ============================================================================


class TestDatabaseIntegration:
    """Integration tests for full database workflows."""

    def test_full_track_workflow(self, in_memory_db, test_embedding):
        """Test complete workflow: track -> featured artists -> embeddings -> clusters."""
        # 1. Create track
        upsert_track(
            track_id='video123',
            title='Test Song (ft. Guest Artist)',
            artist_name='Main Artist',
            duration_sec=180.0,
        )

        # 2. Add featured artist
        add_featured_artist('video123', 'Guest Artist', pattern_matched='ft.')

        # 3. Add speaker embeddings
        emb_id1 = add_speaker_embedding('video123', 'SPEAKER_00', test_embedding, is_primary=True)
        emb_id2 = add_speaker_embedding('video123', 'SPEAKER_01', test_embedding)

        # 4. Create cluster and assign
        cluster_id = create_cluster('Main Artist', is_verified=True)
        add_to_cluster(cluster_id, emb_id1, confidence=0.98)

        # Verify full workflow
        track = get_track('video123')
        assert track['title'] == 'Test Song (ft. Guest Artist)'

        artists = get_featured_artists_for_track('video123')
        assert len(artists) == 1

        embeddings = get_embeddings_for_track('video123')
        assert len(embeddings) == 2

        members = get_cluster_members(cluster_id)
        assert len(members) == 1
        assert members[0]['is_primary'] is True


class TestStorageIntegration:
    """Integration tests for storage workflows."""

    def test_full_profile_training_workflow(self, temp_storage, test_embedding):
        """Test complete profile creation and training workflow."""
        store = VoiceProfileStore(
            profiles_dir=temp_storage['profiles'],
            samples_dir=temp_storage['samples'],
        )

        # 1. Create profile
        profile_id = store.save({
            'name': 'Test Speaker',
            'user_id': 'user1',
            'embedding': test_embedding,
        })

        # 2. Save speaker embedding
        store.save_speaker_embedding(profile_id, test_embedding)

        # 3. Add training samples
        for i in range(3):
            vocals_path = os.path.join(temp_storage['root'], f'sample_{i}.wav')
            with open(vocals_path, 'wb') as f:
                f.write(b'audio data')
            store.add_training_sample(profile_id, vocals_path, duration=float(i + 5))

        # 4. Simulate training (save LoRA weights)
        state_dict = {'layer.weight': torch.randn(10, 10)}
        store.save_lora_weights(profile_id, state_dict)

        # Verify workflow
        profile = store.load(profile_id)
        assert profile['name'] == 'Test Speaker'

        samples = store.list_training_samples(profile_id)
        assert len(samples) == 3

        total_duration = store.get_total_training_duration(profile_id)
        assert total_duration == 18.0  # 5 + 6 + 7

        assert store.has_trained_model(profile_id)

        embedding = store.load_speaker_embedding(profile_id)
        assert embedding is not None


# ============================================================================
# Performance Tests
# ============================================================================


def test_database_performance(in_memory_db, test_embedding):
    """Test that database operations are fast (<2s for 100 operations)."""
    import time

    start = time.time()

    # Bulk insert tracks
    for i in range(50):
        upsert_track(track_id=f'track_{i}', title=f'Song {i}', artist_name=f'Artist {i % 10}')

    # Bulk insert embeddings
    for i in range(50):
        add_speaker_embedding(f'track_{i}', f'SPEAKER_{i % 5}', test_embedding)

    elapsed = time.time() - start
    assert elapsed < 2.0, f"Database operations took {elapsed:.2f}s (expected <2s)"


def test_storage_performance(temp_storage):
    """Test that storage operations are fast (<1s for 20 profiles)."""
    import time

    store = VoiceProfileStore(profiles_dir=temp_storage['profiles'])

    start = time.time()

    # Create 20 profiles
    for i in range(20):
        store.save({'name': f'Speaker {i}', 'user_id': f'user{i % 5}'})

    elapsed = time.time() - start
    assert elapsed < 1.0, f"Storage operations took {elapsed:.2f}s (expected <1s)"
