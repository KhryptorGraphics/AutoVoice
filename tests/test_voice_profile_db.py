"""Tests for VoiceProfile database models and operations.

Task 1.2: Test PostgreSQL schema for voice_profiles and training_samples tables.
"""

import pytest
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from auto_voice.profiles.db.models import Base, VoiceProfileDB, TrainingSampleDB


@pytest.fixture
def test_engine():
    """Create a SQLite in-memory database for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture
def test_session(test_engine):
    """Create a database session for testing."""
    Session = sessionmaker(bind=test_engine)
    session = Session()
    yield session
    session.close()


class TestVoiceProfileDB:
    """Test VoiceProfileDB SQLAlchemy model."""

    def test_create_voice_profile(self, test_session):
        """Profile can be created and persisted."""
        profile = VoiceProfileDB(
            user_id="user-123",
            name="My Singing Voice",
        )
        test_session.add(profile)
        test_session.commit()

        # Retrieve and verify
        retrieved = test_session.query(VoiceProfileDB).filter_by(user_id="user-123").first()
        assert retrieved is not None
        assert retrieved.user_id == "user-123"
        assert retrieved.name == "My Singing Voice"
        assert retrieved.samples_count == 0
        assert retrieved.model_version is None

    def test_profile_id_is_uuid(self, test_session):
        """Profile ID is a valid UUID string."""
        profile = VoiceProfileDB(user_id="user-1", name="Test")
        test_session.add(profile)
        test_session.commit()

        # Should be valid UUID
        UUID(profile.id)  # Raises if invalid

    def test_profile_timestamps(self, test_session):
        """Profile has created and updated timestamps."""
        before = datetime.now(timezone.utc)
        profile = VoiceProfileDB(user_id="user-1", name="Test")
        test_session.add(profile)
        test_session.commit()
        after = datetime.now(timezone.utc)

        assert profile.created is not None
        assert profile.updated is not None
        # Note: SQLite doesn't have timezone support, so we just check existence

    def test_profile_to_dict(self, test_session):
        """Profile can be serialized to dictionary."""
        profile = VoiceProfileDB(
            user_id="user-123",
            name="Test Profile",
            model_version="v1.0.0",
        )
        test_session.add(profile)
        test_session.commit()

        data = profile.to_dict()
        assert data["user_id"] == "user-123"
        assert data["name"] == "Test Profile"
        assert data["model_version"] == "v1.0.0"
        assert "id" in data
        assert "created" in data
        assert "updated" in data

    def test_profile_settings_json(self, test_session):
        """Profile can store JSON settings."""
        profile = VoiceProfileDB(
            user_id="user-1",
            name="Test",
            settings={"pitch_shift": 2, "formant_shift": 0.5},
        )
        test_session.add(profile)
        test_session.commit()

        retrieved = test_session.query(VoiceProfileDB).filter_by(user_id="user-1").first()
        assert retrieved.settings["pitch_shift"] == 2
        assert retrieved.settings["formant_shift"] == 0.5


class TestTrainingSampleDB:
    """Test TrainingSampleDB SQLAlchemy model."""

    def test_create_training_sample(self, test_session):
        """Training sample can be created with profile reference."""
        # Create profile first
        profile = VoiceProfileDB(user_id="user-1", name="Test")
        test_session.add(profile)
        test_session.commit()

        # Create sample
        sample = TrainingSampleDB(
            profile_id=profile.id,
            audio_path="/data/sample.wav",
            duration_seconds=5.5,
            sample_rate=24000,
        )
        test_session.add(sample)
        test_session.commit()

        # Retrieve and verify
        retrieved = test_session.query(TrainingSampleDB).filter_by(profile_id=profile.id).first()
        assert retrieved is not None
        assert retrieved.audio_path == "/data/sample.wav"
        assert retrieved.duration_seconds == 5.5
        assert retrieved.sample_rate == 24000

    def test_sample_quality_score(self, test_session):
        """Sample can have quality score."""
        profile = VoiceProfileDB(user_id="user-1", name="Test")
        test_session.add(profile)
        test_session.commit()

        sample = TrainingSampleDB(
            profile_id=profile.id,
            audio_path="/data/sample.wav",
            duration_seconds=3.0,
            sample_rate=24000,
            quality_score=0.85,
        )
        test_session.add(sample)
        test_session.commit()

        assert sample.quality_score == 0.85

    def test_sample_metadata(self, test_session):
        """Sample can store extra metadata."""
        profile = VoiceProfileDB(user_id="user-1", name="Test")
        test_session.add(profile)
        test_session.commit()

        sample = TrainingSampleDB(
            profile_id=profile.id,
            audio_path="/data/sample.wav",
            duration_seconds=3.0,
            sample_rate=24000,
            extra_metadata={
                "song_id": "song-456",
                "pitch_range": [200, 800],
                "snr_db": 25.5,
            },
        )
        test_session.add(sample)
        test_session.commit()

        retrieved = test_session.query(TrainingSampleDB).filter_by(id=sample.id).first()
        assert retrieved.extra_metadata["song_id"] == "song-456"
        assert retrieved.extra_metadata["pitch_range"] == [200, 800]

    def test_sample_to_dict(self, test_session):
        """Sample can be serialized to dictionary."""
        profile = VoiceProfileDB(user_id="user-1", name="Test")
        test_session.add(profile)
        test_session.commit()

        sample = TrainingSampleDB(
            profile_id=profile.id,
            audio_path="/data/sample.wav",
            duration_seconds=3.0,
            sample_rate=24000,
        )
        test_session.add(sample)
        test_session.commit()

        data = sample.to_dict()
        assert data["profile_id"] == profile.id
        assert data["audio_path"] == "/data/sample.wav"
        assert data["duration_seconds"] == 3.0
        assert "id" in data
        assert "created" in data

    def test_cascade_delete(self, test_session):
        """Samples are deleted when profile is deleted."""
        profile = VoiceProfileDB(user_id="user-1", name="Test")
        test_session.add(profile)
        test_session.commit()
        profile_id = profile.id

        # Add samples
        for i in range(3):
            sample = TrainingSampleDB(
                profile_id=profile_id,
                audio_path=f"/data/sample{i}.wav",
                duration_seconds=3.0,
                sample_rate=24000,
            )
            test_session.add(sample)
        test_session.commit()

        # Verify samples exist
        count = test_session.query(TrainingSampleDB).filter_by(profile_id=profile_id).count()
        assert count == 3

        # Delete profile
        test_session.delete(profile)
        test_session.commit()

        # Samples should be gone
        count = test_session.query(TrainingSampleDB).filter_by(profile_id=profile_id).count()
        assert count == 0

    def test_processing_status(self, test_session):
        """Sample tracks processing status."""
        profile = VoiceProfileDB(user_id="user-1", name="Test")
        test_session.add(profile)
        test_session.commit()

        sample = TrainingSampleDB(
            profile_id=profile.id,
            audio_path="/data/sample.wav",
            duration_seconds=3.0,
            sample_rate=24000,
        )
        test_session.add(sample)
        test_session.commit()

        assert sample.processed == 0  # False
        assert sample.processed_at is None

        # Mark as processed
        sample.processed = 1
        sample.processed_at = datetime.now(timezone.utc)
        test_session.commit()

        retrieved = test_session.query(TrainingSampleDB).filter_by(id=sample.id).first()
        assert retrieved.processed == 1
        assert retrieved.processed_at is not None


class TestDatabaseIntegration:
    """Integration tests for database operations."""

    def test_profile_sample_relationship(self, test_session):
        """Profile has relationship to samples."""
        profile = VoiceProfileDB(user_id="user-1", name="Test")
        test_session.add(profile)
        test_session.commit()

        # Add samples
        for i in range(5):
            sample = TrainingSampleDB(
                profile_id=profile.id,
                audio_path=f"/data/sample{i}.wav",
                duration_seconds=3.0 + i,
                sample_rate=24000,
            )
            test_session.add(sample)
        test_session.commit()

        # Access via relationship
        test_session.refresh(profile)
        assert profile.samples.count() == 5

    def test_multiple_profiles_per_user(self, test_session):
        """User can have multiple profiles."""
        for i in range(3):
            profile = VoiceProfileDB(
                user_id="user-1",
                name=f"Profile {i}",
            )
            test_session.add(profile)
        test_session.commit()

        profiles = test_session.query(VoiceProfileDB).filter_by(user_id="user-1").all()
        assert len(profiles) == 3
