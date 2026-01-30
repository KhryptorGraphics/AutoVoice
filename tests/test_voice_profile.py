"""Tests for VoiceProfile model and storage.

Task 1.1: Test VoiceProfile model with fields:
- user_id: Unique identifier for the user
- name: Display name for the profile
- created: Timestamp when profile was created
- samples_count: Number of training samples accumulated
- model_version: Current model version for this profile
"""
import pytest
from datetime import datetime, timezone
from uuid import UUID

# Import will fail until model is implemented (TDD Red Phase)
from auto_voice.profiles.models import VoiceProfile, TrainingSample


class TestVoiceProfileModel:
    """Test VoiceProfile data model."""

    def test_create_voice_profile_with_required_fields(self):
        """Profile can be created with required fields."""
        profile = VoiceProfile(
            user_id="user-123",
            name="My Singing Voice"
        )

        assert profile.user_id == "user-123"
        assert profile.name == "My Singing Voice"
        assert isinstance(profile.id, (str, UUID))
        assert isinstance(profile.created, datetime)
        assert profile.samples_count == 0
        assert profile.model_version is None

    def test_voice_profile_id_is_unique(self):
        """Each profile gets a unique ID."""
        profile1 = VoiceProfile(user_id="user-1", name="Profile 1")
        profile2 = VoiceProfile(user_id="user-1", name="Profile 2")

        assert profile1.id != profile2.id

    def test_voice_profile_created_timestamp(self):
        """Profile created timestamp is set automatically."""
        before = datetime.now(timezone.utc)
        profile = VoiceProfile(user_id="user-1", name="Test")
        after = datetime.now(timezone.utc)

        assert before <= profile.created <= after

    def test_voice_profile_samples_count_default(self):
        """Profile starts with zero samples."""
        profile = VoiceProfile(user_id="user-1", name="Test")
        assert profile.samples_count == 0

    def test_voice_profile_model_version_default(self):
        """Profile starts with no model version."""
        profile = VoiceProfile(user_id="user-1", name="Test")
        assert profile.model_version is None

    def test_voice_profile_increment_samples(self):
        """Profile sample count can be incremented."""
        profile = VoiceProfile(user_id="user-1", name="Test")
        profile.increment_samples(5)

        assert profile.samples_count == 5

        profile.increment_samples(3)
        assert profile.samples_count == 8

    def test_voice_profile_set_model_version(self):
        """Profile model version can be set."""
        profile = VoiceProfile(user_id="user-1", name="Test")
        profile.set_model_version("v1.0.0")

        assert profile.model_version == "v1.0.0"

    def test_voice_profile_to_dict(self):
        """Profile can be serialized to dictionary."""
        profile = VoiceProfile(user_id="user-123", name="Test Profile")
        profile.set_model_version("v1.0.0")

        data = profile.to_dict()

        assert data["user_id"] == "user-123"
        assert data["name"] == "Test Profile"
        assert data["samples_count"] == 0
        assert data["model_version"] == "v1.0.0"
        assert "id" in data
        assert "created" in data

    def test_voice_profile_from_dict(self):
        """Profile can be deserialized from dictionary."""
        data = {
            "id": "profile-123",
            "user_id": "user-456",
            "name": "Restored Profile",
            "created": "2026-01-24T12:00:00Z",
            "samples_count": 10,
            "model_version": "v2.0.0"
        }

        profile = VoiceProfile.from_dict(data)

        assert profile.id == "profile-123"
        assert profile.user_id == "user-456"
        assert profile.name == "Restored Profile"
        assert profile.samples_count == 10
        assert profile.model_version == "v2.0.0"


class TestTrainingSampleModel:
    """Test TrainingSample data model."""

    def test_create_training_sample(self):
        """Training sample can be created with required fields."""
        sample = TrainingSample(
            profile_id="profile-123",
            audio_path="/data/samples/sample1.wav",
            duration_seconds=5.5,
            sample_rate=24000
        )

        assert sample.profile_id == "profile-123"
        assert sample.audio_path == "/data/samples/sample1.wav"
        assert sample.duration_seconds == 5.5
        assert sample.sample_rate == 24000
        assert isinstance(sample.id, (str, UUID))
        assert isinstance(sample.created, datetime)

    def test_training_sample_quality_score(self):
        """Training sample can have quality score."""
        sample = TrainingSample(
            profile_id="profile-123",
            audio_path="/data/sample.wav",
            duration_seconds=3.0,
            sample_rate=24000,
            quality_score=0.85
        )

        assert sample.quality_score == 0.85

    def test_training_sample_metadata(self):
        """Training sample can store additional metadata."""
        sample = TrainingSample(
            profile_id="profile-123",
            audio_path="/data/sample.wav",
            duration_seconds=3.0,
            sample_rate=24000,
            metadata={
                "song_id": "song-456",
                "pitch_range": [200, 800],
                "snr_db": 25.5
            }
        )

        assert sample.metadata["song_id"] == "song-456"
        assert sample.metadata["pitch_range"] == [200, 800]
        assert sample.metadata["snr_db"] == 25.5

    def test_training_sample_to_dict(self):
        """Training sample can be serialized to dictionary."""
        sample = TrainingSample(
            profile_id="profile-123",
            audio_path="/data/sample.wav",
            duration_seconds=3.0,
            sample_rate=24000
        )

        data = sample.to_dict()

        assert data["profile_id"] == "profile-123"
        assert data["audio_path"] == "/data/sample.wav"
        assert data["duration_seconds"] == 3.0
        assert data["sample_rate"] == 24000
        assert "id" in data
        assert "created" in data

    def test_training_sample_from_dict(self):
        """Training sample can be deserialized from dictionary."""
        data = {
            "id": "sample-789",
            "profile_id": "profile-123",
            "audio_path": "/data/restored.wav",
            "duration_seconds": 4.2,
            "sample_rate": 44100,
            "created": "2026-01-24T12:30:00Z",
            "quality_score": 0.92,
            "metadata": {"source": "karaoke"}
        }

        sample = TrainingSample.from_dict(data)

        assert sample.id == "sample-789"
        assert sample.profile_id == "profile-123"
        assert sample.duration_seconds == 4.2
        assert sample.quality_score == 0.92
        assert sample.metadata["source"] == "karaoke"
