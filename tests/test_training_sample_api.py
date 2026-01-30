"""Tests for TrainingSample storage API endpoints.

Task 1.5: Test training sample storage (audio file + metadata) - TDD Red Phase.
"""

import pytest
import io
import os
import tempfile
from pathlib import Path

from flask import Flask
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from auto_voice.profiles.db.models import Base, VoiceProfileDB
from auto_voice.profiles.db import session as db_session_module


@pytest.fixture
def test_db():
    """Create SQLite in-memory database for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)

    original_engine = db_session_module._engine
    original_session = db_session_module._SessionLocal

    db_session_module._engine = engine
    db_session_module._SessionLocal = SessionLocal

    yield engine

    db_session_module._engine = original_engine
    db_session_module._SessionLocal = original_session


@pytest.fixture
def temp_storage_dir():
    """Create temporary directory for audio storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def app(test_db, temp_storage_dir):
    """Create test Flask app with sample storage configured."""
    from auto_voice.web.app import create_app

    config = {"SAMPLE_STORAGE_PATH": temp_storage_dir}
    app, socketio = create_app(config=config, testing=True)
    app.config["TESTING"] = True
    app.config["SAMPLE_STORAGE_PATH"] = temp_storage_dir
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def test_profile(test_db):
    """Create a test profile for sample tests."""
    with db_session_module.get_db_session() as session:
        profile = VoiceProfileDB(user_id="user-123", name="Test Profile")
        session.add(profile)
        session.flush()
        profile_id = profile.id
    return profile_id


def create_test_audio_file(duration_ms=1000, sample_rate=24000):
    """Create a minimal valid WAV file for testing."""
    import struct

    # Generate simple sine wave
    num_samples = int(duration_ms * sample_rate / 1000)
    audio_data = bytes([128 + int(127 * (i % 100) / 100) for i in range(num_samples)])

    # WAV header
    wav_header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + len(audio_data),
        b'WAVE',
        b'fmt ',
        16,  # Subchunk1Size
        1,   # AudioFormat (PCM)
        1,   # NumChannels
        sample_rate,
        sample_rate,  # ByteRate
        1,   # BlockAlign
        8,   # BitsPerSample
        b'data',
        len(audio_data)
    )

    return io.BytesIO(wav_header + audio_data)


class TestSampleAPIUpload:
    """Test POST /api/v1/profiles/{id}/samples endpoint."""

    def test_upload_sample_success(self, client, test_profile, temp_storage_dir):
        """Upload audio sample returns 201 with metadata."""
        audio_file = create_test_audio_file()

        response = client.post(
            f"/api/v1/profiles/{test_profile}/samples",
            data={
                "audio": (audio_file, "sample.wav", "audio/wav"),
            },
            content_type="multipart/form-data",
        )

        assert response.status_code == 201
        data = response.get_json()
        assert data["profile_id"] == test_profile
        assert "id" in data
        assert "audio_path" in data
        assert data["sample_rate"] == 24000
        assert data["duration_seconds"] > 0

    def test_upload_sample_with_metadata(self, client, test_profile):
        """Upload sample with additional metadata."""
        audio_file = create_test_audio_file()

        response = client.post(
            f"/api/v1/profiles/{test_profile}/samples",
            data={
                "audio": (audio_file, "sample.wav", "audio/wav"),
                "metadata": '{"song_id": "song-123", "pitch_range": [200, 800]}',
            },
            content_type="multipart/form-data",
        )

        assert response.status_code == 201
        data = response.get_json()
        assert data["metadata"]["song_id"] == "song-123"

    def test_upload_sample_profile_not_found(self, client):
        """Upload to non-existent profile returns 404."""
        audio_file = create_test_audio_file()

        response = client.post(
            "/api/v1/profiles/nonexistent-id/samples",
            data={
                "audio": (audio_file, "sample.wav", "audio/wav"),
            },
            content_type="multipart/form-data",
        )

        assert response.status_code == 404

    def test_upload_sample_no_file(self, client, test_profile):
        """Upload without audio file returns 400."""
        response = client.post(
            f"/api/v1/profiles/{test_profile}/samples",
            data={},
            content_type="multipart/form-data",
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_upload_sample_invalid_format(self, client, test_profile):
        """Upload non-audio file returns 400."""
        response = client.post(
            f"/api/v1/profiles/{test_profile}/samples",
            data={
                "audio": (io.BytesIO(b"not audio data"), "file.txt", "text/plain"),
            },
            content_type="multipart/form-data",
        )

        assert response.status_code == 400

    def test_upload_sample_stores_file(self, client, test_profile, temp_storage_dir):
        """Uploaded audio file is stored on disk."""
        audio_file = create_test_audio_file()

        response = client.post(
            f"/api/v1/profiles/{test_profile}/samples",
            data={
                "audio": (audio_file, "sample.wav", "audio/wav"),
            },
            content_type="multipart/form-data",
        )

        assert response.status_code == 201
        data = response.get_json()

        # File should exist at audio_path
        assert os.path.exists(data["audio_path"])

    def test_upload_sample_increments_profile_count(self, client, test_profile):
        """Upload increments profile's samples_count."""
        # Check initial count
        profile_response = client.get(f"/api/v1/profiles/{test_profile}")
        initial_count = profile_response.get_json()["samples_count"]

        # Upload sample
        audio_file = create_test_audio_file()
        client.post(
            f"/api/v1/profiles/{test_profile}/samples",
            data={"audio": (audio_file, "sample.wav", "audio/wav")},
            content_type="multipart/form-data",
        )

        # Check new count
        profile_response = client.get(f"/api/v1/profiles/{test_profile}")
        new_count = profile_response.get_json()["samples_count"]

        assert new_count == initial_count + 1


class TestSampleAPIList:
    """Test GET /api/v1/profiles/{id}/samples endpoint."""

    def test_list_samples_empty(self, client, test_profile):
        """List samples for profile with none returns empty array."""
        response = client.get(f"/api/v1/profiles/{test_profile}/samples")

        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_list_samples_with_data(self, client, test_profile):
        """List samples returns all profile samples."""
        # Upload multiple samples
        for i in range(3):
            audio_file = create_test_audio_file()
            client.post(
                f"/api/v1/profiles/{test_profile}/samples",
                data={"audio": (audio_file, f"sample{i}.wav", "audio/wav")},
                content_type="multipart/form-data",
            )

        response = client.get(f"/api/v1/profiles/{test_profile}/samples")

        assert response.status_code == 200
        data = response.get_json()
        assert len(data) == 3

    def test_list_samples_profile_not_found(self, client):
        """List samples for non-existent profile returns 404."""
        response = client.get("/api/v1/profiles/nonexistent-id/samples")

        assert response.status_code == 404


class TestSampleAPIGet:
    """Test GET /api/v1/profiles/{id}/samples/{sample_id} endpoint."""

    def test_get_sample_by_id(self, client, test_profile):
        """Get sample by ID returns sample data."""
        # Upload a sample
        audio_file = create_test_audio_file()
        upload_response = client.post(
            f"/api/v1/profiles/{test_profile}/samples",
            data={"audio": (audio_file, "sample.wav", "audio/wav")},
            content_type="multipart/form-data",
        )
        sample_id = upload_response.get_json()["id"]

        # Get it
        response = client.get(f"/api/v1/profiles/{test_profile}/samples/{sample_id}")

        assert response.status_code == 200
        data = response.get_json()
        assert data["id"] == sample_id

    def test_get_sample_not_found(self, client, test_profile):
        """Get non-existent sample returns 404."""
        response = client.get(f"/api/v1/profiles/{test_profile}/samples/nonexistent")

        assert response.status_code == 404


class TestSampleAPIDelete:
    """Test DELETE /api/v1/profiles/{id}/samples/{sample_id} endpoint."""

    def test_delete_sample_success(self, client, test_profile, temp_storage_dir):
        """Delete sample removes from DB and disk."""
        # Upload a sample
        audio_file = create_test_audio_file()
        upload_response = client.post(
            f"/api/v1/profiles/{test_profile}/samples",
            data={"audio": (audio_file, "sample.wav", "audio/wav")},
            content_type="multipart/form-data",
        )
        sample_id = upload_response.get_json()["id"]
        audio_path = upload_response.get_json()["audio_path"]

        # Delete it
        response = client.delete(f"/api/v1/profiles/{test_profile}/samples/{sample_id}")

        assert response.status_code == 204

        # Verify removed from DB
        get_response = client.get(f"/api/v1/profiles/{test_profile}/samples/{sample_id}")
        assert get_response.status_code == 404

        # Verify file removed from disk
        assert not os.path.exists(audio_path)

    def test_delete_sample_not_found(self, client, test_profile):
        """Delete non-existent sample returns 404."""
        response = client.delete(f"/api/v1/profiles/{test_profile}/samples/nonexistent")

        assert response.status_code == 404

    def test_delete_sample_decrements_profile_count(self, client, test_profile):
        """Delete decrements profile's samples_count."""
        # Upload sample
        audio_file = create_test_audio_file()
        upload_response = client.post(
            f"/api/v1/profiles/{test_profile}/samples",
            data={"audio": (audio_file, "sample.wav", "audio/wav")},
            content_type="multipart/form-data",
        )
        sample_id = upload_response.get_json()["id"]

        # Check count after upload
        profile_response = client.get(f"/api/v1/profiles/{test_profile}")
        count_after_upload = profile_response.get_json()["samples_count"]

        # Delete sample
        client.delete(f"/api/v1/profiles/{test_profile}/samples/{sample_id}")

        # Check count after delete
        profile_response = client.get(f"/api/v1/profiles/{test_profile}")
        count_after_delete = profile_response.get_json()["samples_count"]

        assert count_after_delete == count_after_upload - 1


class TestSampleStorage:
    """Test file storage organization."""

    def test_samples_organized_by_profile(self, client, test_profile, temp_storage_dir):
        """Samples are stored in profile-specific directories."""
        audio_file = create_test_audio_file()
        response = client.post(
            f"/api/v1/profiles/{test_profile}/samples",
            data={"audio": (audio_file, "sample.wav", "audio/wav")},
            content_type="multipart/form-data",
        )

        assert response.status_code == 201
        audio_path = response.get_json()["audio_path"]

        # Path should contain profile_id
        assert test_profile in audio_path

    def test_sample_filenames_unique(self, client, test_profile):
        """Multiple uploads create unique filenames."""
        paths = []
        for _ in range(3):
            audio_file = create_test_audio_file()
            response = client.post(
                f"/api/v1/profiles/{test_profile}/samples",
                data={"audio": (audio_file, "sample.wav", "audio/wav")},
                content_type="multipart/form-data",
            )
            paths.append(response.get_json()["audio_path"])

        # All paths should be unique
        assert len(set(paths)) == 3
