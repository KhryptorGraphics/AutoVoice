"""Tests for VoiceProfile REST API endpoints.

Task 1.3: Test profile CRUD API endpoints (TDD Red Phase).
"""

import pytest
import json
import os
from unittest.mock import patch, MagicMock

from flask import Flask
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from auto_voice.profiles.db.models import Base
from auto_voice.profiles.db import session as db_session_module


@pytest.fixture
def test_db():
    """Create SQLite in-memory database for testing."""
    # Create in-memory SQLite engine
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)

    # Create session factory
    SessionLocal = sessionmaker(bind=engine)

    # Patch the session module to use test database
    original_engine = db_session_module._engine
    original_session = db_session_module._SessionLocal

    db_session_module._engine = engine
    db_session_module._SessionLocal = SessionLocal

    yield engine

    # Restore original
    db_session_module._engine = original_engine
    db_session_module._SessionLocal = original_session


@pytest.fixture
def app(test_db):
    """Create test Flask app with profile API."""
    from auto_voice.web.app import create_app

    app, socketio = create_app(testing=True)
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


class TestProfileAPICreate:
    """Test POST /api/v1/profiles endpoint."""

    def test_create_profile_success(self, client):
        """Create profile with valid data returns 201."""
        response = client.post(
            "/api/v1/profiles",
            json={"user_id": "user-123", "name": "My Singing Voice"},
            content_type="application/json",
        )

        assert response.status_code == 201
        data = response.get_json()
        assert data["user_id"] == "user-123"
        assert data["name"] == "My Singing Voice"
        assert "id" in data
        assert "created" in data
        assert data["samples_count"] == 0

    def test_create_profile_missing_user_id(self, client):
        """Create profile without user_id returns 400."""
        response = client.post(
            "/api/v1/profiles",
            json={"name": "My Voice"},
            content_type="application/json",
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "user_id" in data["error"].lower()

    def test_create_profile_missing_name(self, client):
        """Create profile without name returns 400."""
        response = client.post(
            "/api/v1/profiles",
            json={"user_id": "user-123"},
            content_type="application/json",
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "name" in data["error"].lower()

    def test_create_profile_empty_name(self, client):
        """Create profile with empty name returns 400."""
        response = client.post(
            "/api/v1/profiles",
            json={"user_id": "user-123", "name": ""},
            content_type="application/json",
        )

        assert response.status_code == 400

    def test_create_profile_with_settings(self, client):
        """Create profile with optional settings."""
        response = client.post(
            "/api/v1/profiles",
            json={
                "user_id": "user-123",
                "name": "Custom Voice",
                "settings": {"pitch_shift": 2, "formant_shift": 0.5},
            },
            content_type="application/json",
        )

        assert response.status_code == 201
        data = response.get_json()
        assert data["settings"]["pitch_shift"] == 2


class TestProfileAPIRead:
    """Test GET /api/v1/profiles endpoints."""

    def test_get_profile_by_id(self, client):
        """Get profile by ID returns profile data."""
        # First create a profile
        create_response = client.post(
            "/api/v1/profiles",
            json={"user_id": "user-123", "name": "Test Profile"},
            content_type="application/json",
        )
        profile_id = create_response.get_json()["id"]

        # Then retrieve it
        response = client.get(f"/api/v1/profiles/{profile_id}")

        assert response.status_code == 200
        data = response.get_json()
        assert data["id"] == profile_id
        assert data["name"] == "Test Profile"

    def test_get_profile_not_found(self, client):
        """Get non-existent profile returns 404."""
        response = client.get("/api/v1/profiles/nonexistent-id")

        assert response.status_code == 404
        data = response.get_json()
        assert "error" in data

    def test_list_profiles_by_user(self, client):
        """List profiles for a user returns array."""
        # Create multiple profiles
        for i in range(3):
            client.post(
                "/api/v1/profiles",
                json={"user_id": "user-123", "name": f"Profile {i}"},
                content_type="application/json",
            )

        response = client.get("/api/v1/profiles?user_id=user-123")

        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)
        assert len(data) >= 3
        for profile in data:
            assert profile["user_id"] == "user-123"

    def test_list_profiles_empty(self, client):
        """List profiles for user with none returns empty array."""
        response = client.get("/api/v1/profiles?user_id=nonexistent-user")

        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_list_profiles_requires_user_id(self, client):
        """List profiles without user_id returns 400."""
        response = client.get("/api/v1/profiles")

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data


class TestProfileAPIUpdate:
    """Test PUT /api/v1/profiles/{id} endpoint."""

    def test_update_profile_name(self, client):
        """Update profile name succeeds."""
        # Create profile
        create_response = client.post(
            "/api/v1/profiles",
            json={"user_id": "user-123", "name": "Original Name"},
            content_type="application/json",
        )
        profile_id = create_response.get_json()["id"]

        # Update it
        response = client.put(
            f"/api/v1/profiles/{profile_id}",
            json={"name": "Updated Name"},
            content_type="application/json",
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["name"] == "Updated Name"

    def test_update_profile_settings(self, client):
        """Update profile settings succeeds."""
        # Create profile
        create_response = client.post(
            "/api/v1/profiles",
            json={"user_id": "user-123", "name": "Test"},
            content_type="application/json",
        )
        profile_id = create_response.get_json()["id"]

        # Update settings
        response = client.put(
            f"/api/v1/profiles/{profile_id}",
            json={"settings": {"pitch_shift": 5}},
            content_type="application/json",
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["settings"]["pitch_shift"] == 5

    def test_update_profile_not_found(self, client):
        """Update non-existent profile returns 404."""
        response = client.put(
            "/api/v1/profiles/nonexistent-id",
            json={"name": "New Name"},
            content_type="application/json",
        )

        assert response.status_code == 404

    def test_update_profile_empty_name(self, client):
        """Update profile with empty name returns 400."""
        # Create profile
        create_response = client.post(
            "/api/v1/profiles",
            json={"user_id": "user-123", "name": "Test"},
            content_type="application/json",
        )
        profile_id = create_response.get_json()["id"]

        # Try to update with empty name
        response = client.put(
            f"/api/v1/profiles/{profile_id}",
            json={"name": ""},
            content_type="application/json",
        )

        assert response.status_code == 400

    def test_update_cannot_change_user_id(self, client):
        """Update cannot change user_id."""
        # Create profile
        create_response = client.post(
            "/api/v1/profiles",
            json={"user_id": "user-123", "name": "Test"},
            content_type="application/json",
        )
        profile_id = create_response.get_json()["id"]

        # Try to change user_id
        response = client.put(
            f"/api/v1/profiles/{profile_id}",
            json={"user_id": "different-user"},
            content_type="application/json",
        )

        # Should either ignore or return 400
        if response.status_code == 200:
            data = response.get_json()
            assert data["user_id"] == "user-123"  # Should not change


class TestProfileAPIDelete:
    """Test DELETE /api/v1/profiles/{id} endpoint."""

    def test_delete_profile_success(self, client):
        """Delete profile returns 204."""
        # Create profile
        create_response = client.post(
            "/api/v1/profiles",
            json={"user_id": "user-123", "name": "To Delete"},
            content_type="application/json",
        )
        profile_id = create_response.get_json()["id"]

        # Delete it
        response = client.delete(f"/api/v1/profiles/{profile_id}")

        assert response.status_code == 204

        # Verify it's gone
        get_response = client.get(f"/api/v1/profiles/{profile_id}")
        assert get_response.status_code == 404

    def test_delete_profile_not_found(self, client):
        """Delete non-existent profile returns 404."""
        response = client.delete("/api/v1/profiles/nonexistent-id")

        assert response.status_code == 404

    def test_delete_profile_cascades_samples(self, client):
        """Delete profile also deletes associated samples."""
        # Create profile
        create_response = client.post(
            "/api/v1/profiles",
            json={"user_id": "user-123", "name": "With Samples"},
            content_type="application/json",
        )
        profile_id = create_response.get_json()["id"]

        # Note: Sample creation will be tested in Task 1.5/1.6
        # For now, just verify delete works
        response = client.delete(f"/api/v1/profiles/{profile_id}")
        assert response.status_code == 204


class TestProfileAPIModelVersion:
    """Test model version management endpoints."""

    def test_set_model_version(self, client):
        """Set model version for profile."""
        # Create profile
        create_response = client.post(
            "/api/v1/profiles",
            json={"user_id": "user-123", "name": "Test"},
            content_type="application/json",
        )
        profile_id = create_response.get_json()["id"]

        # Set model version
        response = client.put(
            f"/api/v1/profiles/{profile_id}",
            json={"model_version": "v1.0.0"},
            content_type="application/json",
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["model_version"] == "v1.0.0"

    def test_get_profile_includes_model_info(self, client):
        """Get profile includes model version and path."""
        # Create and update profile
        create_response = client.post(
            "/api/v1/profiles",
            json={"user_id": "user-123", "name": "Test"},
            content_type="application/json",
        )
        profile_id = create_response.get_json()["id"]

        client.put(
            f"/api/v1/profiles/{profile_id}",
            json={"model_version": "v2.0.0", "model_path": "/models/user-123/v2.0.0"},
            content_type="application/json",
        )

        # Retrieve profile
        response = client.get(f"/api/v1/profiles/{profile_id}")

        assert response.status_code == 200
        data = response.get_json()
        assert data["model_version"] == "v2.0.0"
        assert "model_path" in data
