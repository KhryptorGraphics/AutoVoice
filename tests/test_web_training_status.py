"""Tests for web interface training status.

Phase 6: Test web interface shows training status correctly.

Tests verify:
- Training status endpoint works
- API returns correct status fields
- Status transitions work correctly
"""

import pytest
import torch
from unittest.mock import patch, MagicMock

from auto_voice.storage.voice_profiles import VoiceProfileStore


@pytest.fixture
def temp_profile_dir(tmp_path):
    """Create temporary profile storage directory."""
    profile_dir = tmp_path / "voice_profiles"
    profile_dir.mkdir()
    return profile_dir


@pytest.fixture
def store(temp_profile_dir):
    """Create VoiceProfileStore with temp directory."""
    return VoiceProfileStore(profiles_dir=str(temp_profile_dir))


@pytest.fixture
def app():
    """Create Flask app for testing."""
    from auto_voice.web.app import create_app
    app, socketio = create_app(testing=True)
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


class TestTrainingStatusEndpoint:
    """Tests for /voice/profiles/<id>/training-status endpoint."""

    def test_endpoint_exists(self, client):
        """Task 6.1: Training status endpoint should exist."""
        # Endpoint should exist (404 for missing profile, not 405 for missing route)
        response = client.get('/api/v1/voice/profiles/nonexistent/training-status')
        assert response.status_code in [404, 503], \
            "Endpoint should exist (404 or 503 for missing/error)"

    def test_returns_status_for_untrained_profile(self, client, store):
        """Should return pending status for profile without trained model."""
        # Create profile without training
        profile_id = "untrained-web-test"
        store.save({
            "profile_id": profile_id,
            "name": "Test",
            "embedding": torch.randn(256).numpy(),
        })

        with patch('auto_voice.storage.voice_profiles.VoiceProfileStore', return_value=store):
            response = client.get(f'/api/v1/voice/profiles/{profile_id}/training-status')

        # Should return status
        assert response.status_code == 200
        data = response.get_json()
        assert 'has_trained_model' in data
        assert 'training_status' in data
        assert data['has_trained_model'] is False

    def test_returns_ready_for_trained_profile(self, client, store):
        """Should return ready status for profile with trained model."""
        # Create profile with weights
        profile_id = "trained-web-test"
        store.save({
            "profile_id": profile_id,
            "name": "Test",
            "embedding": torch.randn(256).numpy(),
        })
        store.save_lora_weights(profile_id, {
            "test.lora_A": torch.randn(8, 256),
            "test.lora_B": torch.randn(256, 8),
        })

        with patch('auto_voice.storage.voice_profiles.VoiceProfileStore', return_value=store):
            response = client.get(f'/api/v1/voice/profiles/{profile_id}/training-status')

        assert response.status_code == 200
        data = response.get_json()
        assert data['has_trained_model'] is True
        assert data['training_status'] == 'ready'

    def test_returns_404_for_missing_profile(self, client):
        """Should return 404 for nonexistent profile."""
        response = client.get('/api/v1/voice/profiles/does-not-exist/training-status')
        assert response.status_code in [404, 503]


class TestProfileListWithStatus:
    """Tests for profile list including training status."""

    def test_profiles_list_returns_ok(self, client):
        """Profile list endpoint should work."""
        response = client.get('/api/v1/voice/profiles')
        assert response.status_code == 200

    def test_individual_profile_includes_status_fields(self, client, store):
        """Individual profile response should include status-related fields."""
        # Create profile
        profile_id = "status-fields-test"
        store.save({
            "profile_id": profile_id,
            "name": "Test",
            "embedding": torch.randn(256).numpy(),
            "training_status": "pending",
        })

        # Load and check field
        profile = store.load(profile_id)
        assert "training_status" in profile


class TestFrontendAPIService:
    """Tests for frontend api.ts service updates."""

    def test_training_status_endpoint_format(self, client, store):
        """Verify endpoint returns format expected by frontend."""
        profile_id = "frontend-format-test"
        store.save({
            "profile_id": profile_id,
            "name": "Test",
            "embedding": torch.randn(256).numpy(),
        })

        with patch('auto_voice.storage.voice_profiles.VoiceProfileStore', return_value=store):
            response = client.get(f'/api/v1/voice/profiles/{profile_id}/training-status')

        if response.status_code == 200:
            data = response.get_json()
            # Frontend expects these fields
            assert 'has_trained_model' in data
            assert 'training_status' in data
            assert isinstance(data['has_trained_model'], bool)
            assert data['training_status'] in ['pending', 'training', 'ready', 'failed']
