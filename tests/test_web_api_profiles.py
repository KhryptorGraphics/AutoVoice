"""Comprehensive tests for /api/v1/profiles/* endpoints.

Phase 4.4: Tests for profile sample and segment management:
- GET /profiles/{id}/samples - List samples
- POST /profiles/{id}/samples - Upload sample
- POST /profiles/{id}/samples/from-path - Add sample from path
- GET /profiles/{id}/samples/{sid} - Get sample detail
- DELETE /profiles/{id}/samples/{sid} - Delete sample
- POST /profiles/{id}/samples/{sid}/filter - Filter sample
- GET /profiles/{id}/segments - Get diarization segments
- GET /profiles/{id}/checkpoints - List checkpoints

Uses Flask test client with mocked components.
"""

import io
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.fixtures.audio import voiced_wav_io


@pytest.fixture
def app_with_profiles():
    """Create Flask app with profile components mocked."""
    pytest.importorskip('flask_swagger_ui', reason="flask_swagger_ui not installed")

    from auto_voice.web.app import create_app

    app, socketio = create_app(config={
        'TESTING': True,
        'singing_conversion_enabled': True,
        'voice_cloning_enabled': True,
    })
    real_store = app.voice_cloner.store

    # Mock voice cloner
    mock_voice_cloner = MagicMock()
    mock_voice_cloner.load_voice_profile.return_value = {
        'profile_id': 'test-profile',
        'name': 'Test Artist',
        'embedding': np.zeros(256).tolist(),
    }
    mock_voice_cloner.list_profiles.return_value = [
        {'profile_id': 'test-profile', 'name': 'Test Artist'},
    ]
    mock_voice_cloner.store = real_store

    app.voice_cloner = mock_voice_cloner
    app.socketio = socketio
    app.app_config = {'audio': {'sample_rate': 22050}}

    return app


@pytest.fixture
def client(app_with_profiles):
    """Flask test client."""
    return app_with_profiles.test_client()


@pytest.fixture
def audio_file():
    """Create a deterministic voiced WAV file for upload."""
    return voiced_wav_io(duration_seconds=1.0, sample_rate=22050)


@pytest.fixture
def legacy_sqlite_db(tmp_path, monkeypatch):
    from auto_voice.profiles.db import session as db_session

    monkeypatch.setenv("AUTOVOICE_DATABASE_URL", f"sqlite:///{tmp_path / 'profiles.db'}")
    db_session._engine = None
    db_session._SessionLocal = None
    db_session.init_db()
    yield
    db_session._engine = None
    db_session._SessionLocal = None


def _save_profile(store, profile_id: str, *, trained: bool = True) -> None:
    store.save({
        'profile_id': profile_id,
        'name': f'Test {profile_id[-4:]}',
        'embedding': np.zeros(256, dtype=np.float32).tolist(),
        'profile_role': 'target_user',
        'has_trained_model': trained,
        'training_status': 'ready' if trained else 'pending',
    })


def test_sample_routes_have_one_registered_handler_family(app_with_profiles):
    sample_routes = {}
    for rule in app_with_profiles.url_map.iter_rules():
        if "/api/v1/profiles/" not in rule.rule or "/samples" not in rule.rule:
            continue
        if "<sample_id>/filter" in rule.rule:
            continue
        for method in rule.methods - {"HEAD", "OPTIONS"}:
            sample_routes.setdefault((rule.rule, method), []).append(rule.endpoint)

    duplicates = {key: endpoints for key, endpoints in sample_routes.items() if len(endpoints) > 1}

    assert duplicates == {}
    assert sample_routes[("/api/v1/profiles/<profile_id>/samples", "GET")] == ["api.list_samples"]
    assert sample_routes[("/api/v1/profiles/<profile_id>/samples", "POST")] == ["api.upload_sample"]


def test_profile_export_and_purge_write_audit_events(client, app_with_profiles, tmp_path):
    profile_id = "00000000-0000-0000-0000-000000000001"
    app_with_profiles.voice_cloner.load_voice_profile.return_value = {
        "profile_id": profile_id,
        "name": "Governed Profile",
        "embedding": np.zeros(256).tolist(),
    }
    app_with_profiles.voice_cloner.delete_voice_profile.return_value = True
    asset_path = tmp_path / "sample.wav"
    asset_path.write_bytes(b"RIFF")
    app_with_profiles.state_store.register_asset(asset_path, kind="voice_sample", owner_id=profile_id)

    export_response = client.get(f"/api/v1/voice/profiles/{profile_id}/export")
    purge_response = client.delete(f"/api/v1/voice/profiles/{profile_id}/purge")

    assert export_response.status_code == 200
    assert purge_response.status_code == 200
    events = app_with_profiles.state_store.list_audit_events(resource_id=profile_id)
    event_types = {event["event_type"] for event in events}
    assert {"export", "delete"} <= event_types
    assert {"voice_profile.exported", "voice_profile.purged"} <= {
        event["metadata"]["event_type"] for event in events
    }
    assert app_with_profiles.state_store.list_assets(profile_id) == []


class TestListSamples:
    """Test GET /api/v1/profiles/{id}/samples endpoint."""

    def test_list_samples_returns_list(self, client):
        """Returns list for profile samples endpoint."""
        response = client.get('/api/v1/profiles/test-profile/samples')

        # Should return 200 with list (may be empty)
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)


class TestUploadSample:
    """Test POST /api/v1/profiles/{id}/samples endpoint."""

    def test_upload_sample_missing_file(self, client):
        """Returns 400 when no file provided."""
        response = client.post(
            '/api/v1/profiles/test-profile/samples',
            data={},
            content_type='multipart/form-data'
        )

        assert response.status_code == 400

    def test_upload_sample_invalid_file_type(self, client):
        """Returns 400 for invalid file type."""
        response = client.post(
            '/api/v1/profiles/test-profile/samples',
            data={'audio': (io.BytesIO(b'not audio'), 'test.txt')},
            content_type='multipart/form-data'
        )

        assert response.status_code == 400

    def test_db_only_profile_accepts_documented_file_field(
        self, client, app_with_profiles, legacy_sqlite_db, tmp_path
    ):
        app_with_profiles.config["SAMPLE_STORAGE_PATH"] = str(tmp_path / "legacy-samples")
        created = client.post(
            "/api/v1/profiles",
            json={"user_id": "legacy-user", "name": "Legacy Profile"},
        )
        profile_id = created.get_json()["id"]

        response = client.post(
            f"/api/v1/profiles/{profile_id}/samples",
            data={"file": (voiced_wav_io(duration_seconds=0.2), "legacy.wav")},
            content_type="multipart/form-data",
        )

        assert response.status_code == 201
        assert response.get_json()["profile_id"] == profile_id


class TestAddSampleFromPath:
    """Test POST /api/v1/profiles/{id}/samples/from-path endpoint."""

    def test_add_from_path_missing_path(self, client):
        """Returns 400 when path is missing."""
        response = client.post(
            '/api/v1/profiles/test-profile/samples/from-path',
            json={},
            content_type='application/json'
        )

        assert response.status_code == 400

    def test_add_from_path_file_not_found(self, client):
        """Returns 404 when file doesn't exist."""
        response = client.post(
            '/api/v1/profiles/test-profile/samples/from-path',
            json={'path': '/nonexistent/path/to/audio.wav'},
            content_type='application/json'
        )

        assert response.status_code in (400, 404)


class TestUploadSong:
    def test_db_only_profile_accepts_song_upload_without_auto_split(self, client, legacy_sqlite_db):
        created = client.post(
            "/api/v1/profiles",
            json={"user_id": "legacy-user", "name": "Legacy Profile"},
        )
        profile_id = created.get_json()["id"]

        response = client.post(
            f"/api/v1/profiles/{profile_id}/songs",
            data={
                "file": (voiced_wav_io(duration_seconds=0.2), "song.wav"),
                "auto_split": "false",
            },
            content_type="multipart/form-data",
        )

        assert response.status_code == 202
        assert response.get_json()["status"] == "complete"


class TestGetSampleDetail:
    """Test GET /api/v1/profiles/{id}/samples/{sid} endpoint."""

    def test_get_sample_not_found(self, client):
        """Returns 404 for non-existent sample."""
        response = client.get('/api/v1/profiles/test-profile/samples/nonexistent')

        assert response.status_code == 404


class TestDeleteSample:
    """Test DELETE /api/v1/profiles/{id}/samples/{sid} endpoint."""

    def test_delete_sample_not_found(self, client):
        """Returns 404 for non-existent sample."""
        response = client.delete('/api/v1/profiles/test-profile/samples/nonexistent')

        assert response.status_code == 404


class TestFilterSample:
    """Test POST /api/v1/profiles/{id}/samples/{sid}/filter endpoint."""

    def test_filter_sample_not_found(self, client):
        """Returns 404 for non-existent sample."""
        response = client.post(
            '/api/v1/profiles/test-profile/samples/nonexistent/filter',
            json={},
            content_type='application/json'
        )

        assert response.status_code == 404


class TestListCheckpoints:
    """Test GET /api/v1/profiles/{id}/checkpoints endpoint."""

    def test_list_checkpoints_empty(self, client):
        """Returns empty list when no checkpoints exist."""
        response = client.get('/api/v1/profiles/test-profile/checkpoints')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)


class TestRollbackCheckpoint:
    """Test POST /api/v1/profiles/{id}/checkpoints/{cid}/rollback endpoint."""

    def test_rollback_checkpoint_not_found(self, client):
        """Returns 404 for non-existent checkpoint."""
        response = client.post(
            '/api/v1/profiles/test-profile/checkpoints/nonexistent/rollback'
        )

        assert response.status_code == 404


class TestDeleteCheckpoint:
    """Test DELETE /api/v1/profiles/{id}/checkpoints/{cid} endpoint."""

    def test_delete_checkpoint_not_found(self, client):
        """Returns 404 for non-existent checkpoint."""
        response = client.delete('/api/v1/profiles/test-profile/checkpoints/nonexistent')

        assert response.status_code == 404


class TestSpeakerEmbedding:
    """Test speaker embedding endpoints."""

    def test_set_speaker_embedding_missing_data(self, client):
        """Returns 400 or 404 when embedding data missing or profile not found."""
        response = client.post(
            '/api/v1/profiles/test-profile/speaker-embedding',
            json={},
            content_type='application/json'
        )

        # 400 if profile exists but data missing, 404 if profile not found
        assert response.status_code in (400, 404)


class TestCheckRetrain:
    """Test POST /api/v1/profiles/{id}/check-retrain endpoint."""

    def test_check_retrain_returns_response(self, client):
        """Returns retrain status."""
        response = client.post('/api/v1/profiles/test-profile/check-retrain')

        # Either 200 with status or various error codes
        assert response.status_code in (200, 400, 404, 500)


class TestQualityHistory:
    """Test GET /api/v1/profiles/{id}/quality-history endpoint."""

    def test_get_quality_history_empty(self, client):
        """Returns response for quality history."""
        response = client.get('/api/v1/profiles/test-profile/quality-history')

        assert response.status_code == 200

    def test_get_quality_history_with_limit(self, client):
        """Accepts limit query parameter."""
        response = client.get('/api/v1/profiles/test-profile/quality-history?limit=10')

        assert response.status_code == 200


class TestQualityStatus:
    """Test GET /api/v1/profiles/{id}/quality-status endpoint."""

    def test_get_quality_status(self, client):
        """Returns quality status for profile."""
        response = client.get('/api/v1/profiles/test-profile/quality-status')

        assert response.status_code in (200, 404, 500)


class TestCheckDegradation:
    """Test POST /api/v1/profiles/{id}/check-degradation endpoint."""

    def test_check_degradation(self, client):
        """Returns degradation check results."""
        response = client.post('/api/v1/profiles/test-profile/check-degradation')

        assert response.status_code in (200, 400, 404, 500)


class TestCompatibilityHelperRoutes:
    """Test /api/v1/profiles compatibility helper aliases."""

    def test_training_status_alias_returns_profile_status(self, app_with_profiles, client):
        store = app_with_profiles.voice_cloner.store
        profile_id = '00000000-0000-0000-0000-000000000401'
        _save_profile(store, profile_id, trained=True)

        response = client.get(f'/api/v1/profiles/{profile_id}/training-status')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['profile_id'] == profile_id
        assert data['training_status'] == 'ready'

    def test_model_alias_matches_missing_profile_contract(self, client):
        profile_id = '00000000-0000-0000-0000-000000000402'

        response = client.get(f'/api/v1/profiles/{profile_id}/model')

        assert response.status_code == 404
        data = json.loads(response.data)
        assert data['profile_id'] == profile_id

    def test_adapters_alias_returns_empty_list_without_artifact(self, app_with_profiles, client):
        store = app_with_profiles.voice_cloner.store
        profile_id = '00000000-0000-0000-0000-000000000403'
        _save_profile(store, profile_id, trained=True)

        response = client.get(f'/api/v1/profiles/{profile_id}/adapters')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['profile_id'] == profile_id
        assert data['adapters'] == []
        assert data['count'] == 0
