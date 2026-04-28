from __future__ import annotations

import os
from pathlib import Path

import pytest
from flask import Flask

from auto_voice.web.app import create_app
from auto_voice.web.security import (
    canonical_youtube_url,
    redact_public_paths,
    record_structured_audit_event,
    require_media_consent,
    resolve_server_audio_path,
    socketio_cors_allowed_origins,
    validate_api_token_configuration,
)


def test_api_token_auth_blocks_operator_endpoint(monkeypatch):
    monkeypatch.setenv("AUTOVOICE_REQUIRE_API_AUTH", "true")
    monkeypatch.setenv("AUTOVOICE_API_TOKEN", "unit-token")
    app, _ = create_app(config={"TESTING": True}, testing=True)
    client = app.test_client()

    unauthorized = client.get("/api/v1/system/info")
    authorized = client.get("/api/v1/system/info", headers={"Authorization": "Bearer unit-token"})
    health = client.get("/api/v1/health")

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    assert health.status_code == 200


def test_public_mode_rejects_wildcard_socketio_cors(monkeypatch):
    monkeypatch.setenv("AUTOVOICE_PUBLIC_DEPLOYMENT", "true")
    monkeypatch.setenv("CORS_ORIGINS", "*")
    monkeypatch.delenv("AUTOVOICE_ALLOW_INSECURE_CORS", raising=False)
    app = Flask(__name__)

    with pytest.raises(RuntimeError, match="CORS_ORIGINS='\\*'"):
        socketio_cors_allowed_origins(app)


def test_auth_required_mode_rejects_wildcard_socketio_cors(monkeypatch):
    monkeypatch.setenv("AUTOVOICE_REQUIRE_API_AUTH", "true")
    monkeypatch.setenv("AUTOVOICE_API_TOKEN", "a" * 32)
    monkeypatch.setenv("CORS_ORIGINS", "*")
    monkeypatch.delenv("AUTOVOICE_ALLOW_INSECURE_CORS", raising=False)
    app = Flask(__name__)

    with pytest.raises(RuntimeError, match="API authentication is required"):
        socketio_cors_allowed_origins(app)


def test_public_mode_rejects_placeholder_api_token(monkeypatch):
    monkeypatch.setenv("AUTOVOICE_PUBLIC_DEPLOYMENT", "true")
    monkeypatch.setenv("AUTOVOICE_API_TOKEN", "changeme")
    app = Flask(__name__)

    with pytest.raises(RuntimeError, match="non-placeholder AUTOVOICE_API_TOKEN"):
        validate_api_token_configuration(app)


def test_auth_required_mode_rejects_placeholder_api_token(monkeypatch):
    monkeypatch.setenv("AUTOVOICE_REQUIRE_API_AUTH", "true")
    monkeypatch.setenv("AUTOVOICE_API_TOKEN", "changeme")
    app = Flask(__name__)

    with pytest.raises(RuntimeError, match="Authenticated AutoVoice deployments"):
        validate_api_token_configuration(app)


def test_public_mode_ignores_query_token_and_sets_explicit_cors(monkeypatch):
    monkeypatch.setenv("AUTOVOICE_PUBLIC_DEPLOYMENT", "true")
    monkeypatch.setenv("AUTOVOICE_API_TOKEN", "unit-token")
    monkeypatch.setenv("CORS_ORIGINS", "https://ops.example.test")
    monkeypatch.delenv("AUTOVOICE_ALLOW_QUERY_API_TOKEN", raising=False)
    app, _ = create_app(config={"TESTING": True}, testing=True)
    client = app.test_client()

    query_token = client.get(
        "/api/v1/system/info?api_token=unit-token",
        headers={"Origin": "https://ops.example.test"},
    )
    header_token = client.get(
        "/api/v1/system/info",
        headers={
            "Authorization": "Bearer unit-token",
            "Origin": "https://ops.example.test",
        },
    )
    options = client.options(
        "/api/v1/system/info",
        headers={"Origin": "https://ops.example.test"},
    )

    assert query_token.status_code == 401
    assert header_token.status_code == 200
    assert header_token.headers["Access-Control-Allow-Origin"] == "https://ops.example.test"
    assert "Authorization" in options.headers["Access-Control-Allow-Headers"]


def test_auth_required_mode_rejects_query_token_by_default(monkeypatch):
    monkeypatch.setenv("AUTOVOICE_REQUIRE_API_AUTH", "true")
    monkeypatch.setenv("AUTOVOICE_API_TOKEN", "unit-token")
    monkeypatch.delenv("AUTOVOICE_ALLOW_QUERY_API_TOKEN", raising=False)
    app, _ = create_app(config={"TESTING": True}, testing=True)
    client = app.test_client()

    query_token = client.get("/api/v1/system/info?api_token=unit-token")
    header_token = client.get("/api/v1/system/info", headers={"X-AutoVoice-API-Key": "unit-token"})

    assert query_token.status_code == 401
    assert header_token.status_code == 200


def test_auth_required_mode_allows_query_token_only_when_explicit(monkeypatch):
    monkeypatch.setenv("AUTOVOICE_REQUIRE_API_AUTH", "true")
    monkeypatch.setenv("AUTOVOICE_API_TOKEN", "unit-token")
    monkeypatch.setenv("AUTOVOICE_ALLOW_QUERY_API_TOKEN", "true")
    app, _ = create_app(config={"TESTING": True}, testing=True)
    client = app.test_client()

    response = client.get("/api/v1/system/info?api_token=unit-token")

    assert response.status_code == 200


def test_api_rate_limit_uses_authenticated_bucket(monkeypatch):
    monkeypatch.setenv("AUTOVOICE_REQUIRE_API_AUTH", "true")
    monkeypatch.setenv("AUTOVOICE_API_TOKEN", "unit-token")
    monkeypatch.delenv("AUTOVOICE_ENABLE_RATE_LIMIT", raising=False)
    monkeypatch.setenv("RATE_LIMIT", "1")
    app, _ = create_app(config={"TESTING": True}, testing=True)
    client = app.test_client()
    headers = {"Authorization": "Bearer unit-token"}

    first = client.get("/api/v1/system/info", headers=headers)
    second = client.get("/api/v1/system/info", headers=headers)

    assert first.status_code == 200
    assert second.status_code == 429


def test_strict_audio_path_sandbox_rejects_unmanaged_files(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    allowed = data_dir / "sample.wav"
    outside = tmp_path / "outside.wav"
    allowed.write_bytes(b"RIFF")
    outside.write_bytes(b"RIFF")

    assert resolve_server_audio_path(allowed, data_dir=data_dir, strict=True) == allowed.resolve()
    with pytest.raises(PermissionError):
        resolve_server_audio_path(outside, data_dir=data_dir, strict=True)


def test_youtube_url_validation_canonicalizes_and_rejects_non_youtube():
    assert canonical_youtube_url("https://youtu.be/abc123") == "https://www.youtube.com/watch?v=abc123"
    assert (
        canonical_youtube_url("https://www.youtube.com/watch?v=abc123&feature=share")
        == "https://www.youtube.com/watch?v=abc123"
    )
    with pytest.raises(ValueError, match="Only YouTube URLs"):
        canonical_youtube_url("https://example.com/watch?v=abc123")
    with pytest.raises(ValueError, match="Only http"):
        canonical_youtube_url("file:///etc/passwd")


def test_public_mode_redacts_path_fields_to_asset_ids(monkeypatch, tmp_path):
    from auto_voice.web.persistence import AppStateStore

    monkeypatch.setenv("AUTOVOICE_PUBLIC_DEPLOYMENT", "true")
    app = Flask(__name__)
    store = AppStateStore(str(tmp_path))
    payload = {"audio_path": str(tmp_path / "audio.wav"), "nested": {"file_path": str(tmp_path / "file.wav")}}

    redacted = redact_public_paths(payload, app, store, owner_id="profile-a", kind="voice_sample")

    assert "audio_path" not in redacted
    assert "audio_path_asset_id" in redacted
    assert "file_path" not in redacted["nested"]
    assert len(store.list_assets("profile-a")) == 2


def test_auth_required_mode_redacts_path_fields_to_asset_ids(monkeypatch, tmp_path):
    from auto_voice.web.persistence import AppStateStore

    monkeypatch.setenv("AUTOVOICE_REQUIRE_API_AUTH", "true")
    monkeypatch.setenv("AUTOVOICE_API_TOKEN", "unit-token")
    app = Flask(__name__)
    store = AppStateStore(str(tmp_path))
    payload = {"audio_path": str(tmp_path / "audio.wav")}

    redacted = redact_public_paths(payload, app, store, owner_id="profile-a", kind="voice_sample")

    assert "audio_path" not in redacted
    assert "audio_path_asset_id" in redacted


def test_auth_required_audit_events_redact_asset_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOVOICE_REQUIRE_API_AUTH", "true")
    monkeypatch.setenv("AUTOVOICE_API_TOKEN", "unit-token")
    app, _ = create_app(config={"TESTING": True}, testing=True)
    asset_path = tmp_path / "secret.wav"
    asset_path.write_bytes(b"RIFF")
    with app.app_context():
        record_structured_audit_event(
            "download",
            "conversion",
            app=app,
            resource_id="job-1",
            asset_paths=[asset_path],
            asset_kind="conversion",
        )

    response = app.test_client().get(
        "/api/v1/audit/events",
        headers={"Authorization": "Bearer unit-token"},
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert str(asset_path) not in str(payload)
    asset = payload["events"][0]["metadata"]["assets"][0]
    assert asset["asset_id"]
    assert "path" not in asset


def test_media_consent_rejects_false_strings(monkeypatch):
    monkeypatch.setenv("AUTOVOICE_REQUIRE_MEDIA_CONSENT", "true")
    app = Flask(__name__)

    with pytest.raises(PermissionError):
        require_media_consent(
            {
                "consent_confirmed": "false",
                "source_media_policy_confirmed": "false",
            },
            app,
        )
