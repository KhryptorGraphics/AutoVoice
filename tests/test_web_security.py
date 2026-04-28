from __future__ import annotations

import os
from pathlib import Path

import pytest
from flask import Flask

from auto_voice.web.app import create_app
from auto_voice.web.security import (
    canonical_youtube_url,
    redact_public_paths,
    resolve_server_audio_path,
    socketio_cors_allowed_origins,
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
