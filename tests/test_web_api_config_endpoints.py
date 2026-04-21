"""Targeted coverage for configuration API endpoints."""

from __future__ import annotations

import pytest


@pytest.fixture
def app_config_api(tmp_path):
    pytest.importorskip("flask_swagger_ui", reason="flask_swagger_ui not installed")

    from auto_voice.web.app import create_app

    app, socketio = create_app(
        config={
            "TESTING": True,
            "DATA_DIR": str(tmp_path),
        },
        testing=True,
    )
    app.socketio = socketio
    return app


@pytest.fixture
def client_config_api(app_config_api):
    return app_config_api.test_client()


def test_separation_config_get_and_update(client_config_api):
    original = client_config_api.get("/api/v1/config/separation")
    assert original.status_code == 200
    assert "model" in original.get_json()

    updated = client_config_api.patch(
        "/api/v1/config/separation",
        json={"model": "htdemucs_ft", "segment": 12.5, "shifts": 2},
    )
    assert updated.status_code == 200
    payload = updated.get_json()
    assert payload["model"] == "htdemucs_ft"
    assert payload["segment_length"] == 12.5
    assert payload["shifts"] == 2


def test_pitch_config_get_and_update(client_config_api):
    original = client_config_api.get("/api/v1/config/pitch")
    assert original.status_code == 200
    assert original.get_json()["method"] == "rmvpe"

    updated = client_config_api.patch(
        "/api/v1/config/pitch",
        json={"hop_length": 128, "threshold": 0.45, "use_gpu": False},
    )
    assert updated.status_code == 200
    payload = updated.get_json()
    assert payload["hop_length"] == 128
    assert payload["threshold"] == 0.45
    assert payload["device"] == "cpu"


def test_audio_router_config_get_and_update(client_config_api):
    original = client_config_api.get("/api/v1/audio/router/config")
    assert original.status_code == 200
    assert "speaker_gain" in original.get_json()

    updated = client_config_api.patch(
        "/api/v1/audio/router/config",
        json={"speaker_gain": 1.5, "headphone_enabled": False, "sample_rate": 48000},
    )
    assert updated.status_code == 200
    payload = updated.get_json()
    assert payload["speaker_gain"] == 1.5
    assert payload["headphone_enabled"] is False
    assert payload["sample_rate"] == 48000


def test_audio_router_config_is_persisted(app_config_api, client_config_api):
    from auto_voice.web.persistence import AppStateStore

    response = client_config_api.patch(
        "/api/v1/audio/router/config",
        json={"speaker_device": 2, "headphone_device": 3, "voice_gain": 1.1},
    )

    assert response.status_code == 200
    persisted = AppStateStore(app_config_api.config["DATA_DIR"]).get_audio_router_config()
    assert persisted["speaker_device"] == 2
    assert persisted["headphone_device"] == 3
    assert persisted["voice_gain"] == 1.1


def test_separation_config_is_persisted(app_config_api, client_config_api):
    from auto_voice.web.persistence import AppStateStore

    response = client_config_api.patch(
        "/api/v1/config/separation",
        json={"model": "htdemucs_ft", "segment": 16, "device": "cpu"},
    )

    assert response.status_code == 200
    persisted = AppStateStore(app_config_api.config["DATA_DIR"]).get_separation_config()
    assert persisted["model"] == "htdemucs_ft"
    assert persisted["segment_length"] == 16
    assert persisted["device"] == "cpu"


def test_pitch_config_is_persisted(app_config_api, client_config_api):
    from auto_voice.web.persistence import AppStateStore

    response = client_config_api.patch(
        "/api/v1/config/pitch",
        json={"hop_length": 256, "threshold": 0.4, "use_gpu": False},
    )

    assert response.status_code == 200
    persisted = AppStateStore(app_config_api.config["DATA_DIR"]).get_pitch_config()
    assert persisted["hop_length"] == 256
    assert persisted["threshold"] == 0.4
    assert persisted["device"] == "cpu"


def test_device_config_is_persisted(app_config_api, client_config_api, monkeypatch):
    from auto_voice.web.persistence import AppStateStore

    monkeypatch.setattr(
        "auto_voice.web.audio_router.list_audio_devices",
        lambda device_type=None: (
            [{"device_id": "mic-2", "type": "input"}]
            if device_type == "input"
            else [{"device_id": "spk-2", "type": "output"}]
        ),
    )

    response = client_config_api.post(
        "/api/v1/devices/config",
        json={"input_device_id": "mic-2", "output_device_id": "spk-2", "sample_rate": 32000},
    )

    assert response.status_code == 200
    persisted = AppStateStore(app_config_api.config["DATA_DIR"]).get_device_config()
    assert persisted["input_device_id"] == "mic-2"
    assert persisted["output_device_id"] == "spk-2"
    assert persisted["sample_rate"] == 32000


@pytest.mark.parametrize(
    "endpoint",
    [
        "/api/v1/config/separation",
        "/api/v1/config/pitch",
        "/api/v1/audio/router/config",
    ],
)
def test_config_update_requires_json_payload(client_config_api, endpoint):
    response = client_config_api.patch(endpoint, json={})
    assert response.status_code == 400
    assert "No JSON data provided" in response.get_json()["error"]


def test_config_update_error_paths(client_config_api, monkeypatch):
    from auto_voice.web import api as web_api

    monkeypatch.setattr(web_api.logger, "info", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("config boom")))

    separation_error = client_config_api.patch("/api/v1/config/separation", json={"model": "x"})
    assert separation_error.status_code == 500
    assert "config boom" in separation_error.get_json()["error"]

    pitch_error = client_config_api.patch("/api/v1/config/pitch", json={"method": "harvest"})
    assert pitch_error.status_code == 500
    assert "config boom" in pitch_error.get_json()["error"]

    router_error = client_config_api.patch("/api/v1/audio/router/config", json={"speaker_gain": 1.2})
    assert router_error.status_code == 500
    assert "config boom" in router_error.get_json()["error"]
