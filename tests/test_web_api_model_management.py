"""Targeted coverage for model management API endpoints."""

from __future__ import annotations

import pytest


@pytest.fixture
def app_models(tmp_path):
    pytest.importorskip("flask_swagger_ui", reason="flask_swagger_ui not installed")

    from auto_voice.web.app import create_app
    from auto_voice.web import api as web_api

    app, socketio = create_app(
        config={
            "TESTING": True,
            "DATA_DIR": str(tmp_path),
        },
        testing=True,
    )
    app.socketio = socketio
    web_api._loaded_models.clear()
    yield app
    web_api._loaded_models.clear()


@pytest.fixture
def client_models(app_models):
    return app_models.test_client()


def test_model_load_list_and_unload_round_trip(client_models):
    empty = client_models.get("/api/v1/models/loaded")
    assert empty.status_code == 200
    assert empty.get_json() == {"models": []}

    loaded = client_models.post(
        "/api/v1/models/load",
        json={"model_type": "encoder", "path": "/tmp/encoder.pt"},
    )
    assert loaded.status_code == 201
    payload = loaded.get_json()
    assert payload["model_type"] == "encoder"
    assert payload["status"] == "loaded"

    listed = client_models.get("/api/v1/models/loaded")
    assert listed.status_code == 200
    assert listed.get_json()["models"][0]["model_type"] == "encoder"

    unloaded = client_models.post("/api/v1/models/unload", json={"model_type": "encoder"})
    assert unloaded.status_code == 204

    listed_after = client_models.get("/api/v1/models/loaded")
    assert listed_after.get_json() == {"models": []}


def test_model_load_and_unload_validate_payload(client_models):
    missing_json = client_models.post("/api/v1/models/load", json={})
    assert missing_json.status_code == 400
    assert "No JSON data provided" in missing_json.get_json()["error"]

    missing_type = client_models.post("/api/v1/models/load", json={"path": "/tmp/model.pt"})
    assert missing_type.status_code == 400
    assert "model_type is required" in missing_type.get_json()["error"]

    missing_unload_json = client_models.post("/api/v1/models/unload", json={})
    assert missing_unload_json.status_code == 400
    assert "No JSON data provided" in missing_unload_json.get_json()["error"]

    missing_unload_type = client_models.post("/api/v1/models/unload", json={"path": "/tmp/model.pt"})
    assert missing_unload_type.status_code == 400
    assert "model_type is required" in missing_unload_type.get_json()["error"]


def test_tensorrt_status_reports_engine_inventory(client_models, monkeypatch):
    from auto_voice.web import api as web_api

    monkeypatch.setattr(web_api.os.path, "exists", lambda path: True)
    monkeypatch.setattr(web_api.os, "listdir", lambda path: ["encoder.engine", "notes.txt", "decoder.plan"])
    monkeypatch.setattr(web_api.os.path, "getsize", lambda path: 4096)
    monkeypatch.setattr(web_api, "TORCH_AVAILABLE", False)

    response = client_models.get("/api/v1/models/tensorrt/status")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["available"] is True
    assert [engine["name"] for engine in payload["engines"]] == ["encoder.engine", "decoder.plan"]
    assert payload["cuda_available"] is False


def test_tensorrt_build_and_rebuild_use_defaults_and_overrides(client_models):
    rebuild_default = client_models.post("/api/v1/models/tensorrt/rebuild", json={})
    assert rebuild_default.status_code == 200
    assert rebuild_default.get_json()["precision"] == "fp16"

    build_custom = client_models.post(
        "/api/v1/models/tensorrt/build",
        json={"precision": "fp32", "models": ["encoder", "vocoder"]},
    )
    assert build_custom.status_code == 200
    payload = build_custom.get_json()
    assert payload["precision"] == "fp32"
    assert payload["models"] == ["encoder", "vocoder"]


def test_model_management_error_paths(client_models, monkeypatch):
    from auto_voice.web import api as web_api

    monkeypatch.setattr(web_api.logger, "info", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    load_error = client_models.post("/api/v1/models/load", json={"model_type": "encoder"})
    assert load_error.status_code == 500
    assert "boom" in load_error.get_json()["error"]

    unload_error = client_models.post("/api/v1/models/unload", json={"model_type": "encoder"})
    assert unload_error.status_code == 500
    assert "boom" in unload_error.get_json()["error"]

    rebuild_error = client_models.post("/api/v1/models/tensorrt/rebuild", json={})
    assert rebuild_error.status_code == 500
    assert "boom" in rebuild_error.get_json()["error"]

    build_error = client_models.post("/api/v1/models/tensorrt/build", json={})
    assert build_error.status_code == 500
    assert "boom" in build_error.get_json()["error"]


def test_tensorrt_status_error_path(client_models, monkeypatch):
    from auto_voice.web import api as web_api

    monkeypatch.setattr(web_api.os.path, "exists", lambda path: True)
    monkeypatch.setattr(web_api.os, "listdir", lambda path: (_ for _ in ()).throw(RuntimeError("scan failed")))

    response = client_models.get("/api/v1/models/tensorrt/status")

    assert response.status_code == 500
    assert "scan failed" in response.get_json()["error"]
