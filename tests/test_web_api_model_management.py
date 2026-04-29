"""Targeted coverage for model management API endpoints."""

from __future__ import annotations

import pytest


@pytest.fixture
def app_models(tmp_path):
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
    yield app


@pytest.fixture
def client_models(app_models):
    return app_models.test_client()


def test_model_load_list_and_unload_round_trip(client_models, tmp_path):
    empty = client_models.get("/api/v1/models/loaded")
    assert empty.status_code == 200
    assert empty.get_json() == {"models": []}

    model_path = tmp_path / "encoder.pt"
    model_path.write_bytes(b"encoder")

    loaded = client_models.post(
        "/api/v1/models/load",
        json={"model_type": "encoder", "path": str(model_path)},
    )
    assert loaded.status_code == 201
    payload = loaded.get_json()
    assert payload["model_type"] == "encoder"
    assert payload["status"] == "loaded"
    assert payload["type"] == "encoder"

    listed = client_models.get("/api/v1/models/loaded")
    assert listed.status_code == 200
    model = listed.get_json()["models"][0]
    assert model["model_type"] == "encoder"
    assert model["loaded"] is True
    assert model["runtime_backend"] == "pytorch"

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


def test_model_load_is_persisted(app_models, client_models, tmp_path):
    from auto_voice.web.persistence import AppStateStore

    model_path = tmp_path / "encoder.pt"
    model_path.write_bytes(b"encoder")

    response = client_models.post(
        "/api/v1/models/load",
        json={"model_type": "encoder", "path": str(model_path), "runtime_backend": "tensorrt", "device": "cuda"},
    )
    assert response.status_code == 201

    persisted = AppStateStore(app_models.config["DATA_DIR"]).get_loaded_model("encoder")
    assert persisted is not None
    assert persisted["runtime_backend"] == "tensorrt"
    assert persisted["device"] == "cuda"


def test_tensorrt_status_reports_engine_inventory(client_models, monkeypatch):
    from auto_voice.web import api as web_api

    monkeypatch.setattr(
        web_api,
        "_engine_inventory",
        lambda: [
            {"name": "encoder.engine", "model": "encoder", "path": "/tmp/encoder.engine", "size": 4096},
            {"name": "decoder.plan", "model": "decoder", "path": "/tmp/decoder.plan", "size": 4096},
        ],
    )
    monkeypatch.setattr(web_api, "TORCH_AVAILABLE", False)

    response = client_models.get("/api/v1/models/tensorrt/status")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["available"] is True
    assert payload["engines_available"] is True
    assert "runtime_available" in payload
    assert [engine["name"] for engine in payload["engines"]] == ["encoder.engine", "decoder.plan"]
    assert payload["cuda_available"] is False


def test_tensorrt_status_distinguishes_runtime_from_engine_inventory(client_models, monkeypatch):
    from auto_voice.web import api as web_api

    monkeypatch.setattr(web_api, "_engine_inventory", lambda: [])

    response = client_models.get("/api/v1/models/tensorrt/status")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["available"] is False
    assert payload["engines_available"] is False
    assert isinstance(payload["runtime_available"], bool)


def test_tensorrt_build_and_rebuild_use_defaults_and_overrides(client_models, monkeypatch):
    from auto_voice.web import api as web_api

    monkeypatch.setattr(web_api, "_submit_background_job", lambda *args, **kwargs: None)

    rebuild_default = client_models.post("/api/v1/models/tensorrt/rebuild", json={})
    assert rebuild_default.status_code == 400
    assert "No TensorRT models available to rebuild" in rebuild_default.get_json()["error"]

    monkeypatch.setattr(
        web_api,
        "_engine_inventory",
        lambda: [{"model": "encoder", "name": "encoder.engine", "path": "/tmp/encoder.engine", "size": 1024}],
    )

    rebuild_default = client_models.post("/api/v1/models/tensorrt/rebuild", json={})
    assert rebuild_default.status_code == 202
    assert rebuild_default.get_json()["precision"] == "fp16"
    assert rebuild_default.get_json()["models"] == ["encoder"]

    build_custom = client_models.post(
        "/api/v1/models/tensorrt/build",
        json={"precision": "fp32", "models": ["encoder", "vocoder"]},
    )
    assert build_custom.status_code == 202
    payload = build_custom.get_json()
    assert payload["precision"] == "fp32"
    assert payload["models"] == ["encoder", "vocoder"]


def test_model_management_error_paths(client_models, monkeypatch):
    from auto_voice.web import api as web_api

    monkeypatch.setattr(web_api.logger, "info", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(
        web_api,
        "_engine_inventory",
        lambda: [{"model": "encoder", "name": "encoder.engine", "path": "/tmp/encoder.engine", "size": 1024}],
    )
    monkeypatch.setattr(web_api, "_submit_background_job", lambda *args, **kwargs: None)

    load_error = client_models.post("/api/v1/models/load", json={"model_type": "encoder", "path": __file__})
    assert load_error.status_code == 500
    assert "boom" in load_error.get_json()["error"]

    client_models.application.state_store.save_loaded_model(
        "encoder",
        {"model_type": "encoder", "type": "encoder", "loaded": True, "loaded_at": "2026-01-01T00:00:00Z"},
    )
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

    monkeypatch.setattr(web_api, "_engine_inventory", lambda: (_ for _ in ()).throw(RuntimeError("scan failed")))

    response = client_models.get("/api/v1/models/tensorrt/status")

    assert response.status_code == 500
    assert "scan failed" in response.get_json()["error"]
