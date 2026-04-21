"""Targeted coverage for web preset CRUD endpoints."""

from __future__ import annotations

import pytest


@pytest.fixture
def app_presets(tmp_path):
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
    web_api._presets.clear()
    yield app
    web_api._presets.clear()


@pytest.fixture
def client_presets(app_presets):
    return app_presets.test_client()


def test_presets_crud_round_trip(client_presets):
    create_response = client_presets.post(
        "/api/v1/presets",
        json={"name": "Studio", "config": {"pitch_shift": 2, "reverb": 0.1}},
    )
    assert create_response.status_code == 201
    created = create_response.get_json()
    preset_id = created["id"]
    assert created["name"] == "Studio"
    assert created["config"]["pitch_shift"] == 2
    assert created["config"]["pipeline_type"] == "quality_seedvc"

    list_response = client_presets.get("/api/v1/presets")
    assert list_response.status_code == 200
    assert any(item["id"] == preset_id for item in list_response.get_json())

    get_response = client_presets.get(f"/api/v1/presets/{preset_id}")
    assert get_response.status_code == 200
    assert get_response.get_json()["name"] == "Studio"

    update_response = client_presets.patch(
        f"/api/v1/presets/{preset_id}",
        json={"name": "Stage", "config": {"pitch_shift": -1}},
    )
    assert update_response.status_code == 200
    updated = update_response.get_json()
    assert updated["name"] == "Stage"
    assert updated["config"]["pitch_shift"] == -1

    delete_response = client_presets.delete(f"/api/v1/presets/{preset_id}")
    assert delete_response.status_code == 204

    missing_response = client_presets.get(f"/api/v1/presets/{preset_id}")
    assert missing_response.status_code == 404


def test_create_preset_validates_payload(client_presets):
    no_json = client_presets.post("/api/v1/presets", json={})
    assert no_json.status_code == 400
    assert "No JSON data provided" in no_json.get_json()["error"]

    missing_name = client_presets.post("/api/v1/presets", json={"config": {"pitch_shift": 1}})
    assert missing_name.status_code == 400
    assert "name is required" in missing_name.get_json()["error"]


def test_update_preset_handles_missing_and_invalid_payload(client_presets):
    missing_response = client_presets.patch("/api/v1/presets/missing", json={"name": "Ignored"})
    assert missing_response.status_code == 404

    create_response = client_presets.post("/api/v1/presets", json={"name": "Original"})
    preset_id = create_response.get_json()["id"]

    invalid_response = client_presets.patch(
        f"/api/v1/presets/{preset_id}",
        data="null",
        content_type="application/json",
    )
    assert invalid_response.status_code == 400
    assert "No JSON data provided" in invalid_response.get_json()["error"]


def test_create_preset_rejects_live_only_pipeline(client_presets):
    response = client_presets.post(
        "/api/v1/presets",
        json={"name": "Live only", "config": {"pipeline_type": "realtime_meanvc"}},
    )

    assert response.status_code == 400
    assert "pipeline_type" in response.get_json()["error"]


def test_delete_preset_returns_not_found_for_unknown_id(client_presets):
    response = client_presets.delete("/api/v1/presets/unknown")
    assert response.status_code == 404
