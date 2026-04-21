"""Contract checks for the generated API documentation surface."""

from __future__ import annotations

import pytest


@pytest.fixture
def docs_client():
    pytest.importorskip("flask_swagger_ui", reason="flask_swagger_ui not installed")
    from auto_voice.web.app import create_app

    app, _ = create_app(
        config={
            "TESTING": True,
            "singing_conversion_enabled": False,
            "voice_cloning_enabled": False,
            "karaoke_enabled": False,
        },
        testing=True,
    )
    return app.test_client()


def test_openapi_json_has_required_top_level_fields(docs_client):
    response = docs_client.get("/api/v1/openapi.json")
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["openapi"].startswith("3.")
    assert "info" in payload
    assert "paths" in payload


def test_openapi_json_documents_training_control_routes(docs_client):
    response = docs_client.get("/api/v1/openapi.json")
    payload = response.get_json()
    paths = payload["paths"]

    assert "/api/v1/training/jobs/{job_id}" in paths
    assert "/api/v1/training/jobs/{job_id}/cancel" in paths
    assert "/api/v1/training/jobs/{job_id}/pause" in paths
    assert "/api/v1/training/jobs/{job_id}/resume" in paths
    assert "/api/v1/training/jobs/{job_id}/telemetry" in paths
    assert "/api/v1/training/preview/{job_id}" in paths


def test_openapi_yaml_and_swagger_ui_are_exposed(docs_client):
    yaml_response = docs_client.get("/api/v1/openapi.yaml")
    assert yaml_response.status_code == 200
    assert "openapi:" in yaml_response.get_data(as_text=True)

    docs_response = docs_client.get("/docs", follow_redirects=True)
    assert docs_response.status_code == 200
    assert "swagger-ui" in docs_response.get_data(as_text=True).lower()
