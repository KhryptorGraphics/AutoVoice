"""Contract checks for the generated API documentation surface."""

from __future__ import annotations

import json
from pathlib import Path

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


def test_postman_collection_covers_current_training_control_routes():
    collection = json.loads(Path("docs/api/postman_collection.json").read_text())

    requests = set()

    def walk(items):
        for item in items:
            request = item.get("request")
            if request:
                raw_url = request.get("url", {}).get("raw")
                requests.add((request.get("method"), raw_url))
            for child in item.get("item", []):
                walk([child])

    walk(collection.get("item", []))

    assert ("POST", "{{base_url}}/api/{{api_version}}/training/jobs/{{job_id}}/pause") in requests
    assert ("POST", "{{base_url}}/api/{{api_version}}/training/jobs/{{job_id}}/resume") in requests
    assert ("GET", "{{base_url}}/api/{{api_version}}/training/jobs/{{job_id}}/telemetry") in requests
    assert ("POST", "{{base_url}}/api/{{api_version}}/training/preview/{{job_id}}") in requests
    assert ("GET", "{{base_url}}/api/{{api_version}}/voice/profiles/{{profile_id}}/training-status") in requests
    assert ("GET", "{{base_url}}/api/{{api_version}}/settings/app") in requests
    assert ("PATCH", "{{base_url}}/api/{{api_version}}/settings/app") in requests


def test_secondary_profile_docs_explain_current_route_ownership_and_training_namespace():
    profile_doc = Path("docs/api-voice-profile.md").read_text()
    architecture_doc = Path("docs/continuous-learning-architecture.md").read_text()

    assert "/api/v1/voice/profiles/*" in profile_doc
    assert "/api/v1/profiles/*" in profile_doc
    assert "There is no separate `/training` Socket.IO namespace" in profile_doc
    assert "training currently does not have dedicated room semantics and should be filtered by `job_id`" in architecture_doc
