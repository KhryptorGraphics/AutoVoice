from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest

from auto_voice.storage.voice_profiles import (
    PROFILE_ROLE_SOURCE_ARTIST,
    PROFILE_ROLE_TARGET_USER,
    VoiceProfileStore,
)
from auto_voice.web.app import create_app


@pytest.fixture
def app_workflow(tmp_path):
    app, _socketio = create_app(
        config={
            "TESTING": True,
            "DATA_DIR": str(tmp_path),
            "singing_conversion_enabled": False,
            "voice_cloning_enabled": False,
        }
    )
    return app


@pytest.fixture
def client_workflow(app_workflow):
    return app_workflow.test_client()


class _FakeWorkflowManager:
    def __init__(self):
        self.created = None
        self.attached_job_id = None
        self.resolved = None

    def list_workflows(self):
        return [{"workflow_id": "wf-1", "status": "ready_for_training"}]

    def create_workflow(self, **kwargs):
        self.created = kwargs
        return {
            "workflow_id": "wf-1",
            "status": "processing",
            "stage": "separating_artist_song",
            "progress": 10,
            "artist_song": {"filename": kwargs["artist_song"].filename},
            "user_vocals": [{"filename": upload.filename} for upload in kwargs["user_vocals"]],
            "resolved_source_profiles": [],
            "review_items": [],
            "training_readiness": {"ready": False, "reason": "workflow_incomplete"},
            "conversion_readiness": {"ready": False, "reason": "workflow_incomplete"},
            "created_at": "2026-04-21T00:00:00Z",
            "updated_at": "2026-04-21T00:00:00Z",
        }

    def get_workflow(self, workflow_id):
        if workflow_id != "wf-1":
            return None
        return {
            "workflow_id": workflow_id,
            "status": "ready_for_training",
            "stage": "ready_for_training",
            "progress": 100,
            "artist_song": {"filename": "artist.wav"},
            "user_vocals": [{"filename": "user.wav"}],
            "resolved_source_profiles": [],
            "resolved_target_profile_id": "target-1",
            "review_items": [],
            "training_readiness": {"ready": True, "reason": "ready"},
            "conversion_readiness": {"ready": False, "reason": "target_profile_not_trained"},
            "created_at": "2026-04-21T00:00:00Z",
            "updated_at": "2026-04-21T00:00:00Z",
        }

    def resolve_review_item(self, workflow_id, review_id, **kwargs):
        self.resolved = (workflow_id, review_id, kwargs)
        return {"workflow_id": workflow_id, "review_items": []}

    def attach_training_job(self, workflow_id, job_id):
        self.attached_job_id = (workflow_id, job_id)
        return {"workflow_id": workflow_id, "current_training_job_id": job_id}

    def create_conversion_job(self, workflow_id, settings):
        return {"status": "queued", "job_id": "job-1", "requested_pipeline": settings.get("pipeline_type")}


def test_conversion_workflow_routes_use_manager(client_workflow, app_workflow):
    manager = _FakeWorkflowManager()
    app_workflow.conversion_workflow_manager = manager

    list_response = client_workflow.get("/api/v1/convert/workflows")
    assert list_response.status_code == 200
    assert list_response.get_json()[0]["workflow_id"] == "wf-1"

    create_response = client_workflow.post(
        "/api/v1/convert/workflows",
        data={
            "artist_song": (BytesIO(b"artist"), "artist.wav"),
            "user_vocals": [
                (BytesIO(b"user-1"), "user-1.wav"),
                (BytesIO(b"user-2"), "user-2.wav"),
            ],
            "target_profile_id": "target-override",
            "dominant_source_profile_id": "artist-override",
        },
        content_type="multipart/form-data",
    )
    assert create_response.status_code == 201
    assert manager.created["target_profile_override"] == "target-override"
    assert manager.created["dominant_source_profile_override"] == "artist-override"
    assert len(manager.created["user_vocals"]) == 2

    get_response = client_workflow.get("/api/v1/convert/workflows/wf-1")
    assert get_response.status_code == 200
    assert get_response.get_json()["resolved_target_profile_id"] == "target-1"

    resolve_response = client_workflow.post(
        "/api/v1/convert/workflows/wf-1/resolve-match",
        json={
            "review_id": "review-1",
            "resolution": "create_new",
            "name": "New Artist",
        },
    )
    assert resolve_response.status_code == 200
    assert manager.resolved == ("wf-1", "review-1", {"resolution": "create_new", "profile_id": None, "name": "New Artist"})

    attach_response = client_workflow.post(
        "/api/v1/convert/workflows/wf-1/training-job",
        json={"job_id": "train-1"},
    )
    assert attach_response.status_code == 200
    assert manager.attached_job_id == ("wf-1", "train-1")

    convert_response = client_workflow.post(
        "/api/v1/convert/workflows/wf-1/convert",
        json={"pipeline_type": "quality_seedvc", "return_stems": True},
    )
    assert convert_response.status_code == 202
    assert convert_response.get_json()["job_id"] == "job-1"


def test_conversion_workflow_create_requires_dual_uploads(client_workflow, app_workflow):
    app_workflow.conversion_workflow_manager = _FakeWorkflowManager()

    missing_artist = client_workflow.post(
        "/api/v1/convert/workflows",
        data={"user_vocals": (BytesIO(b"user"), "user.wav")},
        content_type="multipart/form-data",
    )
    assert missing_artist.status_code == 400

    missing_user = client_workflow.post(
        "/api/v1/convert/workflows",
        data={"artist_song": (BytesIO(b"artist"), "artist.wav")},
        content_type="multipart/form-data",
    )
    assert missing_user.status_code == 400


def test_attach_training_job_route_persists_real_workflow_state(client_workflow, app_workflow, tmp_path):
    workflow_id = "wf-persist"
    app_workflow.state_store.save_conversion_workflow(
        {
            "workflow_id": workflow_id,
            "status": "ready_for_training",
            "stage": "ready_for_training",
            "progress": 100,
            "artist_song": {
                "filename": "artist.wav",
                "path": str(tmp_path / "artist.wav"),
            },
            "user_vocals": [],
            "artist_vocals_path": None,
            "instrumental_path": None,
            "diarization_id": None,
            "resolved_source_profiles": [],
            "resolved_target_profile_id": "target-1",
            "review_items": [],
            "target_profile_override": None,
            "dominant_source_profile_override": None,
            "training_readiness": {"ready": True, "reason": "ready"},
            "conversion_readiness": {"ready": False, "reason": "target_profile_not_trained"},
            "user_analysis": {"status": "resolved", "resolved_target_profile_id": "target-1"},
            "artist_analysis": {"status": "resolved"},
            "recovery": {"resume_count": 0, "last_resume_at": None, "last_resume_reason": None},
            "current_training_job_id": None,
            "created_at": "2026-04-22T00:00:00Z",
            "updated_at": "2026-04-22T00:00:00Z",
            "error": None,
        }
    )
    app_workflow.state_store.save_training_job(
        {
            "job_id": "train-persist",
            "profile_id": "target-1",
            "status": "pending",
            "sample_ids": [],
            "progress": 0,
        }
    )

    response = client_workflow.post(
        f"/api/v1/convert/workflows/{workflow_id}/training-job",
        json={"job_id": "train-persist"},
    )

    assert response.status_code == 200
    assert response.get_json()["current_training_job_id"] == "train-persist"
    persisted = app_workflow.state_store.get_conversion_workflow(workflow_id)
    assert persisted is not None
    assert persisted["current_training_job_id"] == "train-persist"
    assert persisted["status"] == "training_in_progress"
    assert persisted["stage"] == "training_in_progress"


def test_rank_speaker_embedding_matches_filters_by_profile_role(tmp_path):
    store = VoiceProfileStore(
        profiles_dir=str(tmp_path / "profiles"),
        samples_dir=str(tmp_path / "samples"),
        trained_models_dir=str(tmp_path / "trained"),
    )
    source_profile_id = store.save({
        "name": "Artist A",
        "profile_role": PROFILE_ROLE_SOURCE_ARTIST,
        "created_from": "manual",
    })
    target_profile_id = store.save({
        "name": "Target User",
        "profile_role": PROFILE_ROLE_TARGET_USER,
        "created_from": "manual",
    })

    source_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    target_embedding = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    store.save_speaker_embedding(source_profile_id, source_embedding)
    store.save_speaker_embedding(target_profile_id, target_embedding)

    ranked_source = store.rank_speaker_embedding_matches(
        np.array([0.99, 0.01, 0.0], dtype=np.float32),
        profile_role=PROFILE_ROLE_SOURCE_ARTIST,
    )
    ranked_target = store.rank_speaker_embedding_matches(
        np.array([0.99, 0.01, 0.0], dtype=np.float32),
        profile_role=PROFILE_ROLE_TARGET_USER,
    )

    assert ranked_source[0]["profile_id"] == source_profile_id
    assert ranked_target[0]["profile_id"] == target_profile_id
    assert store.match_speaker_embedding(
        np.array([0.99, 0.01, 0.0], dtype=np.float32),
        threshold=0.8,
        profile_role=PROFILE_ROLE_SOURCE_ARTIST,
    ) == source_profile_id
