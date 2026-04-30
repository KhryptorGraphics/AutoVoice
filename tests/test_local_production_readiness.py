from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.audio import write_voiced_wav


@pytest.fixture
def local_app(tmp_path):
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
def local_client(local_app):
    return local_app.test_client()


def _store(local_app):
    from auto_voice.storage.paths import resolve_profiles_dir, resolve_samples_dir
    from auto_voice.storage.voice_profiles import VoiceProfileStore

    data_dir = local_app.config["DATA_DIR"]
    return VoiceProfileStore(
        profiles_dir=str(resolve_profiles_dir(data_dir=data_dir)),
        samples_dir=str(resolve_samples_dir(data_dir=data_dir)),
    )


def _create_profile(local_app, profile_id: str, name: str, role: str = "target_user"):
    store = _store(local_app)
    store.save(
        {
            "profile_id": profile_id,
            "name": name,
            "profile_role": role,
            "created_from": "manual",
        }
    )
    return store


def test_sample_review_returns_quality_and_trainable_fields(local_app, local_client, tmp_path):
    store = _create_profile(local_app, "profile-a", "Profile A")
    source = tmp_path / "source.wav"
    write_voiced_wav(source, duration_seconds=1.0, sample_rate=22050)
    store.add_training_sample(
        "profile-a",
        vocals_path=str(source),
        duration=1.0,
        source_file="source.wav",
        extra_metadata={"source": "browser_singalong", "consent_status": "granted"},
    )

    response = local_client.get("/api/v1/samples/review")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["count"] == 1
    sample = payload["samples"][0]
    assert sample["profile_id"] == "profile-a"
    assert sample["profile_name"] == "Profile A"
    assert sample["source"] == "browser_singalong"
    assert sample["consent_status"] == "granted"
    assert sample["duration_seconds"] > 0
    assert isinstance(sample["rms_loudness"], float)
    assert "silence_ratio" in sample
    assert "clipping_ratio" in sample
    assert isinstance(sample["trainable"], bool)
    assert isinstance(sample["issues"], list)
    assert isinstance(sample["recommendations"], list)


def test_duplicate_check_warns_at_configured_similarity_threshold(local_app, local_client):
    store = _create_profile(local_app, "profile-a", "Profile A")
    _create_profile(local_app, "profile-b", "Profile B")
    np.save(store._embedding_path("profile-a"), np.array([1.0, 0.0, 0.0], dtype=np.float32))
    np.save(store._embedding_path("profile-b"), np.array([0.9, 0.1, 0.0], dtype=np.float32))

    response = local_client.get("/api/v1/voice/profiles/duplicate-check?profile_id=profile-a&threshold=0.82")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["duplicate_warning"] is True
    assert payload["candidates"][0]["profile_id"] == "profile-b"
    assert payload["candidates"][0]["similarity"] >= 0.82


def test_backup_export_import_defaults_to_dry_run_and_apply_restores_files(local_client, local_app):
    data_dir = Path(local_app.config["DATA_DIR"])
    profiles_dir = data_dir / "voice_profiles"
    samples_dir = data_dir / "samples" / "profile-a" / "sample_001"
    app_state_dir = data_dir / "app_state"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    app_state_dir.mkdir(parents=True, exist_ok=True)
    (profiles_dir / "profile-a.json").write_text(json.dumps({"profile_id": "profile-a"}))
    (samples_dir / "metadata.json").write_text(json.dumps({"sample_id": "sample_001"}))
    (app_state_dir / "training_jobs.json").write_text("[]")

    export_response = local_client.post("/api/v1/backup/export")
    assert export_response.status_code == 200
    export_payload = export_response.get_json()
    bundle_path = export_payload["backup_path"]
    assert "data/samples" in export_payload["manifest"]["included_paths"]
    assert "data/app_state" in export_payload["manifest"]["included_paths"]

    restored_target = samples_dir / "metadata.json"
    restored_target.unlink()
    dry_run_response = local_client.post("/api/v1/backup/import", data={"backup_path": bundle_path})
    assert dry_run_response.status_code == 200
    assert dry_run_response.get_json()["status"] == "dry_run"
    assert not restored_target.exists()

    apply_response = local_client.post("/api/v1/backup/import?apply=true", data={"backup_path": bundle_path})
    assert apply_response.status_code == 200
    apply_payload = apply_response.get_json()
    assert apply_payload["status"] == "applied"
    assert restored_target.exists()
