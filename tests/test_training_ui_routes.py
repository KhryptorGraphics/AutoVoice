from __future__ import annotations
import json

import threading
from pathlib import Path

import numpy as np
import pytest

from auto_voice.storage.paths import resolve_profiles_dir, resolve_samples_dir
from auto_voice.training.job_manager import TrainingJobManager
from tests.fixtures.audio import write_voiced_wav


def _write_wav(path: Path, sample_rate: int = 22050, duration_seconds: float = 1.0) -> None:
    write_voiced_wav(path, duration_seconds=duration_seconds, sample_rate=sample_rate)


@pytest.fixture
def training_ui_app(tmp_path):
    pytest.importorskip("flask_swagger_ui", reason="flask_swagger_ui not installed")

    from auto_voice.web.app import create_app

    data_dir = tmp_path / "data"
    app, socketio = create_app(
        config={
            "TESTING": True,
            "DATA_DIR": str(data_dir),
            "singing_conversion_enabled": True,
            "voice_cloning_enabled": True,
        }
    )
    app.socketio = socketio

    manager = TrainingJobManager(
        storage_path=data_dir / "app_state",
        require_gpu=False,
        socketio=socketio,
        profiles_dir=str(resolve_profiles_dir(data_dir=str(data_dir))),
        samples_dir=str(resolve_samples_dir(data_dir=str(data_dir))),
    )
    app._training_job_manager = manager
    return app


@pytest.fixture
def training_ui_client(training_ui_app):
    return training_ui_app.test_client()


def _create_target_profile(app, profile_id: str = "profile-training-ui") -> dict:
    store = app.voice_cloner.store
    profile = {
        "profile_id": profile_id,
        "name": "Training UI Profile",
        "embedding": np.zeros(256, dtype=np.float32).tolist(),
        "profile_role": "target_user",
        "created_from": "manual",
        "sample_count": 1,
        "clean_vocal_seconds": 600.0,
        "has_trained_model": True,
        "has_adapter_model": True,
        "has_full_model": False,
        "selected_adapter": "hq",
        "active_model_type": "adapter",
    }
    store.save(profile)
    return store.load(profile_id)


def _prepare_running_job(app):
    store = app.voice_cloner.store
    profile = _create_target_profile(app)
    sample_path = Path(store.samples_dir) / "sample-preview.wav"
    _write_wav(sample_path, duration_seconds=3.5)

    sample = store.add_training_sample(
        profile_id=profile["profile_id"],
        vocals_path=str(sample_path),
        source_file="sample-preview.wav",
        duration=3.5,
    )

    manager = app._training_job_manager
    job = manager.create_job(profile_id=profile["profile_id"], sample_ids=[sample.sample_id])
    job.start(gpu_device=0)
    manager._job_resume_events[job.job_id] = threading.Event()
    manager._job_resume_events[job.job_id].set()
    manager._job_cancel_events[job.job_id] = threading.Event()
    manager._job_runtime_metrics[job.job_id] = {
        "epoch": 2,
        "total_epochs": 10,
        "step": 12,
        "total_steps": 40,
        "loss": 0.24,
        "learning_rate": 1e-4,
        "gpu_metrics": {"memory_used_gb": 3.2, "utilization_percent": 71.0},
        "quality_metrics": {"mos_proxy": 4.1, "speaker_similarity_proxy": 0.92},
        "checkpoint_path": "/tmp/checkpoint_step_1000.pth",
    }
    manager._save_jobs()
    return job, sample


def test_app_settings_round_trip(training_ui_client):
    response = training_ui_client.get("/api/v1/settings/app")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["preferred_pipeline"] == "quality"
    assert payload["preferred_offline_pipeline"] == "quality_seedvc"
    assert payload["preferred_live_pipeline"] == "realtime"

    update = training_ui_client.patch(
        "/api/v1/settings/app",
        json={"preferred_pipeline": "realtime"},
    )

    assert update.status_code == 200
    updated = update.get_json()
    assert updated["preferred_pipeline"] == "realtime"
    assert updated["preferred_offline_pipeline"] == "realtime"
    assert updated["preferred_live_pipeline"] == "realtime"
    assert updated["last_updated"]

    split_update = training_ui_client.patch(
        "/api/v1/settings/app",
        json={
            "preferred_offline_pipeline": "quality_shortcut",
            "preferred_live_pipeline": "realtime_meanvc",
        },
    )

    assert split_update.status_code == 200
    split_payload = split_update.get_json()
    assert split_payload["preferred_offline_pipeline"] == "quality_shortcut"
    assert split_payload["preferred_live_pipeline"] == "realtime_meanvc"
    assert split_payload["preferred_pipeline"] == "quality"


def test_app_settings_multi_speaker_knobs(training_ui_client):
    update = training_ui_client.patch(
        "/api/v1/settings/app",
        json={
            "multi_speaker_separator": "karaoke_model",
            "multi_speaker_backing_gain": 1.4,
            "multi_speaker_backing_voiced_min": 0.55,
        },
    )
    assert update.status_code == 200
    payload = update.get_json()
    assert payload["multi_speaker_separator"] == "karaoke_model"
    assert payload["multi_speaker_backing_gain"] == 1.4
    assert payload["multi_speaker_backing_voiced_min"] == 0.55

    # Persisted: GET returns the stored values, not config defaults.
    fetched = training_ui_client.get("/api/v1/settings/app").get_json()
    assert fetched["multi_speaker_separator"] == "karaoke_model"
    assert fetched["multi_speaker_backing_gain"] == 1.4

    bad_separator = training_ui_client.patch(
        "/api/v1/settings/app",
        json={"multi_speaker_separator": "spectral_magic"},
    )
    assert bad_separator.status_code == 400

    bad_gain = training_ui_client.patch(
        "/api/v1/settings/app",
        json={"multi_speaker_backing_gain": 99},
    )
    assert bad_gain.status_code == 400


def test_training_pause_resume_and_telemetry_routes(training_ui_app, training_ui_client):
    job, sample = _prepare_running_job(training_ui_app)

    pause_response = training_ui_client.post(f"/api/v1/training/jobs/{job.job_id}/pause")
    assert pause_response.status_code == 200
    assert pause_response.get_json()["is_paused"] is True

    telemetry_response = training_ui_client.get(f"/api/v1/training/jobs/{job.job_id}/telemetry")
    assert telemetry_response.status_code == 200
    telemetry = telemetry_response.get_json()
    assert telemetry["preview_available"] is True
    assert telemetry["preview_sample_id"] == sample.sample_id
    assert telemetry["runtime_metrics"]["quality_metrics"]["mos_proxy"] == pytest.approx(4.1)

    resume_response = training_ui_client.post(f"/api/v1/training/jobs/{job.job_id}/resume")
    assert resume_response.status_code == 200
    assert resume_response.get_json()["is_paused"] is False


def test_training_preview_endpoint_returns_audio(training_ui_app, training_ui_client):
    job, _sample = _prepare_running_job(training_ui_app)

    response = training_ui_client.post(
        f"/api/v1/training/preview/{job.job_id}",
        json={"duration_seconds": 1.5},
    )

    assert response.status_code == 200
    assert response.mimetype == "audio/wav"
    assert response.data[:4] == b"RIFF"


def test_training_config_options_include_como_architecture(training_ui_client):
    response = training_ui_client.get("/api/v1/training/config-options")

    assert response.status_code == 200
    payload = response.get_json()
    assert any(item["id"] == "como" for item in payload["architectures"])
    assert "como" in payload["enums"]["architecture"]
    assert payload["defaults"]["training_mode"] == "lora"


def test_create_lora_training_job_uses_unknown_samples_with_warning(
    training_ui_app,
    training_ui_client,
    monkeypatch,
):
    profile = _create_target_profile(training_ui_app, "profile-lora-unknown-qa")
    store = training_ui_app.voice_cloner.store
    sample_path = Path(store.samples_dir) / "unknown-qa.wav"
    _write_wav(sample_path, duration_seconds=3.5)
    sample = store.add_training_sample(
        profile_id=profile["profile_id"],
        vocals_path=str(sample_path),
        source_file="unknown-qa.wav",
        duration=3.5,
    )
    metadata_path = Path(sample.vocals_path).parent / "metadata.json"
    metadata = sample.to_dict()
    metadata.pop("quality_metadata", None)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(TrainingJobManager, "execute_job", lambda self, job_id: None)

    response = training_ui_client.post(
        "/api/v1/training/jobs",
        json={
            "profile_id": profile["profile_id"],
            "config": {"training_mode": "lora"},
        },
    )

    assert response.status_code == 201
    payload = response.get_json()
    assert payload["quality_gate_bypassed"] is True
    assert any("qa_status=unknown" in warning for warning in payload["warnings"])
    assert payload["sample_ids"] == [sample.sample_id]


def test_create_full_training_job_force_overrides_clean_minutes(
    training_ui_app,
    training_ui_client,
    monkeypatch,
):
    profile = _create_target_profile(training_ui_app, "profile-full-force")
    store = training_ui_app.voice_cloner.store
    sample_path = Path(store.samples_dir) / "full-force.wav"
    _write_wav(sample_path, duration_seconds=3.5)
    store.add_training_sample(
        profile_id=profile["profile_id"],
        vocals_path=str(sample_path),
        source_file="full-force.wav",
        duration=3.5,
    )
    payload = {
        "profile_id": profile["profile_id"],
        "config": {"training_mode": "full"},
    }
    monkeypatch.setattr(TrainingJobManager, "execute_job", lambda self, job_id: None)

    blocked_response = training_ui_client.post("/api/v1/training/jobs", json=payload)
    assert blocked_response.status_code == 400

    forced_response = training_ui_client.post(
        "/api/v1/training/jobs",
        json={**payload, "force": True},
    )

    assert forced_response.status_code == 201
    forced_payload = forced_response.get_json()
    assert forced_payload["eligibility_overridden"] is True
    assert any("Full-model eligibility" in warning for warning in forced_payload["warnings"])


def test_training_job_logs_endpoint_returns_buffered_lines(training_ui_app, training_ui_client):
    job, _sample = _prepare_running_job(training_ui_app)
    manager = training_ui_app._training_job_manager

    manager.append_job_log(job.job_id, "hello")

    response = training_ui_client.get(f"/api/v1/training/jobs/{job.job_id}/logs")
    assert response.status_code == 200
    payload = response.get_json()
    assert any("hello" in line for line in payload["lines"])
    assert payload["next_offset"] == 1

    offset_response = training_ui_client.get(
        f"/api/v1/training/jobs/{job.job_id}/logs?offset=1"
    )
    assert offset_response.status_code == 200
    assert offset_response.get_json()["lines"] == []


def test_profile_training_state_prefers_serving_model_path(training_ui_app):
    profile = _create_target_profile(training_ui_app, "profile-serving-model")
    manager = training_ui_app._training_job_manager

    manager._update_profile_training_state(
        profile_id=profile["profile_id"],
        results={
            "adapter_path": "/tmp/profile-serving-model_adapter.pt",
            "serving_model_path": "/tmp/profile-serving-model_adapter_model.pt",
            "manifest_path": "/tmp/profile-serving-model_manifest.json",
            "artifact_type": "adapter",
            "epochs_completed": 3,
            "final_loss": 0.2,
        },
        sample_count=1,
    )

    saved = training_ui_app.voice_cloner.store.load(profile["profile_id"])
    assert saved["model_path"] == "/tmp/profile-serving-model_adapter_model.pt"
