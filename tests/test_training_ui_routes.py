from __future__ import annotations

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
    _write_wav(sample_path, duration_seconds=2.0)

    sample = store.add_training_sample(
        profile_id=profile["profile_id"],
        vocals_path=str(sample_path),
        source_file="sample-preview.wav",
        duration=2.0,
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
