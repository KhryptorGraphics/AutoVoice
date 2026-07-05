"""Web-layer surfacing of svc-fork metadata: conversion_metadata on job status,
fork_backed/fork_engine on profile routes, and the training architecture enum.

Part (a) drives JobManager.get_job_status directly (a full Flask app is
disproportionate for a plain dict passthrough). Parts (b)/(c) build a real app
the same way tests/test_training_ui_routes.py does.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from auto_voice.inference import svc_fork_bridge
from auto_voice.web.job_manager import JobManager


# ---------------------------------------------------------------------------
# (a) conversion_metadata on job status
# ---------------------------------------------------------------------------

def _completed_job(*, conversion_metadata, settings):
    now = time.time()
    return {
        'status': 'completed',
        'progress': 100,
        'file_path': '/tmp/does-not-matter.wav',
        'input_file': 'song.wav',
        'profile_id': 'prof-1',
        'settings': settings,
        'created_at': now - 5.0,
        'started_at': now - 4.0,
        'completed_at': now,
        'result_path': None,
        'stem_paths': {},
        'error': None,
        'metrics': None,
        'duration': 3.0,
        'sample_rate': 44100,
        'conversion_metadata': conversion_metadata,
    }


def test_get_job_status_surfaces_conversion_metadata_and_prefers_its_model_type():
    jm = JobManager(config={}, socketio=MagicMock(), singing_pipeline=None,
                    voice_profile_manager=None)
    try:
        metadata = {
            'active_model_type': 'svc_fork',
            'stereo': True,
            'multi_speaker': True,
            'multi_speaker_info': {'num_speakers': 2, 'primary_speaker': 'SPEAKER_00'},
        }
        # settings say 'adapter' — the pipeline's own metadata must win.
        jm._jobs['job-1'] = _completed_job(
            conversion_metadata=metadata,
            settings={'active_model_type': 'adapter', 'pipeline_type': 'quality'},
        )

        status = jm.get_job_status('job-1')

        assert status['conversion_metadata'] == metadata  # verbatim
        assert status['active_model_type'] == 'svc_fork'  # metadata beats settings
    finally:
        jm.stop()


def test_get_job_status_falls_back_to_settings_model_type_without_metadata():
    jm = JobManager(config={}, socketio=MagicMock(), singing_pipeline=None,
                    voice_profile_manager=None)
    try:
        jm._jobs['job-2'] = _completed_job(
            conversion_metadata={},  # legacy lane reported nothing
            settings={'active_model_type': 'adapter'},
        )
        status = jm.get_job_status('job-2')
        assert status['conversion_metadata'] == {}
        assert status['active_model_type'] == 'adapter'
    finally:
        jm.stop()


# ---------------------------------------------------------------------------
# Shared real-app fixture for (b) and (c)
# ---------------------------------------------------------------------------

@pytest.fixture
def app(tmp_path):
    pytest.importorskip("flask_swagger_ui", reason="flask_swagger_ui not installed")
    from auto_voice.storage.paths import resolve_profiles_dir, resolve_samples_dir
    from auto_voice.training.job_manager import TrainingJobManager
    from auto_voice.web.app import create_app

    svc_fork_bridge.clear_cache()
    data_dir = tmp_path / "data"
    flask_app, socketio = create_app(config={
        "TESTING": True,
        "DATA_DIR": str(data_dir),
        "singing_conversion_enabled": True,
        "voice_cloning_enabled": True,
    })
    flask_app.socketio = socketio
    flask_app._training_job_manager = TrainingJobManager(
        storage_path=data_dir / "app_state",
        require_gpu=False,
        socketio=socketio,
        profiles_dir=str(resolve_profiles_dir(data_dir=str(data_dir))),
        samples_dir=str(resolve_samples_dir(data_dir=str(data_dir))),
    )
    yield flask_app
    svc_fork_bridge.clear_cache()


@pytest.fixture
def client(app):
    return app.test_client()


def _save_target_profile(app, profile_id):
    app.voice_cloner.store.save({
        "profile_id": profile_id,
        "name": f"Profile {profile_id}",
        "embedding": np.zeros(256, dtype=np.float32).tolist(),
        "profile_role": "target_user",
        "created_from": "manual",
        "has_trained_model": True,
        "has_adapter_model": True,
        "active_model_type": "adapter",
        "selected_adapter": "hq",
    })


def _register_fork_model(app, profile_id, *, speaker="connor", epochs=100, f0_method="crepe"):
    """Write a valid fork registry entry (svc_fork_bridge validates the referenced
    model/config exist on disk)."""
    data_dir = Path(app.config["DATA_DIR"])
    model_path = data_dir / "fork_model.pth"
    config_path = data_dir / "fork_config.json"
    model_path.write_bytes(b"\x00")
    config_path.write_text("{}")
    registry_dir = data_dir / svc_fork_bridge.REGISTRY_DIRNAME
    registry_dir.mkdir(parents=True, exist_ok=True)
    (registry_dir / f"{profile_id}.json").write_text(json.dumps({
        "speaker": speaker,
        "model_path": str(model_path),
        "config_path": str(config_path),
        "f0_method": f0_method,
        "trained_epochs": epochs,
    }))
    svc_fork_bridge.clear_cache()


# ---------------------------------------------------------------------------
# (b) fork_backed / fork_engine on profile routes
# ---------------------------------------------------------------------------

def test_profiles_list_reports_fork_backed(client, app):
    fork_id = "11111111-1111-1111-1111-111111111111"
    plain_id = "22222222-2222-2222-2222-222222222222"
    _save_target_profile(app, fork_id)
    _save_target_profile(app, plain_id)
    _register_fork_model(app, fork_id)

    resp = client.get("/api/v1/voice/profiles")
    assert resp.status_code == 200
    by_id = {p["profile_id"]: p for p in resp.get_json()}
    assert by_id[fork_id]["fork_backed"] is True
    assert by_id[plain_id]["fork_backed"] is False


def test_profile_adapters_report_fork_engine(client, app):
    fork_id = "33333333-3333-3333-3333-333333333333"
    _save_target_profile(app, fork_id)
    _register_fork_model(app, fork_id, speaker="connor", epochs=140, f0_method="crepe")

    resp = client.get(f"/api/v1/voice/profiles/{fork_id}/adapters")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["fork_backed"] is True
    assert data["fork_engine"] == {
        "speaker": "connor",
        "trained_epochs": 140,
        "f0_method": "crepe",
    }


def test_profile_adapters_omit_fork_engine_when_not_fork_backed(client, app):
    plain_id = "44444444-4444-4444-4444-444444444444"
    _save_target_profile(app, plain_id)

    resp = client.get(f"/api/v1/voice/profiles/{plain_id}/adapters")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["fork_backed"] is False
    assert "fork_engine" not in data


# ---------------------------------------------------------------------------
# (c) training architecture enum
# ---------------------------------------------------------------------------

def test_config_options_advertises_architectures(client):
    resp = client.get("/api/v1/training/config-options")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["defaults"]["architecture"] == "diffusion_mel"
    assert data["enums"]["architecture"] == ["diffusion_mel", "mel_gan", "svc_fork"]
    ids = {arch["id"] for arch in data["architectures"]}
    assert ids == {"diffusion_mel", "mel_gan", "svc_fork"}


def test_create_training_job_accepts_svc_fork_architecture(client, app, monkeypatch):
    from auto_voice.storage.voice_profiles import VoiceProfileStore
    from auto_voice.storage.paths import resolve_profiles_dir, resolve_samples_dir
    from auto_voice.training.job_manager import TrainingJobManager
    from tests.fixtures.audio import write_voiced_wav

    monkeypatch.setattr(TrainingJobManager, "execute_job", lambda self, job_id: None)

    profile_id = "55555555-5555-5555-5555-555555555555"
    data_dir = app.config["DATA_DIR"]
    store = VoiceProfileStore(
        profiles_dir=str(resolve_profiles_dir(data_dir=data_dir)),
        samples_dir=str(resolve_samples_dir(data_dir=data_dir)),
    )
    store.save({
        "profile_id": profile_id,
        "name": "Fork Train Target",
        "profile_role": "target_user",
        "created_from": "manual",
    })
    sample_wav = Path(data_dir) / "sample.wav"
    write_voiced_wav(sample_wav, duration_seconds=6.0)
    store.add_training_sample(
        profile_id=profile_id,
        vocals_path=str(sample_wav),
        duration=6.0,
        source_file="sample.wav",
    )

    resp = client.post(
        "/api/v1/training/jobs",
        json={"profile_id": profile_id, "config": {"architecture": "svc_fork"}},
        content_type="application/json",
    )
    assert resp.status_code == 201, resp.get_json()


def test_create_training_job_rejects_invalid_architecture(client):
    resp = client.post(
        "/api/v1/training/jobs",
        json={"profile_id": "anything", "config": {"architecture": "bogus"}},
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert "architecture" in resp.get_json()["error"]
