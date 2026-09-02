"""A profile whose only trained artifact is a so-vits-svc-fork checkpoint must
be servable from the web API.

so-vits-svc-fork registers its checkpoint in ``data/fork_models/<id>.json`` and
writes nothing under ``trained_models_dir``. Every gate on the serving path
keyed off artifacts that live *inside* the profile directory, so a fork-only
profile was rejected 404 ("does not have a usable target model") even when fully
trained -- and, once past that, fell through to ``quality_seedvc`` and died
importing SeedVC.

This went unnoticed because both fork voices in this deployment still carry a
``*_full_model.pt`` from the older LoRA/full-model engine, so they satisfied the
gates by historical accident rather than by design. The first genuinely
fork-only profile (William Singe, trained 2026-09-02) hit both failures.
"""
from __future__ import annotations

import json

import pytest

from auto_voice.storage.paths import (
    resolve_profiles_dir,
    resolve_samples_dir,
    resolve_trained_models_dir,
)
from auto_voice.storage.voice_profiles import (
    PROFILE_ROLE_TARGET_USER,
    VoiceProfileStore,
)
from auto_voice.web.app import create_app


@pytest.fixture
def fork_only_app(tmp_path):
    """App whose DATA_DIR holds a profile with a fork registry and nothing else."""
    app, _socketio = create_app(
        config={
            "TESTING": True,
            "DATA_DIR": str(tmp_path),
            # Both must be on or /convert/song short-circuits to 503 before
            # the artifact gate, making this test pass vacuously.
            "singing_conversion_enabled": True,
            "voice_cloning_enabled": True,
        }
    )
    store = VoiceProfileStore(
        profiles_dir=str(resolve_profiles_dir(data_dir=str(tmp_path))),
        samples_dir=str(resolve_samples_dir(data_dir=str(tmp_path))),
        trained_models_dir=str(resolve_trained_models_dir(data_dir=str(tmp_path))),
    )
    profile_id = store.save({
        "name": "Fork Only",
        "profile_role": PROFILE_ROLE_TARGET_USER,
        "created_from": "manual",
        "has_trained_model": True,
        "training_status": "ready",
    })

    # The fork artifacts: a registry pointing at files OUTSIDE the profile dir.
    fork_dir = tmp_path / "fork_models" / f"{profile_id}_svcfork"
    fork_dir.mkdir(parents=True, exist_ok=True)
    model = fork_dir / "G.pth"
    config = fork_dir / "config.json"
    model.write_bytes(b"not-a-real-checkpoint")
    config.write_text(json.dumps({"spk": {"forkspk": 0}}))
    (tmp_path / "fork_models" / f"{profile_id}.json").write_text(json.dumps({
        "profile_id": profile_id,
        "engine": "so-vits-svc-fork",
        "speaker": "forkspk",
        "model_path": str(model),
        "config_path": str(config),
        "f0_method": "crepe",
        "transpose": 0,
        "trained_epochs": 300,
    }))

    # Deliberately NOT created: <id>_adapter.pt and <id>_full_model.pt. Their
    # absence is the whole point - it is what made this profile unservable.
    trained = resolve_trained_models_dir(data_dir=str(tmp_path))
    assert not (trained / f"{profile_id}_full_model.pt").exists()
    assert not (trained / f"{profile_id}_adapter.pt").exists()

    from auto_voice.inference import svc_fork_bridge
    svc_fork_bridge.clear_cache()
    yield app, profile_id, str(tmp_path)
    svc_fork_bridge.clear_cache()


def test_bridge_sees_a_fork_only_profile(fork_only_app):
    """Baseline: the registry alone is enough for the bridge."""
    from auto_voice.inference import svc_fork_bridge
    _app, profile_id, data_dir = fork_only_app
    assert svc_fork_bridge.is_available(profile_id, data_dir) is True


def test_profile_has_neither_adapter_nor_profile_dir_full_model(fork_only_app):
    """The condition the old gates required is genuinely absent."""
    _app, profile_id, data_dir = fork_only_app
    store = VoiceProfileStore(
        profiles_dir=str(resolve_profiles_dir(data_dir=data_dir)),
        samples_dir=str(resolve_samples_dir(data_dir=data_dir)),
        trained_models_dir=str(resolve_trained_models_dir(data_dir=data_dir)),
    )
    profile = store.load(profile_id)
    assert profile["has_trained_model"] is True
    assert not profile.get("has_full_model")
    # ...yet it must still be servable. That is the regression.


def test_convert_does_not_reject_fork_only_profile_as_untrained(fork_only_app):
    """The endpoint must not answer 'no usable target model' for a fork profile.

    Drives the real route. The conversion itself is expected to fail later (the
    checkpoint is a stub) - what is pinned here is that it is NOT rejected by
    the artifact gate before the fork lane is ever reached.
    """
    from io import BytesIO

    import numpy as np
    import soundfile as sf

    app, profile_id, _data_dir = fork_only_app
    # A real WAV: a malformed one fails at sf.read() *after* the gate, which
    # would still pass this assertion but for the wrong reason.
    buf = BytesIO()
    sf.write(buf, np.zeros(4410, dtype="float32"), 44100, format="WAV")
    buf.seek(0)
    client = app.test_client()
    resp = client.post(
        "/api/v1/convert/song",
        data={
            "song": (buf, "in.wav"),
            "profile_id": profile_id,
            "settings": json.dumps({"enable_multi_speaker": False}),
        },
        content_type="multipart/form-data",
    )
    body = resp.get_json() or {}
    message = (body.get("message") or "") + (body.get("error") or "")
    assert "does not have a usable target model" not in message, (
        f"fork-only profile rejected by the artifact gate: {resp.status_code} {body}"
    )
    assert "does not have a trained model" not in message, (
        f"fork-only profile reported untrained: {resp.status_code} {body}"
    )


def test_seedvc_request_redirects_to_quality_for_fork_profiles():
    """quality_seedvc must resolve to the quality lane for a fork-backed profile.

    Asserted against the real method's source: _convert_with_resolved_pipeline
    runs a full GPU conversion, so it cannot be invoked here, and replicating
    its branch in the test would pin the copy rather than the code.
    """
    import inspect
    from auto_voice.web.job_manager import JobManager

    src = inspect.getsource(JobManager._convert_with_resolved_pipeline)
    assert "svc_fork_bridge" in src, "redirect no longer consults the fork bridge"
    assert "fork_backed" in src and "full_model_lane" in src

    redirect = src.split("full_model_lane and requested_pipeline")[1]
    assert "quality_seedvc" in redirect.split("resolved_pipeline = 'quality'")[0], (
        "the seedvc->quality redirect no longer covers fork-backed profiles"
    )
