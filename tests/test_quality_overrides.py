"""Tests for the per-request quality overrides contract.

Covers:
- ``_parse_quality_overrides`` sanitization (coercion, range checks, whitelists).
- ``POST /api/v1/convert/song`` accepting a ``quality_overrides`` JSON form
  field and storing the sanitized dict into the job settings.
- Validation errors (unknown keys, bad floats, bad f0_method, invalid JSON).
- ``JobManager`` forwarding ``quality_overrides`` to the singing pipeline as a
  kwarg ONLY when present, so legacy ``convert_song`` signatures keep working.
"""

from __future__ import annotations

import io
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from auto_voice.web.api_conversion import _parse_quality_overrides


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wav_bytes(sample_rate: int = 22050, duration_seconds: float = 4.0) -> io.BytesIO:
    frames = int(sample_rate * duration_seconds)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00" * frames * 2)
    buffer.seek(0)
    return buffer


def _create_profile(app, *, profile_id: str) -> dict:
    store = app.voice_cloner.store
    profile = {
        "profile_id": profile_id,
        "name": f"quality-{profile_id[-4:]}",
        "embedding": np.zeros(256, dtype=np.float32).tolist(),
        "profile_role": "target_user",
        "created_from": "manual",
        "sample_count": 0,
        "training_sample_count": 0,
        "clean_vocal_seconds": 0.0,
        "has_trained_model": True,
        "has_adapter_model": True,
        "has_full_model": False,
        "active_model_type": "base",
        "selected_adapter": "unified",
    }
    store.save(profile)
    return store.load(profile_id)


def _materialize_trained_artifact(app, profile_id: str) -> Path:
    trained_models_dir = Path(app.voice_cloner.store.trained_models_dir)
    trained_models_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = trained_models_dir / f"{profile_id}_adapter.pt"
    artifact_path.write_bytes(b"artifact")
    return artifact_path


class _FakeJobManager:
    """Captures create_job() arguments and returns a fixed job id."""

    def __init__(self):
        self.captured = {}

    def create_job(self, file_path, target_profile_id, settings):
        self.captured["file_path"] = file_path
        self.captured["profile_id"] = target_profile_id
        self.captured["settings"] = settings
        return "quality-overrides-job-1"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app_quality():
    pytest.importorskip("flask_swagger_ui", reason="flask_swagger_ui not installed")
    from auto_voice.web.app import create_app

    app, socketio = create_app(
        config={
            "TESTING": True,
            "singing_conversion_enabled": True,
            "voice_cloning_enabled": True,
        }
    )
    app.socketio = socketio
    return app


@pytest.fixture
def client_quality(app_quality):
    return app_quality.test_client()


@pytest.fixture
def convert_env(app_quality):
    """Profile + fake job manager ready for POST /api/v1/convert/song."""
    profile_id = "00000000-0000-0000-0000-000000000401"
    _create_profile(app_quality, profile_id=profile_id)
    _materialize_trained_artifact(app_quality, profile_id)
    fake_job_manager = _FakeJobManager()
    app_quality.job_manager = fake_job_manager
    app_quality.singing_conversion_pipeline = object()
    return profile_id, fake_job_manager


def _post_convert(client, profile_id, quality_overrides=None, raw=None):
    data = {
        "audio": (_wav_bytes(), "song.wav"),
        "profile_id": profile_id,
    }
    if raw is not None:
        data["quality_overrides"] = raw
    elif quality_overrides is not None:
        data["quality_overrides"] = json.dumps(quality_overrides)
    return client.post(
        "/api/v1/convert/song",
        data=data,
        content_type="multipart/form-data",
    )


# ---------------------------------------------------------------------------
# _parse_quality_overrides unit tests
# ---------------------------------------------------------------------------

class TestParseQualityOverrides:
    def test_none_and_empty_pass_through(self):
        assert _parse_quality_overrides(None) == (None, None)
        assert _parse_quality_overrides("") == (None, None)
        assert _parse_quality_overrides("{}") == (None, None)

    def test_bool_coercion_from_strings_and_ints(self):
        sanitized, error = _parse_quality_overrides(json.dumps({
            "enable_dereverb": "1",
            "enable_loudness_transfer": "0",
            "enable_nsf_harmonic_enhancement": "true",
            "enable_pupu_vocoder_refinement": "false",
            "enable_hq_super_resolution": 1,
            "enable_consonant_passthrough": 0,
            "enable_f0_postprocess": True,
        }))
        assert error is None
        assert sanitized == {
            "enable_dereverb": True,
            "enable_loudness_transfer": False,
            "enable_nsf_harmonic_enhancement": True,
            "enable_pupu_vocoder_refinement": False,
            "enable_hq_super_resolution": True,
            "enable_consonant_passthrough": False,
            "enable_f0_postprocess": True,
        }
        assert all(isinstance(v, bool) for v in sanitized.values())

    def test_float_range_checks(self):
        sanitized, error = _parse_quality_overrides(
            {"dereverb_strength": "0.5", "consonant_passthrough_mix": 1})
        assert error is None
        assert sanitized == {"dereverb_strength": 0.5, "consonant_passthrough_mix": 1.0}

        for bad in ({"dereverb_strength": 1.5},
                    {"consonant_passthrough_mix": -0.1},
                    {"dereverb_strength": "nope"},
                    {"dereverb_strength": True}):
            sanitized, error = _parse_quality_overrides(bad)
            assert sanitized is None
            assert error

    def test_f0_method_whitelist(self):
        assert _parse_quality_overrides({"f0_method": "rmvpe"}) == ({"f0_method": "rmvpe"}, None)
        assert _parse_quality_overrides({"f0_method": "PYIN"}) == ({"f0_method": "pyin"}, None)
        sanitized, error = _parse_quality_overrides({"f0_method": "crepe"})
        assert sanitized is None
        assert "f0_method" in error

    def test_unknown_keys_rejected(self):
        sanitized, error = _parse_quality_overrides({"enable_dereverb": True, "hax": 1})
        assert sanitized is None
        assert "hax" in error

    def test_non_object_and_invalid_json_rejected(self):
        assert _parse_quality_overrides("[1,2]")[1]
        assert _parse_quality_overrides("{not json")[1]
        assert _parse_quality_overrides(json.dumps("str"))[1]

    def test_bad_bool_rejected(self):
        sanitized, error = _parse_quality_overrides({"enable_dereverb": "maybe"})
        assert sanitized is None
        assert "enable_dereverb" in error


# ---------------------------------------------------------------------------
# POST /api/v1/convert/song endpoint tests
# ---------------------------------------------------------------------------

class TestConvertSongQualityOverrides:
    def test_valid_overrides_stored_in_job_settings(self, client_quality, convert_env):
        profile_id, fake_job_manager = convert_env
        response = _post_convert(client_quality, profile_id, {
            "enable_dereverb": "1",
            "dereverb_strength": 0.7,
            "enable_consonant_passthrough": True,
            "consonant_passthrough_mix": "0.6",
            "enable_f0_postprocess": "false",
            "f0_method": "pyin",
        })
        assert response.status_code == 202, response.get_data(as_text=True)
        settings = fake_job_manager.captured["settings"]
        assert settings["quality_overrides"] == {
            "enable_dereverb": True,
            "dereverb_strength": 0.7,
            "enable_consonant_passthrough": True,
            "consonant_passthrough_mix": 0.6,
            "enable_f0_postprocess": False,
            "f0_method": "pyin",
        }

    def test_absent_field_leaves_settings_untouched(self, client_quality, convert_env):
        profile_id, fake_job_manager = convert_env
        response = _post_convert(client_quality, profile_id)
        assert response.status_code == 202, response.get_data(as_text=True)
        assert "quality_overrides" not in fake_job_manager.captured["settings"]

    def test_unknown_key_rejected(self, client_quality, convert_env):
        profile_id, fake_job_manager = convert_env
        response = _post_convert(client_quality, profile_id, {"enable_mega_bass": True})
        assert response.status_code == 400
        assert "enable_mega_bass" in response.get_data(as_text=True)
        assert not fake_job_manager.captured

    def test_out_of_range_float_rejected(self, client_quality, convert_env):
        profile_id, fake_job_manager = convert_env
        response = _post_convert(client_quality, profile_id, {"dereverb_strength": 2.0})
        assert response.status_code == 400
        assert not fake_job_manager.captured

    def test_bad_f0_method_rejected(self, client_quality, convert_env):
        profile_id, fake_job_manager = convert_env
        response = _post_convert(client_quality, profile_id, {"f0_method": "crepe"})
        assert response.status_code == 400
        assert not fake_job_manager.captured

    def test_invalid_json_rejected(self, client_quality, convert_env):
        profile_id, fake_job_manager = convert_env
        response = _post_convert(client_quality, profile_id, raw="{not json")
        assert response.status_code == 400
        assert not fake_job_manager.captured


# ---------------------------------------------------------------------------
# JobManager kwarg forwarding tests
# ---------------------------------------------------------------------------

class _CapturingPipeline:
    """convert_song accepting the new quality_overrides kwarg."""

    def __init__(self):
        self.calls = []

    def convert_song(self, **kwargs):
        self.calls.append(kwargs)
        return {"mixed_audio": np.zeros(10, dtype=np.float32), "sample_rate": 22050}


class _LegacyPipeline:
    """convert_song with the OLD signature (no quality_overrides kwarg)."""

    def __init__(self):
        self.calls = []

    def convert_song(self, song_path, target_profile_id, vocal_volume,
                     instrumental_volume, pitch_shift, return_stems, preset,
                     enable_multi_speaker=None, convert_backing=None,
                     preserve_speakers=None):
        self.calls.append({"song_path": song_path, "preset": preset})
        return {"mixed_audio": np.zeros(10, dtype=np.float32), "sample_rate": 22050}


def _make_job_manager(pipeline):
    from auto_voice.web.job_manager import JobManager
    return JobManager(
        config={"max_workers": 1},
        socketio=None,
        singing_pipeline=pipeline,
        voice_profile_manager=None,
    )


def _run_quality_conversion(job_manager, settings):
    settings = {"requested_pipeline": "quality", **settings}
    job = {"file_path": "/tmp/in.wav", "profile_id": "p-1"}
    return job_manager._convert_with_resolved_pipeline("job-1", job, settings)


class TestJobManagerForwarding:
    def test_forwards_quality_overrides_when_present(self):
        pipeline = _CapturingPipeline()
        job_manager = _make_job_manager(pipeline)
        overrides = {"enable_dereverb": True, "dereverb_strength": 0.5}
        _run_quality_conversion(job_manager, {"quality_overrides": overrides})
        assert pipeline.calls[0]["quality_overrides"] == overrides

    def test_omits_kwarg_when_absent(self):
        pipeline = _CapturingPipeline()
        job_manager = _make_job_manager(pipeline)
        _run_quality_conversion(job_manager, {})
        assert "quality_overrides" not in pipeline.calls[0]

    def test_omits_kwarg_for_empty_overrides(self):
        pipeline = _CapturingPipeline()
        job_manager = _make_job_manager(pipeline)
        _run_quality_conversion(job_manager, {"quality_overrides": {}})
        assert "quality_overrides" not in pipeline.calls[0]

    def test_legacy_signature_still_works_without_overrides(self):
        pipeline = _LegacyPipeline()
        job_manager = _make_job_manager(pipeline)
        result = _run_quality_conversion(job_manager, {})
        assert result["sample_rate"] == 22050
        assert len(pipeline.calls) == 1
