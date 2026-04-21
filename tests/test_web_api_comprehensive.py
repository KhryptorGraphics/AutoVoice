"""Current-contract integration coverage for the AutoVoice web API."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import types
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


def _wav_bytes(sample_rate: int = 22050, duration_seconds: float = 1.0) -> io.BytesIO:
    frames = int(sample_rate * duration_seconds)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00" * frames * 2)
    buffer.seek(0)
    return buffer


def _write_wav(path: Path, value: float = 0.0, sample_rate: int = 22050) -> None:
    audio = np.full(sample_rate, value, dtype=np.float32)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes((audio * 32767).astype(np.int16).tobytes())


def _create_profile(
    app,
    *,
    profile_id: str,
    role: str = "target_user",
    name: str | None = None,
    has_trained_model: bool = False,
    has_full_model: bool = False,
    clean_vocal_seconds: float = 0.0,
    selected_adapter: str | None = None,
) -> dict:
    store = app.voice_cloner.store
    profile = {
        "profile_id": profile_id,
        "name": name or f"{role}-{profile_id[-4:]}",
        "embedding": np.zeros(256, dtype=np.float32).tolist(),
        "profile_role": role,
        "created_from": "manual",
        "sample_count": 3,
        "clean_vocal_seconds": clean_vocal_seconds,
        "has_trained_model": has_trained_model,
        "has_adapter_model": has_trained_model,
        "has_full_model": has_full_model,
        "selected_adapter": selected_adapter,
        "active_model_type": (
            "full_model"
            if has_full_model
            else "adapter" if has_trained_model else "base"
        ),
    }
    store.save(profile)

    if has_trained_model:
        adapter_path = Path(store.trained_models_dir) / f"{profile_id}_adapter.pt"
        torch.save(
            {
                "module.lora_A.weight": torch.zeros((8, 8)),
                "module.lora_B.weight": torch.zeros((8, 8)),
            },
            adapter_path,
        )
        embedding_path = Path(store.profiles_dir) / f"{profile_id}.npy"
        np.save(embedding_path, np.zeros((256,), dtype=np.float32))

    if has_full_model:
        full_model_path = Path(store.trained_models_dir) / f"{profile_id}_full_model.pt"
        full_model_path.write_bytes(b"full-model")

    return store.load(profile_id)


@pytest.fixture
def app_current():
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

    yield app


@pytest.fixture
def client_current(app_current):
    return app_current.test_client()


@pytest.fixture
def app_disabled():
    pytest.importorskip("flask_swagger_ui", reason="flask_swagger_ui not installed")
    from auto_voice.web.app import create_app

    app, socketio = create_app(
        config={
            "TESTING": True,
            "singing_conversion_enabled": False,
            "voice_cloning_enabled": False,
        }
    )
    app.socketio = socketio
    return app


@pytest.fixture
def client_disabled(app_disabled):
    return app_disabled.test_client()


class TestConversionEndpoints:
    def test_convert_song_async_uses_canonical_adapter(self, client_current, app_current, monkeypatch):
        profile_id = "00000000-0000-0000-0000-000000000201"
        _create_profile(app_current, profile_id=profile_id, has_trained_model=True)

        monkeypatch.setattr(app_current.job_manager, "create_job", lambda *args, **kwargs: "job-123")

        response = client_current.post(
            "/api/v1/convert/song",
            data={
                "song": (_wav_bytes(), "song.wav"),
                "profile_id": profile_id,
            },
            content_type="multipart/form-data",
        )

        assert response.status_code == 202
        data = response.get_json()
        assert data["job_id"] == "job-123"
        assert data["adapter_type"] == "unified"
        assert data["active_model_type"] == "adapter"

    def test_convert_song_accepts_settings_json(self, client_current, app_current, monkeypatch):
        profile_id = "00000000-0000-0000-0000-000000000202"
        _create_profile(app_current, profile_id=profile_id, has_trained_model=True)
        monkeypatch.setattr(app_current.job_manager, "create_job", lambda *args, **kwargs: "job-settings")

        response = client_current.post(
            "/api/v1/convert/song",
            data={
                "song": (_wav_bytes(), "song.wav"),
                "settings": json.dumps(
                    {
                        "target_profile_id": profile_id,
                        "pipeline_type": "quality_seedvc",
                        "output_quality": "studio",
                    }
                ),
            },
            content_type="multipart/form-data",
        )

        assert response.status_code == 202
        assert response.get_json()["job_id"] == "job-settings"

    def test_convert_song_async_realtime_preserves_pipeline_metadata(
        self,
        client_current,
        app_current,
        monkeypatch,
    ):
        profile_id = "00000000-0000-0000-0000-000000000202a"
        _create_profile(app_current, profile_id=profile_id, has_trained_model=True)
        monkeypatch.setattr(app_current.job_manager, "create_job", lambda *args, **kwargs: "job-realtime")

        response = client_current.post(
            "/api/v1/convert/song",
            data={
                "song": (_wav_bytes(), "song.wav"),
                "profile_id": profile_id,
                "pipeline_type": "realtime",
            },
            content_type="multipart/form-data",
        )

        assert response.status_code == 202
        payload = response.get_json()
        assert payload["job_id"] == "job-realtime"
        assert payload["requested_pipeline"] == "realtime"
        assert payload["resolved_pipeline"] == "realtime"
        assert payload["runtime_backend"] == "pytorch"

    def test_convert_song_rejects_invalid_output_quality(self, client_current, app_current):
        profile_id = "00000000-0000-0000-0000-000000000203"
        _create_profile(app_current, profile_id=profile_id, has_trained_model=True)

        response = client_current.post(
            "/api/v1/convert/song",
            data={
                "song": (_wav_bytes(), "song.wav"),
                "profile_id": profile_id,
                "output_quality": "ultra",
            },
            content_type="multipart/form-data",
        )

        assert response.status_code == 400
        assert "output_quality" in response.get_json()["error"]

    def test_convert_song_rejects_invalid_pipeline_type(self, client_current, app_current):
        profile_id = "00000000-0000-0000-0000-000000000204"
        _create_profile(app_current, profile_id=profile_id, has_trained_model=True)

        response = client_current.post(
            "/api/v1/convert/song",
            data={
                "song": (_wav_bytes(), "song.wav"),
                "profile_id": profile_id,
                "pipeline_type": "ultra-low-latency",
            },
            content_type="multipart/form-data",
        )

        assert response.status_code == 400
        assert "pipeline_type" in response.get_json()["error"]

    def test_convert_song_rejects_source_artist_profile(self, client_current, app_current):
        profile_id = "00000000-0000-0000-0000-000000000205"
        _create_profile(app_current, profile_id=profile_id, role="source_artist")

        response = client_current.post(
            "/api/v1/convert/song",
            data={
                "song": (_wav_bytes(), "song.wav"),
                "profile_id": profile_id,
            },
            content_type="multipart/form-data",
        )

        assert response.status_code == 400
        assert "target user profile" in response.get_json()["error"].lower()

    def test_convert_song_allows_full_model_without_adapter(self, client_current, app_current, monkeypatch):
        profile_id = "00000000-0000-0000-0000-000000000206"
        _create_profile(
            app_current,
            profile_id=profile_id,
            has_trained_model=True,
            has_full_model=True,
            clean_vocal_seconds=3600.0,
        )

        monkeypatch.setattr(app_current, "job_manager", None, raising=False)
        monkeypatch.setattr(
            app_current.singing_conversion_pipeline,
            "convert_song",
            lambda **kwargs: {
                "mixed_audio": np.zeros(22050, dtype=np.float32),
                "sample_rate": 22050,
                "duration": 1.0,
                "metadata": {"active_model_type": "full_model"},
                "f0_contour": np.array([], dtype=np.float32),
                "f0_original": np.array([], dtype=np.float32),
            },
        )

        response = client_current.post(
            "/api/v1/convert/song",
            data={
                "song": (_wav_bytes(), "song.wav"),
                "profile_id": profile_id,
                "adapter_type": "hq",
                "pipeline_type": "quality",
            },
            content_type="multipart/form-data",
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["active_model_type"] == "full_model"
        assert data["adapter_type"] is None

    def test_convert_song_sync_realtime_uses_offline_backend(
        self,
        client_current,
        app_current,
        monkeypatch,
    ):
        profile_id = "00000000-0000-0000-0000-000000000206a"
        _create_profile(app_current, profile_id=profile_id, has_trained_model=True)

        monkeypatch.setattr(app_current, "job_manager", None, raising=False)
        monkeypatch.setattr(
            "auto_voice.web.api.run_offline_realtime_conversion",
            lambda *args, **kwargs: {
                "mixed_audio": np.zeros(22050, dtype=np.float32),
                "sample_rate": 22050,
                "duration": 1.0,
                "metadata": {"pipeline": "realtime"},
                "stems": {},
            },
        )

        response = client_current.post(
            "/api/v1/convert/song",
            data={
                "song": (_wav_bytes(), "song.wav"),
                "profile_id": profile_id,
                "pipeline_type": "realtime",
            },
            content_type="multipart/form-data",
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["requested_pipeline"] == "realtime"
        assert payload["resolved_pipeline"] == "realtime"
        assert payload["runtime_backend"] == "pytorch"
        assert payload["metadata"]["resolved_pipeline"] == "realtime"

    def test_download_missing_or_invalid_asset_returns_404(self, client_current, app_current, monkeypatch):
        monkeypatch.setattr(app_current.job_manager, "get_job_asset_path", lambda *args, **kwargs: MagicMock())
        response = client_current.get("/api/v1/convert/download/job-404")
        assert response.status_code == 404

    def test_reassemble_missing_stems_returns_404(self, client_current, app_current, monkeypatch):
        monkeypatch.setattr(app_current.job_manager, "get_job_asset_path", lambda *args, **kwargs: None)
        response = client_current.get("/api/v1/convert/reassemble/job-404")
        assert response.status_code == 404


class TestVoiceProfileEndpoints:
    def test_voice_clone_creates_profile(self, client_current, app_current):
        app_current.voice_cloner.create_voice_profile = MagicMock(
            return_value={
                "profile_id": "new-profile-123",
                "name": "Test Voice",
                "audio_duration": 10.0,
                "created_at": "2026-04-17T00:00:00Z",
            }
        )

        response = client_current.post(
            "/api/v1/voice/clone",
            data={
                "reference_audio": (_wav_bytes(), "reference.wav"),
                "name": "Test Voice",
            },
            content_type="multipart/form-data",
        )

        assert response.status_code == 201
        data = response.get_json()
        assert data["status"] == "success"
        assert data["profile_id"] == "new-profile-123"

    def test_voice_profiles_list_reads_from_store(self, client_current, app_current):
        _create_profile(app_current, profile_id="00000000-0000-0000-0000-000000000211")
        _create_profile(app_current, profile_id="00000000-0000-0000-0000-000000000212", role="source_artist")

        response = client_current.get("/api/v1/voice/profiles")

        assert response.status_code == 200
        data = response.get_json()
        assert len(data) >= 2
        assert {item["profile_role"] for item in data} >= {"target_user", "source_artist"}

    def test_voice_profile_detail_includes_adapter_path(self, client_current, app_current):
        profile_id = "00000000-0000-0000-0000-000000000213"
        profile = _create_profile(app_current, profile_id=profile_id, has_trained_model=True)
        app_current.voice_cloner.load_voice_profile = MagicMock(return_value=profile)

        response = client_current.get(f"/api/v1/voice/profiles/{profile_id}")

        assert response.status_code == 200
        data = response.get_json()
        assert data["profile_id"] == profile_id
        assert data["adapter_path"].endswith(f"{profile_id}_adapter.pt")

    def test_delete_voice_profile_success(self, client_current, app_current):
        app_current.voice_cloner.delete_voice_profile = MagicMock(return_value=True)
        response = client_current.delete("/api/v1/voice/profiles/00000000-0000-0000-0000-000000000214")
        assert response.status_code == 200
        assert response.get_json()["status"] == "success"


class TestModelEndpoints:
    def test_profile_model_rejects_invalid_uuid(self, client_current):
        response = client_current.get("/api/v1/voice/profiles/not-a-uuid/model")
        assert response.status_code == 400

    def test_profile_model_reports_full_model(self, client_current, app_current):
        profile_id = "00000000-0000-0000-0000-000000000221"
        _create_profile(app_current, profile_id=profile_id, has_full_model=True, clean_vocal_seconds=1900.0)

        response = client_current.get(f"/api/v1/voice/profiles/{profile_id}/model")

        assert response.status_code == 200
        data = response.get_json()
        assert data["model_type"] == "full_model"
        assert data["full_model_eligible"] is True

    def test_profile_model_reports_full_model_pth(self, client_current, app_current):
        profile_id = "00000000-0000-0000-0000-000000000225"
        profile = _create_profile(app_current, profile_id=profile_id, clean_vocal_seconds=1900.0)
        full_model_path = Path(app_current.voice_cloner.store.trained_models_dir) / f"{profile_id}_full_model.pth"
        full_model_path.write_bytes(b"full-model-pth")

        response = client_current.get(f"/api/v1/voice/profiles/{profile_id}/model")

        assert response.status_code == 200
        data = response.get_json()
        assert data["model_type"] == "full_model"
        assert data["model_path"] == str(full_model_path)
        assert data["profile_id"] == profile["profile_id"]

    def test_profile_model_reports_adapter_metadata(self, client_current, app_current):
        profile_id = "00000000-0000-0000-0000-000000000222"
        _create_profile(app_current, profile_id=profile_id, has_trained_model=True, clean_vocal_seconds=600.0)

        response = client_current.get(f"/api/v1/voice/profiles/{profile_id}/model")

        assert response.status_code == 200
        data = response.get_json()
        assert data["model_type"] == "adapter"
        assert data["adapter_info"]["rank"] == 8
        assert data["embedding_shape"] == [256]

    def test_profile_model_reports_tensorrt_engine(self, client_current, app_current):
        profile_id = "00000000-0000-0000-0000-000000000226"
        _create_profile(app_current, profile_id=profile_id, clean_vocal_seconds=400.0)
        engine_path = Path(app_current.voice_cloner.store.trained_models_dir) / f"{profile_id}_nvfp4.engine"
        engine_path.write_bytes(b"engine")

        response = client_current.get(f"/api/v1/voice/profiles/{profile_id}/model")

        assert response.status_code == 200
        data = response.get_json()
        assert data["model_type"] == "tensorrt"
        assert data["model_path"] == str(engine_path)
        assert data["tensorrt_engine_path"] == str(engine_path)

    def test_select_adapter_updates_profile(self, client_current, app_current):
        profile_id = "00000000-0000-0000-0000-000000000223"
        _create_profile(app_current, profile_id=profile_id, has_trained_model=True)

        response = client_current.post(
            f"/api/v1/voice/profiles/{profile_id}/adapter/select",
            json={"adapter_type": "nvfp4"},
        )

        assert response.status_code == 200
        assert response.get_json()["selected_adapter"] == "nvfp4"
        saved = app_current.voice_cloner.store.load(profile_id)
        assert saved["selected_adapter"] == "nvfp4"

    def test_adapter_metrics_returns_architecture(self, client_current, app_current):
        profile_id = "00000000-0000-0000-0000-000000000224"
        _create_profile(app_current, profile_id=profile_id, has_trained_model=True)

        response = client_current.get(f"/api/v1/voice/profiles/{profile_id}/adapter/metrics")

        assert response.status_code == 200
        data = response.get_json()
        assert data["adapter_count"] == 1
        metric = next(iter(data["adapters"].values()))
        assert metric["architecture"]["lora_rank"] == 8

    def test_training_status_reports_existing_profile(self, client_current, app_current):
        profile_id = "00000000-0000-0000-0000-000000000225"
        _create_profile(app_current, profile_id=profile_id, has_trained_model=True, clean_vocal_seconds=900.0)

        response = client_current.get(f"/api/v1/voice/profiles/{profile_id}/training-status")

        assert response.status_code == 200
        data = response.get_json()
        assert data["has_trained_model"] is True
        assert data["clean_vocal_seconds"] == 900.0


class TestUtilityAndConfigEndpoints:
    def test_health_and_ready_reflect_disabled_components(self, client_disabled):
        health = client_disabled.get("/api/v1/health")
        ready = client_disabled.get("/api/v1/ready")

        assert health.status_code == 200
        assert health.get_json()["status"] == "degraded"
        assert ready.status_code == 503
        assert ready.get_json()["ready"] is False

    def test_metrics_json_endpoint_uses_prometheus_analytics(self, client_current):
        fake_module = types.SimpleNamespace(
            get_conversion_analytics=lambda: {"total_conversions": 7, "avg_latency_ms": 42.0}
        )
        with patch.dict(sys.modules, {"auto_voice.monitoring.prometheus": fake_module}):
            response = client_current.get("/api/v1/metrics")

        assert response.status_code == 200
        assert response.get_json()["total_conversions"] == 7

    def test_metrics_prometheus_endpoint_returns_text(self, client_current):
        fake_module = types.SimpleNamespace(
            get_metrics=lambda: "# HELP autovoice_jobs 1\n",
            get_content_type=lambda: "text/plain; version=0.0.4",
            update_gpu_metrics=lambda: None,
        )
        with patch.dict(sys.modules, {"auto_voice.monitoring.prometheus": fake_module}):
            response = client_current.get("/api/v1/metrics?format=prometheus")

        assert response.status_code == 200
        assert response.mimetype == "text/plain"

    def test_pipelines_status_uses_factory(self, client_current):
        from auto_voice.web import api as web_api

        class _Factory:
            @staticmethod
            def get_instance():
                return types.SimpleNamespace(
                    get_status=lambda: {"quality": {"loaded": True, "memory_gb": 1.5}}
                )

        with patch.object(web_api, "PIPELINE_FACTORY_AVAILABLE", True), patch.object(web_api, "PipelineFactory", _Factory):
            response = client_current.get("/api/v1/pipelines/status")

        assert response.status_code == 200
        assert response.get_json()["pipelines"]["quality"]["loaded"] is True

    def test_devices_list_rejects_invalid_type(self, client_current):
        response = client_current.get("/api/v1/devices/list?type=monitor")
        assert response.status_code == 400

    def test_set_device_config_validates_devices(self, client_current):
        with patch(
            "auto_voice.web.audio_router.list_audio_devices",
            side_effect=[
                [{"device_id": "mic-1", "name": "Mic", "type": "input"}],
                [{"device_id": "spk-1", "name": "Speaker", "type": "output"}],
            ],
        ):
            response = client_current.post(
                "/api/v1/devices/config",
                json={
                    "input_device_id": "mic-1",
                    "output_device_id": "spk-1",
                    "sample_rate": 48000,
                },
            )

        assert response.status_code == 200
        assert response.get_json()["sample_rate"] == 48000

    def test_model_management_endpoints_round_trip(self, client_current):
        loaded = client_current.get("/api/v1/models/loaded")
        assert loaded.status_code == 200

        created = client_current.post(
            "/api/v1/models/load",
            json={"model_type": "vocoder", "path": "/tmp/vocoder.pt"},
        )
        assert created.status_code == 201
        assert created.get_json()["model_type"] == "vocoder"

        rebuilt = client_current.post("/api/v1/models/tensorrt/rebuild", json={})
        assert rebuilt.status_code == 200
        assert rebuilt.get_json()["precision"] == "fp16"

        built = client_current.post(
            "/api/v1/models/tensorrt/build",
            json={"precision": "fp8", "models": ["encoder"]},
        )
        assert built.status_code == 200
        assert built.get_json()["models"] == ["encoder"]

        unloaded = client_current.post("/api/v1/models/unload", json={"model_type": "vocoder"})
        assert unloaded.status_code == 204

    def test_config_endpoints_update_values(self, client_current):
        separation = client_current.post("/api/v1/config/separation", json={"segment": 12, "overlap": 0.5})
        assert separation.status_code == 200
        assert separation.get_json()["segment_length"] == 12

        pitch = client_current.patch("/api/v1/config/pitch", json={"use_gpu": False, "threshold": 0.2})
        assert pitch.status_code == 200
        assert pitch.get_json()["device"] == "cpu"

        audio_router = client_current.patch(
            "/api/v1/audio/router/config",
            json={"speaker_gain": 1.3, "instrumental_gain": 0.7},
        )
        assert audio_router.status_code == 200
        assert audio_router.get_json()["speaker_gain"] == 1.3


class TestErrorResponses:
    def test_404_returns_json(self, client_current):
        response = client_current.get("/api/v1/does-not-exist")
        assert response.status_code == 404
        assert response.is_json
        assert response.get_json()["status_code"] == 404

    def test_405_returns_json(self, client_current):
        response = client_current.post("/api/v1/health")
        assert response.status_code == 405
        assert response.is_json
        assert response.get_json()["status_code"] == 405
