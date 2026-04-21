#!/usr/bin/env python3
"""Deterministic live backend harness for Playwright acceptance tests."""

from __future__ import annotations

import argparse
import os
import tempfile
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auto_voice.storage.voice_profiles import VoiceProfileStore
from auto_voice.web.app import create_app
from auto_voice.web.voice_model_registry import VoiceModelRegistry


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_wav(path: Path, *, sample_rate: int = 24_000, duration_seconds: float = 1.0, amplitude: float = 0.2) -> None:
    frames = int(sample_rate * duration_seconds)
    t = np.linspace(0, duration_seconds, frames, endpoint=False, dtype=np.float32)
    audio = np.sin(2 * np.pi * 220.0 * t, dtype=np.float32) * amplitude
    pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


class FakePipelineFactory:
    """Stable pipeline status backend for browser tests."""

    @classmethod
    def get_instance(cls) -> "FakePipelineFactory":
        return cls()

    def get_status(self) -> dict[str, dict[str, Any]]:
        return {
            "realtime": {
                "loaded": True,
                "memory_gb": 0.8,
                "latency_target_ms": 100,
                "sample_rate": 24_000,
                "description": "Deterministic realtime path for live browser tests",
            },
            "quality": {
                "loaded": True,
                "memory_gb": 1.6,
                "latency_target_ms": 3_000,
                "sample_rate": 22_050,
                "description": "Deterministic quality path for live browser tests",
            },
            "quality_seedvc": {
                "loaded": True,
                "memory_gb": 2.1,
                "latency_target_ms": 2_000,
                "sample_rate": 44_100,
                "description": "Seed-VC quality path",
            },
            "quality_shortcut": {
                "loaded": True,
                "memory_gb": 1.4,
                "latency_target_ms": 1_400,
                "sample_rate": 44_100,
                "description": "Shortcut quality path",
            },
            "realtime_meanvc": {
                "loaded": True,
                "memory_gb": 1.0,
                "latency_target_ms": 120,
                "sample_rate": 24_000,
                "description": "MeanVC realtime path",
            },
        }


class FakeVoiceCloner:
    def __init__(self, store: VoiceProfileStore):
        self.store = store

    def create_voice_profile(self, audio: str, user_id: str | None = None, name: str | None = None) -> dict[str, Any]:
        profile_id = str(uuid.uuid4())
        embedding = np.linspace(0.01, 1.0, 256, dtype=np.float32)
        profile = {
            "profile_id": profile_id,
            "user_id": user_id,
            "name": name or f"Target {profile_id[:8]}",
            "created_at": _iso_now(),
            "created_from": "voice_clone",
            "embedding": embedding.tolist(),
            "profile_role": "target_user",
            "sample_count": 0,
            "training_sample_count": 0,
            "clean_vocal_seconds": 18.0,
            "has_trained_model": False,
            "has_adapter_model": False,
            "has_full_model": False,
            "active_model_type": "base",
        }
        self.store.save(profile)
        saved = self.store.load(profile_id)
        saved.update(
            {
                "audio_duration": 1.0,
                "vocal_range": {"min_hz": 110.0, "max_hz": 440.0},
            }
        )
        return saved

    def load_voice_profile(self, profile_id: str) -> dict[str, Any] | None:
        try:
            return self.store.load(profile_id)
        except Exception:
            return None

    def delete_voice_profile(self, profile_id: str) -> bool:
        return self.store.delete(profile_id)


class FakeJobManager:
    def __init__(self, app):
        self.app = app
        self.assets_dir = Path(app.config["DATA_DIR"]) / "live_conversion_assets"
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, dict[str, Any]] = {}

    def create_job(self, input_path: str, profile_id: str, settings: dict[str, Any]) -> str:
        job_id = str(uuid.uuid4())
        created_at = _iso_now()
        stem_requested = bool(settings.get("return_stems"))
        mix_path = self.assets_dir / f"{job_id}_mix.wav"
        vocals_path = self.assets_dir / f"{job_id}_vocals.wav"
        instrumental_path = self.assets_dir / f"{job_id}_instrumental.wav"
        _write_wav(mix_path, duration_seconds=1.2, amplitude=0.25)
        _write_wav(vocals_path, duration_seconds=1.2, amplitude=0.18)
        _write_wav(instrumental_path, duration_seconds=1.2, amplitude=0.1)

        requested_pipeline = settings.get("requested_pipeline") or settings.get("pipeline_type") or "quality"
        resolved_pipeline = settings.get("resolved_pipeline") or requested_pipeline
        runtime_backend = settings.get("runtime_backend") or "pytorch"
        profile = self.app.voice_cloner.store.load(profile_id)

        self.jobs[job_id] = {
            "polls": 0,
            "status": {
                "id": job_id,
                "status": "queued",
                "progress": 5,
                "created_at": created_at,
                "input_file": Path(input_path).name,
                "profile_id": profile_id,
                "preset": settings.get("preset", "balanced"),
                "requested_pipeline": requested_pipeline,
                "resolved_pipeline": resolved_pipeline,
                "pipeline_type": resolved_pipeline,
                "runtime_backend": runtime_backend,
                "adapter_type": settings.get("adapter_type"),
                "active_model_type": settings.get("active_model_type", "adapter"),
                "audio_duration_seconds": 1.2,
                "targetVoice": profile.get("name") or profile_id,
            },
            "paths": {
                "mix": mix_path,
                "vocals": vocals_path if stem_requested else None,
                "instrumental": instrumental_path if stem_requested else None,
            },
        }
        return job_id

    def _complete_job(self, job_id: str) -> None:
        job = self.jobs[job_id]
        status = job["status"]
        if status["status"] == "completed":
            return

        status.update(
            {
                "status": "completed",
                "progress": 100,
                "completed_at": _iso_now(),
                "processing_time_seconds": 1.5,
                "rtf": 1.25,
                "output_url": f"/api/v1/convert/download/{job_id}",
                "download_url": f"/api/v1/convert/download/{job_id}",
            }
        )
        if job["paths"]["vocals"] and job["paths"]["instrumental"]:
            status["stem_urls"] = {
                "vocals": f"/api/v1/convert/download/{job_id}?variant=vocals",
                "instrumental": f"/api/v1/convert/download/{job_id}?variant=instrumental",
            }
            status["reassemble_url"] = f"/api/v1/convert/reassemble/{job_id}"

        record = dict(status)
        record["resultUrl"] = status["download_url"]
        self.app.state_store.save_conversion_record(record)

    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None

        job["polls"] += 1
        status = job["status"]
        if status["status"] == "queued":
            status.update({"status": "processing", "progress": 65, "started_at": _iso_now()})
        elif status["status"] == "processing":
            self._complete_job(job_id)
        return dict(status)

    def get_job_result_path(self, job_id: str) -> str | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        return str(job["paths"]["mix"])

    def get_job_asset_path(self, job_id: str, variant: str = "mix") -> str | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        path = job["paths"].get(variant)
        return str(path) if path else None

    def cancel_job(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None:
            return False
        job["status"].update({"status": "cancelled", "progress": 0})
        return True

    def get_job_metrics(self, job_id: str) -> dict[str, Any] | None:
        if job_id not in self.jobs:
            return None
        return {
            "speaker_similarity": {"cosine_similarity": 0.93},
            "naturalness": {"mos_estimate": 4.2},
            "intelligibility": {"stoi": 0.95},
        }


class FakeKaraokeManager:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, dict[str, Any]] = {}

    def start_separation(self, job_id: str, song_path: str) -> None:
        vocals_path = self.root / f"{job_id}_vocals.wav"
        instrumental_path = self.root / f"{job_id}_instrumental.wav"
        _write_wav(vocals_path, duration_seconds=2.0, amplitude=0.16)
        _write_wav(instrumental_path, duration_seconds=2.0, amplitude=0.08)
        self.jobs[job_id] = {
            "status": "completed",
            "progress": 100,
            "vocals_path": str(vocals_path),
            "instrumental_path": str(instrumental_path),
            "source_song_path": song_path,
        }

    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        return self.jobs.get(job_id)


def _seed_profile(store: VoiceProfileStore, *, profile_id: str, name: str, trained: bool) -> None:
    profile = {
        "profile_id": profile_id,
        "name": name,
        "created_at": _iso_now(),
        "created_from": "seed",
        "embedding": np.linspace(0.01, 1.0, 256, dtype=np.float32).tolist(),
        "profile_role": "target_user",
        "sample_count": 3 if trained else 1,
        "training_sample_count": 3 if trained else 1,
        "clean_vocal_seconds": 2_100.0 if trained else 30.0,
        "has_trained_model": trained,
        "has_adapter_model": trained,
        "has_full_model": False,
        "selected_adapter": "unified" if trained else None,
        "active_model_type": "adapter" if trained else "base",
    }
    store.save(profile)

    if trained:
        adapter_path = Path(store.trained_models_dir) / f"{profile_id}_adapter.pt"
        torch.save(
            {
                "module.lora_A.weight": torch.zeros((8, 8)),
                "module.lora_B.weight": torch.zeros((8, 8)),
            },
            adapter_path,
        )


def build_app(data_dir: Path):
    from auto_voice.web import api as web_api
    from auto_voice.web import karaoke_api
    import auto_voice.web.audio_router as audio_router

    app, socketio = create_app(
        config={
            "TESTING": True,
            "DATA_DIR": str(data_dir),
            "SECRET_KEY": "live-backend-secret",
            "SERVER_NAME": None,
        },
        testing=True,
    )

    profiles_dir = data_dir / "voice_profiles"
    samples_dir = data_dir / "training_samples"
    store = VoiceProfileStore(profiles_dir=str(profiles_dir), samples_dir=str(samples_dir))

    _seed_profile(store, profile_id="live-demo-singer", name="Live Demo Singer", trained=True)
    _seed_profile(store, profile_id="fresh-profile", name="Fresh Profile", trained=False)

    app.voice_cloner = FakeVoiceCloner(store)
    app.singing_conversion_pipeline = object()
    app.job_manager = FakeJobManager(app)
    app.karaoke_manager = FakeKaraokeManager(data_dir / "karaoke_assets")
    app._device_config = {
        "input_device_id": None,
        "output_device_id": None,
        "sample_rate": 24_000,
    }

    web_api.PIPELINE_FACTORY_AVAILABLE = True
    web_api.PipelineFactory = FakePipelineFactory

    devices = [
        {
            "index": 0,
            "device_id": 0,
            "name": "Live Monitor",
            "type": "output",
            "sample_rate": 48_000,
            "channels": 2,
            "is_default": True,
            "default_sample_rate": 48_000,
        },
        {
            "index": 1,
            "device_id": 1,
            "name": "Performer Headphones",
            "type": "output",
            "sample_rate": 48_000,
            "channels": 2,
            "is_default": False,
            "default_sample_rate": 48_000,
        },
    ]

    def _list_audio_devices(device_type: str | None = None):
        if device_type and device_type != "output":
            return []
        return devices

    audio_router.list_audio_devices = _list_audio_devices

    registry_dir = data_dir / "voice_models"
    registry_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.linspace(0.01, 1.0, 256, dtype=torch.float32), registry_dir / "live_demo_artist.pt")
    karaoke_api._voice_model_registry = VoiceModelRegistry(models_dir=str(registry_dir))
    karaoke_api._device_config["speaker_device"] = 0
    karaoke_api._device_config["headphone_device"] = 1

    return app, socketio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5051)
    args = parser.parse_args()

    data_dir = Path(tempfile.mkdtemp(prefix="autovoice-live-backend-"))
    app, socketio = build_app(data_dir)

    try:
        socketio.run(app, host=args.host, port=args.port, allow_unsafe_werkzeug=True)
    finally:
        if os.environ.get("AUTOVOICE_KEEP_LIVE_BACKEND_DATA") != "1":
            import shutil

            shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
