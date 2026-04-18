"""Additional endpoint coverage for remaining web/api.py routes."""

from __future__ import annotations

import io
import json
import os
import sys
import types
import wave
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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


def _write_wav(path: Path, *, sample_rate: int = 22050, duration_seconds: float = 1.0) -> None:
    frames = int(sample_rate * duration_seconds)
    audio = np.zeros(frames, dtype=np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())


def _create_profile(
    app,
    *,
    profile_id: str,
    role: str = "target_user",
    name: str | None = None,
    clean_vocal_seconds: float = 0.0,
) -> dict:
    store = app.voice_cloner.store
    profile = {
        "profile_id": profile_id,
        "name": name or f"{role}-{profile_id[-4:]}",
        "embedding": np.zeros(256, dtype=np.float32).tolist(),
        "profile_role": role,
        "created_from": "manual",
        "sample_count": 0,
        "training_sample_count": 0,
        "clean_vocal_seconds": clean_vocal_seconds,
        "has_trained_model": False,
        "has_adapter_model": False,
        "has_full_model": False,
        "active_model_type": "base",
    }
    store.save(profile)
    return store.load(profile_id)


def _add_training_sample(app, profile_id: str, source_path: Path, duration: float = 1.0):
    return app.voice_cloner.store.add_training_sample(
        profile_id=profile_id,
        vocals_path=str(source_path),
        source_file=source_path.name,
        duration=duration,
    )


@dataclass
class _FakeSegment:
    start: float
    end: float
    speaker_id: str
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class _FakeDiarizationResult:
    segments: list[_FakeSegment]
    audio_duration: float
    num_speakers: int

    def get_speaker_total_duration(self, speaker_id: str) -> float:
        return sum(segment.duration for segment in self.segments if segment.speaker_id == speaker_id)

    def get_all_speaker_ids(self) -> list[str]:
        return sorted({segment.speaker_id for segment in self.segments})


@dataclass
class _FakeProfileStatus:
    profile_id: str
    needs_retrain: bool
    needs_training: bool
    is_stale: bool
    quality_ok: bool
    sample_count: int
    issues: list[str]
    recommendations: list[str]


@dataclass
class _FakeAuditSummary:
    total_profiles: int
    profiles_with_adapters: int
    profiles_needing_training: int
    profiles_needing_retrain: int
    stale_adapters: int
    low_quality_adapters: int
    adapter_types: dict[str, int]


def _install_audit_module(monkeypatch, *, statuses=None, summary=None, error: Exception | None = None):
    fake_audit_module = types.ModuleType("scripts.audit_loras")

    class FakeAuditor:
        def __init__(self, verbose=False):
            self.verbose = verbose

        def audit_all(self):
            if error is not None:
                raise error
            return statuses or [], summary

    fake_audit_module.LoRAAuditor = FakeAuditor
    scripts_pkg = types.ModuleType("scripts")
    scripts_pkg.audit_loras = fake_audit_module
    monkeypatch.setitem(sys.modules, "scripts", scripts_pkg)
    monkeypatch.setitem(sys.modules, "scripts.audit_loras", fake_audit_module)


@pytest.fixture
def app_remaining():
    pytest.importorskip("flask_swagger_ui", reason="flask_swagger_ui not installed")
    from auto_voice.web.app import create_app
    from auto_voice.web import api as web_api

    app, socketio = create_app(
        config={
            "TESTING": True,
            "singing_conversion_enabled": True,
            "voice_cloning_enabled": True,
        }
    )
    app.socketio = socketio

    web_api._loaded_models.clear()
    web_api._profile_samples.clear()
    web_api._separation_jobs.clear()
    web_api._diarization_results.clear()
    web_api._segment_assignments.clear()
    web_api._presets.clear()
    web_api._conversion_history.clear()
    web_api._profile_checkpoints.clear()
    web_api._youtube_downloader = None
    yield app
    web_api._loaded_models.clear()
    web_api._profile_samples.clear()
    web_api._separation_jobs.clear()
    web_api._diarization_results.clear()
    web_api._segment_assignments.clear()
    web_api._presets.clear()
    web_api._conversion_history.clear()
    web_api._profile_checkpoints.clear()
    web_api._youtube_downloader = None


@pytest.fixture
def client_remaining(app_remaining):
    return app_remaining.test_client()


class _FakeTrainingJobManager:
    def __init__(self):
        self.jobs = {}
        self.executed = []
        self.full_model_check = {
            "needs_full_model": True,
            "clean_vocal_seconds": 2000.0,
            "remaining_seconds": 0.0,
        }

    def list_jobs(self, profile_id=None):
        jobs = list(self.jobs.values())
        if profile_id:
            jobs = [job for job in jobs if job.profile_id == profile_id]
        return jobs

    def check_needs_full_model(self, profile_id):
        return dict(self.full_model_check)

    def create_job(self, *, profile_id, sample_ids, config):
        job = SimpleNamespace(
            job_id=f"job-{len(self.jobs) + 1}",
            profile_id=profile_id,
            status="pending",
            sample_ids=list(sample_ids),
            config=config,
        )
        job.to_dict = lambda: {
            "job_id": job.job_id,
            "profile_id": job.profile_id,
            "status": job.status,
            "sample_ids": job.sample_ids,
            "config": job.config.to_dict(),
            "progress": 0,
            "created_at": "2026-04-17T00:00:00",
            "started_at": None,
            "completed_at": None,
            "results": None,
            "error": None,
            "gpu_device": None,
        }
        self.jobs[job.job_id] = job
        return job

    def create_full_model_job(self, *, profile_id, config):
        job = self.create_job(profile_id=profile_id, sample_ids=["sample_001"], config=config)
        job.to_dict()["results"] = {"job_type": "full_model"}
        return job

    def execute_job(self, job_id):
        self.executed.append(job_id)

    def get_job(self, job_id):
        return self.jobs.get(job_id)

    def cancel_job(self, job_id):
        job = self.jobs.get(job_id)
        if not job:
            return False
        job.status = "cancelled"
        return True

    def auto_queue_training(self, profile_id):
        job = self.create_job(
            profile_id=profile_id,
            sample_ids=["sample_001"],
            config=SimpleNamespace(to_dict=lambda: {"training_mode": "lora"}),
        )
        return job


class TestTrainingEndpoints:
    def test_create_training_job_rejects_missing_json(self, client_remaining):
        response = client_remaining.post("/api/v1/training/jobs", data=b"", content_type="application/json")
        assert response.status_code == 400
        assert "No JSON data provided" in response.get_json()["error"]

    def test_create_training_job_rejects_profile_without_samples(self, client_remaining, app_remaining):
        profile_id = "00000000-0000-0000-0000-000000000301"
        _create_profile(app_remaining, profile_id=profile_id)

        response = client_remaining.post("/api/v1/training/jobs", json={"profile_id": profile_id})
        assert response.status_code == 400
        assert "No training samples found" in response.get_json()["error"]

    def test_create_training_job_rejects_invalid_training_mode(self, client_remaining, app_remaining, tmp_path):
        profile_id = "00000000-0000-0000-0000-000000000302"
        _create_profile(app_remaining, profile_id=profile_id)
        sample_path = tmp_path / "sample.wav"
        _write_wav(sample_path)
        _add_training_sample(app_remaining, profile_id, sample_path)

        response = client_remaining.post(
            "/api/v1/training/jobs",
            json={"profile_id": profile_id, "config": {"training_mode": "mystery"}},
        )
        assert response.status_code == 400
        assert "training_mode" in response.get_json()["error"]

    def test_create_training_job_full_mode_uses_canonical_manager(self, client_remaining, app_remaining, tmp_path):
        profile_id = "00000000-0000-0000-0000-000000000303"
        _create_profile(app_remaining, profile_id=profile_id, clean_vocal_seconds=3600.0)
        sample_path = tmp_path / "full.wav"
        _write_wav(sample_path)
        _add_training_sample(app_remaining, profile_id, sample_path)

        manager = _FakeTrainingJobManager()
        app_remaining._training_job_manager = manager

        response = client_remaining.post(
            "/api/v1/training/jobs",
            json={"profile_id": profile_id, "config": {"training_mode": "full"}},
        )

        assert response.status_code == 201
        data = response.get_json()
        assert data["job_id"] == "job-1"
        assert manager.executed == ["job-1"]

    def test_cancel_training_job_success(self, client_remaining, app_remaining):
        manager = _FakeTrainingJobManager()
        job = manager.create_job(
            profile_id="00000000-0000-0000-0000-000000000304",
            sample_ids=["sample_001"],
            config=SimpleNamespace(to_dict=lambda: {"training_mode": "lora"}),
        )
        app_remaining._training_job_manager = manager

        response = client_remaining.post(f"/api/v1/training/jobs/{job.job_id}/cancel")
        assert response.status_code == 200
        assert response.get_json()["status"] == "cancelled"


class TestVoiceCloneAndGpuFallbacks:
    def test_clone_voice_invalid_audio_returns_validation_error(self, client_remaining, app_remaining):
        from auto_voice.inference.voice_cloner import InvalidAudioError

        def _raise_invalid(**kwargs):
            raise InvalidAudioError("bad reference")

        app_remaining.voice_cloner.create_voice_profile = _raise_invalid

        response = client_remaining.post(
            "/api/v1/voice/clone",
            data={"reference_audio": (_wav_bytes(), "voice.wav")},
            content_type="multipart/form-data",
        )

        assert response.status_code == 400
        payload = response.get_json()
        assert payload["error"] == "Invalid reference audio"
        assert payload["error_code"] == "invalid_reference_audio"

    def test_clone_voice_quality_error_includes_details(self, client_remaining, app_remaining):
        from auto_voice.inference.voice_cloner import InsufficientQualityError

        def _raise_quality(**kwargs):
            raise InsufficientQualityError(
                "too noisy",
                error_code="noise_floor",
                details={"snr_db": 4.2},
            )

        app_remaining.voice_cloner.create_voice_profile = _raise_quality

        response = client_remaining.post(
            "/api/v1/voice/clone",
            data={"reference_audio": (_wav_bytes(), "voice.wav")},
            content_type="multipart/form-data",
        )

        assert response.status_code == 400
        payload = response.get_json()
        assert payload["error"] == "Insufficient audio quality"
        assert payload["error_code"] == "noise_floor"
        assert payload["details"]["snr_db"] == 4.2

    def test_clone_voice_inconsistent_samples_includes_details(self, client_remaining, app_remaining):
        from auto_voice.inference.voice_cloner import InconsistentSamplesError

        def _raise_inconsistent(**kwargs):
            raise InconsistentSamplesError(
                "multiple singers",
                details={"speaker_count": 2},
            )

        app_remaining.voice_cloner.create_voice_profile = _raise_inconsistent

        response = client_remaining.post(
            "/api/v1/voice/clone",
            data={"reference_audio": (_wav_bytes(), "voice.wav")},
            content_type="multipart/form-data",
        )

        assert response.status_code == 400
        payload = response.get_json()
        assert payload["error"] == "Inconsistent audio samples"
        assert payload["details"]["speaker_count"] == 2

    def test_clone_voice_generic_error_returns_service_unavailable(self, client_remaining, app_remaining):
        def _raise_generic(**kwargs):
            raise RuntimeError("gpu warmup failed")

        app_remaining.voice_cloner.create_voice_profile = _raise_generic

        response = client_remaining.post(
            "/api/v1/voice/clone",
            data={"reference_audio": (_wav_bytes(), "voice.wav")},
            content_type="multipart/form-data",
        )

        assert response.status_code == 503
        payload = response.get_json()
        assert payload["error"] == "Temporary service unavailability during voice cloning"

    def test_gpu_metrics_uses_torch_fallback_when_pynvml_missing(self, client_remaining, monkeypatch):
        from auto_voice.web import api as web_api

        class _FakeProps:
            name = "Fallback GPU"
            total_memory = 8 * 1024**3

        real_import = __import__

        def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "pynvml":
                raise ImportError("missing")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(web_api, "TORCH_AVAILABLE", True)
        monkeypatch.setattr(web_api.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(web_api.torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(web_api.torch.cuda, "get_device_properties", lambda idx: _FakeProps())
        monkeypatch.setattr(web_api.torch.cuda, "memory_allocated", lambda idx: 2 * 1024**3)
        monkeypatch.setattr(web_api.torch.cuda, "memory_reserved", lambda idx: 3 * 1024**3)

        with patch("builtins.__import__", side_effect=_fake_import):
            response = client_remaining.get("/api/v1/gpu/metrics")

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["available"] is True
        assert payload["note"] == "Limited metrics (pynvml not available)"
        assert payload["devices"][0]["name"] == "Fallback GPU"

    def test_gpu_metrics_returns_partial_metrics_on_runtime_failure(self, client_remaining, monkeypatch):
        from auto_voice.web import api as web_api

        class _FakeProps:
            name = "Partial GPU"
            total_memory = 12 * 1024**3

        fake_pynvml = types.ModuleType("pynvml")
        fake_pynvml.nvmlInit = lambda: (_ for _ in ()).throw(RuntimeError("nvml busy"))

        monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)
        monkeypatch.setattr(web_api, "TORCH_AVAILABLE", True)
        monkeypatch.setattr(web_api.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(web_api.torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(web_api.torch.cuda, "get_device_properties", lambda idx: _FakeProps())

        response = client_remaining.get("/api/v1/gpu/metrics")

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["devices"][0]["name"] == "Partial GPU"
        assert "Some metrics unsupported" in payload["note"]


class TestSampleAndDiarizationEndpoints:
    def test_upload_sample_success_with_metadata(self, client_remaining, app_remaining):
        profile_id = "00000000-0000-0000-0000-000000000311"
        _create_profile(app_remaining, profile_id=profile_id)

        response = client_remaining.post(
            f"/api/v1/profiles/{profile_id}/samples",
            data={
                "file": (_wav_bytes(), "sample.wav"),
                "metadata": json.dumps({"source_file": "cover.wav", "note": "clean"}),
            },
            content_type="multipart/form-data",
        )

        assert response.status_code == 201
        payload = response.get_json()
        assert payload["profile_id"] == profile_id
        assert payload["metadata"]["note"] == "clean"

    def test_add_sample_from_path_skip_separation(self, client_remaining, app_remaining, tmp_path):
        profile_id = "00000000-0000-0000-0000-000000000312"
        _create_profile(app_remaining, profile_id=profile_id)
        source_path = tmp_path / "existing.wav"
        _write_wav(source_path)

        response = client_remaining.post(
            f"/api/v1/profiles/{profile_id}/samples/from-path",
            json={"audio_path": str(source_path), "skip_separation": True},
        )

        assert response.status_code == 201
        payload = response.get_json()
        assert payload["metadata"]["source"] == "youtube_download"
        assert payload["metadata"]["original_path"] == str(source_path)

    def test_add_sample_from_path_runs_separation(self, client_remaining, app_remaining, tmp_path, monkeypatch):
        from auto_voice.web import api as web_api
        import soundfile as sf

        profile_id = "00000000-0000-0000-0000-000000000313"
        _create_profile(app_remaining, profile_id=profile_id)
        source_path = tmp_path / "source.wav"
        _write_wav(source_path)

        monkeypatch.setattr(web_api, "TORCH_AVAILABLE", False)
        monkeypatch.setattr(sf, "read", lambda path: (np.zeros(22050, dtype=np.float32), 22050))
        monkeypatch.setattr(sf, "info", lambda path: SimpleNamespace(duration=1.0))
        written = []

        def _fake_write(path, audio, sr):
            Path(path).write_bytes(b"wav")
            written.append((path, sr))

        monkeypatch.setattr(sf, "write", _fake_write)

        fake_sep_module = types.ModuleType("auto_voice.audio.separation")

        class FakeSeparator:
            def __init__(self, segment=10.0):
                self.segment = segment

            def separate(self, audio, sr):
                return {
                    "vocals": np.zeros(22050, dtype=np.float32),
                    "instrumental": np.zeros(22050, dtype=np.float32),
                }

        fake_sep_module.VocalSeparator = FakeSeparator
        monkeypatch.setitem(sys.modules, "auto_voice.audio.separation", fake_sep_module)

        response = client_remaining.post(
            f"/api/v1/profiles/{profile_id}/samples/from-path",
            json={"audio_path": str(source_path), "skip_separation": False},
        )

        assert response.status_code == 201
        assert len(written) == 2
        assert response.get_json()["metadata"]["separated"] is True

    def test_upload_song_auto_split_runs_background_job(self, client_remaining, app_remaining, tmp_path, monkeypatch):
        import soundfile as sf

        profile_id = "00000000-0000-0000-0000-000000000314"
        _create_profile(app_remaining, profile_id=profile_id)

        class ImmediateThread:
            def __init__(self, target=None, daemon=None):
                self._target = target

            def start(self):
                self._target()

        monkeypatch.setattr("threading.Thread", ImmediateThread)
        monkeypatch.setattr(sf, "read", lambda path: (np.zeros((22050, 2), dtype=np.float32), 22050))
        monkeypatch.setattr(sf, "write", lambda path, audio, sr: Path(path).write_bytes(b"wav"))

        fake_sep_module = types.ModuleType("auto_voice.audio.separation")

        class FakeSeparator:
            def separate(self, audio, sr):
                return {
                    "vocals": np.zeros(22050, dtype=np.float32),
                    "instrumental": np.zeros(22050, dtype=np.float32),
                }

        fake_sep_module.VocalSeparator = FakeSeparator
        monkeypatch.setitem(sys.modules, "auto_voice.audio.separation", fake_sep_module)

        response = client_remaining.post(
            f"/api/v1/profiles/{profile_id}/songs",
            data={"file": (_wav_bytes(), "song.wav"), "auto_split": "true"},
            content_type="multipart/form-data",
        )

        assert response.status_code == 202
        job_id = response.get_json()["job_id"]

        status_response = client_remaining.get(f"/api/v1/separation/{job_id}/status")
        assert status_response.status_code == 200
        payload = status_response.get_json()
        assert payload["status"] == "complete"
        assert payload["vocals_path"] is not None

    def test_filter_sample_set_embedding_assign_segment_and_auto_create_profile(
        self, client_remaining, app_remaining, tmp_path, monkeypatch
    ):
        profile_id = "00000000-0000-0000-0000-000000000315"
        _create_profile(app_remaining, profile_id=profile_id)
        sample_path = tmp_path / "sample.wav"
        _write_wav(sample_path, duration_seconds=2.0)
        sample = _add_training_sample(app_remaining, profile_id, sample_path, duration=2.0)

        fake_sd_module = types.ModuleType("auto_voice.audio.speaker_diarization")

        class FakeSpeakerDiarizer:
            def diarize(self, audio_path, num_speakers=None):
                return _FakeDiarizationResult(
                    segments=[
                        _FakeSegment(0.0, 1.0, "SPEAKER_00"),
                        _FakeSegment(1.0, 2.0, "SPEAKER_01"),
                    ],
                    audio_duration=2.0,
                    num_speakers=2,
                )

            def extract_speaker_embedding(self, audio_path, start=None, end=None):
                return np.ones(4, dtype=np.float32)

            def extract_speaker_audio(self, audio_path, diarization, speaker_id, output_path=None):
                path = Path(output_path or (tmp_path / f"{speaker_id}.wav"))
                _write_wav(path)
                return str(path)

        fake_sd_module.SpeakerDiarizer = FakeSpeakerDiarizer
        fake_sd_module.DiarizationResult = _FakeDiarizationResult
        fake_sd_module.SpeakerSegment = _FakeSegment
        monkeypatch.setitem(sys.modules, "auto_voice.audio.speaker_diarization", fake_sd_module)

        fake_filter_module = types.ModuleType("auto_voice.audio.training_filter")

        class FakeTrainingDataFilter:
            def filter_training_audio(self, *, audio_path, target_embedding, similarity_threshold):
                filtered_path = tmp_path / "filtered.wav"
                _write_wav(filtered_path)
                return filtered_path, {
                    "original_duration": 2.0,
                    "filtered_duration": 1.4,
                    "num_segments": 2,
                    "purity": 0.91,
                    "status": "success",
                }

        fake_filter_module.TrainingDataFilter = FakeTrainingDataFilter
        monkeypatch.setitem(sys.modules, "auto_voice.audio.training_filter", fake_filter_module)

        diarize_response = client_remaining.post(
            "/api/v1/audio/diarize",
            data={"file": (_wav_bytes(duration_seconds=2.0), "duet.wav")},
            content_type="multipart/form-data",
        )
        assert diarize_response.status_code == 200
        diarization_id = diarize_response.get_json()["diarization_id"]

        set_embedding_response = client_remaining.post(
            f"/api/v1/profiles/{profile_id}/speaker-embedding",
            json={"use_samples": True},
        )
        assert set_embedding_response.status_code == 200

        get_embedding_response = client_remaining.get(f"/api/v1/profiles/{profile_id}/speaker-embedding")
        assert get_embedding_response.status_code == 200
        assert get_embedding_response.get_json()["has_embedding"] is True

        filter_response = client_remaining.post(
            f"/api/v1/profiles/{profile_id}/samples/{sample.sample_id}/filter",
            json={"similarity_threshold": 0.75},
        )
        assert filter_response.status_code == 200
        assert filter_response.get_json()["status"] == "success"

        assign_response = client_remaining.post(
            "/api/v1/audio/diarize/assign",
            json={
                "diarization_id": diarization_id,
                "segment_index": 0,
                "profile_id": profile_id,
                "extract_audio": True,
            },
        )
        assert assign_response.status_code == 200
        assert assign_response.get_json()["profile_id"] == profile_id

        segments_response = client_remaining.get(f"/api/v1/profiles/{profile_id}/segments")
        assert segments_response.status_code == 200
        assert segments_response.get_json()["total_segments"] >= 2

        create_response = client_remaining.post(
            "/api/v1/profiles/auto-create",
            json={
                "diarization_id": diarization_id,
                "speaker_id": "SPEAKER_01",
                "name": "Extracted Artist",
                "extract_segments": True,
            },
        )
        assert create_response.status_code == 201
        payload = create_response.get_json()
        assert payload["profile_role"] == "source_artist"
        assert payload["num_segments"] == 1

    def test_auto_create_profile_legacy_bulk_contract(
        self, client_remaining, app_remaining, tmp_path, monkeypatch
    ):
        fake_sd_module = types.ModuleType("auto_voice.audio.speaker_diarization")

        class FakeSpeakerDiarizer:
            def diarize(self, audio_path, num_speakers=None):
                return _FakeDiarizationResult(
                    segments=[
                        _FakeSegment(0.0, 1.0, "SPEAKER_00"),
                        _FakeSegment(1.0, 2.0, "SPEAKER_01"),
                    ],
                    audio_duration=2.0,
                    num_speakers=2,
                )

            def extract_speaker_embedding(self, audio_path, start=None, end=None):
                return np.ones(4, dtype=np.float32)

            def extract_speaker_audio(self, audio_path, diarization=None, speaker_id=None, output_path=None):
                path = Path(output_path or (tmp_path / f"{speaker_id}.wav"))
                _write_wav(path)
                return str(path)

        fake_sd_module.SpeakerDiarizer = FakeSpeakerDiarizer
        fake_sd_module.DiarizationResult = _FakeDiarizationResult
        fake_sd_module.SpeakerSegment = _FakeSegment
        monkeypatch.setitem(sys.modules, "auto_voice.audio.speaker_diarization", fake_sd_module)

        diarize_response = client_remaining.post(
            "/api/v1/audio/diarize",
            data={"file": (_wav_bytes(duration_seconds=2.0), "legacy.wav")},
            content_type="multipart/form-data",
        )
        assert diarize_response.status_code == 200
        diarization_payload = diarize_response.get_json()

        response = client_remaining.post(
            "/api/v1/profiles/auto-create",
            json={
                "segment_key": diarization_payload["segment_key"],
                "artist_names": ["Lead", "Guest"],
                "profile_role": "source_artist",
            },
        )

        assert response.status_code == 201
        payload = response.get_json()
        assert payload["status"] == "success"
        assert len(payload["profiles"]) == 2
        assert {profile["name"] for profile in payload["profiles"]} == {"Lead", "Guest"}


class TestLifecycleAndAnalysisErrorBranches:
    def test_identify_speaker_invalid_threshold_defaults_to_safe_value(
        self, client_remaining, monkeypatch
    ):
        captured = {}
        identify_module = types.ModuleType("auto_voice.inference.voice_identifier")

        def _identify_from_file(path, threshold):
            captured["threshold"] = threshold
            return SimpleNamespace(
                profile_id="p1",
                profile_name="Singer",
                similarity=0.9,
                is_match=True,
                all_similarities={"p1": 0.9},
            )

        identify_module.get_voice_identifier = lambda: SimpleNamespace(
            identify_from_file=_identify_from_file
        )
        monkeypatch.setitem(sys.modules, "auto_voice.inference.voice_identifier", identify_module)

        response = client_remaining.post(
            "/api/v1/audio/identify-speaker",
            data={"file": (_wav_bytes(), "voice.wav"), "threshold": "not-a-number"},
            content_type="multipart/form-data",
        )

        assert response.status_code == 200
        assert captured["threshold"] == 0.85

    def test_identify_speaker_failure_returns_error_response(self, client_remaining, monkeypatch):
        identify_module = types.ModuleType("auto_voice.inference.voice_identifier")
        identify_module.get_voice_identifier = lambda: (_ for _ in ()).throw(RuntimeError("identifier down"))
        monkeypatch.setitem(sys.modules, "auto_voice.inference.voice_identifier", identify_module)

        response = client_remaining.post(
            "/api/v1/audio/identify-speaker",
            data={"file": (_wav_bytes(), "voice.wav")},
            content_type="multipart/form-data",
        )

        assert response.status_code == 500
        assert "identifier down" in response.get_json()["error"]

    def test_audit_loras_full_json_response(self, client_remaining, monkeypatch):
        _install_audit_module(
            monkeypatch,
            statuses=[
                _FakeProfileStatus(
                    profile_id="p1",
                    needs_retrain=False,
                    needs_training=False,
                    is_stale=False,
                    quality_ok=True,
                    sample_count=3,
                    issues=[],
                    recommendations=[],
                )
            ],
            summary=_FakeAuditSummary(
                total_profiles=1,
                profiles_with_adapters=1,
                profiles_needing_training=0,
                profiles_needing_retrain=0,
                stale_adapters=0,
                low_quality_adapters=0,
                adapter_types={"unified": 1},
            ),
        )

        response = client_remaining.get("/api/v1/loras/audit")

        assert response.status_code == 200
        payload = response.get_json()
        assert "audit_timestamp" in payload
        assert payload["summary"]["total_profiles"] == 1
        assert payload["profiles"][0]["profile_id"] == "p1"

    def test_audit_loras_failure_returns_error(self, client_remaining, monkeypatch):
        _install_audit_module(monkeypatch, error=RuntimeError("audit crashed"))

        response = client_remaining.get("/api/v1/loras/audit")

        assert response.status_code == 500
        assert "audit crashed" in response.get_json()["error"]

    def test_check_retrain_profile_not_found(self, client_remaining, monkeypatch):
        _install_audit_module(monkeypatch, statuses=[], summary=_FakeAuditSummary(0, 0, 0, 0, 0, 0, {}))

        response = client_remaining.post("/api/v1/profiles/missing/check-retrain", json={})

        assert response.status_code == 404
        assert "Profile missing not found" in response.get_json()["error"]

    def test_check_retrain_queue_failure_returns_training_error(self, client_remaining, monkeypatch):
        profile_id = "profile-retrain"
        _install_audit_module(
            monkeypatch,
            statuses=[
                _FakeProfileStatus(
                    profile_id=profile_id,
                    needs_retrain=True,
                    needs_training=False,
                    is_stale=True,
                    quality_ok=False,
                    sample_count=4,
                    issues=["stale"],
                    recommendations=["retrain"],
                )
            ],
            summary=_FakeAuditSummary(1, 1, 0, 1, 1, 0, {"unified": 1}),
        )
        monkeypatch.setattr(
            "auto_voice.web.api._queue_lora_training_job",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("queue busy")),
        )

        response = client_remaining.post(f"/api/v1/profiles/{profile_id}/check-retrain", json={"trigger": True})

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["training_queued"] is False
        assert payload["training_error"] == "queue busy"

    def test_check_retrain_unhandled_error_returns_error_response(self, client_remaining, monkeypatch):
        _install_audit_module(monkeypatch, error=RuntimeError("audit unavailable"))

        response = client_remaining.post("/api/v1/profiles/p1/check-retrain", json={"trigger": True})

        assert response.status_code == 500
        assert "audit unavailable" in response.get_json()["error"]

    def test_analyze_conversion_requires_paths(self, client_remaining):
        response = client_remaining.post("/api/v1/convert/analyze", json={"source_audio": "src.wav"})
        assert response.status_code == 400
        assert "source_audio and converted_audio required" in response.get_json()["error"]

    def test_analyze_conversion_failure_returns_error(self, client_remaining, monkeypatch):
        analysis_module = types.ModuleType("auto_voice.evaluation.conversion_quality_analyzer")

        class BrokenAnalyzer:
            def analyze(self, **kwargs):
                raise RuntimeError("analysis failed")

        analysis_module.ConversionQualityAnalyzer = BrokenAnalyzer
        monkeypatch.setitem(sys.modules, "auto_voice.evaluation.conversion_quality_analyzer", analysis_module)

        response = client_remaining.post(
            "/api/v1/convert/analyze",
            json={"source_audio": "src.wav", "converted_audio": "out.wav"},
        )

        assert response.status_code == 500
        assert "analysis failed" in response.get_json()["error"]

    def test_compare_methodologies_requires_outputs(self, client_remaining):
        response = client_remaining.post("/api/v1/convert/compare-methodologies", json={"source_audio": "src.wav"})
        assert response.status_code == 400
        assert "source_audio and converted_outputs required" in response.get_json()["error"]

    def test_compare_methodologies_failure_returns_error(self, client_remaining, monkeypatch):
        analysis_module = types.ModuleType("auto_voice.evaluation.conversion_quality_analyzer")

        class BrokenAnalyzer:
            def compare_methodologies(self, **kwargs):
                raise RuntimeError("comparison failed")

        analysis_module.ConversionQualityAnalyzer = BrokenAnalyzer
        monkeypatch.setitem(sys.modules, "auto_voice.evaluation.conversion_quality_analyzer", analysis_module)

        response = client_remaining.post(
            "/api/v1/convert/compare-methodologies",
            json={"source_audio": "src.wav", "converted_outputs": {"quality": "out.wav"}},
        )

        assert response.status_code == 500
        assert "comparison failed" in response.get_json()["error"]

    def test_retrain_lora_failure_returns_error(self, client_remaining, monkeypatch):
        monkeypatch.setattr(
            "auto_voice.web.api._queue_lora_training_job",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("retrain backend down")),
        )

        response = client_remaining.post("/api/v1/loras/retrain/p1", json={})

        assert response.status_code == 500
        assert "retrain backend down" in response.get_json()["error"]


class TestStateAndModelRoutes:
    def test_presets_history_checkpoints_and_tensorrt_routes(self, client_remaining, app_remaining, tmp_path):
        create_preset = client_remaining.post("/api/v1/presets", json={"name": "Studio", "config": {"gain": 2}})
        assert create_preset.status_code == 201
        preset_id = create_preset.get_json()["id"]

        assert client_remaining.get("/api/v1/presets").status_code == 200
        assert client_remaining.put(f"/api/v1/presets/{preset_id}", json={"name": "Studio+"}).status_code == 200
        assert client_remaining.delete(f"/api/v1/presets/{preset_id}").status_code == 204

        app_remaining.state_store.save_conversion_record(
            {"id": "record-1", "profile_id": "p1", "created_at": "2026-04-17T00:00:00Z"}
        )
        assert client_remaining.get("/api/v1/convert/history").status_code == 200
        update_record = client_remaining.patch(
            "/api/v1/convert/history/record-1",
            json={"notes": "great", "isFavorite": True, "tags": ["demo"]},
        )
        assert update_record.status_code == 200
        assert update_record.get_json()["notes"] == "great"
        assert client_remaining.delete("/api/v1/convert/history/record-1").status_code == 204

        app_remaining.state_store.save_checkpoint(
            "profile-a",
            {"id": "checkpoint-1", "created_at": "2026-04-17T00:00:00Z", "epoch": 5},
        )
        assert client_remaining.get("/api/v1/profiles/profile-a/checkpoints").status_code == 200
        rollback = client_remaining.post("/api/v1/profiles/profile-a/checkpoints/checkpoint-1/rollback")
        assert rollback.status_code == 200
        assert rollback.get_json()["status"] == "rolled_back"
        assert client_remaining.delete("/api/v1/profiles/profile-a/checkpoints/checkpoint-1").status_code == 204

        status = client_remaining.get("/api/v1/models/tensorrt/status")
        assert status.status_code == 200
        assert "available" in status.get_json()
        assert client_remaining.post("/api/v1/models/tensorrt/rebuild", json={"precision": "fp32"}).status_code == 200
        assert client_remaining.post(
            "/api/v1/models/tensorrt/build",
            json={"precision": "fp16", "models": ["encoder"]},
        ).status_code == 200


class TestYoutubeAnalysisAndQualityRoutes:
    def test_youtube_history_info_and_download(self, client_remaining, app_remaining, tmp_path, monkeypatch):
        from auto_voice.web import api as web_api

        web_api.YOUTUBE_DOWNLOADER_AVAILABLE = True

        class FakeDownloader:
            def get_video_info(self, url):
                return SimpleNamespace(
                    success=True,
                    title="Video",
                    duration=180.0,
                    main_artist="Artist",
                    featured_artists=["Feat"],
                    is_cover=False,
                    original_artist=None,
                    song_title="Song",
                    thumbnail_url="https://example.com/thumb.jpg",
                    video_id="vid123",
                    error=None,
                )

            def download(self, url, audio_format="wav", sample_rate=44100):
                audio_path = tmp_path / f"download.{audio_format}"
                _write_wav(audio_path, sample_rate=sample_rate)
                return SimpleNamespace(
                    success=True,
                    audio_path=str(audio_path),
                    title="Video",
                    duration=180.0,
                    main_artist="Artist",
                    featured_artists=["Feat"],
                    is_cover=False,
                    original_artist=None,
                    song_title="Song",
                    thumbnail_url="https://example.com/thumb.jpg",
                    video_id="vid123",
                    error=None,
                )

        monkeypatch.setattr(web_api, "get_youtube_downloader", lambda: FakeDownloader())

        fake_sd_module = types.ModuleType("auto_voice.audio.speaker_diarization")

        class FakeSpeakerDiarizer:
            def diarize(self, audio_path):
                return _FakeDiarizationResult(
                    segments=[
                        _FakeSegment(0.0, 1.5, "SPEAKER_00"),
                        _FakeSegment(1.5, 2.0, "SPEAKER_01"),
                    ],
                    audio_duration=2.0,
                    num_speakers=2,
                )

            def extract_speaker_audio(self, audio_path, segments, speaker_id, filtered_path):
                _write_wav(Path(filtered_path))
                return filtered_path

        fake_sd_module.SpeakerDiarizer = FakeSpeakerDiarizer
        monkeypatch.setitem(sys.modules, "auto_voice.audio.speaker_diarization", fake_sd_module)
        fake_filter_module = types.ModuleType("auto_voice.audio.training_filter")
        fake_filter_module.TrainingDataFilter = type("TrainingDataFilter", (), {})
        monkeypatch.setitem(sys.modules, "auto_voice.audio.training_filter", fake_filter_module)

        assert client_remaining.post("/api/v1/youtube/history", json={"url": "https://yt", "title": "Saved"}).status_code == 201
        assert client_remaining.get("/api/v1/youtube/history?limit=1").status_code == 200

        info_response = client_remaining.post("/api/v1/youtube/info", json={"url": "https://youtube.test/watch?v=1"})
        assert info_response.status_code == 200
        assert info_response.get_json()["video_id"] == "vid123"

        download_response = client_remaining.post(
            "/api/v1/youtube/download",
            json={
                "url": "https://youtube.test/watch?v=1",
                "format": "wav",
                "sample_rate": 22050,
                "run_diarization": True,
                "filter_to_main_artist": True,
            },
        )
        assert download_response.status_code == 200
        payload = download_response.get_json()
        assert payload["success"] is True
        assert payload["diarization_result"]["num_speakers"] == 2
        assert payload["filtered_audio_path"].endswith("_filtered.wav")

        items = client_remaining.get("/api/v1/youtube/history").get_json()
        item_id = items[0]["id"]
        assert client_remaining.delete(f"/api/v1/youtube/history/{item_id}").status_code == 204
        assert client_remaining.delete("/api/v1/youtube/history").status_code == 204

    def test_youtube_download_surfaces_filter_and_history_warnings(
        self, client_remaining, app_remaining, tmp_path, monkeypatch
    ):
        from auto_voice.web import api as web_api

        web_api.YOUTUBE_DOWNLOADER_AVAILABLE = True

        class FakeDownloader:
            def download(self, url, audio_format="wav", sample_rate=44100):
                audio_path = tmp_path / "download.wav"
                _write_wav(audio_path, sample_rate=sample_rate)
                return SimpleNamespace(
                    success=True,
                    audio_path=str(audio_path),
                    title="Video",
                    duration=180.0,
                    main_artist="Artist",
                    featured_artists=["Feat"],
                    is_cover=False,
                    original_artist=None,
                    song_title="Song",
                    thumbnail_url=None,
                    video_id="vid999",
                    error=None,
                )

        monkeypatch.setattr(web_api, "get_youtube_downloader", lambda: FakeDownloader())

        fake_sd_module = types.ModuleType("auto_voice.audio.speaker_diarization")

        class FakeSpeakerDiarizer:
            def diarize(self, audio_path):
                return _FakeDiarizationResult(
                    segments=[
                        _FakeSegment(0.0, 1.0, "SPEAKER_00"),
                        _FakeSegment(1.0, 2.0, "SPEAKER_01"),
                    ],
                    audio_duration=2.0,
                    num_speakers=2,
                )

            def extract_speaker_audio(self, audio_path, segments, speaker_id, filtered_path):
                raise RuntimeError("filter failed")

        fake_sd_module.SpeakerDiarizer = FakeSpeakerDiarizer
        monkeypatch.setitem(sys.modules, "auto_voice.audio.speaker_diarization", fake_sd_module)
        fake_filter_module = types.ModuleType("auto_voice.audio.training_filter")
        fake_filter_module.TrainingDataFilter = type("TrainingDataFilter", (), {})
        monkeypatch.setitem(sys.modules, "auto_voice.audio.training_filter", fake_filter_module)
        monkeypatch.setattr(
            app_remaining.state_store,
            "save_youtube_history_item",
            lambda item: (_ for _ in ()).throw(RuntimeError("history offline")),
        )

        response = client_remaining.post(
            "/api/v1/youtube/download",
            json={
                "url": "https://youtube.test/watch?v=warn",
                "format": "wav",
                "sample_rate": 22050,
                "run_diarization": True,
                "filter_to_main_artist": True,
            },
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["success"] is True
        assert payload["filter_error"] == "filter failed"

    def test_youtube_download_surfaces_diarization_warning(
        self, client_remaining, app_remaining, tmp_path, monkeypatch
    ):
        from auto_voice.web import api as web_api

        web_api.YOUTUBE_DOWNLOADER_AVAILABLE = True

        class FakeDownloader:
            def download(self, url, audio_format="wav", sample_rate=44100):
                audio_path = tmp_path / "download.wav"
                _write_wav(audio_path, sample_rate=sample_rate)
                return SimpleNamespace(
                    success=True,
                    audio_path=str(audio_path),
                    title="Video",
                    duration=180.0,
                    main_artist="Artist",
                    featured_artists=[],
                    is_cover=False,
                    original_artist=None,
                    song_title="Song",
                    thumbnail_url=None,
                    video_id="vid998",
                    error=None,
                )

        monkeypatch.setattr(web_api, "get_youtube_downloader", lambda: FakeDownloader())

        fake_sd_module = types.ModuleType("auto_voice.audio.speaker_diarization")

        class BrokenSpeakerDiarizer:
            def diarize(self, audio_path):
                raise RuntimeError("diarizer unavailable")

        fake_sd_module.SpeakerDiarizer = BrokenSpeakerDiarizer
        monkeypatch.setitem(sys.modules, "auto_voice.audio.speaker_diarization", fake_sd_module)

        response = client_remaining.post(
            "/api/v1/youtube/download",
            json={
                "url": "https://youtube.test/watch?v=warn2",
                "format": "wav",
                "sample_rate": 22050,
                "run_diarization": True,
            },
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["success"] is True
        assert payload["diarization_error"] == "diarizer unavailable"

    def test_identify_retrain_analysis_artist_separation_and_quality_routes(
        self, client_remaining, app_remaining, tmp_path, monkeypatch
    ):
        from auto_voice.web import api as web_api

        profile_id = "00000000-0000-0000-0000-000000000321"
        _create_profile(app_remaining, profile_id=profile_id, clean_vocal_seconds=1900.0)
        sample_path = tmp_path / "target.wav"
        _write_wav(sample_path)
        _add_training_sample(app_remaining, profile_id, sample_path)

        _install_audit_module(
            monkeypatch,
            statuses=[
                _FakeProfileStatus(
                    profile_id=profile_id,
                    needs_retrain=True,
                    needs_training=False,
                    is_stale=True,
                    quality_ok=False,
                    sample_count=4,
                    issues=["stale_adapter"],
                    recommendations=["retrain"],
                )
            ],
            summary=_FakeAuditSummary(
                total_profiles=1,
                profiles_with_adapters=1,
                profiles_needing_training=0,
                profiles_needing_retrain=1,
                stale_adapters=1,
                low_quality_adapters=0,
                adapter_types={"unified": 1},
            ),
        )

        manager = _FakeTrainingJobManager()
        app_remaining._training_job_manager = manager

        identify_module = types.ModuleType("auto_voice.inference.voice_identifier")
        identify_module.get_voice_identifier = lambda: SimpleNamespace(
            identify_from_file=lambda path, threshold: SimpleNamespace(
                profile_id=profile_id,
                profile_name="Singer",
                similarity=0.93,
                is_match=True,
                all_similarities={profile_id: 0.93},
            )
        )
        monkeypatch.setitem(sys.modules, "auto_voice.inference.voice_identifier", identify_module)

        analysis_module = types.ModuleType("auto_voice.evaluation.conversion_quality_analyzer")

        class FakeMetrics:
            quality_score = 0.91

            def to_dict(self):
                return {"speaker_similarity": 0.95, "quality_score": 0.91}

        class FakeAnalyzer:
            def analyze(self, **kwargs):
                return SimpleNamespace(
                    methodology=kwargs["methodology"],
                    metrics=FakeMetrics(),
                    passes_thresholds=True,
                    threshold_failures=[],
                    recommendations=["good"],
                    timestamp="2026-04-17T00:00:00Z",
                )

            def compare_methodologies(self, **kwargs):
                analysis = SimpleNamespace(
                    metrics=FakeMetrics(),
                    passes_thresholds=True,
                    threshold_failures=[],
                )
                return SimpleNamespace(
                    best_methodology="quality_seedvc",
                    rankings=["quality_seedvc", "realtime"],
                    summary={"winner": "quality_seedvc"},
                    analyses={"quality_seedvc": analysis, "realtime": analysis},
                )

        analysis_module.ConversionQualityAnalyzer = FakeAnalyzer
        monkeypatch.setitem(sys.modules, "auto_voice.evaluation.conversion_quality_analyzer", analysis_module)

        quality_module = types.ModuleType("auto_voice.monitoring.quality_monitor")

        class FakeAlert:
            def __init__(self, name):
                self.name = name

            def to_dict(self):
                return {"name": self.name}

        class FakeQualityMonitor:
            def get_quality_history(self, profile_id, days=30):
                return {"profile_id": profile_id, "period_days": days, "total_metrics": 1, "metrics": []}

            def get_quality_summary(self, profile_id):
                return {"profile_id": profile_id, "status": "healthy", "rolling_averages": {"speaker_similarity": 0.94}}

            def detect_degradation(self, profile_id):
                return {"profile_id": profile_id, "degradation_detected": True, "alerts": [], "recommendation": "retrain"}

            def record_metric(self, **kwargs):
                return [FakeAlert("speaker_similarity_drop")]

            def get_all_profiles_status(self):
                return [{"profile_id": profile_id, "status": "degraded"}]

        quality_module.get_quality_monitor = lambda: FakeQualityMonitor()
        monkeypatch.setitem(sys.modules, "auto_voice.monitoring.quality_monitor", quality_module)

        separator_module = types.ModuleType("auto_voice.audio.multi_artist_separator")

        class FakeMultiArtistSeparator:
            def __init__(self, auto_create_profiles=True):
                self.auto_create_profiles = auto_create_profiles

            def separate_and_route(self, **kwargs):
                return SimpleNamespace(
                    artists={
                        profile_id: [
                            SimpleNamespace(
                                profile_name="Known Singer",
                                start=0.0,
                                end=1.5,
                                duration=1.5,
                                similarity=0.9,
                            )
                        ]
                    },
                    num_artists=1,
                    new_profiles_created=[],
                    total_duration=1.5,
                )

            def process_batch(self, audio_files, num_speakers=None):
                return {
                    "files_processed": len(audio_files),
                    "files_successful": len(audio_files),
                    "artists_found": 1,
                    "artist_summary": {
                        profile_id: {
                            "profile_name": "Known Singer",
                            "total_segments": 1,
                            "total_duration": 1.5,
                        }
                    },
                }

        separator_module.MultiArtistSeparator = FakeMultiArtistSeparator
        monkeypatch.setitem(sys.modules, "auto_voice.audio.multi_artist_separator", separator_module)

        monkeypatch.setattr(web_api.torchaudio, "load", lambda path: (torch.zeros((1, 22050)), 22050))
        web_api.YOUTUBE_DOWNLOADER_AVAILABLE = True
        monkeypatch.setattr(
            web_api,
            "YouTubeDownloader",
            lambda *args, **kwargs: SimpleNamespace(get_metadata=lambda url: {"title": "Meta"}),
        )

        identify_response = client_remaining.post(
            "/api/v1/audio/identify-speaker",
            data={"file": (_wav_bytes(), "voice.wav"), "threshold": "0.9"},
            content_type="multipart/form-data",
        )
        assert identify_response.status_code == 200
        assert identify_response.get_json()["is_match"] is True

        audit_response = client_remaining.get("/api/v1/loras/audit?format=summary")
        assert audit_response.status_code == 200
        assert audit_response.get_json()["profiles_needing_retrain"] == 1

        check_retrain_response = client_remaining.post(
            f"/api/v1/profiles/{profile_id}/check-retrain",
            json={"trigger": True},
        )
        assert check_retrain_response.status_code == 200
        assert check_retrain_response.get_json()["training_queued"] is True

        analyze_response = client_remaining.post(
            "/api/v1/convert/analyze",
            json={"source_audio": "src.wav", "converted_audio": "out.wav", "methodology": "quality_seedvc"},
        )
        assert analyze_response.status_code == 200
        assert analyze_response.get_json()["passes_thresholds"] is True

        compare_response = client_remaining.post(
            "/api/v1/convert/compare-methodologies",
            json={
                "source_audio": "src.wav",
                "target_profile_id": profile_id,
                "converted_outputs": {"quality_seedvc": "q.wav", "realtime": "r.wav"},
            },
        )
        assert compare_response.status_code == 200
        assert compare_response.get_json()["best_methodology"] == "quality_seedvc"

        retrain_response = client_remaining.post(
            f"/api/v1/loras/retrain/{profile_id}",
            json={"epochs": 25, "batch_size": 2, "learning_rate": 5e-5},
        )
        assert retrain_response.status_code == 200
        assert retrain_response.get_json()["profile_id"] == profile_id

        separate_response = client_remaining.post(
            "/api/v1/audio/separate-artists",
            data={"audio": (_wav_bytes(duration_seconds=2.0), "ensemble.wav"), "youtube_url": "https://youtube.test/watch?v=2"},
            content_type="multipart/form-data",
        )
        assert separate_response.status_code == 200
        assert separate_response.get_json()["num_artists"] == 1

        batch_response = client_remaining.post(
            "/api/v1/audio/batch-separate",
            data={"audio": [(_wav_bytes(), "a.wav"), (_wav_bytes(), "b.wav")]},
            content_type="multipart/form-data",
        )
        assert batch_response.status_code == 200
        assert batch_response.get_json()["files_processed"] == 2

        history_response = client_remaining.get(f"/api/v1/profiles/{profile_id}/quality-history?days=7")
        assert history_response.status_code == 200
        assert history_response.get_json()["period_days"] == 7

        status_response = client_remaining.get(f"/api/v1/profiles/{profile_id}/quality-status")
        assert status_response.status_code == 200
        assert status_response.get_json()["status"] == "healthy"

        degradation_response = client_remaining.post(
            f"/api/v1/profiles/{profile_id}/check-degradation",
            json={"auto_retrain": True},
        )
        assert degradation_response.status_code == 200
        assert degradation_response.get_json()["retrain_queued"] is True

        record_response = client_remaining.post(
            "/api/v1/quality/record",
            json={"profile_id": profile_id, "speaker_similarity": 0.7, "conversion_id": "conv-1"},
        )
        assert record_response.status_code == 200
        assert record_response.get_json()["alert_count"] == 1

        all_profiles_response = client_remaining.get("/api/v1/quality/all-profiles")
        assert all_profiles_response.status_code == 200
        assert all_profiles_response.get_json()["critical_count"] == 0
