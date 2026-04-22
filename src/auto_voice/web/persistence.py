"""Durable local state store for single-user MVP product data."""

from __future__ import annotations

import json
import os
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from auto_voice.runtime_contract import (
    CANONICAL_LIVE_PIPELINE,
    CANONICAL_OFFLINE_PIPELINE,
    LIVE_PIPELINES,
    OFFLINE_PIPELINES,
)


DEFAULT_AUDIO_ROUTER_CONFIG: Dict[str, Any] = {
    "speaker_gain": 1.0,
    "headphone_gain": 1.0,
    "voice_gain": 1.0,
    "instrumental_gain": 0.8,
    "speaker_enabled": True,
    "headphone_enabled": True,
    "speaker_device": None,
    "headphone_device": None,
    "sample_rate": 24000,
}

DEFAULT_DEVICE_CONFIG: Dict[str, Any] = {
    "input_device_id": None,
    "output_device_id": None,
    "sample_rate": 22050,
}

DEFAULT_SEPARATION_CONFIG: Dict[str, Any] = {
    "model": "htdemucs",
    "stems": ["vocals"],
    "overlap": 0.25,
    "segment_length": None,
    "shifts": 1,
    "device": "cpu",
}

DEFAULT_PITCH_CONFIG: Dict[str, Any] = {
    "method": "rmvpe",
    "hop_length": 160,
    "f0_min": 50,
    "f0_max": 1100,
    "threshold": 0.3,
    "use_gpu": False,
    "device": "cpu",
}


def _normalize_app_settings(raw_settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize app settings while preserving one-release legacy compatibility."""
    settings = deepcopy(raw_settings or {})
    legacy_pipeline = settings.get("preferred_pipeline")
    offline_pipeline = settings.get("preferred_offline_pipeline")
    live_pipeline = settings.get("preferred_live_pipeline")

    if offline_pipeline not in OFFLINE_PIPELINES:
        if legacy_pipeline == "realtime":
            offline_pipeline = "realtime"
        else:
            offline_pipeline = CANONICAL_OFFLINE_PIPELINE

    if live_pipeline not in LIVE_PIPELINES:
        if legacy_pipeline == "realtime":
            live_pipeline = "realtime"
        else:
            live_pipeline = CANONICAL_LIVE_PIPELINE

    settings["preferred_offline_pipeline"] = offline_pipeline
    settings["preferred_live_pipeline"] = live_pipeline
    settings["preferred_pipeline"] = (
        "realtime"
        if offline_pipeline == "realtime" and live_pipeline == "realtime"
        else "quality"
    )
    settings.setdefault("last_updated", None)
    return settings


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return bool(value)


def _coerce_float(value: Any, default: float, *, minimum: float = 0.0, maximum: float = 2.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not a valid integer field")
    return int(value)


def _normalize_audio_router_config(raw_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    config = deepcopy(DEFAULT_AUDIO_ROUTER_CONFIG)
    incoming = deepcopy(raw_config or {})

    for key in ("speaker_gain", "headphone_gain", "voice_gain", "instrumental_gain"):
        if key in incoming:
            config[key] = _coerce_float(incoming.get(key), config[key])

    for key in ("speaker_enabled", "headphone_enabled"):
        if key in incoming:
            config[key] = _coerce_bool(incoming.get(key), config[key])

    for key in ("speaker_device", "headphone_device"):
        try:
            if key in incoming:
                config[key] = _coerce_optional_int(incoming.get(key))
        except (TypeError, ValueError):
            config[key] = DEFAULT_AUDIO_ROUTER_CONFIG[key]

    try:
        if "sample_rate" in incoming:
            sample_rate = int(incoming.get("sample_rate"))
            if sample_rate > 0:
                config["sample_rate"] = sample_rate
    except (TypeError, ValueError):
        pass

    return config


def _normalize_device_config(raw_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    config = deepcopy(DEFAULT_DEVICE_CONFIG)
    incoming = deepcopy(raw_config or {})

    for key in ("input_device_id", "output_device_id"):
        value = incoming.get(key, config[key])
        config[key] = None if value in (None, "") else str(value)

    try:
        if "sample_rate" in incoming:
            sample_rate = int(incoming.get("sample_rate"))
            if sample_rate > 0:
                config["sample_rate"] = sample_rate
    except (TypeError, ValueError):
        pass

    return config


def _normalize_separation_config(raw_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    config = deepcopy(DEFAULT_SEPARATION_CONFIG)
    incoming = deepcopy(raw_config or {})

    if "model" in incoming and incoming["model"]:
        config["model"] = str(incoming["model"])
    if "stems" in incoming and isinstance(incoming["stems"], list):
        config["stems"] = [str(stem) for stem in incoming["stems"] if str(stem).strip()]

    for key in ("overlap",):
        if key in incoming:
            config[key] = _coerce_float(incoming.get(key), config[key], minimum=0.0, maximum=1.0)

    if "segment_length" in incoming:
        value = incoming.get("segment_length")
        if value in (None, ""):
            config["segment_length"] = None
        else:
            try:
                segment_length = float(value)
                config["segment_length"] = segment_length if segment_length > 0 else None
            except (TypeError, ValueError):
                config["segment_length"] = DEFAULT_SEPARATION_CONFIG["segment_length"]

    if "shifts" in incoming:
        try:
            shifts = int(incoming.get("shifts"))
            config["shifts"] = shifts if shifts > 0 else DEFAULT_SEPARATION_CONFIG["shifts"]
        except (TypeError, ValueError):
            config["shifts"] = DEFAULT_SEPARATION_CONFIG["shifts"]

    if "device" in incoming and incoming["device"] in {"cpu", "cuda"}:
        config["device"] = incoming["device"]

    return config


def _normalize_pitch_config(raw_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    config = deepcopy(DEFAULT_PITCH_CONFIG)
    incoming = deepcopy(raw_config or {})

    if "method" in incoming and incoming["method"]:
        config["method"] = str(incoming["method"])

    for key in ("hop_length", "f0_min", "f0_max"):
        if key in incoming:
            try:
                value = int(incoming.get(key))
                if value > 0:
                    config[key] = value
            except (TypeError, ValueError):
                config[key] = DEFAULT_PITCH_CONFIG[key]

    if "threshold" in incoming:
        config["threshold"] = _coerce_float(incoming.get("threshold"), config["threshold"], minimum=0.0, maximum=1.0)

    if "use_gpu" in incoming:
        config["use_gpu"] = _coerce_bool(incoming.get("use_gpu"), config["use_gpu"])
        config["device"] = "cuda" if config["use_gpu"] else "cpu"
    elif "device" in incoming and incoming["device"] in {"cpu", "cuda"}:
        config["device"] = incoming["device"]
        config["use_gpu"] = incoming["device"] == "cuda"

    return config


def _normalize_karaoke_session(snapshot: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not snapshot:
        return None

    session = deepcopy(snapshot)
    session_id = str(session.get("session_id") or "").strip()
    if not session_id:
        return None

    session["session_id"] = session_id
    session["song_id"] = str(session.get("song_id") or "").strip()
    session["vocals_path"] = str(session.get("vocals_path") or "")
    session["instrumental_path"] = str(session.get("instrumental_path") or "")
    session["requested_pipeline"] = str(session.get("requested_pipeline") or CANONICAL_LIVE_PIPELINE).strip().lower()
    session["resolved_pipeline"] = str(session.get("resolved_pipeline") or session["requested_pipeline"]).strip().lower()
    session["runtime_backend"] = session.get("runtime_backend")
    session["voice_model_id"] = session.get("voice_model_id")
    session["source_voice_model_id"] = session.get("source_voice_model_id")
    session["target_profile_id"] = session.get("target_profile_id")
    session["active_model_type"] = session.get("active_model_type")
    session["sample_rate"] = int(session.get("sample_rate") or DEFAULT_AUDIO_ROUTER_CONFIG["sample_rate"])
    session["is_active"] = _coerce_bool(session.get("is_active"), False)
    session["collect_samples"] = _coerce_bool(session.get("collect_samples"), False)
    session["sample_collection_enabled"] = _coerce_bool(session.get("sample_collection_enabled"), False)
    session["audio_router_targets"] = _normalize_audio_router_config(session.get("audio_router_targets", {}))
    session["started_at"] = session.get("started_at")
    session["last_activity"] = session.get("last_activity")
    session["stats"] = deepcopy(session.get("stats") or {})

    speaker_embedding = session.get("speaker_embedding")
    if isinstance(speaker_embedding, list):
        session["speaker_embedding"] = list(speaker_embedding)
    else:
        session["speaker_embedding"] = None

    return session


def resolve_data_dir(explicit_data_dir: Optional[str] = None) -> Path:
    """Resolve the application data directory."""
    raw_data_dir = explicit_data_dir or os.environ.get("DATA_DIR") or "data"
    return Path(raw_data_dir)


class AppStateStore:
    """Small JSON-backed store for durable product metadata."""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = resolve_data_dir(data_dir)
        self.base_dir = self.data_dir / "app_state"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._files = {
            "training_jobs": self.base_dir / "training_jobs.json",
            "background_jobs": self.base_dir / "background_jobs.json",
            "presets": self.base_dir / "presets.json",
            "conversion_history": self.base_dir / "conversion_history.json",
            "conversion_workflows": self.base_dir / "conversion_workflows.json",
            "profile_checkpoints": self.base_dir / "profile_checkpoints.json",
            "youtube_history": self.base_dir / "youtube_history.json",
            "app_settings": self.base_dir / "app_settings.json",
            "loaded_models": self.base_dir / "loaded_models.json",
            "separation_config": self.base_dir / "separation_config.json",
            "pitch_config": self.base_dir / "pitch_config.json",
            "audio_router_config": self.base_dir / "audio_router_config.json",
            "device_config": self.base_dir / "device_config.json",
            "karaoke_sessions": self.base_dir / "karaoke_sessions.json",
        }

    def _read(self, name: str, default: Any) -> Any:
        path = self._files[name]
        with self._lock:
            if not path.exists():
                return deepcopy(default)
            with path.open("r", encoding="utf-8") as handle:
                try:
                    return json.load(handle)
                except json.JSONDecodeError:
                    return deepcopy(default)

    def _write(self, name: str, payload: Any) -> None:
        path = self._files[name]
        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        with self._lock:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
            tmp_path.replace(path)

    def list_training_jobs(self, profile_id: Optional[str] = None) -> List[Dict[str, Any]]:
        jobs = list(self._read("training_jobs", {}).values())
        if profile_id:
            jobs = [job for job in jobs if job.get("profile_id") == profile_id]
        jobs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return jobs

    def get_training_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._read("training_jobs", {}).get(job_id)

    def save_training_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        jobs = self._read("training_jobs", {})
        jobs[job["job_id"]] = deepcopy(job)
        self._write("training_jobs", jobs)
        return job

    def list_background_jobs(self, job_type: Optional[str] = None) -> List[Dict[str, Any]]:
        jobs = list(self._read("background_jobs", {}).values())
        if job_type:
            jobs = [job for job in jobs if job.get("job_type") == job_type]
        jobs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return jobs

    def get_background_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._read("background_jobs", {}).get(job_id)

    def save_background_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        jobs = self._read("background_jobs", {})
        jobs[job["job_id"]] = deepcopy(job)
        self._write("background_jobs", jobs)
        return job

    def list_presets(self) -> List[Dict[str, Any]]:
        presets = list(self._read("presets", {}).values())
        presets.sort(key=lambda item: item.get("updated_at") or item.get("created_at", ""), reverse=True)
        return presets

    def get_preset(self, preset_id: str) -> Optional[Dict[str, Any]]:
        return self._read("presets", {}).get(preset_id)

    def save_preset(self, preset: Dict[str, Any]) -> Dict[str, Any]:
        presets = self._read("presets", {})
        presets[preset["id"]] = deepcopy(preset)
        self._write("presets", presets)
        return preset

    def delete_preset(self, preset_id: str) -> bool:
        presets = self._read("presets", {})
        if preset_id not in presets:
            return False
        del presets[preset_id]
        self._write("presets", presets)
        return True

    def list_conversion_history(self, profile_id: Optional[str] = None) -> List[Dict[str, Any]]:
        records = list(self._read("conversion_history", {}).values())
        if profile_id:
            records = [record for record in records if record.get("profile_id") == profile_id]
        records.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return records

    def get_conversion_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        return self._read("conversion_history", {}).get(record_id)

    def save_conversion_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        records = self._read("conversion_history", {})
        records[record["id"]] = deepcopy(record)
        self._write("conversion_history", records)
        return record

    def delete_conversion_record(self, record_id: str) -> bool:
        records = self._read("conversion_history", {})
        if record_id not in records:
            return False
        del records[record_id]
        self._write("conversion_history", records)
        return True

    def list_conversion_workflows(
        self,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        workflows = list(self._read("conversion_workflows", {}).values())
        if status:
            workflows = [workflow for workflow in workflows if workflow.get("status") == status]
        workflows.sort(key=lambda item: item.get("updated_at") or item.get("created_at", ""), reverse=True)
        return workflows

    def get_conversion_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        return self._read("conversion_workflows", {}).get(workflow_id)

    def save_conversion_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        workflows = self._read("conversion_workflows", {})
        workflows[workflow["workflow_id"]] = deepcopy(workflow)
        self._write("conversion_workflows", workflows)
        return workflow

    def delete_conversion_workflow(self, workflow_id: str) -> bool:
        workflows = self._read("conversion_workflows", {})
        if workflow_id not in workflows:
            return False
        del workflows[workflow_id]
        self._write("conversion_workflows", workflows)
        return True

    def list_checkpoints(self, profile_id: str) -> List[Dict[str, Any]]:
        checkpoints = self._read("profile_checkpoints", {})
        profile_checkpoints = list(checkpoints.get(profile_id, {}).values())
        profile_checkpoints.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return profile_checkpoints

    def get_checkpoint(self, profile_id: str, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        checkpoints = self._read("profile_checkpoints", {})
        return checkpoints.get(profile_id, {}).get(checkpoint_id)

    def save_checkpoint(self, profile_id: str, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        checkpoints = self._read("profile_checkpoints", {})
        profile_checkpoints = checkpoints.setdefault(profile_id, {})
        profile_checkpoints[checkpoint["id"]] = deepcopy(checkpoint)
        self._write("profile_checkpoints", checkpoints)
        return checkpoint

    def delete_checkpoint(self, profile_id: str, checkpoint_id: str) -> bool:
        checkpoints = self._read("profile_checkpoints", {})
        profile_checkpoints = checkpoints.get(profile_id, {})
        if checkpoint_id not in profile_checkpoints:
            return False
        del profile_checkpoints[checkpoint_id]
        if not profile_checkpoints:
            checkpoints.pop(profile_id, None)
        self._write("profile_checkpoints", checkpoints)
        return True

    def list_youtube_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        items = self._read("youtube_history", [])
        items.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
        if limit is not None:
            return items[:limit]
        return items

    def save_youtube_history_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        items = self._read("youtube_history", [])
        items = [existing for existing in items if existing.get("id") != item["id"]]
        items.insert(0, deepcopy(item))
        self._write("youtube_history", items[:100])
        return item

    def delete_youtube_history_item(self, item_id: str) -> bool:
        items = self._read("youtube_history", [])
        filtered = [item for item in items if item.get("id") != item_id]
        if len(filtered) == len(items):
            return False
        self._write("youtube_history", filtered)
        return True

    def clear_youtube_history(self) -> None:
        self._write("youtube_history", [])

    def get_app_settings(self) -> Dict[str, Any]:
        return _normalize_app_settings(self._read("app_settings", {}))

    def update_app_settings(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        settings = self.get_app_settings()
        settings.update(deepcopy(updates))
        normalized = _normalize_app_settings(settings)
        self._write("app_settings", normalized)
        return normalized

    def list_loaded_models(self) -> List[Dict[str, Any]]:
        models = list(self._read("loaded_models", {}).values())
        models.sort(key=lambda item: (item.get("model_type", ""), item.get("loaded_at", "")))
        return models

    def get_loaded_model(self, model_type: str) -> Optional[Dict[str, Any]]:
        return self._read("loaded_models", {}).get(model_type)

    def save_loaded_model(self, model_type: str, model_info: Dict[str, Any]) -> Dict[str, Any]:
        models = self._read("loaded_models", {})
        models[str(model_type)] = deepcopy(model_info)
        self._write("loaded_models", models)
        return model_info

    def delete_loaded_model(self, model_type: str) -> bool:
        models = self._read("loaded_models", {})
        if model_type not in models:
            return False
        del models[model_type]
        self._write("loaded_models", models)
        return True

    def clear_loaded_models(self) -> None:
        self._write("loaded_models", {})

    def get_separation_config(self) -> Dict[str, Any]:
        return _normalize_separation_config(self._read("separation_config", DEFAULT_SEPARATION_CONFIG))

    def update_separation_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        config = self.get_separation_config()
        config.update(deepcopy(updates))
        normalized = _normalize_separation_config(config)
        self._write("separation_config", normalized)
        return normalized

    def get_pitch_config(self) -> Dict[str, Any]:
        return _normalize_pitch_config(self._read("pitch_config", DEFAULT_PITCH_CONFIG))

    def update_pitch_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        config = self.get_pitch_config()
        config.update(deepcopy(updates))
        if "device" in updates and "use_gpu" not in updates:
            config["use_gpu"] = updates.get("device") == "cuda"
        normalized = _normalize_pitch_config(config)
        self._write("pitch_config", normalized)
        return normalized

    def get_audio_router_config(self) -> Dict[str, Any]:
        return _normalize_audio_router_config(self._read("audio_router_config", DEFAULT_AUDIO_ROUTER_CONFIG))

    def update_audio_router_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        config = self.get_audio_router_config()
        config.update(deepcopy(updates))
        normalized = _normalize_audio_router_config(config)
        self._write("audio_router_config", normalized)
        return normalized

    def get_device_config(self) -> Dict[str, Any]:
        return _normalize_device_config(self._read("device_config", DEFAULT_DEVICE_CONFIG))

    def update_device_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        config = self.get_device_config()
        config.update(deepcopy(updates))
        normalized = _normalize_device_config(config)
        self._write("device_config", normalized)
        return normalized

    def list_karaoke_sessions(self) -> List[Dict[str, Any]]:
        sessions = [
            normalized
            for normalized in (
                _normalize_karaoke_session(item)
                for item in self._read("karaoke_sessions", {}).values()
            )
            if normalized is not None
        ]
        sessions.sort(
            key=lambda item: item.get("last_activity") or item.get("started_at") or 0,
            reverse=True,
        )
        return sessions

    def get_karaoke_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return _normalize_karaoke_session(self._read("karaoke_sessions", {}).get(session_id))

    def save_karaoke_session(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        normalized = _normalize_karaoke_session(snapshot)
        if normalized is None:
            raise ValueError("karaoke session snapshot requires a session_id")
        sessions = self._read("karaoke_sessions", {})
        sessions[normalized["session_id"]] = normalized
        self._write("karaoke_sessions", sessions)
        return normalized

    def delete_karaoke_session(self, session_id: str) -> bool:
        sessions = self._read("karaoke_sessions", {})
        if session_id not in sessions:
            return False
        del sessions[session_id]
        self._write("karaoke_sessions", sessions)
        return True
