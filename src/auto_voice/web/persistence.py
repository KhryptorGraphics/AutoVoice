"""Durable local state store for single-user MVP product data."""

from __future__ import annotations

import json
import os
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional


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
            "presets": self.base_dir / "presets.json",
            "conversion_history": self.base_dir / "conversion_history.json",
            "profile_checkpoints": self.base_dir / "profile_checkpoints.json",
            "youtube_history": self.base_dir / "youtube_history.json",
            "app_settings": self.base_dir / "app_settings.json",
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
        default_settings = {
            "preferred_pipeline": "quality",
            "last_updated": None,
        }
        return self._read("app_settings", default_settings)

    def update_app_settings(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        settings = self.get_app_settings()
        settings.update(deepcopy(updates))
        self._write("app_settings", settings)
        return settings
