"""Swarm memory integration with optional MemKraft backing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from memkraft import MemKraft
except ImportError:  # pragma: no cover - optional dependency
    MemKraft = None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


@dataclass
class SwarmMemoryBackend:
    backend: str
    available: bool
    base_dir: Path
    channel_id: str
    _memkraft: Optional[Any] = None

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        run_root: Path,
        payload: Dict[str, Any],
        project_root: Path,
    ) -> "SwarmMemoryBackend":
        base_dir = run_root.parent / "swarm_memory"
        channel_id = f"autovoice-{run_id}"

        if MemKraft is None:
            fallback = cls(
                backend="file_fallback",
                available=False,
                base_dir=base_dir,
                channel_id=channel_id,
            )
            fallback._write_fallback(
                channel_id,
                {
                    "run_id": run_id,
                    "manifest_name": payload.get("name"),
                    "project_root": str(project_root),
                    "backend": "file_fallback",
                    "reason": "memkraft_import_unavailable",
                },
            )
            return fallback

        memory = MemKraft(base_dir=str(base_dir))
        memory.init(force=False, verbose=False)
        backend = cls(
            backend="memkraft",
            available=True,
            base_dir=base_dir,
            channel_id=channel_id,
            _memkraft=memory,
        )
        backend.channel_save(
            {
                "run_id": run_id,
                "manifest_name": payload.get("name"),
                "issue_id": payload.get("issue_id"),
                "project_root": str(project_root),
            }
        )
        return backend

    def describe(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "available": self.available,
            "base_dir": str(self.base_dir),
            "channel_id": self.channel_id,
        }

    def _write_fallback(self, name: str, payload: Dict[str, Any]) -> None:
        _write_json(self.base_dir / "fallback" / f"{name}.json", payload)

    def channel_save(self, payload: Dict[str, Any]) -> None:
        if self._memkraft is not None:
            self._memkraft.channel_save(self.channel_id, payload)
            return
        self._write_fallback(self.channel_id, payload)

    def record_run_started(self, payload: Dict[str, Any]) -> None:
        self.channel_save(
            {
                "status": "running",
                "manifest_path": payload.get("manifest_path"),
                "manifest_name": payload.get("manifest_name"),
                "issue_id": payload.get("issue_id"),
                "context_sources": payload.get("context_sources", {}),
                "parallelism": payload.get("parallelism", 1),
            }
        )

    def record_task_started(self, *, task_id: str, description: str, lane: str, role: str) -> None:
        if self._memkraft is not None:
            self._memkraft.task_start(task_id, description, channel_id=self.channel_id, agent=role or lane or "task")
            return
        self._write_fallback(
            f"task-{task_id}",
            {
                "task_id": task_id,
                "description": description,
                "lane": lane,
                "role": role,
                "status": "active",
            },
        )

    def record_task_status(self, task_id: str, status: str, *, note: str = "") -> None:
        if self._memkraft is not None:
            if status == "completed":
                self._memkraft.task_complete(task_id, note)
            else:
                self._memkraft.task_update(task_id, status, note)
            return
        self._write_fallback(
            f"task-{task_id}",
            {
                "task_id": task_id,
                "status": status,
                "note": note,
            },
        )

    def record_agent_context(self, agent_id: str, payload: Dict[str, Any]) -> None:
        if self._memkraft is not None:
            self._memkraft.agent_save(agent_id, payload)
            return
        self._write_fallback(f"agent-{agent_id}", payload)

    def record_run_finished(self, payload: Dict[str, Any]) -> None:
        self.channel_save(
            {
                "status": payload.get("status"),
                "completed_at": payload.get("completed_at"),
                "duration_seconds": payload.get("duration_seconds"),
                "completed_tasks": payload.get("completed_tasks", []),
                "failed_tasks": payload.get("failed_tasks", []),
            }
        )
