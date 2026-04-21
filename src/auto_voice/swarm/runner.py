"""Deterministic repo-native swarm manifest runner."""

from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from auto_voice.swarm.memory import SwarmMemoryBackend


@dataclass(frozen=True)
class TaskSpec:
    id: str
    command: str
    deps: tuple[str, ...]
    retries: int
    description: str
    cwd: str
    artifacts: tuple[str, ...]
    lane: str
    role: str


def load_manifest(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Swarm manifest must be a mapping")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Swarm manifest must contain a non-empty tasks list")
    return payload


def task_specs(payload: Dict[str, Any]) -> List[TaskSpec]:
    specs: List[TaskSpec] = []
    seen: set[str] = set()
    for raw in payload["tasks"]:
        if not isinstance(raw, dict):
            raise ValueError("Each swarm task must be a mapping")
        task_id = str(raw.get("id") or "").strip()
        if not task_id:
            raise ValueError("Each swarm task requires a non-empty id")
        if task_id in seen:
            raise ValueError(f"Duplicate swarm task id: {task_id}")
        seen.add(task_id)
        deps = tuple(str(dep) for dep in raw.get("deps", []) or [])
        specs.append(
            TaskSpec(
                id=task_id,
                command=str(raw.get("command") or "").strip(),
                deps=deps,
                retries=max(0, int(raw.get("retries", 0))),
                description=str(raw.get("description") or task_id),
                cwd=str(raw.get("cwd") or "."),
                artifacts=tuple(str(item) for item in raw.get("artifacts", []) or []),
                lane=str(raw.get("lane") or "default").strip() or "default",
                role=str(raw.get("role") or "").strip(),
            )
        )
    for spec in specs:
        if not spec.command:
            raise ValueError(f"Task {spec.id} requires a command")
        for dep in spec.deps:
            if dep not in seen:
                raise ValueError(f"Task {spec.id} depends on unknown task {dep}")
    return specs


def topological_order(specs: Iterable[TaskSpec]) -> List[TaskSpec]:
    by_id = {spec.id: spec for spec in specs}
    indegree = {spec.id: 0 for spec in specs}
    followers: dict[str, list[str]] = {spec.id: [] for spec in specs}
    for spec in specs:
        for dep in spec.deps:
            indegree[spec.id] += 1
            followers[dep].append(spec.id)

    queue = deque(sorted(task_id for task_id, degree in indegree.items() if degree == 0))
    ordered: List[TaskSpec] = []
    while queue:
        task_id = queue.popleft()
        ordered.append(by_id[task_id])
        for follower in followers[task_id]:
            indegree[follower] -= 1
            if indegree[follower] == 0:
                queue.append(follower)

    if len(ordered) != len(by_id):
        raise ValueError("Swarm manifest contains a dependency cycle")
    return ordered


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _run_task(spec: TaskSpec, run_dir: Path, *, dry_run: bool, project_root: Path) -> Dict[str, Any]:
    env = os.environ.copy()
    env["AUTOVOICE_SWARM_RUN_DIR"] = str(run_dir)
    env.setdefault("AUTOVOICE_PROJECT_ROOT", str(project_root))
    env["AUTOVOICE_SWARM_TASK_ID"] = spec.id
    env["AUTOVOICE_SWARM_LANE"] = spec.lane
    env["AUTOVOICE_SWARM_ROLE"] = spec.role
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{project_root / 'src'}:{current_pythonpath}" if current_pythonpath else str(project_root / "src")
    )

    task_log = run_dir / "tasks" / f"{spec.id}.log"
    task_log.parent.mkdir(parents=True, exist_ok=True)

    attempts = 0
    started = time.time()
    last_error = ""
    status = "failed"

    while attempts <= spec.retries:
        attempts += 1
        if dry_run:
            task_log.write_text(f"[dry-run] {spec.command}\n", encoding="utf-8")
            status = "dry_run"
            last_error = ""
            break

        process = subprocess.run(
            spec.command,
            shell=True,
            cwd=project_root / spec.cwd,
            capture_output=True,
            text=True,
            env=env,
        )
        task_log.write_text(
            process.stdout + ("\n" if process.stdout and process.stderr else "") + process.stderr,
            encoding="utf-8",
        )
        if process.returncode == 0:
            status = "completed"
            last_error = ""
            break
        last_error = process.stderr.strip() or process.stdout.strip() or f"exit {process.returncode}"

    ended = time.time()
    return {
        "task_id": spec.id,
        "description": spec.description,
        "status": status,
        "attempts": attempts,
        "retries": spec.retries,
        "deps": list(spec.deps),
        "cwd": spec.cwd,
        "command": spec.command,
        "artifacts": list(spec.artifacts),
        "lane": spec.lane,
        "role": spec.role,
        "log_path": str(task_log),
        "started_at": started,
        "completed_at": ended,
        "duration_seconds": round(max(0.0, ended - started), 3),
        "error": last_error or None,
    }


def _context_sources(project_root: Path) -> Dict[str, Any]:
    config_path = project_root / "config" / "swarm_config.yaml"
    if not config_path.exists():
        return {}
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return dict(payload.get("runner", {}).get("context_sources", {}))


def execute_manifest(
    manifest_path: Path,
    *,
    run_id: str,
    dry_run: bool,
    run_root: Path,
    project_root: Path,
) -> int:
    payload = load_manifest(manifest_path)
    ordered_specs = topological_order(task_specs(payload))
    specs = {spec.id: spec for spec in ordered_specs}
    parallelism = max(1, int(payload.get("parallelism", 1)))

    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "manifest.snapshot.json", payload)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    memory = SwarmMemoryBackend.create(
        run_id=run_id,
        run_root=run_root,
        payload=payload,
        project_root=project_root,
    )

    ledger: Dict[str, Any] = {
        "run_id": run_id,
        "manifest_path": str(manifest_path),
        "manifest_name": payload.get("name"),
        "issue_id": payload.get("issue_id"),
        "dry_run": dry_run,
        "parallelism": parallelism,
        "context_sources": _context_sources(project_root),
        "memory": memory.describe(),
        "started_at": time.time(),
        "tasks": {},
        "waves": [],
    }
    memory.record_run_started(ledger)

    completed: set[str] = set()
    failed: set[str] = set()
    pending = dict(specs)

    while pending:
        skipped_this_wave: List[str] = []
        for task_id, spec in list(pending.items()):
            if any(dep in failed for dep in spec.deps):
                ledger["tasks"][task_id] = {
                    "task_id": task_id,
                    "description": spec.description,
                    "status": "skipped",
                    "deps": list(spec.deps),
                    "error": "dependency_failed",
                    "artifacts": list(spec.artifacts),
                    "lane": spec.lane,
                    "role": spec.role,
                }
                memory.record_task_started(task_id=task_id, description=spec.description, lane=spec.lane, role=spec.role)
                memory.record_task_status(task_id, "skipped", note="dependency_failed")
                failed.add(task_id)
                skipped_this_wave.append(task_id)
                del pending[task_id]
        if skipped_this_wave:
            ledger["waves"].append({"task_ids": skipped_this_wave, "status": "skipped"})
            _write_json(run_dir / "ledger.json", ledger)
            continue

        ready = [
            spec
            for spec in pending.values()
            if all(dep in completed for dep in spec.deps)
        ]
        ready.sort(key=lambda item: (item.lane, item.id))
        if not ready:
            raise ValueError("Swarm manifest execution stalled due to unsatisfied dependencies")

        batch = ready[:parallelism]
        wave_record = {
            "task_ids": [spec.id for spec in batch],
            "lane_counts": dict(sorted(Counter(spec.lane for spec in batch).items())),
        }
        ledger["waves"].append(wave_record)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures: Dict[concurrent.futures.Future[Dict[str, Any]], TaskSpec] = {}
            for spec in batch:
                memory.record_task_started(task_id=spec.id, description=spec.description, lane=spec.lane, role=spec.role)
                memory.record_agent_context(
                    spec.role or spec.id,
                    {
                        "task_id": spec.id,
                        "lane": spec.lane,
                        "description": spec.description,
                        "artifacts": list(spec.artifacts),
                    },
                )
                futures[executor.submit(_run_task, spec, run_dir, dry_run=dry_run, project_root=project_root)] = spec

            for future in concurrent.futures.as_completed(futures):
                spec = futures[future]
                task_record = future.result()
                ledger["tasks"][spec.id] = task_record
                if task_record["status"] in {"completed", "dry_run"}:
                    completed.add(spec.id)
                    memory.record_task_status(spec.id, "completed", note=task_record["status"])
                else:
                    failed.add(spec.id)
                    memory.record_task_status(spec.id, "failed", note=task_record.get("error") or "failed")
                del pending[spec.id]
                _write_json(run_dir / "ledger.json", ledger)

    finished = time.time()
    completion = {
        "run_id": run_id,
        "manifest_path": str(manifest_path),
        "status": "completed" if not failed else "failed",
        "dry_run": dry_run,
        "started_at": ledger["started_at"],
        "completed_at": finished,
        "duration_seconds": round(max(0.0, finished - ledger["started_at"]), 3),
        "completed_tasks": sorted(completed),
        "failed_tasks": sorted(failed),
        "task_count": len(specs),
        "memory": memory.describe(),
    }
    _write_json(run_dir / "ledger.json", ledger)
    _write_json(run_dir / "completion.json", completion)
    memory.record_run_finished(completion)
    print(json.dumps(completion, indent=2))
    return 0 if completion["status"] == "completed" else 1


def print_status(run_id: str, *, run_root: Path) -> int:
    completion_path = run_root / run_id / "completion.json"
    if not completion_path.exists():
        print(f"Run not found: {run_id}")
        return 1
    print(completion_path.read_text(encoding="utf-8"))
    return 0
