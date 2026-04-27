"""Deterministic repo-native swarm manifest runner."""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import subprocess
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

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


@dataclass(frozen=True)
class RunContext:
    run_id: str
    parent_run_id: str
    manifest_name: str
    data_dir: Path
    run_dir: Path
    channel_id: str
    taxonomy: Dict[str, str]
    required_memory_writes: tuple[str, ...]


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


def _slug(value: str) -> str:
    lowered = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return lowered.strip("-") or "unknown"


def _runner_config(project_root: Path) -> Dict[str, Any]:
    config_path = project_root / "config" / "swarm_config.yaml"
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def _context_sources(project_root: Path) -> Dict[str, Any]:
    payload = _runner_config(project_root)
    return dict(payload.get("runner", {}).get("context_sources", {}))


def _default_required_memory_writes(project_root: Path) -> tuple[str, ...]:
    payload = _runner_config(project_root)
    writes = payload.get("runner", {}).get("required_memory_writes", [])
    if not writes:
        return (
            "sprint_brief",
            "assumptions_and_decisions",
            "findings",
            "artifacts_produced",
            "test_outcomes",
            "handoff_summary",
        )
    return tuple(str(item) for item in writes)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _control_path(run_dir: Path) -> Path:
    return run_dir / "control.json"


def _read_control(run_dir: Path) -> Dict[str, Any]:
    path = _control_path(run_dir)
    if not path.exists():
        return {}
    return _read_json(path)


def request_cancel(*, run_id: str, run_root: Path, reason: str | None = None) -> Dict[str, Any]:
    run_dir = run_root / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_id}")

    completion_path = run_dir / "completion.json"
    if completion_path.exists():
        completion = _read_json(completion_path)
        return {
            "run_id": run_id,
            "cancel_requested": False,
            "reason": "run_already_terminal",
            "terminal_status": completion.get("status"),
        }

    payload = {
        "run_id": run_id,
        "cancel_requested": True,
        "reason": reason or "cancel_requested",
        "requested_at": time.time(),
    }
    _write_json(_control_path(run_dir), payload)
    return payload


def _prepare_followup_manifest(
    *,
    source_run_dir: Path,
    new_run_dir: Path,
    selected_task_ids: Sequence[str],
    mode: str,
) -> Path:
    snapshot_path = source_run_dir / "manifest.snapshot.json"
    payload = _read_json(snapshot_path)
    selected = set(selected_task_ids)
    tasks = []
    for task in payload.get("tasks", []):
        task_id = str(task.get("id") or "")
        if task_id not in selected:
            continue
        next_task = dict(task)
        next_task["deps"] = [dep for dep in (task.get("deps", []) or []) if dep in selected]
        tasks.append(next_task)
    payload["tasks"] = tasks
    new_run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = new_run_dir / f"{mode}.manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def prepare_resume_run(
    *,
    source_run_id: str,
    new_run_id: str,
    run_root: Path,
    mode: str,
) -> Dict[str, Any]:
    source_run_dir = run_root / source_run_id
    ledger_path = source_run_dir / "ledger.json"
    if not ledger_path.exists():
        raise FileNotFoundError(f"Run ledger not found: {source_run_id}")

    ledger = _read_json(ledger_path)
    tasks = ledger.get("tasks", {})
    if not isinstance(tasks, dict):
        raise ValueError(f"Run ledger is malformed: {source_run_id}")
    snapshot = _read_json(source_run_dir / "manifest.snapshot.json")
    manifest_task_ids = [
        str(task.get("id") or "")
        for task in snapshot.get("tasks", [])
        if isinstance(task, dict) and str(task.get("id") or "")
    ]

    completed = sorted(
        task_id
        for task_id, task in tasks.items()
        if isinstance(task, dict) and task.get("status") in {"completed", "dry_run"}
    )
    rerun = []
    for task_id in manifest_task_ids:
        task = tasks.get(task_id, {})
        status = task.get("status") if isinstance(task, dict) else None
        if mode == "resume":
            if status not in {"completed", "dry_run"}:
                rerun.append(task_id)
        elif mode == "retry":
            if status in {"failed", "skipped", "cancelled"}:
                rerun.append(task_id)
        else:
            raise ValueError(f"Unsupported follow-up mode: {mode}")

    if not rerun:
        raise ValueError(f"No tasks available to {mode} for run {source_run_id}")

    manifest_path = _prepare_followup_manifest(
        source_run_dir=source_run_dir,
        new_run_dir=run_root / new_run_id,
        selected_task_ids=rerun,
        mode=mode,
    )
    return {
        "manifest_path": str(manifest_path),
        "completed_tasks": completed,
        "rerun_tasks": sorted(rerun),
        "parent_run_id": source_run_id,
        "run_id": new_run_id,
    }


def build_run_context(
    payload: Dict[str, Any],
    *,
    run_id: str,
    run_root: Path,
    project_root: Path,
    parent_run_id: str | None = None,
) -> RunContext:
    runner_config = _runner_config(project_root).get("runner", {})
    taxonomy = {
        "program": str(payload.get("program") or runner_config.get("program") or project_root.name),
        "phase": str(payload.get("phase") or runner_config.get("phase") or "default"),
        "sprint": str(payload.get("sprint") or runner_config.get("sprint") or "ad-hoc"),
        "lane": str(payload.get("lane") or runner_config.get("lane") or "run"),
    }
    channel_id = "-".join(
        [
            "autovoice",
            _slug(taxonomy["program"]),
            _slug(taxonomy["phase"]),
            _slug(taxonomy["sprint"]),
            _slug(run_id),
        ]
    )
    return RunContext(
        run_id=run_id,
        parent_run_id=(parent_run_id or "").strip(),
        manifest_name=str(payload.get("name") or run_id),
        data_dir=run_root.parent,
        run_dir=run_root / run_id,
        channel_id=channel_id,
        taxonomy=taxonomy,
        required_memory_writes=tuple(
            str(item)
            for item in payload.get("required_memory_writes", _default_required_memory_writes(project_root))
        ),
    )


def _run_task(spec: TaskSpec, context: RunContext, *, dry_run: bool, project_root: Path) -> Dict[str, Any]:
    env = os.environ.copy()
    env["AUTOVOICE_SWARM_RUN_DIR"] = str(context.run_dir)
    env["AUTOVOICE_SWARM_DATA_DIR"] = str(context.data_dir)
    env.setdefault("AUTOVOICE_PROJECT_ROOT", str(project_root))
    env["AUTOVOICE_SWARM_RUN_ID"] = context.run_id
    env["AUTOVOICE_SWARM_PARENT_RUN_ID"] = context.parent_run_id
    env["AUTOVOICE_SWARM_CHANNEL_ID"] = context.channel_id
    env["AUTOVOICE_SWARM_TASK_ID"] = spec.id
    env["AUTOVOICE_SWARM_LANE"] = spec.lane
    env["AUTOVOICE_SWARM_ROLE"] = spec.role
    env["AUTOVOICE_SWARM_TASK_KEY"] = f"{context.run_id}:{spec.lane}:{spec.id}"
    env["AUTOVOICE_SWARM_LANE_KEY"] = f"{context.run_id}:{spec.lane}"
    env["AUTOVOICE_SWARM_AGENT_KEY"] = f"{context.run_id}:{spec.lane}:{spec.role or spec.id}"
    env["AUTOVOICE_SWARM_ARTIFACT_KEY"] = f"{context.run_id}:{spec.lane}:{spec.id}:artifact"
    task_artifact_root = context.run_dir / "artifacts" / spec.lane / spec.id
    task_artifact_root.mkdir(parents=True, exist_ok=True)
    env["AUTOVOICE_SWARM_ARTIFACT_ROOT"] = str(task_artifact_root)
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{project_root / 'src'}:{current_pythonpath}" if current_pythonpath else str(project_root / "src")
    )

    task_log = context.run_dir / "tasks" / f"{spec.id}.log"
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
        "task_key": env["AUTOVOICE_SWARM_TASK_KEY"],
        "lane_key": env["AUTOVOICE_SWARM_LANE_KEY"],
        "agent_key": env["AUTOVOICE_SWARM_AGENT_KEY"],
        "artifact_key": env["AUTOVOICE_SWARM_ARTIFACT_KEY"],
        "artifact_root": env["AUTOVOICE_SWARM_ARTIFACT_ROOT"],
        "log_path": str(task_log),
        "started_at": started,
        "completed_at": ended,
        "duration_seconds": round(max(0.0, ended - started), 3),
        "error": last_error or None,
    }


def execute_manifest(
    manifest_path: Path,
    *,
    run_id: str,
    dry_run: bool,
    run_root: Path,
    project_root: Path,
    parent_run_id: str | None = None,
) -> int:
    payload = load_manifest(manifest_path)
    ordered_specs = topological_order(task_specs(payload))
    specs = {spec.id: spec for spec in ordered_specs}
    parallelism = max(1, int(payload.get("parallelism", 1)))
    context = build_run_context(
        payload,
        run_id=run_id,
        run_root=run_root,
        project_root=project_root,
        parent_run_id=parent_run_id,
    )

    run_dir = context.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "manifest.snapshot.json", payload)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    memory = SwarmMemoryBackend.create(
        run_id=run_id,
        run_root=run_root,
        payload=payload,
        project_root=project_root,
        parent_run_id=context.parent_run_id or None,
    )

    ledger: Dict[str, Any] = {
        "run_id": run_id,
        "parent_run_id": context.parent_run_id or None,
        "manifest_path": str(manifest_path),
        "manifest_name": context.manifest_name,
        "issue_id": payload.get("issue_id"),
        "dry_run": dry_run,
        "parallelism": parallelism,
        "data_dir": str(context.data_dir),
        "channel_id": context.channel_id,
        "taxonomy": context.taxonomy,
        "required_memory_writes": list(context.required_memory_writes),
        "context_sources": _context_sources(project_root),
        "memory": memory.describe(),
        "prior_memory": memory.read_prior_context(),
        "started_at": time.time(),
        "tasks": {},
        "waves": [],
    }
    memory.record_run_started(ledger)

    completed: set[str] = set()
    failed: set[str] = set()
    pending = dict(specs)
    control = _read_control(run_dir)

    while pending:
        control = _read_control(run_dir)
        if control.get("cancel_requested"):
            for task_id, spec in list(pending.items()):
                ledger["tasks"][task_id] = {
                    "task_id": task_id,
                    "description": spec.description,
                    "status": "cancelled",
                    "deps": list(spec.deps),
                    "error": control.get("reason") or "cancel_requested",
                    "artifacts": list(spec.artifacts),
                    "lane": spec.lane,
                    "role": spec.role,
                    "task_key": f"{context.run_id}:{spec.lane}:{spec.id}",
                    "lane_key": f"{context.run_id}:{spec.lane}",
                    "agent_key": f"{context.run_id}:{spec.lane}:{spec.role or spec.id}",
                    "artifact_key": f"{context.run_id}:{spec.lane}:{spec.id}:artifact",
                    "artifact_root": str(run_dir / "artifacts" / spec.lane / spec.id),
                }
                memory.record_task_started(
                    task_id=f"{context.run_id}:{spec.lane}:{spec.id}",
                    description=spec.description,
                    lane=spec.lane,
                    role=spec.role,
                )
                memory.record_task_status(
                    f"{context.run_id}:{spec.lane}:{spec.id}",
                    "cancelled",
                    note=control.get("reason") or "cancel_requested",
                )
                failed.add(task_id)
                del pending[task_id]
            ledger["waves"].append({"task_ids": [], "status": "cancelled"})
            _write_json(run_dir / "ledger.json", ledger)
            break

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
                    "task_key": f"{context.run_id}:{spec.lane}:{spec.id}",
                    "lane_key": f"{context.run_id}:{spec.lane}",
                    "agent_key": f"{context.run_id}:{spec.lane}:{spec.role or spec.id}",
                    "artifact_key": f"{context.run_id}:{spec.lane}:{spec.id}:artifact",
                    "artifact_root": str(run_dir / "artifacts" / spec.lane / spec.id),
                }
                memory.record_task_started(
                    task_id=f"{context.run_id}:{spec.lane}:{spec.id}",
                    description=spec.description,
                    lane=spec.lane,
                    role=spec.role,
                )
                memory.record_task_status(f"{context.run_id}:{spec.lane}:{spec.id}", "skipped", note="dependency_failed")
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
                task_key = f"{context.run_id}:{spec.lane}:{spec.id}"
                agent_key = f"{context.run_id}:{spec.lane}:{spec.role or spec.id}"
                memory.record_task_started(task_id=task_key, description=spec.description, lane=spec.lane, role=spec.role)
                memory.record_agent_context(
                    agent_key,
                    {
                        "run_id": context.run_id,
                        "parent_run_id": context.parent_run_id or None,
                        "task_id": spec.id,
                        "task_key": task_key,
                        "agent_key": agent_key,
                        "lane": spec.lane,
                        "description": spec.description,
                        "taxonomy": context.taxonomy,
                        "artifacts": list(spec.artifacts),
                    },
                )
                futures[executor.submit(_run_task, spec, context, dry_run=dry_run, project_root=project_root)] = spec

            for future in concurrent.futures.as_completed(futures):
                spec = futures[future]
                task_record = future.result()
                ledger["tasks"][spec.id] = task_record
                if task_record["status"] in {"completed", "dry_run"}:
                    completed.add(spec.id)
                    memory.record_task_status(task_record["task_key"], "completed", note=task_record["status"])
                else:
                    failed.add(spec.id)
                    memory.record_task_status(
                        task_record["task_key"],
                        "failed",
                        note=task_record.get("error") or "failed",
                    )
                del pending[spec.id]
                _write_json(run_dir / "ledger.json", ledger)
        control = _read_control(run_dir)

    finished = time.time()
    control = _read_control(run_dir)
    completion = {
        "run_id": run_id,
        "parent_run_id": context.parent_run_id or None,
        "manifest_path": str(manifest_path),
        "status": (
            "cancelled"
            if control.get("cancel_requested")
            else ("completed" if not failed else "failed")
        ),
        "dry_run": dry_run,
        "channel_id": context.channel_id,
        "taxonomy": context.taxonomy,
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
    run_dir = run_root / run_id
    completion_path = run_root / run_id / "completion.json"
    if not completion_path.exists():
        ledger_path = run_dir / "ledger.json"
        if not ledger_path.exists():
            print(f"Run not found: {run_id}")
            return 1
        ledger = _read_json(ledger_path)
        tasks = ledger.get("tasks", {})
        snapshot_path = run_dir / "manifest.snapshot.json"
        snapshot_task_ids = []
        if snapshot_path.exists():
            snapshot = _read_json(snapshot_path)
            snapshot_task_ids = [
                str(task.get("id") or "")
                for task in snapshot.get("tasks", [])
                if isinstance(task, dict) and str(task.get("id") or "")
            ]
        task_ids = snapshot_task_ids or list(tasks.keys())
        status_counts = Counter(
            str(task.get("status"))
            for task in tasks.values()
            if isinstance(task, dict) and task.get("status")
        )
        terminal_statuses = {"completed", "dry_run", "failed", "skipped", "cancelled"}
        pending_ids = [
            task_id
            for task_id in task_ids
            if not (
                isinstance(tasks.get(task_id), dict)
                and tasks[task_id].get("status") in terminal_statuses
            )
        ]
        cancel_requested = bool(_read_control(run_dir).get("cancel_requested"))
        if cancel_requested:
            inferred_status = "cancelling" if pending_ids else "cancelled"
        elif pending_ids:
            inferred_status = "running"
        elif any(status in status_counts for status in ("failed", "skipped")):
            inferred_status = "failed"
        elif status_counts.get("cancelled"):
            inferred_status = "cancelled"
        else:
            inferred_status = "completed"
        payload = {
            "run_id": run_id,
            "status": inferred_status,
            "manifest_path": ledger.get("manifest_path"),
            "started_at": ledger.get("started_at"),
            "channel_id": ledger.get("channel_id"),
            "taxonomy": ledger.get("taxonomy"),
            "task_count": len(task_ids),
            "pending_count": len(pending_ids),
            "status_counts": dict(sorted(status_counts.items())),
            "cancel_requested": cancel_requested,
        }
        print(json.dumps(payload, indent=2))
        return 0
    print(completion_path.read_text(encoding="utf-8"))
    return 0
