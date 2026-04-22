from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from auto_voice.swarm.memory import SwarmMemoryBackend


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = str(PROJECT_ROOT / "src")
    if env.get("PYTHONPATH"):
        pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath
    return env


def test_swarm_cli_validate_and_run(tmp_path: Path):
    manifest = tmp_path / "manifest.yaml"
    output_file = tmp_path / "artifact.txt"
    manifest.write_text(
        "\n".join(
            [
                "version: 1",
                "name: test-swarm",
                "tasks:",
                "  - id: write",
                "    command: \"python -c \\\"from pathlib import Path; Path(r'"
                + str(output_file)
                + "').write_text('ok', encoding='utf-8')\\\"\"",
            ]
        ),
        encoding="utf-8",
    )

    validate = subprocess.run(
        [
            sys.executable,
            "-m",
            "auto_voice.cli",
            "swarm",
            "--data-dir",
            str(tmp_path / "data"),
            "validate",
            "--manifest",
            str(manifest),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=_env(),
    )
    assert validate.returncode == 0, validate.stderr
    assert json.loads(validate.stdout)["status"] == "valid"

    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "auto_voice.cli",
            "swarm",
            "--data-dir",
            str(tmp_path / "data"),
            "run",
            "--manifest",
            str(manifest),
            "--run-id",
            "test-run",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=_env(),
    )
    assert run.returncode == 0, run.stderr
    assert output_file.read_text(encoding="utf-8") == "ok"

    completion_path = tmp_path / "data" / "swarm_runs" / "test-run" / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    assert completion["status"] == "completed"
    assert completion["memory"]["backend"] in {"memkraft", "file_fallback"}

    status = subprocess.run(
        [
            sys.executable,
            "-m",
            "auto_voice.cli",
            "swarm",
            "--data-dir",
            str(tmp_path / "data"),
            "status",
            "--run-id",
            "test-run",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=_env(),
    )
    assert status.returncode == 0, status.stderr
    assert json.loads(status.stdout)["run_id"] == "test-run"


def test_swarm_cli_writes_memkraft_or_fallback_context(tmp_path: Path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "\n".join(
            [
                "version: 1",
                "name: memory-swarm",
                "parallelism: 2",
                "tasks:",
                "  - id: alpha",
                "    lane: research",
                "    role: analyst",
                "    command: \"python -c \\\"print('alpha')\\\"\"",
                "  - id: beta",
                "    lane: research",
                "    role: reviewer",
                "    command: \"python -c \\\"print('beta')\\\"\"",
            ]
        ),
        encoding="utf-8",
    )

    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "auto_voice.cli",
            "swarm",
            "--data-dir",
            str(tmp_path / "data"),
            "run",
            "--manifest",
            str(manifest),
            "--run-id",
            "memory-run",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=_env(),
    )
    assert run.returncode == 0, run.stderr

    completion_path = tmp_path / "data" / "swarm_runs" / "memory-run" / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    assert completion["memory"]["backend"] in {"memkraft", "file_fallback"}

    if completion["memory"]["backend"] == "memkraft":
        channel_path = tmp_path / "data" / "swarm_memory" / ".memkraft" / "channels" / "autovoice-memory-run.json"
        assert channel_path.exists()
    else:
        fallback_path = tmp_path / "data" / "swarm_memory" / "fallback" / "autovoice-memory-run.json"
        assert fallback_path.exists()


def test_swarm_memory_backend_uses_memkraft_when_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    events: list[tuple[str, object]] = []

    class FakeMemKraft:
        def __init__(self, *, base_dir: str):
            self.base_dir = base_dir

        def init(self, *, force: bool, verbose: bool) -> None:
            events.append(("init", {"force": force, "verbose": verbose, "base_dir": self.base_dir}))

        def channel_save(self, channel_id: str, payload: dict) -> None:
            events.append(("channel_save", channel_id, payload))

        def task_start(self, task_id: str, description: str, *, channel_id: str, agent: str) -> None:
            events.append(("task_start", task_id, description, channel_id, agent))

        def task_complete(self, task_id: str, note: str) -> None:
            events.append(("task_complete", task_id, note))

        def task_update(self, task_id: str, status: str, note: str) -> None:
            events.append(("task_update", task_id, status, note))

        def agent_save(self, agent_id: str, payload: dict) -> None:
            events.append(("agent_save", agent_id, payload))

    monkeypatch.setattr("auto_voice.swarm.memory.MemKraft", FakeMemKraft)

    backend = SwarmMemoryBackend.create(
        run_id="memkraft-direct",
        run_root=tmp_path / "data" / "swarm_runs" / "memkraft-direct",
        payload={"name": "mem-test", "issue_id": "AV-test"},
        project_root=PROJECT_ROOT,
    )

    assert backend.available is True
    assert backend.backend == "memkraft"
    assert backend.channel_id == "autovoice-memkraft-direct"
    assert events[0][0] == "init"
    assert events[1][0] == "channel_save"
    assert events[1][1] == "autovoice-memkraft-direct"
