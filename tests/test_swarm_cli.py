from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

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
                "program: next-phase-perfection",
                "phase: hardening",
                "sprint: sprint-0",
                "lane: development",
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
    assert completion["channel_id"] == "autovoice-next-phase-perfection-hardening-sprint-0-test-run"
    assert completion["taxonomy"] == {
        "program": "next-phase-perfection",
        "phase": "hardening",
        "sprint": "sprint-0",
        "lane": "development",
    }
    assert completion["memory"]["backend"] in {"memkraft", "file_fallback"}
    assert completion["memory"]["channel_id"] == completion["channel_id"]

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
    ledger = json.loads((tmp_path / "data" / "swarm_runs" / "memory-run" / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["prior_memory"]["query"]["channel_id"] == completion["memory"]["channel_id"]
    assert "fallback_items" in ledger["prior_memory"]

    if completion["memory"]["backend"] == "memkraft":
        channel_path = (
            tmp_path
            / "data"
            / "swarm_memory"
            / ".memkraft"
            / "channels"
            / f"{completion['memory']['channel_id']}.json"
        )
        assert channel_path.exists()
    else:
        fallback_path = (
            tmp_path
            / "data"
            / "swarm_memory"
            / "fallback"
            / f"{completion['memory']['channel_id']}.json"
        )
        assert fallback_path.exists()


def test_swarm_cli_exports_run_scoped_context(tmp_path: Path):
    manifest = tmp_path / "manifest.yaml"
    output_file = tmp_path / "env.json"
    script = tmp_path / "capture_env.py"
    script.write_text(
        "\n".join(
            [
                "import json",
                "import os",
                "import sys",
                "from pathlib import Path",
                "",
                "keys = [",
                "    'AUTOVOICE_SWARM_RUN_ID',",
                "    'AUTOVOICE_SWARM_PARENT_RUN_ID',",
                "    'AUTOVOICE_SWARM_DATA_DIR',",
                "    'AUTOVOICE_SWARM_CHANNEL_ID',",
                "    'AUTOVOICE_SWARM_TASK_ID',",
                "    'AUTOVOICE_SWARM_TASK_KEY',",
                "    'AUTOVOICE_SWARM_LANE',",
                "    'AUTOVOICE_SWARM_LANE_KEY',",
                "    'AUTOVOICE_SWARM_AGENT_KEY',",
                "    'AUTOVOICE_SWARM_ARTIFACT_KEY',",
                "    'AUTOVOICE_SWARM_ARTIFACT_ROOT',",
                "]",
                "Path(sys.argv[1]).write_text(",
                "    json.dumps({key: os.environ.get(key) for key in keys}, sort_keys=True),",
                "    encoding='utf-8',",
                ")",
            ]
        ),
        encoding="utf-8",
    )
    manifest.write_text(
        "\n".join(
            [
                "version: 1",
                "name: scoped-env",
                "program: next-phase-perfection",
                "phase: hardening",
                "sprint: sprint-0",
                "lane: review",
                "tasks:",
                "  - id: export-env",
                "    lane: review",
                "    role: docs_reviewer",
                f"    command: \"python {script} {output_file}\"",
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
            "scope-run",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=_env(),
    )
    assert run.returncode == 0, run.stderr

    scoped = json.loads(output_file.read_text(encoding="utf-8"))
    assert scoped["AUTOVOICE_SWARM_RUN_ID"] == "scope-run"
    assert scoped["AUTOVOICE_SWARM_PARENT_RUN_ID"] == ""
    assert scoped["AUTOVOICE_SWARM_TASK_ID"] == "export-env"
    assert scoped["AUTOVOICE_SWARM_TASK_KEY"] == "scope-run:review:export-env"
    assert scoped["AUTOVOICE_SWARM_LANE"] == "review"
    assert scoped["AUTOVOICE_SWARM_LANE_KEY"] == "scope-run:review"
    assert scoped["AUTOVOICE_SWARM_AGENT_KEY"] == "scope-run:review:docs_reviewer"
    assert scoped["AUTOVOICE_SWARM_ARTIFACT_KEY"] == "scope-run:review:export-env:artifact"
    assert scoped["AUTOVOICE_SWARM_CHANNEL_ID"] == "autovoice-next-phase-perfection-hardening-sprint-0-scope-run"
    assert scoped["AUTOVOICE_SWARM_DATA_DIR"] == str(tmp_path / "data")
    assert scoped["AUTOVOICE_SWARM_ARTIFACT_ROOT"].endswith("scope-run/artifacts/review/export-env")


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

        def search(self, query: str, *, limit: int = 10) -> list[dict]:
            events.append(("search", query, limit))
            return [{"query": query}]

    monkeypatch.setattr("auto_voice.swarm.memory.MemKraft", FakeMemKraft)

    backend = SwarmMemoryBackend.create(
        run_id="memkraft-direct",
        run_root=tmp_path / "data" / "swarm_runs" / "memkraft-direct",
        payload={
            "name": "mem-test",
            "issue_id": "AV-test",
            "program": "next-phase-perfection",
            "phase": "hardening",
            "sprint": "sprint-0",
            "lane": "testing",
        },
        project_root=PROJECT_ROOT,
    )

    assert backend.available is True
    assert backend.backend == "memkraft"
    assert backend.channel_id == "autovoice-next-phase-perfection-hardening-sprint-0-memkraft-direct"
    assert backend.taxonomy == {
        "program": "next-phase-perfection",
        "phase": "hardening",
        "sprint": "sprint-0",
        "lane": "testing",
    }
    assert backend.task_prefix == "memkraft-direct:testing"
    prior = backend.read_prior_context()
    assert prior["memkraft_items"] == [{"query": "next-phase-perfection hardening sprint-0 testing"}]
    assert events[0][0] == "init"
    assert events[1][0] == "channel_save"
    assert events[1][1] == "autovoice-next-phase-perfection-hardening-sprint-0-memkraft-direct"


def test_full_swarm_manifest_propagates_parent_run_context():
    manifest = yaml.safe_load((PROJECT_ROOT / "config" / "swarm_manifests" / "full.yaml").read_text(encoding="utf-8"))
    commands = [task["command"] for task in manifest["tasks"]]
    assert all('--data-dir "$AUTOVOICE_SWARM_DATA_DIR"' in command for command in commands)
    assert any('${AUTOVOICE_SWARM_RUN_ID}-research' in command for command in commands)
    assert any('${AUTOVOICE_SWARM_RUN_ID}-development' in command for command in commands)
    assert any('${AUTOVOICE_SWARM_RUN_ID}-review' in command for command in commands)
    assert any('${AUTOVOICE_SWARM_RUN_ID}-testing' in command for command in commands)


def test_swarm_cli_cancel_marks_run_cancelled(tmp_path: Path):
    manifest = tmp_path / "manifest.yaml"
    flag_path = tmp_path / "cancel.flag"
    manifest.write_text(
        "\n".join(
            [
                "version: 1",
                "name: cancel-swarm",
                "parallelism: 1",
                "tasks:",
                "  - id: one",
                "    command: \"python -c \\\"from pathlib import Path; Path(r'"
                + str(flag_path)
                + "').write_text('done', encoding='utf-8')\\\"\"",
                "  - id: two",
                "    deps: [one]",
                "    command: \"python -c \\\"print('never-runs')\\\"\"",
            ]
        ),
        encoding="utf-8",
    )
    run_root = tmp_path / "data" / "swarm_runs"
    run_dir = run_root / "cancel-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "control.json").write_text(
        json.dumps({"run_id": "cancel-run", "cancel_requested": True, "reason": "test_cancel"}),
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
            "cancel-run",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=_env(),
    )
    assert run.returncode == 1
    completion = json.loads((run_dir / "completion.json").read_text(encoding="utf-8"))
    assert completion["status"] == "cancelled"
    assert not flag_path.exists()


def test_swarm_cli_resume_and_retry_follow_failed_run(tmp_path: Path):
    marker = tmp_path / "resume-ok.txt"
    state_file = tmp_path / "failure-state.json"
    manifest = tmp_path / "manifest.yaml"
    script = tmp_path / "flaky.py"
    script.write_text(
        "\n".join(
            [
                "import json",
                "import sys",
                "from pathlib import Path",
                "state_path = Path(sys.argv[1])",
                "marker_path = Path(sys.argv[2])",
                "state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {'fail_once': True}",
                "if state.get('fail_once'):",
                "    state['fail_once'] = False",
                "    state_path.write_text(json.dumps(state), encoding='utf-8')",
                "    raise SystemExit(1)",
                "marker_path.write_text('ok', encoding='utf-8')",
            ]
        ),
        encoding="utf-8",
    )
    manifest.write_text(
        "\n".join(
            [
                "version: 1",
                "name: retry-swarm",
                "tasks:",
                "  - id: alpha",
                "    command: \"python -c \\\"print('alpha')\\\"\"",
                f"  - id: beta",
                "    deps: [alpha]",
                f"    command: \"python {script} {state_file} {marker}\"",
            ]
        ),
        encoding="utf-8",
    )

    first_run = subprocess.run(
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
            "base-run",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=_env(),
    )
    assert first_run.returncode == 1

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
            "base-run",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=_env(),
    )
    assert status.returncode == 0
    assert json.loads(status.stdout)["status"] == "failed"

    retry_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "auto_voice.cli",
            "swarm",
            "--data-dir",
            str(tmp_path / "data"),
            "retry",
            "--source-run-id",
            "base-run",
            "--run-id",
            "retry-run",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=_env(),
    )
    assert retry_run.returncode == 0, retry_run.stderr
    assert marker.read_text(encoding="utf-8") == "ok"

    # Build an interrupted run by deleting completion while keeping the ledger.
    interrupted_run_dir = tmp_path / "data" / "swarm_runs" / "base-run"
    (interrupted_run_dir / "completion.json").unlink()
    resume_status = subprocess.run(
        [
            sys.executable,
            "-m",
            "auto_voice.cli",
            "swarm",
            "--data-dir",
            str(tmp_path / "data"),
            "status",
            "--run-id",
            "base-run",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=_env(),
    )
    assert resume_status.returncode == 0
    live_status = json.loads(resume_status.stdout)
    assert live_status["status"] == "failed"
    assert live_status["status_counts"]["completed"] == 1
    assert live_status["status_counts"]["failed"] == 1
    assert live_status["task_count"] == 2
    assert live_status["pending_count"] == 0

    resume_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "auto_voice.cli",
            "swarm",
            "--data-dir",
            str(tmp_path / "data"),
            "resume",
            "--source-run-id",
            "base-run",
            "--run-id",
            "resume-run",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=_env(),
    )
    assert resume_run.returncode == 0, resume_run.stderr


def test_swarm_cli_resume_includes_unwritten_pending_manifest_tasks(tmp_path: Path):
    marker = tmp_path / "pending.txt"
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "\n".join(
            [
                "version: 1",
                "name: interrupted-swarm",
                "tasks:",
                "  - id: alpha",
                "    command: \"python -c \\\"print('alpha')\\\"\"",
                "  - id: beta",
                "    deps: [alpha]",
                f"    command: \"python -c \\\"from pathlib import Path; Path(r'{marker}').write_text('ok', encoding='utf-8')\\\"\"",
            ]
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "data" / "swarm_runs" / "interrupted-run"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.snapshot.json").write_text(
        json.dumps(yaml.safe_load(manifest.read_text(encoding="utf-8"))),
        encoding="utf-8",
    )
    (run_dir / "ledger.json").write_text(
        json.dumps(
            {
                "run_id": "interrupted-run",
                "manifest_path": str(manifest),
                "tasks": {"alpha": {"task_id": "alpha", "status": "completed"}},
            }
        ),
        encoding="utf-8",
    )

    resume_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "auto_voice.cli",
            "swarm",
            "--data-dir",
            str(tmp_path / "data"),
            "resume",
            "--source-run-id",
            "interrupted-run",
            "--run-id",
            "resume-pending",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=_env(),
    )

    assert resume_run.returncode == 0, resume_run.stderr
    assert marker.read_text(encoding="utf-8") == "ok"
    child_manifest = tmp_path / "data" / "swarm_runs" / "resume-pending" / "resume.manifest.json"
    assert child_manifest.exists()
    child_payload = json.loads(child_manifest.read_text(encoding="utf-8"))
    assert [task["id"] for task in child_payload["tasks"]] == ["beta"]


def test_swarm_cli_cancel_completed_run_is_noop(tmp_path: Path):
    run_dir = tmp_path / "data" / "swarm_runs" / "complete-run"
    run_dir.mkdir(parents=True)
    (run_dir / "completion.json").write_text(
        json.dumps({"run_id": "complete-run", "status": "completed"}),
        encoding="utf-8",
    )

    cancel = subprocess.run(
        [
            sys.executable,
            "-m",
            "auto_voice.cli",
            "swarm",
            "--data-dir",
            str(tmp_path / "data"),
            "cancel",
            "--run-id",
            "complete-run",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=_env(),
    )

    assert cancel.returncode == 0
    payload = json.loads(cancel.stdout)
    assert payload["cancel_requested"] is False
    assert payload["terminal_status"] == "completed"
    assert not (run_dir / "control.json").exists()
