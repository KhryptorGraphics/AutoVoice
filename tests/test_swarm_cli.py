from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


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
