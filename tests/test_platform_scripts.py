from __future__ import annotations

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _script_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("AUTOVOICE_ENV_PREFIX", "/home/kp/anaconda3/envs/autovoice-thor")
    env.setdefault("AUTOVOICE_ENV_NAME", "autovoice-thor")
    env.setdefault("AUTOVOICE_DB_PASS", "ci-dry-run-password")
    return env


def test_platform_scripts_have_valid_bash_syntax():
    for script_name in (
        "scripts/setup_jetson_thor.sh",
        "scripts/validate_cuda_stack.sh",
        "scripts/launch_swarms.sh",
    ):
        result = subprocess.run(
            ["bash", "-n", script_name],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env=_script_env(),
        )
        assert result.returncode == 0, result.stderr


def test_setup_jetson_thor_dry_run(tmp_path):
    result = subprocess.run(
        [
            "bash",
            "scripts/setup_jetson_thor.sh",
            "--dry-run",
            "--skip-model-download",
            "--skip-latency-validation",
            "--output-dir",
            str(tmp_path / "reports"),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=_script_env(),
    )
    assert result.returncode == 0, result.stderr
    assert "Service Dependency Snapshot" in result.stdout
    assert "Database Initialization" in result.stdout
    assert "Jetson Thor setup workflow completed." in result.stdout


def test_validate_cuda_stack_dry_run(tmp_path):
    result = subprocess.run(
        [
            "bash",
            "scripts/validate_cuda_stack.sh",
            "--dry-run",
            "--output-dir",
            str(tmp_path / "reports"),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=_script_env(),
    )
    assert result.returncode == 0, result.stderr
    assert "dependency-audit.json" in result.stdout
    assert "all-latency-report.md" in result.stdout


def test_swarm_orchestrator_supports_custom_data_dir(tmp_path):
    result = subprocess.run(
        [
            "python",
            "scripts/swarm_orchestrator.py",
            "--swarm",
            "research",
            "--run-id",
            "wrapper-run",
            "--dry-run",
            "--data-dir",
            str(tmp_path / "swarm-data"),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=_script_env(),
    )
    assert result.returncode == 0, result.stderr
    completion_path = tmp_path / "swarm-data" / "swarm_runs" / "wrapper-run" / "completion.json"
    assert completion_path.exists()
