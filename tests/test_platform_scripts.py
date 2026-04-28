from __future__ import annotations

import os
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _script_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("AUTOVOICE_PYTHON", sys.executable)
    env.setdefault("AUTOVOICE_ENV_PREFIX", sys.prefix)
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


def test_validate_hosted_deployment_with_mock_vhost_files(tmp_path):
    vhost = tmp_path / "autovoice.conf"
    ssl_vhost = tmp_path / "autovoice-ssl.conf"
    content = """
ServerName autovoice.giggahost.com
DocumentRoot frontend/dist
ProxyPass /api http://127.0.0.1:10600/api
ProxyPass /socket.io http://127.0.0.1:10600/socket.io
ProxyPass /ready http://127.0.0.1:10600/ready
SecRequestBodyLimit 262144000
"""
    vhost.write_text(content, encoding="utf-8")
    ssl_vhost.write_text(content, encoding="utf-8")
    report_path = tmp_path / "hosted-preflight.json"

    env = _script_env()
    env["SECRET_KEY"] = "unit-test-secret-key"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_hosted_deployment.py",
            "--hostname",
            "autovoice.giggahost.com",
            "--skip-dns",
            "--skip-tls",
            "--vhost-file",
            str(vhost),
            "--vhost-file",
            str(ssl_vhost),
            "--report",
            str(report_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["checks"]["vhosts"]["ok"] is True


def test_hosted_deployment_discovers_server_alias_vhost(tmp_path):
    from scripts.validate_hosted_deployment import _check_vhost_files, _discover_vhost_files

    sites_dir = tmp_path / "sites-available"
    sites_dir.mkdir()
    canonical = sites_dir / "autovoice.giggadev.com.conf"
    canonical.write_text(
        """
<VirtualHost *:80>
    ServerName autovoice.giggadev.com
    ServerAlias autovoice.giggahost.com
</VirtualHost>
""",
        encoding="utf-8",
    )

    assert _discover_vhost_files("autovoice.giggahost.com", sites_dir=sites_dir) == [canonical]

    ssl_vhost = sites_dir / "autovoice.giggadev.com-le-ssl.conf"
    ssl_vhost.write_text(
        """
<VirtualHost *:443>
    ServerName autovoice.giggadev.com
    ServerAlias autovoice.giggahost.com
    DocumentRoot /home/kp/thordrive/autovoice/frontend/dist
    ProxyPass /api http://127.0.0.1:10600/api
    ProxyPass /socket.io http://127.0.0.1:10600/socket.io
    ProxyPass /ready http://127.0.0.1:10600/ready
    SecRequestBodyLimit 262144000
</VirtualHost>
""",
        encoding="utf-8",
    )

    result = _check_vhost_files(
        [canonical, ssl_vhost],
        hostname="autovoice.giggahost.com",
        backend_port=10600,
        frontend_root="frontend/dist",
        min_body_limit=262144000,
    )
    assert result["ok"] is True
    assert result["serving_vhost_count"] == 1


def test_completion_matrix_smoke_runner(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_completion_matrix.py",
            "--output-dir",
            str(tmp_path / "completion"),
            "--timeout",
            "180",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=_script_env(),
    )
    assert result.returncode == 0, result.stderr
    matrix_path = tmp_path / "completion" / "completion_matrix.json"
    audit_path = tmp_path / "completion" / "skip_audit.json"
    assert matrix_path.exists()
    assert audit_path.exists()

    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert matrix["ok"] is True
    lane_names = {lane["name"] for lane in matrix["lanes"]}
    assert "priority-skip-audit" in lane_names
    assert "benchmark-dashboard-validate" in lane_names
    assert "hosted-preflight-local" in lane_names

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["findings"] == []
    assert audit["environment_gate_evidence"]
    assert all(entry["explained"] for entry in audit["environment_gate_evidence"])
    assert {entry["owner"] for entry in audit["environment_gate_evidence"]} >= {"hardware-runner", "training-runtime"}
