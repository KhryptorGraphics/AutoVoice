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


def test_dependency_contract_script_passes():
    result = subprocess.run(
        [sys.executable, "scripts/check_dependency_contract.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=_script_env(),
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["ok"] is True


def test_dependency_contract_script_detects_lock_drift(tmp_path):
    current_lock = PROJECT_ROOT / "requirements.lock"
    tampered = json.loads(current_lock.read_text(encoding="utf-8"))
    tampered["backend"]["requirements_runtime"]["sha256"] = (
        "000000000000000000000000000000000000000000000000000000000000000000"
    )
    tampered_path = tmp_path / "requirements.lock"
    tampered_path.write_text(json.dumps(tampered, indent=2), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_dependency_contract.py",
            "--lock-path",
            str(tampered_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=_script_env(),
    )
    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    assert report["ok"] is False
    assert any("requirements-runtime.txt hash mismatch" in msg for msg in report["errors"])


def test_dependency_contract_enforces_high_critical_audit_policy(tmp_path):
    pip_report = {"vulnerabilities": [{"id": "CVE-2026-0001", "severity": "high"}]}
    npm_report = {"metadata": {"vulnerabilities": {"high": 1, "moderate": 0}}}
    pip_report_path = tmp_path / "pip-audit.json"
    npm_report_path = tmp_path / "npm-audit.json"
    pip_report_path.write_text(json.dumps(pip_report), encoding="utf-8")
    npm_report_path.write_text(json.dumps(npm_report), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_dependency_contract.py",
            "--lock-path",
            str(PROJECT_ROOT / "requirements.lock"),
            "--pip-audit-report",
            str(pip_report_path),
            "--npm-audit-report",
            str(npm_report_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=_script_env(),
    )
    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    assert report["ok"] is False
    assert report["audits"]["pip-audit"]["ok"] is False
    assert report["audits"]["npm-audit"]["ok"] is False


def test_common_env_prefers_canonical_python_over_stale_python():
    command = (
        "export PYTHON=/tmp/not-autovoice-python; "
        "source scripts/common_env.sh && "
        "autovoice_activate_env && "
        "printf '%s\\n%s\\n%s\\n' \"$PYTHON\" \"$PYTHONNOUSERSITE\" \"$PYTHONPATH\""
    )
    result = subprocess.run(
        ["bash", "-lc", command],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env={
            "HOME": os.environ.get("HOME", ""),
            "PATH": os.environ.get("PATH", ""),
            "AUTOVOICE_ENV_NAME": "autovoice-thor",
        },
    )
    assert result.returncode == 0, result.stderr
    python_path, no_user_site, pythonpath = result.stdout.strip().splitlines()
    assert python_path.endswith("/anaconda3/envs/autovoice-thor/bin/python")
    assert no_user_site == "1"
    assert str(PROJECT_ROOT / "src") in pythonpath.split(":")


def test_common_env_allows_explicit_autovoice_python_override():
    command = (
        "export AUTOVOICE_PYTHON=\"$OVERRIDE_PYTHON\"; "
        "export PYTHON=/tmp/not-autovoice-python; "
        "source scripts/common_env.sh && "
        "autovoice_activate_env && "
        "printf '%s\\n' \"$PYTHON\""
    )
    result = subprocess.run(
        ["bash", "-lc", command],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env={
            **os.environ.copy(),
            "OVERRIDE_PYTHON": sys.executable,
            "AUTOVOICE_ENV_NAME": "autovoice-thor",
        },
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == sys.executable


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
    assert "hardware-lane-gates.json" in result.stdout
    assert "all-latency-report.md" in result.stdout


def test_hardware_lane_gate_report_records_experimental_owner_action(tmp_path):
    output_path = tmp_path / "hardware-lane-gates.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_hardware_lane_gates.py",
            "--pipeline",
            "all",
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=_script_env(),
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["lanes"]["hq_svc"]["support_boundary"] == "experimental:hq_svc"
    assert report["lanes"]["hq_svc"]["owner"] == "model-runtime"
    assert report["lanes"]["hq_svc"]["action"]
    assert report["lanes"]["realtime_meanvc"]["support_boundary"] == "experimental:meanvc"
    assert report["lanes"]["realtime_meanvc"]["owner"] == "model-runtime"
    assert report["lanes"]["realtime_meanvc"]["action"]


def test_hardware_release_evidence_runner_fails_closed_without_execute(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_hardware_release_evidence.py",
            "--dry-run",
            "--allow-blocked",
            "--output-dir",
            str(tmp_path / "release-evidence"),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=_script_env(),
    )
    assert result.returncode == 0, result.stderr
    decision_path = tmp_path / "release-evidence" / "release_decision.json"
    preflight_path = tmp_path / "release-evidence" / "preflight.json"
    assert decision_path.exists()
    assert preflight_path.exists()
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["ready"] is False
    assert decision["git_sha"]
    assert "hardware validation lanes were not executed" in " ".join(decision["blockers"])


def test_hardware_release_evidence_accepts_streaming_tegrastats_probe(monkeypatch):
    import importlib.util

    script_path = PROJECT_ROOT / "scripts/run_hardware_release_evidence.py"
    spec = importlib.util.spec_from_file_location("run_hardware_release_evidence", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def fake_run(*_args, **_kwargs):
        error = subprocess.TimeoutExpired(["tegrastats", "--interval", "1000"], 3)
        error.stdout = "RAM 100/1000MB CPU [1%@1000] gpu@35C\n"
        error.stderr = ""
        raise error

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = module._run(
        ["tegrastats", "--interval", "1000"],
        timeout=3,
        timeout_ok_when_output=True,
    )

    assert result["ok"] is True
    assert result["timed_out"] is True
    assert "RAM 100/1000MB" in result["stdout"]


def test_full_hardware_rc_preflight_records_actionable_blockers(tmp_path):
    output_path = tmp_path / "preflight.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/preflight_full_hardware_rc.py",
            "--output",
            str(output_path),
            "--no-require-docker",
            "--no-require-gitnexus",
            "--no-require-hosted-probes",
            "--no-require-jetson",
            "--no-require-tensorrt-suite",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=_script_env(),
    )
    assert result.returncode == 1
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["ready"] is False
    assert any(blocker["check"] == "benchmark_report" for blocker in report["blockers"])
    assert report["checks"]["hosted_endpoints"]["skipped"] is True


def test_full_hardware_rc_runner_writes_blocked_decision(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_full_hardware_rc.py",
            "--artifact-root",
            str(tmp_path / "release-candidates"),
            "--no-require-docker",
            "--no-require-gitnexus",
            "--no-require-hosted-probes",
            "--no-require-jetson",
            "--no-require-tensorrt-suite",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=_script_env(),
    )
    assert result.returncode == 1
    decision_path = Path(result.stdout.strip())
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["ready_for_release"] is False
    assert decision["status"] == "blocked"
    assert any(blocker["check"] == "benchmark_report" for blocker in decision["blockers"])


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
    enabled_sites = tmp_path / "sites-enabled"
    enabled_sites.mkdir()
    cert_path = tmp_path / "fullchain.pem"
    key_path = tmp_path / "privkey.pem"
    cert_path.write_text("unit-test-certificate", encoding="utf-8")
    key_path.write_text("unit-test-key", encoding="utf-8")
    content = """
ServerName autovoice.giggahost.com
DocumentRoot frontend/dist
ProxyPass /api http://127.0.0.1:10600/api
ProxyPass /socket.io http://127.0.0.1:10600/socket.io
ProxyPass /ready http://127.0.0.1:10600/ready
SecRequestBodyLimit 262144000
"""
    enabled_ssl_content = f"""
<VirtualHost *:443>
SSLCertificateFile {cert_path}
SSLCertificateKeyFile {key_path}
</VirtualHost>
"""
    vhost.write_text(content, encoding="utf-8")
    ssl_vhost.write_text(content, encoding="utf-8")
    (enabled_sites / "autovoice.conf").write_text(enabled_ssl_content, encoding="utf-8")
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
            "--skip-apache-configtest",
            "--enabled-sites-dir",
            str(enabled_sites),
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
    assert report["checks"]["apache_configtest"]["ok"] is True
    assert report["checks"]["apache_configtest"]["skipped"] is True
    assert report["checks"]["enabled_ssl_certificates"]["ok"] is True
    assert report["checks"]["enabled_ssl_certificates"]["checked_path_count"] == 2


def test_hosted_deployment_reports_enabled_ssl_certificate_paths(tmp_path):
    from scripts.validate_hosted_deployment import _check_enabled_ssl_certificates

    enabled_sites = tmp_path / "sites-enabled"
    enabled_sites.mkdir()
    cert_path = tmp_path / "fullchain.pem"
    key_path = tmp_path / "privkey.pem"
    cert_path.write_text("unit-test-certificate", encoding="utf-8")
    key_path.write_text("unit-test-key", encoding="utf-8")
    vhost = enabled_sites / "autovoice.conf"
    vhost.write_text(
        f"""
<VirtualHost *:443>
    SSLCertificateFile {cert_path}
    SSLCertificateKeyFile {key_path}
</VirtualHost>
""",
        encoding="utf-8",
    )

    result = _check_enabled_ssl_certificates(enabled_sites)

    assert result["ok"] is True
    assert result["checked_path_count"] == 2
    assert result["files"][str(vhost)]["certificate_paths"][0]["path"] == str(cert_path)


def test_hosted_deployment_fails_on_missing_enabled_ssl_certificate_path(tmp_path):
    from scripts.validate_hosted_deployment import _check_enabled_ssl_certificates

    enabled_sites = tmp_path / "sites-enabled"
    enabled_sites.mkdir()
    missing_cert = tmp_path / "missing-fullchain.pem"
    key_path = tmp_path / "privkey.pem"
    key_path.write_text("unit-test-key", encoding="utf-8")
    vhost = enabled_sites / "autovoice.conf"
    vhost.write_text(
        f"""
<VirtualHost *:443>
    SSLCertificateFile {missing_cert}
    SSLCertificateKeyFile {key_path}
</VirtualHost>
""",
        encoding="utf-8",
    )

    result = _check_enabled_ssl_certificates(enabled_sites)

    assert result["ok"] is False
    assert result["missing"] == [
        {
            "enabled_path": str(vhost),
            "resolved_path": str(vhost),
            "directive": "SSLCertificateFile",
            "line": 3,
            "path": str(missing_cert),
            "exists": False,
            "size_bytes": 0,
        }
    ]


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
    assert result["canonical_server_name_ok"] is False
    assert result["files"][str(ssl_vhost)]["hostname_covered"] is True


def test_hosted_deployment_can_require_canonical_server_name(tmp_path):
    from scripts.validate_hosted_deployment import _check_vhost_files

    vhost = tmp_path / "autovoice.conf"
    vhost.write_text(
        """
<VirtualHost *:443>
    ServerName autovoice.giggahost.com
    DocumentRoot frontend/dist
    ProxyPass /api http://127.0.0.1:10600/api
    ProxyPass /socket.io http://127.0.0.1:10600/socket.io
    ProxyPass /ready http://127.0.0.1:10600/ready
    SecRequestBodyLimit 262144000
</VirtualHost>
""",
        encoding="utf-8",
    )

    result = _check_vhost_files(
        [vhost],
        hostname="autovoice.giggahost.com",
        backend_port=10600,
        frontend_root="frontend/dist",
        min_body_limit=262144000,
    )
    assert result["ok"] is True
    assert result["canonical_server_name_ok"] is True


def test_completion_matrix_smoke_runner(tmp_path):
    platform_dir = tmp_path / "platform"
    parity_report = tmp_path / "benchmarks" / "parity.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_completion_matrix.py",
            "--output-dir",
            str(tmp_path / "completion"),
            "--platform-report-dir",
            str(platform_dir),
            "--tensorrt-parity-report",
            str(parity_report),
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
    assert matrix["artifacts"]["platform_report_dir"] == str(platform_dir)
    assert matrix["artifacts"]["tensorrt_parity_report"] == str(parity_report)
    lane_names = {lane["name"] for lane in matrix["lanes"]}
    assert "priority-skip-audit" in lane_names
    assert "benchmark-dashboard-validate" in lane_names
    assert "hosted-preflight-local" in lane_names
    assert "real-audio-e2e-lanes" in lane_names
    assert "live-youtube-ingest-smoke" in lane_names
    assert "tensorrt-checkpoint-parity" in lane_names
    assert (platform_dir / "hosted-preflight.json").exists()

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["findings"] == []
    assert audit["environment_gate_evidence"]
    assert all(entry["explained"] for entry in audit["environment_gate_evidence"])
    assert {entry["owner"] for entry in audit["environment_gate_evidence"]} >= {"hardware-runner", "training-runtime"}


def test_completion_matrix_full_mode_defines_real_audio_e2e_lanes(monkeypatch, tmp_path):
    import importlib.util

    script_path = PROJECT_ROOT / "scripts" / "run_completion_matrix.py"
    spec = importlib.util.spec_from_file_location("run_completion_matrix", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    captured: list[tuple[str, list[str]]] = []

    def fake_run_command(name, command, *, report_dir, timeout, env=None):
        del report_dir, timeout, env
        captured.append((name, command))
        return module.LaneResult(name=name, status="passed", command=command)

    monkeypatch.setattr(module, "_run_command", fake_run_command)

    lanes = module.run_supported_real_audio_e2e_lanes(tmp_path, timeout=30)

    lane_names = {lane.name for lane in lanes}
    assert {
        "youtube-ingest-real-audio-e2e",
        "conversion-e2e",
        "karaoke-websocket-e2e",
        "voice-profile-diarization-e2e",
        "voice-profile-training-e2e",
        "benchmark-audio-e2e",
    } <= lane_names
    assert all(command[:3] == [sys.executable, "-m", "pytest"] for _, command in captured)
    assert any("tests/test_youtube_ingest_real_audio_e2e.py" in command for _, command in captured)
    assert any("tests/test_quality_benchmarking_sota.py" in command for _, command in captured)


def test_completion_matrix_real_audio_flag_does_not_enable_docker_or_hardware():
    import importlib.util

    script_path = PROJECT_ROOT / "scripts" / "run_completion_matrix.py"
    spec = importlib.util.spec_from_file_location("run_completion_matrix", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    args = module.parse_args(["--real-audio"])

    assert args.real_audio is True
    assert args.real_compose is False
    assert args.hardware is False
    assert args.full_hosted_preflight is False


def test_tensorrt_parity_benchmark_metadata_validation(tmp_path):
    import importlib.util

    script_path = PROJECT_ROOT / "scripts" / "benchmark_tensorrt_parity.py"
    spec = importlib.util.spec_from_file_location("benchmark_tensorrt_parity", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    metadata_path = tmp_path / "engine_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "artist_key": "unit_artist",
                "checkpoint_path": "checkpoint.pt",
                "engine_path": "engine.trt",
                "onnx_path": "engine.onnx",
                "precision": "fp16",
            }
        ),
        encoding="utf-8",
    )
    metadata = module._load_metadata(metadata_path)
    assert metadata["artist_key"] == "unit_artist"

    metadata_path.write_text(json.dumps({"artist_key": "missing"}), encoding="utf-8")
    try:
        module._load_metadata(metadata_path)
    except ValueError as exc:
        assert "missing required metadata fields" in str(exc)
    else:
        raise AssertionError("expected metadata validation failure")


def test_preflight_full_hardware_rc_reports_blockers(tmp_path):
    report_path = tmp_path / "preflight.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/preflight_full_hardware_rc.py",
            "--output",
            str(report_path),
            "--no-require-hosted-probes",
            "--no-require-jetson",
            "--no-require-docker",
            "--no-require-tensorrt-suite",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=_script_env(),
    )
    assert result.returncode == 1, result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ready"] is False
    assert any(blocker["check"] == "benchmark_report" for blocker in report["blockers"])


def test_preflight_full_hardware_rc_accepts_explicit_benchmark_report(tmp_path):
    report_path = tmp_path / "preflight.json"
    benchmark_report = tmp_path / "benchmark-report.json"
    benchmark_report.write_text(
        json.dumps(
            {
                "timestamp": "2026-04-28T12:00:00Z",
                "pipelines": {
                    "quality_seedvc": {"success": True},
                    "realtime": {"success": True},
                },
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/preflight_full_hardware_rc.py",
            "--output",
            str(report_path),
            "--benchmark-report",
            str(benchmark_report),
            "--no-require-hosted-probes",
            "--no-require-jetson",
            "--no-require-docker",
            "--no-require-tensorrt-suite",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=_script_env(),
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ready"] is True
    assert report["checks"]["benchmark_report"]["ok"] is True


def test_run_full_hardware_rc_writes_blocked_release_decision(tmp_path):
    artifact_root = tmp_path / "release-candidates"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_full_hardware_rc.py",
            "--artifact-root",
            str(artifact_root),
            "--no-require-hosted-probes",
            "--no-require-jetson",
            "--no-require-docker",
            "--no-require-tensorrt-suite",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=_script_env(),
    )
    assert result.returncode == 1, result.stderr
    decision_path = Path(result.stdout.strip())
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["status"] == "blocked"
    assert decision["ready_for_release"] is False
    assert any(blocker["check"] == "benchmark_report" for blocker in decision["blockers"])
    run_dir = decision_path.parent
    assert (run_dir / "preflight.json").exists()
    assert (run_dir / "artifact_manifest.json").exists()


def test_run_full_hardware_rc_records_local_deployment_inputs(tmp_path):
    artifact_root = tmp_path / "release-candidates"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_full_hardware_rc.py",
            "--artifact-root",
            str(artifact_root),
            "--deployment-base-url",
            "http://127.0.0.1:10001",
            "--no-require-production-smoke-stems",
            "--no-run-real-compose",
            "--no-run-full-hosted-preflight",
            "--no-require-hosted-probes",
            "--no-require-jetson",
            "--no-require-docker",
            "--no-require-tensorrt-suite",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=_script_env(),
    )
    assert result.returncode == 1, result.stderr
    decision_path = Path(result.stdout.strip())
    manifest = json.loads((decision_path.parent / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["inputs"]["deployment_base_url"] == "http://127.0.0.1:10001"
    assert manifest["inputs"]["require_production_smoke_stems"] is False
    assert manifest["inputs"]["run_real_compose"] is False
    assert manifest["inputs"]["run_full_hosted_preflight"] is False
