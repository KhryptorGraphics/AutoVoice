from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        payload = None
        status = 200
        if self.path == "/api/v1/health":
            payload = {"status": "healthy"}
        elif self.path == "/ready":
            payload = {"ready": True}
        elif self.path == "/api/v1/metrics":
            payload = {"total_conversions": 1}
        else:
            status = 404
            payload = {"error": "not found"}

        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A003
        return


def _seed_evidence_files() -> None:
    evidence_dir = PROJECT_ROOT / "reports" / "benchmarks" / "latest"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    provenance = {
        "schema_version": 1,
        "generator": "tests.test_release_candidate_validation",
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "source_bundles": ["reports/benchmarks/run-001/summary.json"],
    }
    dashboard = {
        "generated_at": generated_at,
        "provenance": provenance,
        "target_hardware": "NVIDIA Thor",
        "canonical_pipelines": {"offline": "quality_seedvc", "live": "realtime"},
        "pipelines": {
            "quality_seedvc": {"summary": {}, "sample_count": 1},
            "realtime": {"summary": {}, "sample_count": 1},
        },
        "comparisons": {},
        "promotable_candidates": [],
    }
    release_evidence = {
        "generated_at": generated_at,
        "provenance": provenance,
        "target_hardware": "NVIDIA Thor",
        "health_url": "/api/v1/health",
        "canonical_pipelines": {"offline": "quality_seedvc", "live": "realtime"},
        "pipeline_count": 2,
        "comparison_count": 0,
        "promotable_candidates": [],
        "quality_gate_passed": True,
        "quality_failures": [],
    }
    (evidence_dir / "benchmark_dashboard.json").write_text(json.dumps(dashboard), encoding="utf-8")
    (evidence_dir / "release_evidence.json").write_text(json.dumps(release_evidence), encoding="utf-8")


def test_validate_release_candidate_script(tmp_path: Path):
    _seed_evidence_files()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        report_dir = tmp_path / "reports"
        env = os.environ.copy()
        pythonpath = str(PROJECT_ROOT / "src")
        if env.get("PYTHONPATH"):
            pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
        env["PYTHONPATH"] = pythonpath

        result = subprocess.run(
            [
                sys.executable,
                "scripts/validate_release_candidate.py",
                "--base-url",
                f"http://127.0.0.1:{server.server_port}",
                "--skip-compose",
                "--report-dir",
                str(report_dir),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, result.stderr
        report_path = Path(result.stdout.strip())
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["compose"]["ok"] is True
        assert report["repo_boundaries"]["ok"] is True
        assert all(check["ok"] for check in report["checks"])
        checked_urls = {check["url"] for check in report["checks"]}
        assert any(url.endswith("/api/v1/health") for url in checked_urls)
        assert any(url.endswith("/ready") for url in checked_urls)
    finally:
        server.shutdown()
        server.server_close()


def test_validate_release_candidate_waits_for_endpoints(tmp_path: Path):
    _seed_evidence_files()
    report_dir = tmp_path / "reports"
    env = os.environ.copy()
    pythonpath = str(PROJECT_ROOT / "src")
    if env.get("PYTHONPATH"):
        pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    def _start_server_late():
        import time

        time.sleep(0.5)
        delayed_server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
        delayed_server.serve_forever()

    thread = threading.Thread(target=_start_server_late, daemon=True)
    thread.start()

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_release_candidate.py",
            "--base-url",
            f"http://127.0.0.1:{port}",
            "--skip-compose",
            "--wait-seconds",
            "2",
            "--poll-interval",
            "0.1",
            "--report-dir",
            str(report_dir),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr


def test_validate_release_candidate_supports_smoke_report_mode(tmp_path: Path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        report_dir = tmp_path / "reports"
        env = os.environ.copy()
        pythonpath = str(PROJECT_ROOT / "src")
        if env.get("PYTHONPATH"):
            pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
        env["PYTHONPATH"] = pythonpath

        result = subprocess.run(
            [
                sys.executable,
                "scripts/validate_release_candidate.py",
                "--base-url",
                f"http://127.0.0.1:{server.server_port}",
                "--skip-compose",
                "--skip-evidence",
                "--report-dir",
                str(report_dir),
                "--report-name",
                "backend-harness-smoke.json",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, result.stderr
        report_path = Path(result.stdout.strip())
        assert report_path.name == "backend-harness-smoke.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["evidence_files"] == {"skipped": True}
        assert all(check["ok"] for check in report["checks"])
    finally:
        server.shutdown()
        server.server_close()


def test_validate_release_candidate_rejects_mismatched_evidence_git_sha(tmp_path: Path):
    _seed_evidence_files()
    evidence_dir = PROJECT_ROOT / "reports" / "benchmarks" / "latest"
    dashboard_path = evidence_dir / "benchmark_dashboard.json"
    release_path = evidence_dir / "release_evidence.json"
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    release_evidence = json.loads(release_path.read_text(encoding="utf-8"))
    dashboard["provenance"]["git_sha"] = "deadbeef"
    release_evidence["provenance"]["git_sha"] = "deadbeef"
    dashboard_path.write_text(json.dumps(dashboard), encoding="utf-8")
    release_path.write_text(json.dumps(release_evidence), encoding="utf-8")

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        report_dir = tmp_path / "reports"
        env = os.environ.copy()
        pythonpath = str(PROJECT_ROOT / "src")
        if env.get("PYTHONPATH"):
            pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
        env["PYTHONPATH"] = pythonpath

        result = subprocess.run(
            [
                sys.executable,
                "scripts/validate_release_candidate.py",
                "--base-url",
                f"http://127.0.0.1:{server.server_port}",
                "--skip-compose",
                "--report-dir",
                str(report_dir),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 1
        report = json.loads(Path(result.stdout.strip()).read_text(encoding="utf-8"))
        assert "does not match expected" in report["evidence_files"]["error"]
    finally:
        server.shutdown()
        server.server_close()


def test_validate_release_candidate_rejects_failing_quality_gate(tmp_path: Path):
    _seed_evidence_files()
    evidence_dir = PROJECT_ROOT / "reports" / "benchmarks" / "latest"
    release_path = evidence_dir / "release_evidence.json"
    release_evidence = json.loads(release_path.read_text(encoding="utf-8"))
    release_evidence["quality_gate_passed"] = False
    release_evidence["quality_failures"] = [
        {"pipeline": "quality_seedvc", "metric": "mcd_mean", "value": 750.51, "target_status": "fail"}
    ]
    release_path.write_text(json.dumps(release_evidence), encoding="utf-8")

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        report_dir = tmp_path / "reports"
        env = os.environ.copy()
        pythonpath = str(PROJECT_ROOT / "src")
        if env.get("PYTHONPATH"):
            pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
        env["PYTHONPATH"] = pythonpath

        result = subprocess.run(
            [
                sys.executable,
                "scripts/validate_release_candidate.py",
                "--base-url",
                f"http://127.0.0.1:{server.server_port}",
                "--skip-compose",
                "--report-dir",
                str(report_dir),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 1
        report = json.loads(Path(result.stdout.strip()).read_text(encoding="utf-8"))
        assert "quality gate failed" in report["evidence_files"]["error"]
    finally:
        server.shutdown()
        server.server_close()
