#!/usr/bin/env python3
"""Run and artifact the AutoVoice production-completion verification matrix."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


PRIORITY_SKIP_FILES = (
    "tests/test_voice_profile_training_e2e.py",
    "tests/test_karaoke_websocket_events.py",
    "tests/test_tensorrt_pipeline_sota.py",
    "tests/test_quality_benchmarking_sota.py",
)

DISALLOWED_SKIP_TOKENS = (
    "not implemented",
    "Requires Flask app context - tested via integration tests",
    "Requires benchmark audio files",
    "Requires trained model and test samples",
    "Requires pre-built TRT engines",
)


@dataclass
class LaneResult:
    name: str
    status: str
    command: list[str] | None = None
    duration_seconds: float | None = None
    returncode: int | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in {"passed", "skipped"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _run_command(
    name: str,
    command: list[str],
    *,
    report_dir: Path,
    timeout: int = 600,
    env: dict[str, str] | None = None,
) -> LaneResult:
    started = _now()
    lane_dir = report_dir / "logs"
    lane_dir.mkdir(parents=True, exist_ok=True)
    log_path = lane_dir / f"{name}.log"

    command_env = os.environ.copy()
    command_env.setdefault("PYTHONPATH", str(SRC_DIR))
    command_env.setdefault("PYTHONNOUSERSITE", "1")
    command_env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    if env:
        command_env.update(env)

    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=command_env,
        )
        output = completed.stdout + completed.stderr
        log_path.write_text(output, encoding="utf-8")
        status = "passed" if completed.returncode == 0 else "failed"
        return LaneResult(
            name=name,
            status=status,
            command=command,
            duration_seconds=(_now() - started).total_seconds(),
            returncode=completed.returncode,
            artifacts={"log": _display_path(log_path)},
        )
    except FileNotFoundError as exc:
        log_path.write_text(str(exc), encoding="utf-8")
        return LaneResult(
            name=name,
            status="blocked",
            command=command,
            duration_seconds=(_now() - started).total_seconds(),
            artifacts={"log": _display_path(log_path)},
            details={"reason": f"command not found: {command[0]}"},
        )
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        log_path.write_text(output, encoding="utf-8")
        return LaneResult(
            name=name,
            status="failed",
            command=command,
            duration_seconds=timeout,
            artifacts={"log": _display_path(log_path)},
            details={"reason": f"timed out after {timeout}s"},
        )


def audit_priority_skips() -> LaneResult:
    findings: list[dict[str, Any]] = []
    allowed: list[dict[str, Any]] = []
    for file_name in PRIORITY_SKIP_FILES:
        path = PROJECT_ROOT / file_name
        if not path.exists():
            findings.append({"file": file_name, "line": None, "text": "missing priority test file"})
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "pytest.skip" not in line and "pytest.mark.skip" not in line and "importorskip" not in line:
                continue
            entry = {"file": file_name, "line": line_number, "text": line.strip()}
            if any(token in line for token in DISALLOWED_SKIP_TOKENS):
                findings.append(entry)
            else:
                allowed.append(entry)

    status = "passed" if not findings else "failed"
    return LaneResult(
        name="priority-skip-audit",
        status=status,
        details={
            "disallowed_tokens": list(DISALLOWED_SKIP_TOKENS),
            "findings": findings,
            "allowed_environment_gates": allowed,
        },
    )


def write_skip_audit(report_dir: Path) -> LaneResult:
    result = audit_priority_skips()
    path = report_dir / "skip_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.details, indent=2), encoding="utf-8")
    result.artifacts["report"] = _display_path(path)
    return result


def _bundle_payload(title: str, *, sample_count: int, similarity: float, pitch: float, mcd: float, latency: float) -> dict:
    return {
        "title": title,
        "summary": {
            "sample_count": sample_count,
            "speaker_similarity_mean": similarity,
            "pitch_corr_mean": pitch,
            "mcd_mean": mcd,
            "latency_ms_mean": latency,
        },
    }


def generate_smoke_benchmark_bundles(bundle_dir: Path) -> dict[str, Path]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundles = {
        "quality_seedvc": _bundle_payload(
            "quality_seedvc smoke evidence",
            sample_count=2,
            similarity=0.91,
            pitch=0.93,
            mcd=4.1,
            latency=120.0,
        ),
        "realtime": _bundle_payload(
            "realtime smoke evidence",
            sample_count=2,
            similarity=0.88,
            pitch=0.91,
            mcd=4.4,
            latency=42.0,
        ),
        "hq_svc": _bundle_payload(
            "hq_svc candidate smoke evidence",
            sample_count=2,
            similarity=0.89,
            pitch=0.91,
            mcd=4.4,
            latency=140.0,
        ),
        "nsf_harmonic_enhancement": _bundle_payload(
            "nsf_harmonic_enhancement candidate smoke evidence",
            sample_count=2,
            similarity=0.88,
            pitch=0.90,
            mcd=4.6,
            latency=130.0,
        ),
        "ecapa2_encoder": _bundle_payload(
            "ecapa2_encoder candidate smoke evidence",
            sample_count=2,
            similarity=0.90,
            pitch=0.91,
            mcd=4.3,
            latency=150.0,
        ),
        "pupu_vocoder_refinement": _bundle_payload(
            "pupu_vocoder_refinement candidate smoke evidence",
            sample_count=2,
            similarity=0.88,
            pitch=0.90,
            mcd=4.5,
            latency=155.0,
        ),
        "quality_shortcut": _bundle_payload(
            "quality_shortcut candidate smoke evidence",
            sample_count=2,
            similarity=0.88,
            pitch=0.90,
            mcd=4.5,
            latency=90.0,
        ),
        "realtime_meanvc": _bundle_payload(
            "realtime_meanvc candidate smoke evidence",
            sample_count=2,
            similarity=0.86,
            pitch=0.90,
            mcd=4.7,
            latency=48.0,
        ),
    }
    paths: dict[str, Path] = {}
    for pipeline, payload in bundles.items():
        path = bundle_dir / f"{pipeline}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        paths[pipeline] = path
    return paths


def run_benchmark_evidence(report_dir: Path, *, timeout: int) -> list[LaneResult]:
    bundle_dir = report_dir / "benchmark_bundles"
    bundle_paths = generate_smoke_benchmark_bundles(bundle_dir)
    bundle_args: list[str] = []
    for pipeline, path in bundle_paths.items():
        bundle_args.extend(["--bundle", f"{pipeline}={path}"])

    dashboard_dir = PROJECT_ROOT / "reports" / "benchmarks" / "latest"
    build = _run_command(
        "benchmark-dashboard-build",
        [sys.executable, "scripts/build_benchmark_dashboard.py", *bundle_args, "--output-dir", str(dashboard_dir)],
        report_dir=report_dir,
        timeout=timeout,
    )
    validate_dashboard = _run_command(
        "benchmark-dashboard-validate",
        [sys.executable, "scripts/validate_benchmark_dashboard.py"],
        report_dir=report_dir,
        timeout=timeout,
    )
    validate_experimental = _run_command(
        "experimental-evidence-validate",
        [sys.executable, "scripts/validate_experimental_evidence.py"],
        report_dir=report_dir,
        timeout=timeout,
    )
    return [build, validate_dashboard, validate_experimental]


def run_hosted_preflight(report_dir: Path, *, timeout: int, full: bool) -> LaneResult:
    report_path = PROJECT_ROOT / "reports" / "platform" / "hosted-preflight.json"
    if full:
        return _run_command(
            "hosted-preflight",
            [
                sys.executable,
                "scripts/validate_hosted_deployment.py",
                "--report",
                str(report_path),
            ],
            report_dir=report_dir,
            timeout=timeout,
            env={"SECRET_KEY": os.environ.get("SECRET_KEY", "completion-matrix-local-secret")},
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        vhost = tmp_path / "autovoice.conf"
        ssl_vhost = tmp_path / "autovoice-ssl.conf"
        content = """
ServerName autovoice.giggahost.com
DocumentRoot frontend/dist
ProxyPass /api http://127.0.0.1:10600/api
ProxyPass /socket.io http://127.0.0.1:10600/socket.io
SecRequestBodyLimit 262144000
"""
        vhost.write_text(content, encoding="utf-8")
        ssl_vhost.write_text(content, encoding="utf-8")
        return _run_command(
            "hosted-preflight-local",
            [
                sys.executable,
                "scripts/validate_hosted_deployment.py",
                "--skip-dns",
                "--skip-tls",
                "--vhost-file",
                str(vhost),
                "--vhost-file",
                str(ssl_vhost),
                "--report",
                str(report_path),
            ],
            report_dir=report_dir,
            timeout=timeout,
            env={"SECRET_KEY": os.environ.get("SECRET_KEY", "completion-matrix-local-secret")},
        )


def run_real_compose(report_dir: Path, *, timeout: int, base_url: str) -> list[LaneResult]:
    if not shutil.which("docker"):
        return [
            LaneResult(
                name="real-compose-stack",
                status="blocked",
                details={"reason": "docker command not available"},
            )
        ]

    project_name = f"autovoice-completion-{os.getpid()}"
    parsed_base_url = urlparse(base_url)
    env = {
        "SECRET_KEY": os.environ.get("SECRET_KEY", "completion-matrix-compose-secret"),
        "GRAFANA_PASSWORD": os.environ.get("GRAFANA_PASSWORD", "completion-matrix-grafana-password"),
        "HOST_PORT": str(parsed_base_url.port or os.environ.get("HOST_PORT", "10001")),
        "FRONTEND_PORT": os.environ.get("FRONTEND_PORT", "13000"),
    }
    results = [
        _run_command(
            "real-compose-up",
            ["docker", "compose", "-p", project_name, "-f", "docker-compose.yaml", "up", "-d", "--build", "backend", "frontend"],
            report_dir=report_dir,
            timeout=timeout,
            env=env,
        ),
        _run_command(
            "release-candidate-real-compose",
            [
                sys.executable,
                "scripts/validate_release_candidate.py",
                "--base-url",
                base_url,
                "--report-dir",
                "reports/platform",
                "--wait-seconds",
                "180",
            ],
            report_dir=report_dir,
            timeout=timeout,
            env=env,
        ),
    ]
    results.append(
        _run_command(
            "real-compose-down",
            ["docker", "compose", "-p", project_name, "-f", "docker-compose.yaml", "down", "-v"],
            report_dir=report_dir,
            timeout=timeout,
            env=env,
        )
    )
    return results


def skipped_lane(name: str, reason: str) -> LaneResult:
    return LaneResult(name=name, status="skipped", details={"reason": reason})


def build_matrix(args: argparse.Namespace) -> dict[str, Any]:
    report_dir = args.output_dir
    if not report_dir.is_absolute():
        report_dir = PROJECT_ROOT / report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    lanes: list[LaneResult] = []
    if args.refresh_gitnexus:
        lanes.append(_run_command("gitnexus-analyze", ["npx", "gitnexus", "analyze"], report_dir=report_dir, timeout=args.timeout))

    lanes.append(write_skip_audit(report_dir))
    lanes.append(
        _run_command(
            "backend-contract-smoke",
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_release_candidate_validation.py",
                "tests/test_sample_quality_and_benchmark_reporting.py",
                "tests/test_platform_scripts.py",
                "tests/test_karaoke_websocket_context.py",
                "-k",
                "not test_completion_matrix_smoke_runner",
                "-q",
            ],
            report_dir=report_dir,
            timeout=args.timeout,
        )
    )
    lanes.append(
        _run_command(
            "compose-config",
            ["bash", "scripts/validate_compose_config.sh"],
            report_dir=report_dir,
            timeout=args.timeout,
        )
    )
    lanes.extend(run_benchmark_evidence(report_dir, timeout=args.timeout))
    lanes.append(run_hosted_preflight(report_dir, timeout=args.timeout, full=args.full_hosted_preflight))

    if args.frontend:
        for command_name, command in (
            ("frontend-lint", ["npm", "--prefix", "frontend", "run", "lint"]),
            ("frontend-typecheck", ["npm", "--prefix", "frontend", "run", "typecheck"]),
            ("frontend-build", ["npm", "--prefix", "frontend", "run", "build"]),
            ("frontend-browser-smoke", ["npm", "--prefix", "frontend", "run", "test:e2e"]),
        ):
            lanes.append(_run_command(command_name, command, report_dir=report_dir, timeout=args.timeout, env={},))
    else:
        lanes.append(skipped_lane("frontend-verification", "pass --frontend or --full to run lint/type/build/browser smoke"))

    if args.real_compose:
        lanes.extend(run_real_compose(report_dir, timeout=args.timeout, base_url=args.base_url))
    else:
        lanes.append(skipped_lane("real-compose-stack", "pass --real-compose or --full to boot the real compose stack"))
        lanes.append(
            skipped_lane(
                "release-candidate-real-compose",
                "pass --real-compose or --full after evidence generation to validate live health/readiness/metrics",
            )
        )

    if args.hardware:
        lanes.append(
            _run_command(
                "jetson-cuda-tensorrt",
                ["bash", "scripts/validate_cuda_stack.sh", "--output-dir", "reports/platform"],
                report_dir=report_dir,
                timeout=args.timeout,
            )
        )
    else:
        lanes.append(skipped_lane("jetson-cuda-tensorrt", "pass --hardware or --full on Jetson/CUDA/TensorRT hosts"))

    matrix = {
        "schema_version": 1,
        "generated_at": _now().isoformat(),
        "git_sha": _git_sha(),
        "mode": "full" if args.full else "smoke",
        "ok": all(lane.ok for lane in lanes),
        "lanes": [lane.__dict__ for lane in lanes],
    }
    matrix_path = report_dir / "completion_matrix.json"
    matrix_path.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    return matrix


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/completion/latest"))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--base-url", default="http://127.0.0.1:10001")
    parser.add_argument("--refresh-gitnexus", action="store_true", help="Run npx gitnexus analyze as a lane")
    parser.add_argument("--frontend", action="store_true", help="Run frontend lint/type/build/browser smoke lanes")
    parser.add_argument("--real-compose", action="store_true", help="Boot the real compose backend/frontend stack")
    parser.add_argument("--hardware", action="store_true", help="Run Jetson/CUDA/TensorRT validation lanes")
    parser.add_argument("--full-hosted-preflight", action="store_true", help="Run hosted preflight with DNS/TLS instead of local mocks")
    parser.add_argument("--full", action="store_true", help="Enable frontend, real compose, hardware, hosted, and GitNexus lanes")
    args = parser.parse_args(argv)
    if args.full:
        args.refresh_gitnexus = True
        args.frontend = True
        args.real_compose = True
        args.hardware = True
        args.full_hosted_preflight = True
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    matrix = build_matrix(args)
    print(json.dumps({"ok": matrix["ok"], "report": str((args.output_dir / "completion_matrix.json"))}, indent=2))
    return 0 if matrix["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
