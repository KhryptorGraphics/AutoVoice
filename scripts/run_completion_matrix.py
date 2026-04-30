#!/usr/bin/env python3
"""Run and artifact the AutoVoice production-completion verification matrix."""

from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import time
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

REQUIRED_TRT_ENGINE_FILES = (
    "content_extractor.trt",
    "pitch_extractor.trt",
    "decoder.trt",
    "vocoder.trt",
)

SUPPORTED_REAL_AUDIO_E2E_LANES = (
    (
        "youtube-ingest-real-audio-e2e",
        ["tests/test_youtube_ingest_real_audio_e2e.py", "-k", "not live_youtube"],
    ),
    (
        "conversion-e2e",
        ["tests/test_adapter_integration_e2e.py", "tests/test_e2e_pipeline.py"],
    ),
    (
        "karaoke-websocket-e2e",
        ["tests/test_karaoke_websocket_context.py", "tests/test_karaoke_websocket_events.py"],
    ),
    (
        "voice-profile-diarization-e2e",
        ["tests/test_e2e_diarization.py"],
    ),
    (
        "voice-profile-training-e2e",
        ["tests/test_voice_profile_training_e2e.py"],
    ),
    (
        "benchmark-audio-e2e",
        ["tests/test_quality_benchmarking_sota.py", "tests/evaluation/test_benchmark_manifest_reporting.py"],
    ),
)

E2E_ENVIRONMENT_GATES = {
    "CUDA": {
        "lane": "jetson-cuda-tensorrt",
        "owner": "hardware-runner",
        "action": "Run completion matrix with --hardware or --full on a Jetson/CUDA host.",
    },
    "Diarization": {
        "lane": "voice-profile-diarization-e2e",
        "owner": "backend-runtime",
        "action": "Run the diarization E2E lane with the diarization service and real audio fixtures available.",
    },
    "Training": {
        "lane": "voice-profile-training-e2e",
        "owner": "training-runtime",
        "action": "Run the voice-profile training E2E lane with training services and sample fixtures available.",
    },
    "Karaoke": {
        "lane": "karaoke-websocket-e2e",
        "owner": "frontend-runtime",
        "action": "Run the live browser karaoke websocket lane against a live backend.",
    },
    "Jetson": {
        "lane": "jetson-cuda-tensorrt",
        "owner": "hardware-runner",
        "action": "Publish CUDA/TensorRT validation evidence from the self-hosted Jetson runner.",
    },
    "benchmark audio files": {
        "lane": "benchmark-audio-e2e",
        "owner": "benchmark-runtime",
        "action": "Run benchmark-audio validation with release-quality fixture media mounted.",
    },
    "trained model": {
        "lane": "trained-model-e2e",
        "owner": "model-runtime",
        "action": "Run model-backed E2E checks after restoring trained model fixtures.",
    },
    "TRT engines": {
        "lane": "jetson-cuda-tensorrt",
        "owner": "hardware-runner",
        "action": "Build or restore TensorRT engines and rerun the hardware lane.",
    },
    "AUTOVOICE_TRT_ENGINE_DIR": {
        "lane": "jetson-cuda-tensorrt",
        "owner": "hardware-runner",
        "action": "Set AUTOVOICE_TRT_ENGINE_DIR to a directory containing TensorRT engine artifacts.",
    },
    "tensorrt": {
        "lane": "jetson-cuda-tensorrt",
        "owner": "hardware-runner",
        "action": "Install TensorRT in the hardware runner environment and rerun the TensorRT tests.",
    },
    "ContentVec": {
        "lane": "jetson-cuda-tensorrt",
        "owner": "model-runtime",
        "action": "Use the checkpoint-backed TensorRT export path for ContentVec instead of lazy ONNX export.",
    },
    "Sample upload failed": {
        "lane": "voice-profile-training-e2e",
        "owner": "training-runtime",
        "action": "Run with the live backend sample-upload endpoint available and preserve the generated failure report.",
    },
    "Job creation failed": {
        "lane": "voice-profile-training-e2e",
        "owner": "training-runtime",
        "action": "Run with the live training job endpoint available and preserve the generated failure report.",
    },
}


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


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _gitnexus_analyze_command() -> list[str]:
    if shutil.which("gitnexus"):
        return ["gitnexus", "analyze", "."]
    return ["npx", "-y", "gitnexus", "analyze", "."]


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
    existing_pythonpath = command_env.get("PYTHONPATH", "")
    pythonpath_entries = [str(PROJECT_ROOT), str(SRC_DIR)]
    pythonpath_entries.extend(
        entry for entry in existing_pythonpath.split(os.pathsep)
        if entry and entry not in pythonpath_entries
    )
    command_env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    command_env.setdefault("PYTHONNOUSERSITE", "1")
    command_env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    if env:
        command_env.update(env)

    try:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=command_env,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = process.communicate()
            output = (stdout or "") + (stderr or "")
            log_path.write_text(output, encoding="utf-8")
            return LaneResult(
                name=name,
                status="failed",
                command=command,
                duration_seconds=timeout,
                artifacts={"log": _display_path(log_path)},
                details={"reason": f"timed out after {timeout}s"},
            )
        completed_returncode = process.returncode
        output = stdout + stderr
        log_path.write_text(output, encoding="utf-8")
        status = "passed" if completed_returncode == 0 else "failed"
        return LaneResult(
            name=name,
            status=status,
            command=command,
            duration_seconds=(_now() - started).total_seconds(),
            returncode=completed_returncode,
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


def audit_priority_skips() -> LaneResult:
    findings: list[dict[str, Any]] = []
    allowed: list[dict[str, Any]] = []
    environment_gate_evidence: list[dict[str, Any]] = []
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
                gate = next(
                    (metadata for token, metadata in E2E_ENVIRONMENT_GATES.items() if token.lower() in line.lower()),
                    None,
                )
                environment_gate_evidence.append({
                    **entry,
                    "explained": gate is not None,
                    "lane": gate["lane"] if gate else "unclassified-e2e-gate",
                    "owner": gate["owner"] if gate else "unassigned",
                    "action": gate["action"] if gate else "Classify this skip with a concrete service or hardware lane.",
                })

    unexplained = [entry for entry in environment_gate_evidence if not entry["explained"]]
    findings.extend({"file": entry["file"], "line": entry["line"], "text": f"unexplained environment gate: {entry['text']}"} for entry in unexplained)

    status = "passed" if not findings else "failed"
    return LaneResult(
        name="priority-skip-audit",
        status=status,
        details={
            "disallowed_tokens": list(DISALLOWED_SKIP_TOKENS),
            "findings": findings,
            "allowed_environment_gates": allowed,
            "environment_gate_evidence": environment_gate_evidence,
        },
    )


def write_skip_audit(report_dir: Path) -> LaneResult:
    result = audit_priority_skips()
    path = report_dir / "skip_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.details, indent=2), encoding="utf-8")
    result.artifacts["report"] = _display_path(path)
    return result


def _bundle_payload(title: str, *, sample_count: int, similarity: float, pitch: float, mcd: float, latency: float, fixture_tier: str) -> dict:
    return {
        "title": title,
        "fixture_tier": fixture_tier,
        "fixture_suite": "completion-smoke-real-audio",
        "summary": {
            "sample_count": sample_count,
            "fixture_tier": fixture_tier,
            "fixture_suite": "completion-smoke-real-audio",
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
            fixture_tier="smoke",
        ),
        "realtime": _bundle_payload(
            "realtime smoke evidence",
            sample_count=2,
            similarity=0.88,
            pitch=0.91,
            mcd=4.4,
            latency=42.0,
            fixture_tier="smoke",
        ),
        "hq_svc": _bundle_payload(
            "hq_svc candidate smoke evidence",
            sample_count=2,
            similarity=0.89,
            pitch=0.91,
            mcd=4.4,
            latency=140.0,
            fixture_tier="smoke",
        ),
        "nsf_harmonic_enhancement": _bundle_payload(
            "nsf_harmonic_enhancement candidate smoke evidence",
            sample_count=2,
            similarity=0.88,
            pitch=0.90,
            mcd=4.6,
            latency=130.0,
            fixture_tier="smoke",
        ),
        "ecapa2_encoder": _bundle_payload(
            "ecapa2_encoder candidate smoke evidence",
            sample_count=2,
            similarity=0.90,
            pitch=0.91,
            mcd=4.3,
            latency=150.0,
            fixture_tier="smoke",
        ),
        "pupu_vocoder_refinement": _bundle_payload(
            "pupu_vocoder_refinement candidate smoke evidence",
            sample_count=2,
            similarity=0.88,
            pitch=0.90,
            mcd=4.5,
            latency=155.0,
            fixture_tier="smoke",
        ),
        "quality_shortcut": _bundle_payload(
            "quality_shortcut candidate smoke evidence",
            sample_count=2,
            similarity=0.88,
            pitch=0.90,
            mcd=4.5,
            latency=90.0,
            fixture_tier="smoke",
        ),
        "realtime_meanvc": _bundle_payload(
            "realtime_meanvc candidate smoke evidence",
            sample_count=2,
            similarity=0.86,
            pitch=0.90,
            mcd=4.7,
            latency=48.0,
            fixture_tier="smoke",
        ),
    }
    paths: dict[str, Path] = {}
    for pipeline, payload in bundles.items():
        path = bundle_dir / f"{pipeline}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        paths[pipeline] = path
    return paths


def run_benchmark_evidence(report_dir: Path, *, timeout: int, dashboard_dir: Path) -> list[LaneResult]:
    bundle_dir = dashboard_dir / "source_bundles"
    bundle_paths = generate_smoke_benchmark_bundles(bundle_dir)
    bundle_args: list[str] = []
    for pipeline, path in bundle_paths.items():
        bundle_args.extend(["--bundle", f"{pipeline}={path}"])

    build = _run_command(
        "benchmark-dashboard-build",
        [sys.executable, "scripts/build_benchmark_dashboard.py", *bundle_args, "--output-dir", str(dashboard_dir)],
        report_dir=report_dir,
        timeout=timeout,
    )
    publish = LaneResult(
        "benchmark-dashboard-publish-latest",
        "skipped",
        details={"reason": "smoke benchmark evidence is not release-grade; latest is only published from --benchmark-report"},
    )
    validate_dashboard = _run_command(
        "benchmark-dashboard-validate",
        [
            sys.executable,
            "scripts/validate_benchmark_dashboard.py",
            "--dashboard",
            str(dashboard_dir / "benchmark_dashboard.json"),
            "--release-evidence",
            str(dashboard_dir / "release_evidence.json"),
            "--current-git-sha",
        ],
        report_dir=report_dir,
        timeout=timeout,
    )
    validate_experimental = _run_command(
        "experimental-evidence-validate",
        [sys.executable, "scripts/validate_experimental_evidence.py"],
        report_dir=report_dir,
        timeout=timeout,
    )
    return [build, publish, validate_dashboard, validate_experimental]


def run_benchmark_evidence_from_report(
    report_dir: Path,
    *,
    timeout: int,
    benchmark_report: Path,
    dashboard_dir: Path,
) -> list[LaneResult]:
    build = _run_command(
        "benchmark-dashboard-build-real",
        [
            sys.executable,
            "scripts/build_benchmark_dashboard.py",
            "--comprehensive-report",
            str(benchmark_report),
            "--fixture-tier",
            "quality",
            "--fixture-suite",
            "completion-real-benchmark",
            "--output-dir",
            str(dashboard_dir),
        ],
        report_dir=report_dir,
        timeout=timeout,
    )
    publish = _publish_latest_benchmark_evidence(dashboard_dir) if build.ok else LaneResult(
        "benchmark-dashboard-publish-latest",
        "skipped",
        details={"reason": "benchmark dashboard build failed"},
    )
    validate_dashboard = _run_command(
        "benchmark-dashboard-validate",
        [
            sys.executable,
            "scripts/validate_benchmark_dashboard.py",
            "--dashboard",
            str(dashboard_dir / "benchmark_dashboard.json"),
            "--release-evidence",
            str(dashboard_dir / "release_evidence.json"),
            "--current-git-sha",
            "--release-grade",
        ],
        report_dir=report_dir,
        timeout=timeout,
    )
    validate_experimental = _run_command(
        "experimental-evidence-validate",
        [sys.executable, "scripts/validate_experimental_evidence.py"],
        report_dir=report_dir,
        timeout=timeout,
    )
    return [build, publish, validate_dashboard, validate_experimental]


def run_supported_real_audio_e2e_lanes(report_dir: Path, *, timeout: int) -> list[LaneResult]:
    """Run the supported full-mode real-audio E2E lanes and keep per-lane logs."""
    results: list[LaneResult] = []
    for lane_name, test_files in SUPPORTED_REAL_AUDIO_E2E_LANES:
        results.append(
            _run_command(
                lane_name,
                [sys.executable, "-m", "pytest", *test_files, "-q", "--tb=short"],
                report_dir=report_dir,
                timeout=timeout,
            )
        )
    return results


def run_live_youtube_smoke_lane(report_dir: Path, *, timeout: int) -> LaneResult:
    """Run the opt-in live YouTube ingest smoke lane."""
    live_url = os.environ.get("AUTOVOICE_LIVE_YOUTUBE_URL")
    if not live_url:
        return LaneResult(
            name="live-youtube-ingest-smoke",
            status="blocked",
            details={
                "reason": "AUTOVOICE_LIVE_YOUTUBE_URL is required when --live-youtube is set.",
                "owner": "operator",
                "action": "Set AUTOVOICE_LIVE_YOUTUBE_URL to an operator-owned test video and rerun.",
            },
        )

    return _run_command(
        "live-youtube-ingest-smoke",
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_youtube_ingest_real_audio_e2e.py",
            "-q",
            "--tb=short",
            "-k",
            "live_youtube_ingest_smoke",
        ],
        report_dir=report_dir,
        timeout=timeout,
        env={"AUTOVOICE_LIVE_YOUTUBE_URL": live_url},
    )


def run_hosted_preflight(report_dir: Path, *, timeout: int, full: bool, report_path: Path) -> LaneResult:
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
ProxyPass /ready http://127.0.0.1:10600/ready
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
                "--skip-apache-configtest",
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


def _available_port(preferred: int) -> str:
    """Return preferred if free, otherwise ask the OS for an available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", preferred))
            return str(preferred)
        except OSError:
            pass

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _discover_trt_engine_dirs() -> list[Path]:
    """Return candidate TensorRT engine directories, preferring complete suites."""

    candidates: set[Path] = set()
    env_dir = os.environ.get("AUTOVOICE_TRT_ENGINE_DIR")
    if env_dir:
        candidates.add(Path(env_dir))
    for root in (PROJECT_ROOT / "models", PROJECT_ROOT / "data", PROJECT_ROOT / "reports"):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in {".trt", ".engine", ".plan"}:
                candidates.add(path.parent)

    def rank(path: Path) -> tuple[int, str]:
        complete = all((path / name).exists() for name in REQUIRED_TRT_ENGINE_FILES)
        return (0 if complete else 1, str(path))

    return sorted(candidates, key=rank)


def run_tensorrt_engine_suite(report_dir: Path, *, timeout: int) -> LaneResult:
    candidates = _discover_trt_engine_dirs()
    complete = [
        path for path in candidates
        if all((path / name).exists() for name in REQUIRED_TRT_ENGINE_FILES)
    ]
    if not complete:
        missing_by_dir = {
            _display_path(path): [
                name for name in REQUIRED_TRT_ENGINE_FILES if not (path / name).exists()
            ]
            for path in candidates
        }
        return LaneResult(
            name="tensorrt-engine-suite",
            status="blocked",
            details={
                "reason": "No complete TensorRT engine suite found for the hardware pytest lane.",
                "required_files": list(REQUIRED_TRT_ENGINE_FILES),
                "candidate_dirs": [_display_path(path) for path in candidates],
                "missing_by_dir": missing_by_dir,
                "action": (
                    "Build or restore content_extractor.trt, pitch_extractor.trt, "
                    "decoder.trt, and vocoder.trt, then rerun with AUTOVOICE_TRT_ENGINE_DIR."
                ),
            },
        )

    return _run_command(
        "tensorrt-engine-suite",
        [sys.executable, "-m", "pytest", "tests/test_tensorrt_pipeline_sota.py", "-q", "--tb=short"],
        report_dir=report_dir,
        timeout=timeout,
        env={"AUTOVOICE_TRT_ENGINE_DIR": str(complete[0])},
    )


def run_tensorrt_parity_benchmark(report_dir: Path, *, timeout: int, report_path: Path) -> LaneResult:
    result = _run_command(
        "tensorrt-checkpoint-parity",
        [
            sys.executable,
            "scripts/benchmark_tensorrt_parity.py",
            "--output",
            str(report_path),
        ],
        report_dir=report_dir,
        timeout=timeout,
    )
    if report_path.exists():
        result.artifacts["parity_report"] = _display_path(report_path)
    return result


def run_real_compose(
    report_dir: Path,
    *,
    timeout: int,
    base_url: str,
    evidence_dir: Path,
    platform_report_dir: Path,
) -> list[LaneResult]:
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
        "FRONTEND_PORT": os.environ.get("FRONTEND_PORT", _available_port(13000)),
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
                str(platform_report_dir),
                "--evidence-dir",
                str(evidence_dir),
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
            ["docker", "compose", "-p", project_name, "-f", "docker-compose.yaml", "down"],
            report_dir=report_dir,
            timeout=timeout,
            env=env,
        )
    )
    return results


def skipped_lane(
    name: str,
    reason: str,
    *,
    owner: str = "unassigned",
    action: str = "Rerun the lane in an environment where its prerequisites are available.",
) -> LaneResult:
    return LaneResult(
        name=name,
        status="skipped",
        details={"reason": reason, "owner": owner, "action": action},
    )


def attach_hardware_gate_report(result: LaneResult, platform_report_dir: Path) -> LaneResult:
    gate_report = platform_report_dir / "hardware-lane-gates.json"
    if not gate_report.exists():
        return result
    result.artifacts["hardware_lane_gates"] = _display_path(gate_report)
    try:
        result.details["hardware_lane_gates"] = json.loads(gate_report.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        result.details["hardware_lane_gates_error"] = "invalid JSON"
    return result


def build_matrix(args: argparse.Namespace) -> dict[str, Any]:
    report_dir = args.output_dir
    if not report_dir.is_absolute():
        report_dir = PROJECT_ROOT / report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    platform_report_dir = _resolve_repo_path(args.platform_report_dir)
    platform_report_dir.mkdir(parents=True, exist_ok=True)
    tensorrt_parity_report = _resolve_repo_path(args.tensorrt_parity_report)
    tensorrt_parity_report.parent.mkdir(parents=True, exist_ok=True)
    benchmark_archive_dir = _benchmark_archive_dir(report_dir)

    lanes: list[LaneResult] = []
    if args.refresh_gitnexus:
        lanes.append(
            _run_command(
                "gitnexus-analyze",
                _gitnexus_analyze_command(),
                report_dir=report_dir,
                timeout=args.timeout,
            )
        )

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
    if args.benchmark_report:
        lanes.extend(
            run_benchmark_evidence_from_report(
                report_dir,
                timeout=args.timeout,
                benchmark_report=args.benchmark_report,
                dashboard_dir=benchmark_archive_dir,
            )
        )
    else:
        lanes.extend(run_benchmark_evidence(report_dir, timeout=args.timeout, dashboard_dir=benchmark_archive_dir))

    if args.real_audio:
        lanes.extend(run_supported_real_audio_e2e_lanes(report_dir, timeout=args.timeout))
    else:
        lanes.append(
            skipped_lane(
                "real-audio-e2e-lanes",
                "pass --real-audio on Thor to run local deterministic real-audio E2E lanes without Docker",
                owner="backend-runtime",
                action="Run completion matrix with --real-audio in autovoice-thor.",
            )
        )

    if args.live_youtube:
        lanes.append(run_live_youtube_smoke_lane(report_dir, timeout=args.timeout))
    else:
        lanes.append(
            skipped_lane(
                "live-youtube-ingest-smoke",
                "pass --live-youtube with AUTOVOICE_LIVE_YOUTUBE_URL set to run the network-backed YouTube smoke lane",
                owner="operator",
                action="Use only operator-owned media; this lane is intentionally opt-in.",
            )
        )

    lanes.append(
        run_hosted_preflight(
            report_dir,
            timeout=args.timeout,
            full=args.full_hosted_preflight,
            report_path=platform_report_dir / "hosted-preflight.json",
        )
    )

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
        lanes.extend(
            run_real_compose(
                report_dir,
                timeout=args.timeout,
                base_url=args.base_url,
                evidence_dir=benchmark_archive_dir,
                platform_report_dir=platform_report_dir,
            )
        )
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
            attach_hardware_gate_report(
                _run_command(
                    "jetson-cuda-tensorrt",
                    ["bash", "scripts/validate_cuda_stack.sh", "--output-dir", str(platform_report_dir)],
                    report_dir=report_dir,
                    timeout=args.timeout,
                ),
                platform_report_dir,
            )
        )
        lanes.append(run_tensorrt_engine_suite(report_dir, timeout=args.timeout))
        lanes.append(run_tensorrt_parity_benchmark(report_dir, timeout=args.timeout, report_path=tensorrt_parity_report))
    else:
        lanes.append(skipped_lane(
            "jetson-cuda-tensorrt",
            "pass --hardware or --full on Jetson/CUDA/TensorRT hosts",
            owner="hardware-runner",
            action="Run completion matrix with --hardware or --full on a Jetson/CUDA host.",
        ))
        lanes.append(skipped_lane(
            "tensorrt-engine-suite",
            "pass --hardware or --full with AUTOVOICE_TRT_ENGINE_DIR set",
            owner="hardware-runner",
            action="Set AUTOVOICE_TRT_ENGINE_DIR to a complete TensorRT engine suite and rerun.",
        ))
        lanes.append(skipped_lane(
            "tensorrt-checkpoint-parity",
            "pass --hardware or --full with checkpoint-aligned TensorRT engine metadata",
            owner="hardware-runner",
            action="Restore checkpoint-aligned TensorRT metadata and rerun the parity benchmark.",
        ))

    matrix = {
        "schema_version": 1,
        "generated_at": _now().isoformat(),
        "git_sha": _git_sha(),
        "mode": "full" if args.full else "smoke",
        "ok": all(lane.ok for lane in lanes),
        "artifacts": {
            "report_dir": str(report_dir),
            "platform_report_dir": str(platform_report_dir),
            "benchmark_archive_dir": str(benchmark_archive_dir),
            "benchmark_latest_dir": str(PROJECT_ROOT / "reports" / "benchmarks" / "latest"),
            "tensorrt_parity_report": str(tensorrt_parity_report),
        },
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


def _benchmark_archive_dir(report_dir: Path) -> Path:
    git_sha = (_git_sha() or "unknown")[:12]
    date_stamp = _now().strftime("%Y%m%d")
    archive_dir = PROJECT_ROOT / "reports" / "benchmarks" / f"{date_stamp}-{git_sha}"
    report_dir.mkdir(parents=True, exist_ok=True)
    return archive_dir


def _publish_latest_benchmark_evidence(archive_dir: Path) -> LaneResult:
    started = time.time()
    latest_dir = PROJECT_ROOT / "reports" / "benchmarks" / "latest"
    try:
        latest_dir.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, str] = {}
        for name in ("benchmark_dashboard.json", "release_evidence.json", "benchmark_dashboard.md"):
            source = archive_dir / name
            if source.exists():
                dest = latest_dir / name
                shutil.copy2(source, dest)
                artifacts[name] = str(dest)
        return LaneResult(
            "benchmark-dashboard-publish-latest",
            "passed",
            duration_seconds=round(time.time() - started, 3),
            artifacts=artifacts,
            details={"archive_dir": str(archive_dir), "latest_dir": str(latest_dir)},
        )
    except Exception as exc:
        return LaneResult(
            "benchmark-dashboard-publish-latest",
            "failed",
            duration_seconds=round(time.time() - started, 3),
            details={"archive_dir": str(archive_dir), "error": str(exc)},
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/completion/latest"))
    parser.add_argument("--platform-report-dir", type=Path, default=Path("reports/platform"))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--base-url", default="http://127.0.0.1:10001")
    parser.add_argument("--benchmark-report", type=Path, help="Comprehensive benchmark report to use for release evidence.")
    parser.add_argument(
        "--tensorrt-parity-report",
        type=Path,
        default=Path("reports/benchmarks/tensorrt-parity/latest/parity_report.json"),
        help="Parity report path for the TensorRT checkpoint comparison lane.",
    )
    parser.add_argument("--refresh-gitnexus", action="store_true", help="Run npx gitnexus analyze as a lane")
    parser.add_argument("--frontend", action="store_true", help="Run frontend lint/type/build/browser smoke lanes")
    parser.add_argument("--real-audio", action="store_true", help="Run deterministic local real-audio E2E lanes without Docker")
    parser.add_argument("--live-youtube", action="store_true", help="Run opt-in live YouTube ingest smoke using AUTOVOICE_LIVE_YOUTUBE_URL")
    parser.add_argument("--real-compose", action="store_true", help="Boot the real compose backend/frontend stack")
    parser.add_argument("--hardware", action="store_true", help="Run Jetson/CUDA/TensorRT validation lanes")
    parser.add_argument("--full-hosted-preflight", action="store_true", help="Run hosted preflight with DNS/TLS instead of local mocks")
    parser.add_argument("--full", action="store_true", help="Enable frontend, real compose, hardware, hosted, and GitNexus lanes")
    args = parser.parse_args(argv)
    if args.full:
        args.refresh_gitnexus = True
        args.frontend = True
        args.real_audio = True
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
