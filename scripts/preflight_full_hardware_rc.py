#!/usr/bin/env python3
"""Preflight the current-head full hardware RC evidence workflow."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_completion_matrix import REQUIRED_TRT_ENGINE_FILES, _discover_trt_engine_dirs


DEFAULT_HOSTED_BASE_URL = "https://autovoice.giggahost.com"
DEFAULT_ARTIST_SONG = PROJECT_ROOT / "tests/quality_samples/conor_maynard_pillowtalk.wav"
DEFAULT_USER_VOCALS = PROJECT_ROOT / "tests/quality_samples/william_singe_pillowtalk.wav"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _git_dirty() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"ok": False, "error": str(exc), "dirty": None, "entries": []}
    entries = [line for line in result.stdout.splitlines() if line.strip()]
    return {"ok": True, "dirty": bool(entries), "entries": entries}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fetch_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _endpoint_check(base_url: str, path: str, validator: Any) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    try:
        payload = _fetch_json(url)
    except HTTPError as exc:
        return {"url": url, "ok": False, "error": f"http {exc.code}"}
    except URLError as exc:
        return {"url": url, "ok": False, "error": str(exc.reason)}
    except json.JSONDecodeError as exc:
        return {"url": url, "ok": False, "error": f"invalid json: {exc}"}

    ok = bool(validator(payload))
    return {
        "url": url,
        "ok": ok,
        "payload": payload,
        "error": None if ok else f"unexpected payload: {payload!r}",
    }


def _hosted_checks(base_url: str, *, required: bool) -> dict[str, Any]:
    if not required:
        return {"ok": True, "skipped": True, "base_url": base_url, "checks": []}

    checks = [
        _endpoint_check(base_url, "/api/v1/health", lambda payload: payload.get("status") == "healthy"),
        _endpoint_check(base_url, "/ready", lambda payload: bool(payload.get("ready"))),
        _endpoint_check(base_url, "/api/v1/metrics", lambda payload: isinstance(payload, dict) and "error" not in payload),
    ]
    return {"ok": all(check["ok"] for check in checks), "base_url": base_url, "checks": checks}


def _benchmark_report_check(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"ok": False, "error": "missing benchmark report path"}

    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    result: dict[str, Any] = {"path": str(resolved), "ok": False}
    if not resolved.exists():
        result["error"] = "benchmark report not found"
        return result
    try:
        payload = _read_json(resolved)
    except (OSError, json.JSONDecodeError) as exc:
        result["error"] = f"invalid benchmark report: {exc}"
        return result

    pipelines = payload.get("pipelines")
    timestamp = payload.get("timestamp")
    mutable_latest = "reports/benchmarks/latest" in resolved.as_posix()
    result.update(
        {
            "timestamp": timestamp,
            "pipeline_names": sorted(pipelines.keys()) if isinstance(pipelines, dict) else [],
            "mutable_latest_path": mutable_latest,
        }
    )
    if not isinstance(pipelines, dict) or not pipelines:
        result["error"] = "benchmark report missing pipelines"
        return result
    if mutable_latest:
        result["error"] = "benchmark report points at mutable latest evidence"
        return result
    result["ok"] = True
    return result


def _fixture_check(artist_song: Path, user_vocals: list[Path]) -> dict[str, Any]:
    sources = [artist_song, *user_vocals]
    missing = [str(path) for path in sources if not path.exists()]
    return {
        "ok": not missing,
        "artist_song": str(artist_song),
        "user_vocals": [str(path) for path in user_vocals],
        "missing": missing,
    }


def _tensorrt_suite_check(*, required: bool) -> dict[str, Any]:
    if not required:
        return {"ok": True, "skipped": True, "candidate_dirs": [], "required_files": list(REQUIRED_TRT_ENGINE_FILES)}

    candidates = _discover_trt_engine_dirs()
    complete = [
        path for path in candidates
        if all((path / name).exists() for name in REQUIRED_TRT_ENGINE_FILES)
    ]
    return {
        "ok": bool(complete),
        "required_files": list(REQUIRED_TRT_ENGINE_FILES),
        "candidate_dirs": [str(path) for path in candidates],
        "selected_dir": str(complete[0]) if complete else None,
    }


def _jetson_check(*, required: bool) -> dict[str, Any]:
    detected = Path("/etc/nv_tegra_release").exists()
    return {"ok": (detected or not required), "required": required, "jetson_detected": detected}


def _command_check(command: str, *, required: bool) -> dict[str, Any]:
    resolved = shutil.which(command)
    return {"ok": bool(resolved or not required), "required": required, "command": command, "path": resolved}


def _append_blocker(blockers: list[dict[str, Any]], *, check: str, owner: str, action: str, details: Any) -> None:
    blockers.append({"check": check, "owner": owner, "action": action, "details": details})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bead-id", default="AV-j4cd")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hosted-base-url", default=DEFAULT_HOSTED_BASE_URL)
    parser.add_argument("--benchmark-report", type=Path, default=None)
    parser.add_argument("--artist-song", type=Path, default=DEFAULT_ARTIST_SONG)
    parser.add_argument("--user-vocals", type=Path, action="append", default=None)
    parser.add_argument("--require-jetson", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-docker", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-hosted-probes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-tensorrt-suite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-gitnexus", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-clean-head", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    user_vocals = args.user_vocals or [DEFAULT_USER_VOCALS]

    git_state = _git_dirty()
    checks = {
        "git_head": {
            "ok": _git_sha() is not None,
            "git_sha": _git_sha(),
            "git_status": git_state,
        },
        "benchmark_report": _benchmark_report_check(args.benchmark_report),
        "fixtures": _fixture_check(args.artist_song, user_vocals),
        "docker": _command_check("docker", required=args.require_docker),
        "gitnexus": _command_check("gitnexus", required=args.require_gitnexus),
        "jetson": _jetson_check(required=args.require_jetson),
        "tensorrt_suite": _tensorrt_suite_check(required=args.require_tensorrt_suite),
        "hosted_endpoints": _hosted_checks(args.hosted_base_url, required=args.require_hosted_probes),
    }

    blockers: list[dict[str, Any]] = []
    if not checks["git_head"]["ok"]:
        _append_blocker(
            blockers,
            check="git_head",
            owner="release-engineering",
            action="Run the workflow from a git checkout with a resolvable HEAD commit.",
            details=checks["git_head"],
        )
    if args.require_clean_head and checks["git_head"]["git_status"].get("dirty"):
        _append_blocker(
            blockers,
            check="git_status",
            owner="release-engineering",
            action="Commit or stash local edits before recording current-head release evidence.",
            details=checks["git_head"]["git_status"],
        )
    if not checks["benchmark_report"]["ok"]:
        _append_blocker(
            blockers,
            check="benchmark_report",
            owner="benchmark-runtime",
            action="Generate a current-head comprehensive benchmark report and pass it with --benchmark-report.",
            details=checks["benchmark_report"],
        )
    if not checks["fixtures"]["ok"]:
        _append_blocker(
            blockers,
            check="fixtures",
            owner="qa-runtime",
            action="Restore the hosted production-smoke fixture audio or pass alternate fixture paths.",
            details=checks["fixtures"],
        )
    if not checks["docker"]["ok"]:
        _append_blocker(
            blockers,
            check="docker",
            owner="hardware-runner",
            action="Run the full RC workflow on a compose-capable runner with Docker available.",
            details=checks["docker"],
        )
    if not checks["gitnexus"]["ok"]:
        _append_blocker(
            blockers,
            check="gitnexus",
            owner="tooling",
            action="Install the GitNexus CLI or make it available on PATH before running the full matrix.",
            details=checks["gitnexus"],
        )
    if not checks["jetson"]["ok"]:
        _append_blocker(
            blockers,
            check="jetson",
            owner="hardware-runner",
            action="Run the current-head RC evidence workflow on the Jetson/CUDA host.",
            details=checks["jetson"],
        )
    if not checks["tensorrt_suite"]["ok"]:
        _append_blocker(
            blockers,
            check="tensorrt_suite",
            owner="hardware-runner",
            action="Populate AUTOVOICE_TRT_ENGINE_DIR with the complete TensorRT engine suite and rerun preflight.",
            details=checks["tensorrt_suite"],
        )
    if not checks["hosted_endpoints"]["ok"]:
        _append_blocker(
            blockers,
            check="hosted_endpoints",
            owner="hosted-runner",
            action="Expose a healthy hosted AutoVoice deployment and rerun preflight.",
            details=checks["hosted_endpoints"],
        )

    report = {
        "schema_version": 1,
        "generated_at": _now(),
        "bead_id": args.bead_id,
        "hosted_base_url": args.hosted_base_url,
        "ready": not blockers,
        "checks": checks,
        "blockers": blockers,
    }

    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output_path)
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
