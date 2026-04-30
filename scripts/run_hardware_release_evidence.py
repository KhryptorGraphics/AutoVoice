#!/usr/bin/env python3
"""Create current-head hardware release evidence and fail closed when lanes are missing."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tail_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")[-4000:]
    return str(value)[-4000:]


def _run(command: list[str], *, timeout: int = 60, timeout_ok_when_output: bool = False) -> dict[str, Any]:
    started = time.time()
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": _tail_output(result.stdout),
            "stderr": _tail_output(result.stderr),
            "duration_seconds": round(time.time() - started, 3),
            "ok": result.returncode == 0,
        }
    except subprocess.TimeoutExpired as exc:
        stdout = _tail_output(getattr(exc, "stdout", None) or getattr(exc, "output", None))
        stderr = _tail_output(getattr(exc, "stderr", None))
        ok = timeout_ok_when_output and bool(stdout.strip())
        return {
            "command": command,
            "returncode": None,
            "stdout": stdout,
            "stderr": stderr or str(exc),
            "duration_seconds": round(time.time() - started, 3),
            "ok": ok,
            "timed_out": True,
        }
    except OSError as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "duration_seconds": round(time.time() - started, 3),
            "ok": False,
        }


def _git_sha() -> str:
    result = _run(["git", "rev-parse", "HEAD"])
    if not result["ok"]:
        raise RuntimeError(result["stderr"] or "failed to resolve git sha")
    return str(result["stdout"]).strip()


def _short_sha(sha: str) -> str:
    return sha[:12]


def _preflight() -> tuple[list[dict[str, Any]], list[str]]:
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []

    git_status = _run(["git", "status", "--porcelain"])
    checks.append({"name": "git-clean", **git_status})
    if git_status["ok"] and str(git_status["stdout"]).strip():
        blockers.append("working tree is dirty; commit release evidence against a stable tree")
    elif not git_status["ok"]:
        blockers.append("could not inspect git status")

    for name, command in {
        "docker": ["docker", "version", "--format", "{{.Server.Version}}"],
        "nvidia-smi": ["nvidia-smi", "-L"],
        "tegrastats": ["tegrastats", "--interval", "1000"],
    }.items():
        available = shutil.which(command[0]) is not None
        if not available:
            check = {
                "name": name,
                "command": command,
                "ok": False,
                "returncode": None,
                "stdout": "",
                "stderr": f"{command[0]} not found",
                "duration_seconds": 0,
            }
        else:
            timeout = 3 if name == "tegrastats" else 20
            check = {
                "name": name,
                **_run(command, timeout=timeout, timeout_ok_when_output=(name == "tegrastats")),
            }
        checks.append(check)
        if not check["ok"]:
            blockers.append(f"{name} preflight failed")

    torch_check = _run(
        [
            sys.executable,
            "-c",
            "import torch; print(torch.__version__); print(torch.cuda.is_available())",
        ],
        timeout=30,
    )
    checks.append({"name": "torch-cuda", **torch_check})
    if not torch_check["ok"] or "True" not in str(torch_check["stdout"]).splitlines()[-1:]:
        blockers.append("torch CUDA is unavailable")

    return checks, blockers


def _execute_lanes(output_dir: Path, timeout: int) -> tuple[list[dict[str, Any]], list[str]]:
    lanes = [
        {
            "name": "completion-matrix",
            "command": [
                sys.executable,
                "scripts/run_completion_matrix.py",
                "--output-dir",
                str(output_dir / "completion"),
                "--timeout",
                str(timeout),
            ],
        },
        {
            "name": "cuda-stack",
            "command": [
                "bash",
                "scripts/validate_cuda_stack.sh",
                "--output-dir",
                str(output_dir / "cuda-stack"),
            ],
        },
    ]
    results: list[dict[str, Any]] = []
    blockers: list[str] = []
    for lane in lanes:
        result = {"name": lane["name"], **_run(lane["command"], timeout=timeout)}
        results.append(result)
        if not result["ok"]:
            blockers.append(f"{lane['name']} lane failed")
    return results, blockers


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate current-head hardware release evidence.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Run preflight only; do not execute hardware lanes.")
    parser.add_argument("--execute", action="store_true", help="Execute hardware validation lanes after preflight.")
    parser.add_argument("--allow-blocked", action="store_true", help="Write evidence and return 0 even when not release-ready.")
    parser.add_argument("--timeout", type=int, default=900)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sha = _git_sha()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or PROJECT_ROOT / "reports" / "release-evidence" / f"{stamp}-{_short_sha(sha)}"
    output_dir.mkdir(parents=True, exist_ok=True)

    checks, blockers = _preflight()
    lanes: list[dict[str, Any]] = []
    if args.execute and not args.dry_run:
        lane_results, lane_blockers = _execute_lanes(output_dir, args.timeout)
        lanes.extend(lane_results)
        blockers.extend(lane_blockers)
    elif not args.execute:
        blockers.append("hardware validation lanes were not executed; rerun with --execute on Jetson")

    ready = not blockers and bool(lanes)
    decision = {
        "ready": ready,
        "git_sha": sha,
        "git_sha_short": _short_sha(sha),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "dry_run": bool(args.dry_run),
        "executed_lanes": [lane["name"] for lane in lanes],
        "blockers": blockers,
        "preflight_checks": checks,
        "lane_results": lanes,
    }
    _write_json(output_dir / "release_decision.json", decision)
    _write_json(output_dir / "preflight.json", {"git_sha": sha, "checks": checks, "blockers": blockers})

    latest = PROJECT_ROOT / "reports" / "release-evidence" / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    _write_json(latest / "release_decision.json", decision)

    print(json.dumps(decision, indent=2, sort_keys=True))
    if ready or args.allow_blocked:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
