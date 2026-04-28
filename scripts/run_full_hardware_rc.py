#!/usr/bin/env python3
"""Run current-head full hardware RC evidence and emit a release decision artifact."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HOSTED_BASE_URL = "https://autovoice.giggahost.com"
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:10001"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "reports" / "release_candidates"


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


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _copy_if_exists(source: Path, dest: Path) -> str | None:
    if not source.exists():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return str(dest)


def _run_step(name: str, command: list[str], *, log_dir: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    step_env = os.environ.copy()
    step_env.setdefault("PYTHONPATH", str(PROJECT_ROOT / "src"))
    step_env.setdefault("PYTHONNOUSERSITE", "1")
    step_env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    if env:
        step_env.update(env)

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=step_env,
        check=False,
    )
    log_path.write_text((result.stdout or "") + (result.stderr or ""), encoding="utf-8")
    return {
        "name": name,
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "command": command,
        "log": str(log_path),
        "stdout": result.stdout.strip(),
    }


def _boolean_flags(args: argparse.Namespace) -> list[str]:
    flags: list[str] = []
    for attr, name in (
        ("require_jetson", "jetson"),
        ("require_docker", "docker"),
        ("require_gitnexus", "gitnexus"),
        ("require_hosted_probes", "hosted-probes"),
        ("require_tensorrt_suite", "tensorrt-suite"),
        ("require_clean_head", "clean-head"),
    ):
        flags.append(f"--{'require' if getattr(args, attr) else 'no-require'}-{name}")
    return flags


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bead-id", default="AV-j4cd")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--benchmark-report", type=Path, default=None)
    parser.add_argument("--hosted-base-url", default=DEFAULT_HOSTED_BASE_URL)
    parser.add_argument("--local-base-url", default=DEFAULT_LOCAL_BASE_URL)
    parser.add_argument("--timeout", type=int, default=5400)
    parser.add_argument("--production-smoke-timeout-seconds", type=int, default=1800)
    parser.add_argument("--artist-song", type=Path, default=PROJECT_ROOT / "tests/quality_samples/conor_maynard_pillowtalk.wav")
    parser.add_argument("--user-vocals", type=Path, action="append", default=None)
    parser.add_argument("--require-jetson", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-docker", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-gitnexus", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-hosted-probes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-tensorrt-suite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-clean-head", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args(argv)


def _release_decision(
    *,
    bead_id: str,
    git_sha: str | None,
    run_dir: Path,
    preflight: dict[str, Any] | None,
    completion_matrix: dict[str, Any] | None,
    hosted_validation: dict[str, Any] | None,
    production_smoke: dict[str, Any] | None,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    status = "blocked"
    blockers: list[dict[str, Any]] = []
    if preflight and preflight.get("ready"):
        status = "go"
        checks = {
            "completion_matrix": bool(completion_matrix and completion_matrix.get("ok")),
            "hosted_release_candidate": bool(hosted_validation and not hosted_validation.get("error") and hosted_validation.get("evidence_files", {}).get("ok")),
            "production_smoke": bool(production_smoke and production_smoke.get("ok")),
        }
        if not all(checks.values()):
            status = "no-go"
            for name, ok in checks.items():
                if not ok:
                    blockers.append({"check": name, "details": manifest["step_reports"].get(name)})
    elif preflight:
        blockers.extend(preflight.get("blockers", []))

    return {
        "schema_version": 1,
        "generated_at": _now(),
        "bead_id": bead_id,
        "git_sha": git_sha,
        "run_dir": str(run_dir),
        "status": status,
        "ready_for_release": status == "go",
        "blockers": blockers,
        "artifacts": manifest.get("artifacts", {}),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    git_sha = _git_sha()
    git_short = (git_sha or "unknown")[:12]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_root = _resolve_path(args.artifact_root)
    run_dir = artifact_root / args.bead_id / f"{stamp}-{git_short}"
    log_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)

    benchmark_report = _resolve_path(args.benchmark_report) if args.benchmark_report else None
    user_vocals = [_resolve_path(path) for path in (args.user_vocals or [PROJECT_ROOT / "tests/quality_samples/william_singe_pillowtalk.wav"])]

    manifest = {
        "schema_version": 1,
        "generated_at": _now(),
        "bead_id": args.bead_id,
        "git_sha": git_sha,
        "run_dir": str(run_dir),
        "inputs": {
            "benchmark_report": str(benchmark_report) if benchmark_report else None,
            "hosted_base_url": args.hosted_base_url,
            "local_base_url": args.local_base_url,
            "artist_song": str(_resolve_path(args.artist_song)),
            "user_vocals": [str(path) for path in user_vocals],
        },
        "artifacts": {
            "preflight": str(run_dir / "preflight.json"),
            "decision": str(run_dir / "release_decision.json"),
            "completion_matrix": str(run_dir / "completion" / "completion_matrix.json"),
            "hosted_release_candidate": str(run_dir / "platform" / "hosted-release-candidate-validation.json"),
            "production_smoke": str(run_dir / "production_smoke" / "production_smoke.json"),
            "artifact_manifest": str(run_dir / "artifact_manifest.json"),
        },
        "step_reports": {},
    }

    preflight_step = _run_step(
        "preflight",
        [
            sys.executable,
            "scripts/preflight_full_hardware_rc.py",
            "--bead-id",
            args.bead_id,
            "--output",
            str(run_dir / "preflight.json"),
            "--hosted-base-url",
            args.hosted_base_url,
            *([] if benchmark_report is None else ["--benchmark-report", str(benchmark_report)]),
            "--artist-song",
            str(_resolve_path(args.artist_song)),
            *sum([["--user-vocals", str(path)] for path in user_vocals], []),
            *_boolean_flags(args),
        ],
        log_dir=log_dir,
    )
    manifest["step_reports"]["preflight"] = preflight_step
    preflight = _load_json(run_dir / "preflight.json")

    if preflight and preflight.get("ready") and benchmark_report is not None:
        copied = _copy_if_exists(benchmark_report, run_dir / "inputs" / benchmark_report.name)
        if copied:
            manifest["artifacts"]["benchmark_report_copy"] = copied

    if preflight and preflight.get("ready"):
        completion_step = _run_step(
            "completion_matrix",
            [
                sys.executable,
                "scripts/run_completion_matrix.py",
                "--full",
                "--timeout",
                str(args.timeout),
                "--base-url",
                args.local_base_url,
                "--output-dir",
                str(run_dir / "completion"),
                "--platform-report-dir",
                str(run_dir / "platform"),
                "--tensorrt-parity-report",
                str(run_dir / "benchmarks" / "tensorrt-parity.json"),
                *([] if benchmark_report is None else ["--benchmark-report", str(benchmark_report)]),
            ],
            log_dir=log_dir,
        )
        manifest["step_reports"]["completion_matrix"] = completion_step
        completion_matrix = _load_json(run_dir / "completion" / "completion_matrix.json")

        benchmark_archive_dir = None
        if completion_matrix:
            benchmark_archive_dir_raw = completion_matrix.get("artifacts", {}).get("benchmark_archive_dir")
            if benchmark_archive_dir_raw:
                benchmark_archive_dir = Path(benchmark_archive_dir_raw)
                for name in ("benchmark_dashboard.json", "release_evidence.json", "benchmark_dashboard.md"):
                    copied = _copy_if_exists(benchmark_archive_dir / name, run_dir / "benchmarks" / name)
                    if copied:
                        manifest["artifacts"][name] = copied
            parity_report = completion_matrix.get("artifacts", {}).get("tensorrt_parity_report")
            if parity_report:
                manifest["artifacts"]["tensorrt_parity_report"] = parity_report
            platform_report_dir = completion_matrix.get("artifacts", {}).get("platform_report_dir")
            if platform_report_dir:
                manifest["artifacts"]["platform_report_dir"] = platform_report_dir

        hosted_step = _run_step(
            "hosted_release_candidate",
            [
                sys.executable,
                "scripts/validate_release_candidate.py",
                "--base-url",
                args.hosted_base_url,
                "--skip-compose",
                "--report-dir",
                str(run_dir / "platform"),
                "--report-name",
                "hosted-release-candidate-validation.json",
                "--evidence-dir",
                str(benchmark_archive_dir or (run_dir / "benchmarks")),
            ],
            log_dir=log_dir,
        )
        manifest["step_reports"]["hosted_release_candidate"] = hosted_step

        smoke_step = _run_step(
            "production_smoke",
            [
                sys.executable,
                "scripts/run_production_smoke.py",
                "--mode",
                "full",
                "--base-url",
                args.hosted_base_url,
                "--output-dir",
                str(run_dir / "production_smoke"),
                "--artist-song",
                str(_resolve_path(args.artist_song)),
                *sum([["--user-vocals", str(path)] for path in user_vocals], []),
                "--require-stems",
                "--require-quality-evidence",
                "--timeout-seconds",
                str(args.production_smoke_timeout_seconds),
            ],
            log_dir=log_dir,
        )
        manifest["step_reports"]["production_smoke"] = smoke_step
    else:
        completion_matrix = None

    hosted_validation = _load_json(run_dir / "platform" / "hosted-release-candidate-validation.json")
    production_smoke = _load_json(run_dir / "production_smoke" / "production_smoke.json")
    decision = _release_decision(
        bead_id=args.bead_id,
        git_sha=git_sha,
        run_dir=run_dir,
        preflight=preflight,
        completion_matrix=completion_matrix,
        hosted_validation=hosted_validation,
        production_smoke=production_smoke,
        manifest=manifest,
    )

    Path(manifest["artifacts"]["artifact_manifest"]).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    Path(manifest["artifacts"]["decision"]).write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(manifest["artifacts"]["decision"])
    return 0 if decision["ready_for_release"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
