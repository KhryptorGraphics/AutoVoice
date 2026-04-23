#!/usr/bin/env python3
"""Validate an AutoVoice release candidate against health/readiness/metrics contracts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from auto_voice.utils.repo_boundary_audit import run_repo_boundary_audit


def _fetch_json(url: str) -> dict:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _check_url(url: str) -> dict:
    try:
        payload = _fetch_json(url)
        ok = True
        error = None
        if url.endswith("/health"):
            ok = payload.get("status") == "healthy"
            if not ok:
                error = f"unexpected health status: {payload.get('status')!r}"
        elif url.endswith("/ready"):
            ok = bool(payload.get("ready"))
            if not ok:
                error = f"unexpected readiness payload: {payload!r}"
        elif url.endswith("/api/v1/metrics"):
            ok = isinstance(payload, dict) and "error" not in payload
            if not ok:
                error = f"unexpected metrics payload: {payload!r}"
        return {"url": url, "ok": ok, "payload": payload, "error": error}
    except HTTPError as exc:
        return {"url": url, "ok": False, "error": f"http {exc.code}"}
    except URLError as exc:
        return {"url": url, "ok": False, "error": str(exc.reason)}


def _check_url_with_retry(url: str, wait_seconds: float, poll_interval: float) -> dict:
    deadline = datetime.now(timezone.utc).timestamp() + max(0.0, wait_seconds)
    last_result = _check_url(url)
    while not last_result["ok"] and datetime.now(timezone.utc).timestamp() < deadline:
        import time

        time.sleep(max(0.1, poll_interval))
        last_result = _check_url(url)
    return last_result


def _parse_iso8601(raw: str | None) -> datetime | None:
    if not raw or not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _resolve_expected_git_sha() -> str | None:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"].strip() or None
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


def _validate_evidence_payloads(
    dashboard_path: Path,
    release_evidence_path: Path,
    *,
    expected_git_sha: str | None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "paths": {
            "benchmark_dashboard": str(dashboard_path),
            "release_evidence": str(release_evidence_path),
        },
        "ok": False,
        "error": None,
        "expected_git_sha": expected_git_sha,
    }
    if not dashboard_path.exists() or not release_evidence_path.exists():
        report["error"] = "missing evidence artifact"
        return report

    try:
        dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
        release_evidence = json.loads(release_evidence_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report["error"] = f"invalid evidence json: {exc}"
        return report

    if not isinstance(dashboard, dict) or not isinstance(release_evidence, dict):
        report["error"] = "evidence payloads must be JSON objects"
        return report

    dashboard_generated_at = _parse_iso8601(dashboard.get("generated_at"))
    release_generated_at = _parse_iso8601(release_evidence.get("generated_at"))
    dashboard_provenance = dashboard.get("provenance")
    release_provenance = release_evidence.get("provenance")
    now = datetime.now(timezone.utc)

    report["dashboard_generated_at"] = dashboard.get("generated_at")
    report["release_generated_at"] = release_evidence.get("generated_at")
    report["dashboard_provenance"] = dashboard_provenance
    report["release_provenance"] = release_provenance

    if not dashboard_generated_at or not release_generated_at:
        report["error"] = "evidence artifacts require valid generated_at timestamps"
        return report
    if dashboard_generated_at != release_generated_at:
        report["error"] = "dashboard and release evidence generated_at timestamps differ"
        return report
    if dashboard_generated_at > now:
        report["error"] = "evidence timestamp is in the future"
        return report
    if not isinstance(dashboard_provenance, dict) or not isinstance(release_provenance, dict):
        report["error"] = "evidence artifacts require provenance objects"
        return report
    if dashboard_provenance != release_provenance:
        report["error"] = "dashboard and release evidence provenance differ"
        return report
    if dashboard_provenance.get("schema_version") != 1:
        report["error"] = f"unsupported evidence schema_version: {dashboard_provenance.get('schema_version')!r}"
        return report
    if not dashboard_provenance.get("generator"):
        report["error"] = "evidence provenance missing generator"
        return report
    if expected_git_sha and dashboard_provenance.get("git_sha") != expected_git_sha:
        report["error"] = (
            f"evidence git_sha {dashboard_provenance.get('git_sha')!r} "
            f"does not match expected {expected_git_sha!r}"
        )
        return report

    pipelines = dashboard.get("pipelines")
    comparisons = dashboard.get("comparisons")
    canonical_pipelines = dashboard.get("canonical_pipelines")
    if not isinstance(pipelines, dict) or not isinstance(comparisons, dict) or not isinstance(canonical_pipelines, dict):
        report["error"] = "benchmark dashboard missing canonical structure"
        return report
    if release_evidence.get("canonical_pipelines") != canonical_pipelines:
        report["error"] = "release evidence canonical_pipelines do not match dashboard"
        return report
    if release_evidence.get("target_hardware") != dashboard.get("target_hardware"):
        report["error"] = "release evidence target_hardware does not match dashboard"
        return report
    if release_evidence.get("pipeline_count") != len(pipelines):
        report["error"] = "release evidence pipeline_count does not match dashboard"
        return report
    if release_evidence.get("comparison_count") != len(comparisons):
        report["error"] = "release evidence comparison_count does not match dashboard"
        return report

    report["ok"] = True
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:5000", help="Base AutoVoice URL")
    parser.add_argument("--compose-file", default="docker-compose.yaml", help="Compose file to validate")
    parser.add_argument("--report-dir", default="reports/platform", help="Output directory for validation artifacts")
    parser.add_argument(
        "--report-name",
        default="release-candidate-validation.json",
        help="Validation report filename written under --report-dir",
    )
    parser.add_argument("--skip-compose", action="store_true", help="Skip docker compose config validation")
    parser.add_argument(
        "--skip-evidence",
        action="store_true",
        help="Skip benchmark/release evidence checks for non-release smoke lanes",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=0.0,
        help="How long to wait for the release candidate endpoints to become healthy",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Polling interval in seconds while waiting for the release candidate endpoints",
    )
    parser.add_argument(
        "--expected-git-sha",
        default=None,
        help="Expected git SHA for benchmark evidence provenance. Defaults to GITHUB_SHA or HEAD.",
    )
    args = parser.parse_args(argv)

    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = PROJECT_ROOT / report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url.rstrip("/"),
        "compose": None,
        "repo_boundaries": run_repo_boundary_audit(PROJECT_ROOT),
        "expected_git_sha": args.expected_git_sha or _resolve_expected_git_sha(),
        "evidence_files": {},
        "checks": [],
    }

    if not args.skip_compose:
        compose_result = subprocess.run(
            ["docker", "compose", "-f", args.compose_file, "config", "-q"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        results["compose"] = {
            "ok": compose_result.returncode == 0,
            "stderr": compose_result.stderr.strip() or None,
        }
    else:
        results["compose"] = {"ok": True, "skipped": True}

    for path in ("/health", "/api/v1/ready", "/api/v1/metrics"):
        results["checks"].append(
            _check_url_with_retry(
                f"{results['base_url']}{path}",
                wait_seconds=args.wait_seconds,
                poll_interval=args.poll_interval,
            )
        )

    if args.skip_evidence:
        results["evidence_files"] = {"skipped": True}
    else:
        dashboard_path = PROJECT_ROOT / "reports/benchmarks/latest/benchmark_dashboard.json"
        release_evidence_path = PROJECT_ROOT / "reports/benchmarks/latest/release_evidence.json"
        results["evidence_files"] = _validate_evidence_payloads(
            dashboard_path,
            release_evidence_path,
            expected_git_sha=results["expected_git_sha"],
        )

    report_path = report_dir / args.report_name
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    all_ok = (
        bool(results["compose"]["ok"])
        and bool(results["repo_boundaries"]["ok"])
        and (
            args.skip_evidence
            or bool(results["evidence_files"]["ok"])
        )
        and all(check["ok"] for check in results["checks"])
    )
    print(report_path)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
