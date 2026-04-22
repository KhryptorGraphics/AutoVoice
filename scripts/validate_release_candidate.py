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
        return {"url": url, "ok": True, "payload": payload}
    except HTTPError as exc:
        return {"url": url, "ok": False, "error": f"http {exc.code}"}
    except URLError as exc:
        return {"url": url, "ok": False, "error": str(exc.reason)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:5000", help="Base AutoVoice URL")
    parser.add_argument("--compose-file", default="docker-compose.yaml", help="Compose file to validate")
    parser.add_argument("--report-dir", default="reports/platform", help="Output directory for validation artifacts")
    parser.add_argument("--skip-compose", action="store_true", help="Skip docker compose config validation")
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
        results["checks"].append(_check_url(f"{results['base_url']}{path}"))

    for name, relative_path in {
        "benchmark_dashboard": Path("reports/benchmarks/latest/benchmark_dashboard.json"),
        "release_evidence": Path("reports/benchmarks/latest/release_evidence.json"),
    }.items():
        file_path = PROJECT_ROOT / relative_path
        results["evidence_files"][name] = {
            "path": str(file_path),
            "ok": file_path.exists(),
        }

    report_path = report_dir / "release-candidate-validation.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    all_ok = (
        bool(results["compose"]["ok"])
        and bool(results["repo_boundaries"]["ok"])
        and all(item["ok"] for item in results["evidence_files"].values())
        and all(check["ok"] for check in results["checks"])
    )
    print(report_path)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
