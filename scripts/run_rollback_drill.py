#!/usr/bin/env python3
"""Plan or execute an AutoVoice production rollback drill."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_URL = "https://autovoice.giggahost.com"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(command: list[str], *, check: bool = False, env: dict[str, str] | None = None) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )
    result = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "ok": completed.returncode == 0,
    }
    if check and completed.returncode != 0:
        raise RuntimeError(f"Command failed: {command}: {completed.stderr.strip()}")
    return result


def _git(*args: str, check: bool = False) -> dict[str, Any]:
    return _run(["git", *args], check=check)


def _current_ref() -> str | None:
    result = _git("rev-parse", "--short", "HEAD")
    return result["stdout"] or None if result["ok"] else None


def _current_tags() -> list[str]:
    result = _git("tag", "--points-at", "HEAD")
    if not result["ok"] or not result["stdout"]:
        return []
    return result["stdout"].splitlines()


def _previous_tag(current_tag: str | None) -> str | None:
    result = _git("tag", "--sort=-creatordate")
    if not result["ok"]:
        return None
    tags = [tag for tag in result["stdout"].splitlines() if tag]
    if not tags:
        return None
    if current_tag is None:
        return tags[0]
    for tag in tags:
        if tag != current_tag:
            return tag
    return None


def _fetch_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=10) as response:
            return {
                "url": url,
                "status": response.status,
                "ok": response.status == 200,
                "payload": json.loads(response.read().decode("utf-8")),
            }
    except HTTPError as exc:
        return {"url": url, "status": exc.code, "ok": False, "error": exc.read().decode("utf-8", errors="replace")}
    except (URLError, TimeoutError) as exc:
        return {"url": url, "status": None, "ok": False, "error": str(exc)}


def _health_checks(base_url: str) -> list[dict[str, Any]]:
    base = base_url.rstrip("/")
    checks = [
        _fetch_json(f"{base}/api/v1/health"),
        _fetch_json(f"{base}/ready"),
        _fetch_json(f"{base}/api/v1/metrics"),
    ]
    for check in checks:
        payload = check.get("payload")
        if check["url"].endswith("/health"):
            check["ok"] = check["ok"] and isinstance(payload, dict) and payload.get("status") == "healthy"
        elif check["url"].endswith("/ready"):
            check["ok"] = check["ok"] and isinstance(payload, dict) and bool(payload.get("ready"))
        else:
            check["ok"] = check["ok"] and isinstance(payload, dict) and "error" not in payload
    return checks


def _command_plan(*, target_ref: str, project_name: str, base_url: str) -> list[list[str]]:
    return [
        ["git", "fetch", "--tags", "origin"],
        ["git", "checkout", target_ref],
        ["docker", "compose", "-p", project_name, "-f", "docker-compose.yaml", "up", "-d", "--build", "backend", "frontend"],
        ["python", "scripts/validate_release_candidate.py", "--base-url", base_url, "--report-dir", "reports/platform", "--wait-seconds", "180", "--skip-evidence"],
    ]


def run_drill(args: argparse.Namespace) -> dict[str, Any]:
    current_tags = _current_tags()
    current_tag = args.current_tag or (current_tags[0] if current_tags else None)
    target_ref = args.target_ref or _previous_tag(current_tag)
    if not target_ref:
        raise RuntimeError("No rollback target found; pass --target-ref explicitly")

    project_name = args.compose_project or f"autovoice-rollback-{int(time.time())}"
    plan = _command_plan(target_ref=target_ref, project_name=project_name, base_url=args.base_url)
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": _now(),
        "mode": "execute" if args.execute else "dry-run",
        "base_url": args.base_url,
        "current_ref": _current_ref(),
        "current_tags": current_tags,
        "current_tag": current_tag,
        "target_ref": target_ref,
        "compose_project": project_name,
        "preflight_health": _health_checks(args.base_url),
        "compose_config": _run(["docker", "compose", "-f", "docker-compose.yaml", "config", "-q"], env={
            "SECRET_KEY": os.environ.get("SECRET_KEY", "rollback-drill-compose-secret"),
            "GRAFANA_PASSWORD": os.environ.get("GRAFANA_PASSWORD", "rollback-drill-grafana-password"),
        }),
        "command_plan": plan,
        "executed": [],
    }
    report["ok"] = all(check["ok"] for check in report["preflight_health"]) and report["compose_config"]["ok"]

    if args.execute:
        if not report["ok"]:
            raise RuntimeError("Rollback drill preflight failed; refusing --execute")
        for command in plan:
            result = _run(command)
            report["executed"].append(result)
            if not result["ok"]:
                report["ok"] = False
                break

    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output", type=Path, default=Path("reports/platform/rollback-drill.json"))
    parser.add_argument("--current-tag", default=None)
    parser.add_argument("--target-ref", default=None)
    parser.add_argument("--compose-project", default=None)
    parser.add_argument("--execute", action="store_true", help="Execute the rollback command plan after preflight checks")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_drill(args)
    except Exception as exc:
        report = {
            "schema_version": 1,
            "generated_at": _now(),
            "mode": "execute" if args.execute else "dry-run",
            "base_url": args.base_url,
            "ok": False,
            "error": str(exc),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
