#!/usr/bin/env python3
"""Validate hosted deployment assumptions for the Apache-backed AutoVoice install."""

from __future__ import annotations

import argparse
import json
import os
import socket
import ssl
from pathlib import Path
from typing import Any


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _discover_vhost_files(hostname: str, sites_dir: Path = Path("/etc/apache2/sites-available")) -> list[Path]:
    """Find vhosts that serve hostname by ServerName or ServerAlias."""

    if not sites_dir.exists():
        return []

    matches: list[Path] = []
    for path in sorted(sites_dir.glob("*.conf")):
        try:
            text = _load_text(path)
        except OSError:
            continue
        for line in text.splitlines():
            tokens = line.split()
            if not tokens or tokens[0] not in {"ServerName", "ServerAlias"}:
                continue
            if hostname in tokens[1:]:
                matches.append(path)
                break
    return matches


def _check_vhost_files(
    vhost_files: list[Path],
    *,
    hostname: str,
    backend_port: int,
    frontend_root: str,
    min_body_limit: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": True,
        "files": {},
        "hostname": hostname,
        "serving_vhost_count": 0,
    }
    expected_tokens = [
        hostname,
        f"ProxyPass /api http://127.0.0.1:{backend_port}/api",
        f"ProxyPass /socket.io http://127.0.0.1:{backend_port}/socket.io",
        f"ProxyPass /ready http://127.0.0.1:{backend_port}/ready",
        frontend_root,
    ]
    for path in vhost_files:
        if not path.exists():
            result["ok"] = False
            result["files"][str(path)] = {"ok": False, "error": "missing"}
            continue

        text = _load_text(path)
        redirects_to_https = (
            "https://" in text
            and ("Redirect " in text or "RewriteRule " in text)
            and "ProxyPass /api" not in text
            and frontend_root not in text
        )
        body_limit = None
        for line in text.splitlines():
            if "SecRequestBodyLimit" in line:
                tokens = line.split()
                try:
                    body_limit = int(tokens[-1])
                except (ValueError, IndexError):
                    body_limit = None
        if redirects_to_https:
            missing_hostname = hostname not in text
            result["files"][str(path)] = {
                "ok": not missing_hostname,
                "kind": "redirect",
                "missing_tokens": [hostname] if missing_hostname else [],
                "body_limit": body_limit,
                "body_limit_ok": True,
            }
            if missing_hostname:
                result["ok"] = False
            continue

        missing_tokens = [token for token in expected_tokens if token not in text]
        file_ok = not missing_tokens and body_limit is not None and body_limit >= min_body_limit
        if file_ok:
            result["serving_vhost_count"] += 1
        result["files"][str(path)] = {
            "ok": file_ok,
            "kind": "serving",
            "missing_tokens": missing_tokens,
            "body_limit": body_limit,
            "body_limit_ok": body_limit is not None and body_limit >= min_body_limit,
        }
    result["ok"] = result["serving_vhost_count"] > 0
    return result


def _check_dns(hostname: str) -> dict[str, Any]:
    try:
        records = sorted({entry[4][0] for entry in socket.getaddrinfo(hostname, None)})
    except OSError as exc:
        return {"ok": False, "error": str(exc), "hostname": hostname}
    return {"ok": bool(records), "hostname": hostname, "records": records}


def _check_tls(hostname: str, *, port: int = 443) -> dict[str, Any]:
    try:
        with socket.create_connection((hostname, port), timeout=5) as sock:
            context = ssl.create_default_context()
            with context.wrap_socket(sock, server_hostname=hostname) as wrapped:
                cert = wrapped.getpeercert()
    except OSError as exc:
        return {"ok": False, "hostname": hostname, "error": str(exc)}

    sans = []
    for key, value in cert.get("subjectAltName", []):
        if key == "DNS":
            sans.append(value)
    return {
        "ok": hostname in sans,
        "hostname": hostname,
        "subject_alt_names": sans,
    }


def _check_required_secrets(required_env: list[str], secret_env_any: list[str]) -> dict[str, Any]:
    missing = [name for name in required_env if not os.environ.get(name)]
    secret_present = [name for name in secret_env_any if os.environ.get(name)]
    return {
        "ok": not missing and bool(secret_present),
        "required_env": required_env,
        "missing_env": missing,
        "secret_env_any": secret_env_any,
        "secret_present": bool(secret_present),
    }


def _check_jetson_prereqs(require_jetson: bool) -> dict[str, Any]:
    nv_tegra = Path("/etc/nv_tegra_release")
    jetson_detected = nv_tegra.exists()
    if require_jetson and not jetson_detected:
        return {"ok": False, "require_jetson": True, "jetson_detected": False}
    return {"ok": True, "require_jetson": require_jetson, "jetson_detected": jetson_detected}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hostname", default="autovoice.giggahost.com")
    parser.add_argument("--backend-port", type=int, default=10600)
    parser.add_argument("--frontend-root", default="frontend/dist")
    parser.add_argument("--min-body-limit", type=int, default=262144000)
    parser.add_argument("--vhost-file", action="append", default=[])
    parser.add_argument("--report", type=Path, default=Path("reports/platform/hosted-preflight.json"))
    parser.add_argument("--skip-dns", action="store_true")
    parser.add_argument("--skip-tls", action="store_true")
    parser.add_argument("--require-jetson", action="store_true")
    parser.add_argument(
        "--required-env",
        action="append",
        default=[],
        help="Required environment variable for hosted deployment readiness. Can be repeated.",
    )
    parser.add_argument(
        "--secret-env-any",
        action="append",
        default=["SECRET_KEY", "AUTOVOICE_SECRET_FLASK_SECRET_KEY"],
        help="At least one acceptable Flask secret environment variable. Can be repeated.",
    )
    args = parser.parse_args(argv)

    vhost_files = [Path(path) for path in (args.vhost_file or [])]
    if not vhost_files:
        vhost_files = _discover_vhost_files(args.hostname) or [
            Path(f"/etc/apache2/sites-available/{args.hostname}.conf"),
            Path(f"/etc/apache2/sites-available/{args.hostname}-le-ssl.conf"),
        ]

    checks = {
        "vhosts": _check_vhost_files(
            vhost_files,
            hostname=args.hostname,
            backend_port=args.backend_port,
            frontend_root=args.frontend_root,
            min_body_limit=args.min_body_limit,
        ),
        "dns": {"ok": True, "skipped": True} if args.skip_dns else _check_dns(args.hostname),
        "tls": {"ok": True, "skipped": True} if args.skip_tls else _check_tls(args.hostname),
        "secrets": _check_required_secrets(args.required_env, args.secret_env_any),
        "jetson": _check_jetson_prereqs(args.require_jetson),
    }
    report = {
        "hostname": args.hostname,
        "backend_port": args.backend_port,
        "frontend_root": args.frontend_root,
        "checks": checks,
        "ok": all(item.get("ok", False) for item in checks.values()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
