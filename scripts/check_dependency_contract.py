#!/usr/bin/env python3
"""Validate the backend dependency contract and supply-chain lock boundaries."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
import hashlib
import argparse
from json import JSONDecodeError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCK_DEFAULT = PROJECT_ROOT / "requirements.lock"


def _require(path: str) -> Path:
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = PROJECT_ROOT / file_path
    if not file_path.exists():
        raise AssertionError(f"missing required dependency file: {path}")
    return file_path


def _runtime_requirements() -> list[str]:
    lines: list[str] = []
    for raw_line in _require("requirements-runtime.txt").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_refs(text: str, substitutions: dict[str, str]) -> str:
    for key, value in substitutions.items():
        text = text.replace(f"${{{key}}}", value)
    return text


def _extract_docker_from_references() -> dict[str, str]:
    references: dict[str, str] = {}
    dockerfile = _require("Dockerfile")
    substitutions: dict[str, str] = {}
    for raw_line in dockerfile.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("ARG "):
            continue
        if "=" not in line:
            continue
        arg_expr = line[len("ARG "):]
        name, value = arg_expr.split("=", 1)
        substitutions[name] = value

    for raw_line in dockerfile.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        match = re.match(r"^FROM\s+([^\s]+)(?:\s+AS\s+(\w+))?$", stripped, re.IGNORECASE)
        if not match:
            continue
        image_expr = _resolve_refs(match.group(1), substitutions)
        stage = match.group(2) or ""
        if stage.lower() == "base":
            references["backend_base"] = image_expr
        elif stage.lower() == "frontend":
            references["frontend_builder"] = image_expr

    frontend = _require("frontend/Dockerfile.frontend")
    for raw_line in frontend.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        match = re.match(r"^FROM\s+([^\s]+)(?:\s+AS\s+(\w+))?$", stripped, re.IGNORECASE)
        if not match:
            continue
        image_expr = match.group(1)
        stage = match.group(2) or ""
        if stage.lower() == "build":
            references["frontend_builder"] = image_expr
        elif stage.lower() == "":  # pragma: no cover - fallback for unaliased runtime
            if "frontend_builder" not in references:
                references["frontend_builder"] = image_expr
        elif stage.lower() == "nginx":
            references["frontend_runtime"] = image_expr

    # In this repo, frontend runtime image is the second FROM in frontend Dockerfile.
    # Keep a fallback by service alias if stage naming changes later.
    if "frontend_runtime" not in references:
        from_frontend_lines = [
            _resolve_refs(line.strip().split()[1], {})
            for line in frontend.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("FROM ")
        ]
        if len(from_frontend_lines) > 1:
            references["frontend_runtime"] = from_frontend_lines[1]
        elif from_frontend_lines:
            references["frontend_runtime"] = from_frontend_lines[-1]

    if "backend_base" not in references:
        references["backend_base"] = _resolve_refs(substitutions.get("AUTOVOICE_BASE_IMAGE", ""), substitutions)
    return references


def _extract_compose_references() -> dict[str, str]:
    compose = _require("docker-compose.yaml")
    services: dict[str, str] = {}
    current_service: str | None = None
    for raw_line in compose.read_text(encoding="utf-8").splitlines():
        service_match = re.match(r"^  (\S+):$", raw_line)
        if service_match:
            current_service = service_match.group(1)
            continue
        image_match = re.match(r"^\s{4}image:\s*(\S+)\s*$", raw_line)
        if image_match and current_service:
            services[current_service] = image_match.group(1)
    return services


def _check_file_hash(expected_path: str, contract: dict[str, str], label: str, errors: list[str], report: dict[str, dict[str, object]]) -> None:
    required_hash = str(contract.get("sha256", ""))
    actual_path = _require(expected_path)
    actual_hash = _sha256(actual_path)
    ok = required_hash == actual_hash
    report[label] = {
        "path": expected_path,
        "expected": required_hash,
        "actual": actual_hash,
        "ok": ok,
    }
    if not ok:
        errors.append(f"{label} hash mismatch for {expected_path}")


def _check_image_reference(expected_ref: str, actual_ref: str | None, label: str, errors: list[str], report: dict[str, dict[str, object]]) -> None:
    ok = actual_ref is not None and expected_ref == actual_ref
    report[label] = {
        "expected": expected_ref,
        "actual": actual_ref or "",
        "ok": ok,
        "pinned": "@sha256:" in (actual_ref or ""),
    }
    if not ok:
        errors.append(f"image reference mismatch for {label}: expected {expected_ref}, got {actual_ref or 'missing'}")


def _check_audit_report(path: Path, report: dict[str, object], errors: list[str], severity: str = "high") -> None:
    severity_order = ["low", "medium", "high", "critical"]
    cutoff_index = max(0, severity_order.index(severity) if severity in severity_order else 2)
    target = set(severity_order[cutoff_index:])
    if not path.exists():
        errors.append(f"missing audit report: {path}")
        report["exists"] = False
        report["ok"] = False
        return

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except JSONDecodeError:
        errors.append(f"invalid JSON in audit report: {path}")
        report["exists"] = True
        report["ok"] = False
        report["findings"] = ["invalid-json"]
        return
    findings: list[str] = []
    vulnerabilities = payload.get("vulnerabilities", [])
    if isinstance(vulnerabilities, dict):
        for entry in vulnerabilities.values():
            if isinstance(entry, dict):
                if str(entry.get("severity", "")).lower() in target:
                    findings.append(f"{entry.get('title') or entry.get('id') or 'vulnerability'}:{str(entry.get('severity', '')).lower()}")
            elif isinstance(entry, list):
                for item in entry:
                    if isinstance(item, dict) and str(item.get("severity", "")).lower() in target:
                        findings.append(f"{item.get('id') or item.get('vulnerability_id') or 'vulnerability'}:{str(item.get('severity', '')).lower()}")
    elif isinstance(vulnerabilities, list):
        for vuln in vulnerabilities:
            if not isinstance(vuln, dict):
                continue
            vuln_sev = str(vuln.get("severity", vuln.get("fixer", "") or "")).lower()
            if vuln_sev in target:
                if vuln.get("id") or vuln.get("vulnerability_id"):
                    findings.append(f"{vuln.get('id') or vuln.get('vulnerability_id')}:{vuln_sev}")

    # npm format fallback
    if not findings and isinstance(payload.get("metadata"), dict):
        for vuln_level, count in payload["metadata"].get("vulnerabilities", {}).items():
            if vuln_level.lower() in target and int(count or 0) > 0:
                findings.append(f"{vuln_level}:{count}")

    if findings:
        errors.append(f"{path.name} contains high/critical vulnerabilities: {', '.join(findings)}")
        report["ok"] = False
    else:
        report["ok"] = True
    report["exists"] = True
    report["findings"] = findings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dependency and supply-chain contract boundaries.")
    parser.add_argument("--lock-path", default=LOCK_DEFAULT, type=Path, help="Contract lock file.")
    parser.add_argument("--pip-audit-report", type=Path, default=None, help="Path to pip-audit JSON output for policy check.")
    parser.add_argument("--npm-audit-report", type=Path, default=None, help="Path to npm audit JSON output for policy check.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    lock_path = _require(str(args.lock_path))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    report: dict[str, object] = {"ok": True}
    file_report: dict[str, dict[str, object]] = {}
    image_report: dict[str, dict[str, object]] = {}
    audit_report: dict[str, dict[str, object]] = {}

    if "backend" not in lock or "images" not in lock:
        errors.append("requirements.lock is missing required sections")

    runtime = _runtime_requirements()
    if not runtime:
        errors.append("requirements-runtime.txt is empty")

    requirements = _require("requirements.txt").read_text(encoding="utf-8")
    if "-r requirements-runtime.txt" not in requirements:
        errors.append("requirements.txt must include -r requirements-runtime.txt")

    setup_py = _require("setup.py").read_text(encoding="utf-8")
    if "_read_requirements('requirements-runtime.txt')" not in setup_py:
        errors.append("setup.py install_requires must read requirements-runtime.txt")

    _require("pyproject.toml")
    _require("frontend/package-lock.json")

    backend_contract = lock.get("backend", {})
    if runtime and _read_text(_require("requirements-runtime.txt")).strip():
        runtime_entry = backend_contract.get("requirements_runtime", {})
        if runtime_entry:
            _check_file_hash("requirements-runtime.txt", runtime_entry, "requirements-runtime.txt", errors, file_report)
        else:
            errors.append("requirements.lock missing backend.requirements_runtime")
    lock_frontend = backend_contract.get("frontend_lock", {})
    if lock_frontend:
        _check_file_hash("frontend/package-lock.json", lock_frontend, "frontend/package-lock.json", errors, file_report)
    else:
        errors.append("requirements.lock missing backend.frontend_lock")

    image_contract = lock.get("images", {})
    docker_refs = _extract_docker_from_references()
    compose_refs = _extract_compose_references()

    docker_contract = image_contract.get("dockerfiles", {})
    for key in ("backend_base", "frontend_builder", "frontend_runtime"):
        _check_image_reference(docker_contract.get(key, ""), docker_refs.get(key), key, errors, image_report)

    compose_contract = image_contract.get("compose", {})
    for service in ("prometheus", "grafana"):
        expected = compose_contract.get(service, "")
        _check_image_reference(expected, compose_refs.get(service), f"compose.{service}", errors, image_report)

    if args.pip_audit_report is not None:
        _check_audit_report(args.pip_audit_report, audit_report.setdefault("pip-audit", {}), errors)
    if args.npm_audit_report is not None:
        _check_audit_report(args.npm_audit_report, audit_report.setdefault("npm-audit", {}), errors)

    report["files"] = file_report
    report["images"] = image_report
    if audit_report:
        report["audits"] = audit_report

    report["runtime_requirement_count"] = len(runtime)
    report["errors"] = errors
    report["ok"] = not errors

    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
