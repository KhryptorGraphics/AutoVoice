#!/usr/bin/env python3
"""Validate the backend dependency contract stays single-sourced."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _require(path: str) -> Path:
    file_path = PROJECT_ROOT / path
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


def main() -> int:
    errors: list[str] = []
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

    report = {
        "ok": not errors,
        "runtime_requirement_count": len(runtime),
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
