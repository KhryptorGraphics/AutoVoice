#!/usr/bin/env python3
"""Audit tracked dependency/output boundaries for the parent repo."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from auto_voice.utils.repo_boundary_audit import run_repo_boundary_audit


def main() -> int:
    audit = run_repo_boundary_audit()
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
