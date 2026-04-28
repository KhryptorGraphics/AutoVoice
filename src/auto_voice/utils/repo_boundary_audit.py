"""Audit tracked dependency and generated-output boundaries for the parent repo."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List


FORBIDDEN_PREFIXES = (
    "node_modules/",
    "output/",
    "reports/",
)


def _tracked_files(repo_root: Path) -> List[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def run_repo_boundary_audit(repo_root: Path | None = None) -> Dict[str, Any]:
    root = Path(repo_root or Path(__file__).resolve().parents[3])
    tracked = _tracked_files(root)
    violations = [
        path
        for path in tracked
        if path.startswith(FORBIDDEN_PREFIXES)
    ]

    return {
        "repo_root": str(root),
        "forbidden_prefixes": list(FORBIDDEN_PREFIXES),
        "violation_count": len(violations),
        "violations": violations,
        "ok": not violations,
    }
