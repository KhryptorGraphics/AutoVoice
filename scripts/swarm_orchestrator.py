#!/usr/bin/env python3
"""Compatibility shim for the canonical repo-native swarm runner."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "config" / "swarm_manifests" / "full.yaml"
MANIFESTS = {
    "all": DEFAULT_MANIFEST,
    "research": PROJECT_ROOT / "config" / "swarm_manifests" / "research.yaml",
    "development": PROJECT_ROOT / "config" / "swarm_manifests" / "development.yaml",
    "review": PROJECT_ROOT / "config" / "swarm_manifests" / "review.yaml",
    "testing": PROJECT_ROOT / "config" / "swarm_manifests" / "testing.yaml",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swarm", default="all", choices=sorted(MANIFESTS), help="Named swarm manifest to run")
    parser.add_argument("--status", metavar="RUN_ID", help="Print status for an existing run id")
    parser.add_argument("--run-id", default=None, help="Explicit run identifier")
    parser.add_argument("--dry-run", action="store_true", help="Write the run ledger without executing tasks")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = [sys.executable, "-m", "auto_voice.cli", "swarm"]
    env = dict(os.environ)
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT / 'src'}:{current_pythonpath}" if current_pythonpath else str(PROJECT_ROOT / "src")
    )
    if args.status:
        command.extend(["status", "--run-id", args.status])
    else:
        command.extend(["run", "--manifest", str(MANIFESTS[args.swarm])])
        if args.run_id:
            command.extend(["--run-id", args.run_id])
        if args.dry_run:
            command.append("--dry-run")
    return subprocess.call(command, cwd=PROJECT_ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
