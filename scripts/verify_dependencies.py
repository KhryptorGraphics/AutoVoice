#!/usr/bin/env python3
"""Verify the canonical AutoVoice runtime environment and dependency stack."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from auto_voice.utils.dependency_verification import format_audit, run_dependency_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify AutoVoice dependencies")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the audit as JSON instead of text",
    )
    parser.add_argument(
        "--output",
        help="Optional output path for the rendered report",
    )
    parser.add_argument(
        "--require-env",
        action="store_true",
        help="Fail if the current interpreter is not the canonical autovoice-thor env",
    )
    parser.add_argument(
        "--require-tensorrt",
        action="store_true",
        help="Treat TensorRT as a required dependency",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = run_dependency_audit(require_tensorrt=args.require_tensorrt)

    if args.json:
        rendered = json.dumps(audit, indent=2, sort_keys=True)
    else:
        rendered = format_audit(audit)

    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")

    failed = list(audit["failed_required"])
    if args.require_env and not audit["python"]["matches_expected_env"]:
        failed.append("python_env")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
