#!/usr/bin/env python3
"""Audit nested vendor model repositories tracked by the parent repo."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from auto_voice.utils.vendor_repo_audit import format_vendor_repo_audit, run_vendor_repo_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit AutoVoice vendor model repos")
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
        "--require-clean",
        action="store_true",
        help="Fail if any nested vendor repo has tracked or untracked dirt",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = run_vendor_repo_audit()

    if args.json:
        rendered = json.dumps(audit, indent=2, sort_keys=True)
    else:
        rendered = format_vendor_repo_audit(audit)

    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")

    if not audit["contract_ok"]:
        return 1
    if args.require_clean and not audit["clean"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
