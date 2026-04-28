#!/usr/bin/env python3
"""Emit machine-readable support gates for experimental hardware lanes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.performance_validation import experimental_skip_gate


def _dependency_available(import_path: str) -> bool:
    return importlib.util.find_spec(import_path) is not None


def _hq_svc_gate() -> dict[str, Any]:
    missing = [
        name
        for name, import_path in {
            "fairseq": "fairseq",
            "local_attention": "local_attention",
        }.items()
        if not _dependency_available(import_path)
    ]
    full_enabled = os.environ.get("AUTOVOICE_HQSVC_FULL") == "1"
    enabled = full_enabled and not missing
    return {
        "status": "ready" if enabled else "gated",
        "support_boundary": "experimental:hq_svc",
        "owner": "model-runtime",
        "action": (
            "Install the HQ-SVC experimental dependency set, restore HQ-SVC assets, "
            "set AUTOVOICE_HQSVC_FULL=1, and rerun the CUDA validation on hardware."
        ),
        "reason": (
            "HQ-SVC is experimental and disabled unless AUTOVOICE_HQSVC_FULL=1 "
            "and optional dependencies are present."
            if not enabled
            else "HQ-SVC experimental prerequisites are present."
        ),
        "missing_dependencies": missing,
        "full_enabled": full_enabled,
    }


def _meanvc_gate() -> dict[str, Any]:
    skip_gate = experimental_skip_gate("realtime_meanvc")
    if skip_gate is None:
        return {
            "status": "ready",
            "support_boundary": "experimental:meanvc",
            "owner": "model-runtime",
            "action": "Run scripts/validate_cuda_stack.sh --pipeline realtime_meanvc on the target hardware.",
            "reason": "MeanVC runtime assets are present; the performance gate must execute.",
            "missing_assets": [],
        }
    return {
        "status": "gated",
        "support_boundary": skip_gate["support_boundary"],
        "owner": skip_gate["owner"],
        "action": skip_gate["action"],
        "reason": skip_gate["reason"],
        "missing_assets": skip_gate["missing_assets"],
    }


def build_report(*, pipeline: str) -> dict[str, Any]:
    lanes: dict[str, Any] = {"hq_svc": _hq_svc_gate()}
    if pipeline in {"all", "realtime_meanvc"}:
        lanes["realtime_meanvc"] = _meanvc_gate()
    return {
        "schema_version": 1,
        "pipeline": pipeline,
        "lanes": lanes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", default="all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(pipeline=args.pipeline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "report": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
