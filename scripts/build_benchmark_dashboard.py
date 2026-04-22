#!/usr/bin/env python3
"""Aggregate per-pipeline benchmark bundles into dashboard and release evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from auto_voice.evaluation.benchmark_reporting import load_benchmark_bundle, write_benchmark_dashboard


def _parse_bundle_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Expected PIPELINE=PATH")
    pipeline_name, path = raw.split("=", 1)
    if not pipeline_name:
        raise argparse.ArgumentTypeError("Pipeline name is required")
    return pipeline_name, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        action="append",
        default=[],
        type=_parse_bundle_arg,
        help="Pipeline benchmark summary in the form PIPELINE=path/to/summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/benchmarks/latest"),
        help="Directory for dashboard and release evidence output.",
    )
    parser.add_argument("--canonical-offline", default="quality_seedvc")
    parser.add_argument("--canonical-live", default="realtime")
    parser.add_argument("--target-hardware", default="NVIDIA Thor")
    args = parser.parse_args(argv)

    if not args.bundle:
        raise SystemExit("At least one --bundle PIPELINE=PATH is required")

    bundles = {}
    for pipeline_name, path in args.bundle:
        payload = load_benchmark_bundle(path)
        payload["source_bundle"] = str(path)
        bundles[pipeline_name] = payload

    result = write_benchmark_dashboard(
        bundles,
        args.output_dir,
        canonical_offline=args.canonical_offline,
        canonical_live=args.canonical_live,
        target_hardware=args.target_hardware,
    )
    print(json.dumps({
        "dashboard_path": result["dashboard_path"],
        "release_evidence_path": result["release_evidence_path"],
        "markdown_path": result["markdown_path"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
