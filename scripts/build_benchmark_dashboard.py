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


def _bundle_from_comprehensive_report(
    report_path: Path,
    *,
    fixture_tier: str,
    fixture_suite: str,
) -> dict[str, dict]:
    """Convert benchmark_pipelines_comprehensive.py output into dashboard bundles."""

    report = json.loads(report_path.read_text(encoding="utf-8"))
    raw_pipelines = report.get("pipelines")
    if not isinstance(raw_pipelines, dict):
        raise ValueError(f"Comprehensive benchmark report missing pipelines: {report_path}")

    bundles: dict[str, dict] = {}
    for pipeline_name, metrics in raw_pipelines.items():
        if not isinstance(metrics, dict):
            continue
        if metrics.get("success") is False:
            continue

        iterations = int(metrics.get("iterations") or len(metrics.get("times") or []) or 1)
        summary = {
            "sample_count": max(1, iterations),
            "fixture_tier": fixture_tier,
            "fixture_suite": fixture_suite,
            "speaker_similarity_mean": float(metrics.get("speaker_similarity", 0.0) or 0.0),
            "pitch_corr_mean": float(metrics.get("pitch_corr", metrics.get("pitch_corr_mean", 0.0)) or 0.0),
            "mcd_mean": float(metrics.get("mcd", 0.0) or 0.0),
            "latency_ms_mean": float(metrics.get("latency_ms", 0.0) or 0.0),
            "rtf_mean": float(metrics.get("rtf_mean", metrics.get("rtf", 0.0)) or 0.0),
            "vram_mb_peak": float(metrics.get("gpu_memory_peak_mb", 0.0) or 0.0),
        }
        bundles[pipeline_name] = {
            "title": metrics.get("pipeline_name") or pipeline_name,
            "fixture_tier": fixture_tier,
            "fixture_suite": fixture_suite,
            "summary": summary,
            "source_bundle": str(report_path),
        }
    return bundles


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
        "--comprehensive-report",
        type=Path,
        help="Output from scripts/benchmark_pipelines_comprehensive.py to convert into dashboard bundles.",
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
    parser.add_argument("--fixture-tier", default="quality")
    parser.add_argument("--fixture-suite", default="comprehensive-pipeline-benchmark")
    args = parser.parse_args(argv)

    if not args.bundle and not args.comprehensive_report:
        raise SystemExit("At least one --bundle PIPELINE=PATH or --comprehensive-report PATH is required")

    bundles = {}
    for pipeline_name, path in args.bundle:
        payload = load_benchmark_bundle(path)
        payload["source_bundle"] = str(path)
        bundles[pipeline_name] = payload
    if args.comprehensive_report:
        bundles.update(
            _bundle_from_comprehensive_report(
                args.comprehensive_report,
                fixture_tier=args.fixture_tier,
                fixture_suite=args.fixture_suite,
            )
        )

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
