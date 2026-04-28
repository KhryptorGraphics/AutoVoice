#!/usr/bin/env python3
"""Validate canonical benchmark dashboard and release-evidence artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", type=Path, default=Path("reports/benchmarks/latest/benchmark_dashboard.json"))
    parser.add_argument("--release-evidence", type=Path, default=Path("reports/benchmarks/latest/release_evidence.json"))
    parser.add_argument("--suite-config", type=Path, default=Path("config/benchmark_suites.json"))
    args = parser.parse_args(argv)

    errors: list[str] = []
    dashboard: dict = {}
    release_evidence: dict = {}
    suite: dict = {}
    for label, path in (
        ("dashboard", args.dashboard),
        ("release_evidence", args.release_evidence),
        ("suite_config", args.suite_config),
    ):
        try:
            payload = _load_json(path)
        except FileNotFoundError:
            errors.append(f"{label} file missing: {path}")
            continue
        except json.JSONDecodeError as exc:
            errors.append(f"{label} file is not valid JSON: {path}: {exc}")
            continue
        if label == "dashboard":
            dashboard = payload
        elif label == "release_evidence":
            release_evidence = payload
        else:
            suite = payload

    if errors:
        report = {
            "suite_config": str(args.suite_config),
            "dashboard": str(args.dashboard),
            "release_evidence": str(args.release_evidence),
            "ok": False,
            "errors": errors,
        }
        print(json.dumps(report, indent=2))
        return 1

    canonical = suite.get("canonical_pipelines", {})
    if dashboard.get("canonical_pipelines") != canonical:
        errors.append("dashboard canonical_pipelines do not match suite config")
    if release_evidence.get("canonical_pipelines") != canonical:
        errors.append("release evidence canonical_pipelines do not match suite config")

    pipelines = dashboard.get("pipelines", {})
    required_pipelines = [str(name) for name in suite.get("required_pipelines", [])]
    required_metrics = {
        str(name): [str(metric) for metric in metrics]
        for name, metrics in (suite.get("required_metrics", {}) or {}).items()
    }
    minimum_sample_count = int(suite.get("minimum_sample_count", 1))
    required_fixture_tiers = [str(name) for name in suite.get("required_fixture_tiers", [])]

    for pipeline_name in required_pipelines:
        pipeline = pipelines.get(pipeline_name)
        if not isinstance(pipeline, dict):
            errors.append(f"missing required pipeline {pipeline_name}")
            continue
        if int(pipeline.get("sample_count", 0)) < minimum_sample_count:
            errors.append(f"pipeline {pipeline_name} sample_count below minimum {minimum_sample_count}")
        summary = pipeline.get("summary", {})
        if required_fixture_tiers and str(pipeline.get("fixture_tier", "")) not in required_fixture_tiers:
            errors.append(f"pipeline {pipeline_name} fixture_tier is not one of {required_fixture_tiers}")
        for metric in required_metrics.get(pipeline_name, []):
            if metric not in summary:
                errors.append(f"pipeline {pipeline_name} missing required metric {metric}")

    comparisons = dashboard.get("comparisons", {})
    for candidate in suite.get("candidate_pipelines", []):
        if candidate in pipelines and candidate not in comparisons:
            errors.append(f"candidate pipeline {candidate} missing comparison entry")

    if release_evidence.get("pipeline_count") != len(pipelines):
        errors.append("release evidence pipeline_count mismatch")
    if release_evidence.get("comparison_count") != len(comparisons):
        errors.append("release evidence comparison_count mismatch")
    if release_evidence.get("quality_gate_passed") is not True:
        failures = release_evidence.get("quality_failures")
        errors.append(f"release evidence quality gate failed: {failures!r}")
    if required_fixture_tiers:
        tiers = set(str(name) for name in release_evidence.get("fixture_tiers", []))
        if not tiers.intersection(required_fixture_tiers):
            errors.append("release evidence fixture_tiers do not include a required tier")

    report = {
        "suite_config": str(args.suite_config),
        "dashboard": str(args.dashboard),
        "release_evidence": str(args.release_evidence),
        "ok": not errors,
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
