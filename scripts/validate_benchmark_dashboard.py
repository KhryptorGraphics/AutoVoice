#!/usr/bin/env python3
"""Validate canonical benchmark dashboard and release-evidence artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _current_git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _looks_like_tmp_path(value: object) -> bool:
    if not value:
        return False
    try:
        path = Path(str(value)).expanduser()
    except TypeError:
        return False
    return path.is_absolute() and path.parts[:2] == ("/", "tmp")


def _source_bundle_paths(dashboard: dict, release_evidence: dict) -> list[str]:
    paths: list[str] = []
    for pipeline in dashboard.get("pipelines", {}).values():
        if isinstance(pipeline, dict) and pipeline.get("source_bundle"):
            paths.append(str(pipeline["source_bundle"]))
    provenance = release_evidence.get("provenance", {}) or {}
    paths.extend(str(path) for path in provenance.get("source_bundles", []) or [])
    return sorted(set(paths))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", type=Path, default=Path("reports/benchmarks/latest/benchmark_dashboard.json"))
    parser.add_argument("--release-evidence", type=Path, default=Path("reports/benchmarks/latest/release_evidence.json"))
    parser.add_argument("--suite-config", type=Path, default=Path("config/benchmark_suites.json"))
    parser.add_argument(
        "--expected-git-sha",
        default=os.environ.get("GITHUB_SHA"),
        help="Expected provenance git SHA. Defaults to GITHUB_SHA when set.",
    )
    parser.add_argument(
        "--current-git-sha",
        action="store_true",
        help="Require benchmark provenance to match the current repository HEAD.",
    )
    parser.add_argument(
        "--release-grade",
        action="store_true",
        help="Reject smoke-only fixture evidence and temporary source bundles.",
    )
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
                continue
            metric_payload = summary.get(metric)
            if not isinstance(metric_payload, dict):
                errors.append(f"pipeline {pipeline_name} metric {metric} must include value/status metadata")
                continue
            if "value" not in metric_payload:
                errors.append(f"pipeline {pipeline_name} metric {metric} missing value")
            if metric_payload.get("applicable") is False and not metric_payload.get("basis"):
                errors.append(
                    f"pipeline {pipeline_name} non-applicable required metric {metric} must document basis"
                )

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
    exemption_keys = {
        (str(item.get("pipeline")), str(item.get("metric")))
        for item in release_evidence.get("metric_exemptions", []) or []
        if isinstance(item, dict)
    }
    for pipeline_name, pipeline in pipelines.items():
        for metric_name, metric_payload in (pipeline.get("summary", {}) or {}).items():
            if isinstance(metric_payload, dict) and metric_payload.get("applicable") is False:
                if (str(pipeline_name), str(metric_name)) not in exemption_keys:
                    errors.append(
                        f"release evidence missing metric_exemption for {pipeline_name}.{metric_name}"
                    )
    if required_fixture_tiers:
        tiers = set(str(name) for name in release_evidence.get("fixture_tiers", []))
        if not tiers.intersection(required_fixture_tiers):
            errors.append("release evidence fixture_tiers do not include a required tier")

    expected_sha = args.expected_git_sha
    if args.current_git_sha:
        expected_sha = _current_git_sha()
        if not expected_sha:
            errors.append("unable to resolve current git HEAD for provenance validation")
    if expected_sha:
        dashboard_sha = (dashboard.get("provenance", {}) or {}).get("git_sha")
        release_sha = (release_evidence.get("provenance", {}) or {}).get("git_sha")
        if dashboard_sha != expected_sha:
            errors.append(f"dashboard provenance git_sha {dashboard_sha!r} does not match {expected_sha!r}")
        if release_sha != expected_sha:
            errors.append(f"release evidence provenance git_sha {release_sha!r} does not match {expected_sha!r}")

    tmp_sources = [path for path in _source_bundle_paths(dashboard, release_evidence) if _looks_like_tmp_path(path)]
    if tmp_sources:
        errors.append(f"benchmark source bundles must not come from /tmp: {tmp_sources}")

    if args.release_grade:
        tiers = set(str(name) for name in release_evidence.get("fixture_tiers", []))
        if not tiers or tiers == {"smoke"} or "quality" not in tiers and "full" not in tiers:
            errors.append("release-grade benchmark evidence requires quality or full fixture tiers")

    report = {
        "suite_config": str(args.suite_config),
        "dashboard": str(args.dashboard),
        "release_evidence": str(args.release_evidence),
        "expected_git_sha": expected_sha,
        "release_grade": args.release_grade,
        "ok": not errors,
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
