"""Aggregate benchmark bundles into dashboard and release-evidence reports."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping


HIGHER_IS_BETTER = {
    "speaker_similarity_mean",
    "pitch_corr_mean",
    "mos_pred_mean",
    "pesq_mean",
    "stoi_mean",
    "snr_mean",
}
LOWER_IS_BETTER = {
    "mcd_mean",
    "f0_rmse_mean",
    "latency_ms_mean",
}
DEFAULT_TARGETS = {
    "speaker_similarity_mean": 0.85,
    "pitch_corr_mean": 0.90,
    "pesq_mean": 3.5,
    "stoi_mean": 0.85,
    "mcd_mean": 5.0,
    "f0_rmse_mean": 20.0,
}
REPORT_SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_benchmark_bundle(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "summary" not in payload:
        raise ValueError(f"Benchmark bundle missing summary: {path}")
    return payload


def _metric_status(metric: str, value: float) -> str:
    target = DEFAULT_TARGETS.get(metric)
    if target is None:
        return "n/a"
    if metric in HIGHER_IS_BETTER:
        return "pass" if value >= target else "fail"
    if metric in LOWER_IS_BETTER:
        return "pass" if value <= target else "fail"
    return "n/a"


def _comparison_status(metric: str, candidate: float, canonical: float) -> bool:
    if metric in HIGHER_IS_BETTER:
        return candidate >= canonical
    if metric in LOWER_IS_BETTER:
        return candidate <= canonical
    return False


def _resolve_git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _build_report_provenance(
    bundles: Mapping[str, Mapping[str, Any]],
    *,
    generator: str,
) -> Dict[str, Any]:
    source_bundles = sorted(
        str(payload.get("source_bundle"))
        for payload in bundles.values()
        if payload.get("source_bundle")
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generator": generator,
        "git_sha": _resolve_git_sha(),
        "source_bundles": source_bundles,
    }


def build_benchmark_dashboard(
    bundles: Mapping[str, Mapping[str, Any]],
    *,
    canonical_offline: str = "quality_seedvc",
    canonical_live: str = "realtime",
    target_hardware: str = "NVIDIA Thor",
) -> Dict[str, Any]:
    """Build a publishable benchmark dashboard from per-pipeline bundles."""
    generated_at = datetime.now(timezone.utc).isoformat()
    pipelines: Dict[str, Any] = {}

    for pipeline_name, payload in bundles.items():
        summary = dict(payload.get("summary", {}))
        metric_basis = dict(payload.get("metric_basis", {}) or {})
        metric_applicability = dict(payload.get("metric_applicability", {}) or {})
        metrics = {}
        for metric, value in summary.items():
            if isinstance(value, (int, float)):
                applicable = bool(metric_applicability.get(metric, True))
                metrics[metric] = {
                    "value": float(value),
                    "target_status": _metric_status(metric, float(value)) if applicable else "n/a",
                    "applicable": applicable,
                }
                if metric in metric_basis:
                    metrics[metric]["basis"] = str(metric_basis[metric])
        pipelines[pipeline_name] = {
            "title": payload.get("title", pipeline_name),
            "sample_count": int(summary.get("sample_count", payload.get("sample_count", 0) or 0)),
            "fixture_tier": payload.get("fixture_tier", summary.get("fixture_tier", "unspecified")),
            "fixture_suite": payload.get("fixture_suite", summary.get("fixture_suite", "unspecified")),
            "summary": metrics,
            "source_bundle": payload.get("source_bundle"),
        }

    comparisons: Dict[str, Any] = {}
    for candidate_name, candidate in pipelines.items():
        canonical = canonical_live if "realtime" in candidate_name else canonical_offline
        if candidate_name == canonical or canonical not in pipelines:
            continue
        quality_checks = []
        for metric in sorted((HIGHER_IS_BETTER | LOWER_IS_BETTER) - {"latency_ms_mean"}):
            candidate_metric = candidate["summary"].get(metric, {}).get("value")
            canonical_metric = pipelines[canonical]["summary"].get(metric, {}).get("value")
            candidate_applicable = candidate["summary"].get(metric, {}).get("applicable", True)
            canonical_applicable = pipelines[canonical]["summary"].get(metric, {}).get("applicable", True)
            if not candidate_applicable or not canonical_applicable:
                continue
            if candidate_metric is None or canonical_metric is None:
                continue
            quality_checks.append(_comparison_status(metric, candidate_metric, canonical_metric))

        latency_ok = True
        candidate_latency = candidate["summary"].get("latency_ms_mean", {}).get("value")
        canonical_latency = pipelines[canonical]["summary"].get("latency_ms_mean", {}).get("value")
        if candidate_latency is not None and canonical_latency is not None:
            latency_ok = candidate_latency <= canonical_latency * 1.10

        meets_or_beats = bool(quality_checks) and all(quality_checks) and latency_ok
        comparisons[candidate_name] = {
            "canonical_pipeline": canonical,
            "meets_or_beats_canonical": meets_or_beats,
            "quality_checks_passed": all(quality_checks) if quality_checks else False,
            "latency_guard_passed": latency_ok,
        }

    promoted_candidates = [
        name for name, comparison in comparisons.items()
        if comparison["meets_or_beats_canonical"]
    ]
    return {
        "generated_at": generated_at,
        "provenance": _build_report_provenance(
            bundles,
            generator="auto_voice.evaluation.benchmark_reporting.build_benchmark_dashboard",
        ),
        "target_hardware": target_hardware,
        "canonical_pipelines": {
            "offline": canonical_offline,
            "live": canonical_live,
        },
        "pipelines": pipelines,
        "comparisons": comparisons,
        "promotable_candidates": promoted_candidates,
    }


def build_release_evidence(
    dashboard: Mapping[str, Any],
    *,
    health_url: str = "/api/v1/health",
) -> Dict[str, Any]:
    """Build a compact release-evidence payload from the benchmark dashboard."""
    pipelines = dashboard.get("pipelines", {})
    comparisons = dashboard.get("comparisons", {})
    quality_failures = [
        {
            "pipeline": pipeline_name,
            "metric": metric_name,
            "value": metric_data.get("value"),
            "target_status": metric_data.get("target_status"),
        }
        for pipeline_name, pipeline in pipelines.items()
        for metric_name, metric_data in pipeline.get("summary", {}).items()
        if metric_data.get("target_status") == "fail"
    ]
    return {
        "generated_at": dashboard.get("generated_at"),
        "provenance": dashboard.get("provenance", {}),
        "target_hardware": dashboard.get("target_hardware"),
        "health_url": health_url,
        "canonical_pipelines": dashboard.get("canonical_pipelines", {}),
        "pipeline_count": len(pipelines),
        "comparison_count": len(comparisons),
        "promotable_candidates": dashboard.get("promotable_candidates", []),
        "fixture_tiers": sorted({
            str(pipeline.get("fixture_tier", "unspecified"))
            for pipeline in pipelines.values()
        }),
        "quality_gate_passed": not quality_failures,
        "quality_failures": quality_failures,
    }


def write_benchmark_dashboard(
    bundles: Mapping[str, Mapping[str, Any]],
    output_dir: str | Path,
    *,
    canonical_offline: str = "quality_seedvc",
    canonical_live: str = "realtime",
    target_hardware: str = "NVIDIA Thor",
) -> Dict[str, Any]:
    """Write dashboard and release-evidence artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dashboard = build_benchmark_dashboard(
        bundles,
        canonical_offline=canonical_offline,
        canonical_live=canonical_live,
        target_hardware=target_hardware,
    )
    release_evidence = build_release_evidence(dashboard)

    dashboard_path = output_path / "benchmark_dashboard.json"
    release_path = output_path / "release_evidence.json"
    markdown_path = output_path / "benchmark_dashboard.md"

    dashboard_path.write_text(json.dumps(dashboard, indent=2), encoding="utf-8")
    release_path.write_text(json.dumps(release_evidence, indent=2), encoding="utf-8")
    markdown_path.write_text(render_benchmark_dashboard_markdown(dashboard), encoding="utf-8")

    return {
        "dashboard": dashboard,
        "release_evidence": release_evidence,
        "dashboard_path": str(dashboard_path),
        "release_evidence_path": str(release_path),
        "markdown_path": str(markdown_path),
    }


def render_benchmark_dashboard_markdown(dashboard: Mapping[str, Any]) -> str:
    """Render a compact Markdown report for the benchmark dashboard."""
    lines = [
        "# Canonical Benchmark Dashboard",
        "",
        f"Generated: {dashboard.get('generated_at')}",
        f"Target Hardware: {dashboard.get('target_hardware')}",
        "",
        "| Pipeline | Sample Count | Promotable |",
        "|----------|--------------|------------|",
    ]
    comparisons = dashboard.get("comparisons", {})
    for pipeline_name, pipeline in dashboard.get("pipelines", {}).items():
        promotable = comparisons.get(pipeline_name, {}).get("meets_or_beats_canonical")
        lines.append(
            f"| {pipeline_name} | {pipeline.get('sample_count', 0)} | "
            f"{'yes' if promotable else 'no'} |"
        )
    return "\n".join(lines) + "\n"
