from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from auto_voice.evaluation.benchmark_reporting import (
    build_benchmark_dashboard,
    build_release_evidence,
    write_benchmark_dashboard,
)
from auto_voice.experimental_registry import evaluate_evidence_gates
from auto_voice.training.sample_quality import analyze_training_sample, summarize_training_samples
from tests.fixtures.audio import write_voiced_wav


def _write_audio(path: Path, samples: np.ndarray, sample_rate: int = 22050) -> None:
    sf.write(path, samples.astype(np.float32), sample_rate)


def test_analyze_training_sample_flags_silence_and_short_audio(tmp_path):
    audio_path = tmp_path / "silent.wav"
    _write_audio(audio_path, np.zeros(2000, dtype=np.float32))

    quality = analyze_training_sample(audio_path, provenance="unit-test")

    assert quality["qa_status"] == "fail"
    assert "effectively_silent" in quality["issues"]
    assert quality["provenance"] == "unit-test"


def test_analyze_training_sample_accepts_voiced_fixture(tmp_path):
    audio_path = tmp_path / "voiced.wav"
    write_voiced_wav(audio_path, duration_seconds=2.0, sample_rate=22050)

    quality = analyze_training_sample(audio_path, provenance="unit-test")

    assert quality["qa_status"] in {"pass", "warn"}
    assert quality["duration_seconds"] >= 1.5
    assert "effectively_silent" not in quality["issues"]
    assert "sample_too_short" not in quality["issues"]


def test_summarize_training_samples_reports_recommendations():
    samples = [
        type("Sample", (), {"sample_id": "good", "duration": 2.0, "quality_metadata": {"qa_status": "pass"}})(),
        type(
            "Sample",
            (),
            {
                "sample_id": "bad",
                "duration": 1.0,
                "quality_metadata": {"qa_status": "fail", "recommendations": ["Replace the sample."]},
            },
        )(),
    ]

    summary = summarize_training_samples(samples)

    assert summary["sample_count"] == 2
    assert summary["trainable_sample_count"] == 1
    assert "bad" in summary["failing_sample_ids"]
    assert any("Replace" in item for item in summary["recommendations"])


def test_benchmark_dashboard_and_release_evidence_round_trip(tmp_path):
    dashboard_result = write_benchmark_dashboard(
        {
            "quality_seedvc": {
                "title": "quality_seedvc",
                "fixture_tier": "quality",
                "summary": {
                    "sample_count": 2,
                    "speaker_similarity_mean": 0.91,
                    "pitch_corr_mean": 0.93,
                    "mcd_mean": 4.1,
                    "latency_ms_mean": 120.0,
                },
            },
            "hq_svc": {
                "title": "hq_svc",
                "fixture_tier": "quality",
                "summary": {
                    "sample_count": 2,
                    "speaker_similarity_mean": 0.94,
                    "pitch_corr_mean": 0.95,
                    "mcd_mean": 3.9,
                    "latency_ms_mean": 118.0,
                },
            },
            "realtime": {
                "title": "realtime",
                "fixture_tier": "quality",
                "summary": {
                    "sample_count": 2,
                    "speaker_similarity_mean": 0.87,
                    "pitch_corr_mean": 0.91,
                    "mcd_mean": 4.5,
                    "latency_ms_mean": 42.0,
                },
            },
        },
        tmp_path / "reports",
    )

    dashboard = json.loads(Path(dashboard_result["dashboard_path"]).read_text(encoding="utf-8"))
    release = json.loads(Path(dashboard_result["release_evidence_path"]).read_text(encoding="utf-8"))

    assert dashboard["comparisons"]["hq_svc"]["meets_or_beats_canonical"] is True
    assert release["quality_gate_passed"] is True
    assert dashboard["pipelines"]["quality_seedvc"]["fixture_tier"] == "quality"
    assert release["fixture_tiers"] == ["quality"]
    assert dashboard["provenance"]["schema_version"] == 1
    assert dashboard["provenance"]["generator"]
    assert release["provenance"] == dashboard["provenance"]


def test_benchmark_release_evidence_skips_non_applicable_metrics():
    dashboard = build_benchmark_dashboard(
        {
            "quality_seedvc": {
                "summary": {
                    "sample_count": 1,
                    "speaker_similarity_mean": 0.91,
                    "pitch_corr_mean": 0.93,
                    "mcd_mean": 999.0,
                    "latency_ms_mean": 120.0,
                },
                "metric_applicability": {"mcd_mean": False},
                "metric_basis": {"mcd_mean": "not_applicable_without_aligned_same_content_target"},
            },
            "realtime": {
                "summary": {
                    "sample_count": 1,
                    "speaker_similarity_mean": 0.87,
                    "pitch_corr_mean": 0.91,
                    "mcd_mean": 999.0,
                    "latency_ms_mean": 42.0,
                },
                "metric_applicability": {"mcd_mean": False},
                "metric_basis": {"mcd_mean": "not_applicable_without_aligned_same_content_target"},
            },
        }
    )
    release = build_release_evidence(dashboard)

    metric = dashboard["pipelines"]["quality_seedvc"]["summary"]["mcd_mean"]
    assert metric["target_status"] == "n/a"
    assert metric["applicable"] is False
    assert metric["basis"] == "not_applicable_without_aligned_same_content_target"
    assert {
        "pipeline": "quality_seedvc",
        "metric": "mcd_mean",
        "value": 999.0,
        "basis": "not_applicable_without_aligned_same_content_target",
    } in release["metric_exemptions"]
    assert release["quality_gate_passed"] is True


def test_benchmark_release_evidence_deduplicates_source_bundles(tmp_path):
    source_bundle = tmp_path / "comprehensive_report.json"
    source_bundle.write_text("{}", encoding="utf-8")

    dashboard = build_benchmark_dashboard(
        {
            "quality_seedvc": {
                "summary": {"sample_count": 1, "speaker_similarity_mean": 0.91},
                "source_bundle": str(source_bundle),
            },
            "realtime": {
                "summary": {"sample_count": 1, "speaker_similarity_mean": 0.88},
                "source_bundle": str(source_bundle),
            },
        }
    )

    assert dashboard["provenance"]["source_bundles"] == [str(source_bundle)]


def test_experimental_registry_benchmark_gate_requires_candidate_to_win(tmp_path):
    reports_dir = tmp_path / "reports" / "benchmarks" / "latest"
    reports_dir.mkdir(parents=True)
    dashboard = build_benchmark_dashboard(
        {
            "quality_seedvc": {"summary": {"speaker_similarity_mean": 0.9, "mcd_mean": 4.0, "latency_ms_mean": 100.0}},
            "hq_svc": {"summary": {"speaker_similarity_mean": 0.88, "mcd_mean": 4.5, "latency_ms_mean": 130.0}},
        }
    )
    (reports_dir / "benchmark_dashboard.json").write_text(json.dumps(dashboard), encoding="utf-8")

    registry = {
        "schema_version": 1,
        "features": {
            "hq_svc": {
                "status": "promotable",
                "component_paths": ["README.md"],
                "benchmark_gate": {
                    "dashboard_artifact": "reports/benchmarks/latest/benchmark_dashboard.json",
                    "candidate_pipeline": "hq_svc",
                    "canonical_pipeline": "quality_seedvc",
                },
                "evidence": {
                    "quality": {"artifacts": ["reports/benchmarks/latest/benchmark_dashboard.json"], "notes": ""},
                    "latency": {"artifacts": ["reports/benchmarks/latest/benchmark_dashboard.json"], "notes": ""},
                    "packaging": {"artifacts": ["reports/benchmarks/latest/benchmark_dashboard.json"], "notes": ""},
                    "deployability": {"artifacts": ["reports/benchmarks/latest/benchmark_dashboard.json"], "notes": ""},
                },
            }
        },
    }

    result = evaluate_evidence_gates(registry, root_dir=tmp_path)
    feature = result["features"]["hq_svc"]

    assert feature["benchmark_gate"]["configured"] is True
    assert feature["benchmark_gate"]["satisfied"] is False
    assert feature["gate_passed"] is False


def test_validate_benchmark_dashboard_script_round_trip(tmp_path):
    reports_dir = tmp_path / "reports" / "benchmarks" / "latest"
    reports_dir.mkdir(parents=True)
    suite_path = tmp_path / "benchmark_suites.json"
    suite_path.write_text(
        json.dumps(
            {
                "canonical_pipelines": {"offline": "quality_seedvc", "live": "realtime"},
                "required_pipelines": ["quality_seedvc", "realtime"],
                "minimum_sample_count": 1,
                "required_metrics": {
                    "quality_seedvc": ["speaker_similarity_mean", "pitch_corr_mean", "mcd_mean", "latency_ms_mean"],
                    "realtime": ["speaker_similarity_mean", "pitch_corr_mean", "mcd_mean", "latency_ms_mean"],
                },
                "candidate_pipelines": ["hq_svc"],
            }
        ),
        encoding="utf-8",
    )
    result = write_benchmark_dashboard(
        {
            "quality_seedvc": {
                "title": "quality_seedvc",
                "summary": {
                    "sample_count": 1,
                    "speaker_similarity_mean": 0.91,
                    "pitch_corr_mean": 0.93,
                    "mcd_mean": 4.1,
                    "latency_ms_mean": 120.0,
                },
            },
            "realtime": {
                "title": "realtime",
                "summary": {
                    "sample_count": 1,
                    "speaker_similarity_mean": 0.87,
                    "pitch_corr_mean": 0.91,
                    "mcd_mean": 4.5,
                    "latency_ms_mean": 42.0,
                },
            },
            "hq_svc": {
                "title": "hq_svc",
                "summary": {
                    "sample_count": 1,
                    "speaker_similarity_mean": 0.95,
                    "pitch_corr_mean": 0.96,
                    "mcd_mean": 3.8,
                    "latency_ms_mean": 41.0,
                },
            },
        },
        reports_dir,
    )

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_benchmark_dashboard.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--dashboard",
            result["dashboard_path"],
            "--release-evidence",
            result["release_evidence_path"],
            "--suite-config",
            str(suite_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True


def test_validate_benchmark_dashboard_requires_basis_for_non_applicable_metric(tmp_path):
    reports_dir = tmp_path / "reports" / "benchmarks" / "latest"
    reports_dir.mkdir(parents=True)
    suite_path = tmp_path / "benchmark_suites.json"
    suite_path.write_text(
        json.dumps(
            {
                "canonical_pipelines": {"offline": "quality_seedvc", "live": "realtime"},
                "required_pipelines": ["quality_seedvc"],
                "required_metrics": {"quality_seedvc": ["mcd_mean"]},
            }
        ),
        encoding="utf-8",
    )
    result = write_benchmark_dashboard(
        {
            "quality_seedvc": {
                "title": "quality_seedvc",
                "summary": {"sample_count": 1, "mcd_mean": 999.0},
                "metric_applicability": {"mcd_mean": False},
            },
        },
        reports_dir,
    )

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_benchmark_dashboard.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--dashboard",
            result["dashboard_path"],
            "--release-evidence",
            result["release_evidence_path"],
            "--suite-config",
            str(suite_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert any("non-applicable required metric mcd_mean must document basis" in err for err in payload["errors"])


def test_validate_benchmark_dashboard_requires_configured_metric_exemption(tmp_path):
    reports_dir = tmp_path / "reports" / "benchmarks" / "latest"
    reports_dir.mkdir(parents=True)
    suite_path = tmp_path / "benchmark_suites.json"
    suite_path.write_text(
        json.dumps(
            {
                "canonical_pipelines": {"offline": "quality_seedvc", "live": "realtime"},
                "required_pipelines": ["quality_seedvc"],
                "required_metrics": {"quality_seedvc": ["mcd_mean"]},
            }
        ),
        encoding="utf-8",
    )
    result = write_benchmark_dashboard(
        {
            "quality_seedvc": {
                "title": "quality_seedvc",
                "summary": {"sample_count": 1, "mcd_mean": 999.0},
                "metric_applicability": {"mcd_mean": False},
                "metric_basis": {"mcd_mean": "not_applicable_without_aligned_same_content_target"},
            },
        },
        reports_dir,
    )

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_benchmark_dashboard.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--dashboard",
            result["dashboard_path"],
            "--release-evidence",
            result["release_evidence_path"],
            "--suite-config",
            str(suite_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert any("non-applicable required metric mcd_mean is not allowed" in err for err in payload["errors"])


def test_validate_benchmark_dashboard_accepts_configured_metric_exemption(tmp_path):
    reports_dir = tmp_path / "reports" / "benchmarks" / "latest"
    reports_dir.mkdir(parents=True)
    suite_path = tmp_path / "benchmark_suites.json"
    suite_path.write_text(
        json.dumps(
            {
                "canonical_pipelines": {"offline": "quality_seedvc", "live": "realtime"},
                "required_pipelines": ["quality_seedvc"],
                "required_metrics": {"quality_seedvc": ["mcd_mean"]},
                "allowed_metric_exemptions": [
                    {
                        "pipeline": "quality_seedvc",
                        "metric": "mcd_mean",
                        "basis": "not_applicable_without_aligned_same_content_target",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = write_benchmark_dashboard(
        {
            "quality_seedvc": {
                "title": "quality_seedvc",
                "summary": {"sample_count": 1, "mcd_mean": 999.0},
                "metric_applicability": {"mcd_mean": False},
                "metric_basis": {"mcd_mean": "not_applicable_without_aligned_same_content_target"},
            },
        },
        reports_dir,
    )

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_benchmark_dashboard.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--dashboard",
            result["dashboard_path"],
            "--release-evidence",
            result["release_evidence_path"],
            "--suite-config",
            str(suite_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_validate_benchmark_dashboard_rejects_tmp_source_and_sha_drift(tmp_path):
    reports_dir = tmp_path / "reports" / "benchmarks" / "latest"
    reports_dir.mkdir(parents=True)
    suite_path = tmp_path / "benchmark_suites.json"
    suite_path.write_text(
        json.dumps(
            {
                "canonical_pipelines": {"offline": "quality_seedvc", "live": "realtime"},
                "required_pipelines": ["quality_seedvc", "realtime"],
                "minimum_sample_count": 1,
                "required_metrics": {
                    "quality_seedvc": ["speaker_similarity_mean", "pitch_corr_mean", "mcd_mean", "latency_ms_mean"],
                    "realtime": ["speaker_similarity_mean", "pitch_corr_mean", "mcd_mean", "latency_ms_mean"],
                },
            }
        ),
        encoding="utf-8",
    )
    result = write_benchmark_dashboard(
        {
            "quality_seedvc": {
                "title": "quality_seedvc",
                "source_bundle": "/tmp/pytest-stale/quality_seedvc.json",
                "summary": {
                    "sample_count": 1,
                    "speaker_similarity_mean": 0.91,
                    "pitch_corr_mean": 0.93,
                    "mcd_mean": 4.1,
                    "latency_ms_mean": 120.0,
                },
            },
            "realtime": {
                "title": "realtime",
                "summary": {
                    "sample_count": 1,
                    "speaker_similarity_mean": 0.87,
                    "pitch_corr_mean": 0.91,
                    "mcd_mean": 4.5,
                    "latency_ms_mean": 42.0,
                },
            },
        },
        reports_dir,
    )

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_benchmark_dashboard.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--dashboard",
            result["dashboard_path"],
            "--release-evidence",
            result["release_evidence_path"],
            "--suite-config",
            str(suite_path),
            "--expected-git-sha",
            "not-current",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["ok"] is False
    assert any("git_sha" in error for error in payload["errors"])
    assert any("/tmp" in error for error in payload["errors"])


def test_validate_benchmark_dashboard_reports_missing_files_as_json(tmp_path):
    suite_path = tmp_path / "benchmark_suites.json"
    suite_path.write_text(
        json.dumps(
            {
                "canonical_pipelines": {"offline": "quality_seedvc", "live": "realtime"},
                "required_pipelines": ["quality_seedvc", "realtime"],
            }
        ),
        encoding="utf-8",
    )

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_benchmark_dashboard.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--dashboard",
            str(tmp_path / "missing-dashboard.json"),
            "--release-evidence",
            str(tmp_path / "missing-release.json"),
            "--suite-config",
            str(suite_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["ok"] is False
    assert any("file missing" in error for error in payload["errors"])


def test_build_benchmark_dashboard_accepts_comprehensive_report(tmp_path):
    report_path = tmp_path / "comprehensive_report.json"
    report_path.write_text(
        json.dumps(
            {
                "pipelines": {
                    "quality_seedvc": {
                        "pipeline_name": "Quality SeedVC",
                        "success": True,
                        "iterations": 3,
                        "speaker_similarity": 0.91,
                        "pitch_corr": 0.93,
                        "mcd": 4.1,
                        "latency_ms": 120.0,
                        "rtf": 0.8,
                        "gpu_memory_peak_mb": 2048.0,
                    },
                    "realtime": {
                        "pipeline_name": "Realtime",
                        "success": True,
                        "iterations": 3,
                        "speaker_similarity": 0.88,
                        "pitch_corr": 0.91,
                        "mcd": 4.4,
                        "latency_ms": 42.0,
                        "rtf": 0.2,
                        "gpu_memory_peak_mb": 512.0,
                    },
                    "failed_pipeline": {
                        "success": False,
                        "error": "missing model",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "reports"
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_benchmark_dashboard.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--comprehensive-report",
            str(report_path),
            "--fixture-tier",
            "quality",
            "--fixture-suite",
            "unit-real-benchmark",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    dashboard = json.loads((output_dir / "benchmark_dashboard.json").read_text(encoding="utf-8"))
    release = json.loads((output_dir / "release_evidence.json").read_text(encoding="utf-8"))
    assert set(dashboard["pipelines"]) == {"quality_seedvc", "realtime"}
    assert dashboard["pipelines"]["quality_seedvc"]["fixture_suite"] == "unit-real-benchmark"
    assert release["quality_gate_passed"] is True
