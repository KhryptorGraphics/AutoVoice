from __future__ import annotations

import json
from pathlib import Path

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


def _write_audio(path: Path, samples: np.ndarray, sample_rate: int = 22050) -> None:
    sf.write(path, samples.astype(np.float32), sample_rate)


def test_analyze_training_sample_flags_silence_and_short_audio(tmp_path):
    audio_path = tmp_path / "silent.wav"
    _write_audio(audio_path, np.zeros(2000, dtype=np.float32))

    quality = analyze_training_sample(audio_path, provenance="unit-test")

    assert quality["qa_status"] == "fail"
    assert "effectively_silent" in quality["issues"]
    assert quality["provenance"] == "unit-test"


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
    assert dashboard["provenance"]["schema_version"] == 1
    assert dashboard["provenance"]["generator"]
    assert release["provenance"] == dashboard["provenance"]


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
