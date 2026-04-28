"""Tests for benchmark gate semantics."""

from scripts.performance_validation import (
    BenchmarkResult,
    PIPELINE_CONFIGS,
    ReportGenerator,
    evaluate_result,
    experimental_skip_gate,
)


def test_realtime_meanvc_thor_evidence_passes_cpu_streaming_gate():
    """The MeanVC production gate matches its CPU-friendly runtime contract."""
    result = BenchmarkResult(
        pipeline_name="Realtime MeanVC (Streaming)",
        pipeline_type="realtime_meanvc",
        audio_duration_sec=10.0,
        processing_time_sec=15.071,
        rtf=1.507,
        latency_ms=294.3,
        gpu_memory_peak_gb=0.0,
        mcd=192.38,
        speaker_similarity=0.910,
        output_sample_rate=16000,
    )

    gate = evaluate_result(result, PIPELINE_CONFIGS["realtime_meanvc"])

    assert gate["target_met"] is True
    assert gate["mcd_gated"] is False
    assert gate["mcd_ok"] is True


def test_realtime_meanvc_report_marks_mcd_informational():
    result = BenchmarkResult(
        pipeline_name="Realtime MeanVC (Streaming)",
        pipeline_type="realtime_meanvc",
        audio_duration_sec=10.0,
        processing_time_sec=15.071,
        rtf=1.507,
        latency_ms=294.3,
        gpu_memory_peak_gb=0.0,
        mcd=192.38,
        speaker_similarity=0.910,
        output_sample_rate=16000,
    )

    report = ReportGenerator([result]).generate_markdown()

    assert "| Realtime MeanVC (Streaming) | 1.507 | 294ms | 0.00GB | 192.38dB | 0.910 | PASS |" in report
    assert "gate: informational" in report


def test_realtime_meanvc_experimental_skip_has_owner_action(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOVOICE_MEANVC_FULL", "1")
    monkeypatch.setattr(
        "scripts.performance_validation.MEANVC_REQUIRED_ASSETS",
        (tmp_path / "missing.pt",),
    )

    gate = experimental_skip_gate("realtime_meanvc")

    assert gate is not None
    assert gate["support_boundary"] == "experimental:meanvc"
    assert gate["owner"] == "model-runtime"
    assert "prepare_meanvc_assets.py" in gate["action"]
    assert gate["missing_assets"] == [str(tmp_path / "missing.pt")]


def test_realtime_meanvc_skip_requires_explicit_full_opt_in(monkeypatch, tmp_path):
    asset = tmp_path / "present.pt"
    asset.write_text("asset", encoding="utf-8")
    monkeypatch.delenv("AUTOVOICE_MEANVC_FULL", raising=False)
    monkeypatch.setattr(
        "scripts.performance_validation.MEANVC_REQUIRED_ASSETS",
        (asset,),
    )

    gate = experimental_skip_gate("realtime_meanvc")

    assert gate is not None
    assert "AUTOVOICE_MEANVC_FULL=1" in gate["reason"]
    assert gate["missing_assets"] == []


def test_realtime_meanvc_skipped_report_is_explicit():
    result = BenchmarkResult(
        pipeline_name="Realtime MeanVC (Streaming)",
        pipeline_type="realtime_meanvc",
        audio_duration_sec=0,
        processing_time_sec=0,
        rtf=0,
        output_sample_rate=16000,
        status="skipped",
        skip_reason="MeanVC runtime assets are missing.",
        owner="model-runtime",
        action="Run scripts/prepare_meanvc_assets.py",
    )

    gate = evaluate_result(result, PIPELINE_CONFIGS["realtime_meanvc"])
    report = ReportGenerator([result]).generate_markdown()

    assert gate["status"] == "SKIP"
    assert gate["target_met"] is True
    assert "| Realtime MeanVC (Streaming) | 0.000 | 0ms | 0.00GB | 0.00dB | 0.000 | SKIP |" in report
    assert "**SKIPPED:** MeanVC runtime assets are missing." in report
    assert "**Owner:** model-runtime" in report
