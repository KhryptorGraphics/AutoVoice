"""Tests for benchmark gate semantics."""

from scripts.performance_validation import (
    BenchmarkResult,
    PIPELINE_CONFIGS,
    ReportGenerator,
    evaluate_result,
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
