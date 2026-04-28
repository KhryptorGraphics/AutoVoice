from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (PROJECT_ROOT / path).read_text(encoding="utf-8")


def test_operator_console_surfaces_benchmark_evidence_states():
    api = _read("frontend/src/services/api.ts")
    page = _read("frontend/src/pages/SystemStatusPage.tsx")

    assert "getLatestBenchmarkDashboard" in api
    assert "getLatestReleaseEvidence" in api
    assert "/reports/benchmarks/latest" in api
    assert "/reports/release-evidence/latest" in api
    assert "data-testid=\"benchmark-evidence-panel\"" in page
    assert "Benchmark evidence unavailable" in page
    assert "Quality gate passed" in page
    assert "fixture_tier" in page


def test_conversion_history_has_artifact_comparison_empty_states():
    page = _read("frontend/src/pages/ConversionHistoryPage.tsx")

    assert "function ArtifactComparison" in page
    assert "data-testid={`artifact-comparison-${record.id}`}" in page
    assert "Converted vocals" in page
    assert "Instrumental" in page
    assert "Not available for this conversion" in page


def test_voice_profile_training_readiness_covers_threshold_states():
    page = _read("frontend/src/pages/VoiceProfilePage.tsx")

    assert "function TrainingReadinessPanel" in page
    assert "data-testid=\"training-readiness-panel\"" in page
    assert "data-testid=\"select-full-model-training\"" in page
    assert "Full model ready" in page
    assert "30m threshold met" in page
    assert "Needs more clean vocals" in page


def test_training_console_exposes_granular_backend_driven_controls():
    api = _read("frontend/src/services/api.ts")
    panel = _read("frontend/src/components/TrainingConfigPanel.tsx")
    profile = _read("frontend/src/pages/VoiceProfilePage.tsx")
    workflow = _read("frontend/src/pages/ConversionWorkflowPage.tsx")

    assert "getTrainingConfigOptions" in api
    assert "/training/config-options" in api
    assert "checkpoint_every_steps" in api
    assert "training-preset-selector" in panel
    assert "training-runtime-controls" in panel
    assert "training-optimizer-select" in panel
    assert "training-sample-selection-summary" in profile
    assert "workflow-training-sample-selector" in workflow
