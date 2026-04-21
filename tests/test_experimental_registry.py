from __future__ import annotations

import json
from pathlib import Path

import pytest

from auto_voice.experimental_registry import (
    DEFAULT_REGISTRY_PATH,
    REQUIRED_EVIDENCE_CATEGORIES,
    evaluate_evidence_gates,
    load_experimental_registry,
    validate_experimental_registry,
    write_experimental_registry,
)


def test_default_registry_marks_quality_upgrades_experimental():
    registry = load_experimental_registry()

    assert Path(DEFAULT_REGISTRY_PATH).exists()
    for feature_id in ("hq_svc", "nsf_harmonic_enhancement", "ecapa2_encoder", "pupu_vocoder_refinement"):
        assert registry["features"][feature_id]["status"] == "experimental"
        assert set(registry["features"][feature_id]["evidence"].keys()) == set(
            REQUIRED_EVIDENCE_CATEGORIES
        )


def test_evidence_gate_allows_experimental_features_without_artifacts(tmp_path: Path):
    registry = load_experimental_registry()
    result = evaluate_evidence_gates(registry, root_dir=Path.cwd())

    assert all(feature["gate_passed"] for feature in result["features"].values())
    assert all(not feature["promotion_ready"] for feature in result["features"].values())


def test_evidence_gate_requires_artifacts_before_promotion(tmp_path: Path):
    component = tmp_path / "src" / "auto_voice" / "inference" / "hq_svc_wrapper.py"
    artifact = tmp_path / "reports" / "quality.json"
    component.parent.mkdir(parents=True, exist_ok=True)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    component.write_text("# component\n", encoding="utf-8")
    artifact.write_text("{}", encoding="utf-8")

    payload = {
        "schema_version": 1,
        "features": {
            "hq_svc": {
                "display_name": "HQ-SVC",
                "status": "promotable",
                "component_paths": [str(component.relative_to(tmp_path))],
                "evidence": {
                    category: {
                        "artifacts": [str(artifact.relative_to(tmp_path))] if category == "quality" else [],
                        "notes": "",
                    }
                    for category in REQUIRED_EVIDENCE_CATEGORIES
                },
            }
        },
    }

    result = evaluate_evidence_gates(payload, root_dir=tmp_path)
    feature = result["features"]["hq_svc"]

    assert feature["gate_passed"] is False
    assert set(feature["missing_evidence_categories"]) == {
        "latency",
        "packaging",
        "deployability",
    }


def test_write_and_reload_registry_round_trip(tmp_path: Path):
    payload = {
        "schema_version": 1,
        "features": {
            "demo_upgrade": {
                "display_name": "Demo Upgrade",
                "status": "experimental",
                "component_paths": ["src/demo.py"],
                "evidence": {
                    category: {"artifacts": [], "notes": ""}
                    for category in REQUIRED_EVIDENCE_CATEGORIES
                },
            }
        },
    }

    path = tmp_path / "registry.json"
    write_experimental_registry(path, payload)
    reloaded = json.loads(path.read_text(encoding="utf-8"))

    assert validate_experimental_registry(reloaded)["features"]["demo_upgrade"]["status"] == "experimental"


def test_registry_validation_rejects_invalid_status():
    with pytest.raises(ValueError, match="invalid status"):
        validate_experimental_registry(
            {
                "schema_version": 1,
                "features": {
                    "hq_svc": {
                        "display_name": "HQ-SVC",
                        "status": "canonical",
                        "component_paths": [],
                        "evidence": {
                            category: {"artifacts": [], "notes": ""}
                            for category in REQUIRED_EVIDENCE_CATEGORIES
                        },
                    }
                },
            }
        )
