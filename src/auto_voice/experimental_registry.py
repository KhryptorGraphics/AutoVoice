"""Registry and validation helpers for experimental quality upgrades."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping


REQUIRED_EVIDENCE_CATEGORIES = ("quality", "latency", "packaging", "deployability")
VALID_EXPERIMENTAL_STATUSES = ("experimental", "promotable")
DEFAULT_REGISTRY_PATH = Path("config/experimental_evidence.json")


def validate_experimental_registry(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and normalize the experimental evidence registry."""
    features = payload.get("features")
    if not isinstance(features, Mapping) or not features:
        raise ValueError("Experimental evidence registry must define features")

    normalized_features: Dict[str, Any] = {}
    for feature_id, feature in features.items():
        if not isinstance(feature, Mapping):
            raise ValueError(f"Experimental feature {feature_id} must be an object")
        status = str(feature.get("status", "experimental"))
        if status not in VALID_EXPERIMENTAL_STATUSES:
            raise ValueError(f"Experimental feature {feature_id} has invalid status {status}")

        evidence = feature.get("evidence", {})
        if not isinstance(evidence, Mapping):
            raise ValueError(f"Experimental feature {feature_id} evidence must be an object")

        normalized_evidence: Dict[str, Any] = {}
        for category in REQUIRED_EVIDENCE_CATEGORIES:
            category_entry = evidence.get(category, {})
            if not isinstance(category_entry, Mapping):
                raise ValueError(
                    f"Experimental feature {feature_id} evidence.{category} must be an object"
                )
            normalized_evidence[category] = {
                "artifacts": list(category_entry.get("artifacts", [])),
                "notes": str(category_entry.get("notes", "")),
            }

        normalized_features[str(feature_id)] = {
            "display_name": str(feature.get("display_name", feature_id)),
            "status": status,
            "summary": str(feature.get("summary", "")),
            "component_paths": [str(path) for path in feature.get("component_paths", [])],
            "evidence": normalized_evidence,
        }

    return {
        "schema_version": int(payload.get("schema_version", 1)),
        "features": normalized_features,
    }


def load_experimental_registry(path: Path | str = DEFAULT_REGISTRY_PATH) -> Dict[str, Any]:
    """Load and validate the experimental evidence registry."""
    registry_path = Path(path)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    return validate_experimental_registry(payload)


def write_experimental_registry(
    path: Path | str,
    payload: Mapping[str, Any],
) -> Path:
    """Write the experimental evidence registry to disk."""
    registry_path = Path(path)
    normalized = validate_experimental_registry(payload)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    return registry_path


def evaluate_evidence_gates(
    payload: Mapping[str, Any],
    *,
    root_dir: Path | str = ".",
) -> Dict[str, Any]:
    """Evaluate whether experimental features satisfy promotion requirements."""
    normalized = validate_experimental_registry(payload)
    root = Path(root_dir)
    results: Dict[str, Any] = {"schema_version": normalized["schema_version"], "features": {}}

    for feature_id, feature in normalized["features"].items():
        component_paths = [str(path) for path in feature["component_paths"]]
        missing_components = [
            path for path in component_paths if not (root / path).exists()
        ]

        evidence_status: Dict[str, Any] = {}
        missing_evidence = []
        for category, category_entry in feature["evidence"].items():
            artifacts = [str(path) for path in category_entry["artifacts"]]
            missing_artifacts = [
                path for path in artifacts if not (root / path).exists()
            ]
            satisfied = bool(artifacts) and not missing_artifacts
            if not satisfied:
                missing_evidence.append(category)
            evidence_status[category] = {
                "artifacts": artifacts,
                "missing_artifacts": missing_artifacts,
                "satisfied": satisfied,
                "notes": category_entry["notes"],
            }

        promotable = not missing_components and not missing_evidence
        results["features"][feature_id] = {
            "display_name": feature["display_name"],
            "status": feature["status"],
            "component_paths": component_paths,
            "missing_components": missing_components,
            "evidence": evidence_status,
            "missing_evidence_categories": missing_evidence,
            "promotion_ready": promotable,
            "gate_passed": not missing_components and (
                feature["status"] == "experimental" or promotable
            ),
        }

    return results
