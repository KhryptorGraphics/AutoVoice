#!/usr/bin/env python3
"""Write a consolidated Pillowtalk delivery manifest."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"
DATASET_MANIFEST = PROJECT_ROOT / "data" / "training" / "pillowtalk" / "metadata.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _existing(path: Path) -> str | None:
    return str(path) if path.exists() else None


def _collect_model_release(artist_key: str) -> dict:
    release_dir = MODELS_DIR / artist_key
    registry = _load_json(release_dir / "registry_entry.json")
    return {
        "release_dir": _existing(release_dir),
        "registry_entry": registry,
    }


def _collect_swap(name: str, mix_name: str, similarity_gate: float) -> dict:
    report_dir = OUTPUT_DIR / "reports" / name
    summary = _load_json(report_dir / "summary.json")
    summary_metrics = summary.get("summary", {}) if isinstance(summary, dict) else {}
    speaker_similarity = summary_metrics.get("speaker_similarity_mean")
    return {
        "report_dir": _existing(report_dir),
        "summary_json": _existing(report_dir / "summary.json"),
        "report_md": _existing(report_dir / "report.md"),
        "converted_vocals": _existing(OUTPUT_DIR / f"{name}_vocals.wav"),
        "final_mix": _existing(OUTPUT_DIR / mix_name),
        "speaker_similarity_mean": speaker_similarity,
        "passes_similarity_gate": bool(
            speaker_similarity is not None and speaker_similarity >= similarity_gate
        ),
        "summary_metrics": summary_metrics,
    }


def main() -> int:
    dataset = _load_json(DATASET_MANIFEST)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "manifest_path": str(DATASET_MANIFEST),
            "exists": DATASET_MANIFEST.exists(),
            "speaker_backends": dataset.get("speaker_backends", {}),
            "artists": dataset.get("artists", {}),
            "aligned_pairs": dataset.get("aligned_pairs", {}),
        },
        "models": {
            "william_singe": _collect_model_release("william_singe"),
            "conor_maynard": _collect_model_release("conor_maynard"),
        },
        "final_swaps": {
            "william_as_conor": _collect_swap(
                "william_to_conor",
                "william_as_conor_pillowtalk.wav",
                0.80,
            ),
            "conor_as_william": _collect_swap(
                "conor_to_william",
                "conor_as_william_pillowtalk.wav",
                0.80,
            ),
        },
    }

    output_path = OUTPUT_DIR / "pillowtalk_delivery_manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2))
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
