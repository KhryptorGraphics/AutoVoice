#!/usr/bin/env python3
"""Package Pillowtalk model artifacts into release directories.

Creates:
- models/william_singe/
- models/conor_maynard/

Each release directory contains canonical metadata, copied profile assets,
selected adapter artifacts, and a registry entry describing the package.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from auto_voice.runtime_contract import build_packaged_artifact_manifest, write_packaged_artifact_manifest
from pillowtalk_release_paths import resolve_pillowtalk_release_paths


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"


@dataclass(frozen=True)
class ReleaseSpec:
    artist_key: str
    display_name: str
    canonical_profile_id: str
    artifact_profile_ids: tuple[str, ...]


RELEASES: tuple[ReleaseSpec, ...] = (
    ReleaseSpec(
        artist_key="william_singe",
        display_name="William Singe",
        canonical_profile_id="7da05140-1303-40c6-95d9-5b6e2c3624df",
        artifact_profile_ids=("7da05140-1303-40c6-95d9-5b6e2c3624df",),
    ),
    ReleaseSpec(
        artist_key="conor_maynard",
        display_name="Conor Maynard",
        canonical_profile_id="9679a6ec-e6e2-43c4-b64e-1f004fed34f9",
        artifact_profile_ids=(
            "9679a6ec-e6e2-43c4-b64e-1f004fed34f9",
            "c572d02c-c687-4bed-8676-6ad253cf1c91",
        ),
    ),
)

EVALUATION_ARTIFACTS = {
    "william_singe": {
        "report_dir": OUTPUT_DIR / "reports" / "conor_to_william",
        "converted_vocals": OUTPUT_DIR / "conor_to_william_vocals.wav",
        "final_mix": OUTPUT_DIR / "conor_as_william_pillowtalk.wav",
    },
    "conor_maynard": {
        "report_dir": OUTPUT_DIR / "reports" / "william_to_conor",
        "converted_vocals": OUTPUT_DIR / "william_to_conor_vocals.wav",
        "final_mix": OUTPUT_DIR / "william_as_conor_pillowtalk.wav",
    },
}

TRAINING_ARTIFACTS = {
    "william_singe": {
        "hq_checkpoint": Path("checkpoints") / "hq" / "7da05140-1303-40c6-95d9-5b6e2c3624df_hq_lora.pt",
        "nvfp4_checkpoint": Path("checkpoints") / "nvfp4" / "7da05140-1303-40c6-95d9-5b6e2c3624df_lora.pt",
        "log_path": PROJECT_ROOT / "logs" / "training_william_optimal_20260131_020725.log",
    },
    "conor_maynard": {
        "hq_checkpoint": Path("checkpoints") / "hq" / "c572d02c-c687-4bed-8676-6ad253cf1c91_hq_lora.pt",
        "nvfp4_checkpoint": Path("checkpoints") / "nvfp4" / "c572d02c-c687-4bed-8676-6ad253cf1c91_lora.pt",
        "log_path": PROJECT_ROOT / "logs" / "training_tuned_30k_20260131_015654.log",
    },
}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _copy_if_exists(src: Optional[Path], dst: Path) -> Optional[str]:
    if src is None or not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def _find_first_existing(paths: list[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _extract_training_metadata(artifact_path: Optional[Path]) -> Dict[str, Any]:
    if artifact_path is None or not artifact_path.exists():
        return {}

    payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return {}

    metadata: Dict[str, Any] = {}
    for key in ("profile_id", "artist", "epoch", "loss", "status", "precision", "config"):
        if key in payload:
            metadata[key] = payload[key]
    return metadata


def _extract_checkpoint_metadata(checkpoint_path: Optional[Path]) -> Dict[str, Any]:
    if checkpoint_path is None or not checkpoint_path.exists():
        return {}

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return {}

    metadata: Dict[str, Any] = {}
    for key in ("epoch", "current_epoch", "global_step", "loss", "best_loss", "config"):
        if key in payload:
            metadata[key] = payload[key]
    metadata["path"] = str(checkpoint_path)
    return metadata


def _parse_batches_per_epoch(log_path: Optional[Path]) -> Optional[int]:
    if log_path is None or not log_path.exists():
        return None

    marker = "Batches per epoch:"
    for line in log_path.read_text().splitlines():
        if marker in line:
            try:
                return int(line.split(marker, 1)[1].strip())
            except ValueError:
                return None
    return None


def _load_report_summary(report_summary: Optional[Path]) -> Dict[str, Any]:
    if report_summary is None or not report_summary.exists():
        return {}
    data = _load_json(report_summary)
    if not isinstance(data, dict):
        return {}
    return data


def _existing_path_str(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.exists():
        return None
    return str(path)


def _package_release(
    spec: ReleaseSpec,
    mirror_alias_artifacts: bool = True,
    data_dir: str | None = None,
) -> Dict[str, Any]:
    runtime_paths = resolve_pillowtalk_release_paths(data_dir)
    models_dir = runtime_paths["models_dir"]
    trained_models_dir = runtime_paths["trained_models_dir"]
    profiles_dir = runtime_paths["profiles_dir"]
    dataset_manifest_path = runtime_paths["pillowtalk_dataset_manifest"]
    canonical_profile_json = profiles_dir / f"{spec.canonical_profile_id}.json"
    canonical_profile_npy = profiles_dir / f"{spec.canonical_profile_id}.npy"
    canonical_profile = _load_json(canonical_profile_json)

    base_adapter = _find_first_existing(
        [trained_models_dir / f"{profile_id}_adapter.pt" for profile_id in spec.artifact_profile_ids]
    )
    hq_adapter = _find_first_existing(
        [trained_models_dir / "hq" / f"{profile_id}_hq_lora.pt" for profile_id in spec.artifact_profile_ids]
    )
    nvfp4_adapter = _find_first_existing(
        [trained_models_dir / "nvfp4" / f"{profile_id}_nvfp4_lora.pt" for profile_id in spec.artifact_profile_ids]
    )

    release_dir = models_dir / spec.artist_key
    artifacts_dir = release_dir / "artifacts"
    release_dir.mkdir(parents=True, exist_ok=True)

    copied_base = _copy_if_exists(base_adapter, artifacts_dir / "adapter.pt")
    copied_hq = _copy_if_exists(hq_adapter, artifacts_dir / "hq_lora.pt")
    copied_nvfp4 = _copy_if_exists(nvfp4_adapter, artifacts_dir / "nvfp4_lora.pt")
    copied_profile_json = _copy_if_exists(canonical_profile_json, release_dir / "profile.json")
    copied_profile_embedding = _copy_if_exists(
        canonical_profile_npy, release_dir / "speaker_embedding.npy"
    )

    if mirror_alias_artifacts and spec.canonical_profile_id not in spec.artifact_profile_ids[:1]:
        raise ValueError("canonical profile id must be first in artifact_profile_ids")

    if mirror_alias_artifacts:
        if hq_adapter is not None and hq_adapter.stem != f"{spec.canonical_profile_id}_hq_lora":
            canonical_hq = trained_models_dir / "hq" / f"{spec.canonical_profile_id}_hq_lora.pt"
            canonical_hq.parent.mkdir(parents=True, exist_ok=True)
            if not canonical_hq.exists():
                shutil.copy2(hq_adapter, canonical_hq)

        if nvfp4_adapter is not None and nvfp4_adapter.stem != f"{spec.canonical_profile_id}_nvfp4_lora":
            canonical_nvfp4 = trained_models_dir / "nvfp4" / f"{spec.canonical_profile_id}_nvfp4_lora.pt"
            canonical_nvfp4.parent.mkdir(parents=True, exist_ok=True)
            if not canonical_nvfp4.exists():
                shutil.copy2(nvfp4_adapter, canonical_nvfp4)

    dataset_manifest = _load_json(dataset_manifest_path)
    speaker_backend = dataset_manifest.get("speaker_backends", {}).get(spec.artist_key)
    evaluation_paths = EVALUATION_ARTIFACTS.get(spec.artist_key, {})
    report_dir = evaluation_paths.get("report_dir")
    report_summary = report_dir / "summary.json" if report_dir else None
    report_markdown = report_dir / "report.md" if report_dir else None
    report_data = _load_report_summary(report_summary)
    report_metrics = report_data.get("summary", {}) if isinstance(report_data, dict) else {}

    training_paths = TRAINING_ARTIFACTS.get(spec.artist_key, {})
    hq_checkpoint = runtime_paths["data_dir"] / training_paths["hq_checkpoint"]
    nvfp4_checkpoint = runtime_paths["data_dir"] / training_paths["nvfp4_checkpoint"]
    log_path = training_paths.get("log_path")
    hq_checkpoint_metadata = _extract_checkpoint_metadata(hq_checkpoint)
    nvfp4_checkpoint_metadata = _extract_checkpoint_metadata(nvfp4_checkpoint)
    batches_per_epoch = _parse_batches_per_epoch(log_path)
    max_training_steps = max(
        int(hq_checkpoint_metadata.get("global_step", 0) or 0),
        int(nvfp4_checkpoint_metadata.get("global_step", 0) or 0),
    )

    tensorrt_engine = release_dir / "artifacts" / "tensorrt" / "hq_voice_lora.engine"
    tensorrt_metadata = release_dir / "artifacts" / "tensorrt" / "engine_metadata.json"

    training_metadata = {
        "base_adapter": _extract_training_metadata(base_adapter),
        "hq_adapter": _extract_training_metadata(hq_adapter),
        "nvfp4_adapter": _extract_training_metadata(nvfp4_adapter),
        "hq_checkpoint": hq_checkpoint_metadata,
        "nvfp4_checkpoint": nvfp4_checkpoint_metadata,
        "training_log": _existing_path_str(log_path),
        "batches_per_epoch": batches_per_epoch,
        "max_training_steps": max_training_steps,
    }

    registry_entry = {
        "artist_key": spec.artist_key,
        "display_name": spec.display_name,
        "canonical_profile_id": spec.canonical_profile_id,
        "artifact_profile_ids": list(spec.artifact_profile_ids),
        "packaged_at": datetime.now(timezone.utc).isoformat(),
        "speaker_embedding_backend": speaker_backend,
        "dataset_manifest": str(dataset_manifest_path),
        "profile": {
            "json": copied_profile_json,
            "embedding": copied_profile_embedding,
        },
        "artifacts": {
            "adapter": copied_base,
            "hq_lora": copied_hq,
            "nvfp4_lora": copied_nvfp4,
            "tensorrt_engine": _existing_path_str(tensorrt_engine),
            "tensorrt_metadata": _existing_path_str(tensorrt_metadata),
        },
        "evaluation": {
            "report_dir": _existing_path_str(report_dir),
            "summary_json": _existing_path_str(report_summary),
            "report_md": _existing_path_str(report_markdown),
            "converted_vocals": _existing_path_str(evaluation_paths.get("converted_vocals")),
            "final_mix": _existing_path_str(evaluation_paths.get("final_mix")),
            "summary_metrics": report_metrics,
        },
        "training_metadata": training_metadata,
        "profile_metadata": canonical_profile,
        "acceptance": {
            "speaker_similarity_ge_0_85": (
                report_metrics.get("speaker_similarity_mean", 0.0) >= 0.85
            ),
            "speaker_similarity_ge_0_80": (
                report_metrics.get("speaker_similarity_mean", 0.0) >= 0.80
            ),
            "minimum_50000_iterations_met": max_training_steps >= 50000,
            "tensorrt_export_present": tensorrt_engine.exists(),
        },
    }

    packaged_manifest = build_packaged_artifact_manifest(
        profile_id=spec.canonical_profile_id,
        display_name=spec.display_name,
        model_family="seed_vc",
        canonical_pipeline="quality_seedvc",
        sample_rate=44_100,
        speaker_embedding_dim=len(canonical_profile.get("embedding", []) or []),
        mel_bins=80,
        artifacts={
            "profile_json": copied_profile_json,
            "speaker_embedding": copied_profile_embedding,
            "adapter": copied_base,
            "hq_lora": copied_hq,
            "nvfp4_lora": copied_nvfp4,
            "tensorrt_engine": _existing_path_str(tensorrt_engine),
            "tensorrt_metadata": _existing_path_str(tensorrt_metadata),
        },
        compatibility={
            "supported_pipelines": ["quality_seedvc"],
            "supported_runtime_backends": ["pytorch", "tensorrt"],
            "supports_tensorrt": tensorrt_engine.exists(),
        },
        metadata={
            "artist_key": spec.artist_key,
            "artifact_profile_ids": list(spec.artifact_profile_ids),
            "speaker_embedding_backend": speaker_backend,
            "dataset_manifest": str(dataset_manifest_path),
            "evaluation": registry_entry["evaluation"],
            "training_metadata": training_metadata,
            "profile_metadata": canonical_profile,
            "acceptance": registry_entry["acceptance"],
        },
    )

    (release_dir / "registry_entry.json").write_text(json.dumps(registry_entry, indent=2))
    (release_dir / "metadata.json").write_text(json.dumps(registry_entry, indent=2))
    write_packaged_artifact_manifest(release_dir / "artifact_manifest.json", packaged_manifest)
    (release_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {spec.display_name} Release Package",
                "",
                f"- Canonical profile ID: `{spec.canonical_profile_id}`",
                f"- Speaker embedding backend: `{speaker_backend}`",
                f"- Dataset manifest: `{dataset_manifest_path}`",
                "",
                "## Artifacts",
                "",
                f"- adapter.pt: `{copied_base or 'missing'}`",
                f"- hq_lora.pt: `{copied_hq or 'missing'}`",
                f"- nvfp4_lora.pt: `{copied_nvfp4 or 'missing'}`",
                f"- hq_voice_lora.engine: `{_existing_path_str(tensorrt_engine) or 'missing'}`",
                f"- evaluation summary: `{_existing_path_str(report_summary) or 'missing'}`",
                f"- final mix: `{_existing_path_str(evaluation_paths.get('final_mix')) or 'missing'}`",
                "",
                "## Notes",
                "",
                "- `registry_entry.json` is the machine-readable release manifest.",
                "- `artifact_manifest.json` is the canonical packaged runtime/export/deployment manifest.",
                f"- Max recorded training steps: `{max_training_steps}`",
                f"- 50k iteration target met: `{max_training_steps >= 50000}`",
            ]
        )
        + "\n"
    )

    return registry_entry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artist",
        choices=["william_singe", "conor_maynard", "all"],
        default="all",
        help="Which release package to generate.",
    )
    parser.add_argument(
        "--no-mirror-alias-artifacts",
        action="store_true",
        help="Do not copy legacy optimized adapters onto the canonical profile IDs.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override root data directory (defaults to DATA_DIR or data).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    targets = RELEASES if args.artist == "all" else [r for r in RELEASES if r.artist_key == args.artist]

    packaged = []
    for spec in targets:
        packaged.append(
            _package_release(
                spec,
                mirror_alias_artifacts=not args.no_mirror_alias_artifacts,
                data_dir=args.data_dir,
            )
        )

    print(json.dumps({"packaged": packaged}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
