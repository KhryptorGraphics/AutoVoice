#!/usr/bin/env python3
"""Shared runtime-path helpers for legacy Pillowtalk release utilities."""

from __future__ import annotations

import os
from pathlib import Path

from auto_voice.storage.paths import (
    resolve_checkpoints_dir,
    resolve_data_dir,
    resolve_profiles_dir,
    resolve_samples_dir,
    resolve_trained_models_dir,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"


def resolve_pillowtalk_release_paths(
    data_dir: str | None = None,
    *,
    models_dir: str | None = None,
    output_dir: str | None = None,
) -> dict[str, Path]:
    """Resolve canonical paths for release/export/dataset utility scripts."""
    resolved_data_dir = resolve_data_dir(
        data_dir or os.environ.get("DATA_DIR") or str(DEFAULT_DATA_DIR)
    )
    resolved_models_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
    resolved_output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    return {
        "data_dir": resolved_data_dir,
        "models_dir": resolved_models_dir,
        "output_dir": resolved_output_dir,
        "profiles_dir": resolve_profiles_dir(data_dir=str(resolved_data_dir)),
        "samples_dir": resolve_samples_dir(data_dir=str(resolved_data_dir)),
        "trained_models_dir": resolve_trained_models_dir(data_dir=str(resolved_data_dir)),
        "checkpoints_dir": resolve_checkpoints_dir(data_dir=str(resolved_data_dir)),
        "pillowtalk_training_dir": resolved_data_dir / "training" / "pillowtalk",
        "pillowtalk_dataset_manifest": resolved_data_dir / "training" / "pillowtalk" / "metadata.json",
    }


def resolve_profile_separated_dir(profile_id: str, data_dir: str | None = None) -> Path:
    """Resolve the legacy profile-id-based separated-audio directory."""
    return resolve_pillowtalk_release_paths(data_dir)["data_dir"] / "separated" / profile_id
