"""Canonical local storage paths for the single-user MVP."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def resolve_data_dir(explicit_data_dir: Optional[str] = None) -> Path:
    """Resolve the root data directory from config, env, or default."""
    raw = explicit_data_dir or os.environ.get("DATA_DIR") or "data"
    return Path(raw)


def resolve_profiles_dir(
    explicit_dir: Optional[str] = None,
    *,
    data_dir: Optional[str] = None,
) -> Path:
    if explicit_dir:
        return Path(explicit_dir)
    return resolve_data_dir(data_dir) / "voice_profiles"


def resolve_samples_dir(
    explicit_dir: Optional[str] = None,
    *,
    data_dir: Optional[str] = None,
) -> Path:
    if explicit_dir:
        return Path(explicit_dir)
    return resolve_data_dir(data_dir) / "samples"


def resolve_trained_models_dir(
    explicit_dir: Optional[str] = None,
    *,
    data_dir: Optional[str] = None,
) -> Path:
    if explicit_dir:
        return Path(explicit_dir)
    return resolve_data_dir(data_dir) / "trained_models"


def resolve_checkpoints_dir(
    explicit_dir: Optional[str] = None,
    *,
    data_dir: Optional[str] = None,
) -> Path:
    if explicit_dir:
        return Path(explicit_dir)
    return resolve_data_dir(data_dir) / "checkpoints"
