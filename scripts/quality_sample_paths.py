#!/usr/bin/env python3
"""Shared runtime-path helpers for legacy quality/realtime sample scripts."""

from __future__ import annotations

import os
from pathlib import Path

from auto_voice.storage.paths import resolve_data_dir, resolve_separated_audio_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
QUALITY_OUTPUTS_DIR = PROJECT_ROOT / "tests" / "quality_samples" / "outputs"

WILLIAM_SAMPLE_ARTIST = "william_singe"
WILLIAM_SAMPLE_FILE = "2iVFx7f5MMU_vocals.wav"
CONOR_REFERENCE_ARTIST = "conor_maynard"
CONOR_REFERENCE_FILE = "08NWh97_DME_vocals.wav"


def resolve_quality_sample_runtime_paths(data_dir: str | None = None) -> dict[str, Path]:
    """Resolve canonical sample input/output paths for legacy quality scripts."""
    resolved_data_dir = resolve_data_dir(
        data_dir or os.environ.get("DATA_DIR") or str(DEFAULT_DATA_DIR)
    )
    william_root = resolve_separated_audio_dir(
        data_dir=str(resolved_data_dir),
        artist_name=WILLIAM_SAMPLE_ARTIST,
    )
    conor_root = resolve_separated_audio_dir(
        data_dir=str(resolved_data_dir),
        artist_name=CONOR_REFERENCE_ARTIST,
    )
    return {
        "data_dir": resolved_data_dir,
        "william_test_audio": william_root / WILLIAM_SAMPLE_FILE,
        "conor_reference_audio": conor_root / CONOR_REFERENCE_FILE,
        "quality_outputs_dir": QUALITY_OUTPUTS_DIR,
        "realtime_output": QUALITY_OUTPUTS_DIR / "william_as_conor_realtime_30s.wav",
        "quality_output": QUALITY_OUTPUTS_DIR / "william_as_conor_quality_30s.wav",
    }
