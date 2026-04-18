#!/usr/bin/env python3
"""Prepare the canonical Pillowtalk dataset layout and manifest."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import librosa
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from auto_voice.models.ecapa2_encoder import ECAPA2SpeakerEncoder


FIXTURE_FILES = {
    "william_singe": PROJECT_ROOT / "tests/quality_samples/william_singe_pillowtalk.wav",
    "conor_maynard": PROJECT_ROOT / "tests/quality_samples/conor_maynard_pillowtalk.wav",
}


def ensure_relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    relative_source = os.path.relpath(source.resolve(), start=destination.parent.resolve())
    destination.symlink_to(relative_source, target_is_directory=False)


def compute_embedding(audio_path: Path, device: str) -> Dict[str, object]:
    encoder = ECAPA2SpeakerEncoder(device=device)
    audio, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
    result = encoder.extract_embedding(audio, sample_rate)
    return {
        "embedding": result.embedding.astype(np.float32),
        "backend": result.backend,
    }


def build_manifest(dataset_root: Path) -> Dict[str, object]:
    return {
        "dataset_name": "pillowtalk",
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_rate": 24000,
        "defaults": {
            "sample_rate": 24000,
            "evaluation_mode": "offline_quality",
        },
        "samples": [
            {
                "sample_id": "william_to_conor_holdout",
                "split": "holdout",
                "source_audio": "holdout/william_to_conor/source.wav",
                "reference_audio": "holdout/william_to_conor/reference.wav",
                "target_speaker_embedding": "speakers/conor_maynard.npy",
                "metadata": {
                    "source_artist": "william_singe",
                    "target_artist": "conor_maynard",
                    "song": "Pillowtalk",
                    "provenance": "tests/quality_samples fixtures",
                },
            },
            {
                "sample_id": "conor_to_william_holdout",
                "split": "holdout",
                "source_audio": "holdout/conor_to_william/source.wav",
                "reference_audio": "holdout/conor_to_william/reference.wav",
                "target_speaker_embedding": "speakers/william_singe.npy",
                "metadata": {
                    "source_artist": "conor_maynard",
                    "target_artist": "william_singe",
                    "song": "Pillowtalk",
                    "provenance": "tests/quality_samples fixtures",
                },
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data/training/pillowtalk"),
        help="Canonical dataset output directory.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Embedding extraction device for ECAPA2SpeakerEncoder.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.output_dir).resolve()
    raw_dir = dataset_root / "raw"
    holdout_dir = dataset_root / "holdout"
    aligned_dir = dataset_root / "aligned"
    vocals_dir = dataset_root / "vocals"
    speakers_dir = dataset_root / "speakers"

    for directory in (raw_dir, holdout_dir, aligned_dir, vocals_dir, speakers_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Raw fixture links.
    ensure_relative_symlink(FIXTURE_FILES["william_singe"], raw_dir / "william_singe/pillowtalk.wav")
    ensure_relative_symlink(FIXTURE_FILES["conor_maynard"], raw_dir / "conor_maynard/pillowtalk.wav")

    # Holdout evaluation pairs.
    ensure_relative_symlink(
        FIXTURE_FILES["william_singe"],
        holdout_dir / "william_to_conor/source.wav",
    )
    ensure_relative_symlink(
        FIXTURE_FILES["conor_maynard"],
        holdout_dir / "william_to_conor/reference.wav",
    )
    ensure_relative_symlink(
        FIXTURE_FILES["conor_maynard"],
        holdout_dir / "conor_to_william/source.wav",
    )
    ensure_relative_symlink(
        FIXTURE_FILES["william_singe"],
        holdout_dir / "conor_to_william/reference.wav",
    )

    # Placeholder directories for later separation/alignment stages.
    for artist in FIXTURE_FILES:
        (aligned_dir / artist).mkdir(parents=True, exist_ok=True)
        (vocals_dir / artist).mkdir(parents=True, exist_ok=True)

    william_embedding = compute_embedding(FIXTURE_FILES["william_singe"], device=args.device)
    conor_embedding = compute_embedding(FIXTURE_FILES["conor_maynard"], device=args.device)
    np.save(speakers_dir / "william_singe.npy", william_embedding["embedding"])
    np.save(speakers_dir / "conor_maynard.npy", conor_embedding["embedding"])

    manifest = build_manifest(dataset_root)
    manifest["speaker_backends"] = {
        "william_singe": william_embedding["backend"],
        "conor_maynard": conor_embedding["backend"],
    }

    metadata_path = dataset_root / "metadata.json"
    metadata_path.write_text(json.dumps(manifest, indent=2))

    print(f"Wrote manifest: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
