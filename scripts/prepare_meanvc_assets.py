#!/usr/bin/env python3
"""Prepare MeanVC runtime assets that are not fully handled by upstream scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEANVC_DIR = PROJECT_ROOT / "models" / "meanvc"
MODEL_DIR = MEANVC_DIR / "src" / "ckpt"
SV_MODEL_PATH = (
    MEANVC_DIR
    / "src"
    / "runtime"
    / "speaker_verification"
    / "ckpt"
    / "wavlm_large_finetune.pth"
)
SV_MODEL_FILE_ID = "1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP"


def _download_google_drive_file(file_id: str, output_path: Path) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "gdown is required to download the MeanVC speaker verification checkpoint. "
            "Install project requirements or run: pip install gdown"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = gdown.download(id=file_id, output=str(output_path), quiet=False)
    if not result or not output_path.exists():
        raise RuntimeError(f"Failed to download Google Drive checkpoint to {output_path}")


def prepare_meanvc_assets(force: bool = False) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="ASLP-lab/MeanVC",
        allow_patterns=[
            "model_200ms.safetensors",
            "meanvc_200ms.pt",
            "fastu2++.pt",
            "vocos.pt",
        ],
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False,
        repo_type="model",
    )

    if force or not SV_MODEL_PATH.exists():
        _download_google_drive_file(SV_MODEL_FILE_ID, SV_MODEL_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download the Google Drive checkpoint")
    args = parser.parse_args()
    prepare_meanvc_assets(force=args.force)


if __name__ == "__main__":
    main()
