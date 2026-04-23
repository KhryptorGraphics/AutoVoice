#!/usr/bin/env python3
"""Download pretrained models for AutoVoice.

Models:
- hubert-soft-35d9f29f.pt (361MB) - HuBERT-Soft feature extractor
- generator_universal.pth.tar (55MB) - HiFiGAN universal vocoder
- sovits5.0_main_1500.pth (184MB) - Main So-VITS model (requires training)
"""
import os
import sys
import hashlib
from pathlib import Path
from urllib.request import urlretrieve, Request, urlopen
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PROJECT_ROOT / 'models' / 'pretrained'

MODELS = {
    'hubert-soft-35d9f29f.pt': {
        'url': 'https://github.com/bshall/hubert/releases/download/v0.2/hubert-soft-35d9f29f.pt',
        'size_mb': 361,
        'description': 'HuBERT-Soft content encoder (speaker-independent features)',
    },
    'generator_universal.pth.tar': {
        'url': 'https://drive.google.com/uc?id=1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW',
        'size_mb': 55,
        'description': 'HiFiGAN universal vocoder (mel → audio)',
    },
}


def resolve_models_dir() -> Path:
    """Resolve the bootstrap download target for pretrained checkpoints.

    Runtime consumers may use their own path helpers, but this bootstrap script
    writes into the canonical repo-hosted pretrained directory unless an
    explicit AUTOVOICE_PRETRAINED_DIR override is provided.
    """
    override = os.environ.get('AUTOVOICE_PRETRAINED_DIR')
    return Path(override).expanduser() if override else DEFAULT_MODELS_DIR


def download_gdrive(file_id: str, dest: Path):
    """Download from Google Drive."""
    URL = f"https://drive.google.com/uc?export=download&id={file_id}"

    session = None
    response = urlopen(URL)

    # Check for virus scan warning
    for key, value in response.headers.items():
        if key.startswith('Set-Cookie') and 'download_warning' in value:
            # Get confirmation token
            token = value.split('download_warning_')[1].split(';')[0]
            URL = f"{URL}&confirm={token}"
            response = urlopen(URL)
            break

    # Download file
    with open(dest, 'wb') as f:
        shutil.copyfileobj(response, f)


def download_file(url: str, dest: Path, expected_size_mb: int = 0):
    """Download a file with progress."""
    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        if expected_size_mb > 0 and abs(size_mb - expected_size_mb) < 10:
            print(f"  Already exists: {dest.name} ({size_mb:.0f}MB)")
            return True

    print(f"  Downloading: {dest.name} (~{expected_size_mb}MB)...")
    try:
        # Handle Google Drive URLs
        if 'drive.google.com' in url:
            # Extract file ID from URL
            if 'id=' in url:
                file_id = url.split('id=')[1].split('&')[0]
                download_gdrive(file_id, dest)
            else:
                raise ValueError("Invalid Google Drive URL format")
        else:
            # Regular download with progress
            def progress(count, block_size, total_size):
                if total_size > 0:
                    pct = count * block_size * 100 / total_size
                    sys.stdout.write(f'\r    {pct:.0f}%')
                    sys.stdout.flush()

            urlretrieve(url, str(dest), reporthook=progress)
        print()
        return True
    except Exception as e:
        print(f"\n  Failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def verify_model(path: Path) -> bool:
    """Verify a downloaded model is valid (non-zero, loadable)."""
    if not path.exists():
        return False
    if path.stat().st_size < 1000:
        return False
    return True


def main():
    print("=== AutoVoice Pretrained Model Downloader ===\n")

    models_dir = resolve_models_dir()

    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"Models directory: {models_dir}\n")

    success = 0
    failed = 0

    for filename, info in MODELS.items():
        dest = models_dir / filename
        print(f"[{filename}] {info['description']}")
        if download_file(info['url'], dest, info['size_mb']):
            if verify_model(dest):
                success += 1
                print(f"  Verified: {dest.stat().st_size / (1024*1024):.0f}MB")
            else:
                failed += 1
                print(f"  Verification failed!")
        else:
            failed += 1

    print(f"\n=== Results: {success} downloaded, {failed} failed ===")

    if failed > 0:
        print("\nFor models that failed to download, try:")
        print("  1. Check internet connectivity")
        print("  2. Download manually and place in:", models_dir)
        print("  3. Required files:")
        for name in MODELS:
            if not (models_dir / name).exists():
                print(f"     - {name}")

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
