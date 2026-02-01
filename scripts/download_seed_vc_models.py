#!/usr/bin/env python3
"""Download Seed-VC pretrained models for QUALITY_PIPELINE.

Downloads:
1. DiT_seed_v2_uvit_whisper_base_f0_44k (SVC model, 200M params)
2. RMVPE pitch extractor
3. BigVGAN vocoder (auto-downloaded)
4. Whisper-small (auto-downloaded)
5. CAMPPlus (already present)

Usage:
    PYTHONNOUSERSITE=1 python scripts/download_seed_vc_models.py
"""

import os
import sys
from pathlib import Path

# Add seed-vc to path
SEED_VC_DIR = Path(__file__).parent.parent / "models" / "seed-vc"
sys.path.insert(0, str(SEED_VC_DIR))

def download_models():
    """Download all required Seed-VC models."""
    from huggingface_hub import hf_hub_download

    cache_dir = SEED_VC_DIR / "checkpoints"
    cache_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Downloading Seed-VC Models for QUALITY_PIPELINE")
    print("=" * 60)

    # 1. Download Seed-VC SVC checkpoint
    print("\n[1/4] Downloading Seed-VC SVC checkpoint...")
    try:
        svc_path = hf_hub_download(
            repo_id="Plachta/Seed-VC",
            filename="DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
            cache_dir=str(cache_dir)
        )
        print(f"  ✓ SVC model: {svc_path}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

    # 2. Download config
    print("\n[2/4] Downloading Seed-VC config...")
    try:
        config_path = hf_hub_download(
            repo_id="Plachta/Seed-VC",
            filename="config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
            cache_dir=str(cache_dir)
        )
        print(f"  ✓ Config: {config_path}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

    # 3. Download RMVPE
    print("\n[3/4] Downloading RMVPE pitch extractor...")
    try:
        rmvpe_path = hf_hub_download(
            repo_id="lj1995/VoiceConversionWebUI",
            filename="rmvpe.pt",
            cache_dir=str(cache_dir)
        )
        print(f"  ✓ RMVPE: {rmvpe_path}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

    # 4. Pre-download BigVGAN (optional, auto-downloads on first use)
    print("\n[4/4] Pre-downloading BigVGAN vocoder...")
    try:
        from huggingface_hub import snapshot_download
        bigvgan_path = snapshot_download(
            repo_id="nvidia/bigvgan_v2_44khz_128band_512x",
            cache_dir=str(cache_dir),
            ignore_patterns=["*.md", "*.txt"]
        )
        print(f"  ✓ BigVGAN: {bigvgan_path}")
    except Exception as e:
        print(f"  ✗ BigVGAN pre-download failed (will download on first use): {e}")

    # 5. Check CAMPPlus
    print("\n[5/5] Checking CAMPPlus style encoder...")
    campplus_path = SEED_VC_DIR / "campplus_cn_common.bin"
    if campplus_path.exists():
        print(f"  ✓ CAMPPlus already present: {campplus_path}")
    else:
        print("  ✗ CAMPPlus not found, downloading...")
        try:
            campplus_path = hf_hub_download(
                repo_id="funasr/campplus",
                filename="campplus_cn_common.bin",
                cache_dir=str(cache_dir)
            )
            print(f"  ✓ CAMPPlus: {campplus_path}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print("\n" + "=" * 60)
    print("Model download complete!")
    print("=" * 60)

    # List downloaded files
    print("\nDownloaded files in checkpoints/:")
    for f in sorted(cache_dir.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.relative_to(cache_dir)}: {size_mb:.1f} MB")

    return True


if __name__ == "__main__":
    success = download_models()
    sys.exit(0 if success else 1)
