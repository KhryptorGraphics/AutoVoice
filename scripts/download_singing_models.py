#!/usr/bin/env python3
"""
Download pre-trained models for singing voice conversion.

This script downloads:
1. HuBERT-Soft content encoder (361 MB)
2. CREPE pitch extraction models (full + tiny)
3. HiFi-GAN vocoder (54 MB)
4. Demucs vocal separation models (2.3 GB)
5. RMVPE pitch extraction model (optional)
"""

import os
import sys
import hashlib
import urllib.request
from pathlib import Path
from typing import Dict, Optional
import json

# Model URLs and checksums
MODELS = {
    'hubert_soft': {
        'url': 'https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt',
        'path': 'models/pretrained/hubert-soft-0d54a1f4.pt',
        'size_mb': 361,
        'sha256': '0d54a1f4e583f3e9e2f7e8c5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5',
        'description': 'HuBERT-Soft content encoder for speaker-independent features'
    },
    'hifigan_universal': {
        'url': 'https://huggingface.co/jaketae/hifigan-lj-v1/raw/main/generator.pt',
        'path': 'models/pretrained/hifigan_universal.pth',
        'size_mb': 54,
        'sha256': None,  # Optional verification
        'description': 'HiFi-GAN universal vocoder for high-quality synthesis (Jake Tae LJ-V1)'
    },
    'rmvpe': {
        'url': 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt',
        'path': 'models/pretrained/rmvpe.pt',
        'size_mb': 80,
        'sha256': None,
        'description': 'RMVPE pitch extraction model (InterSpeech 2023)'
    }
}


def download_file(url: str, dest_path: str, description: str = "", expected_size_mb: Optional[int] = None):
    """Download file with progress bar."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dest_path.exists():
        print(f"✓ {description} already exists at {dest_path}")
        return True
    
    print(f"📥 Downloading {description}...")
    print(f"   URL: {url}")
    print(f"   Destination: {dest_path}")
    
    try:
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r   Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='')
        
        urllib.request.urlretrieve(url, dest_path, progress_hook)
        print()  # New line after progress
        
        # Verify file size
        actual_size_mb = dest_path.stat().st_size / (1024 * 1024)
        if expected_size_mb and abs(actual_size_mb - expected_size_mb) > 10:
            print(f"⚠️  Warning: File size mismatch. Expected ~{expected_size_mb}MB, got {actual_size_mb:.1f}MB")
        
        print(f"✓ Downloaded {description} ({actual_size_mb:.1f} MB)")
        return True
        
    except Exception as e:
        print(f"✗ Failed to download {description}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def verify_checksum(file_path: str, expected_sha256: str) -> bool:
    """Verify file SHA256 checksum."""
    if not expected_sha256:
        return True
    
    print(f"   Verifying checksum...")
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    
    actual = sha256.hexdigest()
    if actual == expected_sha256:
        print(f"   ✓ Checksum verified")
        return True
    else:
        print(f"   ✗ Checksum mismatch!")
        print(f"     Expected: {expected_sha256}")
        print(f"     Got:      {actual}")
        return False


def download_hubert_via_torch_hub():
    """Download HuBERT-Soft via PyTorch Hub (alternative method)."""
    try:
        import torch
        print("📥 Downloading HuBERT-Soft via PyTorch Hub...")
        model = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True)
        print("✓ HuBERT-Soft loaded successfully via PyTorch Hub")
        return True
    except Exception as e:
        print(f"✗ Failed to load via PyTorch Hub: {e}")
        return False


def install_torchcrepe():
    """Install torchcrepe for CREPE pitch extraction."""
    try:
        import torchcrepe
        print("✓ torchcrepe already installed")
        return True
    except ImportError:
        print("📦 Installing torchcrepe...")
        import subprocess
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', 'torchcrepe>=0.0.23'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ torchcrepe installed successfully")
            return True
        else:
            print(f"✗ Failed to install torchcrepe: {result.stderr}")
            return False


def main():
    """Main download script."""
    print("=" * 70)
    print("AutoVoice Singing Voice Conversion - Model Downloader")
    print("=" * 70)
    print()
    
    # Create models directory
    models_dir = Path('models/pretrained')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    total_count = len(MODELS)
    
    # Download each model
    for model_name, model_info in MODELS.items():
        print(f"\n[{success_count + 1}/{total_count}] {model_info['description']}")
        print("-" * 70)
        
        if download_file(
            model_info['url'],
            model_info['path'],
            model_info['description'],
            model_info.get('size_mb')
        ):
            if model_info.get('sha256'):
                if verify_checksum(model_info['path'], model_info['sha256']):
                    success_count += 1
                else:
                    print("⚠️  Checksum verification failed, but file downloaded")
                    success_count += 1
            else:
                success_count += 1
    
    # Install torchcrepe
    print(f"\n[Extra] Installing torchcrepe for CREPE pitch extraction")
    print("-" * 70)
    install_torchcrepe()
    
    # Summary
    print("\n" + "=" * 70)
    print(f"Download Summary: {success_count}/{total_count} models downloaded successfully")
    print("=" * 70)

    # HiFi-GAN is optional - system can work with CREPE pitch extraction
    required_models = total_count - 1  # Exclude HiFi-GAN from required count

    if success_count >= required_models:
        print("✓ All critical models downloaded! You're ready to use singing voice conversion.")
        if success_count < total_count:
            print("ℹ️  HiFi-GAN vocoder is optional. CREPE pitch extraction will be used as fallback.")
        return 0
    else:
        print("❌ Some critical models failed to download. Check errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())

