#!/usr/bin/env python3
"""Download pre-trained models for AutoVoice singing voice conversion.

This script downloads:
1. So-VITS-SVC 5.0 pre-trained weights from HuggingFace
2. HiFi-GAN vocoder weights
3. HuBERT-Soft content encoder
4. Speaker encoder weights

Usage:
    python scripts/download_pretrained_models.py [--models-dir <path>]
"""

import os
import sys
import argparse
import hashlib
import urllib.request
import time
from pathlib import Path
from typing import Dict, Optional

# Model URLs and metadata
MODELS = {
    'sovits_5.0': {
        'url': 'https://huggingface.co/xihan123/so-vits-svc-5.0-nine/resolve/main/chkpt/sovits5.0/sovits5.0_main_1500.pth',
        'filename': 'sovits5.0_main_1500.pth',
        'size_mb': 300,
        'description': 'So-VITS-SVC 5.0 pre-trained weights (176M parameters)',
        'required': True
    },
    'sovits_4.0_G': {
        'url': 'https://huggingface.co/innnky/sovits_pretrained/resolve/main/sovits4/G_0.pth',
        'filename': 'G_0.pth',
        'size_mb': 150,
        'description': 'So-VITS-SVC 4.0 Generator (fallback option)',
        'required': False
    },
    'sovits_4.0_D': {
        'url': 'https://huggingface.co/innnky/sovits_pretrained/resolve/main/sovits4/D_0.pth',
        'filename': 'D_0.pth',
        'size_mb': 50,
        'description': 'So-VITS-SVC 4.0 Discriminator (fallback option)',
        'required': False
    },
    'hifigan_vocoder': {
        'url': 'https://huggingface.co/speechbrain/tts-hifigan-ljspeech/resolve/main/generator.ckpt',
        'filename': 'hifigan_ljspeech.ckpt',
        'size_mb': 40,
        'description': 'HiFi-GAN vocoder for 80-mel, 22050Hz',
        'required': True
    },
    'hubert_soft': {
        'url': 'https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt',
        'filename': 'hubert-soft-0d54a1f4.pt',
        'size_mb': 95,
        'description': 'HuBERT-Soft content encoder',
        'required': True
    }
}


def download_file(url: str, dest_path: Path, expected_size_mb: Optional[int] = None, max_retries: int = 3) -> bool:
    """Download file with progress reporting and retry logic.
    
    Args:
        url: Download URL
        dest_path: Destination file path
        expected_size_mb: Expected file size in MB for validation
        max_retries: Maximum number of retry attempts
        
    Returns:
        True if download successful
    """
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"  Retry {attempt}/{max_retries-1} in {wait_time}s...")
                time.sleep(wait_time)
            
            print(f"  Downloading from: {url}")
            print(f"  Saving to: {dest_path}")
            
            # Download with progress
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, downloaded * 100 / total_size)
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)
            
            urllib.request.urlretrieve(url, dest_path, reporthook=report_progress)
            print()  # New line after progress
            
            # Validate size
            actual_size_mb = dest_path.stat().st_size / (1024 * 1024)
            print(f"  Downloaded: {actual_size_mb:.1f} MB")
            
            if expected_size_mb and abs(actual_size_mb - expected_size_mb) > expected_size_mb * 0.3:
                print(f"  ⚠️  Warning: Size mismatch (expected ~{expected_size_mb}MB)")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Download attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                continue
            else:
                print(f"  ❌ All {max_retries} attempts failed")
                return False
    
    return False


def main():
    parser = argparse.ArgumentParser(description='Download pre-trained models for AutoVoice')
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models/pretrained',
        help='Directory to save models (default: models/pretrained)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip downloading files that already exist'
    )
    parser.add_argument(
        '--required-only',
        action='store_true',
        help='Download only required models (skip fallbacks)'
    )
    
    args = parser.parse_args()
    
    # Create models directory
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("AutoVoice Pre-trained Model Downloader")
    print("="*70)
    print(f"\nModels directory: {models_dir.absolute()}\n")
    
    # Download each model
    success_count = 0
    total_count = 0
    
    for model_name, model_info in MODELS.items():
        # Skip if not required and --required-only flag set
        if args.required_only and not model_info['required']:
            print(f"⏭️  Skipping {model_name} (not required)\n")
            continue
        
        total_count += 1
        dest_path = models_dir / model_info['filename']
        
        print(f"📦 {model_name}")
        print(f"  {model_info['description']}")
        
        # Check if already exists
        if dest_path.exists():
            if args.skip_existing:
                print(f"  ✓ Already exists (skipping)\n")
                success_count += 1
                continue
            else:
                print(f"  ⚠️  File exists, re-downloading...")
        
        # Download
        if download_file(model_info['url'], dest_path, model_info['size_mb']):
            print(f"  ✅ Success\n")
            success_count += 1
        else:
            print(f"  ❌ Failed\n")
            if model_info['required']:
                print(f"\n⚠️  Required model {model_name} failed to download.")
                print("   System may not work without this model.\n")
    
    # Summary
    print("="*70)
    print(f"Download Summary: {success_count}/{total_count} models downloaded")
    print("="*70)
    
    if success_count == total_count:
        print("✅ All models downloaded successfully!")
        print(f"\nModels saved to: {models_dir.absolute()}")
        print("\nNext steps:")
        print("  1. Fix PyTorch environment: ./scripts/setup_pytorch_env.sh")
        print("  2. Run demo: python examples/demo_voice_conversion.py")
        print("  3. Test web interface: python main.py")
        return 0
    else:
        print(f"⚠️  {total_count - success_count} model(s) failed to download.")
        print("\nYou can:")
        print("  - Check your internet connection")
        print("  - Manually download from URLs listed above")
        print("  - Use --skip-existing flag to retry only failed downloads")
        return 1


if __name__ == '__main__':
    sys.exit(main())
