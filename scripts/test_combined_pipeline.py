#!/usr/bin/env python3
"""Test Task 3.3: Combined pipeline - Seed-VC → HQ-SVC super-resolution.

Tests chaining the quality pipeline with HQ-SVC enhancement for super-resolution.
Pipeline: Source audio → Seed-VC (44kHz) → Downsample (22kHz) → HQ-SVC upsample (44kHz)

This tests whether HQ-SVC can enhance the quality of Seed-VC output.
"""

import os
import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'models' / 'seed-vc'))

import torch
import numpy as np
import librosa
import soundfile as sf
import torchaudio

from auto_voice.inference.hq_svc_wrapper import HQSVCWrapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    print("\n" + "=" * 70)
    print("  TASK 3.3: COMBINED PIPELINE TEST")
    print("  Seed-VC (44kHz) → Downsample (22kHz) → HQ-SVC Super-resolution (44kHz)")
    print("=" * 70 + "\n")

    # Change to repo root
    os.chdir(Path(__file__).parent.parent)

    # Use the quality pipeline output from Task 2.8 as input
    seedvc_output = "tests/quality_samples/outputs/william_as_conor_quality_30s.wav"

    if not Path(seedvc_output).exists():
        print(f"ERROR: Seed-VC output not found: {seedvc_output}")
        print("Run test_quality_pipeline_hq.py first (Task 2.8)")
        return 1

    # Load Seed-VC output
    print(f"Loading Seed-VC output: {seedvc_output}")
    audio, sr = librosa.load(seedvc_output, sr=None, mono=True)
    print(f"  Duration: {len(audio)/sr:.1f}s")
    print(f"  Sample rate: {sr}Hz")
    print(f"  Shape: {audio.shape}")

    # Downsample to 22kHz to simulate lower quality
    print("\nDownsampling to 22kHz (simulating lower quality)...")
    audio_22k = librosa.resample(audio, orig_sr=sr, target_sr=22050)
    print(f"  22kHz shape: {audio_22k.shape}")

    # Initialize HQ-SVC wrapper
    print("\nInitializing HQ-SVC wrapper for super-resolution...")
    try:
        hqsvc = HQSVCWrapper(device=torch.device("cuda"))
    except Exception as e:
        print(f"ERROR: Failed to initialize HQ-SVC: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Super-resolve 22kHz → 44kHz
    print("\n" + "=" * 70)
    print("SUPER-RESOLVING (HQ-SVC Enhancement)...")
    print("=" * 70)
    start_time = time.time()

    try:
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_22k).float()

        # Progress callback
        def progress_callback(stage: str, progress: float):
            logger.info(f"  {stage}: {progress*100:.0f}%")

        # Super-resolve
        result = hqsvc.super_resolve(
            audio=audio_tensor,
            sample_rate=22050,
            on_progress=progress_callback
        )

        enhanced_audio = result['audio'].cpu().numpy()
        enhanced_sr = result['sample_rate']

        elapsed = time.time() - start_time
        rtf = elapsed / (len(audio_22k) / 22050)

        print(f"\n✓ Super-resolution complete!")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  RTF: {rtf:.3f}")
        print(f"  Output SR: {enhanced_sr}Hz")
        print(f"  Output shape: {enhanced_audio.shape}")

    except Exception as e:
        print(f"\n✗ Super-resolution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Save outputs
    output_dir = Path("tests/quality_samples/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save downsampled version (for comparison)
    downsampled_path = output_dir / "william_as_conor_22k_intermediate.wav"
    sf.write(str(downsampled_path), audio_22k, 22050)

    # Save enhanced version
    enhanced_path = output_dir / "william_as_conor_combined_30s.wav"
    sf.write(str(enhanced_path), enhanced_audio, enhanced_sr)

    print(f"\nOutputs saved:")
    print(f"  Original (Seed-VC):  {seedvc_output} (44kHz)")
    print(f"  Downsampled:         {downsampled_path} (22kHz)")
    print(f"  Enhanced (Combined): {enhanced_path} (44kHz)")

    # Cleanup
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("✓ TASK 3.3 COMPLETE")
    print("=" * 70)
    print("\nCombined Pipeline Results:")
    print("  Input:  Seed-VC quality conversion (44kHz)")
    print("  Step 1: Downsample to 22kHz")
    print("  Step 2: HQ-SVC super-resolution to 44kHz")
    print(f"  Output: {enhanced_path}")
    print("\nNext: Task 3.4 - Benchmark quality improvement vs latency cost\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
