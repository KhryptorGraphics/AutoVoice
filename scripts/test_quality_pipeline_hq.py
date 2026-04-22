#!/usr/bin/env python3
"""Test Task 2.8: William->Conor conversion using quality pipeline with HQ LoRA.

Tests the Seed-VC quality voice conversion pipeline with trained voice profiles.
Compares quality vs realtime pipeline.
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

from quality_sample_paths import resolve_quality_sample_runtime_paths
from quality_pipeline import QualityVoiceConverter, QualityConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    print("\n" + "=" * 70)
    print("  TASK 2.8: QUALITY PIPELINE TEST WITH HQ LORA")
    print("  William Singe → Conor Maynard Conversion (Seed-VC)")
    print("=" * 70 + "\n")

    # Change to repo root
    os.chdir(Path(__file__).parent.parent)
    paths = resolve_quality_sample_runtime_paths()

    # Profile IDs from spec
    WILLIAM_ID = "7da05140-1303-40c6-95d9-5b6e2c3624df"
    CONOR_ID = "c572d02c-c687-4bed-8676-6ad253cf1c91"

    # Test audio: Same as realtime test for comparison
    test_audio = paths["william_test_audio"]

    if not test_audio.exists():
        print(f"ERROR: Test audio not found: {test_audio}")
        return 1

    # Load source audio (first 30s for direct comparison)
    print(f"Loading source audio: {test_audio}")
    source_audio, source_sr = librosa.load(str(test_audio), sr=None, mono=True, duration=30.0)
    print(f"  Duration: {len(source_audio)/source_sr:.1f}s")
    print(f"  Sample rate: {source_sr}Hz")
    print(f"  Shape: {source_audio.shape}")

    # Load reference audio (Conor vocals for style)
    print(f"\nLoading reference speaker: Conor")
    reference_audio_path = paths["conor_reference_audio"]
    if not reference_audio_path.exists():
        print(f"ERROR: Reference audio not found: {reference_audio_path}")
        return 1

    reference_audio, reference_sr = librosa.load(
        str(reference_audio_path),
        sr=None,
        mono=True,
        duration=25.0,
    )
    print(f"  Reference: {reference_audio_path}")
    print(f"  Duration: {len(reference_audio)/reference_sr:.1f}s")
    print(f"  Sample rate: {reference_sr}Hz")

    # Initialize quality converter
    print("\nInitializing quality converter (Seed-VC)...")
    config = QualityConfig(
        sample_rate=44100,
        diffusion_steps=30,
        f0_condition=True,
        auto_f0_adjust=False,
        fp16=True,
        device="cuda"
    )
    converter = QualityVoiceConverter(config)

    # Convert
    print("\n" + "=" * 70)
    print("CONVERTING (Quality Pipeline - Seed-VC)...")
    print("=" * 70)
    start_time = time.time()

    try:
        converted, out_sr = converter.convert(
            source_audio=source_audio,
            source_sr=source_sr,
            reference_audio=reference_audio,
            reference_sr=reference_sr,
            pitch_shift=0  # No pitch shift for now
        )

        elapsed = time.time() - start_time
        rtf = elapsed / (len(source_audio) / source_sr)

        print(f"\n✓ Conversion complete!")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  RTF: {rtf:.3f}")
        print(f"  Output SR: {out_sr}Hz")
        print(f"  Output shape: {converted.shape}")

    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Save output
    output_dir = paths["quality_outputs_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = paths["quality_output"]

    print(f"\nSaving output: {output_path}")
    sf.write(str(output_path), converted, out_sr)
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Cleanup
    converter.unload()
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("✓ TASK 2.8 COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {output_path}")
    print("Comparison:")
    print("  - Realtime: tests/quality_samples/outputs/william_as_conor_realtime_30s.wav (22kHz)")
    print(f"  - Quality:  {output_path} (44kHz)")
    print("\nNext: Compare audio quality and proceed to Task 3.3\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
