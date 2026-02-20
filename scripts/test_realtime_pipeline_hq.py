#!/usr/bin/env python3
"""Test Task 1.7: William->Conor conversion using realtime pipeline with HQ LoRA.

Tests the realtime voice conversion pipeline with trained voice profiles.
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

from realtime_pipeline import RealtimeVoiceConverter, RealtimeConfig, load_speaker_embedding

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    print("\n" + "=" * 70)
    print("  TASK 1.7: REALTIME PIPELINE TEST WITH HQ LORA")
    print("  William Singe → Conor Maynard Conversion")
    print("=" * 70 + "\n")

    # Change to repo root
    os.chdir(Path(__file__).parent.parent)

    # Profile IDs from spec
    WILLIAM_ID = "7da05140-1303-40c6-95d9-5b6e2c3624df"
    CONOR_ID = "c572d02c-c687-4bed-8676-6ad253cf1c91"

    # Test audio: Use a short William vocals file (first 30s for testing)
    test_audio = "data/separated_youtube/william_singe/2iVFx7f5MMU_vocals.wav"

    if not Path(test_audio).exists():
        print(f"ERROR: Test audio not found: {test_audio}")
        return 1

    # Load source audio (first 30s only)
    print(f"Loading source audio: {test_audio}")
    audio, sr = librosa.load(test_audio, sr=None, mono=True, duration=30.0)
    print(f"  Duration: {len(audio)/sr:.1f}s")
    print(f"  Sample rate: {sr}Hz")
    print(f"  Shape: {audio.shape}")

    # Load target speaker embedding (Conor)
    print(f"\nLoading target speaker: Conor (ID: {CONOR_ID})")
    try:
        target_embedding = load_speaker_embedding(CONOR_ID)
        print(f"  Embedding shape: {target_embedding.shape}")
        print(f"  Embedding dtype: {target_embedding.dtype}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    # Initialize realtime converter
    print("\nInitializing realtime converter...")
    config = RealtimeConfig(
        sample_rate=22050,
        chunk_size_ms=100,
        overlap_ms=20,
        fp16=True,
        device="cuda"
    )
    converter = RealtimeVoiceConverter(config)

    # Convert
    print("\n" + "=" * 70)
    print("CONVERTING (Realtime Pipeline)...")
    print("=" * 70)
    start_time = time.time()

    try:
        converted, out_sr = converter.convert_full(
            audio=audio,
            sr=sr,
            speaker_embedding=target_embedding,
            pitch_shift=0.0  # No pitch shift for now
        )

        elapsed = time.time() - start_time
        rtf = elapsed / (len(audio) / sr)

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
    output_dir = Path("tests/quality_samples/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "william_as_conor_realtime_30s.wav"

    print(f"\nSaving output: {output_path}")
    sf.write(str(output_path), converted, out_sr)
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Cleanup
    converter.unload()
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("✓ TASK 1.7 COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {output_path}")
    print("Next: Verify audio quality and proceed to Task 2.8\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
