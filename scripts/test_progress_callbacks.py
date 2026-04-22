#!/usr/bin/env python3
"""Task 6.5: Test progress callbacks for long conversions.

Verifies that both pipelines emit progress updates during conversion.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'models' / 'seed-vc'))

import numpy as np
import librosa

from quality_sample_paths import resolve_quality_sample_runtime_paths
from realtime_pipeline import RealtimeVoiceConverter, RealtimeConfig
from quality_pipeline import QualityVoiceConverter, QualityConfig


def test_realtime_progress():
    """Test realtime pipeline progress callbacks."""
    print("\n" + "=" * 70)
    print("TEST 1: REALTIME PIPELINE PROGRESS CALLBACKS")
    print("=" * 70)

    # Track progress updates
    progress_updates = []

    def progress_callback(progress: float, status: str):
        progress_updates.append((progress, status))
        print(f"  [{progress*100:5.1f}%] {status}")

    # Load test audio
    paths = resolve_quality_sample_runtime_paths()
    test_audio = paths["william_test_audio"]
    audio, sr = librosa.load(str(test_audio), sr=None, mono=True, duration=10.0)
    speaker_embedding = np.random.randn(256).astype(np.float32)

    # Initialize and convert
    config = RealtimeConfig(sample_rate=22050, fp16=True, device="cuda")
    converter = RealtimeVoiceConverter(config)

    print("\nConverting with progress callbacks...")
    converted, out_sr = converter.convert_full(
        audio, sr, speaker_embedding,
        progress_callback=progress_callback
    )

    converter.unload()

    # Verify
    print(f"\nTotal progress updates: {len(progress_updates)}")
    print(f"First update: {progress_updates[0] if progress_updates else 'None'}")
    print(f"Last update: {progress_updates[-1] if progress_updates else 'None'}")

    assert len(progress_updates) > 0, "No progress updates received"
    assert progress_updates[0][0] == 0.0, "First progress should be 0.0"
    assert progress_updates[-1][0] == 1.0, "Last progress should be 1.0"
    assert "Complete" in progress_updates[-1][1], "Last status should indicate completion"

    print("\n✓ Realtime pipeline progress callbacks working correctly")
    return True


def test_quality_progress():
    """Test quality pipeline progress callbacks."""
    print("\n" + "=" * 70)
    print("TEST 2: QUALITY PIPELINE PROGRESS CALLBACKS")
    print("=" * 70)

    # Track progress updates
    progress_updates = []

    def progress_callback(progress: float, status: str):
        progress_updates.append((progress, status))
        print(f"  [{progress*100:5.1f}%] {status}")

    # Load test audio
    paths = resolve_quality_sample_runtime_paths()
    test_audio = paths["william_test_audio"]
    audio, sr = librosa.load(str(test_audio), sr=None, mono=True, duration=5.0)
    reference = librosa.load(
        str(paths["conor_reference_audio"]),
        sr=None,
        mono=True,
        duration=5.0,
    )[0]

    # Initialize and convert
    config = QualityConfig(sample_rate=44100, diffusion_steps=10, fp16=True, device="cuda")
    converter = QualityVoiceConverter(config)

    print("\nConverting with progress callbacks...")
    converted, out_sr = converter.convert(
        audio, sr, reference, sr,
        progress_callback=progress_callback
    )

    converter.unload()

    # Verify
    print(f"\nTotal progress updates: {len(progress_updates)}")
    print(f"First update: {progress_updates[0] if progress_updates else 'None'}")
    print(f"Last update: {progress_updates[-1] if progress_updates else 'None'}")

    assert len(progress_updates) > 0, "No progress updates received"
    assert progress_updates[0][0] <= 0.2, "First progress should be early stage"
    assert progress_updates[-1][0] == 1.0, "Last progress should be 1.0"
    assert "Complete" in progress_updates[-1][1], "Last status should indicate completion"

    print("\n✓ Quality pipeline progress callbacks working correctly")
    return True


def main():
    os.chdir(Path(__file__).parent.parent)

    print("\n" + "=" * 70)
    print("  TASK 6.5: PROGRESS CALLBACK TEST")
    print("  Verifying progress updates during long conversions")
    print("=" * 70)

    try:
        # Test both pipelines
        realtime_ok = test_realtime_progress()
        quality_ok = test_quality_progress()

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        if realtime_ok and quality_ok:
            print("\n✓ Both pipelines emit progress callbacks correctly")
            print("  - Realtime: Progress from 0.0 to 1.0 with status updates")
            print("  - Quality: Progress from 0.0 to 1.0 with detailed stage updates")
        else:
            print("\n✗ Some progress callbacks failed")

        print("\n" + "=" * 70)
        print("✓ TASK 6.5 COMPLETE")
        print("=" * 70)
        print("\nProgress callbacks implemented for WebSocket real-time updates")
        print("UI can now show conversion progress to users\n")

        return 0

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
