#!/usr/bin/env python3
"""
Example usage of SingingPitchExtractor production features

This script demonstrates the new features added to SingingPitchExtractor:
1. Vibrato classification
2. Pitch correction suggestions
3. Enhanced real-time streaming with state management
"""

import torch
import numpy as np
from pathlib import Path

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from auto_voice.audio.pitch_extractor import SingingPitchExtractor


def example_vibrato_classification():
    """Example 1: Vibrato Classification"""
    print("=" * 80)
    print("Example 1: Vibrato Classification")
    print("=" * 80)

    # Initialize extractor
    extractor = SingingPitchExtractor(device='cuda' if torch.cuda.is_available() else 'cpu')

    # Create synthetic audio with vibrato (440 Hz with 5 Hz vibrato)
    sample_rate = 22050
    duration = 2.0  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Base frequency 440 Hz with 5 Hz vibrato at 30 cents depth
    carrier_freq = 440.0
    vibrato_rate = 5.0
    vibrato_depth_cents = 30.0

    # Convert cents to frequency ratio
    vibrato_depth_ratio = 2.0 ** (vibrato_depth_cents / 1200.0)

    # Generate signal with vibrato
    frequency_modulation = carrier_freq * (1.0 + (vibrato_depth_ratio - 1.0) * np.sin(2 * np.pi * vibrato_rate * t))
    audio = np.sin(2 * np.pi * frequency_modulation * t)

    # Add some noise for realism
    audio += 0.05 * np.random.randn(len(audio))
    audio = audio / np.max(np.abs(audio))

    print(f"\nGenerated synthetic audio:")
    print(f"  Duration: {duration}s")
    print(f"  Carrier frequency: {carrier_freq} Hz")
    print(f"  Vibrato rate: {vibrato_rate} Hz")
    print(f"  Vibrato depth: {vibrato_depth_cents} cents")

    # Extract F0
    print("\nExtracting F0 contour...")
    f0_data = extractor.extract_f0_contour(audio, sample_rate=sample_rate)

    # Classify vibrato
    print("Classifying vibrato...")
    vibrato = extractor.classify_vibrato(f0_data)

    print("\nVibrato Classification Results:")
    print(f"  Vibrato detected: {vibrato['vibrato_detected']}")
    print(f"  Rate: {vibrato['rate_hz']:.2f} Hz (expected: {vibrato_rate} Hz)")
    print(f"  Extent: {vibrato['extent_cents']:.1f} cents (expected: ~{vibrato_depth_cents} cents)")
    print(f"  Regularity score: {vibrato['regularity_score']:.2f}")
    print(f"  Number of segments: {len(vibrato['segments'])}")

    if vibrato['segments']:
        print("\n  Segments:")
        for i, (start, end, rate, depth) in enumerate(vibrato['segments'][:3]):
            print(f"    {i+1}. Time: {start:.2f}s - {end:.2f}s, Rate: {rate:.2f} Hz, Depth: {depth:.1f} cents")


def example_pitch_correction():
    """Example 2: Pitch Correction Suggestions"""
    print("\n\n" + "=" * 80)
    print("Example 2: Pitch Correction Suggestions")
    print("=" * 80)

    # Initialize extractor
    extractor = SingingPitchExtractor(device='cuda' if torch.cuda.is_available() else 'cpu')

    # Generate audio with intentional pitch errors
    sample_rate = 22050
    duration = 3.0  # seconds

    # Create a sequence of notes (C major scale) with slight detuning
    notes = [
        (261.63, 0.5, 0),     # C4 (perfect)
        (293.66, 0.5, 15),    # D4 (+15 cents, slightly sharp)
        (329.63, 0.5, -20),   # E4 (-20 cents, slightly flat)
        (349.23, 0.5, 0),     # F4 (perfect)
        (392.00, 0.5, 30),    # G4 (+30 cents, noticeably sharp)
        (440.00, 0.5, -25),   # A4 (-25 cents, noticeably flat)
    ]

    print(f"\nGenerating synthetic melody:")
    print("  Notes in C major scale with intentional detuning:")

    audio = []
    for freq, dur, detune_cents in notes:
        # Apply detuning
        detuned_freq = freq * (2.0 ** (detune_cents / 1200.0))
        t = np.linspace(0, dur, int(sample_rate * dur))
        note_audio = 0.5 * np.sin(2 * np.pi * detuned_freq * t)
        audio.append(note_audio)

        note_name = extractor._f0_to_note_name(freq)[0]
        print(f"    {note_name}: {freq:.2f} Hz → {detuned_freq:.2f} Hz ({detune_cents:+d} cents)")

    audio = np.concatenate(audio)

    # Extract F0
    print("\nExtracting F0 contour...")
    f0_data = extractor.extract_f0_contour(audio, sample_rate=sample_rate)

    # Suggest corrections
    print("Analyzing pitch corrections (tolerance: 50 cents)...")
    corrections = extractor.suggest_pitch_corrections(
        f0_data,
        reference_scale='C',
        tolerance_cents=50.0
    )

    print(f"\nPitch Correction Suggestions: {len(corrections)} corrections needed")

    if corrections:
        print("\n  Top 10 corrections:")
        for i, corr in enumerate(corrections[:10]):
            print(f"    {i+1}. Time: {corr['timestamp']:.2f}s")
            print(f"       Detected: {corr['detected_note']} ({corr['detected_f0_hz']:.2f} Hz)")
            print(f"       Target:   {corr['target_note']} ({corr['target_f0_hz']:.2f} Hz)")
            print(f"       Correction: {corr['correction_cents']:+.1f} cents")


def example_realtime_streaming():
    """Example 3: Enhanced Real-time Streaming"""
    print("\n\n" + "=" * 80)
    print("Example 3: Enhanced Real-time Streaming with State Management")
    print("=" * 80)

    # Initialize extractor
    extractor = SingingPitchExtractor(device='cuda' if torch.cuda.is_available() else 'cpu')

    # Generate continuous audio stream
    sample_rate = 22050
    chunk_duration = 0.1  # 100ms chunks
    chunk_size = int(sample_rate * chunk_duration)
    num_chunks = 20

    # Generate a sliding frequency (glissando)
    total_duration = num_chunks * chunk_duration
    t = np.linspace(0, total_duration, int(sample_rate * total_duration))
    freq_start = 200.0  # Hz
    freq_end = 400.0    # Hz
    frequency = np.linspace(freq_start, freq_end, len(t))

    full_audio = np.sin(2 * np.pi * np.cumsum(frequency) / sample_rate)

    print(f"\nGenerating audio stream:")
    print(f"  Frequency sweep: {freq_start} Hz → {freq_end} Hz")
    print(f"  Duration: {total_duration}s")
    print(f"  Chunk size: {chunk_size} samples ({chunk_duration*1000:.0f} ms)")
    print(f"  Number of chunks: {num_chunks}")

    # Process with state (stateful)
    print("\n\n--- Processing with state management (recommended) ---")
    state = extractor.create_realtime_state()
    f0_results_stateful = []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk = torch.from_numpy(full_audio[start_idx:end_idx]).float()

        f0 = extractor.extract_f0_realtime(chunk, sample_rate=sample_rate, state=state, use_cuda_kernel=False)

        # Get mean F0 for this chunk
        if torch.is_tensor(f0):
            mean_f0 = f0.mean().item() if f0.numel() > 0 else 0.0
        else:
            mean_f0 = float(f0)

        f0_results_stateful.append(mean_f0)

        if i % 5 == 0:
            print(f"  Chunk {i+1:2d}: F0 = {mean_f0:6.1f} Hz")

    # Process without state (stateless)
    print("\n\n--- Processing without state (stateless) ---")
    f0_results_stateless = []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk = torch.from_numpy(full_audio[start_idx:end_idx]).float()

        f0 = extractor.extract_f0_realtime(chunk, sample_rate=sample_rate, state=None, use_cuda_kernel=False)

        # Get mean F0 for this chunk
        if torch.is_tensor(f0):
            mean_f0 = f0.mean().item() if f0.numel() > 0 else 0.0
        else:
            mean_f0 = float(f0)

        f0_results_stateless.append(mean_f0)

        if i % 5 == 0:
            print(f"  Chunk {i+1:2d}: F0 = {mean_f0:6.1f} Hz")

    # Compare smoothness
    print("\n\nComparison:")
    stateful_variance = np.var(np.diff(f0_results_stateful))
    stateless_variance = np.var(np.diff(f0_results_stateless))

    print(f"  Stateful processing variance:  {stateful_variance:.2f}")
    print(f"  Stateless processing variance: {stateless_variance:.2f}")
    print(f"  Smoothness improvement: {(1 - stateful_variance/stateless_variance)*100:.1f}%")


def example_configuration():
    """Example 4: Custom Configuration"""
    print("\n\n" + "=" * 80)
    print("Example 4: Custom Configuration")
    print("=" * 80)

    # Custom configuration
    custom_config = {
        'vibrato_regularity_threshold': 0.6,
        'pitch_correction_tolerance_cents': 30.0,
        'pitch_correction_reference_scale': 'G',
        'realtime_smoothing_window': 7,
        'realtime_buffer_size': 8192
    }

    print("\nCustom configuration:")
    for key, value in custom_config.items():
        print(f"  {key}: {value}")

    extractor = SingingPitchExtractor(
        config=custom_config,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("\nExtractor initialized with custom config")
    print(f"  Vibrato regularity threshold: {extractor.vibrato_regularity_threshold}")
    print(f"  Pitch correction tolerance: {extractor.pitch_correction_tolerance_cents} cents")
    print(f"  Pitch correction scale: {extractor.pitch_correction_reference_scale} major")
    print(f"  Real-time smoothing window: {extractor.realtime_smoothing_window}")
    print(f"  Real-time buffer size: {extractor.realtime_buffer_size}")


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("SingingPitchExtractor Production Features - Examples")
    print("=" * 80)

    try:
        example_vibrato_classification()
    except Exception as e:
        print(f"\nError in vibrato classification example: {e}")

    try:
        example_pitch_correction()
    except Exception as e:
        print(f"\nError in pitch correction example: {e}")

    try:
        example_realtime_streaming()
    except Exception as e:
        print(f"\nError in real-time streaming example: {e}")

    try:
        example_configuration()
    except Exception as e:
        print(f"\nError in configuration example: {e}")

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
