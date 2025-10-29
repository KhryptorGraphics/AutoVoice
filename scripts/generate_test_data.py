#!/usr/bin/env python3
"""
Generate synthetic test data for quality evaluation.

Creates simple synthetic waveforms with pitch contours and different timbres
for source and reference audio, plus metadata JSON for test-driven evaluation.

Now creates actual voice profiles using VoiceCloner for realistic testing.
"""

import argparse
import json
import numpy as np
import soundfile as sf
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def generate_sine_with_vibrato(
    duration: float,
    base_freq: float,
    sample_rate: int = 44100,
    vibrato_rate: float = 5.0,
    vibrato_depth: float = 0.02,
    noise_level: float = 0.05,
    timbre_variation: float = 0.0,
    seed: int = 42
) -> np.ndarray:
    """
    Generate synthetic audio with pitch contour and vibrato.

    Args:
        duration: Duration in seconds
        base_freq: Base frequency in Hz
        sample_rate: Audio sample rate
        vibrato_rate: Vibrato frequency in Hz
        vibrato_depth: Vibrato depth as fraction of base_freq
        noise_level: Background noise level
        timbre_variation: Add harmonic variation for different timbres
        seed: Random seed for reproducibility

    Returns:
        Audio waveform as numpy array
    """
    np.random.seed(seed)

    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples)

    # Apply vibrato to frequency
    vibrato = 1.0 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
    instantaneous_freq = base_freq * vibrato

    # Generate phase
    phase = np.cumsum(2 * np.pi * instantaneous_freq / sample_rate)

    # Generate waveform with harmonics
    waveform = np.sin(phase)

    # Add harmonics for richer timbre
    if timbre_variation > 0:
        waveform += timbre_variation * 0.3 * np.sin(2 * phase)  # 2nd harmonic
        waveform += timbre_variation * 0.2 * np.sin(3 * phase)  # 3rd harmonic

    # Add noise
    noise = np.random.normal(0, noise_level, num_samples)
    waveform += noise

    # Normalize
    waveform = waveform / np.max(np.abs(waveform)) * 0.9

    return waveform.astype(np.float32)


def generate_test_case(
    case_id: str,
    output_dir: Path,
    voice_cloner: Optional['VoiceCloner'] = None,
    base_freq: float = 440.0,
    duration: float = 3.0,
    sample_rate: int = 44100,
    seed: int = 42
) -> dict:
    """
    Generate a complete test case with source, reference, and metadata.

    Args:
        case_id: Test case identifier
        output_dir: Directory to save audio files
        voice_cloner: Optional VoiceCloner instance for creating profiles
        base_freq: Base frequency for the test tone
        duration: Audio duration in seconds
        sample_rate: Audio sample rate
        seed: Random seed

    Returns:
        Test case metadata dictionary
    """
    # Generate source audio (clean sine with vibrato)
    source_audio = generate_sine_with_vibrato(
        duration, base_freq, sample_rate,
        vibrato_rate=5.0, vibrato_depth=0.02,
        noise_level=0.02, timbre_variation=0.0,
        seed=seed
    )

    # Generate reference audio (different timbre, similar pitch)
    # Use longer duration for reference to create a better profile
    reference_duration = max(duration, 30.0)  # Minimum 30s for profile creation
    reference_audio = generate_sine_with_vibrato(
        reference_duration, base_freq, sample_rate,
        vibrato_rate=5.5, vibrato_depth=0.025,
        noise_level=0.03, timbre_variation=0.5,
        seed=seed + 1
    )

    # Save audio files
    source_path = output_dir / f"{case_id}_source.wav"
    reference_path = output_dir / f"{case_id}_reference.wav"

    sf.write(source_path, source_audio, sample_rate)
    sf.write(reference_path, reference_audio, sample_rate)

    # Create voice profile from reference audio if VoiceCloner is available
    profile_id = f"synthetic-profile-{case_id}"
    if voice_cloner is not None:
        try:
            # Create profile from reference audio
            profile = voice_cloner.create_voice_profile(
                audio=reference_audio,
                user_id=f"synthetic_test_{case_id}",
                sample_rate=sample_rate,
                metadata={
                    "source": "synthetic_test_data",
                    "case_id": case_id,
                    "base_freq_hz": base_freq,
                    "synthetic": True
                }
            )
            profile_id = profile['profile_id']
            print(f"  Created voice profile: {profile_id}")
        except Exception as e:
            print(f"  Warning: Failed to create profile for {case_id}: {e}")
            # Fall back to synthetic profile ID
            profile_id = f"synthetic-profile-{case_id}"

    # Create test case metadata
    test_case = {
        "id": case_id,
        "source_audio": str(source_path),
        "target_profile_id": profile_id,
        "reference_audio": str(reference_path),
        "metadata": {
            "base_freq_hz": base_freq,
            "duration_sec": duration,
            "sample_rate": sample_rate,
            "synthetic": True,
            "has_real_profile": voice_cloner is not None
        }
    }

    return test_case


def generate_test_dataset(
    output_dir: Path,
    num_samples: int = 6,
    seed: int = 42,
    create_profiles: bool = True
) -> List[dict]:
    """
    Generate complete synthetic test dataset.

    Args:
        output_dir: Directory to save all files
        num_samples: Number of test cases to generate
        seed: Random seed for reproducibility
        create_profiles: Whether to create real voice profiles using VoiceCloner

    Returns:
        List of test case metadata dictionaries
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize VoiceCloner if profiles should be created
    voice_cloner = None
    if create_profiles:
        try:
            from auto_voice.inference.voice_cloner import VoiceCloner

            # Configure VoiceCloner for synthetic test data
            # Use relaxed validation for synthetic audio
            config = {
                'min_duration': 5.0,  # Relaxed for synthetic data
                'max_duration': 300.0,
                'extract_vocal_range': True,
                'extract_timbre_features': True,
                'storage_dir': str(output_dir / 'profiles'),
                'min_snr_db': 5.0  # Relaxed SNR for synthetic audio
            }

            voice_cloner = VoiceCloner(config=config, device='cpu')
            print(f"VoiceCloner initialized for profile creation")
        except Exception as e:
            print(f"Warning: Could not initialize VoiceCloner: {e}")
            print("Falling back to synthetic profile IDs")

    # Generate test cases with different base frequencies
    base_frequencies = [220, 294, 330, 392, 440, 494]  # A3, D4, E4, G4, A4, B4
    test_cases = []

    for i in range(min(num_samples, len(base_frequencies))):
        case_id = f"test_{i+1:03d}"
        base_freq = base_frequencies[i]

        test_case = generate_test_case(
            case_id,
            output_dir,
            voice_cloner=voice_cloner,
            base_freq=base_freq,
            duration=3.0,
            sample_rate=44100,
            seed=seed + i
        )

        test_cases.append(test_case)
        print(f"Generated test case: {case_id} ({base_freq} Hz)")

    return test_cases


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic test data for quality evaluation'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/evaluation/',
        help='Output directory for test data'
    )
    parser.add_argument(
        '--num-samples', '-n',
        type=int,
        default=6,
        help='Number of test samples to generate'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--no-profiles',
        action='store_true',
        help='Skip creating voice profiles (use synthetic profile IDs only)'
    )

    args = parser.parse_args()

    output_dir = Path(args.output)

    # Generate test dataset
    print(f"Generating {args.num_samples} synthetic test cases...")
    print(f"Profile creation: {'disabled' if args.no_profiles else 'enabled'}")

    test_cases = generate_test_dataset(
        output_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        create_profiles=not args.no_profiles
    )

    # Save metadata JSON
    metadata_path = output_dir / 'test_set.json'
    metadata = {
        "test_cases": test_cases,
        "generation_config": {
            "num_samples": args.num_samples,
            "seed": args.seed,
            "synthetic": True,
            "profiles_created": not args.no_profiles
        }
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Count profiles created
    profiles_created = sum(1 for tc in test_cases if not tc['target_profile_id'].startswith('synthetic-profile-'))

    print(f"\nSynthetic test dataset generated:")
    print(f"  Test cases: {len(test_cases)}")
    print(f"  Voice profiles created: {profiles_created}/{len(test_cases)}")
    print(f"  Output directory: {output_dir}")
    print(f"  Metadata file: {metadata_path}")

    if profiles_created > 0:
        print(f"  Profiles directory: {output_dir / 'profiles'}")


if __name__ == '__main__':
    main()
