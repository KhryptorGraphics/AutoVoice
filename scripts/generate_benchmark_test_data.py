#!/usr/bin/env python3
"""
Generate test data for performance benchmarking.

Creates realistic audio test files of various durations (1s, 5s, 10s, 30s, 60s)
for benchmarking TTS synthesis, voice conversion, and CUDA kernel performance.
"""

import argparse
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Tuple
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def generate_sine_with_vibrato(
    duration: float,
    sample_rate: int = 22050,
    base_freq: float = 220.0,
    vibrato_rate: float = 5.0,
    vibrato_depth: float = 0.5
) -> np.ndarray:
    """
    Generate a sine wave with vibrato to simulate singing voice.
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        base_freq: Base frequency in Hz
        vibrato_rate: Vibrato frequency in Hz
        vibrato_depth: Vibrato depth in semitones
        
    Returns:
        Audio signal as numpy array
    """
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    # Add vibrato (frequency modulation)
    vibrato = vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
    freq = base_freq * (2 ** (vibrato / 12))  # Convert semitones to frequency ratio
    
    # Generate base sine wave
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    signal = np.sin(phase)
    
    # Add harmonics for more realistic timbre
    signal += 0.3 * np.sin(2 * phase)  # 2nd harmonic
    signal += 0.15 * np.sin(3 * phase)  # 3rd harmonic
    signal += 0.08 * np.sin(4 * phase)  # 4th harmonic
    
    # Add slight noise for realism
    noise = np.random.normal(0, 0.02, num_samples)
    signal += noise
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    # Apply fade in/out to avoid clicks
    fade_samples = int(0.01 * sample_rate)  # 10ms fade
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    signal[:fade_samples] *= fade_in
    signal[-fade_samples:] *= fade_out
    
    return signal.astype(np.float32)


def create_voice_profile(profile_id: str, output_dir: Path) -> dict:
    """
    Create a synthetic voice profile.
    
    Args:
        profile_id: Profile identifier
        output_dir: Output directory for profile
        
    Returns:
        Profile metadata dictionary
    """
    # Create mock voice profile with random embeddings
    profile_data = {
        'profile_id': profile_id,
        'embedding': np.random.randn(256).astype(np.float32).tolist(),  # 256-dim embedding
        'sample_rate': 22050,
        'created_at': 'benchmark_generation',
        'metadata': {
            'type': 'synthetic',
            'purpose': 'benchmarking'
        }
    }
    
    # Save profile
    profile_file = output_dir / f"{profile_id}.json"
    with open(profile_file, 'w') as f:
        json.dump(profile_data, f, indent=2)
    
    return {
        'profile_id': profile_id,
        'file': str(profile_file.relative_to(output_dir.parent.parent))
    }


def generate_test_data(
    output_dir: Path,
    durations: List[float],
    sample_rates: List[int],
    num_profiles: int,
    skip_profiles: bool = False
) -> dict:
    """
    Generate all test data for benchmarking.
    
    Args:
        output_dir: Output directory
        durations: List of audio durations in seconds
        sample_rates: List of sample rates in Hz
        num_profiles: Number of voice profiles to generate
        skip_profiles: Skip voice profile generation
        
    Returns:
        Metadata dictionary
    """
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir = output_dir / 'profiles'
    profiles_dir.mkdir(exist_ok=True)
    
    metadata = {
        'files': [],
        'profiles': []
    }
    
    # Generate audio files
    print("Generating audio test files...")
    for duration in durations:
        for sample_rate in sample_rates:
            # Generate audio
            base_freq = 220.0 + np.random.uniform(-20, 20)  # Vary base frequency slightly
            audio = generate_sine_with_vibrato(
                duration=duration,
                sample_rate=sample_rate,
                base_freq=base_freq,
                vibrato_rate=5.0 + np.random.uniform(-0.5, 0.5),
                vibrato_depth=0.5 + np.random.uniform(-0.1, 0.1)
            )
            
            # Save audio file
            filename = f"audio_{int(duration)}s_{sample_rate}hz.wav"
            filepath = output_dir / filename
            sf.write(filepath, audio, sample_rate)
            
            # Add to metadata
            metadata['files'].append({
                'path': str(filepath.relative_to(output_dir.parent.parent)),
                'duration': duration,
                'sample_rate': sample_rate,
                'base_freq': base_freq,
                'num_samples': len(audio)
            })
            
            print(f"  Created: {filename} ({duration}s @ {sample_rate}Hz)")
    
    # Generate voice profiles
    if not skip_profiles:
        print(f"\nGenerating {num_profiles} voice profiles...")
        for i in range(num_profiles):
            profile_id = f"test_profile_{i+1}"
            profile_meta = create_voice_profile(profile_id, profiles_dir)
            metadata['profiles'].append(profile_meta)
            print(f"  Created: {profile_id}")
    
    # Save metadata
    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to: {metadata_file}")
    print(f"\nTest data generation complete!")
    print(f"  Audio files: {len(metadata['files'])}")
    print(f"  Voice profiles: {len(metadata['profiles'])}")
    print(f"  Output directory: {output_dir}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Generate test data for performance benchmarking'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('tests/data/benchmark'),
        help='Output directory (default: tests/data/benchmark)'
    )
    parser.add_argument(
        '--durations',
        type=str,
        default='1,5,10,30,60',
        help='Comma-separated durations in seconds (default: 1,5,10,30,60)'
    )
    parser.add_argument(
        '--sample-rates',
        type=str,
        default='22050,44100',
        help='Comma-separated sample rates (default: 22050,44100)'
    )
    parser.add_argument(
        '--num-profiles',
        type=int,
        default=2,
        help='Number of voice profiles to generate (default: 2)'
    )
    parser.add_argument(
        '--no-profiles',
        action='store_true',
        help='Skip voice profile generation'
    )
    
    args = parser.parse_args()
    
    # Parse durations and sample rates
    durations = [float(d) for d in args.durations.split(',')]
    sample_rates = [int(sr) for sr in args.sample_rates.split(',')]
    
    # Generate test data
    generate_test_data(
        output_dir=args.output_dir,
        durations=durations,
        sample_rates=sample_rates,
        num_profiles=args.num_profiles,
        skip_profiles=args.no_profiles
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

