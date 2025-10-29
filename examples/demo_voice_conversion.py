#!/usr/bin/env python3
"""
AutoVoice Song Conversion Demo

Demonstrates single song conversion with voice cloning.

Usage:
    python demo_voice_conversion.py \\
        --voice-sample my_voice.wav \\
        --song song.mp3 \\
        --output converted.wav

    # Or use existing profile:
    python demo_voice_conversion.py \\
        --profile-id 550e8400-e29b-41d4-a716-446655440000 \\
        --song song.mp3 \\
        --output converted.wav
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from auto_voice.inference import VoiceCloner, SingingConversionPipeline
from auto_voice.utils.quality_metrics import QualityMetricsAggregator


def create_voice_profile(
    voice_sample: str,
    user_id: str = "demo_user",
    profile_name: str = "Demo Profile",
    device: str = "cuda"
) -> dict:
    """
    Create a voice profile from an audio sample.

    Args:
        voice_sample: Path to voice audio file (30-60s recommended)
        user_id: User identifier
        profile_name: Human-readable profile name
        device: Device to use ('cuda' or 'cpu')

    Returns:
        Dictionary with profile information including profile_id
    """
    print(f"\n{'='*60}")
    print("Creating Voice Profile")
    print(f"{'='*60}")

    # Initialize cloner
    cloner = VoiceCloner(device=device)

    # Create profile
    print(f"Loading voice sample: {voice_sample}")
    profile = cloner.create_voice_profile(
        audio=voice_sample,
        user_id=user_id,
        profile_name=profile_name
    )

    # Display results
    print(f"\n✅ Voice profile created successfully!")
    print(f"\nProfile Details:")
    print(f"  Profile ID: {profile['profile_id']}")
    print(f"  User ID: {profile['user_id']}")
    print(f"  Profile Name: {profile['profile_name']}")
    print(f"  Created: {profile['created_at']}")

    print(f"\nAudio Info:")
    print(f"  Duration: {profile['audio_info']['duration_seconds']:.2f} seconds")
    print(f"  Sample Rate: {profile['audio_info']['sample_rate']} Hz")
    print(f"  Format: {profile['audio_info']['format']}")

    print(f"\nVocal Range:")
    print(f"  Min: {profile['vocal_range']['min_note']} ({profile['vocal_range']['min_pitch_hz']:.2f} Hz)")
    print(f"  Max: {profile['vocal_range']['max_note']} ({profile['vocal_range']['max_pitch_hz']:.2f} Hz)")
    print(f"  Range: {profile['vocal_range']['range_semitones']} semitones")

    print(f"\nQuality Metrics:")
    print(f"  SNR: {profile['quality_metrics']['snr_db']:.2f} dB")
    print(f"  Quality Score: {profile['quality_metrics']['quality_score']:.2f}")

    # Quality assessment
    duration = profile['audio_info']['duration_seconds']
    snr = profile['quality_metrics']['snr_db']

    print(f"\nQuality Assessment:")
    if duration >= 45:
        print(f"  ✅ Duration optimal (45-60s)")
    elif duration >= 30:
        print(f"  ⚠️  Duration acceptable (30-45s)")
    else:
        print(f"  ❌ Duration too short (<30s)")

    if snr >= 15:
        print(f"  ✅ SNR good (>15 dB)")
    elif snr >= 10:
        print(f"  ⚠️  SNR fair (10-15 dB)")
    else:
        print(f"  ❌ SNR poor (<10 dB)")

    return profile


def convert_song(
    song_path: str,
    target_profile_id: str,
    output_path: str,
    vocal_volume: float = 1.0,
    instrumental_volume: float = 0.9,
    pitch_shift: int = 0,
    temperature: float = 1.0,
    quality_preset: str = "balanced",
    return_stems: bool = False,
    device: str = "cuda"
) -> dict:
    """
    Convert a song to target voice.

    Args:
        song_path: Path to input song
        target_profile_id: UUID of target voice profile
        output_path: Path to save converted song
        vocal_volume: Volume of converted vocals (0.0-2.0)
        instrumental_volume: Volume of instrumental (0.0-2.0)
        pitch_shift: Pitch shift in semitones (±12)
        temperature: Expressiveness control (0.5-2.0)
        quality_preset: Quality preset ('fast', 'balanced', 'quality')
        return_stems: Whether to save separated vocals/instrumental
        device: Device to use ('cuda' or 'cpu')

    Returns:
        Dictionary with conversion results and metrics
    """
    print(f"\n{'='*60}")
    print("Converting Song")
    print(f"{'='*60}")

    # Initialize pipeline
    print(f"Initializing pipeline with '{quality_preset}' preset...")
    pipeline = SingingConversionPipeline(
        device=device,
        quality_preset=quality_preset
    )

    # Display settings
    print(f"\nConversion Settings:")
    print(f"  Input Song: {song_path}")
    print(f"  Target Profile: {target_profile_id}")
    print(f"  Output Path: {output_path}")
    print(f"  Quality Preset: {quality_preset}")
    print(f"  Vocal Volume: {vocal_volume}")
    print(f"  Instrumental Volume: {instrumental_volume}")
    print(f"  Pitch Shift: {pitch_shift:+d} semitones")
    print(f"  Temperature: {temperature}")

    # Start conversion
    print(f"\nStarting conversion...")
    print(f"This may take a few minutes depending on song length and quality preset.")

    start_time = time.time()

    result = pipeline.convert_song(
        song_path=song_path,
        target_profile_id=target_profile_id,
        vocal_volume=vocal_volume,
        instrumental_volume=instrumental_volume,
        pitch_shift_semitones=pitch_shift,
        temperature=temperature,
        return_stems=return_stems
    )

    elapsed_time = time.time() - start_time

    # Display results
    print(f"\n✅ Conversion completed!")
    print(f"\nOutput:")
    print(f"  Converted Song: {result['output_path']}")
    print(f"  Processing Time: {elapsed_time:.1f} seconds")

    if return_stems and result.get('stems'):
        print(f"\nStems:")
        if result['stems'].get('vocals'):
            print(f"  Vocals: {result['stems']['vocals']}")
        if result['stems'].get('instrumental'):
            print(f"  Instrumental: {result['stems']['instrumental']}")

    # Display quality metrics
    if 'quality_metrics' in result:
        metrics = result['quality_metrics']

        print(f"\nQuality Metrics:")
        print(f"  {'='*50}")

        if 'pitch_accuracy' in metrics:
            pitch = metrics['pitch_accuracy']
            rmse_hz = pitch.get('rmse_hz', 0)
            correlation = pitch.get('correlation', 0)

            print(f"  Pitch Accuracy:")
            print(f"    RMSE (Hz): {rmse_hz:.2f} Hz", end="")

            if rmse_hz < 10:
                print(f" ✅ (Excellent)")
            elif rmse_hz < 15:
                print(f" ✅ (Good)")
            else:
                print(f" ⚠️  (Fair)")

            print(f"    Correlation: {correlation:.3f}")

        if 'speaker_similarity' in metrics:
            speaker = metrics['speaker_similarity']
            similarity = speaker.get('cosine_similarity', 0)

            print(f"  Speaker Similarity:")
            print(f"    Cosine Similarity: {similarity:.3f}", end="")

            if similarity > 0.85:
                print(f" ✅ (Excellent)")
            elif similarity > 0.75:
                print(f" ✅ (Good)")
            else:
                print(f" ⚠️  (Fair)")

        if 'f0_statistics' in metrics:
            f0_stats = metrics['f0_statistics']

            print(f"  Pitch Statistics:")
            print(f"    Range: {f0_stats.get('min_note', 'N/A')} - {f0_stats.get('max_note', 'N/A')}")
            print(f"    Mean: {f0_stats.get('mean_hz', 0):.1f} Hz")
            print(f"    Std: {f0_stats.get('std_hz', 0):.1f} Hz")

        print(f"  {'='*50}")

    # Copy to output path if needed
    if result['output_path'] != output_path:
        import shutil
        from pathlib import Path

        # Create parent directory if it doesn't exist
        output_parent = Path(output_path).parent
        output_parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        shutil.copy(result['output_path'], output_path)
        print(f"\nCopied output to: {output_path}")

    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AutoVoice Song Conversion Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Create profile and convert song:
  python demo_voice_conversion.py \\
      --voice-sample my_voice.wav \\
      --song song.mp3 \\
      --output converted.wav

  # Use existing profile:
  python demo_voice_conversion.py \\
      --profile-id 550e8400-e29b-41d4-a716-446655440000 \\
      --song song.mp3 \\
      --output converted.wav

  # With pitch shift and quality preset:
  python demo_voice_conversion.py \\
      --voice-sample my_voice.wav \\
      --song song.mp3 \\
      --output converted.wav \\
      --pitch-shift -2 \\
      --quality quality

  # Save separated stems:
  python demo_voice_conversion.py \\
      --voice-sample my_voice.wav \\
      --song song.mp3 \\
      --output converted.wav \\
      --return-stems
        """
    )

    # Voice profile options
    profile_group = parser.add_mutually_exclusive_group(required=True)
    profile_group.add_argument(
        "--voice-sample",
        type=str,
        help="Path to voice sample audio (30-60s recommended)"
    )
    profile_group.add_argument(
        "--profile-id",
        type=str,
        help="Existing voice profile ID (UUID)"
    )

    # Required arguments
    parser.add_argument(
        "--song",
        type=str,
        required=True,
        help="Path to song to convert"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save converted song"
    )

    # Optional arguments
    parser.add_argument(
        "--profile-name",
        type=str,
        default="Demo Profile",
        help="Name for voice profile (default: Demo Profile)"
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default="demo_user",
        help="User ID for voice profile (default: demo_user)"
    )
    parser.add_argument(
        "--vocal-volume",
        type=float,
        default=1.0,
        help="Volume of converted vocals (0.0-2.0, default: 1.0)"
    )
    parser.add_argument(
        "--instrumental-volume",
        type=float,
        default=0.9,
        help="Volume of instrumental (0.0-2.0, default: 0.9)"
    )
    parser.add_argument(
        "--pitch-shift",
        type=int,
        default=0,
        help="Pitch shift in semitones (±12, default: 0)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Expressiveness control (0.5-2.0, default: 1.0)"
    )
    parser.add_argument(
        "--quality",
        type=str,
        default="balanced",
        choices=["fast", "balanced", "quality"],
        help="Quality preset (default: balanced)"
    )
    parser.add_argument(
        "--return-stems",
        action="store_true",
        help="Save separated vocals and instrumental"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda if available)"
    )

    args = parser.parse_args()

    # Validate inputs
    if args.voice_sample and not os.path.exists(args.voice_sample):
        print(f"❌ Error: Voice sample not found: {args.voice_sample}")
        return 1

    if not os.path.exists(args.song):
        print(f"❌ Error: Song not found: {args.song}")
        return 1

    # Display system info
    print(f"\nAutoVoice Song Conversion Demo")
    print(f"{'='*60}")
    print(f"Device: {args.device}")

    if args.device == "cuda":
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
        else:
            print(f"❌ CUDA not available, falling back to CPU")
            args.device = "cpu"

    try:
        # Step 1: Create or use existing voice profile
        if args.voice_sample:
            profile = create_voice_profile(
                voice_sample=args.voice_sample,
                user_id=args.user_id,
                profile_name=args.profile_name,
                device=args.device
            )
            target_profile_id = profile['profile_id']
        else:
            target_profile_id = args.profile_id
            print(f"\nUsing existing profile: {target_profile_id}")

        # Step 2: Convert song
        result = convert_song(
            song_path=args.song,
            target_profile_id=target_profile_id,
            output_path=args.output,
            vocal_volume=args.vocal_volume,
            instrumental_volume=args.instrumental_volume,
            pitch_shift=args.pitch_shift,
            temperature=args.temperature,
            quality_preset=args.quality,
            return_stems=args.return_stems,
            device=args.device
        )

        # Success
        print(f"\n{'='*60}")
        print("✅ Demo completed successfully!")
        print(f"{'='*60}")
        print(f"\nConverted song saved to: {args.output}")

        if args.return_stems:
            print(f"\nStems saved:")
            if result.get('stems', {}).get('vocals'):
                print(f"  Vocals: {result['stems']['vocals']}")
            if result.get('stems', {}).get('instrumental'):
                print(f"  Instrumental: {result['stems']['instrumental']}")

        print(f"\nNext steps:")
        print(f"  • Listen to the converted song: {args.output}")
        print(f"  • Try different quality presets (--quality fast/balanced/quality)")
        print(f"  • Adjust pitch with --pitch-shift")
        print(f"  • See docs/voice_conversion_guide.md for more options")

        return 0

    except KeyboardInterrupt:
        print(f"\n\n❌ Interrupted by user")
        return 130

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
