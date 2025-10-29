#!/usr/bin/env python3
"""
AutoVoice Batch Song Conversion Demo

Demonstrates batch processing of multiple songs with the same voice profile.

Usage:
    # Convert multiple songs from directory:
    python demo_batch_conversion.py \\
        --profile-id 550e8400-e29b-41d4-a716-446655440000 \\
        --songs-dir data/songs \\
        --output-dir converted/

    # Convert specific songs:
    python demo_batch_conversion.py \\
        --profile-id 550e8400-e29b-41d4-a716-446655440000 \\
        --songs song1.mp3 song2.mp3 song3.mp3 \\
        --output-dir converted/

    # Use metadata file with per-song settings:
    python demo_batch_conversion.py \\
        --metadata batch_config.json \\
        --output-dir converted/
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Optional
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from tqdm import tqdm

from auto_voice.inference import SingingConversionPipeline


def convert_single_song(
    song_path: str,
    target_profile_id: str,
    output_dir: str,
    settings: Optional[Dict] = None,
    quality_preset: str = "balanced",
    device: str = "cuda"
) -> Dict:
    """
    Convert a single song.

    Args:
        song_path: Path to input song
        target_profile_id: UUID of target voice profile
        output_dir: Directory to save output
        settings: Optional per-song settings
        quality_preset: Quality preset
        device: Device to use

    Returns:
        Dictionary with conversion results
    """
    # Default settings
    if settings is None:
        settings = {}

    # Extract settings
    vocal_volume = settings.get('vocal_volume', 1.0)
    instrumental_volume = settings.get('instrumental_volume', 0.9)
    pitch_shift = settings.get('pitch_shift_semitones', 0)
    temperature = settings.get('temperature', 1.0)

    # Initialize pipeline
    pipeline = SingingConversionPipeline(
        device=device,
        quality_preset=quality_preset
    )

    # Generate output path
    song_name = Path(song_path).stem
    output_path = os.path.join(output_dir, f"{song_name}_converted.wav")

    # Convert
    start_time = time.time()

    try:
        result = pipeline.convert_song(
            song_path=song_path,
            target_profile_id=target_profile_id,
            vocal_volume=vocal_volume,
            instrumental_volume=instrumental_volume,
            pitch_shift_semitones=pitch_shift,
            temperature=temperature,
            return_stems=False
        )

        elapsed_time = time.time() - start_time

        # Copy to output directory
        import shutil
        shutil.copy(result['output_path'], output_path)

        return {
            'song': song_path,
            'output': output_path,
            'status': 'success',
            'time': elapsed_time,
            'quality_metrics': result.get('quality_metrics', {})
        }

    except Exception as e:
        elapsed_time = time.time() - start_time

        return {
            'song': song_path,
            'output': None,
            'status': 'failed',
            'time': elapsed_time,
            'error': str(e)
        }


def batch_convert_sequential(
    songs: List[str],
    target_profile_id: str,
    output_dir: str,
    metadata: Optional[Dict] = None,
    quality_preset: str = "balanced",
    device: str = "cuda"
) -> List[Dict]:
    """
    Convert songs sequentially with progress bar.

    Args:
        songs: List of song paths
        target_profile_id: UUID of target voice profile
        output_dir: Directory to save outputs
        metadata: Optional metadata with per-song settings
        quality_preset: Quality preset
        device: Device to use

    Returns:
        List of conversion results
    """
    results = []

    for song_path in tqdm(songs, desc="Converting songs"):
        # Get song-specific settings from metadata
        settings = None
        if metadata and 'songs' in metadata:
            song_name = Path(song_path).name
            settings = next(
                (s for s in metadata['songs'] if s.get('filename') == song_name),
                None
            )

        # Convert song
        result = convert_single_song(
            song_path=song_path,
            target_profile_id=target_profile_id,
            output_dir=output_dir,
            settings=settings,
            quality_preset=quality_preset,
            device=device
        )

        results.append(result)

    return results


def batch_convert_parallel(
    songs: List[str],
    target_profile_id: str,
    output_dir: str,
    metadata: Optional[Dict] = None,
    quality_preset: str = "balanced",
    num_workers: int = 2,
    device: str = "cuda"
) -> List[Dict]:
    """
    Convert songs in parallel (requires multiple GPUs or CPU).

    Args:
        songs: List of song paths
        target_profile_id: UUID of target voice profile
        output_dir: Directory to save outputs
        metadata: Optional metadata with per-song settings
        quality_preset: Quality preset
        num_workers: Number of parallel workers
        device: Device to use

    Returns:
        List of conversion results
    """
    results = []

    # Create conversion tasks
    tasks = []
    for song_path in songs:
        # Get song-specific settings
        settings = None
        if metadata and 'songs' in metadata:
            song_name = Path(song_path).name
            settings = next(
                (s for s in metadata['songs'] if s.get('filename') == song_name),
                None
            )

        tasks.append({
            'song_path': song_path,
            'target_profile_id': target_profile_id,
            'output_dir': output_dir,
            'settings': settings,
            'quality_preset': quality_preset,
            'device': device
        })

    # Process in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(convert_single_song, **task): task
            for task in tasks
        }

        with tqdm(total=len(futures), desc="Converting songs") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)

    return results


def load_metadata(metadata_path: str) -> Dict:
    """
    Load batch conversion metadata from JSON file.

    Expected format:
    {
        "target_profile_id": "uuid",
        "output_dir": "path/to/output",
        "quality_preset": "balanced",
        "songs": [
            {
                "filename": "song1.mp3",
                "vocal_volume": 1.0,
                "instrumental_volume": 0.9,
                "pitch_shift_semitones": 0,
                "temperature": 1.0
            },
            ...
        ]
    }

    Args:
        metadata_path: Path to JSON metadata file

    Returns:
        Dictionary with batch configuration
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return metadata


def print_summary(results: List[Dict]):
    """
    Print batch conversion summary.

    Args:
        results: List of conversion results
    """
    print(f"\n{'='*60}")
    print("Batch Conversion Summary")
    print(f"{'='*60}")

    # Count successes and failures
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']

    print(f"\nResults:")
    print(f"  Total: {len(results)}")
    print(f"  ✅ Successful: {len(successful)}")
    print(f"  ❌ Failed: {len(failed)}")

    # Processing time statistics
    total_time = sum(r['time'] for r in results)
    avg_time = total_time / len(results) if results else 0

    print(f"\nProcessing Time:")
    print(f"  Total: {total_time:.1f} seconds")
    print(f"  Average per song: {avg_time:.1f} seconds")
    print(f"  Fastest: {min(r['time'] for r in results):.1f} seconds")
    print(f"  Slowest: {max(r['time'] for r in results):.1f} seconds")

    # Quality metrics (if available)
    if successful and 'quality_metrics' in successful[0]:
        print(f"\nQuality Metrics (Average):")

        # Average pitch RMSE
        pitch_rmse_values = [
            r['quality_metrics']['pitch_accuracy']['rmse_hz']
            for r in successful
            if 'pitch_accuracy' in r['quality_metrics']
        ]
        if pitch_rmse_values:
            avg_rmse = np.mean(pitch_rmse_values)
            print(f"  Pitch RMSE: {avg_rmse:.2f} Hz", end="")
            if avg_rmse < 10:
                print(f" ✅")
            else:
                print(f" ⚠️")

        # Average speaker similarity
        similarity_values = [
            r['quality_metrics']['speaker_similarity']['cosine_similarity']
            for r in successful
            if 'speaker_similarity' in r['quality_metrics']
        ]
        if similarity_values:
            avg_similarity = np.mean(similarity_values)
            print(f"  Speaker Similarity: {avg_similarity:.3f}", end="")
            if avg_similarity > 0.85:
                print(f" ✅")
            else:
                print(f" ⚠️")

    # List all conversions
    print(f"\nConversion Details:")
    print(f"  {'-'*58}")

    for i, result in enumerate(results, 1):
        song_name = Path(result['song']).name
        status_icon = "✅" if result['status'] == 'success' else "❌"

        print(f"  {i}. {status_icon} {song_name}")
        print(f"     Time: {result['time']:.1f}s", end="")

        if result['status'] == 'success':
            if 'quality_metrics' in result:
                metrics = result['quality_metrics']
                if 'pitch_accuracy' in metrics:
                    rmse = metrics['pitch_accuracy']['rmse_hz']
                    print(f", RMSE: {rmse:.2f} Hz", end="")
                if 'speaker_similarity' in metrics:
                    similarity = metrics['speaker_similarity']['cosine_similarity']
                    print(f", Similarity: {similarity:.3f}", end="")
            print()
            print(f"     Output: {result['output']}")
        else:
            print()
            print(f"     Error: {result.get('error', 'Unknown error')}")

    print(f"  {'-'*58}")

    # Failed conversions
    if failed:
        print(f"\n❌ Failed Conversions:")
        for result in failed:
            song_name = Path(result['song']).name
            print(f"  • {song_name}: {result.get('error', 'Unknown error')}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AutoVoice Batch Song Conversion Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Convert all songs in directory:
  python demo_batch_conversion.py \\
      --profile-id 550e8400-e29b-41d4-a716-446655440000 \\
      --songs-dir data/songs \\
      --output-dir converted/

  # Convert specific songs:
  python demo_batch_conversion.py \\
      --profile-id 550e8400-e29b-41d4-a716-446655440000 \\
      --songs song1.mp3 song2.mp3 song3.mp3 \\
      --output-dir converted/

  # Use metadata file:
  python demo_batch_conversion.py \\
      --metadata batch_config.json

  # Parallel processing (requires multiple GPUs or CPU):
  python demo_batch_conversion.py \\
      --profile-id 550e8400-e29b-41d4-a716-446655440000 \\
      --songs-dir data/songs \\
      --output-dir converted/ \\
      --parallel \\
      --num-workers 2
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--songs-dir",
        type=str,
        help="Directory containing songs to convert"
    )
    input_group.add_argument(
        "--songs",
        type=str,
        nargs="+",
        help="List of song paths to convert"
    )
    input_group.add_argument(
        "--metadata",
        type=str,
        help="JSON metadata file with batch configuration"
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save converted songs"
    )

    # Profile options
    parser.add_argument(
        "--profile-id",
        type=str,
        help="Target voice profile ID (UUID)"
    )

    # Conversion options
    parser.add_argument(
        "--quality",
        type=str,
        default="balanced",
        choices=["fast", "balanced", "quality"],
        help="Quality preset (default: balanced)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing (requires multiple GPUs or CPU)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda if available)"
    )

    args = parser.parse_args()

    # Load metadata if provided
    metadata = None
    if args.metadata:
        print(f"Loading metadata from: {args.metadata}")
        metadata = load_metadata(args.metadata)

        # Extract settings from metadata
        if not args.profile_id:
            args.profile_id = metadata.get('target_profile_id')
        if not args.output_dir:
            args.output_dir = metadata.get('output_dir')
        if 'quality_preset' in metadata:
            args.quality = metadata['quality_preset']

        # Get song paths from metadata
        if 'songs' in metadata:
            songs_dir = metadata.get('songs_dir', '.')
            song_files = [
                os.path.join(songs_dir, s['filename'])
                for s in metadata['songs']
            ]
            args.songs = song_files

    # Validate inputs
    if not args.profile_id:
        print("❌ Error: --profile-id required (or specify in metadata)")
        return 1

    if not args.output_dir:
        print("❌ Error: --output-dir required (or specify in metadata)")
        return 1

    # Collect songs
    songs = []

    if args.songs_dir:
        songs_dir = Path(args.songs_dir)
        if not songs_dir.exists():
            print(f"❌ Error: Songs directory not found: {args.songs_dir}")
            return 1

        # Find all audio files
        for ext in ['*.mp3', '*.wav', '*.flac', '*.ogg']:
            songs.extend(songs_dir.glob(ext))

        songs = [str(s) for s in songs]

    elif args.songs:
        songs = args.songs

        # Validate files exist
        for song in songs:
            if not os.path.exists(song):
                print(f"❌ Error: Song not found: {song}")
                return 1

    else:
        print("❌ Error: Specify --songs-dir, --songs, or --metadata")
        return 1

    if not songs:
        print("❌ Error: No songs found to convert")
        return 1

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # GPU detection and parallel processing guardrails
    if args.device == "cuda" and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()

        # Check for single GPU with multiple workers
        if args.parallel and args.num_workers > 1 and gpu_count == 1:
            print(f"\n⚠️  WARNING: Single GPU detected with {args.num_workers} parallel workers")
            print(f"   This may cause GPU out-of-memory (OOM) errors and resource contention.")
            print(f"   Recommendation:")
            print(f"     • Use sequential processing (remove --parallel)")
            print(f"     • Or set --num-workers 1 for single GPU")
            print(f"     • Or use CPU for parallel: --device cpu")
            print(f"\n   Overriding to sequential processing for safety...")
            args.parallel = False

        # Multi-GPU usage guidance
        if gpu_count > 1 and args.parallel:
            print(f"\n💡 Multi-GPU Setup Detected ({gpu_count} GPUs)")
            print(f"   For multi-GPU parallel processing:")
            print(f"     • Workers will share available GPUs")
            print(f"     • Consider setting CUDA_VISIBLE_DEVICES per worker")
            print(f"     • Example: Set num_workers={gpu_count} for 1 worker/GPU")

    # Display configuration
    print(f"\nAutoVoice Batch Conversion Demo")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Songs: {len(songs)}")
    print(f"  Target Profile: {args.profile_id}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  Quality Preset: {args.quality}")
    print(f"  Device: {args.device}")

    if args.parallel:
        print(f"  Parallel Processing: Enabled ({args.num_workers} workers)")
    else:
        print(f"  Parallel Processing: Disabled (sequential)")

    if args.device == "cuda":
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            if gpu_count > 1:
                print(f"  Available GPUs: {gpu_count}")
        else:
            print(f"  ⚠️  CUDA not available, falling back to CPU")
            args.device = "cpu"

    try:
        # Start batch conversion
        print(f"\nStarting batch conversion...")
        start_time = time.time()

        if args.parallel:
            results = batch_convert_parallel(
                songs=songs,
                target_profile_id=args.profile_id,
                output_dir=args.output_dir,
                metadata=metadata,
                quality_preset=args.quality,
                num_workers=args.num_workers,
                device=args.device
            )
        else:
            results = batch_convert_sequential(
                songs=songs,
                target_profile_id=args.profile_id,
                output_dir=args.output_dir,
                metadata=metadata,
                quality_preset=args.quality,
                device=args.device
            )

        total_time = time.time() - start_time

        # Print summary
        print_summary(results)

        print(f"\nTotal elapsed time: {total_time:.1f} seconds")
        print(f"{'='*60}")
        print("✅ Batch conversion completed!")
        print(f"{'='*60}")

        # Save results to JSON
        results_file = os.path.join(args.output_dir, 'batch_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {results_file}")
        print(f"Converted songs in: {args.output_dir}")

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
