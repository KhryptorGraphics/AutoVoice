#!/usr/bin/env python3
"""YouTube Artist Training Pipeline.

Downloads videos, separates vocals, runs diarization, and trains LoRA models
for Connor Maynard and William Singe.

Usage:
    python scripts/youtube_artist_pipeline.py --artist conor_maynard
    python scripts/youtube_artist_pipeline.py --artist william_singe
    python scripts/youtube_artist_pipeline.py --stage download --artist conor_maynard
    python scripts/youtube_artist_pipeline.py --stage separate --artist william_singe
    python scripts/youtube_artist_pipeline.py --stage train --artist all
"""

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / 'data'

# Add src to path
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from auto_voice.youtube import download_artist_videos, scrape_artist_channel
from auto_voice.audio.separation import VocalSeparator
from auto_voice.storage.paths import (
    resolve_data_dir,
    resolve_diarized_audio_dir,
    resolve_separated_audio_dir,
    resolve_training_vocals_dir,
    resolve_youtube_audio_dir,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Profile IDs from existing voice profiles
ARTIST_PROFILES = {
    'conor_maynard': {
        'profile_id': 'c572d02c-c687-4bed-8676-6ad253cf1c91',
        'channel_key': 'conor_maynard',
        'name': 'Connor Maynard',
    },
    'william_singe': {
        'profile_id': '7da05140-1303-40c6-95d9-5b6e2c3624df',
        'channel_key': 'william_singe',
        'name': 'William Singe',
    },
}


def resolve_runtime_paths(data_dir: str | None = None) -> dict[str, Path]:
    """Resolve runtime directories for this script."""
    resolved_data_dir = resolve_data_dir(
        data_dir or os.environ.get('DATA_DIR') or str(DEFAULT_DATA_DIR)
    )
    return {
        'data_dir': resolved_data_dir,
        'audio_root': resolve_youtube_audio_dir(data_dir=str(resolved_data_dir)),
        'separated_root': resolve_separated_audio_dir(data_dir=str(resolved_data_dir)),
        'diarized_root': resolve_diarized_audio_dir(data_dir=str(resolved_data_dir)),
        'training_vocals_dir': resolve_training_vocals_dir(data_dir=str(resolved_data_dir)),
    }


def resolve_artist_paths(artist_key: str, data_dir: str | None = None) -> dict[str, Path]:
    """Resolve artist-specific runtime directories for this script."""
    paths = resolve_runtime_paths(data_dir)
    return {
        **paths,
        'audio_dir': paths['audio_root'] / artist_key,
        'separated_dir': paths['separated_root'] / artist_key,
        'diarized_dir': paths['diarized_root'] / artist_key,
        'training_vocals_artist_dir': paths['training_vocals_dir'] / artist_key,
    }


def stage_download(
    artist_key: str,
    max_videos: int = 200,
    max_workers: int = 4,
    data_dir: str | None = None,
):
    """Stage 1: Download audio from YouTube channel.

    Args:
        artist_key: Artist key from ARTIST_PROFILES
        max_videos: Maximum videos to download
        max_workers: Parallel download workers
        data_dir: Optional runtime data directory override
    """
    logger.info(f"=== Stage 1: Download {artist_key} videos ===")
    paths = resolve_artist_paths(artist_key, data_dir)

    results = download_artist_videos(
        artist_key,
        output_subdir=str(paths['audio_dir'].resolve()),
        max_videos=max_videos,
        max_workers=max_workers
    )

    success = sum(1 for r in results if r.success)
    logger.info(f"Downloaded {success}/{len(results)} videos successfully")

    return results


def stage_separate(
    artist_key: str,
    gpu_memory_limit_gb: float = 8.0,
    data_dir: str | None = None,
):
    """Stage 2: Separate vocals from downloaded audio.

    Args:
        artist_key: Artist key from ARTIST_PROFILES
        gpu_memory_limit_gb: Max GPU memory to use
        data_dir: Optional runtime data directory override
    """
    logger.info(f"=== Stage 2: Separate vocals for {artist_key} ===")
    import librosa
    import soundfile as sf

    paths = resolve_artist_paths(artist_key, data_dir)
    audio_dir = paths['audio_dir']
    output_dir = paths['separated_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_dir.exists():
        raise RuntimeError(f"Audio directory not found: {audio_dir}. Run download stage first.")

    audio_files = list(audio_dir.glob('*.wav'))
    if not audio_files:
        raise RuntimeError(f"No audio files found in {audio_dir}")

    logger.info(f"Found {len(audio_files)} audio files to process")

    # No segment parameter - let Demucs handle processing internally
    # Memory is managed via explicit cleanup between files
    separator = VocalSeparator(segment=None)
    results = []

    for i, audio_file in enumerate(audio_files, 1):
        output_path = output_dir / f'{audio_file.stem}_vocals.wav'

        if output_path.exists():
            logger.info(f"[{i}/{len(audio_files)}] Already separated: {audio_file.name}")
            results.append({'file': str(audio_file), 'success': True, 'skipped': True})
            continue

        try:
            logger.info(f"[{i}/{len(audio_files)}] Separating: {audio_file.name}")
            # Load audio file
            audio, sr = librosa.load(str(audio_file), sr=None, mono=False)
            # Separate vocals and instrumental
            separated = separator.separate(audio, sr)
            vocals = separated['vocals']

            # Save vocals at original sample rate
            sf.write(str(output_path), vocals, sr)

            results.append({'file': str(audio_file), 'success': True})

            # Aggressive memory cleanup after each file to prevent OOM
            del audio, separated, vocals
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Failed to separate {audio_file.name}: {e}")
            results.append({'file': str(audio_file), 'success': False, 'error': str(e)})

    success = sum(1 for r in results if r.get('success'))
    logger.info(f"Separated {success}/{len(results)} files successfully")

    return results


def stage_diarize(
    artist_key: str,
    max_memory_gb: float = 4.0,
    data_dir: str | None = None,
):
    """Stage 3: Run speaker diarization to identify artist segments.

    Args:
        artist_key: Artist key from ARTIST_PROFILES
        max_memory_gb: Maximum memory per diarization chunk
        data_dir: Optional runtime data directory override
    """
    logger.info(f"=== Stage 3: Diarize vocals for {artist_key} ===")

    from auto_voice.audio.speaker_diarization import SpeakerDiarizer

    paths = resolve_artist_paths(artist_key, data_dir)
    vocals_dir = paths['separated_dir']
    output_dir = paths['diarized_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    if not vocals_dir.exists():
        raise RuntimeError(f"Vocals directory not found: {vocals_dir}. Run separate stage first.")

    vocal_files = list(vocals_dir.glob('*_vocals.wav'))
    if not vocal_files:
        raise RuntimeError(f"No vocal files found in {vocals_dir}")

    logger.info(f"Found {len(vocal_files)} vocal files to diarize")

    diarizer = SpeakerDiarizer(max_memory_gb=max_memory_gb)
    results = []

    for i, vocal_file in enumerate(vocal_files, 1):
        output_json = output_dir / f'{vocal_file.stem}_diarization.json'

        if output_json.exists():
            logger.info(f"[{i}/{len(vocal_files)}] Already diarized: {vocal_file.name}")
            results.append({'file': str(vocal_file), 'success': True, 'skipped': True})
            continue

        try:
            logger.info(f"[{i}/{len(vocal_files)}] Diarizing: {vocal_file.name}")
            result = diarizer.diarize(str(vocal_file))

            # Save diarization result
            with open(output_json, 'w') as f:
                json.dump({
                    'file': str(vocal_file),
                    'num_speakers': result.num_speakers,
                    'segments': [
                        {'start': s.start, 'end': s.end, 'speaker': s.speaker_id}
                        for s in result.segments
                    ]
                }, f, indent=2)

            results.append({
                'file': str(vocal_file),
                'success': True,
                'num_speakers': result.num_speakers,
                'num_segments': len(result.segments)
            })

            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Failed to diarize {vocal_file.name}: {e}")
            results.append({'file': str(vocal_file), 'success': False, 'error': str(e)})

    success = sum(1 for r in results if r.get('success'))
    logger.info(f"Diarized {success}/{len(results)} files successfully")

    return results


def stage_train(artist_key: str, epochs: int = 50, lora_rank: int = 16,
                lora_alpha: int = 32, gradient_checkpointing: bool = True,
                data_dir: str | None = None):
    """Stage 4: Train LoRA adapter with OOM protection.

    Args:
        artist_key: Artist key from ARTIST_PROFILES
        epochs: Training epochs
        lora_rank: LoRA rank (higher = more capacity)
        lora_alpha: LoRA alpha scaling
        gradient_checkpointing: Enable gradient checkpointing for memory
        data_dir: Optional runtime data directory override
    """
    logger.info(f"=== Stage 4: Train LoRA for {artist_key} ===")

    from auto_voice.training.trainer import Trainer
    from auto_voice.training.job_manager import TrainingJobManager

    profile = ARTIST_PROFILES[artist_key]
    profile_id = profile['profile_id']

    # Collect training samples from diarized output
    paths = resolve_artist_paths(artist_key, data_dir)
    samples_dir = paths['diarized_dir']
    if not samples_dir.exists():
        raise RuntimeError(f"Diarized data not found: {samples_dir}. Run diarize stage first.")

    # Create training job
    job_manager = TrainingJobManager(paths['data_dir'])

    job_id = job_manager.create_job(
        profile_id=profile_id,
        epochs=epochs,
        batch_size=4,  # Small batch for OOM protection
        learning_rate=1e-4,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        gradient_checkpointing=gradient_checkpointing,
        mixed_precision=True,
    )

    logger.info(f"Created training job: {job_id}")
    logger.info(f"LoRA config: rank={lora_rank}, alpha={lora_alpha}")
    logger.info(f"OOM protection: gradient_checkpointing={gradient_checkpointing}, mixed_precision=True")

    # Start training
    job_manager.start_job(job_id)

    # Monitor until complete
    while True:
        status = job_manager.get_job_status(job_id)
        if status.get('status') in ('completed', 'failed'):
            break
        logger.info(f"Training progress: {status.get('progress', 0):.1%}")
        import time
        time.sleep(30)

    if status.get('status') == 'completed':
        logger.info(f"Training completed! Adapter saved to: {status.get('adapter_path')}")
    else:
        logger.error(f"Training failed: {status.get('error')}")

    return status


def main():
    parser = argparse.ArgumentParser(description='YouTube Artist Training Pipeline')
    parser.add_argument('--artist', required=True,
                        choices=['conor_maynard', 'william_singe', 'all'],
                        help='Artist to process')
    parser.add_argument('--stage', default='all',
                        choices=['download', 'separate', 'diarize', 'train', 'all'],
                        help='Pipeline stage to run')
    parser.add_argument('--max-videos', type=int, default=200,
                        help='Maximum videos to download')
    parser.add_argument('--max-workers', type=int, default=4,
                        help='Parallel download workers')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs')
    parser.add_argument('--lora-rank', type=int, default=16,
                        help='LoRA rank')
    parser.add_argument('--memory-limit', type=float, default=4.0,
                        help='Max memory per diarization chunk (GB)')
    parser.add_argument('--data-dir', type=Path, default=None,
                        help='Override the runtime data directory')

    args = parser.parse_args()
    data_dir = str(args.data_dir) if args.data_dir else None

    artists = ['conor_maynard', 'william_singe'] if args.artist == 'all' else [args.artist]
    stages = ['download', 'separate', 'diarize', 'train'] if args.stage == 'all' else [args.stage]

    for artist in artists:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {ARTIST_PROFILES[artist]['name']}")
        logger.info(f"{'='*60}\n")

        for stage in stages:
            try:
                if stage == 'download':
                    stage_download(artist, args.max_videos, args.max_workers, data_dir=data_dir)
                elif stage == 'separate':
                    stage_separate(artist, data_dir=data_dir)
                elif stage == 'diarize':
                    stage_diarize(artist, args.memory_limit, data_dir=data_dir)
                elif stage == 'train':
                    stage_train(artist, args.epochs, args.lora_rank, data_dir=data_dir)
            except Exception as e:
                logger.error(f"Stage {stage} failed for {artist}: {e}")
                if stage == 'train':
                    # Training failure is critical
                    raise
                # Other stages can continue


if __name__ == '__main__':
    main()
