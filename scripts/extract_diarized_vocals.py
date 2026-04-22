#!/usr/bin/env python3
"""
Extract speaker-specific vocals from diarization results.

Reads diarization JSON files and creates separate WAV files for each speaker,
identifying the primary artist (longest total speaking time) as the target.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import librosa
import soundfile as sf
from dataclasses import dataclass
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / 'data'
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from auto_voice.storage.paths import (
    resolve_data_dir,
    resolve_diarized_audio_dir,
    resolve_separated_audio_dir,
    resolve_training_vocals_dir,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Segment:
    start: float
    end: float
    speaker: str


def resolve_runtime_paths(data_dir: str | None = None) -> dict[str, Path]:
    """Resolve runtime directories for this script."""
    resolved_data_dir = resolve_data_dir(
        data_dir or os.environ.get('DATA_DIR') or str(DEFAULT_DATA_DIR)
    )
    return {
        'data_dir': resolved_data_dir,
        'diarized_root': resolve_diarized_audio_dir(data_dir=str(resolved_data_dir)),
        'separated_root': resolve_separated_audio_dir(data_dir=str(resolved_data_dir)),
        'training_vocals_dir': resolve_training_vocals_dir(data_dir=str(resolved_data_dir)),
    }


def load_diarization(json_path: Path) -> Tuple[str, List[Segment]]:
    """Load diarization results from JSON."""
    with open(json_path) as f:
        data = json.load(f)

    audio_file = data['file']
    segments = [Segment(**s) for s in data['segments']]
    return audio_file, segments


def identify_primary_speaker(segments: List[Segment]) -> str:
    """Identify the primary speaker (most total speaking time)."""
    speaker_duration = defaultdict(float)
    for seg in segments:
        speaker_duration[seg.speaker] += seg.end - seg.start

    if not speaker_duration:
        return None

    primary = max(speaker_duration.items(), key=lambda x: x[1])
    return primary[0]


def extract_speaker_audio(
    audio: np.ndarray,
    sr: int,
    segments: List[Segment],
    speaker: str,
    fade_ms: float = 10.0,
) -> np.ndarray:
    """Extract audio for a specific speaker with crossfade."""
    # Create output array
    output = np.zeros_like(audio)
    fade_samples = int(fade_ms * sr / 1000)

    for seg in segments:
        if seg.speaker != speaker:
            continue

        start_sample = int(seg.start * sr)
        end_sample = int(seg.end * sr)

        # Clamp to valid range
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)

        if end_sample <= start_sample:
            continue

        # Copy segment
        segment_audio = audio[start_sample:end_sample].copy()

        # Apply fade in/out to reduce clicks
        if len(segment_audio) > 2 * fade_samples:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            segment_audio[:fade_samples] *= fade_in
            segment_audio[-fade_samples:] *= fade_out

        # Add to output (with overlap handling)
        output[start_sample:end_sample] = np.maximum(
            output[start_sample:end_sample],
            segment_audio
        )

    return output


def process_artist(
    artist_name: str,
    diarization_dir: Path,
    separated_dir: Path,
    output_dir: Path,
) -> Dict[str, float]:
    """Process all tracks for an artist."""

    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        'total_files': 0,
        'total_duration_minutes': 0,
        'primary_speaker_duration_minutes': 0,
        'skipped_files': 0,
    }

    # Find all diarization JSON files
    json_files = list(diarization_dir.glob('*_diarization.json'))
    logger.info(f"Found {len(json_files)} diarization files for {artist_name}")

    for json_path in sorted(json_files):
        try:
            # Load diarization
            audio_file, segments = load_diarization(json_path)

            # Check if audio file exists
            audio_path = Path(audio_file)
            if not audio_path.exists():
                # Try relative to separated_dir
                audio_path = separated_dir / audio_path.name

            if not audio_path.exists():
                logger.warning(f"Audio file not found: {audio_file}")
                stats['skipped_files'] += 1
                continue

            # Identify primary speaker
            primary_speaker = identify_primary_speaker(segments)
            if not primary_speaker:
                logger.warning(f"No speakers found in {json_path.name}")
                stats['skipped_files'] += 1
                continue

            # Load audio
            audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
            total_duration = len(audio) / sr

            # Extract primary speaker audio
            primary_audio = extract_speaker_audio(audio, sr, segments, primary_speaker)

            # Calculate primary speaker duration
            primary_duration = sum(
                seg.end - seg.start
                for seg in segments
                if seg.speaker == primary_speaker
            )

            # Save
            output_name = audio_path.stem.replace('_vocals', '') + '_primary.wav'
            output_path = output_dir / output_name
            sf.write(str(output_path), primary_audio, sr)

            stats['total_files'] += 1
            stats['total_duration_minutes'] += total_duration / 60
            stats['primary_speaker_duration_minutes'] += primary_duration / 60

            logger.debug(f"Processed {audio_path.name}: {primary_duration/60:.1f} min primary")

        except Exception as e:
            logger.error(f"Error processing {json_path.name}: {e}")
            stats['skipped_files'] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description='Extract diarized vocals')
    parser.add_argument('--artist', choices=['conor_maynard', 'william_singe', 'all'],
                        default='all', help='Artist to process')
    parser.add_argument('--data-dir', type=Path, default=None,
                        help='Override the runtime data directory')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output directory for extracted vocals')
    args = parser.parse_args()
    data_dir = str(args.data_dir) if args.data_dir else None
    runtime_paths = resolve_runtime_paths(data_dir)
    output_root = args.output_dir or runtime_paths['training_vocals_dir']

    artists = ['conor_maynard', 'william_singe'] if args.artist == 'all' else [args.artist]

    for artist in artists:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {artist}")
        logger.info('='*60)

        diarization_dir = runtime_paths['diarized_root'] / artist
        separated_dir = runtime_paths['separated_root'] / artist
        output_dir = output_root / artist

        if not diarization_dir.exists():
            logger.error(f"Diarization directory not found: {diarization_dir}")
            continue

        stats = process_artist(artist, diarization_dir, separated_dir, output_dir)

        logger.info(f"\n{artist} Summary:")
        logger.info(f"  Total files processed: {stats['total_files']}")
        logger.info(f"  Total audio duration: {stats['total_duration_minutes']:.1f} minutes")
        logger.info(f"  Primary speaker duration: {stats['primary_speaker_duration_minutes']:.1f} minutes")
        logger.info(f"  Skipped files: {stats['skipped_files']}")
        logger.info(f"  Output directory: {output_dir}")


if __name__ == '__main__':
    main()
