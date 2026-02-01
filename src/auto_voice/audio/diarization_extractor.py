"""
Diarization-based speaker extraction for multi-artist vocal tracks.

This module extracts speaker-isolated vocal tracks from diarized audio:
- Each speaker gets a FULL-LENGTH track where they are audible and others are SILENCED
- Automatically creates voice profiles for each detected speaker
- Enables per-artist voice conversion with later remixing

Workflow:
1. Load diarization JSON (speaker segments with timestamps)
2. Load separated vocals WAV
3. For EACH speaker detected:
   - Create full-length track with only that speaker audible
   - Create/update voice profile for that speaker
   - Save to profile's training data directory
4. Expose via web interface for training
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import uuid

import numpy as np
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    """A segment of audio belonging to a single speaker."""
    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class ExtractionResult:
    """Result of extracting speaker-isolated tracks."""
    source_file: str
    total_duration: float
    speakers: Dict[str, 'SpeakerExtractionInfo']


@dataclass
class SpeakerExtractionInfo:
    """Information about a single speaker's extraction."""
    speaker_id: str
    profile_id: str
    output_file: str
    total_duration: float  # Full track duration
    speaker_duration: float  # How much this speaker actually speaks
    segment_count: int
    is_primary: bool  # Is this the main artist?


class DiarizationExtractor:
    """Extract speaker-isolated vocal tracks from diarized audio.

    For each detected speaker, creates a full-length track where:
    - That speaker's segments are audible
    - All other speakers are silenced (zero amplitude)

    This allows:
    - Training separate LoRAs for each artist
    - Converting each artist independently
    - Remixing converted tracks back together
    """

    def __init__(
        self,
        fade_ms: float = 10.0,
        min_segment_duration: float = 0.5,
        profiles_dir: Optional[Path] = None,
        training_vocals_dir: Optional[Path] = None,
    ):
        """Initialize the extractor.

        Args:
            fade_ms: Fade duration at segment boundaries (reduces clicks)
            min_segment_duration: Minimum segment duration to include
            profiles_dir: Directory for voice profiles (default: data/voice_profiles)
            training_vocals_dir: Directory for training vocals (default: data/training_vocals)
        """
        self.fade_ms = fade_ms
        self.min_segment_duration = min_segment_duration
        self.profiles_dir = profiles_dir or Path('data/voice_profiles')
        self.training_vocals_dir = training_vocals_dir or Path('data/training_vocals')

    def load_diarization(self, json_path: Path) -> Tuple[str, List[SpeakerSegment]]:
        """Load diarization results from JSON file.

        Args:
            json_path: Path to diarization JSON

        Returns:
            Tuple of (audio_file_path, list_of_segments)
        """
        with open(json_path) as f:
            data = json.load(f)

        audio_file = data['file']
        segments = [
            SpeakerSegment(
                start=s['start'],
                end=s['end'],
                speaker=s['speaker']
            )
            for s in data['segments']
        ]

        return audio_file, segments

    def get_speaker_durations(self, segments: List[SpeakerSegment]) -> Dict[str, float]:
        """Calculate total speaking duration for each speaker.

        Args:
            segments: List of speaker segments

        Returns:
            Dictionary mapping speaker_id to total duration in seconds
        """
        durations = defaultdict(float)
        for seg in segments:
            if seg.duration >= self.min_segment_duration:
                durations[seg.speaker] += seg.duration
        return dict(durations)

    def identify_primary_speaker(self, segments: List[SpeakerSegment]) -> Optional[str]:
        """Identify the primary speaker (longest total speaking time).

        Args:
            segments: List of speaker segments

        Returns:
            Speaker ID of primary speaker, or None if no segments
        """
        durations = self.get_speaker_durations(segments)
        if not durations:
            return None
        return max(durations.items(), key=lambda x: x[1])[0]

    def extract_speaker_track(
        self,
        audio: np.ndarray,
        sr: int,
        segments: List[SpeakerSegment],
        target_speaker: str,
    ) -> np.ndarray:
        """Create a full-length track with only target speaker audible.

        Args:
            audio: Full audio waveform
            sr: Sample rate
            segments: All speaker segments
            target_speaker: Speaker ID to keep audible

        Returns:
            Full-length audio with only target_speaker audible, others silenced
        """
        # Create output array (starts as silence)
        output = np.zeros_like(audio)
        fade_samples = int(self.fade_ms * sr / 1000)

        for seg in segments:
            if seg.speaker != target_speaker:
                continue

            if seg.duration < self.min_segment_duration:
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

            # Place in output
            output[start_sample:end_sample] = segment_audio

        return output

    def get_or_create_profile(
        self,
        artist_name: str,
        speaker_id: str,
        is_primary: bool,
    ) -> str:
        """Get existing profile ID or create new one for a speaker.

        Args:
            artist_name: Main artist name (e.g., "conor_maynard")
            speaker_id: Speaker ID from diarization (e.g., "SPEAKER_00")
            is_primary: Whether this is the primary/main artist

        Returns:
            Profile UUID
        """
        # Profile mapping file
        mapping_file = self.profiles_dir / artist_name / 'speaker_profiles.json'
        mapping_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing mappings
        if mapping_file.exists():
            with open(mapping_file) as f:
                mappings = json.load(f)
        else:
            mappings = {}

        # Check if speaker already has a profile
        if speaker_id in mappings:
            return mappings[speaker_id]['profile_id']

        # Create new profile
        profile_id = str(uuid.uuid4())

        # Determine profile name
        if is_primary:
            profile_name = artist_name.replace('_', ' ').title()
        else:
            # Featured artist - use speaker ID for now
            # Can be renamed via web interface later
            profile_name = f"{artist_name.replace('_', ' ').title()} - {speaker_id}"

        mappings[speaker_id] = {
            'profile_id': profile_id,
            'profile_name': profile_name,
            'is_primary': is_primary,
            'created_at': str(np.datetime64('now')),
        }

        # Save mappings
        with open(mapping_file, 'w') as f:
            json.dump(mappings, f, indent=2)

        logger.info(f"Created profile for {speaker_id}: {profile_id} ({profile_name})")

        return profile_id

    def process_track(
        self,
        diarization_json: Path,
        audio_path: Path,
        artist_name: str,
        output_dir: Optional[Path] = None,
    ) -> ExtractionResult:
        """Process a single track, extracting all speakers to separate files.

        Args:
            diarization_json: Path to diarization JSON
            audio_path: Path to separated vocals WAV
            artist_name: Main artist name (for profile organization)
            output_dir: Output directory (default: training_vocals_dir/artist_name)

        Returns:
            ExtractionResult with info about all extracted speakers
        """
        # Load diarization
        _, segments = self.load_diarization(diarization_json)

        if not segments:
            logger.warning(f"No segments in {diarization_json}")
            return ExtractionResult(
                source_file=str(audio_path),
                total_duration=0,
                speakers={},
            )

        # Load audio
        audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
        total_duration = len(audio) / sr

        # Identify speakers and their durations
        speaker_durations = self.get_speaker_durations(segments)
        primary_speaker = self.identify_primary_speaker(segments)

        # Output directory
        if output_dir is None:
            output_dir = self.training_vocals_dir

        # Process each speaker
        speakers_info = {}
        track_stem = audio_path.stem.replace('_vocals', '')

        for speaker_id, speaker_duration in speaker_durations.items():
            is_primary = (speaker_id == primary_speaker)

            # Get/create profile for this speaker
            profile_id = self.get_or_create_profile(artist_name, speaker_id, is_primary)

            # Create speaker-isolated track
            speaker_audio = self.extract_speaker_track(audio, sr, segments, speaker_id)

            # Determine output path based on profile
            if is_primary:
                # Primary artist goes to main artist directory
                speaker_output_dir = output_dir / artist_name
            else:
                # Featured artists go to their own directory
                speaker_output_dir = output_dir / 'featured' / profile_id

            speaker_output_dir.mkdir(parents=True, exist_ok=True)

            # Output filename includes track ID and speaker info
            output_filename = f"{track_stem}_{speaker_id}_isolated.wav"
            output_path = speaker_output_dir / output_filename

            # Save
            sf.write(str(output_path), speaker_audio, sr)

            # Count segments for this speaker
            segment_count = sum(
                1 for s in segments
                if s.speaker == speaker_id and s.duration >= self.min_segment_duration
            )

            speakers_info[speaker_id] = SpeakerExtractionInfo(
                speaker_id=speaker_id,
                profile_id=profile_id,
                output_file=str(output_path),
                total_duration=total_duration,
                speaker_duration=speaker_duration,
                segment_count=segment_count,
                is_primary=is_primary,
            )

            logger.debug(
                f"Extracted {speaker_id} ({speaker_duration:.1f}s / {total_duration:.1f}s) "
                f"to {output_path}"
            )

        return ExtractionResult(
            source_file=str(audio_path),
            total_duration=total_duration,
            speakers=speakers_info,
        )

    def process_artist(
        self,
        artist_name: str,
        diarization_dir: Optional[Path] = None,
        separated_dir: Optional[Path] = None,
    ) -> Dict[str, any]:
        """Process all tracks for an artist.

        Args:
            artist_name: Artist name (e.g., "conor_maynard")
            diarization_dir: Directory with diarization JSONs
            separated_dir: Directory with separated vocals WAVs

        Returns:
            Statistics dictionary
        """
        if diarization_dir is None:
            diarization_dir = Path(f'data/diarized_youtube/{artist_name}')
        if separated_dir is None:
            separated_dir = Path(f'data/separated_youtube/{artist_name}')

        if not diarization_dir.exists():
            raise FileNotFoundError(f"Diarization directory not found: {diarization_dir}")

        # Find all diarization files
        json_files = sorted(diarization_dir.glob('*_diarization.json'))
        logger.info(f"Found {len(json_files)} diarization files for {artist_name}")

        stats = {
            'total_tracks': 0,
            'processed_tracks': 0,
            'skipped_tracks': 0,
            'total_duration_minutes': 0,
            'speakers': defaultdict(lambda: {
                'total_duration_minutes': 0,
                'track_count': 0,
                'profile_id': None,
                'is_primary': False,
            }),
        }

        for json_path in json_files:
            stats['total_tracks'] += 1

            try:
                # Find corresponding audio file
                audio_stem = json_path.stem.replace('_diarization', '')
                audio_path = separated_dir / f"{audio_stem}.wav"

                if not audio_path.exists():
                    logger.warning(f"Audio not found: {audio_path}")
                    stats['skipped_tracks'] += 1
                    continue

                # Process track
                result = self.process_track(json_path, audio_path, artist_name)

                if not result.speakers:
                    stats['skipped_tracks'] += 1
                    continue

                stats['processed_tracks'] += 1
                stats['total_duration_minutes'] += result.total_duration / 60

                # Update per-speaker stats
                for speaker_id, info in result.speakers.items():
                    speaker_stats = stats['speakers'][speaker_id]
                    speaker_stats['total_duration_minutes'] += info.speaker_duration / 60
                    speaker_stats['track_count'] += 1
                    speaker_stats['profile_id'] = info.profile_id
                    speaker_stats['is_primary'] = info.is_primary

            except Exception as e:
                logger.error(f"Error processing {json_path.name}: {e}")
                stats['skipped_tracks'] += 1

        # Convert defaultdict to regular dict for JSON serialization
        stats['speakers'] = dict(stats['speakers'])

        return stats


def run_extraction(
    artists: List[str] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Dict]:
    """Run extraction for specified artists.

    Args:
        artists: List of artist names (default: both conor_maynard and william_singe)
        output_dir: Output directory for training vocals

    Returns:
        Dictionary mapping artist name to extraction stats
    """
    if artists is None:
        artists = ['conor_maynard', 'william_singe']

    extractor = DiarizationExtractor(
        training_vocals_dir=output_dir or Path('data/training_vocals'),
    )

    all_stats = {}

    for artist in artists:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {artist}")
        logger.info('='*60)

        try:
            stats = extractor.process_artist(artist)
            all_stats[artist] = stats

            # Print summary
            logger.info(f"\n{artist} Summary:")
            logger.info(f"  Processed: {stats['processed_tracks']}/{stats['total_tracks']} tracks")
            logger.info(f"  Total duration: {stats['total_duration_minutes']:.1f} minutes")
            logger.info(f"  Speakers detected:")
            for speaker_id, speaker_stats in stats['speakers'].items():
                primary_marker = " (PRIMARY)" if speaker_stats['is_primary'] else ""
                logger.info(
                    f"    {speaker_id}{primary_marker}: "
                    f"{speaker_stats['total_duration_minutes']:.1f} min "
                    f"across {speaker_stats['track_count']} tracks"
                )

        except Exception as e:
            logger.error(f"Failed to process {artist}: {e}")
            all_stats[artist] = {'error': str(e)}

    return all_stats


if __name__ == '__main__':
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    parser = argparse.ArgumentParser(
        description='Extract speaker-isolated vocals from diarized audio'
    )
    parser.add_argument(
        '--artist',
        choices=['conor_maynard', 'william_singe', 'all'],
        default='all',
        help='Artist to process'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/training_vocals'),
        help='Output directory'
    )

    args = parser.parse_args()

    artists = ['conor_maynard', 'william_singe'] if args.artist == 'all' else [args.artist]

    stats = run_extraction(artists, args.output_dir)

    # Save stats
    stats_file = args.output_dir / 'extraction_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info(f"\nStats saved to: {stats_file}")
