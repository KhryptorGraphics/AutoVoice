"""Multi-Artist Separation and Profile Routing.

Phase 5 of LoRA Lifecycle Management:
- Demucs vocal separation
- Pyannote/WavLM diarization for speaker segments
- WavLM embedding extraction per segment
- Cluster by speaker similarity (0.85 threshold)
- Match clusters to known profiles
- Create profiles for unknown artists

Cross-Context Dependencies:
- speaker-diarization_20260130: WavLM embeddings (256-dim), speaker diarization
- training-inference-integration_20260130: AdapterManager, JobManager
- voice-profile-training_20260124: VoiceProfileStore
- sota-dual-pipeline_20260130: Demucs separation

Ultimate Goal:
Voice-to-voice conversion where one artist sings another's song EXACTLY as the
original artist sang it - pitch correct, singing abilities matched, synced to instrumental.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class ArtistSegment:
    """A segment of audio belonging to a specific artist."""
    profile_id: str
    profile_name: str
    start: float
    end: float
    audio: np.ndarray
    sample_rate: int
    embedding: np.ndarray
    similarity: float
    is_new_profile: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class SeparationResult:
    """Result of multi-artist separation."""
    artists: Dict[str, List[ArtistSegment]]  # profile_id -> segments
    vocals: np.ndarray
    instrumental: np.ndarray
    sample_rate: int
    total_duration: float
    num_artists: int
    new_profiles_created: List[str] = field(default_factory=list)


class MultiArtistSeparator:
    """Separates multi-artist tracks and routes to voice profiles.

    Pipeline:
    1. Demucs vocal/instrumental separation
    2. WavLM speaker diarization
    3. Cluster embeddings by similarity
    4. Match to existing profiles or create new ones
    5. Return segments organized by artist

    Thresholds (from lora-lifecycle-management spec):
    - speaker_similarity_min: 0.85
    - min_samples_for_training: 5
    """

    SIMILARITY_THRESHOLD = 0.85
    MIN_SEGMENT_DURATION = 1.0  # Minimum segment duration in seconds

    def __init__(
        self,
        profiles_dir: Path = Path("data/voice_profiles"),
        device: str = "cuda",
        auto_create_profiles: bool = True,
        auto_queue_training: bool = True,
    ):
        """Initialize the multi-artist separator.

        Args:
            profiles_dir: Directory for voice profiles
            device: Device for inference ('cuda' or 'cpu')
            auto_create_profiles: Automatically create profiles for unknown speakers
            auto_queue_training: Automatically queue training when thresholds met
        """
        self.profiles_dir = Path(profiles_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.auto_create_profiles = auto_create_profiles
        self.auto_queue_training = auto_queue_training

        # Lazy-loaded components
        self._separator = None
        self._diarizer = None
        self._identifier = None
        self._job_manager = None

        logger.info(f"MultiArtistSeparator initialized on {self.device}")

    def _load_separator(self):
        """Lazy load Demucs vocal separator."""
        if self._separator is None:
            from auto_voice.audio.separation import VocalSeparator
            self._separator = VocalSeparator(device=self.device)
            logger.info("Loaded VocalSeparator (Demucs)")

    def _load_diarizer(self):
        """Lazy load speaker diarizer."""
        if self._diarizer is None:
            from auto_voice.audio.speaker_diarization import SpeakerDiarizer
            self._diarizer = SpeakerDiarizer(device=str(self.device))
            logger.info("Loaded SpeakerDiarizer (WavLM)")

    def _load_identifier(self):
        """Lazy load voice identifier."""
        if self._identifier is None:
            from auto_voice.inference.voice_identifier import VoiceIdentifier
            self._identifier = VoiceIdentifier(
                profiles_dir=self.profiles_dir,
                device=str(self.device),
            )
            self._identifier.load_all_embeddings()
            logger.info(f"Loaded VoiceIdentifier with {len(self._identifier._embeddings)} profiles")

    def _load_job_manager(self):
        """Lazy load training job manager."""
        if self._job_manager is None:
            try:
                from auto_voice.training.job_manager import TrainingJobManager
                self._job_manager = TrainingJobManager(
                    storage_path=Path("data/training_jobs"),
                    require_gpu=False,  # Don't require GPU for queueing
                )
                logger.info("Loaded TrainingJobManager")
            except Exception as e:
                logger.warning(f"Could not load JobManager: {e}")

    def separate_vocals(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Separate vocals from instrumental.

        Args:
            audio: Input audio (mono or stereo)
            sample_rate: Sample rate

        Returns:
            Tuple of (vocals, instrumental) numpy arrays
        """
        self._load_separator()

        result = self._separator.separate(audio, sample_rate)
        return result['vocals'], result['instrumental']

    def diarize_vocals(
        self,
        vocals: np.ndarray,
        sample_rate: int,
        num_speakers: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Perform speaker diarization on vocals.

        Args:
            vocals: Vocal audio
            sample_rate: Sample rate
            num_speakers: Expected number of speakers (optional)

        Returns:
            List of segment dicts with start, end, speaker_id, embedding
        """
        self._load_diarizer()

        import tempfile
        import soundfile as sf

        # Save vocals to temp file for diarizer
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, vocals, sample_rate)

        try:
            result = self._diarizer.diarize(
                temp_path,
                num_speakers=num_speakers,
            )

            # Convert to segment dicts
            segments = []
            for seg in result.segments:
                segments.append({
                    'start': seg.start,
                    'end': seg.end,
                    'speaker_id': seg.speaker_id,
                    'embedding': seg.embedding,
                    'duration': seg.duration,
                })

            return segments

        finally:
            import os
            os.unlink(temp_path)

    def match_segments_to_profiles(
        self,
        segments: List[Dict[str, Any]],
        vocals: np.ndarray,
        sample_rate: int,
        youtube_metadata: Optional[Dict[str, Any]] = None,
        source_file: Optional[str] = None,
    ) -> Dict[str, List[ArtistSegment]]:
        """Match diarization segments to voice profiles.

        Args:
            segments: Diarization segments with embeddings
            vocals: Full vocal audio
            sample_rate: Sample rate
            youtube_metadata: Optional YouTube metadata for naming
            source_file: Optional source filename

        Returns:
            Dict mapping profile_id to list of ArtistSegments
        """
        self._load_identifier()

        artists: Dict[str, List[ArtistSegment]] = {}
        new_profiles: List[str] = []

        for seg in segments:
            if seg['duration'] < self.MIN_SEGMENT_DURATION:
                continue

            # Extract segment audio
            start_sample = int(seg['start'] * sample_rate)
            end_sample = int(seg['end'] * sample_rate)
            segment_audio = vocals[start_sample:end_sample]

            # Get embedding (use existing or extract new)
            embedding = seg.get('embedding')
            if embedding is None:
                embedding = self._identifier.extract_embedding(segment_audio, sample_rate)

            # Identify or create profile
            if self.auto_create_profiles:
                result = self._identifier.identify_or_create(
                    audio=segment_audio,
                    sample_rate=sample_rate,
                    threshold=self.SIMILARITY_THRESHOLD,
                    youtube_metadata=youtube_metadata,
                    source_file=source_file,
                )
                is_new = result.similarity == 1.0 and result.profile_id not in new_profiles
                if is_new and result.profile_id:
                    new_profiles.append(result.profile_id)
            else:
                result = self._identifier.identify(
                    audio=segment_audio,
                    sample_rate=sample_rate,
                    threshold=self.SIMILARITY_THRESHOLD,
                )
                is_new = False

            if result.profile_id:
                artist_segment = ArtistSegment(
                    profile_id=result.profile_id,
                    profile_name=result.profile_name or result.profile_id,
                    start=seg['start'],
                    end=seg['end'],
                    audio=segment_audio,
                    sample_rate=sample_rate,
                    embedding=embedding,
                    similarity=result.similarity,
                    is_new_profile=is_new,
                )

                if result.profile_id not in artists:
                    artists[result.profile_id] = []
                artists[result.profile_id].append(artist_segment)

        return artists, new_profiles

    def separate_and_route(
        self,
        audio: np.ndarray,
        sample_rate: int,
        num_speakers: Optional[int] = None,
        youtube_metadata: Optional[Dict[str, Any]] = None,
        source_file: Optional[str] = None,
    ) -> SeparationResult:
        """Separate multi-artist audio and route to profiles.

        Full pipeline:
        1. Demucs vocal/instrumental separation
        2. WavLM speaker diarization
        3. Match segments to profiles (create new if needed)
        4. Return organized result

        Args:
            audio: Input audio (mono or stereo)
            sample_rate: Sample rate
            num_speakers: Expected number of speakers (optional)
            youtube_metadata: Optional YouTube metadata for naming
            source_file: Optional source filename

        Returns:
            SeparationResult with organized artist segments
        """
        logger.info(f"Starting multi-artist separation ({len(audio) / sample_rate:.1f}s audio)")

        # Step 1: Vocal/instrumental separation
        vocals, instrumental = self.separate_vocals(audio, sample_rate)
        logger.info("Vocal separation complete")

        # Step 2: Speaker diarization
        segments = self.diarize_vocals(vocals, sample_rate, num_speakers)
        logger.info(f"Diarization complete: {len(segments)} segments")

        # Step 3: Match to profiles
        artists, new_profiles = self.match_segments_to_profiles(
            segments=segments,
            vocals=vocals,
            sample_rate=sample_rate,
            youtube_metadata=youtube_metadata,
            source_file=source_file,
        )
        logger.info(f"Matched to {len(artists)} artists, {len(new_profiles)} new profiles")

        # Step 4: Auto-queue training if enabled
        if self.auto_queue_training and new_profiles:
            self._queue_training_for_profiles(artists)

        return SeparationResult(
            artists=artists,
            vocals=vocals,
            instrumental=instrumental,
            sample_rate=sample_rate,
            total_duration=len(audio) / sample_rate,
            num_artists=len(artists),
            new_profiles_created=new_profiles,
        )

    def _queue_training_for_profiles(
        self,
        artists: Dict[str, List[ArtistSegment]],
    ) -> None:
        """Queue training for profiles that meet threshold.

        Args:
            artists: Dict of profile_id -> segments
        """
        self._load_job_manager()

        if self._job_manager is None:
            logger.warning("JobManager not available, skipping auto-queue")
            return

        for profile_id, segments in artists.items():
            # Check if enough samples for training
            total_duration = sum(s.duration for s in segments)
            if total_duration >= 30.0:  # At least 30 seconds of audio
                try:
                    job = self._job_manager.auto_queue_training(
                        profile_id=profile_id,
                        min_samples=5,
                    )
                    if job:
                        logger.info(f"Queued training job {job.job_id} for {profile_id}")
                except Exception as e:
                    logger.warning(f"Failed to queue training for {profile_id}: {e}")

    def process_file(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        youtube_metadata: Optional[Dict[str, Any]] = None,
    ) -> SeparationResult:
        """Process a single audio file.

        Args:
            audio_path: Path to audio file
            num_speakers: Expected number of speakers
            youtube_metadata: Optional YouTube metadata

        Returns:
            SeparationResult with organized artist segments
        """
        import torchaudio

        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

        audio = waveform.numpy()

        return self.separate_and_route(
            audio=audio,
            sample_rate=sample_rate,
            num_speakers=num_speakers,
            youtube_metadata=youtube_metadata,
            source_file=audio_path,
        )

    def process_batch(
        self,
        audio_files: List[str],
        num_speakers: Optional[int] = None,
        aggregate_by_artist: bool = True,
    ) -> Dict[str, Any]:
        """Process multiple audio files (e.g., an album).

        Args:
            audio_files: List of audio file paths
            num_speakers: Expected number of speakers per file
            aggregate_by_artist: If True, aggregate samples per artist

        Returns:
            Dict with per-file results and aggregated artist info
        """
        results = []
        all_artists: Dict[str, List[ArtistSegment]] = {}

        for audio_path in audio_files:
            logger.info(f"Processing: {audio_path}")
            try:
                result = self.process_file(
                    audio_path=audio_path,
                    num_speakers=num_speakers,
                )
                results.append({
                    'file': audio_path,
                    'success': True,
                    'num_artists': result.num_artists,
                    'new_profiles': result.new_profiles_created,
                })

                # Aggregate by artist
                if aggregate_by_artist:
                    for profile_id, segments in result.artists.items():
                        if profile_id not in all_artists:
                            all_artists[profile_id] = []
                        all_artists[profile_id].extend(segments)

            except Exception as e:
                logger.error(f"Failed to process {audio_path}: {e}")
                results.append({
                    'file': audio_path,
                    'success': False,
                    'error': str(e),
                })

        # Generate artist summary
        artist_summary = {}
        for profile_id, segments in all_artists.items():
            total_duration = sum(s.duration for s in segments)
            artist_summary[profile_id] = {
                'profile_name': segments[0].profile_name if segments else profile_id,
                'total_segments': len(segments),
                'total_duration': total_duration,
                'files_appeared_in': len(set(
                    s.audio.tobytes()[:100] for s in segments  # Rough unique file check
                )),
            }

        # Queue training for artists with enough data
        if self.auto_queue_training:
            for profile_id, summary in artist_summary.items():
                if summary['total_duration'] >= 60.0:  # At least 1 minute
                    self._queue_training_for_profiles({
                        profile_id: all_artists[profile_id]
                    })

        return {
            'files_processed': len(audio_files),
            'files_successful': sum(1 for r in results if r.get('success')),
            'file_results': results,
            'artists_found': len(all_artists),
            'artist_summary': artist_summary,
        }

    def save_artist_segments(
        self,
        result: SeparationResult,
        output_dir: Path,
    ) -> Dict[str, List[str]]:
        """Save separated artist segments to files.

        Args:
            result: Separation result
            output_dir: Output directory

        Returns:
            Dict mapping profile_id to list of saved file paths
        """
        import soundfile as sf

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files: Dict[str, List[str]] = {}

        for profile_id, segments in result.artists.items():
            profile_dir = output_dir / profile_id
            profile_dir.mkdir(parents=True, exist_ok=True)

            saved_files[profile_id] = []

            for i, seg in enumerate(segments):
                filename = f"segment_{i:04d}_{seg.start:.1f}s-{seg.end:.1f}s.wav"
                filepath = profile_dir / filename
                sf.write(filepath, seg.audio, seg.sample_rate)
                saved_files[profile_id].append(str(filepath))

        # Also save instrumental
        instrumental_path = output_dir / "instrumental.wav"
        sf.write(instrumental_path, result.instrumental, result.sample_rate)

        logger.info(f"Saved segments for {len(saved_files)} artists to {output_dir}")
        return saved_files


# Global instance
_global_separator: Optional[MultiArtistSeparator] = None


def get_multi_artist_separator() -> MultiArtistSeparator:
    """Get or create global MultiArtistSeparator instance."""
    global _global_separator
    if _global_separator is None:
        _global_separator = MultiArtistSeparator()
    return _global_separator
