"""Training data filtering for speaker-specific audio extraction.

This module provides functionality to filter training audio to only include
segments from a target speaker, based on diarization results and profile matching.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.io import wavfile

from auto_voice.audio.speaker_diarization import (
    DiarizationResult,
    SpeakerDiarizer,
    SpeakerSegment,
    compute_speaker_similarity,
    match_speaker_to_profile,
)

logger = logging.getLogger(__name__)


class TrainingDataFilter:
    """Filter training audio to extract only target speaker vocals."""

    def __init__(
        self,
        diarizer: Optional[SpeakerDiarizer] = None,
        device: Optional[str] = None,
    ):
        """Initialize the training data filter.

        Args:
            diarizer: Optional pre-initialized SpeakerDiarizer.
            device: Device for diarization ('cuda' or 'cpu').
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._diarizer = diarizer

    @property
    def diarizer(self) -> SpeakerDiarizer:
        """Lazy-load the diarizer."""
        if self._diarizer is None:
            self._diarizer = SpeakerDiarizer(device=self.device)
        return self._diarizer

    def filter_training_audio(
        self,
        audio_path: Union[str, Path],
        target_embedding: np.ndarray,
        output_path: Optional[Union[str, Path]] = None,
        similarity_threshold: float = 0.7,
        min_segment_duration: float = 0.5,
        diarization_result: Optional[DiarizationResult] = None,
    ) -> Tuple[Path, Dict]:
        """Filter audio to only include segments matching target speaker.

        Args:
            audio_path: Path to the input audio file.
            target_embedding: Target speaker embedding (512-dim WavLM).
            output_path: Output path for filtered audio (auto-generated if None).
            similarity_threshold: Cosine similarity threshold for matching.
            min_segment_duration: Minimum segment duration to include.
            diarization_result: Pre-computed diarization (computed if None).

        Returns:
            Tuple of (output_path, metadata_dict) where metadata contains
            filtering statistics.
        """
        audio_path = Path(audio_path)

        # Run diarization if not provided
        if diarization_result is None:
            logger.info(f"Running diarization on {audio_path}")
            diarization_result = self.diarizer.diarize(audio_path)

        # Load original audio
        sample_rate, audio_data = wavfile.read(str(audio_path))
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0

        # Handle stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        original_duration = len(audio_data) / sample_rate

        # Find matching segments
        matching_segments = []
        non_matching_segments = []

        for segment in diarization_result.segments:
            if segment.embedding is None:
                continue

            similarity = compute_speaker_similarity(target_embedding, segment.embedding)

            if similarity >= similarity_threshold and segment.duration >= min_segment_duration:
                matching_segments.append((segment, similarity))
            else:
                non_matching_segments.append((segment, similarity))

        if not matching_segments:
            logger.warning("No matching segments found for target speaker")
            # Return empty audio or original based on preference
            metadata = {
                "original_duration": original_duration,
                "filtered_duration": 0.0,
                "num_segments": 0,
                "num_rejected": len(non_matching_segments),
                "purity": 0.0,
                "status": "no_match",
            }
            # Create empty output
            if output_path is None:
                output_path = Path(tempfile.mkdtemp()) / f"{audio_path.stem}_filtered.wav"
            else:
                output_path = Path(output_path)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Write silent audio
            silence = np.zeros(int(0.1 * sample_rate), dtype=np.int16)
            wavfile.write(str(output_path), sample_rate, silence)
            return output_path, metadata

        # Sort by start time
        matching_segments.sort(key=lambda x: x[0].start)

        # Extract and concatenate matching segments
        extracted_parts = []
        total_extracted_duration = 0.0

        for segment, similarity in matching_segments:
            start_sample = int(segment.start * sample_rate)
            end_sample = int(segment.end * sample_rate)

            # Bounds checking
            start_sample = max(0, start_sample)
            end_sample = min(len(audio_data), end_sample)

            if end_sample > start_sample:
                extracted_parts.append(audio_data[start_sample:end_sample])
                total_extracted_duration += (end_sample - start_sample) / sample_rate

        # Concatenate with small crossfade
        if len(extracted_parts) == 1:
            combined = extracted_parts[0]
        else:
            # Add 10ms silence gaps between segments
            gap_samples = int(0.01 * sample_rate)
            silence = np.zeros(gap_samples, dtype=audio_data.dtype)

            combined_parts = []
            for i, part in enumerate(extracted_parts):
                combined_parts.append(part)
                if i < len(extracted_parts) - 1:
                    combined_parts.append(silence)

            combined = np.concatenate(combined_parts)

        # Generate output path
        if output_path is None:
            output_path = Path(tempfile.mkdtemp()) / f"{audio_path.stem}_filtered.wav"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert back to int16 for saving
        combined_int16 = (combined * 32767).astype(np.int16)
        wavfile.write(str(output_path), sample_rate, combined_int16)

        # Calculate purity (ratio of matching to total detected speech)
        total_speech_duration = sum(s.duration for s in diarization_result.segments)
        purity = total_extracted_duration / total_speech_duration if total_speech_duration > 0 else 0.0

        metadata = {
            "original_duration": original_duration,
            "filtered_duration": total_extracted_duration,
            "num_segments": len(matching_segments),
            "num_rejected": len(non_matching_segments),
            "purity": purity,
            "average_similarity": np.mean([s[1] for s in matching_segments]),
            "status": "success",
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "similarity": sim,
                }
                for seg, sim in matching_segments
            ],
        }

        logger.info(
            f"Filtered {audio_path.name}: {len(matching_segments)} segments, "
            f"{total_extracted_duration:.1f}s extracted ({purity*100:.1f}% purity)"
        )

        return output_path, metadata

    def filter_with_profile_matching(
        self,
        audio_path: Union[str, Path],
        profile_embeddings: Dict[str, np.ndarray],
        target_profile_id: str,
        output_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Tuple[Path, Dict]:
        """Filter audio to match a specific profile from a set of profiles.

        Args:
            audio_path: Path to the input audio file.
            profile_embeddings: Dict mapping profile_id to embedding.
            target_profile_id: Profile ID to extract.
            output_path: Output path (auto-generated if None).
            **kwargs: Additional arguments for filter_training_audio.

        Returns:
            Tuple of (output_path, metadata_dict).
        """
        if target_profile_id not in profile_embeddings:
            raise ValueError(f"Target profile {target_profile_id} not in embeddings dict")

        target_embedding = profile_embeddings[target_profile_id]
        return self.filter_training_audio(
            audio_path=audio_path,
            target_embedding=target_embedding,
            output_path=output_path,
            **kwargs,
        )

    def auto_split_by_speakers(
        self,
        audio_path: Union[str, Path],
        output_dir: Union[str, Path],
        diarization_result: Optional[DiarizationResult] = None,
        min_segment_duration: float = 0.5,
    ) -> Dict[str, Tuple[Path, float]]:
        """Split audio into separate files per detected speaker.

        Args:
            audio_path: Path to the input audio file.
            output_dir: Directory for output files.
            diarization_result: Pre-computed diarization (computed if None).
            min_segment_duration: Minimum segment duration to include.

        Returns:
            Dict mapping speaker_id to (output_path, total_duration).
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run diarization if not provided
        if diarization_result is None:
            diarization_result = self.diarizer.diarize(audio_path)

        # Load original audio
        sample_rate, audio_data = wavfile.read(str(audio_path))
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0

        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Group segments by speaker
        speaker_segments: Dict[str, List[SpeakerSegment]] = {}
        for segment in diarization_result.segments:
            if segment.duration >= min_segment_duration:
                if segment.speaker_id not in speaker_segments:
                    speaker_segments[segment.speaker_id] = []
                speaker_segments[segment.speaker_id].append(segment)

        # Extract audio for each speaker
        results = {}
        for speaker_id, segments in speaker_segments.items():
            # Sort by start time
            segments.sort(key=lambda s: s.start)

            # Extract segments
            extracted_parts = []
            total_duration = 0.0

            for segment in segments:
                start_sample = int(segment.start * sample_rate)
                end_sample = int(segment.end * sample_rate)
                start_sample = max(0, start_sample)
                end_sample = min(len(audio_data), end_sample)

                if end_sample > start_sample:
                    extracted_parts.append(audio_data[start_sample:end_sample])
                    total_duration += (end_sample - start_sample) / sample_rate

            if extracted_parts:
                # Concatenate with small gaps
                gap_samples = int(0.01 * sample_rate)
                silence = np.zeros(gap_samples)

                combined_parts = []
                for i, part in enumerate(extracted_parts):
                    combined_parts.append(part)
                    if i < len(extracted_parts) - 1:
                        combined_parts.append(silence)

                combined = np.concatenate(combined_parts)

                # Save
                output_path = output_dir / f"{audio_path.stem}_{speaker_id}.wav"
                combined_int16 = (combined * 32767).astype(np.int16)
                wavfile.write(str(output_path), sample_rate, combined_int16)

                results[speaker_id] = (output_path, total_duration)

                logger.info(f"Extracted {speaker_id}: {total_duration:.1f}s to {output_path}")

        return results


def filter_training_audio(
    audio_path: Union[str, Path],
    target_embedding: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Tuple[Path, Dict]:
    """Convenience function for filtering training audio.

    See TrainingDataFilter.filter_training_audio for full documentation.
    """
    filter_obj = TrainingDataFilter()
    return filter_obj.filter_training_audio(
        audio_path=audio_path,
        target_embedding=target_embedding,
        output_path=output_path,
        **kwargs,
    )
