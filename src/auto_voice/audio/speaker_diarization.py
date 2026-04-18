"""Speaker diarization for multi-speaker audio segmentation.

This module provides speaker diarization capabilities using:
- WavLM for speaker embeddings (512-dim from wavlm-base-sv)
- Agglomerative clustering for speaker segmentation
- Energy-based Voice Activity Detection (VAD)

The pipeline identifies different speakers in audio and extracts their segments.
"""

import gc
import logging
import psutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import tempfile

import numpy as np
import torch
import torchaudio
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

logger = logging.getLogger(__name__)


def get_available_memory_gb() -> float:
    """Get available system memory in GB."""
    return psutil.virtual_memory().available / (1024 ** 3)


def get_gpu_memory_gb() -> Tuple[float, float]:
    """Get GPU memory (used, total) in GB. Returns (0, 0) if no GPU."""
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return used, total
    return 0.0, 0.0


@dataclass
class SpeakerSegment:
    """A segment of audio belonging to a single speaker."""
    start: float  # Start time in seconds
    end: float  # End time in seconds
    speaker_id: str  # Speaker identifier (e.g., "SPEAKER_00")
    embedding: Optional[np.ndarray] = None  # Speaker embedding for this segment
    confidence: float = 1.0  # Confidence score (0-1)

    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return self.end - self.start


@dataclass
class DiarizationResult:
    """Complete diarization result for an audio file."""
    segments: List[SpeakerSegment]
    num_speakers: int
    audio_duration: float
    speaker_embeddings: Dict[str, np.ndarray] = field(default_factory=dict)

    def get_speaker_segments(self, speaker_id: str) -> List[SpeakerSegment]:
        """Get all segments for a specific speaker."""
        return [s for s in self.segments if s.speaker_id == speaker_id]

    def get_speaker_total_duration(self, speaker_id: str) -> float:
        """Get total speaking duration for a speaker."""
        return sum(s.duration for s in self.get_speaker_segments(speaker_id))

    def get_all_speaker_ids(self) -> List[str]:
        """Get list of all unique speaker IDs."""
        return list(set(s.speaker_id for s in self.segments))


class SpeakerDiarizer:
    """Speaker diarization using WavLM embeddings and clustering.

    This provides a simpler alternative to pyannote.audio that works with
    the latest torchaudio versions on Jetson platforms.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        model_name: str = "microsoft/wavlm-base-sv",
        min_segment_duration: float = 0.5,
        max_speakers: int = 10,
        max_memory_gb: Optional[float] = None,
        chunk_duration_sec: float = 60.0,
    ):
        """Initialize the speaker diarizer.

        Args:
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
            model_name: HuggingFace model for speaker embeddings.
            min_segment_duration: Minimum segment duration in seconds.
            max_speakers: Maximum number of speakers to detect.
            max_memory_gb: Maximum memory to use (None = auto 80% of available).
            chunk_duration_sec: Duration of audio chunks for memory-safe processing.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.min_segment_duration = min_segment_duration
        self.max_speakers = max_speakers
        self.chunk_duration_sec = chunk_duration_sec

        # Memory management
        if max_memory_gb is None:
            self.max_memory_gb = get_available_memory_gb() * 0.8
        else:
            self.max_memory_gb = max_memory_gb

        logger.info(f"Memory limit set to {self.max_memory_gb:.1f} GB")

        # Lazy load model
        self._model = None
        self._feature_extractor = None

        logger.info(f"SpeakerDiarizer initialized on {self.device}")

    def _load_model(self):
        """Lazy load the speaker embedding model."""
        if self._model is None:
            from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

            logger.info(f"Loading speaker embedding model: {self.model_name}")
            self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
            self._model = WavLMForXVector.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
            logger.info("Speaker embedding model loaded")

    def _check_memory(self, warn_threshold: float = 0.9) -> bool:
        """Check if memory usage is within limits.

        Args:
            warn_threshold: Fraction of max_memory_gb to warn at.

        Returns:
            True if memory is OK, False if approaching limit.
        """
        available = get_available_memory_gb()
        if available < self.max_memory_gb * (1 - warn_threshold):
            logger.warning(f"Low memory: {available:.1f} GB available")
            return False
        return True

    def _cleanup_memory(self):
        """Force garbage collection and clear CUDA cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("Memory cleanup completed")

    def _get_audio_chunks(
        self,
        audio_duration: float,
        chunk_duration: Optional[float] = None,
    ) -> List[Tuple[float, float]]:
        """Split audio into memory-safe chunks.

        Args:
            audio_duration: Total audio duration in seconds.
            chunk_duration: Chunk duration (uses self.chunk_duration_sec if None).

        Returns:
            List of (start_time, end_time) tuples for each chunk.
        """
        chunk_dur = chunk_duration or self.chunk_duration_sec
        chunks = []
        start = 0.0

        while start < audio_duration:
            end = min(start + chunk_dur, audio_duration)
            if end - start >= self.min_segment_duration:
                chunks.append((start, end))
            start = end

        logger.info(f"Split {audio_duration:.1f}s audio into {len(chunks)} chunks")
        return chunks

    def _load_audio(self, audio_path: Union[str, Path]) -> Tuple[torch.Tensor, int]:
        """Load audio file and resample to 16kHz.

        Args:
            audio_path: Path to audio file.

        Returns:
            Tuple of (waveform, sample_rate).
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Try scipy first (more compatible), fall back to torchaudio
        try:
            from scipy.io import wavfile
            sample_rate, waveform_np = wavfile.read(str(audio_path))

            # Convert to float32
            if waveform_np.dtype == np.int16:
                waveform_np = waveform_np.astype(np.float32) / 32768.0
            elif waveform_np.dtype == np.int32:
                waveform_np = waveform_np.astype(np.float32) / 2147483648.0
            elif waveform_np.dtype == np.float64:
                waveform_np = waveform_np.astype(np.float32)

            # Handle stereo
            if len(waveform_np.shape) > 1:
                waveform_np = waveform_np.mean(axis=1)

            waveform = torch.from_numpy(waveform_np)

        except Exception:
            # Fall back to torchaudio
            waveform, sample_rate = torchaudio.load(str(audio_path))
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.squeeze(0)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform.unsqueeze(0)).squeeze(0)
            sample_rate = 16000

        return waveform, sample_rate

    def _detect_voice_activity(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
        frame_duration: float = 0.025,
        energy_threshold: float = 0.02,
        min_speech_duration: float = 0.3,
    ) -> List[Tuple[float, float]]:
        """Simple energy-based voice activity detection.

        Args:
            waveform: Audio waveform tensor.
            sample_rate: Sample rate in Hz.
            frame_duration: Frame duration in seconds.
            energy_threshold: Energy threshold for speech detection.
            min_speech_duration: Minimum speech segment duration.

        Returns:
            List of (start, end) tuples for speech regions.
        """
        frame_size = int(frame_duration * sample_rate)
        hop_size = frame_size // 2

        # Calculate frame energies
        num_frames = (len(waveform) - frame_size) // hop_size + 1
        energies = []

        waveform_np = waveform.numpy()
        for i in range(num_frames):
            start_idx = i * hop_size
            end_idx = start_idx + frame_size
            frame = waveform_np[start_idx:end_idx]
            energy = np.sqrt(np.mean(frame ** 2))
            energies.append(energy)

        energies = np.array(energies)

        # Normalize energies
        if energies.max() > 0:
            energies = energies / energies.max()

        # Find speech regions
        is_speech = energies > energy_threshold
        speech_regions = []
        in_speech = False
        start_frame = 0

        for i, speech in enumerate(is_speech):
            if speech and not in_speech:
                start_frame = i
                in_speech = True
            elif not speech and in_speech:
                end_frame = i
                start_time = start_frame * hop_size / sample_rate
                end_time = end_frame * hop_size / sample_rate
                if end_time - start_time >= min_speech_duration:
                    speech_regions.append((start_time, end_time))
                in_speech = False

        # Handle case where speech continues to end
        if in_speech:
            end_time = len(waveform) / sample_rate
            start_time = start_frame * hop_size / sample_rate
            if end_time - start_time >= min_speech_duration:
                speech_regions.append((start_time, end_time))

        return speech_regions

    def extract_speaker_embedding(
        self,
        audio_path: Union[str, Path],
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> np.ndarray:
        """Extract speaker embedding from audio segment.

        Args:
            audio_path: Path to audio file.
            start: Start time in seconds (optional).
            end: End time in seconds (optional).

        Returns:
            Speaker embedding as numpy array (512-dim for WavLM-base-sv).
        """
        self._load_model()

        waveform, sample_rate = self._load_audio(audio_path)

        # Extract segment if start/end specified
        if start is not None:
            start_sample = int(start * sample_rate)
            waveform = waveform[start_sample:]
        if end is not None:
            end_sample = int((end - (start or 0)) * sample_rate)
            waveform = waveform[:end_sample]

        # Process through feature extractor
        inputs = self._feature_extractor(
            waveform.numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract embedding
        with torch.no_grad():
            outputs = self._model(**inputs)
            embedding = outputs.embeddings.cpu().numpy().squeeze()

        # L2 normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding

    def _segment_audio_fixed(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        segment_duration: float = 1.5,
        overlap: float = 0.5,
    ) -> List[Tuple[float, float]]:
        """Segment audio into fixed-duration overlapping windows.

        Args:
            waveform: Audio waveform.
            sample_rate: Sample rate.
            segment_duration: Duration of each segment in seconds.
            overlap: Overlap ratio between segments (0-1).

        Returns:
            List of (start, end) tuples.
        """
        audio_duration = len(waveform) / sample_rate
        hop_duration = segment_duration * (1 - overlap)

        segments = []
        start = 0.0

        while start + self.min_segment_duration <= audio_duration:
            end = min(start + segment_duration, audio_duration)
            if end - start >= self.min_segment_duration:
                segments.append((start, end))
            start += hop_duration

        return segments

    def _extract_segment_embeddings(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        segments: List[Tuple[float, float]],
        batch_size: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Extract embeddings for multiple segments with memory-safe batching.

        Args:
            waveform: Audio waveform.
            sample_rate: Sample rate.
            segments: List of (start, end) tuples.
            batch_size: Number of segments to process before memory cleanup.

        Returns:
            List of embeddings.
        """
        self._load_model()

        if batch_size is None:
            batch_size = 32 if self.device.startswith("cuda") else 10

        embeddings: List[Optional[np.ndarray]] = []
        waveform_np = waveform.numpy()
        total_segments = len(segments)
        pending_waveforms: List[np.ndarray] = []
        pending_indices: List[int] = []

        def flush_pending() -> None:
            if not pending_waveforms:
                return

            inputs = self._feature_extractor(
                pending_waveforms,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self._model(**inputs)
                batch_embeddings = outputs.embeddings.cpu().numpy()

            for idx, embedding in zip(pending_indices, batch_embeddings):
                normalized = embedding / (np.linalg.norm(embedding) + 1e-8)
                embeddings[idx] = normalized

            pending_waveforms.clear()
            pending_indices.clear()
            del inputs, outputs, batch_embeddings
            self._cleanup_memory()

        for i, (start, end) in enumerate(segments):
            if i > 0 and i % batch_size == 0:
                flush_pending()
                if not self._check_memory():
                    logger.warning(
                        f"Memory pressure at segment {i}/{total_segments}, "
                        "continuing with cleanup"
                    )

            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            segment_waveform = waveform_np[start_sample:end_sample]

            # Skip very short segments
            if len(segment_waveform) < sample_rate * 0.3:
                embeddings.append(None)
                continue

            embeddings.append(None)
            pending_waveforms.append(segment_waveform)
            pending_indices.append(len(embeddings) - 1)

            if len(pending_waveforms) >= batch_size:
                flush_pending()

        flush_pending()

        # Final cleanup
        self._cleanup_memory()

        return embeddings

    def _cluster_embeddings(
        self,
        embeddings: List[np.ndarray],
        num_speakers: Optional[int] = None,
        distance_threshold: float = 0.5,
    ) -> np.ndarray:
        """Cluster embeddings to assign speaker labels.

        Args:
            embeddings: List of speaker embeddings.
            num_speakers: Number of speakers (if known).
            distance_threshold: Distance threshold for clustering.

        Returns:
            Array of cluster labels.
        """
        # Filter out None embeddings
        valid_embeddings = [e for e in embeddings if e is not None]
        valid_indices = [i for i, e in enumerate(embeddings) if e is not None]

        if len(valid_embeddings) < 2:
            return np.zeros(len(embeddings), dtype=int)

        # Stack embeddings
        embedding_matrix = np.stack(valid_embeddings)

        # Compute pairwise distances (cosine)
        distances = pdist(embedding_matrix, metric='cosine')

        # Hierarchical clustering
        linkage_matrix = linkage(distances, method='average')

        if num_speakers is not None:
            # Use known number of speakers
            labels = fcluster(linkage_matrix, num_speakers, criterion='maxclust')
        else:
            # Use distance threshold
            labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')

        # Limit to max speakers
        unique_labels = np.unique(labels)
        if len(unique_labels) > self.max_speakers:
            labels = fcluster(linkage_matrix, self.max_speakers, criterion='maxclust')

        # Map back to full array
        full_labels = np.zeros(len(embeddings), dtype=int)
        for idx, label in zip(valid_indices, labels):
            full_labels[idx] = label

        return full_labels

    def _merge_adjacent_segments(
        self,
        segments: List[SpeakerSegment],
        max_gap: float = 0.3,
    ) -> List[SpeakerSegment]:
        """Merge adjacent segments from same speaker.

        Args:
            segments: List of speaker segments.
            max_gap: Maximum gap to merge (seconds).

        Returns:
            Merged segments.
        """
        if not segments:
            return []

        # Sort by start time
        segments = sorted(segments, key=lambda s: s.start)

        merged = [segments[0]]
        for seg in segments[1:]:
            last = merged[-1]
            if seg.speaker_id == last.speaker_id and seg.start - last.end <= max_gap:
                # Merge segments
                merged[-1] = SpeakerSegment(
                    start=last.start,
                    end=seg.end,
                    speaker_id=last.speaker_id,
                    embedding=last.embedding,  # Keep first embedding
                    confidence=min(last.confidence, seg.confidence),
                )
            else:
                merged.append(seg)

        return merged

    def diarize(
        self,
        audio_path: Union[str, Path],
        num_speakers: Optional[int] = None,
        segment_duration: float = 1.5,
        overlap: float = 0.5,
        use_chunked_processing: Optional[bool] = None,
    ) -> DiarizationResult:
        """Perform speaker diarization on an audio file.

        Args:
            audio_path: Path to the audio file.
            num_speakers: Number of speakers if known (improves accuracy).
            segment_duration: Duration of analysis segments in seconds.
            overlap: Overlap ratio between segments.
            use_chunked_processing: Force chunked processing (None = auto based on duration).

        Returns:
            DiarizationResult with speaker segments and embeddings.
        """
        audio_path = Path(audio_path)
        logger.info(f"Starting diarization for: {audio_path}")

        # Check initial memory
        available_mem = get_available_memory_gb()
        logger.info(f"Available memory: {available_mem:.1f} GB (limit: {self.max_memory_gb:.1f} GB)")

        # Load audio
        waveform, sample_rate = self._load_audio(audio_path)
        audio_duration = len(waveform) / sample_rate
        logger.info(f"Audio duration: {audio_duration:.2f}s")

        # Decide whether to use chunked processing
        # Rule: use chunks if audio > 2 minutes or memory is tight
        if use_chunked_processing is None:
            use_chunked_processing = (
                audio_duration > 120.0 or
                available_mem < self.max_memory_gb * 0.5
            )

        if use_chunked_processing:
            if self.device.startswith("cuda") and audio_duration >= 180.0:
                # Long GPU jobs do not need as much overlap to preserve speaker
                # identity. Reducing window count cuts end-to-end latency
                # without changing the external API contract.
                segment_duration = max(segment_duration, 2.0)
                overlap = min(overlap, 0.25)
            logger.info(f"Using chunked processing for {audio_duration:.1f}s audio")
            return self._diarize_chunked(
                waveform, sample_rate, audio_duration, audio_path,
                num_speakers, segment_duration, overlap
            )

        # Detect voice activity
        speech_regions = self._detect_voice_activity(waveform, sample_rate)
        logger.info(f"Detected {len(speech_regions)} speech regions")

        if not speech_regions:
            logger.warning("No speech detected in audio")
            return DiarizationResult(
                segments=[],
                num_speakers=0,
                audio_duration=audio_duration,
            )

        # Segment audio
        segments = self._segment_audio_fixed(
            waveform, sample_rate, segment_duration, overlap
        )
        logger.info(f"Created {len(segments)} analysis segments")

        # Extract embeddings
        embeddings = self._extract_segment_embeddings(waveform, sample_rate, segments)
        valid_count = sum(1 for e in embeddings if e is not None)
        logger.info(f"Extracted {valid_count} valid embeddings")

        # Cluster embeddings
        labels = self._cluster_embeddings(embeddings, num_speakers)
        detected_speakers = len(set(labels)) - (1 if 0 in labels else 0)
        logger.info(f"Detected {detected_speakers} speakers")

        # Create speaker segments
        speaker_segments = []
        for (start, end), label, embedding in zip(segments, labels, embeddings):
            if embedding is not None and label > 0:
                speaker_segments.append(SpeakerSegment(
                    start=start,
                    end=end,
                    speaker_id=f"SPEAKER_{label - 1:02d}",
                    embedding=embedding,
                    confidence=0.8,  # Placeholder confidence
                ))

        # Merge adjacent segments
        speaker_segments = self._merge_adjacent_segments(speaker_segments)

        # Compute average embedding per speaker
        speaker_embeddings = {}
        for speaker_id in set(s.speaker_id for s in speaker_segments):
            speaker_embs = [s.embedding for s in speaker_segments
                          if s.speaker_id == speaker_id and s.embedding is not None]
            if speaker_embs:
                avg_embedding = np.mean(speaker_embs, axis=0)
                avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
                speaker_embeddings[speaker_id] = avg_embedding

        result = DiarizationResult(
            segments=speaker_segments,
            num_speakers=detected_speakers,
            audio_duration=audio_duration,
            speaker_embeddings=speaker_embeddings,
        )

        logger.info(f"Diarization complete: {len(speaker_segments)} segments, "
                   f"{detected_speakers} speakers")

        # Cleanup
        self._cleanup_memory()

        return result

    def _diarize_chunked(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        audio_duration: float,
        audio_path: Path,
        num_speakers: Optional[int],
        segment_duration: float,
        overlap: float,
    ) -> DiarizationResult:
        """Diarize long audio in memory-safe chunks.

        Processes audio in chunks, extracts embeddings, then clusters globally.
        """
        # Get audio chunks
        chunks = self._get_audio_chunks(audio_duration)

        all_segments = []
        all_embeddings = []

        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
            logger.info(f"Processing chunk {chunk_idx + 1}/{len(chunks)}: "
                       f"{chunk_start:.1f}s - {chunk_end:.1f}s")

            # Extract chunk waveform
            start_sample = int(chunk_start * sample_rate)
            end_sample = int(chunk_end * sample_rate)
            chunk_waveform = waveform[start_sample:end_sample]

            # Detect voice activity in chunk
            speech_regions = self._detect_voice_activity(chunk_waveform, sample_rate)

            if not speech_regions:
                logger.debug(f"No speech in chunk {chunk_idx + 1}")
                self._cleanup_memory()
                continue

            # Segment chunk
            chunk_segments = self._segment_audio_fixed(
                chunk_waveform, sample_rate, segment_duration, overlap
            )

            # Extract embeddings for this chunk
            chunk_embeddings = self._extract_segment_embeddings(
                chunk_waveform, sample_rate, chunk_segments
            )

            # Adjust segment times to global timeline
            for (seg_start, seg_end), emb in zip(chunk_segments, chunk_embeddings):
                global_start = chunk_start + seg_start
                global_end = chunk_start + seg_end
                all_segments.append((global_start, global_end))
                all_embeddings.append(emb)

            # Cleanup between chunks
            del chunk_waveform, chunk_segments, chunk_embeddings
            self._cleanup_memory()

        if not all_embeddings:
            logger.warning("No speech detected in entire audio")
            return DiarizationResult(
                segments=[],
                num_speakers=0,
                audio_duration=audio_duration,
            )

        # Global clustering of all embeddings
        logger.info(f"Clustering {len(all_embeddings)} embeddings globally")
        labels = self._cluster_embeddings(all_embeddings, num_speakers)
        detected_speakers = len(set(labels)) - (1 if 0 in labels else 0)
        logger.info(f"Detected {detected_speakers} speakers")

        # Create speaker segments
        speaker_segments = []
        for (start, end), label, embedding in zip(all_segments, labels, all_embeddings):
            if embedding is not None and label > 0:
                speaker_segments.append(SpeakerSegment(
                    start=start,
                    end=end,
                    speaker_id=f"SPEAKER_{label - 1:02d}",
                    embedding=embedding,
                    confidence=0.8,
                ))

        # Merge adjacent segments
        speaker_segments = self._merge_adjacent_segments(speaker_segments)

        # Compute average embedding per speaker
        speaker_embeddings = {}
        for speaker_id in set(s.speaker_id for s in speaker_segments):
            speaker_embs = [s.embedding for s in speaker_segments
                          if s.speaker_id == speaker_id and s.embedding is not None]
            if speaker_embs:
                avg_embedding = np.mean(speaker_embs, axis=0)
                avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
                speaker_embeddings[speaker_id] = avg_embedding

        result = DiarizationResult(
            segments=speaker_segments,
            num_speakers=detected_speakers,
            audio_duration=audio_duration,
            speaker_embeddings=speaker_embeddings,
        )

        logger.info(f"Chunked diarization complete: {len(speaker_segments)} segments, "
                   f"{detected_speakers} speakers")

        return result

    def extract_speaker_audio(
        self,
        audio_path: Union[str, Path],
        diarization: DiarizationResult,
        speaker_id: str,
        output_path: Optional[Union[str, Path]] = None,
        min_duration: float = 0.5,
    ) -> Path:
        """Extract audio for a specific speaker from diarization results.

        Args:
            audio_path: Path to the original audio file.
            diarization: Diarization result.
            speaker_id: Speaker ID to extract.
            output_path: Output path (auto-generated if None).
            min_duration: Minimum segment duration to include.

        Returns:
            Path to the extracted audio file.
        """
        audio_path = Path(audio_path)

        # Load original audio
        waveform, sample_rate = self._load_audio(audio_path)

        # Get segments for this speaker
        segments = diarization.get_speaker_segments(speaker_id)
        segments = [s for s in segments if s.duration >= min_duration]

        if not segments:
            raise ValueError(f"No segments found for speaker: {speaker_id}")

        # Extract and concatenate segments
        extracted_parts = []
        for seg in sorted(segments, key=lambda s: s.start):
            start_sample = int(seg.start * sample_rate)
            end_sample = int(seg.end * sample_rate)
            extracted_parts.append(waveform[start_sample:end_sample])

        # Concatenate with small silence gaps
        silence = torch.zeros(int(0.1 * sample_rate))
        combined = []
        for i, part in enumerate(extracted_parts):
            combined.append(part)
            if i < len(extracted_parts) - 1:
                combined.append(silence)

        combined_waveform = torch.cat(combined)

        # Generate output path if not provided
        if output_path is None:
            output_path = Path(tempfile.mkdtemp()) / f"{audio_path.stem}_{speaker_id}.wav"
        else:
            output_path = Path(output_path)

        # Save extracted audio
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_path), combined_waveform.unsqueeze(0), sample_rate)

        logger.info(f"Extracted {len(segments)} segments for {speaker_id} to {output_path}")

        return output_path


def match_speaker_to_profile(
    embedding: np.ndarray,
    profile_embeddings: Dict[str, np.ndarray],
    threshold: float = 0.7,
) -> Optional[str]:
    """Match a speaker embedding to existing profiles.

    Args:
        embedding: Speaker embedding to match.
        profile_embeddings: Dictionary mapping profile ID to embedding.
        threshold: Cosine similarity threshold for matching.

    Returns:
        Profile ID if matched, None otherwise.
    """
    if not profile_embeddings:
        return None

    best_match = None
    best_similarity = threshold

    for profile_id, profile_embedding in profile_embeddings.items():
        # Cosine similarity
        similarity = np.dot(embedding, profile_embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = profile_id

    if best_match:
        logger.info(f"Matched speaker to profile {best_match} "
                   f"(similarity: {best_similarity:.3f})")

    return best_match


def compute_speaker_similarity(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
) -> float:
    """Compute cosine similarity between two speaker embeddings.

    Args:
        embedding1: First speaker embedding.
        embedding2: Second speaker embedding.

    Returns:
        Cosine similarity (0-1, higher = more similar).
    """
    return float(np.dot(embedding1, embedding2))
