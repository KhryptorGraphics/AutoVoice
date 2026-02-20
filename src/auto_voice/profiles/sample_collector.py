"""Training sample collector for karaoke session audio capture.

Captures, validates, and stores high-quality singing samples from karaoke
sessions for continuous voice profile training.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
import wave

import numpy as np

from auto_voice.profiles.db import session as db_session_module
from auto_voice.profiles.db.models import TrainingSampleDB, VoiceProfileDB


@dataclass
class AudioSegment:
    """Represents a segmented audio phrase."""

    audio: np.ndarray
    start_sample: int
    end_sample: int
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        """Duration of the segment in seconds."""
        return len(self.audio) / self.sample_rate


@dataclass
class CapturedSample:
    """Represents a captured training sample."""

    id: str
    profile_id: str
    audio_path: str
    duration_seconds: float
    sample_rate: int
    quality_score: float | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SampleCollector:
    """Collects and validates training samples from karaoke sessions.

    Captures audio samples, applies quality filtering (SNR, duration, pitch
    stability), segments phrases, and stores validated samples for training.
    """

    # Default quality thresholds based on SOTA research
    DEFAULT_MIN_SNR_DB = 20.0
    DEFAULT_MIN_DURATION_SEC = 2.0
    DEFAULT_MAX_DURATION_SEC = 30.0
    DEFAULT_MIN_PITCH_STABILITY = None  # Disabled by default, enable explicitly

    def __init__(
        self,
        storage_path: str | Path,
        min_snr_db: float | None = None,
        min_duration_sec: float | None = None,
        max_duration_sec: float | None = None,
        min_pitch_stability: float | None = None,
    ):
        """Initialize sample collector with storage path and quality thresholds.

        Args:
            storage_path: Directory to store captured audio files.
            min_snr_db: Minimum signal-to-noise ratio in dB (default: 20.0).
            min_duration_sec: Minimum sample duration in seconds (default: 2.0).
            max_duration_sec: Maximum sample duration in seconds (default: 30.0).
            min_pitch_stability: Minimum pitch stability score 0-1 (default: 0.7).
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.min_snr_db = min_snr_db if min_snr_db is not None else self.DEFAULT_MIN_SNR_DB
        self.min_duration_sec = (
            min_duration_sec if min_duration_sec is not None else self.DEFAULT_MIN_DURATION_SEC
        )
        self.max_duration_sec = (
            max_duration_sec if max_duration_sec is not None else self.DEFAULT_MAX_DURATION_SEC
        )
        self.min_pitch_stability = (
            min_pitch_stability
            if min_pitch_stability is not None
            else self.DEFAULT_MIN_PITCH_STABILITY
        )

        # Recording state
        self._recording = False
        self._current_profile_id: str | None = None
        self._current_session_id: str | None = None
        self._current_consent: bool = False
        self._audio_chunks: list[np.ndarray] = []
        self._chunk_sample_rate: int = 24000

    def capture_sample(
        self,
        profile_id: str,
        audio: np.ndarray,
        sample_rate: int,
        metadata: dict[str, Any] | None = None,
        consent_given: bool = True,
    ) -> CapturedSample | None:
        """Capture and validate an audio sample for a profile.

        Args:
            profile_id: ID of the voice profile to associate the sample with.
            audio: Audio data as numpy array (float32, mono).
            sample_rate: Sample rate of the audio in Hz.
            metadata: Optional additional metadata to store with the sample.
            consent_given: Whether user has given consent for data collection.

        Returns:
            CapturedSample if the sample passes quality checks, None otherwise.
        """
        # Require explicit consent
        if not consent_given:
            return None

        # Validate duration
        duration_sec = len(audio) / sample_rate
        if duration_sec < self.min_duration_sec or duration_sec > self.max_duration_sec:
            return None

        # Validate SNR
        snr = self.estimate_snr(audio)
        if snr < self.min_snr_db:
            return None

        # Validate pitch stability if threshold is set
        if self.min_pitch_stability is not None:
            stability = self.measure_pitch_stability(audio, sample_rate)
            if stability < self.min_pitch_stability:
                return None

        # Generate unique ID and save audio
        sample_id = str(uuid4())
        audio_path = self._save_audio(profile_id, sample_id, audio, sample_rate)

        # Create captured sample object
        captured = CapturedSample(
            id=sample_id,
            profile_id=profile_id,
            audio_path=str(audio_path),
            duration_seconds=duration_sec,
            sample_rate=sample_rate,
            quality_score=snr,
            extra_metadata=metadata or {},
        )

        # Save to database
        self._save_to_database(captured)

        return captured

    def estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate signal-to-noise ratio of audio.

        Uses a spectral flatness-based approach. Clean tonal signals (speech,
        singing) have energy concentrated in a few frequency bins, while noise
        is spread across all frequencies.

        Args:
            audio: Audio data as numpy array.

        Returns:
            Estimated SNR in decibels.
        """
        audio = np.asarray(audio, dtype=np.float32)

        if len(audio) == 0:
            return 0.0

        # Calculate overall signal power
        signal_power = np.mean(audio**2)

        if signal_power < 1e-10:
            return 0.0

        # Use spectral analysis
        n_fft = min(4096, len(audio))
        spectrum = np.abs(np.fft.rfft(audio, n=n_fft)) ** 2

        # Spectral flatness: geometric mean / arithmetic mean
        # For white noise: ~1.0, for tonal signal: ~0.0
        log_spectrum = np.log(spectrum + 1e-10)
        geometric_mean = np.exp(np.mean(log_spectrum))
        arithmetic_mean = np.mean(spectrum)

        if arithmetic_mean < 1e-10:
            return 60.0

        spectral_flatness = geometric_mean / arithmetic_mean

        # Convert flatness to SNR estimate
        # Flatness of 0 = pure tone = infinite SNR
        # Flatness of 1 = white noise = 0 dB SNR
        if spectral_flatness < 1e-6:
            return 60.0  # Very clean tonal signal

        # SNR estimate: lower flatness = higher SNR
        # This is a heuristic mapping based on typical values
        snr_db = -10 * np.log10(spectral_flatness + 1e-10)

        # Clamp to reasonable range
        return float(np.clip(snr_db, 0.0, 80.0))

    def measure_pitch_stability(self, audio: np.ndarray, sample_rate: int) -> float:
        """Measure pitch stability of audio.

        Uses autocorrelation-based pitch tracking to measure how stable
        the fundamental frequency is throughout the audio.

        Args:
            audio: Audio data as numpy array.
            sample_rate: Sample rate in Hz.

        Returns:
            Stability score between 0 and 1 (higher = more stable).
        """
        audio = np.asarray(audio, dtype=np.float32)

        if len(audio) < sample_rate // 10:  # Need at least 100ms
            return 0.0

        # Frame-based pitch detection using autocorrelation
        frame_size = int(0.05 * sample_rate)  # 50ms frames
        hop_size = int(0.025 * sample_rate)  # 25ms hop

        # Frequency range for singing: 80-800 Hz
        min_period = int(sample_rate / 800)
        max_period = int(sample_rate / 80)

        pitches = []
        confidences = []

        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i : i + frame_size]

            # Normalize frame
            frame = frame - np.mean(frame)
            if np.std(frame) < 1e-6:
                continue

            frame = frame / (np.std(frame) + 1e-10)

            # Autocorrelation
            autocorr = np.correlate(frame, frame, mode="full")
            autocorr = autocorr[len(autocorr) // 2 :]

            # Find peak in valid range
            if max_period < len(autocorr):
                search_region = autocorr[min_period:max_period]
                if len(search_region) > 0:
                    peak_idx = np.argmax(search_region) + min_period
                    confidence = autocorr[peak_idx] / (autocorr[0] + 1e-10)

                    if confidence > 0.5:  # Only count confident detections
                        pitch = sample_rate / peak_idx
                        pitches.append(pitch)
                        confidences.append(confidence)

        if len(pitches) < 3:
            return 0.0

        # Calculate pitch stability as inverse coefficient of variation
        pitches = np.array(pitches)
        mean_pitch = np.mean(pitches)
        std_pitch = np.std(pitches)

        if mean_pitch < 1e-6:
            return 0.0

        # Coefficient of variation (lower = more stable)
        cv = std_pitch / mean_pitch

        # Convert to stability score using exponential decay
        # CV of 0 = 1.0, CV of 0.2 = 0.55, CV of 0.5 = 0.22, CV of 1.0 = 0.05
        stability = np.exp(-3 * cv)

        return float(np.clip(stability, 0.0, 1.0))

    def segment_phrases(
        self, audio: np.ndarray, sample_rate: int
    ) -> list[AudioSegment]:
        """Segment audio into phrases at silence boundaries.

        Args:
            audio: Audio data as numpy array.
            sample_rate: Sample rate in Hz.

        Returns:
            List of AudioSegment objects representing detected phrases.
        """
        audio = np.asarray(audio, dtype=np.float32)

        if len(audio) == 0:
            return []

        # Parameters for silence detection
        frame_size = int(0.025 * sample_rate)  # 25ms frames
        hop_size = int(0.010 * sample_rate)  # 10ms hop
        silence_threshold_db = -40  # dB below peak

        # Calculate frame energies
        frame_energies = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i : i + frame_size]
            energy = np.mean(frame**2)
            frame_energies.append(energy)

        if not frame_energies:
            return []

        frame_energies = np.array(frame_energies)

        # Convert to dB relative to max
        max_energy = np.max(frame_energies)
        if max_energy < 1e-10:
            return []

        energies_db = 10 * np.log10(frame_energies / max_energy + 1e-10)

        # Find non-silent regions
        is_voice = energies_db > silence_threshold_db

        # Apply minimum duration smoothing
        min_voice_frames = int(0.1 * sample_rate / hop_size)  # 100ms min voice
        min_silence_frames = int(0.2 * sample_rate / hop_size)  # 200ms min silence

        # Morphological opening/closing to clean up
        is_voice = self._smooth_voice_activity(is_voice, min_voice_frames, min_silence_frames)

        # Extract segments
        segments = []
        in_segment = False
        segment_start = 0

        for i, voiced in enumerate(is_voice):
            if voiced and not in_segment:
                segment_start = i * hop_size
                in_segment = True
            elif not voiced and in_segment:
                segment_end = i * hop_size
                segment_audio = audio[segment_start:segment_end]
                segment = AudioSegment(
                    audio=segment_audio,
                    start_sample=segment_start,
                    end_sample=segment_end,
                    sample_rate=sample_rate,
                )

                # Only include segments meeting minimum duration
                if segment.duration_seconds >= self.min_duration_sec:
                    segments.append(segment)

                in_segment = False

        # Handle final segment
        if in_segment:
            segment_audio = audio[segment_start:]
            segment = AudioSegment(
                audio=segment_audio,
                start_sample=segment_start,
                end_sample=len(audio),
                sample_rate=sample_rate,
            )
            if segment.duration_seconds >= self.min_duration_sec:
                segments.append(segment)

        return segments

    def _smooth_voice_activity(
        self, is_voice: np.ndarray, min_voice: int, min_silence: int
    ) -> np.ndarray:
        """Smooth voice activity detection to remove short spurious regions."""
        result = is_voice.copy()

        # Remove short voiced regions
        in_voice = False
        start = 0
        for i in range(len(result)):
            if result[i] and not in_voice:
                start = i
                in_voice = True
            elif not result[i] and in_voice:
                if i - start < min_voice:
                    result[start:i] = False
                in_voice = False

        # Remove short silence gaps
        in_silence = False
        start = 0
        for i in range(len(result)):
            if not result[i] and not in_silence:
                start = i
                in_silence = True
            elif result[i] and in_silence:
                if i - start < min_silence:
                    result[start:i] = True
                in_silence = False

        return result

    def start_recording(
        self,
        profile_id: str,
        session_id: str,
        consent_given: bool = True,
    ) -> None:
        """Start recording audio chunks for a karaoke session.

        Args:
            profile_id: ID of the voice profile.
            session_id: ID of the karaoke session.
            consent_given: Whether user has given consent.
        """
        self._recording = True
        self._current_profile_id = profile_id
        self._current_session_id = session_id
        self._current_consent = consent_given
        self._audio_chunks = []

    def add_chunk(self, chunk: np.ndarray, sample_rate: int = 24000) -> None:
        """Add an audio chunk to the current recording.

        Args:
            chunk: Audio chunk as numpy array.
            sample_rate: Sample rate of the chunk.
        """
        if not self._recording:
            return

        self._audio_chunks.append(np.asarray(chunk, dtype=np.float32))
        self._chunk_sample_rate = sample_rate

    def stop_recording(self) -> list[CapturedSample]:
        """Stop recording and process accumulated audio.

        Returns:
            List of captured samples that passed quality checks.
        """
        if not self._recording:
            return []

        self._recording = False

        # Check consent
        if not self._current_consent:
            self._audio_chunks = []
            return []

        # Concatenate chunks
        if not self._audio_chunks:
            return []

        full_audio = np.concatenate(self._audio_chunks)
        self._audio_chunks = []

        # Segment into phrases
        segments = self.segment_phrases(full_audio, self._chunk_sample_rate)

        # Capture each segment
        captured_samples = []
        for segment in segments:
            result = self.capture_sample(
                profile_id=self._current_profile_id,
                audio=segment.audio,
                sample_rate=segment.sample_rate,
                metadata={
                    "session_id": self._current_session_id,
                    "segment_start_sample": segment.start_sample,
                    "segment_end_sample": segment.end_sample,
                },
                consent_given=self._current_consent,
            )
            if result is not None:
                captured_samples.append(result)

        return captured_samples

    def _save_audio(
        self, profile_id: str, sample_id: str, audio: np.ndarray, sample_rate: int
    ) -> Path:
        """Save audio to WAV file organized by profile.

        Args:
            profile_id: Profile ID for directory organization.
            sample_id: Unique sample ID for filename.
            audio: Audio data.
            sample_rate: Sample rate.

        Returns:
            Path to saved WAV file.
        """
        # Create profile-specific directory
        profile_dir = self.storage_path / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{sample_id}.wav"
        audio_path = profile_dir / filename

        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)

        # Write WAV file
        with wave.open(str(audio_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        return audio_path

    def _save_to_database(self, sample: CapturedSample) -> None:
        """Save captured sample metadata to database.

        Args:
            sample: CapturedSample to save.
        """
        with db_session_module.get_db_session() as session:
            # Verify profile exists
            profile = session.query(VoiceProfileDB).filter_by(id=sample.profile_id).first()
            if profile is None:
                raise ValueError(f"Profile {sample.profile_id} not found")

            # Create training sample record
            db_sample = TrainingSampleDB(
                id=sample.id,
                profile_id=sample.profile_id,
                audio_path=sample.audio_path,
                duration_seconds=sample.duration_seconds,
                sample_rate=sample.sample_rate,
                quality_score=sample.quality_score,
                extra_metadata=sample.extra_metadata,
            )
            session.add(db_sample)

            # Increment profile sample count
            profile.samples_count = (profile.samples_count or 0) + 1

            session.commit()
