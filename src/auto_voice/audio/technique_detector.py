"""Vocal technique detection for singing voice analysis.

Implements detection of singing techniques:
- Vibrato: periodic pitch modulation (typically 4-7 Hz, ±20-50 cents)
- Melisma: rapid pitch transitions across multiple notes

Phase 5: Advanced Vocal Technique Preservation
- Task 5.2: Vibrato detector (frequency modulation analysis)
- Task 5.4: Melisma detector (rapid pitch transitions)
- Task 5.6: Technique-aware pitch extraction
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class VibratoSegment:
    """A segment of audio containing vibrato."""

    start_time: float
    end_time: float
    rate_hz: float
    depth_cents: float


@dataclass
class VibratoResult:
    """Result of vibrato detection."""

    has_vibrato: bool
    vibrato_rate: float = 0.0  # Hz
    vibrato_depth_cents: float = 0.0  # cents
    vibrato_segments: List[VibratoSegment] = field(default_factory=list)


@dataclass
class MelismaSegment:
    """A segment of audio containing melisma/vocal run."""

    start_time: float
    end_time: float
    note_count: int
    pitch_range_cents: float


@dataclass
class MelismaResult:
    """Result of melisma detection."""

    has_melisma: bool
    note_count: int = 0
    melisma_segments: List[MelismaSegment] = field(default_factory=list)


@dataclass
class PitchExtractionResult:
    """Result of technique-aware pitch extraction."""

    f0: np.ndarray  # Fundamental frequency contour
    vibrato_mask: np.ndarray  # Boolean mask for vibrato frames
    melisma_mask: np.ndarray  # Boolean mask for melisma frames
    confidence: np.ndarray  # Confidence for each frame


@dataclass
class TechniqueFlags:
    """Flags for passing technique information through the pipeline."""

    vibrato_mask: torch.Tensor
    vibrato_rate: float
    vibrato_depth_cents: float
    melisma_mask: torch.Tensor

    @property
    def has_vibrato(self) -> bool:
        """Check if any frames are flagged as vibrato."""
        return bool(self.vibrato_mask.any())

    @property
    def has_melisma(self) -> bool:
        """Check if any frames are flagged as melisma."""
        return bool(self.melisma_mask.any())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "vibrato_mask": self.vibrato_mask.cpu().numpy().tolist(),
            "vibrato_rate": self.vibrato_rate,
            "vibrato_depth_cents": self.vibrato_depth_cents,
            "melisma_mask": self.melisma_mask.cpu().numpy().tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TechniqueFlags":
        """Create from dictionary."""
        return cls(
            vibrato_mask=torch.tensor(data["vibrato_mask"], dtype=torch.bool),
            vibrato_rate=data["vibrato_rate"],
            vibrato_depth_cents=data["vibrato_depth_cents"],
            melisma_mask=torch.tensor(data["melisma_mask"], dtype=torch.bool),
        )


class VibratoDetector:
    """Detect vibrato in audio using frequency modulation analysis.

    Vibrato is characterized by:
    - Periodic pitch modulation at 4-7 Hz
    - Depth of ±20-50 cents (sometimes up to 100 cents)
    - Consistent rate and depth within a phrase
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 512,
        hop_size: int = 128,
        min_rate: float = 4.0,
        max_rate: float = 8.0,
        min_depth_cents: float = 15.0,
    ):
        """Initialize vibrato detector.

        Args:
            sample_rate: Audio sample rate
            frame_size: Analysis frame size
            hop_size: Hop between frames
            min_rate: Minimum vibrato rate (Hz)
            max_rate: Maximum vibrato rate (Hz)
            min_depth_cents: Minimum depth to consider as vibrato
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.min_depth_cents = min_depth_cents

    def _extract_f0(self, audio: np.ndarray) -> np.ndarray:
        """Extract F0 contour from audio using autocorrelation."""
        num_frames = (len(audio) - self.frame_size) // self.hop_size + 1
        f0 = np.zeros(num_frames)

        for i in range(num_frames):
            start = i * self.hop_size
            frame = audio[start : start + self.frame_size]

            # Autocorrelation-based pitch detection
            frame = frame * np.hanning(len(frame))
            corr = np.correlate(frame, frame, mode="full")
            corr = corr[len(corr) // 2 :]

            # Find first peak (skip lag 0)
            min_lag = int(self.sample_rate / 500)  # Max 500 Hz
            max_lag = int(self.sample_rate / 80)  # Min 80 Hz

            if max_lag > len(corr):
                max_lag = len(corr) - 1

            if min_lag >= max_lag:
                continue

            search_region = corr[min_lag:max_lag]
            if len(search_region) > 0 and search_region.max() > 0.3 * corr[0]:
                peak_idx = np.argmax(search_region) + min_lag
                f0[i] = self.sample_rate / peak_idx

        return f0

    def _analyze_vibrato(
        self, f0: np.ndarray
    ) -> tuple[bool, float, float, List[VibratoSegment]]:
        """Analyze F0 contour for vibrato characteristics."""
        # Remove unvoiced frames
        voiced_mask = f0 > 0
        if voiced_mask.sum() < 10:
            return False, 0.0, 0.0, []

        f0_voiced = f0[voiced_mask]

        # Convert to cents relative to mean
        mean_f0 = np.mean(f0_voiced)
        if mean_f0 <= 0:
            return False, 0.0, 0.0, []

        cents = 1200 * np.log2(f0_voiced / mean_f0 + 1e-8)

        # Analyze periodicity using FFT on the cents contour
        if len(cents) < 16:
            return False, 0.0, 0.0, []

        # Compute frame rate
        frame_rate = self.sample_rate / self.hop_size

        # FFT of cents deviation
        n = len(cents)
        fft = np.fft.rfft(cents - np.mean(cents))
        freqs = np.fft.rfftfreq(n, 1.0 / frame_rate)
        magnitude = np.abs(fft)

        # Look for peak in vibrato frequency range
        vibrato_mask = (freqs >= self.min_rate) & (freqs <= self.max_rate)
        if not vibrato_mask.any():
            return False, 0.0, 0.0, []

        vibrato_mags = magnitude[vibrato_mask]
        vibrato_freqs = freqs[vibrato_mask]

        if len(vibrato_mags) == 0:
            return False, 0.0, 0.0, []

        peak_idx = np.argmax(vibrato_mags)
        peak_mag = vibrato_mags[peak_idx]
        peak_freq = vibrato_freqs[peak_idx]

        # Check if peak is significant
        noise_floor = np.median(magnitude)
        if peak_mag < 3 * noise_floor:
            return False, 0.0, 0.0, []

        # Estimate depth from standard deviation of cents
        depth_cents = np.std(cents) * 2  # Approximate peak-to-peak

        if depth_cents < self.min_depth_cents:
            return False, 0.0, 0.0, []

        # Find vibrato segments (sliding window analysis)
        segments = self._find_vibrato_segments(f0, peak_freq, depth_cents)

        return True, peak_freq, depth_cents, segments

    def _find_vibrato_segments(
        self, f0: np.ndarray, rate: float, depth: float
    ) -> List[VibratoSegment]:
        """Find continuous segments containing vibrato."""
        frame_duration = self.hop_size / self.sample_rate
        window_frames = int(2.0 / frame_duration)  # 2-second analysis window
        hop_frames = window_frames // 4

        segments = []
        i = 0

        while i < len(f0) - window_frames:
            window = f0[i : i + window_frames]
            voiced = window > 0

            if voiced.sum() > window_frames * 0.5:
                # Check for vibrato in this window
                f0_window = window[voiced]
                if len(f0_window) > 10:
                    mean_f0 = np.mean(f0_window)
                    cents = 1200 * np.log2(f0_window / mean_f0 + 1e-8)
                    std_cents = np.std(cents)

                    if std_cents > self.min_depth_cents * 0.3:
                        start_time = i * frame_duration
                        end_time = (i + window_frames) * frame_duration

                        # Merge with previous segment if overlapping
                        if segments and start_time < segments[-1].end_time + 0.1:
                            segments[-1].end_time = end_time
                        else:
                            segments.append(
                                VibratoSegment(
                                    start_time=start_time,
                                    end_time=end_time,
                                    rate_hz=rate,
                                    depth_cents=depth,
                                )
                            )

            i += hop_frames

        return segments

    def detect(self, audio: np.ndarray) -> VibratoResult:
        """Detect vibrato in audio.

        Args:
            audio: Audio signal (mono, float32)

        Returns:
            VibratoResult with detection results
        """
        f0 = self._extract_f0(audio)
        has_vibrato, rate, depth, segments = self._analyze_vibrato(f0)

        return VibratoResult(
            has_vibrato=has_vibrato,
            vibrato_rate=rate,
            vibrato_depth_cents=depth,
            vibrato_segments=segments,
        )


class MelismaDetector:
    """Detect melisma (vocal runs) in audio.

    Melisma is characterized by:
    - Rapid transitions between distinct pitches
    - Multiple notes sung on a single syllable
    - Typically faster than 3 notes per second
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 512,
        hop_size: int = 128,
        min_note_count: int = 3,
        min_note_rate: float = 3.0,  # notes per second
        semitone_threshold: float = 0.5,  # semitones for note change
    ):
        """Initialize melisma detector.

        Args:
            sample_rate: Audio sample rate
            frame_size: Analysis frame size
            hop_size: Hop between frames
            min_note_count: Minimum notes for melisma
            min_note_rate: Minimum note rate (notes/sec) for melisma
            semitone_threshold: Pitch change threshold for note transition
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.min_note_count = min_note_count
        self.min_note_rate = min_note_rate
        self.semitone_threshold = semitone_threshold

    def _extract_f0(self, audio: np.ndarray) -> np.ndarray:
        """Extract F0 contour from audio."""
        num_frames = (len(audio) - self.frame_size) // self.hop_size + 1
        f0 = np.zeros(num_frames)

        for i in range(num_frames):
            start = i * self.hop_size
            frame = audio[start : start + self.frame_size]

            frame = frame * np.hanning(len(frame))
            corr = np.correlate(frame, frame, mode="full")
            corr = corr[len(corr) // 2 :]

            min_lag = int(self.sample_rate / 500)
            max_lag = int(self.sample_rate / 80)

            if max_lag > len(corr):
                max_lag = len(corr) - 1

            if min_lag >= max_lag:
                continue

            search_region = corr[min_lag:max_lag]
            if len(search_region) > 0 and search_region.max() > 0.3 * corr[0]:
                peak_idx = np.argmax(search_region) + min_lag
                f0[i] = self.sample_rate / peak_idx

        return f0

    def _detect_note_transitions(
        self, f0: np.ndarray
    ) -> tuple[int, List[MelismaSegment]]:
        """Detect rapid note transitions in F0 contour."""
        frame_duration = self.hop_size / self.sample_rate
        voiced_mask = f0 > 0

        if voiced_mask.sum() < 5:
            return 0, []

        # Convert to semitones (MIDI-like)
        f0_voiced = f0.copy()
        f0_voiced[f0_voiced <= 0] = 1  # Avoid log of zero
        semitones = 12 * np.log2(f0_voiced / 440.0) + 69  # MIDI number

        # Find note transitions
        transitions = []
        current_note = None
        current_start = 0

        for i, (voiced, st) in enumerate(zip(voiced_mask, semitones)):
            if not voiced:
                continue

            if current_note is None:
                current_note = round(st)
                current_start = i
            elif abs(st - current_note) > self.semitone_threshold:
                transitions.append((current_start, i, current_note))
                current_note = round(st)
                current_start = i

        if current_note is not None:
            transitions.append((current_start, len(f0), current_note))

        # Analyze transition rate
        if len(transitions) < self.min_note_count:
            return len(transitions), []

        # Find melisma segments (rapid transitions)
        segments = []
        window_frames = int(1.0 / frame_duration)  # 1-second window

        i = 0
        while i < len(transitions) - 2:
            # Count notes in a sliding window
            start_frame = transitions[i][0]
            end_frame = start_frame + window_frames

            notes_in_window = []
            for start, end, note in transitions[i:]:
                if start < end_frame:
                    notes_in_window.append(note)
                else:
                    break

            # Calculate note rate
            if len(notes_in_window) >= self.min_note_count:
                actual_end = min(transitions[i + len(notes_in_window) - 1][1], len(f0))
                duration = (actual_end - start_frame) * frame_duration
                if duration > 0:
                    note_rate = len(notes_in_window) / duration

                    if note_rate >= self.min_note_rate:
                        start_time = start_frame * frame_duration
                        end_time = actual_end * frame_duration
                        pitch_range = (max(notes_in_window) - min(notes_in_window)) * 100

                        # Merge overlapping segments
                        if segments and start_time < segments[-1].end_time + 0.1:
                            segments[-1].end_time = end_time
                            segments[-1].note_count = max(
                                segments[-1].note_count, len(notes_in_window)
                            )
                        else:
                            segments.append(
                                MelismaSegment(
                                    start_time=start_time,
                                    end_time=end_time,
                                    note_count=len(notes_in_window),
                                    pitch_range_cents=pitch_range,
                                )
                            )

            i += 1

        return len(transitions), segments

    def _is_vibrato_not_melisma(self, f0: np.ndarray) -> bool:
        """Check if the pitch variation is vibrato, not melisma.

        Vibrato: periodic oscillation around a center pitch
        Melisma: directional movement through distinct pitches
        """
        voiced_mask = f0 > 0
        if voiced_mask.sum() < 10:
            return False

        f0_voiced = f0[voiced_mask]
        mean_f0 = np.mean(f0_voiced)
        cents = 1200 * np.log2(f0_voiced / mean_f0 + 1e-8)

        # Check for periodicity (vibrato characteristic)
        frame_rate = self.sample_rate / self.hop_size
        fft = np.fft.rfft(cents - np.mean(cents))
        freqs = np.fft.rfftfreq(len(cents), 1.0 / frame_rate)
        magnitude = np.abs(fft)

        # Vibrato range: 4-8 Hz
        vibrato_mask = (freqs >= 4.0) & (freqs <= 8.0)
        if not vibrato_mask.any():
            return False

        vibrato_energy = magnitude[vibrato_mask].sum()
        total_energy = magnitude.sum() + 1e-8

        # Check for strong peak in vibrato range
        vibrato_mags = magnitude[vibrato_mask]
        if len(vibrato_mags) > 0:
            peak_mag = vibrato_mags.max()
            noise_floor = np.median(magnitude)
            # Strong vibrato peak indicates vibrato, not melisma
            if peak_mag > 5 * noise_floor and vibrato_energy > 0.25 * total_energy:
                return True

        # If most energy is in vibrato range, it's vibrato not melisma
        return vibrato_energy > 0.35 * total_energy

    def detect(self, audio: np.ndarray) -> MelismaResult:
        """Detect melisma in audio.

        Args:
            audio: Audio signal (mono, float32)

        Returns:
            MelismaResult with detection results
        """
        f0 = self._extract_f0(audio)

        # First check if it's vibrato (not melisma)
        if self._is_vibrato_not_melisma(f0):
            return MelismaResult(has_melisma=False, note_count=1)

        note_count, segments = self._detect_note_transitions(f0)

        has_melisma = len(segments) > 0 and note_count >= self.min_note_count

        return MelismaResult(
            has_melisma=has_melisma,
            note_count=note_count,
            melisma_segments=segments,
        )


class TechniqueAwarePitchExtractor:
    """Enhanced pitch extractor that detects and preserves vocal techniques.

    Combines pitch extraction with vibrato and melisma detection,
    providing masks for technique-aware processing.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 512,
        hop_size: int = 128,
    ):
        """Initialize technique-aware pitch extractor.

        Args:
            sample_rate: Audio sample rate
            frame_size: Analysis frame size
            hop_size: Hop between frames
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size

        self.vibrato_detector = VibratoDetector(
            sample_rate=sample_rate,
            frame_size=frame_size,
            hop_size=hop_size,
        )
        self.melisma_detector = MelismaDetector(
            sample_rate=sample_rate,
            frame_size=frame_size,
            hop_size=hop_size,
        )

    def _extract_f0(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract F0 with confidence."""
        num_frames = (len(audio) - self.frame_size) // self.hop_size + 1
        f0 = np.zeros(num_frames)
        confidence = np.zeros(num_frames)

        for i in range(num_frames):
            start = i * self.hop_size
            frame = audio[start : start + self.frame_size]

            frame = frame * np.hanning(len(frame))
            corr = np.correlate(frame, frame, mode="full")
            corr = corr[len(corr) // 2 :]

            min_lag = int(self.sample_rate / 500)
            max_lag = int(self.sample_rate / 80)

            if max_lag > len(corr):
                max_lag = len(corr) - 1

            if min_lag >= max_lag:
                continue

            search_region = corr[min_lag:max_lag]
            if len(search_region) > 0:
                peak_val = search_region.max()
                conf = peak_val / (corr[0] + 1e-8)

                if conf > 0.3:
                    peak_idx = np.argmax(search_region) + min_lag
                    f0[i] = self.sample_rate / peak_idx
                    confidence[i] = conf

        return f0, confidence

    def _create_vibrato_mask(
        self, num_frames: int, vibrato_result: VibratoResult
    ) -> np.ndarray:
        """Create frame-level vibrato mask from segments."""
        mask = np.zeros(num_frames, dtype=bool)
        frame_duration = self.hop_size / self.sample_rate

        for seg in vibrato_result.vibrato_segments:
            start_frame = int(seg.start_time / frame_duration)
            end_frame = int(seg.end_time / frame_duration)
            start_frame = max(0, start_frame)
            end_frame = min(num_frames, end_frame)
            mask[start_frame:end_frame] = True

        # If global vibrato detected but no segments, mark all voiced frames
        if vibrato_result.has_vibrato and not vibrato_result.vibrato_segments:
            mask[:] = True

        return mask

    def _create_melisma_mask(
        self, num_frames: int, melisma_result: MelismaResult
    ) -> np.ndarray:
        """Create frame-level melisma mask from segments."""
        mask = np.zeros(num_frames, dtype=bool)
        frame_duration = self.hop_size / self.sample_rate

        for seg in melisma_result.melisma_segments:
            start_frame = int(seg.start_time / frame_duration)
            end_frame = int(seg.end_time / frame_duration)
            start_frame = max(0, start_frame)
            end_frame = min(num_frames, end_frame)
            mask[start_frame:end_frame] = True

        return mask

    def extract(self, audio: np.ndarray) -> PitchExtractionResult:
        """Extract pitch with technique detection.

        Args:
            audio: Audio signal (mono, float32)

        Returns:
            PitchExtractionResult with F0 and technique masks
        """
        f0, confidence = self._extract_f0(audio)
        num_frames = len(f0)

        # Detect techniques
        vibrato_result = self.vibrato_detector.detect(audio)
        melisma_result = self.melisma_detector.detect(audio)

        # Create masks
        vibrato_mask = self._create_vibrato_mask(num_frames, vibrato_result)
        melisma_mask = self._create_melisma_mask(num_frames, melisma_result)

        return PitchExtractionResult(
            f0=f0,
            vibrato_mask=vibrato_mask,
            melisma_mask=melisma_mask,
            confidence=confidence,
        )

    def extract_with_flags(self, audio: np.ndarray) -> tuple[np.ndarray, TechniqueFlags]:
        """Extract pitch and return TechniqueFlags for pipeline.

        Args:
            audio: Audio signal (mono, float32)

        Returns:
            Tuple of (f0 contour, TechniqueFlags)
        """
        result = self.extract(audio)
        vibrato_result = self.vibrato_detector.detect(audio)

        flags = TechniqueFlags(
            vibrato_mask=torch.from_numpy(result.vibrato_mask),
            vibrato_rate=vibrato_result.vibrato_rate,
            vibrato_depth_cents=vibrato_result.vibrato_depth_cents,
            melisma_mask=torch.from_numpy(result.melisma_mask),
        )

        return result.f0, flags
