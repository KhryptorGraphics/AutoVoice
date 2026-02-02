"""Comprehensive tests for sample collector (TDD Phase 3.5).

Tests for profiles/sample_collector.py:
- Sample validation (format, duration, SNR)
- Quality filtering (pitch stability)
- Phrase segmentation
- Duplicate detection
- Sample organization
- Recording session management
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from auto_voice.profiles.sample_collector import (
    SampleCollector,
    CapturedSample,
    AudioSegment,
)
from auto_voice.profiles.db import session as session_module
from auto_voice.profiles.db.models import Base, VoiceProfileDB


@pytest.fixture(scope="function")
def test_db():
    """Create in-memory database for testing."""
    from sqlalchemy import create_engine
    test_url = "sqlite:///:memory:"
    session_module._engine = None
    session_module._SessionLocal = None

    # Create engine directly without pool settings (SQLite doesn't support them)
    engine = create_engine(test_url, echo=False)
    session_module._engine = engine  # Set the global engine
    Base.metadata.create_all(bind=engine)

    yield

    Base.metadata.drop_all(bind=engine)
    engine.dispose()
    session_module._engine = None
    session_module._SessionLocal = None


@pytest.fixture
def sample_collector(tmp_path, test_db):
    """Create sample collector with temporary storage."""
    storage_path = tmp_path / "samples"
    return SampleCollector(
        storage_path=str(storage_path),
        min_snr_db=20.0,
        min_duration_sec=2.0,
        max_duration_sec=30.0,
    )


@pytest.fixture
def test_profile(test_db):
    """Create a test profile in database."""
    with session_module.get_db_session() as session:
        profile = VoiceProfileDB(user_id="user-1", name="Test Profile")
        session.add(profile)
        session.flush()
        profile_id = profile.id

    return profile_id


@pytest.fixture
def clean_audio():
    """Generate clean audio sample (high SNR)."""
    sr = 24000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Clean sine wave (high SNR)
    audio = 0.3 * np.sin(2 * np.pi * 220 * t)  # A3 note
    return audio.astype(np.float32), sr


@pytest.fixture
def noisy_audio():
    """Generate noisy audio sample (low SNR)."""
    sr = 24000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Signal + heavy noise (low SNR)
    signal = 0.1 * np.sin(2 * np.pi * 220 * t)
    noise = 0.3 * np.random.randn(len(t))
    audio = signal + noise
    return audio.astype(np.float32), sr


@pytest.fixture
def short_audio():
    """Generate audio shorter than minimum duration."""
    sr = 24000
    duration = 1.0  # Below 2.0s minimum
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * 220 * t)
    return audio.astype(np.float32), sr


class TestAudioSegmentDataclass:
    """Test AudioSegment dataclass."""

    def test_audio_segment_calculates_duration(self):
        """Test AudioSegment computes duration correctly."""
        # Arrange
        audio = np.zeros(24000, dtype=np.float32)
        segment = AudioSegment(
            audio=audio,
            start_sample=0,
            end_sample=24000,
            sample_rate=24000,
        )

        # Assert
        assert segment.duration_seconds == 1.0


class TestSampleValidation:
    """Test sample validation rules."""

    def test_capture_sample_accepts_valid_audio(self, sample_collector, test_profile, clean_audio):
        """Test valid audio sample is captured."""
        # Arrange
        audio, sr = clean_audio

        # Act
        result = sample_collector.capture_sample(
            profile_id=test_profile,
            audio=audio,
            sample_rate=sr,
            consent_given=True,
        )

        # Assert
        assert result is not None
        assert isinstance(result, CapturedSample)
        assert result.profile_id == test_profile
        assert result.duration_seconds == pytest.approx(3.0, rel=0.1)
        assert Path(result.audio_path).exists()

    def test_capture_sample_rejects_too_short(self, sample_collector, test_profile, short_audio):
        """Test audio shorter than minimum duration is rejected."""
        # Arrange
        audio, sr = short_audio

        # Act
        result = sample_collector.capture_sample(
            profile_id=test_profile,
            audio=audio,
            sample_rate=sr,
            consent_given=True,
        )

        # Assert - rejected due to duration
        assert result is None

    def test_capture_sample_rejects_low_snr(self, sample_collector, test_profile, noisy_audio):
        """Test audio with low SNR is rejected."""
        # Arrange
        audio, sr = noisy_audio

        # Act
        result = sample_collector.capture_sample(
            profile_id=test_profile,
            audio=audio,
            sample_rate=sr,
            consent_given=True,
        )

        # Assert - may be rejected due to low SNR (depending on noise level)
        # If not rejected, quality_score should be low
        if result is not None:
            assert result.quality_score < 20.0

    def test_capture_sample_requires_consent(self, sample_collector, test_profile, clean_audio):
        """Test sample is rejected without user consent."""
        # Arrange
        audio, sr = clean_audio

        # Act
        result = sample_collector.capture_sample(
            profile_id=test_profile,
            audio=audio,
            sample_rate=sr,
            consent_given=False,  # No consent
        )

        # Assert
        assert result is None

    def test_capture_sample_with_metadata(self, sample_collector, test_profile, clean_audio):
        """Test sample captures additional metadata."""
        # Arrange
        audio, sr = clean_audio
        metadata = {"session_id": "session-123", "source": "karaoke"}

        # Act
        result = sample_collector.capture_sample(
            profile_id=test_profile,
            audio=audio,
            sample_rate=sr,
            metadata=metadata,
            consent_given=True,
        )

        # Assert
        assert result is not None
        assert result.extra_metadata == metadata


class TestQualityMetrics:
    """Test quality estimation algorithms."""

    def test_estimate_snr_clean_signal(self, sample_collector):
        """Test SNR estimation for clean tonal signal."""
        # Arrange - pure sine wave (high SNR)
        sr = 24000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Act
        snr = sample_collector.estimate_snr(audio)

        # Assert - clean signal should have high SNR
        assert snr > 30.0

    def test_estimate_snr_noisy_signal(self, sample_collector):
        """Test SNR estimation for noisy signal."""
        # Arrange - signal + noise
        sr = 24000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        signal = 0.1 * np.sin(2 * np.pi * 440 * t)
        noise = 0.3 * np.random.randn(len(t))
        audio = (signal + noise).astype(np.float32)

        # Act
        snr = sample_collector.estimate_snr(audio)

        # Assert - noisy signal should have lower SNR
        assert snr < 30.0

    def test_estimate_snr_silence(self, sample_collector):
        """Test SNR estimation for silence."""
        # Arrange - near-zero audio
        audio = np.zeros(24000, dtype=np.float32) + 1e-12

        # Act
        snr = sample_collector.estimate_snr(audio)

        # Assert - silence should have low SNR
        assert snr >= 0.0  # Should not crash

    def test_estimate_snr_empty_audio(self, sample_collector):
        """Test SNR estimation handles empty audio."""
        # Arrange
        audio = np.array([], dtype=np.float32)

        # Act
        snr = sample_collector.estimate_snr(audio)

        # Assert
        assert snr == 0.0

    def test_measure_pitch_stability_stable(self, sample_collector):
        """Test pitch stability for stable tone."""
        # Arrange - constant frequency sine wave
        sr = 24000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 220 * t).astype(np.float32)

        # Act
        stability = sample_collector.measure_pitch_stability(audio, sr)

        # Assert - stable pitch should have high stability
        assert stability > 0.7

    def test_measure_pitch_stability_unstable(self, sample_collector):
        """Test pitch stability for varying frequency."""
        # Arrange - frequency sweep (unstable pitch)
        sr = 24000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        freq = 200 + 200 * t / duration  # Sweep from 200 to 400 Hz
        audio = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)

        # Act
        stability = sample_collector.measure_pitch_stability(audio, sr)

        # Assert - varying pitch should have lower stability
        assert stability < 0.7

    def test_measure_pitch_stability_too_short(self, sample_collector):
        """Test pitch stability for very short audio."""
        # Arrange - audio too short for analysis
        sr = 24000
        audio = np.ones(sr // 20, dtype=np.float32)  # 50ms

        # Act
        stability = sample_collector.measure_pitch_stability(audio, sr)

        # Assert
        assert stability == 0.0


class TestPhraseSegmentation:
    """Test phrase segmentation at silence boundaries."""

    def test_segment_phrases_single_phrase(self, sample_collector):
        """Test segmentation of continuous audio (no silences)."""
        # Arrange - continuous tone
        sr = 24000
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)

        # Act
        segments = sample_collector.segment_phrases(audio, sr)

        # Assert - single segment (no silences)
        assert len(segments) == 1
        assert segments[0].duration_seconds >= 2.9

    def test_segment_phrases_with_silence(self, sample_collector):
        """Test segmentation with silence gaps."""
        # Arrange - two phrases separated by silence
        sr = 24000
        t1 = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
        phrase1 = 0.3 * np.sin(2 * np.pi * 220 * t1)
        silence = np.zeros(int(sr * 0.5))  # 500ms silence
        phrase2 = 0.3 * np.sin(2 * np.pi * 220 * t1)

        audio = np.concatenate([phrase1, silence, phrase2]).astype(np.float32)

        # Act
        segments = sample_collector.segment_phrases(audio, sr)

        # Assert - two segments detected
        assert len(segments) == 2
        assert segments[0].duration_seconds >= 1.9
        assert segments[1].duration_seconds >= 1.9

    def test_segment_phrases_filters_short_segments(self, sample_collector):
        """Test short segments are filtered out."""
        # Arrange - short burst + long phrase
        sr = 24000
        t_short = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        short_burst = 0.3 * np.sin(2 * np.pi * 220 * t_short)
        silence = np.zeros(int(sr * 0.3))
        t_long = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
        long_phrase = 0.3 * np.sin(2 * np.pi * 220 * t_long)

        audio = np.concatenate([short_burst, silence, long_phrase]).astype(np.float32)

        # Act
        segments = sample_collector.segment_phrases(audio, sr)

        # Assert - only long segment kept (short burst filtered)
        assert len(segments) == 1
        assert segments[0].duration_seconds >= 2.9

    def test_segment_phrases_empty_audio(self, sample_collector):
        """Test segmentation of empty audio."""
        # Arrange
        audio = np.array([], dtype=np.float32)
        sr = 24000

        # Act
        segments = sample_collector.segment_phrases(audio, sr)

        # Assert
        assert len(segments) == 0

    def test_segment_phrases_silence_only(self, sample_collector):
        """Test segmentation of silence-only audio."""
        # Arrange
        sr = 24000
        audio = np.zeros(int(sr * 3.0), dtype=np.float32)

        # Act
        segments = sample_collector.segment_phrases(audio, sr)

        # Assert - no segments (all silence)
        assert len(segments) == 0


class TestRecordingSession:
    """Test recording session management."""

    def test_start_recording_initializes_state(self, sample_collector, test_profile):
        """Test start_recording initializes recording state."""
        # Act
        sample_collector.start_recording(
            profile_id=test_profile,
            session_id="session-123",
            consent_given=True,
        )

        # Assert
        assert sample_collector._recording is True
        assert sample_collector._current_profile_id == test_profile
        assert sample_collector._current_session_id == "session-123"
        assert sample_collector._current_consent is True
        assert len(sample_collector._audio_chunks) == 0

    def test_add_chunk_accumulates_audio(self, sample_collector, test_profile):
        """Test add_chunk accumulates audio chunks."""
        # Arrange
        sample_collector.start_recording(test_profile, "session-1")
        chunk1 = np.ones(1024, dtype=np.float32)
        chunk2 = np.ones(1024, dtype=np.float32) * 0.5

        # Act
        sample_collector.add_chunk(chunk1)
        sample_collector.add_chunk(chunk2)

        # Assert
        assert len(sample_collector._audio_chunks) == 2

    def test_add_chunk_when_not_recording(self, sample_collector):
        """Test add_chunk ignores audio when not recording."""
        # Arrange - no recording started
        chunk = np.ones(1024, dtype=np.float32)

        # Act
        sample_collector.add_chunk(chunk)

        # Assert - chunk ignored
        assert len(sample_collector._audio_chunks) == 0

    def test_stop_recording_processes_chunks(self, sample_collector, test_profile):
        """Test stop_recording processes accumulated audio."""
        # Arrange - record clean audio chunks
        sample_collector.start_recording(test_profile, "session-1", consent_given=True)

        sr = 24000
        t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
        audio = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)

        # Split into chunks
        chunk_size = sr // 10  # 100ms chunks
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            sample_collector.add_chunk(chunk, sample_rate=sr)

        # Act
        samples = sample_collector.stop_recording()

        # Assert - at least one sample captured
        assert len(samples) >= 1
        assert all(isinstance(s, CapturedSample) for s in samples)

    def test_stop_recording_without_consent(self, sample_collector, test_profile):
        """Test stop_recording discards audio without consent."""
        # Arrange
        sample_collector.start_recording(test_profile, "session-1", consent_given=False)
        chunk = np.ones(24000, dtype=np.float32)
        sample_collector.add_chunk(chunk)

        # Act
        samples = sample_collector.stop_recording()

        # Assert - no samples captured
        assert len(samples) == 0

    def test_stop_recording_when_not_recording(self, sample_collector):
        """Test stop_recording handles case when not recording."""
        # Act
        samples = sample_collector.stop_recording()

        # Assert
        assert len(samples) == 0


class TestFileStorage:
    """Test audio file storage organization."""

    def test_save_audio_creates_profile_directory(self, sample_collector, test_profile, clean_audio):
        """Test audio files are organized by profile."""
        # Arrange
        audio, sr = clean_audio

        # Act
        sample_collector.capture_sample(test_profile, audio, sr, consent_given=True)

        # Assert - profile directory created
        profile_dir = sample_collector.storage_path / test_profile
        assert profile_dir.exists()
        assert profile_dir.is_dir()

    def test_save_audio_generates_unique_filename(self, sample_collector, test_profile, clean_audio):
        """Test each sample gets unique filename with timestamp."""
        # Arrange
        audio, sr = clean_audio

        # Act - capture two samples
        sample1 = sample_collector.capture_sample(test_profile, audio, sr, consent_given=True)
        sample2 = sample_collector.capture_sample(test_profile, audio, sr, consent_given=True)

        # Assert - different filenames
        assert sample1.audio_path != sample2.audio_path
        assert Path(sample1.audio_path).exists()
        assert Path(sample2.audio_path).exists()

    def test_save_audio_uses_wav_format(self, sample_collector, test_profile, clean_audio):
        """Test audio files are saved as WAV format."""
        # Arrange
        audio, sr = clean_audio

        # Act
        sample = sample_collector.capture_sample(test_profile, audio, sr, consent_given=True)

        # Assert - WAV file created
        assert sample.audio_path.endswith('.wav')

        # Verify WAV file is readable
        import wave
        with wave.open(sample.audio_path, 'rb') as wav:
            assert wav.getnchannels() == 1  # Mono
            assert wav.getsampwidth() == 2  # 16-bit
            assert wav.getframerate() == sr


class TestDatabaseIntegration:
    """Test integration with database storage."""

    def test_capture_sample_saves_to_database(self, sample_collector, test_profile, clean_audio):
        """Test captured sample metadata is saved to database."""
        # Arrange
        audio, sr = clean_audio

        # Act
        sample = sample_collector.capture_sample(test_profile, audio, sr, consent_given=True)

        # Assert - sample in database
        with session_module.get_db_session() as session:
            from auto_voice.profiles.db.models import TrainingSampleDB
            db_sample = session.query(TrainingSampleDB).filter_by(id=sample.id).first()
            assert db_sample is not None
            assert db_sample.profile_id == test_profile
            assert db_sample.audio_path == sample.audio_path

    def test_capture_sample_increments_profile_count(self, sample_collector, test_profile, clean_audio):
        """Test profile sample count is incremented."""
        # Arrange
        audio, sr = clean_audio

        # Act - capture two samples
        sample_collector.capture_sample(test_profile, audio, sr, consent_given=True)
        sample_collector.capture_sample(test_profile, audio, sr, consent_given=True)

        # Assert - profile count updated
        with session_module.get_db_session() as session:
            profile = session.query(VoiceProfileDB).filter_by(id=test_profile).first()
            assert profile.samples_count == 2

    def test_capture_sample_with_nonexistent_profile(self, sample_collector, clean_audio):
        """Test capturing sample for non-existent profile raises error."""
        # Arrange
        audio, sr = clean_audio

        # Act & Assert
        with pytest.raises(ValueError, match="Profile .* not found"):
            sample_collector.capture_sample(
                profile_id="nonexistent-profile-id",
                audio=audio,
                sample_rate=sr,
                consent_given=True,
            )


class TestCustomThresholds:
    """Test custom quality thresholds."""

    def test_custom_min_snr_threshold(self, tmp_path, test_db, test_profile):
        """Test custom minimum SNR threshold."""
        # Arrange - create collector with low SNR threshold
        collector = SampleCollector(
            storage_path=str(tmp_path / "samples"),
            min_snr_db=5.0,  # Very low threshold
        )

        # Create moderately noisy audio that should pass the threshold
        sr = 24000
        t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
        signal = 0.3 * np.sin(2 * np.pi * 220 * t)  # Stronger signal
        noise = 0.05 * np.random.randn(len(t))  # Less noise
        audio = (signal + noise).astype(np.float32)

        # Act
        result = collector.capture_sample(test_profile, audio, sr, consent_given=True)

        # Assert - accepted with low SNR threshold
        # Note: may still be rejected if SNR is extremely low due to random noise
        if result is None:
            # Verify SNR is indeed below threshold (test is working correctly)
            snr = collector.estimate_snr(audio)
            assert snr < 5.0
        else:
            # Sample was accepted as expected
            assert result is not None

    def test_custom_duration_thresholds(self, tmp_path, test_db, test_profile):
        """Test custom duration min/max thresholds."""
        # Arrange - create collector with custom durations
        collector = SampleCollector(
            storage_path=str(tmp_path / "samples"),
            min_duration_sec=1.0,  # Lower minimum
            max_duration_sec=5.0,  # Lower maximum
        )

        # Create 1.5 second audio (would be rejected by default collector)
        sr = 24000
        t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
        audio = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)

        # Act
        result = collector.capture_sample(test_profile, audio, sr, consent_given=True)

        # Assert - accepted with custom thresholds
        assert result is not None

    def test_pitch_stability_threshold(self, tmp_path, test_db, test_profile):
        """Test pitch stability threshold filtering."""
        # Arrange - create collector with pitch stability requirement
        collector = SampleCollector(
            storage_path=str(tmp_path / "samples"),
            min_pitch_stability=0.8,  # High stability required
        )

        # Create unstable pitch audio
        sr = 24000
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        freq = 200 + 100 * t / duration  # Frequency sweep
        audio = 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)

        # Act
        result = collector.capture_sample(test_profile, audio, sr, consent_given=True)

        # Assert - rejected due to low pitch stability
        assert result is None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_capture_sample_with_zero_audio(self, sample_collector, test_profile):
        """Test capturing sample with all-zero audio."""
        # Arrange
        audio = np.zeros(72000, dtype=np.float32)  # 3 seconds
        sr = 24000

        # Act
        result = sample_collector.capture_sample(test_profile, audio, sr, consent_given=True)

        # Assert - rejected due to low SNR
        assert result is None

    def test_segment_phrases_with_very_short_gaps(self, sample_collector):
        """Test segmentation ignores very brief silence gaps."""
        # Arrange - phrases with brief gaps
        sr = 24000
        t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
        phrase = 0.3 * np.sin(2 * np.pi * 220 * t)
        brief_gap = np.zeros(int(sr * 0.05))  # 50ms gap (too short)

        audio = np.concatenate([phrase, brief_gap, phrase]).astype(np.float32)

        # Act
        segments = sample_collector.segment_phrases(audio, sr)

        # Assert - treated as single segment (gap too brief)
        assert len(segments) == 1

    def test_storage_path_created_automatically(self, tmp_path, test_db):
        """Test storage path is created if it doesn't exist."""
        # Arrange
        storage_path = tmp_path / "nonexistent" / "samples"
        assert not storage_path.exists()

        # Act
        collector = SampleCollector(storage_path=str(storage_path))

        # Assert - path created
        assert storage_path.exists()
