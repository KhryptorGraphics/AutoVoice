"""Tests for training sample collection from karaoke sessions.

Task 3.1: Write failing tests for karaoke session audio capture (TDD Red Phase).
"""

import io
import os
import struct
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from auto_voice.profiles.db.models import Base, VoiceProfileDB
from auto_voice.profiles.db import session as db_session_module


@pytest.fixture
def test_db():
    """Create SQLite in-memory database for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)

    original_engine = db_session_module._engine
    original_session = db_session_module._SessionLocal

    db_session_module._engine = engine
    db_session_module._SessionLocal = SessionLocal

    yield engine

    db_session_module._engine = original_engine
    db_session_module._SessionLocal = original_session


@pytest.fixture
def temp_storage():
    """Create temporary storage directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def test_profile(test_db):
    """Create a test voice profile."""
    with db_session_module.get_db_session() as session:
        profile = VoiceProfileDB(user_id="user-123", name="Test Singer")
        session.add(profile)
        session.flush()
        profile_id = profile.id
    return profile_id


def generate_test_audio(duration_sec: float = 2.0, sample_rate: int = 24000) -> np.ndarray:
    """Generate test audio with a simple sine wave."""
    t = np.linspace(0, duration_sec, int(duration_sec * sample_rate), dtype=np.float32)
    # Generate 440Hz sine wave
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio


def generate_noisy_audio(duration_sec: float = 2.0, sample_rate: int = 24000, snr_db: float = 5.0) -> np.ndarray:
    """Generate noisy audio with low SNR."""
    clean = generate_test_audio(duration_sec, sample_rate)
    noise = np.random.randn(len(clean)).astype(np.float32)

    # Calculate scaling for desired SNR
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    scale = np.sqrt(signal_power / (noise_power * (10 ** (snr_db / 10))))
    noisy = clean + noise * scale
    return noisy


class TestSampleCollector:
    """Test SampleCollector class for capturing training samples."""

    def test_sample_collector_initialization(self, temp_storage):
        """SampleCollector initializes with storage path and quality thresholds."""
        from auto_voice.profiles.sample_collector import SampleCollector

        collector = SampleCollector(
            storage_path=temp_storage,
            min_snr_db=20.0,
            min_duration_sec=2.0,
            max_duration_sec=30.0,
        )

        assert collector.storage_path == Path(temp_storage)
        assert collector.min_snr_db == 20.0
        assert collector.min_duration_sec == 2.0
        assert collector.max_duration_sec == 30.0

    def test_sample_collector_default_thresholds(self, temp_storage):
        """SampleCollector uses sensible defaults for quality thresholds."""
        from auto_voice.profiles.sample_collector import SampleCollector

        collector = SampleCollector(storage_path=temp_storage)

        assert collector.min_snr_db >= 15.0  # At least 15dB SNR
        assert collector.min_duration_sec >= 1.0  # At least 1 second
        assert collector.max_duration_sec <= 60.0  # At most 60 seconds


class TestSampleCapture:
    """Test capturing audio samples from sessions."""

    def test_capture_sample_from_audio(self, temp_storage, test_db, test_profile):
        """Capture audio sample and store to profile."""
        from auto_voice.profiles.sample_collector import SampleCollector

        collector = SampleCollector(storage_path=temp_storage)
        audio = generate_test_audio(duration_sec=3.0)

        result = collector.capture_sample(
            profile_id=test_profile,
            audio=audio,
            sample_rate=24000,
            metadata={"session_id": "session-123", "song_id": "song-456"},
        )

        assert result is not None
        assert result.profile_id == test_profile
        assert result.duration_seconds == pytest.approx(3.0, rel=0.1)
        assert result.sample_rate == 24000
        assert os.path.exists(result.audio_path)

    def test_capture_sample_with_consent(self, temp_storage, test_db, test_profile):
        """Sample capture requires explicit user consent flag."""
        from auto_voice.profiles.sample_collector import SampleCollector

        collector = SampleCollector(storage_path=temp_storage)
        audio = generate_test_audio()

        # Without consent, should not capture
        result_no_consent = collector.capture_sample(
            profile_id=test_profile,
            audio=audio,
            sample_rate=24000,
            consent_given=False,
        )
        assert result_no_consent is None

        # With consent, should capture
        result_with_consent = collector.capture_sample(
            profile_id=test_profile,
            audio=audio,
            sample_rate=24000,
            consent_given=True,
        )
        assert result_with_consent is not None

    def test_capture_sample_stores_metadata(self, temp_storage, test_db, test_profile):
        """Captured sample stores additional metadata."""
        from auto_voice.profiles.sample_collector import SampleCollector

        collector = SampleCollector(storage_path=temp_storage)
        audio = generate_test_audio()

        result = collector.capture_sample(
            profile_id=test_profile,
            audio=audio,
            sample_rate=24000,
            metadata={
                "session_id": "session-123",
                "song_id": "song-456",
                "timestamp_offset_sec": 45.5,
            },
        )

        assert result.extra_metadata["session_id"] == "session-123"
        assert result.extra_metadata["song_id"] == "song-456"
        assert result.extra_metadata["timestamp_offset_sec"] == 45.5


class TestQualityFiltering:
    """Test audio quality filtering before capture."""

    def test_reject_low_snr_audio(self, temp_storage, test_db, test_profile):
        """Reject audio with SNR below threshold."""
        from auto_voice.profiles.sample_collector import SampleCollector

        collector = SampleCollector(storage_path=temp_storage, min_snr_db=20.0)
        noisy_audio = generate_noisy_audio(snr_db=10.0)  # Below threshold

        result = collector.capture_sample(
            profile_id=test_profile,
            audio=noisy_audio,
            sample_rate=24000,
        )

        assert result is None

    def test_accept_high_snr_audio(self, temp_storage, test_db, test_profile):
        """Accept audio with SNR above threshold."""
        from auto_voice.profiles.sample_collector import SampleCollector

        collector = SampleCollector(storage_path=temp_storage, min_snr_db=15.0)
        clean_audio = generate_test_audio()  # Clean sine wave, high SNR

        result = collector.capture_sample(
            profile_id=test_profile,
            audio=clean_audio,
            sample_rate=24000,
        )

        assert result is not None

    def test_reject_too_short_audio(self, temp_storage, test_db, test_profile):
        """Reject audio shorter than minimum duration."""
        from auto_voice.profiles.sample_collector import SampleCollector

        collector = SampleCollector(storage_path=temp_storage, min_duration_sec=2.0)
        short_audio = generate_test_audio(duration_sec=1.0)

        result = collector.capture_sample(
            profile_id=test_profile,
            audio=short_audio,
            sample_rate=24000,
        )

        assert result is None

    def test_reject_too_long_audio(self, temp_storage, test_db, test_profile):
        """Reject audio longer than maximum duration."""
        from auto_voice.profiles.sample_collector import SampleCollector

        collector = SampleCollector(storage_path=temp_storage, max_duration_sec=5.0)
        long_audio = generate_test_audio(duration_sec=10.0)

        result = collector.capture_sample(
            profile_id=test_profile,
            audio=long_audio,
            sample_rate=24000,
        )

        assert result is None

    def test_estimate_snr(self, temp_storage):
        """SNR estimation returns reasonable values."""
        from auto_voice.profiles.sample_collector import SampleCollector

        collector = SampleCollector(storage_path=temp_storage)

        # Clean audio should have high SNR
        clean = generate_test_audio()
        clean_snr = collector.estimate_snr(clean)
        assert clean_snr > 30.0  # Clean sine wave should be very high

        # Noisy audio should have lower SNR
        noisy = generate_noisy_audio(snr_db=10.0)
        noisy_snr = collector.estimate_snr(noisy)
        assert noisy_snr < 20.0


class TestPitchStability:
    """Test pitch stability detection for quality filtering."""

    def test_detect_stable_pitch(self, temp_storage):
        """Detect stable pitch in clean singing audio."""
        from auto_voice.profiles.sample_collector import SampleCollector

        collector = SampleCollector(storage_path=temp_storage)

        # Stable pitch audio (single frequency)
        stable_audio = generate_test_audio()
        stability = collector.measure_pitch_stability(stable_audio, sample_rate=24000)

        assert stability > 0.8  # Should be highly stable

    def test_reject_unstable_pitch(self, temp_storage, test_db, test_profile):
        """Reject audio with highly unstable pitch."""
        from auto_voice.profiles.sample_collector import SampleCollector

        collector = SampleCollector(
            storage_path=temp_storage,
            min_pitch_stability=0.7,
        )

        # Generate audio with discrete frequency jumps between frames
        # Each 50ms segment has a different random frequency (200-800 Hz)
        sample_rate = 24000
        segment_samples = int(0.05 * sample_rate)  # 50ms segments
        total_samples = int(2.0 * sample_rate)
        np.random.seed(42)

        unstable_audio = np.zeros(total_samples, dtype=np.float32)
        for i in range(0, total_samples, segment_samples):
            freq = np.random.uniform(200, 800)  # Wide range for instability
            t = np.arange(min(segment_samples, total_samples - i)) / sample_rate
            unstable_audio[i : i + len(t)] = np.sin(2 * np.pi * freq * t).astype(
                np.float32
            )

        result = collector.capture_sample(
            profile_id=test_profile,
            audio=unstable_audio,
            sample_rate=24000,
        )

        assert result is None


class TestPhraseSegmentation:
    """Test automatic phrase segmentation using silence detection."""

    def test_segment_by_silence(self, temp_storage):
        """Segment audio into phrases at silence boundaries."""
        from auto_voice.profiles.sample_collector import SampleCollector

        collector = SampleCollector(storage_path=temp_storage)

        # Generate audio with silence gaps
        sample_rate = 24000
        phrase1 = generate_test_audio(duration_sec=2.0, sample_rate=sample_rate)
        silence = np.zeros(int(0.5 * sample_rate), dtype=np.float32)
        phrase2 = generate_test_audio(duration_sec=3.0, sample_rate=sample_rate)

        audio_with_gaps = np.concatenate([phrase1, silence, phrase2])

        segments = collector.segment_phrases(audio_with_gaps, sample_rate=sample_rate)

        assert len(segments) >= 2
        # First segment should be around 2 seconds
        assert 1.5 < segments[0].duration_seconds < 2.5
        # Second segment should be around 3 seconds
        assert 2.5 < segments[1].duration_seconds < 3.5

    def test_segment_minimum_length(self, temp_storage):
        """Segments shorter than minimum are discarded."""
        from auto_voice.profiles.sample_collector import SampleCollector

        collector = SampleCollector(
            storage_path=temp_storage,
            min_duration_sec=2.0,
        )

        # Generate short phrase followed by longer phrase
        sample_rate = 24000
        short_phrase = generate_test_audio(duration_sec=0.5, sample_rate=sample_rate)
        silence = np.zeros(int(0.5 * sample_rate), dtype=np.float32)
        long_phrase = generate_test_audio(duration_sec=3.0, sample_rate=sample_rate)

        audio = np.concatenate([short_phrase, silence, long_phrase])

        segments = collector.segment_phrases(audio, sample_rate=sample_rate)

        # Only the long phrase should be returned
        assert len(segments) == 1
        assert segments[0].duration_seconds > 2.0


class TestKaraokeIntegration:
    """Test integration with karaoke session events."""

    def test_collector_receives_session_audio(self, temp_storage, test_db, test_profile):
        """SampleCollector can receive audio from karaoke session."""
        from auto_voice.profiles.sample_collector import SampleCollector

        collector = SampleCollector(storage_path=temp_storage)

        # Simulate receiving audio chunks from karaoke session
        chunk_size = 4800  # 200ms at 24kHz
        audio = generate_test_audio(duration_sec=3.0)

        # Process in chunks (as would happen in real session)
        collector.start_recording(profile_id=test_profile, session_id="session-123")

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            collector.add_chunk(chunk)

        samples = collector.stop_recording()

        assert len(samples) >= 1
        total_duration = sum(s.duration_seconds for s in samples)
        assert total_duration >= 2.0

    def test_collector_respects_consent_flag(self, temp_storage, test_db, test_profile):
        """Recording only happens when consent is enabled."""
        from auto_voice.profiles.sample_collector import SampleCollector

        collector = SampleCollector(storage_path=temp_storage)
        audio = generate_test_audio(duration_sec=3.0)

        # Start without consent
        collector.start_recording(
            profile_id=test_profile,
            session_id="session-123",
            consent_given=False,
        )
        collector.add_chunk(audio)
        samples = collector.stop_recording()

        assert len(samples) == 0

    def test_collector_handles_profile_selection(self, temp_storage, test_db, test_profile):
        """Collector associates samples with correct profile."""
        from auto_voice.profiles.sample_collector import SampleCollector

        collector = SampleCollector(storage_path=temp_storage)

        # Create second profile
        with db_session_module.get_db_session() as session:
            profile2 = VoiceProfileDB(user_id="user-456", name="Another Singer")
            session.add(profile2)
            session.flush()
            profile2_id = profile2.id

        audio = generate_test_audio()

        # Capture to first profile
        result1 = collector.capture_sample(
            profile_id=test_profile,
            audio=audio,
            sample_rate=24000,
        )

        # Capture to second profile
        result2 = collector.capture_sample(
            profile_id=profile2_id,
            audio=audio,
            sample_rate=24000,
        )

        assert result1.profile_id == test_profile
        assert result2.profile_id == profile2_id
        assert result1.audio_path != result2.audio_path


class TestCapturedAudioStorage:
    """Test storage format and organization of captured audio."""

    def test_audio_saved_as_wav(self, temp_storage, test_db, test_profile):
        """Captured audio is saved as WAV format."""
        from auto_voice.profiles.sample_collector import SampleCollector

        collector = SampleCollector(storage_path=temp_storage)
        audio = generate_test_audio()

        result = collector.capture_sample(
            profile_id=test_profile,
            audio=audio,
            sample_rate=24000,
        )

        assert result.audio_path.endswith(".wav")

    def test_audio_organized_by_profile(self, temp_storage, test_db, test_profile):
        """Audio files are organized in profile-specific directories."""
        from auto_voice.profiles.sample_collector import SampleCollector

        collector = SampleCollector(storage_path=temp_storage)
        audio = generate_test_audio()

        result = collector.capture_sample(
            profile_id=test_profile,
            audio=audio,
            sample_rate=24000,
        )

        # Path should contain profile_id
        assert test_profile in result.audio_path

    def test_sample_recorded_in_database(self, temp_storage, test_db, test_profile):
        """Captured sample is recorded in training_samples table."""
        from auto_voice.profiles.sample_collector import SampleCollector
        from auto_voice.profiles.db.models import TrainingSampleDB

        collector = SampleCollector(storage_path=temp_storage)
        audio = generate_test_audio()

        result = collector.capture_sample(
            profile_id=test_profile,
            audio=audio,
            sample_rate=24000,
            metadata={"source": "karaoke_session"},
        )

        # Verify in database
        with db_session_module.get_db_session() as session:
            sample = session.query(TrainingSampleDB).filter_by(id=result.id).first()
            assert sample is not None
            assert sample.profile_id == test_profile
            assert sample.duration_seconds == pytest.approx(result.duration_seconds, rel=0.1)
