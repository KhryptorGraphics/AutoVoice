"""Targeted branch coverage for storage.voice_profiles."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from auto_voice.storage.voice_profiles import VoiceProfileStore


def _write_wav(path: Path, frames: int = 1600) -> None:
    """Create a minimal mono WAV file for storage tests."""
    import wave

    with wave.open(str(path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00" * frames)


@pytest.fixture
def temp_store(tmp_path):
    """Create a temporary voice profile store rooted in the test directory."""
    profiles_dir = tmp_path / "profiles"
    samples_dir = tmp_path / "samples"
    trained_models_dir = tmp_path / "trained_models"
    return VoiceProfileStore(
        profiles_dir=str(profiles_dir),
        samples_dir=str(samples_dir),
        trained_models_dir=str(trained_models_dir),
    )


@pytest.fixture
def saved_profile(temp_store):
    """Persist one profile and return its ID."""
    profile_id = temp_store.save(
        {
            "profile_id": "profile-123",
            "name": "Test Voice",
            "user_id": "user-1",
        }
    )
    return profile_id


def test_init_infers_trained_models_dir_from_samples_dir(tmp_path):
    """Explicit samples_dir should drive the default trained-model location."""
    data_root = tmp_path / "custom-data"
    samples_dir = data_root / "samples"

    store = VoiceProfileStore(samples_dir=str(samples_dir))

    assert store.samples_dir == str(samples_dir)
    assert store.trained_models_dir == str(data_root / "trained_models")


def test_load_marks_full_model_as_active(temp_store, saved_profile):
    """A saved full-model checkpoint should switch the normalized active model type."""
    Path(temp_store._full_model_path(saved_profile)).touch()

    profile = temp_store.load(saved_profile)

    assert profile["has_full_model"] is True
    assert profile["active_model_type"] == "full_model"


def test_get_reference_audio_entries_prefers_training_samples(temp_store, saved_profile, tmp_path):
    """Canonical reference audio should come from stored training samples first."""
    vocals = tmp_path / "vocals.wav"
    _write_wav(vocals)

    sample = temp_store.add_training_sample(
        profile_id=saved_profile,
        vocals_path=str(vocals),
        duration=2.5,
    )
    profile = temp_store.load(saved_profile)

    assert profile["reference_audio"] == [
        {
            "path": sample.vocals_path,
            "source": "training_sample",
            "sample_id": sample.sample_id,
            "duration_seconds": 2.5,
            "created_at": sample.created_at,
        }
    ]
    assert profile["reference_audio_count"] == 1


def test_list_profiles_returns_empty_when_profiles_dir_missing(temp_store):
    """Missing profile directories should be treated as empty storage."""
    os.rmdir(temp_store.profiles_dir)

    assert temp_store.list_profiles() == []


def test_delete_removes_saved_embedding_file(temp_store):
    """Deleting a profile should remove the embedding written via save()."""
    profile_id = temp_store.save(
        {
            "profile_id": "embedded",
            "name": "Embedded Voice",
            "embedding": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        }
    )
    embedding_path = Path(temp_store._embedding_path(profile_id))
    assert embedding_path.exists()

    assert temp_store.delete(profile_id) is True
    assert not embedding_path.exists()


def test_add_training_sample_copies_instrumental(temp_store, saved_profile, tmp_path):
    """Instrumental tracks should be copied alongside vocals when present."""
    vocals = tmp_path / "vocals.wav"
    instrumental = tmp_path / "instrumental.wav"
    _write_wav(vocals)
    _write_wav(instrumental)

    sample = temp_store.add_training_sample(
        profile_id=saved_profile,
        vocals_path=str(vocals),
        instrumental_path=str(instrumental),
        duration=1.25,
    )

    assert sample.instrumental_path is not None
    assert Path(sample.instrumental_path).exists()


def test_list_training_samples_skips_non_dirs_and_warns_on_bad_metadata(
    temp_store, saved_profile, caplog
):
    """Non-directory entries and corrupt sample metadata should be ignored safely."""
    profile_samples_dir = Path(temp_store._samples_dir_for_profile(saved_profile))
    profile_samples_dir.mkdir(parents=True, exist_ok=True)
    (profile_samples_dir / "notes.txt").write_text("not a directory")
    bad_sample_dir = profile_samples_dir / "sample_001"
    bad_sample_dir.mkdir()
    (bad_sample_dir / "metadata.json").write_text("{not-json")

    samples = temp_store.list_training_samples(saved_profile)

    assert samples == []
    assert "Failed to load sample metadata" in caplog.text


def test_update_sample_count_logs_warning_when_profile_refresh_fails(
    temp_store, saved_profile, caplog
):
    """Internal sample-count refresh failures should warn without raising."""
    with patch.object(temp_store, "load", side_effect=RuntimeError("boom")):
        temp_store._update_sample_count(saved_profile)

    assert "Failed to update sample count" in caplog.text


def test_match_speaker_embedding_delegates_to_diarization_matcher(
    temp_store, saved_profile
):
    """Speaker matching should return the top ranked profile above the threshold."""
    temp_store.save_speaker_embedding(saved_profile, np.array([3.0, 4.0], dtype=np.float32))
    with patch.object(
        temp_store,
        "rank_speaker_embedding_matches",
        return_value=[{"profile_id": saved_profile, "similarity": 0.9}],
    ) as rank_matches:
        result = temp_store.match_speaker_embedding(np.array([1.0, 0.0], dtype=np.float32), 0.85)

    assert result == saved_profile
    rank_matches.assert_called_once()


def test_create_profile_from_diarization_adds_metadata_and_segments(
    temp_store, tmp_path
):
    """Diarization profile creation should persist metadata and import valid segments."""
    segment = tmp_path / "segment.wav"
    _write_wav(segment)

    with patch("scipy.io.wavfile.read", return_value=(16000, np.zeros(32000, dtype=np.int16))):
        profile_id = temp_store.create_profile_from_diarization(
            name="Detected Singer",
            speaker_embedding=np.array([2.0, 0.0], dtype=np.float32),
            audio_segments=[str(segment), str(tmp_path / "missing.wav")],
            metadata={"genre": "pop"},
        )

    profile = temp_store.load(profile_id)
    samples = temp_store.list_training_samples(profile_id)

    assert profile["created_from"] == "diarization"
    assert profile["profile_role"] == "source_artist"
    assert profile["genre"] == "pop"
    assert len(samples) == 1
    assert samples[0].source_file == "segment.wav"
    assert pytest.approx(samples[0].duration) == 2.0


def test_create_profile_from_diarization_warns_on_segment_failure(
    temp_store, tmp_path, caplog
):
    """Segment import failures should be logged while still creating the profile."""
    segment = tmp_path / "broken.wav"
    _write_wav(segment)

    with patch("scipy.io.wavfile.read", side_effect=RuntimeError("bad wav")):
        profile_id = temp_store.create_profile_from_diarization(
            name="Broken Segment Singer",
            speaker_embedding=np.array([1.0, 1.0], dtype=np.float32),
            audio_segments=[str(segment)],
        )

    assert temp_store.exists(profile_id) is True
    assert temp_store.list_training_samples(profile_id) == []
    assert "Failed to add segment" in caplog.text
