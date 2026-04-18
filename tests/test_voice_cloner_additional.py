"""Targeted branch coverage for inference.voice_cloner."""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from auto_voice.inference import voice_cloner as voice_cloner_module
from auto_voice.inference.voice_cloner import InvalidAudioError, VoiceCloner, _get_vocal_separator


def _write_wav(path: Path, duration_seconds: float = 5.0, sample_rate: int = 16000) -> None:
    """Create a sine-wave WAV file for tests."""
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    audio = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    sf.write(str(path), audio, sample_rate)


@pytest.fixture
def audio_file(tmp_path):
    path = tmp_path / "sample.wav"
    _write_wav(path)
    return str(path)


@pytest.fixture
def short_audio_file(tmp_path):
    path = tmp_path / "short.wav"
    _write_wav(path, duration_seconds=1.0)
    return str(path)


@pytest.fixture
def cloner(tmp_path):
    return VoiceCloner(
        profiles_dir=str(tmp_path / "profiles"),
        samples_dir=str(tmp_path / "samples"),
        auto_separate_vocals=True,
    )


def test_get_vocal_separator_returns_none_when_backend_unavailable():
    """Lazy separator should degrade cleanly when construction fails."""
    voice_cloner_module._vocal_separator = None
    fake_module = types.SimpleNamespace(VocalSeparator=MagicMock(side_effect=RuntimeError("no demucs")))

    with patch.dict(sys.modules, {"auto_voice.audio.separation": fake_module}):
        separator = _get_vocal_separator("cpu")

    assert separator is None


def test_extract_mel_stat_embedding_short_audio_hits_duration_guard(cloner, short_audio_file):
    """The direct mel-stat path should enforce the minimum duration check."""
    with pytest.raises(InvalidAudioError, match="too short"):
        cloner._extract_mel_stat_embedding(short_audio_file)


def test_extract_embedding_uses_ecapa2_backend_when_available(tmp_path, audio_file):
    """ECAPA2 embeddings should be used directly when the backend loads."""
    embedding = np.array([0.25, 0.75], dtype=np.float32)
    encoder = MagicMock()
    encoder.extract_embedding.return_value = types.SimpleNamespace(
        backend="ecapa2",
        embedding=embedding,
    )
    fake_module = types.SimpleNamespace(ECAPA2SpeakerEncoder=MagicMock(return_value=encoder))
    cloner = VoiceCloner(
        profiles_dir=str(tmp_path / "profiles"),
        samples_dir=str(tmp_path / "samples"),
        speaker_encoder_backend="ecapa2",
    )

    with patch.dict(sys.modules, {"auto_voice.models.ecapa2_encoder": fake_module}):
        result = cloner._extract_embedding(audio_file)

    assert np.array_equal(result, embedding)


def test_extract_embedding_falls_back_to_mel_stats_when_ecapa2_fails(tmp_path, audio_file):
    """ECAPA2 failures should fall back to mel-statistics embedding extraction."""
    failing_encoder = MagicMock()
    failing_encoder.extract_embedding.side_effect = RuntimeError("encoder offline")
    fake_module = types.SimpleNamespace(ECAPA2SpeakerEncoder=MagicMock(return_value=failing_encoder))
    cloner = VoiceCloner(
        profiles_dir=str(tmp_path / "profiles"),
        samples_dir=str(tmp_path / "samples"),
        speaker_encoder_backend="ecapa2",
    )

    with patch.dict(sys.modules, {"auto_voice.models.ecapa2_encoder": fake_module}):
        with patch.object(cloner, "_extract_mel_stat_embedding", return_value=np.array([1.0], dtype=np.float32)) as mel:
            result = cloner._extract_embedding(audio_file)

    assert np.array_equal(result, np.array([1.0], dtype=np.float32))
    mel.assert_called_once_with(audio_file)


def test_create_speaker_embedding_raises_when_average_is_zero(cloner):
    """Averaging zero embeddings should fail with InvalidAudioError."""
    with patch.object(cloner, "_extract_embedding", side_effect=[np.zeros(2), np.zeros(2)]):
        with pytest.raises(InvalidAudioError, match="All embeddings are zero"):
            cloner.create_speaker_embedding(["a.wav", "b.wav"])


def test_estimate_vocal_range_returns_defaults_on_pitch_failure(cloner, audio_file):
    """Pitch tracking failures should return the conservative default vocal range."""
    with patch("librosa.pyin", side_effect=RuntimeError("bad pitch")):
        result = cloner._estimate_vocal_range(audio_file)

    assert result == {"min_hz": 80.0, "max_hz": 800.0, "mean_hz": 200.0}


def test_extract_vocals_handles_disabled_separator_and_runtime_failure(cloner, audio_file):
    """Extraction should return None when disabled or when separation crashes."""
    cloner._auto_separate_vocals = False
    assert cloner._extract_vocals(audio_file, "profile-1") is None

    cloner._auto_separate_vocals = True
    fake_separator = MagicMock()
    fake_separator.separate.side_effect = RuntimeError("broken")
    with patch("auto_voice.inference.voice_cloner._get_vocal_separator", return_value=fake_separator):
        assert cloner._extract_vocals(audio_file, "profile-1") is None


def test_extract_vocals_saves_stems_when_separator_succeeds(cloner, audio_file):
    """Successful vocal separation should persist both stem files."""
    fake_separator = MagicMock()
    fake_separator.separate.return_value = {
        "vocals": np.array([0.1, 0.2], dtype=np.float32),
        "instrumental": np.array([0.3, 0.4], dtype=np.float32),
    }

    with patch("auto_voice.inference.voice_cloner._get_vocal_separator", return_value=fake_separator):
        stems = cloner._extract_vocals(audio_file, "profile-1")

    assert Path(stems["vocals"]).exists()
    assert Path(stems["instrumental"]).exists()


def test_create_voice_profile_handles_missing_separation_and_duration_fallback(cloner, audio_file):
    """Profile creation should still work when separation and duration probing fail."""
    with patch.object(cloner, "_extract_vocals", return_value=None):
        with patch.object(cloner, "_extract_embedding", return_value=np.ones(256, dtype=np.float32)):
            with patch("librosa.load", side_effect=RuntimeError("probe failed")):
                with patch.object(
                    cloner,
                    "_estimate_vocal_range",
                    return_value={"min_hz": 100.0, "max_hz": 500.0, "mean_hz": 250.0},
                ):
                    result = cloner.create_voice_profile(audio_file, user_id="user-1", name="Voice")

    assert result["vocals_extracted"] is False
    assert result["audio_duration"] == 0.0
    assert result["training_sample_count"] == 1


def test_compare_embeddings_returns_zero_for_zero_vector(cloner):
    """Zero vectors should short-circuit to 0 similarity."""
    assert cloner.compare_embeddings(np.zeros(2), np.array([1.0, 0.0])) == 0.0


def test_add_vocal_sample_handles_missing_audio_separation_failure_and_duration_fallback(cloner, audio_file):
    """Sample ingestion should cover not-found, no-separation, and duration-fallback branches."""
    with pytest.raises(InvalidAudioError, match="Audio file not found"):
        cloner.add_vocal_sample("profile-1", "/missing.wav")

    with patch.object(cloner, "_extract_vocals", return_value=None):
        assert cloner.add_vocal_sample("profile-1", audio_file) is None

    separated = {"vocals": audio_file, "instrumental": audio_file}
    with patch.object(cloner, "_extract_vocals", return_value=separated):
        with patch("librosa.load", side_effect=RuntimeError("duration failed")):
            with patch.object(cloner.store, "add_training_sample", return_value="sample-1") as add_sample:
                result = cloner.add_vocal_sample("profile-1", audio_file, source_name="demo")

    assert result == "sample-1"
    assert add_sample.call_args.kwargs["duration"] == 0.0


def test_training_sample_accessors_delegate_to_store(cloner):
    """Getter helpers should forward directly to the backing store."""
    sample = MagicMock()
    cloner.store.list_training_samples = MagicMock(return_value=[sample])
    cloner.store.get_all_vocals_paths = MagicMock(return_value=["a.wav"])
    cloner.store.get_total_training_duration = MagicMock(return_value=12.5)

    assert cloner.get_training_samples("profile-1") == [sample]
    assert cloner.get_training_audio_paths("profile-1") == ["a.wav"]
    assert cloner.get_training_duration("profile-1") == 12.5
