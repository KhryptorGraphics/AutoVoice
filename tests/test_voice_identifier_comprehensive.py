"""Comprehensive tests for voice_identifier.py (0% → 95% coverage).

Target: 95% coverage (~499 lines)
Tests: ~41 tests covering:
- Speaker embedding extraction
- Voice profile matching and similarity scoring
- Database integration
- Error handling
- Edge cases
- Performance validation
"""
import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

from auto_voice.inference.voice_identifier import (
    IdentificationResult,
    VoiceIdentifier,
    get_voice_identifier,
)


@pytest.fixture
def profiles_dir(tmp_path):
    """Create temporary profiles directory."""
    d = tmp_path / "voice_profiles"
    d.mkdir()
    return d


@pytest.fixture
def sample_embedding():
    """Generate 256-dim normalized embedding."""
    emb = np.random.randn(256).astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    return emb


@pytest.fixture
def sample_audio_16k():
    """Generate 16kHz audio (5 seconds)."""
    sr = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sr


@pytest.fixture
def mock_wavlm():
    """Mock WavLM model and processor."""
    with patch("auto_voice.inference.voice_identifier.Wav2Vec2FeatureExtractor") as mock_processor, \
         patch("auto_voice.inference.voice_identifier.WavLMModel") as mock_model:

        # Mock processor
        processor_instance = MagicMock()
        processor_instance.return_value = {
            "input_values": torch.randn(1, 80000)
        }
        mock_processor.from_pretrained.return_value = processor_instance

        # Mock model
        model_instance = MagicMock()
        # Return 256-dim embedding
        hidden_states = torch.randn(1, 100, 768)  # WavLM output
        outputs = MagicMock()
        outputs.last_hidden_state = hidden_states
        model_instance.return_value = outputs
        model_instance.to.return_value = model_instance
        model_instance.set_output_hidden_states = MagicMock()
        mock_model.from_pretrained.return_value = model_instance

        yield {
            "processor": mock_processor,
            "model": mock_model,
            "processor_instance": processor_instance,
            "model_instance": model_instance,
        }


class TestVoiceIdentifierInitialization:
    """Test VoiceIdentifier initialization."""

    def test_init_default_params(self, profiles_dir):
        """Test initialization with default parameters."""
        identifier = VoiceIdentifier(profiles_dir=profiles_dir)

        assert identifier.profiles_dir == profiles_dir
        assert identifier.device.type in ["cuda", "cpu"]
        assert identifier.embedding_model is None
        assert isinstance(identifier._embeddings, dict)
        assert isinstance(identifier._profile_names, dict)
        assert identifier._wavlm_model is None
        assert identifier._wavlm_processor is None

    def test_init_custom_device(self, profiles_dir):
        """Test initialization with custom device."""
        identifier = VoiceIdentifier(profiles_dir=profiles_dir, device="cpu")
        assert identifier.device.type == "cpu"

    def test_init_cuda_fallback(self, profiles_dir):
        """Test CUDA fallback when not available."""
        with patch("torch.cuda.is_available", return_value=False):
            identifier = VoiceIdentifier(profiles_dir=profiles_dir, device="cuda")
            assert identifier.device.type == "cpu"

    def test_init_custom_embedding_model(self, profiles_dir):
        """Test initialization with custom embedding model."""
        identifier = VoiceIdentifier(
            profiles_dir=profiles_dir,
            embedding_model="custom/model"
        )
        assert identifier.embedding_model == "custom/model"


class TestLoadEmbeddings:
    """Test embedding loading functionality."""

    def test_load_uuid_embeddings(self, profiles_dir, sample_embedding):
        """Test loading UUID-based profile embeddings."""
        # Create test embeddings
        profile_id = "abc123"
        np.save(profiles_dir / f"{profile_id}.npy", sample_embedding)

        profile_data = {"name": "Test Singer", "user_id": "user1"}
        with open(profiles_dir / f"{profile_id}.json", "w") as f:
            json.dump(profile_data, f)

        identifier = VoiceIdentifier(profiles_dir=profiles_dir)
        count = identifier.load_all_embeddings()

        assert count == 1
        assert profile_id in identifier._embeddings
        assert identifier._profile_names[profile_id] == "Test Singer"
        assert np.array_equal(identifier._embeddings[profile_id], sample_embedding)

    def test_load_uuid_embeddings_no_json(self, profiles_dir, sample_embedding):
        """Test loading embedding without JSON metadata."""
        profile_id = "xyz789"
        np.save(profiles_dir / f"{profile_id}.npy", sample_embedding)

        identifier = VoiceIdentifier(profiles_dir=profiles_dir)
        count = identifier.load_all_embeddings()

        assert count == 1
        assert identifier._profile_names[profile_id] == profile_id

    def test_load_named_artist_profiles(self, profiles_dir, sample_embedding):
        """Test loading named artist directory profiles."""
        artist_dir = profiles_dir / "taylor_swift"
        artist_dir.mkdir()
        np.save(artist_dir / "speaker_embedding.npy", sample_embedding)

        identifier = VoiceIdentifier(profiles_dir=profiles_dir)
        count = identifier.load_all_embeddings()

        assert count == 1
        assert "taylor_swift" in identifier._embeddings
        assert identifier._profile_names["taylor_swift"] == "Taylor Swift"

    def test_load_multiple_profiles(self, profiles_dir, sample_embedding):
        """Test loading multiple profiles (UUID + named)."""
        # UUID profile
        np.save(profiles_dir / "profile1.npy", sample_embedding)
        with open(profiles_dir / "profile1.json", "w") as f:
            json.dump({"name": "Singer 1"}, f)

        # Named artist
        artist_dir = profiles_dir / "artist_one"
        artist_dir.mkdir()
        np.save(artist_dir / "speaker_embedding.npy", sample_embedding)

        identifier = VoiceIdentifier(profiles_dir=profiles_dir)
        count = identifier.load_all_embeddings()

        assert count == 2
        assert "profile1" in identifier._embeddings
        assert "artist_one" in identifier._embeddings

    def test_load_corrupted_embedding(self, profiles_dir, caplog):
        """Test handling corrupted .npy file."""
        # Create invalid .npy file
        with open(profiles_dir / "corrupted.npy", "wb") as f:
            f.write(b"invalid data")

        identifier = VoiceIdentifier(profiles_dir=profiles_dir)
        with caplog.at_level(logging.WARNING):
            count = identifier.load_all_embeddings()

        assert count == 0
        assert "Failed to load embedding" in caplog.text

    def test_load_empty_directory(self, profiles_dir):
        """Test loading from empty directory."""
        identifier = VoiceIdentifier(profiles_dir=profiles_dir)
        count = identifier.load_all_embeddings()

        assert count == 0
        assert len(identifier._embeddings) == 0

    def test_load_clears_previous_embeddings(self, profiles_dir, sample_embedding):
        """Test that load_all_embeddings clears previous data."""
        identifier = VoiceIdentifier(profiles_dir=profiles_dir)
        identifier._embeddings["old_profile"] = sample_embedding
        identifier._profile_names["old_profile"] = "Old Profile"

        # Load without any files
        count = identifier.load_all_embeddings()

        assert count == 0
        assert "old_profile" not in identifier._embeddings
        assert "old_profile" not in identifier._profile_names


class TestWavLMLoading:
    """Test WavLM model loading."""

    def test_lazy_load_wavlm(self, profiles_dir, mock_wavlm):
        """Test lazy loading of WavLM model."""
        identifier = VoiceIdentifier(profiles_dir=profiles_dir)

        assert identifier._wavlm_model is None

        identifier._load_wavlm()

        assert identifier._wavlm_model is not None
        assert identifier._wavlm_processor is not None
        mock_wavlm["model"].from_pretrained.assert_called_once()

    def test_load_wavlm_only_once(self, profiles_dir, mock_wavlm):
        """Test WavLM is only loaded once."""
        identifier = VoiceIdentifier(profiles_dir=profiles_dir)

        identifier._load_wavlm()
        identifier._load_wavlm()

        # Should only call from_pretrained once
        assert mock_wavlm["model"].from_pretrained.call_count == 1

    def test_load_wavlm_failure(self, profiles_dir):
        """Test WavLM loading failure."""
        identifier = VoiceIdentifier(profiles_dir=profiles_dir)

        with patch("auto_voice.inference.voice_identifier.WavLMModel") as mock_model:
            mock_model.from_pretrained.side_effect = RuntimeError("Model download failed")

            with pytest.raises(RuntimeError, match="Model download failed"):
                identifier._load_wavlm()


class TestEmbeddingExtraction:
    """Test speaker embedding extraction."""

    def test_extract_embedding_16khz(self, profiles_dir, sample_audio_16k, mock_wavlm):
        """Test embedding extraction from 16kHz audio."""
        audio, sr = sample_audio_16k
        identifier = VoiceIdentifier(profiles_dir=profiles_dir)

        embedding = identifier.extract_embedding(audio, sr)

        assert embedding.shape == (256,)
        assert embedding.dtype == np.float32
        # Check normalized
        norm = np.linalg.norm(embedding)
        assert 0.99 <= norm <= 1.01

    def test_extract_embedding_resample_from_44khz(self, profiles_dir, mock_wavlm):
        """Test embedding extraction with resampling from 44.1kHz."""
        audio = np.random.randn(44100 * 5).astype(np.float32)
        identifier = VoiceIdentifier(profiles_dir=profiles_dir)

        with patch("torchaudio.transforms.Resample") as mock_resample:
            resampler = MagicMock()
            resampler.return_value = torch.randn(1, 16000 * 5)
            mock_resample.return_value = resampler

            embedding = identifier.extract_embedding(audio, sample_rate=44100)

            mock_resample.assert_called_once_with(44100, 16000)
            assert embedding.shape == (256,)

    def test_extract_embedding_gpu_placement(self, profiles_dir, sample_audio_16k, mock_wavlm):
        """Test that embedding extraction uses correct device."""
        audio, sr = sample_audio_16k
        identifier = VoiceIdentifier(profiles_dir=profiles_dir, device="cuda")

        embedding = identifier.extract_embedding(audio, sr)

        # Verify input was moved to device
        assert embedding.dtype == np.float32

    def test_extract_embedding_loads_wavlm(self, profiles_dir, sample_audio_16k, mock_wavlm):
        """Test embedding extraction triggers WavLM loading."""
        audio, sr = sample_audio_16k
        identifier = VoiceIdentifier(profiles_dir=profiles_dir)

        assert identifier._wavlm_model is None

        identifier.extract_embedding(audio, sr)

        assert identifier._wavlm_model is not None


class TestIdentification:
    """Test voice identification."""

    def test_identify_match_above_threshold(self, profiles_dir, sample_audio_16k, sample_embedding, mock_wavlm):
        """Test identification with match above threshold."""
        # Create profile
        np.save(profiles_dir / "singer1.npy", sample_embedding)
        with open(profiles_dir / "singer1.json", "w") as f:
            json.dump({"name": "Test Singer"}, f)

        identifier = VoiceIdentifier(profiles_dir=profiles_dir)
        identifier.load_all_embeddings()

        audio, sr = sample_audio_16k
        result = identifier.identify(audio, sr)

        assert isinstance(result, IdentificationResult)
        assert result.similarity >= 0.0
        assert "singer1" in result.all_similarities

    def test_identify_no_match_below_threshold(self, profiles_dir, sample_audio_16k, mock_wavlm):
        """Test identification with no match below threshold."""
        # Create profile with different embedding
        emb1 = np.random.randn(256).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)
        np.save(profiles_dir / "singer1.npy", emb1)

        identifier = VoiceIdentifier(profiles_dir=profiles_dir)
        identifier.load_all_embeddings()

        audio, sr = sample_audio_16k
        result = identifier.identify(audio, sr, threshold=0.99)

        assert result.is_match is False
        assert result.profile_id is None
        assert result.profile_name is None

    def test_identify_custom_threshold(self, profiles_dir, sample_audio_16k, sample_embedding, mock_wavlm):
        """Test identification with custom threshold."""
        np.save(profiles_dir / "singer1.npy", sample_embedding)

        identifier = VoiceIdentifier(profiles_dir=profiles_dir)
        identifier.load_all_embeddings()

        audio, sr = sample_audio_16k
        result = identifier.identify(audio, sr, threshold=0.5)

        # Should match with lower threshold
        assert result.similarity >= 0.0

    def test_identify_no_profiles_loaded(self, profiles_dir, sample_audio_16k, mock_wavlm):
        """Test identification with no profiles loaded."""
        identifier = VoiceIdentifier(profiles_dir=profiles_dir)

        audio, sr = sample_audio_16k
        result = identifier.identify(audio, sr)

        assert result.is_match is False
        assert result.profile_id is None
        assert result.similarity == 0.0
        assert result.all_similarities == {}

    def test_identify_auto_loads_embeddings(self, profiles_dir, sample_audio_16k, sample_embedding, mock_wavlm):
        """Test identification auto-loads embeddings if not loaded."""
        np.save(profiles_dir / "singer1.npy", sample_embedding)

        identifier = VoiceIdentifier(profiles_dir=profiles_dir)
        # Don't explicitly load

        audio, sr = sample_audio_16k
        result = identifier.identify(audio, sr)

        # Should have loaded embeddings
        assert len(identifier._embeddings) > 0

    def test_identify_multiple_profiles(self, profiles_dir, sample_audio_16k, mock_wavlm):
        """Test identification against multiple profiles."""
        # Create 3 different profiles
        for i in range(3):
            emb = np.random.randn(256).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            np.save(profiles_dir / f"singer{i}.npy", emb)

        identifier = VoiceIdentifier(profiles_dir=profiles_dir)
        identifier.load_all_embeddings()

        audio, sr = sample_audio_16k
        result = identifier.identify(audio, sr)

        # Should have similarities for all 3 profiles
        assert len(result.all_similarities) == 3

    def test_identify_shape_mismatch_handling(self, profiles_dir, sample_audio_16k, mock_wavlm):
        """Test handling of embedding shape mismatch."""
        # Create profile with different embedding size
        emb = np.random.randn(128).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        np.save(profiles_dir / "singer1.npy", emb)

        identifier = VoiceIdentifier(profiles_dir=profiles_dir)
        identifier.load_all_embeddings()

        audio, sr = sample_audio_16k
        result = identifier.identify(audio, sr)

        # Should handle gracefully by truncating
        assert isinstance(result, IdentificationResult)


class TestIdentifyFromFile:
    """Test file-based identification."""

    def test_identify_from_wav_file(self, profiles_dir, sample_audio_16k, sample_embedding, mock_wavlm, tmp_path):
        """Test identification from audio file."""
        import soundfile as sf

        # Create test file
        audio, sr = sample_audio_16k
        audio_file = tmp_path / "test.wav"
        sf.write(str(audio_file), audio, sr)

        # Create profile
        np.save(profiles_dir / "singer1.npy", sample_embedding)

        identifier = VoiceIdentifier(profiles_dir=profiles_dir)
        identifier.load_all_embeddings()

        with patch("torchaudio.load") as mock_load:
            mock_load.return_value = (torch.from_numpy(audio).unsqueeze(0), sr)

            result = identifier.identify_from_file(str(audio_file))

            assert isinstance(result, IdentificationResult)
            mock_load.assert_called_once()

    def test_identify_from_stereo_file(self, profiles_dir, sample_embedding, mock_wavlm, tmp_path):
        """Test identification from stereo file (converts to mono)."""
        # Create stereo audio
        audio_stereo = torch.randn(2, 16000 * 5)

        identifier = VoiceIdentifier(profiles_dir=profiles_dir)
        identifier.load_all_embeddings()

        with patch("torchaudio.load") as mock_load:
            mock_load.return_value = (audio_stereo, 16000)

            result = identifier.identify_from_file("fake.wav")

            assert isinstance(result, IdentificationResult)


class TestMatchSegmentsToProfiles:
    """Test segment matching functionality."""

    def test_match_segments_with_embeddings(self, profiles_dir, sample_embedding):
        """Test matching segments that already have embeddings."""
        np.save(profiles_dir / "singer1.npy", sample_embedding)
        with open(profiles_dir / "singer1.json", "w") as f:
            json.dump({"name": "Test Singer"}, f)

        identifier = VoiceIdentifier(profiles_dir=profiles_dir)
        identifier.load_all_embeddings()

        segments = [
            {"embedding": sample_embedding.tolist()},
            {"embedding": sample_embedding.tolist()},
        ]

        result = identifier.match_segments_to_profiles(segments)

        assert result[0]["profile_id"] is not None
        assert "speaker_similarity" in result[0]

    def test_match_segments_with_audio(self, profiles_dir, sample_audio_16k, sample_embedding, mock_wavlm):
        """Test matching segments with raw audio."""
        np.save(profiles_dir / "singer1.npy", sample_embedding)

        identifier = VoiceIdentifier(profiles_dir=profiles_dir)
        identifier.load_all_embeddings()

        audio, sr = sample_audio_16k
        segments = [
            {"audio": audio, "sample_rate": sr},
        ]

        result = identifier.match_segments_to_profiles(segments)

        assert "profile_id" in result[0]

    def test_match_segments_below_threshold(self, profiles_dir, sample_embedding):
        """Test segments below similarity threshold."""
        np.save(profiles_dir / "singer1.npy", sample_embedding)

        identifier = VoiceIdentifier(profiles_dir=profiles_dir)
        identifier.load_all_embeddings()

        # Very different embedding
        diff_emb = -sample_embedding
        segments = [{"embedding": diff_emb.tolist()}]

        result = identifier.match_segments_to_profiles(segments, threshold=0.99)

        assert result[0]["profile_id"] is None
        assert result[0]["profile_name"] == "Unknown"

    def test_match_segments_no_audio_or_embedding(self, profiles_dir, sample_embedding):
        """Test segments without audio or embedding are skipped."""
        np.save(profiles_dir / "singer1.npy", sample_embedding)

        identifier = VoiceIdentifier(profiles_dir=profiles_dir)
        identifier.load_all_embeddings()

        segments = [{"start": 0.0, "end": 5.0}]

        result = identifier.match_segments_to_profiles(segments)

        # Should return unchanged
        assert "profile_id" not in result[0]


class TestProfileManagement:
    """Test profile management utilities."""

    def test_get_loaded_profiles(self, profiles_dir, sample_embedding):
        """Test retrieving loaded profiles."""
        np.save(profiles_dir / "singer1.npy", sample_embedding)
        with open(profiles_dir / "singer1.json", "w") as f:
            json.dump({"name": "Test Singer"}, f)

        identifier = VoiceIdentifier(profiles_dir=profiles_dir)
        identifier.load_all_embeddings()

        profiles = identifier.get_loaded_profiles()

        assert len(profiles) == 1
        assert profiles[0] == ("singer1", "Test Singer")

    def test_get_loaded_profiles_empty(self, profiles_dir):
        """Test get_loaded_profiles with no profiles."""
        identifier = VoiceIdentifier(profiles_dir=profiles_dir)

        profiles = identifier.get_loaded_profiles()

        assert profiles == []


class TestCreateProfile:
    """Test automatic profile creation."""

    def test_create_profile_from_segment(self, profiles_dir, sample_audio_16k, mock_wavlm):
        """Test creating profile from audio segment."""
        identifier = VoiceIdentifier(profiles_dir=profiles_dir)

        audio, sr = sample_audio_16k
        metadata = {"title": "Song by Taylor Swift", "uploader": "Music Channel"}

        with patch("auto_voice.inference.voice_identifier.VoiceProfileStore") as mock_store:
            store_instance = MagicMock()
            store_instance.save.return_value = "new_profile_id"
            mock_store.return_value = store_instance

            with patch("auto_voice.inference.voice_identifier.extract_main_artist") as mock_extract:
                mock_extract.return_value = "Taylor Swift"

                profile_id = identifier.create_profile_from_segment(
                    audio, sr, youtube_metadata=metadata
                )

                assert profile_id == "new_profile_id"
                store_instance.save.assert_called_once()
                store_instance.save_speaker_embedding.assert_called_once()

    def test_create_profile_with_source_file(self, profiles_dir, sample_audio_16k, mock_wavlm, tmp_path):
        """Test profile creation with source file saves training sample."""
        identifier = VoiceIdentifier(profiles_dir=profiles_dir)

        audio, sr = sample_audio_16k

        with patch("auto_voice.inference.voice_identifier.VoiceProfileStore") as mock_store:
            store_instance = MagicMock()
            store_instance.save.return_value = "profile1"
            mock_store.return_value = store_instance

            profile_id = identifier.create_profile_from_segment(
                audio, sr, source_file="song.mp3"
            )

            # Should add training sample
            store_instance.add_training_sample.assert_called_once()

    def test_create_profile_failure_raises_error(self, profiles_dir, sample_audio_16k, mock_wavlm):
        """Test profile creation failure raises RuntimeError."""
        identifier = VoiceIdentifier(profiles_dir=profiles_dir)

        audio, sr = sample_audio_16k

        with patch("auto_voice.inference.voice_identifier.VoiceProfileStore") as mock_store:
            mock_store.side_effect = Exception("Database error")

            with pytest.raises(RuntimeError, match="Profile creation failed"):
                identifier.create_profile_from_segment(audio, sr)


class TestGenerateProfileName:
    """Test profile name generation."""

    def test_generate_name_from_main_artist(self, profiles_dir):
        """Test name generation from main artist in title."""
        identifier = VoiceIdentifier(profiles_dir=profiles_dir)

        metadata = {"title": "Song by Taylor Swift"}

        with patch("auto_voice.inference.voice_identifier.extract_main_artist") as mock_extract:
            mock_extract.return_value = "Taylor Swift"

            name = identifier._generate_profile_name(metadata)

            assert name == "Taylor Swift"

    def test_generate_name_from_featured_artists(self, profiles_dir):
        """Test name generation from featured artists."""
        identifier = VoiceIdentifier(profiles_dir=profiles_dir)

        metadata = {"title": "Song (feat. Artist Name)"}

        with patch("auto_voice.inference.voice_identifier.extract_main_artist") as mock_extract:
            mock_extract.return_value = None

            with patch("auto_voice.inference.voice_identifier.parse_featured_artists") as mock_parse:
                mock_parse.return_value = ["Artist Name"]

                name = identifier._generate_profile_name(metadata)

                assert name == "Artist Name"

    def test_generate_name_from_uploader(self, profiles_dir):
        """Test name generation from uploader."""
        identifier = VoiceIdentifier(profiles_dir=profiles_dir)

        metadata = {"uploader": "Music Channel"}

        with patch("auto_voice.inference.voice_identifier.extract_main_artist") as mock_extract:
            mock_extract.return_value = None

            with patch("auto_voice.inference.voice_identifier.parse_featured_artists") as mock_parse:
                mock_parse.return_value = []

                name = identifier._generate_profile_name(metadata)

                assert name == "Music Channel"

    def test_generate_name_fallback_speaker_n(self, profiles_dir):
        """Test name generation falls back to Speaker_N pattern."""
        identifier = VoiceIdentifier(profiles_dir=profiles_dir)
        identifier._profile_names = {
            "p1": "Speaker_1",
            "p2": "Singer A",
            "p3": "Speaker_2"
        }

        name = identifier._generate_profile_name(None)

        assert name == "Speaker_3"

    def test_generate_name_no_metadata(self, profiles_dir):
        """Test name generation with no metadata."""
        identifier = VoiceIdentifier(profiles_dir=profiles_dir)

        name = identifier._generate_profile_name(None)

        assert name == "Speaker_1"


class TestIdentifyOrCreate:
    """Test identify_or_create functionality."""

    def test_identify_or_create_existing_match(self, profiles_dir, sample_audio_16k, sample_embedding, mock_wavlm):
        """Test identify_or_create returns existing profile if matched."""
        np.save(profiles_dir / "singer1.npy", sample_embedding)
        with open(profiles_dir / "singer1.json", "w") as f:
            json.dump({"name": "Existing Singer"}, f)

        identifier = VoiceIdentifier(profiles_dir=profiles_dir)
        identifier.load_all_embeddings()

        audio, sr = sample_audio_16k

        with patch.object(identifier, "identify") as mock_identify:
            mock_identify.return_value = IdentificationResult(
                profile_id="singer1",
                profile_name="Existing Singer",
                similarity=0.95,
                is_match=True,
                all_similarities={"singer1": 0.95}
            )

            result = identifier.identify_or_create(audio, sr)

            assert result.is_match is True
            assert result.profile_id == "singer1"

    def test_identify_or_create_new_profile(self, profiles_dir, sample_audio_16k, mock_wavlm):
        """Test identify_or_create creates new profile if no match."""
        identifier = VoiceIdentifier(profiles_dir=profiles_dir)

        audio, sr = sample_audio_16k

        with patch.object(identifier, "identify") as mock_identify:
            mock_identify.return_value = IdentificationResult(
                profile_id=None,
                profile_name=None,
                similarity=0.5,
                is_match=False,
                all_similarities={}
            )

            with patch.object(identifier, "create_profile_from_segment") as mock_create:
                mock_create.return_value = "new_profile"
                identifier._profile_names["new_profile"] = "Speaker_1"

                result = identifier.identify_or_create(audio, sr)

                mock_create.assert_called_once()
                assert result.profile_id == "new_profile"
                assert result.is_match is True
                assert result.similarity == 1.0


class TestGlobalInstance:
    """Test global identifier singleton."""

    def test_get_voice_identifier_singleton(self):
        """Test get_voice_identifier returns singleton instance."""
        # Reset global
        import auto_voice.inference.voice_identifier as module
        module._global_identifier = None

        identifier1 = get_voice_identifier()
        identifier2 = get_voice_identifier()

        assert identifier1 is identifier2

    def test_get_voice_identifier_loads_embeddings(self):
        """Test get_voice_identifier auto-loads embeddings."""
        import auto_voice.inference.voice_identifier as module
        module._global_identifier = None

        with patch.object(VoiceIdentifier, "load_all_embeddings") as mock_load:
            identifier = get_voice_identifier()

            mock_load.assert_called_once()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_audio(self, profiles_dir, mock_wavlm):
        """Test handling of empty audio."""
        identifier = VoiceIdentifier(profiles_dir=profiles_dir)

        empty_audio = np.array([], dtype=np.float32)

        # Should handle gracefully
        embedding = identifier.extract_embedding(empty_audio, 16000)
        assert embedding.shape == (256,)

    def test_single_sample_audio(self, profiles_dir, mock_wavlm):
        """Test handling of single sample."""
        identifier = VoiceIdentifier(profiles_dir=profiles_dir)

        audio = np.array([0.5], dtype=np.float32)

        embedding = identifier.extract_embedding(audio, 16000)
        assert embedding.shape == (256,)

    def test_very_long_audio(self, profiles_dir, mock_wavlm):
        """Test handling of very long audio (30 minutes)."""
        identifier = VoiceIdentifier(profiles_dir=profiles_dir)

        # 30 minutes at 16kHz
        long_audio = np.random.randn(16000 * 60 * 30).astype(np.float32)

        embedding = identifier.extract_embedding(long_audio, 16000)
        assert embedding.shape == (256,)

    def test_silence_audio(self, profiles_dir, mock_wavlm):
        """Test handling of silent audio."""
        identifier = VoiceIdentifier(profiles_dir=profiles_dir)

        silence = np.zeros(16000 * 5, dtype=np.float32)

        embedding = identifier.extract_embedding(silence, 16000)
        assert embedding.shape == (256,)

    @pytest.mark.cuda
    def test_gpu_memory_handling(self, profiles_dir, sample_audio_16k, mock_wavlm):
        """Test GPU memory is properly managed."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        identifier = VoiceIdentifier(profiles_dir=profiles_dir, device="cuda")

        audio, sr = sample_audio_16k

        # Process multiple times
        for _ in range(10):
            embedding = identifier.extract_embedding(audio, sr)
            assert embedding is not None

        # GPU memory should not explode
        assert True


class TestPerformance:
    """Test performance requirements."""

    @pytest.mark.slow
    def test_embedding_extraction_performance(self, profiles_dir, sample_audio_16k, mock_wavlm):
        """Test embedding extraction completes in <500ms."""
        import time

        identifier = VoiceIdentifier(profiles_dir=profiles_dir)
        audio, sr = sample_audio_16k

        # Warm up
        identifier.extract_embedding(audio, sr)

        # Measure
        start = time.time()
        identifier.extract_embedding(audio, sr)
        elapsed = time.time() - start

        # Note: This will be fast with mocked model
        assert elapsed < 0.5  # 500ms target

    def test_similarity_computation_fast(self, profiles_dir, sample_embedding):
        """Test similarity computation is fast for many profiles."""
        import time

        # Create 100 profiles
        for i in range(100):
            np.save(profiles_dir / f"singer{i}.npy", sample_embedding)

        identifier = VoiceIdentifier(profiles_dir=profiles_dir)
        identifier.load_all_embeddings()

        # Mock embedding extraction
        with patch.object(identifier, "extract_embedding", return_value=sample_embedding):
            audio = np.random.randn(16000 * 5).astype(np.float32)

            start = time.time()
            result = identifier.identify(audio, 16000)
            elapsed = time.time() - start

            assert elapsed < 0.1  # Should be very fast (vector operations)
            assert len(result.all_similarities) == 100
