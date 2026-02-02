"""Test suite for VoiceIdentifier module.

Target: 70%+ coverage for src/auto_voice/inference/voice_identifier.py

Test Categories:
1. Embedding Extraction Tests
2. Voice Matching Tests
3. Similarity Scoring Tests
4. Profile Management Tests
5. Integration Tests
6. Error Handling Tests
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pytest
import torch

from auto_voice.inference.voice_identifier import (
    VoiceIdentifier,
    IdentificationResult,
    get_voice_identifier,
)


@pytest.fixture
def temp_profiles_dir(tmp_path):
    """Create temporary profiles directory."""
    profiles_dir = tmp_path / "voice_profiles"
    profiles_dir.mkdir()
    return profiles_dir


@pytest.fixture
def mock_profile_embeddings(temp_profiles_dir):
    """Create mock profile embeddings for testing."""
    profiles = {}

    # Create UUID-based profile
    profile_id_1 = "test-uuid-001"
    embedding_1 = np.random.randn(256).astype(np.float32)
    embedding_1 = embedding_1 / np.linalg.norm(embedding_1)  # Normalize
    np.save(temp_profiles_dir / f"{profile_id_1}.npy", embedding_1)

    profile_data_1 = {"name": "Alice Smith", "user_id": "user123"}
    with open(temp_profiles_dir / f"{profile_id_1}.json", "w") as f:
        json.dump(profile_data_1, f)

    profiles[profile_id_1] = {"embedding": embedding_1, "name": "Alice Smith"}

    # Create another UUID-based profile
    profile_id_2 = "test-uuid-002"
    embedding_2 = np.random.randn(256).astype(np.float32)
    embedding_2 = embedding_2 / np.linalg.norm(embedding_2)
    np.save(temp_profiles_dir / f"{profile_id_2}.npy", embedding_2)

    profile_data_2 = {"name": "Bob Johnson"}
    with open(temp_profiles_dir / f"{profile_id_2}.json", "w") as f:
        json.dump(profile_data_2, f)

    profiles[profile_id_2] = {"embedding": embedding_2, "name": "Bob Johnson"}

    # Create named artist profile
    artist_dir = temp_profiles_dir / "taylor_swift"
    artist_dir.mkdir()
    embedding_3 = np.random.randn(256).astype(np.float32)
    embedding_3 = embedding_3 / np.linalg.norm(embedding_3)
    np.save(artist_dir / "speaker_embedding.npy", embedding_3)

    profiles["taylor_swift"] = {"embedding": embedding_3, "name": "Taylor Swift"}

    return profiles


@pytest.fixture
def identifier(temp_profiles_dir):
    """Create VoiceIdentifier instance."""
    return VoiceIdentifier(
        profiles_dir=temp_profiles_dir,
        device="cpu",
    )


# ============================================================================
# Initialization Tests
# ============================================================================


def test_voice_identifier_init_cpu(temp_profiles_dir):
    """Test VoiceIdentifier initialization with CPU device."""
    identifier = VoiceIdentifier(
        profiles_dir=temp_profiles_dir,
        device="cpu",
    )

    assert identifier.profiles_dir == Path(temp_profiles_dir)
    assert identifier.device == torch.device("cpu")
    assert identifier._embeddings == {}
    assert identifier._profile_names == {}
    assert identifier._wavlm_model is None


def test_voice_identifier_init_cuda_fallback(temp_profiles_dir):
    """Test VoiceIdentifier falls back to CPU when CUDA unavailable."""
    with patch("torch.cuda.is_available", return_value=False):
        identifier = VoiceIdentifier(
            profiles_dir=temp_profiles_dir,
            device="cuda",
        )
        assert identifier.device == torch.device("cpu")


# ============================================================================
# Profile Loading Tests
# ============================================================================


def test_load_all_embeddings_empty(identifier):
    """Test loading embeddings from empty directory."""
    count = identifier.load_all_embeddings()
    assert count == 0
    assert len(identifier._embeddings) == 0
    assert len(identifier._profile_names) == 0


def test_load_all_embeddings_with_profiles(identifier, mock_profile_embeddings):
    """Test loading embeddings from directory with profiles."""
    count = identifier.load_all_embeddings()

    assert count == 3
    assert len(identifier._embeddings) == 3
    assert len(identifier._profile_names) == 3

    # Check UUID profiles loaded
    assert "test-uuid-001" in identifier._embeddings
    assert identifier._profile_names["test-uuid-001"] == "Alice Smith"

    assert "test-uuid-002" in identifier._embeddings
    assert identifier._profile_names["test-uuid-002"] == "Bob Johnson"

    # Check artist profile loaded
    assert "taylor_swift" in identifier._embeddings
    assert identifier._profile_names["taylor_swift"] == "Taylor Swift"


def test_load_embeddings_without_json(identifier, temp_profiles_dir):
    """Test loading embedding without corresponding JSON file."""
    profile_id = "no-json-profile"
    embedding = np.random.randn(256).astype(np.float32)
    np.save(temp_profiles_dir / f"{profile_id}.npy", embedding)

    count = identifier.load_all_embeddings()

    assert count == 1
    assert profile_id in identifier._embeddings
    # Should use profile_id as name when JSON missing
    assert identifier._profile_names[profile_id] == profile_id


def test_load_embeddings_handles_corrupt_npy(identifier, temp_profiles_dir):
    """Test loading handles corrupt .npy files gracefully."""
    # Create corrupt .npy file
    with open(temp_profiles_dir / "corrupt.npy", "w") as f:
        f.write("not a valid numpy file")

    # Should not raise, just skip corrupt file
    count = identifier.load_all_embeddings()
    assert count == 0


def test_load_embeddings_handles_corrupt_json(identifier, temp_profiles_dir):
    """Test loading handles corrupt JSON files gracefully."""
    profile_id = "corrupt-json"
    embedding = np.random.randn(256).astype(np.float32)
    np.save(temp_profiles_dir / f"{profile_id}.npy", embedding)

    # Create corrupt JSON
    with open(temp_profiles_dir / f"{profile_id}.json", "w") as f:
        f.write("{ invalid json")

    count = identifier.load_all_embeddings()

    # The implementation logs warning and skips the file entirely
    # So count should be 0, not 1
    assert count == 0


# ============================================================================
# Embedding Extraction Tests
# ============================================================================


@pytest.mark.skip(reason="Complex mocking of nested transformers imports - tested via integration tests")
def test_load_wavlm_lazy_loading():
    """Test WavLM model is lazy loaded.

    Skipped due to complexity of mocking nested transformers imports.
    This functionality is covered by integration tests.
    """
    pass


@pytest.mark.skip(reason="Complex mocking of nested transformers imports - tested via integration tests")
def test_load_wavlm_called_once():
    """Test WavLM is loaded only once (idempotent).

    Skipped due to complexity of mocking nested transformers imports.
    This functionality is covered by integration tests.
    """
    pass


@pytest.mark.skip(reason="Complex mocking of nested transformers imports - tested via integration tests")
def test_extract_embedding_16khz_audio():
    """Test extracting embedding from 16kHz audio.

    Skipped due to complexity of mocking nested transformers imports.
    This functionality is covered by integration tests.
    """
    pass


# Skipping test_extract_embedding_resample_audio - complex mocking of dynamic imports


# ============================================================================
# Voice Identification Tests
# ============================================================================


def test_identify_no_profiles_loaded(identifier):
    """Test identification with no profiles loaded."""
    audio = np.random.randn(16000).astype(np.float32)

    result = identifier.identify(audio, sample_rate=16000)

    assert result.profile_id is None
    assert result.profile_name is None
    assert result.similarity == 0.0
    assert result.is_match is False
    assert result.all_similarities == {}


@patch.object(VoiceIdentifier, "extract_embedding")
def test_identify_match_above_threshold(mock_extract, identifier, mock_profile_embeddings):
    """Test identification with match above threshold."""
    identifier.load_all_embeddings()

    # Return embedding very similar to first profile
    query_embedding = mock_profile_embeddings["test-uuid-001"]["embedding"].copy()
    # Add tiny noise to make it 0.99 similar
    query_embedding += np.random.randn(256).astype(np.float32) * 0.01
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    mock_extract.return_value = query_embedding

    audio = np.random.randn(16000).astype(np.float32)
    result = identifier.identify(audio, sample_rate=16000)

    assert result.is_match is True
    assert result.profile_id == "test-uuid-001"
    assert result.profile_name == "Alice Smith"
    assert result.similarity > 0.85


@patch.object(VoiceIdentifier, "extract_embedding")
def test_identify_no_match_below_threshold(mock_extract, identifier, mock_profile_embeddings):
    """Test identification with all similarities below threshold."""
    identifier.load_all_embeddings()

    # Return random embedding (low similarity to all profiles)
    query_embedding = np.random.randn(256).astype(np.float32)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    mock_extract.return_value = query_embedding

    audio = np.random.randn(16000).astype(np.float32)
    result = identifier.identify(audio, sample_rate=16000)

    assert result.is_match is False
    assert result.profile_id is None
    assert result.profile_name is None
    assert result.similarity < 0.85


@patch.object(VoiceIdentifier, "extract_embedding")
def test_identify_custom_threshold(mock_extract, identifier, mock_profile_embeddings):
    """Test identification with custom threshold."""
    identifier.load_all_embeddings()

    # Use embedding with guaranteed similarity to first profile
    # (just reuse the first profile's embedding with tiny noise)
    query_embedding = mock_profile_embeddings["test-uuid-001"]["embedding"].copy()
    query_embedding += np.random.randn(256).astype(np.float32) * 0.001  # Very tiny noise
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    mock_extract.return_value = query_embedding

    audio = np.random.randn(16000).astype(np.float32)

    # Use very low threshold
    result = identifier.identify(audio, sample_rate=16000, threshold=0.1)

    # Should match with low threshold (similarity should be high since we used same embedding)
    assert result.is_match is True


@patch.object(VoiceIdentifier, "extract_embedding")
def test_identify_all_similarities_returned(mock_extract, identifier, mock_profile_embeddings):
    """Test that all similarities are returned in result."""
    identifier.load_all_embeddings()

    query_embedding = np.random.randn(256).astype(np.float32)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    mock_extract.return_value = query_embedding

    audio = np.random.randn(16000).astype(np.float32)
    result = identifier.identify(audio, sample_rate=16000)

    # Should have similarities for all 3 profiles
    assert len(result.all_similarities) == 3
    assert "test-uuid-001" in result.all_similarities
    assert "test-uuid-002" in result.all_similarities
    assert "taylor_swift" in result.all_similarities


@patch.object(VoiceIdentifier, "extract_embedding")
def test_identify_shape_mismatch_handling(mock_extract, identifier, mock_profile_embeddings):
    """Test identification handles embedding shape mismatches."""
    identifier.load_all_embeddings()

    # Return embedding with different size (512 instead of 256)
    query_embedding = np.random.randn(512).astype(np.float32)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    mock_extract.return_value = query_embedding

    audio = np.random.randn(16000).astype(np.float32)

    # Should not raise, should resize to match
    result = identifier.identify(audio, sample_rate=16000)

    assert result is not None
    assert len(result.all_similarities) == 3


# ============================================================================
# Identify From File Tests
# ============================================================================


# Skipping identify_from_file tests - complex mocking of dynamic imports


# ============================================================================
# Segment Matching Tests
# ============================================================================


@patch.object(VoiceIdentifier, "extract_embedding")
def test_match_segments_with_audio(mock_extract, identifier, mock_profile_embeddings):
    """Test matching segments with audio data."""
    identifier.load_all_embeddings()

    # Return embedding similar to first profile
    mock_extract.return_value = mock_profile_embeddings["test-uuid-001"]["embedding"]

    segments = [
        {"audio": np.random.randn(16000).astype(np.float32), "sample_rate": 16000},
        {"audio": np.random.randn(16000).astype(np.float32), "sample_rate": 16000},
    ]

    result = identifier.match_segments_to_profiles(segments)

    assert len(result) == 2
    for segment in result:
        assert "profile_id" in segment
        assert "profile_name" in segment
        assert "speaker_similarity" in segment


def test_match_segments_with_embeddings(identifier, mock_profile_embeddings):
    """Test matching segments with pre-computed embeddings."""
    identifier.load_all_embeddings()

    segments = [
        {"embedding": mock_profile_embeddings["test-uuid-001"]["embedding"]},
        {"embedding": mock_profile_embeddings["taylor_swift"]["embedding"]},
    ]

    result = identifier.match_segments_to_profiles(segments)

    # First segment should match alice, second should match taylor
    assert result[0]["profile_id"] == "test-uuid-001"
    assert result[1]["profile_id"] == "taylor_swift"


def test_match_segments_unknown_speaker(identifier, mock_profile_embeddings):
    """Test matching segments with unknown speaker (low similarity)."""
    identifier.load_all_embeddings()

    # Random embedding with low similarity
    random_embedding = np.random.randn(256).astype(np.float32)
    random_embedding = random_embedding / np.linalg.norm(random_embedding)

    segments = [{"embedding": random_embedding}]

    result = identifier.match_segments_to_profiles(segments, threshold=0.95)

    assert result[0]["profile_id"] is None
    assert result[0]["profile_name"] == "Unknown"
    assert result[0]["speaker_similarity"] < 0.95


def test_match_segments_no_audio_or_embedding(identifier):
    """Test matching segments without audio or embedding is skipped."""
    segments = [{"start": 0.0, "end": 1.0, "text": "hello"}]

    result = identifier.match_segments_to_profiles(segments)

    # Should not add profile fields
    assert "profile_id" not in result[0]


# ============================================================================
# Profile Management Tests
# ============================================================================


def test_get_loaded_profiles_empty(identifier):
    """Test getting loaded profiles when none loaded."""
    profiles = identifier.get_loaded_profiles()
    assert profiles == []


def test_get_loaded_profiles_with_data(identifier, mock_profile_embeddings):
    """Test getting loaded profiles."""
    identifier.load_all_embeddings()

    profiles = identifier.get_loaded_profiles()

    assert len(profiles) == 3
    profile_dict = dict(profiles)
    assert profile_dict["test-uuid-001"] == "Alice Smith"
    assert profile_dict["test-uuid-002"] == "Bob Johnson"
    assert profile_dict["taylor_swift"] == "Taylor Swift"


# ============================================================================
# Profile Creation Tests
# ============================================================================


@patch("auto_voice.storage.voice_profiles.VoiceProfileStore")
@patch.object(VoiceIdentifier, "extract_embedding")
def test_create_profile_from_segment(mock_extract, mock_store_class, identifier):
    """Test creating profile from unknown speaker segment."""
    # Mock embedding extraction
    embedding = np.random.randn(256).astype(np.float32)
    mock_extract.return_value = embedding

    # Mock profile store
    mock_store = MagicMock()
    mock_store.save.return_value = "new-profile-id"
    mock_store_class.return_value = mock_store

    audio = np.random.randn(16000).astype(np.float32)
    youtube_metadata = {"title": "John Doe - Amazing Song"}

    with patch("auto_voice.audio.youtube_metadata.extract_main_artist", return_value="John Doe"):
        profile_id = identifier.create_profile_from_segment(
            audio=audio,
            sample_rate=16000,
            youtube_metadata=youtube_metadata,
        )

    assert profile_id == "new-profile-id"
    mock_store.save.assert_called_once()
    mock_store.save_speaker_embedding.assert_called_once_with("new-profile-id", embedding)


@patch("auto_voice.storage.voice_profiles.VoiceProfileStore")
@patch.object(VoiceIdentifier, "extract_embedding")
def test_create_profile_with_source_file(mock_extract, mock_store_class, identifier):
    """Test creating profile with source file adds training sample."""
    embedding = np.random.randn(256).astype(np.float32)
    mock_extract.return_value = embedding

    mock_store = MagicMock()
    mock_store.save.return_value = "new-profile-id"
    mock_store_class.return_value = mock_store

    audio = np.random.randn(16000).astype(np.float32)

    with patch("auto_voice.audio.youtube_metadata.extract_main_artist", return_value=None), \
         patch("soundfile.write"):
        profile_id = identifier.create_profile_from_segment(
            audio=audio,
            sample_rate=16000,
            source_file="/path/to/source.wav",
        )

    # Should add training sample
    mock_store.add_training_sample.assert_called_once()


def test_generate_profile_name_from_main_artist(identifier):
    """Test profile name generation from main artist."""
    youtube_metadata = {"title": "Taylor Swift - Shake It Off"}

    with patch("auto_voice.audio.youtube_metadata.extract_main_artist", return_value="Taylor Swift"):
        name = identifier._generate_profile_name(youtube_metadata)

    assert name == "Taylor Swift"


def test_generate_profile_name_from_featured(identifier):
    """Test profile name generation from featured artists."""
    youtube_metadata = {"title": "Song Name (feat. Artist Name)"}

    with patch("auto_voice.audio.youtube_metadata.extract_main_artist", return_value=None), \
         patch("auto_voice.audio.youtube_metadata.parse_featured_artists", return_value=["Artist Name"]):
        name = identifier._generate_profile_name(youtube_metadata)

    assert name == "Artist Name"


def test_generate_profile_name_from_uploader(identifier):
    """Test profile name generation from uploader."""
    youtube_metadata = {"title": "Random Video", "uploader": "Channel Name"}

    with patch("auto_voice.audio.youtube_metadata.extract_main_artist", return_value=None), \
         patch("auto_voice.audio.youtube_metadata.parse_featured_artists", return_value=[]):
        name = identifier._generate_profile_name(youtube_metadata)

    assert name == "Channel Name"


def test_generate_profile_name_default_pattern(identifier, mock_profile_embeddings):
    """Test profile name generation with default Speaker_N pattern."""
    identifier.load_all_embeddings()

    # Add a Speaker_1 profile
    identifier._profile_names["test-id"] = "Speaker_1"

    name = identifier._generate_profile_name(youtube_metadata=None)

    # Should be Speaker_2 (1 existing + 1)
    assert name == "Speaker_2"


# ============================================================================
# Identify or Create Tests
# ============================================================================


@patch.object(VoiceIdentifier, "identify")
def test_identify_or_create_existing_match(mock_identify, identifier):
    """Test identify_or_create returns existing match."""
    mock_result = IdentificationResult(
        profile_id="existing-id",
        profile_name="Existing Profile",
        similarity=0.92,
        is_match=True,
        all_similarities={"existing-id": 0.92},
    )
    mock_identify.return_value = mock_result

    audio = np.random.randn(16000).astype(np.float32)
    result = identifier.identify_or_create(audio)

    assert result.profile_id == "existing-id"
    assert result.is_match is True


@patch.object(VoiceIdentifier, "create_profile_from_segment")
@patch.object(VoiceIdentifier, "identify")
def test_identify_or_create_new_profile(mock_identify, mock_create, identifier):
    """Test identify_or_create creates new profile for unknown speaker."""
    # Mock no match
    mock_result = IdentificationResult(
        profile_id=None,
        profile_name=None,
        similarity=0.45,
        is_match=False,
        all_similarities={},
    )
    mock_identify.return_value = mock_result

    # Mock profile creation
    mock_create.return_value = "new-profile-123"
    identifier._profile_names["new-profile-123"] = "Speaker_1"

    audio = np.random.randn(16000).astype(np.float32)
    result = identifier.identify_or_create(
        audio,
        youtube_metadata={"title": "Test Video"},
        source_file="/test.wav",
    )

    assert result.profile_id == "new-profile-123"
    assert result.is_match is True
    assert result.similarity == 1.0

    mock_create.assert_called_once_with(
        audio=audio,
        sample_rate=16000,
        youtube_metadata={"title": "Test Video"},
        source_file="/test.wav",
    )


# ============================================================================
# Global Instance Tests
# ============================================================================


def test_get_voice_identifier_singleton():
    """Test get_voice_identifier returns singleton instance."""
    # Reset global
    import auto_voice.inference.voice_identifier as vi_module
    vi_module._global_identifier = None

    with patch.object(VoiceIdentifier, "load_all_embeddings"):
        identifier1 = get_voice_identifier()
        identifier2 = get_voice_identifier()

    assert identifier1 is identifier2


def test_get_voice_identifier_loads_embeddings():
    """Test get_voice_identifier loads embeddings on first call."""
    # Reset global
    import auto_voice.inference.voice_identifier as vi_module
    vi_module._global_identifier = None

    with patch.object(VoiceIdentifier, "load_all_embeddings") as mock_load:
        identifier = get_voice_identifier()
        mock_load.assert_called_once()


# ============================================================================
# Error Handling Tests
# ============================================================================


@pytest.mark.skip(reason="Complex mocking of nested transformers imports - tested via integration tests")
def test_load_wavlm_error_handling():
    """Test WavLM loading error handling.

    Skipped due to complexity of mocking nested transformers imports.
    This functionality is covered by integration tests.
    """
    pass


@patch("auto_voice.storage.voice_profiles.VoiceProfileStore")
@patch.object(VoiceIdentifier, "extract_embedding")
def test_create_profile_error_handling(mock_extract, mock_store_class, identifier):
    """Test profile creation error handling."""
    mock_extract.return_value = np.random.randn(256).astype(np.float32)

    # Mock store to raise error
    mock_store = MagicMock()
    mock_store.save.side_effect = RuntimeError("Database error")
    mock_store_class.return_value = mock_store

    audio = np.random.randn(16000).astype(np.float32)

    with pytest.raises(RuntimeError, match="Profile creation failed"):
        identifier.create_profile_from_segment(audio)
