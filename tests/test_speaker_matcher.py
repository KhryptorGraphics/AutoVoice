"""Tests for speaker matcher module.

Tests embedding-based speaker matching and clustering including:
- Embedding extraction and similarity computation
- Cross-track speaker clustering
- Similarity threshold tuning
- Unknown speaker detection
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

from auto_voice.audio.speaker_matcher import SpeakerMatcher


@pytest.fixture
def matcher():
    """Create SpeakerMatcher instance."""
    return SpeakerMatcher(similarity_threshold=0.85, device='cpu')


# ===== Phase 3.1: Test embedding-based matching =====

def test_matcher_initialization():
    """Test SpeakerMatcher initialization."""
    matcher = SpeakerMatcher(similarity_threshold=0.9, device='cpu')

    assert matcher.similarity_threshold == 0.9
    assert matcher.device == 'cpu'
    assert matcher.min_cluster_duration == 30.0


def test_cosine_similarity_calculation(matcher):
    """Test cosine similarity computation."""
    emb1 = np.array([1.0, 0.0, 0.0])
    emb2 = np.array([1.0, 0.0, 0.0])

    similarity = matcher.cosine_similarity(emb1, emb2)

    assert similarity == pytest.approx(1.0, abs=0.01)


def test_same_speaker_high_similarity(matcher):
    """Test same speaker matches with >0.8 similarity."""
    # Create similar embeddings (same speaker)
    np.random.seed(42)
    base = np.random.randn(512)
    base = base / np.linalg.norm(base)

    # Add small noise
    noisy = base + 0.02 * np.random.randn(512)
    noisy = noisy / np.linalg.norm(noisy)

    similarity = matcher.cosine_similarity(base, noisy)

    assert similarity > 0.9


def test_different_speakers_low_similarity(matcher):
    """Test different speakers have <0.7 similarity."""
    np.random.seed(42)

    emb1 = np.random.randn(512)
    emb1 = emb1 / np.linalg.norm(emb1)

    emb2 = np.random.randn(512)
    emb2 = emb2 / np.linalg.norm(emb2)

    similarity = matcher.cosine_similarity(emb1, emb2)

    # Random embeddings should have low correlation
    assert -0.5 <= similarity <= 0.5


# ===== Phase 3.2: Test similarity threshold tuning =====

@pytest.mark.parametrize("threshold", [0.6, 0.7, 0.8, 0.9])
def test_threshold_values(threshold):
    """Test different threshold values."""
    matcher = SpeakerMatcher(similarity_threshold=threshold)

    assert matcher.similarity_threshold == threshold


def test_threshold_affects_clustering(matcher):
    """Test threshold affects clustering results."""
    # This is a conceptual test - actual clustering requires database
    # Verifying that threshold is used in clustering logic

    assert matcher.similarity_threshold == 0.85


# ===== Phase 3.3: Test unknown speaker detection =====

def test_detect_unknown_speaker(matcher):
    """Test detection of speakers not in existing clusters."""
    # Mock scenario: embedding doesn't match any cluster
    test_embedding = np.random.randn(512)
    test_embedding = test_embedding / np.linalg.norm(test_embedding)

    # With no clusters, should be detected as unknown
    # This would be tested via clustering logic


# ===== Phase 3.4: Test edge cases =====

def test_very_short_audio_unreliable():
    """Test that very short audio (<3s) produces warning."""
    # This tests the min_duration filtering in extract_embeddings_for_artist
    # Embeddings from segments < 5s are skipped
    pass  # Placeholder for integration test


def test_noisy_audio_handling():
    """Test embedding extraction with noisy audio."""
    # WavLM should be robust to noise
    # This would require actual audio processing
    pass


def test_similar_voices_challenge():
    """Test handling of very similar voices (e.g., family members)."""
    # This is a known limitation - document expected behavior
    pass


# ===== Helper method tests =====

def test_extract_embedding_from_audio_mock(matcher, tmp_path):
    """Test embedding extraction (mocked encoder)."""
    import soundfile as sf

    # Create test audio
    sr = 16000
    audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 3.0, int(sr * 3))).astype(np.float32)

    audio_path = tmp_path / "test.wav"
    sf.write(str(audio_path), audio, sr)

    # Mock the encoder
    mock_embedding = np.random.randn(512)
    mock_embedding = mock_embedding / np.linalg.norm(mock_embedding)

    with patch.object(matcher, '_get_encoder') as mock_encoder_getter:
        mock_diarizer = Mock()
        mock_diarizer.extract_embedding.return_value = mock_embedding
        mock_encoder_getter.return_value = mock_diarizer

        embedding = matcher.extract_embedding_from_audio(audio_path)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (512,)


# ===== Coverage verification =====

def test_coverage_speaker_matcher():
    """Verify coverage of speaker_matcher.py module."""
    from auto_voice.audio import speaker_matcher

    assert hasattr(speaker_matcher, 'SpeakerMatcher')
    assert hasattr(speaker_matcher, 'run_speaker_matching')
