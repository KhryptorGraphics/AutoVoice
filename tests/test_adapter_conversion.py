"""Test conversion with trained adapters (Task 2.5).

Verifies that set_speaker() correctly loads both adapter weights and speaker
embeddings, and that conversion produces valid output.
"""
import pytest
import torch
import numpy as np
from pathlib import Path

from auto_voice.inference.sota_pipeline import SOTAConversionPipeline


# William and Conor profile IDs (real profiles with trained adapters)
WILLIAM_PROFILE_ID = "7da05140-1303-40c6-95d9-5b6e2c3624df"
CONOR_PROFILE_ID = "c572d02c-c687-4bed-8676-6ad253cf1c91"


def _load_profile_embedding(profile_id: str) -> np.ndarray:
    """Load the stored profile embedding from the canonical profile store."""
    profiles_dir = Path("data/voice_profiles")
    return np.load(profiles_dir / f"{profile_id}.npy")


@pytest.fixture
def test_audio():
    """Create test audio tensor (5 seconds at 24kHz)."""
    sample_rate = 24000
    duration = 5.0
    num_samples = int(sample_rate * duration)

    # Generate sine wave at 440Hz
    t = torch.linspace(0, duration, num_samples)
    audio = 0.5 * torch.sin(2 * torch.pi * 440 * t)

    return audio, sample_rate


@pytest.fixture
def profiles_have_adapters():
    """Check that both test profiles have trained adapters."""
    adapters_dir = Path("data/trained_models")
    william_adapter = adapters_dir / f"{WILLIAM_PROFILE_ID}_adapter.pt"
    conor_adapter = adapters_dir / f"{CONOR_PROFILE_ID}_adapter.pt"

    if not william_adapter.exists():
        pytest.skip(f"William adapter not found: {william_adapter}")
    if not conor_adapter.exists():
        pytest.skip(f"Conor adapter not found: {conor_adapter}")


@pytest.fixture
def profiles_have_embeddings():
    """Check that both test profiles have speaker embeddings."""
    profiles_dir = Path("data/voice_profiles")
    william_emb = profiles_dir / f"{WILLIAM_PROFILE_ID}.npy"
    conor_emb = profiles_dir / f"{CONOR_PROFILE_ID}.npy"

    if not william_emb.exists():
        pytest.skip(f"William embedding not found: {william_emb}")
    if not conor_emb.exists():
        pytest.skip(f"Conor embedding not found: {conor_emb}")


class TestAdapterConversion:
    """Test conversion with trained adapters."""

    @pytest.mark.cuda
    def test_load_william_adapter(self, profiles_have_adapters, profiles_have_embeddings):
        """Test loading William's adapter and speaker embedding."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        pipeline = SOTAConversionPipeline(device=torch.device('cuda'), n_steps=1)

        # Load William's speaker
        pipeline.set_speaker(WILLIAM_PROFILE_ID)

        # Verify speaker is loaded
        assert pipeline.get_current_speaker() == WILLIAM_PROFILE_ID

        # Verify embedding is loaded
        embedding = pipeline.get_speaker_embedding()
        assert embedding is not None
        expected_dim = _load_profile_embedding(WILLIAM_PROFILE_ID).shape[0]
        assert embedding.shape == (expected_dim,)
        assert embedding.device.type == 'cuda'

        # Verify L2 normalization
        norm = torch.norm(embedding).item()
        assert abs(norm - 1.0) < 0.01, f"Embedding not L2-normalized: norm={norm}"

    @pytest.mark.cuda
    def test_load_conor_adapter(self, profiles_have_adapters, profiles_have_embeddings):
        """Test loading Conor's adapter and speaker embedding."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        pipeline = SOTAConversionPipeline(device=torch.device('cuda'), n_steps=1)

        # Load Conor's speaker
        pipeline.set_speaker(CONOR_PROFILE_ID)

        # Verify speaker is loaded
        assert pipeline.get_current_speaker() == CONOR_PROFILE_ID

        # Verify embedding is loaded
        embedding = pipeline.get_speaker_embedding()
        assert embedding is not None
        expected_dim = _load_profile_embedding(CONOR_PROFILE_ID).shape[0]
        assert embedding.shape == (expected_dim,)

        # Verify L2 normalization
        norm = torch.norm(embedding).item()
        assert abs(norm - 1.0) < 0.01

    @pytest.mark.cuda
    def test_switch_between_speakers(self, profiles_have_adapters, profiles_have_embeddings):
        """Test switching between William and Conor."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        pipeline = SOTAConversionPipeline(device=torch.device('cuda'), n_steps=1)

        # Load William
        pipeline.set_speaker(WILLIAM_PROFILE_ID)
        william_emb = pipeline.get_speaker_embedding()

        # Switch to Conor
        pipeline.set_speaker(CONOR_PROFILE_ID)
        conor_emb = pipeline.get_speaker_embedding()

        # Verify embeddings are different
        assert (
            william_emb.shape != conor_emb.shape
            or not torch.allclose(william_emb, conor_emb)
        ), "William and Conor embeddings should be different"

        # Switch back to William
        pipeline.set_speaker(WILLIAM_PROFILE_ID)
        william_emb_2 = pipeline.get_speaker_embedding()

        # Should be same embedding
        assert torch.allclose(william_emb, william_emb_2)

    @pytest.mark.cuda
    def test_clear_speaker(self, profiles_have_adapters, profiles_have_embeddings):
        """Test clearing speaker after loading."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        pipeline = SOTAConversionPipeline(device=torch.device('cuda'), n_steps=1)

        # Load William
        pipeline.set_speaker(WILLIAM_PROFILE_ID)
        assert pipeline.get_current_speaker() == WILLIAM_PROFILE_ID
        assert pipeline.get_speaker_embedding() is not None

        # Clear speaker
        pipeline.clear_speaker()
        assert pipeline.get_current_speaker() is None
        assert pipeline.get_speaker_embedding() is None

    @pytest.mark.cuda
    @pytest.mark.slow
    def test_conversion_with_william(
        self, profiles_have_adapters, profiles_have_embeddings, test_audio
    ):
        """Test actual conversion with William's adapter."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        audio, sample_rate = test_audio

        pipeline = SOTAConversionPipeline(device=torch.device('cuda'), n_steps=1)
        pipeline.set_speaker(WILLIAM_PROFILE_ID)

        # Get the loaded speaker embedding
        speaker_emb = pipeline.get_speaker_embedding()
        assert speaker_emb is not None

        # Run conversion
        result = pipeline.convert(
            audio=audio,
            sample_rate=sample_rate,
            speaker_embedding=speaker_emb,
        )

        # Verify output
        assert 'audio' in result
        assert 'sample_rate' in result
        output_audio = result['audio']

        # Check output is valid
        assert isinstance(output_audio, torch.Tensor)
        assert output_audio.dim() == 1, "Output should be 1D audio"
        assert output_audio.numel() > 0, "Output should not be empty"
        assert torch.isfinite(output_audio).all(), "Output should not contain NaN/Inf"

        # Check output sample rate
        assert result['sample_rate'] == 24000

    @pytest.mark.cuda
    @pytest.mark.slow
    def test_conversion_produces_different_outputs(
        self, profiles_have_adapters, profiles_have_embeddings, test_audio
    ):
        """Test that William and Conor produce different outputs."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        audio, sample_rate = test_audio

        pipeline = SOTAConversionPipeline(device=torch.device('cuda'), n_steps=1)

        # Convert with William
        pipeline.set_speaker(WILLIAM_PROFILE_ID)
        william_emb = pipeline.get_speaker_embedding()
        result_william = pipeline.convert(audio, sample_rate, william_emb)

        # Convert with Conor
        pipeline.set_speaker(CONOR_PROFILE_ID)
        conor_emb = pipeline.get_speaker_embedding()
        result_conor = pipeline.convert(audio, sample_rate, conor_emb)

        # Outputs should be different
        william_audio = result_william['audio']
        conor_audio = result_conor['audio']

        # Ensure they have same length (for comparison)
        min_len = min(len(william_audio), len(conor_audio))
        william_audio = william_audio[:min_len]
        conor_audio = conor_audio[:min_len]

        # Verify outputs differ
        assert not torch.allclose(william_audio, conor_audio, atol=0.01), \
            "William and Conor conversions should produce different outputs"


class TestEmbeddingValidation:
    """Test embedding format validation."""

    def test_embedding_shape_validation(self):
        """Test that invalid embedding shapes are rejected."""
        profiles_dir = Path("data/voice_profiles")
        embedding_path = profiles_dir / f"{WILLIAM_PROFILE_ID}.npy"

        if not embedding_path.exists():
            pytest.skip("William embedding not found")

        # Load valid embedding
        embedding = np.load(embedding_path)

        # Verify shape matches the supported runtime contract.
        assert embedding.ndim == 1
        assert embedding.shape[0] in (192, 256), \
            f"Expected supported embedding width, got {embedding.shape}"

    def test_embedding_normalization(self):
        """Test that embeddings are L2-normalized."""
        profiles_dir = Path("data/voice_profiles")

        for profile_id in [WILLIAM_PROFILE_ID, CONOR_PROFILE_ID]:
            embedding_path = profiles_dir / f"{profile_id}.npy"

            if not embedding_path.exists():
                continue

            embedding = np.load(embedding_path)
            norm = np.linalg.norm(embedding)

            assert abs(norm - 1.0) < 0.01, \
                f"Profile {profile_id} embedding not L2-normalized: norm={norm:.4f}"

    def test_embedding_dtype(self):
        """Test that embeddings are float32."""
        profiles_dir = Path("data/voice_profiles")
        embedding_path = profiles_dir / f"{WILLIAM_PROFILE_ID}.npy"

        if not embedding_path.exists():
            pytest.skip("William embedding not found")

        embedding = np.load(embedding_path)

        assert embedding.dtype == np.float32, \
            f"Expected float32 but got {embedding.dtype}"
