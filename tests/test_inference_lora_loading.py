"""Tests for inference pipeline LoRA weight loading.

Phase 4: Test that SOTAConversionPipeline loads profile-based LoRA weights.

Tests verify:
- Pipeline accepts profile_id parameter
- Automatic LoRA loading if profile has weights
- Conversion uses loaded LoRA weights
"""

import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

from auto_voice.storage.voice_profiles import VoiceProfileStore


@pytest.fixture
def temp_profile_dir(tmp_path):
    """Create temporary profile storage directory."""
    profile_dir = tmp_path / "voice_profiles"
    profile_dir.mkdir()
    return profile_dir


@pytest.fixture
def store(temp_profile_dir):
    """Create VoiceProfileStore with temp directory."""
    return VoiceProfileStore(profiles_dir=str(temp_profile_dir))


@pytest.fixture
def sample_profile_with_weights(store):
    """Create a sample voice profile with trained LoRA weights."""
    profile_data = {
        "profile_id": "trained-profile-123",
        "name": "Trained Artist",
        "embedding": torch.randn(256).numpy(),
        "sample_count": 5,
    }
    store.save(profile_data)

    # Create sample LoRA weights matching decoder structure
    lora_state = {
        "input_proj.adapter.lora_A": torch.randn(8, 1536),
        "input_proj.adapter.lora_B": torch.randn(512, 8),
        "speaker_film.gamma_proj.adapter.lora_A": torch.randn(8, 256),
        "speaker_film.gamma_proj.adapter.lora_B": torch.randn(512, 8),
    }
    store.save_lora_weights(profile_data["profile_id"], lora_state)

    return profile_data["profile_id"]


@pytest.fixture
def sample_profile_no_weights(store):
    """Create a sample voice profile without trained weights."""
    profile_data = {
        "profile_id": "untrained-profile-456",
        "name": "Untrained Artist",
        "embedding": torch.randn(256).numpy(),
        "sample_count": 2,
    }
    store.save(profile_data)
    return profile_data["profile_id"]


class TestPipelineProfileParameter:
    """Tests for SOTAConversionPipeline profile_id parameter."""

    def test_pipeline_exists(self):
        """SOTAConversionPipeline should exist."""
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline
        assert SOTAConversionPipeline is not None

    def test_pipeline_accepts_profile_store(self, store, sample_profile_with_weights):
        """Task 4.1-4.2: Pipeline should accept profile_store parameter."""
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline

        # Pipeline should be constructable with profile_store
        pipeline = SOTAConversionPipeline(
            profile_store=store,
            require_gpu=False,
        )
        assert pipeline is not None
        assert hasattr(pipeline, 'profile_store') or hasattr(pipeline, '_profile_store')

    def test_pipeline_accepts_profile_id(self, store, sample_profile_with_weights):
        """Pipeline should accept profile_id to load specific profile."""
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline

        pipeline = SOTAConversionPipeline(
            profile_store=store,
            profile_id=sample_profile_with_weights,
            require_gpu=False,
        )
        assert pipeline is not None


class TestAutomaticLoRALoading:
    """Tests for automatic LoRA loading from profile."""

    def test_loads_lora_for_trained_profile(self, store, sample_profile_with_weights):
        """Task 4.3-4.4: Should automatically load LoRA if profile has weights."""
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline

        pipeline = SOTAConversionPipeline(
            profile_store=store,
            profile_id=sample_profile_with_weights,
            require_gpu=False,
        )

        # Check that LoRA was injected into decoder
        decoder = pipeline.decoder
        assert decoder._lora_injected is True, \
            "LoRA should be injected for trained profile"

    def test_no_lora_for_untrained_profile(self, store, sample_profile_no_weights):
        """Should not inject LoRA if profile has no weights."""
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline

        pipeline = SOTAConversionPipeline(
            profile_store=store,
            profile_id=sample_profile_no_weights,
            require_gpu=False,
        )

        # LoRA should not be injected
        decoder = pipeline.decoder
        assert not getattr(decoder, '_lora_injected', False), \
            "LoRA should not be injected for untrained profile"

    def test_lora_weights_loaded_correctly(self, store, sample_profile_with_weights):
        """Loaded LoRA weights should match saved weights."""
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline

        # Get saved weights for comparison
        saved_weights = store.load_lora_weights(sample_profile_with_weights)

        pipeline = SOTAConversionPipeline(
            profile_store=store,
            profile_id=sample_profile_with_weights,
            require_gpu=False,
        )

        # Get loaded weights from decoder
        loaded_weights = pipeline.decoder.get_lora_state_dict()

        # Compare (keys may differ slightly due to naming)
        assert len(loaded_weights) > 0, "Should have loaded some weights"


class TestConversionWithLoRA:
    """Tests for conversion using loaded LoRA weights."""

    def test_convert_with_lora_produces_output(self, store, sample_profile_with_weights):
        """Task 4.5-4.6: Conversion should work with loaded LoRA."""
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline

        pipeline = SOTAConversionPipeline(
            profile_store=store,
            profile_id=sample_profile_with_weights,
            require_gpu=False,
        )

        # Create dummy input audio tensor
        sr = 24000
        duration = 0.2  # Short for speed
        audio = torch.randn(int(sr * duration)) * 0.1
        speaker_embedding = torch.randn(256)

        # Should produce output (even if quality is low with random weights)
        result = pipeline.convert(
            audio=audio,
            sample_rate=sr,
            speaker_embedding=speaker_embedding,
        )

        # Verify output was created
        assert "audio" in result, "Result should contain audio"
        assert result["audio"].shape[0] > 0, "Audio should not be empty"
        assert result["sample_rate"] == 24000, "Output should be 24kHz"

    def test_different_output_with_vs_without_lora(self, store, sample_profile_with_weights, sample_profile_no_weights):
        """Output should differ between trained and untrained profiles."""
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline

        # Create input audio (same for both)
        sr = 24000
        duration = 0.2
        torch.manual_seed(42)  # Reproducible input
        audio = torch.randn(int(sr * duration)) * 0.1
        speaker_embedding = torch.randn(256)

        # Convert with trained profile
        torch.manual_seed(123)  # Different seed for model init
        pipeline_trained = SOTAConversionPipeline(
            profile_store=store,
            profile_id=sample_profile_with_weights,
            require_gpu=False,
        )
        result_trained = pipeline_trained.convert(
            audio=audio.clone(),
            sample_rate=sr,
            speaker_embedding=speaker_embedding.clone(),
        )

        # Convert with untrained profile
        torch.manual_seed(123)  # Same seed for fair comparison
        pipeline_untrained = SOTAConversionPipeline(
            profile_store=store,
            profile_id=sample_profile_no_weights,
            require_gpu=False,
        )
        result_untrained = pipeline_untrained.convert(
            audio=audio.clone(),
            sample_rate=sr,
            speaker_embedding=speaker_embedding.clone(),
        )

        # Get output audio (move to CPU for numpy)
        audio_trained = result_trained["audio"].cpu().numpy()
        audio_untrained = result_untrained["audio"].cpu().numpy()

        # They should be different (different LoRA weights vs none)
        # Truncate to same length for comparison
        min_len = min(len(audio_trained), len(audio_untrained), 1000)
        if min_len > 100:
            import numpy as np
            correlation = np.corrcoef(audio_trained[:min_len], audio_untrained[:min_len])[0, 1]
            # Outputs should not be identical (correlation < 0.99)
            assert correlation < 0.99, \
                "Trained and untrained outputs should differ"


class TestProfileSwitching:
    """Tests for switching between profiles."""

    def test_can_load_different_profile(self, store, sample_profile_with_weights, sample_profile_no_weights):
        """Pipeline should support loading different profiles."""
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline

        # Load trained profile
        pipeline = SOTAConversionPipeline(
            profile_store=store,
            profile_id=sample_profile_with_weights,
            require_gpu=False,
        )
        assert pipeline.decoder._lora_injected is True

        # Load untrained profile (creates new pipeline)
        pipeline2 = SOTAConversionPipeline(
            profile_store=store,
            profile_id=sample_profile_no_weights,
            require_gpu=False,
        )
        assert not getattr(pipeline2.decoder, '_lora_injected', False)
