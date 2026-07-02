"""Tests for inference pipeline trained-profile loading.

SOTAConversionPipeline serves a profile's trained weights via the
self-contained serving artifact ({profile}_adapter_model.pt /
{profile}_full_model.pt): the artifact-built decoder replaces the base
decoder. Deltas-only LoRA payloads cannot reproduce the trained model
(random training base) and are skipped at construction with a warning.
"""

import os

import pytest
import torch

from auto_voice.models.svc_decoder import CoMoSVCDecoder
from auto_voice.storage.voice_profiles import VoiceProfileStore

# content/pitch/speaker dims must match the pipeline's runtime features;
# hidden/n_layers are free, so keep the artifact small for test speed
ARTIFACT_DIMS = dict(
    content_dim=768, pitch_dim=768, speaker_dim=256,
    n_mels=100, hidden_dim=64, n_layers=2,
)


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


def _save_serving_artifact(store, profile_id):
    torch.manual_seed(7)
    decoder = CoMoSVCDecoder(device=torch.device("cpu"), **ARTIFACT_DIMS)
    payload = {
        "model_state_dict": decoder.state_dict(),
        "lora_config": {},
    }
    os.makedirs(store.trained_models_dir, exist_ok=True)
    path = os.path.join(store.trained_models_dir, f"{profile_id}_adapter_model.pt")
    torch.save(payload, path)
    return path


@pytest.fixture
def sample_profile_with_weights(store):
    """Profile with a self-contained trained serving artifact."""
    profile_data = {
        "profile_id": "trained-profile-123",
        "name": "Trained Artist",
        "embedding": torch.randn(256).numpy(),
        "sample_count": 5,
    }
    store.save(profile_data)
    _save_serving_artifact(store, profile_data["profile_id"])
    return profile_data["profile_id"]


@pytest.fixture
def sample_profile_deltas_only(store):
    """Legacy profile with only deltas-only LoRA weights (not servable)."""
    profile_data = {
        "profile_id": "deltas-profile-789",
        "name": "Legacy Artist",
        "embedding": torch.randn(256).numpy(),
        "sample_count": 5,
    }
    store.save(profile_data)
    lora_state = {
        "input_proj.adapter.lora_A": torch.randn(8, 1536),
        "input_proj.adapter.lora_B": torch.randn(512, 8),
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
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline
        assert SOTAConversionPipeline is not None

    def test_pipeline_accepts_profile_store(self, store, sample_profile_with_weights):
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline

        pipeline = SOTAConversionPipeline(
            profile_store=store,
            require_gpu=False,
        )
        assert pipeline is not None
        assert hasattr(pipeline, 'profile_store') or hasattr(pipeline, '_profile_store')

    def test_pipeline_accepts_profile_id(self, store, sample_profile_with_weights):
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline

        pipeline = SOTAConversionPipeline(
            profile_store=store,
            profile_id=sample_profile_with_weights,
            require_gpu=False,
        )
        assert pipeline is not None


class TestAutomaticArtifactLoading:
    """Trained serving artifacts replace the base decoder at construction."""

    def test_trained_profile_replaces_decoder(self, store, sample_profile_with_weights):
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline

        pipeline = SOTAConversionPipeline(
            profile_store=store,
            profile_id=sample_profile_with_weights,
            require_gpu=False,
        )

        assert pipeline.decoder is not pipeline._base_decoder, \
            "Trained artifact should replace the base decoder"
        assert pipeline.decoder.hidden_dim == ARTIFACT_DIMS['hidden_dim']

    def test_untrained_profile_keeps_base_decoder(self, store, sample_profile_no_weights):
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline

        pipeline = SOTAConversionPipeline(
            profile_store=store,
            profile_id=sample_profile_no_weights,
            require_gpu=False,
        )

        assert pipeline.decoder is pipeline._base_decoder

    def test_deltas_only_profile_is_skipped_with_base_decoder(
        self, store, sample_profile_deltas_only, caplog
    ):
        """Legacy deltas-only artifacts cannot be served; construction warns."""
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline

        pipeline = SOTAConversionPipeline(
            profile_store=store,
            profile_id=sample_profile_deltas_only,
            require_gpu=False,
        )

        assert pipeline.decoder is pipeline._base_decoder
        assert "deltas-only" in caplog.text

    def test_artifact_weights_loaded_correctly(self, store, sample_profile_with_weights):
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline

        artifact_path = os.path.join(
            store.trained_models_dir, f"{sample_profile_with_weights}_adapter_model.pt"
        )
        saved_state = torch.load(artifact_path, map_location="cpu", weights_only=False)[
            "model_state_dict"
        ]

        pipeline = SOTAConversionPipeline(
            profile_store=store,
            profile_id=sample_profile_with_weights,
            require_gpu=False,
        )

        torch.testing.assert_close(
            pipeline.decoder.state_dict()["input_proj.weight"].cpu(),
            saved_state["input_proj.weight"],
        )


class TestConversionWithTrainedArtifact:
    """Tests for conversion using the trained serving artifact."""

    def test_convert_with_artifact_produces_output(self, store, sample_profile_with_weights):
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline

        pipeline = SOTAConversionPipeline(
            profile_store=store,
            profile_id=sample_profile_with_weights,
            require_gpu=False,
        )

        sr = 24000
        duration = 0.2  # Short for speed
        audio = torch.randn(int(sr * duration)) * 0.1
        speaker_embedding = torch.randn(256)

        result = pipeline.convert(
            audio=audio,
            sample_rate=sr,
            speaker_embedding=speaker_embedding,
        )

        assert "audio" in result, "Result should contain audio"
        assert result["audio"].shape[0] > 0, "Audio should not be empty"
        assert result["sample_rate"] == pipeline.vocoder.sample_rate

    def test_different_output_with_vs_without_artifact(
        self, store, sample_profile_with_weights, sample_profile_no_weights
    ):
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline

        sr = 24000
        duration = 0.2
        torch.manual_seed(42)
        audio = torch.randn(int(sr * duration)) * 0.1
        speaker_embedding = torch.randn(256)

        torch.manual_seed(123)
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

        torch.manual_seed(123)
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

        audio_trained = result_trained["audio"].cpu().numpy()
        audio_untrained = result_untrained["audio"].cpu().numpy()

        min_len = min(len(audio_trained), len(audio_untrained), 1000)
        if min_len > 100:
            import numpy as np
            correlation = np.corrcoef(audio_trained[:min_len], audio_untrained[:min_len])[0, 1]
            assert correlation < 0.99, \
                "Trained and untrained outputs should differ"


class TestProfileSwitching:
    """Tests for switching between profiles."""

    def test_can_load_different_profile(
        self, store, sample_profile_with_weights, sample_profile_no_weights
    ):
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline

        pipeline = SOTAConversionPipeline(
            profile_store=store,
            profile_id=sample_profile_with_weights,
            require_gpu=False,
        )
        assert pipeline.decoder is not pipeline._base_decoder

        pipeline2 = SOTAConversionPipeline(
            profile_store=store,
            profile_id=sample_profile_no_weights,
            require_gpu=False,
        )
        assert pipeline2.decoder is pipeline2._base_decoder
