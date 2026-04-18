"""Targeted branch coverage for singing_conversion_pipeline."""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline


def test_resolve_target_speaker_prefers_full_model(tmp_path):
    """Profiles with a dedicated full model should bypass the adapter speaker ID."""
    trained_models_dir = tmp_path / "trained_models"
    trained_models_dir.mkdir()
    full_model = trained_models_dir / "profile-1_full_model.pt"
    full_model.write_text("weights")

    voice_cloner = MagicMock()
    voice_cloner.store = types.SimpleNamespace(trained_models_dir=str(trained_models_dir))

    pipeline = SingingConversionPipeline(
        config={"speaker_id": "adapter-speaker"},
        voice_cloner=voice_cloner,
    )
    model_manager = MagicMock()
    pipeline._model_manager = model_manager

    speaker_id, model_type = pipeline._resolve_target_speaker(
        "profile-1",
        np.array([1.0, 2.0], dtype=np.float32),
    )

    assert (speaker_id, model_type) == ("profile-1", "full_model")
    args, kwargs = model_manager.load_voice_model.call_args
    assert args == (str(full_model), "profile-1")
    assert np.array_equal(kwargs["speaker_embedding"], np.array([1.0, 2.0], dtype=np.float32))


def test_extract_pitch_falls_back_to_zeros_when_librosa_fails():
    """Pitch extraction should degrade cleanly when pyin errors."""
    pipeline = SingingConversionPipeline()

    with patch("librosa.pyin", side_effect=RuntimeError("bad pitch")):
        f0 = pipeline._extract_pitch(np.ones(2048, dtype=np.float32), 22050)

    assert np.array_equal(f0, np.zeros(4))


def test_resample_audio_uses_librosa_when_sample_rate_changes():
    """Sample-rate conversion should delegate to librosa and keep float32 output."""
    pipeline = SingingConversionPipeline()

    with patch("librosa.resample", return_value=np.array([0.1, 0.2], dtype=np.float64)) as resample:
        result = pipeline._resample_audio(np.array([0.0, 1.0], dtype=np.float32), 22050, 48000)

    assert result.dtype == np.float32
    assert np.allclose(result, [0.1, 0.2])
    resample.assert_called_once()


def test_apply_quality_post_processing_runs_all_enabled_stages():
    """NSF, Pupu, and HQ enhancement branches should compose in order."""

    class FakeNSFEnhancer:
        def __init__(self, harmonic_strength, max_harmonics, blend):
            self.params = (harmonic_strength, max_harmonics, blend)

        def enhance(self, vocals, f0_contour, sample_rate):
            return vocals + 1.0

    class FakePupuEnhancer:
        def __init__(self, brightness, transient_boost):
            self.params = (brightness, transient_boost)

        def refine(self, vocals, sample_rate):
            return vocals + 2.0

    class FakeHQWrapper:
        def __init__(self, device, require_gpu):
            self.device = device
            self.require_gpu = require_gpu

        def super_resolve(self, vocals, sample_rate):
            return {"audio": vocals.numpy() + 3.0, "sample_rate": 48000}

    pipeline = SingingConversionPipeline(
        config={
            "enable_nsf_harmonic_enhancement": True,
            "enable_pupu_vocoder_refinement": True,
            "enable_hq_super_resolution": True,
        }
    )

    fake_modules = {
        "auto_voice.models.nsf_module": types.SimpleNamespace(NSFHarmonicEnhancer=FakeNSFEnhancer),
        "auto_voice.models.pupu_vocoder": types.SimpleNamespace(PupuVocoderEnhancer=FakePupuEnhancer),
        "auto_voice.inference.hq_svc_wrapper": types.SimpleNamespace(HQSVCWrapper=FakeHQWrapper),
    }

    with patch.dict(sys.modules, fake_modules):
        with patch.object(
            pipeline,
            "_resample_audio",
            return_value=np.array([9.0, 8.0], dtype=np.float32),
        ) as resample_audio:
            vocals, backing, output_sr, metadata = pipeline._apply_quality_post_processing(
                np.array([0.0, 1.0], dtype=np.float32),
                np.array([0.5, 0.25], dtype=np.float32),
                22050,
                np.array([100.0, 101.0], dtype=np.float32),
            )

    assert output_sr == 48000
    assert np.allclose(vocals, [6.0, 7.0])
    assert np.allclose(backing, [9.0, 8.0])
    assert metadata["post_processing"] == [
        "nsf_harmonic_enhancement",
        "pupu_vocoder_refinement",
        "hq_super_resolution",
    ]
    args = resample_audio.call_args[0]
    assert np.array_equal(args[0], np.array([0.5, 0.25], dtype=np.float32))
    assert args[1:] == (22050, 48000)


def test_apply_quality_post_processing_records_hq_failure():
    """HQ failures should be logged into metadata without aborting conversion."""

    class FailingHQWrapper:
        def __init__(self, device, require_gpu):
            self.device = device
            self.require_gpu = require_gpu

        def super_resolve(self, vocals, sample_rate):
            raise RuntimeError("gpu missing")

    pipeline = SingingConversionPipeline(config={"enable_hq_super_resolution": True})
    fake_modules = {
        "auto_voice.inference.hq_svc_wrapper": types.SimpleNamespace(HQSVCWrapper=FailingHQWrapper),
    }

    with patch.dict(sys.modules, fake_modules):
        vocals, backing, output_sr, metadata = pipeline._apply_quality_post_processing(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([0.5, 0.5], dtype=np.float32),
            22050,
            np.array([100.0], dtype=np.float32),
        )

    assert output_sr == 22050
    assert np.allclose(vocals, [1.0, 2.0])
    assert np.allclose(backing, [0.5, 0.5])
    assert metadata["hq_super_resolution_skipped"] == "gpu missing"


def test_convert_song_refreshes_pitch_after_post_processing_and_emits_skip_metadata(tmp_path):
    """Post-processing metadata should trigger pitch refresh and propagate skip details."""
    song_path = tmp_path / "song.wav"
    song_path.write_bytes(b"not-a-real-audio-file")

    voice_cloner = MagicMock()
    voice_cloner.load_voice_profile.return_value = {"embedding": [0.1, 0.2, 0.3]}

    pipeline = SingingConversionPipeline(voice_cloner=voice_cloner)
    pipeline._resolve_target_speaker = Mock(return_value=("speaker-1", "adapter"))
    pipeline._separate_vocals = Mock(
        return_value={
            "vocals": np.array([0.5, 0.25], dtype=np.float32),
            "instrumental": np.array([0.1, 0.2], dtype=np.float32),
        }
    )
    pipeline._convert_voice = Mock(return_value=np.array([0.6, 0.4], dtype=np.float32))
    pipeline._extract_pitch = Mock(
        side_effect=[
            np.array([110.0], dtype=np.float32),
            np.array([120.0], dtype=np.float32),
            np.array([130.0], dtype=np.float32),
        ]
    )
    pipeline._apply_quality_post_processing = Mock(
        return_value=(
            np.array([0.9, 0.7], dtype=np.float32),
            np.array([0.05, 0.05], dtype=np.float32),
            48000,
            {
                "post_processing": ["hq_super_resolution"],
                "hq_super_resolution_skipped": "enhancer offline",
            },
        )
    )

    with patch("librosa.load", return_value=(np.array([1.0, 0.5], dtype=np.float32), 22050)):
        result = pipeline.convert_song(str(song_path), "profile-1")

    assert np.allclose(result["f0_contour"], [130.0])
    assert result["metadata"]["quality_post_processing"] == ["hq_super_resolution"]
    assert result["metadata"]["hq_super_resolution_skipped"] == "enhancer offline"
    assert result["metadata"]["speaker_id"] == "speaker-1"
    assert pipeline._extract_pitch.call_count == 3
