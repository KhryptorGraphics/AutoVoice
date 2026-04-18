"""Coverage for optional quality-upgrade modules and integration points."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest


def _write_test_wav(path: Path, sample_rate: int = 16000, duration_seconds: float = 3.5) -> None:
    frames = int(sample_rate * duration_seconds)
    t = np.linspace(0.0, duration_seconds, frames, endpoint=False)
    audio = 0.2 * np.sin(2 * np.pi * 220 * t)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes((audio * 32767).astype(np.int16).tobytes())


def test_nsf_harmonic_enhancer_preserves_shape_and_bounds():
    from auto_voice.models.nsf_module import NSFHarmonicEnhancer

    enhancer = NSFHarmonicEnhancer(harmonic_strength=0.1, max_harmonics=4, blend=0.25)
    audio = np.zeros(22050, dtype=np.float32)
    f0 = np.full(128, 220.0, dtype=np.float32)

    enhanced = enhancer.enhance(audio, f0, 22050)

    assert enhanced.shape == audio.shape
    assert np.max(np.abs(enhanced)) <= 0.98


def test_pupu_vocoder_enhancer_preserves_shape_and_bounds():
    from auto_voice.models.pupu_vocoder import PupuVocoderEnhancer

    enhancer = PupuVocoderEnhancer(brightness=0.05, transient_boost=0.08)
    audio = np.linspace(-0.2, 0.2, 22050, dtype=np.float32)

    refined = enhancer.refine(audio, 22050)

    assert refined.shape == audio.shape
    assert np.max(np.abs(refined)) <= 0.98


def test_ecapa2_encoder_fallback_returns_normalized_embedding():
    from auto_voice.models.ecapa2_encoder import ECAPA2SpeakerEncoder

    encoder = ECAPA2SpeakerEncoder(device="cpu")
    audio = np.random.default_rng(0).normal(0.0, 0.05, 16000 * 4).astype(np.float32)

    result = encoder.extract_embedding(audio, 16000)

    assert result.embedding.shape == (256,)
    assert np.isclose(np.linalg.norm(result.embedding), 1.0, atol=1e-3)
    assert result.backend in {"mel-statistics-fallback", "speechbrain-ecapa"}


def test_voice_cloner_supports_ecapa2_backend_fallback(tmp_path):
    from auto_voice.inference.voice_cloner import VoiceCloner

    audio_path = tmp_path / "reference.wav"
    profiles_dir = tmp_path / "profiles"
    samples_dir = tmp_path / "samples"
    profiles_dir.mkdir()
    samples_dir.mkdir()
    _write_test_wav(audio_path)

    cloner = VoiceCloner(
        profiles_dir=str(profiles_dir),
        samples_dir=str(samples_dir),
        speaker_encoder_backend="ecapa2",
    )

    embedding = cloner._extract_embedding(str(audio_path))

    assert embedding.shape == (256,)
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-3)


def test_pipeline_quality_post_processing_applies_nsf_and_pupu():
    from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

    pipeline = SingingConversionPipeline(
        config={
            "enable_nsf_harmonic_enhancement": True,
            "enable_pupu_vocoder_refinement": True,
        }
    )
    vocals = np.zeros(22050, dtype=np.float32)
    backing = np.zeros(22050, dtype=np.float32)
    f0 = np.full(128, 220.0, dtype=np.float32)

    out_vocals, out_backing, out_sr, metadata = pipeline._apply_quality_post_processing(
        vocals,
        backing,
        22050,
        f0,
    )

    assert out_vocals.shape == vocals.shape
    assert out_backing.shape == backing.shape
    assert out_sr == 22050
    assert metadata["post_processing"] == [
        "nsf_harmonic_enhancement",
        "pupu_vocoder_refinement",
    ]


def test_pipeline_quality_post_processing_uses_hq_super_resolution():
    from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline

    pipeline = SingingConversionPipeline(config={"enable_hq_super_resolution": True})
    pipeline._hq_enhancer = type(
        "FakeHQEnhancer",
        (),
        {
            "super_resolve": staticmethod(
                lambda audio, sample_rate: {
                    "audio": np.zeros(44100, dtype=np.float32),
                    "sample_rate": 44100,
                }
            )
        },
    )()

    vocals = np.zeros(22050, dtype=np.float32)
    backing = np.zeros(22050, dtype=np.float32)
    f0 = np.full(128, 220.0, dtype=np.float32)

    out_vocals, out_backing, out_sr, metadata = pipeline._apply_quality_post_processing(
        vocals,
        backing,
        22050,
        f0,
    )

    assert out_vocals.shape[0] == 44100
    assert out_backing.shape[0] > backing.shape[0]
    assert out_sr == 44100
    assert metadata["post_processing"] == ["hq_super_resolution"]
