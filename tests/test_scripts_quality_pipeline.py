#!/usr/bin/env python3
"""Unit tests for SOTA quality pipeline (scripts/quality_pipeline.py).

Tests the Seed-VC + BigVGAN pipeline for high-quality conversion.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'models' / 'seed-vc'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

import pytest
import torch
import numpy as np
import soundfile as sf

from quality_pipeline import (
    QualityVoiceConverter,
    QualityConfig,
    build_arg_parser,
    main,
    resolve_reference_audio,
)


@pytest.fixture
def converter():
    """Create a quality converter instance."""
    config = QualityConfig(
        sample_rate=44100,
        diffusion_steps=10,  # Use fewer steps for testing
        fp16=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    return QualityVoiceConverter(config)


@pytest.fixture
def sample_audio():
    """Generate synthetic audio for testing."""
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sr


class TestQualityConfig:
    """Test configuration."""

    def test_default_config(self):
        """Test default values."""
        config = QualityConfig()
        assert config.sample_rate == 44100
        assert config.diffusion_steps == 30
        assert config.fp16 is True

    def test_build_arg_parser(self):
        """CLI parser should accept the canonical offline arguments."""
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--source-audio",
                "source.wav",
                "--reference-audio",
                "reference.wav",
                "--output",
                "output.wav",
                "--report-dir",
                "reports",
            ]
        )

        assert args.source_audio == "source.wav"
        assert args.reference_audio == "reference.wav"
        assert args.output == "output.wav"
        assert args.report_dir == "reports"


class TestQualityConverter:
    """Test converter."""

    def test_initialization(self, converter):
        """Test init."""
        assert converter is not None
        assert converter.config.sample_rate == 44100

    def test_unload(self, converter):
        """Test unload."""
        converter.unload()
        # Models are lazily loaded, so unload should succeed even if not loaded

    def test_resolve_reference_audio_from_path(self, tmp_path):
        """CLI helper should load reference audio directly from disk."""
        reference_path = tmp_path / "reference.wav"
        audio = np.sin(2 * np.pi * 220 * np.linspace(0, 1, 16000, endpoint=False)).astype(np.float32)
        sf.write(reference_path, audio, 16000)

        loaded_audio, loaded_sr, resolved_path = resolve_reference_audio(str(reference_path), None)

        assert loaded_sr == 16000
        assert resolved_path == reference_path.resolve()
        assert loaded_audio.shape[0] == audio.shape[0]

    def test_main_cli_with_mocked_converter_and_report(self, tmp_path, monkeypatch):
        """CLI should save output audio and invoke the report writer."""
        source_path = tmp_path / "source.wav"
        reference_path = tmp_path / "reference.wav"
        output_path = tmp_path / "converted.wav"
        report_dir = tmp_path / "reports"

        source_audio = np.sin(2 * np.pi * 220 * np.linspace(0, 1, 16000, endpoint=False)).astype(np.float32)
        reference_audio = np.sin(2 * np.pi * 330 * np.linspace(0, 1, 16000, endpoint=False)).astype(np.float32)
        sf.write(source_path, source_audio, 16000)
        sf.write(reference_path, reference_audio, 16000)

        captured = {}

        class DummyConverter:
            def __init__(self, config):
                self.config = config

            def convert(self, source_audio, source_sr, reference_audio, reference_sr, pitch_shift, progress_callback):
                del source_sr, reference_audio, reference_sr, pitch_shift
                progress_callback(1.0, "Complete!")
                return source_audio * 0.5, 16000

            def unload(self):
                captured["unloaded"] = True

        def fake_report_writer(**kwargs):
            captured["report_kwargs"] = kwargs

        monkeypatch.setattr("quality_pipeline.QualityVoiceConverter", DummyConverter)
        monkeypatch.setattr("quality_pipeline.write_quality_report", fake_report_writer)

        exit_code = main(
            [
                "--source-audio",
                str(source_path),
                "--reference-audio",
                str(reference_path),
                "--output",
                str(output_path),
                "--report-dir",
                str(report_dir),
                "--device",
                "cpu",
                "--fp32",
            ]
        )

        assert exit_code == 0
        assert output_path.exists()
        assert captured["unloaded"] is True
        assert captured["report_kwargs"]["report_dir"] == report_dir.resolve()
        assert captured["report_kwargs"]["output_path"] == output_path.resolve()

    @pytest.mark.cuda
    @pytest.mark.slow
    def test_convert(self, converter, sample_audio):
        """Test conversion."""
        audio, sr = sample_audio
        # Need reference audio for Seed-VC
        reference = audio.copy()
        
        converted, out_sr = converter.convert(
            source_audio=audio,
            source_sr=sr,
            reference_audio=reference,
            reference_sr=sr,
            pitch_shift=0
        )
        
        assert len(converted) > 0
        assert out_sr == 44100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
