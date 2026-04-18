#!/usr/bin/env python3
"""Unit tests for SOTA quality pipeline (scripts/quality_pipeline.py)."""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'models' / 'seed-vc'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

import pytest
import torch
import numpy as np
import soundfile as sf

from quality_pipeline import (
    build_progress_callback,
    QualityVoiceConverter,
    QualityConfig,
    build_arg_parser,
    load_audio_file,
    main,
    resolve_reference_audio,
    write_quality_report,
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
        assert args.report_max_seconds == 30.0

    def test_build_arg_parser_quality_upgrades(self):
        """CLI parser should accept quality-upgrade toggles."""
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--source-audio",
                "source.wav",
                "--reference-audio",
                "reference.wav",
                "--output",
                "output.wav",
                "--hq-super-resolution",
                "--enable-nsf-harmonic-enhancement",
                "--enable-smoothsinger-smoothing",
                "--transfer-reference-dynamics",
            ]
        )

        assert args.hq_super_resolution is True
        assert args.enable_nsf_harmonic_enhancement is True
        assert args.enable_smoothsinger_smoothing is True
        assert args.transfer_reference_dynamics is True


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

    def test_load_audio_file_missing_raises(self, tmp_path):
        missing = tmp_path / "missing.wav"

        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            load_audio_file(missing)

    def test_resolve_reference_audio_from_profile_and_missing_cases(self, tmp_path, monkeypatch):
        reference_path = tmp_path / "profile_reference.wav"
        audio = np.sin(2 * np.pi * 110 * np.linspace(0, 1, 8000, endpoint=False)).astype(np.float32)
        sf.write(reference_path, audio, 8000)

        class DummyStore:
            def __init__(self, paths):
                self._paths = paths

            def get_all_vocals_paths(self, profile_id):
                assert profile_id == "profile-123"
                return self._paths

        module = types.ModuleType("auto_voice.storage.voice_profiles")
        module.VoiceProfileStore = lambda: DummyStore([str(reference_path)])
        monkeypatch.setitem(sys.modules, "auto_voice.storage.voice_profiles", module)

        loaded_audio, loaded_sr, resolved_path = resolve_reference_audio(None, "profile-123")

        assert loaded_sr == 8000
        assert resolved_path == reference_path.resolve()
        assert loaded_audio.shape[0] == audio.shape[0]

        module.VoiceProfileStore = lambda: DummyStore([])
        with pytest.raises(FileNotFoundError, match="No training vocals found"):
            resolve_reference_audio(None, "profile-123")

        with pytest.raises(ValueError, match="Either --reference-audio or --target-profile-id"):
            resolve_reference_audio(None, None)

    def test_build_progress_callback_writes_progress_bar(self, capsys):
        progress = build_progress_callback()
        progress(0.375, "Testing")

        captured = capsys.readouterr()
        assert "Testing" in captured.out
        assert "%" in captured.out

    def test_write_quality_report_uses_fake_evaluation_backend(self, tmp_path, monkeypatch):
        captured = {}

        class FakeQualityMetrics:
            def compute_all(self, reference_audio, converted_audio, sample_rate):
                captured["compute_all"] = {
                    "reference_shape": tuple(reference_audio.shape),
                    "converted_shape": tuple(converted_audio.shape),
                    "sample_rate": sample_rate,
                }
                return {"speaker_similarity": 0.99, "pesq": 1.5}

        class FakeBenchmarkRunner:
            def __init__(self, metrics):
                captured["runner_metrics"] = metrics.__class__.__name__

            def write_report_artifacts(self, results, output_dir, title):
                captured["results"] = results
                captured["output_dir"] = output_dir
                captured["title"] = title

        fake_evaluation = types.ModuleType("auto_voice.evaluation")
        fake_evaluation.QualityMetrics = FakeQualityMetrics
        fake_evaluation.BenchmarkRunner = FakeBenchmarkRunner
        monkeypatch.setitem(sys.modules, "auto_voice.evaluation", fake_evaluation)

        report_dir = tmp_path / "report"
        source_path = tmp_path / "source.wav"
        output_path = tmp_path / "output.wav"
        reference_path = tmp_path / "reference.wav"
        source_audio = np.linspace(-0.25, 0.25, 16000, dtype=np.float32)
        converted_audio = np.linspace(-0.1, 0.1, 12000, dtype=np.float32)
        reference_audio = np.linspace(-0.2, 0.2, 14000, dtype=np.float32)

        write_quality_report(
            report_dir=report_dir,
            source_path=source_path,
            output_path=output_path,
            reference_path=reference_path,
            source_audio=source_audio,
            converted_audio=converted_audio,
            reference_audio=reference_audio,
            output_sample_rate=16000,
            elapsed_seconds=1.25,
            report_max_seconds=None,
            run_metadata={"post_processing": ["nsf_harmonic_enhancement"]},
        )

        assert captured["compute_all"]["sample_rate"] == 16000
        assert captured["results"][0]["metadata"]["pipeline"]["post_processing"] == [
            "nsf_harmonic_enhancement"
        ]
        assert captured["results"][0]["metadata"]["evaluation_window_seconds"] == pytest.approx(0.75)
        assert captured["output_dir"] == str(report_dir)
        assert "Quality Pipeline Report" in captured["title"]

    def test_write_quality_report_honors_report_max_seconds(self, tmp_path, monkeypatch):
        captured = {}

        class FakeQualityMetrics:
            def compute_all(self, reference_audio, converted_audio, sample_rate):
                captured["frames"] = int(reference_audio.numel())
                captured["sample_rate"] = sample_rate
                return {"speaker_similarity": 0.95}

        class FakeBenchmarkRunner:
            def __init__(self, metrics):
                del metrics

            def write_report_artifacts(self, results, output_dir, title):
                captured["window_seconds"] = results[0]["metadata"]["evaluation_window_seconds"]
                captured["output_dir"] = output_dir
                captured["title"] = title

        fake_evaluation = types.ModuleType("auto_voice.evaluation")
        fake_evaluation.QualityMetrics = FakeQualityMetrics
        fake_evaluation.BenchmarkRunner = FakeBenchmarkRunner
        monkeypatch.setitem(sys.modules, "auto_voice.evaluation", fake_evaluation)

        write_quality_report(
            report_dir=tmp_path / "report",
            source_path=tmp_path / "source.wav",
            output_path=tmp_path / "output.wav",
            reference_path=tmp_path / "reference.wav",
            source_audio=np.zeros(16000, dtype=np.float32),
            converted_audio=np.zeros(16000, dtype=np.float32),
            reference_audio=np.zeros(16000, dtype=np.float32),
            output_sample_rate=16000,
            elapsed_seconds=0.25,
            report_max_seconds=0.5,
        )

        assert captured["frames"] == 8000
        assert captured["window_seconds"] == pytest.approx(0.5)

    def test_write_quality_report_rejects_non_positive_sample_rate(self, tmp_path):
        with pytest.raises(ValueError, match="output_sample_rate must be positive"):
            write_quality_report(
                report_dir=tmp_path / "report",
                source_path=tmp_path / "source.wav",
                output_path=tmp_path / "output.wav",
                reference_path=tmp_path / "reference.wav",
                source_audio=np.zeros(4, dtype=np.float32),
                converted_audio=np.zeros(4, dtype=np.float32),
                reference_audio=np.zeros(4, dtype=np.float32),
                output_sample_rate=0,
                elapsed_seconds=0.5,
            )

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
                captured["config"] = config
                self.last_run_metadata = {"post_processing": ["mocked"]}

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
                "--report-max-seconds",
                "12.5",
                "--device",
                "cpu",
                "--fp32",
                "--enable-nsf-harmonic-enhancement",
                "--enable-smoothsinger-smoothing",
                "--transfer-reference-dynamics",
            ]
        )

        assert exit_code == 0
        assert output_path.exists()
        assert captured["unloaded"] is True
        assert captured["report_kwargs"]["report_dir"] == report_dir.resolve()
        assert captured["report_kwargs"]["output_path"] == output_path.resolve()
        assert captured["report_kwargs"]["report_max_seconds"] == 12.5
        assert captured["config"].enable_nsf_harmonic_enhancement is True
        assert captured["config"].enable_smoothsinger_smoothing is True
        assert captured["config"].transfer_reference_dynamics is True

    def test_main_cli_resamples_report_audio_to_output_sample_rate(self, tmp_path, monkeypatch):
        source_path = tmp_path / "source.wav"
        reference_path = tmp_path / "reference.wav"
        output_path = tmp_path / "converted.wav"
        report_dir = tmp_path / "reports"

        source_audio = np.sin(2 * np.pi * 220 * np.linspace(0, 1, 22050, endpoint=False)).astype(np.float32)
        reference_audio = np.sin(2 * np.pi * 330 * np.linspace(0, 1, 22050, endpoint=False)).astype(np.float32)
        sf.write(source_path, source_audio, 22050)
        sf.write(reference_path, reference_audio, 22050)

        captured = {}

        class DummyConverter:
            def __init__(self, config):
                self.last_run_metadata = {"post_processing": ["mocked"]}

            def convert(self, source_audio, source_sr, reference_audio, reference_sr, pitch_shift, progress_callback):
                del source_sr, reference_audio, reference_sr, pitch_shift
                progress_callback(1.0, "Complete!")
                return source_audio[:8000] * 0.25, 8000

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
            ]
        )

        assert exit_code == 0
        assert captured["unloaded"] is True
        assert captured["report_kwargs"]["output_sample_rate"] == 8000
        assert len(captured["report_kwargs"]["source_audio"]) == 8000
        assert len(captured["report_kwargs"]["reference_audio"]) == 8000

    def test_quality_post_processing_chain_with_mocked_hq(self):
        """Post-processing should record SmoothSinger, NSF, and HQ-SVC stages."""
        converter = QualityVoiceConverter(
            QualityConfig(
                device="cpu",
                enable_nsf_harmonic_enhancement=True,
                enable_hq_super_resolution=True,
                enable_smoothsinger_smoothing=True,
                transfer_reference_dynamics=True,
            )
        )
        converter.last_run_metadata = {
            "post_processing": ["smoothsinger_pitch_smoothing"],
            "smooth_pitch_applied": True,
            "f0_conditioned": True,
        }
        converter._hq_wrapper = type(
            "FakeHQWrapper",
            (),
            {
                "super_resolve": staticmethod(
                    lambda audio, sample_rate: {
                        "audio": audio.detach().cpu().numpy() * 0.9,
                        "sample_rate": sample_rate,
                    }
                )
            },
        )()

        audio = np.linspace(-0.2, 0.2, 32000, dtype=np.float32)
        reference = np.sin(np.linspace(0, np.pi * 8, 32000, dtype=np.float32)).astype(np.float32)
        f0 = np.full(128, 220.0, dtype=np.float32)

        processed, sample_rate = converter._apply_quality_post_processing(
            converted_audio=audio,
            output_sample_rate=16000,
            f0_contour=f0,
            reference_audio=reference,
            reference_sr=16000,
        )

        assert processed.shape == audio.shape
        assert sample_rate == 16000
        assert converter.last_run_metadata["smooth_pitch_applied"] is True
        assert "smoothsinger_pitch_smoothing" in converter.last_run_metadata["post_processing"]
        assert "smoothsinger_dynamics_transfer" in converter.last_run_metadata["post_processing"]
        assert "nsf_harmonic_enhancement" in converter.last_run_metadata["post_processing"]
        assert "hq_super_resolution" in converter.last_run_metadata["post_processing"]

    def test_quality_post_processing_records_hq_skip_and_resamples_reference(self):
        converter = QualityVoiceConverter(
            QualityConfig(
                device="cpu",
                enable_hq_super_resolution=True,
            )
        )
        converter._hq_wrapper = type(
            "FailingHQWrapper",
            (),
            {
                "super_resolve": staticmethod(
                    lambda audio, sample_rate: (_ for _ in ()).throw(RuntimeError("hq unavailable"))
                )
            },
        )()

        processed, sample_rate = converter._apply_quality_post_processing(
            converted_audio=np.linspace(-0.2, 0.2, 8000, dtype=np.float32),
            output_sample_rate=8000,
            f0_contour=None,
            reference_audio=np.linspace(-0.2, 0.2, 16000, dtype=np.float32),
            reference_sr=16000,
        )

        assert sample_rate == 8000
        assert processed.shape == (8000,)
        assert converter.last_run_metadata["hq_super_resolution_skipped"] == "hq unavailable"

    def test_get_hq_wrapper_constructs_wrapper(self, monkeypatch):
        captured = {}
        module = types.ModuleType("auto_voice.inference.hq_svc_wrapper")

        class FakeWrapper:
            def __init__(self, device, require_gpu):
                captured["device"] = device
                captured["require_gpu"] = require_gpu

        module.HQSVCWrapper = FakeWrapper
        monkeypatch.setitem(sys.modules, "auto_voice.inference.hq_svc_wrapper", module)

        converter = QualityVoiceConverter(
            QualityConfig(device="cpu", enable_hq_super_resolution=True, hq_require_gpu=True)
        )
        wrapper = converter._get_hq_wrapper()

        assert isinstance(wrapper, FakeWrapper)
        assert str(captured["device"]) == "cpu"
        assert captured["require_gpu"] is True

    def test_crossfade_handles_short_second_chunk(self):
        converter = QualityVoiceConverter(QualityConfig(device="cpu"))

        result = converter._crossfade(
            np.linspace(-1.0, 1.0, 12, dtype=np.float32),
            np.linspace(-0.5, 0.5, 3, dtype=np.float32),
            overlap=6,
        )

        assert result.shape == (3,)
        assert np.isfinite(result).all()

    def test_crossfade_handles_standard_overlap(self):
        converter = QualityVoiceConverter(QualityConfig(device="cpu"))

        result = converter._crossfade(
            np.linspace(-1.0, 1.0, 12, dtype=np.float32),
            np.linspace(-0.5, 0.5, 12, dtype=np.float32),
            overlap=6,
        )

        assert result.shape == (12,)
        assert np.isfinite(result).all()

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
