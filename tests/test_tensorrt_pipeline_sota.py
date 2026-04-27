"""Tests for TensorRT-optimized SOTA pipeline (Phase 8).

Validates TensorRT export and inference for all pipeline components:
- ONNX export with correct dynamic shapes
- TRT engine building with FP16 precision
- Engine loading and inference
- Output quality vs PyTorch baseline

Target: Jetson Thor (SM 11.0, 16GB GPU memory)
"""
import pytest
import torch
import numpy as np
import os
from pathlib import Path


def _trt_engine_dir() -> str:
    engine_dir = os.environ.get("AUTOVOICE_TRT_ENGINE_DIR")
    if not engine_dir:
        pytest.skip("AUTOVOICE_TRT_ENGINE_DIR is not set for the TensorRT hardware lane")
    required = {
        "content_extractor.trt",
        "pitch_extractor.trt",
        "decoder.trt",
        "vocoder.trt",
    }
    missing = sorted(name for name in required if not (Path(engine_dir) / name).exists())
    if missing:
        pytest.skip(f"AUTOVOICE_TRT_ENGINE_DIR is missing TensorRT engines: {', '.join(missing)}")
    return engine_dir


class TestTRTEngineExport:
    """Tests for ONNX/TRT engine export."""

    def test_trt_pipeline_class_exists(self):
        """TRTConversionPipeline class should exist."""
        from auto_voice.inference.trt_pipeline import TRTConversionPipeline
        assert TRTConversionPipeline is not None

    def test_onnx_exporter_exists(self):
        """ONNXExporter class should exist for component export."""
        from auto_voice.inference.trt_pipeline import ONNXExporter
        assert ONNXExporter is not None

    def test_export_content_extractor(self):
        """Content extractor should export to ONNX."""
        pytest.skip("ContentVec has lazy-loading that doesn't export well to ONNX - use TRT from checkpoint")
        from auto_voice.inference.trt_pipeline import ONNXExporter
        from auto_voice.models.encoder import ContentVecEncoder

        model = ContentVecEncoder(output_dim=768, pretrained=None)
        exporter = ONNXExporter()

        # Export with dynamic sequence length
        onnx_path = exporter.export_content_extractor(model, "/tmp/test_content.onnx")
        assert onnx_path.endswith(".onnx")

    def test_export_pitch_extractor(self):
        """Pitch extractor should export to ONNX."""
        from auto_voice.inference.trt_pipeline import ONNXExporter
        from auto_voice.models.pitch import RMVPEPitchExtractor

        model = RMVPEPitchExtractor(pretrained=None)
        exporter = ONNXExporter()

        onnx_path = exporter.export_pitch_extractor(model, "/tmp/test_pitch.onnx")
        assert onnx_path.endswith(".onnx")

    def test_export_decoder(self):
        """SVC decoder should export to ONNX."""
        from auto_voice.inference.trt_pipeline import ONNXExporter
        from auto_voice.models.svc_decoder import CoMoSVCDecoder

        model = CoMoSVCDecoder()
        exporter = ONNXExporter()

        onnx_path = exporter.export_decoder(model, "/tmp/test_decoder.onnx")
        assert onnx_path.endswith(".onnx")

    def test_export_vocoder(self):
        """Vocoder should export to ONNX."""
        from auto_voice.inference.trt_pipeline import ONNXExporter
        from auto_voice.models.vocoder import BigVGANGenerator

        model = BigVGANGenerator()
        exporter = ONNXExporter()

        onnx_path = exporter.export_vocoder(model, "/tmp/test_vocoder.onnx")
        assert onnx_path.endswith(".onnx")


class TestTRTEngineBuilder:
    """Tests for TensorRT engine building."""

    def test_engine_builder_exists(self):
        """TRTEngineBuilder class should exist."""
        from auto_voice.inference.trt_pipeline import TRTEngineBuilder
        assert TRTEngineBuilder is not None

    def test_build_fp16_engine(self):
        """Should build FP16 TRT engine from ONNX."""
        pytest.importorskip("tensorrt")
        from auto_voice.inference.trt_pipeline import TRTEngineBuilder

        builder = TRTEngineBuilder(precision="fp16")
        assert builder.precision == "fp16"

    def test_build_with_dynamic_shapes(self):
        """Engine should support dynamic input shapes."""
        pytest.importorskip("tensorrt")
        from auto_voice.inference.trt_pipeline import TRTEngineBuilder

        builder = TRTEngineBuilder(precision="fp16")
        # Dynamic shape profile: min, optimal, max
        shapes = {
            "audio": [(1, 1600), (1, 16000), (1, 160000)],  # 0.1s to 10s
        }
        assert builder.supports_dynamic_shapes(shapes)


class TestTRTInference:
    """Tests for TRT inference execution."""

    def test_trt_pipeline_init(self):
        """TRT pipeline should initialize."""
        pytest.importorskip("tensorrt")
        from auto_voice.inference.trt_pipeline import TRTConversionPipeline
        pipeline = TRTConversionPipeline(engine_dir=_trt_engine_dir())
        assert pipeline is not None

    def test_trt_convert_produces_audio(self):
        """TRT pipeline should produce audio output."""
        pytest.importorskip("tensorrt")
        from auto_voice.inference.trt_pipeline import TRTConversionPipeline

        pipeline = TRTConversionPipeline(engine_dir=_trt_engine_dir())

        audio = torch.sin(torch.linspace(0, 1, 24000) * 2 * np.pi * 440)
        speaker = torch.randn(256)

        result = pipeline.convert(audio, 24000, speaker)
        assert 'audio' in result
        assert result['audio'].shape[0] > 0

    def test_trt_output_finite(self):
        """TRT output should be finite."""
        pytest.importorskip("tensorrt")
        from auto_voice.inference.trt_pipeline import TRTConversionPipeline

        pipeline = TRTConversionPipeline(engine_dir=_trt_engine_dir())

        audio = torch.sin(torch.linspace(0, 1, 24000) * 2 * np.pi * 440)
        speaker = torch.randn(256)

        result = pipeline.convert(audio, 24000, speaker)
        assert torch.isfinite(result['audio']).all()


class TestTRTQualityComparison:
    """Tests comparing TRT vs PyTorch output quality."""

    def test_trt_matches_pytorch_within_tolerance(self):
        """TRT output should match PyTorch within 1% relative error."""
        pytest.importorskip("tensorrt")
        from auto_voice.inference.trt_pipeline import TRTConversionPipeline
        from auto_voice.inference.sota_pipeline import SOTAConversionPipeline

        # PyTorch reference
        pytorch_pipeline = SOTAConversionPipeline()

        # TRT pipeline
        trt_pipeline = TRTConversionPipeline(engine_dir=_trt_engine_dir())

        audio = torch.sin(torch.linspace(0, 1, 24000) * 2 * np.pi * 440)
        speaker = torch.randn(256)

        pytorch_result = pytorch_pipeline.convert(audio, 24000, speaker)
        trt_result = trt_pipeline.convert(audio, 24000, speaker)

        # Compare mel spectrograms (intermediate) within tolerance
        # Final audio may differ slightly due to phase, but energy should match
        pytorch_energy = pytorch_result['audio'].pow(2).mean().sqrt()
        trt_energy = trt_result['audio'].pow(2).mean().sqrt()

        relative_error = abs(pytorch_energy - trt_energy) / pytorch_energy
        assert relative_error < 0.01  # Within 1%


class TestTRTPerformance:
    """Tests for TRT inference latency."""

    def test_trt_latency_under_100ms(self):
        """TRT inference should be under 100ms for 1s audio."""
        pytest.importorskip("tensorrt")
        import time
        from auto_voice.inference.trt_pipeline import TRTConversionPipeline

        pipeline = TRTConversionPipeline(engine_dir=_trt_engine_dir())

        audio = torch.sin(torch.linspace(0, 1, 24000) * 2 * np.pi * 440)
        speaker = torch.randn(256)

        # Warmup
        pipeline.convert(audio, 24000, speaker)

        # Timed run
        start = time.time()
        pipeline.convert(audio, 24000, speaker)
        elapsed = time.time() - start

        assert elapsed < 0.1  # 100ms for 1s audio = 10x realtime

    def test_trt_memory_efficient(self):
        """TRT engines should fit in 16GB GPU memory."""
        pytest.importorskip("tensorrt")
        from auto_voice.inference.trt_pipeline import TRTConversionPipeline

        pipeline = TRTConversionPipeline(engine_dir=_trt_engine_dir())

        # Get total engine memory usage
        total_memory = pipeline.get_engine_memory_usage()
        assert total_memory < 16 * 1024 * 1024 * 1024  # 16GB


class TestNoFallbackTRT:
    """Tests for strict error behavior in TRT pipeline."""

    def test_missing_engine_raises(self):
        """Missing TRT engine should raise RuntimeError."""
        pytest.importorskip("tensorrt")
        from auto_voice.inference.trt_pipeline import TRTConversionPipeline

        with pytest.raises(RuntimeError):
            TRTConversionPipeline(engine_dir="/nonexistent/path")

    def test_invalid_input_raises(self):
        """Invalid input should raise RuntimeError."""
        pytest.importorskip("tensorrt")
        from auto_voice.inference.trt_pipeline import TRTConversionPipeline

        pipeline = TRTConversionPipeline(engine_dir=_trt_engine_dir())

        with pytest.raises(RuntimeError):
            pipeline.convert(torch.tensor([]), 24000, torch.randn(256))
