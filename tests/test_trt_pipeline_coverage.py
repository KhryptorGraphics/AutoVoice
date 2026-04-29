"""Coverage tests for trt_pipeline.py — mock all TensorRT dependencies."""
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

from auto_voice.models.feature_contract import DEFAULT_PITCH_DIM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_trt_inference_context():
    """Mock TRTInferenceContext that returns plausible tensor shapes."""
    ctx = MagicMock()
    call_count = [0]
    def mock_infer(inputs):
        call_count[0] += 1
        if call_count[0] == 1:  # content extraction
            return {'features': torch.randn(1, 49, 768)}
        elif call_count[0] == 2:  # pitch extraction
            return {'f0': torch.rand(1, 49) * 200 + 80}
        elif call_count[0] == 3:  # decoder
            return {'mel': torch.randn(1, 80, 49)}
        else:  # vocoder
            return {'audio': torch.randn(1, 1, 24000)}
    ctx.infer.side_effect = mock_infer
    ctx.get_memory_usage.return_value = 256 * 1024 * 1024  # 256MB
    return ctx


@pytest.fixture
def mock_separator():
    """Mock vocal separator."""
    sep = MagicMock()
    sep.extract_vocals.return_value = torch.randn(1, 44100)
    return sep


@pytest.fixture
def trt_pipeline(mock_trt_inference_context, mock_separator, tmp_path):
    """Create TRTPipeline with mocked internals."""
    with patch.dict('sys.modules', {'tensorrt': MagicMock()}):
        from auto_voice.inference.trt_pipeline import TRTConversionPipeline as TRTPipeline
        pipeline = TRTPipeline.__new__(TRTPipeline)
        pipeline.device = torch.device('cpu')
        pipeline.precision = 'fp16'
        pipeline.content_ctx = mock_trt_inference_context
        pipeline.pitch_ctx = mock_trt_inference_context
        pipeline.decoder_ctx = mock_trt_inference_context
        pipeline.vocoder_ctx = mock_trt_inference_context
        pipeline.separator = mock_separator
        pipeline.engine_dir = tmp_path
        pipeline.pitch_dim = DEFAULT_PITCH_DIM
        return pipeline


# ---------------------------------------------------------------------------
# ONNXExporter Tests
# ---------------------------------------------------------------------------

class TestONNXExporter:
    def test_init_default_opset(self):
        from auto_voice.inference.trt_pipeline import ONNXExporter
        exporter = ONNXExporter()
        assert exporter.opset_version == 17

    def test_init_custom_opset(self):
        from auto_voice.inference.trt_pipeline import ONNXExporter
        exporter = ONNXExporter(opset_version=14)
        assert exporter.opset_version == 14

    def test_export_content_extractor_calls_onnx_export(self, tmp_path):
        from auto_voice.inference.trt_pipeline import ONNXExporter
        exporter = ONNXExporter()
        # Use a real nn.Module so parameters() works correctly
        model = torch.nn.Linear(10, 10)
        with patch('torch.onnx.export') as mock_export:
            out = exporter.export_content_extractor(model, str(tmp_path / 'content.onnx'))
            assert mock_export.called
            assert out == str(tmp_path / 'content.onnx')

    def test_export_pitch_extractor_calls_onnx_export(self, tmp_path):
        from auto_voice.inference.trt_pipeline import ONNXExporter
        exporter = ONNXExporter()
        model = MagicMock()
        model.parameters.return_value = iter([torch.randn(1)])
        with patch('torch.onnx.export') as mock_export:
            out = exporter.export_pitch_extractor(model, str(tmp_path / 'pitch.onnx'))
            assert mock_export.called

    def test_export_decoder_calls_onnx_export(self, tmp_path):
        from auto_voice.inference.trt_pipeline import ONNXExporter
        exporter = ONNXExporter()
        model = MagicMock()
        model.parameters.return_value = iter([torch.randn(1)])
        with patch('torch.onnx.export') as mock_export:
            out = exporter.export_decoder(model, str(tmp_path / 'decoder.onnx'))
            assert mock_export.called

    def test_export_vocoder_calls_onnx_export(self, tmp_path):
        from auto_voice.inference.trt_pipeline import ONNXExporter
        exporter = ONNXExporter()
        model = MagicMock()
        model.parameters.return_value = iter([torch.randn(1)])
        with patch('torch.onnx.export') as mock_export:
            out = exporter.export_vocoder(model, str(tmp_path / 'vocoder.onnx'))
            assert mock_export.called

    def test_export_content_no_params_uses_cpu(self, tmp_path):
        from auto_voice.inference.trt_pipeline import ONNXExporter
        exporter = ONNXExporter()
        model = MagicMock()
        model.parameters.return_value = iter([])  # no params
        model.train = MagicMock()
        with patch('torch.onnx.export') as mock_export:
            exporter.export_content_extractor(model, str(tmp_path / 'c.onnx'))
            call_args = mock_export.call_args
            assert call_args[0][1].device == torch.device('cpu')


# ---------------------------------------------------------------------------
# TRTEngineBuilder Tests
# ---------------------------------------------------------------------------

class TestTRTEngineBuilder:
    def test_init_defaults(self):
        from auto_voice.inference.trt_pipeline import TRTEngineBuilder
        builder = TRTEngineBuilder()
        assert builder.precision == 'fp16'
        assert builder.workspace_size == 4 * 1024 * 1024 * 1024

    def test_init_fp32(self):
        from auto_voice.inference.trt_pipeline import TRTEngineBuilder
        builder = TRTEngineBuilder(precision='fp32')
        assert builder.precision == 'fp32'

    def test_build_from_onnx_raises_without_trt(self, tmp_path):
        from auto_voice.inference.trt_pipeline import TRTEngineBuilder
        builder = TRTEngineBuilder()
        onnx_path = tmp_path / 'model.onnx'
        onnx_path.touch()
        with patch.dict('sys.modules', {'tensorrt': None}):
            with pytest.raises((ImportError, RuntimeError, AttributeError)):
                builder.build_from_onnx(str(onnx_path), str(tmp_path / 'out.trt'))


# ---------------------------------------------------------------------------
# TRTInferenceContext Tests
# ---------------------------------------------------------------------------

class TestTRTInferenceContext:
    def test_load_engine_file_not_found(self, tmp_path):
        from auto_voice.inference.trt_pipeline import TRTInferenceContext
        with pytest.raises((RuntimeError, FileNotFoundError, OSError)):
            TRTInferenceContext(str(tmp_path / 'nonexistent.trt'))

    def test_get_memory_usage_without_load(self, tmp_path):
        from auto_voice.inference.trt_pipeline import TRTInferenceContext
        with pytest.raises((RuntimeError, AttributeError)):
            ctx = TRTInferenceContext.__new__(TRTInferenceContext)
            ctx.engine = None
            ctx.get_memory_usage()


# ---------------------------------------------------------------------------
# TRTPipeline Core Logic Tests
# ---------------------------------------------------------------------------

class TestTRTPipelineResample:
    def test_resample_identity(self, trt_pipeline):
        audio = torch.randn(44100)
        out = trt_pipeline._resample(audio, 44100, 44100)
        assert torch.equal(out, audio)

    def test_resample_1d(self, trt_pipeline):
        audio = torch.randn(44100)
        out = trt_pipeline._resample(audio, 44100, 22050)
        assert out.shape[0] == 22050

    def test_resample_2d_passthrough(self, trt_pipeline):
        audio = torch.randn(2, 44100)
        out = trt_pipeline._resample(audio, 44100, 22050)
        assert out.shape == audio.shape


class TestTRTPipelineToMono:
    def test_mono_passthrough(self, trt_pipeline):
        audio = torch.randn(1000)
        out = trt_pipeline._to_mono(audio)
        assert out.shape == (1000,)

    def test_stereo_to_mono(self, trt_pipeline):
        audio = torch.randn(2, 1000)
        out = trt_pipeline._to_mono(audio)
        assert out.shape == (1000,)
        assert torch.allclose(out, audio.mean(dim=0))

    def test_invalid_shape_raises(self, trt_pipeline):
        audio = torch.randn(1, 2, 1000)
        with pytest.raises(RuntimeError, match="Unexpected audio shape"):
            trt_pipeline._to_mono(audio)


class TestTRTPipelineEncodePitch:
    def test_output_shape(self, trt_pipeline):
        f0 = torch.rand(1, 100) * 200 + 80  # valid F0 range
        out = trt_pipeline._encode_pitch(f0)
        assert out.shape == (1, 100, DEFAULT_PITCH_DIM)

    def test_contains_sin_and_cos(self, trt_pipeline):
        f0 = torch.rand(1, 50) * 200 + 80
        out = trt_pipeline._encode_pitch(f0)
        half_dim = DEFAULT_PITCH_DIM // 2
        sin_part = out[:, :, :half_dim]
        cos_part = out[:, :, half_dim:]
        assert sin_part.shape == (1, 50, half_dim)
        assert cos_part.shape == (1, 50, half_dim)

    def test_gradient_flows(self, trt_pipeline):
        f0 = torch.rand(1, 20) * 200 + 80
        f0.requires_grad_(True)
        f0.retain_grad()
        out = trt_pipeline._encode_pitch(f0)
        out.sum().backward()
        assert f0.grad is not None

    def test_clamped_low_f0(self, trt_pipeline):
        f0 = torch.zeros(1, 10)  # F0 = 0 Hz
        out = trt_pipeline._encode_pitch(f0)
        assert not torch.any(torch.isnan(out))

    def test_batch_size_2(self, trt_pipeline):
        f0 = torch.rand(2, 50) * 200 + 80
        out = trt_pipeline._encode_pitch(f0)
        assert out.shape == (2, 50, DEFAULT_PITCH_DIM)


class TestTRTPipelineConvert:
    def test_empty_audio_raises(self, trt_pipeline):
        with pytest.raises(RuntimeError, match="Empty audio"):
            trt_pipeline.convert(
                audio=torch.tensor([]),
                sample_rate=44100,
                speaker_embedding=torch.randn(256),
            )

    def test_invalid_speaker_embedding_dim_raises(self, trt_pipeline):
        audio = torch.randn(44100)
        with pytest.raises(RuntimeError, match="Speaker embedding must be \\[256\\]"):
            trt_pipeline.convert(
                audio=audio,
                sample_rate=44100,
                speaker_embedding=torch.randn(128),
            )

    def test_valid_conversion(self, trt_pipeline):
        audio = torch.randn(44100)
        embedding = torch.randn(256)
        result = trt_pipeline.convert(audio, 44100, embedding)
        assert 'audio' in result
        assert 'sample_rate' in result
        assert 'metadata' in result
        assert result['metadata']['backend'] == 'tensorrt'

    def test_stereo_input(self, trt_pipeline):
        audio = torch.randn(2, 44100)
        embedding = torch.randn(256)
        result = trt_pipeline.convert(audio, 44100, embedding)
        assert 'audio' in result

    def test_progress_callback(self, trt_pipeline):
        audio = torch.randn(44100)
        embedding = torch.randn(256)
        callbacks = []
        def on_progress(stage, pct):
            callbacks.append((stage, pct))
        result = trt_pipeline.convert(audio, 44100, embedding, on_progress=on_progress)
        assert len(callbacks) >= 4  # separation, content, pitch, decode, vocoder

    def test_engine_memory_usage(self, trt_pipeline):
        total = trt_pipeline.get_engine_memory_usage()
        assert total == 4 * 256 * 1024 * 1024  # 4 engines * 256MB each
