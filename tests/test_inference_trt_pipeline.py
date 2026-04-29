"""Comprehensive tests for TensorRT pipeline module.

Target: Increase coverage from 23% to 70%+ for trt_pipeline.py

Test Categories:
1. ONNX Export Tests
2. TRT Engine Building Tests
3. TRT Inference Context Tests
4. TRT Conversion Pipeline Tests
5. Error Handling Tests
6. Edge Case Tests
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open

import numpy as np
import pytest
import torch

from auto_voice.inference.trt_pipeline import (
    ONNXExporter,
    TRTEngineBuilder,
    TRTInferenceContext,
    TRTConversionPipeline,
)
from auto_voice.models.feature_contract import DEFAULT_PITCH_DIM


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_engine_dir(tmp_path):
    """Create temporary engine directory."""
    engine_dir = tmp_path / "trt_engines"
    engine_dir.mkdir()
    return str(engine_dir)


@pytest.fixture
def mock_model():
    """Create mock PyTorch model."""
    model = MagicMock(spec=torch.nn.Module)
    model.train = MagicMock()
    model.parameters = MagicMock(return_value=[torch.randn(10, 10)])
    return model


@pytest.fixture
def sample_audio_tensor():
    """Generate sample audio tensor."""
    return torch.randn(1, 16000)


# ============================================================================
# ONNX Exporter Tests
# ============================================================================


def test_onnx_exporter_init():
    """Test ONNXExporter initialization."""
    exporter = ONNXExporter(opset_version=17)
    assert exporter.opset_version == 17


def test_onnx_exporter_default_opset():
    """Test ONNXExporter uses default opset version."""
    exporter = ONNXExporter()
    assert exporter.opset_version == 17


@patch('torch.onnx.export')
def test_export_content_extractor_cpu(mock_export, temp_engine_dir, mock_model):
    """Test exporting content extractor to ONNX on CPU."""
    exporter = ONNXExporter()
    output_path = os.path.join(temp_engine_dir, "content.onnx")

    # Ensure model has no parameters (uses CPU device logic)
    mock_model.parameters.return_value = []

    result = exporter.export_content_extractor(mock_model, output_path)

    assert result == output_path
    mock_export.assert_called_once()

    # Verify export arguments
    call_args = mock_export.call_args
    assert call_args[0][1].shape == torch.Size([1, 16000])  # dummy input
    assert call_args[1]['opset_version'] == 17
    assert 'audio' in call_args[1]['input_names']
    assert 'features' in call_args[1]['output_names']


@patch('torch.onnx.export')
def test_export_content_extractor_cuda(mock_export, temp_engine_dir):
    """Test exporting content extractor with CUDA model."""
    exporter = ONNXExporter()
    output_path = os.path.join(temp_engine_dir, "content.onnx")

    # Create fresh mock model
    mock_model = MagicMock(spec=torch.nn.Module)
    mock_model.train = MagicMock()

    # Mock parameters() to return proper generator
    param = MagicMock()
    param.device = torch.device('cuda:0')

    def param_generator():
        yield param

    mock_model.parameters = MagicMock(side_effect=param_generator)

    result = exporter.export_content_extractor(mock_model, output_path)

    assert result == output_path
    mock_export.assert_called_once()


@patch('torch.onnx.export')
def test_export_pitch_extractor(mock_export, temp_engine_dir, mock_model):
    """Test exporting pitch extractor to ONNX."""
    exporter = ONNXExporter()
    output_path = os.path.join(temp_engine_dir, "pitch.onnx")

    # Mock model device with proper iterator
    param = MagicMock()
    param.device = torch.device('cpu')
    mock_model.parameters.return_value = iter([param])

    result = exporter.export_pitch_extractor(mock_model, output_path)

    assert result == output_path
    mock_export.assert_called_once()

    # Verify dynamic axes for pitch
    call_args = mock_export.call_args
    assert 'f0' in call_args[1]['output_names']


@patch('torch.onnx.export')
def test_export_decoder(mock_export, temp_engine_dir, mock_model):
    """Test exporting decoder to ONNX."""
    exporter = ONNXExporter()
    output_path = os.path.join(temp_engine_dir, "decoder.onnx")

    # Mock model device with proper iterator
    param = MagicMock()
    param.device = torch.device('cpu')
    mock_model.parameters.return_value = iter([param])

    result = exporter.export_decoder(mock_model, output_path)

    assert result == output_path
    mock_export.assert_called_once()

    # Verify multiple inputs
    call_args = mock_export.call_args
    assert call_args[1]['input_names'] == ['content', 'pitch', 'speaker']
    assert call_args[1]['output_names'] == ['mel']


@patch('torch.onnx.export')
def test_export_vocoder(mock_export, temp_engine_dir, mock_model):
    """Test exporting vocoder to ONNX."""
    exporter = ONNXExporter()
    output_path = os.path.join(temp_engine_dir, "vocoder.onnx")

    # Mock model device with proper iterator
    param = MagicMock()
    param.device = torch.device('cpu')
    mock_model.parameters.return_value = iter([param])

    result = exporter.export_vocoder(mock_model, output_path)

    assert result == output_path
    mock_export.assert_called_once()

    # Verify mel input
    call_args = mock_export.call_args
    assert 'mel' in call_args[1]['input_names']
    assert 'audio' in call_args[1]['output_names']


# ============================================================================
# TRT Engine Builder Tests
# ============================================================================


def test_trt_engine_builder_init():
    """Test TRTEngineBuilder initialization."""
    builder = TRTEngineBuilder(precision="fp16", workspace_size=2 * 1024 * 1024 * 1024)
    assert builder.precision == "fp16"
    assert builder.workspace_size == 2 * 1024 * 1024 * 1024


def test_trt_engine_builder_default_params():
    """Test TRTEngineBuilder uses default parameters."""
    builder = TRTEngineBuilder()
    assert builder.precision == "fp16"
    assert builder.workspace_size == 4 * 1024 * 1024 * 1024


def test_supports_dynamic_shapes_valid():
    """Test dynamic shape validation with valid shapes."""
    builder = TRTEngineBuilder()

    shapes = {
        'audio': [(1, 1600), (1, 16000), (1, 160000)]
    }

    result = builder.supports_dynamic_shapes(shapes)
    assert result is True


def test_supports_dynamic_shapes_invalid():
    """Test dynamic shape validation with invalid shapes."""
    builder = TRTEngineBuilder()

    # Optimal > max (invalid)
    shapes = {
        'audio': [(1, 1600), (1, 200000), (1, 160000)]
    }

    result = builder.supports_dynamic_shapes(shapes)
    assert result is False


def test_supports_dynamic_shapes_min_greater_than_opt():
    """Test dynamic shape validation with min > opt."""
    builder = TRTEngineBuilder()

    shapes = {
        'audio': [(1, 20000), (1, 16000), (1, 160000)]
    }

    result = builder.supports_dynamic_shapes(shapes)
    assert result is False


def test_build_engine_no_tensorrt(tmp_path):
    """Test engine building fails gracefully without TensorRT."""
    builder = TRTEngineBuilder()

    # Create dummy ONNX file
    dummy_onnx = tmp_path / "dummy.onnx"
    dummy_onnx.write_bytes(b"fake onnx")

    # Mock tensorrt import to fail
    with patch('builtins.__import__', side_effect=ImportError("No module named 'tensorrt'")):
        with pytest.raises(RuntimeError, match="TensorRT not available"):
            builder.build_engine(str(dummy_onnx), str(tmp_path / "dummy.trt"))


@pytest.mark.skip(reason="Requires TensorRT installation - tested in integration")
def test_build_engine_with_tensorrt():
    """Test engine building with TensorRT available.

    Skipped: Requires TensorRT to be installed.
    """
    pass


# ============================================================================
# TRT Inference Context Tests
# ============================================================================


def test_trt_inference_context_no_tensorrt(tmp_path):
    """Test TRTInferenceContext fails without TensorRT."""
    # Create dummy engine file
    dummy_engine = tmp_path / "dummy.trt"
    dummy_engine.write_bytes(b"fake engine")

    # Mock tensorrt import to fail
    with patch('builtins.__import__', side_effect=ImportError("No module named 'tensorrt'")):
        with pytest.raises(RuntimeError, match="TensorRT not available"):
            TRTInferenceContext(str(dummy_engine))


def test_trt_inference_context_missing_engine(temp_engine_dir):
    """Test TRTInferenceContext fails with missing engine file."""
    engine_path = os.path.join(temp_engine_dir, "nonexistent.trt")

    with pytest.raises(RuntimeError, match="TRT engine not found"):
        TRTInferenceContext(engine_path)


@pytest.mark.skip(reason="Requires TensorRT and valid engine - tested in integration")
def test_trt_inference_context_load_engine():
    """Test loading valid TRT engine.

    Skipped: Requires TensorRT and pre-built engine.
    """
    pass


@pytest.mark.skip(reason="Requires TensorRT - tested in integration")
def test_trt_inference_context_infer():
    """Test inference with TRT context.

    Skipped: Requires TensorRT.
    """
    pass


# ============================================================================
# TRT Conversion Pipeline Tests
# ============================================================================


def test_trt_conversion_pipeline_init_missing_dir():
    """Test TRTConversionPipeline fails with missing engine directory."""
    with pytest.raises(RuntimeError, match="Engine directory not found"):
        TRTConversionPipeline("/nonexistent/path")


@patch('auto_voice.inference.trt_pipeline.TRTInferenceContext')
@patch('auto_voice.audio.separator.MelBandRoFormer')
def test_trt_conversion_pipeline_init(mock_separator, mock_ctx, temp_engine_dir):
    """Test TRTConversionPipeline initialization."""
    # Create mock engine files
    for name in ['content_extractor.trt', 'pitch_extractor.trt',
                 'decoder.trt', 'vocoder.trt']:
        Path(temp_engine_dir, name).touch()

    pipeline = TRTConversionPipeline(temp_engine_dir, device=torch.device('cpu'))

    assert pipeline.engine_dir == Path(temp_engine_dir)
    assert pipeline.device == torch.device('cpu')


@patch('auto_voice.inference.trt_pipeline.TRTInferenceContext')
@patch('auto_voice.audio.separator.MelBandRoFormer')
def test_trt_conversion_pipeline_load_engines(mock_separator, mock_ctx, temp_engine_dir):
    """Test TRTConversionPipeline loads existing engines."""
    # Create mock engine files
    for name in ['content_extractor.trt', 'pitch_extractor.trt',
                 'decoder.trt', 'vocoder.trt']:
        Path(temp_engine_dir, name).touch()

    pipeline = TRTConversionPipeline(temp_engine_dir)

    # Should load all 4 engines
    assert mock_ctx.call_count == 4


@patch('auto_voice.inference.trt_pipeline.TRTInferenceContext')
@patch('auto_voice.inference.trt_pipeline.ONNXExporter')
@patch('auto_voice.inference.trt_pipeline.TRTEngineBuilder')
@patch('auto_voice.audio.separator.MelBandRoFormer')
def test_trt_conversion_pipeline_build_missing_engines(
    mock_separator, mock_builder_class, mock_exporter_class, mock_ctx, temp_engine_dir
):
    """Test TRTConversionPipeline builds missing engines."""
    # Don't create engine files - they should be built
    mock_builder = MagicMock()
    mock_builder_class.return_value = mock_builder

    mock_exporter = MagicMock()
    mock_exporter_class.return_value = mock_exporter

    # Patch model imports
    with patch('auto_voice.models.encoder.ContentVecEncoder'), \
         patch('auto_voice.models.pitch.RMVPEPitchExtractor'), \
         patch('auto_voice.models.svc_decoder.CoMoSVCDecoder'), \
         patch('auto_voice.models.vocoder.BigVGANGenerator'):

        pipeline = TRTConversionPipeline(temp_engine_dir)

        # Should build engines
        assert mock_builder.build_engine.call_count == 4


def test_trt_resample_same_rate():
    """Test resampling with same sample rate is no-op."""
    with patch('auto_voice.inference.trt_pipeline.TRTInferenceContext'), \
         patch('auto_voice.audio.separator.MelBandRoFormer'), \
         patch('os.path.exists', return_value=True):

        pipeline = TRTConversionPipeline.__new__(TRTConversionPipeline)
        pipeline.device = torch.device('cpu')

        audio = torch.randn(16000)
        result = pipeline._resample(audio, 16000, 16000)

        assert torch.equal(result, audio)


def test_trt_resample_different_rate():
    """Test resampling with different sample rates."""
    with patch('auto_voice.inference.trt_pipeline.TRTInferenceContext'), \
         patch('auto_voice.audio.separator.MelBandRoFormer'), \
         patch('os.path.exists', return_value=True):

        pipeline = TRTConversionPipeline.__new__(TRTConversionPipeline)
        pipeline.device = torch.device('cpu')

        audio = torch.randn(16000)
        result = pipeline._resample(audio, 16000, 8000)

        # Should be approximately half length
        assert abs(result.shape[0] - 8000) < 10


def test_trt_to_mono_already_mono():
    """Test converting already mono audio."""
    with patch('auto_voice.inference.trt_pipeline.TRTInferenceContext'), \
         patch('auto_voice.audio.separator.MelBandRoFormer'), \
         patch('os.path.exists', return_value=True):

        pipeline = TRTConversionPipeline.__new__(TRTConversionPipeline)

        audio = torch.randn(16000)
        result = pipeline._to_mono(audio)

        assert torch.equal(result, audio)


def test_trt_to_mono_stereo():
    """Test converting stereo to mono."""
    with patch('auto_voice.inference.trt_pipeline.TRTInferenceContext'), \
         patch('auto_voice.audio.separator.MelBandRoFormer'), \
         patch('os.path.exists', return_value=True):

        pipeline = TRTConversionPipeline.__new__(TRTConversionPipeline)

        audio = torch.randn(2, 16000)
        result = pipeline._to_mono(audio)

        assert result.dim() == 1
        assert result.shape[0] == 16000


def test_trt_to_mono_invalid_shape():
    """Test converting invalid audio shape raises error."""
    with patch('auto_voice.inference.trt_pipeline.TRTInferenceContext'), \
         patch('auto_voice.audio.separator.MelBandRoFormer'), \
         patch('os.path.exists', return_value=True):

        pipeline = TRTConversionPipeline.__new__(TRTConversionPipeline)

        audio = torch.randn(2, 2, 16000)  # 3D tensor

        with pytest.raises(RuntimeError, match="Unexpected audio shape"):
            pipeline._to_mono(audio)


def test_trt_encode_pitch():
    """Test pitch encoding to embeddings."""
    with patch('auto_voice.inference.trt_pipeline.TRTInferenceContext'), \
         patch('auto_voice.audio.separator.MelBandRoFormer'), \
         patch('os.path.exists', return_value=True):

        pipeline = TRTConversionPipeline.__new__(TRTConversionPipeline)
        pipeline.device = torch.device('cpu')

        f0 = torch.randn(1, 100).abs() + 50  # Ensure positive F0
        pitch_embeddings = pipeline._encode_pitch(f0)

        assert pitch_embeddings.shape == (1, 100, DEFAULT_PITCH_DIM)


@patch('auto_voice.inference.trt_pipeline.TRTInferenceContext')
@patch('auto_voice.audio.separator.MelBandRoFormer')
def test_trt_convert_empty_audio(mock_separator, mock_ctx, temp_engine_dir):
    """Test conversion fails with empty audio."""
    # Create engine files
    for name in ['content_extractor.trt', 'pitch_extractor.trt',
                 'decoder.trt', 'vocoder.trt']:
        Path(temp_engine_dir, name).touch()

    pipeline = TRTConversionPipeline(temp_engine_dir)

    empty_audio = torch.tensor([])
    speaker = torch.randn(256)

    with pytest.raises(RuntimeError, match="Empty audio input"):
        pipeline.convert(empty_audio, 44100, speaker)


@patch('auto_voice.inference.trt_pipeline.TRTInferenceContext')
@patch('auto_voice.audio.separator.MelBandRoFormer')
def test_trt_convert_invalid_speaker_embedding(mock_separator, mock_ctx, temp_engine_dir):
    """Test conversion fails with invalid speaker embedding shape."""
    # Create engine files
    for name in ['content_extractor.trt', 'pitch_extractor.trt',
                 'decoder.trt', 'vocoder.trt']:
        Path(temp_engine_dir, name).touch()

    pipeline = TRTConversionPipeline(temp_engine_dir)

    audio = torch.randn(16000)
    invalid_speaker = torch.randn(128)  # Wrong size

    with pytest.raises(RuntimeError, match="Speaker embedding must be"):
        pipeline.convert(audio, 44100, invalid_speaker)


@pytest.mark.skip(reason="Requires TensorRT for full pipeline test")
def test_trt_convert_full_pipeline():
    """Test full conversion pipeline with TRT engines.

    Skipped: Requires TensorRT and pre-built engines.
    """
    pass


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_onnx_export_with_export_failure():
    """Test ONNX export handles torch.onnx.export failures."""
    exporter = ONNXExporter()

    with patch('torch.onnx.export', side_effect=RuntimeError("Export failed")):
        mock_model = MagicMock()
        mock_model.parameters.return_value = []

        with pytest.raises(RuntimeError, match="Export failed"):
            exporter.export_content_extractor(mock_model, "/tmp/output.onnx")


def test_trt_engine_builder_invalid_onnx(tmp_path):
    """Test engine builder handles invalid ONNX files."""
    builder = TRTEngineBuilder()

    onnx_path = tmp_path / "invalid.onnx"
    onnx_path.write_bytes(b"invalid onnx data")

    # Mock tensorrt with proper structure
    mock_trt = MagicMock()
    mock_builder = MagicMock()
    mock_network = MagicMock()
    mock_parser = MagicMock()

    # Parser fails to parse invalid ONNX
    mock_parser.parse.return_value = False
    mock_parser.num_errors = 1
    mock_parser.get_error.return_value = "Invalid ONNX format"

    mock_trt.Builder.return_value = mock_builder
    mock_builder.create_network.return_value = mock_network
    mock_trt.OnnxParser.return_value = mock_parser

    with patch.dict('sys.modules', {'tensorrt': mock_trt}):
        with pytest.raises(RuntimeError, match="ONNX parsing failed"):
            builder.build_engine(str(onnx_path), str(tmp_path / "output.trt"))


# ============================================================================
# TRT Inference Success Path Tests (Mocked)
# ============================================================================


def test_trt_build_engine_success_fp16(tmp_path):
    """Test successful TRT engine building with FP16 precision."""
    builder = TRTEngineBuilder(precision="fp16")

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake onnx model")
    engine_path = tmp_path / "model.trt"

    # Mock tensorrt with successful build
    mock_trt = MagicMock()
    mock_builder = MagicMock()
    mock_network = MagicMock()
    mock_parser = MagicMock()
    mock_config = MagicMock()

    mock_parser.parse.return_value = True
    mock_builder.create_network.return_value = mock_network
    mock_builder.create_builder_config.return_value = mock_config
    mock_builder.build_serialized_network.return_value = b"serialized engine"

    mock_trt.Builder.return_value = mock_builder
    mock_trt.OnnxParser.return_value = mock_parser
    mock_trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH = 0
    mock_trt.MemoryPoolType.WORKSPACE = 0
    mock_trt.BuilderFlag.FP16 = 1

    with patch.dict('sys.modules', {'tensorrt': mock_trt}):
        result = builder.build_engine(str(onnx_path), str(engine_path))

        assert result == str(engine_path)
        assert engine_path.exists()
        mock_config.set_flag.assert_called_once()


def test_trt_build_engine_success_fp32(tmp_path):
    """Test successful TRT engine building with FP32 precision."""
    builder = TRTEngineBuilder(precision="fp32")

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake onnx model")
    engine_path = tmp_path / "model.trt"

    # Mock tensorrt
    mock_trt = MagicMock()
    mock_builder = MagicMock()
    mock_network = MagicMock()
    mock_parser = MagicMock()
    mock_config = MagicMock()

    mock_parser.parse.return_value = True
    mock_builder.create_network.return_value = mock_network
    mock_builder.create_builder_config.return_value = mock_config
    mock_builder.build_serialized_network.return_value = b"serialized engine"

    mock_trt.Builder.return_value = mock_builder
    mock_trt.OnnxParser.return_value = mock_parser
    mock_trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH = 0
    mock_trt.MemoryPoolType.WORKSPACE = 0

    with patch.dict('sys.modules', {'tensorrt': mock_trt}):
        result = builder.build_engine(str(onnx_path), str(engine_path))

        assert result == str(engine_path)
        # Should not call set_flag for FP32
        mock_config.set_flag.assert_not_called()


def test_trt_build_engine_with_dynamic_shapes(tmp_path):
    """Test TRT engine building with dynamic shapes."""
    builder = TRTEngineBuilder()

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake onnx model")
    engine_path = tmp_path / "model.trt"

    dynamic_shapes = {
        'audio': [(1, 1600), (1, 16000), (1, 160000)]
    }

    # Mock tensorrt
    mock_trt = MagicMock()
    mock_builder = MagicMock()
    mock_network = MagicMock()
    mock_parser = MagicMock()
    mock_config = MagicMock()
    mock_profile = MagicMock()

    mock_parser.parse.return_value = True
    mock_builder.create_network.return_value = mock_network
    mock_builder.create_builder_config.return_value = mock_config
    mock_builder.create_optimization_profile.return_value = mock_profile
    mock_builder.build_serialized_network.return_value = b"serialized engine"

    mock_trt.Builder.return_value = mock_builder
    mock_trt.OnnxParser.return_value = mock_parser
    mock_trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH = 0
    mock_trt.MemoryPoolType.WORKSPACE = 0
    mock_trt.BuilderFlag.FP16 = 1

    with patch.dict('sys.modules', {'tensorrt': mock_trt}):
        result = builder.build_engine(str(onnx_path), str(engine_path), dynamic_shapes)

        assert result == str(engine_path)
        mock_profile.set_shape.assert_called_once()
        mock_config.add_optimization_profile.assert_called_once()


def test_trt_build_engine_serialization_failure(tmp_path):
    """Test TRT engine building fails when serialization returns None."""
    builder = TRTEngineBuilder()

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake onnx model")

    # Mock tensorrt with serialization failure
    mock_trt = MagicMock()
    mock_builder = MagicMock()
    mock_network = MagicMock()
    mock_parser = MagicMock()
    mock_config = MagicMock()

    mock_parser.parse.return_value = True
    mock_builder.create_network.return_value = mock_network
    mock_builder.create_builder_config.return_value = mock_config
    mock_builder.build_serialized_network.return_value = None  # Failure

    mock_trt.Builder.return_value = mock_builder
    mock_trt.OnnxParser.return_value = mock_parser
    mock_trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH = 0
    mock_trt.MemoryPoolType.WORKSPACE = 0
    mock_trt.BuilderFlag.FP16 = 1

    with patch.dict('sys.modules', {'tensorrt': mock_trt}):
        with pytest.raises(RuntimeError, match="TRT engine build failed"):
            builder.build_engine(str(onnx_path), str(tmp_path / "output.trt"))


def test_trt_inference_context_successful_init(tmp_path):
    """Test successful TRTInferenceContext initialization."""
    engine_path = tmp_path / "test.trt"
    engine_path.write_bytes(b"fake engine")

    # Mock tensorrt
    mock_trt = MagicMock()
    mock_runtime = MagicMock()
    mock_engine = MagicMock()
    mock_context = MagicMock()

    mock_engine.num_io_tensors = 2
    mock_engine.get_tensor_name.side_effect = ['audio', 'features']
    mock_trt.TensorIOMode.INPUT = 0
    mock_engine.get_tensor_mode.side_effect = [0, 1]  # input, output
    mock_engine.create_execution_context.return_value = mock_context

    mock_runtime.deserialize_cuda_engine.return_value = mock_engine
    mock_trt.Runtime.return_value = mock_runtime

    with patch.dict('sys.modules', {'tensorrt': mock_trt}), \
         patch('torch.cuda.Stream'):
        ctx = TRTInferenceContext(str(engine_path))

        assert ctx.engine == mock_engine
        assert ctx.context == mock_context
        assert ctx.input_names == ['audio']
        assert ctx.output_names == ['features']


def test_trt_inference_context_engine_deserialization_failure(tmp_path):
    """Test TRTInferenceContext fails when engine deserialization returns None."""
    engine_path = tmp_path / "test.trt"
    engine_path.write_bytes(b"corrupted engine")

    # Mock tensorrt with deserialization failure
    mock_trt = MagicMock()
    mock_runtime = MagicMock()
    mock_runtime.deserialize_cuda_engine.return_value = None

    mock_trt.Runtime.return_value = mock_runtime

    with patch.dict('sys.modules', {'tensorrt': mock_trt}):
        with pytest.raises(RuntimeError, match="Failed to load TRT engine"):
            TRTInferenceContext(str(engine_path))


def test_trt_inference_context_infer_mocked(tmp_path):
    """Test TRTInferenceContext.infer with mocked TensorRT."""
    engine_path = tmp_path / "test.trt"
    engine_path.write_bytes(b"fake engine")

    # Mock tensorrt
    mock_trt = MagicMock()
    mock_runtime = MagicMock()
    mock_engine = MagicMock()
    mock_context = MagicMock()
    mock_stream = MagicMock()

    mock_engine.num_io_tensors = 2
    mock_engine.get_tensor_name.side_effect = ['audio', 'features']
    mock_trt.TensorIOMode.INPUT = 0
    mock_engine.get_tensor_mode.side_effect = [0, 1]
    mock_engine.create_execution_context.return_value = mock_context
    mock_engine.get_tensor_dtype.return_value = mock_trt.float32
    mock_trt.nptype.return_value = np.float32

    mock_context.get_tensor_shape.return_value = [1, 100, 768]

    mock_runtime.deserialize_cuda_engine.return_value = mock_engine
    mock_trt.Runtime.return_value = mock_runtime

    with patch.dict('sys.modules', {'tensorrt': mock_trt}), \
         patch('torch.cuda.Stream', return_value=mock_stream):
        ctx = TRTInferenceContext(str(engine_path))

        # Mock inference
        input_tensor = torch.randn(1, 16000, device='cuda')
        outputs = ctx.infer({'audio': input_tensor})

        assert 'features' in outputs
        assert outputs['features'].shape == torch.Size([1, 100, 768])
        mock_context.execute_async_v3.assert_called_once()
        mock_stream.synchronize.assert_called_once()


# ============================================================================
# TRT Conversion Pipeline Full Workflow Tests (Mocked)
# ============================================================================


@patch('auto_voice.inference.trt_pipeline.TRTInferenceContext')
@patch('auto_voice.audio.separator.MelBandRoFormer')
def test_trt_convert_full_mocked_workflow(mock_separator_class, mock_ctx_class, temp_engine_dir):
    """Test full conversion pipeline with mocked TRT engines."""
    # Create engine files
    for name in ['content_extractor.trt', 'pitch_extractor.trt',
                 'decoder.trt', 'vocoder.trt']:
        Path(temp_engine_dir, name).touch()

    # Mock separator
    mock_separator = MagicMock()
    mock_separator.extract_vocals.return_value = torch.randn(1, 44100)
    mock_separator_class.return_value = mock_separator

    # Mock TRT contexts
    mock_content_ctx = MagicMock()
    mock_pitch_ctx = MagicMock()
    mock_decoder_ctx = MagicMock()
    mock_vocoder_ctx = MagicMock()

    mock_content_ctx.infer.return_value = {'features': torch.randn(1, 100, 768, device='cuda')}
    mock_pitch_ctx.infer.return_value = {'f0': torch.randn(1, 100, device='cuda').abs() + 50}
    mock_decoder_ctx.infer.return_value = {'mel': torch.randn(1, 100, 100, device='cuda')}
    mock_vocoder_ctx.infer.return_value = {'audio': torch.randn(1, 24000, device='cuda')}

    mock_ctx_class.side_effect = [mock_content_ctx, mock_pitch_ctx, mock_decoder_ctx, mock_vocoder_ctx]

    with patch('torch.cuda.Stream'):
        pipeline = TRTConversionPipeline(temp_engine_dir, device=torch.device('cuda'))

        audio = torch.randn(44100)
        speaker = torch.randn(256)

        result = pipeline.convert(audio, 44100, speaker)

        assert 'audio' in result
        assert 'sample_rate' in result
        assert 'metadata' in result
        assert result['sample_rate'] == 24000
        assert result['metadata']['backend'] == 'tensorrt'
        assert result['metadata']['precision'] == 'fp16'
        assert 'processing_time' in result['metadata']


@patch('auto_voice.inference.trt_pipeline.TRTInferenceContext')
@patch('auto_voice.audio.separator.MelBandRoFormer')
def test_trt_convert_with_progress_callback(mock_separator_class, mock_ctx_class, temp_engine_dir):
    """Test conversion pipeline calls progress callback."""
    # Create engine files
    for name in ['content_extractor.trt', 'pitch_extractor.trt',
                 'decoder.trt', 'vocoder.trt']:
        Path(temp_engine_dir, name).touch()

    # Mock separator
    mock_separator = MagicMock()
    mock_separator.extract_vocals.return_value = torch.randn(1, 44100)
    mock_separator_class.return_value = mock_separator

    # Mock TRT contexts
    mock_content_ctx = MagicMock()
    mock_pitch_ctx = MagicMock()
    mock_decoder_ctx = MagicMock()
    mock_vocoder_ctx = MagicMock()

    mock_content_ctx.infer.return_value = {'features': torch.randn(1, 100, 768, device='cuda')}
    mock_pitch_ctx.infer.return_value = {'f0': torch.randn(1, 100, device='cuda').abs() + 50}
    mock_decoder_ctx.infer.return_value = {'mel': torch.randn(1, 100, 100, device='cuda')}
    mock_vocoder_ctx.infer.return_value = {'audio': torch.randn(1, 24000, device='cuda')}

    mock_ctx_class.side_effect = [mock_content_ctx, mock_pitch_ctx, mock_decoder_ctx, mock_vocoder_ctx]

    with patch('torch.cuda.Stream'):
        pipeline = TRTConversionPipeline(temp_engine_dir, device=torch.device('cuda'))

        audio = torch.randn(44100)
        speaker = torch.randn(256)

        # Track progress calls
        progress_calls = []

        def on_progress(stage, progress):
            progress_calls.append((stage, progress))

        result = pipeline.convert(audio, 44100, speaker, on_progress=on_progress)

        # Verify progress was reported
        assert len(progress_calls) > 0
        stages = [stage for stage, _ in progress_calls]
        assert 'separation' in stages
        assert 'content_extraction' in stages
        assert 'pitch_extraction' in stages
        assert 'decoding' in stages
        assert 'vocoder' in stages


@patch('auto_voice.inference.trt_pipeline.TRTInferenceContext')
@patch('auto_voice.audio.separator.MelBandRoFormer')
def test_trt_convert_stereo_input(mock_separator_class, mock_ctx_class, temp_engine_dir):
    """Test conversion pipeline converts stereo to mono."""
    # Create engine files
    for name in ['content_extractor.trt', 'pitch_extractor.trt',
                 'decoder.trt', 'vocoder.trt']:
        Path(temp_engine_dir, name).touch()

    # Mock separator
    mock_separator = MagicMock()
    mock_separator.extract_vocals.return_value = torch.randn(1, 44100)
    mock_separator_class.return_value = mock_separator

    # Mock TRT contexts
    mock_ctx_class.return_value = MagicMock()
    mock_ctx_class.return_value.infer.return_value = {
        'features': torch.randn(1, 100, 768, device='cuda'),
        'f0': torch.randn(1, 100, device='cuda').abs() + 50,
        'mel': torch.randn(1, 100, 100, device='cuda'),
        'audio': torch.randn(1, 24000, device='cuda'),
    }

    with patch('torch.cuda.Stream'):
        pipeline = TRTConversionPipeline(temp_engine_dir, device=torch.device('cuda'))

        # Stereo input
        audio = torch.randn(2, 44100)
        speaker = torch.randn(256)

        result = pipeline.convert(audio, 44100, speaker)

        # Should succeed by converting to mono
        assert result['audio'].shape[0] > 0


# ============================================================================
# Memory Usage Tests
# ============================================================================


@patch('auto_voice.inference.trt_pipeline.TRTInferenceContext')
@patch('auto_voice.audio.separator.MelBandRoFormer')
def test_get_engine_memory_usage(mock_separator, mock_ctx_class, temp_engine_dir):
    """Test getting memory usage of TRT engines."""
    # Create engine files
    for name in ['content_extractor.trt', 'pitch_extractor.trt',
                 'decoder.trt', 'vocoder.trt']:
        Path(temp_engine_dir, name).touch()

    # Mock contexts with memory usage
    mock_ctx = MagicMock()
    mock_ctx.get_memory_usage.return_value = 1024 * 1024 * 100  # 100MB
    mock_ctx_class.return_value = mock_ctx

    pipeline = TRTConversionPipeline(temp_engine_dir)

    total_memory = pipeline.get_engine_memory_usage()

    # 4 engines * 100MB each
    assert total_memory == 4 * 1024 * 1024 * 100


def test_trt_context_memory_usage():
    """Test TRTInferenceContext memory reporting with mocked engine."""
    with patch('builtins.open', mock_open(read_data=b'fake engine')), \
         patch('os.path.exists', return_value=True):

        # Mock tensorrt module
        mock_trt = MagicMock()
        mock_runtime = MagicMock()
        mock_engine = MagicMock()
        mock_engine.device_memory_size = 1024 * 1024 * 50  # 50MB
        mock_runtime.deserialize_cuda_engine.return_value = mock_engine
        mock_trt.Runtime.return_value = mock_runtime

        with patch.dict('sys.modules', {'tensorrt': mock_trt}):
            ctx = TRTInferenceContext('/tmp/test.trt')

            memory = ctx.get_memory_usage()
            assert memory == 1024 * 1024 * 50
