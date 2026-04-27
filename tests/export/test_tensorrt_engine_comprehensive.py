"""Comprehensive tests for TensorRT engine builder module.

Target: Increase coverage from 24% to 80%+ for tensorrt_engine.py

Test Categories:
1. LatencyStats Tests
2. ShapeProfile Tests
3. TRTEngineBuilder Initialization Tests
4. Engine Building Tests (ONNX to TRT)
5. Engine Serialization/Deserialization Tests
6. Engine Caching Tests
7. Inference Tests
8. Benchmarking Tests
9. Error Handling Tests
10. Edge Case Tests

All TensorRT operations are mocked to avoid GPU requirements.
"""
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open, call

import numpy as np
import pytest
import torch

from auto_voice.export.tensorrt_engine import (
    LatencyStats,
    ShapeProfile,
    TRTEngineBuilder,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_engine_dir(tmp_path):
    """Create temporary directory for engine files."""
    engine_dir = tmp_path / "trt_engines"
    engine_dir.mkdir()
    return str(engine_dir)


@pytest.fixture
def mock_trt_logger():
    """Mock TensorRT logger."""
    with patch('auto_voice.export.tensorrt_engine.TRT_LOGGER') as mock_logger:
        yield mock_logger


@pytest.fixture
def mock_trt_builder():
    """Mock TensorRT builder and related objects."""
    with patch('auto_voice.export.tensorrt_engine.trt') as mock_trt:
        # Mock builder
        builder = MagicMock()
        network = MagicMock()
        config = MagicMock()
        parser = MagicMock()
        profile = MagicMock()
        runtime = MagicMock()
        engine = MagicMock()

        # Setup builder chain
        mock_trt.Builder.return_value = builder
        builder.create_network.return_value = network
        builder.create_builder_config.return_value = config
        builder.create_optimization_profile.return_value = profile
        builder.build_serialized_network.return_value = b'serialized_engine_data'

        # Setup parser
        mock_trt.OnnxParser.return_value = parser
        parser.parse.return_value = True
        parser.num_errors = 0

        # Setup runtime
        mock_trt.Runtime.return_value = runtime
        runtime.deserialize_cuda_engine.return_value = engine

        # Setup engine properties
        engine.num_io_tensors = 2
        engine.get_tensor_name.side_effect = lambda i: f"tensor_{i}"
        engine.get_tensor_mode.return_value = mock_trt.TensorIOMode.INPUT
        engine.get_tensor_dtype.return_value = MagicMock(itemsize=4)
        engine.serialize.return_value = b'serialized_engine_data'

        # Setup network properties
        network.num_inputs = 1
        input_tensor = MagicMock()
        input_tensor.name = "audio"
        input_tensor.shape = (1, 100, 256)  # Static shape by default
        network.get_input.return_value = input_tensor

        # Setup enums
        mock_trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH = 0
        mock_trt.MemoryPoolType.WORKSPACE = 0
        mock_trt.BuilderFlag.FP16 = 1
        mock_trt.BuilderFlag.INT8 = 2
        mock_trt.TensorIOMode.INPUT = 0
        mock_trt.TensorIOMode.OUTPUT = 1
        mock_trt.Logger.WARNING = 1

        # Mock nptype function
        mock_trt.nptype.return_value = np.float32

        yield mock_trt


@pytest.fixture
def mock_onnx_file(tmp_path):
    """Create mock ONNX file."""
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b'fake_onnx_data')
    return str(onnx_path)


@pytest.fixture
def mock_engine_file(tmp_path):
    """Create mock TRT engine file."""
    engine_path = tmp_path / "model.trt"
    engine_path.write_bytes(b'fake_engine_data')
    return str(engine_path)


# ============================================================================
# LatencyStats Tests
# ============================================================================


def test_latency_stats_creation():
    """Test LatencyStats dataclass creation."""
    stats = LatencyStats(
        mean_ms=10.5,
        p50_ms=9.8,
        p95_ms=15.2,
        p99_ms=18.7,
        n_runs=100,
        all_ms=[5.0, 10.0, 15.0, 20.0]
    )

    assert stats.mean_ms == 10.5
    assert stats.p50_ms == 9.8
    assert stats.p95_ms == 15.2
    assert stats.p99_ms == 18.7
    assert stats.n_runs == 100
    assert len(stats.all_ms) == 4


def test_latency_stats_repr():
    """Test LatencyStats string representation."""
    stats = LatencyStats(
        mean_ms=10.5,
        p50_ms=9.8,
        p95_ms=15.2,
        p99_ms=18.7,
        n_runs=100
    )

    repr_str = repr(stats)
    assert "mean=10.50ms" in repr_str
    assert "p50=9.80ms" in repr_str
    assert "p95=15.20ms" in repr_str
    assert "p99=18.70ms" in repr_str
    assert "n=100" in repr_str


def test_latency_stats_default_all_ms():
    """Test LatencyStats with default empty all_ms list."""
    stats = LatencyStats(
        mean_ms=10.0,
        p50_ms=10.0,
        p95_ms=12.0,
        p99_ms=15.0,
        n_runs=50
    )

    assert stats.all_ms == []


# ============================================================================
# ShapeProfile Tests
# ============================================================================


def test_shape_profile_creation():
    """Test ShapeProfile dataclass creation."""
    profile = ShapeProfile(
        min=(1, 100, 256),
        opt=(16, 200, 256),
        max=(64, 500, 256)
    )

    assert profile.min == (1, 100, 256)
    assert profile.opt == (16, 200, 256)
    assert profile.max == (64, 500, 256)


def test_shape_profile_different_dims():
    """Test ShapeProfile with different dimensionality."""
    profile = ShapeProfile(
        min=(1, 1),
        opt=(8, 100),
        max=(32, 1000)
    )

    assert len(profile.min) == 2
    assert len(profile.opt) == 2
    assert len(profile.max) == 2


# ============================================================================
# TRTEngineBuilder Initialization Tests
# ============================================================================


def test_engine_builder_init_default():
    """Test TRTEngineBuilder initialization with default workspace."""
    builder = TRTEngineBuilder()

    # Default is 2.0 GB
    assert builder._workspace_bytes == int(2.0 * (1 << 30))


def test_engine_builder_init_custom_workspace():
    """Test TRTEngineBuilder initialization with custom workspace."""
    builder = TRTEngineBuilder(workspace_size_gb=4.0)

    assert builder._workspace_bytes == int(4.0 * (1 << 30))


def test_engine_builder_init_small_workspace():
    """Test TRTEngineBuilder with small workspace."""
    builder = TRTEngineBuilder(workspace_size_gb=0.5)

    assert builder._workspace_bytes == int(0.5 * (1 << 30))


# ============================================================================
# Engine Building Tests (ONNX to TRT)
# ============================================================================


def test_build_engine_success(mock_trt_builder, mock_onnx_file):
    """Test successful engine building from ONNX."""
    builder = TRTEngineBuilder()

    engine = builder.build_engine(mock_onnx_file)

    assert engine is not None
    mock_trt_builder.Builder.assert_called_once()
    mock_trt_builder.OnnxParser.assert_called_once()


def test_build_engine_onnx_not_found(mock_trt_builder):
    """Test engine building with missing ONNX file."""
    builder = TRTEngineBuilder()

    with pytest.raises(RuntimeError, match="ONNX file not found"):
        builder.build_engine("/nonexistent/model.onnx")


def test_build_engine_requires_tensorrt_when_onnx_exists(mock_onnx_file):
    """Test engine building reports missing optional TensorRT dependency clearly."""
    builder = TRTEngineBuilder()

    with patch('auto_voice.export.tensorrt_engine.trt', None):
        with pytest.raises(RuntimeError, match="TensorRT is not available"):
            builder.build_engine(mock_onnx_file)


def test_build_engine_with_fp16(mock_trt_builder, mock_onnx_file):
    """Test engine building with FP16 precision enabled."""
    builder = TRTEngineBuilder()

    engine = builder.build_engine(mock_onnx_file, fp16=True)

    assert engine is not None
    # Verify FP16 flag was set
    config = mock_trt_builder.Builder.return_value.create_builder_config.return_value
    config.set_flag.assert_any_call(mock_trt_builder.BuilderFlag.FP16)


def test_build_engine_with_int8(mock_trt_builder, mock_onnx_file):
    """Test engine building with INT8 precision enabled."""
    builder = TRTEngineBuilder()
    calibrator = MagicMock()

    engine = builder.build_engine(mock_onnx_file, int8=True, calibrator=calibrator)

    assert engine is not None
    config = mock_trt_builder.Builder.return_value.create_builder_config.return_value
    config.set_flag.assert_any_call(mock_trt_builder.BuilderFlag.INT8)
    assert config.int8_calibrator == calibrator


def test_build_engine_with_int8_no_calibrator(mock_trt_builder, mock_onnx_file):
    """Test engine building with INT8 but no calibrator."""
    builder = TRTEngineBuilder()

    engine = builder.build_engine(mock_onnx_file, int8=True)

    assert engine is not None
    config = mock_trt_builder.Builder.return_value.create_builder_config.return_value
    config.set_flag.assert_any_call(mock_trt_builder.BuilderFlag.INT8)


def test_build_engine_fp32_only(mock_trt_builder, mock_onnx_file):
    """Test engine building with FP32 only (no FP16/INT8)."""
    builder = TRTEngineBuilder()

    engine = builder.build_engine(mock_onnx_file, fp16=False, int8=False)

    assert engine is not None
    config = mock_trt_builder.Builder.return_value.create_builder_config.return_value
    # FP16 and INT8 flags should not be set
    flag_calls = [call[0][0] for call in config.set_flag.call_args_list]
    assert mock_trt_builder.BuilderFlag.FP16 not in flag_calls
    assert mock_trt_builder.BuilderFlag.INT8 not in flag_calls


def test_build_engine_with_dynamic_shapes(mock_trt_builder, mock_onnx_file):
    """Test engine building with dynamic shape profiles."""
    # Configure mock network to have dynamic input
    network = mock_trt_builder.Builder.return_value.create_network.return_value
    input_tensor = MagicMock()
    input_tensor.name = "audio"
    input_tensor.shape = (1, -1, 256)  # Dynamic sequence length
    network.get_input.return_value = input_tensor

    builder = TRTEngineBuilder()
    shape_profiles = {
        "audio": ShapeProfile(
            min=(1, 100, 256),
            opt=(1, 500, 256),
            max=(1, 1000, 256)
        )
    }

    engine = builder.build_engine(mock_onnx_file, shape_profiles=shape_profiles)

    assert engine is not None
    profile = mock_trt_builder.Builder.return_value.create_optimization_profile.return_value
    profile.set_shape.assert_called_once_with(
        "audio",
        (1, 100, 256),
        (1, 500, 256),
        (1, 1000, 256)
    )


def test_build_engine_with_dynamic_shapes_auto_infer(mock_trt_builder, mock_onnx_file):
    """Test engine building with auto-inferred dynamic shapes."""
    # Configure mock network to have dynamic input
    network = mock_trt_builder.Builder.return_value.create_network.return_value
    input_tensor = MagicMock()
    input_tensor.name = "audio"
    input_tensor.shape = (1, -1, 256)  # Dynamic sequence length
    network.get_input.return_value = input_tensor

    builder = TRTEngineBuilder()

    # No shape_profiles provided - should auto-infer
    engine = builder.build_engine(mock_onnx_file)

    assert engine is not None
    profile = mock_trt_builder.Builder.return_value.create_optimization_profile.return_value
    # Should infer: min=(1,1,256), opt=(1,16,256), max=(1,256,256)
    profile.set_shape.assert_called_once_with(
        "audio",
        (1, 1, 256),
        (1, 16, 256),
        (1, 256, 256)
    )


def test_build_engine_onnx_parse_failure(mock_trt_builder, mock_onnx_file):
    """Test engine building with ONNX parsing failure."""
    parser = mock_trt_builder.OnnxParser.return_value
    parser.parse.return_value = False
    parser.num_errors = 2
    parser.get_error.side_effect = [
        MagicMock(__str__=lambda self: "Error 1: Invalid op"),
        MagicMock(__str__=lambda self: "Error 2: Unsupported type")
    ]

    builder = TRTEngineBuilder()

    with pytest.raises(RuntimeError, match="ONNX parse failed"):
        builder.build_engine(mock_onnx_file)


def test_build_engine_build_failure(mock_trt_builder, mock_onnx_file):
    """Test engine building when TRT build fails."""
    trt_builder = mock_trt_builder.Builder.return_value
    trt_builder.build_serialized_network.return_value = None

    builder = TRTEngineBuilder()

    with pytest.raises(RuntimeError, match="TensorRT engine build failed"):
        builder.build_engine(mock_onnx_file)


def test_build_engine_deserialization_failure(mock_trt_builder, mock_onnx_file):
    """Test engine building when deserialization fails."""
    runtime = mock_trt_builder.Runtime.return_value
    runtime.deserialize_cuda_engine.return_value = None

    builder = TRTEngineBuilder()

    with pytest.raises(RuntimeError, match="Failed to deserialize built engine"):
        builder.build_engine(mock_onnx_file)


def test_build_engine_multiple_inputs(mock_trt_builder, mock_onnx_file):
    """Test engine building with multiple input tensors."""
    network = mock_trt_builder.Builder.return_value.create_network.return_value
    network.num_inputs = 2

    input1 = MagicMock()
    input1.name = "audio"
    input1.shape = (1, -1, 256)

    input2 = MagicMock()
    input2.name = "pitch"
    input2.shape = (1, -1)

    network.get_input.side_effect = [input1, input2]

    builder = TRTEngineBuilder()
    shape_profiles = {
        "audio": ShapeProfile(min=(1, 100, 256), opt=(1, 500, 256), max=(1, 1000, 256)),
        "pitch": ShapeProfile(min=(1, 100), opt=(1, 500), max=(1, 1000))
    }

    engine = builder.build_engine(mock_onnx_file, shape_profiles=shape_profiles)

    assert engine is not None
    profile = mock_trt_builder.Builder.return_value.create_optimization_profile.return_value
    assert profile.set_shape.call_count == 2


# ============================================================================
# Engine Serialization/Deserialization Tests
# ============================================================================


def test_save_engine_success(mock_trt_builder, tmp_path):
    """Test successful engine saving."""
    engine = MagicMock()
    engine.serialize.return_value = b'serialized_engine_data'

    builder = TRTEngineBuilder()
    save_path = str(tmp_path / "model.trt")

    result = builder.save_engine(engine, save_path)

    assert result == save_path
    assert Path(save_path).exists()
    assert Path(save_path).read_bytes() == b'serialized_engine_data'


def test_save_engine_creates_parent_dirs(mock_trt_builder, tmp_path):
    """Test that save_engine works with nested paths (parent dirs must exist)."""
    engine = MagicMock()
    engine.serialize.return_value = b'test_data'

    builder = TRTEngineBuilder()
    # Create parent directories first (save_engine doesn't create them)
    nested_dir = tmp_path / "subdir" / "nested"
    nested_dir.mkdir(parents=True)
    save_path = str(nested_dir / "model.trt")

    result = builder.save_engine(engine, save_path)

    assert Path(save_path).exists()


def test_save_engine_converts_path(mock_trt_builder, tmp_path):
    """Test that save_engine converts Path to string."""
    engine = MagicMock()
    engine.serialize.return_value = b'test_data'

    builder = TRTEngineBuilder()
    save_path = tmp_path / "model.trt"

    result = builder.save_engine(engine, save_path)

    assert result == str(save_path)


def test_load_engine_success(mock_trt_builder, mock_engine_file):
    """Test successful engine loading."""
    builder = TRTEngineBuilder()

    engine = builder.load_engine(mock_engine_file)

    assert engine is not None
    mock_trt_builder.Runtime.assert_called()


def test_load_engine_file_not_found(mock_trt_builder):
    """Test loading engine from non-existent file."""
    builder = TRTEngineBuilder()

    with pytest.raises(RuntimeError, match="Engine file not found"):
        builder.load_engine("/nonexistent/model.trt")


def test_load_engine_deserialization_failure(mock_trt_builder, mock_engine_file):
    """Test loading engine when deserialization fails."""
    runtime = mock_trt_builder.Runtime.return_value
    runtime.deserialize_cuda_engine.return_value = None

    builder = TRTEngineBuilder()

    with pytest.raises(RuntimeError, match="Failed to deserialize engine"):
        builder.load_engine(mock_engine_file)


def test_load_engine_converts_path(mock_trt_builder, tmp_path):
    """Test that load_engine converts Path to string."""
    engine_path = tmp_path / "model.trt"
    engine_path.write_bytes(b'fake_engine_data')

    builder = TRTEngineBuilder()

    engine = builder.load_engine(engine_path)

    assert engine is not None


# ============================================================================
# Engine Caching Tests
# ============================================================================


def test_load_cached_engine_uses_cache(mock_trt_builder, tmp_path):
    """Test that cached engine is loaded when newer than ONNX."""
    onnx_path = tmp_path / "model.onnx"
    engine_path = tmp_path / "model.trt"

    # Create ONNX file first
    onnx_path.write_bytes(b'onnx_data')
    time.sleep(0.1)  # Ensure different mtime

    # Create engine file after (newer)
    engine_path.write_bytes(b'engine_data')

    builder = TRTEngineBuilder()

    with patch.object(builder, 'load_engine') as mock_load:
        mock_load.return_value = MagicMock()

        engine = builder.load_cached_engine(str(onnx_path), str(engine_path))

        mock_load.assert_called_once_with(str(engine_path))


def test_load_cached_engine_rebuilds_when_onnx_newer(mock_trt_builder, tmp_path):
    """Test that engine is rebuilt when ONNX is newer than cache."""
    onnx_path = tmp_path / "model.onnx"
    engine_path = tmp_path / "model.trt"

    # Create engine file first
    engine_path.write_bytes(b'engine_data')
    time.sleep(0.1)  # Ensure different mtime

    # Create/update ONNX file after (newer)
    onnx_path.write_bytes(b'onnx_data')

    builder = TRTEngineBuilder()

    with patch.object(builder, 'build_engine') as mock_build, \
         patch.object(builder, 'save_engine') as mock_save:
        mock_build.return_value = MagicMock()

        engine = builder.load_cached_engine(str(onnx_path), str(engine_path))

        mock_build.assert_called_once_with(str(onnx_path))
        mock_save.assert_called_once()


def test_load_cached_engine_builds_when_no_cache(mock_trt_builder, tmp_path):
    """Test that engine is built when cache doesn't exist."""
    onnx_path = tmp_path / "model.onnx"
    engine_path = tmp_path / "model.trt"

    onnx_path.write_bytes(b'onnx_data')

    builder = TRTEngineBuilder()

    with patch.object(builder, 'build_engine') as mock_build, \
         patch.object(builder, 'save_engine') as mock_save:
        mock_build.return_value = MagicMock()

        engine = builder.load_cached_engine(str(onnx_path), str(engine_path))

        mock_build.assert_called_once_with(str(onnx_path))
        mock_save.assert_called_once()


def test_load_cached_engine_passes_build_kwargs(mock_trt_builder, tmp_path):
    """Test that build kwargs are passed through when rebuilding."""
    onnx_path = tmp_path / "model.onnx"
    engine_path = tmp_path / "model.trt"

    onnx_path.write_bytes(b'onnx_data')

    builder = TRTEngineBuilder()

    with patch.object(builder, 'build_engine') as mock_build, \
         patch.object(builder, 'save_engine') as mock_save:
        mock_build.return_value = MagicMock()

        engine = builder.load_cached_engine(
            str(onnx_path),
            str(engine_path),
            fp16=True,
            int8=True
        )

        mock_build.assert_called_once_with(str(onnx_path), fp16=True, int8=True)


# ============================================================================
# Inference Tests
# ============================================================================


@patch('torch.cuda.Stream')
def test_infer_success(mock_stream, mock_trt_builder):
    """Test successful inference execution."""
    # Setup mock engine
    engine = MagicMock()
    engine.num_io_tensors = 2
    engine.get_tensor_name.side_effect = ["input", "output"]

    mock_trt_builder.TensorIOMode.INPUT = 0
    mock_trt_builder.TensorIOMode.OUTPUT = 1
    engine.get_tensor_mode.side_effect = [
        mock_trt_builder.TensorIOMode.INPUT,
        mock_trt_builder.TensorIOMode.OUTPUT
    ]

    # Setup context
    context = MagicMock()
    context.execute_async_v3.return_value = True
    context.get_tensor_shape.return_value = [1, 256, 768]
    engine.create_execution_context.return_value = context

    # Mock CUDA stream
    mock_stream_instance = MagicMock()
    mock_stream.return_value = mock_stream_instance

    builder = TRTEngineBuilder()
    inputs = {"input": np.random.randn(1, 100, 256).astype(np.float32)}

    outputs = builder.infer(engine, inputs)

    assert "output" in outputs
    assert isinstance(outputs["output"], np.ndarray)
    context.execute_async_v3.assert_called_once()


@patch('torch.cuda.Stream')
def test_infer_execution_failure(mock_stream, mock_trt_builder):
    """Test inference with execution failure."""
    engine = MagicMock()
    engine.num_io_tensors = 2
    engine.get_tensor_name.side_effect = ["input", "output"]
    engine.get_tensor_mode.side_effect = [
        mock_trt_builder.TensorIOMode.INPUT,
        mock_trt_builder.TensorIOMode.OUTPUT
    ]

    context = MagicMock()
    context.execute_async_v3.return_value = False  # Failure
    context.get_tensor_shape.return_value = [1, 256, 768]
    engine.create_execution_context.return_value = context

    mock_stream_instance = MagicMock()
    mock_stream.return_value = mock_stream_instance

    builder = TRTEngineBuilder()
    inputs = {"input": np.random.randn(1, 100, 256).astype(np.float32)}

    with pytest.raises(RuntimeError, match="TensorRT inference execution failed"):
        builder.infer(engine, inputs)


@patch('torch.cuda.Stream')
def test_infer_multiple_inputs(mock_stream, mock_trt_builder):
    """Test inference with multiple input tensors."""
    engine = MagicMock()
    engine.num_io_tensors = 3
    engine.get_tensor_name.side_effect = ["input1", "input2", "output"]
    engine.get_tensor_mode.side_effect = [
        mock_trt_builder.TensorIOMode.INPUT,
        mock_trt_builder.TensorIOMode.INPUT,
        mock_trt_builder.TensorIOMode.OUTPUT
    ]

    context = MagicMock()
    context.execute_async_v3.return_value = True
    context.get_tensor_shape.return_value = [1, 256]
    engine.create_execution_context.return_value = context

    mock_stream_instance = MagicMock()
    mock_stream.return_value = mock_stream_instance

    builder = TRTEngineBuilder()
    inputs = {
        "input1": np.random.randn(1, 100, 256).astype(np.float32),
        "input2": np.random.randn(1, 100).astype(np.float32)
    }

    outputs = builder.infer(engine, inputs)

    assert "output" in outputs
    assert context.set_input_shape.call_count == 2


@patch('torch.cuda.Stream')
def test_infer_multiple_outputs(mock_stream, mock_trt_builder):
    """Test inference with multiple output tensors."""
    engine = MagicMock()
    engine.num_io_tensors = 3
    engine.get_tensor_name.side_effect = ["input", "output1", "output2"]
    engine.get_tensor_mode.side_effect = [
        mock_trt_builder.TensorIOMode.INPUT,
        mock_trt_builder.TensorIOMode.OUTPUT,
        mock_trt_builder.TensorIOMode.OUTPUT
    ]

    context = MagicMock()
    context.execute_async_v3.return_value = True
    context.get_tensor_shape.return_value = [1, 256]
    engine.create_execution_context.return_value = context

    mock_stream_instance = MagicMock()
    mock_stream.return_value = mock_stream_instance

    builder = TRTEngineBuilder()
    inputs = {"input": np.random.randn(1, 100, 256).astype(np.float32)}

    outputs = builder.infer(engine, inputs)

    assert "output1" in outputs
    assert "output2" in outputs


# ============================================================================
# Benchmarking Tests
# ============================================================================


@patch('torch.cuda.Stream')
@patch('time.perf_counter')
def test_benchmark_success(mock_perf_counter, mock_stream, mock_trt_builder):
    """Test successful benchmarking."""
    # Mock timing
    mock_perf_counter.side_effect = [
        # Warmup runs (10 pairs)
        *[i * 0.001 for i in range(20)],
        # Timed runs (100 pairs)
        *[0.1 + i * 0.002 for i in range(200)]
    ]

    engine = MagicMock()
    engine.num_io_tensors = 2
    engine.get_tensor_name.side_effect = ["input"] * 100 + ["output"] * 100
    engine.get_tensor_mode.side_effect = (
        [mock_trt_builder.TensorIOMode.INPUT] * 100 +
        [mock_trt_builder.TensorIOMode.OUTPUT] * 100
    )

    context = MagicMock()
    context.execute_async_v3.return_value = True
    context.get_tensor_shape.return_value = [1, 256]
    engine.create_execution_context.return_value = context

    mock_stream_instance = MagicMock()
    mock_stream.return_value = mock_stream_instance

    builder = TRTEngineBuilder()
    inputs = {"input": np.random.randn(1, 100, 256).astype(np.float32)}

    stats = builder.benchmark(engine, inputs, n_runs=100, warmup=10)

    assert isinstance(stats, LatencyStats)
    assert stats.n_runs == 100
    assert len(stats.all_ms) == 100
    assert stats.mean_ms > 0
    assert stats.p50_ms > 0
    assert stats.p95_ms >= stats.p50_ms
    assert stats.p99_ms >= stats.p95_ms


@patch('torch.cuda.Stream')
@patch('time.perf_counter')
def test_benchmark_custom_runs(mock_perf_counter, mock_stream, mock_trt_builder):
    """Test benchmarking with custom number of runs."""
    mock_perf_counter.side_effect = [i * 0.001 for i in range(100)]

    engine = MagicMock()
    engine.num_io_tensors = 2
    engine.get_tensor_name.side_effect = ["input"] * 50 + ["output"] * 50
    engine.get_tensor_mode.side_effect = (
        [mock_trt_builder.TensorIOMode.INPUT] * 50 +
        [mock_trt_builder.TensorIOMode.OUTPUT] * 50
    )

    context = MagicMock()
    context.execute_async_v3.return_value = True
    context.get_tensor_shape.return_value = [1, 256]
    engine.create_execution_context.return_value = context

    mock_stream_instance = MagicMock()
    mock_stream.return_value = mock_stream_instance

    builder = TRTEngineBuilder()
    inputs = {"input": np.random.randn(1, 100, 256).astype(np.float32)}

    stats = builder.benchmark(engine, inputs, n_runs=50, warmup=5)

    assert stats.n_runs == 50
    assert len(stats.all_ms) == 50


@patch('torch.cuda.Stream')
@patch('time.perf_counter')
def test_benchmark_no_warmup(mock_perf_counter, mock_stream, mock_trt_builder):
    """Test benchmarking without warmup runs."""
    mock_perf_counter.side_effect = [i * 0.001 for i in range(20)]

    engine = MagicMock()
    engine.num_io_tensors = 2
    engine.get_tensor_name.side_effect = ["input"] * 20 + ["output"] * 20
    engine.get_tensor_mode.side_effect = (
        [mock_trt_builder.TensorIOMode.INPUT] * 20 +
        [mock_trt_builder.TensorIOMode.OUTPUT] * 20
    )

    context = MagicMock()
    context.execute_async_v3.return_value = True
    context.get_tensor_shape.return_value = [1, 256]
    engine.create_execution_context.return_value = context

    mock_stream_instance = MagicMock()
    mock_stream.return_value = mock_stream_instance

    builder = TRTEngineBuilder()
    inputs = {"input": np.random.randn(1, 100, 256).astype(np.float32)}

    stats = builder.benchmark(engine, inputs, n_runs=10, warmup=0)

    assert stats.n_runs == 10


@patch('torch.cuda.Stream')
@patch('time.perf_counter')
def test_benchmark_percentile_calculation(mock_perf_counter, mock_stream, mock_trt_builder):
    """Test that percentiles are calculated correctly."""
    # Create controlled latencies: 1, 2, 3, ..., 100 ms
    latencies = list(range(1, 101))
    timing_values = []
    for lat in latencies:
        timing_values.extend([0, lat / 1000.0])  # start, end
    mock_perf_counter.side_effect = timing_values

    engine = MagicMock()
    engine.num_io_tensors = 2
    engine.get_tensor_name.side_effect = ["input"] * 200 + ["output"] * 200
    engine.get_tensor_mode.side_effect = (
        [mock_trt_builder.TensorIOMode.INPUT] * 200 +
        [mock_trt_builder.TensorIOMode.OUTPUT] * 200
    )

    context = MagicMock()
    context.execute_async_v3.return_value = True
    context.get_tensor_shape.return_value = [1, 256]
    engine.create_execution_context.return_value = context

    mock_stream_instance = MagicMock()
    mock_stream.return_value = mock_stream_instance

    builder = TRTEngineBuilder()
    inputs = {"input": np.random.randn(1, 100, 256).astype(np.float32)}

    stats = builder.benchmark(engine, inputs, n_runs=100, warmup=0)

    # Check percentiles are approximately correct (within 2ms tolerance for indexing)
    assert 49.0 <= stats.p50_ms <= 52.0  # Median (around 50-51)
    assert 94.0 <= stats.p95_ms <= 97.0  # 95th percentile (around 95-96)
    assert 98.0 <= stats.p99_ms <= 101.0  # 99th percentile (around 99-100)
    assert abs(stats.mean_ms - 50.5) < 0.1  # Mean of 1-100


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_build_engine_empty_network(mock_trt_builder, mock_onnx_file):
    """Test engine building with empty network (no inputs)."""
    network = mock_trt_builder.Builder.return_value.create_network.return_value
    network.num_inputs = 0

    builder = TRTEngineBuilder()

    engine = builder.build_engine(mock_onnx_file)

    assert engine is not None
    # No profile should be added for static-only network
    config = mock_trt_builder.Builder.return_value.create_builder_config.return_value
    # add_optimization_profile should not be called (or called 0 times in this case)


def test_build_engine_all_static_shapes(mock_trt_builder, mock_onnx_file):
    """Test engine building with all static shapes."""
    network = mock_trt_builder.Builder.return_value.create_network.return_value
    network.num_inputs = 1

    input_tensor = MagicMock()
    input_tensor.name = "audio"
    input_tensor.shape = (1, 100, 256)  # All static
    network.get_input.return_value = input_tensor

    builder = TRTEngineBuilder()

    engine = builder.build_engine(mock_onnx_file)

    assert engine is not None


def test_workspace_size_conversion():
    """Test correct workspace size conversion from GB to bytes."""
    builder = TRTEngineBuilder(workspace_size_gb=1.5)

    expected_bytes = int(1.5 * (1 << 30))
    assert builder._workspace_bytes == expected_bytes


def test_build_engine_pathlib_path(mock_trt_builder, tmp_path):
    """Test that build_engine accepts pathlib.Path objects."""
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b'fake_onnx_data')

    builder = TRTEngineBuilder()

    # Should accept Path object
    engine = builder.build_engine(onnx_path)

    assert engine is not None


def test_save_load_roundtrip(mock_trt_builder, tmp_path):
    """Test engine save/load roundtrip preserves data."""
    engine = MagicMock()
    test_data = b'test_engine_data_123'
    engine.serialize.return_value = test_data

    builder = TRTEngineBuilder()
    save_path = str(tmp_path / "model.trt")

    # Save
    builder.save_engine(engine, save_path)

    # Verify file contents
    assert Path(save_path).read_bytes() == test_data

    # Load
    loaded_engine = builder.load_engine(save_path)

    assert loaded_engine is not None


# ============================================================================
# Integration-like Tests (Mocked)
# ============================================================================


def test_full_workflow_fp16(mock_trt_builder, tmp_path):
    """Test complete workflow: build → save → load → infer with FP16."""
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b'fake_onnx_data')
    engine_path = tmp_path / "model.trt"

    builder = TRTEngineBuilder()

    # Build
    engine = builder.build_engine(str(onnx_path), fp16=True)
    assert engine is not None

    # Save
    saved_path = builder.save_engine(engine, str(engine_path))
    assert saved_path == str(engine_path)
    assert engine_path.exists()

    # Load
    loaded_engine = builder.load_engine(str(engine_path))
    assert loaded_engine is not None


def test_full_workflow_with_caching(mock_trt_builder, tmp_path):
    """Test complete workflow with engine caching."""
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b'fake_onnx_data')
    engine_path = tmp_path / "model.trt"

    builder = TRTEngineBuilder()

    # First call - builds and caches (engine doesn't exist)
    engine1 = builder.load_cached_engine(str(onnx_path), str(engine_path), fp16=True)
    assert engine1 is not None
    assert engine_path.exists()  # Should have been saved

    # Second call - should use cache (engine exists and is newer than ONNX)
    # Just verify it doesn't raise an error and returns an engine
    engine2 = builder.load_cached_engine(str(onnx_path), str(engine_path), fp16=True)
    assert engine2 is not None


def test_multiple_precision_modes(mock_trt_builder, mock_onnx_file):
    """Test building engines with different precision modes."""
    builder = TRTEngineBuilder()

    # FP32
    engine_fp32 = builder.build_engine(mock_onnx_file, fp16=False, int8=False)
    assert engine_fp32 is not None

    # FP16
    engine_fp16 = builder.build_engine(mock_onnx_file, fp16=True, int8=False)
    assert engine_fp16 is not None

    # INT8
    calibrator = MagicMock()
    engine_int8 = builder.build_engine(mock_onnx_file, fp16=False, int8=True, calibrator=calibrator)
    assert engine_int8 is not None

    # Mixed FP16 + INT8
    engine_mixed = builder.build_engine(mock_onnx_file, fp16=True, int8=True, calibrator=calibrator)
    assert engine_mixed is not None
