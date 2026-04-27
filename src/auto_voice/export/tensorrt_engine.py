"""TensorRT engine builder and inference for AutoVoice models.

Builds optimized TensorRT engines from ONNX models with dynamic shape
support, FP16/INT8 precision, and engine caching. Targets Jetson Thor
(SM 11.0, CUDA 13.0, TensorRT 10.x).
"""

from __future__ import annotations

import gc
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    trt = None
    TRT_AVAILABLE = False

logger = logging.getLogger(__name__)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING) if trt else None


def _require_tensorrt() -> Any:
    """Return the TensorRT module or fail with an actionable optional-dependency error."""
    if trt is None:
        raise RuntimeError(
            "TensorRT is not available. Install the optional 'tensorrt' package "
            "to build, load, or run TensorRT engines."
        )
    return trt


@dataclass
class LatencyStats:
    """Benchmark latency statistics in milliseconds."""

    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    n_runs: int
    all_ms: List[float] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"LatencyStats(mean={self.mean_ms:.2f}ms, p50={self.p50_ms:.2f}ms, "
            f"p95={self.p95_ms:.2f}ms, p99={self.p99_ms:.2f}ms, n={self.n_runs})"
        )


@dataclass
class ShapeProfile:
    """Min/opt/max shapes for a single input tensor."""

    min: Tuple[int, ...]
    opt: Tuple[int, ...]
    max: Tuple[int, ...]


class TRTEngineBuilder:
    """Builds and runs TensorRT engines from ONNX models.

    Supports dynamic shapes via optimization profiles, FP16/INT8 precision,
    engine serialization/deserialization with cache validation, and latency
    benchmarking.

    Example:
        builder = TRTEngineBuilder()
        engine = builder.build_engine("model.onnx", fp16=True)
        builder.save_engine(engine, "model.trt")

        engine = builder.load_engine("model.trt")
        outputs = builder.infer(engine, {"input": np.randn(1, 100, 256).astype(np.float32)})
    """

    def __init__(self, workspace_size_gb: float = 2.0):
        """Initialize the TRT engine builder.

        Args:
            workspace_size_gb: Maximum GPU memory for TRT builder workspace.
        """
        self._workspace_bytes = int(workspace_size_gb * (1 << 30))

    @staticmethod
    def _cleanup_cuda() -> None:
        """Release Python refs and return cached CUDA memory before heavy TRT work."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def build_engine(
        self,
        onnx_path: str,
        fp16: bool = True,
        int8: bool = False,
        calibrator: Optional[Any] = None,
        shape_profiles: Optional[Dict[str, ShapeProfile]] = None,
    ) -> trt.ICudaEngine:
        """Build a TensorRT engine from an ONNX model.

        Args:
            onnx_path: Path to the ONNX model file.
            fp16: Enable FP16 precision (default True for Jetson Thor).
            int8: Enable INT8 precision (requires calibrator).
            calibrator: INT8 calibration data provider.
            shape_profiles: Dynamic shape specifications per input name.
                If None, profiles are inferred from the ONNX model's
                dynamic axes with reasonable defaults.

        Returns:
            Compiled TensorRT engine.

        Raises:
            RuntimeError: If ONNX parsing or engine build fails.
        """
        onnx_path = str(onnx_path)
        if not Path(onnx_path).exists():
            raise RuntimeError(f"ONNX file not found: {onnx_path}")

        trt_module = _require_tensorrt()
        self._cleanup_cuda()
        builder = trt_module.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt_module.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt_module.OnnxParser(network, TRT_LOGGER)

        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                errors = []
                for i in range(parser.num_errors):
                    errors.append(str(parser.get_error(i)))
                raise RuntimeError(
                    f"ONNX parse failed for {onnx_path}:\n" + "\n".join(errors)
                )

        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt_module.MemoryPoolType.WORKSPACE,
            self._workspace_bytes,
        )

        if fp16:
            config.set_flag(trt_module.BuilderFlag.FP16)
        if int8:
            config.set_flag(trt_module.BuilderFlag.INT8)
            if calibrator is not None:
                config.int8_calibrator = calibrator

        # Set up optimization profile for dynamic shapes
        profile = builder.create_optimization_profile()
        has_dynamic = False

        for i in range(network.num_inputs):
            inp = network.get_input(i)
            name = inp.name
            dims = inp.shape  # tuple with -1 for dynamic dims

            if -1 in dims:
                has_dynamic = True
                if shape_profiles and name in shape_profiles:
                    sp = shape_profiles[name]
                    profile.set_shape(name, sp.min, sp.opt, sp.max)
                else:
                    # Infer reasonable defaults for dynamic dims
                    min_shape = tuple(1 if d == -1 else d for d in dims)
                    opt_shape = tuple(16 if d == -1 else d for d in dims)
                    max_shape = tuple(256 if d == -1 else d for d in dims)
                    profile.set_shape(name, min_shape, opt_shape, max_shape)

        if has_dynamic:
            config.add_optimization_profile(profile)

        logger.info(
            f"Building TRT engine from {onnx_path} "
            f"(fp16={fp16}, int8={int8}, workspace={self._workspace_bytes / (1 << 30):.1f}GB)"
        )

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(f"TensorRT engine build failed for {onnx_path}")

        runtime = trt_module.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(serialized)
        if engine is None:
            raise RuntimeError("Failed to deserialize built engine")

        del serialized, runtime, parser, network, config, builder
        self._cleanup_cuda()

        logger.info(
            f"Engine built: {engine.num_io_tensors} tensors, "
            f"{sum(engine.get_tensor_dtype(engine.get_tensor_name(i)).itemsize for i in range(engine.num_io_tensors))} bytes/element total"
        )
        return engine

    def save_engine(self, engine: trt.ICudaEngine, path: str) -> str:
        """Serialize and save a TensorRT engine to disk.

        Args:
            engine: Compiled TensorRT engine.
            path: Output file path for the serialized engine.

        Returns:
            The output path string.
        """
        path = str(path)
        serialized = engine.serialize()
        # IHostMemory supports buffer protocol; convert to bytes for len/write
        data = bytes(serialized)
        with open(path, 'wb') as f:
            f.write(data)
        logger.info(f"Saved TRT engine to {path} ({len(data)} bytes)")
        return path

    def load_engine(self, path: str) -> trt.ICudaEngine:
        """Load a serialized TensorRT engine from disk.

        Args:
            path: Path to the serialized engine file.

        Returns:
            Deserialized TensorRT engine.

        Raises:
            RuntimeError: If the file doesn't exist or deserialization fails.
        """
        path = str(path)
        if not Path(path).exists():
            raise RuntimeError(f"Engine file not found: {path}")

        trt_module = _require_tensorrt()
        self._cleanup_cuda()
        runtime = trt_module.Runtime(TRT_LOGGER)
        with open(path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        if engine is None:
            raise RuntimeError(f"Failed to deserialize engine from {path}")

        del runtime
        self._cleanup_cuda()
        logger.info(f"Loaded TRT engine from {path}")
        return engine

    def load_cached_engine(
        self, onnx_path: str, engine_path: str, **build_kwargs
    ) -> trt.ICudaEngine:
        """Load a cached engine, rebuilding if ONNX is newer.

        Checks modification times: if the cached engine exists and is newer
        than the ONNX source, loads it directly. Otherwise rebuilds.

        Args:
            onnx_path: Path to the source ONNX model.
            engine_path: Path for the cached TRT engine.
            **build_kwargs: Passed to build_engine() if rebuild needed.

        Returns:
            TensorRT engine (cached or freshly built).
        """
        onnx_path = str(onnx_path)
        engine_path = str(engine_path)

        if Path(engine_path).exists():
            onnx_mtime = os.path.getmtime(onnx_path)
            engine_mtime = os.path.getmtime(engine_path)
            if engine_mtime > onnx_mtime:
                logger.info(f"Using cached engine: {engine_path}")
                return self.load_engine(engine_path)
            else:
                logger.info("ONNX newer than cached engine, rebuilding...")

        engine = self.build_engine(onnx_path, **build_kwargs)
        self.save_engine(engine, engine_path)
        return engine

    def infer(
        self,
        engine: trt.ICudaEngine,
        inputs: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Run inference on a TensorRT engine.

        Args:
            engine: Compiled TensorRT engine.
            inputs: Dict mapping input tensor names to numpy arrays.

        Returns:
            Dict mapping output tensor names to numpy arrays.

        Raises:
            RuntimeError: If inference execution fails.
        """
        trt_module = _require_tensorrt()
        context = engine.create_execution_context()
        self._cleanup_cuda()

        # Set input shapes for dynamic dims
        for name, arr in inputs.items():
            context.set_input_shape(name, arr.shape)

        # Allocate GPU memory and set tensor addresses
        device_buffers = {}
        output_names = []

        stream = torch.cuda.Stream()

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)

            if mode == trt_module.TensorIOMode.INPUT:
                # Copy input to GPU
                arr = inputs[name]
                device_tensor = torch.from_numpy(arr).cuda()
                device_buffers[name] = device_tensor
                context.set_tensor_address(name, device_tensor.data_ptr())
            else:
                # Allocate output buffer
                shape = context.get_tensor_shape(name)
                dtype = trt_module.nptype(engine.get_tensor_dtype(name))
                device_tensor = torch.empty(
                    tuple(shape), dtype=torch.from_numpy(np.empty(1, dtype=dtype)).dtype
                ).cuda()
                device_buffers[name] = device_tensor
                context.set_tensor_address(name, device_tensor.data_ptr())
                output_names.append(name)

        # Execute inference
        success = context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

        if not success:
            raise RuntimeError("TensorRT inference execution failed")

        # Copy outputs to CPU
        outputs = {}
        for name in output_names:
            outputs[name] = device_buffers[name].cpu().numpy()

        del context, device_buffers, stream
        self._cleanup_cuda()
        return outputs

    def benchmark(
        self,
        engine: trt.ICudaEngine,
        inputs: Dict[str, np.ndarray],
        n_runs: int = 100,
        warmup: int = 10,
    ) -> LatencyStats:
        """Benchmark TensorRT engine latency.

        Args:
            engine: Compiled TensorRT engine.
            inputs: Dict mapping input tensor names to numpy arrays.
            n_runs: Number of timed inference runs.
            warmup: Number of warmup runs before timing.

        Returns:
            LatencyStats with mean, p50, p95, p99 in milliseconds.
        """
        trt_module = _require_tensorrt()
        context = engine.create_execution_context()
        self._cleanup_cuda()

        # Set input shapes
        for name, arr in inputs.items():
            context.set_input_shape(name, arr.shape)

        # Allocate buffers once
        device_buffers = {}
        stream = torch.cuda.Stream()

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)

            if mode == trt_module.TensorIOMode.INPUT:
                device_tensor = torch.from_numpy(inputs[name]).cuda()
                device_buffers[name] = device_tensor
                context.set_tensor_address(name, device_tensor.data_ptr())
            else:
                shape = context.get_tensor_shape(name)
                dtype = trt_module.nptype(engine.get_tensor_dtype(name))
                device_tensor = torch.empty(
                    tuple(shape), dtype=torch.from_numpy(np.empty(1, dtype=dtype)).dtype
                ).cuda()
                device_buffers[name] = device_tensor
                context.set_tensor_address(name, device_tensor.data_ptr())

        # Warmup
        for _ in range(warmup):
            context.execute_async_v3(stream.cuda_stream)
            stream.synchronize()

        # Timed runs
        latencies = []
        for _ in range(n_runs):
            start = time.perf_counter()
            context.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)

        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)

        stats = LatencyStats(
            mean_ms=sum(latencies) / n,
            p50_ms=latencies_sorted[n // 2],
            p95_ms=latencies_sorted[int(n * 0.95)],
            p99_ms=latencies_sorted[int(n * 0.99)],
            n_runs=n_runs,
            all_ms=latencies,
        )
        del context, device_buffers, stream
        self._cleanup_cuda()
        return stats
