"""TensorRT engine management for inference."""

import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import torch

logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    logger.warning("TensorRT not available. Install tensorrt for accelerated inference.")
    TRT_AVAILABLE = False


class TensorRTEngine:
    """TensorRT engine wrapper for model inference."""

    def __init__(self, engine_path: Union[str, Path]):
        """Initialize TensorRT engine from serialized file."""
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")

        self.engine_path = Path(engine_path)
        self.engine = None
        self.context = None
        self.bindings = []
        self.input_specs = {}
        self.output_specs = {}

        if self.engine_path.exists():
            self.load_engine()
        else:
            logger.info(f"Engine file not found at {engine_path}. Will create on first build.")

    def load_engine(self) -> None:
        """Load serialized TensorRT engine."""
        logger.info(f"Loading TensorRT engine from {self.engine_path}")

        # Initialize TensorRT
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        # Load engine
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()

        self.engine = runtime.deserialize_cuda_engine(engine_data)

        # CONSISTENCY FIX: Check if deserialization failed
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        self.context = self.engine.create_execution_context()

        # Parse input/output specs
        self._parse_engine_specs()

        logger.info("TensorRT engine loaded successfully")

    def _parse_engine_specs(self) -> None:
        """
        Parse engine input/output specifications.

        Note: For engines with dynamic shapes, dimensions may be -1.
        Size computation is guarded to handle dynamic shapes properly.
        """
        self.bindings = []

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = self.engine.get_binding_shape(i)

            # Guard against -1 (dynamic) dimensions when computing size
            # For dynamic engines, size will be determined at runtime
            if -1 in shape:
                size = None  # Dynamic shape - size determined at inference time
                logger.debug(f"Binding '{name}' has dynamic shape: {shape}")
            else:
                size = trt.volume(shape) * np.dtype(dtype).itemsize

            spec = {
                'name': name,
                'dtype': dtype,
                'shape': tuple(shape),
                'size': size  # None for dynamic shapes
            }

            if self.engine.binding_is_input(i):
                self.input_specs[name] = spec
            else:
                self.output_specs[name] = spec

            self.bindings.append(None)  # Will be filled with device pointers

    def build_engine(self, onnx_path: Union[str, Path], max_batch_size: int = 1,
                    fp16: bool = True, workspace_size: int = 1 << 30,
                    dynamic_shapes: Optional[Dict[str, Tuple[Tuple, Tuple, Tuple]]] = None) -> None:
        """Build TensorRT engine from ONNX model."""
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")

        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        logger.info(f"Building TensorRT engine from {onnx_path}")

        # Initialize TensorRT
        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        config = builder.create_builder_config()

        # Set workspace size
        config.max_workspace_size = workspace_size

        # Enable FP16 if requested
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 mode enabled")

        # Parse ONNX
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, trt_logger)

        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX model")

        # Setup dynamic shapes AFTER parsing - use network.get_input(i).name to match inputs
        if dynamic_shapes:
            profile = builder.create_optimization_profile()

            for input_name, (min_shape, opt_shape, max_shape) in dynamic_shapes.items():
                # Find input by name in the parsed network
                input_found = False
                for i in range(network.num_inputs):
                    if network.get_input(i).name == input_name:
                        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                        logger.info(f"Dynamic shape for {input_name}:")
                        logger.info(f"  Min: {min_shape}")
                        logger.info(f"  Opt: {opt_shape}")
                        logger.info(f"  Max: {max_shape}")
                        input_found = True
                        break

                if not input_found:
                    logger.warning(f"Dynamic shape input '{input_name}' not found in ONNX network")

            config.add_optimization_profile(profile)
            logger.info("Dynamic shape optimization profile added")

        # Build engine
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        # Save engine
        self.engine_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.engine_path, 'wb') as f:
            f.write(serialized_engine)

        logger.info(f"TensorRT engine saved to {self.engine_path}")

        # Load the built engine
        self.load_engine()

    def allocate_buffers(self) -> Tuple[List, List]:
        """
        Allocate GPU memory for inputs and outputs.

        DEPRECATED: This method is deprecated for engines with dynamic shapes.
        Use the infer() method instead, which handles dynamic shapes correctly
        by allocating buffers based on actual input shapes at runtime.

        For static engines only.
        """
        logger.warning(
            "allocate_buffers() is deprecated for dynamic engines. "
            "Use infer() method which handles dynamic shapes correctly."
        )

        inputs = []
        outputs = []

        for name, spec in self.input_specs.items():
            # Check if shape is dynamic
            if spec['size'] is None:
                raise RuntimeError(
                    f"Cannot allocate buffers for dynamic shape input '{name}'. "
                    "Use infer() method instead."
                )
            # Allocate device memory
            device_mem = cuda.mem_alloc(spec['size'])
            inputs.append({
                'name': name,
                'host': np.empty(spec['shape'], dtype=spec['dtype']),
                'device': device_mem
            })

        for name, spec in self.output_specs.items():
            # Check if shape is dynamic
            if spec['size'] is None:
                raise RuntimeError(
                    f"Cannot allocate buffers for dynamic shape output '{name}'. "
                    "Use infer() method instead."
                )
            # Allocate device memory
            device_mem = cuda.mem_alloc(spec['size'])
            outputs.append({
                'name': name,
                'host': np.empty(spec['shape'], dtype=spec['dtype']),
                'device': device_mem
            })

        return inputs, outputs

    def infer(self, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run inference with input data with dynamic shape handling."""
        if self.engine is None:
            raise RuntimeError("Engine not loaded. Call load_engine() first.")

        # Step 1: Set binding shapes for dynamic inputs
        for name, data in input_data.items():
            if name in self.input_specs:
                # Find binding index
                binding_idx = self.engine.get_binding_index(name)
                if binding_idx < 0:
                    logger.warning(f"Input '{name}' not found in engine bindings")
                    continue

                # Set the binding shape based on actual input shape
                input_shape = data.shape
                self.context.set_binding_shape(binding_idx, input_shape)
                logger.debug(f"Set binding shape for '{name}': {input_shape}")

        # Step 2: Resolve all binding shapes after setting dynamic inputs
        num_bindings = self.engine.num_bindings
        bindings = [None] * num_bindings
        inputs_data = []
        outputs_data = []

        for i in range(num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))

            # Get resolved shape from context
            shape = tuple(self.context.get_binding_shape(i))
            size = trt.volume(shape) * np.dtype(dtype).itemsize

            # Allocate device memory
            device_mem = cuda.mem_alloc(size)
            bindings[i] = int(device_mem)

            if self.engine.binding_is_input(i):
                # Allocate host memory and copy input data
                # Handle boolean dtype - cast to int8 for TensorRT compatibility
                if input_data[name].dtype == np.bool_ and dtype == np.int8:
                    host_mem = np.ascontiguousarray(input_data[name].astype(np.int8)).ravel()
                else:
                    host_mem = np.ascontiguousarray(input_data[name]).astype(dtype).ravel()

                if host_mem.size * np.dtype(dtype).itemsize != size:
                    # Reshape if needed
                    host_mem = host_mem.reshape(-1)
                cuda.memcpy_htod(device_mem, host_mem)
                inputs_data.append({'name': name, 'device': device_mem, 'size': size})
            else:
                # Allocate host memory for output
                host_mem = np.empty(shape, dtype=dtype)
                outputs_data.append({'name': name, 'host': host_mem, 'device': device_mem, 'size': size})

        # Step 3: Run inference
        stream = cuda.Stream()
        success = self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        stream.synchronize()

        if not success:
            raise RuntimeError("TensorRT inference failed")

        # Step 4: Copy results back to host
        results = {}
        for out in outputs_data:
            cuda.memcpy_dtoh(out['host'], out['device'])
            results[out['name']] = out['host'].copy()

        # Step 5: Free allocated device memory
        for inp in inputs_data:
            cuda.mem_free(inp['device'])
        for out in outputs_data:
            cuda.mem_free(out['device'])

        return results

    def infer_torch(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run inference with PyTorch tensors."""
        # Convert to numpy
        input_data = {}
        for name, tensor in input_tensors.items():
            input_data[name] = tensor.cpu().numpy()

        # Run inference
        results = self.infer(input_data)

        # Convert back to torch tensors
        output_tensors = {}
        for name, array in results.items():
            output_tensors[name] = torch.from_numpy(array)

        return output_tensors
    
    def __del__(self):
        """Cleanup allocated resources."""
        try:
            # Note: Dynamic shape handling in infer() now frees memory per call
            # No persistent input_buffers/output_buffers attributes to clean up
            # Just ensure context and engine are properly released
            if hasattr(self, 'context') and self.context is not None:
                del self.context
            if hasattr(self, 'engine') and self.engine is not None:
                del self.engine
            logger.debug("TensorRT engine resources cleaned up")
        except Exception as e:
            logger.warning(f"Error during TensorRT cleanup: {e}")


class TensorRTEngineBuilder:
    """Builder class for creating latency-optimized TensorRT engines."""
    
    def __init__(self, logger_severity=None):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")
            
        if logger_severity is None:
            logger_severity = trt.Logger.WARNING
            
        self.logger = trt.Logger(logger_severity)
        self.builder = trt.Builder(self.logger)
        self.config = self.builder.create_builder_config()
        
    def build_from_onnx(self, onnx_path: Union[str, Path],
                       engine_path: Union[str, Path],
                       max_batch_size: int = 1,
                       fp16: bool = True,
                       int8: bool = False,
                       workspace_size: int = 2 << 30,  # 2GB for better optimization
                       dynamic_shapes: Optional[Dict[str, Tuple[Tuple, Tuple, Tuple]]] = None,
                       optimize_for_latency: bool = True,
                       calibrator: Optional[Any] = None,
                       calibration_npz: Optional[Union[str, Path]] = None) -> bool:
        """Build latency-optimized TensorRT engine from ONNX model.

        Args:
            onnx_path: Path to ONNX model file
            engine_path: Path to save TensorRT engine
            max_batch_size: Maximum batch size for engine
            fp16: Enable FP16 precision
            int8: Enable INT8 precision (requires calibrator or calibration_npz)
            workspace_size: Maximum workspace size in bytes
            dynamic_shapes: Dictionary mapping input names to (min, opt, max) shape tuples
            optimize_for_latency: Enable latency-specific optimizations
            calibrator: Pre-created INT8 calibrator object (optional)
            calibration_npz: Path to calibration dataset NPZ file (optional, alternative to calibrator)
        """
        
        onnx_path = Path(onnx_path)
        engine_path = Path(engine_path)
        
        if not onnx_path.exists():
            logger.error(f"ONNX file not found: {onnx_path}")
            return False
            
        logger.info(f"Building latency-optimized TensorRT engine from {onnx_path}")
        
        # Configure builder for maximum performance
        self.config.max_workspace_size = workspace_size
        
        # Enable precision modes
        if fp16 and self.builder.platform_has_fast_fp16:
            self.config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 precision enabled for latency optimization")
            
        if int8 and self.builder.platform_has_fast_int8:
            self.config.set_flag(trt.BuilderFlag.INT8)

            # Wire INT8 calibrator if provided
            if calibrator is not None:
                self.config.int8_calibrator = calibrator
                logger.info("INT8 precision enabled with provided calibrator")
            elif calibration_npz is not None:
                # Load calibration data and create calibrator
                try:
                    calibration_npz = Path(calibration_npz)
                    if not calibration_npz.exists():
                        raise FileNotFoundError(f"Calibration NPZ not found: {calibration_npz}")

                    calibration_data = np.load(calibration_npz, allow_pickle=True)
                    component_name = engine_path.stem  # Use engine filename as component identifier

                    # Create calibrator from NPZ data
                    calibrator = self._create_int8_calibrator(
                        calibration_data=calibration_data,
                        component_name=component_name,
                        cache_file=str(engine_path.parent / f"{component_name}_calibration.cache")
                    )
                    self.config.int8_calibrator = calibrator
                    logger.info(f"INT8 precision enabled with calibration data from {calibration_npz}")
                except Exception as e:
                    logger.error(f"Failed to load calibration data: {e}")
                    raise RuntimeError(f"INT8 calibration setup failed: {e}")
            else:
                logger.warning("INT8 enabled but no calibrator or calibration_npz provided - may result in sub-optimal quantization")
                logger.info("INT8 precision enabled without calibration data")
            
        # Latency-specific optimizations
        if optimize_for_latency:
            if TRT_AVAILABLE:
                self.config.set_flag(trt.BuilderFlag.STRICT_TYPES)
                self.config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
                # Optimize for single batch inference
                # Note: DISABLE_TIMING_CACHE may not be available in all TensorRT versions
                try:
                    self.config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
                except AttributeError:
                    logger.debug("DISABLE_TIMING_CACHE not available in this TensorRT version")
                
                # Set optimization level for latency
                if hasattr(self.config, 'builder_optimization_level'):
                    self.config.builder_optimization_level = 5  # Maximum optimization
                    
                logger.info("Latency optimization flags enabled")
        
        # Create network
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = self.builder.create_network(network_flags)
        parser = trt.OnnxParser(network, self.logger)

        # Parse ONNX
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                logger.error("Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                return False

        # Configure dynamic shapes AFTER parsing - use network.get_input(i).name to match inputs
        if dynamic_shapes:
            profile = self.builder.create_optimization_profile()

            for input_name, (min_shape, opt_shape, max_shape) in dynamic_shapes.items():
                # Find input by name in the parsed network
                input_found = False
                for i in range(network.num_inputs):
                    if network.get_input(i).name == input_name:
                        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                        logger.info(f"Dynamic shape for {input_name}: {min_shape} -> {opt_shape} -> {max_shape}")
                        input_found = True
                        break

                if not input_found:
                    logger.warning(f"Dynamic shape input '{input_name}' not found in ONNX network")

            self.config.add_optimization_profile(profile)
            
        # Build engine with error handling
        try:
            logger.info("Building TensorRT engine (this may take several minutes)...")
            serialized_engine = self.builder.build_serialized_network(network, self.config)
            
            if serialized_engine is None:
                logger.error("Failed to build TensorRT engine")
                return False
                
            # Save engine
            engine_path.parent.mkdir(parents=True, exist_ok=True)
            with open(engine_path, 'wb') as f:
                f.write(serialized_engine)
                
            logger.info(f"Latency-optimized TensorRT engine saved to {engine_path}")
            logger.info(f"Engine size: {len(serialized_engine) / (1024*1024):.2f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"Engine build failed: {e}")
            return False

    def _create_int8_calibrator(self, calibration_data: Dict[str, np.ndarray],
                               component_name: str,
                               cache_file: str):
        """Create INT8 calibrator from calibration data.

        Args:
            calibration_data: Dictionary of calibration data arrays
            component_name: Name of the component being calibrated
            cache_file: Path to calibration cache file

        Returns:
            INT8 calibrator instance
        """
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")

        class INT8EntropyCalibrator(trt.IInt8EntropyCalibrator2):
            """INT8 calibrator using entropy minimization."""

            def __init__(self, data_dict, cache_path):
                trt.IInt8EntropyCalibrator2.__init__(self)
                self.data_dict = data_dict
                self.cache_path = cache_path
                self.current_index = 0
                self.device_buffers = {}

                # Convert data_dict to list of batches
                # Expect data_dict format: {"input_name": array}
                self.batches = []
                if data_dict:
                    # Get number of samples from first key
                    num_samples = len(next(iter(data_dict.values())))
                    for i in range(num_samples):
                        batch = {key: val[i:i+1] for key, val in data_dict.items()}
                        self.batches.append(batch)

            def get_batch_size(self):
                """Return batch size for calibration."""
                return 1

            def get_batch(self, names):
                """Get next calibration batch.

                Args:
                    names: List of input names

                Returns:
                    List of device pointers or None if no more batches
                """
                if self.current_index >= len(self.batches):
                    return None

                batch = self.batches[self.current_index]
                self.current_index += 1

                # Allocate device buffers if needed
                for name in names:
                    if name not in self.device_buffers and name in batch:
                        data = batch[name].astype(np.float32)
                        self.device_buffers[name] = cuda.mem_alloc(data.nbytes)

                # Copy batch data to device
                device_ptrs = []
                for name in names:
                    if name in batch:
                        data = batch[name].astype(np.float32)
                        cuda.memcpy_htod(self.device_buffers[name], data)
                        device_ptrs.append(int(self.device_buffers[name]))
                    else:
                        logger.warning(f"Input {name} not found in calibration batch")
                        device_ptrs.append(0)

                return device_ptrs

            def read_calibration_cache(self):
                """Read calibration cache if it exists."""
                if Path(self.cache_path).exists():
                    with open(self.cache_path, 'rb') as f:
                        return f.read()
                return None

            def write_calibration_cache(self, cache):
                """Write calibration cache to disk."""
                with open(self.cache_path, 'wb') as f:
                    f.write(cache)

        logger.info(f"Creating INT8 calibrator for {component_name} with {len(calibration_data)} input(s)")
        return INT8EntropyCalibrator(calibration_data, cache_file)

    def create_calibration_dataset(self, component_datasets: Dict[str, Dict[str, np.ndarray]],
                                  output_path: Union[str, Path]) -> bool:
        """Create calibration dataset NPZ with per-component keys.

        Args:
            component_datasets: Dictionary mapping component names to their input data
                               Format: {"content_encoder": {"input_audio": array}, ...}
            output_path: Path to save calibration NPZ file

        Returns:
            True if successful, False otherwise
        """
        output_path = Path(output_path)

        try:
            # Validate input structure
            if not component_datasets:
                raise ValueError("component_datasets cannot be empty")

            # Prepare NPZ data with per-component keys
            npz_data = {}
            for component_name, inputs_dict in component_datasets.items():
                logger.info(f"Adding calibration data for {component_name}")

                # Validate that inputs are numpy arrays
                for input_name, data in inputs_dict.items():
                    if not isinstance(data, np.ndarray):
                        raise ValueError(f"Data for {component_name}/{input_name} must be numpy array, got {type(data)}")

                    # Store with component-prefixed key
                    key = f"{component_name}/{input_name}"
                    npz_data[key] = data
                    logger.info(f"  {key}: shape {data.shape}, dtype {data.dtype}")

            # Save to NPZ file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(output_path, **npz_data)

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"Calibration dataset saved to {output_path} ({file_size_mb:.2f} MB)")
            logger.info(f"Total components: {len(component_datasets)}, Total arrays: {len(npz_data)}")

            return True

        except Exception as e:
            logger.error(f"Failed to create calibration dataset: {e}")
            return False