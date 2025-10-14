"""TensorRT engine management for inference."""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
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
        self.context = self.engine.create_execution_context()

        # Parse input/output specs
        self._parse_engine_specs()

        logger.info("TensorRT engine loaded successfully")

    def _parse_engine_specs(self) -> None:
        """Parse engine input/output specifications."""
        self.bindings = []

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = self.engine.get_binding_shape(i)

            spec = {
                'name': name,
                'dtype': dtype,
                'shape': tuple(shape),
                'size': trt.volume(shape) * np.dtype(dtype).itemsize
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

        # Setup dynamic shapes if provided
        if dynamic_shapes:
            profile = builder.create_optimization_profile()

            for input_name, (min_shape, opt_shape, max_shape) in dynamic_shapes.items():
                # Find input index
                for i in range(network.num_inputs):
                    if network.get_input(i).name == input_name:
                        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                        logger.info(f"Dynamic shape for {input_name}:")
                        logger.info(f"  Min: {min_shape}")
                        logger.info(f"  Opt: {opt_shape}")
                        logger.info(f"  Max: {max_shape}")
                        break

            config.add_optimization_profile(profile)
            logger.info("Dynamic shape optimization profile added")

        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX model")

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
        """Allocate GPU memory for inputs and outputs."""
        inputs = []
        outputs = []

        for name, spec in self.input_specs.items():
            # Allocate device memory
            device_mem = cuda.mem_alloc(spec['size'])
            inputs.append({
                'name': name,
                'host': np.empty(spec['shape'], dtype=spec['dtype']),
                'device': device_mem
            })

        for name, spec in self.output_specs.items():
            # Allocate device memory
            device_mem = cuda.mem_alloc(spec['size'])
            outputs.append({
                'name': name,
                'host': np.empty(spec['shape'], dtype=spec['dtype']),
                'device': device_mem
            })

        return inputs, outputs

    def infer(self, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run inference with input data."""
        if self.engine is None:
            raise RuntimeError("Engine not loaded. Call load_engine() first.")

        # Allocate buffers
        inputs, outputs = self.allocate_buffers()

        # Copy input data to GPU
        for inp in inputs:
            if inp['name'] in input_data:
                np.copyto(inp['host'], input_data[inp['name']])
                cuda.memcpy_htod(inp['device'], inp['host'])

        # Prepare bindings
        bindings = [inp['device'] for inp in inputs] + [out['device'] for out in outputs]

        # Run inference
        stream = cuda.Stream()
        success = self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        stream.synchronize()

        if not success:
            raise RuntimeError("TensorRT inference failed")

        # Copy results back to host
        results = {}
        for out in outputs:
            cuda.memcpy_dtoh(out['host'], out['device'])
            results[out['name']] = out['host'].copy()

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
            # Free GPU memory
            for buffer in self.input_buffers.values():
                if isinstance(buffer, int):
                    cuda.mem_free(buffer)
            
            for buffer in self.output_buffers.values():
                if isinstance(buffer, int):
                    cuda.mem_free(buffer)
            
            logger.info("TensorRT engine resources cleaned up")
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
                       optimize_for_latency: bool = True) -> bool:
        """Build latency-optimized TensorRT engine from ONNX model."""
        
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
            logger.info("INT8 precision enabled")
            
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
                
        # Configure dynamic shapes if provided
        if dynamic_shapes:
            profile = self.builder.create_optimization_profile()
            
            for input_name, (min_shape, opt_shape, max_shape) in dynamic_shapes.items():
                # Find input in network
                for i in range(network.num_inputs):
                    if network.get_input(i).name == input_name:
                        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                        logger.info(f"Dynamic shape for {input_name}: {min_shape} -> {opt_shape} -> {max_shape}")
                        break
                        
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
    
    def create_calibration_dataset(self, onnx_path: Union[str, Path], 
                                 dataset_path: Union[str, Path],
                                 num_samples: int = 500) -> bool:
        """Create calibration dataset for INT8 optimization."""
        # This would create a calibration dataset for INT8 quantization
        # Implementation depends on specific model and data format
        logger.info(f"Creating calibration dataset with {num_samples} samples")
        # Placeholder implementation
        return True