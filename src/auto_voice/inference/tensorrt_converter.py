"""TensorRT converter for ONNX export and optimization of voice conversion components."""

import logging
import os
import math
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import tempfile

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    logger.warning("TensorRT not available. Install tensorrt for acceleration capabilities.")
    TRT_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    logger.warning("ONNX Runtime not available. Install onnxruntime for inference capabilities.")
    ORT_AVAILABLE = False


class TensorRTConverter:
    """
    Converter for exporting voice conversion components to ONNX and optimizing with TensorRT.

    Handles ONNX export for individual components and their TensorRT optimization with
    support for dynamic shapes and different precision modes.

    Args:
        export_dir: Directory to save exported models
        trt_logger: TensorRT logger (optional)
        device: Device for ONNX export ('cpu' or 'cuda')

    Example:
        >>> converter = TensorRTConverter(export_dir="./models")
        >>> converter.export_content_encoder(content_encoder)
        >>> converter.optimize_with_tensorrt("content_encoder.onnx", fp16=True)
    """

    def __init__(
        self,
        export_dir: Union[str, Path],
        trt_logger: Optional[Any] = None,
        device: str = 'cpu'
    ):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # Setup TensorRT logger
        if trt_logger is None and TRT_AVAILABLE:
            self.trt_logger = trt.Logger(trt.Logger.WARNING)
        else:
            self.trt_logger = trt_logger

        logger.info(f"TensorRT Converter initialized with device: {device}, export_dir: {export_dir}")

    def export_content_encoder(
        self,
        content_encoder: nn.Module,
        model_name: str = "content_encoder",
        opset_version: int = 17,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        input_sample: Optional[torch.Tensor] = None
    ) -> Path:
        """
        Export ContentEncoder to ONNX format.

        Note: This export assumes input audio is already at the correct sample rate (16kHz)
        to avoid unsupported torchaudio.transforms.Resample operations in ONNX.

        Args:
            content_encoder: ContentEncoder model to export
            model_name: Base name for exported model files
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes specification for variable-length inputs
            input_sample: Sample input tensor for shape inference

        Returns:
            Path to exported ONNX model
        """
        content_encoder.eval()

        # Default dynamic axes for variable-length audio
        # Fixed: Use correct input/output names and dimensions [B, T] not [B, C, T]
        if dynamic_axes is None:
            dynamic_axes = {
                'input_audio': {0: 'batch_size', 1: 'audio_length'},
                'content_features': {0: 'batch_size', 1: 'time_frames'}
            }

        # Create sample input if not provided
        if input_sample is None:
            # Realistic sample: 1 second at 16kHz, shape [1, T] not [1, C, T]
            input_sample = torch.randn(1, 16000).to(self.device)

        # Get model device
        model_device = next(content_encoder.parameters()).device

        # Move sample input to model device
        input_sample = input_sample.to(model_device)

        # Get sample rate from encoder
        sample_rate = getattr(content_encoder, 'sample_rate', 16000)

        # Create wrapper that fixes sample_rate to avoid runtime resampling
        class ContentEncoderWrapper(nn.Module):
            """
            Wrapper that assumes input audio is already at correct sample rate.
            This avoids unsupported torchaudio.transforms.Resample in ONNX export.
            """
            def __init__(self, encoder, fixed_sample_rate):
                super().__init__()
                self.encoder = encoder
                self.fixed_sample_rate = fixed_sample_rate

            def forward(self, input_audio):
                # Always pass fixed sample_rate to avoid resampling branch
                return self.encoder(input_audio, sample_rate=self.fixed_sample_rate)

        wrapper = ContentEncoderWrapper(content_encoder, sample_rate).to(model_device)

        # Export model
        onnx_path = self.export_dir / f"{model_name}.onnx"

        logger.info(f"Exporting ContentEncoder to ONNX: {onnx_path}")
        logger.info(f"NOTE: Input audio must be at {sample_rate}Hz for this ONNX model")

        try:
            torch.onnx.export(
                wrapper,
                input_sample,  # Single input: audio only
                onnx_path,
                export_params=True,
                verbose=False,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input_audio'],  # Removed 'sample_rate' input
                output_names=['content_features'],
                dynamic_axes=dynamic_axes
            )
            logger.info(f"ContentEncoder exported successfully: {onnx_path}")
            logger.info(f"Preprocessing required: Resample audio to {sample_rate}Hz before inference")
            return onnx_path

        except Exception as e:
            logger.error(f"ContentEncoder ONNX export failed: {e}")
            raise

    def export_pitch_encoder(
        self,
        pitch_encoder: nn.Module,
        model_name: str = "pitch_encoder",
        opset_version: int = 17,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        input_sample: Optional[Dict[str, torch.Tensor]] = None
    ) -> Path:
        """
        Export PitchEncoder to ONNX format.

        Args:
            pitch_encoder: PitchEncoder model to export
            model_name: Base name for exported model files
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes specification
            input_sample: Sample input tensors

        Returns:
            Path to exported ONNX model
        """
        pitch_encoder.eval()

        # Default dynamic axes - include voiced_mask
        if dynamic_axes is None:
            dynamic_axes = {
                'f0_input': {0: 'batch_size', 1: 'time_steps'},
                'voiced_mask': {0: 'batch_size', 1: 'time_steps'},
                'pitch_features': {0: 'batch_size', 1: 'time_steps'}
            }

        # Create sample input if not provided - always pass boolean tensor for voiced
        if input_sample is None:
            # Sample F0 contour: 50 frames at 50Hz
            f0_sample = torch.randn(1, 50).to(self.device)
            # Always pass a boolean tensor for voiced_mask (all ones = all voiced)
            voiced_sample = torch.ones(1, 50, dtype=torch.bool).to(self.device)
            input_sample = {'f0': f0_sample, 'voiced': voiced_sample}

        # Get model device
        model_device = next(pitch_encoder.parameters()).device

        # Move inputs to model device and ensure voiced is a boolean tensor
        if isinstance(input_sample, dict):
            f0_input = input_sample['f0'].to(model_device)
            voiced_input = input_sample.get('voiced')
            if voiced_input is not None:
                voiced_input = voiced_input.to(model_device)
                # Ensure it's a boolean tensor, not None
                if voiced_input.dtype != torch.bool:
                    voiced_input = voiced_input.bool()
            else:
                # If None provided, create default all-ones boolean tensor
                voiced_input = torch.ones_like(f0_input, dtype=torch.bool).to(model_device)
        else:
            f0_input = input_sample.to(model_device)
            # Create default all-ones boolean tensor
            voiced_input = torch.ones_like(f0_input, dtype=torch.bool).to(model_device)

        # Create wrapper for ONNX export
        class PitchEncoderWrapper(nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder

            def forward(self, f0_input, voiced_input):
                return self.encoder(f0_input, voiced_input)

        wrapper = PitchEncoderWrapper(pitch_encoder).to(model_device)

        # Export model
        onnx_path = self.export_dir / f"{model_name}.onnx"

        logger.info(f"Exporting PitchEncoder to ONNX: {onnx_path}")

        try:
            torch.onnx.export(
                wrapper,
                (f0_input, voiced_input),
                onnx_path,
                export_params=True,
                verbose=False,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['f0_input', 'voiced_mask'],
                output_names=['pitch_features'],
                dynamic_axes=dynamic_axes
            )
            logger.info(f"PitchEncoder exported successfully: {onnx_path}")
            return onnx_path

        except Exception as e:
            logger.error(f"PitchEncoder ONNX export failed: {e}")
            raise

    def export_flow_decoder(
        self,
        flow_decoder: nn.Module,
        model_name: str = "flow_decoder",
        opset_version: int = 17,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        input_sample: Optional[Dict[str, torch.Tensor]] = None,
        cond_channels: int = 704  # content(256) + pitch(192) + speaker(256)
    ) -> Path:
        """
        Export FlowDecoder to ONNX format.

        Args:
            flow_decoder: FlowDecoder model to export
            model_name: Base name for exported model files
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes specification
            input_sample: Sample input tensors
            cond_channels: Conditioning channels (matches FlowDecoder parameter name)

        Returns:
            Path to exported ONNX model
        """
        flow_decoder.eval()

        # Default dynamic axes for sequences - NO 'inverse' input
        if dynamic_axes is None:
            dynamic_axes = {
                'latent_input': {0: 'batch_size', 2: 'time_steps'},
                'conditioning': {0: 'batch_size', 2: 'time_steps'},
                'mask': {0: 'batch_size', 2: 'time_steps'},
                'output_latent': {0: 'batch_size', 2: 'time_steps'}
            }

        # Get model device
        model_device = next(flow_decoder.parameters()).device

        # Create sample input if not provided
        if input_sample is None:
            batch_size, latent_dim, time_steps = 1, flow_decoder.in_channels, 50
            latent_input = torch.randn(batch_size, latent_dim, time_steps).to(model_device)
            conditioning = torch.randn(batch_size, cond_channels, time_steps).to(model_device)
            mask = torch.ones(batch_size, 1, time_steps).to(model_device)
            input_sample = {
                'latent': latent_input,
                'mask': mask,
                'conditioning': conditioning
            }

        # Create wrapper that FREEZES inverse=True internally (not an input)
        class FlowDecoderWrapper(nn.Module):
            def __init__(self, decoder):
                super().__init__()
                self.decoder = decoder

            def forward(self, latent_input, mask, conditioning):
                # inverse=True is frozen, NOT an input parameter
                return self.decoder(latent_input, mask, cond=conditioning, inverse=True)

        wrapper = FlowDecoderWrapper(flow_decoder).to(model_device)

        # Prepare inputs (NO inverse parameter)
        latent_input = input_sample['latent']
        mask = input_sample['mask']
        conditioning = input_sample['conditioning']

        # Export model
        onnx_path = self.export_dir / f"{model_name}.onnx"

        logger.info(f"Exporting FlowDecoder to ONNX: {onnx_path}")

        try:
            torch.onnx.export(
                wrapper,
                (latent_input, mask, conditioning),  # NO inverse in inputs
                onnx_path,
                export_params=True,
                verbose=False,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['latent_input', 'mask', 'conditioning'],  # NO 'inverse'
                output_names=['output_latent'],
                dynamic_axes=dynamic_axes
            )
            logger.info(f"FlowDecoder exported successfully: {onnx_path}")
            return onnx_path

        except Exception as e:
            logger.error(f"FlowDecoder ONNX export failed: {e}")
            raise

    def export_mel_projection(
        self,
        mel_projection: nn.Module,
        model_name: str = "mel_projection",
        opset_version: int = 17,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        input_sample: Optional[torch.Tensor] = None,
        latent_dim: int = 192,
        mel_channels: int = 80
    ) -> Path:
        """
        Export mel-spectrogram projection layer to ONNX.

        Args:
            mel_projection: Conv1d layer for latent to mel conversion
            model_name: Base name for exported model files
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes specification
            input_sample: Sample input tensor
            latent_dim: Latent dimension
            mel_channels: Number of mel channels

        Returns:
            Path to exported ONNX model
        """
        # Default dynamic axes for sequences
        if dynamic_axes is None:
            dynamic_axes = {
                'latent_input': {0: 'batch_size', 2: 'time_steps'},
                'mel_output': {0: 'batch_size', 2: 'time_steps'}
            }

        # Get model device
        model_device = next(mel_projection.parameters()).device

        # Create sample input if not provided
        if input_sample is None:
            # Sample latent: [batch_size, latent_dim, time_steps]
            input_sample = torch.randn(1, latent_dim, 50).to(model_device)

        # Export model
        onnx_path = self.export_dir / f"{model_name}.onnx"

        logger.info(f"Exporting MelProjection to ONNX: {onnx_path}")

        try:
            torch.onnx.export(
                mel_projection,
                input_sample,
                onnx_path,
                export_params=True,
                verbose=False,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['latent_input'],
                output_names=['mel_output'],
                dynamic_axes=dynamic_axes
            )
            logger.info(f"MelProjection exported successfully: {onnx_path}")
            return onnx_path

        except Exception as e:
            logger.error(f"MelProjection ONNX export failed: {e}")
            raise

    def export_vocoder(
        self,
        vocoder: nn.Module,
        model_name: str = "vocoder",
        opset_version: int = 17,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        input_sample: Optional[torch.Tensor] = None,
        mel_channels: int = 80
    ) -> Path:
        """
        Export vocoder (e.g., HiFiGAN) to ONNX.

        Args:
            vocoder: Vocoder model (HiFiGAN or similar)
            model_name: Base name for exported model files
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes specification
            input_sample: Sample input tensor
            mel_channels: Number of mel channels

        Returns:
            Path to exported ONNX model
        """
        # Default dynamic axes for mel-spectrogram input
        if dynamic_axes is None:
            dynamic_axes = {
                'mel_input': {0: 'batch_size', 2: 'time_steps'},
                'audio_output': {0: 'batch_size', 2: 'samples'}
            }

        # Get model device
        model_device = next(vocoder.parameters()).device

        # Create sample input if not provided
        if input_sample is None:
            # Sample mel-spectrogram: [batch_size, mel_channels, time_steps]
            input_sample = torch.randn(1, mel_channels, 50).to(model_device)

        # Export model
        onnx_path = self.export_dir / f"{model_name}.onnx"

        logger.info(f"Exporting Vocoder to ONNX: {onnx_path}")

        try:
            torch.onnx.export(
                vocoder,
                input_sample,
                onnx_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['mel_input'],
                output_names=['audio_output'],
                dynamic_axes=dynamic_axes
            )
            logger.info(f"Vocoder exported successfully: {onnx_path}")
            return onnx_path

        except Exception as e:
            logger.error(f"Vocoder ONNX export failed: {e}")
            raise

    def optimize_with_tensorrt(
        self,
        onnx_path: Union[str, Path],
        engine_path: Union[str, Path] = None,
        max_batch_size: int = 1,
        fp16: bool = True,
        int8: bool = False,
        workspace_size: int = 2 << 30,  # 2GB
        dynamic_shapes: Optional[Dict[str, Tuple[Tuple, Tuple, Tuple]]] = None,
        calibrate: bool = False,
        component_name: Optional[str] = None,
        calibration_npz: Optional[str] = None
    ) -> Path:
        """
        Optimize ONNX model with TensorRT.

        Args:
            onnx_path: Path to ONNX model
            engine_path: Output path for TensorRT engine (auto-generated if None)
            max_batch_size: Maximum batch size
            fp16: Enable FP16 precision
            int8: Enable INT8 precision (requires calibration)
            workspace_size: Workspace size in bytes
            dynamic_shapes: Dynamic shape specification for optimization profile
            calibrate: Perform INT8 calibration

        Returns:
            Path to optimized TensorRT engine
        """
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not available for optimization")

        onnx_path = Path(onnx_path)
        if engine_path is None:
            engine_path = onnx_path.with_suffix('.engine')
        engine_path = Path(engine_path)

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        logger.info(f"Optimizing {onnx_path} with TensorRT")

        # Initialize TensorRT
        builder = trt.Builder(self.trt_logger)
        config = builder.create_builder_config()
        config.max_workspace_size = workspace_size

        # Set precision modes
        if trt.__version__ >= '8.0':
            if fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("FP16 precision enabled")
            if int8 and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                # When int8=True, prepare calibrator
                if calibration_npz or calibrate:
                    calibrator = self._prepare_int8_calibrator(
                        component_name=component_name,
                        calibration_npz=calibration_npz
                    )
                    config.int8_calibrator = calibrator
                logger.info("INT8 precision enabled")
        else:
            # Legacy API for older versions
            if fp16 and builder.platform_has_fast_fp16:
                builder.fp16_mode = True
                logger.info("FP16 precision enabled (legacy API)")
            if int8 and builder.platform_has_fast_int8:
                builder.int8_mode = True
                if calibration_npz or calibrate:
                    calibrator = self._prepare_int8_calibrator(
                        component_name=component_name,
                        calibration_npz=calibration_npz
                    )
                    builder.int8_calibrator = calibrator
                logger.info("INT8 precision enabled (legacy API)")

        # Enable other optimizations
        if trt.__version__ >= '8.0':
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        # Parse ONNX
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, self.trt_logger)

        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                logger.error("Failed to parse ONNX model")
                for error_idx in range(parser.num_errors):
                    logger.error(parser.get_error(error_idx))
                raise RuntimeError("ONNX parsing failed")

        # Configure dynamic shapes if provided
        if dynamic_shapes:
            profile = builder.create_optimization_profile()

            for input_name, (min_shape, opt_shape, max_shape) in dynamic_shapes.items():
                # Find input index
                input_idx = -1
                for i in range(network.num_inputs):
                    if network.get_input(i).name == input_name:
                        input_idx = i
                        break

                if input_idx >= 0:
                    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                    logger.info(f"Dynamic shape for {input_name}: {min_shape} -> {opt_shape} -> {max_shape}")
                else:
                    logger.warning(f"Input {input_name} not found in network")

            config.add_optimization_profile(profile)
            logger.info("Dynamic shape optimization profile added")

        # Build engine
        logger.info("Building TensorRT engine...")
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        # Save engine
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)

        # Get engine size
        engine_size_mb = len(serialized_engine) / (1024 * 1024)
        logger.info(".2f")

        return engine_path

    def _prepare_int8_calibrator(self, component_name: Optional[str], calibration_npz: Optional[str]):
        """
        Prepare INT8 calibrator with proper calibration data for the component.

        Args:
            component_name: Name of the component being calibrated
            calibration_npz: Path to calibration dataset NPZ file

        Returns:
            INT8 calibrator instance
        """
        if not calibration_npz:
            # Use default placeholder calibrator if no dataset provided
            return self._create_calibrator(
                calibration_data=[],
                calibration_cache_file=str(self.export_dir / f"{component_name}_calibration.cache")
            )

        # Load calibration data from NPZ
        calibration_data = self._load_calibration_data(component_name, calibration_npz)

        # Create calibrator with loaded data
        return self._create_calibrator(
            calibration_data=calibration_data,
            calibration_cache_file=str(self.export_dir / f"{component_name}_calibration.cache")
        )

    def _load_calibration_data(self, component_name: str, calibration_npz: str) -> List[Dict[str, np.ndarray]]:
        """
        Load calibration data for a specific component from NPZ file.

        Args:
            component_name: Name of the component
            calibration_npz: Path to NPZ file containing calibration data

        Returns:
            List of calibration data batches
        """
        calibration_data = []

        with np.load(calibration_npz) as data:
            # Construct component-specific keys
            if component_name == 'content_encoder':
                input_keys = ['input_audio', 'sample_rate']
                if all(key in data for key in ['content/input_audio', 'content/sample_rate']):
                    # Grouped format
                    audio_data = data['content/input_audio']
                    sample_rate_data = data['content/sample_rate']
                    # Ensure sample_rate is int32
                    sample_rate_data = sample_rate_data.astype(np.int32)

                    for i in range(len(audio_data)):
                        calibration_data.append({
                            'input_audio': audio_data[i].astype(np.float32),
                            'sample_rate': sample_rate_data[i:i+1].astype(np.int32)  # Keep as array, ensure int32
                        })
                else:
                    logger.warning(f"Content encoder calibration data missing in {calibration_npz}")

            elif component_name == 'pitch_encoder':
                input_keys = ['f0_input', 'voiced_mask']
                if all(key in data for key in ['pitch/f0_input', 'pitch/voiced_mask']):
                    # Grouped format
                    f0_data = data['pitch/f0_input']
                    voiced_data = data['pitch/voiced_mask']

                    for i in range(len(f0_data)):
                        calibration_data.append({
                            'f0_input': f0_data[i].astype(np.float32),
                            'voiced_mask': voiced_data[i].astype(np.bool_)  # Ensure bool dtype
                        })
                else:
                    logger.warning(f"Pitch encoder calibration data missing in {calibration_npz}")

            elif component_name == 'flow_decoder':
                input_keys = ['latent_input', 'mask', 'conditioning']
                required_keys = ['flow/latent_input', 'flow/mask', 'flow/conditioning']
                if all(key in data for key in required_keys):
                    # Grouped format
                    latent_data = data['flow/latent_input']
                    mask_data = data['flow/mask']
                    conditioning_data = data['flow/conditioning']

                    for i in range(len(latent_data)):
                        calibration_data.append({
                            'latent_input': latent_data[i].astype(np.float32),
                            'mask': mask_data[i].astype(np.float32),
                            'conditioning': conditioning_data[i].astype(np.float32)
                        })
                else:
                    logger.warning(f"Flow decoder calibration data missing in {calibration_npz}")

            elif component_name == 'mel_projection':
                input_keys = ['latent_input']
                if 'flow/latent_input' in data:
                    # Reuse flow decoder latent input
                    latent_data = data['flow/latent_input']

                    for i in range(len(latent_data)):
                        calibration_data.append({
                            'latent_input': latent_data[i].astype(np.float32)
                        })
                else:
                    logger.warning(f"Mel projection calibration data missing in {calibration_npz}")

        logger.info(f"Loaded {len(calibration_data)} calibration samples for {component_name}")
        return calibration_data

    def create_calibration_dataset(self, dataset, num_samples: int, output_path: str) -> str:
        """
        Create calibration dataset from provided dataset and save to NPZ.

        Args:
            dataset: Dataset or list of samples to create calibration from
            num_samples: Number of samples to include
            output_path: Path where to save the NPZ file

        Returns:
            Path to created NPZ file
        """
        logger.info(f"Creating calibration dataset with {num_samples} samples")

        # Collect calibration data
        calibration_samples = {
            'content/input_audio': [],
            'content/sample_rate': [],
            'pitch/f0_input': [],
            'pitch/voiced_mask': [],
            'flow/latent_input': [],
            'flow/mask': [],
            'flow/conditioning': []
        }

        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break

            # Extract inputs for each component
            # Content encoder: source audio at 16kHz
            if hasattr(sample, 'source_audio'):
                audio = sample.source_audio
                sample_rate = 16000  # Standard rate
                calibration_samples['content/input_audio'].append(audio)
                calibration_samples['content/sample_rate'].append(sample_rate)

            # Pitch encoder: F0 and voiced mask
            if hasattr(sample, 'source_f0'):
                f0 = sample.source_f0
                voiced_mask = (f0 > 0).astype(np.bool_)
                calibration_samples['pitch/f0_input'].append(f0)
                calibration_samples['pitch/voiced_mask'].append(voiced_mask)

            # Flow decoder: random latent, mask, and conditioning
            # Derive consistent lengths from audio
            if hasattr(sample, 'source_audio'):
                audio_length = len(sample.source_audio)
                hop_length = 512  # From config
                T = math.ceil(audio_length / hop_length)

                # Random latent [192, T]
                latent = np.random.randn(192, T).astype(np.float32)
                # All-ones mask [1, T]
                mask = np.ones((1, T), dtype=np.float32)
                # Random conditioning [704, T] (content + pitch + speaker)
                conditioning = np.random.randn(704, T).astype(np.float32)

                calibration_samples['flow/latent_input'].append(latent)
                calibration_samples['flow/mask'].append(mask)
                calibration_samples['flow/conditioning'].append(conditioning)

        # Convert to arrays and ensure consistent shapes and dtypes
        for key, samples in calibration_samples.items():
            if samples:
                # Ensure correct dtype for each component
                if 'content/sample_rate' in key:
                    calibration_samples[key] = np.array(samples, dtype=np.int32)
                elif 'pitch/voiced_mask' in key:
                    calibration_samples[key] = np.array(samples, dtype=np.bool_)
                else:
                    calibration_samples[key] = np.array(samples, dtype=np.float32)
            else:
                # Create placeholder with reasonable shape and correct dtype
                if 'content/input_audio' in key:
                    calibration_samples[key] = np.random.randn(num_samples, 16000).astype(np.float32)
                elif 'content/sample_rate' in key:
                    calibration_samples[key] = np.full(num_samples, 16000, dtype=np.int32)
                elif 'pitch/f0_input' in key:
                    calibration_samples[key] = np.random.randn(num_samples, 50).astype(np.float32)
                elif 'pitch/voiced_mask' in key:
                    calibration_samples[key] = (np.random.randn(num_samples, 50) > 0).astype(np.bool_)
                elif 'flow/latent_input' in key:
                    calibration_samples[key] = np.random.randn(num_samples, 192, 50).astype(np.float32)
                elif 'flow/mask' in key:
                    calibration_samples[key] = np.ones((num_samples, 1, 50), dtype=np.float32)
                elif 'flow/conditioning' in key:
                    calibration_samples[key] = np.random.randn(num_samples, 704, 50).astype(np.float32)

        # Save to NPZ
        np.savez(output_path, **calibration_samples)
        logger.info(f"Calibration dataset saved to {output_path}")

        return output_path

    def _create_calibrator(self, calibration_cache_file: Optional[str] = None,
                          calibration_data: Optional[List[Dict[str, np.ndarray]]] = None):
        """
        Create INT8 calibrator for calibration.

        Args:
            calibration_cache_file: Path to calibration cache file
            calibration_data: Pre-loaded calibration data (optional)

        Returns:
            INT8 calibrator instance
        """
        if calibration_cache_file is None:
            calibration_cache_file = str(self.export_dir / "calibration.cache")

        logger.info(f"Creating INT8 calibrator with cache: {calibration_cache_file}")

        class INT8EntropyCalibrator(trt.IInt8EntropyCalibrator2):
            """INT8 calibrator using entropy minimization."""

            def __init__(self, calibration_data: List[Dict[str, np.ndarray]], cache_file: str):
                trt.IInt8EntropyCalibrator2.__init__(self)
                self.calibration_data = calibration_data if calibration_data is not None else []
                self.cache_file = cache_file
                self.current_index = 0
                self.device_buffers = {}

            def get_batch_size(self):
                """Return batch size for calibration."""
                return 1

            def get_batch(self, names):
                """
                Get next calibration batch.

                Args:
                    names: List of input names

                Returns:
                    List of device pointers or None if no more batches
                """
                if self.current_index >= len(self.calibration_data):
                    # No more calibration data - free buffers
                    for buf in self.device_buffers.values():
                        try:
                            buf.free()
                        except:
                            pass
                    return None

                # Get current batch data
                batch_data = self.calibration_data[self.current_index]
                self.current_index += 1

                # Allocate device buffers on first call based on batch data
                if not self.device_buffers:
                    for name in names:
                        if name in batch_data:
                            data = batch_data[name]
                            # Ensure contiguous array
                            data = np.ascontiguousarray(data)
                            size = data.nbytes
                            try:
                                device_mem = cuda.mem_alloc(size)
                                self.device_buffers[name] = device_mem
                                logger.debug(f"Allocated {size} bytes for input '{name}', dtype={data.dtype}, shape={data.shape}")
                            except Exception as e:
                                logger.error(f"Failed to allocate device buffer for {name}: {e}")
                        else:
                            logger.warning(f"Input '{name}' not found in calibration data, skipping allocation")

                # Copy data to device and build pointer list
                device_ptrs = []
                for name in names:
                    if name in batch_data and name in self.device_buffers:
                        data = batch_data[name]
                        # Ensure proper dtype for TensorRT
                        if name == 'sample_rate':
                            data = data.astype(np.int32)
                        elif name == 'voiced_mask':
                            # Cast boolean inputs to INT8 for TensorRT compatibility
                            # TensorRT expects INT8, not bool dtype
                            data = data.astype(np.int8)
                        else:
                            data = data.astype(np.float32)

                        # Ensure contiguous
                        data = np.ascontiguousarray(data)

                        try:
                            cuda.memcpy_htod(self.device_buffers[name], data)
                            device_ptrs.append(int(self.device_buffers[name]))
                            logger.debug(f"Copied data to device for '{name}', size={data.nbytes} bytes")
                        except Exception as e:
                            logger.error(f"Failed to copy data for {name}: {e}")
                            device_ptrs.append(0)
                    else:
                        if name not in batch_data:
                            logger.warning(f"Calibration data missing for input: {name}")
                        device_ptrs.append(0)

                return device_ptrs

            def read_calibration_cache(self):
                """Read calibration cache from file."""
                if Path(self.cache_file).exists():
                    logger.info(f"Reading calibration cache from {self.cache_file}")
                    with open(self.cache_file, 'rb') as f:
                        return f.read()
                return None

            def write_calibration_cache(self, cache):
                """Write calibration cache to file."""
                logger.info(f"Writing calibration cache to {self.cache_file}")
                with open(self.cache_file, 'wb') as f:
                    f.write(cache)

        # Honor the provided calibration_data parameter
        if calibration_data is None:
            calibration_data = []
            logger.warning("No calibration data provided, using empty calibration dataset")

        logger.info(f"INT8 calibrator created with {len(calibration_data)} calibration samples")
        return INT8EntropyCalibrator(calibration_data, calibration_cache_file)

    def validate_onnx_model(
        self,
        onnx_path: Union[str, Path],
        input_shapes: Dict[str, Tuple],
        output_shapes: Optional[Dict[str, Tuple]] = None,
        rtol: float = 1e-5,
        atol: float = 1e-5
    ) -> bool:
        """
        Validate ONNX model by comparing outputs with PyTorch model.

        Args:
            onnx_path: Path to ONNX model
            input_shapes: Expected input shapes for validation
            output_shapes: Expected output shapes (optional)
            rtol: Relative tolerance for output comparison
            atol: Absolute tolerance for output comparison

        Returns:
            True if validation passes
        """
        if not ORT_AVAILABLE:
            logger.warning("ONNX Runtime not available for validation")
            return False

        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        try:
            # Create ONNX Runtime session
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

            session = ort.InferenceSession(str(onnx_path), session_options)
            logger.info(f"Validated ONNX model: {onnx_path}")

            return True

        except Exception as e:
            logger.error(f"ONNX model validation failed: {e}")
            return False

    def export_voice_conversion_pipeline(
        self,
        singing_voice_converter: nn.Module,
        model_name: str = "voice_conversion_pipeline",
        opset_version: int = 17,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    ) -> Dict[str, Path]:
        """
        Export complete voice conversion pipeline as separate ONNX models.

        This method exports each component separately for better optimization
        and fallback capabilities.

        Args:
            singing_voice_converter: SingingVoiceConverter model
            model_name: Base name for exported model files
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes specification

        Returns:
            Dictionary mapping component names to ONNX file paths
        """
        logger.info(f"Exporting voice conversion pipeline components: {model_name}")

        exported_models = {}
        device = self.device

        try:
            # Export Content Encoder
            content_encoder_onnx = self.export_content_encoder(
                singing_voice_converter.content_encoder,
                f"{model_name}_content_encoder",
                opset_version,
                dynamic_axes
            )
            exported_models['content_encoder'] = content_encoder_onnx

            # Export Pitch Encoder
            pitch_encoder_onnx = self.export_pitch_encoder(
                singing_voice_converter.pitch_encoder,
                f"{model_name}_pitch_encoder",
                opset_version,
                dynamic_axes
            )
            exported_models['pitch_encoder'] = pitch_encoder_onnx

            # Export Flow Decoder
            flow_decoder_onnx = self.export_flow_decoder(
                singing_voice_converter.flow_decoder,
                f"{model_name}_flow_decoder",
                opset_version,
                dynamic_axes,
                cond_channels=singing_voice_converter.cond_dim
            )
            exported_models['flow_decoder'] = flow_decoder_onnx

            # Export Mel Projection
            mel_proj_onnx = self.export_mel_projection(
                singing_voice_converter.latent_to_mel,
                f"{model_name}_mel_projection",
                opset_version,
                dynamic_axes,
                latent_dim=singing_voice_converter.latent_dim,
                mel_channels=singing_voice_converter.mel_channels
            )
            exported_models['mel_projection'] = mel_proj_onnx

            logger.info(f"Successfully exported {len(exported_models)} pipeline components")
            return exported_models

        except Exception as e:
            logger.error(f"Voice conversion pipeline export failed: {e}")
            raise
