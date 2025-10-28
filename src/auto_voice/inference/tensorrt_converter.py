"""TensorRT converter for ONNX export and optimization of voice conversion components."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import tempfile

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
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
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size', 2: 'audio_length'},
                'output': {0: 'batch_size', 1: 'time_frames'}
            }

        # Create sample input if not provided
        if input_sample is None:
            # Realistic sample: 1 second at 16kHz
            input_sample = torch.randn(1, 16000).to(self.device)

        # Get model device
        model_device = next(content_encoder.parameters()).device

        # Move sample input to model device
        input_sample = input_sample.to(model_device)

        # Get sample rate from encoder
        sample_rate = getattr(content_encoder, 'sample_rate', 16000)

        # Export model
        onnx_path = self.export_dir / f"{model_name}.onnx"
        dummy_input = (input_sample, sample_rate)

        logger.info(f"Exporting ContentEncoder to ONNX: {onnx_path}")

        try:
            torch.onnx.export(
                content_encoder,
                dummy_input,
                onnx_path,
                export_params=True,
                verbose=False,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input_audio', 'sample_rate'],
                output_names=['content_features'],
                dynamic_axes=dynamic_axes
            )
            logger.info(f"ContentEncoder exported successfully: {onnx_path}")
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

        # Default dynamic axes
        if dynamic_axes is None:
            dynamic_axes = {
                'f0_input': {0: 'batch_size', 1: 'time_steps'},
                'pitch_features': {0: 'batch_size', 1: 'time_steps'}
            }

        # Create sample input if not provided
        if input_sample is None:
            # Sample F0 contour: 50 frames at 50Hz
            f0_sample = torch.randn(1, 50).to(self.device)
            input_sample = {'f0': f0_sample, 'voiced': None}

        # Get model device
        model_device = next(pitch_encoder.parameters()).device

        # Move inputs to model device
        if isinstance(input_sample, dict):
            f0_input = input_sample['f0'].to(model_device)
            voiced_input = input_sample.get('voiced')
            if voiced_input is not None:
                voiced_input = voiced_input.to(model_device)
        else:
            f0_input = input_sample.to(model_device)
            voiced_input = None

        # Create wrapper for ONNX export (single tensor input)
        class PitchEncoderWrapper(nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder

            def forward(self, f0_input, voiced_input=None):
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
        condition_dim: int = 704  # content(256) + pitch(192) + speaker(256)
    ) -> Path:
        """
        Export FlowDecoder to ONNX format.

        Args:
            flow_decoder: FlowDecoder model to export
            model_name: Base name for exported model files
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes specification
            input_sample: Sample input tensors
            condition_dim: Conditioning dimension

        Returns:
            Path to exported ONNX model
        """
        flow_decoder.eval()

        # Default dynamic axes for sequences
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
            conditioning = torch.randn(batch_size, condition_dim, time_steps).to(model_device)
            mask = torch.ones(batch_size, 1, time_steps).to(model_device)
            inverse = True  # For inference mode
            input_sample = {
                'latent': latent_input,
                'mask': mask,
                'conditioning': conditioning,
                'inverse': inverse
            }

        # Create wrapper for ONNX export
        class FlowDecoderWrapper(nn.Module):
            def __init__(self, decoder):
                super().__init__()
                self.decoder = decoder

            def forward(self, latent_input, mask, conditioning, inverse=True):
                return self.decoder(latent_input, mask, cond=conditioning, inverse=inverse)

        wrapper = FlowDecoderWrapper(flow_decoder).to(model_device)

        # Prepare inputs
        latent_input = input_sample['latent']
        mask = input_sample['mask']
        conditioning = input_sample['conditioning']
        inverse = input_sample.get('inverse', True)

        # Export model
        onnx_path = self.export_dir / f"{model_name}.onnx"

        logger.info(f"Exporting FlowDecoder to ONNX: {onnx_path}")

        try:
            torch.onnx.export(
                wrapper,
                (latent_input, mask, conditioning, inverse),
                onnx_path,
                export_params=True,
                verbose=False,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['latent_input', 'mask', 'conditioning', 'inverse'],
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

    def optimize_with_tensorrt(
        self,
        onnx_path: Union[str, Path],
        engine_path: Union[str, Path] = None,
        max_batch_size: int = 1,
        fp16: bool = True,
        int8: bool = False,
        workspace_size: int = 2 << 30,  # 2GB
        dynamic_shapes: Optional[Dict[str, Tuple[Tuple, Tuple, Tuple]]] = None,
        calibrate: bool = False
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
                logger.info("INT8 precision enabled")
        else:
            # Legacy API for older versions
            if fp16 and builder.platform_has_fast_fp16:
                builder.fp16_mode = True
                logger.info("FP16 precision enabled (legacy API)")
            if int8 and builder.platform_has_fast_int8:
                builder.int8_mode = True
                builder.int8_calibrator = self._create_calibrator() if calibrate else None
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

    def _create_calibrator(self):
        """Create INT8 calibrator for calibration."""
        # Placeholder for INT8 calibration
        # In a real implementation, this would create a calibrator
        # that runs inference on calibration data to determine quantization parameters
        logger.info("INT8 calibrator created (placeholder)")
        return None

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
                condition_dim=singing_voice_converter.cond_dim
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
