"""Model export utilities for converting PyTorch models to ONNX and TensorRT."""

import torch
import onnx
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import numpy as np
import sys

# Add models directory to path
models_dir = Path(__file__).parent.parent / 'models'
sys.path.insert(0, str(models_dir))

from transformer import VoiceTransformer
from hifigan import HiFiGANGenerator

logger = logging.getLogger(__name__)


def export_transformer_to_onnx(
    checkpoint_path: str,
    output_path: str,
    config: Optional[Dict[str, Any]] = None,
    input_shape: Tuple[int, int, int] = (1, 100, 80),
    validate: bool = True,
    verbose: bool = True
) -> str:
    """Export VoiceTransformer to ONNX format.

    Args:
        checkpoint_path: Path to trained PyTorch checkpoint
        output_path: Path to save ONNX model
        config: Model configuration (if None, uses defaults)
        input_shape: Input tensor shape (batch, seq_len, input_dim)
        validate: Whether to validate exported ONNX model
        verbose: Enable verbose logging

    Returns:
        Path to exported ONNX model
    """
    if config is None:
        config = {}

    # Instantiate model
    model = VoiceTransformer(
        input_dim=config.get('input_dim', 80),
        d_model=config.get('d_model', 512),
        n_heads=config.get('n_heads', 8),
        n_layers=config.get('num_layers', 6),
        d_ff=config.get('d_ff', 2048),
        max_seq_len=config.get('max_seq_len', 1024),
        dropout=config.get('dropout', 0.1)
    )

    # Load checkpoint
    if verbose:
        print(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # Export to ONNX
    model.export_to_onnx(
        output_path=output_path,
        input_shape=input_shape,
        verbose=verbose
    )

    # Validate ONNX model
    if validate:
        try:
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            if verbose:
                print(f"✓ ONNX model validation passed")
        except Exception as e:
            logger.error(f"ONNX validation failed: {e}")
            raise

    return output_path


def export_vocoder_to_onnx(
    checkpoint_path: str,
    output_path: str,
    config: Optional[Dict[str, Any]] = None,
    mel_shape: Tuple[int, int, int] = (1, 80, 100),
    validate: bool = True,
    verbose: bool = True
) -> str:
    """Export HiFiGAN vocoder to ONNX format.

    Args:
        checkpoint_path: Path to trained PyTorch checkpoint
        output_path: Path to save ONNX model
        config: Model configuration (if None, uses defaults)
        mel_shape: Input mel-spectrogram shape (batch, mel_channels, time_steps)
        validate: Whether to validate exported ONNX model
        verbose: Enable verbose logging

    Returns:
        Path to exported ONNX model
    """
    if config is None:
        config = {}

    # Get generator config
    gen_config = config.get('generator', {})

    # Instantiate HiFiGAN generator
    model = HiFiGANGenerator(
        mel_channels=gen_config.get('mel_channels', 80),
        upsample_rates=gen_config.get('upsample_rates', [8, 8, 2, 2]),
        upsample_kernel_sizes=gen_config.get('upsample_kernel_sizes', [16, 16, 4, 4]),
        upsample_initial_channel=gen_config.get('upsample_initial_channel', 512),
        resblock_kernel_sizes=gen_config.get('resblock_kernel_sizes', [3, 7, 11]),
        resblock_dilation_sizes=gen_config.get('resblock_dilation_sizes', [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    )

    # Load checkpoint
    if verbose:
        print(f"Loading vocoder checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # Export to ONNX
    model.export_to_onnx(
        output_path=output_path,
        mel_shape=mel_shape,
        verbose=verbose
    )

    # Validate ONNX model
    if validate:
        try:
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            if verbose:
                print(f"✓ ONNX vocoder model validation passed")
        except Exception as e:
            logger.error(f"ONNX validation failed: {e}")
            raise

    return output_path


def build_tensorrt_engines(
    config: Dict[str, Any],
    model_checkpoints: Dict[str, str],
    output_dir: str,
    fp16: bool = True,
    dynamic_shapes: bool = True,
    verbose: bool = True
) -> Dict[str, str]:
    """Build TensorRT engines for all models in the pipeline.

    Args:
        config: Configuration dictionary with model parameters
        model_checkpoints: Dictionary mapping model names to checkpoint paths
        output_dir: Directory to save TensorRT engines
        fp16: Whether to use FP16 precision
        dynamic_shapes: Whether to enable dynamic shape support
        verbose: Enable verbose logging

    Returns:
        Dictionary mapping model names to TensorRT engine paths
    """
    from tensorrt_engine import TensorRTEngine

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine_paths = {}
    onnx_dir = output_dir / 'onnx'
    onnx_dir.mkdir(exist_ok=True)

    # Export and build transformer (encoder/decoder)
    if 'transformer' in model_checkpoints:
        transformer_onnx = str(onnx_dir / 'transformer.onnx')
        transformer_engine = str(output_dir / 'transformer.trt')

        if verbose:
            print("\n=== Exporting Transformer ===")

        # Export to ONNX
        export_transformer_to_onnx(
            checkpoint_path=model_checkpoints['transformer'],
            output_path=transformer_onnx,
            config=config.get('transformer', {}),
            verbose=verbose
        )

        # Build TensorRT engine
        if verbose:
            print("\n=== Building Transformer TensorRT Engine ===")

        trt_engine = TensorRTEngine(transformer_engine)

        # Dynamic shapes for variable-length sequences
        dynamic_shape_config = None
        if dynamic_shapes:
            dynamic_shape_config = {
                'input': (
                    (1, 10, 80),    # min: batch=1, seq_len=10
                    (1, 100, 80),   # opt: batch=1, seq_len=100
                    (1, 512, 80)    # max: batch=1, seq_len=512
                )
            }

        trt_engine.build_engine(
            onnx_path=transformer_onnx,
            fp16=fp16,
            dynamic_shapes=dynamic_shape_config,
            workspace_size=1 << 30  # 1GB
        )

        engine_paths['transformer'] = transformer_engine

    # Export and build HiFiGAN vocoder
    if 'vocoder' in model_checkpoints:
        vocoder_onnx = str(onnx_dir / 'vocoder.onnx')
        vocoder_engine = str(output_dir / 'vocoder.trt')

        if verbose:
            print("\n=== Exporting Vocoder ===")

        # Export to ONNX
        export_vocoder_to_onnx(
            checkpoint_path=model_checkpoints['vocoder'],
            output_path=vocoder_onnx,
            config=config.get('hifigan', {}),
            verbose=verbose
        )

        # Build TensorRT engine
        if verbose:
            print("\n=== Building Vocoder TensorRT Engine ===")

        trt_engine = TensorRTEngine(vocoder_engine)

        # Dynamic shapes for variable-length mel-spectrograms
        dynamic_shape_config = None
        if dynamic_shapes:
            dynamic_shape_config = {
                'mel_spectrogram': (
                    (1, 80, 10),    # min
                    (1, 80, 100),   # opt
                    (1, 80, 512)    # max
                )
            }

        trt_engine.build_engine(
            onnx_path=vocoder_onnx,
            fp16=fp16,
            dynamic_shapes=dynamic_shape_config,
            workspace_size=1 << 30  # 1GB
        )

        engine_paths['vocoder'] = vocoder_engine

    if verbose:
        print("\n=== TensorRT Engine Build Complete ===")
        for model_name, engine_path in engine_paths.items():
            print(f"  {model_name}: {engine_path}")

    return engine_paths


def validate_exported_model(
    pytorch_checkpoint: str,
    onnx_path: str,
    model_type: str = 'transformer',
    config: Optional[Dict[str, Any]] = None,
    tolerance: float = 1e-3,
    verbose: bool = True
) -> Dict[str, Any]:
    """Validate exported ONNX model against PyTorch original.

    Args:
        pytorch_checkpoint: Path to PyTorch checkpoint
        onnx_path: Path to ONNX model
        model_type: Type of model ('transformer' or 'vocoder')
        config: Model configuration
        tolerance: Maximum allowed difference between outputs
        verbose: Enable verbose logging

    Returns:
        Validation results dictionary
    """
    import onnxruntime as ort

    if config is None:
        config = {}

    # Load PyTorch model
    if model_type == 'transformer':
        pytorch_model = VoiceTransformer(
            input_dim=config.get('input_dim', 80),
            d_model=config.get('d_model', 512),
            n_heads=config.get('n_heads', 8),
            n_layers=config.get('num_layers', 6),
            d_ff=config.get('d_ff', 2048),
            max_seq_len=config.get('max_seq_len', 1024),
            dropout=config.get('dropout', 0.1)
        )
        dummy_input = torch.randn(1, 100, 80)
    else:  # vocoder
        gen_config = config.get('generator', {})
        pytorch_model = HiFiGANGenerator(
            mel_channels=gen_config.get('mel_channels', 80),
            upsample_rates=gen_config.get('upsample_rates', [8, 8, 2, 2]),
            upsample_kernel_sizes=gen_config.get('upsample_kernel_sizes', [16, 16, 4, 4]),
            upsample_initial_channel=gen_config.get('upsample_initial_channel', 512),
            resblock_kernel_sizes=gen_config.get('resblock_kernel_sizes', [3, 7, 11]),
            resblock_dilation_sizes=gen_config.get('resblock_dilation_sizes', [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
        )
        dummy_input = torch.randn(1, 80, 100)

    # Load checkpoint
    checkpoint = torch.load(pytorch_checkpoint, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        pytorch_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        pytorch_model.load_state_dict(checkpoint)

    pytorch_model.eval()

    # PyTorch inference
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input).numpy()

    # ONNX inference
    ort_session = ort.InferenceSession(onnx_path)
    onnx_output = ort_session.run(None, {'input' if model_type == 'transformer' else 'mel_spectrogram': dummy_input.numpy()})[0]

    # Compare outputs
    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    mean_diff = np.mean(np.abs(pytorch_output - onnx_output))

    results = {
        'max_difference': float(max_diff),
        'mean_difference': float(mean_diff),
        'tolerance': tolerance,
        'passed': max_diff < tolerance,
        'pytorch_output_shape': pytorch_output.shape,
        'onnx_output_shape': onnx_output.shape
    }

    if verbose:
        print(f"\n=== Validation Results for {model_type} ===")
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        print(f"  Tolerance: {tolerance}")
        print(f"  Status: {'✓ PASSED' if results['passed'] else '✗ FAILED'}")

    return results
