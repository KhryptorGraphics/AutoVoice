"""Testing utility functions and helpers.

Provides reusable testing utilities for assertions, comparisons,
and common test operations.
"""

import numpy as np
import torch
from typing import Any, Optional, Union
import pytest


# ============================================================================
# Audio Comparison Utilities
# ============================================================================

def assert_audio_equal(
    audio1: np.ndarray,
    audio2: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    msg: Optional[str] = None
):
    """Assert two audio arrays are equal within tolerance.

    Args:
        audio1: First audio array
        audio2: Second audio array
        rtol: Relative tolerance
        atol: Absolute tolerance
        msg: Custom error message
    """
    assert audio1.shape == audio2.shape, \
        f"Shape mismatch: {audio1.shape} != {audio2.shape}"

    if not np.allclose(audio1, audio2, rtol=rtol, atol=atol):
        max_diff = np.max(np.abs(audio1 - audio2))
        error_msg = f"Audio arrays not equal. Max diff: {max_diff}"
        if msg:
            error_msg = f"{msg}: {error_msg}"
        pytest.fail(error_msg)


def assert_audio_shape(
    audio: np.ndarray,
    expected_shape: tuple,
    msg: Optional[str] = None
):
    """Assert audio has expected shape.

    Args:
        audio: Audio array
        expected_shape: Expected shape
        msg: Custom error message
    """
    if audio.shape != expected_shape:
        error_msg = f"Shape mismatch: {audio.shape} != {expected_shape}"
        if msg:
            error_msg = f"{msg}: {error_msg}"
        pytest.fail(error_msg)


def assert_audio_normalized(
    audio: np.ndarray,
    max_value: float = 1.0,
    msg: Optional[str] = None
):
    """Assert audio is properly normalized.

    Args:
        audio: Audio array
        max_value: Maximum allowed absolute value
        msg: Custom error message
    """
    actual_max = np.max(np.abs(audio))

    if actual_max > max_value:
        error_msg = f"Audio not normalized. Max value: {actual_max} > {max_value}"
        if msg:
            error_msg = f"{msg}: {error_msg}"
        pytest.fail(error_msg)


def compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio.

    Args:
        signal: Clean signal
        noise: Noise component

    Returns:
        SNR in dB
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float('inf')

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def compute_similarity(audio1: np.ndarray, audio2: np.ndarray) -> float:
    """Compute audio similarity (normalized correlation).

    Args:
        audio1: First audio
        audio2: Second audio

    Returns:
        Similarity score (0-1)
    """
    # Ensure same length
    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]

    # Normalize
    audio1 = audio1 - np.mean(audio1)
    audio2 = audio2 - np.mean(audio2)

    # Correlation
    correlation = np.dot(audio1, audio2)
    norm = np.linalg.norm(audio1) * np.linalg.norm(audio2)

    if norm == 0:
        return 0.0

    return correlation / norm


# ============================================================================
# Model Testing Utilities
# ============================================================================

def assert_model_outputs_valid(
    output: Union[torch.Tensor, np.ndarray],
    check_nan: bool = True,
    check_inf: bool = True,
    check_range: Optional[tuple] = None,
    msg: Optional[str] = None
):
    """Assert model output is valid.

    Args:
        output: Model output
        check_nan: Check for NaN values
        check_inf: Check for Inf values
        check_range: Expected value range (min, max)
        msg: Custom error message
    """
    if isinstance(output, torch.Tensor):
        output_np = output.detach().cpu().numpy()
    else:
        output_np = output

    errors = []

    if check_nan and np.isnan(output_np).any():
        errors.append("Output contains NaN values")

    if check_inf and np.isinf(output_np).any():
        errors.append("Output contains Inf values")

    if check_range is not None:
        min_val, max_val = check_range
        actual_min = np.min(output_np)
        actual_max = np.max(output_np)

        if actual_min < min_val or actual_max > max_val:
            errors.append(
                f"Output range [{actual_min}, {actual_max}] outside "
                f"expected [{min_val}, {max_val}]"
            )

    if errors:
        error_msg = "; ".join(errors)
        if msg:
            error_msg = f"{msg}: {error_msg}"
        pytest.fail(error_msg)


def count_parameters(model: torch.nn.Module) -> dict:
    """Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dict with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }


def assert_gradients_exist(
    model: torch.nn.Module,
    msg: Optional[str] = None
):
    """Assert all trainable parameters have gradients.

    Args:
        model: PyTorch model
        msg: Custom error message
    """
    params_without_grad = []

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            params_without_grad.append(name)

    if params_without_grad:
        error_msg = f"Parameters without gradients: {params_without_grad}"
        if msg:
            error_msg = f"{msg}: {error_msg}"
        pytest.fail(error_msg)


# ============================================================================
# GPU Testing Utilities
# ============================================================================

def assert_gpu_memory_efficient(
    max_memory_mb: float = 1000.0,
    device: str = 'cuda',
    msg: Optional[str] = None
):
    """Assert GPU memory usage is within limits.

    Args:
        max_memory_mb: Maximum allowed memory in MB
        device: CUDA device
        msg: Custom error message
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    allocated = torch.cuda.memory_allocated(device) / 1024 / 1024

    if allocated > max_memory_mb:
        error_msg = f"GPU memory {allocated:.2f}MB exceeds limit {max_memory_mb}MB"
        if msg:
            error_msg = f"{msg}: {error_msg}"
        pytest.fail(error_msg)


def get_gpu_utilization(device: str = 'cuda') -> dict:
    """Get GPU utilization statistics.

    Args:
        device: CUDA device

    Returns:
        Dict with GPU stats
    """
    if not torch.cuda.is_available():
        return {'available': False}

    return {
        'available': True,
        'allocated_mb': torch.cuda.memory_allocated(device) / 1024 / 1024,
        'reserved_mb': torch.cuda.memory_reserved(device) / 1024 / 1024,
        'max_allocated_mb': torch.cuda.max_memory_allocated(device) / 1024 / 1024,
    }


# ============================================================================
# Performance Testing Utilities
# ============================================================================

def assert_performance_threshold(
    execution_time: float,
    max_time: float,
    operation: str = "operation",
    msg: Optional[str] = None
):
    """Assert execution time meets performance threshold.

    Args:
        execution_time: Actual execution time
        max_time: Maximum allowed time
        operation: Operation name
        msg: Custom error message
    """
    if execution_time > max_time:
        error_msg = f"{operation} took {execution_time:.4f}s, exceeds {max_time:.4f}s"
        if msg:
            error_msg = f"{msg}: {error_msg}"
        pytest.fail(error_msg)


def assert_realtime_factor(
    processing_time: float,
    audio_duration: float,
    max_rtf: float = 1.0,
    msg: Optional[str] = None
):
    """Assert Real-Time Factor is within threshold.

    Args:
        processing_time: Processing time in seconds
        audio_duration: Audio duration in seconds
        max_rtf: Maximum allowed RTF (1.0 = real-time)
        msg: Custom error message
    """
    rtf = processing_time / audio_duration

    if rtf > max_rtf:
        error_msg = f"RTF {rtf:.3f} exceeds threshold {max_rtf}"
        if msg:
            error_msg = f"{msg}: {error_msg}"
        pytest.fail(error_msg)


# ============================================================================
# Data Validation Utilities
# ============================================================================

def validate_audio_file(filepath: str) -> dict:
    """Validate audio file and return metadata.

    Args:
        filepath: Path to audio file

    Returns:
        Dict with audio metadata

    Raises:
        AssertionError: If file is invalid
    """
    import soundfile as sf

    try:
        info = sf.info(filepath)

        return {
            'valid': True,
            'sample_rate': info.samplerate,
            'channels': info.channels,
            'duration': info.duration,
            'format': info.format,
            'subtype': info.subtype,
        }

    except Exception as e:
        pytest.fail(f"Invalid audio file: {e}")


def assert_tensor_device(
    tensor: torch.Tensor,
    expected_device: str,
    msg: Optional[str] = None
):
    """Assert tensor is on expected device.

    Args:
        tensor: PyTorch tensor
        expected_device: Expected device ('cpu', 'cuda', etc.)
        msg: Custom error message
    """
    actual_device = str(tensor.device)

    if not actual_device.startswith(expected_device):
        error_msg = f"Tensor on {actual_device}, expected {expected_device}"
        if msg:
            error_msg = f"{msg}: {error_msg}"
        pytest.fail(error_msg)


__all__ = [
    # Audio assertions
    'assert_audio_equal',
    'assert_audio_shape',
    'assert_audio_normalized',
    'compute_snr',
    'compute_similarity',

    # Model assertions
    'assert_model_outputs_valid',
    'count_parameters',
    'assert_gradients_exist',

    # GPU assertions
    'assert_gpu_memory_efficient',
    'get_gpu_utilization',

    # Performance assertions
    'assert_performance_threshold',
    'assert_realtime_factor',

    # Validation
    'validate_audio_file',
    'assert_tensor_device',
]
