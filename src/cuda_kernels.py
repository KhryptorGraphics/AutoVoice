"""
CUDA Kernels Wrapper Module for Profiling Scripts

This module provides a compatibility layer for profiling scripts that expect
to import cuda_kernels directly. It re-exports all launch functions from the
main auto_voice.gpu.cuda_kernels module.

Usage:
    import cuda_kernels
    cuda_kernels.launch_pitch_detection(...)
"""

# Import all launch functions from the actual implementation
from auto_voice.gpu.cuda_kernels import (
    launch_optimized_stft,
    launch_optimized_istft,
    launch_pitch_detection,
    launch_mel_spectrogram_singing,
    launch_formant_extraction,
    CUDAKernelError,
    KernelConfig,
    CUDA_KERNELS_AVAILABLE
)

# Re-export for direct access
__all__ = [
    'launch_optimized_stft',
    'launch_optimized_istft',
    'launch_pitch_detection',
    'launch_mel_spectrogram_singing',
    'launch_formant_extraction',
    'CUDAKernelError',
    'KernelConfig',
    'CUDA_KERNELS_AVAILABLE'
]
