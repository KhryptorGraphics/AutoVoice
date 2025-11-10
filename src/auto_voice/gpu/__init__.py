"""GPU management module for AutoVoice"""

from .gpu_manager import GPUManager, GPUConfig, OptimizationLevel, ModelPrecision
from .cuda_manager import CUDAManager, DeviceState, DeviceInfo
from .memory_manager import MemoryManager, AllocationStrategy
from .performance_monitor import PerformanceMonitor, AlertLevel, MetricType
from .cuda_kernels import (
    PitchDetectionKernel,
    SpectrogramKernel,
    VoiceSynthesisKernel,
    FeatureExtractionKernel,
    KernelConfig,
    CUDAKernelError,
    create_kernel_suite,
    CUDA_KERNELS_AVAILABLE
)

__all__ = [
    'GPUManager',
    'GPUConfig',
    'OptimizationLevel',
    'ModelPrecision',
    'CUDAManager',
    'DeviceState',
    'DeviceInfo',
    'MemoryManager',
    'AllocationStrategy',
    'PerformanceMonitor',
    'AlertLevel',
    'MetricType',
    'PitchDetectionKernel',
    'SpectrogramKernel',
    'VoiceSynthesisKernel',
    'FeatureExtractionKernel',
    'KernelConfig',
    'CUDAKernelError',
    'create_kernel_suite',
    'CUDA_KERNELS_AVAILABLE'
]
