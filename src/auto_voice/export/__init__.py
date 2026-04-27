"""ONNX and TensorRT export utilities for AutoVoice inference models."""

from .onnx_export import (
    export_content_encoder,
    export_sovits,
    export_bigvgan,
)

# TensorRT is optional - only import if available
try:
    from .tensorrt_engine import (
        LatencyStats,
        ShapeProfile,
        TRT_AVAILABLE as _TENSORRT_AVAILABLE,
        TRTEngineBuilder,
    )
except ImportError:
    # TensorRT not available - define placeholder classes
    LatencyStats = None
    ShapeProfile = None
    TRTEngineBuilder = None
    _TENSORRT_AVAILABLE = False

__all__ = [
    'export_content_encoder',
    'export_sovits',
    'export_bigvgan',
    'LatencyStats',
    'ShapeProfile',
    'TRTEngineBuilder',
]
