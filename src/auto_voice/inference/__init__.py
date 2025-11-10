"""Inference module for AutoVoice - Optimized for <100ms latency"""

# Lazy import implementation to avoid loading heavy modules at import time
# and to handle optional dependencies (torch, TensorRT) gracefully

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import VoiceInferenceEngine
    from .tensorrt_engine import TensorRTEngine, TensorRTEngineBuilder
    from .tensorrt_converter import TensorRTConverter
    from .synthesizer import VoiceSynthesizer
    from .realtime_processor import RealtimeProcessor, AsyncRealtimeProcessor
    from .cuda_graphs import CUDAGraphManager, GraphOptimizedModel
    from .inference_manager import InferenceManager
    from .voice_cloner import VoiceCloner
    from .singing_conversion_pipeline import SingingConversionPipeline
    from .voice_conversion_pipeline import VoiceConversionPipeline, PipelineConfig, VoiceConversionError

__all__ = [
    'VoiceInferenceEngine',
    'TensorRTEngine',
    'TensorRTEngineBuilder',
    'TensorRTConverter',
    'VoiceSynthesizer',
    'RealtimeProcessor',
    'AsyncRealtimeProcessor',
    'CUDAGraphManager',
    'GraphOptimizedModel',
    'InferenceManager',
    'VoiceCloner',
    'SingingConversionPipeline',
    'VoiceConversionPipeline',
    'PipelineConfig',
    'VoiceConversionError'
]

# Module-level cache for lazy-loaded classes
_module_cache = {}

class LazyImportError(ImportError):
    """Exception raised when a lazy import fails due to missing dependencies."""
    pass


def __getattr__(name):
    """Lazy import mechanism for inference engine classes.

    This delays importing torch, TensorRT, and other heavy dependencies until
    they're actually accessed, improving import speed and handling optional
    dependencies gracefully.
    """
    if name not in __all__:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    # Check cache first
    if name in _module_cache:
        return _module_cache[name]

    # Lazy import with proper error handling
    try:
        if name == 'VoiceInferenceEngine':
            from .engine import VoiceInferenceEngine
            _module_cache[name] = VoiceInferenceEngine
            return VoiceInferenceEngine
            
        elif name == 'TensorRTEngine':
            from .tensorrt_engine import TensorRTEngine
            _module_cache[name] = TensorRTEngine
            return TensorRTEngine
            
        elif name == 'TensorRTEngineBuilder':
            from .tensorrt_engine import TensorRTEngineBuilder
            _module_cache[name] = TensorRTEngineBuilder
            return TensorRTEngineBuilder

        elif name == 'TensorRTConverter':
            from .tensorrt_converter import TensorRTConverter
            _module_cache[name] = TensorRTConverter
            return TensorRTConverter

        elif name == 'VoiceSynthesizer':
            from .synthesizer import VoiceSynthesizer
            _module_cache[name] = VoiceSynthesizer
            return VoiceSynthesizer
            
        elif name == 'RealtimeProcessor':
            from .realtime_processor import RealtimeProcessor
            _module_cache[name] = RealtimeProcessor
            return RealtimeProcessor
            
        elif name == 'AsyncRealtimeProcessor':
            from .realtime_processor import AsyncRealtimeProcessor
            _module_cache[name] = AsyncRealtimeProcessor
            return AsyncRealtimeProcessor
            
        elif name == 'CUDAGraphManager':
            from .cuda_graphs import CUDAGraphManager
            _module_cache[name] = CUDAGraphManager
            return CUDAGraphManager
            
        elif name == 'GraphOptimizedModel':
            from .cuda_graphs import GraphOptimizedModel
            _module_cache[name] = GraphOptimizedModel
            return GraphOptimizedModel
            
        elif name == 'InferenceManager':
            from .inference_manager import InferenceManager
            _module_cache[name] = InferenceManager
            return InferenceManager

        elif name == 'VoiceCloner':
            from .voice_cloner import VoiceCloner
            _module_cache[name] = VoiceCloner
            return VoiceCloner

        elif name == 'SingingConversionPipeline':
            from .singing_conversion_pipeline import SingingConversionPipeline
            _module_cache[name] = SingingConversionPipeline
            return SingingConversionPipeline

        elif name == 'VoiceConversionPipeline':
            from .voice_conversion_pipeline import VoiceConversionPipeline
            _module_cache[name] = VoiceConversionPipeline
            return VoiceConversionPipeline

        elif name == 'PipelineConfig':
            from .voice_conversion_pipeline import PipelineConfig
            _module_cache[name] = PipelineConfig
            return PipelineConfig

        elif name == 'VoiceConversionError':
            from .voice_conversion_pipeline import VoiceConversionError
            _module_cache[name] = VoiceConversionError
            return VoiceConversionError

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to import {name}: {e}")
        raise LazyImportError(f"{name} is unavailable due to missing dependencies: {e}") from e

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
