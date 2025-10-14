"""Audio processing module for AutoVoice"""

# Lazy import implementation to avoid loading heavy modules at import time
# and to handle optional dependencies gracefully

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .processor import AudioProcessor
    from .gpu_processor import GPUAudioProcessor

__all__ = ['AudioProcessor', 'GPUAudioProcessor']

# Module-level cache for lazy-loaded classes
_module_cache = {}


class LazyImportError(ImportError):
    pass


def __getattr__(name):
    """Lazy import mechanism for audio processor classes.

    This delays importing torch and other heavy dependencies until they're actually accessed,
    improving import speed and handling optional dependencies gracefully.
    """
    if name not in __all__:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    # Check cache first
    if name in _module_cache:
        return _module_cache[name]

    # Lazy import with proper error handling
    try:
        if name == 'AudioProcessor':
            from .processor import AudioProcessor
            _module_cache[name] = AudioProcessor
            return AudioProcessor
        elif name == 'GPUAudioProcessor':
            from .gpu_processor import GPUAudioProcessor
            _module_cache[name] = GPUAudioProcessor
            return GPUAudioProcessor
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to import {name}: {e}")
        raise LazyImportError(f"{name} is unavailable due to missing dependencies: {e}") from e

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
