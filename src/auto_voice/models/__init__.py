"""Models module for AutoVoice"""

# Lazy import implementation to avoid loading heavy modules at import time
# and to handle optional dependencies gracefully

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .transformer import VoiceTransformer
    from .hifigan import HiFiGANGenerator
    from .voice_model import VoiceModel
    from .speaker_encoder import SpeakerEncoder

class LazyImportError(ImportError):
    """Raised when a lazy import fails due to missing optional dependencies."""
    pass

import logging
logger = logging.getLogger(__name__)

__all__ = ['VoiceModel', 'VoiceTransformer', 'HiFiGANGenerator', 'SpeakerEncoder']

# Module-level cache for lazy-loaded classes
_module_cache = {}


def __getattr__(name):
    """Lazy import mechanism for heavy model classes.

    This delays importing torch and model classes until they're actually accessed,
    improving import speed and handling optional dependencies gracefully.
    """
    if name not in __all__:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    # Check cache first
    if name in _module_cache:
        return _module_cache[name]

    # Lazy import with proper error handling
    try:
        if name == 'VoiceModel':
            from .voice_model import VoiceModel
            _module_cache[name] = VoiceModel
            return VoiceModel
        elif name == 'VoiceTransformer':
            from .transformer import VoiceTransformer
            _module_cache[name] = VoiceTransformer
            return VoiceTransformer
        elif name == 'HiFiGANGenerator':
            from .hifigan import HiFiGANGenerator
            _module_cache[name] = HiFiGANGenerator
            return HiFiGANGenerator
        elif name == 'SpeakerEncoder':
            from .speaker_encoder import SpeakerEncoder
            _module_cache[name] = SpeakerEncoder
            return SpeakerEncoder
    except Exception as e:
        logger.warning(f"Failed to import {name}: {e}")
        raise LazyImportError(f"{name} is unavailable due to missing dependencies: {e}") from e

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
