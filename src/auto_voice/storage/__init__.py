"""Storage module for AutoVoice - Voice profile management"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .voice_profiles import VoiceProfileStorage

__all__ = ['VoiceProfileStorage']

# Module-level cache for lazy-loaded classes
_module_cache = {}


class LazyImportError(ImportError):
    """Raised when a lazy import fails."""
    pass


def __getattr__(name):
    """Lazy import mechanism for storage classes."""
    if name not in __all__:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    # Check cache first
    if name in _module_cache:
        return _module_cache[name]

    # Lazy import
    try:
        if name == 'VoiceProfileStorage':
            from .voice_profiles import VoiceProfileStorage
            _module_cache[name] = VoiceProfileStorage
            return VoiceProfileStorage
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to import {name}: {e}")
        raise LazyImportError(f"{name} is unavailable: {e}") from e

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
