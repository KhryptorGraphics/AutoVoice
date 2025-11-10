"""AutoVoice: Advanced Voice Synthesis and Processing System"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .web.app import create_app
    from .utils.config_loader import load_config
    from .gpu.gpu_manager import GPUManager
    from .audio.processor import AudioProcessor
    from .audio.gpu_processor import GPUAudioProcessor
    from .models.voice_model import VoiceModel
    from .models.transformer import VoiceTransformer
    from .models.hifigan import HiFiGANGenerator
    from .training.trainer import VoiceTrainer
    from .inference.engine import VoiceInferenceEngine
    from .inference.synthesizer import VoiceSynthesizer

__version__ = "1.0.0"

__all__ = [
    'create_app',
    'load_config',
    'GPUManager',
    'AudioProcessor',
    'GPUAudioProcessor',
    'VoiceModel',
    'VoiceTransformer',
    'HiFiGANGenerator',
    'VoiceTrainer',
    'VoiceInferenceEngine',
    'VoiceSynthesizer',
    'initialize_system',
    'run_app'
]

# Module-level cache for lazy-loaded components
_module_cache = {}


def __getattr__(name):
    """Lazy import mechanism for AutoVoice components.

    This delays importing heavy modules (torch, TensorRT, etc.) until they're
    actually accessed, improving import speed and handling optional dependencies
    gracefully.
    """
    if name not in __all__:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    # Check cache first
    if name in _module_cache:
        return _module_cache[name]

    # Lazy import with proper error handling
    try:
        if name == 'create_app':
            from .web.app import create_app
            _module_cache[name] = create_app
            return create_app
        elif name == 'load_config':
            from .utils.config_loader import load_config
            _module_cache[name] = load_config
            return load_config
        elif name == 'GPUManager':
            from .gpu.gpu_manager import GPUManager
            _module_cache[name] = GPUManager
            return GPUManager
        elif name == 'AudioProcessor':
            from .audio.processor import AudioProcessor
            _module_cache[name] = AudioProcessor
            return AudioProcessor
        elif name == 'GPUAudioProcessor':
            from .audio.gpu_processor import GPUAudioProcessor
            _module_cache[name] = GPUAudioProcessor
            return GPUAudioProcessor
        elif name == 'VoiceModel':
            from .models.voice_model import VoiceModel
            _module_cache[name] = VoiceModel
            return VoiceModel
        elif name == 'VoiceTransformer':
            from .models.transformer import VoiceTransformer
            _module_cache[name] = VoiceTransformer
            return VoiceTransformer
        elif name == 'HiFiGANGenerator':
            from .models.hifigan import HiFiGANGenerator
            _module_cache[name] = HiFiGANGenerator
            return HiFiGANGenerator
        elif name == 'VoiceTrainer':
            from .training.trainer import VoiceTrainer
            _module_cache[name] = VoiceTrainer
            return VoiceTrainer
        elif name == 'VoiceInferenceEngine':
            from .inference.engine import VoiceInferenceEngine
            _module_cache[name] = VoiceInferenceEngine
            return VoiceInferenceEngine
        elif name == 'VoiceSynthesizer':
            from .inference.synthesizer import VoiceSynthesizer
            _module_cache[name] = VoiceSynthesizer
            return VoiceSynthesizer
        elif name == 'initialize_system':
            # Import from main.py at project root
            import sys
            import os
            _parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if _parent_dir not in sys.path:
                sys.path.insert(0, _parent_dir)
            from main import initialize_system
            _module_cache[name] = initialize_system
            return initialize_system
        elif name == 'run_app':
            # Import from main.py at project root
            import sys
            import os
            _parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if _parent_dir not in sys.path:
                sys.path.insert(0, _parent_dir)
            from main import run_app
            _module_cache[name] = run_app
            return run_app
    except ImportError as e:
        # If import fails (e.g., torch/dependencies not available), set to None and cache
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to import {name}: {e}. Setting to None.")
        _module_cache[name] = None
        return None

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
