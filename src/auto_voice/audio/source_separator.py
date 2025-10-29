"""Vocal separation utilities for AutoVoice using Demucs and Spleeter"""

from __future__ import annotations
import hashlib
import logging
import os
import shutil
import threading
import time
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, Callable, List

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False

try:
    from spleeter.separator import Separator
    SPLEETER_AVAILABLE = True
except ImportError:
    SPLEETER_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from .processor import AudioProcessor

logger = logging.getLogger(__name__)


class VocalSeparationError(Exception):
    """Base exception for vocal separation errors"""
    pass


class ModelLoadError(VocalSeparationError):
    """Exception raised when model loading fails"""
    pass


class SeparationError(VocalSeparationError):
    """Exception raised when separation process fails"""
    pass


class VocalSeparator:
    """High-level interface for vocal/instrumental separation with GPU acceleration and caching

    This class provides seamless vocal separation using Demucs (primary) and Spleeter (fallback)
    with intelligent caching, GPU acceleration, and comprehensive error handling.

    Multi-Channel Audio Handling:
        - Input audio with more than 2 channels: selects first two channels for stereo separation
        - Mono audio is converted to stereo by duplicating the single channel
        - Output is always stereo (2 channels) regardless of input channel count
        - For best results with multi-channel audio, consider pre-mixing to stereo with custom weights

    Sample Rate Preservation:
        - When preserve_sample_rate=True (default), outputs are resampled to match input sample rate
        - Processing occurs at the configured sample_rate (default 44.1kHz)
        - Original sample rate is detected and outputs are resampled back if different

    Thread Safety:
        - Model inference is protected by an internal lock for thread-safe batch processing
        - Multiple threads can safely call separate_vocals() concurrently
        - The lock serializes access to the shared model instance to prevent race conditions
        - For best parallel performance, consider using separate VocalSeparator instances per thread

    Example:
        >>> separator = VocalSeparator(device='cuda', gpu_manager=gpu_manager)
        >>> vocals, instrumental, sr = separator.separate_vocals('song.mp3')
        >>> # vocals and instrumental are numpy arrays of the separated tracks
        >>> # sr is the output sample rate (matches input if preserve_sample_rate=True)

    Attributes:
        config (Dict[str, Any]): Configuration dictionary
        device (str): Device for processing ('cuda', 'cpu', etc.)
        gpu_manager: Optional GPUManager for GPU acceleration
        audio_processor (AudioProcessor): Audio I/O handler
        model: Loaded separation model (Demucs or Spleeter)
        cache_dir (Path): Directory for cached separations
        backend (str): Active backend ('demucs' or 'spleeter')
        lock (threading.RLock): Reentrant lock for thread-safe model access
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        gpu_manager = None
    ):
        """Initialize VocalSeparator with configuration and device settings

        Args:
            config: Optional configuration dict with separation parameters.
                   Keys: model, cache_enabled, cache_dir, cache_size_limit_gb,
                         sample_rate, shifts, overlap, split, backend_priority
            device: Optional device string ('cuda', 'cpu', 'cuda:0', etc.)
            gpu_manager: Optional GPUManager instance for GPU acceleration

        Raises:
            ModelLoadError: If no separation backend can be loaded
        """
        # Initialize configuration with defaults
        self.config = config or {}
        self._load_default_config()

        # Load YAML config if available and merge with user config (user config takes precedence)
        self._load_yaml_config()

        # Device setup
        self.gpu_manager = gpu_manager
        if device is not None:
            self.device = device
        elif gpu_manager and gpu_manager.is_cuda_available():
            self.device = str(gpu_manager.get_device())
        else:
            self.device = 'cpu'

        # Initialize audio processor for I/O
        self.audio_processor = AudioProcessor(
            config={'sample_rate': self.config['sample_rate']},
            device=self.device
        )

        # State tracking
        self.model = None
        self.backend = None
        self.lock = threading.RLock()

        # Setup cache directory
        cache_dir = self.config.get('cache_dir', '~/.cache/autovoice/separated/')
        self.cache_dir = Path(cache_dir).expanduser()
        if self.config.get('cache_enabled', True):
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize separation backend
        self._initialize_backend()

        logger.info(f"VocalSeparator initialized with backend={self.backend}, device={self.device}")

    def _load_default_config(self):
        """Load default configuration values"""
        defaults = {
            'model': 'htdemucs',
            'sample_rate': 44100,
            'shifts': 1,
            'overlap': 0.25,
            'split': True,
            'cache_enabled': True,
            'cache_dir': '~/.cache/autovoice/separated/',
            'cache_size_limit_gb': 10,
            'cache_ttl_days': 30,
            'backend_priority': ['demucs', 'spleeter'],
            'fallback_enabled': True,
            'normalize_output': True,
            'preserve_sample_rate': True,
            'mixed_precision': True,
            'show_progress': False,
            'defer_model_load': True,
            'quality_preset': 'balanced',
            'batch_max_workers': 4,
            'lru_access_tracking': True
        }

        # Merge with user config (user config takes precedence)
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value

    def _load_yaml_config(self):
        """Load configuration from YAML file and environment variables"""
        # Try to read YAML config
        config_path = Path(__file__).parent.parent.parent.parent / 'config' / 'audio_config.yaml'
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    if yaml_config and 'vocal_separation' in yaml_config:
                        # Merge YAML config (only if not already set by user)
                        for key, value in yaml_config['vocal_separation'].items():
                            if key not in self.config or self.config[key] == self._get_default_for_key(key):
                                self.config[key] = value
            except Exception as e:
                logger.warning(f"Failed to load YAML config from {config_path}: {e}")

        # Read environment overrides (highest priority)
        env_mapping = {
            'AUTOVOICE_SEPARATION_MODEL': 'model',
            'AUTOVOICE_SEPARATION_BACKEND': 'backend_priority',
            'AUTOVOICE_SEPARATION_CACHE_DIR': 'cache_dir',
            'AUTOVOICE_SEPARATION_CACHE_ENABLED': 'cache_enabled',
            'AUTOVOICE_SEPARATION_SAMPLE_RATE': 'sample_rate',
            'AUTOVOICE_SEPARATION_SHIFTS': 'shifts',
            'AUTOVOICE_SEPARATION_OVERLAP': 'overlap',
            'AUTOVOICE_SEPARATION_SPLIT': 'split',
            'AUTOVOICE_SEPARATION_SHOW_PROGRESS': 'show_progress'
        }

        for env_var, config_key in env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Type conversion
                if config_key in ['cache_enabled', 'split', 'show_progress']:
                    self.config[config_key] = env_value.lower() in ('true', '1', 'yes')
                elif config_key in ['sample_rate', 'shifts']:
                    self.config[config_key] = int(env_value)
                elif config_key == 'overlap':
                    self.config[config_key] = float(env_value)
                elif config_key == 'backend_priority':
                    self.config[config_key] = [b.strip() for b in env_value.split(',')]
                else:
                    self.config[config_key] = env_value

    def _get_default_for_key(self, key: str) -> Any:
        """Get default value for a config key"""
        defaults = {
            'model': 'htdemucs',
            'sample_rate': 44100,
            'shifts': 1,
            'overlap': 0.25,
            'split': True,
            'cache_enabled': True,
            'cache_dir': '~/.cache/autovoice/separated/',
            'cache_size_limit_gb': 10,
            'cache_ttl_days': 30,
            'backend_priority': ['demucs', 'spleeter'],
            'fallback_enabled': True,
            'normalize_output': True,
            'preserve_sample_rate': True,
            'mixed_precision': True,
            'show_progress': False,
            'quality_preset': 'balanced',
            'batch_max_workers': 4,
            'lru_access_tracking': True
        }
        return defaults.get(key)

    def _initialize_backend(self):
        """Initialize separation backend (Demucs or Spleeter)

        When defer_model_load=True (default), only sets self.backend without loading models.
        Models are loaded lazily on first use in separate_vocals().
        """
        backend_priority = self.config.get('backend_priority', ['demucs', 'spleeter'])
        defer_load = self.config.get('defer_model_load', True)

        for backend_name in backend_priority:
            try:
                if backend_name == 'demucs' and DEMUCS_AVAILABLE:
                    if not defer_load:
                        self._load_demucs_model()
                        logger.info(f"Loaded Demucs model: {self.config['model']}")
                    else:
                        logger.debug("Demucs backend selected, model loading deferred")
                    self.backend = 'demucs'
                    return
                elif backend_name == 'spleeter' and SPLEETER_AVAILABLE:
                    if not defer_load:
                        self._load_spleeter_model()
                        logger.info("Loaded Spleeter model: 2stems")
                    else:
                        logger.debug("Spleeter backend selected, model loading deferred")
                    self.backend = 'spleeter'
                    return
            except Exception as e:
                logger.warning(f"Failed to load {backend_name} backend: {e}")
                continue

        # No backend available
        error_msg = "No separation backend available. Install demucs or spleeter."
        if not DEMUCS_AVAILABLE and not SPLEETER_AVAILABLE:
            error_msg += "\n  pip install demucs"
            if not SPLEETER_AVAILABLE:
                error_msg += "\n  pip install spleeter>=2.4.0,<3.0.0"
        logger.error(error_msg)
        raise ModelLoadError(error_msg)

    def _load_demucs_model(self):
        """Load Demucs model"""
        if not DEMUCS_AVAILABLE:
            raise ModelLoadError("Demucs not available")

        model_name = self.config.get('model', 'htdemucs')

        try:
            # Load pretrained Demucs model
            self.model = get_model(model_name)

            # Move to device
            if TORCH_AVAILABLE:
                device = torch.device(self.device)
                self.model.to(device)
                self.model.eval()  # Set to evaluation mode

            # Optimize with GPUManager if available
            if hasattr(self, 'gpu_manager') and self.gpu_manager is not None:
                self.model = self.gpu_manager.optimize_model(self.model, model_name='demucs')

            logger.debug(f"Demucs model '{model_name}' loaded on {self.device}")

        except Exception as e:
            raise ModelLoadError(f"Failed to load Demucs model '{model_name}': {e}")

    def _load_spleeter_model(self):
        """Load Spleeter model"""
        if not SPLEETER_AVAILABLE:
            raise ModelLoadError("Spleeter not available")

        try:
            # Initialize Spleeter separator with 2stems model
            self.model = Separator('spleeter:2stems')
            logger.debug("Spleeter 2stems model loaded")

        except Exception as e:
            raise ModelLoadError(f"Failed to load Spleeter model: {e}")

    def separate_vocals(
        self,
        audio_path: str,
        use_cache: bool = True,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Separate vocals from instrumental in audio file

        This is the primary method for vocal separation. It handles caching,
        format conversion, and fallback between backends automatically.

        Processing Pipeline:
            1. Load audio and detect original sample rate
            2. Resample to processing sample rate (default 44.1kHz) if needed
            3. Handle multi-channel inputs (>2 channels: select first two channels; mono: duplicate to stereo)
            4. Perform separation using Demucs (or Spleeter fallback)
            5. Resample back to original sample rate if preserve_sample_rate=True
            6. Return stereo outputs

        Args:
            audio_path: Path to audio file (MP3/WAV/FLAC/OGG supported)
            use_cache: Whether to use cached results if available
            progress_callback: Optional callback for progress updates (0.0 to 1.0)

        Returns:
            Tuple of (vocals, instrumental) where:
            - vocals: numpy array (2, samples) - stereo separated vocals
            - instrumental: numpy array (2, samples) - stereo separated instrumental

        Raises:
            FileNotFoundError: If audio file doesn't exist
            SeparationError: If separation fails on all backends

        Example:
            >>> vocals, instrumental = separator.separate_vocals('song.mp3')
            >>> print(vocals.shape, instrumental.shape)
            (2, 3528000) (2, 3528000)  # Stereo, ~80 seconds
        """
        audio_path = Path(audio_path)

        # Validate file exists
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Check cache first
        if use_cache and self.config.get('cache_enabled', True):
            cache_key = self._get_cache_key(str(audio_path))
            cached_result = self._load_from_cache(cache_key)
            if cached_result is not None:
                logger.info(f"Cache hit for {audio_path.name}")
                if progress_callback:
                    progress_callback(1.0)
                # Cache returns (vocals, instrumental)
                vocals_cached, instrumental_cached = cached_result
                return vocals_cached, instrumental_cached

        # Load audio
        if progress_callback:
            progress_callback(0.1)

        logger.info(f"Separating vocals from {audio_path.name} using {self.backend}")

        try:
            # Load audio using AudioProcessor with preserve_channels=True to avoid premature mono downmixing
            audio, original_sr = self.audio_processor.load_audio(
                str(audio_path),
                target_sr=self.config['sample_rate'],
                return_sr=True,
                preserve_channels=True
            )

            # Store processing sample rate
            processing_sr = self.config['sample_rate']

            # Determine final output sample rate
            if self.config.get('preserve_sample_rate', True):
                output_sr = original_sr
            else:
                output_sr = processing_sr

            if progress_callback:
                progress_callback(0.2)

            # Handle multi-channel inputs - ensure stereo for separation
            # IMPORTANT: Do NOT downmix to mono first - preserve stereo information for Demucs
            if TORCH_AVAILABLE:
                if audio.dim() == 1:
                    # Mono: upmix to stereo by duplicating
                    audio = audio.unsqueeze(0).repeat(2, 1)
                elif audio.dim() == 2:
                    if audio.shape[0] == 1:
                        # Single channel: upmix to stereo by duplicating
                        audio = audio.repeat(2, 1)
                    elif audio.shape[0] > 2:
                        # Multi-channel (>2): downmix to stereo WITHOUT averaging to mono first
                        logger.info(f"Downmixing {audio.shape[0]} channels to stereo (preserving spatial information)")
                        # Keep first 2 channels or use proper stereo downmix
                        audio = audio[:2, :]  # Use first two channels directly to preserve stereo
            else:
                if audio.ndim == 1:
                    # Mono: upmix to stereo by duplicating
                    audio = np.stack([audio, audio])
                elif audio.ndim == 2:
                    if audio.shape[0] == 1:
                        # Single channel: upmix to stereo by duplicating
                        audio = np.repeat(audio, 2, axis=0)
                    elif audio.shape[0] > 2:
                        # Multi-channel (>2): downmix to stereo WITHOUT averaging to mono first
                        logger.info(f"Downmixing {audio.shape[0]} channels to stereo (preserving spatial information)")
                        # Keep first 2 channels to preserve stereo
                        audio = audio[:2, :]

            if progress_callback:
                progress_callback(0.3)

            # Lazy load model if not already loaded
            if self.model is None:
                logger.info(f"Lazy loading {self.backend} model on first use...")
                if self.backend == 'demucs':
                    self._load_demucs_model()
                elif self.backend == 'spleeter':
                    self._load_spleeter_model()
                else:
                    raise SeparationError(f"Unknown backend: {self.backend}")

            # Perform separation based on backend with thread-safety
            # Lock is required for thread-safe batch processing when using shared model
            try:
                with self.lock:
                    if self.backend == 'demucs':
                        vocals, instrumental = self._separate_with_demucs(audio, progress_callback)
                    elif self.backend == 'spleeter':
                        vocals, instrumental = self._separate_with_spleeter(audio, progress_callback)
                    else:
                        raise SeparationError(f"Unknown backend: {self.backend}")

            except Exception as e:
                # Try fallback if enabled
                if self.config.get('fallback_enabled', True):
                    logger.warning(f"Primary backend failed: {e}. Trying fallback...")
                    with self.lock:
                        vocals, instrumental = self._try_fallback(audio, progress_callback, e)
                else:
                    raise

            if progress_callback:
                progress_callback(0.9)

            # Post-processing
            if self.config.get('normalize_output', True):
                vocals = self._normalize_audio(vocals)
                instrumental = self._normalize_audio(instrumental)

            # Resample back to original sample rate if preserve_sample_rate is enabled
            if self.config.get('preserve_sample_rate', True) and original_sr != processing_sr:
                logger.info(f"Resampling output from {processing_sr} Hz to original {original_sr} Hz")

                if TORCH_AVAILABLE:
                    # Use torchaudio for resampling if available
                    try:
                        import torchaudio.transforms as T
                        resampler = T.Resample(orig_freq=processing_sr, new_freq=original_sr)

                        # Ensure tensors for resampling
                        if isinstance(vocals, np.ndarray):
                            vocals_tensor = torch.from_numpy(vocals)
                            instrumental_tensor = torch.from_numpy(instrumental)
                        else:
                            vocals_tensor = vocals
                            instrumental_tensor = instrumental

                        # Move tensors to CPU before resampling (torchaudio Resample expects CPU tensors)
                        vocals_tensor = vocals_tensor.detach().cpu()
                        instrumental_tensor = instrumental_tensor.detach().cpu()

                        # Resample on CPU
                        vocals = resampler(vocals_tensor)
                        instrumental = resampler(instrumental_tensor)

                    except Exception as e:
                        logger.warning(f"Failed to resample with torchaudio: {e}, trying librosa")
                        # Fallback to librosa
                        if LIBROSA_AVAILABLE:
                            import librosa
                            if isinstance(vocals, torch.Tensor):
                                vocals = vocals.detach().cpu().numpy()
                                instrumental = instrumental.detach().cpu().numpy()

                            # Resample each channel
                            if vocals.ndim == 2:
                                vocals = np.stack([
                                    librosa.resample(vocals[i], orig_sr=processing_sr, target_sr=original_sr)
                                    for i in range(vocals.shape[0])
                                ])
                                instrumental = np.stack([
                                    librosa.resample(instrumental[i], orig_sr=processing_sr, target_sr=original_sr)
                                    for i in range(instrumental.shape[0])
                                ])
                            else:
                                vocals = librosa.resample(vocals, orig_sr=processing_sr, target_sr=original_sr)
                                instrumental = librosa.resample(instrumental, orig_sr=processing_sr, target_sr=original_sr)
                        else:
                            logger.warning("Cannot resample: neither torchaudio nor librosa available. Outputs at processing sample rate.")

            # Convert to numpy if needed
            if TORCH_AVAILABLE and isinstance(vocals, torch.Tensor):
                vocals = vocals.detach().cpu().numpy()
                instrumental = instrumental.detach().cpu().numpy()

            # Save to cache with output sample rate
            if use_cache and self.config.get('cache_enabled', True):
                self._save_to_cache(cache_key, vocals, instrumental, output_sr)

            if progress_callback:
                progress_callback(1.0)

            logger.info(f"Separation complete for {audio_path.name}")
            return vocals, instrumental

        except Exception as e:
            logger.error(f"Separation failed for {audio_path.name}: {e}")
            raise SeparationError(f"Failed to separate vocals: {e}")

    def _separate_with_demucs(
        self,
        audio: torch.Tensor,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Separate vocals using Demucs model

        Args:
            audio: Input audio tensor (channels, samples)
            progress_callback: Optional progress callback

        Returns:
            Tuple of (vocals, instrumental) where:
            - vocals: tensor (2, samples)
            - instrumental: tensor (2, samples)
        """
        if not DEMUCS_AVAILABLE or self.backend != 'demucs':
            raise SeparationError("Demucs backend not available")

        # Use GPUManager device context if available
        context_manager = self.gpu_manager.device_context() if self.gpu_manager else torch.no_grad()

        try:
            with context_manager as device:
                # Use the yielded device from context or fallback to self.device
                if device is None or not self.gpu_manager:
                    device = torch.device(self.device)

                # Ensure audio is on correct device
                audio = audio.to(device)

                # Ensure model is on same device as audio
                if self.model is not None and hasattr(self.model, 'to'):
                    if next(self.model.parameters(), None) is not None:
                        model_device = next(self.model.parameters()).device
                        if model_device != device:
                            self.model.to(device)

                # Add batch dimension: (1, channels, samples)
                if audio.dim() == 2:
                    audio = audio.unsqueeze(0)

                # Determine if we can use AMP (only on CUDA devices)
                # Use the same device variable that was yielded/assigned above
                use_amp = (getattr(device, 'type', 'cpu') == 'cuda' and
                          torch.cuda.is_available() and
                          self.config.get('mixed_precision', True))

                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        sources = apply_model(
                            self.model,
                            audio,
                            shifts=self.config.get('shifts', 1),
                            split=self.config.get('split', True),
                            overlap=self.config.get('overlap', 0.25),
                            progress=self.config.get('show_progress', False)
                        )

                # Extract vocals and instrumental based on model's source order
                # sources shape: (batch, sources, channels, samples)
                # Dynamically determine source indices from model.sources
                source_names = getattr(self.model, 'sources', ['drums', 'bass', 'other', 'vocals'])

                # Robustly set vocals_idx
                if 'vocals' in source_names:
                    vocals_idx = source_names.index('vocals')
                    vocals = sources[0, vocals_idx]
                else:
                    # Fallback: assume vocals is last source
                    logger.warning(f"'vocals' not found in model sources {source_names}, using last source")
                    vocals_idx = len(source_names) - 1
                    vocals = sources[0, vocals_idx]

                # For instrumental, sum all non-vocal sources
                # Handle both 2-stem (vocals/accompaniment) and 4-stem models
                if len(source_names) == 2:
                    # 2-stem model: use the non-vocals stem directly
                    accompaniment_idx = 1 - vocals_idx
                    instrumental = sources[0, accompaniment_idx]
                else:
                    # 4-stem model: sum all non-vocal sources
                    non_vocal_indices = [i for i, name in enumerate(source_names) if name != 'vocals']
                    if non_vocal_indices:
                        instrumental = sources[0, non_vocal_indices].sum(0)
                    else:
                        # Fallback: sum all but vocals index
                        instrumental = torch.cat([sources[0, :vocals_idx], sources[0, vocals_idx+1:]], dim=0).sum(0)

                if progress_callback:
                    progress_callback(0.8)

                return vocals, instrumental

        except RuntimeError as e:
            # Handle GPU OOM error with fallback to CPU
            if 'out of memory' in str(e).lower() and self.gpu_manager:
                logger.warning(f"GPU OOM detected, attempting CPU fallback: {e}")
                try:
                    # Clear GPU cache
                    self.gpu_manager.clear_cache()

                    # Move model to CPU
                    logger.info("Moving model to CPU for retry")
                    self.model.cpu()
                    self.device = 'cpu'

                    # Optimize memory before retry
                    if hasattr(self.gpu_manager, 'optimize_memory'):
                        self.gpu_manager.optimize_memory()

                    # Retry on CPU
                    return self._separate_with_demucs(audio.cpu(), progress_callback)

                except Exception as fallback_error:
                    logger.error(f"CPU fallback also failed: {fallback_error}")
                    raise SeparationError(f"Demucs separation failed on both GPU and CPU: {fallback_error}")
            else:
                logger.error(f"Demucs separation failed: {e}")
                raise SeparationError(f"Demucs separation error: {e}")

        except Exception as e:
            logger.error(f"Demucs separation failed: {e}")
            raise SeparationError(f"Demucs separation error: {e}")

    def _separate_with_spleeter(
        self,
        audio: np.ndarray,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Separate vocals using Spleeter model

        Spleeter was trained on 44.1kHz audio. If input sample rate differs,
        this method resamples to 44100 Hz before separation.

        Args:
            audio: Input audio array (channels, samples) or tensor
            progress_callback: Optional progress callback

        Returns:
            Tuple of (vocals, instrumental) where:
            - vocals: numpy array (2, samples)
            - instrumental: numpy array (2, samples)
        """
        if not SPLEETER_AVAILABLE or self.backend != 'spleeter':
            raise SeparationError("Spleeter backend not available")

        try:
            # Convert to numpy if torch tensor
            if TORCH_AVAILABLE and isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()

            # Spleeter expects (samples, channels) format
            if audio.ndim == 2 and audio.shape[0] == 2:  # (channels, samples)
                audio = audio.T  # Transpose to (samples, channels)
            elif audio.ndim == 1:
                # If mono, make it (samples, 1)
                audio = audio.reshape(-1, 1)

            # Ensure proper shape: (samples, channels)
            if audio.ndim == 1:
                audio = audio.reshape(-1, 1)

            # Detect GPU availability for Spleeter (TensorFlow backend)
            try:
                import tensorflow as tf
                gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
                if gpu_available:
                    logger.info("Spleeter using GPU acceleration (TensorFlow)")
                else:
                    logger.warning("Spleeter running on CPU only (no TensorFlow GPU support detected)")
            except Exception:
                logger.warning("Could not detect TensorFlow GPU availability; assuming CPU-only")

            # Resample to 44.1kHz if needed (Spleeter expects 44100 Hz)
            expected_sr = 44100
            input_sr = self.config.get('sample_rate', 44100)
            if input_sr != expected_sr:
                logger.info(f"Resampling audio from {input_sr} Hz to {expected_sr} Hz for Spleeter")

                if LIBROSA_AVAILABLE:
                    import librosa
                    # Resample each channel
                    if audio.shape[1] == 2:  # Stereo
                        # Resample each channel independently
                        left_resampled = librosa.resample(audio[:, 0], orig_sr=input_sr, target_sr=expected_sr)
                        right_resampled = librosa.resample(audio[:, 1], orig_sr=input_sr, target_sr=expected_sr)

                        # Handle potential off-by-one length mismatch from resampling
                        min_len = min(len(left_resampled), len(right_resampled))
                        audio = np.stack([
                            left_resampled[:min_len],
                            right_resampled[:min_len]
                        ], axis=1)
                    else:  # Mono
                        audio = librosa.resample(audio[:, 0], orig_sr=input_sr, target_sr=expected_sr).reshape(-1, 1)
                elif TORCH_AVAILABLE:
                    # Fallback to torchaudio for resampling if librosa unavailable
                    try:
                        import torchaudio.transforms as T
                        logger.info("Using torchaudio for resampling (librosa not available)")

                        # Convert to torch tensor if needed
                        if isinstance(audio, np.ndarray):
                            audio_tensor = torch.from_numpy(audio.astype(np.float32))
                        else:
                            audio_tensor = audio

                        # audio is in (samples, channels) format, need to transpose to (channels, samples)
                        audio_tensor = audio_tensor.T  # Now (channels, samples)

                        # Create resampler and resample
                        resampler = T.Resample(orig_freq=input_sr, new_freq=expected_sr)
                        audio_tensor = resampler(audio_tensor)

                        # Convert back to numpy and transpose to (samples, channels)
                        audio = audio_tensor.T.cpu().numpy()

                    except Exception as e:
                        logger.warning(f"Failed to resample with torchaudio: {e}. Separation quality may be affected.")
                else:
                    logger.warning(
                        f"Neither librosa nor torchaudio available; cannot resample from {input_sr} Hz to {expected_sr} Hz. "
                        f"Separation quality may be affected."
                    )

            # Run separation
            prediction = self.model.separate(audio)

            # Extract vocals and accompaniment
            vocals = prediction['vocals'].T  # Transpose back to (channels, samples)
            instrumental = prediction['accompaniment'].T

            if progress_callback:
                progress_callback(0.8)

            return vocals, instrumental

        except Exception as e:
            logger.error(f"Spleeter separation failed: {e}")
            raise SeparationError(f"Spleeter separation error: {e}")

    def _try_fallback(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        progress_callback: Optional[Callable[[float], None]] = None,
        original_error: Optional[Exception] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Try fallback backend if primary fails

        Args:
            audio: Input audio array or tensor
            progress_callback: Optional progress callback
            original_error: Optional original exception from primary backend

        Returns:
            Tuple of (vocals, instrumental)

        Raises:
            SeparationError: If fallback backend is not available or fails
        """
        # Check if fallback backend is available before attempting
        if self.backend == 'demucs':
            if not SPLEETER_AVAILABLE:
                error_msg = f"Primary backend (Demucs) failed: {original_error}. Fallback backend (Spleeter) not available."
                logger.error(error_msg)
                if original_error:
                    raise SeparationError(error_msg) from original_error
                else:
                    raise SeparationError(error_msg)

            logger.warning(f"Primary backend (Demucs) failed: {original_error}. Falling back to Spleeter")
            old_backend = self.backend
            original_model = self.model
            try:
                self._load_spleeter_model()
                self.backend = 'spleeter'
                result = self._separate_with_spleeter(audio, progress_callback)
                return result
            except Exception as fallback_error:
                error_msg = f"All separation backends failed. Demucs error: {original_error}. Spleeter error: {fallback_error}"
                logger.error(error_msg)
                raise SeparationError(error_msg) from fallback_error
            finally:
                # Restore original backend and model regardless of success or failure
                self.backend = old_backend
                self.model = original_model

        elif self.backend == 'spleeter':
            if not DEMUCS_AVAILABLE:
                error_msg = f"Primary backend (Spleeter) failed: {original_error}. Fallback backend (Demucs) not available."
                logger.error(error_msg)
                if original_error:
                    raise SeparationError(error_msg) from original_error
                else:
                    raise SeparationError(error_msg)

            logger.warning(f"Primary backend (Spleeter) failed: {original_error}. Falling back to Demucs")
            old_backend = self.backend
            original_model = self.model
            try:
                self._load_demucs_model()
                self.backend = 'demucs'
                result = self._separate_with_demucs(audio, progress_callback)
                return result
            except Exception as fallback_error:
                error_msg = f"All separation backends failed. Spleeter error: {original_error}. Demucs error: {fallback_error}"
                logger.error(error_msg)
                raise SeparationError(error_msg) from fallback_error
            finally:
                # Restore original backend and model regardless of success or failure
                self.backend = old_backend
                self.model = original_model

        else:
            error_msg = f"No fallback backend available for {self.backend}"
            if original_error:
                error_msg += f". Original error: {original_error}"
                raise SeparationError(error_msg) from original_error
            else:
                raise SeparationError(error_msg)

    def _normalize_audio(self, audio: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Normalize audio to prevent clipping"""
        if TORCH_AVAILABLE and isinstance(audio, torch.Tensor):
            max_val = torch.abs(audio).max()
            if max_val > 0:
                return audio / max_val * 0.95  # Leave headroom
            return audio
        else:
            max_val = np.abs(audio).max()
            if max_val > 0:
                return audio / max_val * 0.95
            return audio

    def _get_cache_key(self, audio_path: str, output_sr: Optional[int] = None) -> str:
        """Generate unique cache key for audio file

        Args:
            audio_path: Path to audio file
            output_sr: Optional output sample rate (if known)

        Returns:
            MD5 hash string for cache filename
        """
        path = Path(audio_path)

        # Determine output sample rate for cache key
        preserve_sr = self.config.get('preserve_sample_rate', True)
        processing_sr = self.config.get('sample_rate', 44100)

        if output_sr is None:
            # If output_sr not provided, compute it based on preserve_sample_rate setting
            if preserve_sr:
                # Need to detect original sample rate to know output SR
                # Quick detection without loading full audio
                try:
                    if LIBROSA_AVAILABLE:
                        import librosa
                        original_sr = librosa.get_samplerate(str(path))
                    else:
                        # Fallback: assume processing SR if can't detect
                        original_sr = processing_sr
                    output_sr = original_sr
                except Exception as e:
                    logger.debug(f"Could not detect sample rate for cache key, using processing SR: {e}")
                    output_sr = processing_sr
            else:
                output_sr = processing_sr

        # Include file path, modification time, model name, processing SR, and output SR in key
        key_components = [
            str(path.absolute()),
            str(path.stat().st_mtime),
            self.backend,
            self.config.get('model', 'default'),
            str(processing_sr),
            str(preserve_sr),
            str(output_sr)
        ]

        key_string = '|'.join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _update_cache_access_time(self, cache_key: str):
        """Update cache access time for LRU tracking

        Writes a timestamp file to track when cache entries are accessed.
        This provides more accurate LRU tracking than relying solely on filesystem atime.

        Args:
            cache_key: Cache key hash
        """
        if not self.config.get('lru_access_tracking', True):
            return

        access_time_path = self.cache_dir / f"{cache_key}_access.timestamp"

        try:
            # Write current timestamp
            with open(access_time_path, 'w') as f:
                f.write(str(time.time()))
        except Exception as e:
            logger.warning(f"Failed to update cache access time for {cache_key}: {e}")

    def _load_from_cache(self, cache_key: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load separated audio from cache

        Args:
            cache_key: Cache key hash

        Returns:
            Tuple of (vocals, instrumental, sample_rate) where:
            - vocals: numpy array (2, samples)
            - instrumental: numpy array (2, samples)
            - sample_rate: int - output sample rate in Hz

            Returns None if cache miss.
        """
        if not NUMPY_AVAILABLE:
            return None

        vocals_path = self.cache_dir / f"{cache_key}_vocals.npy"
        instrumental_path = self.cache_dir / f"{cache_key}_instrumental.npy"
        metadata_path = self.cache_dir / f"{cache_key}_metadata.json"

        if vocals_path.exists() and instrumental_path.exists():
            try:
                # Check TTL if configured
                ttl_days = self.config.get('cache_ttl_days')
                if ttl_days is not None and ttl_days > 0:
                    # Check file age using modification time
                    file_age_seconds = time.time() - vocals_path.stat().st_mtime
                    file_age_days = file_age_seconds / 86400

                    if file_age_days > ttl_days:
                        logger.debug(f"Cache entry expired (age: {file_age_days:.1f} days > TTL: {ttl_days} days)")
                        # Delete expired files
                        try:
                            vocals_path.unlink()
                            instrumental_path.unlink()
                            if metadata_path.exists():
                                metadata_path.unlink()
                            logger.debug(f"Deleted expired cache files: {cache_key}")
                        except Exception as e:
                            logger.warning(f"Failed to delete expired cache files: {e}")
                        return None

                vocals = np.load(vocals_path)
                instrumental = np.load(instrumental_path)

                # Update access time for LRU tracking
                self._update_cache_access_time(cache_key)

                # Also touch the files for filesystem-level tracking (fallback)
                vocals_path.touch()
                instrumental_path.touch()

                return vocals, instrumental
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
                return None

        return None

    def _save_to_cache(self, cache_key: str, vocals: np.ndarray, instrumental: np.ndarray, sample_rate: int):
        """Save separated audio to cache

        Args:
            cache_key: Cache key hash
            vocals: Vocals array
            instrumental: Instrumental array
            sample_rate: Output sample rate in Hz
        """
        if not NUMPY_AVAILABLE:
            return

        with self.lock:
            try:
                # Check cache size and cleanup if needed
                self._enforce_cache_limit()

                # Save to temporary files first (atomic write)
                vocals_temp = self.cache_dir / f"{cache_key}_vocals.npy.tmp"
                instrumental_temp = self.cache_dir / f"{cache_key}_instrumental.npy.tmp"
                metadata_temp = self.cache_dir / f"{cache_key}_metadata.json.tmp"

                np.save(vocals_temp, vocals)
                np.save(instrumental_temp, instrumental)

                # Save metadata
                import json
                metadata = {
                    'cached_at': time.time()
                }
                with open(metadata_temp, 'w') as f:
                    json.dump(metadata, f)

                # Rename to final paths (atomic on POSIX)
                vocals_path = self.cache_dir / f"{cache_key}_vocals.npy"
                instrumental_path = self.cache_dir / f"{cache_key}_instrumental.npy"
                metadata_path = self.cache_dir / f"{cache_key}_metadata.json"

                vocals_temp.rename(vocals_path)
                instrumental_temp.rename(instrumental_path)
                metadata_temp.rename(metadata_path)

                logger.debug(f"Saved to cache: {cache_key} (sample_rate={sample_rate})")

            except Exception as e:
                logger.warning(f"Failed to save to cache: {e}")

    def _get_cache_access_time(self, cache_key: str) -> float:
        """Get the last access time for a cache entry

        Reads from timestamp file if available, otherwise falls back to file mtime.

        Args:
            cache_key: Cache key hash

        Returns:
            Access timestamp (seconds since epoch)
        """
        access_time_path = self.cache_dir / f"{cache_key}_access.timestamp"

        # Try to read from timestamp file first
        if access_time_path.exists():
            try:
                with open(access_time_path, 'r') as f:
                    return float(f.read().strip())
            except Exception as e:
                logger.warning(f"Failed to read access timestamp for {cache_key}: {e}")

        # Fallback to file modification time
        vocals_path = self.cache_dir / f"{cache_key}_vocals.npy"
        if vocals_path.exists():
            try:
                return vocals_path.stat().st_mtime
            except Exception:
                pass

        return 0.0

    def _delete_cache_pair(self, cache_key: str) -> int:
        """Delete both vocal and instrumental files for a cache key atomically

        Args:
            cache_key: Cache key hash

        Returns:
            Total size deleted in bytes (0 if deletion failed)
        """
        vocals_path = self.cache_dir / f"{cache_key}_vocals.npy"
        instrumental_path = self.cache_dir / f"{cache_key}_instrumental.npy"
        access_time_path = self.cache_dir / f"{cache_key}_access.timestamp"

        total_deleted = 0
        vocals_deleted = False
        instrumental_deleted = False

        try:
            # Get sizes before deletion
            vocals_size = vocals_path.stat().st_size if vocals_path.exists() else 0
            instrumental_size = instrumental_path.stat().st_size if instrumental_path.exists() else 0

            # Try to delete both files
            if vocals_path.exists():
                vocals_path.unlink()
                vocals_deleted = True
                total_deleted += vocals_size

            if instrumental_path.exists():
                instrumental_path.unlink()
                instrumental_deleted = True
                total_deleted += instrumental_size

            # Delete timestamp file if exists
            if access_time_path.exists():
                access_time_path.unlink()

            # If we successfully deleted at least one file, count it as success
            if vocals_deleted or instrumental_deleted:
                return total_deleted
            else:
                return 0

        except Exception as e:
            logger.warning(f"Failed to delete cache pair {cache_key}: {e}")
            # Try to clean up partial deletion - if one file was deleted but not the other
            # leave the orphaned file for cleanup in next run
            return 0

    def _enforce_cache_limit(self):
        """Enforce cache size limit using LRU eviction and TTL-based deletion

        Handles vocal/instrumental file pairs atomically to avoid partial cache entries.
        """
        if not self.cache_dir.exists():
            return

        limit_bytes = self.config.get('cache_size_limit_gb', 10) * 1024**3
        ttl_days = self.config.get('cache_ttl_days')
        lru_tracking = self.config.get('lru_access_tracking', True)

        # Get all cache entries (grouped by cache_key)
        cache_entries = {}  # cache_key -> (vocals_path, instrumental_path, total_size, access_time, mtime)
        current_time = time.time()

        for file_path in self.cache_dir.glob("*_vocals.npy"):
            try:
                # Extract cache_key from filename
                cache_key = file_path.stem.replace('_vocals', '')
                instrumental_path = self.cache_dir / f"{cache_key}_instrumental.npy"

                if not instrumental_path.exists():
                    # Orphaned vocal file - try to delete it
                    logger.warning(f"Found orphaned vocal file: {cache_key}, deleting")
                    try:
                        file_path.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete orphaned vocal file: {e}")
                    continue

                # Get file sizes
                vocals_size = file_path.stat().st_size
                instrumental_size = instrumental_path.stat().st_size
                total_size = vocals_size + instrumental_size

                # Get access time (use LRU tracking if enabled)
                if lru_tracking:
                    access_time = self._get_cache_access_time(cache_key)
                else:
                    access_time = file_path.stat().st_atime

                mtime = file_path.stat().st_mtime

                cache_entries[cache_key] = (file_path, instrumental_path, total_size, access_time, mtime)

            except Exception as e:
                logger.warning(f"Failed to process cache file {file_path}: {e}")
                continue

        # Also check for orphaned instrumental files
        for file_path in self.cache_dir.glob("*_instrumental.npy"):
            try:
                cache_key = file_path.stem.replace('_instrumental', '')
                vocals_path = self.cache_dir / f"{cache_key}_vocals.npy"

                if not vocals_path.exists() and cache_key not in cache_entries:
                    # Orphaned instrumental file
                    logger.warning(f"Found orphaned instrumental file: {cache_key}, deleting")
                    try:
                        file_path.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete orphaned instrumental file: {e}")
            except Exception as e:
                logger.warning(f"Failed to check instrumental file {file_path}: {e}")

        # Calculate total cache size
        total_size = sum(entry[2] for entry in cache_entries.values())

        # First pass: Delete expired files based on TTL (atomic pair deletion)
        if ttl_days is not None and ttl_days > 0:
            ttl_seconds = ttl_days * 86400
            keys_to_delete = []

            for cache_key, (vocals_path, instrumental_path, size, _, mtime) in cache_entries.items():
                file_age = current_time - mtime
                if file_age > ttl_seconds:
                    deleted_size = self._delete_cache_pair(cache_key)
                    if deleted_size > 0:
                        total_size -= deleted_size
                        keys_to_delete.append(cache_key)
                        logger.debug(f"Deleted expired cache entry (age: {file_age/86400:.1f} days): {cache_key}")

            # Remove deleted entries
            for cache_key in keys_to_delete:
                del cache_entries[cache_key]

        # Second pass: If still over limit, use LRU eviction (atomic pair deletion)
        if total_size > limit_bytes:
            # Sort by access time (oldest first), fallback to mtime if access_time is 0
            sorted_entries = sorted(
                cache_entries.items(),
                key=lambda x: x[1][3] if x[1][3] > 0 else x[1][4]  # access_time or mtime
            )

            # Remove oldest entries until under limit
            for cache_key, (vocals_path, instrumental_path, size, _, _) in sorted_entries:
                if total_size <= limit_bytes:
                    break

                deleted_size = self._delete_cache_pair(cache_key)
                if deleted_size > 0:
                    total_size -= deleted_size
                    logger.debug(f"Evicted from cache (LRU): {cache_key}")

    def clear_cache(self, max_age_days: Optional[int] = None):
        """Clear cached separation results

        Handles vocal/instrumental file pairs atomically to avoid leaving orphaned files.

        Args:
            max_age_days: If specified, only clear files older than this many days
        """
        if not self.cache_dir.exists():
            return

        with self.lock:
            if max_age_days is None:
                # Clear all cache files - process vocal files and delete pairs
                cache_keys = set()
                for file_path in self.cache_dir.glob("*_vocals.npy"):
                    try:
                        cache_key = file_path.stem.replace('_vocals', '')
                        cache_keys.add(cache_key)
                    except Exception as e:
                        logger.warning(f"Failed to extract cache key from {file_path}: {e}")

                # Delete all pairs
                for cache_key in cache_keys:
                    self._delete_cache_pair(cache_key)

                # Clean up any orphaned files
                for file_path in self.cache_dir.glob("*.npy"):
                    try:
                        file_path.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete orphaned file {file_path}: {e}")

                # Clean up timestamp files
                for file_path in self.cache_dir.glob("*.timestamp"):
                    try:
                        file_path.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete timestamp file {file_path}: {e}")

                logger.info("Cache cleared")
            else:
                # Clear files older than threshold - process as pairs
                threshold = time.time() - (max_age_days * 86400)
                cache_keys_to_delete = []

                for file_path in self.cache_dir.glob("*_vocals.npy"):
                    try:
                        cache_key = file_path.stem.replace('_vocals', '')
                        instrumental_path = self.cache_dir / f"{cache_key}_instrumental.npy"

                        # Check if both files exist and are old enough
                        vocals_old = file_path.stat().st_mtime < threshold
                        instrumental_old = instrumental_path.exists() and instrumental_path.stat().st_mtime < threshold

                        # Delete pair if both are old or if one is missing (orphaned)
                        if vocals_old or not instrumental_path.exists():
                            cache_keys_to_delete.append(cache_key)
                    except Exception as e:
                        logger.warning(f"Failed to check cache file age {file_path}: {e}")

                # Delete old cache pairs
                for cache_key in cache_keys_to_delete:
                    self._delete_cache_pair(cache_key)

                logger.info(f"Cleared {len(cache_keys_to_delete)} cache entries older than {max_age_days} days")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics

        Returns:
            Dictionary with cache information:
                - total_size_mb: Total cache size in MB
                - num_files: Number of cached files
                - cache_dir: Cache directory path
        """
        if not self.cache_dir.exists():
            return {
                'total_size_mb': 0,
                'num_files': 0,
                'cache_dir': str(self.cache_dir)
            }

        total_size = 0
        num_files = 0

        for file_path in self.cache_dir.glob("*.npy"):
            try:
                total_size += file_path.stat().st_size
                num_files += 1
            except Exception:
                pass

        return {
            'total_size_mb': total_size / (1024**2),
            'num_files': num_files,
            'cache_dir': str(self.cache_dir)
        }

    def set_model(self, model_name: str):
        """Switch separation model

        Args:
            model_name: Model to use ('htdemucs', 'htdemucs_ft', 'spleeter:2stems', etc.)

        Raises:
            ModelLoadError: If model loading fails
        """
        with self.lock:
            # Update config
            self.config['model'] = model_name

            # Determine backend from model name
            if 'spleeter' in model_name.lower():
                if not SPLEETER_AVAILABLE:
                    raise ModelLoadError("Spleeter not available")
                self._load_spleeter_model()
                self.backend = 'spleeter'
            else:
                if not DEMUCS_AVAILABLE:
                    raise ModelLoadError("Demucs not available")
                self._load_demucs_model()
                self.backend = 'demucs'

            # Clear cache since results will differ
            self.clear_cache()

            logger.info(f"Switched to model: {model_name} (backend: {self.backend})")

    def separate_vocals_batch(
        self,
        audio_files: List[str],
        use_cache: bool = True,
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Separate vocals from multiple audio files in parallel

        Uses ThreadPoolExecutor for concurrent processing. Each file is processed
        independently, leveraging multiple CPU cores and GPU parallelism.

        Args:
            audio_files: List of audio file paths
            use_cache: Whether to use cached results if available
            max_workers: Maximum number of parallel workers (default: from config)
            progress_callback: Optional callback(completed, total, current_file)

        Returns:
            List of (vocals, instrumental) tuples, in same order as input files

        Raises:
            ValueError: If audio_files is empty
            SeparationError: If any separation fails (includes partial results)

        Example:
            >>> files = ['song1.mp3', 'song2.mp3', 'song3.mp3']
            >>> results = separator.separate_vocals_batch(files, max_workers=4)
            >>> for vocals, instrumental in results:
            ...     print(vocals.shape, instrumental.shape)
        """
        if not audio_files:
            raise ValueError("audio_files list cannot be empty")

        max_workers = max_workers or self.config.get('batch_max_workers', 4)
        results = [None] * len(audio_files)
        errors = {}

        logger.info(f"Starting batch separation of {len(audio_files)} files with {max_workers} workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.separate_vocals, audio_path, use_cache): i
                for i, audio_path in enumerate(audio_files)
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                audio_path = audio_files[index]

                try:
                    result = future.result()
                    results[index] = result
                    completed += 1

                    if progress_callback:
                        progress_callback(completed, len(audio_files), audio_path)

                    logger.debug(f"Completed {completed}/{len(audio_files)}: {audio_path}")

                except Exception as e:
                    errors[audio_path] = str(e)
                    completed += 1
                    logger.error(f"Failed to separate {audio_path}: {e}")

                    if progress_callback:
                        progress_callback(completed, len(audio_files), audio_path)

        # Check for errors
        if errors:
            error_msg = f"Batch separation completed with {len(errors)} error(s):\n"
            for path, err in errors.items():
                error_msg += f"  - {path}: {err}\n"

            # Filter out None results (failed files)
            successful_results = [r for r in results if r is not None]

            if not successful_results:
                raise SeparationError(f"All batch separations failed:\n{error_msg}")
            else:
                logger.warning(error_msg)
                # Return results with None for failed entries
                return results

        logger.info(f"Batch separation completed successfully: {len(audio_files)} files")
        return results

    def separate_vocals_batch_sequential(
        self,
        audio_files: List[str],
        use_cache: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Separate vocals from multiple audio files sequentially

        Processes files one at a time. Useful for debugging or when parallel
        processing causes resource contention.

        Args:
            audio_files: List of audio file paths
            use_cache: Whether to use cached results if available
            progress_callback: Optional callback(completed, total, current_file)

        Returns:
            List of (vocals, instrumental) tuples, in same order as input files

        Raises:
            ValueError: If audio_files is empty
            SeparationError: If any separation fails

        Example:
            >>> files = ['song1.mp3', 'song2.mp3']
            >>> results = separator.separate_vocals_batch_sequential(files)
        """
        if not audio_files:
            raise ValueError("audio_files list cannot be empty")

        results = []
        errors = {}

        logger.info(f"Starting sequential separation of {len(audio_files)} files")

        for i, audio_path in enumerate(audio_files):
            try:
                result = self.separate_vocals(audio_path, use_cache)
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, len(audio_files), audio_path)

                logger.debug(f"Completed {i + 1}/{len(audio_files)}: {audio_path}")

            except Exception as e:
                errors[audio_path] = str(e)
                results.append(None)
                logger.error(f"Failed to separate {audio_path}: {e}")

                if progress_callback:
                    progress_callback(i + 1, len(audio_files), audio_path)

        # Check for errors
        if errors:
            error_msg = f"Sequential separation completed with {len(errors)} error(s):\n"
            for path, err in errors.items():
                error_msg += f"  - {path}: {err}\n"

            successful_results = [r for r in results if r is not None]

            if not successful_results:
                raise SeparationError(f"All sequential separations failed:\n{error_msg}")
            else:
                logger.warning(error_msg)

        logger.info(f"Sequential separation completed: {len(audio_files)} files")
        return results

    def set_quality_preset(self, preset_name: str):
        """Set quality preset for separation

        Presets adjust processing parameters for different quality/speed tradeoffs.
        These presets align with config/voice_conversion_presets.yaml.

        Available presets:
            - 'fast': Fastest processing, lower quality (shifts=0, split=True, overlap=0.15)
            - 'balanced': Balanced quality/speed (shifts=1, split=True, overlap=0.25) [default]
            - 'quality': Best quality, slower (shifts=5, split=True, overlap=0.35)

        Args:
            preset_name: Name of preset ('fast', 'balanced', 'quality')

        Raises:
            ValueError: If preset_name is not recognized

        Example:
            >>> separator.set_quality_preset('quality')
            >>> vocals, instrumental, sr = separator.separate_vocals('song.mp3')
        """
        presets = {
            'fast': {
                'shifts': 0,
                'overlap': 0.15,  # Reduced from 0.25 per YAML
                'split': True,    # Changed from False per YAML
                'mixed_precision': True
            },
            'balanced': {
                'shifts': 1,
                'overlap': 0.25,
                'split': True,
                'mixed_precision': True
            },
            'quality': {
                'shifts': 5,      # Reduced from 10 to reasonable value per YAML
                'overlap': 0.35,  # Reduced from 0.5 per YAML
                'split': True,
                'mixed_precision': False
            }
        }

        preset_name = preset_name.lower()
        if preset_name not in presets:
            raise ValueError(
                f"Invalid preset '{preset_name}'. "
                f"Available presets: {', '.join(presets.keys())}"
            )

        # Apply preset settings
        preset_config = presets[preset_name]
        for key, value in preset_config.items():
            self.config[key] = value

        self.config['quality_preset'] = preset_name

        logger.info(f"Quality preset set to '{preset_name}': {preset_config}")

    def get_quality_preset_details(self) -> Dict[str, Any]:
        """Get detailed information about the active quality preset

        Returns:
            Dictionary with preset name and active parameters:
            {
                'preset': str,
                'model': str,
                'shifts': int,
                'overlap': float,
                'split': bool,
                'mixed_precision': bool
            }

        Example:
            >>> details = separator.get_quality_preset_details()
            >>> print(f"Preset: {details['preset']}, Shifts: {details['shifts']}")
        """
        preset_name = self.config.get('quality_preset', 'balanced')

        return {
            'preset': preset_name,
            'model': self.config.get('model', 'htdemucs'),
            'shifts': self.config.get('shifts', 1),
            'overlap': self.config.get('overlap', 0.25),
            'split': self.config.get('split', True),
            'mixed_precision': self.config.get('mixed_precision', True)
        }

    def get_current_quality_preset(self) -> str:
        """Get the current quality preset name

        Returns:
            Current preset name ('fast', 'balanced', 'quality', or 'custom')

        Example:
            >>> separator.get_current_quality_preset()
            'balanced'
        """
        return self.config.get('quality_preset', 'custom')

    def estimate_separation_time(
        self,
        audio_duration: float,
        preset: Optional[str] = None
    ) -> float:
        """Estimate separation time for given audio duration

        Provides rough time estimate based on preset and typical GPU performance.
        Actual time varies based on hardware, model, and system load.

        Estimates are based on typical NVIDIA RTX 3080 performance:
            - fast: ~0.1x realtime (10s audio -> 1s processing)
            - balanced: ~0.3x realtime (10s audio -> 3s processing)
            - quality: ~2.0x realtime (10s audio -> 20s processing)

        Args:
            audio_duration: Duration of audio in seconds
            preset: Optional preset name (uses current if not specified)

        Returns:
            Estimated processing time in seconds

        Raises:
            ValueError: If preset is invalid

        Example:
            >>> separator.estimate_separation_time(180.0, 'balanced')
            54.0  # ~54 seconds for 3-minute song
        """
        preset = preset or self.get_current_quality_preset()

        # Time multipliers based on typical GPU performance
        time_multipliers = {
            'fast': 0.1,      # 10x faster than realtime
            'balanced': 0.3,  # 3.3x faster than realtime
            'quality': 2.0    # 2x slower than realtime
        }

        if preset not in time_multipliers and preset != 'custom':
            raise ValueError(
                f"Invalid preset '{preset}'. "
                f"Available presets: {', '.join(time_multipliers.keys())}"
            )

        # For custom preset, estimate based on shifts parameter
        if preset == 'custom':
            shifts = self.config.get('shifts', 1)
            # Base time + additional time per shift
            multiplier = 0.2 + (shifts * 0.15)
        else:
            multiplier = time_multipliers[preset]

        # Apply CPU multiplier if not using GPU
        if self.device == 'cpu':
            multiplier *= 5.0  # CPU is roughly 5x slower

        estimated_time = audio_duration * multiplier

        logger.debug(
            f"Estimated separation time for {audio_duration:.1f}s audio "
            f"with preset '{preset}': {estimated_time:.1f}s"
        )

        return estimated_time
