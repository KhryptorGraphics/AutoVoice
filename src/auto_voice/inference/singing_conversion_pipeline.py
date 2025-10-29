"""Singing voice conversion pipeline for end-to-end song conversion.

This module provides the SingingConversionPipeline class that orchestrates
the complete workflow: vocal separation → pitch extraction → voice conversion → audio mixing.
"""

import os
import hashlib
import json
import threading
import time
from typing import Optional, Dict, Any, Callable, Tuple
import numpy as np

from ..audio.source_separator import VocalSeparator
from ..audio.pitch_extractor import SingingPitchExtractor
from ..models.singing_voice_converter import SingingVoiceConverter
from ..audio.mixer import AudioMixer
from ..audio.processor import AudioProcessor
from ..storage.voice_profiles import VoiceProfileStorage
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class SingingConversionError(Exception):
    """Base exception for singing voice conversion errors."""
    pass


class SeparationError(SingingConversionError):
    """Exception raised when vocal separation fails."""
    pass


class ConversionError(SingingConversionError):
    """Exception raised when voice conversion fails."""
    pass


class SingingConversionPipeline:
    """End-to-end singing voice conversion pipeline.

    This class orchestrates the complete workflow:
    1. Vocal separation (vocals vs instrumental)
    2. Pitch extraction (F0 contour)
    3. Voice conversion (target speaker embedding)
    4. Audio mixing (final output)

    Features:
    - Intermediate result caching
    - Progress tracking with callbacks
    - GPU acceleration
    - Error recovery
    - Thread-safe operations

    Attributes:
        config (Dict[str, Any]): Pipeline configuration
        vocal_separator (VocalSeparator): Vocal separation component
        pitch_extractor (SingingPitchExtractor): Pitch extraction component
        voice_converter (SingingVoiceConverter): Voice conversion model
        audio_mixer (AudioMixer): Audio mixing component
        audio_processor (AudioProcessor): Audio I/O processor
        voice_storage (VoiceProfileStorage): Voice profile storage
        device (str): Device for GPU/CPU processing
        cache_dir (str): Directory for caching results
        lock (threading.RLock): Thread safety lock

    Example:
        >>> pipeline = SingingConversionPipeline(device='cuda')
        >>>
        >>> def on_progress(percent, message):
        ...     print(f"[{percent:.1f}%] {message}")
        >>>
        >>> result = pipeline.convert_song(
        ...     song_path='original_song.mp3',
        ...     target_profile_id='user-profile-uuid',
        ...     progress_callback=on_progress
        ... )
        >>>
        >>> import soundfile as sf
        >>> sf.write('converted.wav', result['mixed_audio'], result['sample_rate'])
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        gpu_manager: Optional[Any] = None,
        preset: Optional[str] = None,
        voice_cloner: Optional[Any] = None,
        use_tensorrt: bool = False,
        tensorrt_precision: str = 'fp16'
    ):
        """Initialize SingingConversionPipeline.

        Args:
            config: Optional configuration dict
            device: Device string ('cuda', 'cpu', etc.)
            gpu_manager: Optional GPU manager instance
            preset: Preset name ('fast', 'balanced', 'quality', 'custom')
            voice_cloner: Optional VoiceCloner instance (for loading voice profiles)
            use_tensorrt: Enable TensorRT optimization (default: False)
            tensorrt_precision: TensorRT precision ('fp16' or 'fp32', default: 'fp16')
        """
        self.config = self._load_config(config, preset)
        self.device = device or self.config.get('device', 'cuda')
        self.gpu_manager = gpu_manager
        self.current_preset = preset or self.config.get('preset', 'balanced')
        self.voice_cloner = voice_cloner
        self.use_tensorrt = use_tensorrt
        self.tensorrt_precision = tensorrt_precision

        # Initialize cache directory
        self.cache_dir = os.path.expanduser(self.config['cache_dir'])
        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize components with preset configs
        logger.info("Initializing pipeline components...")

        # VocalSeparator config
        vocal_separator_config = self.config.get('vocal_separator', {})
        self.vocal_separator = VocalSeparator(
            device=device,
            gpu_manager=gpu_manager,
            config=vocal_separator_config if vocal_separator_config else None
        )
        logger.info("✓ VocalSeparator initialized")

        # SingingPitchExtractor config
        pitch_extractor_config = self.config.get('pitch_extractor', {})
        self.pitch_extractor = SingingPitchExtractor(
            device=device,
            gpu_manager=gpu_manager,
            config=pitch_extractor_config if pitch_extractor_config else None
        )
        logger.info("✓ SingingPitchExtractor initialized")

        # Initialize voice converter with preset-specific model config
        model_config = self.config.get('model_config', {})
        voice_converter_config = self.config.get('voice_converter', {})
        # Merge voice_converter preset config into model_config
        if voice_converter_config:
            model_config = {**model_config, **voice_converter_config}
        self.voice_converter = SingingVoiceConverter(model_config)
        # Guard model API calls with hasattr checks
        if device and hasattr(self.voice_converter, 'to'):
            converted_model = self.voice_converter.to(device)
            # Only reassign if to() returns a new instance
            if converted_model is not None:
                self.voice_converter = converted_model
        if hasattr(self.voice_converter, 'eval'):
            self.voice_converter.eval()
        if hasattr(self.voice_converter, 'prepare_for_inference'):
            self.voice_converter.prepare_for_inference()
        logger.info("✓ SingingVoiceConverter initialized")

        # Initialize TensorRT if requested
        if self.use_tensorrt:
            try:
                import tensorrt as trt
                logger.info(f"TensorRT requested with precision: {self.tensorrt_precision}")

                # Try to load existing TensorRT engines
                engine_dir = os.path.expanduser('~/.cache/autovoice/tensorrt_engines')
                if hasattr(self.voice_converter, 'load_tensorrt_engines'):
                    engines_loaded = self.voice_converter.load_tensorrt_engines(engine_dir=engine_dir)

                    if engines_loaded:
                        logger.info("✓ TensorRT engines loaded successfully")
                    else:
                        # Engines not found, try to build them
                        logger.info("TensorRT engines not found, attempting to build...")
                        onnx_dir = os.path.expanduser('~/.cache/autovoice/onnx_models')

                        # Export to ONNX if needed
                        if not os.path.exists(onnx_dir) or not os.listdir(onnx_dir):
                            logger.info("Exporting models to ONNX...")
                            if hasattr(self.voice_converter, 'export_to_onnx'):
                                self.voice_converter.export_to_onnx(export_dir=onnx_dir)

                        # Build TensorRT engines
                        if hasattr(self.voice_converter, 'create_tensorrt_engines'):
                            fp16 = (self.tensorrt_precision == 'fp16')
                            self.voice_converter.create_tensorrt_engines(
                                onnx_dir=onnx_dir,
                                engine_dir=engine_dir,
                                fp16=fp16
                            )
                            logger.info("✓ TensorRT engines built successfully")
                        else:
                            logger.warning("Voice converter does not support TensorRT engine creation")
                            self.use_tensorrt = False
                else:
                    logger.warning("Voice converter does not support TensorRT, falling back to PyTorch")
                    self.use_tensorrt = False

            except ImportError:
                logger.warning("TensorRT not available, falling back to PyTorch inference")
                self.use_tensorrt = False
            except Exception as e:
                logger.warning(f"TensorRT initialization failed: {e}, falling back to PyTorch")
                self.use_tensorrt = False

        # AudioMixer config
        mixer_config = self.config.get('mixer_config', {})
        audio_mixer_config = self.config.get('audio_mixer', {})
        # Merge audio_mixer preset config into mixer_config
        if audio_mixer_config:
            mixer_config = {**mixer_config, **audio_mixer_config}
        self.audio_mixer = AudioMixer(config=mixer_config if mixer_config else None)
        logger.info("✓ AudioMixer initialized")

        self.audio_processor = AudioProcessor()
        logger.info("✓ AudioProcessor initialized")

        # Use VoiceCloner if provided, otherwise fallback to direct storage with consistent path
        if self.voice_cloner is None:
            # Get storage_dir from config to match VoiceCloner's storage path
            # Accept both 'storage_dir' and 'voice_profile_dir' as aliases for backward compatibility
            storage_dir = (
                self.config.get('storage_dir') or
                self.config.get('voice_profile_dir') or
                os.path.expanduser('~/.cache/autovoice/voice_profiles/')
            )
            self.voice_storage = VoiceProfileStorage(storage_dir=storage_dir)
            logger.info(f"✓ VoiceProfileStorage initialized (direct access, storage_dir={storage_dir})")
        else:
            self.voice_storage = None
            logger.info("✓ Using VoiceCloner for profile access")

        # Initialize thread lock
        self.lock = threading.RLock()

        logger.info(f"SingingConversionPipeline initialized (device={self.device}, cache={self.cache_dir})")

    def _load_config(self, config: Optional[Dict[str, Any]], preset: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from multiple sources.

        Priority: constructor config > preset config > YAML file > environment variables > defaults

        Args:
            config: Optional constructor configuration
            preset: Optional preset name to load

        Returns:
            Merged configuration dictionary
        """
        import yaml

        # Default configuration
        default_config = {
            'cache_enabled': True,
            'cache_dir': '~/.cache/autovoice/converted/',
            'cache_size_limit_gb': 5.0,
            'cache_ttl_days': 7,
            'enable_progress_tracking': True,
            'save_intermediate_results': False,
            'intermediate_dir': '~/.cache/autovoice/intermediate/',
            'vocal_volume': 1.0,
            'instrumental_volume': 0.9,
            'output_sample_rate': 44100,
            'output_format': 'wav',
            'gpu_acceleration': True,
            'device': 'cuda',
            'fallback_on_mixing_error': True,
            'max_retries': 1,
            'preset': 'balanced',
            # Component-specific configs (will be populated from preset)
            'vocal_separator': {},
            'pitch_extractor': {},
            'voice_converter': {},
            'audio_mixer': {}
        }

        # Try to load from YAML
        try:
            config_path = os.path.join(
                os.path.dirname(__file__), '..', '..', '..', 'config', 'audio_config.yaml'
            )
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    if yaml_config and 'singing_conversion_pipeline' in yaml_config:
                        default_config.update(yaml_config['singing_conversion_pipeline'])
        except Exception as e:
            logger.warning(f"Could not load pipeline config from YAML: {e}")

        # Load preset configuration
        preset_config = self._load_preset_config(preset or default_config.get('preset', 'balanced'))
        if preset_config:
            # Merge preset's component configs
            default_config['preset'] = preset or default_config.get('preset', 'balanced')
            default_config['vocal_separator'].update(preset_config.get('vocal_separator', {}))
            default_config['pitch_extractor'].update(preset_config.get('pitch_extractor', {}))
            default_config['voice_converter'].update(preset_config.get('voice_converter', {}))
            default_config['audio_mixer'].update(preset_config.get('audio_mixer', {}))

        # Override with environment variables
        env_mappings = {
            'AUTOVOICE_PIPELINE_CACHE_ENABLED': ('cache_enabled', lambda x: x.lower() == 'true'),
            'AUTOVOICE_PIPELINE_CACHE_DIR': 'cache_dir',
            'AUTOVOICE_PIPELINE_DEVICE': 'device',
            'AUTOVOICE_PIPELINE_PRESET': 'preset',
        }

        for env_var, config_key in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                if isinstance(config_key, tuple):
                    key, converter = config_key
                    try:
                        default_config[key] = converter(env_value)
                    except Exception as e:
                        logger.warning(f"Invalid env var {env_var}={env_value}: {e}")
                else:
                    default_config[config_key] = env_value

        # Override with constructor config (user config takes precedence)
        if config:
            # Deep merge component configs if provided
            for component in ['vocal_separator', 'pitch_extractor', 'voice_converter', 'audio_mixer']:
                if component in config:
                    default_config[component].update(config[component])
            # Merge other top-level config
            default_config.update({k: v for k, v in config.items()
                                 if k not in ['vocal_separator', 'pitch_extractor', 'voice_converter', 'audio_mixer']})

        return default_config

    def _load_preset_config(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """Load preset configuration from voice_conversion_presets.yaml.

        Args:
            preset_name: Name of preset to load ('fast', 'balanced', 'quality', 'custom')

        Returns:
            Preset configuration dict or None if not found
        """
        try:
            import yaml
            preset_path = os.path.join(
                os.path.dirname(__file__), '..', '..', '..', 'config', 'voice_conversion_presets.yaml'
            )

            if not os.path.exists(preset_path):
                logger.warning(f"Preset file not found: {preset_path}, using defaults")
                return None

            with open(preset_path, 'r') as f:
                preset_data = yaml.safe_load(f)

            if not preset_data or 'presets' not in preset_data:
                logger.warning("Invalid preset file format, using defaults")
                return None

            presets = preset_data['presets']

            # Validate preset name
            if preset_name not in presets:
                available = ', '.join(presets.keys())
                logger.warning(f"Preset '{preset_name}' not found. Available presets: {available}. Using default.")
                default_preset = preset_data.get('default_preset', 'balanced')
                preset_name = default_preset if default_preset in presets else 'balanced'

            preset_config = presets[preset_name]
            logger.info(f"Loaded preset '{preset_name}': {preset_config.get('description', 'No description')}")

            return preset_config

        except Exception as e:
            logger.warning(f"Failed to load preset '{preset_name}': {e}, using defaults")
            return None

    def convert_song(
        self,
        song_path: str,
        target_profile_id: str,
        vocal_volume: float = 1.0,
        instrumental_volume: float = 0.9,
        pitch_shift: float = 0.0,
        preset: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        profiling_callback: Optional[Callable[[str, float], None]] = None,
        return_stems: bool = False
    ) -> Dict[str, Any]:
        """Convert singing voice in a song to target voice.

        Primary method for end-to-end conversion. Performs:
        1. Vocal separation (0-25%)
        2. F0 extraction (25-40%)
        3. Voice conversion (40-80%)
        4. Audio mixing (80-100%)

        Progress callback stages: 'source_separation', 'pitch_extraction', 'voice_conversion', 'audio_mixing'

        Args:
            song_path: Path to input song file (MP3/WAV/FLAC)
            target_profile_id: Voice profile ID for target speaker
            vocal_volume: Volume multiplier for vocals (0.0-2.0, default: 1.0)
            instrumental_volume: Volume multiplier for instrumental (0.0-2.0, default: 0.9)
            pitch_shift: Pitch shift in semitones (default: 0.0, range: -12 to +12)
            preset: Quality preset name ('fast', 'balanced', 'quality', default: None uses current)
            progress_callback: Optional callback(progress: float, stage: str) - progress 0.0-100.0 first, then stage name
            profiling_callback: Optional callback(stage_name: str, elapsed_ms: float) - called at completion of each stage with timing
            return_stems: Return separated vocals/instrumental in addition to mix

        Returns:
            Dictionary with conversion results:
            {
                'mixed_audio': np.ndarray,
                'sample_rate': int,
                'duration': float,
                'vocals': np.ndarray (if return_stems=True),
                'instrumental': np.ndarray (if return_stems=True),
                'f0_contour': np.ndarray,
                'metadata': {
                    'target_profile_id': str,
                    'vocal_volume': float,
                    'instrumental_volume': float,
                    'pitch_shift': float,
                    'preset': str,
                    'f0_stats': dict
                }
            }

        Raises:
            SingingConversionError: If conversion fails
            FileNotFoundError: If song file or profile not found
        """
        with self.lock:
            start_time = time.time()

            # Apply preset if specified
            if preset is not None:
                self.set_preset(preset, clear_cache=False)

            # Initial progress callback
            if progress_callback:
                progress_callback(0.0, 'source_separation')

            # Validate inputs
            if not os.path.exists(song_path):
                raise FileNotFoundError(f"Song file not found: {song_path}")

            # Check cache (bypass if return_stems=True since cache doesn't store stems)
            if self.config['cache_enabled'] and not return_stems:
                cache_key = self._get_cache_key(
                    song_path, target_profile_id,
                    {
                        'vocal_volume': vocal_volume,
                        'instrumental_volume': instrumental_volume,
                        'pitch_shift': pitch_shift,
                        'preset': preset or self.current_preset
                    }
                )
                cached_result = self._load_from_cache(cache_key)
                if cached_result:
                    logger.info(f"Cache hit for conversion: {cache_key}")
                    if progress_callback:
                        progress_callback(100.0, 'audio_mixing')
                    return cached_result
            elif self.config['cache_enabled'] and return_stems:
                logger.info("Bypassing cache because return_stems=True (stems not cached)")

            logger.info(f"Starting song conversion: {song_path} -> profile {target_profile_id}")

            # Start timing for profiling
            import time
            t0 = time.time()
            stage_start = t0

            # Load original audio to get sample rate
            _, original_sr = self.audio_processor.load_audio(song_path, return_sr=True)
            sample_rate = original_sr

            # Load target speaker embedding
            try:
                # Use VoiceCloner if available, otherwise fallback to direct storage
                if self.voice_cloner is not None:
                    profile = self.voice_cloner.load_voice_profile(target_profile_id)
                else:
                    profile = self.voice_storage.load_profile(target_profile_id)

                if profile is None:
                    raise FileNotFoundError(f"Voice profile not found: {target_profile_id}")
                target_embedding = profile.get('embedding')
                if target_embedding is None:
                    raise ValueError(f"Profile {target_profile_id} has no embedding")
                target_embedding = np.array(target_embedding)
                logger.info(f"Loaded target embedding: shape {target_embedding.shape}")
            except FileNotFoundError:
                # Backward-compatible fallback: try loading .npz file directly
                storage_dir = (
                    self.config.get('storage_dir') or
                    self.config.get('voice_profile_dir') or
                    os.path.expanduser('~/.cache/autovoice/voice_profiles/')
                )
                npz_path = Path(storage_dir) / f'{target_profile_id}.npz'
                if npz_path.exists():
                    logger.info(f"Loading legacy .npz profile: {npz_path}")
                    try:
                        npz_data = np.load(str(npz_path))
                        target_embedding = npz_data['embedding']
                        logger.info(f"Loaded target embedding from .npz: shape {target_embedding.shape}")
                    except Exception as npz_error:
                        logger.error(f"Failed to load .npz profile: {npz_error}")
                        raise FileNotFoundError(f"Voice profile not found: {target_profile_id}")
                else:
                    # Re-raise FileNotFoundError to allow API to map it to 404
                    logger.error(f"Voice profile not found: {target_profile_id}")
                    raise
            except Exception as e:
                # Wrap other exceptions as SingingConversionError
                logger.error(f"Failed to load profile: {e}", exc_info=True)
                raise SingingConversionError(f"Failed to load voice profile: {e}")

            # Stage 1: Vocal Separation (0-25%)
            try:
                if progress_callback:
                    progress_callback(0.0, 'source_separation')

                # VocalSeparator returns (vocals, instrumental)
                vocals, instrumental = self.vocal_separator.separate_vocals(song_path)

                if progress_callback:
                    progress_callback(25.0, 'source_separation')

                logger.info(f"Separated vocals: {vocals.shape}, instrumental: {instrumental.shape} at {sample_rate} Hz")

                # Profiling callback for separation stage
                if profiling_callback:
                    t_sep = (time.time() - stage_start) * 1000  # Convert to ms
                    try:
                        profiling_callback('separation', t_sep)
                    except Exception as e:
                        logger.warning(f"Profiling callback error: {e}")
                    stage_start = time.time()

                # Optional GPU memory cleanup after separation
                if self.config.get('enable_memory_cleanup', False):
                    try:
                        import torch
                        if torch.cuda.is_available():
                            mem_before = torch.cuda.memory_allocated() / (1024**2)
                            torch.cuda.empty_cache()
                            mem_after = torch.cuda.memory_allocated() / (1024**2)
                            logger.debug(f"GPU memory cleanup: {mem_before:.1f}MB -> {mem_after:.1f}MB")
                    except Exception as cleanup_error:
                        logger.debug(f"GPU memory cleanup skipped: {cleanup_error}")

            except Exception as e:
                logger.error(f"Vocal separation failed: {e}", exc_info=True)
                raise SeparationError(f"Failed to separate vocals: {e}")

            # Stage 2: F0 Extraction (25-40%)
            try:
                if progress_callback:
                    progress_callback(25.0, 'pitch_extraction')

                f0_data = self.pitch_extractor.extract_f0_contour(vocals, sample_rate)

                if progress_callback:
                    progress_callback(40.0, 'pitch_extraction')

                # Get pitch statistics
                f0_stats = self.pitch_extractor.get_pitch_statistics(f0_data)
                logger.info(f"Extracted F0: min={f0_stats['min_f0']:.1f}Hz, max={f0_stats['max_f0']:.1f}Hz")

                # Profiling callback for f0_extraction stage
                if profiling_callback:
                    t_f0 = (time.time() - stage_start) * 1000  # Convert to ms
                    try:
                        profiling_callback('f0_extraction', t_f0)
                    except Exception as e:
                        logger.warning(f"Profiling callback error: {e}")
                    stage_start = time.time()

                # Optional GPU memory cleanup after pitch extraction
                if self.config.get('enable_memory_cleanup', False):
                    try:
                        import torch
                        if torch.cuda.is_available():
                            mem_before = torch.cuda.memory_allocated() / (1024**2)
                            torch.cuda.empty_cache()
                            mem_after = torch.cuda.memory_allocated() / (1024**2)
                            logger.debug(f"GPU memory cleanup: {mem_before:.1f}MB -> {mem_after:.1f}MB")
                    except Exception as cleanup_error:
                        logger.debug(f"GPU memory cleanup skipped: {cleanup_error}")

            except Exception as e:
                logger.warning(f"Pitch extraction failed: {e}, continuing without F0 guidance")
                f0_data = None
                f0_stats = {}

            # Stage 3: Voice Conversion (40-80%)
            try:
                if progress_callback:
                    progress_callback(40.0, 'voice_conversion')

                # Prepare F0 for converter
                source_f0 = f0_data['f0'] if f0_data else None

                # Convert vocals with pitch shift using TensorRT if enabled
                if self.use_tensorrt and hasattr(self.voice_converter, 'convert_with_tensorrt'):
                    logger.info("Using TensorRT-accelerated conversion")
                    converted_vocals = self.voice_converter.convert_with_tensorrt(
                        vocals,
                        target_embedding,
                        source_f0=source_f0,
                        source_sample_rate=sample_rate,
                        output_sample_rate=self.config['output_sample_rate'],
                        pitch_shift_semitones=pitch_shift
                    )
                    # If convert_with_tensorrt returns tuple (audio, timing_info), extract audio
                    if isinstance(converted_vocals, tuple):
                        converted_vocals = converted_vocals[0]
                else:
                    converted_vocals = self.voice_converter.convert(
                        vocals,
                        target_embedding,
                        source_f0=source_f0,
                        source_sample_rate=sample_rate,
                        output_sample_rate=self.config['output_sample_rate'],
                        pitch_shift_semitones=pitch_shift
                    )

                if progress_callback:
                    progress_callback(80.0, 'voice_conversion')

                logger.info(f"Converted vocals: {converted_vocals.shape}")

                # Profiling callback for conversion stage
                if profiling_callback:
                    t_conv = (time.time() - stage_start) * 1000  # Convert to ms
                    try:
                        profiling_callback('conversion', t_conv)
                    except Exception as e:
                        logger.warning(f"Profiling callback error: {e}")
                    stage_start = time.time()

                # Optional GPU memory cleanup after voice conversion
                if self.config.get('enable_memory_cleanup', False):
                    try:
                        import torch
                        if torch.cuda.is_available():
                            mem_before = torch.cuda.memory_allocated() / (1024**2)
                            torch.cuda.empty_cache()
                            mem_after = torch.cuda.memory_allocated() / (1024**2)
                            logger.debug(f"GPU memory cleanup: {mem_before:.1f}MB -> {mem_after:.1f}MB")
                    except Exception as cleanup_error:
                        logger.debug(f"GPU memory cleanup skipped: {cleanup_error}")

            except Exception as e:
                logger.error(f"Voice conversion failed: {e}", exc_info=True)
                raise ConversionError(f"Failed to convert vocals: {e}")

            # Stage 4: Audio Mixing (80-100%)
            try:
                if progress_callback:
                    progress_callback(80.0, 'audio_mixing')

                # Resample instrumental if needed using AudioMixer's resampling utility
                if sample_rate != self.config['output_sample_rate']:
                    logger.info(f"Resampling instrumental from {sample_rate} Hz to {self.config['output_sample_rate']} Hz")

                    # Try torchaudio first, fallback to librosa
                    try:
                        # Use AudioMixer's _resample_if_needed for consistency
                        instrumental = self.audio_mixer._resample_if_needed(
                            instrumental,
                            sample_rate,
                            self.config['output_sample_rate']
                        )
                    except Exception as resample_error:
                        logger.warning(f"AudioMixer resampling failed: {resample_error}, trying direct methods")

                        # Fallback: try torchaudio directly
                        try:
                            import torchaudio.transforms as T
                            import torch

                            # Convert to tensor if needed
                            if isinstance(instrumental, np.ndarray):
                                inst_tensor = torch.from_numpy(instrumental.astype(np.float32))
                            else:
                                inst_tensor = instrumental

                            # Handle shape
                            if inst_tensor.ndim == 1:
                                inst_tensor = inst_tensor.unsqueeze(0)
                            elif inst_tensor.ndim == 2 and inst_tensor.shape[0] > 2:
                                # Multi-channel, convert to mono then duplicate
                                inst_tensor = inst_tensor.mean(dim=0, keepdim=True)

                            resampler = T.Resample(orig_freq=sample_rate, new_freq=self.config['output_sample_rate'])
                            resampled = resampler(inst_tensor)

                            # Convert back to numpy
                            if resampled.ndim > 1:
                                instrumental = resampled.squeeze(0).numpy() if resampled.shape[0] == 1 else resampled.numpy()
                            else:
                                instrumental = resampled.numpy()

                            logger.debug("Resampled using torchaudio")

                        except Exception as torch_error:
                            logger.warning(f"torchaudio resampling failed: {torch_error}, trying librosa")

                            # Final fallback: librosa
                            try:
                                import librosa

                                # Ensure numpy array
                                if hasattr(instrumental, 'detach'):
                                    instrumental = instrumental.detach().cpu().numpy()

                                # Resample
                                if instrumental.ndim == 2:
                                    # Multi-channel: resample each channel
                                    instrumental = np.stack([
                                        librosa.resample(instrumental[i], orig_sr=sample_rate, target_sr=self.config['output_sample_rate'])
                                        for i in range(instrumental.shape[0])
                                    ])
                                else:
                                    instrumental = librosa.resample(
                                        instrumental,
                                        orig_sr=sample_rate,
                                        target_sr=self.config['output_sample_rate']
                                    )

                                logger.debug("Resampled using librosa")

                            except Exception as librosa_error:
                                logger.error(f"All resampling methods failed: {librosa_error}")
                                # If all resampling fails, continue without resampling
                                logger.warning("Continuing without resampling - mixing may fail or produce unexpected results")

                # Mix converted vocals with instrumental
                mixed_audio, final_sample_rate = self.audio_mixer.mix(
                    converted_vocals,
                    instrumental,
                    vocal_volume=vocal_volume,
                    instrumental_volume=instrumental_volume,
                    sample_rate=self.config['output_sample_rate']
                )

                if progress_callback:
                    progress_callback(100.0, 'audio_mixing')

                logger.info(f"Final mix: {mixed_audio.shape} at {final_sample_rate}Hz")

                # Profiling callback for mixing stage
                if profiling_callback:
                    t_mix = (time.time() - stage_start) * 1000  # Convert to ms
                    try:
                        profiling_callback('mixing', t_mix)
                    except Exception as e:
                        logger.warning(f"Profiling callback error: {e}")

                # Optional GPU memory cleanup after mixing
                if self.config.get('enable_memory_cleanup', False):
                    try:
                        import torch
                        if torch.cuda.is_available():
                            mem_before = torch.cuda.memory_allocated() / (1024**2)
                            torch.cuda.empty_cache()
                            mem_after = torch.cuda.memory_allocated() / (1024**2)
                            logger.debug(f"GPU memory cleanup: {mem_before:.1f}MB -> {mem_after:.1f}MB")
                    except Exception as cleanup_error:
                        logger.debug(f"GPU memory cleanup skipped: {cleanup_error}")

            except Exception as e:
                logger.error(f"Audio mixing failed: {e}", exc_info=True)
                if self.config['fallback_on_mixing_error']:
                    logger.warning("Falling back to dry vocals")
                    mixed_audio = converted_vocals
                    final_sample_rate = self.config['output_sample_rate']
                else:
                    raise SingingConversionError(f"Failed to mix audio: {e}")

            # Build result dictionary
            # Duration is always computed from first dimension (time axis) for both mono (T,) and stereo (T, 2)
            duration = mixed_audio.shape[0] / final_sample_rate

            # Convert f0_contour to list for JSON serialization if present
            f0_contour = None
            if f0_data is not None and 'f0' in f0_data:
                import numpy as np
                f0_array = f0_data['f0']
                # Convert to list for serialization
                f0_contour = f0_array.tolist() if isinstance(f0_array, np.ndarray) else f0_array

            result = {
                'mixed_audio': mixed_audio,
                'sample_rate': final_sample_rate,
                'duration': duration,
                'vocals': converted_vocals if return_stems else None,
                'instrumental': instrumental if return_stems else None,
                'f0_contour': f0_contour,
                'metadata': {
                    'target_profile_id': target_profile_id,
                    'vocal_volume': vocal_volume,
                    'instrumental_volume': instrumental_volume,
                    'pitch_shift': pitch_shift,
                    'preset': preset or self.current_preset,
                    'f0_stats': f0_stats,
                    'processing_time': time.time() - t0,
                    'input_sample_rate': sample_rate,
                    'output_sample_rate': final_sample_rate,
                    'tensorrt': {
                        'enabled': self.use_tensorrt and hasattr(self.voice_converter, 'trt_enabled') and self.voice_converter.trt_enabled,
                        'precision': self.tensorrt_precision if self.use_tensorrt else None
                    }
                }
            }

            # Save to cache (skip if return_stems=True since stems are not cached)
            if self.config['cache_enabled'] and not return_stems:
                try:
                    # Ensure cache_key is available (may not be computed if cache was bypassed earlier)
                    if 'cache_key' not in locals():
                        cache_key = self._get_cache_key(
                            song_path, target_profile_id,
                            {'vocal_volume': vocal_volume, 'instrumental_volume': instrumental_volume}
                        )
                    self._save_to_cache(cache_key, result)
                except Exception as e:
                    logger.warning(f"Failed to save to cache: {e}")
            elif self.config['cache_enabled'] and return_stems:
                logger.debug("Skipping cache save: stems are not cached (return_stems=True)")

            logger.info(f"Song conversion complete in {time.time() - t0:.2f}s")

            # Profiling callback for total time
            if profiling_callback:
                total_time = (time.time() - t0) * 1000  # Convert to ms
                try:
                    profiling_callback('total', total_time)
                except Exception as e:
                    logger.warning(f"Profiling callback error: {e}")

            return result

    def convert_vocals_only(
        self,
        vocals_path: str,
        target_profile_id: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """Convert isolated vocals (no separation needed).

        Simplified method for converting already-separated vocals.

        Progress callback stages: 'pitch_extraction', 'voice_conversion'

        Args:
            vocals_path: Path to vocals file
            target_profile_id: Voice profile ID for target speaker
            progress_callback: Optional progress callback

        Returns:
            Dictionary with conversion results (vocals only, no mix)

        Raises:
            SingingConversionError: If conversion fails
        """
        with self.lock:
            try:
                # Load vocals with sample rate
                vocals, sample_rate = self.audio_processor.load_audio(vocals_path, return_sr=True)

                # Load target embedding
                # Use VoiceCloner if available, otherwise fallback to direct storage
                if self.voice_cloner is not None:
                    profile = self.voice_cloner.load_voice_profile(target_profile_id)
                else:
                    profile = self.voice_storage.load_profile(target_profile_id)

                if profile is None:
                    raise FileNotFoundError(f"Voice profile not found: {target_profile_id}")
                target_embedding = np.array(profile['embedding'])

                # Extract pitch with error handling
                if progress_callback:
                    progress_callback(0.0, 'pitch_extraction')

                try:
                    f0_data = self.pitch_extractor.extract_f0_contour(vocals, sample_rate)
                    f0_stats = self.pitch_extractor.get_pitch_statistics(f0_data)

                    if progress_callback:
                        progress_callback(33.0, 'pitch_extraction')
                except Exception as e:
                    logger.warning(f"Pitch extraction failed: {e}, continuing without F0 guidance")
                    f0_data = None
                    f0_stats = {}

                    if progress_callback:
                        progress_callback(33.0, 'pitch_extraction')

                # Convert vocals
                if progress_callback:
                    progress_callback(33.0, 'voice_conversion')

                converted_vocals = self.voice_converter.convert(
                    vocals,
                    target_embedding,
                    source_f0=f0_data['f0'] if f0_data else None,
                    source_sample_rate=sample_rate,
                    output_sample_rate=self.config['output_sample_rate']
                )

                if progress_callback:
                    progress_callback(100.0, 'voice_conversion')

                return {
                    'vocals': converted_vocals,
                    'sample_rate': self.config['output_sample_rate'],
                    'metadata': {
                        'target_profile_id': target_profile_id,
                        'f0_stats': f0_stats
                    }
                }

            except Exception as e:
                logger.error(f"Vocals-only conversion failed: {e}", exc_info=True)
                raise SingingConversionError(f"Failed to convert vocals: {e}")

    def _get_cache_key(self, song_path: str, target_profile_id: str, params: Dict) -> str:
        """Generate unique cache key for conversion.

        Args:
            song_path: Path to song file
            target_profile_id: Target profile ID
            params: Conversion parameters (vocal_volume, instrumental_volume, pitch_shift, preset, etc.)

        Returns:
            Cache key string (hex digest)
        """
        # Get file modification time
        mtime = os.path.getmtime(song_path) if os.path.exists(song_path) else 0

        # Create hash from song path, mtime, profile, and all conversion-affecting params
        # Include all parameters that affect the conversion output
        import json

        # Build stable key with all conversion parameters
        key_parts = [
            song_path,
            str(mtime),
            target_profile_id,
            str(params.get('vocal_volume', 1.0)),
            str(params.get('instrumental_volume', 0.9)),
            str(params.get('pitch_shift', 0.0)),
            str(params.get('preset', self.current_preset)),
        ]

        # Include converter options if present
        converter_options = params.get('converter_options', {})
        if converter_options:
            # Sort keys for stable serialization
            key_parts.append(json.dumps(converter_options, sort_keys=True))

        key_string = ':'.join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load cached conversion result.

        IMPORTANT: Cache does NOT store stems. Cached results will always have
        vocals=None and instrumental=None, regardless of the original return_stems setting.
        Only the mixed audio and metadata are cached.

        Args:
            cache_key: Cache key

        Returns:
            Cached result dict or None if cache miss. Result structure:
            {
                'mixed_audio': np.ndarray,
                'sample_rate': int,
                'duration': float,
                'vocals': None,  # Never returned from cache
                'instrumental': None,  # Never returned from cache
                'metadata': {
                    'target_profile_id': str,
                    'vocal_volume': float,
                    'instrumental_volume': float,
                    'f0_stats': dict,
                    'from_cache': True  # Indicates this was loaded from cache
                }
            }
        """
        try:
            audio_path = os.path.join(self.cache_dir, f"{cache_key}_mixed.npy")
            metadata_path = os.path.join(self.cache_dir, f"{cache_key}_metadata.json")

            if not os.path.exists(audio_path) or not os.path.exists(metadata_path):
                return None

            # Check TTL
            if self.config['cache_ttl_days'] > 0:
                age_days = (time.time() - os.path.getmtime(audio_path)) / (24 * 3600)
                if age_days > self.config['cache_ttl_days']:
                    logger.info(f"Cache entry expired: {cache_key}")
                    return None

            # Load audio and metadata
            mixed_audio = np.load(audio_path)
            with open(metadata_path, 'r') as f:
                cached_metadata = json.load(f)

            # Extract metadata consistently - use nested metadata if it exists
            result_metadata = cached_metadata.get('metadata', {})

            # Add cache indicator
            result_metadata['from_cache'] = True
            result_metadata['cached_at'] = cached_metadata.get('cached_at')

            return {
                'mixed_audio': mixed_audio,
                'sample_rate': cached_metadata.get('sample_rate', self.config['output_sample_rate']),
                'duration': cached_metadata.get('duration', 0.0),
                'vocals': None,  # Cache never returns stems
                'instrumental': None,  # Cache never returns stems
                'metadata': result_metadata
            }

        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            return None

    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]):
        """Save conversion result to cache.

        Args:
            cache_key: Cache key
            result: Conversion result dictionary
        """
        try:
            audio_path = os.path.join(self.cache_dir, f"{cache_key}_mixed.npy")
            metadata_path = os.path.join(self.cache_dir, f"{cache_key}_metadata.json")

            # Save audio
            np.save(audio_path, result['mixed_audio'])

            # Save metadata
            metadata = {
                'sample_rate': result['sample_rate'],
                'duration': result['duration'],
                'metadata': result['metadata'],
                'cached_at': time.time()
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Enforce cache size limits (simple LRU)
            self._enforce_cache_limits()

        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")

    def _enforce_cache_limits(self):
        """Enforce cache size limits with LRU eviction."""
        try:
            # Get all cache files
            cache_files = []
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('_mixed.npy'):
                    filepath = os.path.join(self.cache_dir, filename)
                    cache_files.append((filepath, os.path.getmtime(filepath), os.path.getsize(filepath)))

            # Sort by modification time (oldest first)
            cache_files.sort(key=lambda x: x[1])

            # Calculate total size
            total_size_gb = sum(size for _, _, size in cache_files) / (1024 ** 3)

            # Evict oldest files if over limit
            if total_size_gb > self.config['cache_size_limit_gb']:
                logger.info(f"Cache size {total_size_gb:.2f}GB exceeds limit, evicting oldest entries")
                for filepath, _, size in cache_files:
                    try:
                        # Remove audio and metadata files
                        os.unlink(filepath)
                        metadata_path = filepath.replace('_mixed.npy', '_metadata.json')
                        if os.path.exists(metadata_path):
                            os.unlink(metadata_path)
                        total_size_gb -= size / (1024 ** 3)
                        if total_size_gb <= self.config['cache_size_limit_gb'] * 0.8:
                            break
                    except Exception as e:
                        logger.warning(f"Failed to evict cache file: {e}")

        except Exception as e:
            logger.warning(f"Failed to enforce cache limits: {e}")

    def clear_cache(self, max_age_days: Optional[int] = None):
        """Clear cached conversion results.

        Args:
            max_age_days: Only clear files older than this many days.
                If None, clear all cache files.
        """
        try:
            cleared = 0
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('_mixed.npy') or filename.endswith('_metadata.json'):
                    filepath = os.path.join(self.cache_dir, filename)
                    if max_age_days is None:
                        os.unlink(filepath)
                        cleared += 1
                    else:
                        age_days = (time.time() - os.path.getmtime(filepath)) / (24 * 3600)
                        if age_days > max_age_days:
                            os.unlink(filepath)
                            cleared += 1

            logger.info(f"Cleared {cleared} cache files")

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}", exc_info=True)

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache information:
            {
                'cache_dir': str,
                'total_size_mb': float,
                'num_conversions': int,
                'oldest_entry_age_days': float
            }
        """
        try:
            cache_files = []
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('_mixed.npy'):
                    filepath = os.path.join(self.cache_dir, filename)
                    cache_files.append((filepath, os.path.getmtime(filepath), os.path.getsize(filepath)))

            if not cache_files:
                return {
                    'cache_dir': self.cache_dir,
                    'total_size_mb': 0.0,
                    'num_conversions': 0,
                    'oldest_entry_age_days': 0.0
                }

            total_size = sum(size for _, _, size in cache_files)
            oldest_mtime = min(mtime for _, mtime, _ in cache_files)
            oldest_age_days = (time.time() - oldest_mtime) / (24 * 3600)

            return {
                'cache_dir': self.cache_dir,
                'total_size_mb': total_size / (1024 ** 2),
                'num_conversions': len(cache_files),
                'oldest_entry_age_days': oldest_age_days
            }

        except Exception as e:
            logger.error(f"Failed to get cache info: {e}", exc_info=True)
            return {
                'cache_dir': self.cache_dir,
                'total_size_mb': 0.0,
                'num_conversions': 0,
                'oldest_entry_age_days': 0.0
            }

    def set_preset(self, preset_name: str, clear_cache: bool = True):
        """Switch to a different preset at runtime.

        This method reloads the configuration with the new preset and reinitializes
        all pipeline components. Optionally clears the cache since results from
        different presets are not compatible.

        Args:
            preset_name: Name of preset to switch to ('fast', 'balanced', 'quality', 'custom')
            clear_cache: Whether to clear the conversion cache (default: True)

        Raises:
            ValueError: If preset name is invalid

        Example:
            >>> pipeline = SingingConversionPipeline(preset='balanced')
            >>> # Switch to quality preset for final output
            >>> pipeline.set_preset('quality')
        """
        with self.lock:
            logger.info(f"Switching preset from '{self.current_preset}' to '{preset_name}'")

            # Validate preset exists
            preset_config = self._load_preset_config(preset_name)
            if preset_config is None:
                raise ValueError(f"Invalid preset name: {preset_name}")

            # Clear cache if requested (recommended since presets produce different results)
            if clear_cache:
                logger.info("Clearing cache due to preset change")
                self.clear_cache()

            # Reload config with new preset
            self.config = self._load_config(None, preset_name)
            self.current_preset = preset_name

            # Reinitialize components with new configs
            logger.info("Reinitializing pipeline components with new preset...")

            # VocalSeparator
            vocal_separator_config = self.config.get('vocal_separator', {})
            self.vocal_separator = VocalSeparator(
                device=self.device,
                gpu_manager=self.gpu_manager,
                config=vocal_separator_config if vocal_separator_config else None
            )
            logger.info("✓ VocalSeparator reinitialized")

            # SingingPitchExtractor
            pitch_extractor_config = self.config.get('pitch_extractor', {})
            self.pitch_extractor = SingingPitchExtractor(
                device=self.device,
                gpu_manager=self.gpu_manager,
                config=pitch_extractor_config if pitch_extractor_config else None
            )
            logger.info("✓ SingingPitchExtractor reinitialized")

            # SingingVoiceConverter
            model_config = self.config.get('model_config', {})
            voice_converter_config = self.config.get('voice_converter', {})
            if voice_converter_config:
                model_config = {**model_config, **voice_converter_config}
            self.voice_converter = SingingVoiceConverter(model_config)
            if self.device:
                self.voice_converter = self.voice_converter.to(self.device)
            self.voice_converter.eval()
            self.voice_converter.prepare_for_inference()
            logger.info("✓ SingingVoiceConverter reinitialized")

            # AudioMixer
            mixer_config = self.config.get('mixer_config', {})
            audio_mixer_config = self.config.get('audio_mixer', {})
            if audio_mixer_config:
                mixer_config = {**mixer_config, **audio_mixer_config}
            self.audio_mixer = AudioMixer(config=mixer_config if mixer_config else None)
            logger.info("✓ AudioMixer reinitialized")

            logger.info(f"Successfully switched to preset '{preset_name}'")

    def get_current_preset(self) -> Dict[str, Any]:
        """Get information about the currently active preset.

        Returns:
            Dictionary with preset information:
            {
                'name': str,  # Current preset name
                'description': str,  # Preset description
                'config': dict  # Full preset configuration
            }

        Example:
            >>> pipeline = SingingConversionPipeline(preset='balanced')
            >>> info = pipeline.get_current_preset()
            >>> print(f"Using preset: {info['name']}")
            Using preset: balanced
        """
        preset_config = self._load_preset_config(self.current_preset)

        return {
            'name': self.current_preset,
            'description': preset_config.get('description', 'No description') if preset_config else 'No description',
            'config': {
                'vocal_separator': self.config.get('vocal_separator', {}),
                'pitch_extractor': self.config.get('pitch_extractor', {}),
                'voice_converter': self.config.get('voice_converter', {}),
                'audio_mixer': self.config.get('audio_mixer', {})
            }
        }
