"""Voice cloning system for creating user voice profiles from audio samples"""

from __future__ import annotations
import logging
import os
import threading
import uuid
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

logger = logging.getLogger(__name__)


class VoiceCloningError(Exception):
    """Base exception for voice cloning errors"""
    pass


class InvalidAudioError(VoiceCloningError):
    """Exception raised when audio validation fails

    Attributes:
        error_code (str): Machine-readable error code
        details (dict): Additional error details
    """
    def __init__(self, message: str, error_code: str = 'invalid_audio', details: Optional[Dict] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}


class VoiceCloner:
    """High-level interface for voice cloning - extract speaker embeddings and create voice profiles

    This class orchestrates the voice cloning pipeline:
    1. Load and validate audio (5-60 seconds recommended)
    2. Extract 256-dim speaker embedding using SpeakerEncoder
    3. Extract vocal range using SingingPitchExtractor (optional)
    4. Extract timbre features from mel-spectrogram (optional)
    5. Create and save voice profile with VoiceProfileStorage

    The resulting voice profiles can be used for:
    - Voice conversion and voice cloning
    - Speaker verification
    - Personalized TTS

    Features:
        - Audio validation (duration, sample rate, quality)
        - Multi-feature extraction (embedding + vocal range + timbre)
        - Persistent storage with JSON metadata + NumPy embeddings
        - Profile management (create, load, list, delete, compare)
        - GPU acceleration for feature extraction

    Example:
        >>> cloner = VoiceCloner(device='cuda', gpu_manager=gpu_manager)
        >>> profile = cloner.create_voice_profile(
        ...     audio='user_voice.wav',
        ...     user_id='user123'
        ... )
        >>> print(f"Profile ID: {profile['profile_id']}")
        >>> print(f"Vocal range: {profile['vocal_range']['min_f0']:.1f} - {profile['vocal_range']['max_f0']:.1f} Hz")

    Attributes:
        config (Dict[str, Any]): Configuration dictionary
        device (str): Device for processing ('cuda', 'cpu')
        gpu_manager: Optional GPUManager for GPU acceleration
        speaker_encoder: SpeakerEncoder for embedding extraction
        audio_processor: AudioProcessor for audio I/O
        pitch_extractor: Optional SingingPitchExtractor for vocal range
        storage: VoiceProfileStorage for profile persistence
        lock (threading.RLock): Thread safety lock
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        gpu_manager: Optional[Any] = None
    ):
        """Initialize VoiceCloner with configuration and components

        Args:
            config: Optional configuration dictionary with voice cloning parameters
            device: Optional device string ('cuda', 'cpu', 'cuda:0', etc.)
            gpu_manager: Optional GPUManager instance for GPU acceleration
        """
        # Check numpy availability
        if not NUMPY_AVAILABLE:
            raise VoiceCloningError("numpy is required for voice cloning")

        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        self.gpu_manager = gpu_manager

        # Load configuration
        self.config = self._load_config(config)

        # Set device
        if device is not None:
            self.device = device
        elif gpu_manager is not None and hasattr(gpu_manager, 'device'):
            self.device = str(gpu_manager.device) if hasattr(gpu_manager.device, '__str__') else 'cpu'
        elif TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # Initialize components
        self._initialize_components()

        self.logger.info(
            f"VoiceCloner initialized: device={self.device}, "
            f"extract_vocal_range={self.config.get('extract_vocal_range')}, "
            f"extract_timbre={self.config.get('extract_timbre_features')}"
        )

    def _load_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load configuration from multiple sources

        Priority: constructor config > YAML file > environment variables > defaults

        Args:
            config: Configuration dictionary from constructor

        Returns:
            Merged configuration dictionary
        """
        # Start with defaults
        final_config = {
            'min_duration': 30.0,  # 30-60s recommended for best quality
            'max_duration': 60.0,
            'embedding_dim': 256,
            'storage_dir': '~/.cache/autovoice/voice_profiles/',
            'cache_enabled': True,
            'cache_size': 100,
            'extract_vocal_range': True,
            'extract_timbre_features': True,
            'min_sample_rate': 8000,
            'max_sample_rate': 48000,
            'silence_threshold': 0.01,
            'similarity_threshold': 0.75,
            'gpu_acceleration': True,
            # SNR validation
            'min_snr_db': 10.0,  # Minimum SNR threshold in dB
            # Multi-sample support
            'multi_sample_quality_weighting': True,  # Weight by SNR when averaging
            'multi_sample_min_samples': 1,  # Minimum samples for profile
            'multi_sample_max_samples': 10,  # Maximum samples to store
            # Versioning
            'versioning_enabled': True,
            'version_history_max_entries': 50,  # Max version history entries
            'schema_version': '1.0.0'  # Profile schema version
        }

        # Load from YAML if available - resolve path relative to package root
        # Go up 3 levels from this file: voice_cloner.py -> inference -> auto_voice -> src -> repo root
        config_path = Path(__file__).resolve().parents[3] / 'config' / 'audio_config.yaml'
        yaml_config = None
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    if yaml_config and 'voice_cloning' in yaml_config:
                        final_config.update(yaml_config['voice_cloning'])
            except Exception as e:
                self.logger.warning(f"Failed to load YAML config: {e}")
        else:
            self.logger.debug(f"Config file not found at {config_path}, using defaults")

        # Override with environment variables
        env_mapping = {
            'AUTOVOICE_VOICE_CLONING_MIN_DURATION': 'min_duration',
            'AUTOVOICE_VOICE_CLONING_MAX_DURATION': 'max_duration',
            'AUTOVOICE_VOICE_CLONING_STORAGE_DIR': 'storage_dir',
            'AUTOVOICE_VOICE_CLONING_CACHE_ENABLED': 'cache_enabled'
        }
        for env_var, config_key in env_mapping.items():
            if env_var in os.environ:
                try:
                    value = os.environ[env_var]
                    if config_key in ['min_duration', 'max_duration']:
                        value = float(value)
                    elif config_key == 'cache_enabled':
                        value = value.lower() in ('true', '1', 'yes')
                    final_config[config_key] = value
                except ValueError:
                    self.logger.warning(f"Invalid value for {env_var}: {os.environ[env_var]}")

        # Override with constructor config (highest priority)
        if config:
            final_config.update(config)

        # Auto-populate audio_config from YAML if not already set
        if yaml_config and 'audio' in yaml_config and 'audio_config' not in final_config:
            final_config['audio_config'] = yaml_config['audio']

        return final_config

    def _initialize_components(self):
        """Initialize all required components for voice cloning"""
        try:
            # Initialize SpeakerEncoder
            from ..models.speaker_encoder import SpeakerEncoder
            self.speaker_encoder = SpeakerEncoder(
                device=self.device,
                gpu_manager=self.gpu_manager
            )
            self.logger.info("SpeakerEncoder initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize SpeakerEncoder: {e}")
            raise VoiceCloningError(f"SpeakerEncoder initialization failed: {e}")

        try:
            # Initialize AudioProcessor with config from app
            from ..audio.processor import AudioProcessor
            audio_config = self.config.get('audio_config', {'sample_rate': 22050})
            self.audio_processor = AudioProcessor(
                config=audio_config,
                device=self.device
            )
            self.logger.info("AudioProcessor initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize AudioProcessor: {e}")
            raise VoiceCloningError(f"AudioProcessor initialization failed: {e}")

        # Initialize SingingPitchExtractor (optional)
        if self.config.get('extract_vocal_range', True):
            try:
                from ..audio.pitch_extractor import SingingPitchExtractor
                self.pitch_extractor = SingingPitchExtractor(
                    device=self.device,
                    gpu_manager=self.gpu_manager
                )
                self.logger.info("SingingPitchExtractor initialized")
            except Exception as e:
                self.logger.warning(f"SingingPitchExtractor unavailable, vocal range extraction disabled: {e}")
                self.pitch_extractor = None
        else:
            self.pitch_extractor = None

        try:
            # Initialize VoiceProfileStorage
            from ..storage.voice_profiles import VoiceProfileStorage
            self.storage = VoiceProfileStorage(
                storage_dir=self.config.get('storage_dir'),
                cache_enabled=self.config.get('cache_enabled', True),
                cache_size=self.config.get('cache_size', 100)
            )
            self.logger.info("VoiceProfileStorage initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize VoiceProfileStorage: {e}")
            raise VoiceCloningError(f"VoiceProfileStorage initialization failed: {e}")

    def create_voice_profile(
        self,
        audio: Union[np.ndarray, torch.Tensor, str],
        user_id: Optional[str] = None,
        sample_rate: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Create voice profile from audio sample

        This is the primary method for creating voice profiles. It performs:
        1. Audio validation (duration, quality)
        2. Speaker embedding extraction
        3. Vocal range analysis (optional)
        4. Timbre feature extraction (optional)
        5. Profile creation and storage

        Args:
            audio: Audio as numpy array, torch tensor, or file path
            user_id: Optional user identifier for profile management
            sample_rate: Sample rate (required for array/tensor input)
            metadata: Optional metadata dict (filename, format, etc.)

        Returns:
            Profile dictionary with keys:
                - profile_id: Unique identifier (UUID)
                - user_id: User identifier (if provided)
                - created_at: ISO timestamp
                - audio_duration: Duration in seconds
                - sample_rate: Sample rate used
                - embedding: 256-dim numpy array
                - vocal_range: {'min_f0', 'max_f0', 'range_semitones'}
                - timbre_features: {'spectral_centroid', 'spectral_rolloff'}
                - embedding_stats: {'mean', 'std', 'norm'}
                - metadata: Additional metadata

        Raises:
            InvalidAudioError: If audio validation fails
            VoiceCloningError: If profile creation fails

        Example:
            >>> profile = cloner.create_voice_profile(
            ...     audio='voice.wav',
            ...     user_id='user123',
            ...     metadata={'source': 'microphone'}
            ... )
            >>> print(f"Profile ID: {profile['profile_id']}")
        """
        with self.lock:
            try:
                # Load audio if file path
                if isinstance(audio, str):
                    audio_path = audio
                    audio, original_sr = self.audio_processor.load_audio(
                        audio_path,
                        return_sr=True
                    )
                    if isinstance(audio, torch.Tensor):
                        audio = audio.detach().cpu().numpy()

                    # Set resampled sample rate (target sample rate after resampling)
                    resampled_sr = self.audio_processor.sample_rate

                    # Update metadata with filename and original sample rate
                    if metadata is None:
                        metadata = {}
                    metadata.setdefault('filename', Path(audio_path).name)
                    metadata.setdefault('format', Path(audio_path).suffix[1:])
                    metadata['original_sample_rate'] = int(original_sr)

                    # Use resampled_sr for all downstream operations
                    sample_rate = resampled_sr

                # Convert to numpy if torch tensor
                if TORCH_AVAILABLE and isinstance(audio, torch.Tensor):
                    audio = audio.detach().cpu().numpy()

                if sample_rate is None:
                    raise InvalidAudioError(
                        "sample_rate must be provided for array/tensor input",
                        error_code='missing_sample_rate'
                    )

                # Validate audio (using resampled sample rate)
                is_valid, error_message, error_code = self.validate_audio(audio, sample_rate)
                if not is_valid:
                    raise InvalidAudioError(error_message, error_code=error_code or 'validation_failed')

                # Compute audio duration (using resampled sample rate)
                audio_duration = len(audio) / sample_rate

                self.logger.info(f"Creating voice profile from {audio_duration:.1f}s audio")

                # Extract speaker embedding
                embedding = self.speaker_encoder.extract_embedding(audio, sample_rate)
                embedding_stats = self.speaker_encoder.get_embedding_stats(embedding)

                # Extract vocal range (optional)
                vocal_range = {}
                if self.config.get('extract_vocal_range', True) and self.pitch_extractor is not None:
                    try:
                        vocal_range = self._extract_vocal_range(audio, sample_rate)
                    except Exception as e:
                        self.logger.warning(f"Vocal range extraction failed: {e}")

                # Extract timbre features (optional)
                timbre_features = {}
                if self.config.get('extract_timbre_features', True):
                    try:
                        timbre_features = self._extract_timbre_features(audio, sample_rate)
                    except Exception as e:
                        self.logger.warning(f"Timbre feature extraction failed: {e}")

                # Generate profile ID
                profile_id = str(uuid.uuid4())

                # Create profile dictionary
                profile = {
                    'profile_id': profile_id,
                    'user_id': user_id,
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'audio_duration': float(audio_duration),
                    'sample_rate': int(sample_rate),
                    'embedding': embedding,
                    'vocal_range': vocal_range,
                    'timbre_features': timbre_features,
                    'embedding_stats': embedding_stats,
                    'metadata': metadata or {}
                }

                # Add versioning metadata if enabled
                if self.config.get('versioning_enabled', True):
                    profile['schema_version'] = self.config.get('schema_version', '1.0.0')
                    profile['profile_version'] = 1
                    profile['version_history'] = [{
                        'version': 1,
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'change_description': 'Initial profile creation',
                        'audio_duration': float(audio_duration)
                    }]

                # Save profile
                self.storage.save_profile(profile)

                self.logger.info(f"Voice profile created: {profile_id}")

                # Return profile without embedding for API response (too large)
                response_profile = {k: v for k, v in profile.items() if k != 'embedding'}
                return response_profile

            except InvalidAudioError:
                raise
            except Exception as e:
                self.logger.error(f"Failed to create voice profile: {e}")
                raise VoiceCloningError(f"Profile creation failed: {e}")

    def load_voice_profile(self, profile_id: str) -> Dict[str, Any]:
        """Load voice profile by ID

        Args:
            profile_id: Profile identifier

        Returns:
            Profile dictionary with embedding

        Raises:
            ProfileNotFoundError: If profile doesn't exist (from storage)

        Example:
            >>> profile = cloner.load_voice_profile('uuid-1234')
            >>> embedding = profile['embedding']
        """
        from ..storage.voice_profiles import ProfileNotFoundError
        try:
            profile = self.storage.load_profile(profile_id, include_embedding=True)
            return profile
        except ProfileNotFoundError:
            # Re-raise the storage ProfileNotFoundError directly
            raise
        except Exception as e:
            raise VoiceCloningError(f"Failed to load profile: {e}")

    def list_voice_profiles(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all voice profiles, optionally filtered by user_id

        Args:
            user_id: Optional user ID filter

        Returns:
            List of profile dictionaries (without embeddings)

        Example:
            >>> all_profiles = cloner.list_voice_profiles()
            >>> user_profiles = cloner.list_voice_profiles(user_id='user123')
        """
        try:
            return self.storage.list_profiles(user_id=user_id)
        except Exception as e:
            self.logger.error(f"Failed to list profiles: {e}")
            raise VoiceCloningError(f"Profile listing failed: {e}")

    def delete_voice_profile(self, profile_id: str) -> bool:
        """Delete voice profile by ID

        Args:
            profile_id: Profile identifier

        Returns:
            True if deleted, False if not found

        Example:
            >>> deleted = cloner.delete_voice_profile('uuid-1234')
        """
        try:
            return self.storage.delete_profile(profile_id)
        except Exception as e:
            self.logger.error(f"Failed to delete profile: {e}")
            raise VoiceCloningError(f"Profile deletion failed: {e}")

    def get_embedding(self, profile_id: str) -> np.ndarray:
        """Get speaker embedding for profile

        Args:
            profile_id: Profile identifier

        Returns:
            256-dimensional speaker embedding

        Raises:
            ProfileNotFoundError: If profile doesn't exist

        Example:
            >>> embedding = cloner.get_embedding('uuid-1234')
            >>> print(embedding.shape)  # (256,)
        """
        profile = self.load_voice_profile(profile_id)
        return profile['embedding']

    def compare_profiles(
        self,
        profile_id1: str,
        profile_id2: str
    ) -> Dict[str, float]:
        """Compare two voice profiles using cosine similarity

        Args:
            profile_id1: First profile ID
            profile_id2: Second profile ID

        Returns:
            Dictionary with similarity score and interpretation

        Example:
            >>> result = cloner.compare_profiles('uuid-1', 'uuid-2')
            >>> print(f"Similarity: {result['similarity']:.3f}")
            >>> print(f"Same speaker: {result['is_same_speaker']}")
        """
        try:
            emb1 = self.get_embedding(profile_id1)
            emb2 = self.get_embedding(profile_id2)

            similarity = self.speaker_encoder.compute_similarity(emb1, emb2)
            threshold = self.config.get('similarity_threshold', 0.75)

            return {
                'similarity': float(similarity),
                'is_same_speaker': similarity >= threshold,
                'threshold': float(threshold)
            }

        except Exception as e:
            self.logger.error(f"Failed to compare profiles: {e}")
            raise VoiceCloningError(f"Profile comparison failed: {e}")

    def validate_audio(
        self,
        audio: Union[np.ndarray, str],
        sample_rate: Optional[int] = None
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """Validate audio for voice cloning with SNR checking

        Args:
            audio: Audio array or file path
            sample_rate: Sample rate (required for array input)

        Returns:
            Tuple of (is_valid, error_message, error_code)

        Example:
            >>> is_valid, error = cloner.validate_audio(audio, sample_rate=22050)
            >>> if not is_valid:
            ...     print(f"Invalid audio: {error}")
        """
        try:
            # Load audio if file path
            if isinstance(audio, str):
                audio, sample_rate = self.audio_processor.load_audio(audio, return_sr=True)
                if TORCH_AVAILABLE and isinstance(audio, torch.Tensor):
                    audio = audio.detach().cpu().numpy()

            if sample_rate is None:
                return False, "sample_rate is required", 'missing_sample_rate'

            # Normalize audio to mono float32 before validation
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio, dtype=np.float32)

            # Convert to mono if stereo/multi-channel
            if audio.ndim > 1:
                audio = np.mean(audio, axis=0 if audio.shape[0] < audio.shape[1] else 1)

            # Convert to float32
            audio = audio.astype(np.float32)

            # Normalize to [-1, 1] if needed
            max_abs = np.max(np.abs(audio))
            if max_abs > 1.0:
                audio = audio / max_abs

            # Check duration
            duration = len(audio) / sample_rate
            min_duration = self.config.get('min_duration', 5.0)
            max_duration = self.config.get('max_duration', 60.0)

            if duration < min_duration:
                return False, f"Audio too short: {duration:.1f}s (minimum: {min_duration}s)", 'duration_too_short'

            if duration > max_duration:
                return False, f"Audio too long: {duration:.1f}s (maximum: {max_duration}s)", 'duration_too_long'

            # Check sample rate
            min_sr = self.config.get('min_sample_rate', 8000)
            max_sr = self.config.get('max_sample_rate', 48000)

            if sample_rate < min_sr or sample_rate > max_sr:
                return False, f"Invalid sample rate: {sample_rate} Hz (range: {min_sr}-{max_sr} Hz)", 'invalid_sample_rate'

            # Check audio quality (not silent, not pure noise)
            rms = np.sqrt(np.mean(audio ** 2))
            silence_threshold = self.config.get('silence_threshold', 0.001)  # Lowered from 0.01 to 0.001

            if rms < silence_threshold:
                return False, f"Audio too quiet (RMS: {rms:.4f} < threshold: {silence_threshold})", 'audio_too_quiet'

            # Check SNR if threshold is configured
            min_snr_db = self.config.get('min_snr_db', None)
            if min_snr_db is not None:
                snr_db = self._compute_snr(audio)
                if snr_db is not None and snr_db < min_snr_db:
                    return False, f"Audio SNR too low: {snr_db:.1f} dB (minimum: {min_snr_db} dB)", 'snr_too_low'

            return True, None, None

        except Exception as e:
            return False, f"Audio validation error: {str(e)}", 'validation_error'

    def _compute_snr(self, audio: np.ndarray) -> Optional[float]:
        """Compute Signal-to-Noise Ratio (SNR) in dB

        Estimates SNR using RMS-based signal power and noise floor estimation.

        Args:
            audio: Audio array (mono, float32, normalized to [-1, 1])

        Returns:
            SNR in dB, or None if computation fails

        Notes:
            SNR = 10 * log10(signal_power / noise_power)
            - Signal power estimated from RMS of entire audio
            - Noise floor estimated from quietest 10% of frames
        """
        try:
            # Ensure audio is 1D
            if audio.ndim > 1:
                audio = audio.flatten()

            # Frame the audio for noise floor estimation
            frame_length = 2048
            hop_length = 512

            # Compute frame-wise RMS
            num_frames = (len(audio) - frame_length) // hop_length + 1
            if num_frames < 10:
                # Too short for reliable SNR estimation
                return None

            frame_rms = np.zeros(num_frames)
            for i in range(num_frames):
                start = i * hop_length
                end = start + frame_length
                if end > len(audio):
                    break
                frame = audio[start:end]
                frame_rms[i] = np.sqrt(np.mean(frame ** 2))

            # Remove zero/near-zero frames
            frame_rms = frame_rms[frame_rms > 1e-10]
            if len(frame_rms) < 10:
                return None

            # Estimate signal power from RMS of entire audio
            signal_rms = np.sqrt(np.mean(audio ** 2))
            signal_power = signal_rms ** 2

            # Estimate noise floor from quietest 10% of frames
            noise_percentile = 10
            noise_rms = np.percentile(frame_rms, noise_percentile)
            noise_power = noise_rms ** 2

            # Guard against zero division
            if noise_power < 1e-10 or signal_power < 1e-10:
                return None

            # Compute SNR in dB
            snr_db = 10.0 * np.log10(signal_power / noise_power)

            return float(snr_db)

        except Exception as e:
            self.logger.warning(f"SNR computation failed: {e}")
            return None

    def get_audio_quality_report(
        self,
        audio: Union[np.ndarray, str],
        sample_rate: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate detailed audio quality diagnostic report

        Args:
            audio: Audio array or file path
            sample_rate: Sample rate (required for array input)

        Returns:
            Dictionary with quality metrics:
                - duration: Audio duration in seconds
                - sample_rate: Sample rate in Hz
                - rms: Root Mean Square amplitude
                - peak: Peak amplitude
                - snr_db: Signal-to-Noise Ratio in dB
                - dynamic_range_db: Dynamic range in dB
                - is_valid: Whether audio passes validation
                - validation_errors: List of validation error messages

        Example:
            >>> report = cloner.get_audio_quality_report('voice.wav')
            >>> print(f"SNR: {report['snr_db']:.1f} dB")
            >>> print(f"Valid: {report['is_valid']}")
        """
        try:
            # Load audio if file path
            if isinstance(audio, str):
                audio, sample_rate = self.audio_processor.load_audio(audio, return_sr=True)
                if TORCH_AVAILABLE and isinstance(audio, torch.Tensor):
                    audio = audio.detach().cpu().numpy()

            if sample_rate is None:
                raise InvalidAudioError(
                    "sample_rate is required for array input",
                    error_code='missing_sample_rate'
                )

            # Normalize audio
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio, dtype=np.float32)

            # Convert to mono
            if audio.ndim > 1:
                audio = np.mean(audio, axis=0 if audio.shape[0] < audio.shape[1] else 1)

            audio = audio.astype(np.float32)

            # Normalize to [-1, 1]
            max_abs = np.max(np.abs(audio))
            if max_abs > 1.0:
                audio = audio / max_abs

            # Compute metrics
            duration = len(audio) / sample_rate
            rms = np.sqrt(np.mean(audio ** 2))
            peak = float(np.max(np.abs(audio)))
            snr_db = self._compute_snr(audio)

            # Dynamic range (difference between peak and noise floor)
            # Use percentile-based noise floor
            noise_floor = np.percentile(np.abs(audio), 10)
            dynamic_range_db = 20.0 * np.log10(peak / (noise_floor + 1e-10))

            # Validate audio
            is_valid, error_message, error_code = self.validate_audio(audio, sample_rate)
            validation_errors = []
            if not is_valid:
                validation_errors.append({
                    'message': error_message,
                    'code': error_code
                })

            return {
                'duration': float(duration),
                'sample_rate': int(sample_rate),
                'rms': float(rms),
                'peak': float(peak),
                'snr_db': snr_db,
                'dynamic_range_db': float(dynamic_range_db),
                'is_valid': is_valid,
                'validation_errors': validation_errors
            }

        except Exception as e:
            self.logger.error(f"Failed to generate quality report: {e}")
            raise VoiceCloningError(f"Quality report generation failed: {e}")

    def create_voice_profile_from_multiple_samples(
        self,
        audio_samples: List[Union[np.ndarray, str]],
        user_id: Optional[str] = None,
        sample_rate: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Create voice profile by averaging multiple audio samples

        This method creates a more robust voice profile by combining multiple
        audio samples from the same speaker. Embeddings are averaged (optionally
        quality-weighted by SNR), and vocal ranges/timbre features are merged.

        Args:
            audio_samples: List of audio arrays or file paths
            user_id: Optional user identifier
            sample_rate: Sample rate (required if passing arrays)
            metadata: Optional metadata dict

        Returns:
            Profile dictionary with merged features

        Raises:
            InvalidAudioError: If any sample fails validation
            VoiceCloningError: If profile creation fails

        Example:
            >>> samples = ['voice1.wav', 'voice2.wav', 'voice3.wav']
            >>> profile = cloner.create_voice_profile_from_multiple_samples(
            ...     audio_samples=samples,
            ...     user_id='user123'
            ... )
            >>> print(f"Profile from {len(samples)} samples")
        """
        with self.lock:
            try:
                min_samples = self.config.get('multi_sample_min_samples', 1)
                max_samples = self.config.get('multi_sample_max_samples', 10)

                if len(audio_samples) < min_samples:
                    raise InvalidAudioError(
                        f"Need at least {min_samples} samples (got {len(audio_samples)})",
                        error_code='insufficient_samples'
                    )

                if len(audio_samples) > max_samples:
                    self.logger.warning(f"Too many samples ({len(audio_samples)}), using first {max_samples}")
                    audio_samples = audio_samples[:max_samples]

                self.logger.info(f"Creating voice profile from {len(audio_samples)} samples")

                # Process each sample
                embeddings = []
                snr_scores = []
                vocal_ranges = []
                timbre_features_list = []
                sample_metadata = []

                for i, audio in enumerate(audio_samples):
                    try:
                        # Load audio if file path
                        if isinstance(audio, str):
                            audio_data, sr = self.audio_processor.load_audio(audio, return_sr=True)
                            if isinstance(audio_data, torch.Tensor):
                                audio_data = audio_data.detach().cpu().numpy()
                            current_sr = sr

                            sample_info = {
                                'index': i,
                                'filename': Path(audio).name if isinstance(audio, str) else f'sample_{i}',
                                'original_sample_rate': int(sr)
                            }
                        else:
                            audio_data = audio
                            current_sr = sample_rate
                            if current_sr is None:
                                raise InvalidAudioError(
                                    "sample_rate required for array input",
                                    error_code='missing_sample_rate'
                                )
                            sample_info = {
                                'index': i,
                                'filename': f'sample_{i}',
                                'original_sample_rate': int(current_sr)
                            }

                        # Convert to numpy
                        if TORCH_AVAILABLE and isinstance(audio_data, torch.Tensor):
                            audio_data = audio_data.detach().cpu().numpy()

                        # Validate
                        is_valid, error_msg, error_code = self.validate_audio(audio_data, current_sr)
                        if not is_valid:
                            raise InvalidAudioError(
                                f"Sample {i} validation failed: {error_msg}",
                                error_code=error_code
                            )

                        # Compute SNR for quality weighting
                        snr_db = self._compute_snr(audio_data)
                        snr_scores.append(snr_db if snr_db is not None else 0.0)
                        sample_info['snr_db'] = snr_db

                        # Extract embedding
                        embedding = self.speaker_encoder.extract_embedding(audio_data, current_sr)
                        embeddings.append(embedding)

                        # Extract vocal range
                        if self.config.get('extract_vocal_range', True) and self.pitch_extractor is not None:
                            try:
                                vocal_range = self._extract_vocal_range(audio_data, current_sr)
                                if vocal_range:
                                    vocal_ranges.append(vocal_range)
                            except Exception as e:
                                self.logger.warning(f"Vocal range extraction failed for sample {i}: {e}")

                        # Extract timbre features
                        if self.config.get('extract_timbre_features', True):
                            try:
                                timbre = self._extract_timbre_features(audio_data, current_sr)
                                if timbre:
                                    timbre_features_list.append(timbre)
                            except Exception as e:
                                self.logger.warning(f"Timbre extraction failed for sample {i}: {e}")

                        sample_metadata.append(sample_info)

                    except InvalidAudioError:
                        raise
                    except Exception as e:
                        raise VoiceCloningError(f"Failed to process sample {i}: {e}")

                # Average embeddings (optionally quality-weighted)
                if self.config.get('multi_sample_quality_weighting', True) and any(s > 0 for s in snr_scores):
                    # Compute weights from SNR (softmax-like normalization)
                    weights = np.array(snr_scores)
                    # Convert SNR to linear scale for weighting
                    weights = 10 ** (weights / 10.0)  # Convert dB to linear power
                    weights = weights / np.sum(weights)  # Normalize

                    # Weighted average
                    averaged_embedding = np.average(embeddings, axis=0, weights=weights)
                else:
                    # Simple average
                    averaged_embedding = np.mean(embeddings, axis=0)

                # Merge vocal ranges (min of mins, max of maxs, average of means)
                merged_vocal_range = {}
                if vocal_ranges:
                    merged_vocal_range = {
                        'min_f0': float(min(vr['min_f0'] for vr in vocal_ranges)),
                        'max_f0': float(max(vr['max_f0'] for vr in vocal_ranges)),
                        'mean_f0': float(np.mean([vr['mean_f0'] for vr in vocal_ranges])),
                        'range_semitones': float(max(vr['range_semitones'] for vr in vocal_ranges))
                    }

                # Average timbre features
                merged_timbre = {}
                if timbre_features_list:
                    if all('spectral_centroid' in t for t in timbre_features_list):
                        merged_timbre['spectral_centroid'] = float(
                            np.mean([t['spectral_centroid'] for t in timbre_features_list])
                        )
                    if all('spectral_rolloff' in t for t in timbre_features_list):
                        merged_timbre['spectral_rolloff'] = float(
                            np.mean([t['spectral_rolloff'] for t in timbre_features_list])
                        )

                # Compute embedding stats
                embedding_stats = self.speaker_encoder.get_embedding_stats(averaged_embedding)

                # Generate profile ID
                profile_id = str(uuid.uuid4())

                # Create profile with versioning metadata
                profile = {
                    'profile_id': profile_id,
                    'user_id': user_id,
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'sample_rate': int(sample_rate or 22050),
                    'embedding': averaged_embedding,
                    'vocal_range': merged_vocal_range,
                    'timbre_features': merged_timbre,
                    'embedding_stats': embedding_stats,
                    'metadata': metadata or {},
                    'multi_sample_info': {
                        'num_samples': len(audio_samples),
                        'sample_metadata': sample_metadata,
                        'quality_weighted': self.config.get('multi_sample_quality_weighting', True),
                        'average_snr_db': float(np.mean(snr_scores)) if snr_scores else None
                    }
                }

                # Add versioning metadata if enabled
                if self.config.get('versioning_enabled', True):
                    profile['schema_version'] = self.config.get('schema_version', '1.0.0')
                    profile['profile_version'] = 1
                    profile['version_history'] = [{
                        'version': 1,
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'change_description': f'Initial profile from {len(audio_samples)} samples',
                        'num_samples': len(audio_samples)
                    }]

                # Save profile
                self.storage.save_profile(profile)

                self.logger.info(f"Multi-sample voice profile created: {profile_id}")

                # Return without embedding
                response_profile = {k: v for k, v in profile.items() if k != 'embedding'}
                return response_profile

            except InvalidAudioError:
                raise
            except Exception as e:
                self.logger.error(f"Failed to create multi-sample profile: {e}")
                raise VoiceCloningError(f"Multi-sample profile creation failed: {e}")

    def add_sample_to_profile(
        self,
        profile_id: str,
        audio: Union[np.ndarray, str],
        sample_rate: Optional[int] = None,
        weight: Optional[float] = None
    ) -> Dict[str, Any]:
        """Add a new sample to existing voice profile and update embedding

        This method updates an existing profile by incorporating a new audio sample.
        The embedding is re-averaged with the new sample, and vocal range/timbre
        features are updated.

        Args:
            profile_id: Existing profile identifier
            audio: New audio sample (array or file path)
            sample_rate: Sample rate (required for array input)
            weight: Optional manual weight (if None, uses SNR-based weighting)

        Returns:
            Updated profile dictionary

        Raises:
            ProfileNotFoundError: If profile doesn't exist
            InvalidAudioError: If sample validation fails
            VoiceCloningError: If update fails

        Example:
            >>> profile = cloner.add_sample_to_profile(
            ...     profile_id='uuid-1234',
            ...     audio='new_voice.wav'
            ... )
        """
        with self.lock:
            try:
                # Load existing profile
                from ..storage.voice_profiles import ProfileNotFoundError
                profile = self.load_voice_profile(profile_id)

                # Check max samples limit
                max_samples = self.config.get('multi_sample_max_samples', 10)
                current_num_samples = profile.get('multi_sample_info', {}).get('num_samples', 1)

                if current_num_samples >= max_samples:
                    raise VoiceCloningError(
                        f"Profile already has maximum {max_samples} samples"
                    )

                # Load and validate new sample
                if isinstance(audio, str):
                    audio_data, sr = self.audio_processor.load_audio(audio, return_sr=True)
                    if isinstance(audio_data, torch.Tensor):
                        audio_data = audio_data.detach().cpu().numpy()
                    current_sr = sr
                    sample_name = Path(audio).name
                else:
                    audio_data = audio
                    current_sr = sample_rate
                    if current_sr is None:
                        raise InvalidAudioError(
                            "sample_rate required for array input",
                            error_code='missing_sample_rate'
                        )
                    sample_name = f'sample_{current_num_samples}'

                # Convert to numpy
                if TORCH_AVAILABLE and isinstance(audio_data, torch.Tensor):
                    audio_data = audio_data.detach().cpu().numpy()

                # Validate
                is_valid, error_msg, error_code = self.validate_audio(audio_data, current_sr)
                if not is_valid:
                    raise InvalidAudioError(f"Sample validation failed: {error_msg}", error_code=error_code)

                # Extract features from new sample
                new_embedding = self.speaker_encoder.extract_embedding(audio_data, current_sr)
                new_snr = self._compute_snr(audio_data) if weight is None else None

                # Extract vocal range and timbre
                new_vocal_range = {}
                if self.config.get('extract_vocal_range', True) and self.pitch_extractor is not None:
                    try:
                        new_vocal_range = self._extract_vocal_range(audio_data, current_sr)
                    except Exception as e:
                        self.logger.warning(f"Vocal range extraction failed: {e}")

                new_timbre = {}
                if self.config.get('extract_timbre_features', True):
                    try:
                        new_timbre = self._extract_timbre_features(audio_data, current_sr)
                    except Exception as e:
                        self.logger.warning(f"Timbre extraction failed: {e}")

                # Compute new averaged embedding
                old_embedding = profile['embedding']

                if weight is not None:
                    # Manual weight provided
                    total_weight = current_num_samples + weight
                    averaged_embedding = (old_embedding * current_num_samples + new_embedding * weight) / total_weight
                elif self.config.get('multi_sample_quality_weighting', True) and new_snr is not None:
                    # Quality-weighted averaging
                    # Approximate old average SNR
                    old_avg_snr = profile.get('multi_sample_info', {}).get('average_snr_db', 15.0)

                    # Convert to linear power scale
                    old_weight = 10 ** (old_avg_snr / 10.0) * current_num_samples
                    new_weight = 10 ** (new_snr / 10.0)
                    total_weight = old_weight + new_weight

                    averaged_embedding = (old_embedding * old_weight + new_embedding * new_weight) / total_weight
                else:
                    # Simple averaging
                    averaged_embedding = (old_embedding * current_num_samples + new_embedding) / (current_num_samples + 1)

                # Update vocal range (merge with existing)
                if new_vocal_range and profile.get('vocal_range'):
                    old_range = profile['vocal_range']
                    profile['vocal_range'] = {
                        'min_f0': float(min(old_range.get('min_f0', float('inf')), new_vocal_range.get('min_f0', float('inf')))),
                        'max_f0': float(max(old_range.get('max_f0', 0), new_vocal_range.get('max_f0', 0))),
                        'mean_f0': float((old_range.get('mean_f0', 0) * current_num_samples + new_vocal_range.get('mean_f0', 0)) / (current_num_samples + 1)),
                        'range_semitones': float(max(old_range.get('range_semitones', 0), new_vocal_range.get('range_semitones', 0)))
                    }
                elif new_vocal_range:
                    profile['vocal_range'] = new_vocal_range

                # Update timbre features (average with existing)
                if new_timbre and profile.get('timbre_features'):
                    old_timbre = profile['timbre_features']
                    for key in new_timbre:
                        if key in old_timbre:
                            profile['timbre_features'][key] = float(
                                (old_timbre[key] * current_num_samples + new_timbre[key]) / (current_num_samples + 1)
                            )
                elif new_timbre:
                    profile['timbre_features'] = new_timbre

                # Update embedding and stats
                profile['embedding'] = averaged_embedding
                profile['embedding_stats'] = self.speaker_encoder.get_embedding_stats(averaged_embedding)

                # Update multi-sample info
                if 'multi_sample_info' not in profile:
                    profile['multi_sample_info'] = {
                        'num_samples': 1,
                        'sample_metadata': [],
                        'quality_weighted': self.config.get('multi_sample_quality_weighting', True),
                        'average_snr_db': None
                    }

                sample_info = {
                    'index': current_num_samples,
                    'filename': sample_name,
                    'original_sample_rate': int(current_sr),
                    'snr_db': new_snr,
                    'added_at': datetime.now(timezone.utc).isoformat()
                }

                profile['multi_sample_info']['num_samples'] = current_num_samples + 1
                profile['multi_sample_info']['sample_metadata'].append(sample_info)

                # Update average SNR
                all_snrs = [s.get('snr_db') for s in profile['multi_sample_info']['sample_metadata'] if s.get('snr_db') is not None]
                if all_snrs:
                    profile['multi_sample_info']['average_snr_db'] = float(np.mean(all_snrs))

                # Update version history
                if self.config.get('versioning_enabled', True):
                    current_version = profile.get('profile_version', 1)
                    new_version = current_version + 1
                    profile['profile_version'] = new_version

                    if 'version_history' not in profile:
                        profile['version_history'] = []

                    max_history = self.config.get('version_history_max_entries', 50)
                    history_entry = {
                        'version': new_version,
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'change_description': f'Added sample: {sample_name}',
                        'num_samples': current_num_samples + 1,
                        'sample_added': sample_name
                    }
                    profile['version_history'].append(history_entry)

                    # Trim history if needed
                    if len(profile['version_history']) > max_history:
                        profile['version_history'] = profile['version_history'][-max_history:]

                profile['updated_at'] = datetime.now(timezone.utc).isoformat()

                # Save updated profile
                self.storage.save_profile(profile)

                self.logger.info(f"Added sample to profile {profile_id}, now has {current_num_samples + 1} samples")

                # Return without embedding
                response_profile = {k: v for k, v in profile.items() if k != 'embedding'}
                return response_profile

            except (InvalidAudioError, ProfileNotFoundError):
                raise
            except Exception as e:
                self.logger.error(f"Failed to add sample to profile: {e}")
                raise VoiceCloningError(f"Sample addition failed: {e}")

    def get_profile_version_history(self, profile_id: str) -> List[Dict[str, Any]]:
        """Get version history for a voice profile

        Args:
            profile_id: Profile identifier

        Returns:
            List of version history entries with timestamps and descriptions

        Raises:
            ProfileNotFoundError: If profile doesn't exist

        Example:
            >>> history = cloner.get_profile_version_history('uuid-1234')
            >>> for entry in history:
            ...     print(f"v{entry['version']}: {entry['change_description']}")
        """
        try:
            from ..storage.voice_profiles import ProfileNotFoundError
            profile = self.load_voice_profile(profile_id)
            return profile.get('version_history', [])
        except ProfileNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to get version history: {e}")
            raise VoiceCloningError(f"Version history retrieval failed: {e}")

    def _extract_vocal_range(self, audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Extract vocal range using SingingPitchExtractor

        Args:
            audio: Audio array
            sample_rate: Sample rate

        Returns:
            Dictionary with min_f0, max_f0, range_semitones
        """
        if self.pitch_extractor is None:
            return {}

        try:
            # Extract F0 contour
            f0_data = self.pitch_extractor.extract_f0_contour(
                audio,
                sample_rate,
                return_confidence=False,
                return_times=False
            )

            # Get pitch statistics
            stats = self.pitch_extractor.get_pitch_statistics(f0_data)

            return {
                'min_f0': float(stats['min_f0']),
                'max_f0': float(stats['max_f0']),
                'range_semitones': float(stats['range_semitones']),
                'mean_f0': float(stats['mean_f0'])
            }

        except Exception as e:
            self.logger.warning(f"Vocal range extraction failed: {e}")
            return {}

    def _extract_timbre_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Extract timbre features from linear-frequency STFT

        Args:
            audio: Audio array
            sample_rate: Sample rate

        Returns:
            Dictionary with spectral_centroid, spectral_rolloff
        """
        try:
            # Ensure audio is numpy for librosa
            if TORCH_AVAILABLE and isinstance(audio, torch.Tensor):
                audio_np = audio.detach().cpu().numpy()
            else:
                audio_np = audio

            # Compute spectral features using librosa
            if LIBROSA_AVAILABLE:
                spectral_centroid = librosa.feature.spectral_centroid(
                    y=audio_np,
                    sr=sample_rate
                ).mean()

                spectral_rolloff = librosa.feature.spectral_rolloff(
                    y=audio_np,
                    sr=sample_rate
                ).mean()

                return {
                    'spectral_centroid': float(spectral_centroid),
                    'spectral_rolloff': float(spectral_rolloff)
                }
            else:
                # Fallback: estimate from linear-frequency STFT magnitude
                # Only use torch if available; otherwise work with numpy directly
                if TORCH_AVAILABLE:
                    if not isinstance(audio, torch.Tensor):
                        audio_tensor = torch.from_numpy(audio_np).float()
                    else:
                        audio_tensor = audio
                    # Compute linear-frequency STFT
                    stft_mag = self.audio_processor.compute_spectrogram(audio_tensor)
                    if isinstance(stft_mag, torch.Tensor):
                        stft_mag_np = stft_mag.detach().cpu().numpy()
                    else:
                        stft_mag_np = stft_mag
                else:
                    # Pure numpy fallback - compute STFT manually or use AudioProcessor
                    stft_mag = self.audio_processor.compute_spectrogram(audio_np)
                    if hasattr(stft_mag, 'numpy'):
                        stft_mag_np = stft_mag.numpy()
                    else:
                        stft_mag_np = stft_mag

                # Guard-rails for empty or zero-energy frames
                if stft_mag_np.size == 0 or stft_mag_np.shape[1] == 0:
                    self.logger.warning("Empty spectrogram, returning empty features")
                    return {}

                # Create linear frequency axis
                freqs = np.linspace(0, sample_rate / 2, stft_mag_np.shape[0])

                # Spectral centroid on linear frequency axis
                # Guard against zero-energy frames
                frame_energy = np.sum(stft_mag_np, axis=0)
                valid_frames = frame_energy > 1e-10

                if not np.any(valid_frames):
                    self.logger.warning("All frames have zero energy, returning empty features")
                    return {}

                centroid = np.sum(freqs[:, None] * stft_mag_np, axis=0) / (frame_energy + 1e-10)
                # Only use valid frames for mean calculation
                centroid_valid = centroid[valid_frames]
                centroid_valid = centroid_valid[np.isfinite(centroid_valid)]

                if len(centroid_valid) == 0:
                    self.logger.warning("No valid centroid values, returning empty features")
                    return {}

                spectral_centroid = np.mean(centroid_valid)

                # Spectral rolloff approximation (85th percentile)
                cumsum = np.cumsum(stft_mag_np, axis=0)
                total = cumsum[-1, :] + 1e-10
                rolloff_idx = np.argmax(cumsum >= 0.85 * total[None, :], axis=0)
                # Only use valid frames for rolloff
                rolloff_freqs = freqs[rolloff_idx[valid_frames]]
                rolloff_freqs = rolloff_freqs[np.isfinite(rolloff_freqs)]

                if len(rolloff_freqs) == 0:
                    self.logger.warning("No valid rolloff values, returning partial features")
                    return {'spectral_centroid': float(spectral_centroid)}

                spectral_rolloff = np.mean(rolloff_freqs)

                return {
                    'spectral_centroid': float(spectral_centroid),
                    'spectral_rolloff': float(spectral_rolloff)
                }

        except Exception as e:
            self.logger.warning(f"Timbre feature extraction failed: {e}")
            return {}
