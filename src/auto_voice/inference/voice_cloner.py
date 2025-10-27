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


class ProfileNotFoundError(VoiceCloningError):
    """Exception raised when profile doesn't exist"""
    pass


class InvalidAudioError(VoiceCloningError):
    """Exception raised when audio validation fails"""
    pass


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
            'min_duration': 5.0,
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
            'gpu_acceleration': True
        }

        # Load from YAML if available
        config_path = Path('config/audio_config.yaml')
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    if yaml_config and 'voice_cloning' in yaml_config:
                        final_config.update(yaml_config['voice_cloning'])
            except Exception as e:
                self.logger.warning(f"Failed to load YAML config: {e}")

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
            # Initialize AudioProcessor
            from ..audio.processor import AudioProcessor
            self.audio_processor = AudioProcessor(
                config={'sample_rate': 22050},
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
                    audio, sample_rate = self.audio_processor.load_audio(
                        audio_path,
                        return_sr=True
                    )
                    if isinstance(audio, torch.Tensor):
                        audio = audio.detach().cpu().numpy()

                    # Update metadata with filename
                    if metadata is None:
                        metadata = {}
                    metadata.setdefault('filename', Path(audio_path).name)
                    metadata.setdefault('format', Path(audio_path).suffix[1:])

                # Convert to numpy if torch tensor
                if TORCH_AVAILABLE and isinstance(audio, torch.Tensor):
                    audio = audio.detach().cpu().numpy()

                if sample_rate is None:
                    raise InvalidAudioError("sample_rate must be provided for array/tensor input")

                # Validate audio
                is_valid, error_message = self.validate_audio(audio, sample_rate)
                if not is_valid:
                    raise InvalidAudioError(error_message)

                # Compute audio duration
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

                # Save profile
                self.storage.save_profile(profile)

                self.logger.info(f"Voice profile created: {profile_id}")

                # Return profile without embedding for API response
                return profile

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
            ProfileNotFoundError: If profile doesn't exist

        Example:
            >>> profile = cloner.load_voice_profile('uuid-1234')
            >>> embedding = profile['embedding']
        """
        try:
            profile = self.storage.load_profile(profile_id, include_embedding=True)
            return profile
        except Exception as e:
            if 'not found' in str(e).lower():
                raise ProfileNotFoundError(f"Profile not found: {profile_id}")
            else:
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
    ) -> tuple[bool, Optional[str]]:
        """Validate audio for voice cloning

        Args:
            audio: Audio array or file path
            sample_rate: Sample rate (required for array input)

        Returns:
            Tuple of (is_valid, error_message)

        Example:
            >>> is_valid, error = cloner.validate_audio(audio, sample_rate=22050)
            >>> if not is_valid:
            ...     print(f"Invalid audio: {error}")
        """
        try:
            # Load audio if file path
            if isinstance(audio, str):
                audio, sample_rate = self.audio_processor.load_audio(audio, return_sr=True)
                if isinstance(audio, torch.Tensor):
                    audio = audio.detach().cpu().numpy()

            if sample_rate is None:
                return False, "sample_rate is required"

            # Check duration
            duration = len(audio) / sample_rate
            min_duration = self.config.get('min_duration', 5.0)
            max_duration = self.config.get('max_duration', 60.0)

            if duration < min_duration:
                return False, f"Audio too short: {duration:.1f}s (minimum: {min_duration}s)"

            if duration > max_duration:
                return False, f"Audio too long: {duration:.1f}s (maximum: {max_duration}s)"

            # Check sample rate
            min_sr = self.config.get('min_sample_rate', 8000)
            max_sr = self.config.get('max_sample_rate', 48000)

            if sample_rate < min_sr or sample_rate > max_sr:
                return False, f"Invalid sample rate: {sample_rate} Hz (range: {min_sr}-{max_sr} Hz)"

            # Check audio quality (not silent, not pure noise)
            rms = np.sqrt(np.mean(audio ** 2))
            silence_threshold = self.config.get('silence_threshold', 0.01)

            if rms < silence_threshold:
                return False, f"Audio too quiet (RMS: {rms:.4f} < threshold: {silence_threshold})"

            return True, None

        except Exception as e:
            return False, f"Audio validation error: {str(e)}"

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
        """Extract timbre features from mel-spectrogram

        Args:
            audio: Audio array
            sample_rate: Sample rate

        Returns:
            Dictionary with spectral_centroid, spectral_rolloff
        """
        try:
            # Convert to torch tensor
            if not isinstance(audio, torch.Tensor):
                audio_tensor = torch.from_numpy(audio).float()
            else:
                audio_tensor = audio

            # Compute mel-spectrogram
            mel_spec = self.audio_processor.to_mel_spectrogram(
                audio_tensor,
                sample_rate=sample_rate
            )

            # Convert to numpy for librosa
            if isinstance(mel_spec, torch.Tensor):
                mel_spec_np = mel_spec.detach().cpu().numpy()
            else:
                mel_spec_np = mel_spec

            # Ensure audio is numpy for librosa
            if isinstance(audio, torch.Tensor):
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
                # Fallback: estimate from mel-spectrogram
                # Spectral centroid approximation
                freqs = np.linspace(0, sample_rate / 2, mel_spec_np.shape[0])
                centroid = np.sum(freqs[:, None] * mel_spec_np, axis=0) / np.sum(mel_spec_np, axis=0)
                spectral_centroid = np.mean(centroid[np.isfinite(centroid)])

                # Spectral rolloff approximation (85th percentile)
                cumsum = np.cumsum(mel_spec_np, axis=0)
                total = cumsum[-1, :]
                rolloff_idx = np.argmax(cumsum >= 0.85 * total[None, :], axis=0)
                spectral_rolloff = np.mean(freqs[rolloff_idx])

                return {
                    'spectral_centroid': float(spectral_centroid),
                    'spectral_rolloff': float(spectral_rolloff)
                }

        except Exception as e:
            self.logger.warning(f"Timbre feature extraction failed: {e}")
            return {}
