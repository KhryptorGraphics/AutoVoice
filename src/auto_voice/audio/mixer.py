"""Audio mixing utilities for combining vocals and instrumental tracks.

This module provides the AudioMixer class for mixing converted vocals with
instrumental tracks, applying volume normalization, and ensuring synchronization.
"""

import os
import threading
from typing import Optional, Dict, Any, Union, Tuple, Callable
import numpy as np

# Conditional imports
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

from .processor import AudioProcessor
from ..utils.data_utils import DataPreprocessor
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class MixingError(Exception):
    """Exception raised for audio mixing errors."""
    pass


class AudioMixer:
    """Mix converted vocals with instrumental tracks.

    This class provides utilities for:
    - Mixing vocals and instrumental with volume control
    - Volume normalization (RMS, peak, LUFS)
    - Length alignment and synchronization
    - Stereo/mono conversion
    - Clipping prevention

    Attributes:
        config (Dict[str, Any]): Mixer configuration
        audio_processor (AudioProcessor): Audio I/O processor
        data_preprocessor (DataPreprocessor): Audio preprocessing utilities
        lock (threading.RLock): Thread safety lock

    Example:
        >>> mixer = AudioMixer()
        >>> mixed, sr = mixer.mix(
        ...     vocals='converted_vocals.wav',
        ...     instrumental='original_instrumental.wav',
        ...     vocal_volume=1.0,
        ...     instrumental_volume=0.9
        ... )
        >>> import soundfile as sf
        >>> sf.write('final_mix.wav', mixed, sr)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize AudioMixer.

        Args:
            config: Optional configuration dict with mixing parameters.
                If None, loads from YAML and environment variables.
        """
        self.config = self._load_config(config)
        self.audio_processor = AudioProcessor()
        self.data_preprocessor = DataPreprocessor()
        self.lock = threading.RLock()

        logger.info(f"AudioMixer initialized with normalization: {self.config['normalization_method']}")

    def _load_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load configuration from multiple sources.

        Priority: constructor config > YAML file > environment variables > defaults

        Args:
            config: Optional constructor configuration

        Returns:
            Merged configuration dictionary
        """
        # Default configuration
        default_config = {
            'normalization_method': 'rms',
            'target_vocal_level_db': -20.0,
            'target_instrumental_level_db': -23.0,
            'auto_align_length': True,
            'alignment_method': 'trim',
            'fade_in_ms': 10.0,
            'fade_out_ms': 10.0,
            'fade_curve': 'linear',
            'output_format': 'stereo',
            'prevent_clipping': True,
            'headroom_db': -1.0
        }

        # Try to load from YAML
        try:
            import yaml
            config_path = os.path.join(
                os.path.dirname(__file__), '..', '..', '..', 'config', 'audio_config.yaml'
            )
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    if yaml_config and 'audio_mixing' in yaml_config:
                        default_config.update(yaml_config['audio_mixing'])
        except Exception as e:
            logger.warning(f"Could not load mixer config from YAML: {e}")

        # Override with environment variables
        env_mappings = {
            'AUTOVOICE_MIXER_NORMALIZATION': 'normalization_method',
            'AUTOVOICE_MIXER_VOCAL_LEVEL_DB': ('target_vocal_level_db', float),
            'AUTOVOICE_MIXER_INSTRUMENTAL_LEVEL_DB': ('target_instrumental_level_db', float),
            'AUTOVOICE_MIXER_AUTO_ALIGN': ('auto_align_length', lambda x: x.lower() == 'true'),
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

        # Override with constructor config
        if config:
            default_config.update(config)

        return default_config

    def mix(
        self,
        vocals: Union[np.ndarray, str],
        instrumental: Union[np.ndarray, str],
        vocal_volume: float = 1.0,
        instrumental_volume: float = 1.0,
        sample_rate: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """Mix vocals and instrumental tracks.

        This is the primary mixing method. It:
        1. Loads audio if file paths provided
        2. Validates and aligns lengths
        3. Normalizes volumes
        4. Applies target level adjustments
        5. Mixes tracks
        6. Prevents clipping
        7. Converts to output format

        Args:
            vocals: Vocals audio array or file path
            instrumental: Instrumental audio array or file path
            vocal_volume: Volume multiplier for vocals (0.0-2.0)
            instrumental_volume: Volume multiplier for instrumental (0.0-2.0)
            sample_rate: Sample rate (required if inputs are arrays)

        Returns:
            Tuple of (mixed_audio, sample_rate)

        Raises:
            MixingError: If mixing fails
        """
        with self.lock:
            try:
                # Load audio if paths provided
                if isinstance(vocals, str):
                    vocals, vocal_sr = self.audio_processor.load_audio(vocals, return_sr=True)
                    # Convert torch.Tensor to NumPy array
                    if hasattr(vocals, 'detach'):
                        vocals = vocals.detach().cpu().numpy()
                    sample_rate = vocal_sr

                if isinstance(instrumental, str):
                    instrumental, inst_sr = self.audio_processor.load_audio(instrumental, return_sr=True)
                    # Convert torch.Tensor to NumPy array
                    if hasattr(instrumental, 'detach'):
                        instrumental = instrumental.detach().cpu().numpy()
                    if sample_rate is None:
                        sample_rate = inst_sr
                    elif inst_sr != sample_rate:
                        # Resample instrumental to match vocals sample rate
                        instrumental = self._resample_if_needed(instrumental, inst_sr, sample_rate)

                # Convert any remaining tensors to numpy
                if hasattr(vocals, 'detach'):
                    vocals = vocals.detach().cpu().numpy()
                if hasattr(instrumental, 'detach'):
                    instrumental = instrumental.detach().cpu().numpy()

                if sample_rate is None:
                    raise MixingError("sample_rate must be provided for array inputs")

                # Normalize shape to 1D mono before further processing
                # Explicit channel detection: if dimension is 2, average across it
                if vocals.ndim > 1:
                    if vocals.shape[0] == 2:
                        # (2, T) format - average across channels (axis 0)
                        vocals = np.mean(vocals, axis=0)
                    elif vocals.shape[1] == 2:
                        # (T, 2) format - average across channels (axis 1)
                        vocals = np.mean(vocals, axis=1)
                    else:
                        # Ambiguous shape - assume first dimension with smaller size is channels
                        vocals = np.mean(vocals, axis=0) if vocals.shape[0] <= vocals.shape[1] else np.mean(vocals, axis=1)
                vocals = vocals.flatten()

                if instrumental.ndim > 1:
                    if instrumental.shape[0] == 2:
                        # (2, T) format - average across channels (axis 0)
                        instrumental = np.mean(instrumental, axis=0)
                    elif instrumental.shape[1] == 2:
                        # (T, 2) format - average across channels (axis 1)
                        instrumental = np.mean(instrumental, axis=1)
                    else:
                        # Ambiguous shape - assume first dimension with smaller size is channels
                        instrumental = np.mean(instrumental, axis=0) if instrumental.shape[0] <= instrumental.shape[1] else np.mean(instrumental, axis=1)
                instrumental = instrumental.flatten()

                # Validate inputs after conversion
                if vocals.size == 0 or instrumental.size == 0:
                    raise MixingError("Audio inputs cannot be empty")

                # Align lengths
                if self.config['auto_align_length']:
                    vocals, instrumental = self._align_audio_lengths(vocals, instrumental, sample_rate)

                # Normalize individual tracks to initial baseline
                vocals_normalized = self.data_preprocessor.normalize_audio(
                    vocals,
                    method=self.config['normalization_method']
                )
                instrumental_normalized = self.data_preprocessor.normalize_audio(
                    instrumental,
                    method=self.config['normalization_method']
                )

                # Apply target level adjustments first
                vocals_adjusted = self._normalize_to_target_level(
                    vocals_normalized,
                    self.config['target_vocal_level_db'],
                    method='rms'
                )
                instrumental_adjusted = self._normalize_to_target_level(
                    instrumental_normalized,
                    self.config['target_instrumental_level_db'],
                    method='rms'
                )

                # Then apply volume multipliers to preserve user balance settings
                vocals_adjusted = vocals_adjusted * vocal_volume
                instrumental_adjusted = instrumental_adjusted * instrumental_volume

                # Mix tracks
                mixed = vocals_adjusted + instrumental_adjusted

                # Prevent clipping
                if self.config['prevent_clipping']:
                    max_amplitude = np.abs(mixed).max()
                    if max_amplitude > 0:
                        # Target peak based on headroom
                        target_peak = 10 ** (self.config['headroom_db'] / 20.0)
                        if max_amplitude > target_peak:
                            mixed = mixed * (target_peak / max_amplitude)

                # Convert to output format (channel-first convention: (2, T))
                if self.config['output_format'] == 'stereo':
                    mixed = self._convert_to_stereo(mixed)

                logger.info(f"Mixed audio: {len(mixed) if mixed.ndim == 1 else mixed.shape} (channels, samples) at {sample_rate}Hz")

                return mixed, sample_rate

            except Exception as e:
                logger.error(f"Mixing failed: {e}", exc_info=True)
                raise MixingError(f"Failed to mix audio: {e}")

    def mix_with_balance(
        self,
        vocals: np.ndarray,
        instrumental: np.ndarray,
        vocal_balance: float = 0.5,
        sample_rate: int = 44100
    ) -> Tuple[np.ndarray, int]:
        """Mix with balance parameter.

        Alternative mixing method where balance controls the ratio:
        - vocal_balance = 0.0: only instrumental
        - vocal_balance = 0.5: equal mix
        - vocal_balance = 1.0: only vocals

        Args:
            vocals: Vocals audio array
            instrumental: Instrumental audio array
            vocal_balance: Balance between vocals (1.0) and instrumental (0.0)
            sample_rate: Sample rate

        Returns:
            Tuple of (mixed_audio, sample_rate)
        """
        with self.lock:
            try:
                # Ensure mono
                if vocals.ndim > 1:
                    vocals = np.mean(vocals, axis=0)
                if instrumental.ndim > 1:
                    instrumental = np.mean(instrumental, axis=0)

                # Align lengths
                if self.config['auto_align_length']:
                    vocals, instrumental = self._align_audio_lengths(vocals, instrumental, sample_rate)

                # Compute weights
                vocal_weight = np.clip(vocal_balance, 0.0, 1.0)
                instrumental_weight = 1.0 - vocal_weight

                # Mix with weights
                mixed = vocals * vocal_weight + instrumental * instrumental_weight

                # Normalize to prevent clipping
                max_amplitude = np.abs(mixed).max()
                if max_amplitude > 0.95:
                    mixed = mixed * (0.95 / max_amplitude)

                return mixed, sample_rate

            except Exception as e:
                logger.error(f"Balance mixing failed: {e}", exc_info=True)
                raise MixingError(f"Failed to mix with balance: {e}")

    def _apply_crossfade(
        self,
        audio: np.ndarray,
        fade_in_samples: int,
        fade_out_samples: int,
        curve: str = 'linear'
    ) -> np.ndarray:
        """Apply crossfade to audio.

        Args:
            audio: Audio array
            fade_in_samples: Number of samples for fade-in
            fade_out_samples: Number of samples for fade-out
            curve: Fade curve type ('linear', 'cosine', 'exponential')

        Returns:
            Audio with crossfade applied
        """
        audio_len = len(audio)
        result = audio.copy()

        # Apply fade-in
        if fade_in_samples > 0 and fade_in_samples < audio_len:
            if curve == 'cosine':
                # Cosine fade (smoother)
                fade_in = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_in_samples)))
            elif curve == 'exponential':
                # Exponential fade
                fade_in = np.power(np.linspace(0, 1, fade_in_samples), 2)
            else:  # linear
                fade_in = np.linspace(0, 1, fade_in_samples)

            result[:fade_in_samples] *= fade_in

        # Apply fade-out
        if fade_out_samples > 0 and fade_out_samples < audio_len:
            if curve == 'cosine':
                # Cosine fade (smoother)
                fade_out = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_out_samples)))
            elif curve == 'exponential':
                # Exponential fade
                fade_out = np.power(np.linspace(1, 0, fade_out_samples), 2)
            else:  # linear
                fade_out = np.linspace(1, 0, fade_out_samples)

            result[-fade_out_samples:] *= fade_out

        return result

    def _align_audio_lengths(
        self,
        audio1: np.ndarray,
        audio2: np.ndarray,
        sample_rate: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align two audio arrays to same length and apply crossfade if configured.

        Args:
            audio1: First audio array
            audio2: Second audio array
            sample_rate: Sample rate for crossfade calculation

        Returns:
            Tuple of aligned audio arrays
        """
        len1, len2 = len(audio1), len(audio2)

        if len1 == len2:
            result1, result2 = audio1, audio2
        elif self.config['alignment_method'] == 'trim':
            # Trim longer to match shorter
            min_len = min(len1, len2)
            result1, result2 = audio1[:min_len], audio2[:min_len]
        else:  # pad
            # Pad shorter with zeros
            max_len = max(len1, len2)
            if len1 < max_len:
                audio1 = np.pad(audio1, (0, max_len - len1), mode='constant')
            if len2 < max_len:
                audio2 = np.pad(audio2, (0, max_len - len2), mode='constant')
            result1, result2 = audio1, audio2

        # Apply crossfade if configured
        fade_in_ms = self.config.get('fade_in_ms', 0)
        fade_out_ms = self.config.get('fade_out_ms', 0)

        if fade_in_ms > 0 or fade_out_ms > 0:
            # Use provided sample_rate for accurate fade calculation
            fade_in_samples = int(fade_in_ms * sample_rate / 1000)
            fade_out_samples = int(fade_out_ms * sample_rate / 1000)
            curve = self.config.get('fade_curve', 'linear')

            result1 = self._apply_crossfade(result1, fade_in_samples, fade_out_samples, curve)
            result2 = self._apply_crossfade(result2, fade_in_samples, fade_out_samples, curve)

        return result1, result2

    def _normalize_to_target_level(
        self,
        audio: np.ndarray,
        target_db: float,
        method: str = 'rms'
    ) -> np.ndarray:
        """Normalize audio to target dB level.

        Args:
            audio: Audio array
            target_db: Target level in dB
            method: Normalization method ('rms', 'peak', 'lufs')

        Returns:
            Normalized audio array
        """
        if method == 'rms':
            # Compute current RMS
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 0:
                # Compute gain to reach target
                target_amplitude = 10 ** (target_db / 20.0)
                gain = target_amplitude / rms
                return audio * gain
            return audio

        elif method == 'peak':
            # Peak normalization
            peak = np.abs(audio).max()
            if peak > 0:
                target_amplitude = 10 ** (target_db / 20.0)
                gain = target_amplitude / peak
                return audio * gain
            return audio

        elif method == 'lufs':
            # LUFS normalization (simplified)
            # For full LUFS, would need pyloudnorm library
            logger.warning("LUFS normalization not fully implemented, using RMS")
            return self._normalize_to_target_level(audio, target_db, method='rms')

        else:
            logger.warning(f"Unknown normalization method: {method}, using RMS")
            return self._normalize_to_target_level(audio, target_db, method='rms')

    def _resample_if_needed(
        self,
        audio: np.ndarray,
        source_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """Resample audio if source and target sample rates differ.

        Args:
            audio: Audio array to resample
            source_sr: Source sample rate
            target_sr: Target sample rate

        Returns:
            Resampled audio array
        """
        if source_sr == target_sr:
            return audio

        logger.info(f"Resampling audio from {source_sr} Hz to {target_sr} Hz")

        try:
            # Try torchaudio first
            try:
                import torchaudio.transforms as T
                import torch

                # Convert to tensor if needed
                if isinstance(audio, np.ndarray):
                    audio_tensor = torch.from_numpy(audio.astype(np.float32))
                else:
                    audio_tensor = audio

                # Add channel dim if mono
                if audio_tensor.ndim == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)

                resampler = T.Resample(orig_freq=source_sr, new_freq=target_sr)
                resampled = resampler(audio_tensor)

                # Convert back to numpy and remove channel dim
                result = resampled.squeeze(0).numpy()
                logger.debug(f"Resampled using torchaudio: {source_sr} Hz -> {target_sr} Hz")
                return result

            except (ImportError, Exception) as e:
                logger.debug(f"torchaudio resampling failed: {e}, trying librosa")

                # Fallback to librosa
                if LIBROSA_AVAILABLE:
                    result = librosa.resample(
                        audio.astype(np.float32) if isinstance(audio, np.ndarray) else audio,
                        orig_sr=source_sr,
                        target_sr=target_sr
                    )
                    logger.debug(f"Resampled using librosa: {source_sr} Hz -> {target_sr} Hz")
                    return result
                else:
                    logger.warning("Neither torchaudio nor librosa available for resampling")
                    return audio

        except Exception as e:
            logger.error(f"Resampling failed: {e}")
            return audio

    def _convert_to_stereo(self, audio: np.ndarray) -> np.ndarray:
        """Convert mono audio to stereo in (2, T) channel-first format.

        Args:
            audio: Mono audio array (1D)

        Returns:
            Stereo audio array in (2, T) shape (channel-first convention)
        """
        if audio.ndim == 1:
            # Duplicate mono to stereo in (2, T) format (channel-first)
            return np.stack([audio, audio], axis=0)
        elif audio.ndim == 2:
            if audio.shape[0] == 2:
                # Already in (2, T) format
                return audio
            elif audio.shape[1] == 2:
                # Convert from (T, 2) to (2, T)
                return audio.T
            elif audio.shape[0] == 1:
                # Convert (1, T) to (2, T)
                return np.concatenate([audio, audio], axis=0)
            elif audio.shape[1] == 1:
                # Convert (T, 1) to (2, T)
                return np.stack([audio[:, 0], audio[:, 0]], axis=0)

        logger.warning(f"Unexpected audio shape: {audio.shape}, returning as-is")
        return audio

    def separate_and_mix(
        self,
        mixed_audio: np.ndarray,
        new_vocals: np.ndarray,
        sample_rate: int,
        device: Optional[str] = None,
        gpu_manager: Optional[Any] = None,
        separator: Optional[Any] = None
    ) -> np.ndarray:
        """Extract instrumental from mixed audio and mix with new vocals.

        Helper method that separates vocals from mixed audio, then mixes
        the extracted instrumental with new vocals.

        Args:
            mixed_audio: Original mixed audio
            new_vocals: New vocals to mix in
            sample_rate: Sample rate
            device: Optional device string ('cuda', 'cpu', etc.)
            gpu_manager: Optional GPU manager instance
            separator: Optional pre-initialized VocalSeparator instance

        Returns:
            Final mixed audio

        Raises:
            MixingError: If separation or mixing fails
        """
        try:
            from ..audio.source_separator import VocalSeparator

            # Save mixed audio to temp file for separator
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                if SOUNDFILE_AVAILABLE:
                    sf.write(tmp.name, mixed_audio, sample_rate)
                else:
                    raise MixingError("soundfile not available for separation")

                # Use provided separator or create new one with device/gpu_manager
                if separator is not None:
                    sep = separator
                else:
                    sep = VocalSeparator(device=device, gpu_manager=gpu_manager)

                vocals, instrumental = sep.separate_vocals(tmp.name)

                # Clean up temp file
                os.unlink(tmp.name)

            # Mix new vocals with extracted instrumental
            mixed, sr = self.mix(new_vocals, instrumental, sample_rate=sample_rate)
            return mixed

        except Exception as e:
            logger.error(f"Separate and mix failed: {e}", exc_info=True)
            raise MixingError(f"Failed to separate and mix: {e}")
