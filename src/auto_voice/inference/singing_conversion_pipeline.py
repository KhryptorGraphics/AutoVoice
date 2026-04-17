"""Singing voice conversion pipeline.

Orchestrates: audio separation -> content encoding -> voice conversion -> vocoder -> mixing.
"""
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


class SeparationError(Exception):
    """Raised when vocal/instrumental separation fails."""
    pass


class ConversionError(Exception):
    """Raised when voice conversion fails."""
    pass


# Preset configurations
PRESETS = {
    'draft': {'n_steps': 10, 'denoise': 0.3},
    'fast': {'n_steps': 20, 'denoise': 0.5},
    'balanced': {'n_steps': 50, 'denoise': 0.7},
    'high': {'n_steps': 100, 'denoise': 0.8},
    'studio': {'n_steps': 200, 'denoise': 0.9},
}


class SingingConversionPipeline:
    """Main voice conversion pipeline for singing audio."""

    def __init__(self, device=None, config: Optional[Dict] = None, voice_cloner=None):
        """Initialize the singing voice conversion pipeline.

        Args:
            device: PyTorch device (cpu/cuda). Auto-detects if None.
            config: Configuration dict with model paths and settings.
                Expected keys: hubert_path, vocoder_path, vocoder_type,
                encoder_backend, encoder_type, conformer_config,
                voice_model_path, speaker_id.
            voice_cloner: Optional VoiceCloner instance for profile loading.
        """
        import torch
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or {}
        self._separator = None
        self._voice_cloner = voice_cloner
        self._sample_rate = 22050
        self._model_manager = None

        logger.info(f"SingingConversionPipeline initialized on {self.device}")

    def _get_separator(self):
        """Lazy-load vocal separator on first use.

        Returns:
            VocalSeparator instance configured for this device.
        """
        if self._separator is None:
            from ..audio.separation import VocalSeparator
            self._separator = VocalSeparator(device=self.device)
            logger.info("Vocal separator loaded")
        return self._separator

    def _separate_vocals(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Separate vocals from instrumental using Demucs model.

        Args:
            audio: Input audio signal (mono or stereo)
            sr: Sample rate of the audio

        Returns:
            Dict with 'vocals' and 'instrumental' keys, each containing
            separated audio arrays at the same sample rate.

        Raises:
            SeparationError: If vocal separation fails
        """
        separator = self._get_separator()
        try:
            return separator.separate(audio, sr)
        except Exception as e:
            raise SeparationError(f"Vocal separation failed: {e}")

    def _get_model_manager(self):
        """Get or create ModelManager and load models from config.

        Lazy-loads ModelManager on first call, then loads models using paths
        from self.config (hubert_path, vocoder_path, voice_model_path, etc).

        Returns:
            ModelManager instance with loaded models ready for inference.

        Raises:
            RuntimeError: If model loading fails or required paths not in config.
        """
        if self._model_manager is None:
            from .model_manager import ModelManager
            self._model_manager = ModelManager(device=self.device, config=self.config)

            # Load models from config paths
            hubert_path = self.config.get('hubert_path')
            vocoder_path = self.config.get('vocoder_path')
            vocoder_type = self.config.get('vocoder_type', 'hifigan')
            encoder_backend = self.config.get('encoder_backend', 'hubert')
            encoder_type = self.config.get('encoder_type', 'linear')
            conformer_config = self.config.get('conformer_config')
            self._model_manager.load(
                hubert_path=hubert_path,
                vocoder_path=vocoder_path,
                vocoder_type=vocoder_type,
                encoder_backend=encoder_backend,
                encoder_type=encoder_type,
                conformer_config=conformer_config,
            )

            # Load voice model if configured
            voice_model_path = self.config.get('voice_model_path')
            speaker_id = self.config.get('speaker_id', 'default')
            if voice_model_path:
                self._model_manager.load_voice_model(voice_model_path, speaker_id)
        return self._model_manager

    def _resolve_target_speaker(self, target_profile_id: str, target_embedding: np.ndarray) -> tuple[str, str]:
        """Resolve which target model should drive conversion for a profile.

        Returns:
            Tuple of (speaker_id, model_type), where model_type is one of
            ``full_model`` or ``adapter``.
        """
        model_manager = self._get_model_manager()

        store = getattr(self._voice_cloner, 'store', None)
        trained_models_dir = getattr(store, 'trained_models_dir', None)
        if trained_models_dir:
            full_model_path = Path(trained_models_dir) / f"{target_profile_id}_full_model.pt"
            if full_model_path.exists():
                model_manager.load_voice_model(
                    str(full_model_path),
                    target_profile_id,
                    speaker_embedding=target_embedding,
                )
                logger.info(
                    "Using dedicated full model for target profile %s from %s",
                    target_profile_id,
                    full_model_path,
                )
                return target_profile_id, 'full_model'

        return self.config.get('speaker_id', 'default'), 'adapter'

    def _convert_voice(self, vocals: np.ndarray, target_embedding: np.ndarray,
                       sr: int, speaker_id: str, preset: str = 'balanced') -> np.ndarray:
        """Convert vocals to target voice using trained So-VITS-SVC model.

        Args:
            vocals: Input vocal audio signal (mono)
            target_embedding: Target speaker embedding (256-dim L2-normalized)
            sr: Sample rate of vocal audio
            preset: Quality preset name (unused, reserved for future use)

        Returns:
            Converted vocal audio at the same sample rate.

        Raises:
            ConversionError: If voice conversion fails or models not loaded.
        """
        try:
            model_manager = self._get_model_manager()
            return model_manager.infer(vocals, speaker_id, target_embedding, sr)
        except RuntimeError as e:
            raise ConversionError(f"Voice conversion failed: {e}")

    def _extract_pitch(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract pitch contour (F0) from audio using librosa pyin.

        Args:
            audio: Input audio signal (mono)
            sr: Sample rate of the audio

        Returns:
            F0 contour array with NaN replaced by 0.0.
            Falls back to zero array if extraction fails.
        """
        try:
            import librosa
            f0, voiced, _ = librosa.pyin(audio, fmin=50, fmax=1100, sr=sr)
            f0 = np.nan_to_num(f0, nan=0.0)
            return f0
        except Exception:
            hop_length = 512
            n_frames = len(audio) // hop_length
            return np.zeros(n_frames)

    def _detect_techniques(self, audio: np.ndarray, sr: int) -> Optional[Dict[str, Any]]:
        """Detect vocal techniques (vibrato, melisma) in audio.

        Args:
            audio: Vocal audio signal
            sr: Sample rate

        Returns:
            Dict with technique information or None if detection fails
        """
        try:
            from ..audio.technique_detector import TechniqueAwarePitchExtractor

            extractor = TechniqueAwarePitchExtractor(sample_rate=sr)
            f0, flags = extractor.extract_with_flags(audio)

            return {
                'f0': f0,
                'technique_flags': flags,
                'has_vibrato': flags.has_vibrato,
                'has_melisma': flags.has_melisma,
                'vibrato_rate': flags.vibrato_rate,
                'vibrato_depth_cents': flags.vibrato_depth_cents,
            }
        except Exception as e:
            logger.warning(f"Technique detection failed: {e}")
            return None

    def convert_song(self, song_path: str, target_profile_id: str,
                     vocal_volume: float = 1.0, instrumental_volume: float = 0.9,
                     pitch_shift: float = 0.0, return_stems: bool = False,
                     preset: str = 'balanced',
                     preserve_techniques: bool = True) -> Dict[str, Any]:
        """Convert a song to target voice.

        Args:
            song_path: Path to input audio file
            target_profile_id: Target voice profile ID
            vocal_volume: Vocal volume multiplier [0.0-2.0]
            instrumental_volume: Instrumental volume [0.0-2.0]
            pitch_shift: Pitch shift in semitones [-12 to 12]
            return_stems: Whether to return separate stems
            preset: Quality preset (draft/fast/balanced/high/studio)
            preserve_techniques: Detect and preserve vocal techniques (vibrato, melisma)

        Returns:
            Dict with: mixed_audio, sample_rate, duration, metadata,
                       f0_contour, f0_original, stems (optional),
                       techniques (if preserve_techniques=True)
        """
        import librosa

        start_time = time.time()

        # Load audio
        if not os.path.exists(song_path):
            raise ConversionError(f"Song file not found: {song_path}")

        try:
            audio, sr = librosa.load(song_path, sr=self._sample_rate, mono=True)
        except Exception as e:
            raise ConversionError(f"Failed to load audio: {e}")

        if len(audio) == 0:
            raise ConversionError("Empty audio file")

        # Load target profile
        cloner = self._voice_cloner
        if cloner is None:
            from .voice_cloner import VoiceCloner
            cloner = VoiceCloner(device=self.device)
        try:
            profile = cloner.load_voice_profile(target_profile_id)
        except Exception as e:
            from ..storage.voice_profiles import ProfileNotFoundError
            raise ProfileNotFoundError(f"Profile {target_profile_id} not found: {e}")

        target_embedding = profile.get('embedding')
        if target_embedding is None:
            raise ConversionError("Profile missing embedding data")
        if isinstance(target_embedding, list):
            target_embedding = np.array(target_embedding)

        speaker_id, model_type = self._resolve_target_speaker(
            target_profile_id,
            target_embedding,
        )

        # Separate vocals and instrumental
        stems = self._separate_vocals(audio, sr)
        vocals = stems['vocals']
        instrumental = stems['instrumental']

        # Extract original pitch
        f0_original = self._extract_pitch(vocals, sr)

        # Detect vocal techniques (vibrato, melisma) if requested
        techniques = None
        if preserve_techniques:
            techniques = self._detect_techniques(vocals, sr)
            if techniques:
                logger.info(
                    f"Techniques detected - vibrato: {techniques['has_vibrato']}, "
                    f"melisma: {techniques['has_melisma']}"
                )

        # Apply pitch shift if requested
        if abs(pitch_shift) > 0.01:
            try:
                vocals = librosa.effects.pitch_shift(
                    vocals, sr=sr, n_steps=pitch_shift
                )
            except Exception as e:
                logger.warning(f"Pitch shift failed: {e}")

        # Convert vocals to target voice
        converted_vocals = self._convert_voice(
            vocals,
            target_embedding,
            sr,
            speaker_id=speaker_id,
            preset=preset,
        )

        # Extract converted pitch
        f0_contour = self._extract_pitch(converted_vocals, sr)

        # Mix with volume adjustments
        converted_vocals = converted_vocals * vocal_volume
        instrumental = instrumental * instrumental_volume

        # Ensure same length
        min_len = min(len(converted_vocals), len(instrumental))
        mixed_audio = converted_vocals[:min_len] + instrumental[:min_len]

        # Normalize to prevent clipping
        peak = np.abs(mixed_audio).max()
        if peak > 0.95:
            mixed_audio = mixed_audio * (0.95 / peak)

        duration = len(mixed_audio) / sr
        elapsed = time.time() - start_time

        result = {
            'mixed_audio': mixed_audio,
            'sample_rate': sr,
            'duration': duration,
            'metadata': {
                'preset': preset,
                'pitch_shift': pitch_shift,
                'vocal_volume': vocal_volume,
                'instrumental_volume': instrumental_volume,
                'processing_time': elapsed,
                'target_profile_id': target_profile_id,
                'active_model_type': model_type,
                'speaker_id': speaker_id,
            },
            'f0_contour': f0_contour,
            'f0_original': f0_original,
        }

        if return_stems:
            result['stems'] = {
                'vocals': converted_vocals[:min_len],
                'instrumental': instrumental[:min_len],
            }

        if techniques:
            # Include technique info (excluding non-serializable flags)
            result['techniques'] = {
                'has_vibrato': techniques['has_vibrato'],
                'has_melisma': techniques['has_melisma'],
                'vibrato_rate': techniques['vibrato_rate'],
                'vibrato_depth_cents': techniques['vibrato_depth_cents'],
            }

        logger.info(
            f"Song conversion complete: {duration:.1f}s audio in {elapsed:.1f}s "
            f"(preset={preset}, profile={target_profile_id}, model_type={model_type})"
        )

        return result
