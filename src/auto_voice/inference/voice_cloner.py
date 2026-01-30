"""Voice cloner - speaker embeddings and profile management."""
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import numpy as np

from ..storage.voice_profiles import VoiceProfileStore, ProfileNotFoundError

logger = logging.getLogger(__name__)


class InvalidAudioError(Exception):
    """Raised when audio input is invalid (corrupt, too short, wrong format)."""
    pass


class InsufficientQualityError(Exception):
    """Raised when audio quality is too low for voice cloning."""
    def __init__(self, message: str, error_code: str = 'insufficient_quality', details: Optional[Dict] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details


class InconsistentSamplesError(Exception):
    """Raised when multiple audio samples are inconsistent (different speakers)."""
    def __init__(self, message: str, error_code: str = 'inconsistent_samples', details: Optional[Dict] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details


class VoiceCloner:
    """Creates and manages voice profiles using speaker embeddings."""

    def __init__(self, device=None, profiles_dir: str = 'data/voice_profiles'):
        import torch
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.store = VoiceProfileStore(profiles_dir)
        self._sample_rate = 16000
        self._min_duration = 3.0  # seconds

    def _extract_embedding(self, audio_path: str) -> np.ndarray:
        """Extract speaker embedding from mel-spectrogram statistics."""
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=self._sample_rate, mono=True)
        except Exception as e:
            raise InvalidAudioError(f"Failed to load audio: {e}")

        duration = len(audio) / sr
        if duration < self._min_duration:
            raise InvalidAudioError(
                f"Audio too short ({duration:.1f}s). Minimum {self._min_duration}s required."
            )

        # Mel-statistics embedding (deterministic, discriminative)
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mean = mel_db.mean(axis=1)   # [128] average spectral shape = timbre
        std = mel_db.std(axis=1)     # [128] spectral variability
        embedding = np.concatenate([mean, std])  # [256]
        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise InvalidAudioError("Silent audio - cannot extract embedding")
        return (embedding / norm).astype(np.float32)

    def create_speaker_embedding(self, audio_paths: List[str]) -> np.ndarray:
        """Create averaged speaker embedding from multiple singing recordings."""
        if not audio_paths:
            raise InvalidAudioError("No audio files provided")
        embeddings = [self._extract_embedding(p) for p in audio_paths]
        avg = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg)
        if norm == 0:
            raise InvalidAudioError("All embeddings are zero")
        return (avg / norm).astype(np.float32)

    def _estimate_vocal_range(self, audio_path: str) -> Dict[str, float]:
        """Estimate vocal range from audio."""
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=self._sample_rate)
            f0, voiced_flag, _ = librosa.pyin(
                audio, fmin=50, fmax=1100, sr=sr
            )
            voiced_f0 = f0[voiced_flag] if voiced_flag is not None else f0[f0 > 0]
            if len(voiced_f0) > 0:
                return {
                    'min_hz': float(np.min(voiced_f0)),
                    'max_hz': float(np.max(voiced_f0)),
                    'mean_hz': float(np.mean(voiced_f0)),
                }
        except Exception as e:
            logger.warning(f"Vocal range estimation failed: {e}")

        return {'min_hz': 80.0, 'max_hz': 800.0, 'mean_hz': 200.0}

    def create_voice_profile(self, audio: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a voice profile from audio file.

        Args:
            audio: Path to reference audio file
            user_id: Optional user identifier

        Returns:
            Dict with profile_id, user_id, audio_duration, vocal_range, created_at
        """
        if not os.path.exists(audio):
            raise InvalidAudioError(f"Audio file not found: {audio}")

        # Extract embedding
        embedding = self._extract_embedding(audio)

        # Get audio duration
        try:
            import librosa
            y, sr = librosa.load(audio, sr=self._sample_rate)
            audio_duration = len(y) / sr
        except Exception:
            audio_duration = 0.0

        # Estimate vocal range
        vocal_range = self._estimate_vocal_range(audio)

        # Build profile
        profile_id = str(uuid.uuid4())
        profile_data = {
            'profile_id': profile_id,
            'user_id': user_id,
            'audio_duration': audio_duration,
            'vocal_range': vocal_range,
            'embedding': embedding,
            'created_at': datetime.now(timezone.utc).isoformat(),
        }

        # Save (note: store.save() pops embedding from dict, so preserve it)
        saved_embedding = embedding  # Keep reference before save mutates dict
        self.store.save(profile_data)

        # Restore embedding that was popped by store.save()
        profile_data['embedding'] = saved_embedding

        logger.info(f"Created voice profile {profile_id} (duration={audio_duration:.1f}s)")
        return profile_data

    def load_voice_profile(self, profile_id: str) -> Dict[str, Any]:
        """Load a voice profile. Raises ProfileNotFoundError if not found."""
        return self.store.load(profile_id)

    def list_voice_profiles(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List voice profiles, optionally filtered by user_id."""
        return self.store.list_profiles(user_id=user_id)

    def delete_voice_profile(self, profile_id: str) -> bool:
        """Delete a voice profile. Returns True if deleted."""
        return self.store.delete(profile_id)

    def compare_embeddings(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """Compute cosine similarity between two speaker embeddings."""
        norm_a = np.linalg.norm(embedding_a)
        norm_b = np.linalg.norm(embedding_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(embedding_a, embedding_b) / (norm_a * norm_b))
