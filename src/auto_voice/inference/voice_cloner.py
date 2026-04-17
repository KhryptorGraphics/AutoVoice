"""Voice cloner - speaker embeddings and profile management."""
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import numpy as np

from ..storage.voice_profiles import VoiceProfileStore, ProfileNotFoundError, TrainingSample

logger = logging.getLogger(__name__)

# Lazy-loaded vocal separator
_vocal_separator = None


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


def _get_vocal_separator(device):
    """Get or create the vocal separator (lazy singleton).

    Args:
        device: PyTorch device (CPU or CUDA) for model inference

    Returns:
        VocalSeparator instance, or None if unavailable
    """
    global _vocal_separator
    if _vocal_separator is None:
        try:
            from ..audio.separation import VocalSeparator
            _vocal_separator = VocalSeparator(device=device)
            logger.info("Vocal separator initialized for automatic voice extraction")
        except Exception as e:
            logger.warning(f"Vocal separator unavailable: {e}. Will use audio as-is.")
            return None
    return _vocal_separator


class VoiceCloner:
    """Creates and manages voice profiles using speaker embeddings."""

    def __init__(
        self,
        device=None,
        profiles_dir: str = 'data/voice_profiles',
        samples_dir: str = 'data/samples',
        auto_separate_vocals: bool = True,
    ):
        """Initialize voice cloner with storage and processing settings.

        Args:
            device: PyTorch device for inference (defaults to CUDA if available)
            profiles_dir: Directory for storing voice profile metadata and embeddings
            auto_separate_vocals: If True, automatically separate vocals from instrumentals
        """
        import torch
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.store = VoiceProfileStore(profiles_dir=profiles_dir, samples_dir=samples_dir)
        self._sample_rate = 16000
        self._min_duration = 3.0  # seconds
        self._auto_separate_vocals = auto_separate_vocals

    def _extract_embedding(self, audio_path: str) -> np.ndarray:
        """Extract speaker embedding from mel-spectrogram statistics.

        Creates a 256-dimensional L2-normalized embedding by computing mean and
        standard deviation across 128 mel-frequency bins. This captures the speaker's
        timbre and spectral characteristics.

        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)

        Returns:
            L2-normalized 256-dim float32 array [mel_mean(128), mel_std(128)]

        Raises:
            InvalidAudioError: If audio file cannot be loaded, is too short (< 3s),
                or is silent (all zeros)
        """
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
        """Create averaged speaker embedding from multiple singing recordings.

        Computes individual embeddings from each audio file and averages them to
        create a more robust speaker representation. Useful when multiple reference
        samples are available.

        Args:
            audio_paths: List of paths to audio files containing target speaker

        Returns:
            L2-normalized 256-dim averaged embedding

        Raises:
            InvalidAudioError: If no audio files provided, any file is invalid,
                or all embeddings are zero
        """
        if not audio_paths:
            raise InvalidAudioError("No audio files provided")
        embeddings = [self._extract_embedding(p) for p in audio_paths]
        avg = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg)
        if norm == 0:
            raise InvalidAudioError("All embeddings are zero")
        return (avg / norm).astype(np.float32)

    def _estimate_vocal_range(self, audio_path: str) -> Dict[str, float]:
        """Estimate vocal range from audio using pitch tracking.

        Uses probabilistic YIN (pYIN) algorithm to track fundamental frequency
        and compute min/max/mean pitch over voiced regions.

        Args:
            audio_path: Path to audio file

        Returns:
            Dict with 'min_hz', 'max_hz', 'mean_hz' keys. Returns default range
            (80-800 Hz) if pitch tracking fails.
        """
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

    def _extract_vocals(self, audio_path: str, profile_id: str) -> Optional[Dict[str, str]]:
        """Extract vocals from audio using Demucs separation.

        Saves both vocals and instrumental tracks permanently for later use in
        training and voice conversion.

        Args:
            audio_path: Path to input audio file
            profile_id: Profile ID for organizing separated tracks

        Returns:
            Dict with 'vocals' and 'instrumental' paths, or None if separation fails.
        """
        if not self._auto_separate_vocals:
            return None

        separator = _get_vocal_separator(self.device)
        if separator is None:
            return None

        try:
            import librosa
            import soundfile as sf

            # Load audio
            audio, sr = librosa.load(audio_path, sr=None, mono=False)
            logger.info(f"Separating vocals from {audio_path} (sr={sr})")

            # Separate vocals and instrumental
            separated = separator.separate(audio, sr)
            vocals = separated['vocals']
            instrumental = separated['instrumental']

            # Create permanent storage directory
            separated_dir = os.path.join(self.store.profiles_dir, '..', 'separated', profile_id)
            separated_dir = os.path.abspath(separated_dir)
            os.makedirs(separated_dir, exist_ok=True)

            # Save vocals and instrumental permanently
            vocals_path = os.path.join(separated_dir, 'vocals.wav')
            instrumental_path = os.path.join(separated_dir, 'instrumental.wav')

            sf.write(vocals_path, vocals, sr)
            sf.write(instrumental_path, instrumental, sr)

            logger.info(f"Saved separated tracks to {separated_dir}")

            return {
                'vocals': vocals_path,
                'instrumental': instrumental_path,
            }

        except Exception as e:
            logger.warning(f"Vocal separation failed, using original audio: {e}")
            return None

    def create_voice_profile(self, audio: str, user_id: Optional[str] = None, name: Optional[str] = None) -> Dict[str, Any]:
        """Create a voice profile from audio file.

        Automatically separates vocals from instrumentals when possible,
        ensuring the voice model is trained only on the vocal track.
        Separated tracks are saved permanently for later use.

        Args:
            audio: Path to reference audio file
            user_id: Optional user identifier
            name: Optional profile name (e.g., artist name)

        Returns:
            Dict with profile_id, user_id, name, audio_duration, vocal_range,
            separated_tracks (if available), and created_at
        """
        if not os.path.exists(audio):
            raise InvalidAudioError(f"Audio file not found: {audio}")

        # Generate profile_id first (needed for separation directory)
        profile_id = str(uuid.uuid4())

        # Extract and save vocals/instrumental (automatic separation)
        separated_tracks = self._extract_vocals(audio, profile_id)

        # Use vocals for embedding if separation succeeded
        if separated_tracks:
            audio_for_embedding = separated_tracks['vocals']
            vocals_extracted = True
        else:
            audio_for_embedding = audio
            vocals_extracted = False

        # Extract embedding from vocals (or original if separation failed)
        embedding = self._extract_embedding(audio_for_embedding)

        # Get audio duration
        try:
            import librosa
            y, sr = librosa.load(audio_for_embedding, sr=self._sample_rate)
            audio_duration = len(y) / sr
        except Exception:
            audio_duration = 0.0

        # Estimate vocal range from vocals
        vocal_range = self._estimate_vocal_range(audio_for_embedding)

        # Build profile with separation paths
        profile_data = {
            'profile_id': profile_id,
            'user_id': user_id,
            'name': name,
            'audio_duration': audio_duration,
            'vocal_range': vocal_range,
            'vocals_extracted': vocals_extracted,
            'separated_tracks': separated_tracks,  # Contains vocals/instrumental paths
            'embedding': embedding,
            'created_at': datetime.now(timezone.utc).isoformat(),
        }

        # Save (note: store.save() pops embedding from dict, so preserve it)
        saved_embedding = embedding  # Keep reference before save mutates dict
        self.store.save(profile_data)

        # Restore embedding that was popped by store.save()
        profile_data['embedding'] = saved_embedding

        separation_note = " (vocals extracted)" if vocals_extracted else ""
        if separated_tracks:
            logger.info(f"Created voice profile {profile_id} (duration={audio_duration:.1f}s){separation_note}")
            logger.info(f"  Vocals: {separated_tracks['vocals']}")
            logger.info(f"  Instrumental: {separated_tracks['instrumental']}")

            # Add as first training sample for progressive training
            source_name = os.path.basename(audio)
            self.store.add_training_sample(
                profile_id=profile_id,
                vocals_path=separated_tracks['vocals'],
                instrumental_path=separated_tracks['instrumental'],
                source_file=source_name,
                duration=audio_duration,
            )
            logger.info(f"  Added as training sample for progressive improvement")
        else:
            logger.info(f"Created voice profile {profile_id} (duration={audio_duration:.1f}s){separation_note}")
        return profile_data

    def load_voice_profile(self, profile_id: str) -> Dict[str, Any]:
        """Load a voice profile by ID.

        Args:
            profile_id: UUID of the voice profile to load

        Returns:
            Dict containing profile metadata and embedding (profile_id, user_id,
            name, audio_duration, vocal_range, embedding, created_at, etc.)

        Raises:
            ProfileNotFoundError: If profile does not exist
        """
        return self.store.load(profile_id)

    def list_voice_profiles(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List voice profiles, optionally filtered by user_id.

        Args:
            user_id: Optional user ID to filter profiles. If None, returns all profiles.

        Returns:
            List of profile dicts (without embedding data for efficiency)
        """
        return self.store.list_profiles(user_id=user_id)

    def delete_voice_profile(self, profile_id: str) -> bool:
        """Delete a voice profile and its associated data.

        Removes profile metadata, embedding file, and any training samples.

        Args:
            profile_id: UUID of the voice profile to delete

        Returns:
            True if profile was deleted, False if profile did not exist
        """
        return self.store.delete(profile_id)

    def compare_embeddings(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """Compute cosine similarity between two speaker embeddings.

        Measures speaker similarity in range [-1, 1], where 1 means identical,
        0 means orthogonal (unrelated), and -1 means opposite.

        Args:
            embedding_a: First speaker embedding (256-dim)
            embedding_b: Second speaker embedding (256-dim)

        Returns:
            Cosine similarity in range [-1, 1]. Returns 0.0 if either embedding
            is zero-vector.
        """
        norm_a = np.linalg.norm(embedding_a)
        norm_b = np.linalg.norm(embedding_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(embedding_a, embedding_b) / (norm_a * norm_b))

    # ─────────────────────────────────────────────────────────────────────────
    # Progressive Training - Add samples from songs/karaoke sessions
    # ─────────────────────────────────────────────────────────────────────────

    def add_vocal_sample(
        self,
        profile_id: str,
        audio_path: str,
        source_name: Optional[str] = None,
    ) -> Optional[TrainingSample]:
        """Add a new vocal sample for progressive model improvement.

        Call this when processing a new song to accumulate training data.
        The vocals will be separated automatically and saved for future
        model retraining.

        Args:
            profile_id: Voice profile to add sample to
            audio_path: Path to audio file (song with vocals)
            source_name: Optional name/description of the source

        Returns:
            TrainingSample if successful, None if separation failed
        """
        if not os.path.exists(audio_path):
            raise InvalidAudioError(f"Audio file not found: {audio_path}")

        # Separate vocals from the audio
        separated_tracks = self._extract_vocals(audio_path, profile_id)

        if not separated_tracks:
            logger.warning(f"Could not separate vocals from {audio_path}")
            return None

        # Get duration
        try:
            import librosa
            y, sr = librosa.load(separated_tracks['vocals'], sr=self._sample_rate)
            duration = len(y) / sr
        except Exception:
            duration = 0.0

        # Add to training samples
        sample = self.store.add_training_sample(
            profile_id=profile_id,
            vocals_path=separated_tracks['vocals'],
            instrumental_path=separated_tracks['instrumental'],
            source_file=source_name or os.path.basename(audio_path),
            duration=duration,
        )

        logger.info(f"Added vocal sample from {source_name or audio_path} to profile {profile_id}")
        return sample

    def get_training_samples(self, profile_id: str) -> List[TrainingSample]:
        """Get all training samples for a profile.

        Args:
            profile_id: Voice profile ID

        Returns:
            List of TrainingSample objects
        """
        return self.store.list_training_samples(profile_id)

    def get_training_audio_paths(self, profile_id: str) -> List[str]:
        """Get all vocal audio paths for training.

        Args:
            profile_id: Voice profile ID

        Returns:
            List of paths to vocals WAV files for training
        """
        return self.store.get_all_vocals_paths(profile_id)

    def get_training_duration(self, profile_id: str) -> float:
        """Get total accumulated training duration.

        Args:
            profile_id: Voice profile ID

        Returns:
            Total duration in seconds
        """
        return self.store.get_total_training_duration(profile_id)
