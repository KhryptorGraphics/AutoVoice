"""Voice Identification Pipeline.

Identifies vocalist from audio by matching against existing voice profile embeddings.

Cross-Context Dependencies:
- speaker-diarization_20260130: WavLM embeddings (256-dim)
- voice-profile-training_20260124: Profile management
- training-inference-integration_20260130: AdapterManager

Features:
- Load all profile embeddings
- Compute cosine similarity
- Return best match above threshold (0.85 default)
- Auto-route separated vocals to correct profiles
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class IdentificationResult:
    """Result of voice identification."""
    profile_id: Optional[str]
    profile_name: Optional[str]
    similarity: float
    is_match: bool
    all_similarities: Dict[str, float]


class VoiceIdentifier:
    """Identifies voices by matching embeddings against known profiles.

    Thresholds:
    - speaker_similarity_min: 0.85 (from lora-lifecycle-management spec)
    """

    SIMILARITY_THRESHOLD = 0.85

    def __init__(
        self,
        profiles_dir: Path = Path("data/voice_profiles"),
        device: str = "cuda",
        embedding_model: Optional[str] = None,
    ):
        self.profiles_dir = Path(profiles_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.embedding_model = embedding_model

        self._embeddings: Dict[str, np.ndarray] = {}
        self._profile_names: Dict[str, str] = {}
        self._wavlm_model = None
        self._wavlm_processor = None

        logger.info(f"VoiceIdentifier initialized on {self.device}")

    def load_all_embeddings(self) -> int:
        """Load all profile embeddings from disk.

        Returns:
            Number of embeddings loaded
        """
        self._embeddings.clear()
        self._profile_names.clear()

        count = 0

        # Load UUID-based profile embeddings (.npy files)
        for npy_file in self.profiles_dir.glob("*.npy"):
            profile_id = npy_file.stem
            try:
                embedding = np.load(npy_file)
                self._embeddings[profile_id] = embedding

                # Load profile name from JSON
                json_file = self.profiles_dir / f"{profile_id}.json"
                if json_file.exists():
                    with open(json_file) as f:
                        data = json.load(f)
                        self._profile_names[profile_id] = data.get("name", profile_id)
                else:
                    self._profile_names[profile_id] = profile_id

                count += 1
                logger.debug(f"Loaded embedding for {self._profile_names[profile_id]}")
            except Exception as e:
                logger.warning(f"Failed to load embedding {npy_file}: {e}")

        # Load named artist profiles
        for artist_dir in self.profiles_dir.iterdir():
            if artist_dir.is_dir():
                embedding_file = artist_dir / "speaker_embedding.npy"
                if embedding_file.exists():
                    try:
                        embedding = np.load(embedding_file)
                        profile_id = artist_dir.name
                        self._embeddings[profile_id] = embedding
                        self._profile_names[profile_id] = artist_dir.name.replace("_", " ").title()
                        count += 1
                    except Exception as e:
                        logger.warning(f"Failed to load {embedding_file}: {e}")

        logger.info(f"Loaded {count} profile embeddings")
        return count

    def _load_wavlm(self) -> None:
        """Lazy load WavLM model for embedding extraction."""
        if self._wavlm_model is not None:
            return

        try:
            from transformers import Wav2Vec2FeatureExtractor, WavLMModel

            logger.info("Loading WavLM model for embedding extraction...")
            self._wavlm_processor = Wav2Vec2FeatureExtractor.from_pretrained(
                "microsoft/wavlm-base-plus"
            )
            self._wavlm_model = WavLMModel.from_pretrained(
                "microsoft/wavlm-base-plus"
            ).to(self.device)
            self._wavlm_model.set_output_hidden_states(False)  # Disable to save memory
            logger.info("WavLM model loaded")
        except Exception as e:
            logger.error(f"Failed to load WavLM: {e}")
            raise

    def extract_embedding(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> np.ndarray:
        """Extract speaker embedding from audio.

        Args:
            audio: Audio waveform (float32, -1 to 1)
            sample_rate: Sample rate (should be 16kHz for WavLM)

        Returns:
            256-dim speaker embedding (normalized)
        """
        self._load_wavlm()

        # Ensure 16kHz
        if sample_rate != 16000:
            import torchaudio
            audio_tensor = torch.from_numpy(audio).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            audio_tensor = resampler(audio_tensor)
            audio = audio_tensor.squeeze().numpy()

        # Process through WavLM
        inputs = self._wavlm_processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            outputs = self._wavlm_model(input_values)
            # Use mean pooling over time dimension
            hidden_states = outputs.last_hidden_state
            embedding = hidden_states.mean(dim=1).squeeze()

            # Take first 256 dims for consistency with diarization
            if embedding.shape[0] > 256:
                embedding = embedding[:256]

            # L2 normalize
            embedding = F.normalize(embedding, dim=0)

        return embedding.cpu().numpy()

    def identify(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        threshold: Optional[float] = None
    ) -> IdentificationResult:
        """Identify the speaker in the audio.

        Args:
            audio: Audio waveform (float32)
            sample_rate: Sample rate
            threshold: Similarity threshold (default: 0.85)

        Returns:
            IdentificationResult with match info
        """
        threshold = threshold or self.SIMILARITY_THRESHOLD

        # Ensure embeddings are loaded
        if not self._embeddings:
            self.load_all_embeddings()

        if not self._embeddings:
            return IdentificationResult(
                profile_id=None,
                profile_name=None,
                similarity=0.0,
                is_match=False,
                all_similarities={},
            )

        # Extract embedding from input audio
        query_embedding = self.extract_embedding(audio, sample_rate)

        # Compute similarities against all profiles
        similarities = {}
        best_match_id = None
        best_similarity = 0.0

        for profile_id, profile_embedding in self._embeddings.items():
            # Ensure same shape
            if query_embedding.shape != profile_embedding.shape:
                # Resize to match
                min_len = min(len(query_embedding), len(profile_embedding))
                q = query_embedding[:min_len]
                p = profile_embedding[:min_len]
            else:
                q = query_embedding
                p = profile_embedding

            # Cosine similarity
            norm_product = np.linalg.norm(q) * np.linalg.norm(p) + 1e-8
            similarity = float(np.dot(q, p) / norm_product)
            similarities[profile_id] = similarity

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = profile_id

        # Determine if match
        is_match = best_similarity >= threshold

        return IdentificationResult(
            profile_id=best_match_id if is_match else None,
            profile_name=self._profile_names.get(best_match_id) if is_match else None,
            similarity=best_similarity,
            is_match=is_match,
            all_similarities=similarities,
        )

    def identify_from_file(
        self,
        audio_path: str,
        threshold: Optional[float] = None
    ) -> IdentificationResult:
        """Identify speaker from audio file."""
        import torchaudio

        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        audio = waveform.squeeze().numpy()
        return self.identify(audio, sample_rate, threshold)

    def match_segments_to_profiles(
        self,
        segments: List[Dict[str, Any]],
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Match diarization segments to known profiles.

        Args:
            segments: List of segment dicts with 'audio' or 'embedding' keys
            threshold: Similarity threshold

        Returns:
            Segments with 'profile_id' and 'profile_name' added
        """
        threshold = threshold or self.SIMILARITY_THRESHOLD

        if not self._embeddings:
            self.load_all_embeddings()

        for segment in segments:
            if "embedding" in segment:
                embedding = np.array(segment["embedding"])
            elif "audio" in segment:
                embedding = self.extract_embedding(
                    segment["audio"],
                    segment.get("sample_rate", 16000)
                )
            else:
                continue

            # Find best match
            best_id = None
            best_sim = 0.0

            for profile_id, profile_emb in self._embeddings.items():
                min_len = min(len(embedding), len(profile_emb))
                sim = float(np.dot(embedding[:min_len], profile_emb[:min_len]))
                if sim > best_sim:
                    best_sim = sim
                    best_id = profile_id

            if best_sim >= threshold:
                segment["profile_id"] = best_id
                segment["profile_name"] = self._profile_names.get(best_id, best_id)
                segment["speaker_similarity"] = best_sim
            else:
                segment["profile_id"] = None
                segment["profile_name"] = "Unknown"
                segment["speaker_similarity"] = best_sim

        return segments

    def get_loaded_profiles(self) -> List[Tuple[str, str]]:
        """Get list of loaded profiles.

        Returns:
            List of (profile_id, profile_name) tuples
        """
        return [(pid, self._profile_names.get(pid, pid)) for pid in self._embeddings.keys()]

    def create_profile_from_segment(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        youtube_metadata: Optional[Dict[str, Any]] = None,
        source_file: Optional[str] = None,
    ) -> str:
        """Create a new voice profile from an unknown speaker segment.

        Phase 3: Automatic Profile Creation
        - Generate profile name from YouTube metadata
        - Extract and save speaker embedding
        - Set status="needs_training"
        - Queue for training when samples >= 5

        Cross-context dependencies:
        - voice-profile-training_20260124: VoiceProfileStore
        - speaker-diarization_20260130: WavLM embeddings

        Args:
            audio: Audio waveform for the speaker segment
            sample_rate: Sample rate of audio
            youtube_metadata: Optional YouTube metadata dict with 'title', 'description'
            source_file: Optional source filename for reference

        Returns:
            profile_id of created profile

        Raises:
            RuntimeError: If profile creation fails
        """
        from auto_voice.storage.voice_profiles import VoiceProfileStore
        from auto_voice.audio.youtube_metadata import extract_main_artist, parse_featured_artists

        try:
            # Extract speaker embedding
            embedding = self.extract_embedding(audio, sample_rate)

            # Generate profile name from metadata
            profile_name = self._generate_profile_name(youtube_metadata)

            # Create profile
            store = VoiceProfileStore()

            profile_data = {
                "name": profile_name,
                "user_id": "system",
                "created_from": "auto_identification",
                "status": "needs_training",
                "training_sample_count": 0,
            }

            profile_id = store.save(profile_data)

            # Save speaker embedding
            store.save_speaker_embedding(profile_id, embedding)

            # Add this segment as first training sample
            if source_file:
                import tempfile
                import soundfile as sf

                with tempfile.NamedTemporaryFile(
                    suffix='.wav', delete=False, mode='wb'
                ) as tmp_file:
                    sf.write(tmp_file.name, audio, sample_rate)

                    duration = len(audio) / sample_rate
                    store.add_training_sample(
                        profile_id=profile_id,
                        vocals_path=tmp_file.name,
                        source_file=source_file,
                        duration=duration,
                    )

            # Update local cache
            self._embeddings[profile_id] = embedding
            self._profile_names[profile_id] = profile_name

            logger.info(
                f"Created profile '{profile_name}' ({profile_id}) from unknown speaker. "
                f"Status: needs_training"
            )

            return profile_id

        except Exception as e:
            logger.error(f"Failed to create profile from segment: {e}", exc_info=True)
            raise RuntimeError(f"Profile creation failed: {e}") from e

    def _generate_profile_name(
        self,
        youtube_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate profile name from YouTube metadata or default pattern.

        Args:
            youtube_metadata: Optional dict with 'title', 'description', 'uploader'

        Returns:
            Generated profile name
        """
        from auto_voice.audio.youtube_metadata import extract_main_artist, parse_featured_artists

        if youtube_metadata:
            title = youtube_metadata.get('title', '')
            description = youtube_metadata.get('description', '')

            # Try extracting main artist
            main_artist = extract_main_artist(title)
            if main_artist:
                return main_artist

            # Try featured artists
            featured = parse_featured_artists(title, description)
            if featured:
                return featured[0]

            # Try uploader/channel
            uploader = youtube_metadata.get('uploader') or youtube_metadata.get('channel')
            if uploader:
                return uploader

        # Fall back to Speaker_N pattern
        # Count existing Speaker_N profiles
        speaker_count = sum(
            1 for name in self._profile_names.values()
            if name.startswith("Speaker_")
        )
        return f"Speaker_{speaker_count + 1}"

    def identify_or_create(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        threshold: Optional[float] = None,
        youtube_metadata: Optional[Dict[str, Any]] = None,
        source_file: Optional[str] = None,
    ) -> IdentificationResult:
        """Identify speaker or create new profile if unknown.

        Phase 3: Auto-route unknown speakers to profile creation.

        Args:
            audio: Audio waveform
            sample_rate: Sample rate
            threshold: Similarity threshold (default: 0.85)
            youtube_metadata: Optional YouTube metadata for naming
            source_file: Optional source filename

        Returns:
            IdentificationResult with profile_id set (existing or newly created)
        """
        # Try identification first
        result = self.identify(audio, sample_rate, threshold)

        if result.is_match:
            return result

        # Unknown speaker - create new profile
        logger.info(
            f"Unknown speaker detected (best_similarity={result.similarity:.3f}). "
            "Creating new profile..."
        )

        profile_id = self.create_profile_from_segment(
            audio=audio,
            sample_rate=sample_rate,
            youtube_metadata=youtube_metadata,
            source_file=source_file,
        )

        # Return result with new profile
        return IdentificationResult(
            profile_id=profile_id,
            profile_name=self._profile_names.get(profile_id, profile_id),
            similarity=1.0,
            is_match=True,
            all_similarities={profile_id: 1.0},
        )


# Global instance
_global_identifier: Optional[VoiceIdentifier] = None


def get_voice_identifier() -> VoiceIdentifier:
    """Get or create global VoiceIdentifier instance."""
    global _global_identifier
    if _global_identifier is None:
        _global_identifier = VoiceIdentifier()
        _global_identifier.load_all_embeddings()
    return _global_identifier
