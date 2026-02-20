"""Voice profile and training sample data models.

These models represent the core data structures for persistent voice profiles
and accumulated training samples used for continuous learning.
"""

import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
from uuid import uuid4


@dataclass
class VoiceProfile:
    """Represents a user's voice profile for voice conversion.

    Attributes:
        user_id: Unique identifier for the user owning this profile.
        name: Display name for the profile.
        id: Unique identifier for this profile (auto-generated).
        created: Timestamp when profile was created (auto-generated).
        samples_count: Number of training samples accumulated.
        model_version: Current model version trained for this profile.
        speaker_embedding: Optional speaker embedding for diarization matching (512-dim WavLM).
    """

    user_id: str
    name: str
    id: str = field(default_factory=lambda: str(uuid4()))
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    samples_count: int = 0
    model_version: str | None = None
    speaker_embedding: Optional[np.ndarray] = None

    def increment_samples(self, count: int) -> None:
        """Increment the samples count by the given amount."""
        self.samples_count += count

    def set_model_version(self, version: str) -> None:
        """Set the current model version."""
        self.model_version = version

    def set_speaker_embedding(self, embedding: np.ndarray) -> None:
        """Set the speaker embedding for diarization matching.

        Args:
            embedding: Speaker embedding (512-dim WavLM, L2 normalized).
        """
        # Ensure L2 normalization
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        self.speaker_embedding = embedding

    def to_dict(self) -> dict[str, Any]:
        """Serialize profile to dictionary."""
        result = {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "created": self.created.isoformat(),
            "samples_count": self.samples_count,
            "model_version": self.model_version,
        }
        # Serialize embedding as base64-encoded bytes
        if self.speaker_embedding is not None:
            embedding_bytes = self.speaker_embedding.astype(np.float32).tobytes()
            result["speaker_embedding"] = base64.b64encode(embedding_bytes).decode('ascii')
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoiceProfile":
        """Deserialize profile from dictionary."""
        created = data.get("created")
        if isinstance(created, str):
            # Handle ISO format with Z suffix
            if created.endswith("Z"):
                created = created[:-1] + "+00:00"
            created = datetime.fromisoformat(created)

        # Deserialize embedding from base64
        speaker_embedding = None
        if "speaker_embedding" in data and data["speaker_embedding"]:
            embedding_bytes = base64.b64decode(data["speaker_embedding"])
            speaker_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)

        return cls(
            id=data["id"],
            user_id=data["user_id"],
            name=data["name"],
            created=created,
            samples_count=data.get("samples_count", 0),
            model_version=data.get("model_version"),
            speaker_embedding=speaker_embedding,
        )


@dataclass
class TrainingSample:
    """Represents a training sample for voice profile learning.

    Attributes:
        profile_id: ID of the profile this sample belongs to.
        audio_path: Path to the audio file on disk.
        duration_seconds: Duration of the audio in seconds.
        sample_rate: Sample rate of the audio in Hz.
        id: Unique identifier for this sample (auto-generated).
        created: Timestamp when sample was created (auto-generated).
        quality_score: Optional quality score (0.0 to 1.0).
        metadata: Optional additional metadata dictionary.
    """

    profile_id: str
    audio_path: str
    duration_seconds: float
    sample_rate: int
    id: str = field(default_factory=lambda: str(uuid4()))
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    quality_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize sample to dictionary."""
        return {
            "id": self.id,
            "profile_id": self.profile_id,
            "audio_path": self.audio_path,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "created": self.created.isoformat(),
            "quality_score": self.quality_score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingSample":
        """Deserialize sample from dictionary."""
        created = data.get("created")
        if isinstance(created, str):
            # Handle ISO format with Z suffix
            if created.endswith("Z"):
                created = created[:-1] + "+00:00"
            created = datetime.fromisoformat(created)

        return cls(
            id=data["id"],
            profile_id=data["profile_id"],
            audio_path=data["audio_path"],
            duration_seconds=data["duration_seconds"],
            sample_rate=data["sample_rate"],
            created=created,
            quality_score=data.get("quality_score"),
            metadata=data.get("metadata", {}),
        )
