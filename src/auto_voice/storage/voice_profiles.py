"""Voice profile storage - file-based CRUD operations."""
import json
import logging
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import torch

from auto_voice.runtime_contract import (
    load_packaged_artifact_manifest,
    normalize_reference_audio_entries,
    write_packaged_artifact_manifest,
)
from auto_voice.training.sample_quality import analyze_training_sample, summarize_training_samples

from .paths import (
    resolve_profiles_dir,
    resolve_samples_dir,
    resolve_trained_models_dir,
)

logger = logging.getLogger(__name__)

DEFAULT_PROFILES_DIR = None
DEFAULT_SAMPLES_DIR = None
PROFILE_ROLE_SOURCE_ARTIST = "source_artist"
PROFILE_ROLE_TARGET_USER = "target_user"
FULL_MODEL_TRAINING_UNLOCK_SECONDS = 30 * 60.0


class ProfileNotFoundError(Exception):
    """Raised when a voice profile is not found."""
    pass


class TrainingSample:
    """Represents a training sample for progressive voice model improvement."""

    def __init__(self, sample_id: str, vocals_path: str, instrumental_path: Optional[str] = None,
                 source_file: Optional[str] = None, duration: float = 0.0,
                 created_at: Optional[str] = None,
                 quality_metadata: Optional[Dict[str, Any]] = None,
                 extra_metadata: Optional[Dict[str, Any]] = None):
        self.sample_id = sample_id
        self.vocals_path = vocals_path
        self.instrumental_path = instrumental_path
        self.source_file = source_file
        self.duration = duration
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.quality_metadata = dict(quality_metadata or {})
        self.extra_metadata = dict(extra_metadata or {})

    def to_dict(self) -> Dict[str, Any]:
        data = {
            'sample_id': self.sample_id,
            'vocals_path': self.vocals_path,
            'instrumental_path': self.instrumental_path,
            'source_file': self.source_file,
            'duration': self.duration,
            'created_at': self.created_at,
        }
        if self.quality_metadata:
            data['quality_metadata'] = self.quality_metadata
        data.update(self.extra_metadata)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingSample':
        known_fields = {
            'sample_id',
            'vocals_path',
            'instrumental_path',
            'source_file',
            'duration',
            'created_at',
            'quality_metadata',
            'extra_metadata',
        }
        payload = {k: v for k, v in data.items() if k in known_fields}
        extras = {k: v for k, v in data.items() if k not in known_fields}
        extra_metadata = dict(payload.pop('extra_metadata', {}) or {})
        extra_metadata.update(extras)
        payload['extra_metadata'] = extra_metadata
        return cls(**payload)


class VoiceProfileStore:
    """File-based voice profile storage with progressive training sample support."""

    def __init__(
        self,
        profiles_dir: Optional[str] = DEFAULT_PROFILES_DIR,
        samples_dir: Optional[str] = DEFAULT_SAMPLES_DIR,
        trained_models_dir: Optional[str] = None,
    ):
        profiles_path = resolve_profiles_dir(profiles_dir)
        samples_path = resolve_samples_dir(samples_dir)
        inferred_data_dir = None
        if profiles_dir:
            inferred_data_dir = str(profiles_path.parent)
        elif samples_dir:
            inferred_data_dir = str(samples_path.parent)

        self.profiles_dir = str(profiles_path)
        self.samples_dir = str(samples_path)
        self.trained_models_dir = str(
            resolve_trained_models_dir(
                trained_models_dir,
                data_dir=inferred_data_dir,
            )
        )
        os.makedirs(self.profiles_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)
        os.makedirs(self.trained_models_dir, exist_ok=True)

    def _profile_path(self, profile_id: str) -> str:
        return os.path.join(self.profiles_dir, f"{profile_id}.json")

    def _embedding_path(self, profile_id: str) -> str:
        return os.path.join(self.profiles_dir, f"{profile_id}.npy")

    def _normalize_profile(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure profile metadata exposes the canonical workflow fields."""
        normalized = dict(profile_data)
        profile_id = normalized.get("profile_id")
        has_adapter = False
        has_full_model = False
        runtime_manifest = None
        manifest_path = None
        reference_audio: list[Dict[str, Any]] = normalize_reference_audio_entries(
            normalized.get("reference_audio")
        )

        if profile_id:
            manifest_path = self._artifact_manifest_path(profile_id)
            if os.path.exists(manifest_path):
                try:
                    runtime_manifest = load_packaged_artifact_manifest(manifest_path)
                except Exception as exc:
                    logger.warning("Failed to load runtime artifact manifest for %s: %s", profile_id, exc)
            has_adapter = any(
                os.path.exists(path)
                for path in (
                    self._lora_weights_path(profile_id),
                    self._legacy_lora_weights_path(profile_id),
                )
            )
            has_full_model = os.path.exists(self._full_model_path(profile_id))
            if runtime_manifest:
                artifacts = runtime_manifest.get("artifacts", {})
                has_adapter = has_adapter or bool(artifacts.get("adapter"))
                has_full_model = has_full_model or bool(artifacts.get("full_model"))
            reference_audio = self.get_reference_audio_entries(
                profile_id,
                profile_data=normalized,
                runtime_manifest=runtime_manifest,
            )

        sample_count = normalized.get("training_sample_count")
        total_training_duration = normalized.get("total_training_duration")

        if profile_id:
            samples = self.list_training_samples(profile_id)
            if sample_count is None:
                sample_count = len(samples)
            if total_training_duration is None:
                total_training_duration = sum(sample.duration for sample in samples)
            normalized["training_quality_summary"] = summarize_training_samples(samples)

        sample_count = int(sample_count or normalized.get("sample_count") or 0)
        total_training_duration = float(
            total_training_duration
            or normalized.get("clean_vocal_seconds")
            or 0.0
        )
        clean_vocal_seconds = float(
            normalized.get("clean_vocal_seconds", total_training_duration)
        )
        remaining_seconds = max(
            FULL_MODEL_TRAINING_UNLOCK_SECONDS - clean_vocal_seconds,
            0.0,
        )

        normalized.setdefault("profile_role", PROFILE_ROLE_TARGET_USER)
        normalized.setdefault("created_from", "manual")
        normalized["reference_audio"] = reference_audio
        normalized["reference_audio_paths"] = [entry["path"] for entry in reference_audio]
        normalized["reference_audio_count"] = len(reference_audio)
        normalized["primary_reference_audio_path"] = (
            reference_audio[0]["path"] if reference_audio else None
        )
        normalized["reference_audio_total_duration"] = float(
            sum(float(entry.get("duration_seconds") or 0.0) for entry in reference_audio)
        )
        normalized["training_sample_count"] = sample_count
        normalized["sample_count"] = sample_count
        normalized["total_training_duration"] = total_training_duration
        normalized["clean_vocal_seconds"] = clean_vocal_seconds
        normalized["full_model_unlock_seconds"] = FULL_MODEL_TRAINING_UNLOCK_SECONDS
        normalized["full_model_remaining_seconds"] = remaining_seconds
        normalized["full_model_eligible"] = (
            clean_vocal_seconds >= FULL_MODEL_TRAINING_UNLOCK_SECONDS
        )
        normalized["has_adapter_model"] = bool(has_adapter)
        normalized["has_full_model"] = bool(has_full_model)
        normalized["has_trained_model"] = bool(
            normalized.get("has_trained_model") or has_adapter or has_full_model
        )
        if runtime_manifest:
            normalized["runtime_artifact_manifest_path"] = manifest_path
            normalized["runtime_artifact_pipeline"] = runtime_manifest.get("canonical_pipeline")
            normalized["runtime_artifact_model_family"] = runtime_manifest.get("model_family")
            normalized["runtime_artifact_compatibility_version"] = runtime_manifest.get(
                "compatibility_version"
            )

        if normalized["has_full_model"]:
            normalized["active_model_type"] = "full_model"
        elif normalized["has_trained_model"]:
            normalized.setdefault("active_model_type", "adapter")
        else:
            normalized["active_model_type"] = "base"

        if normalized["has_adapter_model"]:
            normalized.setdefault("selected_adapter", "unified")

        return normalized

    def save(self, profile_data: Dict[str, Any]) -> str:
        """Save a voice profile. Returns profile_id."""
        profile_id = profile_data.get('profile_id', str(uuid.uuid4()))
        profile_data = self._normalize_profile(dict(profile_data))
        profile_data['profile_id'] = profile_id
        profile_data.setdefault('created_at', datetime.now(timezone.utc).isoformat())

        # Save embedding separately as numpy
        embedding = profile_data.pop('embedding', None)
        if embedding is not None:
            if isinstance(embedding, np.ndarray):
                np.save(self._embedding_path(profile_id), embedding)
            elif isinstance(embedding, list):
                np.save(self._embedding_path(profile_id), np.array(embedding))

        # Save metadata as JSON
        with open(self._profile_path(profile_id), 'w') as f:
            json.dump(profile_data, f, indent=2, default=str)

        logger.info(f"Saved voice profile: {profile_id}")
        return profile_id

    def load(self, profile_id: str) -> Dict[str, Any]:
        """Load a voice profile by ID. Raises ProfileNotFoundError if not found."""
        path = self._profile_path(profile_id)
        if not os.path.exists(path):
            raise ProfileNotFoundError(f"Profile {profile_id} not found")

        with open(path) as f:
            profile = json.load(f)

        # Load embedding if exists
        emb_path = self._embedding_path(profile_id)
        if os.path.exists(emb_path):
            profile['embedding'] = np.load(emb_path)

        return self._normalize_profile(profile)

    def list_profiles(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all profiles, optionally filtered by user_id."""
        profiles = []
        if not os.path.exists(self.profiles_dir):
            return profiles

        for fname in os.listdir(self.profiles_dir):
            if not fname.endswith('.json'):
                continue
            try:
                with open(os.path.join(self.profiles_dir, fname)) as f:
                    profile = json.load(f)
                if user_id is None or profile.get('user_id') == user_id:
                    profiles.append(self._normalize_profile(profile))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to read profile {fname}: {e}")

        return profiles

    def delete(self, profile_id: str) -> bool:
        """Delete a profile and profile-owned samples/artifacts."""
        path = self._profile_path(profile_id)
        if not os.path.exists(path):
            return False

        os.remove(path)
        emb_path = self._embedding_path(profile_id)
        if os.path.exists(emb_path):
            os.remove(emb_path)

        samples_path = self._samples_dir_for_profile(profile_id)
        if os.path.exists(samples_path):
            shutil.rmtree(samples_path)

        self._delete_profile_artifacts(profile_id)

        logger.info(f"Deleted voice profile: {profile_id}")
        return True

    def _delete_profile_artifacts(self, profile_id: str) -> None:
        """Remove trained model artifacts owned by a profile."""
        explicit_paths = {
            Path(self._lora_weights_path(profile_id)),
            Path(self._legacy_lora_weights_path(profile_id)),
            Path(self._full_model_path(profile_id)),
            Path(self.trained_models_dir) / f"{profile_id}_full_model.pth",
            Path(self.trained_models_dir) / profile_id,
        }
        for artifact_path in explicit_paths:
            if artifact_path.is_dir():
                shutil.rmtree(artifact_path)
            elif artifact_path.exists():
                artifact_path.unlink()

        trained_root = Path(self.trained_models_dir)
        if not trained_root.exists():
            return
        for pattern in (
            f"{profile_id}*.engine",
            f"{profile_id}*.plan",
            f"{profile_id}*.trt",
            f"{profile_id}_*.pt",
            f"{profile_id}_*.pth",
        ):
            for artifact_path in trained_root.rglob(pattern):
                if artifact_path.is_dir():
                    shutil.rmtree(artifact_path)
                elif artifact_path.exists():
                    artifact_path.unlink()

    def exists(self, profile_id: str) -> bool:
        """Check if a profile exists."""
        return os.path.exists(self._profile_path(profile_id))

    def _lora_weights_path(self, profile_id: str) -> str:
        """Get path to the canonical adapter file for a profile."""
        return os.path.join(self.trained_models_dir, f"{profile_id}_adapter.pt")

    def _legacy_lora_weights_path(self, profile_id: str) -> str:
        """Get the legacy in-profile-dir adapter path kept for backward compatibility."""
        return os.path.join(self.profiles_dir, f"{profile_id}_lora_weights.pt")

    def _full_model_path(self, profile_id: str) -> str:
        """Get path to a full model checkpoint for a profile."""
        return os.path.join(self.trained_models_dir, f"{profile_id}_full_model.pt")

    def _artifact_manifest_path(self, profile_id: str) -> str:
        """Get path to the canonical runtime artifact manifest for a profile."""
        return os.path.join(self.trained_models_dir, profile_id, "artifact_manifest.json")

    def save_runtime_artifact_manifest(self, profile_id: str, manifest: Dict[str, Any]) -> str:
        """Persist a runtime artifact manifest for a profile."""
        if not self.exists(profile_id):
            raise ValueError(f"Profile {profile_id} not found")
        profile = self.load(profile_id)
        payload = dict(manifest)
        metadata = dict(payload.get("metadata", {}))
        metadata.setdefault("profile_role", profile.get("profile_role", PROFILE_ROLE_TARGET_USER))
        metadata.setdefault("created_from", profile.get("created_from", "manual"))
        metadata["reference_audio"] = profile.get("reference_audio", [])
        payload["metadata"] = metadata
        manifest_path = self._artifact_manifest_path(profile_id)
        write_packaged_artifact_manifest(manifest_path, payload)
        return manifest_path

    def load_runtime_artifact_manifest(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """Load a profile runtime artifact manifest when present."""
        manifest_path = self._artifact_manifest_path(profile_id)
        if not os.path.exists(manifest_path):
            return None
        manifest = load_packaged_artifact_manifest(manifest_path)
        metadata = dict(manifest.get("metadata", {}))
        if "reference_audio" in metadata:
            metadata["reference_audio"] = normalize_reference_audio_entries(
                metadata["reference_audio"]
            )
            manifest["metadata"] = metadata
        return manifest

    def get_reference_audio_entries(
        self,
        profile_id: str,
        *,
        profile_data: Optional[Dict[str, Any]] = None,
        runtime_manifest: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Resolve the canonical reference-audio contract for a profile."""
        entries: List[Dict[str, Any]] = []

        for sample in self.list_training_samples(profile_id):
            entries.append(
                {
                    "path": sample.vocals_path,
                    "source": "training_sample",
                    "sample_id": sample.sample_id,
                    "source_file": sample.source_file,
                    "duration_seconds": sample.duration,
                    "created_at": sample.created_at,
                }
            )

        profile_payload = dict(profile_data or {})
        if not entries:
            entries.extend(profile_payload.get("reference_audio", []))

        separated_tracks = profile_payload.get("separated_tracks", {})
        if isinstance(separated_tracks, dict) and separated_tracks.get("vocals"):
            entries.append(
                {
                    "path": separated_tracks["vocals"],
                    "source": "separated_track",
                    "created_at": profile_payload.get("created_at"),
                }
            )

        manifest_payload = runtime_manifest
        if manifest_payload is None:
            manifest_payload = self.load_runtime_artifact_manifest(profile_id)
        manifest_reference_audio = ((manifest_payload or {}).get("metadata") or {}).get(
            "reference_audio"
        )
        if manifest_reference_audio:
            entries.extend(manifest_reference_audio)

        return normalize_reference_audio_entries(entries, require_exists=True)

    def save_lora_weights(
        self, profile_id: str, state_dict: Dict[str, torch.Tensor]
    ) -> None:
        """Save LoRA adapter weights for a voice profile.

        Args:
            profile_id: ID of the voice profile
            state_dict: Dict of LoRA parameter tensors

        Raises:
            ValueError: If profile does not exist
        """
        if not self.exists(profile_id):
            raise ValueError(f"Profile {profile_id} not found")

        canonical_path = self._lora_weights_path(profile_id)
        legacy_path = self._legacy_lora_weights_path(profile_id)
        torch.save(state_dict, canonical_path)
        torch.save(state_dict, legacy_path)
        logger.info(f"Saved LoRA weights for profile {profile_id} at {canonical_path}")

    def load_lora_weights(self, profile_id: str) -> Dict[str, torch.Tensor]:
        """Load LoRA adapter weights for a voice profile.

        Args:
            profile_id: ID of the voice profile

        Returns:
            Dict of LoRA parameter tensors

        Raises:
            ValueError: If profile does not exist
            FileNotFoundError: If no weights saved for profile
        """
        if not self.exists(profile_id):
            raise ValueError(f"Profile {profile_id} not found")

        weights_path = self._lora_weights_path(profile_id)
        if not os.path.exists(weights_path):
            weights_path = self._legacy_lora_weights_path(profile_id)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"No LoRA weights saved for profile {profile_id}")

        return torch.load(weights_path, map_location="cpu")

    def has_trained_model(self, profile_id: str) -> bool:
        """Check if a profile has trained LoRA weights.

        Args:
            profile_id: ID of the voice profile

        Returns:
            True if weights file exists, False otherwise
        """
        if not self.exists(profile_id):
            return False
        if os.path.exists(self._artifact_manifest_path(profile_id)):
            return True
        return any(
            os.path.exists(path)
            for path in (
                self._lora_weights_path(profile_id),
                self._legacy_lora_weights_path(profile_id),
                self._full_model_path(profile_id),
            )
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Progressive Training Sample Management
    # ─────────────────────────────────────────────────────────────────────────

    def _samples_dir_for_profile(self, profile_id: str) -> str:
        """Get the samples directory for a profile."""
        return os.path.join(self.samples_dir, profile_id)

    def add_training_sample(
        self,
        profile_id: str,
        vocals_path: str,
        instrumental_path: Optional[str] = None,
        source_file: Optional[str] = None,
        duration: float = 0.0,
        quality_metadata: Optional[Dict[str, Any]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> TrainingSample:
        """Add a training sample for progressive model improvement.

        Each time vocals are separated from a song, they can be saved as a
        training sample to improve the voice model through retraining.

        Args:
            profile_id: Voice profile ID
            vocals_path: Path to extracted vocals WAV file
            instrumental_path: Path to instrumental track (optional)
            source_file: Original source filename for reference
            duration: Duration of the vocals in seconds

        Returns:
            TrainingSample object with paths and metadata
        """
        if not self.exists(profile_id):
            raise ProfileNotFoundError(f"Profile {profile_id} not found")

        # Create sample directory structure
        profile_samples_dir = self._samples_dir_for_profile(profile_id)
        os.makedirs(profile_samples_dir, exist_ok=True)

        # Generate sample ID based on count
        existing_samples = self.list_training_samples(profile_id)
        sample_num = len(existing_samples) + 1
        sample_id = f"sample_{sample_num:03d}"
        sample_dir = os.path.join(profile_samples_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        # Copy/move vocals to permanent location
        import shutil
        dest_vocals = os.path.join(sample_dir, 'vocals.wav')
        shutil.copy2(vocals_path, dest_vocals)

        dest_instrumental = None
        if instrumental_path and os.path.exists(instrumental_path):
            dest_instrumental = os.path.join(sample_dir, 'instrumental.wav')
            shutil.copy2(instrumental_path, dest_instrumental)

        analyzed_quality = analyze_training_sample(
            dest_vocals,
            quality_metadata=quality_metadata,
            provenance=(extra_metadata or {}).get("provenance") if isinstance(extra_metadata, dict) else source_file,
        )

        # Create sample metadata
        sample = TrainingSample(
            sample_id=sample_id,
            vocals_path=dest_vocals,
            instrumental_path=dest_instrumental,
            source_file=source_file,
            duration=float(duration or analyzed_quality.get("duration_seconds") or 0.0),
            quality_metadata=analyzed_quality,
            extra_metadata=extra_metadata,
        )

        # Save metadata
        metadata_path = os.path.join(sample_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(sample.to_dict(), f, indent=2)

        # Update profile with sample count
        self._update_sample_count(profile_id)

        logger.info(f"Added training sample {sample_id} to profile {profile_id}")
        return sample

    def list_training_samples(self, profile_id: str) -> List[TrainingSample]:
        """List all training samples for a profile.

        Args:
            profile_id: Voice profile ID

        Returns:
            List of TrainingSample objects sorted by creation time
        """
        samples = []
        profile_samples_dir = self._samples_dir_for_profile(profile_id)

        if not os.path.exists(profile_samples_dir):
            return samples

        for sample_name in sorted(os.listdir(profile_samples_dir)):
            sample_dir = os.path.join(profile_samples_dir, sample_name)
            if not os.path.isdir(sample_dir):
                continue

            metadata_path = os.path.join(sample_dir, 'metadata.json')
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path) as f:
                        data = json.load(f)
                    samples.append(TrainingSample.from_dict(data))
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"Failed to load sample metadata: {e}")

        return samples

    def get_all_vocals_paths(self, profile_id: str) -> List[str]:
        """Get paths to all vocal samples for training.

        Args:
            profile_id: Voice profile ID

        Returns:
            List of paths to vocals WAV files
        """
        samples = self.list_training_samples(profile_id)
        return [s.vocals_path for s in samples if os.path.exists(s.vocals_path)]

    def get_total_training_duration(self, profile_id: str) -> float:
        """Get total duration of all training samples in seconds.

        Args:
            profile_id: Voice profile ID

        Returns:
            Total duration in seconds
        """
        samples = self.list_training_samples(profile_id)
        return sum(s.duration for s in samples)

    def list_trainable_samples(self, profile_id: str) -> List[TrainingSample]:
        """Return samples with explicit pass/warn QA state."""
        samples = self.list_training_samples(profile_id)
        return [
            sample
            for sample in samples
            if (sample.quality_metadata or {}).get("qa_status") in {"pass", "warn"}
        ]

    def get_training_quality_summary(self, profile_id: str) -> Dict[str, Any]:
        """Summarize QA metadata across a profile's training samples."""
        return summarize_training_samples(self.list_training_samples(profile_id))

    def _update_sample_count(self, profile_id: str) -> None:
        """Update the profile with current sample count and total duration."""
        try:
            profile = self.load(profile_id)
            samples = self.list_training_samples(profile_id)
            quality_summary = summarize_training_samples(samples)
            profile['training_sample_count'] = len(samples)
            profile['total_training_duration'] = sum(s.duration for s in samples)
            profile['training_quality_summary'] = quality_summary

            # Re-add embedding for save (since load includes it)
            embedding = profile.pop('embedding', None)
            if embedding is not None:
                profile['embedding'] = embedding

            self.save(profile)
        except Exception as e:
            logger.warning(f"Failed to update sample count: {e}")

    def delete_training_sample(self, profile_id: str, sample_id: str) -> bool:
        """Delete a specific training sample.

        Args:
            profile_id: Voice profile ID
            sample_id: Sample ID to delete

        Returns:
            True if deleted, False if not found
        """
        sample_dir = os.path.join(self._samples_dir_for_profile(profile_id), sample_id)
        if not os.path.exists(sample_dir):
            return False

        import shutil
        shutil.rmtree(sample_dir)
        self._update_sample_count(profile_id)
        logger.info(f"Deleted training sample {sample_id} from profile {profile_id}")
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Speaker Diarization & Embedding Methods
    # ─────────────────────────────────────────────────────────────────────────

    def _speaker_embedding_path(self, profile_id: str) -> str:
        """Get path to speaker embedding file for diarization matching."""
        return os.path.join(self.profiles_dir, f"{profile_id}_speaker_embedding.npy")

    def save_speaker_embedding(
        self,
        profile_id: str,
        embedding: np.ndarray,
    ) -> None:
        """Save speaker embedding for diarization matching.

        Args:
            profile_id: Voice profile ID
            embedding: Speaker embedding (512-dim WavLM)

        Raises:
            ProfileNotFoundError: If profile does not exist
        """
        if not self.exists(profile_id):
            raise ProfileNotFoundError(f"Profile {profile_id} not found")

        # Ensure L2 normalization
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        emb_path = self._speaker_embedding_path(profile_id)
        np.save(emb_path, embedding.astype(np.float32))
        logger.info(f"Saved speaker embedding for profile {profile_id}")

    def load_speaker_embedding(self, profile_id: str) -> Optional[np.ndarray]:
        """Load speaker embedding for diarization matching.

        Args:
            profile_id: Voice profile ID

        Returns:
            Speaker embedding array or None if not set
        """
        emb_path = self._speaker_embedding_path(profile_id)
        if os.path.exists(emb_path):
            return np.load(emb_path)
        return None

    def get_all_speaker_embeddings(self) -> Dict[str, np.ndarray]:
        """Load speaker embeddings for all profiles.

        Returns:
            Dictionary mapping profile_id to speaker embedding
        """
        embeddings = {}
        for profile in self.list_profiles():
            profile_id = profile.get('profile_id')
            if profile_id:
                emb = self.load_speaker_embedding(profile_id)
                if emb is not None:
                    embeddings[profile_id] = emb
        return embeddings

    def match_speaker_embedding(
        self,
        embedding: np.ndarray,
        threshold: float = 0.7,
        profile_role: Optional[str] = None,
    ) -> Optional[str]:
        """Match a speaker embedding to existing profiles.

        Args:
            embedding: Speaker embedding to match (512-dim WavLM)
            threshold: Cosine similarity threshold (0-1)
            profile_role: Optional profile role filter.

        Returns:
            Profile ID of best match, or None if no match above threshold
        """
        matches = self.rank_speaker_embedding_matches(
            embedding,
            profile_role=profile_role,
            limit=1,
        )
        if not matches:
            return None
        best_match = matches[0]
        if float(best_match.get("similarity") or 0.0) < threshold:
            return None
        return str(best_match["profile_id"])

    def rank_speaker_embedding_matches(
        self,
        embedding: np.ndarray,
        profile_role: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return the highest-similarity speaker matches for an embedding.

        Args:
            embedding: Speaker embedding to compare against saved profile embeddings.
            profile_role: Optional profile-role filter.
            limit: Maximum number of matches to return.

        Returns:
            Sorted list of match dictionaries containing profile metadata and similarity.
        """
        normalized_embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(normalized_embedding))
        if norm > 0:
            normalized_embedding = normalized_embedding / norm

        matches: List[Dict[str, Any]] = []
        for profile in self.list_profiles():
            if profile_role and profile.get("profile_role") != profile_role:
                continue

            profile_id = profile.get("profile_id")
            if not profile_id:
                continue

            profile_embedding = self.load_speaker_embedding(profile_id)
            if profile_embedding is None:
                continue

            normalized_profile = np.asarray(profile_embedding, dtype=np.float32).reshape(-1)
            profile_norm = float(np.linalg.norm(normalized_profile))
            if profile_norm > 0:
                normalized_profile = normalized_profile / profile_norm

            min_len = min(len(normalized_embedding), len(normalized_profile))
            if min_len == 0:
                continue

            similarity = float(np.dot(normalized_embedding[:min_len], normalized_profile[:min_len]))
            matches.append({
                "profile_id": profile_id,
                "name": profile.get("name") or profile_id,
                "profile_role": profile.get("profile_role"),
                "similarity": similarity,
                "active_model_type": profile.get("active_model_type"),
                "has_trained_model": profile.get("has_trained_model"),
                "sample_count": profile.get("sample_count", 0),
            })

        matches.sort(key=lambda item: item["similarity"], reverse=True)
        return matches[: max(limit, 0)]

    def create_profile_from_diarization(
        self,
        name: str,
        speaker_embedding: np.ndarray,
        user_id: str = "system",
        audio_segments: Optional[List[str]] = None,
        profile_role: str = PROFILE_ROLE_SOURCE_ARTIST,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new profile from diarization results.

        Args:
            name: Name for the new profile
            speaker_embedding: Speaker embedding from diarization
            user_id: User ID for the profile
            audio_segments: Optional list of audio file paths to add as samples
            profile_role: Profile role for workflow routing
            metadata: Additional metadata to persist into the manifest

        Returns:
            Profile ID of the created profile
        """
        # Create profile
        profile_data = {
            "name": name,
            "user_id": user_id,
            "created_from": "diarization",
            "profile_role": profile_role,
        }
        if metadata:
            profile_data.update(metadata)
        profile_id = self.save(profile_data)

        # Save speaker embedding
        self.save_speaker_embedding(profile_id, speaker_embedding)

        # Add audio segments as training samples
        if audio_segments:
            for audio_path in audio_segments:
                if os.path.exists(audio_path):
                    try:
                        # Get duration using scipy
                        from scipy.io import wavfile
                        sr, data = wavfile.read(audio_path)
                        duration = len(data) / sr

                        self.add_training_sample(
                            profile_id=profile_id,
                            vocals_path=audio_path,
                            duration=duration,
                            source_file=os.path.basename(audio_path),
                        )
                    except Exception as e:
                        logger.warning(f"Failed to add segment {audio_path}: {e}")

        logger.info(f"Created profile '{name}' ({profile_id}) from diarization")
        return profile_id
