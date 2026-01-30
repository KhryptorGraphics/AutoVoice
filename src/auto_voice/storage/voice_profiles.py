"""Voice profile storage - file-based CRUD operations."""
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

DEFAULT_PROFILES_DIR = 'data/voice_profiles'


class ProfileNotFoundError(Exception):
    """Raised when a voice profile is not found."""
    pass


class VoiceProfileStore:
    """File-based voice profile storage."""

    def __init__(self, profiles_dir: str = DEFAULT_PROFILES_DIR):
        self.profiles_dir = profiles_dir
        os.makedirs(profiles_dir, exist_ok=True)

    def _profile_path(self, profile_id: str) -> str:
        return os.path.join(self.profiles_dir, f"{profile_id}.json")

    def _embedding_path(self, profile_id: str) -> str:
        return os.path.join(self.profiles_dir, f"{profile_id}.npy")

    def save(self, profile_data: Dict[str, Any]) -> str:
        """Save a voice profile. Returns profile_id."""
        profile_id = profile_data.get('profile_id', str(uuid.uuid4()))
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

        return profile

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
                    profiles.append(profile)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to read profile {fname}: {e}")

        return profiles

    def delete(self, profile_id: str) -> bool:
        """Delete a profile. Returns True if deleted, False if not found."""
        path = self._profile_path(profile_id)
        if not os.path.exists(path):
            return False

        os.remove(path)
        emb_path = self._embedding_path(profile_id)
        if os.path.exists(emb_path):
            os.remove(emb_path)

        logger.info(f"Deleted voice profile: {profile_id}")
        return True

    def exists(self, profile_id: str) -> bool:
        """Check if a profile exists."""
        return os.path.exists(self._profile_path(profile_id))

    def _lora_weights_path(self, profile_id: str) -> str:
        """Get path to LoRA weights file for a profile."""
        return os.path.join(self.profiles_dir, f"{profile_id}_lora_weights.pt")

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

        weights_path = self._lora_weights_path(profile_id)
        torch.save(state_dict, weights_path)
        logger.info(f"Saved LoRA weights for profile {profile_id}")

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
        return os.path.exists(self._lora_weights_path(profile_id))
