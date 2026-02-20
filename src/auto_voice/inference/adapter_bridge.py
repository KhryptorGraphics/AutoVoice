"""LoRA Adapter Bridge for SeedVC Pipeline.

Bridges trained LoRA adapters (from our MLP-based decoder) to work with
SeedVC's in-context learning approach. Since Seed-VC uses reference audio
rather than speaker embeddings, this bridge provides reference audio paths
from voice profiles.

For the original pipeline (realtime/quality), LoRAs directly modify the decoder.
For Seed-VC, we provide reference audio that captures the voice characteristics.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class VoiceReference:
    """Reference audio information for a voice profile."""
    profile_id: str
    profile_name: str
    reference_paths: List[Path]  # Paths to reference audio files
    speaker_embedding: Optional[np.ndarray]  # Pre-computed embedding (for original pipeline)
    lora_path: Optional[Path]  # Path to trained LoRA (for original pipeline)
    total_duration: float  # Total duration of reference audio


class AdapterBridge:
    """Bridge between trained LoRAs and Seed-VC pipeline.

    This bridge serves two purposes:
    1. For original pipelines (realtime/quality): Load and apply LoRA adapters
    2. For Seed-VC pipeline: Provide reference audio paths for in-context learning

    Usage:
        bridge = AdapterBridge()

        # Get reference for Seed-VC
        ref = bridge.get_voice_reference("7da05140-...")
        pipeline.set_reference_audio(ref.reference_paths[0])

        # Or load LoRA for original pipeline
        lora_state = bridge.load_lora("7da05140-...")
        pipeline.apply_adapter(lora_state)
    """

    def __init__(
        self,
        profiles_dir: Union[str, Path] = "data/voice_profiles",
        training_audio_dir: Union[str, Path] = "data/separated_youtube",
        lora_dir: Union[str, Path] = "data/trained_models/hq",
        device: str = "cuda",
    ):
        """Initialize the adapter bridge with profile and LoRA storage locations.

        Loads all voice profile mappings and initializes caches for efficient
        reference audio and LoRA weight access.

        Args:
            profiles_dir: Directory containing voice profile JSON files and embeddings
            training_audio_dir: Directory containing separated vocals for training/reference
            lora_dir: Directory containing trained LoRA checkpoints
            device: Device for loading tensors ('cuda' or 'cpu'). LoRA weights will be
                moved to this device when loaded.
        """
        self.profiles_dir = Path(profiles_dir)
        self.training_audio_dir = Path(training_audio_dir)
        self.lora_dir = Path(lora_dir)
        self.device = torch.device(device)

        # Cache loaded references
        self._reference_cache: Dict[str, VoiceReference] = {}
        self._lora_cache: Dict[str, Dict[str, torch.Tensor]] = {}

        # Profile ID to artist name mapping
        self._profile_to_artist: Dict[str, str] = {}
        self._load_profile_mappings()

        logger.info(f"AdapterBridge initialized with {len(self._profile_to_artist)} profiles")

    def _load_profile_mappings(self) -> None:
        """Load profile ID to artist name mappings from JSON files.

        Scans the profiles directory for JSON files and extracts profile_id and name
        fields. Mappings are stored in _profile_to_artist for use in finding
        reference audio directories. Failures to read individual profiles are logged
        as warnings but do not stop the loading process.
        """
        for profile_file in self.profiles_dir.glob("*.json"):
            try:
                with open(profile_file) as f:
                    data = json.load(f)
                profile_id = data.get("profile_id", profile_file.stem)
                name = data.get("name", "Unknown")
                self._profile_to_artist[profile_id] = name
                logger.debug(f"Loaded profile mapping: {profile_id} -> {name}")
            except Exception as e:
                logger.warning(f"Failed to load profile {profile_file}: {e}")

    def _find_artist_audio_dir(self, profile_id: str) -> Optional[Path]:
        """Find the audio directory for a profile's artist using fuzzy matching.

        Searches for the artist's separated audio directory by normalizing the profile
        name and attempting exact matches, substring matches, and fuzzy matches on
        directory names. This handles variations like spaces vs underscores and
        minor spelling differences (e.g., "Connor" vs "Conor").

        Args:
            profile_id: Voice profile UUID to look up

        Returns:
            Path to artist's audio directory if found, None otherwise
        """
        name = self._profile_to_artist.get(profile_id, "")
        if not name:
            return None

        # Normalize name for directory matching
        normalized = name.lower().replace(" ", "_")

        # Check for exact match
        artist_dir = self.training_audio_dir / normalized
        if artist_dir.exists():
            return artist_dir

        # Try variations
        for subdir in self.training_audio_dir.iterdir():
            if subdir.is_dir():
                subdir_normalized = subdir.name.lower().replace(" ", "_").replace("-", "_")
                # Check if normalized name is contained in directory name
                if normalized in subdir_normalized or subdir_normalized in normalized:
                    return subdir
                # Check if first part of profile name matches first part of directory
                # e.g., "connor" matches "conor_maynard"
                name_parts = normalized.split("_")
                dir_parts = subdir_normalized.split("_")
                if name_parts and dir_parts:
                    # Fuzzy match on first name (handles connor vs conor)
                    if self._fuzzy_match(name_parts[0], dir_parts[0]):
                        return subdir

        return None

    @staticmethod
    def _fuzzy_match(s1: str, s2: str, max_distance: int = 2) -> bool:
        """Simple fuzzy string matching using Levenshtein distance.

        Args:
            s1: First string
            s2: Second string
            max_distance: Maximum allowed edit distance

        Returns:
            True if strings are within max_distance edits of each other
        """
        if not s1 or not s2:
            return False
        if s1 == s2:
            return True
        # Quick length check
        if abs(len(s1) - len(s2)) > max_distance:
            return False

        # Simple Levenshtein distance calculation
        if len(s1) < len(s2):
            s1, s2 = s2, s1

        # Only need to track previous row for space efficiency
        previous = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous[j + 1] + 1
                deletions = current[j] + 1
                substitutions = previous[j] + (c1 != c2)
                current.append(min(insertions, deletions, substitutions))
            previous = current

        return previous[-1] <= max_distance

    def get_voice_reference(
        self,
        profile_id: str,
        max_references: int = 5,
        min_duration: float = 10.0,
    ) -> VoiceReference:
        """Get voice reference information for Seed-VC pipeline.

        Args:
            profile_id: Voice profile UUID
            max_references: Maximum number of reference audio files to return
            min_duration: Minimum duration per reference in seconds

        Returns:
            VoiceReference with paths to reference audio

        Raises:
            ValueError: If profile not found or no audio available
        """
        # Check cache
        if profile_id in self._reference_cache:
            return self._reference_cache[profile_id]

        # Load profile metadata
        profile_file = self.profiles_dir / f"{profile_id}.json"
        if not profile_file.exists():
            raise ValueError(f"Profile not found: {profile_id}")

        with open(profile_file) as f:
            profile_data = json.load(f)

        profile_name = profile_data.get("name", "Unknown")

        # Find artist audio directory
        audio_dir = self._find_artist_audio_dir(profile_id)
        reference_paths = []
        total_duration = 0.0

        if audio_dir:
            # Get vocal files sorted by size (larger = longer = better reference)
            vocal_files = sorted(
                audio_dir.glob("*_vocals.wav"),
                key=lambda p: p.stat().st_size,
                reverse=True
            )

            for vf in vocal_files[:max_references]:
                reference_paths.append(vf)
                # Estimate duration from file size (rough approximation)
                # 44.1kHz * 2 bytes * 1 channel = ~88.2 KB/s
                estimated_duration = vf.stat().st_size / 88200
                total_duration += estimated_duration

        # Load speaker embedding if available
        embedding_file = self.profiles_dir / f"{profile_id}.npy"
        speaker_embedding = None
        if embedding_file.exists():
            speaker_embedding = np.load(embedding_file)

        # Find LoRA path
        lora_path = self.lora_dir / f"{profile_id}_hq_lora.pt"
        if not lora_path.exists():
            lora_path = None

        reference = VoiceReference(
            profile_id=profile_id,
            profile_name=profile_name,
            reference_paths=reference_paths,
            speaker_embedding=speaker_embedding,
            lora_path=lora_path,
            total_duration=total_duration,
        )

        self._reference_cache[profile_id] = reference

        logger.info(
            f"Loaded voice reference for {profile_name}: "
            f"{len(reference_paths)} files, {total_duration:.1f}s total"
        )

        return reference

    def load_lora(
        self,
        profile_id: str,
        use_cache: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Load LoRA weights for the original pipeline.

        Args:
            profile_id: Voice profile UUID
            use_cache: Whether to cache loaded weights

        Returns:
            LoRA state dict

        Raises:
            FileNotFoundError: If LoRA not found
        """
        if use_cache and profile_id in self._lora_cache:
            return self._lora_cache[profile_id]

        lora_path = self.lora_dir / f"{profile_id}_hq_lora.pt"
        if not lora_path.exists():
            raise FileNotFoundError(f"No LoRA found for profile: {profile_id}")

        checkpoint = torch.load(lora_path, map_location=self.device, weights_only=False)

        # Extract LoRA state from checkpoint
        if "lora_state" in checkpoint:
            lora_state = checkpoint["lora_state"]
        else:
            lora_state = checkpoint

        # Move to device
        lora_state = {k: v.to(self.device) for k, v in lora_state.items()}

        if use_cache:
            self._lora_cache[profile_id] = lora_state

        logger.info(f"Loaded LoRA for profile {profile_id}: {len(lora_state)} parameters")

        return lora_state

    def get_lora_metadata(self, profile_id: str) -> Dict:
        """Get metadata from a trained LoRA checkpoint without loading weights.

        Loads only the metadata portion of the checkpoint to retrieve training
        information such as artist name, epoch, loss, precision, and configuration.
        Returns an empty dict if the checkpoint does not exist.

        Args:
            profile_id: Voice profile UUID

        Returns:
            Metadata dict with fields: artist, epoch, loss, precision, status, config.
            Empty dict if checkpoint not found.
        """
        lora_path = self.lora_dir / f"{profile_id}_hq_lora.pt"
        if not lora_path.exists():
            return {}

        checkpoint = torch.load(lora_path, map_location="cpu", weights_only=False)

        return {
            "artist": checkpoint.get("artist", "Unknown"),
            "epoch": checkpoint.get("epoch", 0),
            "loss": float(checkpoint.get("loss", 0)),
            "precision": checkpoint.get("precision", "unknown"),
            "status": checkpoint.get("status", "unknown"),
            "config": checkpoint.get("config", {}),
        }

    def list_available_profiles(self) -> List[Tuple[str, str, bool, bool]]:
        """List all available voice profiles with their capabilities.

        Scans all loaded profiles and checks for the presence of trained LoRA weights
        and reference audio files. Useful for determining which profiles are ready
        for different pipeline types (original pipeline needs LoRA, Seed-VC needs
        reference audio).

        Returns:
            List of tuples, each containing:
                - profile_id (str): Profile UUID
                - name (str): Artist/profile name
                - has_lora (bool): Whether trained LoRA weights exist
                - has_reference_audio (bool): Whether reference vocals are available
        """
        results = []

        for profile_id, name in self._profile_to_artist.items():
            lora_path = self.lora_dir / f"{profile_id}_hq_lora.pt"
            has_lora = lora_path.exists()

            audio_dir = self._find_artist_audio_dir(profile_id)
            has_reference = audio_dir is not None and any(audio_dir.glob("*_vocals.wav"))

            results.append((profile_id, name, has_lora, has_reference))

        return results

    def clear_cache(self) -> None:
        """Clear all cached references and LoRA weights.

        Frees memory by clearing the reference cache and LoRA cache. Use this when
        switching between many different profiles or to force reloading of updated
        files from disk.
        """
        self._reference_cache.clear()
        self._lora_cache.clear()
        logger.info("Adapter bridge cache cleared")


# Singleton instance
_bridge_instance: Optional[AdapterBridge] = None


def get_adapter_bridge() -> AdapterBridge:
    """Get the global AdapterBridge singleton instance.

    Creates the instance on first call with default configuration. Subsequent calls
    return the same instance. Use this for consistent bridge state across the
    application.

    Returns:
        The global AdapterBridge instance
    """
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = AdapterBridge()
    return _bridge_instance
