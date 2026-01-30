"""Model versioning and rollback for voice profile adapters.

Implements:
- Version creation with unique IDs and sequential numbering
- Metadata tracking (metrics, timestamps)
- Rollback to previous versions
- A/B version comparison
- Retention policy (keep last N versions)

Task 4.6: Implement model version management
"""

import json
import logging
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Represents a stored model version."""

    version_id: str
    version_number: int
    profile_id: str
    created_at: datetime
    metrics: Dict[str, float]
    weights_path: Path
    is_active: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "version_id": self.version_id,
            "version_number": self.version_number,
            "profile_id": self.profile_id,
            "created_at": self.created_at.isoformat(),
            "metrics": self.metrics,
            "weights_path": str(self.weights_path),
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        """Deserialize from dictionary."""
        return cls(
            version_id=data["version_id"],
            version_number=data["version_number"],
            profile_id=data["profile_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            metrics=data["metrics"],
            weights_path=Path(data["weights_path"]),
            is_active=data.get("is_active", False),
        )


class ModelVersionManager:
    """Manages model versions for a voice profile."""

    def __init__(
        self,
        profile_id: str,
        storage_dir: Path,
        max_versions: Optional[int] = 10,
    ):
        """Initialize version manager.

        Args:
            profile_id: Unique identifier for the voice profile
            storage_dir: Base directory for version storage
            max_versions: Maximum versions to retain (None = unlimited)
        """
        self.profile_id = profile_id
        self.storage_dir = Path(storage_dir)
        self.max_versions = max_versions
        self._profile_dir = self.storage_dir / profile_id
        self._profile_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._profile_dir / "versions.json"
        self._versions: Dict[str, ModelVersion] = {}
        self._active_version_id: Optional[str] = None
        self._load_index()

    def _load_index(self) -> None:
        """Load version index from disk."""
        if self._index_path.exists():
            with open(self._index_path) as f:
                data = json.load(f)
            self._active_version_id = data.get("active_version_id")
            for v_data in data.get("versions", []):
                version = ModelVersion.from_dict(v_data)
                version.is_active = version.version_id == self._active_version_id
                self._versions[version.version_id] = version

    def _save_index(self) -> None:
        """Save version index to disk."""
        data = {
            "active_version_id": self._active_version_id,
            "versions": [v.to_dict() for v in self._versions.values()],
        }
        with open(self._index_path, "w") as f:
            json.dump(data, f, indent=2)

    def _generate_version_id(self) -> str:
        """Generate a unique version ID."""
        return f"v{len(self._versions) + 1}-{uuid.uuid4().hex[:8]}"

    def _next_version_number(self) -> int:
        """Get the next sequential version number."""
        if not self._versions:
            return 1
        return max(v.version_number for v in self._versions.values()) + 1

    def _enforce_retention(self) -> None:
        """Remove old versions beyond retention limit."""
        if self.max_versions is None or len(self._versions) <= self.max_versions:
            return

        # Sort by version number (oldest first)
        sorted_versions = sorted(
            self._versions.values(), key=lambda v: v.version_number
        )

        # Remove oldest versions beyond limit
        to_remove = len(self._versions) - self.max_versions
        for version in sorted_versions[:to_remove]:
            self._delete_version(version.version_id)

    def _delete_version(self, version_id: str) -> None:
        """Delete a version and its files."""
        if version_id not in self._versions:
            return

        version = self._versions[version_id]
        version_dir = self._profile_dir / version_id

        # Remove directory and files
        if version_dir.exists():
            shutil.rmtree(version_dir)

        del self._versions[version_id]
        logger.debug(f"Deleted version {version_id}")

    def create_version(
        self,
        weights: Dict[str, Any],
        metrics: Dict[str, float],
    ) -> ModelVersion:
        """Create a new model version.

        Args:
            weights: Adapter weights to store
            metrics: Training metrics for this version

        Returns:
            The created ModelVersion
        """
        version_id = self._generate_version_id()
        version_number = self._next_version_number()
        version_dir = self._profile_dir / version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save weights
        weights_path = version_dir / "adapter.pt"
        torch.save(weights, weights_path)

        # Save metadata
        metadata = {
            "version_id": version_id,
            "version_number": version_number,
            "profile_id": self.profile_id,
            "created_at": datetime.now().isoformat(),
            "metrics": metrics,
        }
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Create version object
        version = ModelVersion(
            version_id=version_id,
            version_number=version_number,
            profile_id=self.profile_id,
            created_at=datetime.now(),
            metrics=metrics,
            weights_path=weights_path,
            is_active=False,
        )

        self._versions[version_id] = version

        # Set as active if first version or newest
        self._active_version_id = version_id
        version.is_active = True

        # Enforce retention policy
        self._enforce_retention()

        # Save index
        self._save_index()

        logger.info(f"Created version {version_id} (v{version_number})")
        return version

    def list_versions(self, newest_first: bool = True) -> List[ModelVersion]:
        """List all versions for this profile.

        Args:
            newest_first: If True, return newest versions first

        Returns:
            List of ModelVersion objects
        """
        versions = list(self._versions.values())
        versions.sort(key=lambda v: v.version_number, reverse=newest_first)
        return versions

    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get a specific version by ID."""
        return self._versions.get(version_id)

    def get_latest_version(self) -> Optional[ModelVersion]:
        """Get the most recently created version."""
        if not self._versions:
            return None
        return max(self._versions.values(), key=lambda v: v.version_number)

    def get_active_version(self) -> Optional[ModelVersion]:
        """Get the currently active version."""
        if self._active_version_id and self._active_version_id in self._versions:
            return self._versions[self._active_version_id]
        return self.get_latest_version()

    def load_weights(self, version_id: str) -> Dict[str, Any]:
        """Load adapter weights for a version.

        Args:
            version_id: ID of version to load

        Returns:
            Loaded weights dictionary

        Raises:
            ValueError: If version not found
        """
        version = self.get_version(version_id)
        if version is None:
            raise ValueError(f"Version {version_id} not found")

        return torch.load(version.weights_path, map_location="cpu")

    def rollback_to(self, version_id: str) -> ModelVersion:
        """Set a previous version as active.

        Args:
            version_id: ID of version to roll back to

        Returns:
            The now-active version

        Raises:
            ValueError: If version not found
        """
        version = self.get_version(version_id)
        if version is None:
            raise ValueError(f"Version {version_id} not found")

        # Update active version
        if self._active_version_id:
            old_active = self._versions.get(self._active_version_id)
            if old_active:
                old_active.is_active = False

        self._active_version_id = version_id
        version.is_active = True
        self._save_index()

        logger.info(f"Rolled back to version {version_id}")
        return version

    def compare_versions(
        self,
        version_id_1: str,
        version_id_2: str,
    ) -> Dict[str, Dict[str, float]]:
        """Compare metrics between two versions.

        Args:
            version_id_1: First version ID
            version_id_2: Second version ID

        Returns:
            Dict mapping metric names to comparison data
        """
        v1 = self.get_version(version_id_1)
        v2 = self.get_version(version_id_2)

        if v1 is None or v2 is None:
            raise ValueError("One or both versions not found")

        comparison = {}
        all_metrics = set(v1.metrics.keys()) | set(v2.metrics.keys())

        for metric in all_metrics:
            val1 = v1.metrics.get(metric, 0.0)
            val2 = v2.metrics.get(metric, 0.0)
            comparison[metric] = {
                "v1": val1,
                "v2": val2,
                "improvement": val2 - val1,  # Positive if v2 improved over v1
            }

        return comparison

    def compare_with_latest(self, version_id: str) -> Dict[str, Dict[str, float]]:
        """Compare a version with the latest version.

        Args:
            version_id: Version to compare

        Returns:
            Dict mapping metric names to comparison data
        """
        v1 = self.get_version(version_id)
        latest = self.get_latest_version()

        if v1 is None or latest is None:
            raise ValueError("Version not found")

        comparison = {}
        all_metrics = set(v1.metrics.keys()) | set(latest.metrics.keys())

        for metric in all_metrics:
            val1 = v1.metrics.get(metric, 0.0)
            val_latest = latest.metrics.get(metric, 0.0)
            comparison[metric] = {
                "v1": val1,
                "latest": val_latest,
                "improvement": val1 - val_latest,
            }

        return comparison

    def get_best_version(
        self,
        metric: str,
        minimize: bool = True,
    ) -> Optional[ModelVersion]:
        """Find the best version according to a metric.

        Args:
            metric: Metric name to optimize
            minimize: If True, lower is better; if False, higher is better

        Returns:
            The best version, or None if no versions exist
        """
        if not self._versions:
            return None

        # Filter to versions that have this metric
        candidates = [v for v in self._versions.values() if metric in v.metrics]
        if not candidates:
            return None

        if minimize:
            return min(candidates, key=lambda v: v.metrics[metric])
        else:
            return max(candidates, key=lambda v: v.metrics[metric])
