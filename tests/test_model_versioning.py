"""TDD tests for model versioning and rollback (Task 4.5).

Tests cover:
- Model version creation and storage
- Version metadata tracking
- Rollback to previous versions
- A/B version comparison
- Version retention policy (keep last N)
"""

import json
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


# === Fixtures ===


@pytest.fixture
def temp_version_storage():
    """Temporary directory for version storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_model():
    """Simple mock model for versioning tests."""

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(128, 64)

        def forward(self, x):
            return self.fc(x)

    return SimpleModel()


@pytest.fixture
def mock_adapter_weights():
    """Mock LoRA adapter weights."""
    return {
        "adapters": {
            "fc": {
                "lora_A": torch.randn(8, 128),
                "lora_B": torch.randn(64, 8),
                "rank": 8,
                "alpha": 16,
            }
        },
        "config": {
            "rank": 8,
            "alpha": 16,
            "target_modules": ["fc"],
        },
    }


# === Test Classes ===


class TestModelVersionCreation:
    """Tests for creating and storing model versions."""

    def test_create_version_generates_unique_id(self, temp_version_storage, mock_adapter_weights):
        """Each version should have a unique identifier."""
        from auto_voice.training.model_versioning import ModelVersionManager

        manager = ModelVersionManager(
            profile_id="test-profile",
            storage_dir=temp_version_storage,
        )

        v1 = manager.create_version(mock_adapter_weights, metrics={"loss": 0.5})
        v2 = manager.create_version(mock_adapter_weights, metrics={"loss": 0.4})

        assert v1.version_id != v2.version_id

    def test_create_version_stores_weights(self, temp_version_storage, mock_adapter_weights):
        """Version weights should be persisted to storage."""
        from auto_voice.training.model_versioning import ModelVersionManager

        manager = ModelVersionManager(
            profile_id="test-profile",
            storage_dir=temp_version_storage,
        )

        version = manager.create_version(mock_adapter_weights, metrics={"loss": 0.5})

        # Verify weights file exists
        weights_path = temp_version_storage / "test-profile" / version.version_id / "adapter.pt"
        assert weights_path.exists()

    def test_create_version_stores_metadata(self, temp_version_storage, mock_adapter_weights):
        """Version metadata (metrics, timestamp) should be stored."""
        from auto_voice.training.model_versioning import ModelVersionManager

        manager = ModelVersionManager(
            profile_id="test-profile",
            storage_dir=temp_version_storage,
        )

        metrics = {"loss": 0.5, "speaker_similarity": 0.85}
        version = manager.create_version(mock_adapter_weights, metrics=metrics)

        # Verify metadata file exists and contains expected data
        metadata_path = temp_version_storage / "test-profile" / version.version_id / "metadata.json"
        assert metadata_path.exists()

        with open(metadata_path) as f:
            stored_metadata = json.load(f)

        assert stored_metadata["metrics"] == metrics
        assert "created_at" in stored_metadata

    def test_create_version_increments_number(self, temp_version_storage, mock_adapter_weights):
        """Version numbers should increment sequentially."""
        from auto_voice.training.model_versioning import ModelVersionManager

        manager = ModelVersionManager(
            profile_id="test-profile",
            storage_dir=temp_version_storage,
        )

        v1 = manager.create_version(mock_adapter_weights, metrics={"loss": 0.5})
        v2 = manager.create_version(mock_adapter_weights, metrics={"loss": 0.4})
        v3 = manager.create_version(mock_adapter_weights, metrics={"loss": 0.3})

        assert v1.version_number == 1
        assert v2.version_number == 2
        assert v3.version_number == 3


class TestVersionRetrieval:
    """Tests for retrieving stored versions."""

    def test_list_versions_returns_all(self, temp_version_storage, mock_adapter_weights):
        """List all versions for a profile."""
        from auto_voice.training.model_versioning import ModelVersionManager

        manager = ModelVersionManager(
            profile_id="test-profile",
            storage_dir=temp_version_storage,
        )

        manager.create_version(mock_adapter_weights, metrics={"loss": 0.5})
        manager.create_version(mock_adapter_weights, metrics={"loss": 0.4})
        manager.create_version(mock_adapter_weights, metrics={"loss": 0.3})

        versions = manager.list_versions()
        assert len(versions) == 3

    def test_list_versions_ordered_by_number(self, temp_version_storage, mock_adapter_weights):
        """Versions should be returned in order (newest first by default)."""
        from auto_voice.training.model_versioning import ModelVersionManager

        manager = ModelVersionManager(
            profile_id="test-profile",
            storage_dir=temp_version_storage,
        )

        manager.create_version(mock_adapter_weights, metrics={"loss": 0.5})
        manager.create_version(mock_adapter_weights, metrics={"loss": 0.4})
        manager.create_version(mock_adapter_weights, metrics={"loss": 0.3})

        versions = manager.list_versions()
        assert versions[0].version_number == 3  # Newest first
        assert versions[2].version_number == 1

    def test_get_version_by_id(self, temp_version_storage, mock_adapter_weights):
        """Retrieve a specific version by its ID."""
        from auto_voice.training.model_versioning import ModelVersionManager

        manager = ModelVersionManager(
            profile_id="test-profile",
            storage_dir=temp_version_storage,
        )

        created = manager.create_version(mock_adapter_weights, metrics={"loss": 0.5})
        retrieved = manager.get_version(created.version_id)

        assert retrieved is not None
        assert retrieved.version_id == created.version_id
        assert retrieved.metrics["loss"] == 0.5

    def test_get_latest_version(self, temp_version_storage, mock_adapter_weights):
        """Retrieve the most recent version."""
        from auto_voice.training.model_versioning import ModelVersionManager

        manager = ModelVersionManager(
            profile_id="test-profile",
            storage_dir=temp_version_storage,
        )

        manager.create_version(mock_adapter_weights, metrics={"loss": 0.5})
        manager.create_version(mock_adapter_weights, metrics={"loss": 0.4})
        latest = manager.create_version(mock_adapter_weights, metrics={"loss": 0.3})

        retrieved = manager.get_latest_version()
        assert retrieved.version_id == latest.version_id

    def test_load_version_weights(self, temp_version_storage, mock_adapter_weights):
        """Load adapter weights from a stored version."""
        from auto_voice.training.model_versioning import ModelVersionManager

        manager = ModelVersionManager(
            profile_id="test-profile",
            storage_dir=temp_version_storage,
        )

        version = manager.create_version(mock_adapter_weights, metrics={"loss": 0.5})
        loaded_weights = manager.load_weights(version.version_id)

        assert "adapters" in loaded_weights
        assert "fc" in loaded_weights["adapters"]


class TestVersionRollback:
    """Tests for rolling back to previous versions."""

    def test_rollback_sets_active_version(self, temp_version_storage, mock_adapter_weights):
        """Rollback should update the active version pointer."""
        from auto_voice.training.model_versioning import ModelVersionManager

        manager = ModelVersionManager(
            profile_id="test-profile",
            storage_dir=temp_version_storage,
        )

        v1 = manager.create_version(mock_adapter_weights, metrics={"loss": 0.5})
        v2 = manager.create_version(mock_adapter_weights, metrics={"loss": 0.4})
        v3 = manager.create_version(mock_adapter_weights, metrics={"loss": 0.3})

        # Active should be v3 (latest)
        assert manager.get_active_version().version_id == v3.version_id

        # Rollback to v1
        manager.rollback_to(v1.version_id)
        assert manager.get_active_version().version_id == v1.version_id

    def test_rollback_preserves_versions(self, temp_version_storage, mock_adapter_weights):
        """Rollback should not delete newer versions."""
        from auto_voice.training.model_versioning import ModelVersionManager

        manager = ModelVersionManager(
            profile_id="test-profile",
            storage_dir=temp_version_storage,
        )

        v1 = manager.create_version(mock_adapter_weights, metrics={"loss": 0.5})
        v2 = manager.create_version(mock_adapter_weights, metrics={"loss": 0.4})
        v3 = manager.create_version(mock_adapter_weights, metrics={"loss": 0.3})

        manager.rollback_to(v1.version_id)

        # All versions should still exist
        versions = manager.list_versions()
        assert len(versions) == 3

    def test_rollback_to_nonexistent_raises(self, temp_version_storage, mock_adapter_weights):
        """Rollback to non-existent version should raise error."""
        from auto_voice.training.model_versioning import ModelVersionManager

        manager = ModelVersionManager(
            profile_id="test-profile",
            storage_dir=temp_version_storage,
        )

        manager.create_version(mock_adapter_weights, metrics={"loss": 0.5})

        with pytest.raises(ValueError, match="not found"):
            manager.rollback_to("nonexistent-version-id")


class TestVersionRetentionPolicy:
    """Tests for version retention (keep last N versions)."""

    def test_retention_deletes_old_versions(self, temp_version_storage, mock_adapter_weights):
        """Old versions beyond retention limit should be deleted."""
        from auto_voice.training.model_versioning import ModelVersionManager

        manager = ModelVersionManager(
            profile_id="test-profile",
            storage_dir=temp_version_storage,
            max_versions=3,
        )

        v1 = manager.create_version(mock_adapter_weights, metrics={"loss": 0.5})
        v2 = manager.create_version(mock_adapter_weights, metrics={"loss": 0.4})
        v3 = manager.create_version(mock_adapter_weights, metrics={"loss": 0.3})
        v4 = manager.create_version(mock_adapter_weights, metrics={"loss": 0.2})

        versions = manager.list_versions()
        assert len(versions) == 3

        # v1 should be deleted, v2-v4 retained
        version_ids = [v.version_id for v in versions]
        assert v1.version_id not in version_ids
        assert v4.version_id in version_ids

    def test_retention_removes_weights_files(self, temp_version_storage, mock_adapter_weights):
        """Deleted versions should have their weight files removed."""
        from auto_voice.training.model_versioning import ModelVersionManager

        manager = ModelVersionManager(
            profile_id="test-profile",
            storage_dir=temp_version_storage,
            max_versions=2,
        )

        v1 = manager.create_version(mock_adapter_weights, metrics={"loss": 0.5})
        v1_path = temp_version_storage / "test-profile" / v1.version_id

        manager.create_version(mock_adapter_weights, metrics={"loss": 0.4})
        manager.create_version(mock_adapter_weights, metrics={"loss": 0.3})

        # v1's directory should be deleted
        assert not v1_path.exists()

    def test_no_retention_limit_keeps_all(self, temp_version_storage, mock_adapter_weights):
        """With no limit, all versions should be retained."""
        from auto_voice.training.model_versioning import ModelVersionManager

        manager = ModelVersionManager(
            profile_id="test-profile",
            storage_dir=temp_version_storage,
            max_versions=None,  # No limit
        )

        for i in range(10):
            manager.create_version(mock_adapter_weights, metrics={"loss": 0.5 - i * 0.05})

        versions = manager.list_versions()
        assert len(versions) == 10


class TestVersionComparison:
    """Tests for A/B version comparison."""

    def test_compare_versions_by_metric(self, temp_version_storage, mock_adapter_weights):
        """Compare two versions by a specific metric."""
        from auto_voice.training.model_versioning import ModelVersionManager

        manager = ModelVersionManager(
            profile_id="test-profile",
            storage_dir=temp_version_storage,
        )

        v1 = manager.create_version(mock_adapter_weights, metrics={"loss": 0.5, "similarity": 0.80})
        v2 = manager.create_version(mock_adapter_weights, metrics={"loss": 0.3, "similarity": 0.85})

        comparison = manager.compare_versions(v1.version_id, v2.version_id)

        assert comparison["loss"]["v1"] == 0.5
        assert comparison["loss"]["v2"] == 0.3
        # improvement = v2 - v1: negative means v2 decreased the metric
        assert comparison["loss"]["improvement"] == pytest.approx(-0.2)
        assert comparison["similarity"]["improvement"] == pytest.approx(0.05)

    def test_compare_with_latest(self, temp_version_storage, mock_adapter_weights):
        """Compare a version with the latest version."""
        from auto_voice.training.model_versioning import ModelVersionManager

        manager = ModelVersionManager(
            profile_id="test-profile",
            storage_dir=temp_version_storage,
        )

        v1 = manager.create_version(mock_adapter_weights, metrics={"loss": 0.5})
        manager.create_version(mock_adapter_weights, metrics={"loss": 0.4})
        v3 = manager.create_version(mock_adapter_weights, metrics={"loss": 0.3})

        comparison = manager.compare_with_latest(v1.version_id)

        assert comparison["loss"]["v1"] == 0.5
        assert comparison["loss"]["latest"] == 0.3

    def test_get_best_version_by_metric(self, temp_version_storage, mock_adapter_weights):
        """Find the best version according to a metric."""
        from auto_voice.training.model_versioning import ModelVersionManager

        manager = ModelVersionManager(
            profile_id="test-profile",
            storage_dir=temp_version_storage,
        )

        manager.create_version(mock_adapter_weights, metrics={"loss": 0.5, "similarity": 0.80})
        v2 = manager.create_version(mock_adapter_weights, metrics={"loss": 0.3, "similarity": 0.90})
        manager.create_version(mock_adapter_weights, metrics={"loss": 0.4, "similarity": 0.85})

        # Best by loss (lower is better)
        best_loss = manager.get_best_version(metric="loss", minimize=True)
        assert best_loss.version_id == v2.version_id

        # Best by similarity (higher is better)
        best_sim = manager.get_best_version(metric="similarity", minimize=False)
        assert best_sim.version_id == v2.version_id


class TestModelVersionModel:
    """Tests for the ModelVersion data class."""

    def test_model_version_fields(self, temp_version_storage):
        """ModelVersion should have required fields."""
        from auto_voice.training.model_versioning import ModelVersion

        version = ModelVersion(
            version_id="v1-abc123",
            version_number=1,
            profile_id="test-profile",
            created_at=datetime.now(),
            metrics={"loss": 0.5},
            weights_path=temp_version_storage / "weights.pt",
        )

        assert version.version_id == "v1-abc123"
        assert version.version_number == 1
        assert version.profile_id == "test-profile"
        assert version.metrics["loss"] == 0.5

    def test_model_version_to_dict(self, temp_version_storage):
        """ModelVersion should serialize to dict."""
        from auto_voice.training.model_versioning import ModelVersion

        now = datetime.now()
        version = ModelVersion(
            version_id="v1-abc123",
            version_number=1,
            profile_id="test-profile",
            created_at=now,
            metrics={"loss": 0.5},
            weights_path=temp_version_storage / "weights.pt",
        )

        d = version.to_dict()
        assert d["version_id"] == "v1-abc123"
        assert d["version_number"] == 1
        assert d["metrics"]["loss"] == 0.5

    def test_model_version_from_dict(self, temp_version_storage):
        """ModelVersion should deserialize from dict."""
        from auto_voice.training.model_versioning import ModelVersion

        d = {
            "version_id": "v1-abc123",
            "version_number": 1,
            "profile_id": "test-profile",
            "created_at": "2026-01-25T10:00:00",
            "metrics": {"loss": 0.5},
            "weights_path": str(temp_version_storage / "weights.pt"),
        }

        version = ModelVersion.from_dict(d)
        assert version.version_id == "v1-abc123"
        assert version.version_number == 1
