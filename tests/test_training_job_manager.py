"""TDD tests for TrainingJobManager and incremental training jobs.

Task 4.1: Write failing tests for incremental training job creation
Task 4.2: Implement TrainingJobManager with job queue (GPU-only execution)

Tests cover:
- TrainingJob model/dataclass
- TrainingJobManager initialization
- Job creation for voice profiles
- Job states (pending, running, completed, failed)
- Job queue management
- GPU requirement enforcement
"""

import pytest
import tempfile
import os
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import torch


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_job_storage():
    """Temporary directory for job artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_profile():
    """Mock VoiceProfile for testing."""
    profile = Mock()
    profile.profile_id = "test-profile-123"
    profile.user_id = "user-456"
    profile.name = "Test Voice"
    profile.samples_count = 15
    profile.model_version = "v1"
    return profile


@pytest.fixture
def mock_training_samples():
    """Mock training samples for a profile."""
    samples = []
    for i in range(10):
        sample = Mock()
        sample.sample_id = f"sample-{i}"
        sample.profile_id = "test-profile-123"
        sample.duration_seconds = 5.0 + i * 0.5  # 5-9.5 seconds
        sample.audio_path = f"/data/samples/sample-{i}.wav"
        sample.quality_score = 0.85 + i * 0.01
        samples.append(sample)
    return samples


@pytest.fixture
def job_manager(temp_job_storage):
    """TrainingJobManager instance for testing."""
    from auto_voice.training.job_manager import TrainingJobManager
    return TrainingJobManager(storage_path=temp_job_storage)


# ============================================================================
# Test: TrainingJob Model
# ============================================================================

class TestTrainingJobModel:
    """Tests for TrainingJob dataclass/model."""

    def test_training_job_has_required_fields(self):
        """TrainingJob must have job_id, profile_id, status, created_at."""
        from auto_voice.training.job_manager import TrainingJob

        job = TrainingJob(
            job_id="job-001",
            profile_id="profile-123",
        )

        assert job.job_id == "job-001"
        assert job.profile_id == "profile-123"
        assert job.status == "pending"  # Default status
        assert job.created_at is not None
        assert isinstance(job.created_at, datetime)

    def test_training_job_status_values(self):
        """TrainingJob status must be one of: pending, running, completed, failed, cancelled."""
        from auto_voice.training.job_manager import TrainingJob, JobStatus

        # Valid statuses
        assert JobStatus.PENDING == "pending"
        assert JobStatus.RUNNING == "running"
        assert JobStatus.COMPLETED == "completed"
        assert JobStatus.FAILED == "failed"
        assert JobStatus.CANCELLED == "cancelled"

    def test_training_job_tracks_progress(self):
        """TrainingJob must track training progress (0-100%)."""
        from auto_voice.training.job_manager import TrainingJob

        job = TrainingJob(
            job_id="job-001",
            profile_id="profile-123",
        )

        assert job.progress == 0
        job.update_progress(50)
        assert job.progress == 50
        job.update_progress(100)
        assert job.progress == 100

    def test_training_job_stores_config(self):
        """TrainingJob must store training configuration."""
        from auto_voice.training.job_manager import TrainingJob, TrainingConfig

        config = TrainingConfig(
            learning_rate=1e-4,
            epochs=10,
            batch_size=4,
            lora_rank=8,
            lora_alpha=16,
            use_ewc=True,
            ewc_lambda=1000.0,
        )

        job = TrainingJob(
            job_id="job-001",
            profile_id="profile-123",
            config=config,
        )

        assert job.config.learning_rate == 1e-4
        assert job.config.lora_rank == 8
        assert job.config.use_ewc is True

    def test_training_job_tracks_sample_ids(self):
        """TrainingJob must track which samples are used for training."""
        from auto_voice.training.job_manager import TrainingJob

        sample_ids = ["sample-1", "sample-2", "sample-3"]
        job = TrainingJob(
            job_id="job-001",
            profile_id="profile-123",
            sample_ids=sample_ids,
        )

        assert job.sample_ids == sample_ids
        assert len(job.sample_ids) == 3

    def test_training_job_to_dict(self):
        """TrainingJob must be serializable to dict."""
        from auto_voice.training.job_manager import TrainingJob

        job = TrainingJob(
            job_id="job-001",
            profile_id="profile-123",
            sample_ids=["s1", "s2"],
        )

        job_dict = job.to_dict()
        assert job_dict["job_id"] == "job-001"
        assert job_dict["profile_id"] == "profile-123"
        assert job_dict["status"] == "pending"
        assert "created_at" in job_dict

    def test_training_job_from_dict(self):
        """TrainingJob must be deserializable from dict."""
        from auto_voice.training.job_manager import TrainingJob

        job_dict = {
            "job_id": "job-002",
            "profile_id": "profile-456",
            "status": "completed",
            "created_at": "2026-01-25T10:00:00",
            "progress": 100,
        }

        job = TrainingJob.from_dict(job_dict)
        assert job.job_id == "job-002"
        assert job.profile_id == "profile-456"
        assert job.status == "completed"
        assert job.progress == 100


# ============================================================================
# Test: TrainingJobManager Initialization
# ============================================================================

class TestTrainingJobManagerInit:
    """Tests for TrainingJobManager initialization."""

    def test_job_manager_initialization(self, temp_job_storage):
        """TrainingJobManager initializes with storage path."""
        from auto_voice.training.job_manager import TrainingJobManager

        manager = TrainingJobManager(storage_path=temp_job_storage)
        assert manager.storage_path == temp_job_storage
        assert manager.is_initialized

    def test_job_manager_creates_storage_directory(self, temp_job_storage):
        """TrainingJobManager creates storage directory if not exists."""
        from auto_voice.training.job_manager import TrainingJobManager

        new_path = temp_job_storage / "jobs"
        manager = TrainingJobManager(storage_path=new_path)
        assert new_path.exists()

    def test_job_manager_has_empty_queue_initially(self, job_manager):
        """TrainingJobManager starts with empty job queue."""
        assert job_manager.queue_size == 0
        assert job_manager.get_pending_jobs() == []

    def test_job_manager_requires_gpu(self, temp_job_storage):
        """TrainingJobManager raises RuntimeError if CUDA unavailable."""
        from auto_voice.training.job_manager import TrainingJobManager

        with patch.object(torch.cuda, 'is_available', return_value=False):
            with pytest.raises(RuntimeError, match="CUDA.*required"):
                TrainingJobManager(storage_path=temp_job_storage, require_gpu=True)

    def test_job_manager_accepts_gpu_check_skip_for_testing(self, temp_job_storage):
        """TrainingJobManager allows GPU check skip for testing."""
        from auto_voice.training.job_manager import TrainingJobManager

        # Should not raise even if CUDA unavailable
        manager = TrainingJobManager(
            storage_path=temp_job_storage,
            require_gpu=False  # Skip GPU check for testing
        )
        assert manager.is_initialized


# ============================================================================
# Test: Job Creation
# ============================================================================

class TestJobCreation:
    """Tests for creating training jobs."""

    def test_create_job_for_profile(self, job_manager, mock_profile, mock_training_samples):
        """Create training job for a voice profile."""
        job = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=[s.sample_id for s in mock_training_samples],
        )

        assert job is not None
        assert job.job_id is not None
        assert job.profile_id == mock_profile.profile_id
        assert job.status == "pending"
        assert len(job.sample_ids) == len(mock_training_samples)

    def test_create_job_generates_unique_id(self, job_manager, mock_profile):
        """Each job gets a unique ID."""
        job1 = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1", "s2"],
        )
        job2 = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s3", "s4"],
        )

        assert job1.job_id != job2.job_id

    def test_create_job_with_custom_config(self, job_manager, mock_profile):
        """Create job with custom training configuration."""
        from auto_voice.training.job_manager import TrainingConfig

        config = TrainingConfig(
            learning_rate=5e-5,
            epochs=20,
            lora_rank=16,
        )

        job = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1"],
            config=config,
        )

        assert job.config.learning_rate == 5e-5
        assert job.config.epochs == 20
        assert job.config.lora_rank == 16

    def test_create_job_requires_samples(self, job_manager, mock_profile):
        """Creating job without samples raises ValueError."""
        with pytest.raises(ValueError, match="sample.*required"):
            job_manager.create_job(
                profile_id=mock_profile.profile_id,
                sample_ids=[],
            )

    def test_create_job_adds_to_queue(self, job_manager, mock_profile):
        """Created job is added to pending queue."""
        assert job_manager.queue_size == 0

        job = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1", "s2"],
        )

        assert job_manager.queue_size == 1
        pending = job_manager.get_pending_jobs()
        assert len(pending) == 1
        assert pending[0].job_id == job.job_id


# ============================================================================
# Test: Job Queue Management
# ============================================================================

class TestJobQueueManagement:
    """Tests for job queue operations."""

    def test_get_job_by_id(self, job_manager, mock_profile):
        """Retrieve job by its ID."""
        created_job = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1"],
        )

        retrieved_job = job_manager.get_job(created_job.job_id)
        assert retrieved_job is not None
        assert retrieved_job.job_id == created_job.job_id

    def test_get_nonexistent_job_returns_none(self, job_manager):
        """Getting non-existent job returns None."""
        job = job_manager.get_job("nonexistent-job-id")
        assert job is None

    def test_list_jobs_for_profile(self, job_manager, mock_profile):
        """List all jobs for a specific profile."""
        job1 = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1"],
        )
        job2 = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s2"],
        )
        # Job for different profile
        job3 = job_manager.create_job(
            profile_id="other-profile",
            sample_ids=["s3"],
        )

        profile_jobs = job_manager.get_jobs_for_profile(mock_profile.profile_id)
        assert len(profile_jobs) == 2
        job_ids = [j.job_id for j in profile_jobs]
        assert job1.job_id in job_ids
        assert job2.job_id in job_ids
        assert job3.job_id not in job_ids

    def test_cancel_pending_job(self, job_manager, mock_profile):
        """Cancel a pending job."""
        job = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1"],
        )

        success = job_manager.cancel_job(job.job_id)
        assert success is True

        updated_job = job_manager.get_job(job.job_id)
        assert updated_job.status == "cancelled"

    def test_cancel_running_job(self, job_manager, mock_profile):
        """Cancelling running job sets status to cancelled."""
        job = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1"],
        )
        # Simulate job starting
        job_manager._set_job_status(job.job_id, "running")

        success = job_manager.cancel_job(job.job_id)
        assert success is True

        updated_job = job_manager.get_job(job.job_id)
        assert updated_job.status == "cancelled"

    def test_cannot_cancel_completed_job(self, job_manager, mock_profile):
        """Cannot cancel already completed job."""
        job = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1"],
        )
        job_manager._set_job_status(job.job_id, "completed")

        success = job_manager.cancel_job(job.job_id)
        assert success is False

        updated_job = job_manager.get_job(job.job_id)
        assert updated_job.status == "completed"  # Unchanged

    def test_get_next_pending_job(self, job_manager, mock_profile):
        """Get next job from queue (FIFO order)."""
        job1 = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1"],
        )
        time.sleep(0.01)  # Ensure different timestamps
        job2 = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s2"],
        )

        next_job = job_manager.get_next_pending_job()
        assert next_job.job_id == job1.job_id  # First in, first out


# ============================================================================
# Test: Job Status Updates
# ============================================================================

class TestJobStatusUpdates:
    """Tests for job status transitions."""

    def test_update_job_status_to_running(self, job_manager, mock_profile):
        """Update job status from pending to running."""
        job = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1"],
        )

        job_manager.update_job_status(job.job_id, "running")
        updated = job_manager.get_job(job.job_id)
        assert updated.status == "running"
        assert updated.started_at is not None

    def test_update_job_status_to_completed(self, job_manager, mock_profile):
        """Update job status to completed with results."""
        job = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1"],
        )
        job_manager.update_job_status(job.job_id, "running")

        results = {
            "adapter_path": "/models/profile-123/adapter_v2.pt",
            "metrics": {
                "speaker_similarity": 0.92,
                "loss_final": 0.015,
            }
        }

        job_manager.update_job_status(job.job_id, "completed", results=results)
        updated = job_manager.get_job(job.job_id)

        assert updated.status == "completed"
        assert updated.completed_at is not None
        assert updated.results["adapter_path"] == results["adapter_path"]
        assert updated.results["metrics"]["speaker_similarity"] == 0.92

    def test_update_job_status_to_failed(self, job_manager, mock_profile):
        """Update job status to failed with error message."""
        job = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1"],
        )
        job_manager.update_job_status(job.job_id, "running")

        error_msg = "CUDA out of memory"
        job_manager.update_job_status(job.job_id, "failed", error=error_msg)

        updated = job_manager.get_job(job.job_id)
        assert updated.status == "failed"
        assert updated.error == error_msg

    def test_update_job_progress(self, job_manager, mock_profile):
        """Update job training progress."""
        job = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1"],
        )
        job_manager.update_job_status(job.job_id, "running")

        job_manager.update_job_progress(job.job_id, 25)
        assert job_manager.get_job(job.job_id).progress == 25

        job_manager.update_job_progress(job.job_id, 75)
        assert job_manager.get_job(job.job_id).progress == 75

    def test_invalid_status_transition_raises(self, job_manager, mock_profile):
        """Invalid status transitions raise ValueError."""
        job = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1"],
        )

        # Cannot go from pending directly to completed
        with pytest.raises(ValueError, match="Invalid.*transition"):
            job_manager.update_job_status(job.job_id, "completed")


# ============================================================================
# Test: GPU Enforcement
# ============================================================================

class TestGPUEnforcement:
    """Tests for GPU-only execution requirement."""

    def test_job_execution_requires_cuda(self, temp_job_storage, mock_profile):
        """Job execution raises RuntimeError if CUDA unavailable."""
        from auto_voice.training.job_manager import TrainingJobManager

        # Create manager with GPU check disabled (for queue operations)
        manager = TrainingJobManager(
            storage_path=temp_job_storage,
            require_gpu=False,
        )

        job = manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1"],
        )

        # But execution should fail without GPU
        with patch.object(torch.cuda, 'is_available', return_value=False):
            with pytest.raises(RuntimeError, match="CUDA.*required.*training"):
                manager.execute_job(job.job_id)

    def test_job_tracks_gpu_device(self, job_manager, mock_profile):
        """Job records which GPU device was used."""
        job = job_manager.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1"],
        )

        # Simulate job running on GPU 0
        job_manager.update_job_status(job.job_id, "running", gpu_device=0)

        updated = job_manager.get_job(job.job_id)
        assert updated.gpu_device == 0


# ============================================================================
# Test: Job Persistence
# ============================================================================

class TestJobPersistence:
    """Tests for job state persistence."""

    def test_jobs_persist_to_storage(self, temp_job_storage, mock_profile):
        """Jobs are persisted to storage directory."""
        from auto_voice.training.job_manager import TrainingJobManager

        manager1 = TrainingJobManager(
            storage_path=temp_job_storage,
            require_gpu=False,
        )

        job = manager1.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1", "s2"],
        )
        job_id = job.job_id

        # Create new manager instance - should load existing jobs
        manager2 = TrainingJobManager(
            storage_path=temp_job_storage,
            require_gpu=False,
        )

        loaded_job = manager2.get_job(job_id)
        assert loaded_job is not None
        assert loaded_job.profile_id == mock_profile.profile_id
        assert loaded_job.sample_ids == ["s1", "s2"]

    def test_job_status_updates_persist(self, temp_job_storage, mock_profile):
        """Job status updates are persisted."""
        from auto_voice.training.job_manager import TrainingJobManager

        manager1 = TrainingJobManager(
            storage_path=temp_job_storage,
            require_gpu=False,
        )

        job = manager1.create_job(
            profile_id=mock_profile.profile_id,
            sample_ids=["s1"],
        )
        manager1.update_job_status(job.job_id, "running")
        manager1.update_job_progress(job.job_id, 50)

        # Reload
        manager2 = TrainingJobManager(
            storage_path=temp_job_storage,
            require_gpu=False,
        )

        loaded_job = manager2.get_job(job.job_id)
        assert loaded_job.status == "running"
        assert loaded_job.progress == 50


# ============================================================================
# Test: TrainingConfig Defaults
# ============================================================================

class TestTrainingConfigDefaults:
    """Tests for TrainingConfig with sensible defaults."""

    def test_training_config_defaults(self):
        """TrainingConfig has sensible defaults from SOTA research."""
        from auto_voice.training.job_manager import TrainingConfig

        config = TrainingConfig()

        # LoRA defaults from research doc
        assert config.lora_rank == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.1

        # Training defaults
        assert config.learning_rate == 1e-4
        assert config.batch_size == 4
        assert config.epochs == 10

        # EWC defaults
        assert config.use_ewc is True
        assert config.ewc_lambda == 1000.0

    def test_training_config_serialization(self):
        """TrainingConfig serializes to/from dict."""
        from auto_voice.training.job_manager import TrainingConfig

        config = TrainingConfig(
            learning_rate=5e-5,
            epochs=20,
        )

        config_dict = config.to_dict()
        assert config_dict["learning_rate"] == 5e-5
        assert config_dict["epochs"] == 20

        loaded = TrainingConfig.from_dict(config_dict)
        assert loaded.learning_rate == 5e-5
        assert loaded.epochs == 20
