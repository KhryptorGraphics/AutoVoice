"""Training Job Manager for continuous voice profile learning.

Provides GPU-only training job management with:
- Job queue with FIFO ordering
- LoRA adapter configuration (from SOTA research)
- EWC regularization support
- Job persistence to disk
- Status tracking and progress updates

Task 4.2: Implement TrainingJobManager with job queue (GPU-only execution)
"""

import json
import logging
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


# ============================================================================
# Job Status Enum
# ============================================================================

class JobStatus(str, Enum):
    """Training job status values."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    def __str__(self) -> str:
        return self.value


# Valid status transitions
VALID_TRANSITIONS = {
    JobStatus.PENDING: [JobStatus.RUNNING, JobStatus.CANCELLED],
    JobStatus.RUNNING: [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED],
    JobStatus.COMPLETED: [],  # Terminal state
    JobStatus.FAILED: [],  # Terminal state
    JobStatus.CANCELLED: [],  # Terminal state
}


# ============================================================================
# Training Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Training configuration with SOTA defaults from research.

    Defaults based on:
    - LoRA config: rank=8, alpha=16 (from federated LoRA research)
    - EWC lambda: 1000.0 (from EVCL paper)
    """

    # LoRA configuration
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "content_encoder"]
    )

    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 4
    epochs: int = 10
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # EWC configuration (prevent catastrophic forgetting)
    use_ewc: bool = True
    ewc_lambda: float = 1000.0

    # Prior preservation (from Stable-TTS research)
    use_prior_preservation: bool = False
    prior_loss_weight: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Deserialize from dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================================
# Training Job
# ============================================================================

@dataclass
class TrainingJob:
    """Represents a training job for a voice profile.

    Tracks job state, configuration, and results.
    """

    job_id: str
    profile_id: str
    status: str = JobStatus.PENDING.value
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: int = 0
    sample_ids: List[str] = field(default_factory=list)
    config: Optional[TrainingConfig] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    gpu_device: Optional[int] = None

    def __post_init__(self):
        """Set defaults after initialization."""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.config is None:
            self.config = TrainingConfig()

    def update_progress(self, progress: int) -> None:
        """Update training progress (0-100)."""
        self.progress = max(0, min(100, progress))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for persistence."""
        return {
            "job_id": self.job_id,
            "profile_id": self.profile_id,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "sample_ids": self.sample_ids,
            "config": self.config.to_dict() if self.config else None,
            "results": self.results,
            "error": self.error,
            "gpu_device": self.gpu_device,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingJob":
        """Deserialize from dict."""
        # Parse datetime fields
        created_at = None
        if data.get("created_at"):
            created_at = datetime.fromisoformat(data["created_at"])

        started_at = None
        if data.get("started_at"):
            started_at = datetime.fromisoformat(data["started_at"])

        completed_at = None
        if data.get("completed_at"):
            completed_at = datetime.fromisoformat(data["completed_at"])

        # Parse config
        config = None
        if data.get("config"):
            config = TrainingConfig.from_dict(data["config"])

        return cls(
            job_id=data["job_id"],
            profile_id=data["profile_id"],
            status=data.get("status", JobStatus.PENDING.value),
            created_at=created_at,
            started_at=started_at,
            completed_at=completed_at,
            progress=data.get("progress", 0),
            sample_ids=data.get("sample_ids", []),
            config=config,
            results=data.get("results"),
            error=data.get("error"),
            gpu_device=data.get("gpu_device"),
        )


# ============================================================================
# Training Job Manager
# ============================================================================

class TrainingJobManager:
    """Manages training jobs for voice profile continuous learning.

    Features:
    - Job queue with FIFO ordering
    - GPU-only execution enforcement
    - Job persistence to disk
    - Status tracking and progress updates
    """

    JOBS_FILENAME = "training_jobs.json"

    def __init__(
        self,
        storage_path: Path | str,
        require_gpu: bool = True,
        socketio: Optional[Any] = None,
    ):
        """Initialize job manager.

        Args:
            storage_path: Directory for job persistence
            require_gpu: If True, raise RuntimeError if CUDA unavailable
            socketio: Optional SocketIO instance for real-time events

        Raises:
            RuntimeError: If require_gpu=True and CUDA is not available
        """
        self.storage_path = Path(storage_path)
        self._require_gpu = require_gpu
        self._socketio = socketio
        self._jobs: Dict[str, TrainingJob] = {}
        self._is_initialized = False

        # Check GPU availability
        if require_gpu and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for TrainingJobManager. "
                "Training must run on GPU for acceptable performance."
            )

        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Load existing jobs
        self._load_jobs()
        self._is_initialized = True

        logger.info(
            f"TrainingJobManager initialized with storage at {self.storage_path}, "
            f"loaded {len(self._jobs)} existing jobs"
        )

    @property
    def is_initialized(self) -> bool:
        """Check if manager is initialized."""
        return self._is_initialized

    @property
    def queue_size(self) -> int:
        """Number of pending jobs in queue."""
        return len([j for j in self._jobs.values() if j.status == JobStatus.PENDING.value])

    def _jobs_file_path(self) -> Path:
        """Path to jobs persistence file."""
        return self.storage_path / self.JOBS_FILENAME

    def _load_jobs(self) -> None:
        """Load jobs from persistence file."""
        jobs_file = self._jobs_file_path()
        if jobs_file.exists():
            try:
                with open(jobs_file, "r") as f:
                    data = json.load(f)
                    for job_data in data.get("jobs", []):
                        job = TrainingJob.from_dict(job_data)
                        self._jobs[job.job_id] = job
                logger.debug(f"Loaded {len(self._jobs)} jobs from {jobs_file}")
            except Exception as e:
                logger.warning(f"Failed to load jobs from {jobs_file}: {e}")

    def _save_jobs(self) -> None:
        """Save jobs to persistence file."""
        jobs_file = self._jobs_file_path()
        try:
            data = {
                "jobs": [job.to_dict() for job in self._jobs.values()],
                "updated_at": datetime.now().isoformat(),
            }
            with open(jobs_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(self._jobs)} jobs to {jobs_file}")
        except Exception as e:
            logger.error(f"Failed to save jobs to {jobs_file}: {e}")

    def create_job(
        self,
        profile_id: str,
        sample_ids: List[str],
        config: Optional[TrainingConfig] = None,
    ) -> TrainingJob:
        """Create a new training job.

        Args:
            profile_id: Voice profile to train
            sample_ids: Training sample IDs to use
            config: Optional custom training config

        Returns:
            Created TrainingJob

        Raises:
            ValueError: If sample_ids is empty
        """
        if not sample_ids:
            raise ValueError("At least one sample is required for training")

        job_id = f"job-{uuid.uuid4().hex[:12]}"

        job = TrainingJob(
            job_id=job_id,
            profile_id=profile_id,
            sample_ids=sample_ids,
            config=config or TrainingConfig(),
        )

        self._jobs[job_id] = job
        self._save_jobs()

        logger.info(
            f"Created training job {job_id} for profile {profile_id} "
            f"with {len(sample_ids)} samples"
        )

        return job

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get job by ID.

        Args:
            job_id: Job ID to retrieve

        Returns:
            TrainingJob or None if not found
        """
        return self._jobs.get(job_id)

    def get_pending_jobs(self) -> List[TrainingJob]:
        """Get all pending jobs in FIFO order.

        Returns:
            List of pending jobs sorted by creation time
        """
        pending = [
            job for job in self._jobs.values()
            if job.status == JobStatus.PENDING.value
        ]
        return sorted(pending, key=lambda j: j.created_at or datetime.min)

    def get_next_pending_job(self) -> Optional[TrainingJob]:
        """Get next job from queue (FIFO).

        Returns:
            Next pending job or None if queue empty
        """
        pending = self.get_pending_jobs()
        return pending[0] if pending else None

    def get_jobs_for_profile(self, profile_id: str) -> List[TrainingJob]:
        """Get all jobs for a specific profile.

        Args:
            profile_id: Profile ID to filter by

        Returns:
            List of jobs for the profile
        """
        return [
            job for job in self._jobs.values()
            if job.profile_id == profile_id
        ]

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if cancelled, False if job cannot be cancelled
        """
        job = self._jobs.get(job_id)
        if not job:
            return False

        # Check if job can be cancelled
        current_status = JobStatus(job.status)
        if JobStatus.CANCELLED not in VALID_TRANSITIONS.get(current_status, []):
            logger.warning(
                f"Cannot cancel job {job_id} in status {job.status}"
            )
            return False

        job.status = JobStatus.CANCELLED.value
        self._save_jobs()

        logger.info(f"Cancelled job {job_id}")
        return True

    def _set_job_status(self, job_id: str, status: str) -> None:
        """Internal method to set job status (for testing)."""
        job = self._jobs.get(job_id)
        if job:
            job.status = status
            self._save_jobs()

    def update_job_status(
        self,
        job_id: str,
        status: str,
        results: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        gpu_device: Optional[int] = None,
    ) -> None:
        """Update job status with validation.

        Args:
            job_id: Job ID to update
            status: New status
            results: Optional results dict (for completed jobs)
            error: Optional error message (for failed jobs)
            gpu_device: Optional GPU device ID (for running jobs)

        Raises:
            ValueError: If status transition is invalid
        """
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        current_status = JobStatus(job.status)
        new_status = JobStatus(status)

        # Validate transition
        valid_next = VALID_TRANSITIONS.get(current_status, [])
        if new_status not in valid_next:
            raise ValueError(
                f"Invalid status transition from {current_status} to {new_status}. "
                f"Valid transitions: {valid_next}"
            )

        job.status = new_status.value

        # Update timestamps
        if new_status == JobStatus.RUNNING:
            job.started_at = datetime.now()
            if gpu_device is not None:
                job.gpu_device = gpu_device

        elif new_status in (JobStatus.COMPLETED, JobStatus.FAILED):
            job.completed_at = datetime.now()

        # Store results/error
        if results is not None:
            job.results = results
        if error is not None:
            job.error = error

        self._save_jobs()

        # Emit WebSocket events based on new status
        if new_status == JobStatus.RUNNING:
            self._emit_started_event(job)
        elif new_status == JobStatus.COMPLETED:
            self._emit_completed_event(job)
        elif new_status == JobStatus.FAILED:
            self._emit_failed_event(job)

        logger.info(f"Updated job {job_id} status to {new_status}")

    def update_job_progress(self, job_id: str, progress: int) -> None:
        """Update job training progress.

        Args:
            job_id: Job ID to update
            progress: Progress percentage (0-100)
        """
        job = self._jobs.get(job_id)
        if job:
            job.update_progress(progress)
            self._save_jobs()

    # =========================================================================
    # WebSocket Event Emission
    # =========================================================================

    def _emit_event(self, event_name: str, data: Dict[str, Any]) -> None:
        """Emit a WebSocket event if socketio is available.

        Args:
            event_name: Event name (e.g., 'training.started')
            data: Event data to send
        """
        if self._socketio is not None:
            try:
                self._socketio.emit(event_name, data)
            except Exception as e:
                logger.debug(f"Failed to emit {event_name}: {e}")

    def _emit_started_event(self, job: TrainingJob) -> None:
        """Emit training.started event when job begins."""
        self._emit_event('training.started', {
            'job_id': job.job_id,
            'profile_id': job.profile_id,
            'sample_count': len(job.sample_ids),
            'config': job.config.to_dict() if job.config else {},
            'started_at': job.started_at.isoformat() if job.started_at else None,
        })

    def _emit_completed_event(self, job: TrainingJob) -> None:
        """Emit training.completed event when job finishes successfully."""
        self._emit_event('training.completed', {
            'job_id': job.job_id,
            'profile_id': job.profile_id,
            'results': job.results or {},
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
        })

    def _emit_failed_event(self, job: TrainingJob) -> None:
        """Emit training.failed event when job fails."""
        self._emit_event('training.failed', {
            'job_id': job.job_id,
            'profile_id': job.profile_id,
            'error': job.error or 'Unknown error',
            'failed_at': job.completed_at.isoformat() if job.completed_at else None,
        })

    def emit_training_progress(
        self,
        job_id: str,
        epoch: int,
        total_epochs: int,
        step: int,
        total_steps: int,
        loss: float,
        learning_rate: float,
    ) -> None:
        """Emit training.progress event with current training metrics.

        Args:
            job_id: Training job ID
            epoch: Current epoch number
            total_epochs: Total epochs to train
            step: Current step within epoch
            total_steps: Total steps in epoch
            loss: Current loss value
            learning_rate: Current learning rate
        """
        job = self._jobs.get(job_id)
        if not job:
            return

        # Calculate overall progress
        epoch_progress = step / total_steps if total_steps > 0 else 0
        overall_progress = ((epoch - 1) + epoch_progress) / total_epochs * 100

        self._emit_event('training.progress', {
            'job_id': job_id,
            'profile_id': job.profile_id,
            'epoch': epoch,
            'total_epochs': total_epochs,
            'step': step,
            'total_steps': total_steps,
            'loss': loss,
            'learning_rate': learning_rate,
            'progress_percent': round(overall_progress, 1),
        })

        # Also update job progress
        job.update_progress(int(overall_progress))

    def execute_job(self, job_id: str) -> None:
        """Execute a training job.

        Args:
            job_id: Job ID to execute

        Raises:
            RuntimeError: If CUDA is not available
            ValueError: If job not found or not in pending state
        """
        # Enforce GPU requirement at execution time
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for training execution. "
                "Training jobs must run on GPU."
            )

        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        if job.status != JobStatus.PENDING.value:
            raise ValueError(
                f"Job {job_id} is not in pending state (current: {job.status})"
            )

        # TODO: Implement actual training execution in Task 4.4
        # This will integrate with the fine-tuning pipeline
        logger.info(f"Would execute job {job_id} (not yet implemented)")

    def _mark_job_completed(self, job_id: str, results: Optional[Dict[str, Any]] = None) -> None:
        """Mark a job as completed (internal method for testing).

        Args:
            job_id: Job ID to mark complete
            results: Optional results dict
        """
        job = self._jobs.get(job_id)
        if not job:
            return

        # Force status to completed (bypasses validation for testing)
        job.status = JobStatus.COMPLETED.value
        job.completed_at = datetime.now()
        job.progress = 100
        if results:
            job.results = results
        self._save_jobs()

    def cleanup_completed_jobs(self, keep_count: int = 10) -> List[str]:
        """Remove old completed jobs, keeping only the most recent.

        Args:
            keep_count: Number of completed jobs to keep

        Returns:
            List of removed job IDs
        """
        # Get completed jobs sorted by completion time (newest first)
        completed = [
            job for job in self._jobs.values()
            if job.status == JobStatus.COMPLETED.value
        ]
        completed.sort(
            key=lambda j: j.completed_at or datetime.min,
            reverse=True
        )

        # Remove old jobs
        removed = []
        for job in completed[keep_count:]:
            del self._jobs[job.job_id]
            removed.append(job.job_id)

        if removed:
            self._save_jobs()
            logger.info(f"Cleaned up {len(removed)} old completed jobs")

        return removed

    def get_completed_jobs(self) -> List[TrainingJob]:
        """Get all completed jobs.

        Returns:
            List of completed jobs sorted by completion time
        """
        completed = [
            job for job in self._jobs.values()
            if job.status == JobStatus.COMPLETED.value
        ]
        return sorted(
            completed,
            key=lambda j: j.completed_at or datetime.min,
            reverse=True
        )
