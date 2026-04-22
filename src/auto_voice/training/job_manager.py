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
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from auto_voice.runtime_contract import build_packaged_artifact_manifest
from auto_voice.storage.paths import (
    resolve_checkpoints_dir,
    resolve_profiles_dir,
    resolve_samples_dir,
    resolve_trained_models_dir,
)
from auto_voice.storage.voice_profiles import (
    FULL_MODEL_TRAINING_UNLOCK_SECONDS,
    PROFILE_ROLE_TARGET_USER,
)
from auto_voice.training.artifacts import (
    build_lora_checkpoint_payload,
    extract_lora_state_dict,
)

logger = logging.getLogger(__name__)


class _FallbackTrainingCancelledError(RuntimeError):
    """Local fallback when a patched trainer module omits the cancel exception."""


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

    training_mode: str = "lora"
    initialization_mode: str = "scratch"

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
    is_paused: bool = False

    def __post_init__(self):
        """Set defaults after initialization."""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.config is None:
            self.config = TrainingConfig()

    def update_progress(self, progress: int) -> None:
        """Update training progress (0-100)."""
        self.progress = max(0, min(100, progress))

    def start(self, gpu_device: Optional[int] = None) -> None:
        self.status = JobStatus.RUNNING.value
        self.started_at = datetime.now()
        self.is_paused = False
        if gpu_device is not None:
            self.gpu_device = gpu_device

    def complete(self, results: Optional[Dict[str, Any]] = None) -> None:
        self.status = JobStatus.COMPLETED.value
        self.completed_at = datetime.now()
        self.progress = 100
        if results is not None:
            self.results = results

    def fail(self, error: str) -> None:
        self.status = JobStatus.FAILED.value
        self.completed_at = datetime.now()
        self.error = error
        self.is_paused = False

    def cancel(self, error: Optional[str] = None) -> None:
        self.status = JobStatus.CANCELLED.value
        self.completed_at = datetime.now()
        if error is not None:
            self.error = error
        self.is_paused = False

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
            "is_paused": self.is_paused,
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
            is_paused=data.get("is_paused", False),
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
        profiles_dir: Optional[str] = None,
        samples_dir: Optional[str] = None,
    ):
        """Initialize job manager.

        Args:
            storage_path: Directory for job persistence
            require_gpu: If True, raise RuntimeError if CUDA unavailable
            socketio: Optional SocketIO instance for real-time events
            profiles_dir: Optional custom profiles directory for VoiceProfileStore
            samples_dir: Optional custom samples directory for VoiceProfileStore

        Raises:
            RuntimeError: If require_gpu=True and CUDA is not available
        """
        self.storage_path = Path(storage_path)
        self._require_gpu = require_gpu
        self._socketio = socketio
        self._profiles_dir = profiles_dir
        self._samples_dir = samples_dir
        if self._profiles_dir:
            self._data_dir = Path(self._profiles_dir).parent
        elif self._samples_dir:
            self._data_dir = Path(self._samples_dir).parent
        elif self.storage_path.name == "app_state":
            self._data_dir = self.storage_path.parent
        else:
            self._data_dir = self.storage_path
        self._jobs: Dict[str, TrainingJob] = {}
        self._job_resume_events: Dict[str, threading.Event] = {}
        self._job_cancel_events: Dict[str, threading.Event] = {}
        self._job_runtime_metrics: Dict[str, Dict[str, Any]] = {}
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

    def _resolve_profiles_dir(self) -> Path:
        return resolve_profiles_dir(self._profiles_dir, data_dir=str(self._data_dir))

    def _resolve_samples_dir(self) -> Path:
        return resolve_samples_dir(self._samples_dir, data_dir=str(self._data_dir))

    def _resolve_trained_models_dir(self) -> Path:
        return resolve_trained_models_dir(data_dir=str(self._data_dir))

    def _resolve_checkpoints_dir(self, profile_id: str) -> Path:
        return resolve_checkpoints_dir(data_dir=str(self._data_dir)) / profile_id

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

    def _resolve_profile_artifact_path(self, profile_id: str, training_mode: str) -> Optional[Path]:
        """Return the canonical artifact path for a profile/mode when present."""
        store = self._get_profile_store()

        if training_mode == "full":
            candidate = Path(store._full_model_path(profile_id))
            return candidate if candidate.exists() else None

        for candidate_path in (
            Path(store._lora_weights_path(profile_id)),
            Path(store._legacy_lora_weights_path(profile_id)),
        ):
            if candidate_path.exists():
                return candidate_path
        return None

    def _find_latest_checkpoint(
        self,
        profile_id: str,
        training_mode: str,
    ) -> Optional[Dict[str, Any]]:
        """Find the latest checkpoint compatible with the requested training mode."""
        checkpoint_dir = self._resolve_checkpoints_dir(profile_id)
        if not checkpoint_dir.exists():
            return None

        preferred_names = ["latest.pth", "final.pth", "best.pth"]
        candidates: List[Path] = []

        for name in preferred_names:
            candidate = checkpoint_dir / name
            if candidate.exists():
                candidates.append(candidate)

        extra_candidates = sorted(
            checkpoint_dir.glob("checkpoint_*.pth"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        candidates.extend(extra_candidates)

        seen: set[str] = set()
        for candidate in candidates:
            if str(candidate) in seen:
                continue
            seen.add(str(candidate))
            try:
                payload = torch.load(str(candidate), map_location="cpu", weights_only=False)
            except Exception as exc:
                logger.warning("Failed to inspect checkpoint %s: %s", candidate, exc)
                continue

            is_lora = bool(payload.get("is_lora", False))
            if training_mode == "lora" and not is_lora:
                continue
            if training_mode == "full" and is_lora:
                continue

            return {
                "path": str(candidate),
                "current_epoch": int(payload.get("current_epoch", 0) or 0),
                "global_step": int(payload.get("global_step", 0) or 0),
                "is_lora": is_lora,
            }

        return None

    def _resolve_initialization_state(
        self,
        profile_id: str,
        training_mode: str,
        initialization_mode: str,
    ) -> Dict[str, Any]:
        """Resolve whether a job should start from scratch or continue existing artifacts."""
        if initialization_mode not in {"scratch", "continue"}:
            initialization_mode = "scratch"

        if initialization_mode == "scratch":
            return {
                "initialization_mode": "scratch",
                "resume_checkpoint": None,
                "resume_epoch": 0,
                "artifact_path": None,
                "source": "scratch",
            }

        checkpoint_info = self._find_latest_checkpoint(profile_id, training_mode)
        if checkpoint_info:
            return {
                "initialization_mode": "continue",
                "resume_checkpoint": checkpoint_info["path"],
                "resume_epoch": checkpoint_info["current_epoch"],
                "artifact_path": None,
                "source": "checkpoint",
            }

        artifact_path = self._resolve_profile_artifact_path(profile_id, training_mode)
        if artifact_path is not None:
            return {
                "initialization_mode": "continue",
                "resume_checkpoint": None,
                "resume_epoch": 0,
                "artifact_path": str(artifact_path),
                "source": "artifact",
            }

        raise ValueError(
            f"No existing {training_mode} artifact or checkpoint is available to continue training"
        )

    def _load_existing_training_state(
        self,
        *,
        model: Any,
        artifact_path: str,
        training_mode: str,
        device: torch.device,
    ) -> None:
        """Load the current artifact into a fresh training model for continuation jobs."""
        payload = torch.load(artifact_path, map_location=device, weights_only=False)

        if training_mode == "lora":
            state_dict = extract_lora_state_dict(payload)
            model.load_lora_state_dict(state_dict)
            logger.info("Loaded LoRA continuation artifact from %s", artifact_path)
            return

        if isinstance(payload, dict) and "model" in payload:
            model_state = payload["model"]
        else:
            model_state = payload
        model.load_state_dict(model_state, strict=False)
        logger.info("Loaded full-model continuation artifact from %s", artifact_path)

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get job by ID.

        Args:
            job_id: Job ID to retrieve

        Returns:
            TrainingJob or None if not found
        """
        return self._jobs.get(job_id)

    def list_jobs(self, profile_id: Optional[str] = None) -> List[TrainingJob]:
        """List all jobs, optionally filtered by profile."""
        jobs = list(self._jobs.values())
        if profile_id:
            jobs = [job for job in jobs if job.profile_id == profile_id]
        return sorted(jobs, key=lambda job: job.created_at or datetime.min, reverse=True)

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

        current_status = JobStatus(job.status)
        if current_status == JobStatus.RUNNING:
            cancel_event = self._job_cancel_events.get(job_id)
            resume_event = self._job_resume_events.get(job_id)
            if cancel_event is not None:
                cancel_event.set()
            if resume_event is not None:
                resume_event.set()

            if cancel_event is not None or resume_event is not None:
                job.error = "Cancellation requested"
                self._save_jobs()
                logger.info("Requested cancellation for running job %s", job_id)
                return True

            # Preserve legacy behavior for callers/tests that mark a job running
            # without creating runtime coordination events.
            logger.warning("Cancelling running job %s without runtime events", job_id)
            job.cancel("Cancellation requested")
            self._save_jobs()
            self._emit_cancelled_event(job)
            return True

        if JobStatus.CANCELLED not in VALID_TRANSITIONS.get(current_status, []):
            logger.warning(
                f"Cannot cancel job {job_id} in status {job.status}"
            )
            return False

        job.cancel()
        self._save_jobs()
        self._emit_cancelled_event(job)

        logger.info(f"Cancelled job {job_id}")
        return True

    def pause_job(self, job_id: str) -> bool:
        """Pause an active training job."""
        job = self._jobs.get(job_id)
        if job is None or job.status != JobStatus.RUNNING.value or job.is_paused:
            return False

        resume_event = self._job_resume_events.get(job_id)
        if resume_event is None:
            return False

        resume_event.clear()
        job.is_paused = True
        self._save_jobs()
        self._emit_paused_event(job)
        return True

    def resume_job(self, job_id: str) -> bool:
        """Resume a paused training job."""
        job = self._jobs.get(job_id)
        if job is None or job.status != JobStatus.RUNNING.value or not job.is_paused:
            return False

        resume_event = self._job_resume_events.get(job_id)
        if resume_event is None:
            return False

        resume_event.set()
        job.is_paused = False
        self._save_jobs()
        self._emit_resumed_event(job)
        return True

    def get_job_runtime_metrics(self, job_id: str) -> Dict[str, Any]:
        """Return the latest runtime metrics for a job."""
        return dict(self._job_runtime_metrics.get(job_id, {}))

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
        payload = {
            'job_id': job.job_id,
            'profile_id': job.profile_id,
            'sample_count': len(job.sample_ids),
            'config': job.config.to_dict() if job.config else {},
            'started_at': job.started_at.isoformat() if job.started_at else None,
        }
        self._emit_event('training.started', payload)

    def _emit_completed_event(self, job: TrainingJob) -> None:
        """Emit training.completed event when job finishes successfully."""
        payload = {
            'job_id': job.job_id,
            'profile_id': job.profile_id,
            'results': job.results or {},
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
        }
        self._emit_event('training.completed', payload)
        self._emit_event('training_complete', payload)

    def _emit_failed_event(self, job: TrainingJob) -> None:
        """Emit training.failed event when job fails."""
        payload = {
            'job_id': job.job_id,
            'profile_id': job.profile_id,
            'error': job.error or 'Unknown error',
            'failed_at': job.completed_at.isoformat() if job.completed_at else None,
        }
        self._emit_event('training.failed', payload)
        self._emit_event('training_error', payload)

    def _emit_paused_event(self, job: TrainingJob) -> None:
        payload = {
            'job_id': job.job_id,
            'profile_id': job.profile_id,
            'paused_at': datetime.now().isoformat(),
        }
        self._emit_event('training.paused', payload)
        self._emit_event('training_paused', payload)

    def _emit_resumed_event(self, job: TrainingJob) -> None:
        payload = {
            'job_id': job.job_id,
            'profile_id': job.profile_id,
            'resumed_at': datetime.now().isoformat(),
        }
        self._emit_event('training.resumed', payload)
        self._emit_event('training_resumed', payload)

    def _emit_cancelled_event(self, job: TrainingJob) -> None:
        payload = {
            'job_id': job.job_id,
            'profile_id': job.profile_id,
            'cancelled_at': job.completed_at.isoformat() if job.completed_at else datetime.now().isoformat(),
            'error': job.error or 'Training cancelled by user',
        }
        self._emit_event('training.cancelled', payload)
        self._emit_event('training_cancelled', payload)

    def _get_gpu_metrics(self, device: torch.device) -> Dict[str, Any]:
        if device.type != 'cuda' or not torch.cuda.is_available():
            return {
                'available': False,
                'memory_used_gb': 0.0,
                'memory_reserved_gb': 0.0,
                'memory_total_gb': 0.0,
                'utilization_percent': 0.0,
            }

        device_idx = device.index or 0
        props = torch.cuda.get_device_properties(device_idx)
        total = float(props.total_memory)
        allocated = float(torch.cuda.memory_allocated(device_idx))
        reserved = float(torch.cuda.memory_reserved(device_idx))
        return {
            'available': True,
            'memory_used_gb': round(allocated / (1024 ** 3), 3),
            'memory_reserved_gb': round(reserved / (1024 ** 3), 3),
            'memory_total_gb': round(total / (1024 ** 3), 3),
            'utilization_percent': round((allocated / total) * 100.0, 2) if total else 0.0,
        }

    def _estimate_quality_metrics(self, loss: float) -> Dict[str, float]:
        bounded_loss = max(0.0, min(float(loss), 2.0))
        return {
            'mos_proxy': round(max(1.0, min(5.0, 5.0 - bounded_loss * 1.5)), 3),
            'speaker_similarity_proxy': round(max(0.0, min(0.995, 1.0 - bounded_loss * 0.25)), 4),
        }

    def emit_training_progress(
        self,
        job_id: str,
        epoch: int,
        total_epochs: int,
        step: int,
        total_steps: int,
        loss: float,
        learning_rate: float,
        *,
        gpu_metrics: Optional[Dict[str, Any]] = None,
        quality_metrics: Optional[Dict[str, Any]] = None,
        checkpoint_path: Optional[str] = None,
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

        runtime_metrics = {
            'epoch': epoch,
            'total_epochs': total_epochs,
            'step': step,
            'total_steps': total_steps,
            'loss': float(loss),
            'learning_rate': float(learning_rate),
            'progress_percent': round(overall_progress, 1),
            'gpu_metrics': gpu_metrics or {},
            'quality_metrics': quality_metrics or {},
            'checkpoint_path': checkpoint_path,
        }
        self._job_runtime_metrics[job_id] = runtime_metrics

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
            'gpu_metrics': gpu_metrics or {},
            'quality_metrics': quality_metrics or {},
            'checkpoint_path': checkpoint_path,
            'is_paused': job.is_paused,
        })
        self._emit_event('training_progress', {
            'job_id': job_id,
            'profile_id': job.profile_id,
            'epoch': epoch,
            'total_epochs': total_epochs,
            'step': step,
            'total_steps': total_steps,
            'loss': loss,
            'learning_rate': learning_rate,
            'gpu_metrics': gpu_metrics or {},
            'quality_metrics': quality_metrics or {},
            'checkpoint_path': checkpoint_path,
            'is_paused': job.is_paused,
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
        # Training execution remains GPU-only; require_gpu=False only skips
        # the constructor-time availability check so queue operations can run.
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Mark job as running
        gpu_device = device.index if device.type == "cuda" else None
        job.start(gpu_device=gpu_device)
        resume_event = threading.Event()
        resume_event.set()
        cancel_event = threading.Event()
        self._job_resume_events[job_id] = resume_event
        self._job_cancel_events[job_id] = cancel_event
        self._save_jobs()
        self._emit_started_event(job)
        logger.info(f"Starting training job {job_id}")

        try:
            import shutil
            import tempfile

            # Get sample audio paths
            store = self._get_profile_store()
            training_samples = store.list_training_samples(job.profile_id)

            if not training_samples:
                raise ValueError(f"No training samples found for profile {job.profile_id}")

            # Create temporary training directory with audio files
            train_dir = Path(tempfile.mkdtemp(prefix=f"autovoice-training-{job_id}-"))

            sample_files = []
            selected_sample_ids = set(job.sample_ids)
            for sample in training_samples:
                if sample.sample_id in selected_sample_ids or not selected_sample_ids:
                    src_path = Path(sample.vocals_path)
                    if src_path.exists():
                        dst_path = train_dir / f"{sample.sample_id}.wav"
                        shutil.copy2(src_path, dst_path)
                        source_metadata = src_path.parent / "metadata.json"
                        if source_metadata.exists():
                            shutil.copy2(
                                source_metadata,
                                train_dir / f"{sample.sample_id}.json",
                            )
                        sample_files.append(str(dst_path))
                        logger.info(f"Copied sample: {src_path} -> {dst_path}")

            if not sample_files:
                raise ValueError("No valid audio samples could be loaded")

            # Run training in background thread to not block
            def run_training():
                training_cancelled_error = _FallbackTrainingCancelledError
                try:
                    import importlib

                    from auto_voice.models.svc_decoder import CoMoSVCDecoder

                    trainer_module = importlib.import_module("auto_voice.training.trainer")
                    Trainer = trainer_module.Trainer
                    training_cancelled_error = getattr(
                        trainer_module,
                        "TrainingCancelledError",
                        _FallbackTrainingCancelledError,
                    )

                    training_mode = job.config.training_mode if job.config else "lora"
                    if training_mode not in {"lora", "full"}:
                        training_mode = "lora"
                    job_type = "full_model" if training_mode == "full" else "lora"
                    initialization_mode = (
                        job.config.initialization_mode if job.config else "scratch"
                    )
                    if initialization_mode not in {"scratch", "continue"}:
                        initialization_mode = "scratch"
                    initialization_state = self._resolve_initialization_state(
                        job.profile_id,
                        training_mode,
                        initialization_mode,
                    )

                    default_epochs = 500 if training_mode == "full" else 100
                    default_lr = 5e-5 if training_mode == "full" else 1e-4
                    batch_size = job.config.batch_size if job.config else 4
                    requested_epochs = job.config.epochs if job.config else default_epochs
                    resume_epoch = int(initialization_state.get("resume_epoch", 0) or 0)
                    epochs = requested_epochs + resume_epoch if initialization_state["resume_checkpoint"] else requested_epochs
                    learning_rate = job.config.learning_rate if job.config else default_lr

                    config = {
                        'epochs': epochs,
                        'learning_rate': learning_rate,
                        'batch_size': batch_size,
                        'checkpoint_dir': str(self._resolve_checkpoints_dir(job.profile_id)),
                        'checkpoint_interval_steps': 1000,
                        'n_mels': 80,
                    }

                    model = CoMoSVCDecoder(
                        content_dim=768,
                        pitch_dim=256,
                        speaker_dim=256,
                        n_mels=80,
                        hidden_dim=512,
                        n_layers=8,
                        device=device,
                    )

                    if training_mode == "lora":
                        lora_rank = job.config.lora_rank if job.config else 8
                        lora_alpha = job.config.lora_alpha if job.config else 16
                        lora_dropout = job.config.lora_dropout if job.config else 0.1
                        model.inject_lora(rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)

                    if initialization_state["artifact_path"]:
                        self._load_existing_training_state(
                            model=model,
                            artifact_path=initialization_state["artifact_path"],
                            training_mode=training_mode,
                            device=device,
                        )

                    trainer = Trainer(model=model, config=config, device=device)
                    trainer.resume_event = resume_event
                    trainer.cancel_event = cancel_event
                    logger.info(
                        "Training job %s on %s in %s mode",
                        job_id,
                        device,
                        training_mode,
                    )

                    def on_batch_end(batch_metrics: Dict[str, Any]) -> None:
                        job.update_progress(
                            int(batch_metrics.get('progress_percent') or (
                                ((batch_metrics['epoch'] - 1) + (
                                    batch_metrics['step'] / max(batch_metrics['total_steps'], 1)
                                )) / max(batch_metrics['total_epochs'], 1) * 100
                            ))
                        )
                        checkpoint_path = None
                        if trainer.global_step > 0 and trainer.global_step % 1000 == 0:
                            checkpoint_path = str(
                                trainer.checkpoint_dir / f'checkpoint_step_{trainer.global_step}.pth'
                            )
                        gpu_metrics = self._get_gpu_metrics(device)
                        quality_metrics = self._estimate_quality_metrics(batch_metrics['loss'])
                        job.results = job.results or {}
                        job.results.update({
                            'current_loss': batch_metrics['loss'],
                            'current_epoch': batch_metrics['epoch'],
                            'current_step': batch_metrics['global_step'],
                            'job_type': job_type,
                            'quality_metrics': quality_metrics,
                            'gpu_metrics': gpu_metrics,
                        })
                        if checkpoint_path:
                            job.results['latest_checkpoint'] = checkpoint_path
                        self._save_jobs()
                        self.emit_training_progress(
                            job_id=job_id,
                            epoch=batch_metrics['epoch'],
                            total_epochs=batch_metrics['total_epochs'],
                            step=batch_metrics['step'],
                            total_steps=batch_metrics['total_steps'],
                            loss=batch_metrics['loss'],
                            learning_rate=batch_metrics['learning_rate'],
                            gpu_metrics=gpu_metrics,
                            quality_metrics=quality_metrics,
                            checkpoint_path=checkpoint_path,
                        )

                    trainer.on_batch_end = on_batch_end

                    # Set speaker embedding before training
                    trainer.set_speaker_embedding(str(train_dir))

                    # Run training
                    trainer.train(
                        str(train_dir),
                        resume_from=initialization_state["resume_checkpoint"],
                    )

                    # Save adapter and speaker embedding to correct location
                    # Task 3.1: Verify training output format matches adapter spec
                    adapter_saved = self._save_trained_adapter(
                        trainer=trainer,
                        profile_id=job.profile_id,
                        job_id=job_id,
                        training_mode=training_mode,
                    )

                    # Get results
                    results = {
                        'final_loss': trainer.train_losses[-1] if trainer.train_losses else 0,
                        'best_loss': trainer.best_loss,
                        'epochs_completed': config['epochs'],
                        'requested_epochs': requested_epochs,
                        'checkpoint_path': str(trainer.checkpoint_dir / 'final.pth'),
                        'job_type': job_type,
                        'adapter_path': adapter_saved.get('adapter_path') if adapter_saved else None,
                        'embedding_path': adapter_saved.get('embedding_path') if adapter_saved else None,
                        'manifest_path': adapter_saved.get('manifest_path') if adapter_saved else None,
                        'artifact_type': adapter_saved.get('artifact_type') if adapter_saved else None,
                        'initialization_mode': initialization_mode,
                        'resume_source': initialization_state["source"],
                        'resume_checkpoint': initialization_state["resume_checkpoint"],
                        'resumed_from_epoch': resume_epoch if initialization_state["resume_checkpoint"] else None,
                        'artifact_reused': initialization_state["artifact_path"],
                    }

                    # Mark as completed
                    job.complete(results)
                    self._update_profile_training_state(
                        profile_id=job.profile_id,
                        results=results,
                        sample_count=len(sample_files),
                    )
                    self._save_jobs()

                    # Task 3.3: Emit training_complete event with profile_id
                    self._emit_completed_event(job)
                    logger.info(f"Training job {job_id} completed successfully")

                    # Cleanup temp directory
                    shutil.rmtree(train_dir, ignore_errors=True)

                except training_cancelled_error as e:
                    logger.info("Training job %s cancelled: %s", job_id, e)
                    job.cancel(str(e))
                    self._save_jobs()
                    self._emit_cancelled_event(job)
                except Exception as e:
                    logger.error(f"Training job {job_id} failed: {e}", exc_info=True)
                    job.fail(str(e))
                    self._mark_profile_training_failed(job.profile_id, str(e))
                    self._save_jobs()
                    self._emit_failed_event(job)
                finally:
                    self._job_resume_events.pop(job_id, None)
                    self._job_cancel_events.pop(job_id, None)
                    shutil.rmtree(train_dir, ignore_errors=True)

            # Start training in background
            training_thread = threading.Thread(target=run_training, daemon=True)
            training_thread.start()
            logger.info(f"Training job {job_id} started in background thread")

        except Exception as e:
            logger.error(f"Failed to start training job {job_id}: {e}", exc_info=True)
            job.fail(str(e))
            self._save_jobs()
            self._emit_failed_event(job)
            raise

    def _save_trained_adapter(
        self,
        trainer: Any,
        profile_id: str,
        job_id: str,
        training_mode: str = "lora",
    ) -> Dict[str, str]:
        """Save trained adapter and speaker embedding to correct location.

        Tasks 3.1-3.4: Save adapter in correct format, validate, and prepare for inference.

        Args:
            trainer: Trainer instance with trained model
            profile_id: Voice profile ID
            job_id: Training job ID

        Returns:
            Dict with 'adapter_path' and 'embedding_path'

        Raises:
            RuntimeError: If adapter cannot be saved or validated
        """
        import numpy as np
        from auto_voice.models.adapter_manager import AdapterManager, AdapterManagerConfig

        try:
            trained_models_dir = self._resolve_trained_models_dir()
            profiles_dir = self._resolve_profiles_dir()
            trained_models_dir.mkdir(parents=True, exist_ok=True)
            profiles_dir.mkdir(parents=True, exist_ok=True)

            store = self._get_profile_store()
            profile = store.load(profile_id)
            display_name = profile.get("name") or profile.get("display_name") or profile_id

            # Check if model has LoRA injected
            has_lora = getattr(trainer.model, '_lora_injected', False)
            if training_mode == "full":
                full_checkpoint_path = trained_models_dir / f"{profile_id}_full_model.pt"
                torch.save(trainer.model.state_dict(), full_checkpoint_path)
                logger.info(f"Saved full model checkpoint: {full_checkpoint_path}")

                # Still need speaker embedding
                if trainer.speaker_embedding is None:
                    raise RuntimeError("Trainer has no speaker embedding set")

                embedding_np = trainer.speaker_embedding.cpu().numpy()

                # Verify and normalize embedding
                norm = np.linalg.norm(embedding_np)
                if abs(norm - 1.0) > 0.01:
                    logger.warning(f"Normalizing speaker embedding (norm={norm:.4f})")
                    embedding_np = embedding_np / norm

                embedding_path = profiles_dir / f"{profile_id}.npy"
                np.save(embedding_path, embedding_np)
                logger.info(f"Saved speaker embedding: {embedding_path}")

                manifest = build_packaged_artifact_manifest(
                    profile_id=profile_id,
                    display_name=display_name,
                    model_family="realtime",
                    canonical_pipeline="realtime",
                    sample_rate=int(getattr(trainer, "sample_rate", 22050)),
                    speaker_embedding_dim=int(embedding_np.shape[0]),
                    mel_bins=int(getattr(trainer.model, "n_mels", trainer.config.get("n_mels", 80))),
                    artifacts={
                        "profile_json": str(profiles_dir / f"{profile_id}.json"),
                        "speaker_embedding": str(embedding_path),
                        "adapter": None,
                        "full_model": str(full_checkpoint_path),
                        "checkpoint": str(getattr(trainer, "checkpoint_dir", profiles_dir) / "final.pth"),
                    },
                    metadata={
                        "job_id": job_id,
                        "training_mode": training_mode,
                    },
                )
                manifest_path = store.save_runtime_artifact_manifest(profile_id, manifest.to_dict())

                return {
                    'adapter_path': str(full_checkpoint_path),
                    'embedding_path': str(embedding_path),
                    'manifest_path': str(manifest_path),
                    'artifact_type': 'full_model',
                }

            if not has_lora:
                raise RuntimeError(
                    f"Training job {job_id} requested LoRA output but the model has no injected LoRA adapters"
                )

            # Task 3.1: Extract LoRA adapter weights
            lora_state = trainer.model.get_lora_state_dict()
            adapter_payload = build_lora_checkpoint_payload(
                lora_state,
                config=getattr(trainer.model, "_lora_config", {}),
                metadata={
                    "profile_id": profile_id,
                    "job_id": job_id,
                    "training_mode": training_mode,
                },
            )

            # Save adapter with correct naming: {profile_id}_adapter.pt
            adapter_path = trained_models_dir / f"{profile_id}_adapter.pt"
            torch.save(adapter_payload, adapter_path)
            size_kb = adapter_path.stat().st_size / 1024
            logger.info(f"Saved LoRA adapter: {adapter_path} ({size_kb:.1f} KB)")

            # Task 3.1: Save speaker embedding as .npy
            if trainer.speaker_embedding is None:
                raise RuntimeError("Trainer has no speaker embedding set")

            embedding_np = trainer.speaker_embedding.cpu().numpy()

            # Verify embedding format (256-dim, L2-normalized)
            if embedding_np.shape != (256,):
                raise RuntimeError(
                    f"Invalid speaker embedding shape: {embedding_np.shape}, expected (256,)"
                )

            norm = np.linalg.norm(embedding_np)
            if abs(norm - 1.0) > 0.01:
                logger.warning(f"Normalizing speaker embedding (norm={norm:.4f})")
                embedding_np = embedding_np / norm

            embedding_path = profiles_dir / f"{profile_id}.npy"
            np.save(embedding_path, embedding_np)
            logger.info(f"Saved speaker embedding: {embedding_path}")

            # Task 3.2: Post-training validation
            logger.info(f"Validating saved adapter for profile {profile_id}")

            # Use AdapterManager to validate the saved adapter can be loaded
            adapter_manager = AdapterManager(AdapterManagerConfig(
                adapters_dir=trained_models_dir,
                profiles_dir=profiles_dir,
            ))

            # Try loading the adapter to verify it's valid
            loaded_state = adapter_manager.load_adapter(profile_id, use_cache=False)

            # Verify non-empty
            if not loaded_state:
                raise RuntimeError("Loaded adapter state is empty")

            # Verify contains expected keys
            expected_keys = ['lora_A', 'lora_B']
            has_expected = any(
                any(key in param_name for key in expected_keys)
                for param_name in loaded_state.keys()
            )
            if not has_expected:
                logger.warning(
                    f"Adapter may not contain expected LoRA structure. "
                    f"Keys: {list(loaded_state.keys())[:5]}"
                )

            logger.info(
                f"Adapter validation successful: {len(loaded_state)} parameters, "
                f"{sum(p.numel() for p in loaded_state.values())} total elements"
            )

            manifest = build_packaged_artifact_manifest(
                profile_id=profile_id,
                display_name=display_name,
                model_family="realtime",
                canonical_pipeline="realtime",
                sample_rate=int(getattr(trainer, "sample_rate", 22050)),
                speaker_embedding_dim=int(embedding_np.shape[0]),
                mel_bins=int(getattr(trainer.model, "n_mels", trainer.config.get("n_mels", 80))),
                artifacts={
                    "profile_json": str(profiles_dir / f"{profile_id}.json"),
                    "speaker_embedding": str(embedding_path),
                    "adapter": str(adapter_path),
                    "full_model": None,
                    "checkpoint": str(getattr(trainer, "checkpoint_dir", profiles_dir) / "final.pth"),
                },
                metadata={
                    "job_id": job_id,
                    "training_mode": training_mode,
                    "adapter_parameters": len(loaded_state),
                },
            )
            manifest_path = store.save_runtime_artifact_manifest(profile_id, manifest.to_dict())

            return {
                'adapter_path': str(adapter_path),
                'embedding_path': str(embedding_path),
                'manifest_path': str(manifest_path),
                'artifact_type': 'adapter',
            }

        except Exception as e:
            logger.error(f"Failed to save trained adapter for {profile_id}: {e}", exc_info=True)
            raise RuntimeError(f"Adapter save failed: {e}") from e

    def _update_profile_training_state(
        self,
        profile_id: str,
        results: Dict[str, Any],
        sample_count: int,
    ) -> None:
        """Persist successful training metadata into the canonical profile manifest."""
        store = self._get_profile_store()
        profile = store.load(profile_id)
        profile['training_status'] = 'ready'
        profile['has_trained_model'] = True
        profile['last_trained_at'] = datetime.now().isoformat()
        profile['model_version'] = profile.get('model_version') or '1.0'
        profile['model_path'] = results.get('adapter_path')
        profile['runtime_artifact_manifest_path'] = results.get('manifest_path')
        profile['training_epochs'] = results.get('epochs_completed')
        profile['loss_final'] = results.get('final_loss')
        profile['sample_count'] = sample_count
        artifact_type = results.get('artifact_type')
        if artifact_type == 'adapter':
            profile['selected_adapter'] = profile.get('selected_adapter') or 'unified'
            profile['active_model_type'] = 'adapter'
        elif artifact_type == 'full_model':
            profile['active_model_type'] = 'full_model'
        profile.pop('embedding', None)
        store.save(profile)

    def _mark_profile_training_failed(self, profile_id: str, error: str) -> None:
        """Persist failed training state without deleting prior artifacts."""
        try:
            store = self._get_profile_store()
            profile = store.load(profile_id)
            profile['training_status'] = 'failed'
            profile['last_training_error'] = error
            profile.pop('embedding', None)
            store.save(profile)
        except Exception as exc:
            logger.warning(f"Failed to persist training failure for {profile_id}: {exc}")

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

    # =========================================================================
    # Phase 4: Auto-Training Logic
    # =========================================================================

    def _get_profile_store(self) -> "VoiceProfileStore":
        """Get VoiceProfileStore with configured paths."""
        from auto_voice.storage.voice_profiles import VoiceProfileStore

        return VoiceProfileStore(
            profiles_dir=str(self._resolve_profiles_dir()),
            samples_dir=str(self._resolve_samples_dir()),
        )

    def check_needs_training(
        self,
        profile_id: str,
        min_samples: int = 5,
    ) -> Dict[str, Any]:
        """Check if a profile needs initial training.

        Phase 4: Training trigger conditions
        - Profile has >= min_samples (default 5)
        - No existing LoRA adapter
        - Profile flagged "needs_training"

        Args:
            profile_id: Voice profile ID
            min_samples: Minimum samples required for training (default: 5)

        Returns:
            Dict with keys:
            - needs_training: bool
            - reason: str (why training is needed)
            - sample_count: int
            - has_adapter: bool
        """
        from auto_voice.models.adapter_manager import AdapterManager, AdapterManagerConfig

        store = self._get_profile_store()

        try:
            profile = store.load(profile_id)
        except Exception as e:
            logger.warning(f"Could not load profile {profile_id}: {e}")
            return {
                "needs_training": False,
                "reason": "profile_not_found",
                "sample_count": 0,
                "has_adapter": False,
            }

        # Count samples
        samples = store.list_training_samples(profile_id)
        sample_count = len(samples)

        # Check if adapter exists
        adapter_manager = AdapterManager(AdapterManagerConfig(
            adapters_dir=self._resolve_trained_models_dir(),
            profiles_dir=self._resolve_profiles_dir(),
        ))

        try:
            adapter_state = adapter_manager.load_adapter(profile_id, use_cache=False)
            has_adapter = adapter_state is not None and len(adapter_state) > 0
        except Exception:
            has_adapter = False

        # Check status flag
        status = profile.get("status", "")
        needs_training_flag = status == "needs_training"
        profile_role = profile.get("profile_role", PROFILE_ROLE_TARGET_USER)

        # Determine if training needed
        needs_training = False
        reason = "none"

        if profile_role != PROFILE_ROLE_TARGET_USER:
            reason = f"unsupported_profile_role ({profile_role})"
        elif sample_count < min_samples:
            reason = f"insufficient_samples (have {sample_count}, need {min_samples})"
        elif not has_adapter:
            needs_training = True
            reason = "no_adapter"
        elif needs_training_flag:
            needs_training = True
            reason = "flagged_needs_training"

        return {
            "needs_training": needs_training,
            "reason": reason,
            "sample_count": sample_count,
            "has_adapter": has_adapter,
        }

    def check_needs_retraining(
        self,
        profile_id: str,
        retrain_new_samples: int = 3,
        freshness_days: int = 30,
        speaker_similarity_min: float = 0.85,
        mcd_max: float = 4.5,
    ) -> Dict[str, Any]:
        """Check if a profile needs retraining.

        Phase 4: Retrain trigger conditions
        - >3 new samples since last training
        - Quality degradation (speaker_sim drops below threshold)
        - LoRA >30 days old with new samples available

        Args:
            profile_id: Voice profile ID
            retrain_new_samples: Trigger retrain after N new samples (default: 3)
            freshness_days: Max age before retrain recommended (default: 30)
            speaker_similarity_min: Min speaker similarity threshold (default: 0.85)
            mcd_max: Max MCD threshold (default: 4.5)

        Returns:
            Dict with keys:
            - needs_retraining: bool
            - reasons: List[str] (all reasons for retraining)
            - new_sample_count: int
            - days_since_training: int
            - quality_metrics: Dict
        """
        store = self._get_profile_store()

        try:
            profile = store.load(profile_id)
        except Exception as e:
            logger.warning(f"Could not load profile {profile_id}: {e}")
            return {
                "needs_retraining": False,
                "reasons": [],
                "new_sample_count": 0,
                "days_since_training": 0,
                "quality_metrics": {},
            }

        reasons = []

        # Check for new samples since last training
        last_trained_at = profile.get("last_trained_at")
        samples = store.list_training_samples(profile_id)

        if last_trained_at:
            last_trained_dt = datetime.fromisoformat(last_trained_at)
            new_samples = [
                s for s in samples
                if datetime.fromisoformat(s.created_at) > last_trained_dt
            ]
            new_sample_count = len(new_samples)

            if new_sample_count >= retrain_new_samples:
                reasons.append(
                    f"new_samples ({new_sample_count} >= {retrain_new_samples})"
                )

            # Check freshness
            days_since_training = (datetime.now() - last_trained_dt).days
            if days_since_training > freshness_days and new_sample_count > 0:
                reasons.append(
                    f"stale_adapter ({days_since_training} days > {freshness_days})"
                )
        else:
            new_sample_count = len(samples)
            days_since_training = 0

        # Check quality metrics
        quality_metrics = profile.get("quality_metrics", {})
        speaker_sim = quality_metrics.get("speaker_similarity")
        mcd = quality_metrics.get("mcd")

        if speaker_sim is not None and speaker_sim < speaker_similarity_min:
            reasons.append(
                f"quality_degradation (speaker_sim={speaker_sim:.3f} < {speaker_similarity_min})"
            )

        if mcd is not None and mcd > mcd_max:
            reasons.append(
                f"quality_degradation (mcd={mcd:.2f} > {mcd_max})"
            )

        return {
            "needs_retraining": len(reasons) > 0,
            "reasons": reasons,
            "new_sample_count": new_sample_count,
            "days_since_training": days_since_training,
            "quality_metrics": quality_metrics,
        }

    def auto_queue_training(
        self,
        profile_id: str,
        min_samples: int = 5,
        retrain_new_samples: int = 3,
        freshness_days: int = 30,
        config: Optional[TrainingConfig] = None,
    ) -> Optional[TrainingJob]:
        """Automatically queue training if conditions are met.

        Phase 4: Background training scheduler integration
        Checks both initial training and retraining conditions.

        Args:
            profile_id: Voice profile ID
            min_samples: Minimum samples for initial training
            retrain_new_samples: New samples threshold for retrain
            freshness_days: Max days before retrain
            config: Optional custom training config

        Returns:
            TrainingJob if queued, None if conditions not met
        """
        store = self._get_profile_store()

        # Check if profile exists
        try:
            profile = store.load(profile_id)
        except Exception as e:
            logger.warning(f"Profile {profile_id} not found: {e}")
            return None

        if profile.get("profile_role", PROFILE_ROLE_TARGET_USER) != PROFILE_ROLE_TARGET_USER:
            logger.debug(
                f"Skipping auto-training for non-target profile {profile_id}: "
                f"{profile.get('profile_role')}"
            )
            return None

        # Check if already queued
        existing_jobs = self.get_jobs_for_profile(profile_id)
        pending_or_running = [
            j for j in existing_jobs
            if j.status in [JobStatus.PENDING.value, JobStatus.RUNNING.value]
        ]

        if pending_or_running:
            logger.info(
                f"Profile {profile_id} already has pending/running job: "
                f"{pending_or_running[0].job_id}"
            )
            return None

        # Check initial training conditions
        training_check = self.check_needs_training(profile_id, min_samples)

        if training_check["needs_training"]:
            logger.info(
                f"Queueing initial training for {profile_id}: "
                f"{training_check['reason']}"
            )

            samples = store.list_training_samples(profile_id)
            sample_ids = [s.sample_id for s in samples]

            job = self.create_job(
                profile_id=profile_id,
                sample_ids=sample_ids,
                config=config,
            )

            # Update profile status
            profile["status"] = "training_queued"
            store.save(profile)

            return job

        # Check retraining conditions
        retrain_check = self.check_needs_retraining(
            profile_id=profile_id,
            retrain_new_samples=retrain_new_samples,
            freshness_days=freshness_days,
        )

        if retrain_check["needs_retraining"]:
            logger.info(
                f"Queueing retraining for {profile_id}: "
                f"{', '.join(retrain_check['reasons'])}"
            )

            samples = store.list_training_samples(profile_id)
            sample_ids = [s.sample_id for s in samples]

            job = self.create_job(
                profile_id=profile_id,
                sample_ids=sample_ids,
                config=config,
            )

            return job

        logger.debug(
            f"Profile {profile_id} does not need training: "
            f"has_adapter={training_check['has_adapter']}, "
            f"sample_count={training_check['sample_count']}"
        )
        return None

    def auto_queue_all_profiles(
        self,
        min_samples: int = 5,
        retrain_new_samples: int = 3,
        freshness_days: int = 30,
    ) -> List[TrainingJob]:
        """Check all profiles and queue training as needed.

        Phase 4: Background scheduler integration
        Scans all profiles and auto-queues training/retraining.

        Args:
            min_samples: Minimum samples for initial training
            retrain_new_samples: New samples threshold for retrain
            freshness_days: Max days before retrain

        Returns:
            List of queued training jobs
        """
        store = self._get_profile_store()
        profiles = store.list_profiles()

        queued_jobs = []

        for profile in profiles:
            profile_id = profile.get("profile_id")
            if not profile_id:
                continue

            job = self.auto_queue_training(
                profile_id=profile_id,
                min_samples=min_samples,
                retrain_new_samples=retrain_new_samples,
                freshness_days=freshness_days,
            )

            if job:
                queued_jobs.append(job)

        if queued_jobs:
            logger.info(
                f"Auto-queued {len(queued_jobs)} training jobs across "
                f"{len(profiles)} profiles"
            )

        return queued_jobs

    # =========================================================================
    # Phase 4.4: Full Model Training for High-Sample Profiles
    # =========================================================================

    FULL_MODEL_UNLOCK_SECONDS = FULL_MODEL_TRAINING_UNLOCK_SECONDS

    def check_needs_full_model(
        self,
        profile_id: str,
        duration_threshold_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Check if a profile should upgrade to full model training.

        Phase 4.4: Full model training for target-user profiles with sufficient
        accumulated clean singing vocals.

        Args:
            profile_id: Voice profile ID
            duration_threshold_seconds: Override default threshold in seconds

        Returns:
            Dict with keys:
            - needs_full_model: bool
            - reason: str
            - sample_count: int
            - clean_vocal_seconds: float
            - remaining_seconds: float
            - current_adapter_type: str
        """
        threshold_seconds = float(
            duration_threshold_seconds or self.FULL_MODEL_UNLOCK_SECONDS
        )

        store = self._get_profile_store()

        try:
            profile = store.load(profile_id)
        except Exception as e:
            logger.warning(f"Could not load profile {profile_id}: {e}")
            return {
                "needs_full_model": False,
                "reason": "profile_not_found",
                "sample_count": 0,
                "clean_vocal_seconds": 0.0,
                "remaining_seconds": threshold_seconds,
                "current_adapter_type": "none",
            }

        # Count samples
        samples = store.list_training_samples(profile_id)
        sample_count = len(samples)
        clean_vocal_seconds = float(
            profile.get("clean_vocal_seconds")
            or profile.get("total_training_duration")
            or sum(sample.duration for sample in samples)
        )
        remaining_seconds = max(threshold_seconds - clean_vocal_seconds, 0.0)
        profile_role = profile.get("profile_role", PROFILE_ROLE_TARGET_USER)

        # Check current adapter type
        trained_models_dir = self._resolve_trained_models_dir()

        current_adapter_type = "none"
        if (trained_models_dir / f"{profile_id}_full_model.pt").exists():
            current_adapter_type = "full_model"
        elif (trained_models_dir / "hq" / f"{profile_id}_hq_lora.pt").exists():
            current_adapter_type = "hq_lora"
        elif (trained_models_dir / "nvfp4" / f"{profile_id}_nvfp4_lora.pt").exists():
            current_adapter_type = "nvfp4_lora"
        elif (trained_models_dir / f"{profile_id}_adapter.pt").exists():
            current_adapter_type = "standard_lora"

        # Determine if upgrade needed
        needs_full_model = False
        reason = "none"

        if profile_role != PROFILE_ROLE_TARGET_USER:
            reason = f"unsupported_profile_role ({profile_role})"
        elif clean_vocal_seconds < threshold_seconds:
            reason = (
                "insufficient_clean_vocals "
                f"(have {clean_vocal_seconds:.1f}s, need {threshold_seconds:.1f}s)"
            )
        elif current_adapter_type == "full_model":
            reason = "already_full_model"
        else:
            needs_full_model = True
            reason = (
                "upgrade_recommended "
                f"(has {clean_vocal_seconds:.1f}s clean vocals, current: {current_adapter_type})"
            )

        return {
            "needs_full_model": needs_full_model,
            "reason": reason,
            "sample_count": sample_count,
            "clean_vocal_seconds": clean_vocal_seconds,
            "remaining_seconds": remaining_seconds,
            "current_adapter_type": current_adapter_type,
        }

    def create_full_model_job(
        self,
        profile_id: str,
        config: Optional[TrainingConfig] = None,
        initialization_mode: str = "scratch",
    ) -> TrainingJob:
        """Create a full model training job (not LoRA).

        Phase 4.4: For profiles with >=30 minutes of clean user vocals, train
        full model
        for higher quality conversion.

        Args:
            profile_id: Voice profile ID
            config: Optional training config (uses enhanced defaults for full model)

        Returns:
            Created TrainingJob with full_model flag

        Raises:
            ValueError: If insufficient clean vocals for full model
        """
        if initialization_mode not in {"scratch", "continue"}:
            initialization_mode = "scratch"

        # Check if eligible
        check = self.check_needs_full_model(profile_id)

        if not check["needs_full_model"]:
            if check["reason"] == "already_full_model":
                logger.info(
                    "Profile %s already has a full model; allowing manual %s training",
                    profile_id,
                    initialization_mode,
                )
            elif check["clean_vocal_seconds"] < self.FULL_MODEL_UNLOCK_SECONDS:
                raise ValueError(
                    f"Profile {profile_id} has only {check['clean_vocal_seconds']:.1f}s "
                    f"of clean vocals. Need at least "
                    f"{self.FULL_MODEL_UNLOCK_SECONDS:.1f}s for full model training."
                )
            else:
                raise ValueError(check["reason"])

        store = self._get_profile_store()
        samples = store.list_training_samples(profile_id)
        sample_ids = [s.sample_id for s in samples]

        # Enhanced config for full model training
        if config is None:
            config = TrainingConfig(
                training_mode="full",
                initialization_mode=initialization_mode,
                # Full model uses more epochs and lower LR
                epochs=50,
                learning_rate=5e-5,
                batch_size=8,
                # Disable LoRA for full model
                lora_rank=0,  # 0 indicates full model
                lora_alpha=0,
                # Enable EWC for preserving base model behavior
                use_ewc=True,
                ewc_lambda=500.0,
            )
        else:
            # Override LoRA settings for full model
            config.training_mode = "full"
            config.initialization_mode = initialization_mode
            config.lora_rank = 0
            config.lora_alpha = 0

        job = self.create_job(
            profile_id=profile_id,
            sample_ids=sample_ids,
            config=config,
        )

        # Mark as full model job in results
        job.results = job.results or {}
        job.results["job_type"] = "full_model"
        job.results["sample_count"] = len(sample_ids)

        self._save_jobs()

        logger.info(
            f"Created full model training job {job.job_id} for {profile_id} "
            f"with {check['clean_vocal_seconds']:.1f}s clean vocals"
        )

        return job

    def auto_queue_full_model_training(
        self,
        profile_id: str,
        duration_threshold_seconds: Optional[float] = None,
    ) -> Optional[TrainingJob]:
        """Automatically queue full model training if threshold met.

        Phase 4.4: Auto-upgrade from LoRA to full model when clean vocals
        meet the canonical duration threshold.

        Args:
            profile_id: Voice profile ID
            duration_threshold_seconds: Override threshold (default: canonical unlock)

        Returns:
            TrainingJob if queued, None otherwise
        """
        check = self.check_needs_full_model(profile_id, duration_threshold_seconds)

        if not check["needs_full_model"]:
            logger.debug(
                f"Profile {profile_id} does not need full model: {check['reason']}"
            )
            return None

        # Check if already has pending full model job
        existing_jobs = self.get_jobs_for_profile(profile_id)
        for job in existing_jobs:
            if job.status in [JobStatus.PENDING.value, JobStatus.RUNNING.value]:
                if job.results and job.results.get("job_type") == "full_model":
                    logger.info(
                        f"Profile {profile_id} already has full model job: {job.job_id}"
                    )
                    return None

        return self.create_full_model_job(profile_id)
