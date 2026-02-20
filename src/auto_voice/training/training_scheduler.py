"""Training scheduler for automatic incremental training.

Automatically triggers training jobs when sample thresholds are reached.
Implements configurable thresholds and cooldown periods.

Task 4.8: Implement training scheduler with configurable thresholds
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


class SampleStorage(Protocol):
    """Protocol for sample storage interface."""

    def get_sample_count(self, profile_id: str) -> int:
        """Get total sample count for a profile."""
        ...

    def get_unprocessed_samples(self, profile_id: str) -> List[Any]:
        """Get samples that haven't been used for training."""
        ...

    def mark_samples_processed(self, sample_ids: List[str]) -> None:
        """Mark samples as processed after training."""
        ...


class JobManager(Protocol):
    """Protocol for job manager interface."""

    def create_job(self, profile_id: str, sample_ids: List[str], config: Any = None) -> Any:
        """Create a new training job."""
        ...


@dataclass
class SchedulerConfig:
    """Configuration for training scheduler."""

    min_samples_threshold: int = 10
    max_samples_threshold: int = 100
    check_interval_seconds: int = 60
    cooldown_seconds: int = 3600  # 1 hour default

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "min_samples_threshold": self.min_samples_threshold,
            "max_samples_threshold": self.max_samples_threshold,
            "check_interval_seconds": self.check_interval_seconds,
            "cooldown_seconds": self.cooldown_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchedulerConfig":
        """Deserialize from dictionary."""
        return cls(
            min_samples_threshold=data.get("min_samples_threshold", 10),
            max_samples_threshold=data.get("max_samples_threshold", 100),
            check_interval_seconds=data.get("check_interval_seconds", 60),
            cooldown_seconds=data.get("cooldown_seconds", 3600),
        )


class TrainingScheduler:
    """Automatically schedules training based on sample accumulation."""

    def __init__(
        self,
        profile_id: str,
        job_manager: JobManager,
        sample_storage: SampleStorage,
        config: SchedulerConfig,
        state_dir: Path,
    ):
        """Initialize training scheduler.

        Args:
            profile_id: Voice profile to schedule training for
            job_manager: Manager for creating training jobs
            sample_storage: Storage for training samples
            config: Scheduler configuration
            state_dir: Directory for persisting scheduler state
        """
        self.profile_id = profile_id
        self.job_manager = job_manager
        self.sample_storage = sample_storage
        self.config = config
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self._state_path = self.state_dir / f"{profile_id}_scheduler.json"
        self._last_training_time: Optional[datetime] = None
        self._load_state()

    def _load_state(self) -> None:
        """Load scheduler state from disk."""
        if self._state_path.exists():
            with open(self._state_path) as f:
                data = json.load(f)
            if data.get("last_training_time"):
                self._last_training_time = datetime.fromisoformat(
                    data["last_training_time"]
                )

    def _save_state(self) -> None:
        """Save scheduler state to disk."""
        data = {
            "profile_id": self.profile_id,
            "last_training_time": (
                self._last_training_time.isoformat()
                if self._last_training_time
                else None
            ),
        }
        with open(self._state_path, "w") as f:
            json.dump(data, f, indent=2)

    def _is_in_cooldown(self) -> bool:
        """Check if scheduler is in cooldown period."""
        if self._last_training_time is None:
            return False

        cooldown_end = self._last_training_time + timedelta(
            seconds=self.config.cooldown_seconds
        )
        return datetime.now() < cooldown_end

    def _get_unprocessed_count(self) -> int:
        """Get count of unprocessed samples."""
        samples = self.sample_storage.get_unprocessed_samples(self.profile_id)
        return len(samples)

    def check_should_train(self) -> Tuple[bool, str]:
        """Check if training should be triggered.

        Returns:
            Tuple of (should_train, reason)
        """
        sample_count = self._get_unprocessed_count()

        # Max threshold overrides cooldown
        if sample_count >= self.config.max_samples_threshold:
            return True, f"Max threshold reached ({sample_count} samples)"

        # Check cooldown for min threshold
        if self._is_in_cooldown():
            return False, "In cooldown period"

        # Check min threshold
        if sample_count >= self.config.min_samples_threshold:
            return True, f"Min threshold reached ({sample_count} samples)"

        return False, f"Below threshold ({sample_count}/{self.config.min_samples_threshold} samples)"

    def trigger_training(self) -> Any:
        """Trigger a training job.

        Returns:
            The created training job
        """
        samples = self.sample_storage.get_unprocessed_samples(self.profile_id)
        sample_ids = [getattr(s, "sample_id", str(i)) for i, s in enumerate(samples)]

        # Create job
        job = self.job_manager.create_job(
            profile_id=self.profile_id,
            sample_ids=sample_ids,
        )

        # Mark samples as processed
        self.sample_storage.mark_samples_processed(sample_ids)

        # Update last training time
        self._last_training_time = datetime.now()
        self._save_state()

        logger.info(
            f"Triggered training for profile {self.profile_id} "
            f"with {len(samples)} samples"
        )

        return job

    def check_and_trigger(self) -> bool:
        """Check conditions and trigger training if needed.

        Returns:
            True if training was triggered, False otherwise
        """
        should_train, reason = self.check_should_train()

        if should_train:
            logger.info(f"Training triggered: {reason}")
            self.trigger_training()
            return True

        logger.debug(f"Training not triggered: {reason}")
        return False

    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status.

        Returns:
            Dictionary with scheduler state information
        """
        current_samples = self._get_unprocessed_count()
        samples_until_trigger = max(
            0, self.config.min_samples_threshold - current_samples
        )

        return {
            "profile_id": self.profile_id,
            "current_samples": current_samples,
            "min_threshold": self.config.min_samples_threshold,
            "max_threshold": self.config.max_samples_threshold,
            "samples_until_trigger": samples_until_trigger,
            "in_cooldown": self._is_in_cooldown(),
            "last_training_time": (
                self._last_training_time.isoformat()
                if self._last_training_time
                else None
            ),
        }
