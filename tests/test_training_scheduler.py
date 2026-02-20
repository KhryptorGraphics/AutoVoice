"""TDD tests for training scheduler (Task 4.7).

Tests cover:
- Auto-trigger training after N samples accumulated
- Configurable sample thresholds
- Scheduler state persistence
- Integration with TrainingJobManager
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# === Fixtures ===


@pytest.fixture
def temp_scheduler_storage():
    """Temporary directory for scheduler state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_job_manager():
    """Mock TrainingJobManager for testing."""
    manager = MagicMock()
    manager.create_job.return_value = MagicMock(job_id="job-123")
    return manager


@pytest.fixture
def mock_sample_storage():
    """Mock sample storage with configurable sample counts."""
    storage = MagicMock()
    storage.get_sample_count.return_value = 0
    storage.get_unprocessed_samples.return_value = []
    return storage


@pytest.fixture
def scheduler_config():
    """Default scheduler configuration."""
    from auto_voice.training.training_scheduler import SchedulerConfig

    return SchedulerConfig(
        min_samples_threshold=10,
        max_samples_threshold=100,
        check_interval_seconds=60,
        cooldown_seconds=3600,
    )


# === Test Classes ===


class TestSchedulerConfig:
    """Tests for scheduler configuration."""

    def test_config_default_values(self):
        """Config should have sensible defaults."""
        from auto_voice.training.training_scheduler import SchedulerConfig

        config = SchedulerConfig()

        assert config.min_samples_threshold > 0
        assert config.max_samples_threshold > config.min_samples_threshold
        assert config.check_interval_seconds > 0
        assert config.cooldown_seconds >= 0

    def test_config_custom_thresholds(self):
        """Config should accept custom thresholds."""
        from auto_voice.training.training_scheduler import SchedulerConfig

        config = SchedulerConfig(
            min_samples_threshold=5,
            max_samples_threshold=50,
        )

        assert config.min_samples_threshold == 5
        assert config.max_samples_threshold == 50

    def test_config_serialization(self):
        """Config should serialize to/from dict."""
        from auto_voice.training.training_scheduler import SchedulerConfig

        config = SchedulerConfig(
            min_samples_threshold=15,
            max_samples_threshold=150,
            check_interval_seconds=120,
            cooldown_seconds=7200,
        )

        d = config.to_dict()
        restored = SchedulerConfig.from_dict(d)

        assert restored.min_samples_threshold == 15
        assert restored.max_samples_threshold == 150
        assert restored.check_interval_seconds == 120
        assert restored.cooldown_seconds == 7200


class TestSchedulerInitialization:
    """Tests for scheduler initialization."""

    def test_scheduler_initialization(
        self, temp_scheduler_storage, mock_job_manager, mock_sample_storage, scheduler_config
    ):
        """Scheduler should initialize with required dependencies."""
        from auto_voice.training.training_scheduler import TrainingScheduler

        scheduler = TrainingScheduler(
            profile_id="test-profile",
            job_manager=mock_job_manager,
            sample_storage=mock_sample_storage,
            config=scheduler_config,
            state_dir=temp_scheduler_storage,
        )

        assert scheduler.profile_id == "test-profile"
        assert scheduler.config == scheduler_config

    def test_scheduler_loads_persisted_state(
        self, temp_scheduler_storage, mock_job_manager, mock_sample_storage, scheduler_config
    ):
        """Scheduler should restore state from disk."""
        from auto_voice.training.training_scheduler import TrainingScheduler

        # Create scheduler and update state
        scheduler1 = TrainingScheduler(
            profile_id="test-profile",
            job_manager=mock_job_manager,
            sample_storage=mock_sample_storage,
            config=scheduler_config,
            state_dir=temp_scheduler_storage,
        )
        scheduler1._last_training_time = datetime.now()
        scheduler1._save_state()

        # Create new scheduler instance - should load state
        scheduler2 = TrainingScheduler(
            profile_id="test-profile",
            job_manager=mock_job_manager,
            sample_storage=mock_sample_storage,
            config=scheduler_config,
            state_dir=temp_scheduler_storage,
        )

        assert scheduler2._last_training_time is not None


class TestSampleThresholdTrigger:
    """Tests for sample count threshold triggering."""

    def test_trigger_when_min_threshold_reached(
        self, temp_scheduler_storage, mock_job_manager, mock_sample_storage, scheduler_config
    ):
        """Training should trigger when minimum samples reached."""
        from auto_voice.training.training_scheduler import TrainingScheduler

        # Set up mock to return enough samples
        mock_sample_storage.get_sample_count.return_value = 10
        mock_sample_storage.get_unprocessed_samples.return_value = [MagicMock() for _ in range(10)]

        scheduler = TrainingScheduler(
            profile_id="test-profile",
            job_manager=mock_job_manager,
            sample_storage=mock_sample_storage,
            config=scheduler_config,
            state_dir=temp_scheduler_storage,
        )

        should_train, reason = scheduler.check_should_train()

        assert should_train is True
        assert "threshold" in reason.lower()

    def test_no_trigger_below_threshold(
        self, temp_scheduler_storage, mock_job_manager, mock_sample_storage, scheduler_config
    ):
        """Training should not trigger below minimum threshold."""
        from auto_voice.training.training_scheduler import TrainingScheduler

        # Set up mock to return fewer samples than threshold
        mock_sample_storage.get_sample_count.return_value = 5
        mock_sample_storage.get_unprocessed_samples.return_value = [MagicMock() for _ in range(5)]

        scheduler = TrainingScheduler(
            profile_id="test-profile",
            job_manager=mock_job_manager,
            sample_storage=mock_sample_storage,
            config=scheduler_config,
            state_dir=temp_scheduler_storage,
        )

        should_train, reason = scheduler.check_should_train()

        assert should_train is False

    def test_trigger_at_max_threshold_overrides_cooldown(
        self, temp_scheduler_storage, mock_job_manager, mock_sample_storage, scheduler_config
    ):
        """Max threshold should trigger even during cooldown."""
        from auto_voice.training.training_scheduler import TrainingScheduler

        # Set up mock to return max samples
        mock_sample_storage.get_sample_count.return_value = 100
        mock_sample_storage.get_unprocessed_samples.return_value = [MagicMock() for _ in range(100)]

        scheduler = TrainingScheduler(
            profile_id="test-profile",
            job_manager=mock_job_manager,
            sample_storage=mock_sample_storage,
            config=scheduler_config,
            state_dir=temp_scheduler_storage,
        )

        # Simulate recent training (within cooldown)
        scheduler._last_training_time = datetime.now()

        should_train, reason = scheduler.check_should_train()

        assert should_train is True
        assert "max" in reason.lower()


class TestCooldownBehavior:
    """Tests for training cooldown period."""

    def test_cooldown_prevents_training(
        self, temp_scheduler_storage, mock_job_manager, mock_sample_storage, scheduler_config
    ):
        """Training should be blocked during cooldown period."""
        from auto_voice.training.training_scheduler import TrainingScheduler

        mock_sample_storage.get_sample_count.return_value = 15
        mock_sample_storage.get_unprocessed_samples.return_value = [MagicMock() for _ in range(15)]

        scheduler = TrainingScheduler(
            profile_id="test-profile",
            job_manager=mock_job_manager,
            sample_storage=mock_sample_storage,
            config=scheduler_config,
            state_dir=temp_scheduler_storage,
        )

        # Simulate recent training
        scheduler._last_training_time = datetime.now()

        should_train, reason = scheduler.check_should_train()

        assert should_train is False
        assert "cooldown" in reason.lower()

    def test_cooldown_expires_allows_training(
        self, temp_scheduler_storage, mock_job_manager, mock_sample_storage, scheduler_config
    ):
        """Training should be allowed after cooldown expires."""
        from auto_voice.training.training_scheduler import TrainingScheduler

        mock_sample_storage.get_sample_count.return_value = 15
        mock_sample_storage.get_unprocessed_samples.return_value = [MagicMock() for _ in range(15)]

        scheduler = TrainingScheduler(
            profile_id="test-profile",
            job_manager=mock_job_manager,
            sample_storage=mock_sample_storage,
            config=scheduler_config,
            state_dir=temp_scheduler_storage,
        )

        # Simulate old training (cooldown expired)
        scheduler._last_training_time = datetime.now() - timedelta(seconds=scheduler_config.cooldown_seconds + 1)

        should_train, reason = scheduler.check_should_train()

        assert should_train is True


class TestTrainingJobCreation:
    """Tests for automatic training job creation."""

    def test_trigger_training_creates_job(
        self, temp_scheduler_storage, mock_job_manager, mock_sample_storage, scheduler_config
    ):
        """trigger_training should create a job via job manager."""
        from auto_voice.training.training_scheduler import TrainingScheduler

        samples = [MagicMock() for _ in range(10)]
        mock_sample_storage.get_sample_count.return_value = 10
        mock_sample_storage.get_unprocessed_samples.return_value = samples

        scheduler = TrainingScheduler(
            profile_id="test-profile",
            job_manager=mock_job_manager,
            sample_storage=mock_sample_storage,
            config=scheduler_config,
            state_dir=temp_scheduler_storage,
        )

        job = scheduler.trigger_training()

        mock_job_manager.create_job.assert_called_once()
        assert job is not None

    def test_trigger_training_updates_last_time(
        self, temp_scheduler_storage, mock_job_manager, mock_sample_storage, scheduler_config
    ):
        """trigger_training should update last training timestamp."""
        from auto_voice.training.training_scheduler import TrainingScheduler

        mock_sample_storage.get_sample_count.return_value = 10
        mock_sample_storage.get_unprocessed_samples.return_value = [MagicMock() for _ in range(10)]

        scheduler = TrainingScheduler(
            profile_id="test-profile",
            job_manager=mock_job_manager,
            sample_storage=mock_sample_storage,
            config=scheduler_config,
            state_dir=temp_scheduler_storage,
        )

        before = scheduler._last_training_time
        scheduler.trigger_training()
        after = scheduler._last_training_time

        assert after is not None
        assert after != before

    def test_trigger_training_marks_samples_processed(
        self, temp_scheduler_storage, mock_job_manager, mock_sample_storage, scheduler_config
    ):
        """trigger_training should mark samples as processed."""
        from auto_voice.training.training_scheduler import TrainingScheduler

        samples = [MagicMock() for _ in range(10)]
        mock_sample_storage.get_sample_count.return_value = 10
        mock_sample_storage.get_unprocessed_samples.return_value = samples

        scheduler = TrainingScheduler(
            profile_id="test-profile",
            job_manager=mock_job_manager,
            sample_storage=mock_sample_storage,
            config=scheduler_config,
            state_dir=temp_scheduler_storage,
        )

        scheduler.trigger_training()

        mock_sample_storage.mark_samples_processed.assert_called_once()


class TestSchedulerCheckCycle:
    """Tests for the scheduler check cycle."""

    def test_check_and_trigger_when_needed(
        self, temp_scheduler_storage, mock_job_manager, mock_sample_storage, scheduler_config
    ):
        """check_and_trigger should create job when conditions met."""
        from auto_voice.training.training_scheduler import TrainingScheduler

        mock_sample_storage.get_sample_count.return_value = 15
        mock_sample_storage.get_unprocessed_samples.return_value = [MagicMock() for _ in range(15)]

        scheduler = TrainingScheduler(
            profile_id="test-profile",
            job_manager=mock_job_manager,
            sample_storage=mock_sample_storage,
            config=scheduler_config,
            state_dir=temp_scheduler_storage,
        )

        result = scheduler.check_and_trigger()

        assert result is True
        mock_job_manager.create_job.assert_called_once()

    def test_check_and_trigger_skips_when_not_needed(
        self, temp_scheduler_storage, mock_job_manager, mock_sample_storage, scheduler_config
    ):
        """check_and_trigger should not create job when conditions not met."""
        from auto_voice.training.training_scheduler import TrainingScheduler

        mock_sample_storage.get_sample_count.return_value = 5
        mock_sample_storage.get_unprocessed_samples.return_value = [MagicMock() for _ in range(5)]

        scheduler = TrainingScheduler(
            profile_id="test-profile",
            job_manager=mock_job_manager,
            sample_storage=mock_sample_storage,
            config=scheduler_config,
            state_dir=temp_scheduler_storage,
        )

        result = scheduler.check_and_trigger()

        assert result is False
        mock_job_manager.create_job.assert_not_called()


class TestSchedulerStatus:
    """Tests for scheduler status reporting."""

    def test_get_status_returns_info(
        self, temp_scheduler_storage, mock_job_manager, mock_sample_storage, scheduler_config
    ):
        """get_status should return scheduler state info."""
        from auto_voice.training.training_scheduler import TrainingScheduler

        mock_sample_storage.get_sample_count.return_value = 5

        scheduler = TrainingScheduler(
            profile_id="test-profile",
            job_manager=mock_job_manager,
            sample_storage=mock_sample_storage,
            config=scheduler_config,
            state_dir=temp_scheduler_storage,
        )

        status = scheduler.get_status()

        assert "profile_id" in status
        assert "current_samples" in status
        assert "min_threshold" in status
        assert "max_threshold" in status

    def test_status_includes_next_trigger_estimate(
        self, temp_scheduler_storage, mock_job_manager, mock_sample_storage, scheduler_config
    ):
        """Status should estimate samples needed for next trigger."""
        from auto_voice.training.training_scheduler import TrainingScheduler

        mock_sample_storage.get_sample_count.return_value = 5
        mock_sample_storage.get_unprocessed_samples.return_value = [MagicMock() for _ in range(5)]

        scheduler = TrainingScheduler(
            profile_id="test-profile",
            job_manager=mock_job_manager,
            sample_storage=mock_sample_storage,
            config=scheduler_config,
            state_dir=temp_scheduler_storage,
        )

        status = scheduler.get_status()

        assert "samples_until_trigger" in status
        assert status["samples_until_trigger"] == 5  # 10 - 5
