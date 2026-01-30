"""Tests for training WebSocket events.

Phase 7: Test WebSocket events for training progress.

Tests verify:
- training.started event is emitted when job begins
- training.progress events are emitted with epoch/loss
- training.completed/failed events are emitted at end
"""

import pytest
from unittest.mock import MagicMock, patch, call
from datetime import datetime

from auto_voice.training.job_manager import (
    TrainingJobManager,
    TrainingJob,
    TrainingConfig,
    JobStatus,
)


@pytest.fixture
def temp_storage(tmp_path):
    """Create temporary storage directory."""
    storage_dir = tmp_path / "training_jobs"
    storage_dir.mkdir()
    return storage_dir


@pytest.fixture
def mock_socketio():
    """Create mock SocketIO instance."""
    socketio = MagicMock()
    socketio.emit = MagicMock()
    return socketio


@pytest.fixture
def manager_with_socketio(temp_storage, mock_socketio):
    """Create TrainingJobManager with mock SocketIO."""
    return TrainingJobManager(
        storage_path=temp_storage,
        require_gpu=False,
        socketio=mock_socketio,
    )


class TestTrainingStartedEvent:
    """Tests for training.started WebSocket event."""

    def test_manager_accepts_socketio_parameter(self, temp_storage, mock_socketio):
        """Task 7.1: TrainingJobManager should accept socketio parameter."""
        manager = TrainingJobManager(
            storage_path=temp_storage,
            require_gpu=False,
            socketio=mock_socketio,
        )
        assert manager is not None

    def test_emits_started_event_on_job_start(self, manager_with_socketio, mock_socketio):
        """Task 7.2: Should emit training.started when job begins."""
        # Create and start a job
        job = manager_with_socketio.create_job(
            profile_id="test-profile",
            sample_ids=["sample1", "sample2"],
        )

        # Start the job
        manager_with_socketio.update_job_status(
            job.job_id,
            JobStatus.RUNNING.value,
            gpu_device=0,
        )

        # Check emit was called
        mock_socketio.emit.assert_called()

        # Find the training.started call
        started_calls = [
            c for c in mock_socketio.emit.call_args_list
            if c[0][0] == 'training.started'
        ]
        assert len(started_calls) == 1

        # Check event data
        event_data = started_calls[0][0][1]
        assert event_data['job_id'] == job.job_id
        assert event_data['profile_id'] == 'test-profile'
        assert 'config' in event_data

    def test_started_event_includes_config(self, manager_with_socketio, mock_socketio):
        """Started event should include training configuration."""
        config = TrainingConfig(epochs=15, learning_rate=5e-5)
        job = manager_with_socketio.create_job(
            profile_id="test-profile",
            sample_ids=["sample1"],
            config=config,
        )

        manager_with_socketio.update_job_status(job.job_id, JobStatus.RUNNING.value)

        started_calls = [
            c for c in mock_socketio.emit.call_args_list
            if c[0][0] == 'training.started'
        ]
        assert len(started_calls) == 1

        event_data = started_calls[0][0][1]
        assert event_data['config']['epochs'] == 15


class TestTrainingProgressEvent:
    """Tests for training.progress WebSocket event."""

    def test_emits_progress_event(self, manager_with_socketio, mock_socketio):
        """Task 7.3-7.4: Should emit training.progress during training."""
        job = manager_with_socketio.create_job(
            profile_id="test-profile",
            sample_ids=["sample1"],
        )
        manager_with_socketio.update_job_status(job.job_id, JobStatus.RUNNING.value)

        # Reset mock to clear started event
        mock_socketio.emit.reset_mock()

        # Emit progress
        manager_with_socketio.emit_training_progress(
            job_id=job.job_id,
            epoch=3,
            total_epochs=10,
            step=150,
            total_steps=500,
            loss=0.45,
            learning_rate=1e-4,
        )

        # Check progress event was emitted
        mock_socketio.emit.assert_called()
        progress_calls = [
            c for c in mock_socketio.emit.call_args_list
            if c[0][0] == 'training.progress'
        ]
        assert len(progress_calls) == 1

        event_data = progress_calls[0][0][1]
        assert event_data['job_id'] == job.job_id
        assert event_data['epoch'] == 3
        assert event_data['total_epochs'] == 10
        assert event_data['loss'] == 0.45

    def test_progress_event_includes_all_fields(self, manager_with_socketio, mock_socketio):
        """Progress event should include all training metrics."""
        job = manager_with_socketio.create_job(
            profile_id="test-profile",
            sample_ids=["sample1"],
        )
        manager_with_socketio.update_job_status(job.job_id, JobStatus.RUNNING.value)
        mock_socketio.emit.reset_mock()

        manager_with_socketio.emit_training_progress(
            job_id=job.job_id,
            epoch=5,
            total_epochs=10,
            step=250,
            total_steps=500,
            loss=0.32,
            learning_rate=8e-5,
        )

        progress_calls = [
            c for c in mock_socketio.emit.call_args_list
            if c[0][0] == 'training.progress'
        ]
        event_data = progress_calls[0][0][1]

        # Verify all fields
        assert 'epoch' in event_data
        assert 'total_epochs' in event_data
        assert 'step' in event_data
        assert 'total_steps' in event_data
        assert 'loss' in event_data
        assert 'learning_rate' in event_data
        assert 'progress_percent' in event_data


class TestTrainingCompletedEvent:
    """Tests for training.completed WebSocket event."""

    def test_emits_completed_event(self, manager_with_socketio, mock_socketio):
        """Task 7.5-7.6: Should emit training.completed when job finishes."""
        job = manager_with_socketio.create_job(
            profile_id="test-profile",
            sample_ids=["sample1"],
        )
        manager_with_socketio.update_job_status(job.job_id, JobStatus.RUNNING.value)
        mock_socketio.emit.reset_mock()

        # Complete the job
        results = {
            'final_loss': 0.15,
            'epochs_trained': 10,
            'training_time_seconds': 120.5,
        }
        manager_with_socketio.update_job_status(
            job.job_id,
            JobStatus.COMPLETED.value,
            results=results,
        )

        # Check completed event was emitted
        completed_calls = [
            c for c in mock_socketio.emit.call_args_list
            if c[0][0] == 'training.completed'
        ]
        assert len(completed_calls) == 1

        event_data = completed_calls[0][0][1]
        assert event_data['job_id'] == job.job_id
        assert event_data['profile_id'] == 'test-profile'
        assert event_data['results']['final_loss'] == 0.15

    def test_completed_event_includes_results(self, manager_with_socketio, mock_socketio):
        """Completed event should include training results."""
        job = manager_with_socketio.create_job(
            profile_id="test-profile",
            sample_ids=["sample1", "sample2"],
        )
        manager_with_socketio.update_job_status(job.job_id, JobStatus.RUNNING.value)
        mock_socketio.emit.reset_mock()

        results = {
            'final_loss': 0.12,
            'initial_loss': 1.5,
            'epochs_trained': 10,
            'loss_curve': [1.5, 0.8, 0.5, 0.3, 0.2, 0.15, 0.13, 0.12, 0.12, 0.12],
        }
        manager_with_socketio.update_job_status(
            job.job_id,
            JobStatus.COMPLETED.value,
            results=results,
        )

        completed_calls = [
            c for c in mock_socketio.emit.call_args_list
            if c[0][0] == 'training.completed'
        ]
        event_data = completed_calls[0][0][1]

        assert 'results' in event_data
        assert event_data['results']['initial_loss'] == 1.5
        assert len(event_data['results']['loss_curve']) == 10


class TestTrainingFailedEvent:
    """Tests for training.failed WebSocket event."""

    def test_emits_failed_event(self, manager_with_socketio, mock_socketio):
        """Should emit training.failed when job fails."""
        job = manager_with_socketio.create_job(
            profile_id="test-profile",
            sample_ids=["sample1"],
        )
        manager_with_socketio.update_job_status(job.job_id, JobStatus.RUNNING.value)
        mock_socketio.emit.reset_mock()

        # Fail the job
        manager_with_socketio.update_job_status(
            job.job_id,
            JobStatus.FAILED.value,
            error="Out of GPU memory",
        )

        # Check failed event was emitted
        failed_calls = [
            c for c in mock_socketio.emit.call_args_list
            if c[0][0] == 'training.failed'
        ]
        assert len(failed_calls) == 1

        event_data = failed_calls[0][0][1]
        assert event_data['job_id'] == job.job_id
        assert event_data['error'] == "Out of GPU memory"

    def test_failed_event_includes_error_details(self, manager_with_socketio, mock_socketio):
        """Failed event should include error details."""
        job = manager_with_socketio.create_job(
            profile_id="test-profile",
            sample_ids=["sample1"],
        )
        manager_with_socketio.update_job_status(job.job_id, JobStatus.RUNNING.value)
        mock_socketio.emit.reset_mock()

        manager_with_socketio.update_job_status(
            job.job_id,
            JobStatus.FAILED.value,
            error="CUDA error: device-side assert triggered",
        )

        failed_calls = [
            c for c in mock_socketio.emit.call_args_list
            if c[0][0] == 'training.failed'
        ]
        event_data = failed_calls[0][0][1]

        assert 'error' in event_data
        assert 'CUDA error' in event_data['error']
        assert 'profile_id' in event_data


class TestWebSocketWithoutSocketIO:
    """Tests for graceful handling when socketio is not provided."""

    def test_works_without_socketio(self, temp_storage):
        """Manager should work without socketio (no events emitted)."""
        manager = TrainingJobManager(
            storage_path=temp_storage,
            require_gpu=False,
            socketio=None,
        )

        job = manager.create_job(
            profile_id="test-profile",
            sample_ids=["sample1"],
        )

        # These should not raise even without socketio
        manager.update_job_status(job.job_id, JobStatus.RUNNING.value)
        manager.update_job_progress(job.job_id, 50)
        manager.update_job_status(
            job.job_id,
            JobStatus.COMPLETED.value,
            results={'final_loss': 0.1},
        )

        assert manager.get_job(job.job_id).status == JobStatus.COMPLETED.value
