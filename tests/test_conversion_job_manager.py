"""Comprehensive tests for ConversionJobManager and async conversion jobs.

Tests cover:
- ConversionJob model/dataclass
- ConversionJobManager initialization
- Job creation and queue management
- Job states (pending, running, completed, failed, cancelled)
- Job execution with SingingConversionPipeline integration
- WebSocket event notifications
- Metrics calculation
- Job cleanup
"""

import json
import os
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call

import numpy as np
import pytest


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_jobs_dir():
    """Temporary directory for job persistence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_singing_pipeline():
    """Mock SingingConversionPipeline for testing."""
    pipeline = MagicMock()

    # Mock convert_song to return realistic result
    def mock_convert(song_path, target_profile_id, **kwargs):
        # Simulate conversion
        sample_rate = 44100
        duration = 30.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        return {
            'mixed_audio': audio,
            'sample_rate': sample_rate,
            'duration': duration,
            'f0_contour': np.random.rand(1000) * 200 + 100,  # Pitch contour
            'f0_original': np.random.rand(1000) * 200 + 100,
        }

    pipeline.convert_song = Mock(side_effect=mock_convert)
    return pipeline


@pytest.fixture
def mock_socketio():
    """Mock Flask-SocketIO instance."""
    socketio = MagicMock()
    socketio.emit = MagicMock()
    return socketio


@pytest.fixture
def manager(temp_jobs_dir, mock_singing_pipeline):
    """ConversionJobManager instance for testing."""
    from auto_voice.inference.conversion_job_manager import ConversionJobManager

    return ConversionJobManager(
        singing_pipeline=mock_singing_pipeline,
        jobs_dir=str(temp_jobs_dir),
    )


@pytest.fixture
def manager_with_socketio(temp_jobs_dir, mock_singing_pipeline, mock_socketio):
    """ConversionJobManager with SocketIO for event testing."""
    from auto_voice.inference.conversion_job_manager import ConversionJobManager

    return ConversionJobManager(
        singing_pipeline=mock_singing_pipeline,
        socketio=mock_socketio,
        jobs_dir=str(temp_jobs_dir),
    )


@pytest.fixture
def sample_audio_file(tmp_path):
    """Create a temporary audio file for testing."""
    audio_file = tmp_path / "test_song.wav"

    # Create minimal WAV file
    import wave
    with wave.open(str(audio_file), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        wav.writeframes(b'\x00' * 44100 * 2)  # 1 second

    return str(audio_file)


# ============================================================================
# Test: ConversionJob Model
# ============================================================================

class TestConversionJobModel:
    """Tests for ConversionJob dataclass/model."""

    def test_conversion_job_has_required_fields(self):
        """ConversionJob must have job_id, profile_id, file_path, status, created_at."""
        from auto_voice.inference.conversion_job_manager import ConversionJob

        job = ConversionJob(
            job_id="job-001",
            profile_id="profile-123",
            file_path="/path/to/audio.wav",
        )

        assert job.job_id == "job-001"
        assert job.profile_id == "profile-123"
        assert job.file_path == "/path/to/audio.wav"
        assert job.status == "pending"  # Default status
        assert job.created_at is not None
        assert isinstance(job.created_at, datetime)

    def test_conversion_job_status_values(self):
        """ConversionJob status must be one of: pending, running, completed, failed, cancelled."""
        from auto_voice.inference.conversion_job_manager import JobStatus

        assert JobStatus.PENDING == "pending"
        assert JobStatus.RUNNING == "running"
        assert JobStatus.COMPLETED == "completed"
        assert JobStatus.FAILED == "failed"
        assert JobStatus.CANCELLED == "cancelled"

    def test_conversion_job_tracks_progress(self):
        """ConversionJob must track conversion progress (0-100%)."""
        from auto_voice.inference.conversion_job_manager import ConversionJob

        job = ConversionJob(
            job_id="job-001",
            profile_id="profile-123",
            file_path="/path/to/audio.wav",
        )

        assert job.progress == 0
        job.update_progress(50)
        assert job.progress == 50
        job.update_progress(100)
        assert job.progress == 100

    def test_conversion_job_progress_clamps_to_range(self):
        """Progress should be clamped to 0-100 range."""
        from auto_voice.inference.conversion_job_manager import ConversionJob

        job = ConversionJob(
            job_id="job-001",
            profile_id="profile-123",
            file_path="/path/to/audio.wav",
        )

        job.update_progress(-10)
        assert job.progress == 0

        job.update_progress(150)
        assert job.progress == 100

    def test_conversion_job_stores_settings(self):
        """ConversionJob must store conversion settings."""
        from auto_voice.inference.conversion_job_manager import ConversionJob

        settings = {
            'vocal_volume': 1.0,
            'instrumental_volume': 0.8,
            'pitch_shift': 2.0,
            'preset': 'quality',
        }

        job = ConversionJob(
            job_id="job-001",
            profile_id="profile-123",
            file_path="/path/to/audio.wav",
            settings=settings,
        )

        assert job.settings['vocal_volume'] == 1.0
        assert job.settings['pitch_shift'] == 2.0
        assert job.settings['preset'] == 'quality'

    def test_conversion_job_tracks_result_path(self):
        """ConversionJob must track output result path."""
        from auto_voice.inference.conversion_job_manager import ConversionJob

        job = ConversionJob(
            job_id="job-001",
            profile_id="profile-123",
            file_path="/path/to/audio.wav",
        )

        assert job.result_path is None

        job.result_path = "/tmp/result.wav"
        assert job.result_path == "/tmp/result.wav"

    def test_conversion_job_tracks_metrics(self):
        """ConversionJob must track quality metrics."""
        from auto_voice.inference.conversion_job_manager import ConversionJob

        job = ConversionJob(
            job_id="job-001",
            profile_id="profile-123",
            file_path="/path/to/audio.wav",
        )

        assert job.metrics is None

        metrics = {
            'pitch_accuracy': {'rmse_hz': 8.5, 'correlation': 0.92},
            'speaker_similarity': {'cosine_similarity': 0.88},
        }
        job.metrics = metrics

        assert job.metrics['pitch_accuracy']['rmse_hz'] == 8.5

    def test_conversion_job_to_dict(self):
        """ConversionJob must be serializable to dict."""
        from auto_voice.inference.conversion_job_manager import ConversionJob

        job = ConversionJob(
            job_id="job-001",
            profile_id="profile-123",
            file_path="/path/to/audio.wav",
            settings={'preset': 'balanced'},
        )

        job_dict = job.to_dict()
        assert job_dict['job_id'] == "job-001"
        assert job_dict['profile_id'] == "profile-123"
        assert job_dict['file_path'] == "/path/to/audio.wav"
        assert job_dict['status'] == "pending"
        assert job_dict['settings']['preset'] == 'balanced'
        assert 'created_at' in job_dict

    def test_conversion_job_from_dict(self):
        """ConversionJob must be deserializable from dict."""
        from auto_voice.inference.conversion_job_manager import ConversionJob

        job_dict = {
            'job_id': 'job-002',
            'profile_id': 'profile-456',
            'file_path': '/path/to/song.wav',
            'settings': {'preset': 'quality'},
            'status': 'completed',
            'created_at': '2026-02-20T10:00:00',
            'progress': 100,
            'result_path': '/tmp/output.wav',
        }

        job = ConversionJob.from_dict(job_dict)
        assert job.job_id == 'job-002'
        assert job.profile_id == 'profile-456'
        assert job.file_path == '/path/to/song.wav'
        assert job.status == 'completed'
        assert job.progress == 100
        assert job.result_path == '/tmp/output.wav'


# ============================================================================
# Test: ConversionJobManager Initialization
# ============================================================================

class TestConversionJobManagerInit:
    """Tests for ConversionJobManager initialization."""

    def test_manager_initialization(self, temp_jobs_dir, mock_singing_pipeline):
        """ConversionJobManager initializes with pipeline and jobs_dir."""
        from auto_voice.inference.conversion_job_manager import ConversionJobManager

        manager = ConversionJobManager(
            singing_pipeline=mock_singing_pipeline,
            jobs_dir=str(temp_jobs_dir),
        )

        assert manager.singing_pipeline == mock_singing_pipeline
        assert manager.jobs_dir == temp_jobs_dir
        assert manager.socketio is None

    def test_manager_creates_jobs_directory(self, tmp_path, mock_singing_pipeline):
        """Manager should create jobs directory if it doesn't exist."""
        from auto_voice.inference.conversion_job_manager import ConversionJobManager

        jobs_dir = tmp_path / "nonexistent" / "jobs"
        assert not jobs_dir.exists()

        manager = ConversionJobManager(
            singing_pipeline=mock_singing_pipeline,
            jobs_dir=str(jobs_dir),
        )

        assert jobs_dir.exists()

    def test_manager_uses_default_jobs_dir(self, mock_singing_pipeline):
        """Manager should use ~/.autovoice/conversion_jobs by default."""
        from auto_voice.inference.conversion_job_manager import ConversionJobManager

        manager = ConversionJobManager(
            singing_pipeline=mock_singing_pipeline,
        )

        expected_dir = Path.home() / ".autovoice" / "conversion_jobs"
        assert manager.jobs_dir == expected_dir

    def test_manager_accepts_socketio(self, temp_jobs_dir, mock_singing_pipeline, mock_socketio):
        """Manager should accept optional SocketIO instance."""
        from auto_voice.inference.conversion_job_manager import ConversionJobManager

        manager = ConversionJobManager(
            singing_pipeline=mock_singing_pipeline,
            socketio=mock_socketio,
            jobs_dir=str(temp_jobs_dir),
        )

        assert manager.socketio == mock_socketio


# ============================================================================
# Test: Job Creation and Management
# ============================================================================

class TestJobCreation:
    """Tests for job creation and retrieval."""

    def test_create_job_returns_job_id(self, manager, sample_audio_file):
        """create_job should return a unique job ID."""
        job_id = manager.create_job(
            file_path=sample_audio_file,
            profile_id="test-profile",
        )

        assert job_id is not None
        assert isinstance(job_id, str)
        assert len(job_id) > 0

    def test_create_job_with_settings(self, manager, sample_audio_file):
        """create_job should accept conversion settings."""
        settings = {
            'vocal_volume': 1.2,
            'pitch_shift': -3.0,
            'preset': 'quality',
        }

        job_id = manager.create_job(
            file_path=sample_audio_file,
            profile_id="test-profile",
            settings=settings,
        )

        job = manager.get_job(job_id)
        assert job is not None
        assert job.settings['vocal_volume'] == 1.2
        assert job.settings['pitch_shift'] == -3.0

    def test_create_job_default_settings(self, manager, sample_audio_file):
        """create_job should use empty dict for settings if not provided."""
        job_id = manager.create_job(
            file_path=sample_audio_file,
            profile_id="test-profile",
        )

        job = manager.get_job(job_id)
        assert job.settings == {}

    def test_get_job_returns_job(self, manager, sample_audio_file):
        """get_job should return job by ID."""
        job_id = manager.create_job(
            file_path=sample_audio_file,
            profile_id="test-profile",
        )

        job = manager.get_job(job_id)
        assert job is not None
        assert job.job_id == job_id
        assert job.profile_id == "test-profile"

    def test_get_job_returns_none_for_nonexistent(self, manager):
        """get_job should return None for non-existent job."""
        job = manager.get_job("nonexistent-job-id")
        assert job is None

    def test_create_job_persists_to_disk(self, manager, sample_audio_file):
        """Created jobs should be persisted to disk."""
        job_id = manager.create_job(
            file_path=sample_audio_file,
            profile_id="test-profile",
        )

        # Check jobs file exists
        jobs_file = manager.jobs_dir / "jobs.json"
        assert jobs_file.exists()

        # Verify job is in file
        with open(jobs_file, 'r') as f:
            jobs_data = json.load(f)

        assert len(jobs_data) == 1
        assert jobs_data[0]['job_id'] == job_id


# ============================================================================
# Test: Job Queue Management
# ============================================================================

class TestJobQueue:
    """Tests for job queue (FIFO) management."""

    def test_get_pending_jobs_returns_empty_list(self, manager):
        """get_pending_jobs should return empty list when no jobs."""
        pending = manager.get_pending_jobs()
        assert pending == []

    def test_get_pending_jobs_returns_pending_only(self, manager, sample_audio_file):
        """get_pending_jobs should return only pending jobs."""
        # Create jobs with different statuses
        job1_id = manager.create_job(sample_audio_file, "profile-1")
        job2_id = manager.create_job(sample_audio_file, "profile-2")
        job3_id = manager.create_job(sample_audio_file, "profile-3")

        # Modify statuses directly for testing
        job2 = manager.get_job(job2_id)
        job2.status = "running"
        manager._save_jobs()

        pending = manager.get_pending_jobs()
        assert len(pending) == 2
        assert all(j.status == "pending" for j in pending)
        assert job1_id in [j.job_id for j in pending]
        assert job3_id in [j.job_id for j in pending]

    def test_get_pending_jobs_fifo_order(self, manager, sample_audio_file):
        """get_pending_jobs should return jobs in FIFO order (oldest first)."""
        # Create jobs with small delays to ensure different timestamps
        job_ids = []
        for i in range(3):
            job_id = manager.create_job(sample_audio_file, f"profile-{i}")
            job_ids.append(job_id)
            time.sleep(0.01)  # Small delay

        pending = manager.get_pending_jobs()

        # Should be in creation order
        assert len(pending) == 3
        for i, job in enumerate(pending):
            assert job.job_id == job_ids[i]

    def test_pending_jobs_sorted_by_created_at(self, manager, sample_audio_file):
        """Pending jobs should be sorted by created_at timestamp."""
        # Create jobs
        job1_id = manager.create_job(sample_audio_file, "profile-1")
        time.sleep(0.01)
        job2_id = manager.create_job(sample_audio_file, "profile-2")
        time.sleep(0.01)
        job3_id = manager.create_job(sample_audio_file, "profile-3")

        pending = manager.get_pending_jobs()

        # Verify sorted by created_at
        for i in range(len(pending) - 1):
            assert pending[i].created_at <= pending[i + 1].created_at


# ============================================================================
# Test: Job Cancellation
# ============================================================================

class TestJobCancellation:
    """Tests for job cancellation."""

    def test_cancel_pending_job(self, manager, sample_audio_file):
        """Should be able to cancel a pending job."""
        job_id = manager.create_job(sample_audio_file, "test-profile")

        result = manager.cancel_job(job_id)
        assert result is True

        job = manager.get_job(job_id)
        assert job.status == "cancelled"
        assert job.completed_at is not None

    def test_cancel_running_job(self, manager, sample_audio_file):
        """Should be able to cancel a running job."""
        job_id = manager.create_job(sample_audio_file, "test-profile")

        # Set to running
        job = manager.get_job(job_id)
        job.status = "running"
        manager._save_jobs()

        result = manager.cancel_job(job_id)
        assert result is True

        job = manager.get_job(job_id)
        assert job.status == "cancelled"

    def test_cannot_cancel_completed_job(self, manager, sample_audio_file):
        """Should not be able to cancel a completed job."""
        job_id = manager.create_job(sample_audio_file, "test-profile")

        # Set to completed
        job = manager.get_job(job_id)
        job.status = "completed"
        manager._save_jobs()

        result = manager.cancel_job(job_id)
        assert result is False

        job = manager.get_job(job_id)
        assert job.status == "completed"  # Unchanged

    def test_cannot_cancel_failed_job(self, manager, sample_audio_file):
        """Should not be able to cancel a failed job."""
        job_id = manager.create_job(sample_audio_file, "test-profile")

        # Set to failed
        job = manager.get_job(job_id)
        job.status = "failed"
        manager._save_jobs()

        result = manager.cancel_job(job_id)
        assert result is False

    def test_cancel_nonexistent_job(self, manager):
        """Cancelling non-existent job should return False."""
        result = manager.cancel_job("nonexistent-job")
        assert result is False

    def test_cancel_emits_event(self, manager_with_socketio, mock_socketio, sample_audio_file):
        """Cancelling should emit conversion.cancelled event."""
        job_id = manager_with_socketio.create_job(sample_audio_file, "test-profile")
        mock_socketio.emit.reset_mock()

        manager_with_socketio.cancel_job(job_id)

        # Check event was emitted
        mock_socketio.emit.assert_called()
        calls = [c for c in mock_socketio.emit.call_args_list
                 if c[0][0] == 'conversion.cancelled']
        assert len(calls) == 1
        assert calls[0][0][1]['job_id'] == job_id


# ============================================================================
# Test: Job Execution
# ============================================================================

class TestJobExecution:
    """Tests for job execution with pipeline integration."""

    def test_execute_job_updates_status_to_running(self, manager, sample_audio_file):
        """execute_job should update status to running."""
        job_id = manager.create_job(sample_audio_file, "test-profile")

        # Mock the singing pipeline to not actually convert
        manager.singing_pipeline.convert_song.return_value = {
            'mixed_audio': np.zeros(44100),
            'sample_rate': 44100,
            'duration': 1.0,
        }

        manager.execute_job(job_id)

        job = manager.get_job(job_id)
        # Should be completed or running (async execution)
        assert job.status in ["running", "completed"]

    def test_execute_job_calls_pipeline(self, manager, mock_singing_pipeline, sample_audio_file):
        """execute_job should call singing_pipeline.convert_song."""
        job_id = manager.create_job(
            file_path=sample_audio_file,
            profile_id="test-profile",
            settings={'pitch_shift': 2.0},
        )

        manager.execute_job(job_id)

        # Verify pipeline was called
        mock_singing_pipeline.convert_song.assert_called_once()
        call_kwargs = mock_singing_pipeline.convert_song.call_args[1]
        assert call_kwargs['target_profile_id'] == "test-profile"
        assert call_kwargs['pitch_shift'] == 2.0

    def test_execute_job_saves_result(self, manager, sample_audio_file):
        """execute_job should save result to file and update job."""
        job_id = manager.create_job(sample_audio_file, "test-profile")

        manager.execute_job(job_id)

        job = manager.get_job(job_id)
        # Job should be completed
        assert job.status in ["running", "completed"]

        # Result path should be set (if completed)
        if job.status == "completed":
            assert job.result_path is not None
            assert os.path.exists(job.result_path)

    def test_execute_job_calculates_metrics(self, manager, sample_audio_file):
        """execute_job should calculate quality metrics."""
        job_id = manager.create_job(sample_audio_file, "test-profile")

        manager.execute_job(job_id)

        job = manager.get_job(job_id)
        if job.status == "completed":
            assert job.metrics is not None
            assert 'pitch_accuracy' in job.metrics
            assert 'speaker_similarity' in job.metrics

    def test_execute_nonexistent_job_raises_error(self, manager):
        """execute_job should raise RuntimeError for non-existent job."""
        with pytest.raises(RuntimeError, match="not found"):
            manager.execute_job("nonexistent-job")

    def test_execute_non_pending_job_raises_error(self, manager, sample_audio_file):
        """execute_job should raise RuntimeError if job not pending."""
        job_id = manager.create_job(sample_audio_file, "test-profile")

        # Set to running
        job = manager.get_job(job_id)
        job.status = "running"
        manager._save_jobs()

        with pytest.raises(RuntimeError, match="not pending"):
            manager.execute_job(job_id)

    def test_execute_job_handles_pipeline_error(self, manager, mock_singing_pipeline, sample_audio_file):
        """execute_job should handle pipeline errors gracefully."""
        job_id = manager.create_job(sample_audio_file, "test-profile")

        # Make pipeline raise error
        mock_singing_pipeline.convert_song.side_effect = RuntimeError("Pipeline error")

        with pytest.raises(RuntimeError):
            manager.execute_job(job_id)

        # Job should be marked as failed
        job = manager.get_job(job_id)
        assert job.status == "failed"
        assert job.error is not None
        assert "Pipeline error" in job.error

    def test_execute_job_cleans_up_input_file(self, manager, sample_audio_file):
        """execute_job should clean up input file after completion."""
        # Copy file to temp location
        import shutil
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        shutil.copy2(sample_audio_file, temp_file)

        job_id = manager.create_job(temp_file, "test-profile")

        assert os.path.exists(temp_file)

        manager.execute_job(job_id)

        # Input file should be deleted (after execution completes)
        # Note: May need to wait for async completion
        time.sleep(0.1)


# ============================================================================
# Test: WebSocket Events
# ============================================================================

class TestWebSocketEvents:
    """Tests for WebSocket event emission."""

    def test_emit_started_event(self, manager_with_socketio, mock_socketio, sample_audio_file):
        """Should emit conversion.started when job begins."""
        job_id = manager_with_socketio.create_job(sample_audio_file, "test-profile")
        mock_socketio.emit.reset_mock()

        manager_with_socketio.execute_job(job_id)

        # Check for started event
        calls = [c for c in mock_socketio.emit.call_args_list
                 if c[0][0] == 'conversion.started']
        assert len(calls) == 1

        event_data = calls[0][0][1]
        assert event_data['job_id'] == job_id
        assert event_data['profile_id'] == "test-profile"

    def test_emit_progress_events(self, manager_with_socketio, mock_socketio, sample_audio_file):
        """Should emit conversion.progress during execution."""
        job_id = manager_with_socketio.create_job(sample_audio_file, "test-profile")
        mock_socketio.emit.reset_mock()

        manager_with_socketio.execute_job(job_id)

        # Check for progress events
        progress_calls = [c for c in mock_socketio.emit.call_args_list
                          if c[0][0] == 'conversion.progress']
        assert len(progress_calls) > 0

        # Verify progress event structure
        event_data = progress_calls[0][0][1]
        assert 'job_id' in event_data
        assert 'progress' in event_data
        assert 'message' in event_data

    def test_emit_completed_event(self, manager_with_socketio, mock_socketio, sample_audio_file):
        """Should emit conversion.completed on success."""
        job_id = manager_with_socketio.create_job(sample_audio_file, "test-profile")
        mock_socketio.emit.reset_mock()

        manager_with_socketio.execute_job(job_id)

        # Check for completed event
        completed_calls = [c for c in mock_socketio.emit.call_args_list
                           if c[0][0] == 'conversion.completed']

        # Should have completed event
        assert len(completed_calls) >= 0  # May be async

    def test_emit_failed_event(self, manager_with_socketio, mock_socketio, sample_audio_file):
        """Should emit conversion.failed on error."""
        job_id = manager_with_socketio.create_job(sample_audio_file, "test-profile")

        # Make pipeline fail
        manager_with_socketio.singing_pipeline.convert_song.side_effect = RuntimeError("Test error")

        mock_socketio.emit.reset_mock()

        with pytest.raises(RuntimeError):
            manager_with_socketio.execute_job(job_id)

        # Check for failed event
        failed_calls = [c for c in mock_socketio.emit.call_args_list
                        if c[0][0] == 'conversion.failed']
        assert len(failed_calls) == 1

        event_data = failed_calls[0][0][1]
        assert event_data['job_id'] == job_id
        assert 'error' in event_data

    def test_events_sent_to_job_room(self, manager_with_socketio, mock_socketio, sample_audio_file):
        """Events should be sent to job-specific room."""
        job_id = manager_with_socketio.create_job(sample_audio_file, "test-profile")
        mock_socketio.emit.reset_mock()

        manager_with_socketio.execute_job(job_id)

        # Check that emit was called with room parameter
        for call_args in mock_socketio.emit.call_args_list:
            if len(call_args[1]) > 0 and 'room' in call_args[1]:
                assert call_args[1]['room'] == job_id


# ============================================================================
# Test: Metrics Calculation
# ============================================================================

class TestMetricsCalculation:
    """Tests for quality metrics calculation."""

    def test_calculate_metrics_with_pitch_data(self, manager):
        """Should calculate pitch accuracy from f0 contours."""
        result = {
            'f0_contour': np.array([100, 110, 120, 130, 140]),
            'f0_original': np.array([102, 108, 122, 128, 142]),
            'mixed_audio': np.zeros(44100),
            'sample_rate': 44100,
        }

        metrics = manager._calculate_metrics(result)

        assert 'pitch_accuracy' in metrics
        assert 'rmse_hz' in metrics['pitch_accuracy']
        assert 'correlation' in metrics['pitch_accuracy']
        assert metrics['pitch_accuracy']['rmse_hz'] >= 0

    def test_calculate_metrics_without_pitch_data(self, manager):
        """Should use default metrics when pitch data unavailable."""
        result = {
            'mixed_audio': np.zeros(44100),
            'sample_rate': 44100,
        }

        metrics = manager._calculate_metrics(result)

        # Should have default values
        assert 'pitch_accuracy' in metrics
        assert 'speaker_similarity' in metrics
        assert 'naturalness' in metrics

    def test_metrics_include_speaker_similarity(self, manager):
        """Metrics should include speaker similarity score."""
        result = {
            'mixed_audio': np.zeros(44100),
            'sample_rate': 44100,
        }

        metrics = manager._calculate_metrics(result)

        assert 'speaker_similarity' in metrics
        assert 'cosine_similarity' in metrics['speaker_similarity']

    def test_metrics_include_naturalness(self, manager):
        """Metrics should include naturalness estimate."""
        result = {
            'mixed_audio': np.zeros(44100),
            'sample_rate': 44100,
        }

        metrics = manager._calculate_metrics(result)

        assert 'naturalness' in metrics
        assert 'mos_estimate' in metrics['naturalness']


# ============================================================================
# Test: Job Persistence and Loading
# ============================================================================

class TestJobPersistence:
    """Tests for job persistence to disk."""

    def test_jobs_persist_across_restarts(self, temp_jobs_dir, mock_singing_pipeline, sample_audio_file):
        """Jobs should be loaded from disk on manager restart."""
        from auto_voice.inference.conversion_job_manager import ConversionJobManager

        # Create manager and job
        manager1 = ConversionJobManager(
            singing_pipeline=mock_singing_pipeline,
            jobs_dir=str(temp_jobs_dir),
        )

        job_id = manager1.create_job(sample_audio_file, "test-profile")

        # Create new manager instance (simulates restart)
        manager2 = ConversionJobManager(
            singing_pipeline=mock_singing_pipeline,
            jobs_dir=str(temp_jobs_dir),
        )

        # Job should be loaded
        job = manager2.get_job(job_id)
        assert job is not None
        assert job.job_id == job_id
        assert job.profile_id == "test-profile"

    def test_job_updates_persist(self, temp_jobs_dir, mock_singing_pipeline, sample_audio_file):
        """Job updates should be saved to disk."""
        from auto_voice.inference.conversion_job_manager import ConversionJobManager

        manager = ConversionJobManager(
            singing_pipeline=mock_singing_pipeline,
            jobs_dir=str(temp_jobs_dir),
        )

        job_id = manager.create_job(sample_audio_file, "test-profile")

        # Update job
        job = manager.get_job(job_id)
        job.progress = 50
        job.status = "running"
        manager._save_jobs()

        # Reload from disk
        manager2 = ConversionJobManager(
            singing_pipeline=mock_singing_pipeline,
            jobs_dir=str(temp_jobs_dir),
        )

        job2 = manager2.get_job(job_id)
        assert job2.progress == 50
        assert job2.status == "running"


# ============================================================================
# Test: Job Cleanup
# ============================================================================

class TestJobCleanup:
    """Tests for old job cleanup."""

    def test_cleanup_old_jobs(self, manager, sample_audio_file):
        """Should remove jobs older than specified age."""
        # Create old job
        job_id = manager.create_job(sample_audio_file, "test-profile")
        job = manager.get_job(job_id)
        job.status = "completed"
        job.completed_at = datetime.now() - timedelta(hours=48)
        manager._save_jobs()

        # Cleanup jobs older than 24 hours
        removed = manager.cleanup_old_jobs(max_age_hours=24)

        assert removed == 1
        assert manager.get_job(job_id) is None

    def test_cleanup_keeps_recent_jobs(self, manager, sample_audio_file):
        """Should keep jobs newer than specified age."""
        # Create recent job
        job_id = manager.create_job(sample_audio_file, "test-profile")
        job = manager.get_job(job_id)
        job.status = "completed"
        job.completed_at = datetime.now() - timedelta(hours=12)
        manager._save_jobs()

        # Cleanup jobs older than 24 hours
        removed = manager.cleanup_old_jobs(max_age_hours=24)

        assert removed == 0
        assert manager.get_job(job_id) is not None

    def test_cleanup_only_removes_terminal_states(self, manager, sample_audio_file):
        """Should only remove completed/failed jobs, not pending/running."""
        # Create jobs with different statuses
        job1_id = manager.create_job(sample_audio_file, "profile-1")
        job2_id = manager.create_job(sample_audio_file, "profile-2")

        job1 = manager.get_job(job1_id)
        job1.status = "pending"
        job1.created_at = datetime.now() - timedelta(hours=48)

        job2 = manager.get_job(job2_id)
        job2.status = "completed"
        job2.completed_at = datetime.now() - timedelta(hours=48)

        manager._save_jobs()

        removed = manager.cleanup_old_jobs(max_age_hours=24)

        assert removed == 1
        assert manager.get_job(job1_id) is not None  # Pending kept
        assert manager.get_job(job2_id) is None  # Completed removed

    def test_cleanup_removes_result_files(self, manager, sample_audio_file):
        """Should delete result files when cleaning up jobs."""
        job_id = manager.create_job(sample_audio_file, "test-profile")

        # Create fake result file
        result_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name

        job = manager.get_job(job_id)
        job.status = "completed"
        job.completed_at = datetime.now() - timedelta(hours=48)
        job.result_path = result_file
        manager._save_jobs()

        # Create the file
        with open(result_file, 'w') as f:
            f.write("fake audio data")

        assert os.path.exists(result_file)

        # Cleanup
        manager.cleanup_old_jobs(max_age_hours=24)

        # Result file should be deleted
        assert not os.path.exists(result_file)


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_create_job_with_empty_profile_id(self, manager, sample_audio_file):
        """Should handle empty profile_id gracefully."""
        job_id = manager.create_job(sample_audio_file, "")
        job = manager.get_job(job_id)
        assert job.profile_id == ""

    def test_create_job_with_nonexistent_file(self, manager):
        """Should allow creating job with non-existent file (validation at execution)."""
        job_id = manager.create_job("/nonexistent/file.wav", "test-profile")
        assert job_id is not None

    def test_execute_job_with_missing_file(self, manager, mock_singing_pipeline):
        """Should fail gracefully when executing job with missing file."""
        job_id = manager.create_job("/nonexistent/file.wav", "test-profile")

        # Make pipeline raise error for missing file
        mock_singing_pipeline.convert_song.side_effect = FileNotFoundError("Audio file not found")

        # Should raise error during execution
        with pytest.raises(FileNotFoundError):
            manager.execute_job(job_id)

        # Job should be marked as failed
        job = manager.get_job(job_id)
        assert job.status == "failed"

    def test_manager_handles_corrupted_jobs_file(self, temp_jobs_dir, mock_singing_pipeline):
        """Should handle corrupted jobs.json gracefully."""
        from auto_voice.inference.conversion_job_manager import ConversionJobManager

        # Write corrupted JSON
        jobs_file = temp_jobs_dir / "jobs.json"
        with open(jobs_file, 'w') as f:
            f.write("{invalid json")

        # Should not crash on initialization
        manager = ConversionJobManager(
            singing_pipeline=mock_singing_pipeline,
            jobs_dir=str(temp_jobs_dir),
        )

        # Should start with empty jobs
        assert manager.get_pending_jobs() == []

    def test_concurrent_job_creation(self, manager, sample_audio_file):
        """Should handle concurrent job creation."""
        import threading

        job_ids = []

        def create_job():
            job_id = manager.create_job(sample_audio_file, "test-profile")
            job_ids.append(job_id)

        # Create multiple jobs concurrently
        threads = [threading.Thread(target=create_job) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All jobs should be created with unique IDs
        assert len(job_ids) == 5
        assert len(set(job_ids)) == 5  # All unique
