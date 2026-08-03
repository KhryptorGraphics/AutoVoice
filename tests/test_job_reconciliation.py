"""Startup reconciliation of jobs orphaned by a dead process.

A job that was RUNNING (or still queued) when the service died stays that way
in `training_jobs.json` forever - two jobs in the live deployment had been
`running` since 2026-07-12, through every restart since. That permanently
disables auto-training and auto-full-model promotion for the profile, because
both triggers skip a profile that has a pending-or-running job.
"""
import json
import tempfile
from pathlib import Path

import pytest

from auto_voice.training.job_manager import (
    JobStatus,
    TrainingConfig,
    TrainingJob,
    TrainingJobManager,
)


def _write_jobs(storage: Path, jobs):
    (storage / "training_jobs.json").write_text(
        json.dumps({"jobs": jobs, "updated_at": "2026-08-03T00:00:00"})
    )


def _job(job_id, status, profile_id="p-1", **extra):
    d = {
        "job_id": job_id,
        "profile_id": profile_id,
        "status": status,
        "created_at": "2026-07-12T16:18:14.941781",
        "progress": 40,
        "sample_ids": [],
        "config": None,
        "results": None,
        "error": None,
        "gpu_device": None,
        "is_paused": False,
    }
    d.update(extra)
    return d


@pytest.fixture
def storage():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def _manager_with_profile(storage, monkeypatch, profile, saved):
    """Manager whose profile store is a stub, patched per-test.

    Patched on the class (the manager is constructed inside the call under
    test, so there is no instance to patch yet) but via monkeypatch, so it is
    undone at test teardown rather than leaking into the rest of the suite.
    """
    class _Store:
        def load(self, pid):
            return dict(profile)

        def save(self, p):
            saved.clear()
            saved.update(p)

    monkeypatch.setattr(
        TrainingJobManager, "_get_profile_store", lambda self: _Store()
    )
    return TrainingJobManager(storage_path=storage)


class TestReconciliation:
    def test_running_job_is_failed(self, storage):
        _write_jobs(storage, [_job("job-dead", "running")])
        mgr = TrainingJobManager(storage_path=storage)
        job = mgr._jobs["job-dead"]
        assert job.status == JobStatus.FAILED.value
        assert "did not survive a restart" in job.error

    def test_pending_job_is_failed(self, storage):
        """Nothing re-queues work at startup, so a pending job never runs."""
        _write_jobs(storage, [_job("job-queued", "pending")])
        mgr = TrainingJobManager(storage_path=storage)
        job = mgr._jobs["job-queued"]
        assert job.status == JobStatus.FAILED.value
        assert "still queued" in job.error

    def test_terminal_jobs_untouched(self, storage):
        _write_jobs(storage, [
            _job("job-ok", "completed"),
            _job("job-bad", "failed", error="original error"),
            _job("job-gone", "cancelled"),
        ])
        mgr = TrainingJobManager(storage_path=storage)
        assert mgr._jobs["job-ok"].status == JobStatus.COMPLETED.value
        assert mgr._jobs["job-bad"].error == "original error"
        assert mgr._jobs["job-gone"].status == JobStatus.CANCELLED.value

    def test_paused_running_job_is_reconciled(self, storage):
        """A paused job is still RUNNING + is_paused; the process is gone too."""
        _write_jobs(storage, [_job("job-paused", "running", is_paused=True)])
        mgr = TrainingJobManager(storage_path=storage)
        assert mgr._jobs["job-paused"].status == JobStatus.FAILED.value

    def test_reconciliation_persists(self, storage):
        """The fix must survive - otherwise it re-runs on every start."""
        _write_jobs(storage, [_job("job-dead", "running")])
        TrainingJobManager(storage_path=storage)
        on_disk = json.loads((storage / "training_jobs.json").read_text())
        statuses = {j["job_id"]: j["status"] for j in on_disk["jobs"]}
        assert statuses["job-dead"] == "failed"

    def test_no_write_when_nothing_orphaned(self, storage):
        _write_jobs(storage, [_job("job-ok", "completed")])
        path = storage / "training_jobs.json"
        before = path.read_text()
        TrainingJobManager(storage_path=storage)
        assert path.read_text() == before

    def test_empty_and_missing_files_are_fine(self, storage):
        TrainingJobManager(storage_path=storage)  # no file at all
        _write_jobs(storage, [])
        TrainingJobManager(storage_path=storage)


class TestProfileMarkClearing:
    def test_owning_profile_marks_cleared(self, storage, monkeypatch):
        saved = {}
        profile = {
            "profile_id": "p-1",
            "training_status": "training",
            "current_job_id": "job-dead",
            "current_architecture": "svc_fork",
        }
        _write_jobs(storage, [_job("job-dead", "running", profile_id="p-1")])
        _manager_with_profile(storage, monkeypatch, profile, saved)
        assert saved["training_status"] == "pending"
        assert saved["current_job_id"] is None
        assert "current_architecture" not in saved

    def test_profile_claimed_by_newer_run_untouched(self, storage, monkeypatch):
        """A profile a newer job already owns must not be clobbered."""
        saved = {}
        profile = {
            "profile_id": "p-1",
            "training_status": "training",
            "current_job_id": "job-newer",
        }
        _write_jobs(storage, [_job("job-dead", "running", profile_id="p-1")])
        mgr = _manager_with_profile(storage, monkeypatch, profile, saved)
        assert mgr._jobs["job-dead"].status == JobStatus.FAILED.value
        assert saved == {}, "reconciliation clobbered another job's profile"


class TestUnblocksAutoTraining:
    def test_orphan_no_longer_blocks_auto_training(self, storage):
        """The whole point: a stuck job gates both auto-triggers via a
        pending_or_running filter, disabling retraining for that profile."""
        _write_jobs(storage, [_job("job-dead", "running", profile_id="p-1")])
        mgr = TrainingJobManager(storage_path=storage)
        blocking = [
            j for j in mgr.get_jobs_for_profile("p-1")
            if j.status in (JobStatus.PENDING.value, JobStatus.RUNNING.value)
        ]
        assert blocking == []
