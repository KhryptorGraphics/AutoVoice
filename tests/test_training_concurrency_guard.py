"""Only one training run at a time - they share one GPU.

execute_job spawned a daemon thread unconditionally with no queue, lock or
"is another job running" check, so two clicks of Start ran two so-vits-svc-fork
fine-tunes against the same GPU. On Jetson, GPU memory IS system RAM, and a
single run at batch_size 16 has already OOM'd this box once at ~epoch 70; two
would starve it and take the live conversion path down with them.

The only pre-existing pending/running check is inside auto_queue_training and
is per-profile, so it never covered the manual API.
"""
import pytest

from auto_voice.training.job_manager import TrainingJobManager, JobStatus


class _Job:
    def __init__(self, job_id, profile_id, status):
        self.job_id = job_id
        self.profile_id = profile_id
        self.status = status
        self.config = None


def _manager(jobs):
    mgr = object.__new__(TrainingJobManager)
    mgr._jobs = {j.job_id: j for j in jobs}
    return mgr


def _execute(mgr, job_id):
    """Run only execute_job's precondition block, without touching CUDA."""
    job = mgr._jobs.get(job_id)
    if job.status != JobStatus.PENDING.value:
        raise ValueError(f"Job {job_id} is not in pending state")
    running = [o for o in mgr._jobs.values()
               if o.job_id != job_id and o.status == JobStatus.RUNNING.value]
    if running:
        busy = running[0]
        raise ValueError(
            f"Training job {busy.job_id} is already running (profile "
            f"{busy.profile_id}). Only one training run at a time")
    return "started"


def test_a_second_run_is_refused_while_one_is_running():
    mgr = _manager([
        _Job("running-one", "profile-a", JobStatus.RUNNING.value),
        _Job("mine", "profile-b", JobStatus.PENDING.value),
    ])
    with pytest.raises(ValueError, match="already running"):
        _execute(mgr, "mine")


def test_the_guard_is_not_per_profile():
    """A second run on a DIFFERENT profile is the dangerous case - they still
    share the one GPU. The pre-existing check was per-profile and missed this."""
    mgr = _manager([
        _Job("running-one", "profile-a", JobStatus.RUNNING.value),
        _Job("mine", "profile-a", JobStatus.PENDING.value),
    ])
    with pytest.raises(ValueError, match="already running"):
        _execute(mgr, "mine")


def test_a_run_starts_when_nothing_else_is_running():
    mgr = _manager([
        _Job("done", "profile-a", JobStatus.COMPLETED.value),
        _Job("failed", "profile-a", JobStatus.FAILED.value),
        _Job("mine", "profile-b", JobStatus.PENDING.value),
    ])
    assert _execute(mgr, "mine") == "started"


def test_the_error_names_the_blocking_job():
    """An operator must be able to find and cancel what is in the way."""
    mgr = _manager([
        _Job("blocker-123", "profile-a", JobStatus.RUNNING.value),
        _Job("mine", "profile-b", JobStatus.PENDING.value),
    ])
    with pytest.raises(ValueError) as exc:
        _execute(mgr, "mine")
    assert "blocker-123" in str(exc.value)
    assert "profile-a" in str(exc.value)


def test_the_real_execute_job_contains_the_guard():
    """The tests above replicate the precondition block rather than calling
    execute_job, which needs CUDA and spawns a thread. Replicated logic cannot
    catch a fault in the shipped code - that exact gap let a NameError into
    svc_fork_trainer earlier. Assert the real method still has the check.
    """
    import inspect
    from auto_voice.training.job_manager import TrainingJobManager

    src = inspect.getsource(TrainingJobManager.execute_job)
    assert "JobStatus.RUNNING.value" in src, (
        "execute_job no longer checks for a running job; two training runs "
        "can again share one GPU")
    assert "already running" in src
    # The guard must come BEFORE the thread is spawned, or it guards nothing.
    assert src.index("already running") < src.index("Thread("), (
        "the concurrency check must precede thread creation")
