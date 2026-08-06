"""Tests for the svc_fork training lane (TrainingJobManager._run_fork_training).

Covers the job-lifecycle contract that the fork lane has to honour:

* a fork *cancel* must land as CANCELLED, not FAILED. ``ForkTrainingError`` is a
  plain ``RuntimeError`` and is deliberately NOT a subclass of the in-repo
  trainer's ``TrainingCancelledError``, so if it escapes ``_run_fork_training``
  it is swallowed by ``run_training``'s generic ``except Exception`` and the job
  is misreported.
* an OOM-shaped failure must keep the original message *and* gain the
  remediation hint.
* the metrics poller must be stopped before the results dict is assembled, so
  it cannot mutate ``job.results`` concurrently.

Plus the svc-fork config writer's precision/batch-size handling.
"""
import json
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from auto_voice.training.job_manager import (
    JobStatus,
    TrainingConfig,
    TrainingJob,
    TrainingJobManager,
    _resolve_fork_precision,
)
from auto_voice.training.svc_fork_trainer import (
    ForkTrainingError,
    _meaningful_tail,
    _set_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield TrainingJobManager(storage_path=Path(tmpdir))


@pytest.fixture
def fork_job(manager):
    job = TrainingJob(
        job_id="fork-job-1",
        profile_id="11111111-2222-3333-4444-555555555555",
        config=TrainingConfig(architecture="svc_fork", epochs=2),
    )
    job.status = JobStatus.RUNNING.value
    manager._jobs[job.job_id] = job
    return job


def _run_fork(manager, job, raises=None, returns=None, poller=None):
    """Drive _run_fork_training with train_svc_fork stubbed out."""
    def _fake_train(**kwargs):
        if raises is not None:
            raise raises
        return returns

    with tempfile.TemporaryDirectory() as train_dir:
        with patch(
            "auto_voice.training.svc_fork_trainer.train_svc_fork",
            side_effect=_fake_train,
        ), patch(
            "auto_voice.training.svc_fork_metrics.start_fork_metrics_poller",
            return_value=poller,
        ):
            manager._run_fork_training(
                job, job.job_id, Path(train_dir), [], threading.Event()
            )


# ---------------------------------------------------------------------------
# D1 - exception handling must cover the train_svc_fork call
# ---------------------------------------------------------------------------

class TestForkFailureLifecycle:
    def test_cancel_reports_cancelled_not_failed(self, manager, fork_job):
        """A fork cancel must not be misreported as a failure."""
        _run_fork(
            manager, fork_job,
            raises=ForkTrainingError("training cancelled by user"),
        )
        assert fork_job.status == JobStatus.CANCELLED.value

    def test_cancel_clears_profile_training_marks(self, manager, fork_job):
        """_mark_profile_training_started flips the profile to 'training' and
        records current_job_id; a cancel has to undo that or the GUI shows a
        job running forever for one that is already cancelled."""
        saved = {}

        class _Store:
            def load(self, pid):
                return {"profile_id": pid, "training_status": "training",
                        "current_job_id": fork_job.job_id,
                        "current_architecture": "svc_fork"}

            def save(self, profile):
                saved.update(profile)

        manager._get_profile_store = lambda: _Store()
        _run_fork(
            manager, fork_job,
            raises=ForkTrainingError("training cancelled by user"),
        )
        assert fork_job.status == JobStatus.CANCELLED.value
        assert saved["training_status"] == "pending"
        assert saved["current_job_id"] is None
        assert "current_architecture" not in saved

    def test_cancel_returns_trained_profile_to_ready(self, manager, fork_job):
        saved = {}

        class _Store:
            def load(self, pid):
                return {"profile_id": pid, "training_status": "training",
                        "current_job_id": fork_job.job_id,
                        "has_trained_model": True}

            def save(self, profile):
                saved.update(profile)

        manager._get_profile_store = lambda: _Store()
        _run_fork(manager, fork_job, raises=ForkTrainingError("cancelled"))
        assert saved["training_status"] == "ready"

    def test_cancel_leaves_another_jobs_marks_alone(self, manager, fork_job):
        """Cancelling a job that never owned the profile - a PENDING one, or a
        stale job for a profile another run has since claimed - must not
        clobber real state such as a 'failed' status and its error."""
        saved = {}

        class _Store:
            def load(self, pid):
                return {"profile_id": pid, "training_status": "failed",
                        "current_job_id": "some-other-job",
                        "last_training_error": "CUDA out of memory"}

            def save(self, profile):
                saved.update(profile)

        manager._get_profile_store = lambda: _Store()
        _run_fork(manager, fork_job, raises=ForkTrainingError("cancelled"))
        assert fork_job.status == JobStatus.CANCELLED.value
        assert saved == {}, "cancel overwrote another job's profile state"

    def test_oom_failure_gets_remediation_hint(self, manager, fork_job):
        """OOM stderr keeps its original text and gains the operator hint."""
        _run_fork(
            manager, fork_job,
            raises=ForkTrainingError(
                "train failed: torch.cuda.OutOfMemoryError: CUDA out of memory."
            ),
        )
        assert fork_job.status == JobStatus.FAILED.value
        assert "CUDA out of memory" in fork_job.error
        assert "svc_fork OOM:" in fork_job.error

    def test_non_oom_failure_keeps_message_unchanged(self, manager, fork_job):
        """A non-memory failure must not be dressed up as an OOM."""
        _run_fork(
            manager, fork_job,
            raises=ForkTrainingError("pre-hubert failed: exit code 1"),
        )
        assert fork_job.status == JobStatus.FAILED.value
        assert "svc_fork OOM:" not in fork_job.error

    def test_fork_error_does_not_escape(self, manager, fork_job):
        """ForkTrainingError must be handled here, not propagate to the caller.

        ForkTrainingError is not a TrainingCancelledError subclass, so anything
        that escapes reaches run_training's generic handler and loses the
        cancel/OOM distinction entirely.
        """
        from auto_voice.training.trainer import TrainingCancelledError
        assert not issubclass(ForkTrainingError, TrainingCancelledError)
        _run_fork(manager, fork_job, raises=ForkTrainingError("boom"))
        assert fork_job.status == JobStatus.FAILED.value


class TestPollerShutdown:
    def test_poller_stopped_before_results_assembly(self, manager, fork_job):
        """The stop event must be set (and the thread joined) as soon as
        training returns, so it cannot race the results dict."""
        joined = {"value": False}

        class _FakeThread:
            def join(self, timeout=None):
                joined["value"] = True

        _run_fork(
            manager, fork_job,
            returns={
                "model_path": "/tmp/G_2.pth",
                "registry_path": "/tmp/reg.json",
                "config_path": "/tmp/config.json",
                "epochs": 2,
                "speaker": "spk_test",
            },
            poller=_FakeThread(),
        )
        assert joined["value"] is True

    def test_poller_stopped_even_when_training_fails(self, manager, fork_job):
        joined = {"value": False}

        class _FakeThread:
            def join(self, timeout=None):
                joined["value"] = True

        _run_fork(
            manager, fork_job,
            raises=ForkTrainingError("CUDA out of memory"),
            poller=_FakeThread(),
        )
        assert joined["value"] is True


# ---------------------------------------------------------------------------
# D2 - precision / batch size written into the svc-fork config
# ---------------------------------------------------------------------------

class TestForkConfigWriter:
    def _write(self, tmp_path, **kwargs):
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps({"train": {}}))
        _set_config(cfg_path, 10, **kwargs)
        return json.loads(cfg_path.read_text())["train"]

    def test_defaults_to_fp32(self, tmp_path):
        """fp16_run=True stalls the GPU to ~0-5% util on the ComplexHalf STFT
        path; fp32 is the known-good default."""
        train = self._write(tmp_path)
        assert train["fp16_run"] is False
        assert train["bf16_run"] is False

    def test_default_batch_size_is_4(self, tmp_path):
        """batch_size 4 (not upstream's 16) is the actual Jetson OOM fix."""
        assert self._write(tmp_path)["batch_size"] == 4

    def test_explicit_batch_size_honoured(self, tmp_path):
        assert self._write(tmp_path, batch_size=8)["batch_size"] == 8

    def test_fp16_opt_in(self, tmp_path):
        assert self._write(tmp_path, precision="fp16")["fp16_run"] is True

    def test_bf16_does_not_enable_fp16(self, tmp_path):
        """svc-fork has no bf16 path; it must not silently become fp16."""
        train = self._write(tmp_path, precision="bf16")
        assert train["fp16_run"] is False
        assert train["bf16_run"] is False

    def test_epochs_and_keep_ckpts_preserved(self, tmp_path):
        train = self._write(tmp_path)
        assert train["epochs"] == 10
        assert train["keep_ckpts"] == 5


class TestSubprocessErrorTail:
    """Regression for the unreadable failure recorded against Conor:
    ``pre-hubert failed: 21%|##  | 195/911 [02:44<10:23,`` - a tqdm bar where
    the actual cause should be."""

    def _bars(self, n=200):
        return "".join(
            f"\r {i * 100 // n:2d}%|{'#' * 10}    | {i}/{n} [02:44<10:23,  4.3it/s]"
            for i in range(n)
        )

    def test_traceback_preferred_over_progress_bars(self):
        raw = (
            "loading checkpoint\n"
            + self._bars()
            + "\nTraceback (most recent call last):\n"
            '  File "hubert.py", line 42, in extract\n'
            "torch.cuda.OutOfMemoryError: CUDA out of memory.\n"
            + self._bars()
        )
        tail = _meaningful_tail(raw)
        assert "Traceback (most recent call last)" in tail
        assert "CUDA out of memory" in tail

    def test_oom_survives_for_the_signal_matcher(self):
        """The job manager's OOM detector greps this text; if the bars win,
        the remediation hint can never fire."""
        from auto_voice.training.job_manager import _FORK_OOM_SIGNAL

        raw = "torch.cuda.OutOfMemoryError: CUDA out of memory.\n" + self._bars()
        assert _FORK_OOM_SIGNAL.search(_meaningful_tail(raw))

    def test_progress_bars_stripped(self):
        tail = _meaningful_tail("real error here\n" + self._bars())
        assert "real error here" in tail
        # the hundreds of redraws are gone; only the single kept marker remains
        assert tail.count("it/s]") <= 1

    def test_furthest_progress_is_kept(self):
        """How far it got is half the diagnosis: these steps run per file over
        hundreds of clips, so 'died at 195/911' says the failure is
        data-dependent and points at which file to inspect."""
        raw = self._bars(n=200) + "\nValueError: something broke\n"
        tail = _meaningful_tail(raw)
        assert tail.startswith("[progress: ")
        assert "199/200" in tail.splitlines()[0]
        assert "ValueError: something broke" in tail

    def test_no_progress_marker_when_none_logged(self):
        tail = _meaningful_tail("Traceback (most recent call last):\nboom\n")
        assert "[progress:" not in tail

    def test_progress_prefix_respects_limit(self):
        raw = self._bars(n=50) + "\n" + ("x" * 5000)
        assert len(_meaningful_tail(raw, limit=300)) <= 300

    def test_bars_only_log_still_returns_something(self):
        assert _meaningful_tail(self._bars()).strip() != ""

    def test_respects_length_limit(self):
        assert len(_meaningful_tail("x\n" * 5000, limit=200)) <= 200


class TestForkMetricsApplication:
    def test_metrics_promoted_to_top_level(self, manager, fork_job):
        manager._apply_fork_metrics(
            fork_job, {"current_loss": 2.5, "current_step": 40}
        )
        payload = fork_job.to_dict()
        assert payload["current_loss"] == 2.5
        assert payload["current_step"] == 40

    def test_existing_results_preserved(self, manager, fork_job):
        fork_job.results = {"stage": "training", "engine": "svc_fork"}
        manager._apply_fork_metrics(fork_job, {"current_loss": 1.0})
        assert fork_job.results["stage"] == "training"
        assert fork_job.results["current_loss"] == 1.0

    def test_results_rebound_not_mutated(self, manager, fork_job):
        """A reader holding the old dict must not see it change under them."""
        fork_job.results = {"stage": "training"}
        original = fork_job.results
        manager._apply_fork_metrics(fork_job, {"current_loss": 1.0})
        assert fork_job.results is not original
        assert "current_loss" not in original

    def test_progress_not_clobbered(self, manager, fork_job):
        """emit_training_progress would recompute and overwrite progress; the
        fork lane's stage-based value must survive."""
        fork_job.update_progress(30)
        manager._apply_fork_metrics(
            fork_job, {"current_loss": 1.0, "current_epoch": 7}
        )
        assert fork_job.progress == 30

    def test_emits_socket_event(self, manager, fork_job):
        emitted = []
        manager._socketio = type(
            "S", (), {"emit": lambda self, n, d: emitted.append((n, d))}
        )()
        manager._apply_fork_metrics(fork_job, {"current_loss": 3.0})
        names = [n for n, _ in emitted]
        assert "training.progress" in names
        payload = dict(emitted[names.index("training.progress")][1])
        assert payload["current_loss"] == 3.0
        assert payload["job_id"] == fork_job.job_id


class TestForkWorkspaceRoot:
    def test_poller_uses_trainer_workspace_constant(self, manager, fork_job):
        """The poller must glob the same root the trainer writes into.

        A duplicated literal here is how the two drift apart - and a poller
        looking in the wrong place presents exactly like the 0/0-metrics bug
        this feature exists to fix.
        """
        from auto_voice.training import svc_fork_trainer

        seen = {}

        def _capture(**kwargs):
            seen.update(kwargs)
            return None

        with tempfile.TemporaryDirectory() as train_dir:
            with patch(
                "auto_voice.training.svc_fork_trainer.train_svc_fork",
                return_value={
                    "model_path": "/tmp/G.pth", "registry_path": "/tmp/r.json",
                    "config_path": "/tmp/c.json", "epochs": 1, "speaker": "s",
                },
            ), patch(
                "auto_voice.training.svc_fork_metrics.start_fork_metrics_poller",
                side_effect=_capture,
            ):
                manager._run_fork_training(
                    fork_job, fork_job.job_id, Path(train_dir), [],
                    threading.Event(),
                )

        assert seen["workspace_root"] == svc_fork_trainer.DEFAULT_WORKSPACE_ROOT


class TestForkPrecisionResolution:
    """The fork lane pins fp32 unless fp16 is explicitly unlocked.

    Four built-in training presets ship precision='fp16' (api_training.py),
    so honouring the config verbatim would silently re-introduce the
    ComplexHalf GPU stall for anyone selecting those presets.
    """

    def test_fp32_passes_through(self):
        assert _resolve_fork_precision("fp32", "job-1") == "fp32"

    def test_missing_precision_defaults_fp32(self):
        assert _resolve_fork_precision(None, "job-1") == "fp32"

    def test_bf16_collapses_to_fp32(self):
        """svc-fork has no bf16 path."""
        assert _resolve_fork_precision("bf16", "job-1") == "fp32"

    def test_preset_fp16_is_pinned_to_fp32(self, monkeypatch):
        monkeypatch.delenv("AUTOVOICE_SVCFORK_ALLOW_FP16", raising=False)
        assert _resolve_fork_precision("fp16", "job-1") == "fp32"

    def test_fp16_honoured_when_explicitly_unlocked(self, monkeypatch):
        monkeypatch.setenv("AUTOVOICE_SVCFORK_ALLOW_FP16", "1")
        assert _resolve_fork_precision("fp16", "job-1") == "fp16"
