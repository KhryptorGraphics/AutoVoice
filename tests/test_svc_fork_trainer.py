"""Tests for so-vits-svc-fork training + its job-manager wiring.

The fork runs in a separate conda env; all subprocess steps are mocked, so these
tests need neither the fork nor a GPU.
"""
import json
import threading
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

from auto_voice.training import svc_fork_trainer as fork


def _fake_train_dir(tmp_path):
    d = tmp_path / "train"
    d.mkdir()
    sf.write(str(d / "s1.wav"), np.zeros(44100, dtype=np.float32), 44100)
    sf.write(str(d / "s2.wav"), np.zeros(44100, dtype=np.float32), 44100)
    return d


def _patch_pipeline(monkeypatch):
    """Mock the fork subprocess steps; pre-config writes config.json, train
    writes a final checkpoint (both consumed by train_svc_fork)."""
    calls = []

    def fake_step(cmd, cwd=None, cancel_event=None, timeout=None):
        sub = cmd[1]
        calls.append(sub)
        if sub == "pre-config":
            cfg = Path(cwd) / "configs" / "44k"
            cfg.mkdir(parents=True, exist_ok=True)
            (cfg / "config.json").write_text('{"train": {}}')

    def fake_train(cmd, cwd, epochs, cancel_event=None, progress_cb=None):
        logs = Path(cwd) / "logs" / "44k"
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "G_0.pth").write_bytes(b"base")
        (logs / "G_100.pth").write_bytes(b"trained")
        if progress_cb:
            progress_cb(95, "training")

    monkeypatch.setattr(fork, "_run_step", fake_step)
    monkeypatch.setattr(fork, "_run_train", fake_train)
    return calls


def test_train_svc_fork_happy_path(tmp_path, monkeypatch):
    calls = _patch_pipeline(monkeypatch)
    train_dir = _fake_train_dir(tmp_path)
    data_dir = tmp_path / "data"
    seen = []

    res = fork.train_svc_fork(
        str(train_dir), "prof-1", "spk_prof", 100, str(data_dir),
        workspace_root=str(tmp_path / "ws"),
        progress_cb=lambda p, s: seen.append((p, s)),
    )

    # pipeline invoked in order
    assert calls == ["pre-split", "pre-resample", "pre-config", "pre-hubert"]
    # model promoted into the durable location
    assert (data_dir / "fork_models" / "prof-1_svcfork" / "G.pth").exists()
    assert (data_dir / "fork_models" / "prof-1_svcfork" / "config.json").exists()
    # registry written and points at the promoted model
    reg = json.loads((data_dir / "fork_models" / "prof-1.json").read_text())
    assert reg["speaker"] == "spk_prof"
    assert reg["model_path"].endswith("prof-1_svcfork/G.pth")
    assert reg["trained_epochs"] == 100
    assert res["epochs"] == 100 and res["engine"] == "svc_fork"
    assert (100, "completed") in seen


def test_train_svc_fork_no_audio_raises(tmp_path, monkeypatch):
    _patch_pipeline(monkeypatch)
    empty = tmp_path / "empty"; empty.mkdir()
    with pytest.raises(fork.ForkTrainingError, match="no audio"):
        fork.train_svc_fork(str(empty), "p", "spk", 100, str(tmp_path / "d"),
                            workspace_root=str(tmp_path / "ws"))


def test_train_svc_fork_cancel_raises(tmp_path, monkeypatch):
    _patch_pipeline(monkeypatch)
    train_dir = _fake_train_dir(tmp_path)
    ev = threading.Event(); ev.set()  # cancelled before the first step
    with pytest.raises(fork.ForkTrainingError, match="cancel"):
        fork.train_svc_fork(str(train_dir), "p", "spk", 100, str(tmp_path / "d"),
                            workspace_root=str(tmp_path / "ws"), cancel_event=ev)


def test_train_svc_fork_no_checkpoint_raises(tmp_path, monkeypatch):
    _patch_pipeline(monkeypatch)
    # train that writes only the base checkpoint -> no trained model
    monkeypatch.setattr(fork, "_run_train",
                        lambda cmd, cwd, epochs, cancel_event=None, progress_cb=None:
                        (Path(cwd) / "logs" / "44k").mkdir(parents=True, exist_ok=True))
    train_dir = _fake_train_dir(tmp_path)
    with pytest.raises(fork.ForkTrainingError, match="no checkpoint"):
        fork.train_svc_fork(str(train_dir), "p", "spk", 100, str(tmp_path / "d"),
                            workspace_root=str(tmp_path / "ws"))


def test_run_step_interrupted_mid_run_by_cancel():
    """A REAL subprocess is terminated promptly when cancel fires mid-step --
    the gap the mocked tests missed and the live smoke exposed (job stayed
    'running' because a blocking pre-hubert couldn't be interrupted)."""
    import time as _time
    ev = threading.Event()
    threading.Thread(target=lambda: (_time.sleep(0.4), ev.set()), daemon=True).start()
    t0 = _time.time()
    with pytest.raises(fork.ForkTrainingError, match="cancel"):
        fork._run_step(["sleep", "30"], cancel_event=ev)
    elapsed = _time.time() - t0
    assert elapsed < 5, f"cancel took {elapsed:.1f}s; step was not interrupted"


# ── job-manager wiring (call the method with a mock self) ────────────────────
def _mock_result(**kw):
    return {"engine": "svc_fork", "speaker": kw["speaker"], "epochs": 100,
            "model_path": "/m/G.pth", "config_path": "/m/config.json",
            "registry_path": "/m/p.json"}


def _call_fork_method(monkeypatch, tmp_path, train_impl):
    from auto_voice.training.job_manager import TrainingJobManager
    monkeypatch.setattr(fork, "train_svc_fork", train_impl)
    mgr = MagicMock()
    mgr._data_dir = tmp_path
    job = MagicMock()
    job.profile_id = "prof-1"
    job.config = None
    job.results = None
    TrainingJobManager._run_fork_training(mgr, job, "job1",
                                          str(tmp_path / "train"), ["a.wav"], None)
    return mgr, job


def test_run_fork_training_completes_job(monkeypatch, tmp_path):
    mgr, job = _call_fork_method(monkeypatch, tmp_path, _mock_result)
    job.complete.assert_called_once()
    completed = job.complete.call_args[0][0]
    assert completed["artifact_type"] == "full_model"
    assert completed["engine"] == "svc_fork"
    assert completed["adapter_path"] == "/m/G.pth"
    mgr._update_profile_training_state.assert_called_once()
    mgr._emit_completed_event.assert_called_once()
    job.fail.assert_not_called()


def test_run_fork_training_failure_marks_failed(monkeypatch, tmp_path):
    def boom(**kw):
        raise fork.ForkTrainingError("pre-hubert failed: boom")
    mgr, job = _call_fork_method(monkeypatch, tmp_path, boom)
    job.fail.assert_called_once()
    mgr._emit_failed_event.assert_called_once()
    job.complete.assert_not_called()


def test_run_fork_training_cancel_marks_cancelled(monkeypatch, tmp_path):
    def cancelled(**kw):
        raise fork.ForkTrainingError("cancelled")
    mgr, job = _call_fork_method(monkeypatch, tmp_path, cancelled)
    job.cancel.assert_called_once()
    mgr._emit_cancelled_event.assert_called_once()
    job.fail.assert_not_called()
