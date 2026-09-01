"""Train a so-vits-svc-fork model for a profile and register it for serving.

Mirrors the pipeline that produced the first working voice (melody corr ~0.98
where the in-repo decoder gave ~0): stage the profile's clean vocals, run the
fork's pre-split / pre-resample / pre-config / pre-hubert / train steps in the
isolated ``svcfork`` conda env (subprocess), then promote the final checkpoint
into ``<data_dir>/fork_models/<profile_id>_svcfork/`` and write the per-profile
registry that :mod:`auto_voice.inference.svc_fork_bridge` reads at serving time.

Runs in a separate conda env from the trainer/server, so all interaction is via
subprocess. Import-light so the job manager can call it without pulling torch.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

DEFAULT_SVC_BIN = os.environ.get(
    "AUTOVOICE_SVCFORK_BIN", "/home/kp/anaconda3/envs/svcfork/bin/svc")
DEFAULT_WORKSPACE_ROOT = os.environ.get(
    "AUTOVOICE_SVCFORK_WORKSPACE",
    "/home/kp/thordrive/autofusion/svcfork_ws/profiles")
_AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg")
_EPOCH_RE = re.compile(r"epoch[^0-9]*([0-9]+)", re.IGNORECASE)
# A tqdm bar redraw, e.g. " 21%|##########    | 195/911 [02:44<10:23, 4.3it/s]"
_TQDM_BAR = re.compile(r"^\s*\d{1,3}%\|.*\|\s*\d+/\d+")

ProgressCB = Optional[Callable[[int, str], None]]


class ForkTrainingError(RuntimeError):
    """A so-vits-svc-fork subprocess step failed."""


# Allocator config handed to every fork subprocess. `expandable_segments:True`
# never returns the segments it grows: measured on Thor over 162 steps, live
# tensors stayed flat at 2,401.8 MiB while the reserve climbed from 4 GB to
# 91 GB. GPU memory IS system RAM on Jetson, so the run degraded from ~1 s/step
# to 180 s/step and the box drifted toward OOM. Pinned here rather than left to
# the environment, because otherwise a training run's memory behaviour depends
# on whichever shell happened to launch gunicorn.
_ALLOC_CONF = "max_split_size_mb:512"


def _clean_env() -> Dict[str, str]:
    """Env for fork subprocesses: drop the serving PYTHONPATH so the fork imports
    only its own packages; pin PYTHONNOUSERSITE and the CUDA allocator."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    # Override rather than setdefault: the serving environment has been seen
    # exporting expandable_segments:True, and inheriting it is the regression
    # this guards against.
    env["PYTORCH_CUDA_ALLOC_CONF"] = _ALLOC_CONF
    return env


def _report(cb: ProgressCB, pct: int, stage: str) -> None:
    if cb is not None:
        cb(int(pct), stage)


def _terminate(proc) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _meaningful_tail(raw: str, limit: int = 800) -> str:
    """Extract the useful tail of a failed fork subprocess log.

    tqdm redraws its progress bar in place with ``\\r`` many times a second,
    so a naive ``raw[-600:]`` is almost always bar spam with the real cause
    scrolled out of view - which is how a genuine failure ends up recorded as
    ``pre-hubert failed: 21%|##  | 195/911 [02:44<10:23,`` and the operator
    has no idea what actually went wrong. Strip the bar redraws, and when a
    traceback is present prefer it over whatever happened to come last.

    The furthest progress marker is kept as a one-line prefix, because *how
    far it got* is the other half of the diagnosis: these steps run per file
    over hundreds of clips, so "died at 195/911" is what tells you the
    failure is data-dependent rather than a bad setup, and which file to go
    look at. It is the count that is useful, not the hundreds of redraws.
    """
    all_lines = [ln.strip() for ln in raw.replace("\r", "\n").splitlines()]
    progress = [ln for ln in all_lines if _TQDM_BAR.match(ln)]
    lines = [ln for ln in all_lines if ln and not _TQDM_BAR.match(ln)]

    furthest = f"[progress: {progress[-1]}]\n" if progress else ""
    budget = limit - len(furthest)

    if not lines:
        # Nothing but progress bars - fall back to the raw tail so we at
        # least return *something* rather than an empty error.
        return (furthest + raw[-budget:].strip()).strip()
    for i, line in enumerate(lines):
        if line.startswith("Traceback (most recent call last)"):
            return furthest + "\n".join(lines[i:])[-budget:]
    return furthest + "\n".join(lines)[-budget:]


def _supervise(proc, label: str, cancel_event, timeout: Optional[int],
               logf, on_new_output=None) -> None:
    """Poll a running fork subprocess: terminate promptly when ``cancel_event``
    fires (so a cancel lands mid-step, not only between steps) or on timeout,
    streaming new stdout to ``on_new_output``. Raises on nonzero exit."""
    start = time.time()
    read_pos = 0
    while proc.poll() is None:
        if cancel_event is not None and cancel_event.is_set():
            _terminate(proc)
            raise ForkTrainingError("cancelled")
        if timeout and (time.time() - start) > timeout:
            _terminate(proc)
            raise ForkTrainingError(f"{label} timed out after {timeout}s")
        if on_new_output is not None:
            logf.seek(read_pos)
            chunk = logf.read()
            read_pos = logf.tell()
            if chunk:
                on_new_output(chunk)
        time.sleep(0.3)
    if proc.returncode != 0:
        logf.seek(0)
        raise ForkTrainingError(f"{label} failed (rc={proc.returncode}): "
                                f"{_meaningful_tail(logf.read())}")


def _run_step(cmd: List[str], cwd: Optional[str] = None, cancel_event=None,
              timeout: Optional[int] = 1800) -> None:
    """Run a preprocessing step, interruptible mid-run by ``cancel_event``."""
    label = cmd[1] if len(cmd) > 1 else cmd[0]
    with tempfile.TemporaryFile(mode="w+") as logf:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=logf,
                                stderr=subprocess.STDOUT, text=True, env=_clean_env())
        _supervise(proc, label, cancel_event, timeout, logf)


def _run_train(cmd: List[str], cwd: str, epochs: int, cancel_event=None,
               progress_cb: ProgressCB = None, timeout: Optional[int] = None) -> None:
    """Run ``svc train``, mapping epochs -> 50..95% progress and terminating
    promptly if ``cancel_event`` fires (checked every 0.3s, not per-line)."""
    def _on_out(chunk: str) -> None:
        if progress_cb is not None and epochs > 0:
            eps = _EPOCH_RE.findall(chunk)
            if eps:
                frac = min(int(eps[-1]) / float(epochs), 1.0)
                _report(progress_cb, 50 + int(frac * 45), "training")

    with tempfile.TemporaryFile(mode="w+") as logf:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=logf,
                                stderr=subprocess.STDOUT, text=True, env=_clean_env())
        _supervise(proc, "train", cancel_event, timeout, logf, on_new_output=_on_out)


def _latest_epoch_checkpoint(logs_dir: Path) -> Optional[Path]:
    best, best_ep = None, -1
    for p in logs_dir.glob("G_*.pth"):
        m = re.match(r"G_(\d+)\.pth$", p.name)
        if m and p.name != "G_0.pth":
            ep = int(m.group(1))
            if ep > best_ep:
                best, best_ep = p, ep
    return best


def train_svc_fork(
    train_dir: str,
    profile_id: str,
    speaker: str,
    epochs: int,
    data_dir: str,
    *,
    svc_bin: str = DEFAULT_SVC_BIN,
    workspace_root: str = DEFAULT_WORKSPACE_ROOT,
    f0_method: str = "crepe",
    max_split_seconds: float = 10.0,
    progress_cb: ProgressCB = None,
    cancel_event=None,
    batch_size: Optional[int] = None,
    precision: str = "fp32",
) -> Dict[str, object]:
    """Fine-tune a so-vits-svc-fork model on the WAVs in ``train_dir`` and
    register it for ``profile_id``.

    Returns a result dict: model_path, config_path, speaker, epochs, engine.
    Raises ForkTrainingError on any failure (job manager maps it to job.fail).
    """
    wavs = [p for p in Path(train_dir).rglob("*") if p.suffix.lower() in _AUDIO_EXTS]
    if not wavs:
        raise ForkTrainingError(f"no audio files in train_dir {train_dir}")
    if epochs < 20:
        epochs = 100  # dataclass default (10) is far too few for a fine-tune

    ws = Path(workspace_root) / profile_id
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)
    raw = ws / "dataset_raw_raw" / speaker
    raw.mkdir(parents=True, exist_ok=True)
    for i, w in enumerate(wavs):
        shutil.copy2(w, raw / f"{i:04d}_{w.stem}.wav")
    _report(progress_cb, 5, "staging")

    def _check_cancel():
        if cancel_event is not None and cancel_event.is_set():
            raise ForkTrainingError("cancelled")

    ws_s = str(ws)
    # 1. split long clips -> dataset_raw/<speaker>/
    _run_step([svc_bin, "pre-split", "-i", str(ws / "dataset_raw_raw"),
               "-o", str(ws / "dataset_raw"), "-l", str(max_split_seconds)],
              cancel_event=cancel_event)
    _check_cancel(); _report(progress_cb, 15, "preprocessing")
    # 2-3. resample + config (cwd-relative dirs)
    _run_step([svc_bin, "pre-resample"], cwd=ws_s, cancel_event=cancel_event)
    _run_step([svc_bin, "pre-config"], cwd=ws_s, cancel_event=cancel_event)
    _set_config(ws / "configs" / "44k" / "config.json", epochs,
                batch_size=batch_size, precision=precision)
    _check_cancel(); _report(progress_cb, 30, "extracting features")
    # 4. ContentVec + F0 (single-process avoids the CUDA-fork deadlock)
    _run_step([svc_bin, "pre-hubert", "-n", "1", "-fm", f0_method], cwd=ws_s,
              cancel_event=cancel_event)
    _check_cancel(); _report(progress_cb, 50, "training")
    # 5. train from the auto-downloaded base (no tensorboard)
    _run_train([svc_bin, "train", "-nt"], cwd=ws_s, epochs=epochs,
               cancel_event=cancel_event, progress_cb=progress_cb)

    final = _latest_epoch_checkpoint(ws / "logs" / "44k")
    if final is None:
        raise ForkTrainingError("training produced no checkpoint")
    trained_ep = int(re.match(r"G_(\d+)\.pth$", final.name).group(1))

    dest = Path(data_dir) / "fork_models" / f"{profile_id}_svcfork"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final, dest / "G.pth")
    shutil.copy2(ws / "configs" / "44k" / "config.json", dest / "config.json")
    registry = Path(data_dir) / "fork_models" / f"{profile_id}.json"
    entry = {
        "profile_id": profile_id, "engine": "so-vits-svc-fork", "speaker": speaker,
        "model_path": str(dest / "G.pth"), "config_path": str(dest / "config.json"),
        "svc_bin": svc_bin, "f0_method": f0_method, "transpose": 0,
        "trained_epochs": trained_ep,
    }
    registry.write_text(json.dumps(entry, indent=2))
    _report(progress_cb, 100, "completed")

    from ..inference import svc_fork_bridge
    svc_fork_bridge.clear_cache()  # so a re-trained profile is picked up live
    return {"engine": "svc_fork", "speaker": speaker, "epochs": trained_ep,
            "model_path": entry["model_path"], "config_path": entry["config_path"],
            "registry_path": str(registry)}


def _set_config(
    config_path: Path,
    epochs: int,
    *,
    batch_size: Optional[int] = None,
    precision: str = "fp32",
) -> None:
    """Bound the epoch count and tighten memory-sensitive defaults for the
    so-vits-svc-fork trainer on Jetson / constrained GPUs.

    The upstream config template ships with batch_size=16, which OOMs at
    training time on Thor-class hardware once the scheduler holds activations
    across the discriminator + generator at long segment_size. Cap at 4 unless
    the caller explicitly asks for more. ``keep_ckpts`` stays at 5 so the
    existing retention policy is unchanged.

    Precision defaults to fp32 deliberately. ``fp16_run=True`` on torch 2.13
    hits an experimental ComplexHalf STFT path that stalls the GPU to ~0-5%
    utilisation - training looks alive but makes no progress - so fp16 is
    opt-in per run rather than the default. svc-fork has no bf16 path at all,
    so 'bf16' behaves as fp32 rather than silently degrading to fp16.
    """
    cfg = json.loads(config_path.read_text())
    cfg.setdefault("train", {})
    cfg["train"]["epochs"] = int(epochs)
    cfg["train"]["keep_ckpts"] = 5
    cfg["train"]["batch_size"] = int(batch_size) if batch_size else 4
    cfg["train"]["fp16_run"] = (precision == "fp16")
    cfg["train"]["bf16_run"] = False
    config_path.write_text(json.dumps(cfg, indent=2))
