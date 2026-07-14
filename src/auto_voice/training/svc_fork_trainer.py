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
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional

DEFAULT_SVC_BIN = os.environ.get(
    "AUTOVOICE_SVCFORK_BIN", "/home/kp/anaconda3/envs/svcfork/bin/svc")
DEFAULT_WORKSPACE_ROOT = os.environ.get(
    "AUTOVOICE_SVCFORK_WORKSPACE",
    "/home/kp/thordrive/autofusion/svcfork_ws/profiles")
_AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg")
_EPOCH_RE = re.compile(r"epoch[^0-9]*([0-9]+)", re.IGNORECASE)
# Best-effort metric parsing from the so-vits-svc-fork (PyTorch-Lightning) train
# output. Epoch is reliable; loss/lr appear in the Lightning progress bar
# ("loss_gen_all=..", "lr=..") but key names vary, so they are optional.
_LOSS_RE = re.compile(r"loss[/_a-z0-9]*\s*[=:]\s*([0-9]+\.[0-9]+)", re.IGNORECASE)
_LR_RE = re.compile(r"\blr\s*[=:]\s*([0-9]+\.?[0-9]*(?:[eE][-+]?[0-9]+)?)", re.IGNORECASE)

ProgressCB = Optional[Callable[..., None]]


class ForkTrainingError(RuntimeError):
    """A so-vits-svc-fork subprocess step failed."""


def _clean_env() -> Dict[str, str]:
    """Env for fork subprocesses: drop the serving PYTHONPATH so the fork imports
    only its own packages; pin PYTHONNOUSERSITE."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _report(cb: ProgressCB, pct: int, stage: str,
            metrics: Optional[Dict] = None) -> None:
    if cb is not None:
        cb(int(pct), stage, metrics)


def _terminate(proc) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


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
                                f"{logf.read()[-600:]}")


def _run_step(cmd: List[str], cwd: Optional[str] = None, cancel_event=None,
              timeout: Optional[int] = 1800) -> None:
    """Run a preprocessing step, interruptible mid-run by ``cancel_event``."""
    label = cmd[1] if len(cmd) > 1 else cmd[0]
    with tempfile.TemporaryFile(mode="w+") as logf:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=logf,
                                stderr=subprocess.STDOUT, text=True, env=_clean_env())
        _supervise(proc, label, cancel_event, timeout, logf)


def _parse_train_metrics(text: str, epochs: int) -> Optional[Dict]:
    """Parse the latest epoch (reliable) plus loss from a rolling buffer of fork
    train output. so-vits-svc-fork (Rich) prints ``loss/g/total:`` then the value
    a few chars later, often split across 0.3s read-chunks, so the caller passes
    a buffer spanning chunks rather than a single chunk. Returns None if no epoch
    token is present in the buffer."""
    eps = _EPOCH_RE.findall(text)
    if not eps:
        return None
    epoch = int(eps[-1])
    frac = min(epoch / float(epochs), 1.0) if epochs > 0 else 0.0
    metrics: Dict[str, object] = {
        "epoch": epoch,
        "total_epochs": epochs,
        "pct": 50 + int(frac * 45),
    }
    # Main training loss = generator total. Rich renders "loss/g/total:" then the
    # numeric value after whitespace/box-glyphs (bounded lookahead avoids grabbing
    # a stray digit from an unrelated column).
    losses = re.findall(r"loss/g/total:[^0-9]{0,80}([0-9]+\.[0-9]+)", text) \
        or _LOSS_RE.findall(text)
    if losses:
        try:
            metrics["loss"] = float(losses[-1])
        except (ValueError, TypeError):
            pass
    lrs = _LR_RE.findall(text)
    if lrs:
        try:
            metrics["lr"] = float(lrs[-1])
        except (ValueError, TypeError):
            pass
    return metrics


def _latest_tb_scalars(logs_dir: Path) -> Optional[Dict]:
    """Read the latest ``loss/g/total`` and ``lr`` scalars from the fork's
    TensorBoard event files. so-vits-svc-fork (Lightning ``TensorBoardLogger``)
    always writes these under ``logs/44k/lightning_logs`` — the ``-nt`` flag only
    skips launching the TB *web server*, not the event writing — and its per-step
    losses appear only in Rich's terminal display, never the redirected pipe, so
    this is the reliable live-loss source. Best-effort: returns None on any error."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except Exception:
        return None
    try:
        events = sorted(
            Path(logs_dir).rglob("events.out.tfevents*"),
            key=lambda p: p.stat().st_mtime,
        )
        if not events:
            return None
        ea = EventAccumulator(str(events[-1]), size_guidance={"scalars": 20})
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        out: Dict[str, object] = {}
        if "loss/g/total" in tags:
            pts = ea.Scalars("loss/g/total")
            if pts:
                out["loss"] = float(pts[-1].value)
                out["step"] = int(pts[-1].step)
        if "lr" in tags:
            pts = ea.Scalars("lr")
            if pts:
                out["lr"] = float(pts[-1].value)
        return out or None
    except Exception:
        return None


def _run_train(cmd: List[str], cwd: str, epochs: int, cancel_event=None,
               progress_cb: ProgressCB = None, timeout: Optional[int] = None,
               debug_log_path: Optional[str] = None) -> None:
    """Run ``svc train``, forwarding live metrics via ``progress_cb`` and
    terminating promptly if ``cancel_event`` fires (checked every 0.3s).

    Epoch/progress come from the subprocess stdout; loss and lr come from a
    background poller over the fork's TensorBoard event files (its per-step
    losses render only in Rich's terminal display, not the redirected pipe).
    Raw output is mirrored to ``debug_log_path`` (capped) for inspection."""
    dbg = (Path(debug_log_path) if debug_log_path
           else (Path(cwd) / "fork_train_debug.log" if cwd else None))
    if dbg is not None:
        try:
            dbg.parent.mkdir(parents=True, exist_ok=True)
            dbg.write_text("")
        except Exception:
            dbg = None

    logs_dir = (Path(cwd) / "logs" / "44k" / "lightning_logs") if cwd else None
    buf = {"s": ""}
    latest = {"epoch": 0, "pct": 50, "loss": None, "lr": None}
    stop = threading.Event()

    def _emit() -> None:
        if progress_cb is None:
            return
        m: Dict[str, object] = {
            "epoch": latest["epoch"],
            "total_epochs": epochs,
            "pct": latest["pct"],
        }
        if latest["loss"] is not None:
            m["loss"] = latest["loss"]
        if latest["lr"] is not None:
            m["lr"] = latest["lr"]
        _report(progress_cb, latest["pct"], "training", m)

    def _on_out(chunk: str) -> None:
        if dbg is not None:
            try:
                with open(dbg, "a", encoding="utf-8") as fh:
                    fh.write(chunk)
                if dbg.stat().st_size > 2_000_000:
                    tail = dbg.read_text(encoding="utf-8", errors="ignore")[-1_000_000:]
                    dbg.write_text(tail)
            except Exception:
                pass
        if progress_cb is not None and epochs > 0:
            buf["s"] = (buf["s"] + chunk)[-8000:]
            metrics = _parse_train_metrics(buf["s"], epochs)
            if metrics is not None:
                latest["epoch"] = metrics["epoch"]
                latest["pct"] = metrics["pct"]
                if metrics.get("loss") is not None and latest["loss"] is None:
                    latest["loss"] = metrics["loss"]
                _emit()

    def _tb_poller() -> None:
        while not stop.wait(6.0):
            if logs_dir is None or progress_cb is None:
                continue
            sc = _latest_tb_scalars(logs_dir)
            if not sc:
                continue
            if sc.get("loss") is not None:
                latest["loss"] = sc["loss"]
            if sc.get("lr") is not None:
                latest["lr"] = sc["lr"]
            _emit()

    poller = threading.Thread(target=_tb_poller, name="svcfork-tb-poll",
                              daemon=True)
    poller.start()
    try:
        with tempfile.TemporaryFile(mode="w+") as logf:
            proc = subprocess.Popen(cmd, cwd=cwd, stdout=logf,
                                    stderr=subprocess.STDOUT, text=True,
                                    env=_clean_env())
            _supervise(proc, "train", cancel_event, timeout, logf,
                       on_new_output=_on_out)
    finally:
        stop.set()


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
    precision: str = "fp16",
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
    _set_config(ws / "configs" / "44k" / "config.json", epochs, precision=precision)
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


def _set_config(config_path: Path, epochs: int, precision: str = "fp16") -> None:
    """Bound the epoch count and keep a few checkpoints in the fork config."""
    cfg = json.loads(config_path.read_text())
    cfg.setdefault("train", {})
    cfg["train"]["epochs"] = int(epochs)
    cfg["train"]["keep_ckpts"] = 5
    # Precision: fp16 mixed precision -> Lightning "16-mixed" (train.py) uses
    # Thor's tensor cores and halves memory; the caller may request fp32 (full
    # precision) explicitly. fp32 roughly doubles activation/gradient memory, so
    # pair it with a smaller batch to keep peak near the fp16+batch-4 baseline.
    fp32 = str(precision).lower() == "fp32"
    cfg["train"]["fp16_run"] = not fp32
    # Cap batch size for Thor's shared GPU. Full-model GAN activations at the
    # fork default (16, batching the whole small set at once) reached ~63GB in
    # FP32 and OOM'd the unified memory (NVRM NV_ERR_NO_MEMORY). The fork has no
    # drop_last, but the model tolerates a size-1 tail batch (validation runs
    # batch_size=1). fp32 -> batch 2 (~= fp16+batch-4 peak); fp16 -> batch 4.
    cfg["train"]["batch_size"] = 2 if fp32 else 4
    # Log losses far more often than the fork default (100 steps) so the live
    # monitor's loss curve populates within a few epochs instead of ~epoch 33.
    cfg["train"]["log_interval"] = 10
    config_path.write_text(json.dumps(cfg, indent=2))
