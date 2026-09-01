"""Live-metrics poller for svc-fork training subprocesses.

The upstream so-vits-svc-fork trainer routes loss / lr / GPU stats only
to a TensorBoard events file written by its Lightning logger; nothing about
the in-flight metrics reaches stdout. The AutoVoice GUI's live-monitor
view expects top-level ``current_loss`` / ``current_step`` /
``current_epoch`` / ``learning_rate`` fields on the training job so it
can render the loss curve and step counter while training runs.

This module tails the Lightning events file from a background thread,
parses the latest values for the metrics we care about, and hands them
back via a callback. The caller (``TrainingJobManager._run_fork_training``)
writes them into ``job.results`` which is then surfaced at the top level
by ``TrainingJob.to_dict``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import glob
import os
import threading
import time
from typing import Any, Callable, Dict, Optional

_LOSS_TAGS = (
    "loss/g/total",
    "loss/g/fm",
    "loss/g/mel",
    "loss/g/kl",
)
# Byte-valued tags only. These were `active.all.current`, which is the COUNT of
# active allocation blocks (~2,800 on a real run), not a size - and _read_metrics
# divides by 1 MiB unconditionally, so it rendered 0.0 GB on the live training
# card for every run that has ever trained here. A count tag fails silently
# rather than loudly, which is why it survived.
#
# Reserved and allocated are kept apart because the PAIR is the diagnostic:
# reserve climbing while allocated stays flat means the allocator is holding
# segments it will not return, not that the model is leaking. That is exactly
# the failure this deployment hit - reserve 4 GB -> 91 GB with live tensors flat
# at 2,401.8 MiB - and this telemetry is what would have shown it.
_GPU_MEM_TAGS = (
    "DeviceStatsMonitor.on_train_batch_end/reserved_bytes.all.current",
    "DeviceStatsMonitor.on_train_batch_start/reserved_bytes.all.current",
)
_GPU_ALLOC_TAGS = (
    "DeviceStatsMonitor.on_train_batch_end/allocated_bytes.all.current",
    "DeviceStatsMonitor.on_train_batch_start/allocated_bytes.all.current",
)


def _total_steps(events_path: str) -> Optional[int]:
    """Total optimizer steps a run will take, or None if it cannot be known.

    svc-fork logs a step COUNTER but never a total, so a multi-hour run had no
    denominator and the UI could not offer a remaining-time estimate. The total
    is derivable from the workspace the events file sits in:

        epochs * ceil(clips / batch_size)

    Rounds UP: a trailing partial batch is still a step, and flooring
    under-reports the total, which makes a run appear to overshoot 100%.

    Returns None - never 0 - on anything unknown or degenerate. The caller
    hides the estimate on None, whereas 0 would sail through as a real total
    and render a nonsense ETA. A missing number is better than a wrong one.
    """
    try:
        # <ws>/logs/44k/lightning_logs/version_0/events.out.tfevents.N
        ws = Path(events_path).resolve().parents[4]
        config_path = ws / "configs" / "44k" / "config.json"
        filelist = ws / "filelists" / "44k" / "train.txt"
        if not config_path.is_file() or not filelist.is_file():
            return None
        cfg = json.loads(config_path.read_text())
        train_cfg = cfg.get("train") or {}
        epochs = int(train_cfg.get("epochs") or 0)
        batch_size = int(train_cfg.get("batch_size") or 0)
        clips = sum(1 for line in filelist.read_text().splitlines() if line.strip())
        if epochs <= 0 or batch_size <= 0 or clips <= 0:
            return None
        return epochs * math.ceil(clips / batch_size)
    except (OSError, ValueError, TypeError, IndexError, KeyError):
        # Telemetry must never take down the thing it is reporting on.
        return None
# Slack on the "ignore files older than this run" cutoff, to absorb
# filesystem mtime granularity and thread-scheduling jitter. A previous
# training run is minutes-to-days older, so this never readmits one.
_STALE_GRACE_S = 5.0
_GPU_UTIL_TAGS = (
    "DeviceStatsMonitor.on_train_batch_end/utilization.gpu.0",
    "DeviceStatsMonitor.on_train_batch_start/utilization.gpu.0",
)


def _find_events_file(
    workspace_root: str,
    profile_id: str,
    min_mtime: float = 0.0,
) -> Optional[str]:
    """Locate the active Lightning events file for a profile run.

    The filename includes the host + PID + version suffix; we glob and
    return the newest write. Returns ``None`` while Lightning is still
    starting up.

    ``min_mtime`` discards events files older than the current run. A
    retrained profile keeps its previous ``version_*`` directories, so
    without this the poller reports the *last* run's final loss/step as if
    they were live until the new version dir is created - which on a real
    run meant the GUI showed "loss 33.9 @ step 141" for the first ~90s of a
    job that had not taken a step yet.
    """
    pattern = os.path.join(
        workspace_root,
        profile_id,
        "logs",
        "44k",
        "lightning_logs",
        "version_*",
        "events.out.tfevents.*",
    )
    candidates = [
        path for path in glob.glob(pattern)
        if _mtime_or_zero(path) >= min_mtime
    ]
    if not candidates:
        return None
    return max(candidates, key=_mtime_or_zero)


def _mtime_or_zero(path: str) -> float:
    """mtime, tolerating a file that vanished between glob and stat."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _read_metrics(events_file: str) -> Dict[str, Any]:
    """Parse the events file with TensorBoard's EventAccumulator.

    Lazy-imported because tensorboard is a heavyweight dependency and this
    poller runs in a daemon thread for hours at a time.
    """
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )
    try:
        ea = EventAccumulator(
            events_file,
            size_guidance={"scalars": 4096},
        )
        ea.Reload()
    except Exception:
        return {}

    scalars = ea.Tags().get("scalars", [])
    out: Dict[str, Any] = {}

    # Loss: prefer the combined "loss/g/total" tag, fall back to the most
    # recent of the per-component tags so the GUI's loss curve still has
    # something to plot when the trainer logs only one component.
    loss = None
    loss_step = None
    loss_tag = None
    for tag in _LOSS_TAGS:
        if tag in scalars:
            events = ea.Scalars(tag)
            if events:
                last = events[-1]
                if loss is None or last.step > loss_step:
                    loss = float(last.value)
                    loss_step = int(last.step)
                    loss_tag = tag
    if loss is not None:
        out["current_loss"] = loss
        out["current_step"] = loss_step
        out["current_loss_tag"] = loss_tag

    # Epoch: tensorboard doesn't log a dedicated "epoch" scalar for svc-fork
    # but the Lightning log_dict_ uses total_batch_idx as the global step.
    # step / (steps_per_epoch) approximates the epoch. The trainer also
    # writes step counts via total_batch_idx; if the trainer exposes an
    # explicit "epoch" scalar we surface it instead.
    if "epoch" in scalars:
        events = ea.Scalars("epoch")
        if events:
            out["current_epoch"] = int(events[-1].value)

    if "lr" in scalars:
        events = ea.Scalars("lr")
        if events:
            out["learning_rate"] = float(events[-1].value)

    # GPU memory. These tags are byte-valued (see _GPU_MEM_TAGS), so convert
    # unconditionally - a size-dependent heuristic would report a genuinely
    # small allocation as if it were already in MiB.
    for tag in _GPU_MEM_TAGS:
        if tag in scalars:
            events = ea.Scalars(tag)
            if events:
                val = float(events[-1].value) / (1024 * 1024)
                out["gpu_memory_mb"] = round(val, 1)
                break

    # Reported alongside, never merged: the gap between reserved and allocated
    # is what distinguishes an allocator holding segments from a model leaking.
    for tag in _GPU_ALLOC_TAGS:
        if tag in scalars:
            events = ea.Scalars(tag)
            if events:
                val = float(events[-1].value) / (1024 * 1024)
                out["gpu_allocated_mb"] = round(val, 1)
                break
    if "gpu_memory_mb" in out and "gpu_allocated_mb" in out:
        held = out["gpu_memory_mb"] - out["gpu_allocated_mb"]
        out["gpu_reserved_overhead_mb"] = round(max(held, 0.0), 1)

    for tag in _GPU_UTIL_TAGS:
        if tag in scalars:
            events = ea.Scalars(tag)
            if events:
                out["gpu_util_pct"] = round(float(events[-1].value), 1)
                break

    # Health: derive an MOS-proxy from the loss. Lower is better; map the
    # typical svc-fork loss range (1.0 -> 5.0) onto a 1.0 -> 5.0 MOS scale.
    if "current_loss" in out:
        loss = out["current_loss"]
        mos = max(1.0, min(5.0, 5.0 - (loss - 1.0)))
        out["mos_proxy"] = round(mos, 2)

    return out


def start_fork_metrics_poller(
    profile_id: str,
    workspace_root: str,
    on_metrics: Callable[[Dict[str, Any]], None],
    stop_event: threading.Event,
    interval_seconds: float = 2.0,
    max_age_seconds: float = 3600.0,
) -> Optional[threading.Thread]:
    """Start a daemon thread that polls the tb events file every
    ``interval_seconds`` and forwards new metrics through ``on_metrics``.

    The poller exits when ``stop_event`` is set, or when no events file has
    appeared for ``max_age_seconds``. Returns the thread so the caller can
    join cleanly; returns ``None`` if the thread could not be started.

    ``max_age_seconds`` only bounds the *initial* wait for Lightning to show
    up - once an events file exists the deadline keeps resetting. It must
    therefore cover the whole svc-fork data-prep phase, which runs before
    ``svc train`` is invoked: pre-resample, pre-config and pre-hubert
    (ContentVec + F0 over every clip) routinely take several minutes and can
    exceed ten on a large dataset. A short deadline here makes the poller
    give up before training ever starts and the GUI sits on 0/0 for the whole
    run - the exact symptom this module exists to fix. The authoritative stop
    is ``stop_event``, which ``_run_fork_training`` always sets in a
    ``finally``, so a generous deadline costs nothing.
    """
    # Captured here, not inside the thread: the cutoff must be "when we
    # decided to start watching", otherwise a thread that is scheduled late
    # can adopt a cutoff newer than the events file it is meant to read and
    # then ignore it forever. _STALE_GRACE_S absorbs filesystem timestamp
    # granularity - it is orders of magnitude smaller than the gap to a
    # previous training run, which is what min_mtime is really excluding.
    started_at = time.time() - _STALE_GRACE_S
    try:
        thread = threading.Thread(
            target=_run_fork_metrics_poller,
            args=(
                profile_id,
                workspace_root,
                on_metrics,
                stop_event,
                interval_seconds,
                max_age_seconds,
                started_at,
            ),
            daemon=True,
            name=f"svc-fork-metrics-{profile_id[:8]}",
        )
        thread.start()
        return thread
    except Exception:
        return None


def _run_fork_metrics_poller(
    profile_id: str,
    workspace_root: str,
    on_metrics: Callable[[Dict[str, Any]], None],
    stop_event: threading.Event,
    interval_seconds: float,
    max_age_seconds: float,
    started_at: float,
) -> None:
    """Poller loop. Tracks the last seen file mtime to skip re-parsing
    when nothing changed; still re-reads after a fresh events file is
    rotated so we don't lose the initial 0-step metrics.

    Only events files written at or after the poller started are considered,
    so a retrain never reports the previous run's trailing metrics as live.
    """
    last_mtime = 0.0
    last_metrics: Dict[str, Any] = {}
    last_progress_at = started_at
    events_file: Optional[str] = None

    while not stop_event.is_set():
        events_file = _find_events_file(
            workspace_root, profile_id, min_mtime=started_at
        )
        if events_file is None:
            if time.time() - last_progress_at > max_age_seconds:
                return
            stop_event.wait(interval_seconds)
            continue

        try:
            mtime = os.path.getmtime(events_file)
        except OSError:
            stop_event.wait(interval_seconds)
            continue

        last_progress_at = time.time()
        if mtime == last_mtime and last_metrics:
            stop_event.wait(interval_seconds)
            continue

        last_mtime = mtime
        metrics = _read_metrics(events_file)
        if metrics and metrics != last_metrics:
            last_metrics = metrics
            try:
                on_metrics(metrics)
            except Exception:
                # Never let a callback failure kill the poller.
                pass
        stop_event.wait(interval_seconds)
