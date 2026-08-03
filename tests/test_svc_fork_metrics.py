"""Tests for the svc-fork live-metrics poller.

The poller exists because so-vits-svc-fork routes loss / lr / GPU stats only to
a TensorBoard events file - nothing reaches stdout - so without it the GUI's
live-monitor card sits on 0/0 forever. These tests drive it against a real
events file written by SummaryWriter rather than a mock, so a change in how
tensorboard serialises scalars is caught here.
"""
import threading
import time
from pathlib import Path

import pytest
from torch.utils.tensorboard import SummaryWriter

from auto_voice.training.svc_fork_metrics import (
    _find_events_file,
    _read_metrics,
    start_fork_metrics_poller,
)

PROFILE_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def _write_events(root: Path, profile_id: str, version: int = 0, **scalars) -> Path:
    """Write a Lightning-shaped events file under the layout the poller globs."""
    log_dir = (
        root / profile_id / "logs" / "44k" / "lightning_logs" / f"version_{version}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    for tag, points in scalars.items():
        # tags are passed as kwargs so '/' is spelled '__'
        real_tag = tag.replace("__", "/")
        for step, value in points:
            writer.add_scalar(real_tag, value, global_step=step)
    writer.flush()
    writer.close()
    return log_dir


# ---------------------------------------------------------------------------
# Locating the events file
# ---------------------------------------------------------------------------

class TestFindEventsFile:
    def test_returns_none_before_lightning_starts(self, tmp_path):
        assert _find_events_file(str(tmp_path), PROFILE_ID) is None

    def test_finds_written_events_file(self, tmp_path):
        _write_events(tmp_path, PROFILE_ID, **{"loss__g__total": [(1, 5.0)]})
        found = _find_events_file(str(tmp_path), PROFILE_ID)
        assert found is not None
        assert "events.out.tfevents" in found

    def test_picks_newest_version_dir(self, tmp_path):
        _write_events(tmp_path, PROFILE_ID, 0, **{"loss__g__total": [(1, 5.0)]})
        time.sleep(1.1)  # mtime resolution
        _write_events(tmp_path, PROFILE_ID, 1, **{"loss__g__total": [(9, 1.0)]})
        found = _find_events_file(str(tmp_path), PROFILE_ID)
        assert "version_1" in found

    def test_other_profile_not_matched(self, tmp_path):
        _write_events(tmp_path, PROFILE_ID, **{"loss__g__total": [(1, 5.0)]})
        assert _find_events_file(str(tmp_path), "some-other-profile") is None

    def test_stale_run_ignored_via_min_mtime(self, tmp_path):
        """A retrain must not surface the previous run's trailing metrics.

        Observed live 2026-08-03: a second job on the same profile reported
        the first run's final 'loss 33.9 @ step 141' for ~90s before its own
        version dir existed.
        """
        _write_events(tmp_path, PROFILE_ID, 0, **{"loss__g__total": [(141, 33.9)]})
        cutoff = time.time() + 1
        assert _find_events_file(str(tmp_path), PROFILE_ID) is not None
        assert _find_events_file(str(tmp_path), PROFILE_ID, min_mtime=cutoff) is None

    def test_new_run_found_after_cutoff(self, tmp_path):
        _write_events(tmp_path, PROFILE_ID, 0, **{"loss__g__total": [(141, 33.9)]})
        cutoff = time.time()
        time.sleep(1.1)
        _write_events(tmp_path, PROFILE_ID, 1, **{"loss__g__total": [(3, 40.0)]})
        found = _find_events_file(str(tmp_path), PROFILE_ID, min_mtime=cutoff)
        assert found is not None and "version_1" in found


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

class TestReadMetrics:
    def test_extracts_loss_and_step(self, tmp_path):
        _write_events(
            tmp_path, PROFILE_ID,
            **{"loss__g__total": [(10, 4.0), (20, 3.25)]},
        )
        m = _read_metrics(_find_events_file(str(tmp_path), PROFILE_ID))
        assert m["current_loss"] == pytest.approx(3.25)
        assert m["current_step"] == 20
        assert m["current_loss_tag"] == "loss/g/total"

    def test_extracts_learning_rate(self, tmp_path):
        _write_events(
            tmp_path, PROFILE_ID,
            **{"loss__g__total": [(1, 4.0)], "lr": [(1, 0.0001)]},
        )
        m = _read_metrics(_find_events_file(str(tmp_path), PROFILE_ID))
        assert m["learning_rate"] == pytest.approx(0.0001)

    def test_extracts_epoch(self, tmp_path):
        _write_events(
            tmp_path, PROFILE_ID,
            **{"loss__g__total": [(1, 4.0)], "epoch": [(1, 7)]},
        )
        m = _read_metrics(_find_events_file(str(tmp_path), PROFILE_ID))
        assert m["current_epoch"] == 7

    def test_falls_back_to_component_loss(self, tmp_path):
        """When the trainer logs only a component, the curve still has data."""
        _write_events(tmp_path, PROFILE_ID, **{"loss__g__mel": [(5, 2.0)]})
        m = _read_metrics(_find_events_file(str(tmp_path), PROFILE_ID))
        assert m["current_loss"] == pytest.approx(2.0)
        assert m["current_loss_tag"] == "loss/g/mel"

    def test_gpu_memory_converted_from_bytes_to_mib(self, tmp_path):
        tag = "DeviceStatsMonitor.on_train_batch_end/active.all.current"
        _write_events(
            tmp_path, PROFILE_ID,
            **{tag.replace("/", "__"): [(1, 512 * 1024 * 1024)]},
        )
        m = _read_metrics(_find_events_file(str(tmp_path), PROFILE_ID))
        assert m["gpu_memory_mb"] == pytest.approx(512.0, abs=1.0)

    def test_small_allocation_not_mislabelled_as_mib(self, tmp_path):
        """A size-dependent heuristic used to report small byte counts as MiB."""
        tag = "DeviceStatsMonitor.on_train_batch_end/active.all.current"
        _write_events(
            tmp_path, PROFILE_ID,
            **{tag.replace("/", "__"): [(1, 500_000)]},
        )
        m = _read_metrics(_find_events_file(str(tmp_path), PROFILE_ID))
        assert m["gpu_memory_mb"] < 1.0

    def test_mos_proxy_derived_from_loss(self, tmp_path):
        _write_events(tmp_path, PROFILE_ID, **{"loss__g__total": [(1, 1.0)]})
        m = _read_metrics(_find_events_file(str(tmp_path), PROFILE_ID))
        assert 1.0 <= m["mos_proxy"] <= 5.0

    def test_empty_dir_yields_no_metrics(self, tmp_path):
        log_dir = _write_events(tmp_path, PROFILE_ID)
        assert _read_metrics(str(log_dir)) == {}

    def test_unreadable_file_returns_empty(self, tmp_path):
        bad = tmp_path / "not-an-events-file"
        bad.write_text("garbage")
        assert _read_metrics(str(bad)) == {}


# ---------------------------------------------------------------------------
# Poller lifecycle
# ---------------------------------------------------------------------------

class TestPollerLifecycle:
    def test_delivers_metrics_then_stops(self, tmp_path):
        received = []
        stop = threading.Event()
        thread = start_fork_metrics_poller(
            profile_id=PROFILE_ID,
            workspace_root=str(tmp_path),
            on_metrics=received.append,
            stop_event=stop,
            interval_seconds=0.05,
        )
        assert thread is not None
        # Written after the poller starts, as in the real lane: the poller is
        # launched before train_svc_fork, so Lightning's file is always newer.
        _write_events(tmp_path, PROFILE_ID, **{"loss__g__total": [(3, 2.0)]})
        for _ in range(100):
            if received:
                break
            time.sleep(0.05)
        stop.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert received and received[0]["current_loss"] == pytest.approx(2.0)

    def test_waits_through_data_prep_for_a_late_events_file(self, tmp_path):
        """Regression: the poller must survive the pre-train data-prep phase.

        svc-fork runs pre-resample / pre-config / pre-hubert before invoking
        `svc train`, so Lightning's events file does not exist for minutes
        after the job starts. A poller that gives up during that window
        delivers nothing for the entire run - observed live on 2026-08-03,
        where a 30s deadline expired ~50s before the file appeared and the
        GUI sat on 0/0 through a successful training run.
        """
        received = []
        stop = threading.Event()
        thread = start_fork_metrics_poller(
            profile_id=PROFILE_ID,
            workspace_root=str(tmp_path),
            on_metrics=received.append,
            stop_event=stop,
            interval_seconds=0.05,
        )
        # Nothing to find yet - stand in for the data-prep phase.
        time.sleep(1.0)
        assert thread.is_alive(), "poller gave up before training started"

        _write_events(tmp_path, PROFILE_ID, **{"loss__g__total": [(7, 1.5)]})
        for _ in range(100):
            if received:
                break
            time.sleep(0.05)
        stop.set()
        thread.join(timeout=5)
        assert received, "poller never picked up the late events file"
        assert received[0]["current_step"] == 7

    def test_default_deadline_covers_data_prep(self):
        """A default measured in seconds cannot cover a multi-minute prep."""
        import inspect

        default = inspect.signature(
            start_fork_metrics_poller
        ).parameters["max_age_seconds"].default
        assert default >= 600, (
            "max_age_seconds must outlast pre-hubert feature extraction"
        )

    def test_exits_when_no_events_file_appears(self, tmp_path):
        """Lightning never started - the poller must give up, not spin forever."""
        stop = threading.Event()
        thread = start_fork_metrics_poller(
            profile_id=PROFILE_ID,
            workspace_root=str(tmp_path),
            on_metrics=lambda m: None,
            stop_event=stop,
            interval_seconds=0.05,
            max_age_seconds=0.2,
        )
        thread.join(timeout=5)
        assert not thread.is_alive()

    def test_callback_failure_does_not_kill_poller(self, tmp_path):
        calls = []

        def _boom(metrics):
            calls.append(metrics)
            raise RuntimeError("subscriber exploded")

        stop = threading.Event()
        thread = start_fork_metrics_poller(
            profile_id=PROFILE_ID,
            workspace_root=str(tmp_path),
            on_metrics=_boom,
            stop_event=stop,
            interval_seconds=0.05,
        )
        _write_events(tmp_path, PROFILE_ID, **{"loss__g__total": [(1, 2.0)]})
        for _ in range(100):
            if calls:
                break
            time.sleep(0.05)
        assert calls, "callback was never invoked"
        assert thread.is_alive(), "poller died on a subscriber exception"
        stop.set()
        thread.join(timeout=5)
