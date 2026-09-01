"""Guards for the training-telemetry fixes.

Three separate defects, all of which presented to a user as a blank or absurd
number on the live training card:

1. ``_GPU_MEM_TAGS`` read ``active.all.current``, which is a COUNT of active
   allocation blocks (~2,800), not a size. Divided by 1MiB it rendered 0.0 for
   every run that ever trained here.
2. svc-fork logs no step total, so the monitor had no denominator and could
   not offer a remaining estimate on a multi-hour run.
3. The Socket.IO heartbeat used the engineio defaults (25s + 20s), and with
   eventlet running un-monkey-patched the hub lost the race often enough that
   browsers reconnected on a ~45s cycle, invalidating the query cache each time.
"""
import json
import math

import pytest

from auto_voice.training.svc_fork_metrics import (
    _GPU_ALLOC_TAGS,
    _GPU_MEM_TAGS,
    _total_steps,
)


class TestGpuTagsMeasureBytes:
    @pytest.mark.parametrize("tag", [*_GPU_MEM_TAGS, *_GPU_ALLOC_TAGS])
    def test_tags_are_byte_valued_not_counts(self, tag):
        """``active.all.current`` and friends are counts; only ``*_bytes.*`` are sizes.

        _read_metrics divides these by 1024**2 unconditionally, so a count tag
        silently becomes 0.0 GB rather than failing.
        """
        assert "_bytes." in tag, (
            f"{tag!r} is a block count, not a size - dividing it by 1MiB renders 0.0"
        )

    def test_reserved_and_allocated_are_kept_distinct(self):
        """The pair is the diagnostic: reserve climbing while allocated is flat
        means the allocator is holding segments, not that the model leaks."""
        assert all("reserved_bytes." in t for t in _GPU_MEM_TAGS)
        assert all("allocated_bytes." in t for t in _GPU_ALLOC_TAGS)
        assert not set(_GPU_MEM_TAGS) & set(_GPU_ALLOC_TAGS)


class TestTotalSteps:
    def _workspace(self, tmp_path, *, clips, batch_size, epochs):
        ws = tmp_path / "ws"
        (ws / "configs" / "44k").mkdir(parents=True)
        (ws / "filelists" / "44k").mkdir(parents=True)
        (ws / "logs" / "44k" / "lightning_logs" / "version_0").mkdir(parents=True)
        (ws / "configs" / "44k" / "config.json").write_text(
            json.dumps({"train": {"epochs": epochs, "batch_size": batch_size}})
        )
        (ws / "filelists" / "44k" / "train.txt").write_text(
            "\n".join(f"clip_{i}.wav" for i in range(clips))
        )
        events = ws / "logs" / "44k" / "lightning_logs" / "version_0" / "events.out.tfevents.1"
        events.write_text("")
        return str(events)

    def test_matches_the_real_run(self, tmp_path):
        """The live run: 906 clips, batch 16, 400 epochs -> 22,800."""
        events = self._workspace(tmp_path, clips=906, batch_size=16, epochs=400)
        assert _total_steps(events) == 400 * math.ceil(906 / 16) == 22800

    def test_partial_final_batch_rounds_up(self, tmp_path):
        """A trailing partial batch is still a step; floor would under-report."""
        events = self._workspace(tmp_path, clips=10, batch_size=4, epochs=2)
        assert _total_steps(events) == 2 * 3

    @pytest.mark.parametrize(
        "clips, batch_size, epochs",
        [(0, 16, 400), (906, 0, 400), (906, 16, 0)],
    )
    def test_degenerate_config_returns_none_not_zero(self, tmp_path, clips, batch_size, epochs):
        """None means "unknown" and hides the estimate; 0 would render an ETA."""
        events = self._workspace(tmp_path, clips=clips, batch_size=batch_size, epochs=epochs)
        assert _total_steps(events) is None

    def test_missing_workspace_is_not_fatal(self, tmp_path):
        assert _total_steps(str(tmp_path / "nope" / "events.out.tfevents.1")) is None

    def test_unreadable_config_is_not_fatal(self, tmp_path):
        events = self._workspace(tmp_path, clips=906, batch_size=16, epochs=400)
        config = tmp_path / "ws" / "configs" / "44k" / "config.json"
        config.write_text("{not json")
        assert _total_steps(events) is None


class TestSocketHeartbeat:
    def test_timeout_exceeds_the_engineio_default(self):
        """45s of silence (25+20) was dropping healthy clients under GIL load."""
        from auto_voice.web.app import SOCKETIO_PING_INTERVAL, SOCKETIO_PING_TIMEOUT

        assert SOCKETIO_PING_TIMEOUT > 20, "must be more tolerant than the engineio default"
        assert SOCKETIO_PING_INTERVAL + SOCKETIO_PING_TIMEOUT > 45, (
            "clients must survive longer than the ~45s stall that was observed"
        )
        # Still bounded, so a genuinely dead client is reaped rather than leaking a room.
        assert SOCKETIO_PING_INTERVAL <= 30
        assert SOCKETIO_PING_TIMEOUT <= 120
