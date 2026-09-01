"""Regression guard for the mixed-timestamp sort hazard.

The state store persists ``created_at`` as epoch floats today, and the read
paths sort on it. Before ``timestamp_sort_key`` those sorts compared raw
values, so the first ISO string written into any of those files would raise

    TypeError: '<' not supported between instances of 'str' and 'float'

and take the endpoint down permanently for every user - ``/convert/history``
returns 500 until someone migrates the file by hand. The plain missing-key
case (``"" vs None``) was already latent for the same reason.

These tests pin the tolerance so a future "just normalise the writer" change
cannot reintroduce it. See src/auto_voice/web/utils.py::timestamp_sort_key.
"""
import json

import pytest

from auto_voice.web.persistence import AppStateStore
from auto_voice.web.utils import timestamp_sort_key


# One of each shape the files can legitimately hold, plus two that should not
# exist but must not be fatal if they do.
MIXED_RECORDS = [
    {"id": "float-old", "created_at": 1783184417.6167185},
    {"id": "iso-z", "created_at": "2026-08-27T10:00:00Z"},
    {"id": "iso-offset", "created_at": "2026-08-26T10:00:00+00:00"},
    {"id": "iso-naive", "created_at": "2026-08-25T10:00:00"},
    {"id": "int-epoch", "created_at": 1783184000},
    {"id": "missing"},
    {"id": "garbage", "created_at": "not-a-timestamp"},
    {"id": "null", "created_at": None},
]


class TestTimestampSortKey:
    def test_mixed_types_do_not_raise(self):
        """The exact comparison that used to be a permanent 500."""
        ordered = sorted(
            MIXED_RECORDS,
            key=lambda item: timestamp_sort_key(item.get("created_at")),
            reverse=True,
        )
        assert len(ordered) == len(MIXED_RECORDS)

    def test_the_old_naive_key_really_would_have_raised(self):
        """Guard against the guard: prove this test exercises a real hazard.

        If this ever stops raising, the fixture no longer reproduces the bug
        and the test above has quietly become vacuous.
        """
        with pytest.raises(TypeError):
            sorted(
                MIXED_RECORDS,
                key=lambda item: item.get("created_at", ""),
                reverse=True,
            )

    def test_unparseable_values_sort_last_under_reverse(self):
        ordered = [
            item["id"]
            for item in sorted(
                MIXED_RECORDS,
                key=lambda item: timestamp_sort_key(item.get("created_at")),
                reverse=True,
            )
        ]
        for sentinel in ("missing", "garbage", "null"):
            assert ordered.index(sentinel) > ordered.index("float-old"), (
                f"{sentinel!r} should sort after real timestamps, got {ordered}"
            )

    def test_epoch_and_iso_are_ordered_against_each_other(self):
        """A float and an ISO string for the same instant must compare equal-ish."""
        epoch = 1783184417.6167185
        iso = "2026-07-04T17:00:17.616719Z"
        assert abs(timestamp_sort_key(epoch) - timestamp_sort_key(iso)) < 1.0


class TestStateStoreReadPaths:
    """The four call sites, through the real store rather than the helper."""

    @pytest.fixture
    def store(self, tmp_path):
        return AppStateStore(str(tmp_path))

    @pytest.mark.parametrize(
        "key, reader",
        [
            ("conversion_history", "list_conversion_history"),
            ("training_jobs", "list_training_jobs"),
            ("background_jobs", "list_background_jobs"),
            ("presets", "list_presets"),
        ],
    )
    def test_reader_survives_a_mixed_file(self, store, key, reader):
        # Write through the store's own path mapping rather than guessing the
        # filename: these do not match their keys (``training_jobs`` lives in
        # web_training_jobs.json) and they sit under data_dir/app_state/.
        payload = {record["id"]: dict(record) for record in MIXED_RECORDS}
        store._files[key].write_text(json.dumps(payload))

        rows = getattr(store, reader)()

        assert len(rows) == len(MIXED_RECORDS), (
            f"{reader} dropped rows from a mixed-timestamp file"
        )
