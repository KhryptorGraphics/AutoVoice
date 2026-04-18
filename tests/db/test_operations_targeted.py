"""Targeted branch coverage for db.operations."""

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from auto_voice.db import schema
from auto_voice.db.operations import (
    _get_upsert,
    add_speaker_embedding,
    add_to_cluster,
    create_cluster,
    get_cluster_members,
    get_embeddings_for_track,
    get_track,
    upsert_track,
)
from auto_voice.db.schema import init_database


@pytest.fixture
def temp_db():
    """Create an isolated SQLite database file for operations coverage tests."""
    old_db_type = os.environ.get("AUTOVOICE_DB_TYPE")
    old_db_path = schema.DATABASE_PATH
    os.environ["AUTOVOICE_DB_TYPE"] = "sqlite"

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "targeted.db"
        schema.DATABASE_PATH = db_path
        schema.close_database()
        schema._engine = None
        schema._SessionFactory = None
        init_database(db_type="sqlite")
        yield db_path
        schema.close_database()
        schema.DATABASE_PATH = old_db_path
        schema._engine = None
        schema._SessionFactory = None

    if old_db_type is not None:
        os.environ["AUTOVOICE_DB_TYPE"] = old_db_type
    else:
        os.environ.pop("AUTOVOICE_DB_TYPE", None)


def test_get_upsert_selects_backend_insert_function():
    """Choose MySQL or SQLite upsert helper based on the engine dialect."""
    mysql_engine = SimpleNamespace(dialect=SimpleNamespace(name="mysql"))
    sqlite_engine = SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))

    assert _get_upsert(mysql_engine).__name__ == "insert"
    assert _get_upsert(sqlite_engine).__name__ == "insert"
    assert _get_upsert(mysql_engine).__module__.endswith(".mysql.dml")
    assert _get_upsert(sqlite_engine).__module__.endswith(".sqlite.dml")


def test_upsert_track_updates_optional_fields(temp_db):
    """Existing tracks preserve untouched fields and update optional metadata."""
    upsert_track("yt123", title="Original", artist_name="Artist", db_path=temp_db)

    upsert_track(
        "yt123",
        channel="Updated Channel",
        upload_date="2026-04-18",
        vocals_path="/tmp/vocals.wav",
        diarization_path="/tmp/diarization.json",
        db_path=temp_db,
    )

    track = get_track("yt123", db_path=temp_db)
    assert track["title"] == "Original"
    assert track["channel"] == "Updated Channel"
    assert track["upload_date"] == "2026-04-18"
    assert track["vocals_path"] == "/tmp/vocals.wav"
    assert track["diarization_path"] == "/tmp/diarization.json"


def test_embedding_and_cluster_updates_refresh_optional_fields(temp_db):
    """Update paths/profile IDs for embeddings and confidence for cluster members."""
    upsert_track("yt123", title="Track", artist_name="Artist", db_path=temp_db)
    first = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    second = np.array([0.4, 0.5, 0.6], dtype=np.float32)

    emb_id = add_speaker_embedding(
        "yt123",
        "SPEAKER_00",
        first,
        duration_sec=10.0,
        is_primary=False,
        db_path=temp_db,
    )
    updated_id = add_speaker_embedding(
        "yt123",
        "SPEAKER_00",
        second,
        duration_sec=22.0,
        is_primary=True,
        profile_id="profile-123",
        isolated_vocals_path="/tmp/isolated.wav",
        db_path=temp_db,
    )

    assert updated_id == emb_id
    [embedding] = get_embeddings_for_track("yt123", db_path=temp_db)
    assert embedding["duration_sec"] == 22.0
    assert embedding["is_primary"] is True
    assert embedding["profile_id"] == "profile-123"
    assert embedding["isolated_vocals_path"] == "/tmp/isolated.wav"
    assert np.allclose(embedding["embedding"], second)

    cluster_id = create_cluster("Singer", db_path=temp_db)
    add_to_cluster(cluster_id, emb_id, confidence=0.15, db_path=temp_db)
    add_to_cluster(cluster_id, emb_id, confidence=0.98, db_path=temp_db)

    [member] = get_cluster_members(cluster_id, db_path=temp_db)
    assert member["confidence"] == 0.98
