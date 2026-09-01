"""Backup exports must not grow without bound.

Each bundle is a full copy of profiles, samples and trained models - ~3.6 GB
on this deployment - and nothing pruned them. 11 bundles had accumulated to
30 GB, three of them byte-identical duplicates from repeated clicks. That is
not just untidy: a backup directory that grows forever eventually fills the
disk the service runs on.
"""
from pathlib import Path

import pytest

from auto_voice.web.api_backup import _prune_old_backups


def _bundle(d: Path, name: str, mtime: float, size: int = 16) -> Path:
    p = d / name
    p.write_bytes(b"x" * size)
    import os
    os.utime(p, (mtime, mtime))
    return p


@pytest.fixture
def backups(tmp_path):
    return [
        _bundle(tmp_path, f"autovoice-backup-2026043{i}-000000.zip", 1000.0 + i)
        for i in range(6)
    ]


def test_keeps_only_the_newest(tmp_path, backups):
    newest = backups[-1]
    removed = _prune_old_backups(tmp_path, keep=3, protect=newest)
    left = sorted(p.name for p in tmp_path.glob("*.zip"))
    assert len(left) == 3, left
    assert newest.name in left
    assert len(removed) == 3


def test_the_bundle_just_written_is_never_deleted(tmp_path, backups):
    """Even when it is the OLDEST file, the caller just created it."""
    oldest = backups[0]
    _prune_old_backups(tmp_path, keep=1, protect=oldest)
    assert oldest.exists(), "pruning deleted the bundle it was told to protect"


def test_retention_disabled_deletes_nothing(tmp_path, backups):
    """0 means 'disabled', not 'delete everything' - the destructive reading of
    a misconfigured value is not one to guess at."""
    for keep in (0, -1):
        removed = _prune_old_backups(tmp_path, keep=keep, protect=backups[-1])
        assert removed == []
    assert len(list(tmp_path.glob("*.zip"))) == 6


def test_reports_what_it_deleted(tmp_path, backups):
    removed = _prune_old_backups(tmp_path, keep=2, protect=backups[-1])
    assert all('path' in r and 'bytes' in r for r in removed)
    assert sum(r['bytes'] for r in removed) > 0


def test_unrelated_files_are_left_alone(tmp_path, backups):
    keeper = tmp_path / "important-notes.txt"
    keeper.write_text("not a backup")
    _prune_old_backups(tmp_path, keep=1, protect=backups[-1])
    assert keeper.exists()
