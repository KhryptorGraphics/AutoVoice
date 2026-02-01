"""
Database operations for AutoVoice persistent storage.

CRUD operations for:
- Tracks
- Featured artists
- Speaker embeddings
- Speaker clusters
"""

import sqlite3
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from .schema import get_db_context, DATABASE_PATH

logger = logging.getLogger(__name__)


# ============================================================================
# Track Operations
# ============================================================================

def upsert_track(
    track_id: str,
    title: Optional[str] = None,
    channel: Optional[str] = None,
    upload_date: Optional[str] = None,
    duration_sec: Optional[float] = None,
    artist_name: Optional[str] = None,
    vocals_path: Optional[str] = None,
    diarization_path: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> None:
    """Insert or update a track record.

    Args:
        track_id: YouTube video ID
        title: Video title
        channel: YouTube channel name
        upload_date: Upload date string
        duration_sec: Track duration in seconds
        artist_name: Main artist name
        vocals_path: Path to extracted vocals file
        diarization_path: Path to diarization JSON
        db_path: Database path
    """
    with get_db_context(db_path) as conn:
        conn.execute("""
            INSERT INTO tracks (id, title, channel, upload_date, duration_sec,
                               artist_name, vocals_path, diarization_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = COALESCE(excluded.title, title),
                channel = COALESCE(excluded.channel, channel),
                upload_date = COALESCE(excluded.upload_date, upload_date),
                duration_sec = COALESCE(excluded.duration_sec, duration_sec),
                artist_name = COALESCE(excluded.artist_name, artist_name),
                vocals_path = COALESCE(excluded.vocals_path, vocals_path),
                diarization_path = COALESCE(excluded.diarization_path, diarization_path),
                fetched_at = CURRENT_TIMESTAMP
        """, (track_id, title, channel, upload_date, duration_sec,
              artist_name, vocals_path, diarization_path))


def get_track(track_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Get a track by ID.

    Args:
        track_id: YouTube video ID
        db_path: Database path

    Returns:
        Track dict or None if not found
    """
    with get_db_context(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM tracks WHERE id = ?", (track_id,)
        ).fetchone()
        return dict(row) if row else None


def get_all_tracks(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Get all tracks.

    Args:
        db_path: Database path

    Returns:
        List of track dicts
    """
    with get_db_context(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM tracks ORDER BY artist_name, title"
        ).fetchall()
        return [dict(row) for row in rows]


def get_tracks_by_artist(artist_name: str, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Get all tracks for an artist.

    Args:
        artist_name: Artist name
        db_path: Database path

    Returns:
        List of track dicts
    """
    with get_db_context(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM tracks WHERE artist_name = ? ORDER BY title",
            (artist_name,)
        ).fetchall()
        return [dict(row) for row in rows]


# ============================================================================
# Featured Artist Operations
# ============================================================================

def add_featured_artist(
    track_id: str,
    name: str,
    pattern_matched: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Add a featured artist for a track.

    Args:
        track_id: YouTube video ID
        name: Featured artist name
        pattern_matched: Pattern used to match (ft., feat., vs., etc.)
        db_path: Database path

    Returns:
        Row ID of inserted/existing record
    """
    with get_db_context(db_path) as conn:
        cursor = conn.execute("""
            INSERT INTO featured_artists (track_id, name, pattern_matched)
            VALUES (?, ?, ?)
            ON CONFLICT(track_id, name) DO UPDATE SET
                pattern_matched = COALESCE(excluded.pattern_matched, pattern_matched)
            RETURNING id
        """, (track_id, name, pattern_matched))
        return cursor.fetchone()[0]


def get_featured_artists_for_track(
    track_id: str,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Get all featured artists for a track.

    Args:
        track_id: YouTube video ID
        db_path: Database path

    Returns:
        List of featured artist dicts
    """
    with get_db_context(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM featured_artists WHERE track_id = ? ORDER BY name",
            (track_id,)
        ).fetchall()
        return [dict(row) for row in rows]


def get_all_featured_artists(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Get all featured artists with track counts.

    Args:
        db_path: Database path

    Returns:
        List of dicts with name, track_count, tracks
    """
    with get_db_context(db_path) as conn:
        rows = conn.execute("""
            SELECT
                name,
                COUNT(DISTINCT track_id) as track_count,
                GROUP_CONCAT(track_id) as track_ids
            FROM featured_artists
            GROUP BY name
            ORDER BY track_count DESC
        """).fetchall()
        return [dict(row) for row in rows]


# ============================================================================
# Speaker Embedding Operations
# ============================================================================

def add_speaker_embedding(
    track_id: str,
    speaker_id: str,
    embedding: np.ndarray,
    duration_sec: Optional[float] = None,
    is_primary: bool = False,
    profile_id: Optional[str] = None,
    isolated_vocals_path: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Add a speaker embedding for a track.

    Args:
        track_id: YouTube video ID
        speaker_id: Speaker ID (SPEAKER_00, etc.)
        embedding: 512-dim numpy array
        duration_sec: Total speaking duration
        is_primary: Whether this is the primary speaker
        profile_id: Associated voice profile UUID
        isolated_vocals_path: Path to isolated vocals
        db_path: Database path

    Returns:
        Row ID of inserted/updated record
    """
    # Convert numpy array to bytes for storage
    embedding_bytes = embedding.astype(np.float32).tobytes()

    with get_db_context(db_path) as conn:
        cursor = conn.execute("""
            INSERT INTO speaker_embeddings
                (track_id, speaker_id, embedding, duration_sec, is_primary,
                 profile_id, isolated_vocals_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(track_id, speaker_id) DO UPDATE SET
                embedding = excluded.embedding,
                duration_sec = COALESCE(excluded.duration_sec, duration_sec),
                is_primary = excluded.is_primary,
                profile_id = COALESCE(excluded.profile_id, profile_id),
                isolated_vocals_path = COALESCE(excluded.isolated_vocals_path, isolated_vocals_path)
            RETURNING id
        """, (track_id, speaker_id, embedding_bytes, duration_sec, is_primary,
              profile_id, isolated_vocals_path))
        return cursor.fetchone()[0]


def get_embeddings_for_track(
    track_id: str,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Get all speaker embeddings for a track.

    Args:
        track_id: YouTube video ID
        db_path: Database path

    Returns:
        List of embedding dicts with numpy arrays
    """
    with get_db_context(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM speaker_embeddings WHERE track_id = ? ORDER BY speaker_id",
            (track_id,)
        ).fetchall()

        results = []
        for row in rows:
            d = dict(row)
            # Convert bytes back to numpy array
            d['embedding'] = np.frombuffer(d['embedding'], dtype=np.float32)
            results.append(d)
        return results


def get_all_embeddings(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Get all speaker embeddings.

    Args:
        db_path: Database path

    Returns:
        List of embedding dicts with numpy arrays
    """
    with get_db_context(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM speaker_embeddings ORDER BY track_id, speaker_id"
        ).fetchall()

        results = []
        for row in rows:
            d = dict(row)
            d['embedding'] = np.frombuffer(d['embedding'], dtype=np.float32)
            results.append(d)
        return results


def get_embeddings_by_cluster(
    cluster_id: str,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Get all embeddings belonging to a cluster.

    Args:
        cluster_id: Cluster UUID
        db_path: Database path

    Returns:
        List of embedding dicts with confidence scores
    """
    with get_db_context(db_path) as conn:
        rows = conn.execute("""
            SELECT se.*, cm.confidence
            FROM speaker_embeddings se
            JOIN cluster_members cm ON se.id = cm.embedding_id
            WHERE cm.cluster_id = ?
            ORDER BY cm.confidence DESC
        """, (cluster_id,)).fetchall()

        results = []
        for row in rows:
            d = dict(row)
            d['embedding'] = np.frombuffer(d['embedding'], dtype=np.float32)
            results.append(d)
        return results


# ============================================================================
# Speaker Cluster Operations
# ============================================================================

def create_cluster(
    name: str,
    is_verified: bool = False,
    voice_profile_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> str:
    """Create a new speaker cluster.

    Args:
        name: Cluster name (e.g., "Anth", "Unknown 1")
        is_verified: Whether identity is user-confirmed
        voice_profile_id: Associated voice profile UUID
        db_path: Database path

    Returns:
        Cluster UUID
    """
    cluster_id = str(uuid.uuid4())

    with get_db_context(db_path) as conn:
        conn.execute("""
            INSERT INTO speaker_clusters (id, name, is_verified, voice_profile_id)
            VALUES (?, ?, ?, ?)
        """, (cluster_id, name, is_verified, voice_profile_id))

    return cluster_id


def get_cluster(cluster_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Get a cluster by ID.

    Args:
        cluster_id: Cluster UUID
        db_path: Database path

    Returns:
        Cluster dict or None
    """
    with get_db_context(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM speaker_clusters WHERE id = ?", (cluster_id,)
        ).fetchone()
        return dict(row) if row else None


def get_all_clusters(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Get all clusters with member counts.

    Args:
        db_path: Database path

    Returns:
        List of cluster dicts with member_count
    """
    with get_db_context(db_path) as conn:
        rows = conn.execute("""
            SELECT sc.*,
                   COUNT(cm.embedding_id) as member_count,
                   SUM(se.duration_sec) as total_duration_sec
            FROM speaker_clusters sc
            LEFT JOIN cluster_members cm ON sc.id = cm.cluster_id
            LEFT JOIN speaker_embeddings se ON cm.embedding_id = se.id
            GROUP BY sc.id
            ORDER BY sc.name
        """).fetchall()
        return [dict(row) for row in rows]


def update_cluster_name(
    cluster_id: str,
    name: str,
    is_verified: bool = True,
    db_path: Optional[Path] = None,
) -> None:
    """Update a cluster's name.

    Args:
        cluster_id: Cluster UUID
        name: New name
        is_verified: Whether identity is confirmed
        db_path: Database path
    """
    with get_db_context(db_path) as conn:
        conn.execute("""
            UPDATE speaker_clusters
            SET name = ?, is_verified = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (name, is_verified, cluster_id))


def merge_clusters(
    target_cluster_id: str,
    source_cluster_id: str,
    db_path: Optional[Path] = None,
) -> None:
    """Merge source cluster into target cluster.

    All members from source are moved to target, source is deleted.

    Args:
        target_cluster_id: Cluster to keep
        source_cluster_id: Cluster to merge and delete
        db_path: Database path
    """
    with get_db_context(db_path) as conn:
        # Move all members
        conn.execute("""
            UPDATE cluster_members
            SET cluster_id = ?
            WHERE cluster_id = ?
        """, (target_cluster_id, source_cluster_id))

        # Delete source cluster
        conn.execute(
            "DELETE FROM speaker_clusters WHERE id = ?",
            (source_cluster_id,)
        )

        # Update target timestamp
        conn.execute("""
            UPDATE speaker_clusters
            SET updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (target_cluster_id,))


def add_to_cluster(
    cluster_id: str,
    embedding_id: int,
    confidence: Optional[float] = None,
    db_path: Optional[Path] = None,
) -> None:
    """Add an embedding to a cluster.

    Args:
        cluster_id: Cluster UUID
        embedding_id: Speaker embedding row ID
        confidence: Cosine similarity score
        db_path: Database path
    """
    with get_db_context(db_path) as conn:
        conn.execute("""
            INSERT INTO cluster_members (cluster_id, embedding_id, confidence)
            VALUES (?, ?, ?)
            ON CONFLICT(cluster_id, embedding_id) DO UPDATE SET
                confidence = excluded.confidence
        """, (cluster_id, embedding_id, confidence))


def remove_from_cluster(
    cluster_id: str,
    embedding_id: int,
    db_path: Optional[Path] = None,
) -> None:
    """Remove an embedding from a cluster.

    Args:
        cluster_id: Cluster UUID
        embedding_id: Speaker embedding row ID
        db_path: Database path
    """
    with get_db_context(db_path) as conn:
        conn.execute("""
            DELETE FROM cluster_members
            WHERE cluster_id = ? AND embedding_id = ?
        """, (cluster_id, embedding_id))


def get_cluster_members(
    cluster_id: str,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Get all members of a cluster with track info.

    Args:
        cluster_id: Cluster UUID
        db_path: Database path

    Returns:
        List of member dicts with track and embedding info
    """
    with get_db_context(db_path) as conn:
        rows = conn.execute("""
            SELECT
                cm.confidence,
                se.id as embedding_id,
                se.track_id,
                se.speaker_id,
                se.duration_sec,
                se.is_primary,
                se.isolated_vocals_path,
                t.title as track_title,
                t.artist_name
            FROM cluster_members cm
            JOIN speaker_embeddings se ON cm.embedding_id = se.id
            JOIN tracks t ON se.track_id = t.id
            WHERE cm.cluster_id = ?
            ORDER BY cm.confidence DESC
        """, (cluster_id,)).fetchall()
        return [dict(row) for row in rows]


# ============================================================================
# Utility Functions
# ============================================================================

def find_unclustered_embeddings(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Find all embeddings not assigned to any cluster.

    Args:
        db_path: Database path

    Returns:
        List of embedding dicts
    """
    with get_db_context(db_path) as conn:
        rows = conn.execute("""
            SELECT se.*, t.title as track_title, t.artist_name
            FROM speaker_embeddings se
            JOIN tracks t ON se.track_id = t.id
            LEFT JOIN cluster_members cm ON se.id = cm.embedding_id
            WHERE cm.cluster_id IS NULL
            ORDER BY se.track_id, se.speaker_id
        """).fetchall()

        results = []
        for row in rows:
            d = dict(row)
            d['embedding'] = np.frombuffer(d['embedding'], dtype=np.float32)
            results.append(d)
        return results


def get_embedding_by_id(
    embedding_id: int,
    db_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Get a single embedding by ID.

    Args:
        embedding_id: Embedding row ID
        db_path: Database path

    Returns:
        Embedding dict or None
    """
    with get_db_context(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM speaker_embeddings WHERE id = ?",
            (embedding_id,)
        ).fetchone()

        if row:
            d = dict(row)
            d['embedding'] = np.frombuffer(d['embedding'], dtype=np.float32)
            return d
        return None
