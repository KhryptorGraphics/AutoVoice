"""
Database schema for AutoVoice persistent storage.

Tables:
- tracks: YouTube track metadata
- featured_artists: Artists parsed from video titles
- speaker_embeddings: WavLM embeddings for speaker matching
- speaker_clusters: Global speaker identities across tracks
- cluster_members: Mapping of embeddings to clusters
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Database location
DATABASE_PATH = Path('data/autovoice.db')

# Schema SQL
SCHEMA_SQL = """
-- Track metadata from YouTube
CREATE TABLE IF NOT EXISTS tracks (
    id TEXT PRIMARY KEY,           -- YouTube video ID
    title TEXT,
    channel TEXT,
    upload_date TEXT,
    duration_sec REAL,
    artist_name TEXT,              -- Main artist (conor_maynard, william_singe)
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    vocals_path TEXT,              -- Path to extracted vocals
    diarization_path TEXT          -- Path to diarization JSON
);

-- Featured artists parsed from video titles
CREATE TABLE IF NOT EXISTS featured_artists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id TEXT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    name TEXT NOT NULL,            -- Parsed name (e.g., "Anth")
    pattern_matched TEXT,          -- Pattern used (e.g., "ft.", "feat.", "vs.")
    UNIQUE(track_id, name)
);

-- Speaker embeddings for cross-track matching
CREATE TABLE IF NOT EXISTS speaker_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id TEXT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    speaker_id TEXT NOT NULL,      -- SPEAKER_00, SPEAKER_01, etc.
    embedding BLOB NOT NULL,       -- 512-dim WavLM embedding (numpy tobytes)
    duration_sec REAL,             -- Total speaking duration for this speaker
    is_primary BOOLEAN DEFAULT FALSE,
    profile_id TEXT,               -- Voice profile UUID if assigned
    isolated_vocals_path TEXT,     -- Path to isolated vocals file
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(track_id, speaker_id)
);

-- Global speaker clusters (same person across tracks)
CREATE TABLE IF NOT EXISTS speaker_clusters (
    id TEXT PRIMARY KEY,           -- UUID
    name TEXT NOT NULL,            -- "Anth", "Tayler Holder", "Unknown 1"
    is_verified BOOLEAN DEFAULT FALSE,  -- User confirmed identity
    voice_profile_id TEXT,         -- Associated voice profile UUID
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Mapping: which embeddings belong to which cluster
CREATE TABLE IF NOT EXISTS cluster_members (
    cluster_id TEXT NOT NULL REFERENCES speaker_clusters(id) ON DELETE CASCADE,
    embedding_id INTEGER NOT NULL REFERENCES speaker_embeddings(id) ON DELETE CASCADE,
    confidence REAL,               -- Cosine similarity score (0-1)
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (cluster_id, embedding_id)
);

-- Indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_tracks_artist ON tracks(artist_name);
CREATE INDEX IF NOT EXISTS idx_featured_artists_track ON featured_artists(track_id);
CREATE INDEX IF NOT EXISTS idx_featured_artists_name ON featured_artists(name);
CREATE INDEX IF NOT EXISTS idx_embeddings_track ON speaker_embeddings(track_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_profile ON speaker_embeddings(profile_id);
CREATE INDEX IF NOT EXISTS idx_cluster_members_cluster ON cluster_members(cluster_id);
CREATE INDEX IF NOT EXISTS idx_cluster_members_embedding ON cluster_members(embedding_id);
"""


def init_database(db_path: Optional[Path] = None) -> None:
    """Initialize the database with schema.

    Args:
        db_path: Path to database file. Defaults to DATABASE_PATH.
    """
    if db_path is None:
        db_path = DATABASE_PATH

    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initializing database at {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        logger.info("Database schema initialized successfully")
    finally:
        conn.close()


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Get a database connection.

    Args:
        db_path: Path to database file. Defaults to DATABASE_PATH.

    Returns:
        SQLite connection with row factory set to sqlite3.Row
    """
    if db_path is None:
        db_path = DATABASE_PATH

    # Initialize if database doesn't exist
    if not db_path.exists():
        init_database(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_db_context(db_path: Optional[Path] = None):
    """Context manager for database operations.

    Args:
        db_path: Path to database file. Defaults to DATABASE_PATH.

    Yields:
        SQLite connection
    """
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def reset_database(db_path: Optional[Path] = None) -> None:
    """Reset the database by deleting and reinitializing.

    WARNING: This deletes all data!

    Args:
        db_path: Path to database file. Defaults to DATABASE_PATH.
    """
    if db_path is None:
        db_path = DATABASE_PATH

    if db_path.exists():
        logger.warning(f"Deleting existing database at {db_path}")
        db_path.unlink()

    init_database(db_path)


def get_database_stats(db_path: Optional[Path] = None) -> dict:
    """Get statistics about the database contents.

    Args:
        db_path: Path to database file. Defaults to DATABASE_PATH.

    Returns:
        Dictionary with counts for each table
    """
    with get_db_context(db_path) as conn:
        cursor = conn.cursor()

        stats = {}
        tables = ['tracks', 'featured_artists', 'speaker_embeddings',
                  'speaker_clusters', 'cluster_members']

        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cursor.fetchone()[0]

        # Additional stats
        cursor.execute("SELECT COUNT(DISTINCT artist_name) FROM tracks")
        stats['unique_artists'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT name) FROM featured_artists")
        stats['unique_featured_artists'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM speaker_clusters WHERE is_verified = TRUE")
        stats['verified_clusters'] = cursor.fetchone()[0]

        return stats
