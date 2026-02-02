"""
Database schema for AutoVoice persistent storage.

Supports both SQLite (for testing) and MySQL (for production).

Tables:
- tracks: YouTube track metadata
- featured_artists: Artists parsed from video titles
- speaker_embeddings: WavLM embeddings for speaker matching
- speaker_clusters: Global speaker identities across tracks
- cluster_members: Mapping of embeddings to clusters
"""

import os
import logging
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Boolean,
    Text, LargeBinary, DateTime, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.sql import func

logger = logging.getLogger(__name__)

# Database configuration from environment
DATABASE_TYPE = os.environ.get('AUTOVOICE_DB_TYPE', 'mysql')  # 'mysql' or 'sqlite'
DATABASE_HOST = os.environ.get('AUTOVOICE_DB_HOST', '127.0.0.1')
DATABASE_PORT = os.environ.get('AUTOVOICE_DB_PORT', '3306')
DATABASE_NAME = os.environ.get('AUTOVOICE_DB_NAME', 'autovoice')
DATABASE_USER = os.environ.get('AUTOVOICE_DB_USER', 'root')
DATABASE_PASS = os.environ.get('AUTOVOICE_DB_PASS', 'teamrsi123teamrsi123teamrsi123')

# Legacy SQLite path (for backward compatibility and testing)
DATABASE_PATH = Path('data/autovoice.db')

# SQLAlchemy base
Base = declarative_base()

# Global engine and session factory
_engine = None
_SessionFactory = None


def get_database_url(db_type: Optional[str] = None) -> str:
    """Get database connection URL.

    Args:
        db_type: 'mysql' or 'sqlite'. Defaults to DATABASE_TYPE env var.

    Returns:
        SQLAlchemy connection URL
    """
    db_type = db_type or DATABASE_TYPE

    if db_type == 'sqlite':
        DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{DATABASE_PATH}"
    else:
        # MySQL with PyMySQL driver
        return f"mysql+pymysql://{DATABASE_USER}:{DATABASE_PASS}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"


def get_engine(db_type: Optional[str] = None):
    """Get or create database engine.

    Args:
        db_type: 'mysql' or 'sqlite'. Defaults to DATABASE_TYPE env var.

    Returns:
        SQLAlchemy engine
    """
    global _engine

    if _engine is None:
        url = get_database_url(db_type)

        # Engine options
        if 'sqlite' in url:
            _engine = create_engine(url, echo=False)
        else:
            _engine = create_engine(
                url,
                echo=False,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,  # Handle stale connections
            )

        logger.info(f"Database engine created: {url.split('@')[-1] if '@' in url else url}")

    return _engine


def get_session_factory():
    """Get session factory."""
    global _SessionFactory

    if _SessionFactory is None:
        engine = get_engine()
        _SessionFactory = sessionmaker(bind=engine)

    return _SessionFactory


# ============================================================================
# ORM Models
# ============================================================================

class Track(Base):
    """YouTube track metadata."""
    __tablename__ = 'tracks'

    id = Column(String(64), primary_key=True)  # YouTube video ID
    title = Column(String(512))
    channel = Column(String(256))
    upload_date = Column(String(32))
    duration_sec = Column(Float)
    artist_name = Column(String(256), index=True)
    fetched_at = Column(DateTime, default=func.now())
    vocals_path = Column(String(1024))
    diarization_path = Column(String(1024))


class FeaturedArtist(Base):
    """Featured artists parsed from video titles."""
    __tablename__ = 'featured_artists'

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(String(64), ForeignKey('tracks.id', ondelete='CASCADE'), nullable=False, index=True)
    name = Column(String(256), nullable=False, index=True)
    pattern_matched = Column(String(64))

    __table_args__ = (
        UniqueConstraint('track_id', 'name', name='unique_track_artist'),
    )


class SpeakerEmbedding(Base):
    """Speaker embeddings for cross-track matching."""
    __tablename__ = 'speaker_embeddings'

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(String(64), ForeignKey('tracks.id', ondelete='CASCADE'), nullable=False, index=True)
    speaker_id = Column(String(32), nullable=False)
    embedding = Column(LargeBinary, nullable=False)  # 512-dim numpy array as bytes
    duration_sec = Column(Float)
    is_primary = Column(Boolean, default=False)
    profile_id = Column(String(64), index=True)
    isolated_vocals_path = Column(String(1024))
    created_at = Column(DateTime, default=func.now())

    __table_args__ = (
        UniqueConstraint('track_id', 'speaker_id', name='unique_track_speaker'),
    )


class SpeakerCluster(Base):
    """Global speaker clusters (same person across tracks)."""
    __tablename__ = 'speaker_clusters'

    id = Column(String(64), primary_key=True)  # UUID
    name = Column(String(256), nullable=False)
    is_verified = Column(Boolean, default=False)
    voice_profile_id = Column(String(64))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class ClusterMember(Base):
    """Mapping: which embeddings belong to which cluster."""
    __tablename__ = 'cluster_members'

    cluster_id = Column(String(64), ForeignKey('speaker_clusters.id', ondelete='CASCADE'), primary_key=True)
    embedding_id = Column(Integer, ForeignKey('speaker_embeddings.id', ondelete='CASCADE'), primary_key=True, index=True)
    confidence = Column(Float)
    added_at = Column(DateTime, default=func.now())


# ============================================================================
# Database Operations
# ============================================================================

def init_database(db_type: Optional[str] = None) -> None:
    """Initialize the database with schema.

    Args:
        db_type: 'mysql' or 'sqlite'. Defaults to DATABASE_TYPE env var.
    """
    engine = get_engine(db_type)
    Base.metadata.create_all(engine)
    logger.info("Database schema initialized successfully")


@contextmanager
def get_db_session(db_path: Optional[Path] = None) -> Session:
    """Context manager for database sessions.

    Args:
        db_path: Deprecated, kept for backward compatibility.

    Yields:
        SQLAlchemy session
    """
    SessionFactory = get_session_factory()
    session = SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# Legacy aliases for backward compatibility
get_db_context = get_db_session


def get_connection(db_path: Optional[Path] = None):
    """Legacy function for backward compatibility.

    Returns a session instead of raw connection.
    """
    SessionFactory = get_session_factory()
    return SessionFactory()


def reset_database(db_type: Optional[str] = None) -> None:
    """Reset the database by dropping and recreating all tables.

    WARNING: This deletes all data!

    Args:
        db_type: 'mysql' or 'sqlite'. Defaults to DATABASE_TYPE env var.
    """
    global _engine, _SessionFactory

    engine = get_engine(db_type)
    logger.warning("Dropping all tables...")
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    logger.info("Database reset complete")


def get_database_stats(db_path: Optional[Path] = None) -> dict:
    """Get statistics about the database contents.

    Args:
        db_path: Deprecated, kept for backward compatibility.

    Returns:
        Dictionary with counts for each table
    """
    with get_db_session() as session:
        stats = {
            'tracks': session.query(Track).count(),
            'featured_artists': session.query(FeaturedArtist).count(),
            'speaker_embeddings': session.query(SpeakerEmbedding).count(),
            'speaker_clusters': session.query(SpeakerCluster).count(),
            'cluster_members': session.query(ClusterMember).count(),
        }

        # Additional stats
        from sqlalchemy import func as sqlfunc
        stats['unique_artists'] = session.query(
            sqlfunc.count(sqlfunc.distinct(Track.artist_name))
        ).scalar() or 0

        stats['unique_featured_artists'] = session.query(
            sqlfunc.count(sqlfunc.distinct(FeaturedArtist.name))
        ).scalar() or 0

        stats['verified_clusters'] = session.query(SpeakerCluster).filter(
            SpeakerCluster.is_verified == True
        ).count()

        return stats


def close_database() -> None:
    """Close database connections and reset globals."""
    global _engine, _SessionFactory

    if _engine is not None:
        _engine.dispose()
        _engine = None
    _SessionFactory = None
    logger.info("Database connections closed")
