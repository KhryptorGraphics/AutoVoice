"""Database session management and initialization.

Provides connection pooling, session factories, and database initialization
for PostgreSQL with SQLAlchemy.
"""

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from auto_voice.profiles.db.models import Base

# Default database URL - uses environment variable or falls back to local PostgreSQL
DEFAULT_DATABASE_URL = "postgresql://autovoice:autovoice@localhost:5432/autovoice"

_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def get_database_url() -> str:
    """Get the database URL from environment or use default."""
    return os.environ.get("AUTOVOICE_DATABASE_URL", DEFAULT_DATABASE_URL)


def get_engine(database_url: str | None = None, **kwargs) -> Engine:
    """Get or create the SQLAlchemy engine.

    Args:
        database_url: Optional database URL. If not provided, uses environment
            variable AUTOVOICE_DATABASE_URL or default.
        **kwargs: Additional arguments passed to create_engine.

    Returns:
        SQLAlchemy Engine instance.
    """
    global _engine

    if _engine is None:
        url = database_url or get_database_url()
        default_kwargs = {
            "pool_pre_ping": True,
            "pool_recycle": 3600,
        }
        # Connection pool settings only valid for PostgreSQL/MySQL, not SQLite
        if not url.startswith("sqlite"):
            default_kwargs["pool_size"] = 5
            default_kwargs["max_overflow"] = 10
        default_kwargs.update(kwargs)
        _engine = create_engine(url, **default_kwargs)

    return _engine


def get_session_factory(engine: Engine | None = None) -> sessionmaker:
    """Get or create the session factory.

    Args:
        engine: Optional engine. If not provided, uses default engine.

    Returns:
        SQLAlchemy sessionmaker instance.
    """
    global _SessionLocal

    if _SessionLocal is None:
        eng = engine or get_engine()
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=eng,
        )

    return _SessionLocal


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Get a database session as a context manager.

    Yields:
        SQLAlchemy Session that automatically commits on success
        or rolls back on exception.

    Example:
        with get_db_session() as session:
            profile = VoiceProfileDB(user_id="user-1", name="My Voice")
            session.add(profile)
            # Commits automatically on context exit
    """
    session_factory = get_session_factory()
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(engine: Engine | None = None) -> None:
    """Initialize the database by creating all tables.

    Args:
        engine: Optional engine. If not provided, uses default engine.

    Note:
        This creates tables if they don't exist. For production migrations,
        use Alembic instead.
    """
    eng = engine or get_engine()
    Base.metadata.create_all(bind=eng)


def reset_engine() -> None:
    """Reset the global engine and session factory.

    Useful for testing or reconfiguring the database connection.
    """
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
