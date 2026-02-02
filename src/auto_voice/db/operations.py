"""
Database operations for AutoVoice persistent storage.

CRUD operations using SQLAlchemy ORM for:
- Tracks
- Featured artists
- Speaker embeddings
- Speaker clusters
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np

from sqlalchemy import func as sqlfunc
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from .schema import (
    get_db_session, get_engine, DATABASE_TYPE,
    Track, FeaturedArtist, SpeakerEmbedding, SpeakerCluster, ClusterMember
)

logger = logging.getLogger(__name__)


def _get_upsert(engine):
    """Get the appropriate upsert function for the database type."""
    dialect = engine.dialect.name
    if dialect == 'mysql':
        return mysql_insert
    else:
        return sqlite_insert


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
        db_path: Deprecated, kept for compatibility
    """
    with get_db_session() as session:
        existing = session.query(Track).filter(Track.id == track_id).first()

        if existing:
            # Update existing record
            if title is not None:
                existing.title = title
            if channel is not None:
                existing.channel = channel
            if upload_date is not None:
                existing.upload_date = upload_date
            if duration_sec is not None:
                existing.duration_sec = duration_sec
            if artist_name is not None:
                existing.artist_name = artist_name
            if vocals_path is not None:
                existing.vocals_path = vocals_path
            if diarization_path is not None:
                existing.diarization_path = diarization_path
            existing.fetched_at = datetime.utcnow()
        else:
            # Insert new record
            track = Track(
                id=track_id,
                title=title,
                channel=channel,
                upload_date=upload_date,
                duration_sec=duration_sec,
                artist_name=artist_name,
                vocals_path=vocals_path,
                diarization_path=diarization_path,
            )
            session.add(track)


def get_track(track_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Get a track by ID.

    Args:
        track_id: YouTube video ID
        db_path: Deprecated

    Returns:
        Track dict or None if not found
    """
    with get_db_session() as session:
        track = session.query(Track).filter(Track.id == track_id).first()
        if track:
            return {
                'id': track.id,
                'title': track.title,
                'channel': track.channel,
                'upload_date': track.upload_date,
                'duration_sec': track.duration_sec,
                'artist_name': track.artist_name,
                'fetched_at': track.fetched_at,
                'vocals_path': track.vocals_path,
                'diarization_path': track.diarization_path,
            }
        return None


def get_all_tracks(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Get all tracks.

    Args:
        db_path: Deprecated

    Returns:
        List of track dicts
    """
    with get_db_session() as session:
        tracks = session.query(Track).order_by(Track.artist_name, Track.title).all()
        return [{
            'id': t.id,
            'title': t.title,
            'channel': t.channel,
            'upload_date': t.upload_date,
            'duration_sec': t.duration_sec,
            'artist_name': t.artist_name,
            'fetched_at': t.fetched_at,
            'vocals_path': t.vocals_path,
            'diarization_path': t.diarization_path,
        } for t in tracks]


def get_tracks_by_artist(artist_name: str, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Get all tracks for an artist.

    Args:
        artist_name: Artist name
        db_path: Deprecated

    Returns:
        List of track dicts
    """
    with get_db_session() as session:
        tracks = session.query(Track).filter(
            Track.artist_name == artist_name
        ).order_by(Track.title).all()
        return [{
            'id': t.id,
            'title': t.title,
            'channel': t.channel,
            'upload_date': t.upload_date,
            'duration_sec': t.duration_sec,
            'artist_name': t.artist_name,
            'fetched_at': t.fetched_at,
            'vocals_path': t.vocals_path,
            'diarization_path': t.diarization_path,
        } for t in tracks]


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
        db_path: Deprecated

    Returns:
        Row ID of inserted/existing record
    """
    with get_db_session() as session:
        existing = session.query(FeaturedArtist).filter(
            FeaturedArtist.track_id == track_id,
            FeaturedArtist.name == name
        ).first()

        if existing:
            if pattern_matched is not None:
                existing.pattern_matched = pattern_matched
            return existing.id
        else:
            artist = FeaturedArtist(
                track_id=track_id,
                name=name,
                pattern_matched=pattern_matched,
            )
            session.add(artist)
            session.flush()  # Get the ID
            return artist.id


def get_featured_artists_for_track(
    track_id: str,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Get all featured artists for a track.

    Args:
        track_id: YouTube video ID
        db_path: Deprecated

    Returns:
        List of featured artist dicts
    """
    with get_db_session() as session:
        artists = session.query(FeaturedArtist).filter(
            FeaturedArtist.track_id == track_id
        ).order_by(FeaturedArtist.name).all()
        return [{
            'id': a.id,
            'track_id': a.track_id,
            'name': a.name,
            'pattern_matched': a.pattern_matched,
        } for a in artists]


def get_all_featured_artists(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Get all featured artists with track counts.

    Args:
        db_path: Deprecated

    Returns:
        List of dicts with name, track_count, tracks
    """
    with get_db_session() as session:
        results = session.query(
            FeaturedArtist.name,
            sqlfunc.count(sqlfunc.distinct(FeaturedArtist.track_id)).label('track_count'),
            sqlfunc.group_concat(FeaturedArtist.track_id).label('track_ids')
        ).group_by(FeaturedArtist.name).order_by(
            sqlfunc.count(sqlfunc.distinct(FeaturedArtist.track_id)).desc()
        ).all()

        return [{
            'name': r.name,
            'track_count': r.track_count,
            'track_ids': r.track_ids,
        } for r in results]


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
        db_path: Deprecated

    Returns:
        Row ID of inserted/updated record
    """
    embedding_bytes = embedding.astype(np.float32).tobytes()

    with get_db_session() as session:
        existing = session.query(SpeakerEmbedding).filter(
            SpeakerEmbedding.track_id == track_id,
            SpeakerEmbedding.speaker_id == speaker_id
        ).first()

        if existing:
            existing.embedding = embedding_bytes
            if duration_sec is not None:
                existing.duration_sec = duration_sec
            existing.is_primary = is_primary
            if profile_id is not None:
                existing.profile_id = profile_id
            if isolated_vocals_path is not None:
                existing.isolated_vocals_path = isolated_vocals_path
            return existing.id
        else:
            emb = SpeakerEmbedding(
                track_id=track_id,
                speaker_id=speaker_id,
                embedding=embedding_bytes,
                duration_sec=duration_sec,
                is_primary=is_primary,
                profile_id=profile_id,
                isolated_vocals_path=isolated_vocals_path,
            )
            session.add(emb)
            session.flush()
            return emb.id


def get_embeddings_for_track(
    track_id: str,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Get all speaker embeddings for a track.

    Args:
        track_id: YouTube video ID
        db_path: Deprecated

    Returns:
        List of embedding dicts with numpy arrays
    """
    with get_db_session() as session:
        embeddings = session.query(SpeakerEmbedding).filter(
            SpeakerEmbedding.track_id == track_id
        ).order_by(SpeakerEmbedding.speaker_id).all()

        return [{
            'id': e.id,
            'track_id': e.track_id,
            'speaker_id': e.speaker_id,
            'embedding': np.frombuffer(e.embedding, dtype=np.float32),
            'duration_sec': e.duration_sec,
            'is_primary': e.is_primary,
            'profile_id': e.profile_id,
            'isolated_vocals_path': e.isolated_vocals_path,
            'created_at': e.created_at,
        } for e in embeddings]


def get_all_embeddings(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Get all speaker embeddings.

    Args:
        db_path: Deprecated

    Returns:
        List of embedding dicts with numpy arrays
    """
    with get_db_session() as session:
        embeddings = session.query(SpeakerEmbedding).order_by(
            SpeakerEmbedding.track_id, SpeakerEmbedding.speaker_id
        ).all()

        return [{
            'id': e.id,
            'track_id': e.track_id,
            'speaker_id': e.speaker_id,
            'embedding': np.frombuffer(e.embedding, dtype=np.float32),
            'duration_sec': e.duration_sec,
            'is_primary': e.is_primary,
            'profile_id': e.profile_id,
            'isolated_vocals_path': e.isolated_vocals_path,
            'created_at': e.created_at,
        } for e in embeddings]


def get_embeddings_by_cluster(
    cluster_id: str,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Get all embeddings belonging to a cluster.

    Args:
        cluster_id: Cluster UUID
        db_path: Deprecated

    Returns:
        List of embedding dicts with confidence scores
    """
    with get_db_session() as session:
        results = session.query(
            SpeakerEmbedding, ClusterMember.confidence
        ).join(
            ClusterMember, SpeakerEmbedding.id == ClusterMember.embedding_id
        ).filter(
            ClusterMember.cluster_id == cluster_id
        ).order_by(ClusterMember.confidence.desc()).all()

        return [{
            'id': e.id,
            'track_id': e.track_id,
            'speaker_id': e.speaker_id,
            'embedding': np.frombuffer(e.embedding, dtype=np.float32),
            'duration_sec': e.duration_sec,
            'is_primary': e.is_primary,
            'profile_id': e.profile_id,
            'isolated_vocals_path': e.isolated_vocals_path,
            'created_at': e.created_at,
            'confidence': conf,
        } for e, conf in results]


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
        db_path: Deprecated

    Returns:
        Cluster UUID
    """
    cluster_id = str(uuid.uuid4())

    with get_db_session() as session:
        cluster = SpeakerCluster(
            id=cluster_id,
            name=name,
            is_verified=is_verified,
            voice_profile_id=voice_profile_id,
        )
        session.add(cluster)

    return cluster_id


def get_cluster(cluster_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Get a cluster by ID.

    Args:
        cluster_id: Cluster UUID
        db_path: Deprecated

    Returns:
        Cluster dict or None
    """
    with get_db_session() as session:
        cluster = session.query(SpeakerCluster).filter(
            SpeakerCluster.id == cluster_id
        ).first()

        if cluster:
            return {
                'id': cluster.id,
                'name': cluster.name,
                'is_verified': cluster.is_verified,
                'voice_profile_id': cluster.voice_profile_id,
                'created_at': cluster.created_at,
                'updated_at': cluster.updated_at,
            }
        return None


def get_all_clusters(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Get all clusters with member counts.

    Args:
        db_path: Deprecated

    Returns:
        List of cluster dicts with member_count
    """
    with get_db_session() as session:
        results = session.query(
            SpeakerCluster,
            sqlfunc.count(ClusterMember.embedding_id).label('member_count'),
            sqlfunc.sum(SpeakerEmbedding.duration_sec).label('total_duration_sec')
        ).outerjoin(
            ClusterMember, SpeakerCluster.id == ClusterMember.cluster_id
        ).outerjoin(
            SpeakerEmbedding, ClusterMember.embedding_id == SpeakerEmbedding.id
        ).group_by(SpeakerCluster.id).order_by(SpeakerCluster.name).all()

        return [{
            'id': c.id,
            'name': c.name,
            'is_verified': c.is_verified,
            'voice_profile_id': c.voice_profile_id,
            'created_at': c.created_at,
            'updated_at': c.updated_at,
            'member_count': count or 0,
            'total_duration_sec': duration,
        } for c, count, duration in results]


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
        db_path: Deprecated
    """
    with get_db_session() as session:
        cluster = session.query(SpeakerCluster).filter(
            SpeakerCluster.id == cluster_id
        ).first()
        if cluster:
            cluster.name = name
            cluster.is_verified = is_verified
            cluster.updated_at = datetime.utcnow()


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
        db_path: Deprecated
    """
    with get_db_session() as session:
        # Move all members
        session.query(ClusterMember).filter(
            ClusterMember.cluster_id == source_cluster_id
        ).update({'cluster_id': target_cluster_id})

        # Delete source cluster
        session.query(SpeakerCluster).filter(
            SpeakerCluster.id == source_cluster_id
        ).delete()

        # Update target timestamp
        target = session.query(SpeakerCluster).filter(
            SpeakerCluster.id == target_cluster_id
        ).first()
        if target:
            target.updated_at = datetime.utcnow()


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
        db_path: Deprecated
    """
    with get_db_session() as session:
        existing = session.query(ClusterMember).filter(
            ClusterMember.cluster_id == cluster_id,
            ClusterMember.embedding_id == embedding_id
        ).first()

        if existing:
            existing.confidence = confidence
        else:
            member = ClusterMember(
                cluster_id=cluster_id,
                embedding_id=embedding_id,
                confidence=confidence,
            )
            session.add(member)


def remove_from_cluster(
    cluster_id: str,
    embedding_id: int,
    db_path: Optional[Path] = None,
) -> None:
    """Remove an embedding from a cluster.

    Args:
        cluster_id: Cluster UUID
        embedding_id: Speaker embedding row ID
        db_path: Deprecated
    """
    with get_db_session() as session:
        session.query(ClusterMember).filter(
            ClusterMember.cluster_id == cluster_id,
            ClusterMember.embedding_id == embedding_id
        ).delete()


def get_cluster_members(
    cluster_id: str,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Get all members of a cluster with track info.

    Args:
        cluster_id: Cluster UUID
        db_path: Deprecated

    Returns:
        List of member dicts with track and embedding info
    """
    with get_db_session() as session:
        results = session.query(
            ClusterMember.confidence,
            SpeakerEmbedding.id.label('embedding_id'),
            SpeakerEmbedding.track_id,
            SpeakerEmbedding.speaker_id,
            SpeakerEmbedding.duration_sec,
            SpeakerEmbedding.is_primary,
            SpeakerEmbedding.isolated_vocals_path,
            Track.title.label('track_title'),
            Track.artist_name,
        ).join(
            SpeakerEmbedding, ClusterMember.embedding_id == SpeakerEmbedding.id
        ).join(
            Track, SpeakerEmbedding.track_id == Track.id
        ).filter(
            ClusterMember.cluster_id == cluster_id
        ).order_by(ClusterMember.confidence.desc()).all()

        return [{
            'confidence': r.confidence,
            'embedding_id': r.embedding_id,
            'track_id': r.track_id,
            'speaker_id': r.speaker_id,
            'duration_sec': r.duration_sec,
            'is_primary': r.is_primary,
            'isolated_vocals_path': r.isolated_vocals_path,
            'track_title': r.track_title,
            'artist_name': r.artist_name,
        } for r in results]


# ============================================================================
# Utility Functions
# ============================================================================

def find_unclustered_embeddings(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Find all embeddings not assigned to any cluster.

    Args:
        db_path: Deprecated

    Returns:
        List of embedding dicts
    """
    with get_db_session() as session:
        # Subquery for clustered embeddings
        clustered = session.query(ClusterMember.embedding_id).subquery()

        results = session.query(
            SpeakerEmbedding,
            Track.title.label('track_title'),
            Track.artist_name,
        ).join(
            Track, SpeakerEmbedding.track_id == Track.id
        ).filter(
            ~SpeakerEmbedding.id.in_(session.query(clustered))
        ).order_by(
            SpeakerEmbedding.track_id, SpeakerEmbedding.speaker_id
        ).all()

        return [{
            'id': e.id,
            'track_id': e.track_id,
            'speaker_id': e.speaker_id,
            'embedding': np.frombuffer(e.embedding, dtype=np.float32),
            'duration_sec': e.duration_sec,
            'is_primary': e.is_primary,
            'profile_id': e.profile_id,
            'isolated_vocals_path': e.isolated_vocals_path,
            'created_at': e.created_at,
            'track_title': title,
            'artist_name': artist,
        } for e, title, artist in results]


def get_embedding_by_id(
    embedding_id: int,
    db_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Get a single embedding by ID.

    Args:
        embedding_id: Embedding row ID
        db_path: Deprecated

    Returns:
        Embedding dict or None
    """
    with get_db_session() as session:
        e = session.query(SpeakerEmbedding).filter(
            SpeakerEmbedding.id == embedding_id
        ).first()

        if e:
            return {
                'id': e.id,
                'track_id': e.track_id,
                'speaker_id': e.speaker_id,
                'embedding': np.frombuffer(e.embedding, dtype=np.float32),
                'duration_sec': e.duration_sec,
                'is_primary': e.is_primary,
                'profile_id': e.profile_id,
                'isolated_vocals_path': e.isolated_vocals_path,
                'created_at': e.created_at,
            }
        return None
