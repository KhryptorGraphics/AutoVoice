"""
Database module for AutoVoice persistent storage.

Provides SQLite-based storage for:
- Track metadata from YouTube
- Featured artist information
- Speaker embeddings for cross-track matching
- Speaker clusters for global identity resolution
"""

from .schema import init_database, get_connection, DATABASE_PATH
from .operations import (
    # Track operations
    upsert_track,
    get_track,
    get_all_tracks,
    get_tracks_by_artist,
    # Featured artist operations
    add_featured_artist,
    get_featured_artists_for_track,
    get_all_featured_artists,
    # Speaker embedding operations
    add_speaker_embedding,
    get_embeddings_for_track,
    get_all_embeddings,
    get_embeddings_by_cluster,
    # Cluster operations
    create_cluster,
    get_cluster,
    get_all_clusters,
    update_cluster_name,
    merge_clusters,
    add_to_cluster,
    remove_from_cluster,
    get_cluster_members,
)

__all__ = [
    'init_database',
    'get_connection',
    'DATABASE_PATH',
    'upsert_track',
    'get_track',
    'get_all_tracks',
    'get_tracks_by_artist',
    'add_featured_artist',
    'get_featured_artists_for_track',
    'get_all_featured_artists',
    'add_speaker_embedding',
    'get_embeddings_for_track',
    'get_all_embeddings',
    'get_embeddings_by_cluster',
    'create_cluster',
    'get_cluster',
    'get_all_clusters',
    'update_cluster_name',
    'merge_clusters',
    'add_to_cluster',
    'remove_from_cluster',
    'get_cluster_members',
]
