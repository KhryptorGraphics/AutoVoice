"""Speaker identification API endpoints.

Endpoints for cross-track speaker identification, clustering, and management.
"""

import json
import logging
import os
import uuid
from pathlib import Path
from flask import Blueprint, request, jsonify, send_file
from typing import Dict, Any, Optional
import tempfile

from .karaoke_api import log_request, rate_limit

logger = logging.getLogger(__name__)

# Blueprint with /api/v1/speakers prefix
speaker_bp = Blueprint('speakers', __name__, url_prefix='/api/v1/speakers')

# Track extraction jobs in memory
_extraction_jobs: Dict[str, Dict[str, Any]] = {}


def _get_db_operations():
    """Lazy import database operations."""
    from ..db.operations import (
        get_all_tracks, get_track, get_tracks_by_artist,
        get_all_clusters, get_cluster, get_cluster_members,
        update_cluster_name, merge_clusters, create_cluster,
        get_embeddings_by_cluster, get_featured_artists_for_track,
        get_all_featured_artists,
    )
    return {
        'get_all_tracks': get_all_tracks,
        'get_track': get_track,
        'get_tracks_by_artist': get_tracks_by_artist,
        'get_all_clusters': get_all_clusters,
        'get_cluster': get_cluster,
        'get_cluster_members': get_cluster_members,
        'update_cluster_name': update_cluster_name,
        'merge_clusters': merge_clusters,
        'create_cluster': create_cluster,
        'get_embeddings_by_cluster': get_embeddings_by_cluster,
        'get_featured_artists_for_track': get_featured_artists_for_track,
        'get_all_featured_artists': get_all_featured_artists,
    }


# =============================================================================
# Extraction Endpoints
# =============================================================================

@speaker_bp.route('/extraction/run', methods=['POST'])
def run_extraction():
    """Trigger speaker extraction for an artist.

    Request JSON:
        artist_name: Artist name (e.g., 'conor_maynard')
        run_clustering: Whether to run cross-track clustering (default: True)

    Returns:
        JSON with job_id for tracking progress
    """
    data = request.get_json() or {}
    artist_name = data.get('artist_name')

    if not artist_name:
        return jsonify({'error': 'artist_name is required'}), 400

    job_id = str(uuid.uuid4())

    # Start extraction in background
    _extraction_jobs[job_id] = {
        'status': 'pending',
        'artist_name': artist_name,
        'progress': 0,
        'message': 'Starting extraction...',
        'error': None,
    }

    # TODO: Run in background thread/celery
    # For now, run synchronously but track progress
    try:
        _extraction_jobs[job_id]['status'] = 'running'

        from ..audio.speaker_matcher import SpeakerMatcher

        matcher = SpeakerMatcher()

        # Extract embeddings
        _extraction_jobs[job_id]['message'] = f'Extracting embeddings for {artist_name}...'
        _extraction_jobs[job_id]['progress'] = 25

        stats = matcher.extract_embeddings_for_artist(artist_name)

        # Run clustering if requested
        if data.get('run_clustering', True):
            _extraction_jobs[job_id]['message'] = 'Clustering speakers across tracks...'
            _extraction_jobs[job_id]['progress'] = 50

            clusters = matcher.cluster_speakers()

            _extraction_jobs[job_id]['message'] = 'Auto-matching clusters to artists...'
            _extraction_jobs[job_id]['progress'] = 75

            match_stats = matcher.auto_match_clusters_to_artists()
            stats['clustering'] = {
                'clusters_created': len(clusters),
            }
            stats['matching'] = match_stats

        _extraction_jobs[job_id]['status'] = 'complete'
        _extraction_jobs[job_id]['progress'] = 100
        _extraction_jobs[job_id]['message'] = 'Extraction complete'
        _extraction_jobs[job_id]['result'] = stats

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        _extraction_jobs[job_id]['status'] = 'failed'
        _extraction_jobs[job_id]['error'] = str(e)

    return jsonify({
        'job_id': job_id,
        'status': _extraction_jobs[job_id]['status'],
    })


@speaker_bp.route('/extraction/status/<job_id>', methods=['GET'])
def get_extraction_status(job_id: str):
    """Get extraction job status.

    Returns:
        JSON with job status and progress
    """
    if job_id not in _extraction_jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = _extraction_jobs[job_id]
    return jsonify({
        'job_id': job_id,
        'status': job['status'],
        'progress': job['progress'],
        'message': job['message'],
        'error': job.get('error'),
        'result': job.get('result'),
    })


# =============================================================================
# Track Endpoints
# =============================================================================

@speaker_bp.route('/tracks', methods=['GET'])
def list_tracks():
    """List all tracks with YouTube metadata.

    Query params:
        artist: Filter by artist name
        has_featured: Only tracks with featured artists

    Returns:
        JSON array of tracks
    """
    db = _get_db_operations()

    artist = request.args.get('artist')
    has_featured = request.args.get('has_featured', '').lower() == 'true'

    if artist:
        tracks = db['get_tracks_by_artist'](artist)
    else:
        tracks = db['get_all_tracks']()

    # Add featured artists to each track
    for track in tracks:
        featured = db['get_featured_artists_for_track'](track['id'])
        track['featured_artists'] = [f['name'] for f in featured]

    # Filter if requested
    if has_featured:
        tracks = [t for t in tracks if t['featured_artists']]

    return jsonify({
        'tracks': tracks,
        'count': len(tracks),
    })


@speaker_bp.route('/tracks/<track_id>', methods=['GET'])
def get_track_details(track_id: str):
    """Get track details with featured artists.

    Returns:
        JSON with track info and featured artists
    """
    db = _get_db_operations()

    track = db['get_track'](track_id)
    if not track:
        return jsonify({'error': 'Track not found'}), 404

    featured = db['get_featured_artists_for_track'](track_id)
    track['featured_artists'] = featured

    return jsonify(track)


@speaker_bp.route('/tracks/fetch-metadata', methods=['POST'])
def fetch_metadata():
    """Fetch YouTube metadata for all tracks.

    Request JSON:
        artist_name: Optional - only fetch for this artist

    Returns:
        JSON with fetch statistics
    """
    data = request.get_json() or {}
    artist_name = data.get('artist_name')

    try:
        from ..audio.youtube_metadata import (
            YouTubeMetadataFetcher, populate_database_from_files
        )

        if artist_name:
            # Fetch for specific artist
            stats = populate_database_from_files(
                Path(f'data/separated_youtube/{artist_name}'),
                artist_name=artist_name,
            )
        else:
            # Fetch for all artists
            stats = {'total_tracks': 0, 'total_featured': 0}
            for artist in ['conor_maynard', 'william_singe']:
                artist_dir = Path(f'data/separated_youtube/{artist}')
                if artist_dir.exists():
                    artist_stats = populate_database_from_files(
                        artist_dir, artist_name=artist
                    )
                    stats['total_tracks'] += artist_stats.get('tracks_processed', 0)
                    stats['total_featured'] += artist_stats.get('featured_found', 0)

        return jsonify({
            'success': True,
            'stats': stats,
        })

    except Exception as e:
        logger.error(f"Metadata fetch failed: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Cluster Endpoints
# =============================================================================

@speaker_bp.route('/clusters', methods=['GET'])
def list_clusters():
    """List all speaker clusters.

    Returns:
        JSON array of clusters with member counts
    """
    db = _get_db_operations()
    clusters = db['get_all_clusters']()

    return jsonify({
        'clusters': clusters,
        'count': len(clusters),
    })


@speaker_bp.route('/clusters/<cluster_id>', methods=['GET'])
def get_cluster_details(cluster_id: str):
    """Get cluster details with member tracks.

    Returns:
        JSON with cluster info and members
    """
    db = _get_db_operations()

    cluster = db['get_cluster'](cluster_id)
    if not cluster:
        return jsonify({'error': 'Cluster not found'}), 404

    members = db['get_cluster_members'](cluster_id)

    # Group by track
    tracks = {}
    for member in members:
        track_id = member['track_id']
        if track_id not in tracks:
            tracks[track_id] = {
                'track_id': track_id,
                'title': member.get('track_title'),
                'artist_name': member.get('artist_name'),
                'speakers': [],
            }
        tracks[track_id]['speakers'].append({
            'speaker_id': member['speaker_id'],
            'duration_sec': member['duration_sec'],
            'is_primary': member['is_primary'],
            'confidence': member.get('confidence'),
        })

    return jsonify({
        'cluster': cluster,
        'members': members,
        'tracks': list(tracks.values()),
        'track_count': len(tracks),
    })


@log_request
@rate_limit(5, 60)
@speaker_bp.route('/clusters/<cluster_id>/name', methods=['PUT'])
def update_cluster_name_endpoint(cluster_id: str):
    """Update cluster name.

    Request JSON:
        name: New name for the cluster
        is_verified: Whether the name is user-verified (default: True)

    Returns:
        JSON with updated cluster
    """
    db = _get_db_operations()
    data = request.get_json() or {}

    name = data.get('name')
    if not name:
        return jsonify({'error': 'name is required'}), 400

    is_verified = data.get('is_verified', True)

    try:
        db['update_cluster_name'](cluster_id, name, is_verified)
        cluster = db['get_cluster'](cluster_id)
        return jsonify({
            'success': True,
            'cluster': cluster,
        })
    except Exception as e:
        logger.error(f"Failed to update cluster name: {e}")
        return jsonify({'error': str(e)}), 500


@log_request
@rate_limit(5, 60)
@speaker_bp.route('/clusters/merge', methods=['POST'])
def merge_clusters_endpoint():
    """Merge two clusters.

    Request JSON:
        target_id: Cluster to keep
        source_id: Cluster to merge into target (will be deleted)

    Returns:
        JSON with merged cluster
    """
    db = _get_db_operations()
    data = request.get_json() or {}

    target_id = data.get('target_id')
    source_id = data.get('source_id')

    if not target_id or not source_id:
        return jsonify({'error': 'target_id and source_id are required'}), 400

    if target_id == source_id:
        return jsonify({'error': 'Cannot merge cluster with itself'}), 400

    try:
        db['merge_clusters'](target_id, source_id)
        cluster = db['get_cluster'](target_id)
        members = db['get_cluster_members'](target_id)

        return jsonify({
            'success': True,
            'cluster': cluster,
            'member_count': len(members),
        })
    except Exception as e:
        logger.error(f"Failed to merge clusters: {e}")
        return jsonify({'error': str(e)}), 500


@log_request
@rate_limit(10, 60)
@speaker_bp.route('/clusters/split', methods=['POST'])
def split_cluster_endpoint():
    """Split a cluster by moving members to new cluster.

    Request JSON:
        cluster_id: Cluster to split from
        embedding_ids: List of embedding IDs to move to new cluster
        new_name: Name for the new cluster

    Returns:
        JSON with both clusters
    """
    db = _get_db_operations()
    data = request.get_json() or {}

    cluster_id = data.get('cluster_id')
    embedding_ids = data.get('embedding_ids', [])
    new_name = data.get('new_name', 'Split Cluster')

    if not cluster_id:
        return jsonify({'error': 'cluster_id is required'}), 400

    if not embedding_ids:
        return jsonify({'error': 'embedding_ids is required'}), 400

    try:
        from ..db.operations import remove_from_cluster, add_to_cluster

        # Create new cluster
        new_cluster_id = db['create_cluster'](new_name, is_verified=False)

        # Move embeddings
        for emb_id in embedding_ids:
            remove_from_cluster(cluster_id, emb_id)
            add_to_cluster(new_cluster_id, emb_id)

        original_cluster = db['get_cluster'](cluster_id)
        new_cluster = db['get_cluster'](new_cluster_id)

        return jsonify({
            'success': True,
            'original_cluster': original_cluster,
            'new_cluster': new_cluster,
        })
    except Exception as e:
        logger.error(f"Failed to split cluster: {e}")
        return jsonify({'error': str(e)}), 500


@speaker_bp.route('/clusters/<cluster_id>/sample', methods=['GET'])
def get_cluster_sample(cluster_id: str):
    """Get audio sample for a cluster.

    Query params:
        max_duration: Maximum sample duration in seconds (default: 10)

    Returns:
        WAV audio file
    """
    max_duration = float(request.args.get('max_duration', 10.0))

    try:
        from ..audio.speaker_matcher import SpeakerMatcher
        import soundfile as sf

        matcher = SpeakerMatcher()
        audio, sr = matcher.get_cluster_sample_audio(cluster_id, max_duration)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio, sr)
            return send_file(
                f.name,
                mimetype='audio/wav',
                as_attachment=True,
                download_name=f'cluster_{cluster_id[:8]}_sample.wav',
            )

    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Failed to get cluster sample: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Speaker Identification Endpoints
# =============================================================================

@speaker_bp.route('/identify', methods=['POST'])
def run_speaker_identification():
    """Run cross-track speaker matching pipeline.

    Request JSON:
        artists: List of artist names (default: all)
        threshold: Similarity threshold (default: 0.85)
        min_duration: Minimum cluster duration (default: 30)

    Returns:
        JSON with identification results
    """
    data = request.get_json() or {}

    artists = data.get('artists')
    threshold = data.get('threshold', 0.85)
    min_duration = data.get('min_duration', 30.0)

    try:
        from ..audio.speaker_matcher import SpeakerMatcher

        matcher = SpeakerMatcher(
            similarity_threshold=threshold,
            min_cluster_duration=min_duration,
        )

        stats = {
            'artists': {},
            'clustering': {},
            'matching': {},
        }

        # Extract embeddings for each artist
        if artists is None:
            artists = ['conor_maynard', 'william_singe']

        for artist in artists:
            artist_stats = matcher.extract_embeddings_for_artist(artist)
            stats['artists'][artist] = artist_stats

        # Cluster speakers
        clusters = matcher.cluster_speakers()
        stats['clustering'] = {
            'clusters_created': len(clusters),
            'clusters': [
                {
                    'cluster_id': c['cluster_id'],
                    'member_count': c['member_count'],
                    'duration_sec': c['total_duration_sec'],
                }
                for c in clusters
            ],
        }

        # Auto-match to featured artists
        match_stats = matcher.auto_match_clusters_to_artists()
        stats['matching'] = match_stats

        return jsonify({
            'success': True,
            'stats': stats,
        })

    except Exception as e:
        logger.error(f"Speaker identification failed: {e}")
        return jsonify({'error': str(e)}), 500


@speaker_bp.route('/featured-artists', methods=['GET'])
def list_featured_artists():
    """List all featured artists with track counts.

    Returns:
        JSON array of featured artists
    """
    db = _get_db_operations()
    artists = db['get_all_featured_artists']()

    return jsonify({
        'artists': artists,
        'count': len(artists),
    })


# =============================================================================
# File Organization Endpoints
# =============================================================================

@speaker_bp.route('/organize', methods=['POST'])
def organize_files():
    """Organize vocal files by identified artist.

    Request JSON:
        dry_run: If true, only report what would be done (default: true)

    Returns:
        JSON with organization statistics
    """
    data = request.get_json() or {}
    dry_run = data.get('dry_run', True)

    try:
        from ..audio.file_organizer import organize_by_identified_artist

        stats = organize_by_identified_artist(dry_run=dry_run)

        return jsonify({
            'success': True,
            'dry_run': dry_run,
            'stats': stats,
        })

    except Exception as e:
        logger.error(f"File organization failed: {e}")
        return jsonify({'error': str(e)}), 500
