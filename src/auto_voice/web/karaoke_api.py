"""Karaoke API endpoints for live voice conversion.

Provides endpoints for song upload, vocal separation, and session management
for the live karaoke voice conversion feature.
"""
import atexit
import functools
import logging
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable

from flask import Blueprint, request, jsonify, current_app, g
from werkzeug.utils import secure_filename

from .persistence import DEFAULT_AUDIO_ROUTER_CONFIG
from .utils import (
    allowed_file,
    ALLOWED_AUDIO_EXTENSIONS,
    validation_error_response,
    not_found_response,
    service_unavailable_response,
    error_response,
)

logger = logging.getLogger(__name__)

karaoke_bp = Blueprint('karaoke', __name__, url_prefix='/api/v1/karaoke')
LIVE_PIPELINES = {'realtime', 'realtime_meanvc'}


# ============================================================================
# Production Logging & Request Tracking (Task 8.1)
# ============================================================================

def _generate_request_id() -> str:
    """Generate unique request ID for tracing."""
    return f"req_{uuid.uuid4().hex[:12]}"


def log_request(f: Callable) -> Callable:
    """Decorator to log API requests with timing and request IDs."""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        request_id = _generate_request_id()
        g.request_id = request_id
        start_time = time.perf_counter()

        # Log request
        logger.info(
            f"[{request_id}] {request.method} {request.path} started",
            extra={
                'request_id': request_id,
                'method': request.method,
                'path': request.path,
                'remote_addr': request.remote_addr,
            }
        )

        try:
            response = f(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Extract status code from response
            if isinstance(response, tuple):
                status_code = response[1] if len(response) > 1 else 200
            else:
                status_code = 200

            logger.info(
                f"[{request_id}] {request.method} {request.path} completed "
                f"status={status_code} time={elapsed_ms:.1f}ms",
                extra={
                    'request_id': request_id,
                    'status_code': status_code,
                    'elapsed_ms': elapsed_ms,
                }
            )
            return response

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"[{request_id}] {request.method} {request.path} failed: {e}",
                extra={
                    'request_id': request_id,
                    'error': str(e),
                    'elapsed_ms': elapsed_ms,
                },
                exc_info=True
            )
            raise

    return wrapper


# ============================================================================
# Health Check Endpoint (Task 8.2)
# ============================================================================

@karaoke_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring.

    Returns:
        HTTP 200: All components healthy
        HTTP 503: One or more components unhealthy

    Example Response:
        {
            "status": "healthy",
            "timestamp": "2026-01-24T19:00:00Z",
            "components": {
                "karaoke_manager": {"status": "healthy"},
                "voice_model_registry": {"status": "healthy", "model_count": 2},
                "storage": {"status": "healthy", "songs": 5, "jobs": 3}
            }
        }
    """
    components = {}
    overall_healthy = True

    # Check KaraokeManager
    karaoke_mgr = getattr(current_app, 'karaoke_manager', None)
    if karaoke_mgr:
        components['karaoke_manager'] = {'status': 'healthy'}
    else:
        components['karaoke_manager'] = {'status': 'unavailable', 'message': 'not initialized'}

    # Check voice model registry
    try:
        registry = _get_voice_model_registry()
        models = registry.list_models()
        components['voice_model_registry'] = {
            'status': 'healthy',
            'model_count': len(models)
        }
    except Exception as e:
        components['voice_model_registry'] = {
            'status': 'unhealthy',
            'error': str(e)
        }
        overall_healthy = False

    # Check storage state
    components['storage'] = {
        'status': 'healthy',
        'songs_uploaded': len(_uploaded_songs),
        'active_jobs': len([j for j in _separation_jobs.values() if j['status'] in ('queued', 'processing')]),
        'completed_jobs': len([j for j in _separation_jobs.values() if j['status'] == 'completed']),
        'active_sessions': len(_active_sessions),
    }

    # Check temp directory
    tmp_dir = '/tmp/autovoice_karaoke'
    try:
        os.makedirs(tmp_dir, exist_ok=True)
        test_file = os.path.join(tmp_dir, '.health_check')
        with open(test_file, 'w') as f:
            f.write('ok')
        os.unlink(test_file)
        components['temp_storage'] = {'status': 'healthy', 'path': tmp_dir}
    except Exception as e:
        components['temp_storage'] = {'status': 'unhealthy', 'error': str(e)}
        overall_healthy = False

    status = 'healthy' if overall_healthy else 'degraded'
    status_code = 200 if overall_healthy else 503

    return jsonify({
        'status': status,
        'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'components': components,
        'version': '1.0.0',
    }), status_code


@karaoke_bp.route('/metrics', methods=['GET'])
def get_metrics():
    """Get usage metrics endpoint for monitoring.

    Returns aggregate, privacy-respecting usage analytics.

    Returns:
        HTTP 200: Usage metrics

    Example Response:
        {
            "total_sessions": 42,
            "total_audio_minutes": 128.5,
            "avg_latency_ms": 23.4,
            "sessions_last_24h": 12
        }
    """
    from .karaoke_events import get_karaoke_analytics
    metrics = get_karaoke_analytics()
    return jsonify(metrics)


# ============================================================================
# Rate Limiting (Task 8.6)
# ============================================================================

# Simple in-memory rate limiter
_rate_limit_store: Dict[str, Dict[str, Any]] = {}


def _get_client_ip() -> str:
    """Get client IP address, handling proxies."""
    # Check X-Forwarded-For header for proxy setups
    forwarded = request.headers.get('X-Forwarded-For', '')
    if forwarded:
        return forwarded.split(',')[0].strip()
    return request.remote_addr or 'unknown'


def rate_limit(max_requests: int, window_seconds: int) -> Callable:
    """Rate limiting decorator.

    Args:
        max_requests: Maximum requests allowed in the window
        window_seconds: Time window in seconds

    Returns:
        Decorator function
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            client_ip = _get_client_ip()
            endpoint = f'{request.method}:{request.path}'
            key = f'{client_ip}:{endpoint}'
            now = time.time()

            # Get or create rate limit entry
            if key not in _rate_limit_store:
                _rate_limit_store[key] = {
                    'count': 0,
                    'window_start': now,
                }

            entry = _rate_limit_store[key]

            # Reset window if expired
            if now - entry['window_start'] > window_seconds:
                entry['count'] = 0
                entry['window_start'] = now

            # Check rate limit
            if entry['count'] >= max_requests:
                remaining = int(entry['window_start'] + window_seconds - now)
                logger.warning(
                    f"Rate limit exceeded for {client_ip} on {endpoint}"
                )
                return error_response(
                    'Rate limit exceeded',
                    status_code=429,
                    retry_after=remaining
                )

            # Increment counter
            entry['count'] += 1

            return f(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# Session Cleanup (Task 8.3)
# ============================================================================

# Track active karaoke sessions for cleanup
_active_sessions: Dict[str, Dict[str, Any]] = {}


def register_session(session_id: str, song_id: str, client_id: str) -> None:
    """Register an active karaoke session."""
    _active_sessions[session_id] = {
        'session_id': session_id,
        'song_id': song_id,
        'client_id': client_id,
        'started_at': time.time(),
        'last_activity': time.time(),
    }
    logger.info(f"Session registered: {session_id} for client {client_id}")


def update_session_activity(session_id: str) -> None:
    """Update last activity timestamp for a session."""
    if session_id in _active_sessions:
        _active_sessions[session_id]['last_activity'] = time.time()


def cleanup_session(session_id: str, reason: str = 'unknown') -> None:
    """Clean up a karaoke session and its resources."""
    session = _active_sessions.pop(session_id, None)
    if session:
        duration = time.time() - session['started_at']
        logger.info(
            f"Session cleaned up: {session_id} reason={reason} "
            f"duration={duration:.1f}s client={session['client_id']}"
        )


def cleanup_stale_sessions(max_idle_seconds: int = 300) -> int:
    """Clean up sessions that have been idle for too long.

    Args:
        max_idle_seconds: Maximum idle time before cleanup

    Returns:
        Number of sessions cleaned up
    """
    now = time.time()
    stale_sessions = [
        sid for sid, sess in _active_sessions.items()
        if now - sess['last_activity'] > max_idle_seconds
    ]

    for session_id in stale_sessions:
        cleanup_session(session_id, reason='idle_timeout')

    if stale_sessions:
        logger.info(f"Cleaned up {len(stale_sessions)} stale sessions")

    return len(stale_sessions)


def cleanup_old_songs(max_age_seconds: int = 3600) -> int:
    """Clean up uploaded songs older than max_age_seconds.

    Args:
        max_age_seconds: Maximum age in seconds

    Returns:
        Number of songs cleaned up
    """
    now = time.time()
    old_songs = [
        sid for sid, song in _uploaded_songs.items()
        if now - song['uploaded_at'] > max_age_seconds
    ]

    for song_id in old_songs:
        song = _uploaded_songs.pop(song_id, None)
        if song:
            # Clean up file
            try:
                if os.path.exists(song['path']):
                    os.unlink(song['path'])
            except OSError as e:
                logger.warning(f"Failed to delete song file: {e}")

            # Clean up associated job
            job_id = song.get('separation_job_id')
            if job_id:
                job = _separation_jobs.pop(job_id, None)
                if job:
                    # Clean up separated files
                    for path_key in ('vocals_path', 'instrumental_path'):
                        path = job.get(path_key)
                        if path and os.path.exists(path):
                            try:
                                os.unlink(path)
                            except OSError:
                                pass

    if old_songs:
        logger.info(f"Cleaned up {len(old_songs)} old songs")

    return len(old_songs)


def _cleanup_on_shutdown() -> None:
    """Clean up all resources on application shutdown."""
    logger.info("Karaoke API shutdown: cleaning up resources")

    # Clean up all sessions
    for session_id in list(_active_sessions.keys()):
        cleanup_session(session_id, reason='shutdown')

    # Clean up temp files
    for song in _uploaded_songs.values():
        try:
            path = song.get('path')
            if path and os.path.exists(path):
                os.unlink(path)
        except OSError:
            pass

    for job in _separation_jobs.values():
        for path_key in ('vocals_path', 'instrumental_path'):
            path = job.get(path_key)
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass

    logger.info("Karaoke API shutdown complete")


# Register shutdown cleanup
atexit.register(_cleanup_on_shutdown)

# In-memory storage for uploaded songs and separation jobs
# In production, this would be Redis or a database
_uploaded_songs: Dict[str, Dict[str, Any]] = {}
_separation_jobs: Dict[str, Dict[str, Any]] = {}

# Configuration
MAX_UPLOAD_SIZE_MB = 100
MAX_SONG_DURATION_SECONDS = 600  # 10 minutes


def _get_audio_duration(file_path: str) -> float:
    """Get audio duration in seconds.

    Args:
        file_path: Path to audio file

    Returns:
        Duration in seconds

    Raises:
        RuntimeError: If duration cannot be determined
    """
    try:
        import torchaudio
        info = torchaudio.info(file_path)
        return info.num_frames / info.sample_rate
    except Exception:
        pass

    try:
        import librosa
        duration = librosa.get_duration(path=file_path)
        return duration
    except Exception:
        pass

    try:
        import soundfile as sf
        info = sf.info(file_path)
        return info.duration
    except Exception:
        pass

    raise RuntimeError(f"Could not determine audio duration for {file_path}")


@karaoke_bp.route('/upload', methods=['POST'])
@log_request
@rate_limit(max_requests=10, window_seconds=60)  # 10 uploads per minute per IP
def upload_song():
    """Upload a song for karaoke processing.

    Request (multipart/form-data):
        song (file): Audio file to upload (required)
            Supported formats: wav, mp3, flac, m4a, ogg, aac

    Returns:
        HTTP 201: JSON with song_id, duration, sample_rate, format
        HTTP 400: Invalid request (no file, empty filename, invalid format)
        HTTP 413: File too large
        HTTP 429: Rate limit exceeded
        HTTP 503: Service unavailable

    Example Response:
        {
            "song_id": "abc123-def456",
            "duration": 180.5,
            "sample_rate": 44100,
            "format": "wav",
            "status": "uploaded"
        }
    """
    # Check for song file
    if 'song' not in request.files:
        return validation_error_response('No song file provided')

    song_file = request.files['song']

    if song_file.filename == '':
        return validation_error_response('No file selected')

    if not allowed_file(song_file.filename):
        return validation_error_response(
            f'Invalid file format. Supported formats: {", ".join(ALLOWED_AUDIO_EXTENSIONS)}'
        )

    # Check file size (approximate, before saving)
    song_file.seek(0, 2)  # Seek to end
    file_size = song_file.tell()
    song_file.seek(0)  # Reset to beginning

    if file_size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        return error_response(
            f'File too large. Maximum size: {MAX_UPLOAD_SIZE_MB}MB',
            status_code=413
        )

    tmp_file = None
    try:
        # Save to temporary file
        secure_name = secure_filename(song_file.filename)
        file_ext = os.path.splitext(secure_name)[1].lower()

        tmp_file = tempfile.NamedTemporaryFile(
            suffix=file_ext,
            delete=False,
            dir='/tmp/autovoice_karaoke'
        )

        # Ensure directory exists
        os.makedirs(os.path.dirname(tmp_file.name), exist_ok=True)

        song_file.save(tmp_file.name)
        tmp_file.close()

        # Get audio info
        try:
            duration = _get_audio_duration(tmp_file.name)
        except RuntimeError as e:
            logger.warning(f"Could not get audio duration: {e}")
            # Allow upload but with unknown duration
            duration = 0.0

        # Check duration limit
        if duration > MAX_SONG_DURATION_SECONDS:
            os.unlink(tmp_file.name)
            return validation_error_response(
                f'Song too long. Maximum duration: {MAX_SONG_DURATION_SECONDS // 60} minutes'
            )

        # Get sample rate if possible
        sample_rate = 44100  # Default
        try:
            import torchaudio
            info = torchaudio.info(tmp_file.name)
            sample_rate = info.sample_rate
        except Exception:
            pass

        # Generate song ID
        song_id = str(uuid.uuid4())

        # Store song metadata
        _uploaded_songs[song_id] = {
            'id': song_id,
            'path': tmp_file.name,
            'original_filename': song_file.filename,
            'format': file_ext.lstrip('.'),
            'duration': duration,
            'sample_rate': sample_rate,
            'file_size': file_size,
            'uploaded_at': time.time(),
            'status': 'uploaded'
        }

        logger.info(f"Song uploaded: {song_id} ({song_file.filename}, {duration:.1f}s)")

        return jsonify({
            'song_id': song_id,
            'duration': duration,
            'sample_rate': sample_rate,
            'format': file_ext.lstrip('.'),
            'status': 'uploaded'
        }), 201

    except Exception as e:
        logger.error(f"Song upload error: {e}", exc_info=True)
        if tmp_file and os.path.exists(tmp_file.name):
            try:
                os.unlink(tmp_file.name)
            except OSError:
                pass
        return service_unavailable_response('Failed to process uploaded file')


@karaoke_bp.route('/songs/<song_id>', methods=['GET'])
def get_song_info(song_id: str):
    """Get information about an uploaded song.

    Args:
        song_id: Unique song identifier

    Returns:
        HTTP 200: Song information
        HTTP 404: Song not found
    """
    song = _uploaded_songs.get(song_id)
    if not song:
        return error_response('Song not found', status_code=404, song_id=song_id)

    # Return public fields only
    return jsonify({
        'song_id': song['id'],
        'duration': song['duration'],
        'sample_rate': song['sample_rate'],
        'format': song['format'],
        'status': song['status'],
        'uploaded_at': song['uploaded_at']
    })


@karaoke_bp.route('/separate', methods=['POST'])
@log_request
@rate_limit(max_requests=5, window_seconds=60)  # 5 separations per minute per IP
def start_separation():
    """Start vocal/instrumental separation for an uploaded song.

    Request (JSON):
        song_id (str): ID of uploaded song (required)

    Returns:
        HTTP 202: Separation job queued
        HTTP 400: Invalid request
        HTTP 404: Song not found
        HTTP 503: Service unavailable

    Example Response:
        {
            "job_id": "xyz789",
            "song_id": "abc123",
            "status": "queued",
            "estimated_time": 30
        }
    """
    # Get song_id from request - handle missing/invalid JSON gracefully
    data = request.get_json(silent=True) or {}
    song_id = data.get('song_id')

    if not song_id:
        return validation_error_response('song_id is required')

    # Verify song exists
    song = _uploaded_songs.get(song_id)
    if not song:
        return error_response('Song not found', status_code=404, song_id=song_id)

    # Check if separator is available
    separator = getattr(current_app, 'vocal_separator', None)
    karaoke_manager = getattr(current_app, 'karaoke_manager', None)

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Estimate separation time based on duration
    estimated_time = max(10, int(song['duration'] / 6))  # ~6x realtime

    # Create job record
    _separation_jobs[job_id] = {
        'job_id': job_id,
        'song_id': song_id,
        'status': 'queued',
        'progress': 0,
        'created_at': time.time(),
        'estimated_time': estimated_time,
        'vocals_path': None,
        'instrumental_path': None,
        'error': None
    }

    # Update song status
    song['status'] = 'separating'
    song['separation_job_id'] = job_id

    # Start async separation via KaraokeManager
    karaoke_mgr = getattr(current_app, 'karaoke_manager', None)
    if karaoke_mgr:
        try:
            karaoke_mgr.start_separation(job_id, song['path'])
        except Exception as e:
            logger.error(f"Failed to start separation: {e}")
            _separation_jobs[job_id]['status'] = 'failed'
            _separation_jobs[job_id]['error'] = str(e)
    else:
        logger.info(f"KaraokeManager not available, job {job_id} queued only")

    logger.info(f"Separation job created: {job_id} for song {song_id}")

    return jsonify({
        'job_id': job_id,
        'song_id': song_id,
        'status': 'queued',
        'estimated_time': estimated_time
    }), 202


@karaoke_bp.route('/separate/<job_id>', methods=['GET'])
def get_separation_status(job_id: str):
    """Get status of a separation job.

    Args:
        job_id: Unique job identifier

    Returns:
        HTTP 200: Job status and progress
        HTTP 404: Job not found

    Example Response (queued/processing):
        {
            "job_id": "xyz789",
            "status": "processing",
            "progress": 45,
            "estimated_remaining": 15
        }

    Example Response (completed):
        {
            "job_id": "xyz789",
            "status": "completed",
            "progress": 100,
            "vocals_ready": true,
            "instrumental_ready": true
        }
    """
    job = _separation_jobs.get(job_id)
    if not job:
        return error_response('Job not found', status_code=404, job_id=job_id)

    # Sync status from KaraokeManager if available
    karaoke_mgr = getattr(current_app, 'karaoke_manager', None)
    if karaoke_mgr:
        mgr_status = karaoke_mgr.get_job_status(job_id)
        if mgr_status:
            job['status'] = mgr_status.get('status', job['status'])
            job['progress'] = mgr_status.get('progress', job['progress'])
            if mgr_status.get('vocals_path'):
                job['vocals_path'] = mgr_status['vocals_path']
            if mgr_status.get('instrumental_path'):
                job['instrumental_path'] = mgr_status['instrumental_path']
            if mgr_status.get('error'):
                job['error'] = mgr_status['error']

    response = {
        'job_id': job['job_id'],
        'song_id': job['song_id'],
        'status': job['status'],
        'progress': job['progress']
    }

    if job['status'] == 'processing':
        elapsed = time.time() - job['created_at']
        remaining = max(0, job['estimated_time'] - elapsed)
        response['estimated_remaining'] = int(remaining)

    elif job['status'] == 'completed':
        response['vocals_ready'] = job['vocals_path'] is not None
        response['instrumental_ready'] = job['instrumental_path'] is not None

    elif job['status'] == 'failed':
        response['error'] = job.get('error', 'Unknown error')

    return jsonify(response)


# Utility functions for managing karaoke state

def get_uploaded_song(song_id: str) -> Optional[Dict[str, Any]]:
    """Get uploaded song by ID (for internal use)."""
    return _uploaded_songs.get(song_id)


def get_separation_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Get separation job by ID (for internal use)."""
    return _separation_jobs.get(job_id)


def update_separation_progress(job_id: str, progress: int, status: str = None):
    """Update separation job progress (for internal use)."""
    job = _separation_jobs.get(job_id)
    if job:
        job['progress'] = progress
        if status:
            job['status'] = status


def complete_separation(job_id: str, vocals_path: str, instrumental_path: str):
    """Mark separation as complete (for internal use)."""
    job = _separation_jobs.get(job_id)
    if job:
        job['status'] = 'completed'
        job['progress'] = 100
        job['vocals_path'] = vocals_path
        job['instrumental_path'] = instrumental_path

        # Update song status
        song = _uploaded_songs.get(job['song_id'])
        if song:
            song['status'] = 'separated'


# Audio device configuration cache.
# The canonical source of truth is AppStateStore; keep this module-level copy in
# sync for older tests and helpers that still patch the karaoke module directly.
_device_config = {
    'speaker_device': None,
    'headphone_device': None,
}


def _load_audio_router_config() -> Dict[str, Any]:
    try:
        state_store = getattr(current_app, 'state_store', None)
    except RuntimeError:
        state_store = None

    if state_store is None:
        config = dict(DEFAULT_AUDIO_ROUTER_CONFIG)
        config.update(_device_config)
        return config

    config = state_store.get_audio_router_config()
    if (
        config.get('speaker_device') is None
        and config.get('headphone_device') is None
        and (
            _device_config.get('speaker_device') is not None
            or _device_config.get('headphone_device') is not None
        )
    ):
        config = state_store.update_audio_router_config(_device_config)
    _device_config['speaker_device'] = config.get('speaker_device')
    _device_config['headphone_device'] = config.get('headphone_device')
    return config


def _persist_audio_router_targets(updates: Dict[str, Any]) -> Dict[str, Any]:
    try:
        state_store = getattr(current_app, 'state_store', None)
    except RuntimeError:
        state_store = None

    if state_store is None:
        _device_config.update({
            'speaker_device': updates.get('speaker_device', _device_config.get('speaker_device')),
            'headphone_device': updates.get('headphone_device', _device_config.get('headphone_device')),
        })
        config = dict(DEFAULT_AUDIO_ROUTER_CONFIG)
        config.update(_device_config)
        return config

    config = state_store.update_audio_router_config(updates)
    _device_config['speaker_device'] = config.get('speaker_device')
    _device_config['headphone_device'] = config.get('headphone_device')
    return config


def load_karaoke_session_snapshot(session_id: str) -> Optional[Dict[str, Any]]:
    try:
        state_store = getattr(current_app, 'state_store', None)
    except RuntimeError:
        state_store = None
    if state_store is None:
        return None
    return state_store.get_karaoke_session(session_id)


def save_karaoke_session_snapshot(snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        state_store = getattr(current_app, 'state_store', None)
    except RuntimeError:
        state_store = None
    if state_store is None:
        return None
    return state_store.save_karaoke_session(snapshot)


@karaoke_bp.route('/devices', methods=['GET'])
def list_audio_devices():
    """List available audio output devices.

    Returns:
        HTTP 200: List of audio devices

    Example Response:
        {
            "devices": [
                {
                    "index": 0,
                    "name": "Built-in Audio",
                    "channels": 2,
                    "default_sample_rate": 48000,
                    "is_default": true
                }
            ]
        }
    """
    from .audio_router import list_audio_devices as get_devices

    devices = get_devices()

    return jsonify({
        'devices': devices,
        'count': len(devices)
    })


@karaoke_bp.route('/devices/output', methods=['GET'])
def get_output_device_config():
    """Get current output device configuration.

    Returns:
        HTTP 200: Current device config

    Example Response:
        {
            "speaker_device": 0,
            "headphone_device": 1
        }
    """
    config = _load_audio_router_config()
    return jsonify({
        'speaker_device': config.get('speaker_device'),
        'headphone_device': config.get('headphone_device'),
    })


@karaoke_bp.route('/preflight', methods=['POST'])
def karaoke_preflight():
    """Validate the current karaoke session inputs before starting live conversion."""
    data = request.get_json(silent=True) or {}
    song_id = data.get('song_id')
    profile_id = data.get('profile_id')
    voice_model_id = data.get('voice_model_id')
    pipeline_type = str(data.get('pipeline_type', 'realtime')).strip().lower()

    audio_router_config = _load_audio_router_config()
    issues = []
    warnings = []

    song = _uploaded_songs.get(song_id) if song_id else None
    song_ready = bool(song)
    if not song_ready:
        issues.append('Upload a song before starting karaoke.')

    separation_job = None
    if song:
        separation_job = _separation_jobs.get(song.get('separation_job_id', ''))
    assets_ready = bool(
        separation_job
        and separation_job.get('status') == 'completed'
        and separation_job.get('vocals_path')
        and separation_job.get('instrumental_path')
    )
    if song and not assets_ready:
        issues.append('Vocals and instrumental stems are not ready yet.')

    pipeline_valid = pipeline_type in LIVE_PIPELINES
    if not pipeline_valid:
        issues.append('Live karaoke only supports the realtime or realtime_meanvc pipelines.')

    profile_ready = True
    active_model_type = None
    if profile_id:
        voice_cloner = getattr(current_app, 'voice_cloner', None)
        profile_store = getattr(voice_cloner, 'store', None)
        profile = profile_store.load(profile_id) if profile_store is not None else None
        profile_ready = bool(profile and profile.get('has_trained_model'))
        active_model_type = profile.get('active_model_type') if profile else None
        if not profile_ready:
            issues.append('Selected voice profile is not trained and ready for live conversion.')

    voice_model_ready = True
    if voice_model_id:
        registry = _get_voice_model_registry()
        voice_model_ready = registry.get_model(voice_model_id) is not None
        if not voice_model_ready:
            issues.append('Selected voice model is unavailable.')

    routing_ready = (
        audio_router_config.get('speaker_device') is not None
        or audio_router_config.get('headphone_device') is not None
    )
    if not routing_ready:
        warnings.append('No explicit output devices selected; system defaults will be used.')

    return jsonify({
        'ok': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'checks': {
            'song_ready': song_ready,
            'assets_ready': assets_ready,
            'pipeline_valid': pipeline_valid,
            'profile_ready': profile_ready,
            'voice_model_ready': voice_model_ready,
            'routing_ready': routing_ready,
        },
        'requested_pipeline': pipeline_type,
        'active_model_type': active_model_type,
        'audio_router_targets': {
            'speaker_device': audio_router_config.get('speaker_device'),
            'headphone_device': audio_router_config.get('headphone_device'),
        },
    })


@karaoke_bp.route('/sessions/<session_id>', methods=['GET'])
def get_karaoke_session_snapshot(session_id: str):
    """Return the latest durable karaoke session snapshot for recovery/debugging."""
    snapshot = load_karaoke_session_snapshot(session_id)
    if not snapshot:
        return error_response('Session not found', status_code=404, session_id=session_id)
    return jsonify(snapshot)


@karaoke_bp.route('/devices/output', methods=['POST'])
def set_output_device_config():
    """Set output device configuration.

    Request (JSON):
        speaker_device (int, optional): Device index for speaker output
        headphone_device (int, optional): Device index for headphone output

    Returns:
        HTTP 200: Configuration updated
        HTTP 400: Invalid device index

    Example Request:
        {
            "speaker_device": 0,
            "headphone_device": 1
        }
    """
    from .audio_router import list_audio_devices as get_devices

    data = request.get_json(silent=True) or {}

    # Get available devices for validation
    devices = get_devices()

    def _normalize_device_index(value):
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("boolean is not a valid device index")
        return int(value)

    device_indices = set()
    for device in devices:
        raw_index = device.get('index', device.get('device_id'))
        if raw_index is None:
            continue
        try:
            device_indices.add(_normalize_device_index(raw_index))
        except (TypeError, ValueError):
            logger.debug("Skipping device with invalid index payload: %s", device)

    persisted_updates = {}

    # Validate and set speaker device
    if 'speaker_device' in data:
        try:
            speaker_idx = _normalize_device_index(data['speaker_device'])
        except (TypeError, ValueError):
            return error_response(
                f"Invalid speaker device index: {data['speaker_device']}",
                status_code=400,
                available_devices=sorted(device_indices),
            )
        if speaker_idx is not None and speaker_idx not in device_indices:
            return error_response(
                f'Invalid speaker device index: {speaker_idx}',
                status_code=400,
                available_devices=sorted(device_indices)
            )
        persisted_updates['speaker_device'] = speaker_idx

    # Validate and set headphone device
    if 'headphone_device' in data:
        try:
            headphone_idx = _normalize_device_index(data['headphone_device'])
        except (TypeError, ValueError):
            return error_response(
                f"Invalid headphone device index: {data['headphone_device']}",
                status_code=400,
                available_devices=sorted(device_indices),
            )
        if headphone_idx is not None and headphone_idx not in device_indices:
            return error_response(
                f'Invalid headphone device index: {headphone_idx}',
                status_code=400,
                available_devices=sorted(device_indices)
            )
        persisted_updates['headphone_device'] = headphone_idx

    config = _persist_audio_router_targets(persisted_updates)

    logger.info(
        f"Device config updated: speaker={config['speaker_device']}, "
        f"headphone={config['headphone_device']}"
    )

    return jsonify({
        'speaker_device': config['speaker_device'],
        'headphone_device': config['headphone_device'],
        'status': 'updated'
    })


# Voice model registry (singleton for the blueprint)
_voice_model_registry = None


def reset_test_state() -> None:
    """Reset in-memory karaoke API state for isolated test app instances."""
    global _voice_model_registry

    _uploaded_songs.clear()
    _separation_jobs.clear()
    _active_sessions.clear()
    _rate_limit_store.clear()
    _voice_model_registry = None
    _device_config['speaker_device'] = None
    _device_config['headphone_device'] = None


def _get_voice_model_registry():
    """Get or create the voice model registry singleton."""
    global _voice_model_registry
    if _voice_model_registry is None:
        from .voice_model_registry import VoiceModelRegistry
        _voice_model_registry = VoiceModelRegistry()
    return _voice_model_registry


@karaoke_bp.route('/voice-models', methods=['GET'])
def list_voice_models():
    """List available voice models for conversion.

    Returns:
        HTTP 200: List of voice models

    Example Response:
        {
            "models": [
                {
                    "id": "artist_xyz",
                    "name": "Artist XYZ",
                    "type": "pretrained"
                }
            ],
            "count": 1
        }
    """
    registry = _get_voice_model_registry()
    models = registry.list_models()

    return jsonify({
        'models': models,
        'count': len(models)
    })


@karaoke_bp.route('/voice-models/<model_id>', methods=['GET'])
def get_voice_model(model_id: str):
    """Get voice model details by ID.

    Args:
        model_id: Unique model identifier

    Returns:
        HTTP 200: Model details
        HTTP 404: Model not found
    """
    registry = _get_voice_model_registry()
    model = registry.get_model(model_id)

    if not model:
        return error_response('Voice model not found', status_code=404, model_id=model_id)

    return jsonify(model)


@karaoke_bp.route('/voice-models/extract', methods=['POST'])
@log_request
@rate_limit(max_requests=10, window_seconds=60)  # 10 extractions per minute per IP
def extract_voice_model():
    """Extract voice model from uploaded song vocals.

    Request (JSON):
        song_id (str): ID of uploaded song with separated vocals
        name (str): Display name for the extracted model

    Returns:
        HTTP 201: Model created
        HTTP 400: Invalid request
        HTTP 404: Song not found or not separated
    """
    data = request.get_json(silent=True) or {}

    song_id = data.get('song_id')
    name = data.get('name')

    if not song_id:
        return validation_error_response('song_id is required')

    if not name:
        return validation_error_response('name is required')

    # Get song info
    song = _uploaded_songs.get(song_id)
    if not song:
        return error_response('Song not found', status_code=404, song_id=song_id)

    # Check if song has been separated
    job_id = song.get('separation_job_id')
    if not job_id:
        return error_response(
            'Song has not been separated',
            status_code=400,
            song_id=song_id
        )

    job = _separation_jobs.get(job_id)
    if not job or job['status'] != 'completed':
        return error_response(
            'Separation not complete',
            status_code=400,
            song_id=song_id
        )

    vocals_path = job.get('vocals_path')
    if not vocals_path:
        return error_response(
            'Vocals not available',
            status_code=400,
            song_id=song_id
        )

    try:
        import torch
        import soundfile as sf
        from .voice_model_registry import extract_speaker_embedding

        # Load vocals using soundfile (more reliable than torchaudio)
        audio_np, sr = sf.read(vocals_path)
        audio = torch.from_numpy(audio_np).float()
        if audio.dim() > 1:
            audio = audio.mean(dim=1)  # stereo to mono

        # Extract embedding
        embedding = extract_speaker_embedding(audio, sample_rate=sr)

        # Register model
        registry = _get_voice_model_registry()
        model_id = registry.register_extracted_model(
            name=name,
            embedding=embedding,
            source_song_id=song_id
        )

        logger.info(f"Extracted voice model {model_id} from song {song_id}")

        return jsonify({
            'model_id': model_id,
            'name': name,
            'type': 'extracted',
            'source_song_id': song_id,
            'status': 'created'
        }), 201

    except Exception as e:
        logger.error(f"Failed to extract voice model: {e}", exc_info=True)
        return service_unavailable_response('Failed to extract voice model')
