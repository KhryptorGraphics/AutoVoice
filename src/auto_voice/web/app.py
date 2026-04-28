"""Flask application factory for AutoVoice."""
import logging
import os
import secrets
import tempfile
import threading
import time
from typing import Optional, Dict, Any, Tuple

from flask import Flask, request, g
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.exceptions import HTTPException

from auto_voice.config.secrets import SecretsManager
from auto_voice.web.persistence import AppStateStore
from auto_voice.storage.paths import resolve_profiles_dir, resolve_samples_dir

logger = logging.getLogger(__name__)


def _release_tracked_request(app: Flask) -> None:
    """Decrement active request count exactly once for a tracked request."""
    if not getattr(g, '_request_tracked', False):
        return

    with app._request_lock:
        app._active_requests = max(0, app._active_requests - 1)
    g._request_tracked = False


def register_default_socket_handlers(socketio: SocketIO) -> None:
    """Register default-namespace room subscription handlers.

    The frontend already relies on `join_job` / `leave_job` acknowledgements for
    conversion job rooms. Keep this contract explicit in the backend so the
    default namespace remains the canonical path for non-karaoke realtime
    updates.
    """

    @socketio.on('join_job')
    def handle_join_job(payload: Optional[Dict[str, Any]]) -> None:
        job_id = str((payload or {}).get('job_id') or '').strip()
        if not job_id:
            emit('job_subscription_error', {'message': 'job_id is required'})
            return

        join_room(job_id)
        emit('joined_job', {'job_id': job_id})

    @socketio.on('leave_job')
    def handle_leave_job(payload: Optional[Dict[str, Any]]) -> None:
        job_id = str((payload or {}).get('job_id') or '').strip()
        if not job_id:
            emit('job_subscription_error', {'message': 'job_id is required'})
            return

        leave_room(job_id)
        emit('left_job', {'job_id': job_id})


def create_app(config: Optional[Dict[str, Any]] = None, testing: Optional[bool] = None) -> Tuple[Flask, SocketIO]:
    """Create and configure the Flask application.

    Args:
        config: Optional configuration dictionary
        testing: If True, configure for testing (no ML components).
                 If None (default), infer from config['TESTING'].
                 If False, always use production mode.

    Returns:
        Tuple of (Flask app, SocketIO instance)
    """
    app = Flask(__name__)
    app.start_time = time.time()

    # Apply user config first so we can check TESTING flag
    if config:
        app.config.update(config)

    if 'DATA_DIR' not in app.config:
        if os.environ.get('DATA_DIR'):
            app.config['DATA_DIR'] = os.environ['DATA_DIR']
        elif app.config.get('TESTING') or testing is True:
            app.config['DATA_DIR'] = tempfile.mkdtemp(prefix='autovoice-test-data-')
        else:
            app.config['DATA_DIR'] = 'data'

    # Determine testing mode for SECRET_KEY:
    # - If testing parameter explicitly set, use it (for SECRET_KEY behavior)
    # - Otherwise, infer from config TESTING flag (default False)
    # Note: We DON'T overwrite app.config['TESTING'] if it was set in config,
    # because tests may need TESTING=True for SocketIO threading mode
    # while testing=False for production SECRET_KEY behavior
    if testing is not None:
        is_testing = testing
        # Only set TESTING in config if it wasn't already set
        if 'TESTING' not in app.config:
            app.config['TESTING'] = testing
    else:
        is_testing = app.config.get('TESTING', False)

    # Configure SECRET_KEY using SecretsManager
    if is_testing:
        # Generate secure random key for testing (32 bytes = 64 hex chars)
        # Don't overwrite if SECRET_KEY already in config
        if not app.config.get('SECRET_KEY'):
            app.config['SECRET_KEY'] = secrets.token_hex(32)
    else:
        # Use SecretsManager for production - requires AUTOVOICE_SECRET_FLASK_SECRET_KEY env var
        secrets_manager = SecretsManager()
        if app.config.get('SECRET_KEY'):
            # Config already has SECRET_KEY (e.g., from YAML or test), keep it
            pass
        elif os.environ.get('SECRET_KEY'):
            app.config['SECRET_KEY'] = os.environ['SECRET_KEY']
        else:
            # Require secret from environment or secrets file
            app.config['SECRET_KEY'] = secrets_manager.get_required('flask_secret_key')

    # Store app-level config for API access
    app.app_config = config if config else {}
    app.state_store = AppStateStore(app.config['DATA_DIR'])

    # Request tracking for graceful shutdown
    app._active_requests = 0
    app._shutdown_event = threading.Event()
    app._request_lock = threading.Lock()

    @app.before_request
    def track_request_start():
        """Track active requests for graceful shutdown."""
        if app._shutdown_event.is_set():
            from flask import jsonify
            return jsonify({'error': 'Server is shutting down'}), 503

        with app._request_lock:
            app._active_requests += 1
            g._request_tracked = True

    @app.after_request
    def track_request_end(response):
        """Decrement active request counter."""
        _release_tracked_request(app)
        return response

    @app.errorhandler(HTTPException)
    def handle_http_exception(error: HTTPException):
        """Return JSON for uncaught HTTP errors while preserving status codes."""
        _release_tracked_request(app)
        payload = {
            'error': error.name,
            'message': error.description,
            'status_code': error.code,
        }
        return payload, error.code

    @app.errorhandler(Exception)
    def handle_exception(error):
        """Ensure request tracking is decremented and surface JSON 500s."""
        _release_tracked_request(app)
        logger.exception("Unhandled application error")
        return {
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred',
            'status_code': 500,
        }, 500

    def wait_for_requests(timeout=30.0):
        """Wait for active requests to complete during shutdown.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if all requests completed, False if timeout
        """
        app._shutdown_event.set()
        logger.info(f"Waiting for {app._active_requests} active requests to complete...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            with app._request_lock:
                if app._active_requests == 0:
                    logger.info("All requests completed")
                    return True
            time.sleep(0.1)

        with app._request_lock:
            remaining = app._active_requests
        logger.warning(f"Shutdown timeout reached with {remaining} requests still active")
        return False

    app.wait_for_requests = wait_for_requests

    # Initialize SocketIO
    # Use threading mode in testing to avoid eventlet dependency
    # Check both testing parameter and config TESTING flag for SocketIO mode
    async_mode = 'threading' if (testing is True or app.config.get('TESTING', False)) else 'eventlet'
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode=async_mode)
    app.socketio = socketio

    # Register API blueprints
    from .api import api_bp, health_check as api_health_check, readiness_check as api_readiness_check
    from .karaoke_api import karaoke_bp, reset_test_state as reset_karaoke_test_state
    from .speaker_api import speaker_bp
    from .training_ui import training_ui_bp
    from auto_voice.profiles.api import profiles_bp
    from .api_docs import docs_bp, swagger_ui_blueprint

    if app.config.get('TESTING', False):
        reset_karaoke_test_state()
        try:
            from auto_voice.inference.pipeline_factory import PipelineFactory

            PipelineFactory.reset_instance()
        except Exception:
            logger.debug("Unable to reset pipeline factory test state", exc_info=True)

    app.register_blueprint(api_bp)
    app.register_blueprint(karaoke_bp)
    app.register_blueprint(speaker_bp)
    app.register_blueprint(training_ui_bp)
    app.register_blueprint(profiles_bp)
    app.register_blueprint(docs_bp, url_prefix='/api/v1')
    app.register_blueprint(swagger_ui_blueprint)
    app.add_url_rule('/health', 'health_check_alias', api_health_check)
    app.add_url_rule('/ready', 'readiness_check_alias', api_readiness_check)

    # Register WebSocket namespaces
    from .karaoke_events import register_karaoke_namespace
    register_default_socket_handlers(socketio)
    register_karaoke_namespace(socketio)

    # Initialize components (skip in testing mode)
    # Only skip if testing parameter is explicitly True
    # (maintains backward compatibility: if testing not provided, initialize components)
    if testing is not True:
        _init_components(app, socketio, config)

    logger.info("AutoVoice Flask app created")
    return app, socketio


def _init_components(app: Flask, socketio: SocketIO, config: Optional[Dict]):
    """Initialize ML components if enabled."""
    try:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    except ImportError:
        logger.warning("PyTorch not available, ML components disabled")
        return

    # Initialize KaraokeManager for separation jobs
    if not config or config.get('karaoke_enabled', True):
        try:
            from .karaoke_manager import KaraokeManager

            def progress_callback(job_id: str, progress: int, status: str):
                socketio.emit('separation_progress', {
                    'job_id': job_id,
                    'progress': progress,
                    'status': status
                }, namespace='/karaoke')

            app.karaoke_manager = KaraokeManager(
                device=device,
                progress_callback=progress_callback
            )
            logger.info(f"KaraokeManager initialized on {device}")
        except Exception as e:
            logger.warning(f"Failed to initialize KaraokeManager: {e}")

    # Initialize singing conversion pipeline
    singing_pipeline = None
    if not config or config.get('singing_conversion_enabled', True):
        try:
            from ..inference.singing_conversion_pipeline import SingingConversionPipeline
            singing_pipeline = SingingConversionPipeline(device=device, config=config)
            app.singing_conversion_pipeline = singing_pipeline
            logger.info(f"Singing conversion pipeline initialized on {device}")
        except Exception as e:
            logger.warning(f"Failed to initialize singing conversion pipeline: {e}")

    # Initialize voice cloner
    voice_cloner = None
    if not config or config.get('voice_cloning_enabled', True):
        try:
            from ..inference.voice_cloner import VoiceCloner
            voice_cloner = VoiceCloner(
                device=device,
                profiles_dir=str(resolve_profiles_dir(data_dir=app.config['DATA_DIR'])),
                samples_dir=str(resolve_samples_dir(data_dir=app.config['DATA_DIR'])),
                speaker_encoder_backend=(
                    (config or {}).get('speaker_encoder_backend', 'mel_stats')
                ),
            )
            app.voice_cloner = voice_cloner
            logger.info(f"Voice cloner initialized on {device}")
        except Exception as e:
            logger.warning(f"Failed to initialize voice cloner: {e}")

    # Initialize JobManager if components available
    if singing_pipeline and voice_cloner:
        try:
            from .job_manager import JobManager
            jm_config = config.get('job_manager', {}) if config else {}
            jm_config.setdefault('max_workers', 4)
            jm_config.setdefault('ttl_seconds', 3600)
            jm_config.setdefault('in_progress_ttl_seconds', 7200)
            jm_config['audio'] = app.config

            job_manager = JobManager(
                config=jm_config,
                socketio=socketio,
                singing_pipeline=singing_pipeline,
                voice_profile_manager=voice_cloner,
                state_store=app.state_store,
            )
            app.job_manager = job_manager
            job_manager.start_cleanup_thread()
            logger.info("JobManager initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize JobManager: {e}")
    else:
        logger.info("JobManager skipped (missing pipeline or cloner)")
