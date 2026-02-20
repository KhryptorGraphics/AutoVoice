"""Flask application factory for AutoVoice."""
import logging
import time
import threading
from typing import Optional, Dict, Any, Tuple

from flask import Flask, request, g
from flask_socketio import SocketIO

logger = logging.getLogger(__name__)


def create_app(config: Optional[Dict[str, Any]] = None, testing: bool = False) -> Tuple[Flask, SocketIO]:
    """Create and configure the Flask application.

    Args:
        config: Optional configuration dictionary
        testing: If True, configure for testing (no ML components)

    Returns:
        Tuple of (Flask app, SocketIO instance)
    """
    app = Flask(__name__)
    app.config['SECRET_KEY'] = config.get('SECRET_KEY', 'autovoice-dev-key') if config else 'autovoice-dev-key'
    app.config['TESTING'] = testing

    if config:
        app.config.update(config)

    # Store app-level config for API access
    app.app_config = config if config else {}

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
        if getattr(g, '_request_tracked', False):
            with app._request_lock:
                app._active_requests = max(0, app._active_requests - 1)
        return response

    @app.errorhandler(Exception)
    def handle_exception(error):
        """Ensure request tracking is decremented on errors."""
        if getattr(g, '_request_tracked', False):
            with app._request_lock:
                app._active_requests = max(0, app._active_requests - 1)
        raise error

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
    async_mode = 'threading' if (testing or app.config.get('TESTING')) else 'eventlet'
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode=async_mode)
    app.socketio = socketio

    # Register API blueprints
    from .api import api_bp
    from .karaoke_api import karaoke_bp
    from .speaker_api import speaker_bp
    from auto_voice.profiles.api import profiles_bp
    from .api_docs import docs_bp, swagger_ui_blueprint
    app.register_blueprint(api_bp)
    app.register_blueprint(karaoke_bp)
    app.register_blueprint(speaker_bp)
    app.register_blueprint(profiles_bp)
    app.register_blueprint(docs_bp, url_prefix='/api/v1')
    app.register_blueprint(swagger_ui_blueprint)

    # Register WebSocket namespaces
    from .karaoke_events import register_karaoke_namespace
    register_karaoke_namespace(socketio)

    # Initialize components (skip in testing mode)
    if not testing:
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
            voice_cloner = VoiceCloner(device=device)
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
                voice_profile_manager=voice_cloner
            )
            app.job_manager = job_manager
            job_manager.start_cleanup_thread()
            logger.info("JobManager initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize JobManager: {e}")
    else:
        logger.info("JobManager skipped (missing pipeline or cloner)")
