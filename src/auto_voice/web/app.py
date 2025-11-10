"""Web application for AutoVoice"""

import os
import logging
import tempfile
from pathlib import Path
from io import BytesIO
import base64

try:
    from flask import Flask, render_template, request, jsonify, send_file
    from flask_socketio import SocketIO
    from werkzeug.utils import secure_filename
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import core modules
from ..utils.config_loader import load_config
from ..utils.logging_config import setup_logging
from ..monitoring.metrics import get_metrics_collector, start_gpu_metrics_collection, set_model_loaded
from ..gpu.gpu_manager import GPUManager
from ..audio.processor import AudioProcessor
from ..models.voice_model import VoiceModel
from ..inference.synthesizer import VoiceSynthesizer
from ..inference.voice_cloner import VoiceCloner
from ..inference.singing_conversion_pipeline import SingingConversionPipeline
from .api import api_bp
from .websocket_handler import WebSocketHandler
from .utils import allowed_file, ALLOWED_AUDIO_EXTENSIONS

# Initialize structured logging
setup_logging()
logger = logging.getLogger(__name__)

# Global instances
app = None
socketio = None
gpu_manager = None
audio_processor = None
voice_model = None
synthesizer = None
voice_cloner = None
singing_conversion_pipeline = None
config = None

UPLOAD_FOLDER = '/tmp/autovoice_uploads'
ALLOWED_EXTENSIONS = ALLOWED_AUDIO_EXTENSIONS  # Use shared constant

def create_app(config_path=None, config=None):
    """Create and configure Flask application"""
    global app, socketio, gpu_manager, audio_processor, voice_model, synthesizer, voice_cloner, singing_conversion_pipeline

    if not FLASK_AVAILABLE:
        logger.warning("Flask not installed. Web interface not available.")
        return None, None

    app = Flask(__name__,
                template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
                static_folder=os.path.join(os.path.dirname(__file__), 'static'))

    # Apply configuration if provided (for testing)
    if config:
        app.config.update(config)

    # Enable CORS if available
    if CORS_AVAILABLE:
        CORS(app)

    # Initialize SocketIO with secure CORS configuration
    allowed_origins = os.getenv('CORS_ALLOWED_ORIGINS', '*').split(',')
    if '*' in allowed_origins and os.getenv('FLASK_ENV') == 'production':
        logger.warning("CORS wildcard (*) should not be used in production")
    socketio = SocketIO(app, cors_allowed_origins=allowed_origins)

    # Load configuration (skip if testing)
    app_config = config if config else {}
    if not config:
        try:
            app_config = load_config(config_path) if config_path else load_config()
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            app_config = {}

    # Configure upload folder
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = app_config.get('web', {}).get('max_content_length', 100 * 1024 * 1024)
    
    # Set additional config defaults expected by tests
    app.config['JSON_SORT_KEYS'] = False

    # Initialize components (skip complex initialization for testing)
    if app.config.get('TESTING'):
        # Mock components for testing
        logger.info("Setting up test environment...")
        gpu_manager = type('MockGPUManager', (), {
            'is_cuda_available': lambda: False,
            'get_device_count': lambda: 0
        })()
        
        audio_processor = type('MockAudioProcessor', (), {
            'extract_pitch': lambda self, audio: np.zeros(100) if NUMPY_AVAILABLE else [0] * 100,
            'voice_activity_detection': lambda self, audio: np.ones(100) if NUMPY_AVAILABLE else [1] * 100,
            'compute_spectrogram': lambda self, audio: np.random.rand(80, 100) if NUMPY_AVAILABLE else [[0] * 100] * 80
        })()
        
        voice_model = type('MockVoiceModel', (), {
            'is_loaded': lambda: True
        })()
        
        synthesizer = type('MockSynthesizer', (), {
            'synthesize_speech': lambda self, text, speaker_id=0: np.random.rand(22050) if NUMPY_AVAILABLE else [0] * 22050,
            'text_to_speech': lambda self, text, speaker_id=0, **kwargs: np.random.rand(22050) if NUMPY_AVAILABLE else [0] * 22050
        })()

        # COMMENT 8 FIX: Implement in-memory profile store for MockVoiceCloner
        class MockVoiceCloner:
            def __init__(self):
                self._profiles = {}  # In-memory profile store keyed by profile_id

            def create_voice_profile(self, audio, **kwargs):
                import uuid
                import datetime
                profile_id = kwargs.get('profile_id', f'test-profile-{uuid.uuid4().hex[:8]}')
                profile = {
                    'profile_id': profile_id,
                    'user_id': kwargs.get('user_id'),
                    'audio_duration': 30.0,
                    'vocal_range': {'min_f0': 100.0, 'max_f0': 400.0},
                    'created_at': datetime.datetime.utcnow().isoformat() + 'Z'
                }
                self._profiles[profile_id] = profile
                return profile

            def list_voice_profiles(self, user_id=None):
                if user_id is None:
                    return list(self._profiles.values())
                return [p for p in self._profiles.values() if p.get('user_id') == user_id]

            def load_voice_profile(self, profile_id):
                """Load voice profile by ID

                COMMENT 2 FIX: Return None for missing profiles instead of creating fake profile.
                API route will detect None and return 404.
                """
                return self._profiles.get(profile_id)

            def delete_voice_profile(self, profile_id):
                if profile_id in self._profiles:
                    del self._profiles[profile_id]
                    return True
                return False

        voice_cloner = MockVoiceCloner()

        singing_conversion_pipeline = type('MockSingingConversionPipeline', (), {
            'convert_song': lambda self, song_path, target_profile_id, **kwargs: {
                'mixed_audio': np.random.rand(44100) if NUMPY_AVAILABLE else [0] * 44100,
                'sample_rate': 44100,
                'duration': 1.0,
                'metadata': {
                    'target_profile_id': target_profile_id,
                    'vocal_volume': kwargs.get('vocal_volume', 1.0),
                    'instrumental_volume': kwargs.get('instrumental_volume', 0.9),
                    'f0_stats': {'min_f0': 100.0, 'max_f0': 400.0}
                }
            },
            'convert_vocals_only': lambda self, vocals_path, target_profile_id, **kwargs: {
                'vocals': np.random.rand(44100) if NUMPY_AVAILABLE else [0] * 44100,
                'sample_rate': 44100
            },
            'clear_cache': lambda self: None,
            'get_cache_info': lambda self: {'total_size_mb': 0.0, 'num_conversions': 0}
        })()
    else:
        # Initialize real components
        try:
            logger.info("Initializing GPU Manager...")
            gpu_manager = GPUManager(app_config.get('gpu', {}))

            logger.info("Initializing Audio Processor...")
            audio_processor = AudioProcessor(app_config.get('audio', {}))

            logger.info("Loading Voice Model...")
            voice_model = VoiceModel(app_config.get('model', {}))

            logger.info("Initializing Synthesizer...")
            synthesizer = VoiceSynthesizer(voice_model, audio_processor, gpu_manager)

            logger.info("Initializing Voice Cloner...")
            # Merge voice cloning config with audio config
            vc_config = {**app_config.get('voice_cloning', {}), 'audio_config': app_config.get('audio', {})}
            # COMMENT 3 FIX: Pass existing audio_processor to avoid config drift
            voice_cloner = VoiceCloner(
                config=vc_config,
                device=gpu_manager.get_device() if hasattr(gpu_manager, 'get_device') else None,
                gpu_manager=gpu_manager,
                audio_processor=audio_processor
            )

            logger.info("Initializing Singing Conversion Pipeline...")
            # Merge pipeline config with model config and add storage_dir from voice_cloning config
            pipeline_config = {
                **app_config.get('singing_conversion_pipeline', {}),
                'model_config': app_config.get('singing_voice_converter', {}),
                'mixer_config': app_config.get('audio_mixing', {}),
                'storage_dir': app_config.get('voice_cloning', {}).get('storage_dir', '~/.cache/autovoice/voice_profiles/')
            }
            singing_conversion_pipeline = SingingConversionPipeline(
                config=pipeline_config,
                device=gpu_manager.get_device() if hasattr(gpu_manager, 'get_device') else None,
                gpu_manager=gpu_manager,
                voice_cloner=voice_cloner  # Pass VoiceCloner instance
            )

            # Update metrics
            set_model_loaded(voice_model.is_loaded() if hasattr(voice_model, 'is_loaded') else True)

            # Start GPU metrics collection if enabled
            if os.getenv('PROMETHEUS_ENABLED', 'true').lower() == 'true':
                logger.info("Starting GPU metrics collection...")
                start_gpu_metrics_collection(interval=10)
        except Exception as e:
            logger.warning(f"Failed to initialize components: {e}, using mock components")
            # Fallback to mock components
            gpu_manager = type('MockGPUManager', (), {'is_cuda_available': lambda: False})()
            audio_processor = type('MockAudioProcessor', (), {})()
            voice_model = type('MockVoiceModel', (), {'is_loaded': lambda: False})()
            synthesizer = type('MockSynthesizer', (), {})()
            voice_cloner = type('MockVoiceCloner', (), {})()
            singing_conversion_pipeline = type('MockSingingConversionPipeline', (), {})()

    # Set app context attributes for blueprints
    app.app_config = app_config
    app.audio_processor = audio_processor
    app.inference_engine = synthesizer  # API expects this name
    app.gpu_manager = gpu_manager
    app.voice_model = voice_model
    app.voice_cloner = voice_cloner
    app.singing_conversion_pipeline = singing_conversion_pipeline

    # Register API blueprint
    app.register_blueprint(api_bp)

    # COMMENT 1 FIX: Add backward-compatible redirect routes from /api/* to /api/v1/*
    # This maintains compatibility for existing consumers while supporting versioned API
    @app.route('/api/synthesize', methods=['POST'])
    def api_synthesize_redirect():
        """Redirect /api/synthesize to /api/v1/synthesize for backward compatibility"""
        from flask import redirect, url_for
        return redirect(url_for('api.synthesize_voice'), code=307)  # 307 preserves POST method

    @app.route('/api/process_audio', methods=['POST'])
    def api_process_audio_redirect():
        """Redirect /api/process_audio to /api/v1/process_audio for backward compatibility"""
        from flask import redirect, url_for
        return redirect(url_for('api.process_audio'), code=307)

    @app.route('/api/analyze', methods=['POST'])
    def api_analyze_redirect():
        """Redirect /api/analyze to /api/v1/analyze for backward compatibility"""
        from flask import redirect, url_for
        return redirect(url_for('api.analyze_audio'), code=307)

    @app.route('/api/speakers', methods=['GET'])
    def api_speakers_redirect():
        """Redirect /api/speakers to /api/v1/speakers for backward compatibility"""
        from flask import redirect, url_for
        return redirect(url_for('api.get_speakers'), code=307)

    @app.route('/api/convert/song', methods=['POST'])
    def api_convert_song_redirect():
        """Redirect /api/convert/song to /api/v1/convert/song for backward compatibility"""
        from flask import redirect, url_for
        return redirect(url_for('api.convert_song'), code=307)

    @app.route('/api/voice/clone', methods=['POST'])
    def api_voice_clone_redirect():
        """Redirect /api/voice/clone to /api/v1/voice/clone for backward compatibility"""
        from flask import redirect, url_for
        return redirect(url_for('api.clone_voice'), code=307)

    @app.route('/api/voice/profiles', methods=['GET'])
    def api_voice_profiles_redirect():
        """Redirect /api/voice/profiles to /api/v1/voice/profiles for backward compatibility"""
        from flask import redirect, url_for
        return redirect(url_for('api.get_voice_profiles'), code=307)

    @app.route('/api/gpu_status', methods=['GET'])
    def api_gpu_status_redirect():
        """Redirect /api/gpu_status to /api/v1/gpu_status for backward compatibility"""
        from flask import redirect, url_for
        return redirect(url_for('api.get_gpu_status'), code=307)

    # Initialize WebSocket handlers
    try:
        websocket_handler = WebSocketHandler(socketio)
    except Exception as e:
        logger.warning(f"Failed to initialize WebSocket handler: {e}")

    # Add request logging middleware
    @app.before_request
    def log_request():
        """Log incoming requests."""
        logger.debug(
            "Incoming request",
            extra={
                "method": request.method,
                "path": request.path,
                "remote_addr": request.remote_addr,
                "user_agent": request.user_agent.string
            }
        )

    @app.after_request
    def log_response(response):
        """Log outgoing responses."""
        logger.debug(
            "Outgoing response",
            extra={
                "method": request.method,
                "path": request.path,
                "status": response.status_code,
                "content_length": response.content_length
            }
        )
        return response

    # Verify URL map for duplicate routes
    if app.debug:
        routes = {}
        for rule in app.url_map.iter_rules():
            route_key = f"{rule.rule} [{','.join(rule.methods)}]"
            if route_key in routes:
                logger.warning(f"Duplicate route detected: {route_key}")
            routes[route_key] = str(rule.endpoint)
        logger.info(f"Registered {len(routes)} unique routes")

    # Routes
    @app.route('/')
    def index():
        """Serve main page or API info"""
        # Return JSON for API-only mode
        if request.headers.get('Accept') == 'application/json':
            return jsonify({
                'message': 'AutoVoice service running',
                'status': 'healthy',
                'components': {
                    'gpu_available': gpu_manager.is_cuda_available() if gpu_manager else False,
                    'model_loaded': voice_model.is_loaded() if voice_model else False,
                    'api': True
                }
            })
        # Otherwise return HTML template if it exists
        template_path = os.path.join(os.path.dirname(__file__), 'templates', 'index.html')
        if os.path.exists(template_path):
            return render_template('index.html')
        # Fallback to JSON
        return jsonify({
            'message': 'AutoVoice service running',
            'status': 'healthy',
            'components': {
                'gpu_available': gpu_manager.is_cuda_available() if gpu_manager else False,
                'model_loaded': voice_model.is_loaded() if voice_model else False,
                'api': True
            }
        })

    @app.route('/song-conversion')
    def song_conversion():
        """Render the song conversion page."""
        return render_template('song_conversion.html')

    @app.route('/profiles')
    def profile_management():
        """Render the profile management page."""
        return render_template('profile_management.html')

    @app.route('/health')
    def health_check():
        """Health check endpoint"""
        # Check if psutil is available for system metrics
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent(interval=0.1)
        except ImportError:
            memory_percent = None
            cpu_percent = None

        # Get GPU status
        gpu_status = {}
        if gpu_manager:
            try:
                gpu_status = {
                    'available': gpu_manager.is_cuda_available(),
                    'device_count': gpu_manager.get_device_count() if hasattr(gpu_manager, 'get_device_count') else 0
                }
            except Exception as e:
                logger.warning(f"Failed to get GPU status: {e}")

        health_status = {
            'status': 'healthy',
            'components': {
                'gpu_available': gpu_status.get('available', False),
                'model_loaded': voice_model.is_loaded() if voice_model and hasattr(voice_model, 'is_loaded') else False,
                'api': True,
                'synthesizer': synthesizer is not None,
                'voice_cloner': voice_cloner is not None,
                'singing_conversion_pipeline': singing_conversion_pipeline is not None
            },
            'system': {}
        }

        if memory_percent is not None:
            health_status['system']['memory_percent'] = memory_percent
            health_status['system']['cpu_percent'] = cpu_percent

        if gpu_status:
            health_status['system']['gpu'] = gpu_status

        return jsonify(health_status)

    @app.route('/health/live')
    def liveness():
        """Liveness probe - checks if application is running"""
        return jsonify({'status': 'alive'}), 200

    @app.route('/health/ready')
    def readiness():
        """Readiness probe - checks if application is ready to serve traffic"""
        # Check critical components
        is_ready = True
        components = {}

        # Check model
        if voice_model:
            model_ready = voice_model.is_loaded() if hasattr(voice_model, 'is_loaded') else True
            components['model'] = 'ready' if model_ready else 'not_ready'
            is_ready = is_ready and model_ready
        else:
            components['model'] = 'not_initialized'
            is_ready = False

        # Check GPU (optional, not critical for readiness)
        if gpu_manager:
            gpu_ready = gpu_manager.is_cuda_available()
            components['gpu'] = 'available' if gpu_ready else 'unavailable'

        # Check synthesizer
        if synthesizer:
            components['synthesizer'] = 'ready'
        else:
            components['synthesizer'] = 'not_initialized'
            is_ready = False

        # Check voice_cloner (optional, not critical for readiness)
        if voice_cloner:
            components['voice_cloner'] = 'ready'
        else:
            components['voice_cloner'] = 'not_initialized'

        # Check singing conversion pipeline (optional, not critical for readiness)
        if singing_conversion_pipeline:
            components['singing_conversion_pipeline'] = 'ready'
        else:
            components['singing_conversion_pipeline'] = 'not_initialized'
            # Pipeline is optional, don't fail readiness

        status_code = 200 if is_ready else 503
        return jsonify({
            'status': 'ready' if is_ready else 'not_ready',
            'components': components
        }), status_code

    @app.route('/metrics')
    def metrics():
        """Prometheus metrics endpoint"""
        metrics_collector = get_metrics_collector()
        if not metrics_collector.enabled:
            return "Metrics not enabled", 503

        return metrics_collector.generate_metrics(), 200, {
            'Content-Type': metrics_collector.get_content_type()
        }

    @app.route('/ws/audio_stream', methods=['GET'])
    def websocket_stream_info():
        """Information about WebSocket audio streaming endpoint."""
        return jsonify({
            'endpoint': '/ws/audio_stream',
            'protocol': 'WebSocket',
            'supported_events': [
                'connect', 'disconnect', 'join', 'leave',
                'audio_stream', 'synthesize_stream', 'audio_analysis',
                'voice_config', 'get_status'
            ],
            'message': 'Use WebSocket connection for real-time audio streaming',
            'connection_url': f"ws://{request.host}/socket.io/"
        })

    # Note: All /api/* endpoints are now handled by the API blueprint (api.py)
    # The blueprint is registered with url_prefix='/api' and provides all API functionality

    @app.errorhandler(404)
    def not_found(error):
        """Handle not found error"""
        return jsonify({'error': 'Not found', 'message': 'The requested resource was not found'}), 404

    @app.errorhandler(413)
    def request_entity_too_large(error):
        """Handle file too large error"""
        return jsonify({'error': 'File too large. Maximum size is 100MB'}), 413

    @app.errorhandler(500)
    def internal_error(error):
        """Handle internal server error"""
        logger.error(f"Internal server error: {str(error)}")
        return jsonify({'error': 'Internal server error'}), 500

    return app, socketio

def run_server(host='0.0.0.0', port=5000, debug=False):
    """Run the Flask development server with SocketIO"""
    global app, socketio
    if app is None or socketio is None:
        app, socketio = create_app()

    logger.info(f"Starting AutoVoice server on {host}:{port}")
    # Use SocketIO's run method for WebSocket support
    if socketio:
        socketio.run(app, host=host, port=port, debug=debug)
    else:
        app.run(host=host, port=port, debug=debug)
