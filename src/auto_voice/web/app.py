"""Flask web application for AutoVoice."""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import torch
import os
import tempfile
import logging
from werkzeug.utils import secure_filename
from auto_voice.inference import VoiceSynthesizer
from auto_voice.audio import AudioProcessor
from auto_voice.web.api import api_bp
from auto_voice.web.job_manager import JobManager
from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
from auto_voice.inference.voice_cloner import VoiceCloner

logger = logging.getLogger(__name__)

UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}


def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_app(config_path: str = None, model_path: str = None, vocoder_path: str = None, config: dict = None):
    """Create Flask application with SocketIO support.

    Args:
        config_path: Path to configuration file (for compatibility with main.py)
        model_path: Path to acoustic model
        vocoder_path: Path to vocoder model
        config: Optional configuration dictionary to merge into app.config

    Returns:
        Tuple of (Flask app instance, SocketIO instance)
    """
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

    # Merge provided config
    if config:
        app.config.update(config)
    
    # Store app-level config for API access
    app.app_config = config if config else {}
    # SECURITY: Require SECRET_KEY in production
    secret_key = os.environ.get('SECRET_KEY')
    if not secret_key:
        if os.environ.get('FLASK_ENV') == 'production':
            raise ValueError("SECRET_KEY environment variable must be set in production")
        else:
            secret_key = 'dev-secret-key-not-for-production'
            logger.warning("Using default secret key - not suitable for production")
    app.config['SECRET_KEY'] = secret_key

    # Enable CORS
    CORS(app)

    # Initialize SocketIO
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

    # Initialize synthesizer if models provided
    synthesizer = None
    if model_path and vocoder_path:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        synthesizer = VoiceSynthesizer(model_path, vocoder_path, device)
        app.config['synthesizer'] = synthesizer

    # Audio processor
    audio_processor = AudioProcessor()
    app.config['audio_processor'] = audio_processor

    # Initialize singing conversion pipeline and voice cloner for job_manager
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize singing conversion pipeline
    singing_pipeline = None
    if config and config.get('singing_conversion_enabled', True):
        try:
            singing_pipeline = SingingConversionPipeline(device=device)
            app.singing_conversion_pipeline = singing_pipeline
            logger.info(f"Singing conversion pipeline initialized on {device}")
        except Exception as e:
            logger.warning(f"Failed to initialize singing conversion pipeline: {e}")
    
    # Initialize voice cloner
    voice_cloner = None
    if config and config.get('voice_cloning_enabled', True):
        try:
            voice_cloner = VoiceCloner(device=device)
            app.voice_cloner = voice_cloner
            logger.info(f"Voice cloner initialized on {device}")
        except Exception as e:
            logger.warning(f"Failed to initialize voice cloner: {e}")
    
    # Initialize JobManager if components are available
    job_manager = None
    if singing_pipeline and voice_cloner:
        try:
            job_manager_config = config.get('job_manager', {}) if config else {}
            job_manager_config.setdefault('max_workers', 4)
            job_manager_config.setdefault('ttl_seconds', 3600)
            job_manager_config.setdefault('in_progress_ttl_seconds', 7200)
            
            # Pass app.config for audio settings
            job_manager_config['audio'] = app.config
            
            job_manager = JobManager(
                config=job_manager_config,
                socketio=socketio,
                singing_pipeline=singing_pipeline,
                voice_profile_manager=voice_cloner
            )
            app.job_manager = job_manager
            job_manager.start_cleanup_thread()
            logger.info("JobManager initialized and cleanup thread started")
        except Exception as e:
            logger.warning(f"Failed to initialize JobManager: {e}")
    else:
        logger.info("JobManager not initialized (missing singing_pipeline or voice_cloner)")

    @app.route('/')
    def index():
        """Render main page."""
        return render_template('index.html')

    @app.route('/api/synthesize', methods=['POST'])
    def synthesize():
        """Synthesize speech from text."""
        if not synthesizer:
            return jsonify({'error': 'Synthesizer not initialized'}), 500

        data = request.json
        text = data.get('text', '')
        speaker_id = data.get('speaker_id', 0)
        pitch_scale = data.get('pitch_scale', 1.0)
        speed_scale = data.get('speed_scale', 1.0)

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        try:
            # Synthesize
            waveform = synthesizer.synthesize(
                text=text,
                speaker_id=speaker_id,
                pitch_scale=pitch_scale,
                speed_scale=speed_scale
            )

            # Save to temporary file
            output_path = os.path.join(UPLOAD_FOLDER, 'output.wav')
            synthesizer.save_audio(waveform, output_path)

            return send_file(output_path, mimetype='audio/wav')

        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/clone', methods=['POST'])
    def clone_voice():
        """Clone voice from reference audio."""
        if not synthesizer:
            return jsonify({'error': 'Synthesizer not initialized'}), 500

        # Check for file
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format'}), 400

        # Get text
        text = request.form.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Clone voice
            waveform = synthesizer.clone_voice(filepath, text)

            # Save output
            output_path = os.path.join(UPLOAD_FOLDER, 'cloned.wav')
            synthesizer.save_audio(waveform, output_path)

            # Clean up input file
            os.remove(filepath)

            return send_file(output_path, mimetype='audio/wav')

        except Exception as e:
            logger.error(f"Voice cloning error: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/convert', methods=['POST'])
    def convert_voice():
        """Convert voice between speakers."""
        if not synthesizer:
            return jsonify({'error': 'Synthesizer not initialized'}), 500

        # Check for file
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        file = request.files['audio']
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format'}), 400

        # Get target speaker
        target_speaker = request.form.get('target_speaker', 0)

        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Convert voice
            waveform = synthesizer.convert_voice(filepath, int(target_speaker))

            # Save output
            output_path = os.path.join(UPLOAD_FOLDER, 'converted.wav')
            synthesizer.save_audio(waveform, output_path)

            # Clean up
            os.remove(filepath)

            return send_file(output_path, mimetype='audio/wav')

        except Exception as e:
            logger.error(f"Voice conversion error: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/analyze', methods=['POST'])
    def analyze_voice():
        """Analyze voice characteristics."""
        # Check for file
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        file = request.files['audio']
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format'}), 400

        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Load and analyze
            from auto_voice.audio import VoiceAnalyzer
            analyzer = VoiceAnalyzer()
            waveform, _ = audio_processor.load_audio(filepath)
            characteristics = analyzer.analyze_voice_characteristics(waveform)

            # Clean up
            os.remove(filepath)

            return jsonify(characteristics)

        except Exception as e:
            logger.error(f"Voice analysis error: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/speakers', methods=['GET'])
    def list_speakers():
        """List available speakers."""
        # Placeholder speaker list
        speakers = [
            {'id': 0, 'name': 'Speaker 1', 'gender': 'female'},
            {'id': 1, 'name': 'Speaker 2', 'gender': 'male'},
            {'id': 2, 'name': 'Speaker 3', 'gender': 'female'},
            {'id': 3, 'name': 'Speaker 4', 'gender': 'male'},
            {'id': 4, 'name': 'Speaker 5', 'gender': 'neutral'}
        ]
        return jsonify(speakers)

    @app.route('/api/health', methods=['GET'])
    def health():
        """Health check endpoint."""
        status = {
            'status': 'healthy',
            'synthesizer': synthesizer is not None,
            'device': str(synthesizer.device) if synthesizer else 'cpu'
        }
        return jsonify(status)

    # SocketIO event handlers
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection."""
        logger.info(f"Client connected: {request.sid}")
        emit('connected', {'data': 'Connected to AutoVoice server'})

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection."""
        logger.info(f"Client disconnected: {request.sid}")

    @socketio.on('synthesize_stream')
    def handle_synthesize_stream(data):
        """Handle streaming synthesis request."""
        text = data.get('text', '')
        speaker_id = data.get('speaker_id', 0)

        if not text:
            emit('error', {'message': 'No text provided'})
            return

        if synthesizer:
            try:
                # Synthesize audio
                waveform = synthesizer.synthesize(
                    text=text,
                    speaker_id=speaker_id
                )
                # Convert to base64 for streaming
                import base64
                import io
                import soundfile as sf

                buffer = io.BytesIO()
                sf.write(buffer, waveform.cpu().numpy(), 22050, format='WAV')
                buffer.seek(0)
                audio_data = base64.b64encode(buffer.read()).decode('utf-8')

                emit('audio_chunk', {'audio': audio_data, 'finished': True})
            except Exception as e:
                logger.error(f"Streaming synthesis error: {e}")
                emit('error', {'message': str(e)})
        else:
            emit('error', {'message': 'Synthesizer not initialized'})

    # Register API blueprint
    app.register_blueprint(api_bp)

    return app, socketio


def run_server(host='0.0.0.0', port=5000, debug=False,
              model_path=None, vocoder_path=None):
    """Run the web server.

    Args:
        host: Host address
        port: Port number
        debug: Debug mode
        model_path: Path to acoustic model
        vocoder_path: Path to vocoder model
    """
    app, socketio = create_app(model_path=model_path, vocoder_path=vocoder_path)
    socketio.run(app, host=host, port=port, debug=debug)
