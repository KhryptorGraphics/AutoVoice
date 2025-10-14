"""REST API endpoints for AutoVoice with comprehensive voice synthesis and audio processing"""
import base64
import io
import os
import json
import logging
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from typing import Optional, Dict, Any
import tempfile

# Graceful imports with fallback
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    torchaudio = None
    TORCHAUDIO_AVAILABLE = False

logger = logging.getLogger(__name__)

# Changed URL prefix from /api/v1 to /api to match tests
api_bp = Blueprint('api', __name__, url_prefix='/api')

UPLOAD_FOLDER = '/tmp/autovoice_uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a', 'webm'}
MAX_TEXT_LENGTH = 5000
MAX_AUDIO_DURATION = 600  # 10 minutes


def allowed_file(filename):
    """Check if the file extension is allowed and filename is safe."""
    if not filename or '.' not in filename:
        return False
    
    # Check for path traversal attempts
    if '..' in filename or '/' in filename or '\\' in filename:
        return False
    
    # Check file extension
    extension = filename.rsplit('.', 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS


def validate_request_json(required_fields):
    """Decorator to validate JSON request data."""
    def decorator(f):
        def wrapped(*args, **kwargs):
            if not request.is_json:
                return jsonify({'error': 'Content-Type must be application/json'}), 400

            data = request.get_json()
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400

            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return jsonify({
                    'error': f'Missing required fields: {", ".join(missing_fields)}'
                }), 400

            return f(*args, **kwargs)
        wrapped.__name__ = f.__name__
        return wrapped
    return decorator


@api_bp.route('/health', methods=['GET'])
def health_check():
    """API-specific health check endpoint."""
    import time
    from datetime import datetime, timezone
    
    audio_processor = getattr(current_app, 'audio_processor', None)
    inference_engine = getattr(current_app, 'inference_engine', None)
    gpu_manager = getattr(current_app, 'gpu_manager', None)

    # Determine GPU availability
    gpu_available = False
    if gpu_manager:
        try:
            gpu_available = gpu_manager.is_cuda_available()
        except Exception:
            gpu_available = False

    # Determine model status
    model_loaded = False
    if inference_engine:
        if hasattr(inference_engine, 'is_loaded'):
            model_loaded = inference_engine.is_loaded()
        else:
            model_loaded = True  # Assume loaded if inference engine exists

    # Determine overall status
    status = 'healthy'
    if not inference_engine:
        status = 'degraded'
    elif not audio_processor:
        status = 'degraded'

    return jsonify({
        'status': status,
        'gpu_available': gpu_available,
        'model_loaded': model_loaded,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'service': 'AutoVoice API',
        'endpoints': {
            'synthesize': bool(inference_engine),
            'process_audio': bool(audio_processor),
            'analyze': bool(audio_processor)
        },
        'dependencies': {
            'numpy': NUMPY_AVAILABLE,
            'torch': TORCH_AVAILABLE,
            'torchaudio': TORCHAUDIO_AVAILABLE
        }
    })


@api_bp.route('/synthesize', methods=['POST'])
@validate_request_json(['text', 'speaker_id'])
def synthesize_voice():
    """Synthesize speech from text using the VoiceInferenceEngine.

    Request JSON:
        text (str): Text to synthesize (required)
        speaker_id (int): Speaker ID for multi-speaker models (optional)
        voice_config (dict): Voice synthesis parameters (currently informational only)
            - temperature (float): Sampling temperature (not applied)
            - speed (float): Speech speed multiplier (not applied)
            - pitch (float): Pitch adjustment (not applied)
            Note: voice_config is accepted but not currently applied to the synthesis

    Returns:
        JSON with base64-encoded audio and metadata
    """
    if not TORCH_AVAILABLE:
        return jsonify({
            'error': 'Voice synthesis service unavailable',
            'message': 'PyTorch not installed'
        }), 503

    inference_engine = getattr(current_app, 'inference_engine', None)
    if not inference_engine:
        return jsonify({
            'error': 'Voice synthesis service unavailable',
            'message': 'Inference engine not initialized'
        }), 503

    data = request.get_json()
    text = data['text']

    # Validate text length
    if len(text) > MAX_TEXT_LENGTH:
        return jsonify({
            'error': f'Text too long. Maximum {MAX_TEXT_LENGTH} characters allowed'
        }), 400

    if not text.strip():
        return jsonify({'error': 'Text cannot be empty'}), 400

    # Get required parameters
    speaker_id = data['speaker_id']
    voice_config = data.get('voice_config', {})
    
    # Validate speaker_id
    if not isinstance(speaker_id, int) or speaker_id < 0:
        return jsonify({'error': 'Invalid speaker_id. Must be a non-negative integer'}), 400
    
    # Check if speaker_id is available (based on model configuration)
    app_config = getattr(current_app, 'app_config', {})
    max_speakers = app_config.get('model', {}).get('num_speakers', 1)
    if speaker_id >= max_speakers:
        return jsonify({
            'error': f'Invalid speaker_id. Must be between 0 and {max_speakers - 1}'
        }), 404

    try:
        # Synthesize speech
        logger.info(f"Synthesizing speech for text: {text[:50]}...")

        # Call the inference engine with standardized method name
        # The synthesizer should have synthesize_speech method
        if hasattr(inference_engine, 'synthesize_speech'):
            audio_data = inference_engine.synthesize_speech(
                text=text,
                speaker_id=speaker_id
            )
        elif hasattr(inference_engine, 'text_to_speech'):
            # Fallback to text_to_speech method
            audio_data = inference_engine.text_to_speech(
                text=text,
                speaker_id=speaker_id,
                speed=voice_config.get('speed', 1.0),
                pitch=voice_config.get('pitch', 1.0)
            )
        else:
            return jsonify({
                'error': 'Voice synthesis method not found',
                'message': 'Inference engine does not support synthesis'
            }), 500

        # Convert audio to base64
        if torch and isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.cpu().numpy()

        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(1, -1)

        # Fix config access - use correct nesting
        sample_rate = current_app.app_config.get('audio', {}).get('sample_rate', 22050)

        # Save to temporary WAV buffer
        buffer = io.BytesIO()

        if TORCHAUDIO_AVAILABLE:
            try:
                torchaudio.save(buffer, torch.from_numpy(audio_data), sample_rate, format='wav')
            except Exception as e:
                logger.warning(f"torchaudio.save failed, using wave fallback: {e}")
                import wave
                buffer = io.BytesIO()
                with wave.open(buffer, 'wb') as wav_file:
                    wav_file.setnchannels(audio_data.shape[0])
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    # Convert to int16
                    audio_int16 = (audio_data * 32767).astype(np.int16) if NUMPY_AVAILABLE else audio_data
                    wav_file.writeframes(audio_int16.tobytes())
        else:
            # Fallback to wave module
            import wave
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(audio_data.shape[0])
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                # Convert to int16
                audio_int16 = (audio_data * 32767).astype(np.int16) if NUMPY_AVAILABLE else audio_data
                wav_file.writeframes(audio_int16.tobytes())

        buffer.seek(0)

        # Encode to base64
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        return jsonify({
            'status': 'success',
            'audio': audio_base64,
            'format': 'wav',
            'sample_rate': sample_rate,
            'duration': len(audio_data[0]) / sample_rate,
            'metadata': {
                'text_length': len(text),
                'speaker_id': speaker_id,
                'voice_config': voice_config
            }
        })

    except Exception as e:
        logger.error(f"Voice synthesis error: {e}", exc_info=True)
        return jsonify({
            'error': 'Voice synthesis failed',
            'message': str(e) if current_app.debug else 'Internal processing error'
        }), 500


@api_bp.route('/process_audio', methods=['POST'])
def process_audio():
    """Process audio file or data using the AudioProcessor.

    Accepts either:
    1. Multipart form with 'audio' file
    2. JSON with 'audio_data' base64-encoded audio

    Request:
        audio (file): Audio file upload
        OR
        audio_data (str): Base64-encoded audio data
        processing_config (dict): Processing parameters (optional)
            - enable_vad (bool): Voice activity detection
            - enable_denoising (bool): Noise reduction
            - enable_pitch_extraction (bool): Extract pitch

    Returns:
        JSON with processed audio features and analysis
    """
    if not TORCHAUDIO_AVAILABLE:
        return jsonify({
            'error': 'Audio processing service unavailable',
            'message': 'torchaudio not installed'
        }), 503

    audio_processor = getattr(current_app, 'audio_processor', None)
    if not audio_processor:
        return jsonify({
            'error': 'Audio processing service unavailable',
            'message': 'Audio processor not initialized'
        }), 503

    audio_data = None
    # Fix config access
    sample_rate = current_app.app_config.get('audio', {}).get('sample_rate', 22050)

    # Handle file upload
    if 'audio' in request.files:
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if file and allowed_file(file.filename):
            # Save to temporary file with secure filename
            secure_name = secure_filename(file.filename)
            if not secure_name:
                return jsonify({'error': 'Invalid filename'}), 400
            
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(secure_name)[1], delete=False) as tmp_file:
                file.save(tmp_file.name)
                try:
                    # Load audio
                    waveform, sr = torchaudio.load(tmp_file.name)
                    # Resample if needed
                    if sr != sample_rate:
                        resampler = torchaudio.transforms.Resample(sr, sample_rate)
                        waveform = resampler(waveform)
                    audio_data = waveform.numpy() if NUMPY_AVAILABLE else waveform
                finally:
                    os.unlink(tmp_file.name)
        else:
            return jsonify({'error': 'Invalid file format'}), 400

    # Handle base64 audio data
    elif request.is_json and 'audio_data' in request.get_json():
        data = request.get_json()
        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(data['audio_data'])
            audio_buffer = io.BytesIO(audio_bytes)
            waveform, sr = torchaudio.load(audio_buffer)
            # Resample if needed
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(sr, sample_rate)
                waveform = resampler(waveform)
            audio_data = waveform.numpy() if NUMPY_AVAILABLE else waveform
        except Exception as e:
            return jsonify({'error': f'Invalid audio data: {str(e)}'}), 400
    else:
        return jsonify({'error': 'No audio data provided'}), 400

    # Check audio duration
    duration = audio_data.shape[-1] / sample_rate
    if duration > MAX_AUDIO_DURATION:
        return jsonify({
            'error': f'Audio too long. Maximum {MAX_AUDIO_DURATION} seconds allowed'
        }), 400

    # Get processing configuration
    processing_config = request.get_json().get('processing_config', {}) if request.is_json else {}

    try:
        # Process audio
        logger.info(f"Processing audio: duration={duration:.2f}s")

        # Convert to tensor for processing
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(1, -1)
        audio_tensor = torch.from_numpy(audio_data).float() if torch and NUMPY_AVAILABLE else audio_data

        results = {}

        # Pitch extraction
        if processing_config.get('enable_pitch_extraction', True):
            # Ensure audio tensor is 1D
            audio_1d = audio_tensor.squeeze(0) if hasattr(audio_tensor, 'ndim') and audio_tensor.ndim > 1 else audio_tensor
            pitch = audio_processor.extract_pitch(audio_1d)
            # Move to CPU if it's a tensor
            if torch and isinstance(pitch, torch.Tensor):
                pitch = pitch.detach().cpu().numpy()
            elif not isinstance(pitch, np.ndarray) and NUMPY_AVAILABLE:
                pitch = np.array(pitch)
            results['pitch'] = {
                'values': pitch.tolist() if hasattr(pitch, 'tolist') else list(pitch),
                'mean': float(np.mean(pitch[pitch > 0])) if NUMPY_AVAILABLE and len(pitch[pitch > 0]) > 0 else 0,
                'std': float(np.std(pitch[pitch > 0])) if NUMPY_AVAILABLE and len(pitch[pitch > 0]) > 0 else 0
            }

        # Voice activity detection
        if processing_config.get('enable_vad', True):
            # Ensure audio tensor is 1D
            audio_1d = audio_tensor.squeeze(0) if hasattr(audio_tensor, 'ndim') and audio_tensor.ndim > 1 else audio_tensor
            vad = audio_processor.voice_activity_detection(audio_1d)
            # Move to CPU if it's a tensor
            if torch and isinstance(vad, torch.Tensor):
                vad = vad.detach().cpu().numpy()
            elif not isinstance(vad, np.ndarray) and NUMPY_AVAILABLE:
                vad = np.array(vad)
            results['vad'] = {
                'segments': vad.tolist() if hasattr(vad, 'tolist') else list(vad),
                'voice_ratio': float(np.mean(vad)) if NUMPY_AVAILABLE and len(vad) > 0 else 0
            }

        # Compute spectrogram
        audio_1d = audio_tensor.squeeze(0) if hasattr(audio_tensor, 'ndim') and audio_tensor.ndim > 1 else audio_tensor
        spectrogram = audio_processor.compute_spectrogram(audio_1d)
        # Move to CPU if it's a tensor
        if torch and isinstance(spectrogram, torch.Tensor):
            spectrogram = spectrogram.detach().cpu()
        results['spectrogram'] = {
            'shape': list(spectrogram.shape),
            'min': float(spectrogram.min()),
            'max': float(spectrogram.max()),
            'mean': float(spectrogram.mean())
        }

        # Apply denoising if requested
        processed_audio = audio_tensor
        if processing_config.get('enable_denoising', False):
            # Apply denoising (placeholder - implement actual denoising)
            processed_audio = audio_tensor  # TODO: Implement denoising

        # Convert processed audio to base64
        buffer = io.BytesIO()
        if TORCHAUDIO_AVAILABLE:
            try:
                torchaudio.save(buffer, processed_audio, sample_rate, format='wav')
            except Exception as e:
                # Fallback to wave module if torchaudio fails
                logger.warning(f"torchaudio.save failed, using wave fallback: {e}")
                import wave
                buffer = io.BytesIO()
                with wave.open(buffer, 'wb') as wav_file:
                    processed_audio_np = processed_audio.detach().cpu().numpy() if torch and isinstance(processed_audio, torch.Tensor) else processed_audio
                    wav_file.setnchannels(processed_audio_np.shape[0])
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    # Convert to int16
                    audio_int16 = (processed_audio_np * 32767).astype(np.int16) if NUMPY_AVAILABLE else processed_audio_np
                    wav_file.writeframes(audio_int16.tobytes())
        else:
            import wave
            with wave.open(buffer, 'wb') as wav_file:
                processed_audio_np = processed_audio.detach().cpu().numpy() if torch and isinstance(processed_audio, torch.Tensor) else processed_audio
                wav_file.setnchannels(processed_audio_np.shape[0])
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                # Convert to int16
                audio_int16 = (processed_audio_np * 32767).astype(np.int16) if NUMPY_AVAILABLE else processed_audio_np
                wav_file.writeframes(audio_int16.tobytes())
        buffer.seek(0)
        processed_audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        return jsonify({
            'status': 'success',
            'processed_audio': processed_audio_base64,
            'format': 'wav',
            'sample_rate': sample_rate,
            'duration': duration,
            'analysis': results,
            'processing_config': processing_config
        })

    except Exception as e:
        logger.error(f"Audio processing error: {e}", exc_info=True)
        return jsonify({
            'error': 'Audio processing failed',
            'message': str(e) if current_app.debug else 'Internal processing error'
        }), 500


@api_bp.route('/analyze', methods=['POST'])
def analyze_audio():
    """Quick audio analysis without full processing.

    Request:
        audio_data (str): Base64-encoded audio data
        OR
        audio_url (str): URL to audio file

    Returns:
        JSON with audio analysis results
    """
    if not TORCHAUDIO_AVAILABLE:
        return jsonify({
            'error': 'Audio analysis service unavailable',
            'message': 'torchaudio not installed'
        }), 503

    audio_processor = getattr(current_app, 'audio_processor', None)
    if not audio_processor:
        return jsonify({
            'error': 'Audio analysis service unavailable',
            'message': 'Audio processor not initialized'
        }), 503

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    # Fix config access
    sample_rate = current_app.app_config.get('audio', {}).get('sample_rate', 22050)

    try:
        # Load audio from base64
        if 'audio_data' in data:
            audio_bytes = base64.b64decode(data['audio_data'])
            audio_buffer = io.BytesIO(audio_bytes)
            waveform, sr = torchaudio.load(audio_buffer)
        else:
            return jsonify({'error': 'No audio data provided'}), 400

        # Resample if needed
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)

        # Quick analysis
        duration = waveform.shape[-1] / sample_rate

        # Compute basic statistics
        analysis = {
            'duration': duration,
            'sample_rate': sample_rate,
            'channels': waveform.shape[0],
            'samples': waveform.shape[-1],
            'statistics': {
                'mean': float(waveform.mean()),
                'std': float(waveform.std()),
                'min': float(waveform.min()),
                'max': float(waveform.max()),
                'rms': float(torch.sqrt(torch.mean(waveform ** 2))) if torch else 0
            }
        }

        # Quick pitch analysis
        waveform_1d = waveform.squeeze(0) if waveform.ndim > 1 else waveform
        pitch = audio_processor.extract_pitch(waveform_1d)
        if pitch is not None:
            # Move to CPU if it's a tensor
            if torch and isinstance(pitch, torch.Tensor):
                pitch = pitch.detach().cpu().numpy()
            elif not isinstance(pitch, np.ndarray) and NUMPY_AVAILABLE:
                pitch = np.array(pitch)
            if len(pitch) > 0:
                valid_pitch = pitch[pitch > 0] if NUMPY_AVAILABLE else [p for p in pitch if p > 0]
                if len(valid_pitch) > 0:
                    analysis['pitch'] = {
                        'mean_hz': float(np.mean(valid_pitch)) if NUMPY_AVAILABLE else sum(valid_pitch) / len(valid_pitch),
                        'std_hz': float(np.std(valid_pitch)) if NUMPY_AVAILABLE else 0,
                        'min_hz': float(np.min(valid_pitch)) if NUMPY_AVAILABLE else min(valid_pitch),
                        'max_hz': float(np.max(valid_pitch)) if NUMPY_AVAILABLE else max(valid_pitch)
                    }

        return jsonify({
            'status': 'success',
            'analysis': analysis
        })

    except Exception as e:
        logger.error(f"Audio analysis error: {e}", exc_info=True)
        return jsonify({
            'error': 'Audio analysis failed',
            'message': str(e) if current_app.debug else 'Internal processing error'
        }), 500


@api_bp.route('/models/info', methods=['GET'])
def get_models_info():
    """Get information about available models and their capabilities."""
    inference_engine = getattr(current_app, 'inference_engine', None)
    app_config = getattr(current_app, 'app_config', {})

    model_info = {
        'status': 'available' if inference_engine else 'unavailable',
        'models': []
    }

    if inference_engine and app_config:
        model_config = app_config.get('model', {})
        model_info['models'].append({
            'name': model_config.get('name', 'unknown'),
            'version': model_config.get('version', 'unknown'),
            'type': model_config.get('type', 'unknown'),
            'capabilities': {
                'multi_speaker': model_config.get('num_speakers', 1) > 1,
                'num_speakers': model_config.get('num_speakers', 1),
                'style_transfer': model_config.get('enable_style_transfer', False),
                'tensorrt': hasattr(inference_engine, 'get_model_info') and inference_engine.get_model_info().get('tensorrt_available', False),
                'device': str(inference_engine.device) if hasattr(inference_engine, 'device') else 'unknown'
            },
            'parameters': {
                'max_length': app_config.get('inference', {}).get('max_length', 1000),
                'temperature_range': [0.1, 2.0],
                'speed_range': [0.5, 2.0],
                'pitch_range': [-12, 12]
            }
        })

    return jsonify(model_info)


@api_bp.route('/config', methods=['GET'])
def get_config():
    """Get current API configuration (sanitized)."""
    app_config = getattr(current_app, 'app_config', {})

    # Return sanitized configuration - fix config access
    safe_config = {
        'audio': {
            'sample_rate': app_config.get('audio', {}).get('sample_rate', 22050),
            'channels': app_config.get('audio', {}).get('channels', 1),
            'formats': list(ALLOWED_EXTENSIONS)
        },
        'limits': {
            'max_text_length': MAX_TEXT_LENGTH,
            'max_audio_duration': MAX_AUDIO_DURATION,
            'max_file_size': current_app.config.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024)
        },
        'processing': {
            'vad_enabled': app_config.get('audio', {}).get('enable_vad', True),
            'denoising_enabled': app_config.get('audio', {}).get('enable_denoising', True),
            'pitch_extraction_enabled': app_config.get('audio', {}).get('enable_pitch_extraction', True)
        }
    }

    return jsonify(safe_config)


@api_bp.route('/config', methods=['POST'])
def update_config():
    """Update voice synthesis parameters (runtime only, not persistent)."""
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No configuration data provided'}), 400

    # Only allow certain parameters to be updated
    allowed_updates = ['temperature', 'speed', 'pitch', 'speaker_id']
    updates = {k: v for k, v in data.items() if k in allowed_updates}

    if not updates:
        return jsonify({'error': 'No valid configuration parameters provided'}), 400

    # Store in app context (runtime only)
    if not hasattr(current_app, 'runtime_config'):
        current_app.runtime_config = {}

    current_app.runtime_config.update(updates)

    return jsonify({
        'status': 'success',
        'message': 'Configuration updated',
        'updates': updates
    })


# Error handlers for the API blueprint
@api_bp.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors."""
    max_size = current_app.config.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024)
    return jsonify({
        'error': 'File too large',
        'message': f'Maximum file size is {max_size / (1024 * 1024):.1f} MB'
    }), 413


@api_bp.route('/convert', methods=['POST'])
def convert_voice():
    """Convert voice from one speaker to another.
    
    Request (multipart/form-data):
        audio (file): Audio file to convert (required)
        target_speaker (str): Target speaker ID (required)
        
    Returns:
        JSON with converted audio and metadata
    """
    if not TORCH_AVAILABLE:
        return jsonify({
            'error': 'Voice conversion service unavailable',
            'message': 'PyTorch not installed'
        }), 503

    inference_engine = getattr(current_app, 'inference_engine', None)
    if not inference_engine:
        return jsonify({
            'error': 'Voice conversion service unavailable',
            'message': 'Inference engine not initialized'
        }), 503

    # Check for audio file
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    target_speaker = request.form.get('target_speaker')
    if not target_speaker:
        return jsonify({'error': 'No target speaker provided'}), 400

    if file and allowed_file(file.filename):
        try:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file.filename)[1], delete=False) as tmp_file:
                file.save(tmp_file.name)
                
                try:
                    # For now, return a placeholder response since voice conversion is complex
                    return jsonify({
                        'status': 'success',
                        'message': 'Voice conversion not yet implemented',
                        'target_speaker': target_speaker,
                        'original_filename': file.filename
                    })
                finally:
                    os.unlink(tmp_file.name)
        except Exception as e:
            logger.error(f"Voice conversion error: {e}", exc_info=True)
            return jsonify({
                'error': 'Voice conversion failed',
                'message': str(e) if current_app.debug else 'Internal processing error'
            }), 500
    else:
        return jsonify({'error': 'Invalid file format'}), 400


@api_bp.route('/clone', methods=['POST'])
def clone_voice():
    """Clone a voice from reference audio.
    
    Request (multipart/form-data):
        reference_audio (file): Reference audio for cloning (required)
        target_text (str): Text to synthesize with cloned voice (required)
        
    Returns:
        JSON with cloned voice audio and metadata
    """
    if not TORCH_AVAILABLE:
        return jsonify({
            'error': 'Voice cloning service unavailable',
            'message': 'PyTorch not installed'
        }), 503

    # Check for audio file
    if 'reference_audio' not in request.files:
        return jsonify({'error': 'No reference audio file provided'}), 400

    file = request.files['reference_audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    target_text = request.form.get('target_text')
    if not target_text:
        return jsonify({'error': 'No target text provided'}), 400

    if file and allowed_file(file.filename):
        try:
            # For now, return a placeholder response since voice cloning is complex
            return jsonify({
                'status': 'success', 
                'message': 'Voice cloning not yet implemented',
                'target_text': target_text,
                'reference_filename': file.filename
            })
        except Exception as e:
            logger.error(f"Voice cloning error: {e}", exc_info=True)
            return jsonify({
                'error': 'Voice cloning failed',
                'message': str(e) if current_app.debug else 'Internal processing error'
            }), 500
    else:
        return jsonify({'error': 'Invalid file format'}), 400


@api_bp.route('/speakers', methods=['GET'])
def get_speakers():
    """Get list of available speakers.
    
    Query parameters:
        language (str): Filter by language code (optional)
        
    Returns:
        JSON array of speaker information
    """
    inference_engine = getattr(current_app, 'inference_engine', None)
    app_config = getattr(current_app, 'app_config', {})
    
    # Get language filter
    language_filter = request.args.get('language')
    
    speakers = []
    
    # Generate speaker list based on model configuration
    if inference_engine and app_config:
        model_config = app_config.get('model', {})
        num_speakers = model_config.get('num_speakers', 1)
        
        for i in range(num_speakers):
            speaker = {
                'id': i,
                'name': f'Speaker {i}',
                'language': 'en',  # Default language
                'gender': 'neutral',
                'description': f'Generated speaker voice {i}'
            }
            
            # Apply language filter if specified
            if language_filter is None or speaker['language'] == language_filter:
                speakers.append(speaker)
    
    # If no speakers from config, provide default speaker
    if not speakers:
        speakers = [{
            'id': 0,
            'name': 'Default Speaker',
            'language': 'en',
            'gender': 'neutral',
            'description': 'Default voice'
        }]
    
    return jsonify(speakers)


@api_bp.route('/gpu_status', methods=['GET'])
def get_gpu_status():
    """Get GPU status and utilization information.
    
    Returns:
        JSON with GPU status and metrics
    """
    gpu_manager = getattr(current_app, 'gpu_manager', None)
    
    status = {
        'cuda_available': False,
        'device': 'cpu',
        'device_count': 0
    }
    
    if gpu_manager:
        try:
            status['cuda_available'] = gpu_manager.is_cuda_available()
            if hasattr(gpu_manager, 'get_device_count'):
                status['device_count'] = gpu_manager.get_device_count()
            
            if status['cuda_available']:
                status['device'] = 'cuda'
                
                # Get GPU details if CUDA is available
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    try:
                        status['device_name'] = torch.cuda.get_device_name(0)
                        status['memory_total'] = torch.cuda.get_device_properties(0).total_memory
                        status['memory_allocated'] = torch.cuda.memory_allocated(0)
                        status['memory_reserved'] = torch.cuda.memory_reserved(0)
                        status['memory_free'] = status['memory_total'] - status['memory_allocated']
                    except Exception as e:
                        logger.warning(f"Failed to get detailed GPU info: {e}")
        except Exception as e:
            logger.warning(f"Failed to get GPU status: {e}")
    
    return jsonify(status)


@api_bp.route('/ws/audio_stream', methods=['GET'])
def websocket_info():
    """Information about WebSocket audio streaming endpoint.
    
    Returns:
        JSON with WebSocket connection information
    """
    return jsonify({
        'endpoint': '/ws/audio_stream',
        'protocol': 'WebSocket',
        'supported_events': [
            'connect',
            'disconnect', 
            'join',
            'leave',
            'audio_stream',
            'synthesize_stream',
            'audio_analysis',
            'voice_config',
            'get_status'
        ],
        'message': 'Use WebSocket connection for real-time audio streaming'
    })


@api_bp.errorhandler(415)
def unsupported_media_type(error):
    """Handle unsupported media type errors."""
    return jsonify({
        'error': 'Unsupported media type',
        'message': 'This endpoint requires JSON or multipart/form-data'
    }), 415