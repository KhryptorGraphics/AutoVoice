"""REST API endpoints for AutoVoice with comprehensive voice synthesis and audio processing"""
import base64
import io
import os
import json
import logging
import time
import uuid
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

try:
    import soundfile
    SOUNDFILE_AVAILABLE = True
except ImportError:
    soundfile = None
    SOUNDFILE_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    librosa = None
    LIBROSA_AVAILABLE = False

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    nr = None
    NOISEREDUCE_AVAILABLE = False

# Import voice cloning exceptions
try:
    from ..inference.voice_cloner import InvalidAudioError
    INVALID_AUDIO_ERROR_AVAILABLE = True
except ImportError:
    InvalidAudioError = None
    INVALID_AUDIO_ERROR_AVAILABLE = False

try:
    from ..storage.voice_profiles import ProfileNotFoundError
    PROFILE_NOT_FOUND_ERROR_AVAILABLE = True
except ImportError:
    ProfileNotFoundError = None
    PROFILE_NOT_FOUND_ERROR_AVAILABLE = False

# Import singing conversion exceptions
try:
    from ..inference.singing_conversion_pipeline import SeparationError, ConversionError
    SINGING_CONVERSION_ERRORS_AVAILABLE = True
except ImportError:
    SeparationError = None
    ConversionError = None
    SINGING_CONVERSION_ERRORS_AVAILABLE = False

# Import shared utilities
from .utils import allowed_file, ALLOWED_AUDIO_EXTENSIONS

logger = logging.getLogger(__name__)

# Use /api/v1 prefix for versioned API
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

UPLOAD_FOLDER = '/tmp/autovoice_uploads'
ALLOWED_EXTENSIONS = ALLOWED_AUDIO_EXTENSIONS  # Use shared constant
MAX_TEXT_LENGTH = 5000
MAX_AUDIO_DURATION = 600  # 10 minutes


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


@api_bp.route('/health/live', methods=['GET'])
def health_liveness():
    """Kubernetes liveness probe endpoint - checks if service is running."""
    return jsonify({
        'status': 'live',
        'timestamp': time.time()
    }), 200


@api_bp.route('/health/ready', methods=['GET'])
def health_readiness():
    """Kubernetes readiness probe endpoint - checks if service can handle requests."""
    from datetime import datetime, timezone

    inference_engine = getattr(current_app, 'inference_engine', None)
    audio_processor = getattr(current_app, 'audio_processor', None)

    # Check if critical components are initialized
    ready = bool(inference_engine and audio_processor)

    if ready:
        return jsonify({
            'status': 'ready',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'components': {
                'inference_engine': bool(inference_engine),
                'audio_processor': bool(audio_processor)
            }
        }), 200
    else:
        return jsonify({
            'status': 'not_ready',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'components': {
                'inference_engine': bool(inference_engine),
                'audio_processor': bool(audio_processor)
            }
        }), 503


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
    # Check inference engine availability
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
            if NOISEREDUCE_AVAILABLE and NUMPY_AVAILABLE:
                try:
                    # Convert to CPU numpy float array
                    audio_np = processed_audio.detach().cpu().numpy() if torch and isinstance(processed_audio, torch.Tensor) else processed_audio

                    # Store original shape
                    original_shape = audio_np.shape

                    # Apply denoising per channel if multi-channel
                    if audio_np.ndim > 1 and audio_np.shape[0] > 1:
                        # Multi-channel: apply per channel
                        denoised_channels = []
                        for ch in range(audio_np.shape[0]):
                            denoised_ch = nr.reduce_noise(
                                y=audio_np[ch],
                                sr=sample_rate
                            )
                            denoised_channels.append(denoised_ch)
                        denoised_audio = np.stack(denoised_channels, axis=0)
                    else:
                        # Single channel or 1D
                        if audio_np.ndim > 1:
                            audio_np = audio_np.squeeze()
                        denoised_audio = nr.reduce_noise(
                            y=audio_np,
                            sr=sample_rate
                        )
                        # Restore shape if needed
                        if len(original_shape) > 1:
                            denoised_audio = denoised_audio.reshape(original_shape)

                    # Convert back to torch tensor
                    processed_audio = torch.from_numpy(denoised_audio).float()

                except Exception as e:
                    logger.error(f"Denoising failed: {e}", exc_info=True)
                    processed_audio = audio_tensor
            else:
                logger.warning("Denoising requested but noisereduce not available")

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
    # Check inference engine availability
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
                        'success': True,
                        'converted_audio': converted_audio_b64,
                        'format': 'wav',
                        'sample_rate': sample_rate
                    }), 200
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


# Import new modules for NEXT PHASE features
try:
    from ..inference.model_deployment_service import ModelDeploymentService
    MODEL_DEPLOYMENT_AVAILABLE = True
except ImportError:
    ModelDeploymentService = None
    MODEL_DEPLOYMENT_AVAILABLE = False

try:
    from ..inference.realtime_voice_conversion_pipeline import (
        RealtimeVoiceConversionPipeline,
        AdvancedVocalProcessor
    )
    REALTIME_VOICE_CONVERSION_AVAILABLE = True
except ImportError:
    RealtimeVoiceConversionPipeline = None
    AdvancedVocalProcessor = None
    REALTIME_VOICE_CONVERSION_AVAILABLE = False

try:
    from ..inference.professional_music_integration import (
        ProfessionalMusicAPI,
        ProfessionalMetadata
    )
    PROFESSIONAL_MUSIC_INTEGRATION_AVAILABLE = True
except ImportError:
    ProfessionalMusicAPI = None
    ProfessionalMetadata = None
    PROFESSIONAL_MUSIC_INTEGRATION_AVAILABLE = False

# Initialize new services if available
model_deployment_service = None
realtime_voice_conversion_pipeline = None
advanced_vocal_processor = None
professional_music_api = None
professional_metadata = None

if MODEL_DEPLOYMENT_AVAILABLE:
    model_deployment_service = ModelDeploymentService(current_app.app_config)

if REALTIME_VOICE_CONVERSION_AVAILABLE:
    realtime_voice_conversion_pipeline = RealtimeVoiceConversionPipeline(current_app.app_config)
    advanced_vocal_processor = AdvancedVocalProcessor(current_app.app_config)

if PROFESSIONAL_MUSIC_INTEGRATION_AVAILABLE:
    professional_music_api = ProfessionalMusicAPI(current_app.app_config)
    professional_metadata = ProfessionalMetadata(current_app.app_config)


# MODEL DEPLOYMENT ENDPOINTS
@api_bp.route('/models/deploy', methods=['POST'])
def deploy_model():
    """Deploy a new ML model for voice analysis.

    Request (multipart/form-data):
        model_name (str): Name for the deployed model (required)
        model_file (file): Model file (.pth, .pt, .onnx, or .engine)
        model_config (str): JSON configuration for the model
        auto_activate (bool): Whether to automatically activate the model

    Returns:
        JSON with deployment status
    """
    if not MODEL_DEPLOYMENT_AVAILABLE:
        return jsonify({
            'error': 'Model deployment service unavailable',
            'message': 'Model deployment components not loaded'
        }), 503

    # Get model name
    model_name = request.form.get('model_name')
    if not model_name:
        return jsonify({'error': 'model_name is required'}), 400

    # Check for model file
    if 'model_file' not in request.files:
        return jsonify({'error': 'No model file provided'}), 400

    file = request.files['model_file']
    if file.filename == '':
        return jsonify({'error': 'No model file selected'}), 400

    # Read model data
    model_data = file.read()

    # Get additional parameters
    model_config = request.form.get('model_config', '{}')
    auto_activate = request.form.get('auto_activate', 'true').lower() == 'true'

    try:
        model_config_dict = json.loads(model_config)
    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid JSON in model_config'}), 400

    # Deploy model
    success = model_deployment_service.deploy_model(
        model_name=model_name,
        model_data=model_data,
        auto_activate=auto_activate
    )

    if success:
        return jsonify({
            'status': 'success',
            'message': f'Model {model_name} deployed successfully',
            'model_name': model_name,
            'auto_activated': auto_activate
        }), 201
    else:
        return jsonify({
            'error': 'Model deployment failed',
            'model_name': model_name
        }), 500


@api_bp.route('/models/<model_name>/activate', methods=['POST'])
def activate_model(model_name: str):
    """Activate a deployed model.

    Returns:
        JSON with activation status
    """
    if not MODEL_DEPLOYMENT_AVAILABLE:
        return jsonify({
            'error': 'Model deployment service unavailable'
        }), 503

    success = model_deployment_service.activate_model(model_name)

    if success:
        return jsonify({
            'status': 'success',
            'message': f'Model {model_name} activated',
            'model_name': model_name
        }), 200
    else:
        return jsonify({
            'error': f'Failed to activate model {model_name}'
        }), 500


@api_bp.route('/models/<model_name>/inference', methods=['POST'])
def run_model_inference(model_name: str):
    """Run inference on a deployed model.

    Request JSON:
        inputs (dict): Model inputs
        **kwargs: Additional inference parameters

    Returns:
        JSON with inference results
    """
    if not MODEL_DEPLOYMENT_AVAILABLE:
        return jsonify({
            'error': 'Model deployment service unavailable'
        }), 503

    data = request.get_json()
    if not data or 'inputs' not in data:
        return jsonify({'error': 'inputs field is required'}), 400

    inputs = data['inputs']
    kwargs = {k: v for k, v in data.items() if k != 'inputs'}

    try:
        result = model_deployment_service.run_inference(model_name, inputs, **kwargs)
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Inference failed for model {model_name}: {e}")
        return jsonify({'error': 'Inference failed', 'details': str(e)}), 500


@api_bp.route('/models/ab-test', methods=['POST'])
def setup_ab_test():
    """Set up A/B testing between two models.

    Request JSON:
        test_name (str): Name for the A/B test
        model_a (str): First model name
        model_b (str): Second model name
        traffic_split (float): Traffic split ratio (0.0-1.0)

    Returns:
        JSON with A/B test setup status
    """
    if not MODEL_DEPLOYMENT_AVAILABLE:
        return jsonify({
            'error': 'Model deployment service unavailable'
        }), 503

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request data is required'}), 400

    required_fields = ['test_name', 'model_a', 'model_b']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'{field} is required'}), 400

    test_name = data['test_name']
    model_a = data['model_a']
    model_b = data['model_b']
    traffic_split = data.get('traffic_split', 0.5)

    success = model_deployment_service.setup_ab_test(
        test_name=test_name,
        model_a=model_a,
        model_b=model_b,
        traffic_split=traffic_split
    )

    if success:
        return jsonify({
            'status': 'success',
            'message': f'A/B test {test_name} set up between {model_a} and {model_b}',
            'traffic_split': traffic_split
        }), 200
    else:
        return jsonify({
            'error': 'Failed to set up A/B test',
            'test_name': test_name
        }), 500


@api_bp.route('/models/ab-test/<test_name>/inference', methods=['POST'])
def run_ab_test_inference(test_name: str):
    """Run inference with A/B testing.

    Request JSON:
        inputs (dict): Model inputs

    Returns:
        JSON with inference results including A/B test info
    """
    if not MODEL_DEPLOYMENT_AVAILABLE:
        return jsonify({
            'error': 'Model deployment service unavailable'
        }), 503

    data = request.get_json()
    if not data or 'inputs' not in data:
        return jsonify({'error': 'inputs field is required'}), 400

    inputs = data['inputs']

    try:
        result = model_deployment_service.run_ab_test_inference(test_name, inputs)
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"A/B test inference failed for {test_name}: {e}")
        return jsonify({'error': 'A/B test inference failed', 'details': str(e)}), 500


@api_bp.route('/models/stats', methods=['GET'])
def get_model_deployment_stats():
    """Get model deployment statistics."""
    if not MODEL_DEPLOYMENT_AVAILABLE:
        return jsonify({
            'error': 'Model deployment service unavailable'
        }), 503

    stats = model_deployment_service.get_deployment_stats()
    return jsonify(stats), 200


# REAL-TIME VOICE CONVERSION ENDPOINTS
@api_bp.route('/realtime/start', methods=['POST'])
def start_realtime_conversion():
    """Start real-time voice conversion streaming.

    Request JSON:
        target_profile_id (str): ID of target voice profile (optional)
        quality_mode (str): Quality mode ('fast', 'balanced', 'quality')
        sample_rate (int): Audio sample rate
        buffer_size (int): Processing buffer size

    Returns:
        JSON with streaming status
    """
    if not REALTIME_VOICE_CONVERSION_AVAILABLE:
        return jsonify({
            'error': 'Real-time voice conversion unavailable'
        }), 503

    data = request.get_json() or {}
    target_profile_id = data.get('target_profile_id')
    quality_mode = data.get('quality_mode', 'balanced')
    sample_rate = data.get('sample_rate', 44100)
    buffer_size = data.get('buffer_size', 2048)

    # Update pipeline config
    config = current_app.app_config.copy()
    config.update({
        'sample_rate': sample_rate,
        'buffer_size': buffer_size,
        'quality_mode': quality_mode
    })

    global realtime_voice_conversion_pipeline
    realtime_voice_conversion_pipeline = RealtimeVoiceConversionPipeline(config)

    success = realtime_voice_conversion_pipeline.start_streaming(target_profile_id)

    if success:
        return jsonify({
            'status': 'started',
            'message': 'Real-time voice conversion streaming started',
            'config': {
                'quality_mode': quality_mode,
                'sample_rate': sample_rate,
                'target_profile': target_profile_id
            }
        }), 200
    else:
        return jsonify({
            'error': 'Failed to start real-time streaming'
        }), 500


@api_bp.route('/realtime/stop', methods=['POST'])
def stop_realtime_conversion():
    """Stop real-time voice conversion streaming."""
    if not REALTIME_VOICE_CONVERSION_AVAILABLE or not realtime_voice_conversion_pipeline:
        return jsonify({
            'error': 'Real-time voice conversion not active'
        }), 503

    success = realtime_voice_conversion_pipeline.stop_streaming()

    if success:
        return jsonify({
            'status': 'stopped',
            'message': 'Real-time voice conversion streaming stopped'
        }), 200
    else:
        return jsonify({
            'error': 'Failed to stop real-time streaming'
        }), 500


@api_bp.route('/realtime/process', methods=['POST'])
def process_realtime_chunk():
    """Process a chunk of audio in real-time.

    Request JSON:
        audio_data (str): Base64 encoded audio chunk
        format (str): Audio format ('wav', 'pcm', etc.)

    Returns:
        JSON with processed audio chunk
    """
    if not REALTIME_VOICE_CONVERSION_AVAILABLE or not realtime_voice_conversion_pipeline:
        return jsonify({
            'error': 'Real-time voice conversion not active'
        }), 503

    data = request.get_json()
    if not data or 'audio_data' not in data:
        return jsonify({'error': 'audio_data is required'}), 400

    try:
        # Decode audio (simplified - would need proper format handling)
        audio_b64 = data['audio_data']
        audio_bytes = base64.b64decode(audio_b64)

        # Convert to numpy array (simplified - would need proper audio decoding)
        audio_chunk = np.frombuffer(audio_bytes, dtype=np.float32)

        # Process chunk (placeholder - in real implementation would handle various formats)
        processed_chunk = realtime_voice_conversion_pipeline.process_audio_chunk(audio_chunk)

        if processed_chunk is not None:
            # Encode back to base64 (simplified)
            processed_bytes = (processed_chunk * 32767).astype(np.int16).tobytes()
            processed_b64 = base64.b64encode(processed_bytes).decode('utf-8')

            return jsonify({
                'status': 'success',
                'processed_audio': processed_b64,
                'format': 'pcm_s16le'  # Example format
            }), 200
        else:
            return jsonify({
                'status': 'buffering',
                'message': 'Buffer needs more data'
            }), 202

    except Exception as e:
        logger.error(f"Real-time processing error: {e}")
        return jsonify({'error': 'Processing failed', 'details': str(e)}), 500


@api_bp.route('/realtime/stats', methods=['GET'])
def get_realtime_stats():
    """Get real-time streaming statistics."""
    if not REALTIME_VOICE_CONVERSION_AVAILABLE or not realtime_voice_conversion_pipeline:
        return jsonify({
            'error': 'Real-time voice conversion not active'
        }), 503

    stats = realtime_voice_conversion_pipeline.get_streaming_stats()
    return jsonify(stats), 200


@api_bp.route('/realtime/profile', methods=['POST'])
def update_realtime_profile():
    """Update target profile during real-time streaming.

    Request JSON:
        profile_id (str): New target voice profile ID
    """
    if not REALTIME_VOICE_CONVERSION_AVAILABLE or not realtime_voice_conversion_pipeline:
        return jsonify({
            'error': 'Real-time voice conversion not active'
        }), 503

    data = request.get_json()
    if not data or 'profile_id' not in data:
        return jsonify({'error': 'profile_id is required'}), 400

    profile_id = data['profile_id']
    success = realtime_voice_conversion_pipeline.update_target_profile(profile_id)

    if success:
        return jsonify({
            'status': 'success',
            'message': f'Updated to profile {profile_id}',
            'profile_id': profile_id
        }), 200
    else:
        return jsonify({
            'error': f'Failed to update profile {profile_id}'
        }), 500


# ADVANCED VOCAL PROCESSING ENDPOINTS
@api_bp.route('/vocal/emotion', methods=['POST'])
def inject_emotion():
    """Inject emotional characteristics into vocal audio.

    Request JSON:
        audio_data (str): Base64 encoded audio
        emotion (str): Emotion type ('excited', 'sad', 'calm', etc.)
        intensity (float): Emotion intensity (0.0-1.0)

    Returns:
        JSON with processed audio
    """
    if not REALTIME_VOICE_CONVERSION_AVAILABLE or not advanced_vocal_processor:
        return jsonify({
            'error': 'Advanced vocal processing unavailable'
        }), 503

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request data is required'}), 400

    required_fields = ['audio_data', 'emotion']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'{field} is required'}), 400

    audio_data = data['audio_data']
    emotion = data['emotion']
    intensity = data.get('intensity', 0.5)

    try:
        # Decode audio (simplified)
        audio_bytes = base64.b64decode(audio_data)
        audio = np.frombuffer(audio_bytes, dtype=np.float32)

        # Apply emotion injection
        processed_audio = advanced_vocal_processor.inject_emotion(audio, emotion, intensity)

        # Encode back
        processed_bytes = (processed_audio * 32767).astype(np.int16).tobytes()
        processed_b64 = base64.b64encode(processed_bytes).decode('utf-8')

        return jsonify({
            'status': 'success',
            'processed_audio': processed_b64,
            'emotion': emotion,
            'intensity': intensity
        }), 200

    except Exception as e:
        logger.error(f"Emotion injection failed: {e}")
        return jsonify({'error': 'Emotion injection failed', 'details': str(e)}), 500


@api_bp.route('/vocal/style-transfer', methods=['POST'])
def transfer_vocal_style():
    """Transfer vocal style from a reference source.

    Request JSON:
        audio_data (str): Base64 encoded target audio
        style_source (str): Style reference identifier
        strength (float): Style transfer strength (0.0-1.0)

    Returns:
        JSON with processed audio
    """
    if not REALTIME_VOICE_CONVERSION_AVAILABLE or not advanced_vocal_processor:
        return jsonify({
            'error': 'Advanced vocal processing unavailable'
        }), 503

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request data is required'}), 400

    required_fields = ['audio_data', 'style_source']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'{field} is required'}), 400

    audio_data = data['audio_data']
    style_source = data['style_source']
    strength = data.get('strength', 0.5)

    try:
        # Decode audio
        audio_bytes = base64.b64decode(audio_data)
        audio = np.frombuffer(audio_bytes, dtype=np.float32)

        # Apply style transfer (placeholder)
        processed_audio = advanced_vocal_processor.transfer_style(audio, style_source)

        # Encode back
        processed_bytes = (processed_audio * 32767).astype(np.int16).tobytes()
        processed_b64 = base64.b64encode(processed_bytes).decode('utf-8')

        return jsonify({
            'status': 'success',
            'processed_audio': processed_b64,
            'style_source': style_source,
            'strength': strength
        }), 200

    except Exception as e:
        logger.error(f"Style transfer failed: {e}")
        return jsonify({'error': 'Style transfer failed', 'details': str(e)}), 500


# PROFESSIONAL MUSIC PRODUCTION ENDPOINTS
@api_bp.route('/sessions', methods=['POST'])
def create_session():
    """Create a new production session.

    Request JSON:
        session_config (dict): Session configuration with metadata, audio settings, etc.

    Returns:
        JSON with session information
    """
    if not PROFESSIONAL_MUSIC_INTEGRATION_AVAILABLE:
        return jsonify({
            'error': 'Professional music integration unavailable'
        }), 503

    data = request.get_json()
    if not data or 'session_config' not in data:
        return jsonify({'error': 'session_config is required'}), 400

    session_config = data['session_config']

    session = professional_music_api.create_session(session_config)

    return jsonify(session), 201


@api_bp.route('/sessions/<session_id>', methods=['GET'])
def get_session(session_id: str):
    """Get session information."""
    if not PROFESSIONAL_MUSIC_INTEGRATION_AVAILABLE:
        return jsonify({
            'error': 'Professional music integration unavailable'
        }), 503

    session = professional_music_api.get_session(session_id)

    if session:
        return jsonify(session), 200
    else:
        return jsonify({'error': 'Session not found'}), 404


@api_bp.route('/sessions/<session_id>/metadata', methods=['POST'])
def update_session_metadata(session_id: str):
    """Update session metadata."""
    if not PROFESSIONAL_MUSIC_INTEGRATION_AVAILABLE:
        return jsonify({
            'error': 'Professional music integration unavailable'
        }), 503

    data = request.get_json()
    if not data or 'metadata' not in data:
        return jsonify({'error': 'metadata is required'}), 400

    metadata = data['metadata']
    success = professional_music_api.update_session_metadata(session_id, metadata)

    if success:
        return jsonify({
            'status': 'success',
            'message': f'Session {session_id} metadata updated'
        }), 200
    else:
        return jsonify({'error': f'Session {session_id} not found'}), 404


@api_bp.route('/batch', methods=['POST'])
def submit_batch_job():
    """Submit a batch processing job.

    Request JSON:
        job_config (dict): Job configuration with tasks, priorities, deadlines

    Returns:
        JSON with job information
    """
    if not PROFESSIONAL_MUSIC_INTEGRATION_AVAILABLE:
        return jsonify({
            'error': 'Professional music integration unavailable'
        }), 503

    data = request.get_json()
    if not data or 'job_config' not in data:
        return jsonify({'error': 'job_config is required'}), 400

    job_config = data['job_config']

    job_info = professional_music_api.process_batch_job(job_config)

    return jsonify(job_info), 201


@api_bp.route('/batch/<job_id>', methods=['GET'])
def get_batch_job_status(job_id: str):
    """Get batch job status and progress."""
    if not PROFESSIONAL_MUSIC_INTEGRATION_AVAILABLE:
        return jsonify({
            'error': 'Professional music integration unavailable'
        }), 503

    job_status = professional_music_api.get_batch_job_status(job_id)

    if job_status:
        return jsonify(job_status), 200
    else:
        return jsonify({'error': 'Job not found'}), 404


@api_bp.route('/batch/<job_id>', methods=['DELETE'])
def cancel_batch_job(job_id: str):
    """Cancel a batch job."""
    if not PROFESSIONAL_MUSIC_INTEGRATION_AVAILABLE:
        return jsonify({
            'error': 'Professional music integration unavailable'
        }), 503

    success = professional_music_api.cancel_batch_job(job_id)

    if success:
        return jsonify({
            'status': 'success',
            'message': f'Job {job_id} cancelled'
        }), 200
    else:
        return jsonify({'error': f'Job {job_id} not found or cannot be cancelled'}), 404


@api_bp.route('/batch/stats', methods=['GET'])
def get_batch_processing_stats():
    """Get batch processing statistics."""
    if not PROFESSIONAL_MUSIC_INTEGRATION_AVAILABLE:
        return jsonify({
            'error': 'Professional music integration unavailable'
        }), 503

    stats = professional_music_api.get_batch_processing_stats()
    return jsonify(stats), 200


@api_bp.route('/sessions/<session_id>/export', methods=['POST'])
def export_session(session_id: str):
    """Export session audio.

    Request JSON:
        export_config (dict): Export settings (format, bitrate, metadata, etc.)
    """
    if not PROFESSIONAL_MUSIC_INTEGRATION_AVAILABLE:
        return jsonify({
            'error': 'Professional music integration unavailable'
        }), 503

    data = request.get_json()
    if not data or 'export_config' not in data:
        return jsonify({'error': 'export_config is required'}), 400

    export_config = data['export_config']

    try:
        result = professional_music_api.export_session_audio(session_id, export_config)
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Session export failed: {e}")
        return jsonify({'error': 'Export failed', 'details': str(e)}), 500


@api_bp.route('/daw/integration', methods=['GET'])
def get_daw_integration_info():
    """Get DAW integration information."""
    if not PROFESSIONAL_MUSIC_INTEGRATION_AVAILABLE:
        return jsonify({
            'error': 'Professional music integration unavailable'
        }), 503

    info = professional_music_api.get_daw_integration_info()
    return jsonify(info), 200


@api_bp.route('/daw/project', methods=['POST'])
def process_daw_project():
    """Process a DAW project file or data.

    Request JSON:
        project_data (dict): Project data with tracks, regions, automation, etc.
    """
    if not PROFESSIONAL_MUSIC_INTEGRATION_AVAILABLE:
        return jsonify({
            'error': 'Professional music integration unavailable'
        }), 503

    data = request.get_json()
    if not data or 'project_data' not in data:
        return jsonify({'error': 'project_data is required'}), 400

    project_data = data['project_data']

    result = professional_music_api.process_daw_project(project_data)
    return jsonify(result), 200


@api_bp.route('/metadata/professional', methods=['POST'])
def create_professional_metadata():
    """Create professional metadata for a session."""
    if not PROFESSIONAL_MUSIC_INTEGRATION_AVAILABLE:
        return jsonify({
            'error': 'Professional music integration unavailable'
        }), 503

    data = request.get_json()
    if not data or 'session_metadata' not in data:
        return jsonify({'error': 'session_metadata is required'}), 400

    session_metadata = data['session_metadata']

    metadata = professional_metadata.create_professional_metadata(session_metadata)
    return jsonify(metadata), 200


@api_bp.route('/metadata/midi', methods=['POST'])
def extract_midi_data():
    """Extract data from MIDI file."""
    if not PROFESSIONAL_MUSIC_INTEGRATION_AVAILABLE:
        return jsonify({
            'error': 'Professional music integration unavailable'
        }), 503

    # Check for MIDI file
    if 'midi_file' not in request.files:
        return jsonify({'error': 'No MIDI file provided'}), 400

    file = request.files['midi_file']
    if file.filename == '':
        return jsonify({'error': 'No MIDI file selected'}), 400

    midi_data = file.read()

    result = professional_metadata.extract_midi_data(midi_data)
    return jsonify(result), 200


@api_bp.route('/daw/template', methods=['POST'])
def create_daw_project_template():
    """Create DAW project template."""
    if not PROFESSIONAL_MUSIC_INTEGRATION_AVAILABLE:
        return jsonify({
            'error': 'Professional music integration unavailable'
        }), 503

    data = request.get_json()
    if not data or 'session_id' not in data:
        return jsonify({'error': 'session_id is required'}), 400

    session_id = data['session_id']
    template_type = data.get('template_type', 'voice_conversion')

    template = professional_metadata.create_daw_project_template(session_id, template_type)
    return jsonify(template), 200
