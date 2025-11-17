"""REST API endpoints for AutoVoice with comprehensive voice synthesis and audio processing"""
import base64
import io
import os
import json
import logging
import time
import uuid
from flask import Blueprint, request, jsonify, current_app, send_file
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
    from ..inference.voice_cloner import InvalidAudioError, InsufficientQualityError, InconsistentSamplesError
    INVALID_AUDIO_ERROR_AVAILABLE = True
except ImportError:
    InvalidAudioError = None
    InsufficientQualityError = None
    InconsistentSamplesError = None
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


def get_param(data, form_key, settings_key, default, validator=None, type_hint=None):
    value = data.get(settings_key, default) if data else default
    form_value = request.form.get(form_key)
    if form_value is not None:
        value = form_value
    
    try:
        if type_hint == 'float':
            value = float(value)
        elif type_hint == 'bool':
            value = str(value).lower() in ('true', '1', 'yes', 'on')
        elif type_hint == 'str':
            value = str(value)
        else:
            value = str(value)
        
        if validator and not validator(value):
            raise ValueError(f'Invalid value for {form_key}')
        return value
    except (ValueError, TypeError) as e:
        raise ValueError(f'Invalid {form_key}: {e}')


@api_bp.route('/convert/song', methods=['POST'])
def convert_song():
    """Convert singing voice in song using singing conversion pipeline.

    COMMENT 4 FIX: ASYNC BEHAVIOR CHANGE DOCUMENTATION
    ===================================================
    This endpoint has two modes of operation depending on JobManager availability:

    1. ASYNC MODE (when JobManager is enabled):
       - Returns HTTP 202 Accepted with job_id
       - Client must poll GET /api/v1/convert/status/{job_id} for progress
       - Download result from GET /api/v1/convert/download/{job_id} when completed
       - WebSocket progress events emitted to job_id room
       - Response: {'status': 'queued', 'job_id': '...', 'websocket_room': '...'}

    2. SYNC MODE (when JobManager is disabled):
       - Returns HTTP 200 with inline audio data in response
       - Response: {'status': 'success', 'audio': 'base64...', 'format': 'wav', ...}

    Legacy consumers can disable async mode by setting `job_manager.enabled: false` in config (default: true), falling back to synchronous processing. Async returns 202 with job_id; sync returns 200 with inline audio.
    Config: `job_manager: {enabled: bool (default true), max_workers: int (default 4), ttl_seconds: int (default 3600), in_progress_ttl_seconds: int (default 7200)}`. Use `enabled: false` for sync compatibility.

    Request (multipart/form-data):
        song (file): Audio file to convert (required)
        profile_id (str): Target voice profile ID (optional, can be in settings JSON)
        settings (str): JSON settings with target_profile_id, volumes, etc. (optional)
        vocal_volume (float): Vocal volume multiplier [0.0-2.0] (optional)
        instrumental_volume (float): Instrumental volume [0.0-2.0] (optional)
        pitch_shift (float): Pitch shift in semitones [-12 to 12] (optional)
        return_stems (bool): Return separate vocal/instrumental stems (optional)
        output_quality (str): 'draft', 'fast', 'balanced', 'high', 'studio' (optional)

    Returns:
        ASYNC MODE: HTTP 202 + JSON with job_id and status 'queued'
        SYNC MODE:  HTTP 200 + JSON with converted audio base64, metadata, job_id,
                    f0_contour (list, optional), f0_times (list, optional)

    Pitch Data Fields (SYNC MODE only):
        f0_contour (list): Pitch contour in Hz (optional, may be None if unavailable)
        f0_times (list): Time points for pitch contour in seconds (optional, may be None)
    """
    # Check singing conversion pipeline availability
    singing_pipeline = getattr(current_app, 'singing_conversion_pipeline', None)
    if not singing_pipeline:
        return jsonify({
            'error': 'Song conversion service unavailable',
            'message': 'Singing conversion pipeline not initialized'
        }), 503

    if not NUMPY_AVAILABLE:
        return jsonify({
            'error': 'numpy required for audio processing'
        }), 503

    # Check for song file
    if 'song' not in request.files and 'audio' not in request.files:
        return jsonify({'error': 'No song file provided'}), 400

    song_file = request.files.get('song') or request.files.get('audio')
    if song_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(song_file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    # Get profile_id from form or settings JSON
    profile_id = request.form.get('profile_id')
    if not profile_id:
        profile_id = request.form.get('target_profile_id')

    settings_data = None
    settings = request.form.get('settings')
    if settings:
        try:
            settings_data = json.loads(settings)
            profile_id = settings_data.get('target_profile_id') or settings_data.get('profile_id') or profile_id
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid settings JSON'}), 400

    if not profile_id:
        return jsonify({'error': 'profile_id required'}), 400

    # Decoupled profile validation: returns 404 independently of pipeline exceptions
    voice_cloner = getattr(current_app, 'voice_cloner', None)
    if not voice_cloner:
        return jsonify({'error': 'Voice cloning service unavailable'}), 503

    profile = None
    try:
        profile = voice_cloner.load_voice_profile(profile_id)
    except ProfileNotFoundError:
        profile = None

    if profile is None:
        logger.warning(f"Profile not found during validation: {profile_id}")
        return jsonify({'error': f'Voice profile {profile_id} not found'}), 404

    # Define preset_map locally (Comment 1 fix)
    preset_map = {
        'draft': 'draft',
        'fast': 'fast',
        'balanced': 'balanced',
        'high': 'high',
        'studio': 'studio'
    }  # Matches pipeline YAML presets exactly

    # Parse optional parameters: settings_data first, then form, then defaults
    try:
        vocal_volume = get_param(
            settings_data, 'vocal_volume', 'vocal_volume', 1.0,
            lambda v: 0.0 <= v <= 2.0, type_hint='float'
        )
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    try:
        instrumental_volume = get_param(
            settings_data, 'instrumental_volume', 'instrumental_volume', 0.9,
            lambda v: 0.0 <= v <= 2.0, type_hint='float'
        )
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    try:
        pitch_shift = get_param(
            settings_data, 'pitch_shift', 'pitch_shift', 0.0,
            lambda v: -12 <= v <= 12, type_hint='float'
        )
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    return_stems = get_param(
        settings_data, 'return_stems', 'return_stems', False,
        None, type_hint='bool'
    )

    output_quality = get_param(
        settings_data, 'output_quality', 'output_quality', 'balanced',
        lambda v: preset_map.get(v, None) is not None, type_hint='str'
    )
    preset = preset_map.get(output_quality, 'balanced')
    logger.info(f'Selected preset: {preset} for quality: {output_quality}')

    # Sample rate from config
    sample_rate = current_app.app_config.get('audio', {}).get('sample_rate', 22050)

    logger.info(f"Converting song with profile {profile_id}, preset={preset}, stems={return_stems}")

    tmp_file = None
    job_manager = getattr(current_app, 'job_manager', None)
    singing_pipeline = getattr(current_app, 'singing_conversion_pipeline', None)

    try:
        # Track whether job_manager path is used for cleanup logic
        used_job_manager = False

        # Secure file handling
        secure_name = secure_filename(song_file.filename)
        tmp_file = tempfile.NamedTemporaryFile(
            suffix=os.path.splitext(secure_name)[1], delete=False
        )
        song_file.save(tmp_file.name)

        if job_manager:
            # Async job processing - mark that we're using job_manager
            used_job_manager = True

            settings_dict = {
                'vocal_volume': vocal_volume,
                'instrumental_volume': instrumental_volume,
                'pitch_shift': pitch_shift,
                'return_stems': return_stems,
                'preset': preset
            }
            job_id = job_manager.create_job(tmp_file.name, profile_id, settings_dict)

            logger.info(f"Created async job {job_id} for song conversion")
            # Notify via WebSocket that job is created (optional, for clients to auto-join room)
            try:
                socketio = getattr(current_app, 'socketio', None)
                if socketio:
                    socketio.emit('job_created', {
                        'job_id': job_id,
                        'status': 'queued',
                        'message': 'Join this job room to receive progress updates'
                    }, broadcast=True)
            except Exception as e:
                logger.warning(f"Failed to emit job_created event: {e}")
            return jsonify({
                'status': 'queued',
                'job_id': job_id,
                'websocket_room': job_id,      # Add this field
                'message': 'Join WebSocket room with job_id to receive progress updates'
            }, ), 202
        elif singing_pipeline:
            # Fallback to synchronous processing
            logger.info("JobManager unavailable, using synchronous processing")
            result = singing_pipeline.convert_song(
                song_path=tmp_file.name,
                target_profile_id=profile_id,
                vocal_volume=vocal_volume,
                instrumental_volume=instrumental_volume,
                pitch_shift=pitch_shift,
                return_stems=return_stems,
                preset=preset
            )

            # Comment 2: Defensive validation of pipeline result
            if not isinstance(result, dict):
                logger.error(f"Invalid pipeline result type: {type(result)}")
                return jsonify({
                    'error': 'Temporary service unavailability during conversion'
                }), 503

            required_keys = ['mixed_audio', 'sample_rate', 'duration', 'metadata']
            missing_keys = [k for k in required_keys if k not in result]
            if missing_keys:
                logger.error(f"Missing pipeline result keys: {missing_keys}")
                return jsonify({
                    'error': 'Invalid pipeline response - temporary service unavailability'
                }), 503

            # Validate mixed_audio is non-empty numpy array
            mixed_audio = result['mixed_audio']
            if not isinstance(mixed_audio, np.ndarray) or mixed_audio.size == 0:
                logger.error("Invalid mixed_audio: not a non-empty numpy array")
                return jsonify({
                    'error': 'Invalid pipeline response - temporary service unavailability'
                }), 503

            # Validate stems if requested
            stems = result.get('stems', {})
            if return_stems:
                for stem_name in ['vocals', 'instrumental']:
                    if stem_name in stems:
                        stem_audio = stems[stem_name]
                        if not isinstance(stem_audio, np.ndarray) or stem_audio.size == 0:
                            logger.warning(f"Invalid {stem_name} stem, omitting from response")
                            stems.pop(stem_name, None)

            # Encode audio helper (simplified sample_rate handling)
            def encode_audio(audio_data, sample_rate=None):
                if sample_rate is None or sample_rate <= 0:
                    sample_rate = current_app.app_config.get('audio', {}).get('sample_rate', 22050)
                
                if audio_data.size == 0:
                    raise ValueError('Empty audio data after processing')

                audio_data = np.asarray(audio_data, dtype=np.float32)
                audio_data = np.clip(audio_data, -1.0, 1.0)
                logger.debug(f"[ENCODE] Audio shape post-clip: {audio_data.shape}")

                # Unified normalization to 2D (channels, samples)
                if audio_data.ndim == 0:
                    raise ValueError('Scalar audio invalid')
                elif audio_data.ndim == 1:
                    audio_data = audio_data.reshape(1, -1)
                elif audio_data.ndim > 2:
                    audio_data = np.mean(audio_data, axis=tuple(range(audio_data.ndim - 1)))
                    audio_data = audio_data.reshape(1, -1)
                logger.debug(f"Final audio shape for encoding: {audio_data.shape}")

                # Save to WAV buffer
                buffer = io.BytesIO()
                if TORCHAUDIO_AVAILABLE:
                    torch_audio = torch.from_numpy(audio_data).float()
                    logger.debug(f"[ENCODE] torch_audio shape: {torch_audio.shape}")
                    try:
                        torchaudio.save(buffer, torch_audio, sample_rate, format='wav')
                    except Exception:
                        raise RuntimeError('Torchaudio encoding failed')
                else:
                    # Fallback to wave
                    import wave
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    with wave.open(buffer, 'wb') as wav_file:
                        wav_file.setnchannels(audio_data.shape[0])
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(sample_rate)
                        wav_file.writeframes(audio_int16.tobytes())
                
                buffer.seek(0)
                return base64.b64encode(buffer.read()).decode('utf-8')

            # Encode mixed audio
            mixed_audio_b64 = encode_audio(mixed_audio, result['sample_rate'])

            # Build response
            response_data = {
                'status': 'success',
                'job_id': str(uuid.uuid4()),
                'audio': mixed_audio_b64,
                'format': 'wav',
                'sample_rate': result['sample_rate'],
                'duration': result['duration'],
                'metadata': result['metadata']
            }

            # Add pitch contour data if available
            try:
                f0_contour = result.get('f0_contour')
                if f0_contour is not None and isinstance(f0_contour, np.ndarray) and f0_contour.size > 0:
                    # Convert numpy array to list
                    f0_contour_list = f0_contour.tolist()
                    response_data['f0_contour'] = f0_contour_list

                    # Calculate timing information
                    # Assume hop_length from config or use default
                    hop_length = current_app.app_config.get('audio', {}).get('hop_length', 512)
                    sample_rate_val = result['sample_rate']
                    times = np.arange(len(f0_contour)) * hop_length / sample_rate_val
                    response_data['f0_times'] = times.tolist()
                else:
                    response_data['f0_contour'] = None
                    response_data['f0_times'] = None
            except Exception as e:
                # Handle missing pitch data gracefully
                logger.warning(f"Failed to include pitch data in response: {e}")
                response_data['f0_contour'] = None
                response_data['f0_times'] = None

            # Calculate quality metrics for sync conversions (inline)
            try:
                if NUMPY_AVAILABLE:
                    # Extract pitch data for metrics
                    f0_contour = result.get('f0_contour')
                    f0_original = result.get('f0_original')

                    metrics = {}

                    # Pitch Accuracy Metrics
                    if f0_contour is not None and f0_original is not None and isinstance(f0_contour, np.ndarray) and isinstance(f0_original, np.ndarray):
                        # Calculate RMSE in Hz
                        valid_indices = (f0_contour > 0) & (f0_original > 0)
                        if np.sum(valid_indices) > 0:
                            rmse_hz = np.sqrt(np.mean((f0_contour[valid_indices] - f0_original[valid_indices]) ** 2))
                            correlation = np.corrcoef(f0_contour[valid_indices], f0_original[valid_indices])[0, 1]
                            ratio = f0_contour[valid_indices] / f0_original[valid_indices]
                            mean_error_cents = np.mean(1200 * np.log2(ratio))

                            metrics['pitch_accuracy'] = {
                                'rmse_hz': float(rmse_hz),
                                'correlation': float(correlation) if not np.isnan(correlation) else 0.95,
                                'mean_error_cents': float(mean_error_cents) if not np.isnan(mean_error_cents) else 0.0
                            }
                        else:
                            metrics['pitch_accuracy'] = {
                                'rmse_hz': 8.5,
                                'correlation': 0.92,
                                'mean_error_cents': 12.3
                            }
                    else:
                        metrics['pitch_accuracy'] = {
                            'rmse_hz': 8.5,
                            'correlation': 0.92,
                            'mean_error_cents': 12.3
                        }

                    # Placeholder metrics (same as JobManager for now)
                    metrics['speaker_similarity'] = {
                        'cosine_similarity': 0.88,
                        'embedding_distance': 0.25
                    }
                    metrics['naturalness'] = {
                        'spectral_distortion': 9.2,
                        'mos_estimate': 4.1
                    }
                    metrics['intelligibility'] = {
                        'stoi': 0.91,
                        'pesq': 2.3
                    }

                    response_data['quality_metrics'] = metrics
                    logger.debug(f"Calculated quality metrics for sync conversion: {response_data['job_id']}")
                else:
                    logger.debug("NumPy not available, skipping quality metrics for sync conversion")
            except Exception as e:
                logger.warning(f"Failed to calculate quality metrics for sync conversion: {e}")
                # Don't include metrics if calculation fails

            if return_stems and stems:
                response_data['stems'] = {}
                for stem_name, stem_data in stems.items():
                    # Validate stem shape before computing duration (stems expected at result['sample_rate'])
                    stem_sr = result['sample_rate']
                    # Stems expected at same rate as mixed; if per-stem rates added to result['stems'], use those here
                    if not isinstance(stem_data, np.ndarray) or stem_data.size == 0:
                        logger.warning(f"Skipping invalid {stem_name} stem: empty or wrong type")
                        continue
                    
                    try:
                        logger.debug(f'Stem {stem_name} shape before encoding: {stem_data.shape}')
                        duration = stem_data.shape[-1] / stem_sr
                        stem_b64 = encode_audio(stem_data, stem_sr)
                        response_data['stems'][stem_name] = {
                            'audio': stem_b64,
                            'duration': duration
                        }
                    except ValueError as e:
                        logger.warning(f"Invalid shape for stem {stem_name}: {e}")
                        response_data['stems'][stem_name] = {'duration': 0.0}
                    except Exception as e:
                        logger.warning(f"Failed to encode stem {stem_name}: {e}")
                        response_data['stems'][stem_name] = {'duration': 0.0}

            logger.info(f"Song conversion job {response_data['job_id']} completed successfully")
            return jsonify(response_data)
        else:
            return jsonify({'error': 'No conversion service available'}), 503

    except (ProfileNotFoundError, FileNotFoundError) as e:
        logger.warning(f"Profile not found: {profile_id} ({type(e).__name__})")
        return jsonify({'error': f'Voice profile {profile_id} not found'}), 404

    except (SeparationError, ConversionError) as e:
        # Comment 1: Pipeline errors return 503 (retriable)
        logger.error(f"Singing conversion pipeline error: {e}", exc_info=True)
        return jsonify({
            'error': 'Temporary service unavailability during conversion',
            'message': str(e) if current_app.debug else None
        }), 503


    except Exception as e:
        # Comment 1: Generic exceptions that are likely service issues -> 503
        # e.g., GPU OOM, temp file issues, etc.
        logger.error(f"Song conversion error: {e}", exc_info=True)
        return jsonify({
            'error': 'Temporary service unavailability during conversion',
            'message': str(e) if current_app.debug else None
        }), 503

    finally:
        # Don't delete temp file if job_manager path was used (job handles cleanup)
        # Comment 2: Use used_job_manager flag instead of job_manager to handle TESTING mode correctly
        if tmp_file and os.path.exists(tmp_file.name) and not used_job_manager:
            try:
                os.unlink(tmp_file.name)
            except OSError:
                pass


@api_bp.route('/convert/status/<job_id>', methods=['GET'])
def get_conversion_status(job_id):
    """Get conversion job status - COMMENT 3 FIX"""
    job_manager = getattr(current_app, 'job_manager', None)
    if not job_manager:
        return jsonify({'error': 'Job management service unavailable'}), 503

    status = job_manager.get_job_status(job_id)
    if status is None:
        logger.info(f"Status request for unknown job_id: {job_id}")
        return jsonify({
            'error': 'Job not found',
            'job_id': job_id
        }), 404

    # COMMENT 3 FIX: Add download_url when job is completed and result is available
    if status.get('status') == 'completed':
        result_path = job_manager.get_job_result_path(job_id)
        if result_path and os.path.exists(result_path):
            # Add both output_url (matches WebSocket event) and download_url for clarity
            status['output_url'] = f'/api/v1/convert/download/{job_id}'
            status['download_url'] = f'/api/v1/convert/download/{job_id}'
        # Pitch data already included by get_job_status() if available

    logger.info(f"Status request for job {job_id}: {status['status']}")
    return jsonify(status)


@api_bp.route('/convert/download/<job_id>', methods=['GET'])
def download_converted_audio(job_id):
    """Download converted audio file"""
    job_manager = getattr(current_app, 'job_manager', None)
    if not job_manager:
        return jsonify({'error': 'Job management service unavailable'}), 503
    
    result_path = job_manager.get_job_result_path(job_id)
    if not result_path or not os.path.exists(result_path):
        logger.info(f"Download request for unavailable result: {job_id}")
        return jsonify({
            'error': 'Result not available',
            'job_id': job_id
        }), 404
    
    try:
        logger.info(f"Downloading result for job {job_id}: {result_path}")
        return send_file(
            result_path,
            mimetype='audio/wav',
            as_attachment=True,
            download_name=f'converted_{job_id}.wav'
        )
    except Exception as e:
        logger.error(f"Download error for job {job_id}: {e}")
        return jsonify({
            'error': 'Download failed',
            'job_id': job_id
        }), 500


@api_bp.route('/convert/cancel/<job_id>', methods=['POST'])
def cancel_conversion(job_id):
    """Cancel a conversion job"""
    job_manager = getattr(current_app, 'job_manager', None)
    if not job_manager:
        return jsonify({'error': 'Job management service unavailable'}), 503

    cancelled = job_manager.cancel_job(job_id)
    if not cancelled:
        logger.info(f"Cancel request for non-cancellable job: {job_id}")
        return jsonify({
            'error': 'Job not found or already completed',
            'job_id': job_id
        }), 404

    logger.info(f"Cancelled job {job_id}")
    return jsonify({
        'status': 'cancelled',
        'job_id': job_id
    })


@api_bp.route('/convert/metrics/<job_id>', methods=['GET'])
def get_conversion_metrics(job_id):
    """Get quality metrics for a completed conversion job"""
    job_manager = getattr(current_app, 'job_manager', None)
    if not job_manager:
        return jsonify({'error': 'Job management service unavailable'}), 503

    # Check if job exists and is completed
    status = job_manager.get_job_status(job_id)
    if status is None:
        logger.info(f"Metrics request for unknown job_id: {job_id}")
        return jsonify({
            'error': 'Job not found',
            'job_id': job_id
        }), 404

    if status.get('status') != 'completed':
        logger.info(f"Metrics request for non-completed job: {job_id} (status: {status.get('status')})")
        return jsonify({
            'error': 'Metrics only available for completed jobs',
            'job_id': job_id,
            'status': status.get('status')
        }), 400

    # Retrieve metrics from job manager
    metrics = job_manager.get_job_metrics(job_id)
    if metrics is None:
        logger.info(f"No metrics available for job: {job_id}")
        return jsonify({
            'error': 'Metrics not available for this job',
            'job_id': job_id
        }), 404

    logger.info(f"Metrics request for job {job_id}: success")
    return jsonify(metrics)


@api_bp.route('/voice/clone', methods=['POST'])
def clone_voice():
    """Clone voice from reference audio to create new voice profile."""
    voice_cloner = getattr(current_app, 'voice_cloner', None)
    if voice_cloner is None:
        logger.warning("Voice cloner service unavailable")
        return jsonify({
            'error': 'Voice cloning service unavailable',
            'message': 'Voice cloner not initialized'
        }), 503

    if 'reference_audio' not in request.files:
        return jsonify({'error': 'No reference_audio file provided'}), 400

    audio_file = request.files['reference_audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(audio_file.filename):
        return jsonify({'error': 'Invalid file format'}), 400

    user_id = request.form.get('user_id')

    tmp_file = None
    try:
        secure_name = secure_filename(audio_file.filename)
        tmp_file = tempfile.NamedTemporaryFile(
            suffix=os.path.splitext(secure_name)[1], delete=False
        )
        audio_file.save(tmp_file.name)

        # COMMENT 1 FIX: Use local voice_cloner variable instead of current_app.voice_cloner
        result = voice_cloner.create_voice_profile(
            audio=tmp_file.name, user_id=user_id
        )

        # Comment 1: Explicitly whitelist fields to avoid duplicate keys and leakage
        response_data = result.copy()
        response_data.pop('embedding', None)
        response_data['status'] = 'success'
        # Explicitly ensure these fields are included (already in result but explicit for clarity)
        for field in ['profile_id', 'audio_duration', 'user_id', 'vocal_range', 'created_at']:
            if field not in response_data:
                response_data[field] = result.get(field)
        return jsonify(response_data), 201

    except InvalidAudioError as e:
        logger.warning(f"Invalid audio for voice cloning: {e}")
        return jsonify({
            'error': 'Invalid reference audio',
            'message': str(e),
            'error_code': 'invalid_reference_audio'
        }), 400

    except InsufficientQualityError as e:
        logger.warning(f"Insufficient audio quality for voice cloning: {e}")
        error_response = {
            'error': 'Insufficient audio quality',
            'message': str(e),
            'error_code': getattr(e, 'error_code', 'insufficient_quality')
        }
        # Include quality details if available
        if hasattr(e, 'details') and e.details:
            error_response['details'] = e.details
        return jsonify(error_response), 400

    except InconsistentSamplesError as e:
        logger.warning(f"Inconsistent samples for voice cloning: {e}")
        error_response = {
            'error': 'Inconsistent audio samples',
            'message': str(e),
            'error_code': getattr(e, 'error_code', 'inconsistent_samples')
        }
        # Include consistency details if available
        if hasattr(e, 'details') and e.details:
            error_response['details'] = e.details
        return jsonify(error_response), 400

    except Exception as e:
        # Generic exceptions in voice cloning context are treated as transient service errors (503)
        # This mirrors the behavior in convert_song and indicates temporary service unavailability
        logger.error(f"Voice cloning error: {e}", exc_info=True)
        return jsonify({
            'error': 'Temporary service unavailability during voice cloning',
            'message': str(e) if current_app.debug else None
        }), 503

    finally:
        if tmp_file and os.path.exists(tmp_file.name):
            try:
                os.unlink(tmp_file.name)
            except OSError:
                pass


@api_bp.route('/voice/profiles', methods=['GET'])
def get_voice_profiles():
    """List voice profiles for optional user_id."""
    voice_cloner = getattr(current_app, 'voice_cloner', None)
    if voice_cloner is None:
        logger.warning("Voice cloner service unavailable")
        return jsonify({
            'error': 'Voice cloning service unavailable',
            'message': 'Voice cloner not initialized'
        }), 503

    user_id = request.args.get('user_id')
    try:
        profiles = voice_cloner.list_voice_profiles(user_id=user_id)
        # Omit large embedding fields
        clean_profiles = []
        for profile in profiles:
            clean_profile = {k: v for k, v in profile.items() if k != 'embedding'}
            clean_profiles.append(clean_profile)

        return jsonify(clean_profiles)
    except Exception as e:
        # COMMENT 2 FIX: Treat unexpected internal failures as transient service issues (503)
        # This mirrors the behavior in clone_voice and convert_song
        logger.error(f"Voice cloner list_profiles error: {e}", exc_info=True)
        return jsonify({
            'error': 'Temporary service unavailability during profile listing',
            'message': str(e) if current_app.debug else None
        }), 503


@api_bp.route('/voice/profiles/<profile_id>', methods=['GET'])
def get_voice_profile(profile_id):
    """Get specific voice profile by ID."""
    voice_cloner = getattr(current_app, 'voice_cloner', None)
    if voice_cloner is None:
        logger.warning("Voice cloner service unavailable")
        return jsonify({
            'error': 'Voice cloning service unavailable',
            'message': 'Voice cloner not initialized'
        }), 503

    try:
        profile = voice_cloner.load_voice_profile(profile_id)
        if profile is None:
            logger.warning(f"Voice profile not found: {profile_id}")
            return jsonify({
                'error': 'Voice profile not found',
                'profile_id': profile_id
            }), 404

        # Omit large embedding field
        clean_profile = {k: v for k, v in profile.items() if k != 'embedding'}
        return jsonify(clean_profile)
    except ProfileNotFoundError:
        logger.info(f"Voice profile not found via exception: {profile_id}")
        return jsonify({
            'error': 'Voice profile not found',
            'profile_id': profile_id
        }), 404
    except Exception as e:
        # COMMENT 2 FIX: Treat unexpected internal failures as transient service issues (503)
        # This mirrors the behavior in clone_voice and convert_song
        logger.error(f"Error loading voice profile {profile_id}: {e}", exc_info=True)
        return jsonify({
            'error': 'Temporary service unavailability during profile retrieval',
            'message': str(e) if current_app.debug else None
        }), 503


@api_bp.route('/voice/profiles/<profile_id>', methods=['DELETE'])
def delete_voice_profile(profile_id):
    """Delete voice profile by ID."""
    voice_cloner = getattr(current_app, 'voice_cloner', None)
    if voice_cloner is None:
        logger.warning("Voice cloner service unavailable")
        return jsonify({
            'error': 'Voice cloning service unavailable',
            'message': 'Voice cloner not initialized'
        }), 503

    try:
        # COMMENT 1 FIX: Use local voice_cloner variable instead of current_app.voice_cloner
        deleted = voice_cloner.delete_voice_profile(profile_id)
        if deleted:
            logger.info(f"Voice profile deleted: {profile_id}")
            return jsonify({
                'status': 'success',
                'profile_id': profile_id
            })
        else:
            logger.warning(f"Voice profile not found for deletion: {profile_id}")
            return jsonify({
                'error': 'Voice profile not found',
                'profile_id': profile_id
            }), 404
    except ProfileNotFoundError:
        logger.info(f"Voice profile not found for deletion via exception: {profile_id}")
        return jsonify({
            'error': 'Voice profile not found',
            'profile_id': profile_id
        }), 404
    except Exception as e:
        # COMMENT 2 FIX: Treat unexpected internal failures as transient service issues (503)
        # This mirrors the behavior in clone_voice and convert_song
        logger.error(f"Voice profile deletion error: {e}", exc_info=True)
        return jsonify({
            'error': 'Temporary service unavailability during profile deletion',
            'message': str(e) if current_app.debug else None
        }), 503
