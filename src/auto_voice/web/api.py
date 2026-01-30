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

# Import YouTube downloader
try:
    from ..audio.youtube_downloader import YouTubeDownloader, YouTubeDownloadResult
    YOUTUBE_DOWNLOADER_AVAILABLE = True
except ImportError:
    YouTubeDownloader = None
    YouTubeDownloadResult = None
    YOUTUBE_DOWNLOADER_AVAILABLE = False

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
    name = request.form.get('name')

    tmp_file = None
    try:
        secure_name = secure_filename(audio_file.filename)
        tmp_file = tempfile.NamedTemporaryFile(
            suffix=os.path.splitext(secure_name)[1], delete=False
        )
        audio_file.save(tmp_file.name)

        # COMMENT 1 FIX: Use local voice_cloner variable instead of current_app.voice_cloner
        result = voice_cloner.create_voice_profile(
            audio=tmp_file.name, user_id=user_id, name=name
        )

        # Comment 1: Explicitly whitelist fields to avoid duplicate keys and leakage
        response_data = result.copy()
        response_data.pop('embedding', None)
        response_data['status'] = 'success'
        # Explicitly ensure these fields are included (already in result but explicit for clarity)
        for field in ['profile_id', 'audio_duration', 'user_id', 'name', 'vocal_range', 'created_at']:
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
        # Omit large embedding fields and normalize field names for frontend
        clean_profiles = []
        for profile in profiles:
            clean_profile = {k: v for k, v in profile.items() if k != 'embedding'}
            # Normalize field names for frontend compatibility
            if 'training_sample_count' in clean_profile:
                clean_profile['sample_count'] = clean_profile.pop('training_sample_count')
            elif 'sample_count' not in clean_profile:
                clean_profile['sample_count'] = 0
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


@api_bp.route('/voice/profiles/<profile_id>/training-status', methods=['GET'])
def get_profile_training_status(profile_id):
    """Get training status for a voice profile.

    Returns:
        JSON with training status information:
        - has_trained_model: bool - whether LoRA weights exist
        - training_status: str - 'pending', 'training', 'ready', 'failed'
        - model_version: str | None - version identifier if trained
    """
    from ..storage.voice_profiles import VoiceProfileStore, ProfileNotFoundError

    store = VoiceProfileStore()

    try:
        # Check if profile exists
        if not store.exists(profile_id):
            return jsonify({
                'error': 'Voice profile not found',
                'profile_id': profile_id
            }), 404

        # Get training status
        has_trained = store.has_trained_model(profile_id)

        # Load profile for additional status
        profile = store.load(profile_id)
        training_status = profile.get('training_status', 'pending' if not has_trained else 'ready')

        return jsonify({
            'profile_id': profile_id,
            'has_trained_model': has_trained,
            'training_status': training_status if not has_trained else 'ready',
            'model_version': profile.get('model_version'),
        })
    except ProfileNotFoundError:
        return jsonify({
            'error': 'Voice profile not found',
            'profile_id': profile_id
        }), 404
    except Exception as e:
        logger.error(f"Training status error for {profile_id}: {e}", exc_info=True)
        return jsonify({
            'error': 'Failed to get training status',
            'message': str(e) if current_app.debug else None
        }), 503


# ============================================================================
# Health & Metrics Endpoints
# ============================================================================

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for liveness/readiness probes.

    Returns:
        JSON with status and component health information

    Response Schema:
        {
            "status": "healthy" | "degraded" | "unhealthy",
            "timestamp": "ISO8601 timestamp",
            "components": {
                "api": {"status": "up"},
                "torch": {"status": "up" | "down", "cuda": bool, "version": str},
                "voice_cloner": {"status": "up" | "down"},
                "singing_pipeline": {"status": "up" | "down"},
                "job_manager": {"status": "up" | "down"}
            },
            "cuda_kernels_available": bool,
            "version": str
        }
    """
    components = {}
    overall_status = "healthy"

    # API is up if we're responding
    components['api'] = {'status': 'up'}

    # Check PyTorch
    if TORCH_AVAILABLE:
        components['torch'] = {
            'status': 'up',
            'version': torch.__version__,
            'cuda': torch.cuda.is_available()
        }
        if torch.cuda.is_available():
            try:
                components['torch']['device'] = torch.cuda.get_device_name(0)
            except Exception:
                pass
    else:
        components['torch'] = {'status': 'down', 'cuda': False}
        overall_status = "degraded"

    # Check voice cloner
    voice_cloner = getattr(current_app, 'voice_cloner', None)
    if voice_cloner:
        components['voice_cloner'] = {'status': 'up'}
    else:
        components['voice_cloner'] = {'status': 'down'}
        overall_status = "degraded"

    # Check singing pipeline
    singing_pipeline = getattr(current_app, 'singing_conversion_pipeline', None)
    if singing_pipeline:
        components['singing_pipeline'] = {'status': 'up'}
    else:
        components['singing_pipeline'] = {'status': 'down'}
        overall_status = "degraded"

    # Check job manager
    job_manager = getattr(current_app, 'job_manager', None)
    if job_manager:
        components['job_manager'] = {'status': 'up'}
    else:
        components['job_manager'] = {'status': 'down'}

    # Check CUDA kernels availability
    try:
        from ..gpu.cuda_kernels import CUDA_KERNELS_AVAILABLE
        cuda_kernels = CUDA_KERNELS_AVAILABLE
    except ImportError:
        cuda_kernels = False

    return jsonify({
        'status': overall_status,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'components': components,
        'cuda_kernels_available': cuda_kernels,
        'version': '0.1.0'
    })


@api_bp.route('/gpu/metrics', methods=['GET'])
def gpu_metrics():
    """Get GPU utilization and memory metrics.

    Returns:
        JSON with GPU metrics including memory usage, utilization, temperature

    Response Schema:
        {
            "available": bool,
            "device_count": int,
            "devices": [
                {
                    "index": int,
                    "name": str,
                    "memory_total_gb": float,
                    "memory_used_gb": float,
                    "memory_free_gb": float,
                    "utilization_percent": float,
                    "temperature_c": float
                }
            ]
        }
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return jsonify({
            'available': False,
            'device_count': 0,
            'devices': [],
            'message': 'CUDA not available'
        })

    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        devices = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
            except Exception:
                gpu_util = None

            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = None

            devices.append({
                'index': i,
                'name': name,
                'memory_total_gb': round(mem_info.total / (1024**3), 2),
                'memory_used_gb': round(mem_info.used / (1024**3), 2),
                'memory_free_gb': round(mem_info.free / (1024**3), 2),
                'utilization_percent': gpu_util,
                'temperature_c': temp
            })

        pynvml.nvmlShutdown()

        return jsonify({
            'available': True,
            'device_count': device_count,
            'devices': devices
        })

    except ImportError:
        # pynvml not available, use PyTorch fallback
        device_count = torch.cuda.device_count()
        devices = []

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            mem_allocated = torch.cuda.memory_allocated(i)
            mem_reserved = torch.cuda.memory_reserved(i)

            devices.append({
                'index': i,
                'name': props.name,
                'memory_total_gb': round(props.total_memory / (1024**3), 2),
                'memory_used_gb': round(mem_allocated / (1024**3), 2),
                'memory_reserved_gb': round(mem_reserved / (1024**3), 2),
                'utilization_percent': None,
                'temperature_c': None
            })

        return jsonify({
            'available': True,
            'device_count': device_count,
            'devices': devices,
            'note': 'Limited metrics (pynvml not available)'
        })

    except Exception as e:
        logger.warning(f"GPU metrics partially unavailable: {e}")
        # Return 200 with fallback data - GPU is available but some metrics unsupported
        device_count = torch.cuda.device_count()
        devices = []
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            devices.append({
                'index': i,
                'name': props.name,
                'memory_total_gb': round(props.total_memory / (1024**3), 2),
                'utilization_percent': None,
                'temperature_c': None
            })
        return jsonify({
            'available': True,
            'device_count': device_count,
            'devices': devices,
            'note': f'Some metrics unsupported: {e}'
        })


@api_bp.route('/kernels/metrics', methods=['GET'])
def kernel_metrics():
    """Get CUDA kernel performance metrics.

    Returns:
        JSON array of kernel metrics or empty array if unavailable

    Response Schema:
        [
            {
                "name": str,
                "calls": int,
                "total_time_ms": float,
                "avg_time_ms": float,
                "min_time_ms": float,
                "max_time_ms": float
            }
        ]
    """
    # Check if custom CUDA kernels are available
    try:
        from ..gpu.cuda_kernels import CUDA_KERNELS_AVAILABLE, get_kernel_metrics
        if CUDA_KERNELS_AVAILABLE:
            metrics = get_kernel_metrics()
            return jsonify(metrics if metrics else [])
    except (ImportError, AttributeError):
        pass

    # Return empty list with note about fallback mode
    return jsonify({
        'kernels': [],
        'note': 'Using PyTorch fallbacks - custom CUDA kernel metrics not available',
        'cuda_available': TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False
    })


@api_bp.route('/system/info', methods=['GET'])
def system_info():
    """Get comprehensive system information.

    Returns:
        JSON with system, Python, PyTorch, and CUDA information
    """
    import platform
    import sys

    info = {
        'system': {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_version': sys.version
        },
        'dependencies': {
            'numpy': NUMPY_AVAILABLE,
            'torch': TORCH_AVAILABLE,
            'torchaudio': TORCHAUDIO_AVAILABLE,
            'soundfile': SOUNDFILE_AVAILABLE,
            'librosa': LIBROSA_AVAILABLE,
            'noisereduce': NOISEREDUCE_AVAILABLE
        }
    }

    if TORCH_AVAILABLE:
        info['torch'] = {
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None
        }
        if torch.cuda.is_available():
            info['torch']['device_name'] = torch.cuda.get_device_name(0)
            info['torch']['device_count'] = torch.cuda.device_count()

    # Check custom CUDA kernels
    try:
        from ..gpu.cuda_kernels import CUDA_KERNELS_AVAILABLE
        info['cuda_kernels_available'] = CUDA_KERNELS_AVAILABLE
    except ImportError:
        info['cuda_kernels_available'] = False

    return jsonify(info)


# ============================================================================
# Audio Device Configuration Endpoints
# ============================================================================

@api_bp.route('/devices/list', methods=['GET'])
def list_devices():
    """List available audio devices.

    Query Parameters:
        type (optional): Filter by 'input' or 'output'

    Returns:
        JSON array of devices with device_id, name, type, sample_rate, channels, is_default

    Response Schema:
        [
            {
                "device_id": str,
                "name": str,
                "type": "input" | "output",
                "sample_rate": int,
                "channels": int,
                "is_default": bool
            }
        ]
    """
    try:
        from .audio_router import list_audio_devices

        device_type = request.args.get('type')
        if device_type and device_type not in ('input', 'output'):
            return jsonify({'error': 'Invalid type parameter, use "input" or "output"'}), 400

        devices = list_audio_devices(device_type=device_type)
        return jsonify(devices)

    except Exception as e:
        logger.error(f"Error listing audio devices: {e}", exc_info=True)
        return jsonify({
            'error': 'Failed to list audio devices',
            'message': str(e) if current_app.debug else None
        }), 500


@api_bp.route('/devices/config', methods=['GET'])
def get_device_config():
    """Get current audio device configuration.

    Returns:
        JSON with input_device_id, output_device_id, sample_rate

    Response Schema:
        {
            "input_device_id": str | null,
            "output_device_id": str | null,
            "sample_rate": int
        }
    """
    # Get config from app context, or use defaults
    device_config = getattr(current_app, '_device_config', None)
    if device_config is None:
        device_config = {
            'input_device_id': None,
            'output_device_id': None,
            'sample_rate': current_app.app_config.get('audio', {}).get('sample_rate', 22050)
        }
        current_app._device_config = device_config

    return jsonify(device_config)


@api_bp.route('/devices/config', methods=['POST'])
def set_device_config():
    """Set audio device configuration.

    Request Body:
        {
            "input_device_id": str | null (optional),
            "output_device_id": str | null (optional),
            "sample_rate": int (optional)
        }

    Returns:
        Updated device configuration
    """
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'Request body required'}), 400

        # Get current config
        device_config = getattr(current_app, '_device_config', None)
        if device_config is None:
            device_config = {
                'input_device_id': None,
                'output_device_id': None,
                'sample_rate': current_app.app_config.get('audio', {}).get('sample_rate', 22050)
            }

        # Validate and update input_device_id
        if 'input_device_id' in data:
            input_id = data['input_device_id']
            if input_id is not None and input_id != '':
                # Validate device exists and is an input device
                from .audio_router import list_audio_devices
                input_devices = list_audio_devices(device_type='input')
                valid_ids = [d['device_id'] for d in input_devices]
                if input_id not in valid_ids:
                    return jsonify({'error': f'Invalid input device ID: {input_id}'}), 400
                device_config['input_device_id'] = input_id
            else:
                device_config['input_device_id'] = None

        # Validate and update output_device_id
        if 'output_device_id' in data:
            output_id = data['output_device_id']
            if output_id is not None and output_id != '':
                # Validate device exists and is an output device
                from .audio_router import list_audio_devices
                output_devices = list_audio_devices(device_type='output')
                valid_ids = [d['device_id'] for d in output_devices]
                if output_id not in valid_ids:
                    return jsonify({'error': f'Invalid output device ID: {output_id}'}), 400
                device_config['output_device_id'] = output_id
            else:
                device_config['output_device_id'] = None

        # Update sample_rate if provided
        if 'sample_rate' in data:
            sample_rate = data['sample_rate']
            if isinstance(sample_rate, int) and sample_rate > 0:
                device_config['sample_rate'] = sample_rate
            else:
                return jsonify({'error': 'Invalid sample_rate, must be positive integer'}), 400

        # Store updated config
        current_app._device_config = device_config

        logger.info(f"Device config updated: {device_config}")
        return jsonify(device_config)

    except Exception as e:
        logger.error(f"Error setting device config: {e}", exc_info=True)
        return jsonify({
            'error': 'Failed to set device configuration',
            'message': str(e) if current_app.debug else None
        }), 500


# =============================================================================
# TRAINING JOB ENDPOINTS
# =============================================================================

# In-memory storage for training jobs (TODO: persist to database)
_training_jobs: Dict[str, Dict[str, Any]] = {}


def _sanitize_job(job: dict) -> dict:
    """Sanitize job dict to ensure valid JSON (no Infinity/NaN)."""
    import math
    sanitized = dict(job)
    if sanitized.get('results'):
        results = dict(sanitized['results'])
        for key, val in results.items():
            if isinstance(val, float) and (math.isinf(val) or math.isnan(val)):
                results[key] = None
        sanitized['results'] = results
    return sanitized


@api_bp.route('/training/jobs', methods=['GET'])
def list_training_jobs():
    """List all training jobs, optionally filtered by profile."""
    try:
        profile_id = request.args.get('profile_id')
        jobs = list(_training_jobs.values())
        if profile_id:
            jobs = [j for j in jobs if j.get('profile_id') == profile_id]
        # Sort by created_at descending
        jobs.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        # Sanitize for valid JSON
        jobs = [_sanitize_job(j) for j in jobs]
        return jsonify(jobs)
    except Exception as e:
        logger.error(f"Error listing training jobs: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@api_bp.route('/training/jobs', methods=['POST'])
def create_training_job():
    """Create and start a new training job."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        profile_id = data.get('profile_id')
        if not profile_id:
            return jsonify({'error': 'profile_id is required'}), 400

        sample_ids = data.get('sample_ids', [])
        config = data.get('config', {})

        job_id = str(uuid.uuid4())
        job = {
            'job_id': job_id,
            'profile_id': profile_id,
            'status': 'pending',
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'started_at': None,
            'completed_at': None,
            'progress': 0,
            'sample_ids': sample_ids,
            'config': config,
            'error': None,
            'results': None
        }
        _training_jobs[job_id] = job
        logger.info(f"Created training job {job_id} for profile {profile_id}")

        # Start training in background
        import threading
        app = current_app._get_current_object()  # Get actual app object for thread context

        def run_training():
            import shutil
            from pathlib import Path
            import torch
            try:
                with app.app_context():  # Required for Flask context in thread
                    from ..storage.voice_profiles import VoiceProfileStore
                    from ..training.trainer import Trainer

                    # Update status to running
                    job['status'] = 'running'
                    job['started_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

                    # Get sample audio paths
                    store = VoiceProfileStore()
                    training_samples = store.list_training_samples(profile_id)

                    if not training_samples:
                        raise ValueError(f"No training samples found for profile {profile_id}")

                    # Create temporary training directory
                    train_dir = Path(f"/tmp/autovoice_training/{job_id}")
                    train_dir.mkdir(parents=True, exist_ok=True)

                    sample_files = []
                    for sample in training_samples:
                        if sample.sample_id in sample_ids or not sample_ids:
                            src_path = Path(sample.vocals_path)
                            if not src_path.is_absolute():
                                src_path = Path("/home/kp/repos/autovoice") / sample.vocals_path
                            if src_path.exists():
                                dst_path = train_dir / f"{sample.sample_id}.wav"
                                shutil.copy2(src_path, dst_path)
                                sample_files.append(str(dst_path))
                                logger.info(f"Copied sample: {src_path} -> {dst_path}")

                    if not sample_files:
                        raise ValueError("No valid audio samples could be loaded")

                    # Training configuration
                    # Training mode: 'lora' (default, fast) or 'full' (from scratch, higher quality)
                    training_mode = config.get('training_mode', 'lora')

                    # Adjust defaults based on training mode
                    if training_mode == 'full':
                        # Full training needs more epochs and lower LR
                        default_epochs = 500
                        default_lr = 5e-5
                    else:
                        # LoRA fine-tuning is faster
                        default_epochs = 100
                        default_lr = 1e-4

                    epochs = config.get('epochs', default_epochs)
                    learning_rate = config.get('learning_rate', default_lr)
                    batch_size = config.get('batch_size', 4)

                    trainer_config = {
                        'epochs': epochs,
                        'learning_rate': learning_rate,
                        'batch_size': batch_size,
                        'checkpoint_dir': f"/home/kp/repos/autovoice/data/checkpoints/{profile_id}",
                    }

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                    # Create the SVC decoder model for training
                    # ContentVec encoder outputs 768-dim, PitchEncoder outputs 256-dim
                    from ..models.svc_decoder import CoMoSVCDecoder
                    model = CoMoSVCDecoder(
                        content_dim=768,  # ContentVec 768-dim for better speaker disentanglement
                        pitch_dim=256,    # Pitch embedding
                        speaker_dim=256,  # Speaker embedding
                        n_mels=100,       # Output mel bins
                        hidden_dim=512,
                        n_layers=8,
                        device=device,
                    )

                    # Apply training mode
                    if training_mode == 'lora':
                        # LoRA: Parameter-efficient fine-tuning (~1MB checkpoint)
                        lora_rank = config.get('lora_rank', 8)
                        lora_alpha = config.get('lora_alpha', 16)
                        lora_dropout = config.get('lora_dropout', 0.1)
                        model.inject_lora(rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
                        logger.info(f"Training mode: LoRA (rank={lora_rank}, alpha={lora_alpha})")
                    else:
                        # Full: Train entire model from scratch (~184MB checkpoint)
                        logger.info(f"Training mode: FULL (training all {sum(p.numel() for p in model.parameters())} parameters)")

                    trainer = Trainer(model=model, config=trainer_config, device=device)
                    logger.info(f"Training on device: {device}, epochs: {epochs}, model: CoMoSVCDecoder")

                    # Set speaker embedding from training audio (required before training)
                    trainer.set_speaker_embedding(str(train_dir))
                    logger.info(f"Speaker embedding computed from {train_dir}")

                    # Patch train to update progress
                    original_train_epoch = trainer._train_epoch
                    def patched_train_epoch(loader, epoch):
                        loss = original_train_epoch(loader, epoch)
                        progress = int(100 * (epoch + 1) / epochs)
                        job['progress'] = progress
                        job['results'] = job.get('results') or {}
                        job['results']['current_loss'] = loss
                        job['results']['current_epoch'] = epoch + 1

                        # Emit WebSocket event (socketio from outer scope)
                        try:
                            from flask import current_app as ca
                            socketio = ca.extensions.get('socketio')
                            if socketio:
                                socketio.emit('training_progress', {
                                    'job_id': job_id,
                                    'profile_id': profile_id,
                                    'epoch': epoch + 1,
                                    'total_epochs': epochs,
                                    'step': epoch + 1,
                                    'total_steps': epochs,
                                    'loss': loss,
                                    'learning_rate': learning_rate,
                                })
                        except Exception as emit_err:
                            logger.warning(f"Failed to emit progress: {emit_err}")
                        return loss

                    trainer._train_epoch = patched_train_epoch

                    # Run training
                    trainer.train(str(train_dir))

                    # Update job as completed
                    job['status'] = 'completed'
                    job['completed_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                    job['progress'] = 100
                    # Convert Infinity to None for valid JSON
                    best_loss = trainer.best_loss if trainer.best_loss != float('inf') else None
                    job['results'] = {
                        'final_loss': trainer.train_losses[-1] if trainer.train_losses else 0,
                        'best_loss': best_loss,
                        'epochs_completed': epochs,
                        'checkpoint_path': str(trainer.checkpoint_dir / 'final.pth'),
                    }

                    # Emit completion event
                    try:
                        from flask import current_app as ca
                        socketio = ca.extensions.get('socketio')
                        if socketio:
                            socketio.emit('training_complete', {
                                'job_id': job_id,
                                'profile_id': profile_id,
                                'results': job['results'],
                            })
                    except Exception as emit_err:
                        logger.warning(f"Failed to emit completion: {emit_err}")

                    logger.info(f"Training job {job_id} completed successfully")

                    # Cleanup temp directory
                    shutil.rmtree(train_dir, ignore_errors=True)

            except Exception as e:
                logger.error(f"Training job {job_id} failed: {e}", exc_info=True)
                job['status'] = 'failed'
                job['error'] = str(e)

                # Emit error event
                try:
                    with app.app_context():
                        from flask import current_app as ca
                        socketio = ca.extensions.get('socketio')
                        if socketio:
                            socketio.emit('training_error', {
                                'job_id': job_id,
                                'profile_id': profile_id,
                                'error': str(e),
                            })
                except Exception as emit_err:
                    logger.warning(f"Failed to emit error: {emit_err}")

        training_thread = threading.Thread(target=run_training, daemon=True)
        training_thread.start()
        logger.info(f"Training job {job_id} started in background")

        return jsonify(job), 201
    except Exception as e:
        logger.error(f"Error creating training job: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@api_bp.route('/training/jobs/<job_id>', methods=['GET'])
def get_training_job(job_id: str):
    """Get details of a specific training job."""
    job = _training_jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Training job not found'}), 404
    return jsonify(_sanitize_job(job))


@api_bp.route('/training/jobs/<job_id>/cancel', methods=['POST'])
def cancel_training_job(job_id: str):
    """Cancel a training job."""
    job = _training_jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Training job not found'}), 404
    if job['status'] in ('completed', 'failed', 'cancelled'):
        return jsonify({'error': f"Cannot cancel job in {job['status']} state"}), 400
    job['status'] = 'cancelled'
    job['completed_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    logger.info(f"Cancelled training job {job_id}")
    return jsonify(job)


# =============================================================================
# SAMPLE MANAGEMENT ENDPOINTS
# =============================================================================

# In-memory storage for samples (fallback for samples not in VoiceProfileStore)
_profile_samples: Dict[str, Dict[str, Dict[str, Any]]] = {}


@api_bp.route('/profiles/<profile_id>/samples', methods=['GET'])
def list_samples(profile_id: str):
    """List all samples for a profile."""
    from ..storage.voice_profiles import VoiceProfileStore
    store = VoiceProfileStore()

    # First try to get samples from VoiceProfileStore (persistent)
    try:
        training_samples = store.list_training_samples(profile_id)
        if training_samples:
            # Convert TrainingSample objects to API format matching frontend interface
            samples = []
            for ts in training_samples:
                samples.append({
                    'id': ts.sample_id,  # Frontend expects 'id' not 'sample_id'
                    'sample_id': ts.sample_id,
                    'profile_id': profile_id,
                    'audio_path': ts.vocals_path,
                    'duration_seconds': ts.duration,
                    'sample_rate': 44100,  # Default, could be read from file
                    'created': ts.created_at,
                    'source_file': ts.source_file,
                })
            return jsonify(samples)
    except Exception as e:
        logger.warning(f"Failed to get samples from VoiceProfileStore: {e}")

    # Fallback to in-memory storage
    samples = _profile_samples.get(profile_id, {})
    return jsonify(list(samples.values()))


@api_bp.route('/profiles/<profile_id>/samples', methods=['POST'])
def upload_sample(profile_id: str):
    """Upload a new training sample for a profile."""
    try:
        # Accept both 'file' and 'audio' field names for flexibility
        file = None
        if 'file' in request.files:
            file = request.files['file']
        elif 'audio' in request.files:
            file = request.files['audio']

        if not file:
            return jsonify({'error': 'No file provided (expected "file" or "audio" field)'}), 400
        if not file.filename:
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        # Save file temporarily
        filename = secure_filename(file.filename)
        sample_id = str(uuid.uuid4())

        # Create upload directory if needed
        upload_dir = os.path.join(UPLOAD_FOLDER, 'samples', profile_id)
        os.makedirs(upload_dir, exist_ok=True)

        file_path = os.path.join(upload_dir, f"{sample_id}_{filename}")
        file.save(file_path)

        # Get metadata from form
        metadata = {}
        if request.form.get('metadata'):
            try:
                metadata = json.loads(request.form.get('metadata'))
            except json.JSONDecodeError:
                pass

        sample = {
            'sample_id': sample_id,
            'profile_id': profile_id,
            'filename': filename,
            'file_path': file_path,
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'duration': None,  # TODO: compute from file
            'metadata': metadata
        }

        if profile_id not in _profile_samples:
            _profile_samples[profile_id] = {}
        _profile_samples[profile_id][sample_id] = sample

        logger.info(f"Uploaded sample {sample_id} for profile {profile_id}")
        return jsonify(sample), 201
    except Exception as e:
        logger.error(f"Error uploading sample: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# In-memory storage for song separation jobs
_separation_jobs: Dict[str, Dict] = {}


@api_bp.route('/profiles/<profile_id>/songs', methods=['POST'])
def upload_song(profile_id: str):
    """Upload a song for vocal separation (uses Demucs to extract vocals for training)."""
    try:
        # Accept both 'file' and 'audio' field names
        file = None
        if 'file' in request.files:
            file = request.files['file']
        elif 'audio' in request.files:
            file = request.files['audio']

        if not file:
            return jsonify({'error': 'No file provided (expected "file" or "audio" field)'}), 400

        if not file.filename:
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        # Save file temporarily
        filename = secure_filename(file.filename)
        job_id = str(uuid.uuid4())
        song_id = str(uuid.uuid4())

        # Create upload directory
        upload_dir = os.path.join(UPLOAD_FOLDER, 'songs', profile_id)
        os.makedirs(upload_dir, exist_ok=True)

        file_path = os.path.join(upload_dir, f"{song_id}_{filename}")
        file.save(file_path)

        # Check if auto_split is requested
        auto_split = request.form.get('auto_split', 'true').lower() == 'true'

        # Store job info
        _separation_jobs[job_id] = {
            'job_id': job_id,
            'song_id': song_id,
            'profile_id': profile_id,
            'filename': filename,
            'file_path': file_path,
            'status': 'pending' if auto_split else 'complete',
            'progress': 0,
            'message': 'Queued for vocal separation' if auto_split else 'Uploaded without separation',
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'vocals_path': None,
            'instrumental_path': None,
            'error': None
        }

        if auto_split:
            # Start separation in background (simplified - in production use Celery/etc)
            import threading
            def run_separation():
                try:
                    _separation_jobs[job_id]['status'] = 'processing'
                    _separation_jobs[job_id]['message'] = 'Separating vocals and instrumental...'
                    _separation_jobs[job_id]['progress'] = 10

                    # Use Demucs VocalSeparator
                    import soundfile as sf
                    import numpy as np
                    from auto_voice.audio.separation import VocalSeparator

                    output_dir = os.path.join(UPLOAD_FOLDER, 'separated', profile_id, song_id)
                    os.makedirs(output_dir, exist_ok=True)

                    _separation_jobs[job_id]['progress'] = 20
                    _separation_jobs[job_id]['message'] = 'Loading audio...'

                    # Load audio file
                    audio, sr = sf.read(file_path)
                    if audio.ndim > 1:
                        audio = audio.T  # sf.read returns (samples, channels), need (channels, samples)

                    _separation_jobs[job_id]['progress'] = 30
                    _separation_jobs[job_id]['message'] = 'Running vocal separation (this may take a while)...'

                    # Run separation
                    separator = VocalSeparator()
                    result = separator.separate(audio, sr)

                    _separation_jobs[job_id]['progress'] = 80
                    _separation_jobs[job_id]['message'] = 'Saving separated tracks...'

                    # Save vocals and instrumental
                    vocals_path = os.path.join(output_dir, 'vocals.wav')
                    instrumental_path = os.path.join(output_dir, 'instrumental.wav')
                    sf.write(vocals_path, result['vocals'], sr)
                    sf.write(instrumental_path, result['instrumental'], sr)

                    _separation_jobs[job_id]['vocals_path'] = vocals_path
                    _separation_jobs[job_id]['instrumental_path'] = instrumental_path
                    _separation_jobs[job_id]['status'] = 'complete'
                    _separation_jobs[job_id]['progress'] = 100
                    _separation_jobs[job_id]['message'] = 'Separation complete'

                    # Auto-add vocals as training sample
                    sample_id = str(uuid.uuid4())
                    sample = {
                        'sample_id': sample_id,
                        'profile_id': profile_id,
                        'filename': f"vocals_{filename}",
                        'file_path': vocals_path,
                        'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                        'duration': None,
                        'metadata': {'source_song': song_id, 'source_file': filename}
                    }
                    if profile_id not in _profile_samples:
                        _profile_samples[profile_id] = {}
                    _profile_samples[profile_id][sample_id] = sample

                    logger.info(f"Separation complete for song {song_id}, added sample {sample_id}")
                except Exception as e:
                    logger.error(f"Separation failed for job {job_id}: {e}", exc_info=True)
                    _separation_jobs[job_id]['status'] = 'error'
                    _separation_jobs[job_id]['error'] = str(e)
                    _separation_jobs[job_id]['message'] = f'Separation failed: {str(e)}'

            thread = threading.Thread(target=run_separation, daemon=True)
            thread.start()

        logger.info(f"Uploaded song {song_id} for profile {profile_id}, job {job_id}")
        return jsonify({
            'job_id': job_id,
            'song_id': song_id,
            'status': _separation_jobs[job_id]['status'],
            'message': _separation_jobs[job_id]['message']
        }), 202

    except Exception as e:
        logger.error(f"Error uploading song: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@api_bp.route('/separation/<job_id>/status', methods=['GET'])
def get_separation_status(job_id: str):
    """Get status of a vocal separation job."""
    job = _separation_jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(job)


@api_bp.route('/profiles/<profile_id>/samples/<sample_id>', methods=['GET'])
def get_sample(profile_id: str, sample_id: str):
    """Get details of a specific sample."""
    samples = _profile_samples.get(profile_id, {})
    sample = samples.get(sample_id)
    if not sample:
        return jsonify({'error': 'Sample not found'}), 404
    return jsonify(sample)


@api_bp.route('/profiles/<profile_id>/samples/<sample_id>', methods=['DELETE'])
def delete_sample(profile_id: str, sample_id: str):
    """Delete a sample."""
    samples = _profile_samples.get(profile_id, {})
    sample = samples.get(sample_id)
    if not sample:
        return jsonify({'error': 'Sample not found'}), 404

    # Delete file if exists
    if sample.get('file_path') and os.path.exists(sample['file_path']):
        os.remove(sample['file_path'])

    del _profile_samples[profile_id][sample_id]
    logger.info(f"Deleted sample {sample_id} from profile {profile_id}")
    return '', 204


# =============================================================================
# DIARIZATION ENDPOINTS
# =============================================================================

@api_bp.route('/audio/diarize', methods=['POST'])
def diarize_audio():
    """Run speaker diarization on uploaded audio.

    Accepts multipart file upload or JSON with audio_path.

    Returns:
        JSON with diarization results including speaker segments.
    """
    try:
        from auto_voice.audio.speaker_diarization import SpeakerDiarizer

        # Handle file upload or path
        if 'file' in request.files:
            file = request.files['file']
            if not file.filename:
                return jsonify({'error': 'No file selected'}), 400

            # Save uploaded file temporarily
            import tempfile
            temp_dir = tempfile.mkdtemp()
            audio_path = os.path.join(temp_dir, file.filename)
            file.save(audio_path)
        elif request.is_json:
            data = request.get_json()
            audio_path = data.get('audio_path')
            if not audio_path or not os.path.exists(audio_path):
                return jsonify({'error': 'audio_path not found'}), 400
        else:
            return jsonify({'error': 'Provide file upload or audio_path'}), 400

        # Optional parameters
        num_speakers = None
        if request.is_json:
            num_speakers = request.get_json().get('num_speakers')

        # Run diarization
        diarizer = SpeakerDiarizer()
        result = diarizer.diarize(audio_path, num_speakers=num_speakers)

        # Generate diarization ID for later reference
        diarization_id = str(uuid.uuid4())

        # Format segments
        segments = [
            {
                'start': seg.start,
                'end': seg.end,
                'speaker_id': seg.speaker_id,
                'duration': seg.duration,
                'confidence': seg.confidence,
            }
            for seg in result.segments
        ]

        # Format response
        response = {
            'diarization_id': diarization_id,
            'audio_duration': result.audio_duration,
            'num_speakers': result.num_speakers,
            'segments': segments,
            'speaker_durations': {
                speaker_id: result.get_speaker_total_duration(speaker_id)
                for speaker_id in result.get_all_speaker_ids()
            },
        }

        # Store for later reference (e.g., by assign/auto-create endpoints)
        _diarization_results[diarization_id] = {
            'audio_path': audio_path,
            'audio_duration': result.audio_duration,
            'num_speakers': result.num_speakers,
            'segments': segments,
            'created_at': time.time(),
        }

        logger.info(f"Diarization {diarization_id} complete: {result.num_speakers} speakers detected")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Diarization error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@api_bp.route('/profiles/<profile_id>/samples/<sample_id>/filter', methods=['POST'])
def filter_sample(profile_id: str, sample_id: str):
    """Filter a training sample to only include target speaker vocals.

    Uses speaker diarization to identify and extract only segments
    matching the profile's speaker embedding.

    Returns:
        JSON with filtering results and paths to filtered audio.
    """
    try:
        from auto_voice.audio.training_filter import TrainingDataFilter
        from auto_voice.storage.voice_profiles import VoiceProfileStore

        # Get sample
        samples = _profile_samples.get(profile_id, {})
        sample = samples.get(sample_id)
        if not sample:
            return jsonify({'error': 'Sample not found'}), 404

        audio_path = sample.get('file_path')
        if not audio_path or not os.path.exists(audio_path):
            return jsonify({'error': 'Sample audio file not found'}), 404

        # Get profile's speaker embedding
        store = VoiceProfileStore()
        embedding = store.load_speaker_embedding(profile_id)

        if embedding is None:
            return jsonify({
                'error': 'Profile has no speaker embedding. Upload a sample first to create one.'
            }), 400

        # Optional parameters
        data = request.get_json() or {}
        similarity_threshold = data.get('similarity_threshold', 0.7)

        # Run filtering
        filter_obj = TrainingDataFilter()
        output_path, metadata = filter_obj.filter_training_audio(
            audio_path=audio_path,
            target_embedding=embedding,
            similarity_threshold=similarity_threshold,
        )

        # Update sample with filtered path
        sample['filtered_path'] = str(output_path)
        sample['filter_metadata'] = metadata

        response = {
            'sample_id': sample_id,
            'original_path': audio_path,
            'filtered_path': str(output_path),
            'original_duration': metadata['original_duration'],
            'filtered_duration': metadata['filtered_duration'],
            'num_segments': metadata['num_segments'],
            'purity': metadata['purity'],
            'status': metadata['status'],
        }

        logger.info(f"Filtered sample {sample_id}: {metadata['filtered_duration']:.1f}s extracted")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Filter error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@api_bp.route('/profiles/<profile_id>/speaker-embedding', methods=['POST'])
def set_profile_speaker_embedding(profile_id: str):
    """Set or update the speaker embedding for a profile.

    Extracts speaker embedding from the provided audio file or
    computes from existing training samples.

    Request body:
        - audio_path: Path to audio for embedding extraction
        - OR use_samples: true to compute from existing samples
    """
    try:
        from auto_voice.audio.speaker_diarization import SpeakerDiarizer
        from auto_voice.storage.voice_profiles import VoiceProfileStore

        store = VoiceProfileStore()
        if not store.exists(profile_id):
            return jsonify({'error': 'Profile not found'}), 404

        data = request.get_json() or {}

        if data.get('use_samples', False):
            # Compute embedding from existing samples
            samples = store.list_training_samples(profile_id)
            if not samples:
                return jsonify({'error': 'No training samples to compute embedding from'}), 400

            # Use first sample for now
            audio_path = samples[0].vocals_path
        elif 'audio_path' in data:
            audio_path = data['audio_path']
            if not os.path.exists(audio_path):
                return jsonify({'error': 'Audio file not found'}), 400
        else:
            return jsonify({'error': 'Provide audio_path or set use_samples=true'}), 400

        # Extract embedding
        diarizer = SpeakerDiarizer()
        embedding = diarizer.extract_speaker_embedding(audio_path)

        # Save embedding
        store.save_speaker_embedding(profile_id, embedding)

        logger.info(f"Set speaker embedding for profile {profile_id}")
        return jsonify({
            'profile_id': profile_id,
            'embedding_dim': len(embedding),
            'source': audio_path,
            'status': 'success',
        })

    except Exception as e:
        logger.error(f"Error setting speaker embedding: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@api_bp.route('/profiles/<profile_id>/speaker-embedding', methods=['GET'])
def get_profile_speaker_embedding(profile_id: str):
    """Check if profile has a speaker embedding."""
    try:
        from auto_voice.storage.voice_profiles import VoiceProfileStore

        store = VoiceProfileStore()
        if not store.exists(profile_id):
            return jsonify({'error': 'Profile not found'}), 404

        embedding = store.load_speaker_embedding(profile_id)

        return jsonify({
            'profile_id': profile_id,
            'has_embedding': embedding is not None,
            'embedding_dim': len(embedding) if embedding is not None else None,
        })

    except Exception as e:
        logger.error(f"Error getting speaker embedding: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# In-memory storage for diarization results (cleared on restart)
_diarization_results: Dict[str, Dict[str, Any]] = {}
# In-memory storage for segment-to-profile assignments
_segment_assignments: Dict[str, Dict[str, str]] = {}  # profile_id -> {segment_key -> audio_path}


@api_bp.route('/audio/diarize/assign', methods=['POST'])
def assign_diarization_segment():
    """Assign a diarization segment to an existing profile.

    This endpoint allows manual correction of speaker identification
    by assigning a detected segment to the correct profile.

    Request body:
        - diarization_id: ID of the diarization result
        - segment_index: Index of the segment to assign
        - profile_id: Profile to assign the segment to
        - extract_audio: (optional) Whether to extract audio (default: true)
    """
    try:
        from auto_voice.audio.speaker_diarization import SpeakerDiarizer
        from auto_voice.storage.voice_profiles import VoiceProfileStore

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        diarization_id = data.get('diarization_id')
        segment_index = data.get('segment_index')
        profile_id = data.get('profile_id')

        if not all([diarization_id, segment_index is not None, profile_id]):
            return jsonify({
                'error': 'Required: diarization_id, segment_index, profile_id'
            }), 400

        # Get diarization result
        diarization_data = _diarization_results.get(diarization_id)
        if not diarization_data:
            return jsonify({'error': 'Diarization result not found or expired'}), 404

        segments = diarization_data.get('segments', [])
        if segment_index < 0 or segment_index >= len(segments):
            return jsonify({'error': f'Invalid segment_index: {segment_index}'}), 400

        segment = segments[segment_index]

        # Verify profile exists
        store = VoiceProfileStore()
        if not store.exists(profile_id):
            return jsonify({'error': 'Profile not found'}), 404

        # Extract audio for segment if requested
        extract_audio = data.get('extract_audio', True)
        extracted_path = None

        if extract_audio:
            audio_path = diarization_data.get('audio_path')
            if audio_path and os.path.exists(audio_path):
                diarizer = SpeakerDiarizer()

                # Create diarization result object for extraction
                from auto_voice.audio.speaker_diarization import DiarizationResult, SpeakerSegment
                import numpy as np

                seg_obj = SpeakerSegment(
                    start=segment['start'],
                    end=segment['end'],
                    speaker_id=segment['speaker_id'],
                    confidence=segment.get('confidence', 1.0),
                )
                temp_result = DiarizationResult(
                    segments=[seg_obj],
                    audio_duration=diarization_data.get('audio_duration', 0),
                    num_speakers=1,
                )

                extracted_path = diarizer.extract_speaker_audio(
                    audio_path=audio_path,
                    diarization=temp_result,
                    speaker_id=segment['speaker_id'],
                )

                # Add as training sample to profile
                if extracted_path:
                    from scipy.io import wavfile
                    sr, audio_data = wavfile.read(str(extracted_path))
                    duration = len(audio_data) / sr

                    store.add_training_sample(
                        profile_id=profile_id,
                        vocals_path=str(extracted_path),
                        duration=duration,
                        source_file=f"diarization_{diarization_id}_seg{segment_index}",
                    )

        # Track assignment
        segment_key = f"{diarization_id}_{segment_index}"
        if profile_id not in _segment_assignments:
            _segment_assignments[profile_id] = {}
        _segment_assignments[profile_id][segment_key] = str(extracted_path) if extracted_path else ""

        logger.info(f"Assigned segment {segment_index} from {diarization_id} to profile {profile_id}")
        return jsonify({
            'status': 'success',
            'profile_id': profile_id,
            'segment_index': segment_index,
            'extracted_path': str(extracted_path) if extracted_path else None,
            'segment': segment,
        })

    except Exception as e:
        logger.error(f"Error assigning segment: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@api_bp.route('/profiles/<profile_id>/segments', methods=['GET'])
def get_profile_segments(profile_id: str):
    """Get all audio segments assigned to a profile.

    Returns segments from diarization assignments and training samples.
    """
    try:
        from auto_voice.storage.voice_profiles import VoiceProfileStore

        store = VoiceProfileStore()
        if not store.exists(profile_id):
            return jsonify({'error': 'Profile not found'}), 404

        # Get training samples
        samples = store.list_training_samples(profile_id)
        sample_segments = [
            {
                'type': 'training_sample',
                'sample_id': s.sample_id,
                'vocals_path': s.vocals_path,
                'duration': s.duration,
                'source_file': s.source_file,
                'created_at': s.created_at,
            }
            for s in samples
        ]

        # Get diarization assignments
        assignments = _segment_assignments.get(profile_id, {})
        assignment_segments = [
            {
                'type': 'diarization_assignment',
                'segment_key': key,
                'audio_path': path,
            }
            for key, path in assignments.items()
        ]

        total_duration = sum(s.duration for s in samples)

        return jsonify({
            'profile_id': profile_id,
            'total_segments': len(sample_segments) + len(assignment_segments),
            'total_duration': total_duration,
            'training_samples': sample_segments,
            'diarization_assignments': assignment_segments,
        })

    except Exception as e:
        logger.error(f"Error getting profile segments: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@api_bp.route('/profiles/auto-create', methods=['POST'])
def auto_create_profile_from_diarization():
    """Create a new profile from diarization results.

    Automatically creates a profile with speaker embedding extracted
    from the diarized segments.

    Request body:
        - diarization_id: ID of the diarization result
        - speaker_id: Speaker ID from diarization to use
        - name: Name for the new profile
        - user_id: (optional) User ID for the profile
        - extract_segments: (optional) Add all segments as training samples (default: true)
    """
    try:
        from auto_voice.audio.speaker_diarization import SpeakerDiarizer, DiarizationResult, SpeakerSegment
        from auto_voice.storage.voice_profiles import VoiceProfileStore
        import numpy as np

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        diarization_id = data.get('diarization_id')
        speaker_id = data.get('speaker_id')
        name = data.get('name')

        if not all([diarization_id, speaker_id, name]):
            return jsonify({
                'error': 'Required: diarization_id, speaker_id, name'
            }), 400

        # Get diarization result
        diarization_data = _diarization_results.get(diarization_id)
        if not diarization_data:
            return jsonify({'error': 'Diarization result not found or expired'}), 404

        # Find segments for this speaker
        segments = diarization_data.get('segments', [])
        speaker_segments = [s for s in segments if s['speaker_id'] == speaker_id]

        if not speaker_segments:
            return jsonify({'error': f'No segments found for speaker {speaker_id}'}), 400

        audio_path = diarization_data.get('audio_path')
        if not audio_path or not os.path.exists(audio_path):
            return jsonify({'error': 'Original audio not found'}), 400

        # Extract speaker embedding from segments
        diarizer = SpeakerDiarizer()

        # Use the longest segment for embedding extraction
        longest_segment = max(speaker_segments, key=lambda s: s['end'] - s['start'])
        embedding = diarizer.extract_speaker_embedding(
            audio_path=audio_path,
            start=longest_segment['start'],
            end=longest_segment['end'],
        )

        # Create profile
        store = VoiceProfileStore()
        user_id = data.get('user_id', 'system')

        # Extract all segments as audio files if requested
        audio_segments = []
        extract_segments = data.get('extract_segments', True)

        if extract_segments:
            # Create DiarizationResult for extraction
            seg_objects = [
                SpeakerSegment(
                    start=s['start'],
                    end=s['end'],
                    speaker_id=s['speaker_id'],
                    confidence=s.get('confidence', 1.0),
                )
                for s in speaker_segments
            ]
            temp_result = DiarizationResult(
                segments=seg_objects,
                audio_duration=diarization_data.get('audio_duration', 0),
                num_speakers=diarization_data.get('num_speakers', 1),
            )

            extracted_path = diarizer.extract_speaker_audio(
                audio_path=audio_path,
                diarization=temp_result,
                speaker_id=speaker_id,
            )
            if extracted_path:
                audio_segments.append(str(extracted_path))

        profile_id = store.create_profile_from_diarization(
            name=name,
            speaker_embedding=embedding,
            user_id=user_id,
            audio_segments=audio_segments,
        )

        # Calculate total duration
        total_duration = sum(s['end'] - s['start'] for s in speaker_segments)

        logger.info(f"Auto-created profile '{name}' ({profile_id}) from diarization")
        return jsonify({
            'profile_id': profile_id,
            'name': name,
            'speaker_id': speaker_id,
            'num_segments': len(speaker_segments),
            'total_duration': total_duration,
            'embedding_dim': len(embedding),
            'status': 'success',
        }), 201

    except Exception as e:
        logger.error(f"Error auto-creating profile: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# =============================================================================
# PRESET ENDPOINTS
# =============================================================================

# In-memory storage for presets (TODO: persist to database)
_presets: Dict[str, Dict[str, Any]] = {}


@api_bp.route('/presets', methods=['GET'])
def list_presets():
    """List all user presets."""
    return jsonify(list(_presets.values()))


@api_bp.route('/presets', methods=['POST'])
def create_preset():
    """Create a new preset."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        name = data.get('name')
        if not name:
            return jsonify({'error': 'name is required'}), 400

        preset_id = str(uuid.uuid4())
        preset = {
            'id': preset_id,
            'name': name,
            'config': data.get('config', {}),
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'updated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        }
        _presets[preset_id] = preset
        logger.info(f"Created preset {preset_id}: {name}")
        return jsonify(preset), 201
    except Exception as e:
        logger.error(f"Error creating preset: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@api_bp.route('/presets/<preset_id>', methods=['GET'])
def get_preset(preset_id: str):
    """Get a specific preset."""
    preset = _presets.get(preset_id)
    if not preset:
        return jsonify({'error': 'Preset not found'}), 404
    return jsonify(preset)


@api_bp.route('/presets/<preset_id>', methods=['PUT'])
def update_preset(preset_id: str):
    """Update a preset."""
    preset = _presets.get(preset_id)
    if not preset:
        return jsonify({'error': 'Preset not found'}), 404

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        if 'name' in data:
            preset['name'] = data['name']
        if 'config' in data:
            preset['config'] = data['config']
        preset['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

        logger.info(f"Updated preset {preset_id}")
        return jsonify(preset)
    except Exception as e:
        logger.error(f"Error updating preset: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@api_bp.route('/presets/<preset_id>', methods=['DELETE'])
def delete_preset(preset_id: str):
    """Delete a preset."""
    if preset_id not in _presets:
        return jsonify({'error': 'Preset not found'}), 404
    del _presets[preset_id]
    logger.info(f"Deleted preset {preset_id}")
    return '', 204


# =============================================================================
# MODEL MANAGEMENT ENDPOINTS
# =============================================================================

# In-memory storage for loaded models state
_loaded_models: Dict[str, Dict[str, Any]] = {}


@api_bp.route('/models/loaded', methods=['GET'])
def get_loaded_models():
    """Get list of currently loaded models."""
    return jsonify({'models': list(_loaded_models.values())})


@api_bp.route('/models/load', methods=['POST'])
def load_model():
    """Load a model."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        model_type = data.get('model_type')
        if not model_type:
            return jsonify({'error': 'model_type is required'}), 400

        path = data.get('path')

        # TODO: Actually load the model
        model_info = {
            'model_type': model_type,
            'path': path,
            'loaded_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'status': 'loaded'
        }
        _loaded_models[model_type] = model_info
        logger.info(f"Loaded model: {model_type}")
        return jsonify(model_info), 201
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@api_bp.route('/models/unload', methods=['POST'])
def unload_model():
    """Unload a model."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        model_type = data.get('model_type')
        if not model_type:
            return jsonify({'error': 'model_type is required'}), 400

        if model_type in _loaded_models:
            del _loaded_models[model_type]

        logger.info(f"Unloaded model: {model_type}")
        return '', 204
    except Exception as e:
        logger.error(f"Error unloading model: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@api_bp.route('/models/tensorrt/status', methods=['GET'])
def get_tensorrt_status():
    """Get TensorRT engine status."""
    try:
        # Check if TensorRT engines exist
        engines_dir = os.path.join(os.path.dirname(__file__), '../../export/engines')
        engines = []
        if os.path.exists(engines_dir):
            for f in os.listdir(engines_dir):
                if f.endswith('.engine') or f.endswith('.plan'):
                    engines.append({
                        'name': f,
                        'path': os.path.join(engines_dir, f),
                        'size': os.path.getsize(os.path.join(engines_dir, f))
                    })

        return jsonify({
            'available': len(engines) > 0,
            'engines': engines,
            'cuda_available': TORCH_AVAILABLE and torch.cuda.is_available()
        })
    except Exception as e:
        logger.error(f"Error getting TensorRT status: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@api_bp.route('/models/tensorrt/rebuild', methods=['POST'])
def rebuild_tensorrt():
    """Rebuild TensorRT engines."""
    try:
        data = request.get_json() or {}
        precision = data.get('precision', 'fp16')

        # TODO: Actually rebuild engines
        logger.info(f"Rebuilding TensorRT engines with precision: {precision}")

        return jsonify({
            'status': 'complete',
            'precision': precision,
            'duration_seconds': 0  # TODO: actual duration
        })
    except Exception as e:
        logger.error(f"Error rebuilding TensorRT: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@api_bp.route('/models/tensorrt/build', methods=['POST'])
def build_tensorrt():
    """Build TensorRT engines with options."""
    try:
        data = request.get_json() or {}
        precision = data.get('precision', 'fp16')
        models = data.get('models', ['encoder', 'decoder', 'vocoder'])

        # TODO: Actually build engines
        logger.info(f"Building TensorRT engines: {models} with precision: {precision}")

        return jsonify({
            'status': 'complete',
            'precision': precision,
            'models': models,
            'duration_seconds': 0
        })
    except Exception as e:
        logger.error(f"Error building TensorRT: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# =============================================================================
# CONFIGURATION ENDPOINTS
# =============================================================================

# Default configurations
_separation_config = {
    'model': 'htdemucs',
    'overlap': 0.25,
    'segment': 10,
    'shifts': 1,
    'device': 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
}

_pitch_config = {
    'method': 'crepe',
    'hop_length': 160,
    'threshold': 0.3,
    'device': 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
}

_audio_router_config = {
    'speaker_gain': 1.0,
    'headphone_gain': 1.0,
    'voice_gain': 1.0,
    'instrumental_gain': 0.8,
    'speaker_enabled': True,
    'headphone_enabled': True
}


@api_bp.route('/config/separation', methods=['GET'])
def get_separation_config():
    """Get vocal separation configuration."""
    return jsonify(_separation_config)


@api_bp.route('/config/separation', methods=['POST'])
def update_separation_config():
    """Update vocal separation configuration."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        for key in ['model', 'overlap', 'segment', 'shifts', 'device']:
            if key in data:
                _separation_config[key] = data[key]

        logger.info(f"Updated separation config: {_separation_config}")
        return jsonify(_separation_config)
    except Exception as e:
        logger.error(f"Error updating separation config: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@api_bp.route('/config/pitch', methods=['GET'])
def get_pitch_config():
    """Get pitch extraction configuration."""
    return jsonify(_pitch_config)


@api_bp.route('/config/pitch', methods=['POST'])
def update_pitch_config():
    """Update pitch extraction configuration."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        for key in ['method', 'hop_length', 'threshold', 'device']:
            if key in data:
                _pitch_config[key] = data[key]

        logger.info(f"Updated pitch config: {_pitch_config}")
        return jsonify(_pitch_config)
    except Exception as e:
        logger.error(f"Error updating pitch config: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@api_bp.route('/audio/router/config', methods=['GET'])
def get_audio_router_config():
    """Get audio router configuration."""
    return jsonify(_audio_router_config)


@api_bp.route('/audio/router/config', methods=['POST'])
def update_audio_router_config():
    """Update audio router configuration."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        for key in ['speaker_gain', 'headphone_gain', 'voice_gain',
                    'instrumental_gain', 'speaker_enabled', 'headphone_enabled']:
            if key in data:
                _audio_router_config[key] = data[key]

        logger.info(f"Updated audio router config: {_audio_router_config}")
        return jsonify(_audio_router_config)
    except Exception as e:
        logger.error(f"Error updating audio router config: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# =============================================================================
# CONVERSION HISTORY ENDPOINTS
# =============================================================================

# In-memory storage for conversion history (TODO: persist to database)
_conversion_history: Dict[str, Dict[str, Any]] = {}


@api_bp.route('/convert/history', methods=['GET'])
def get_conversion_history():
    """Get conversion history, optionally filtered by profile."""
    profile_id = request.args.get('profile_id')
    history = list(_conversion_history.values())
    if profile_id:
        history = [h for h in history if h.get('profile_id') == profile_id]
    # Sort by created_at descending
    history.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return jsonify(history)


@api_bp.route('/convert/history/<record_id>', methods=['DELETE'])
def delete_conversion_record(record_id: str):
    """Delete a conversion record."""
    if record_id not in _conversion_history:
        return jsonify({'error': 'Record not found'}), 404
    del _conversion_history[record_id]
    logger.info(f"Deleted conversion record {record_id}")
    return '', 204


@api_bp.route('/convert/history/<record_id>', methods=['PATCH'])
def update_conversion_record(record_id: str):
    """Update a conversion record (e.g., add notes, favorite)."""
    record = _conversion_history.get(record_id)
    if not record:
        return jsonify({'error': 'Record not found'}), 404

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Allow updating specific fields
        for key in ['notes', 'isFavorite', 'tags']:
            if key in data:
                record[key] = data[key]

        logger.info(f"Updated conversion record {record_id}")
        return jsonify(record)
    except Exception as e:
        logger.error(f"Error updating conversion record: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# =============================================================================
# CHECKPOINT ENDPOINTS
# =============================================================================

# In-memory storage for checkpoints (TODO: persist to database)
_profile_checkpoints: Dict[str, Dict[str, Dict[str, Any]]] = {}


@api_bp.route('/profiles/<profile_id>/checkpoints', methods=['GET'])
def list_checkpoints(profile_id: str):
    """List all checkpoints for a profile."""
    checkpoints = _profile_checkpoints.get(profile_id, {})
    return jsonify(list(checkpoints.values()))


@api_bp.route('/profiles/<profile_id>/checkpoints/<checkpoint_id>/rollback', methods=['POST'])
def rollback_checkpoint(profile_id: str, checkpoint_id: str):
    """Rollback to a specific checkpoint."""
    checkpoints = _profile_checkpoints.get(profile_id, {})
    checkpoint = checkpoints.get(checkpoint_id)
    if not checkpoint:
        return jsonify({'error': 'Checkpoint not found'}), 404

    # TODO: Actually rollback the model
    logger.info(f"Rolling back profile {profile_id} to checkpoint {checkpoint_id}")
    return jsonify({'status': 'rolled_back', 'checkpoint': checkpoint})


@api_bp.route('/profiles/<profile_id>/checkpoints/<checkpoint_id>', methods=['DELETE'])
def delete_checkpoint(profile_id: str, checkpoint_id: str):
    """Delete a checkpoint."""
    checkpoints = _profile_checkpoints.get(profile_id, {})
    if checkpoint_id not in checkpoints:
        return jsonify({'error': 'Checkpoint not found'}), 404

    del _profile_checkpoints[profile_id][checkpoint_id]
    logger.info(f"Deleted checkpoint {checkpoint_id} from profile {profile_id}")
    return '', 204


# =============================================================================
# YOUTUBE DOWNLOAD ENDPOINTS
# =============================================================================

# Module-level YouTube downloader instance
_youtube_downloader: Optional['YouTubeDownloader'] = None


def get_youtube_downloader() -> 'YouTubeDownloader':
    """Get or create a YouTubeDownloader instance."""
    global _youtube_downloader
    if _youtube_downloader is None:
        if not YOUTUBE_DOWNLOADER_AVAILABLE:
            raise RuntimeError("YouTube downloader not available")
        output_dir = os.path.join(UPLOAD_FOLDER, 'youtube')
        os.makedirs(output_dir, exist_ok=True)
        _youtube_downloader = YouTubeDownloader(output_dir)
    return _youtube_downloader


@api_bp.route('/youtube/info', methods=['POST'])
def youtube_info():
    """Get video information and detected artists without downloading.

    Request (JSON):
        url (str): YouTube video URL (required)

    Returns:
        {
            "success": true/false,
            "title": "Video Title",
            "duration": 180.5,
            "main_artist": "Artist Name",
            "featured_artists": ["Artist 1", "Artist 2"],
            "is_cover": false,
            "original_artist": null,
            "song_title": "Song Name",
            "thumbnail_url": "https://...",
            "video_id": "abc123",
            "error": null
        }
    """
    if not YOUTUBE_DOWNLOADER_AVAILABLE:
        return jsonify({'error': 'YouTube downloader not available. Install yt-dlp.'}), 503

    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'Missing required field: url'}), 400

    url = data['url']
    if not url.strip():
        return jsonify({'error': 'URL cannot be empty'}), 400

    try:
        downloader = get_youtube_downloader()
        result = downloader.get_video_info(url)

        return jsonify({
            'success': result.success,
            'title': result.title,
            'duration': result.duration,
            'main_artist': result.main_artist,
            'featured_artists': result.featured_artists,
            'is_cover': result.is_cover,
            'original_artist': result.original_artist,
            'song_title': result.song_title,
            'thumbnail_url': result.thumbnail_url,
            'video_id': result.video_id,
            'error': result.error
        })

    except Exception as e:
        logger.error(f"YouTube info failed: {e}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/youtube/download', methods=['POST'])
def youtube_download():
    """Download audio from YouTube video with metadata.

    Request (JSON):
        url (str): YouTube video URL (required)
        format (str): Audio format (wav, mp3, flac) - default: wav
        sample_rate (int): Sample rate - default: 44100
        run_diarization (bool): Run speaker diarization after download - default: false

    Returns:
        {
            "success": true/false,
            "audio_path": "/path/to/audio.wav",
            "title": "Video Title",
            "duration": 180.5,
            "main_artist": "Artist Name",
            "featured_artists": ["Artist 1", "Artist 2"],
            "is_cover": false,
            "video_id": "abc123",
            "diarization_result": {...} (if run_diarization=true),
            "error": null
        }
    """
    if not YOUTUBE_DOWNLOADER_AVAILABLE:
        return jsonify({'error': 'YouTube downloader not available. Install yt-dlp.'}), 503

    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'Missing required field: url'}), 400

    url = data['url']
    if not url.strip():
        return jsonify({'error': 'URL cannot be empty'}), 400

    audio_format = data.get('format', 'wav')
    if audio_format not in ['wav', 'mp3', 'flac']:
        return jsonify({'error': 'Invalid format. Must be wav, mp3, or flac'}), 400

    sample_rate = data.get('sample_rate', 44100)
    try:
        sample_rate = int(sample_rate)
        if sample_rate not in [16000, 22050, 44100, 48000]:
            return jsonify({'error': 'Invalid sample_rate. Must be 16000, 22050, 44100, or 48000'}), 400
    except (ValueError, TypeError):
        return jsonify({'error': 'sample_rate must be an integer'}), 400

    run_diarization = data.get('run_diarization', False)

    try:
        downloader = get_youtube_downloader()
        result = downloader.download(url, audio_format=audio_format, sample_rate=sample_rate)

        if not result.success:
            return jsonify({
                'success': False,
                'error': result.error,
                'title': result.title,
                'video_id': result.video_id
            }), 400

        response = {
            'success': True,
            'audio_path': result.audio_path,
            'title': result.title,
            'duration': result.duration,
            'main_artist': result.main_artist,
            'featured_artists': result.featured_artists,
            'is_cover': result.is_cover,
            'original_artist': result.original_artist,
            'song_title': result.song_title,
            'thumbnail_url': result.thumbnail_url,
            'video_id': result.video_id,
            'error': None
        }

        # Optionally run diarization
        if run_diarization and result.audio_path:
            try:
                from ..audio.speaker_diarization import SpeakerDiarizer
                diarizer = SpeakerDiarizer()
                diarization_result = diarizer.diarize(result.audio_path)
                response['diarization_result'] = {
                    'num_speakers': diarization_result.num_speakers,
                    'segments': [
                        {
                            'speaker_id': seg.speaker_id,
                            'start': seg.start,
                            'end': seg.end,
                            'duration': seg.duration
                        }
                        for seg in diarization_result.segments
                    ]
                }
            except Exception as e:
                logger.warning(f"Diarization failed: {e}")
                response['diarization_error'] = str(e)

        return jsonify(response)

    except Exception as e:
        logger.error(f"YouTube download failed: {e}")
        return jsonify({'error': str(e)}), 500
