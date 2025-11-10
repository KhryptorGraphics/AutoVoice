"""WebSocket handler for AutoVoice real-time audio processing and conversions."""

import os
import io
import base64
import time
import uuid
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

try:
    from flask import Flask, request, current_app
    from flask_socketio import SocketIO, emit, join_room, leave_room
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

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

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

logger = logging.getLogger(__name__)


class WebSocketHandler:
    """Handles WebSocket connections for real-time audio processing and conversions."""

    def __init__(self, socketio: SocketIO):
        """Initialize WebSocket handler
        
        Args:
            socketio: Flask-SocketIO instance
        """
        if not FLASK_AVAILABLE:
            logger.warning("Flask not available, WebSocket handler not initialized")
            return

        self.socketio = socketio
        self.sessions: Dict[str, Dict] = {}  # Store session data
        self.audio_buffers: Dict[str, List[np.ndarray]] = {}  # Store audio buffers

        # Register event handlers
        self._register_handlers()

    def _register_handlers(self):
        """Register all WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            sid = request.sid
            logger.info(f"Client connected: {sid}")
            
            # Initialize session
            self.sessions[sid] = {
                'connected_at': time.time(),
                'config': {
                    'enable_pitch': True,
                    'enable_vad': True,
                    'pitch_shift': 0,
                    'speed': 1.0
                },
                '_warned_librosa_missing': False,
                '_warned_effects_clamped': False
            }
            self.audio_buffers[sid] = []
            
            # Send capabilities
            capabilities = self._get_capabilities()
            emit('connected', {
                'session_id': sid,
                'capabilities': capabilities,
                'message': 'Connected to AutoVoice WebSocket server'
            })

        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            sid = request.sid
            logger.info(f"Client disconnected: {sid}")
            
            # Clean up session data
            self.cleanup_session(sid)

        @self.socketio.on('join')
        def handle_join(data):
            """Handle joining a room"""
            room = data.get('room')
            if room:
                join_room(room)
                emit('joined', {'room': room})

        @self.socketio.on('leave')
        def handle_leave(data):
            """Handle leaving a room"""
            room = data.get('room')
            if room:
                leave_room(room)
                emit('left', {'room': room})

        @self.socketio.on('get_status')
        def handle_get_status():
            """Get server status and capabilities"""
            capabilities = self._get_capabilities()
            metrics = self._get_performance_metrics()
            
            emit('status', {
                'capabilities': capabilities,
                'metrics': metrics,
                'timestamp': time.time()
            })

        @self.socketio.on('config')
        def handle_config(data):
            """Update client configuration"""
            sid = request.sid
            if sid in self.sessions:
                self.sessions[sid].get('config', {}).update(data)
                emit('config_updated', {'config': self.sessions[sid]['config']})

        @self.socketio.on('audio_stream')
        def handle_audio_stream(data):
            """Handle real-time audio streaming"""
            sid = request.sid
            
            try:
                # Decode audio data
                audio_data = data.get('audio_data')
                sample_rate = data.get('sample_rate', 44100)
                chunk_size = data.get('chunk_size', 1024)
                
                if not audio_data:
                    emit('error', {'message': 'No audio data provided'})
                    return

                # Decode base64 audio
                audio_bytes = base64.b64decode(audio_data)
                if NUMPY_AVAILABLE:
                    # Convert bytes to numpy array
                    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                else:
                    # Fallback to raw bytes
                    audio_array = audio_bytes

                # Get audio processor from app context
                audio_processor = getattr(current_app, 'audio_processor', None)
                if not audio_processor:
                    emit('error', {'message': 'Audio processor not available'})
                    return

                # Process audio chunk
                results = self.process_audio_chunk(
                    audio_array, sample_rate, audio_processor, sid
                )

                # Send results back
                emit('audio_processed', results)

            except Exception as e:
                logger.error(f"Error processing audio stream: {e}", exc_info=True)
                emit('error', {'message': str(e)})

        @self.socketio.on('synthesize_stream')
        def handle_synthesize_stream(data):
            """Handle streaming voice synthesis"""
            sid = request.sid
            
            try:
                text = data.get('text')
                speaker_id = data.get('speaker_id', 0)
                
                if not text:
                    emit('error', {'message': 'No text provided'})
                    return

                # Get synthesizer from app context
                synthesizer = getattr(current_app, 'inference_engine', None)
                if not synthesizer:
                    emit('error', {'message': 'Voice synthesizer not available'})
                    return

                # Synthesize speech
                audio_output = synthesizer.text_to_speech(
                    text=text, 
                    speaker_id=speaker_id,
                    stream=True
                )

                # Convert to base64
                if NUMPY_AVAILABLE and isinstance(audio_output, np.ndarray):
                    audio_bytes = (audio_output * 32767).astype(np.int16).tobytes()
                else:
                    audio_bytes = audio_output

                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

                emit('synthesis_complete', {
                    'audio': audio_base64,
                    'text': text,
                    'speaker_id': speaker_id
                })

            except Exception as e:
                logger.error(f"Error in stream synthesis: {e}", exc_info=True)
                emit('error', {'message': str(e)})

        @self.socketio.on('audio_analysis')
        def handle_audio_analysis(data):
            """Handle real-time audio analysis"""
            sid = request.sid
            
            try:
                # Decode audio data
                audio_data = data.get('audio_data')
                sample_rate = data.get('sample_rate', 44100)
                
                if not audio_data:
                    emit('error', {'message': 'No audio data provided'})
                    return

                # Decode base64 audio
                audio_bytes = base64.b64decode(audio_data)
                if NUMPY_AVAILABLE:
                    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                else:
                    audio_array = audio_bytes

                # Get audio processor from app context
                audio_processor = getattr(current_app, 'audio_processor', None)
                if not audio_processor:
                    emit('error', {'message': 'Audio processor not available'})
                    return

                # Analyze audio
                results = {}
                
                # Pitch extraction
                if NUMPY_AVAILABLE and isinstance(audio_array, np.ndarray):
                    # Ensure 1D array for pitch extraction
                    audio_1d = audio_array.flatten()
                    pitch = audio_processor.extract_pitch(audio_1d)
                    if pitch is not None:
                        results['pitch'] = {
                            'current': float(np.mean(pitch)) if len(pitch) > 0 else 0.0,
                            'min': float(np.min(pitch)) if len(pitch) > 0 else 0.0,
                            'max': float(np.max(pitch)) if len(pitch) > 0 else 0.0
                        }

                # Voice activity detection
                vad = audio_processor.voice_activity_detection(audio_array)
                if vad is not None:
                    if NUMPY_AVAILABLE and isinstance(vad, np.ndarray):
                        results['voice_detected'] = bool(np.any(vad))
                        results['voice_confidence'] = float(np.mean(vad))
                    else:
                        results['voice_detected'] = bool(vad)

                # Spectrogram
                spectrogram = audio_processor.compute_spectrogram(audio_array)
                if spectrogram is not None and NUMPY_AVAILABLE:
                    results['spectrogram'] = {
                        'shape': spectrogram.shape,
                        'mean': float(np.mean(spectrogram)),
                        'std': float(np.std(spectrogram))
                    }

                emit('analysis_complete', {
                    'analysis': results,
                    'sample_rate': sample_rate
                })

            except Exception as e:
                logger.error(f"Error in audio analysis: {e}", exc_info=True)
                emit('error', {'message': str(e)})

        @self.socketio.on('convert_song_stream')
        def handle_convert_song_stream(data):
            """Handle streaming song conversion with real-time progress updates"""
            sid = request.sid
            song_path = None
            conversion_id = data.get('conversion_id', str(uuid.uuid4()))
            try:
                from flask import current_app

                # Generate unique conversion ID
                song_data = data.get('song_data')
                target_profile_id = data.get('target_profile_id')
                vocal_volume = data.get('vocal_volume', 1.0)
                instrumental_volume = data.get('instrumental_volume', 0.9)
                return_stems = data.get('return_stems', False)

                # Validate inputs
                if not song_data:
                    self.socketio.emit('conversion_error', {
                        'conversion_id': conversion_id,
                        'error': 'Missing song data',
                        'code': 'MISSING_SONG'
                    }, to=sid)
                    return

                if not target_profile_id:
                    self.socketio.emit('conversion_error', {
                        'conversion_id': conversion_id,
                        'error': 'Missing target profile ID',
                        'code': 'MISSING_PROFILE'
                    }, to=sid)
                    return

                # Validate volume parameters within [0.0, 2.0] like REST API
                if not (0.0 <= vocal_volume <= 2.0):
                    self.socketio.emit('conversion_error', {
                        'conversion_id': conversion_id,
                        'error': 'Volume must be between 0.0 and 2.0',
                        'message': 'Volume must be between 0.0 and 2.0',
                        'code': 'INVALID_PARAMS'
                    }, to=sid)
                    return

                if not (0.0 <= instrumental_volume <= 2.0):
                    self.socketio.emit('conversion_error', {
                        'conversion_id': conversion_id,
                        'error': 'Volume must be between 0.0 and 2.0',
                        'message': 'Volume must be between 0.0 and 2.0',
                        'code': 'INVALID_PARAMS'
                    }, to=sid)
                    return

                # Get pipeline from app context
                pipeline = getattr(current_app, 'singing_conversion_pipeline', None)
                if not pipeline:
                    self.socketio.emit('conversion_error', {
                        'conversion_id': conversion_id,
                        'error': 'Singing conversion pipeline not available',
                        'code': 'SERVICE_UNAVAILABLE'
                    }, to=sid)
                    return

                # Store conversion state
                self._store_conversion_state(sid, conversion_id, {
                    'progress': 0,
                    'stage': 'Starting conversion',
                    'start_time': time.time(),
                    'profile_id': target_profile_id,
                    'status': 'processing',
                    'cancel_flag': False
                })

                # Define progress callback
                def progress_callback(percent, stage_name):
                    # Check for cancellation
                    state = self.sessions.get(f'conversion_{sid}_{conversion_id}', {})
                    if state.get('cancel_flag', False):
                        raise InterruptedError('Conversion cancelled by user')

                    # Update state
                    self._store_conversion_state(sid, conversion_id, {
                        'progress': percent,
                        'stage': stage_name,
                        'status': 'processing'
                    })

                    # Emit progress event
                    self.socketio.emit('conversion_progress', {
                        'conversion_id': conversion_id,
                        'progress': percent,
                        'stage': stage_name,
                        'timestamp': time.time()
                    }, to=sid)

                # Emit initial progress
                progress_callback(0, 'Starting conversion')

                # Decode song data
                progress_callback(5, 'Decoding audio data')

                if isinstance(song_data, str):
                    song_bytes = base64.b64decode(song_data)

                    # Determine file suffix from MIME type or filename
                    song_mime = data.get('song_mime', 'audio/wav')
                    song_filename = data.get('song_filename', 'song.wav')

                    # Map MIME types to extensions
                    mime_to_ext = {
                        'audio/wav': '.wav',
                        'audio/wave': '.wav',
                        'audio/x-wav': '.wav',
                        'audio/mpeg': '.mp3',
                        'audio/mp3': '.mp3',
                        'audio/flac': '.flac',
                        'audio/x-flac': '.flac',
                        'audio/ogg': '.ogg',
                        'audio/vorbis': '.ogg'
                    }

                    # Get extension from MIME or filename
                    suffix = mime_to_ext.get(song_mime)
                    if not suffix and song_filename:
                        _, ext = os.path.splitext(song_filename)
                        suffix = ext if ext else '.wav'
                    if not suffix:
                        suffix = '.wav'  # Default fallback

                    # Save to temporary file with correct extension
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(song_bytes)
                        song_path = tmp.name
                else:
                    song_path = song_data  # Assume file path

                # Starting pipeline conversion
                progress_callback(10, 'Initializing conversion pipeline')

                # Run conversion with progress tracking
                result = pipeline.convert_song(
                    song_path=song_path,
                    target_profile_id=target_profile_id,
                    vocal_volume=vocal_volume,
                    instrumental_volume=instrumental_volume,
                    return_stems=return_stems,
                    progress_callback=progress_callback
                )

                # Encoding results
                progress_callback(95, 'Encoding conversion results')

                # Emit completion event
                audio_output = result.get('mixed_audio') or result.get('audio')
                sample_rate = result.get('sample_rate', 22050)

                if audio_output is not None and not isinstance(audio_output, str):
                    # Convert NumPy array to WAV bytes
                    import wave
                    if NUMPY_AVAILABLE and isinstance(audio_output, np.ndarray):
                        # Detect channels like api.py convert_song()
                        # Determine number of channels and audio data shape
                        if audio_output.ndim == 1:
                            n_channels = 1
                            audio_data = audio_output
                        elif audio_output.shape[1] == 2:  # (T, 2) format
                            n_channels = 2
                            audio_data = audio_output
                        elif audio_output.shape[0] == 2:  # (2, T) format
                            n_channels = 2
                            audio_data = audio_output.T
                        else:
                            n_channels = 1
                            audio_data = audio_output.flatten()

                        # Clip to prevent wrap-around artifacts, then convert to int16
                        audio_data = np.clip(audio_data, -1.0, 1.0)
                        audio_int16 = (audio_data * 32767).astype(np.int16)

                        # Write to WAV format in memory
                        buffer = io.BytesIO()
                        with wave.open(buffer, 'wb') as wav_file:
                            wav_file.setnchannels(n_channels)
                            wav_file.setsampwidth(2)  # 16-bit
                            wav_file.setframerate(sample_rate)
                            wav_file.writeframes(audio_int16.tobytes())

                        audio_output = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    elif isinstance(audio_output, bytes):
                        audio_output = base64.b64encode(audio_output).decode('utf-8')
                    else:
                        audio_output = base64.b64encode(bytes(audio_output)).decode('utf-8')

                # Encode stems to WAV format
                stems_output = None
                if return_stems and result.get('stems'):
                    stems_output = {
                        'format': 'wav',
                        'sample_rate': sample_rate
                    }

                    for stem_name in ['vocals', 'instrumental']:
                        stem_data = result['stems'].get(stem_name)
                        if stem_data is not None and NUMPY_AVAILABLE and isinstance(stem_data, np.ndarray):
                            # Detect channels like api.py convert_song() and main audio above
                            # Determine number of channels and audio data shape for stems
                            if stem_data.ndim == 1:
                                n_channels = 1
                                stem_audio_data = stem_data
                            elif stem_data.shape[1] == 2:  # (T, 2) format
                                n_channels = 2
                                stem_audio_data = stem_data
                            elif stem_data.shape[0] == 2:  # (2, T) format
                                n_channels = 2
                                stem_audio_data = stem_data.T
                            else:
                                n_channels = 1
                                stem_audio_data = stem_data.flatten()

                            # Clip to prevent wrap-around artifacts, then convert to int16
                            stem_audio_data = np.clip(stem_audio_data, -1.0, 1.0)
                            stem_int16 = (stem_audio_data * 32767).astype(np.int16)

                            # Write to WAV format in memory
                            buffer = io.BytesIO()
                            with wave.open(buffer, 'wb') as wav_file:
                                wav_file.setnchannels(n_channels)
                                wav_file.setsampwidth(2)  # 16-bit
                                wav_file.setframerate(sample_rate)
                                wav_file.writeframes(stem_int16.tobytes())

                            stems_output[stem_name] = base64.b64encode(buffer.getvalue()).decode('utf-8')

                self.socketio.emit('conversion_complete', {
                    'conversion_id': conversion_id,
                    'audio': audio_output,
                    'format': 'wav',
                    'sample_rate': sample_rate,
                    'duration': result.get('duration'),
                    'metadata': result.get('metadata', {}),
                    'stems': stems_output
                }, to=sid)

                # Clean up conversion state
                self._cleanup_conversion(sid, conversion_id)

            except InterruptedError as e:
                self.socketio.emit('conversion_cancelled', {
                    'conversion_id': conversion_id,
                    'message': str(e)
                }, to=sid)
                self._cleanup_conversion(sid, conversion_id)

            except Exception as e:
                logger.error(f"Error during song conversion: {e}", exc_info=True)
                self.socketio.emit('conversion_error', {
                    'conversion_id': conversion_id,
                    'error': str(e),
                    'code': 'CONVERSION_FAILED',
                    'stage': self.sessions.get(f'conversion_{sid}_{conversion_id}', {}).get('stage', 'Unknown')
                }, to=sid)
                self._cleanup_conversion(sid, conversion_id)

            finally:
                if song_path and os.path.exists(song_path):
                    os.unlink(song_path)

        @self.socketio.on('cancel_conversion')
        def handle_cancel_conversion(data):
            """Handle conversion cancellation request"""
            try:
                conversion_id = data.get('conversion_id')
                if not conversion_id:
                    emit('error', {'message': 'Missing conversion_id'})
                    return

                # Get SID and compute state key consistently
                sid = request.sid
                state_key = f'conversion_{sid}_{conversion_id}'

                # Set cancellation flag if conversion exists
                if state_key in self.sessions:
                    self.sessions[state_key]['cancel_flag'] = True
                    emit('conversion_cancelled', {
                        'conversion_id': conversion_id,
                        'message': 'Cancellation requested'
                    }, to=sid)
                else:
                    # Conversion not found - might have already finished or never started
                    # Still emit cancellation event for idempotence
                    emit('conversion_cancelled', {
                        'conversion_id': conversion_id,
                        'message': 'Conversion not active (may have already completed or failed)'
                    }, to=sid)

            except Exception as e:
                logger.error(f"Error in cancel_conversion handler: {e}", exc_info=True)
                emit('error', {'message': str(e)})

        @self.socketio.on('get_conversion_status')
        def handle_get_conversion_status(data):
            """Get current status of a conversion"""
            try:
                conversion_id = data.get('conversion_id')
                if not conversion_id:
                    emit('error', {'message': 'Missing conversion_id'})
                    return

                # Get SID and compute state key consistently
                sid = request.sid
                state_key = f'conversion_{sid}_{conversion_id}'

                if state_key in self.sessions:
                    state = self.sessions[state_key]
                    emit('conversion_status', {
                        'conversion_id': conversion_id,
                        'progress': state.get('progress', 0),
                        'stage': state.get('stage', 'Unknown'),
                        'status': state.get('status', 'unknown'),
                        'start_time': state.get('start_time'),
                        'profile_id': state.get('profile_id')
                    }, to=sid)
                else:
                    emit('error', {
                        'message': f'Conversion {conversion_id} not found',
                        'code': 'NOT_FOUND'
                    }, to=sid)

            except Exception as e:
                logger.error(f"Error in get_conversion_status handler: {e}", exc_info=True)
                emit('error', {'message': str(e)})

    def _store_conversion_state(self, sid: str, conversion_id: str, state: Dict):
        """Store conversion state for tracking and reconnection

        Args:
            sid: The session ID of the client.
            conversion_id: Unique conversion identifier.
            state: State data to store (progress, stage, status, etc.).
        """
        state_key = f'conversion_{sid}_{conversion_id}'
        if state_key not in self.sessions:
            self.sessions[state_key] = {
                'conversion_id': conversion_id,
                'start_time': time.time()
            }
        self.sessions[state_key].update(state)

    def _cleanup_conversion(self, sid: str, conversion_id: str):
        """Clean up conversion state after completion or cancellation

        Args:
            sid: The session ID of the client.
            conversion_id: Unique conversion identifier.
        """
        state_key = f'conversion_{sid}_{conversion_id}'
        if state_key in self.sessions:
            del self.sessions[state_key]
            logger.info(f"Cleaned up conversion state for {conversion_id}")

    def process_audio_chunk(self, audio_chunk: np.ndarray, sample_rate: int,
                           audio_processor, session_id: str) -> Dict[str, Any]:
        """Process a chunk of audio data using the AudioProcessor

        Args:
            audio_chunk: Audio data as numpy array
            sample_rate: Sample rate of the audio
            audio_processor: AudioProcessor instance
            session_id: Session identifier

        Returns:
            Dictionary with processed audio and analysis results
        """
        try:
            # Check dependencies
            if not TORCH_AVAILABLE or not NUMPY_AVAILABLE:
                return {
                    'error': 'Dependencies not available',
                    'audio': base64.b64encode(audio_chunk.tobytes()).decode('utf-8')
                }

            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_chunk).float()
            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            config = self.sessions[session_id].get('config', {})
            results = {}

            # Pitch extraction for real-time pitch tracking
            if config.get('enable_pitch', True):
                # Ensure audio tensor is 1D
                audio_1d = audio_tensor.squeeze(0) if audio_tensor.ndim > 1 else audio_tensor
                pitch = audio_processor.extract_pitch(audio_1d)
                if pitch is not None:
                    # Move to CPU if it's a tensor
                    if isinstance(pitch, torch.Tensor):
                        pitch = pitch.detach().cpu().numpy()
                    elif not isinstance(pitch, np.ndarray):
                        pitch = np.array(pitch)
                    if len(pitch) > 0:
                        valid_pitch = pitch[pitch > 0]
                        if len(valid_pitch) > 0:
                            results['pitch'] = {
                                'current': float(np.mean(valid_pitch)),
                                'min': float(np.min(valid_pitch)),
                                'max': float(np.max(valid_pitch))
                            }

            # Voice activity detection
            if config.get('enable_vad', True):
                # Ensure audio tensor is 1D
                audio_1d = audio_tensor.squeeze(0) if audio_tensor.ndim > 1 else audio_tensor
                vad = audio_processor.voice_activity_detection(audio_1d)
                if vad is not None:
                    # Move to CPU if it's a tensor
                    if isinstance(vad, torch.Tensor):
                        vad = vad.detach().cpu().numpy()
                    elif not isinstance(vad, np.ndarray):
                        vad = np.array(vad)
                    results['voice_detected'] = bool(np.any(vad))
                    results['voice_confidence'] = float(np.mean(vad))

            # Apply real-time effects if requested
            processed_audio = audio_tensor
            original_length = audio_tensor.shape[-1]
            had_batch_dim = audio_tensor.ndim > 1

            # Pitch shifting
            if 'pitch_shift' in config and config['pitch_shift'] != 0:
                if LIBROSA_AVAILABLE and NUMPY_AVAILABLE:
                    try:
                        # Convert to CPU numpy float array
                        audio_np = processed_audio.detach().cpu().numpy() if isinstance(processed_audio, torch.Tensor) else processed_audio

                        # Squeeze to 1D if needed
                        if audio_np.ndim > 1:
                            audio_np = audio_np.squeeze()

                        # Check buffer length guard
                        min_len = 2048
                        buf_len = audio_np.shape[-1]
                        if buf_len < min_len:
                            logger.debug(f"Skipping pitch shift: short buffer {buf_len}<{min_len}")
                        else:
                            # Check buffer length guard
                        min_len = 2048
                        buf_len = audio_np.shape[-1]
                        if buf_len < min_len:
                            logger.debug(f"Skipping pitch shift: short buffer {buf_len}<{min_len}")
                        else:
                            # Apply pitch shift
                        shifted_audio = librosa.effects.pitch_shift(
                            y=audio_np,
                            sr=sample_rate,
                            n_steps=config['pitch_shift']
                        )

                        # Convert back to torch tensor
                        processed_audio = torch.from_numpy(shifted_audio).float()

                        # Restore batch dimension if it existed
                        if had_batch_dim:
                            processed_audio = processed_audio.unsqueeze(0)

                        # Normalize length to match original chunk size for streaming
                        current_length = processed_audio.shape[-1]
                        if current_length != original_length:
                            if current_length > original_length:
                                # Truncate
                                processed_audio = processed_audio[..., :original_length]
                            else:
                                # Pad with zeros
                                pad_length = original_length - current_length
                                if had_batch_dim:
                                    padding = torch.zeros(1, pad_length, dtype=processed_audio.dtype)
                                    processed_audio = torch.cat([processed_audio, padding], dim=-1)
                                else:
                                    padding = torch.zeros(pad_length, dtype=processed_audio.dtype)
                                    processed_audio = torch.cat([processed_audio, padding], dim=-1)

                    except Exception as e:
                        logger.warning(f"Pitch shifting failed: {e}, using original audio")
                        processed_audio = audio_tensor
                else:
                    logger.warning("Pitch shifting requested but librosa not available")

            # Speed adjustment
            if 'speed' in config and config['speed'] != 1.0:
                # Validate speed parameter
                if config['speed'] > 0:
                    if LIBROSA_AVAILABLE and NUMPY_AVAILABLE:
                        try:
                            # Convert to CPU numpy float array
                            audio_np = processed_audio.detach().cpu().numpy() if isinstance(processed_audio, torch.Tensor) else processed_audio

                            # Squeeze to 1D if needed
                            if audio_np.ndim > 1:
                                audio_np = audio_np.squeeze()

                            # Check buffer length guard
                        min_len = 2048
                        buf_len = audio_np.shape[-1]
                        if buf_len < min_len:
                            logger.debug(f"Skipping time stretch: short buffer {buf_len}<{min_len}")
                        else:
                            # Apply time stretch
                            stretched_audio = librosa.effects.time_stretch(
                                y=audio_np,
                                rate=config['speed']
                            )

                            # Convert back to torch tensor
                            processed_audio = torch.from_numpy(stretched_audio).float()

                            # Restore batch dimension if it existed
                            if had_batch_dim:
                                processed_audio = processed_audio.unsqueeze(0)

                            # Normalize length to match original chunk size for streaming
                            current_length = processed_audio.shape[-1]
                            if current_length != original_length:
                                if current_length > original_length:
                                    # Truncate
                                    processed_audio = processed_audio[..., :original_length]
                                else:
                                    # Pad with zeros
                                    pad_length = original_length - current_length
                                    if had_batch_dim:
                                        padding = torch.zeros(1, pad_length, dtype=processed_audio.dtype)
                                        processed_audio = torch.cat([processed_audio, padding], dim=-1)
                                    else:
                                        padding = torch.zeros(pad_length, dtype=processed_audio.dtype)
                                        processed_audio = torch.cat([processed_audio, padding], dim=-1)

                        except Exception as e:
                            logger.warning(f"Time stretching failed: {e}, using original audio")
                            processed_audio = audio_tensor
                    else:
                        logger.warning("Time stretching requested but librosa not available")
                else:
                    logger.warning(f"Invalid speed value {config['speed']}, must be > 0")

            # Convert back to bytes
            processed_bytes = processed_audio.cpu().numpy().astype(np.float32).tobytes()

            return {
                'audio': base64.b64encode(processed_bytes).decode('utf-8'),
                'analysis': results,
                'sample_rate': sample_rate
            }

        except Exception as e:
            logger.error(f"Chunk processing error: {e}", exc_info=True)
            return {
                'error': str(e),
                'audio': base64.b64encode(audio_chunk.tobytes()).decode('utf-8')
            }

    def _get_capabilities(self) -> Dict[str, bool]:
        """Get current system capabilities"""
        return {
            'audio_processing': bool(getattr(current_app, 'audio_processor', None)),
            'voice_synthesis': bool(getattr(current_app, 'inference_engine', None)),
            'cuda_available': torch.cuda.is_available() if TORCH_AVAILABLE else False,
            'real_time_processing': True,
            'pitch_detection': True,
            'voice_activity_detection': True,
            'multi_speaker': True
        }

    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        metrics = {
            'active_sessions': len(self.sessions),
            'total_buffers': sum(len(buf) for buf in self.audio_buffers.values())
        }

        # Add GPU metrics if CUDA is available
        if torch.cuda.is_available():
            try:
                metrics['gpu'] = {
                    'memory_allocated_mb': torch.cuda.memory_allocated(0) / 1024**2,
                    'memory_reserved_mb': torch.cuda.memory_reserved(0) / 1024**2,
                    'utilization': torch.cuda.utilization(0) if hasattr(torch.cuda, 'utilization') else 'N/A'
                }
            except Exception as e:
                logger.warning(f"Failed to get GPU metrics: {e}")

        return metrics

    def broadcast_to_room(self, room: str, event: str, data: Any):
        """Broadcast data to all clients in a room"""
        self.socketio.emit(event, data, room=room)

    def cleanup_session(self, session_id: str):
        """Clean up session data and associated conversion states"""
        # Clean up main session data
        if session_id in self.sessions:
            del self.sessions[session_id]
        if session_id in self.audio_buffers:
            del self.audio_buffers[session_id]

        # Clean up conversion states associated with this session
        cleaned_count = 0
        conversion_keys_to_remove = []

        # Find all conversion states for this session
        for key in list(self.sessions.keys()):
            if key.startswith(f'conversion_{session_id}_'):
                # Set cancel flag first to stop any active conversions
                if 'cancel_flag' in self.sessions[key]:
                    self.sessions[key]['cancel_flag'] = True
                conversion_keys_to_remove.append(key)

        # Remove conversion states
        for key in conversion_keys_to_remove:
            del self.sessions[key]
            cleaned_count += 1

        # Log cleanup activity
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} conversion states for disconnected session {session_id}")
