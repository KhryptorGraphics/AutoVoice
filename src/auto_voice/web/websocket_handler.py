"""WebSocket handler for real-time audio streaming with full component integration"""
import base64
import io
import json
import logging
from flask import current_app, request
from flask_socketio import emit, join_room, leave_room, rooms
from typing import Dict, Any, Optional
from collections import defaultdict
import time

# Graceful imports
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


class WebSocketHandler:
    """Handles WebSocket connections for real-time audio processing and voice synthesis"""

    def __init__(self, socketio):
        self.socketio = socketio
        self.sessions = defaultdict(dict)  # Store session data per connection
        self.audio_buffers = defaultdict(list)  # Audio buffers per connection
        self.setup_handlers()

    def setup_handlers(self):
        """Setup WebSocket event handlers with full audio processing integration"""

        @self.socketio.on('connect')
        def on_connect():
            """Handle client connection and initialize session"""
            session_id = request.sid  # Use SocketIO session ID as default

            # Initialize session with default configuration
            self.sessions[session_id] = {
                'room': 'default',
                'config': {},
                'start_time': time.time()
            }

            emit('status', {
                'message': 'Connected to AutoVoice',
                'session_id': session_id,
                'capabilities': self._get_capabilities()
            })
            logger.info(f"Client connected: {session_id}")

        @self.socketio.on('disconnect')
        def on_disconnect():
            """Handle client disconnect and cleanup session"""
            session_id = request.sid

            # Cleanup session data
            self.cleanup_session(session_id)

            logger.info(f"Client disconnected: {session_id}")

        @self.socketio.on('join')
        def on_join(data):
            """Join a room for multi-user support"""
            room = data.get('room', 'default')
            session_id = data.get('session_id', request.sid)  # Default to request.sid
            join_room(room)

            # Update session with room information
            if session_id not in self.sessions:
                self.sessions[session_id] = {
                    'start_time': time.time()
                }

            self.sessions[session_id]['room'] = room
            self.sessions[session_id]['config'] = data.get('config', self.sessions[session_id].get('config', {}))

            emit('status', {
                'message': f'Joined room {room}',
                'session_id': session_id,
                'capabilities': self._get_capabilities()
            })
            logger.info(f"Client {session_id} joined room {room}")

        @self.socketio.on('leave')
        def on_leave(data):
            """Leave a room (not full disconnect)"""
            room = data.get('room', 'default')
            session_id = data.get('session_id', request.sid)  # Default to request.sid
            leave_room(room)

            # Update session room to default
            if session_id in self.sessions:
                self.sessions[session_id]['room'] = 'default'

            emit('status', {'message': f'Left room {room}'})
            logger.info(f"Client {session_id} left room {room}")

        @self.socketio.on('audio_stream')
        def on_audio_stream(data):
            """Handle incoming audio stream data with real-time processing"""
            session_id = data.get('session_id', request.sid)  # Default to request.sid

            try:
                audio_processor = getattr(current_app, 'audio_processor', None)
                if not audio_processor:
                    emit('error', {
                        'message': 'Audio processor not available',
                        'code': 'PROCESSOR_UNAVAILABLE'
                    })
                    return

                # Decode base64 audio data
                audio_bytes = base64.b64decode(data['audio'])
                sample_rate = data.get('sample_rate', 22050)

                # Convert bytes to numpy array if numpy is available
                if not NUMPY_AVAILABLE:
                    emit('error', {
                        'message': 'NumPy not available for audio processing',
                        'code': 'NUMPY_UNAVAILABLE'
                    })
                    return

                dtype = data.get('dtype', 'float32')
                if dtype == 'float32':
                    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                elif dtype == 'int16':
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

                # Add to buffer for continuous processing
                self.audio_buffers[session_id].append(audio_array)

                # Process when buffer is large enough
                buffer_size = self.sessions[session_id].get('config', {}).get('buffer_size', 1024)
                if sum(len(chunk) for chunk in self.audio_buffers[session_id]) >= buffer_size:
                    # Concatenate buffer
                    full_audio = np.concatenate(self.audio_buffers[session_id])

                    # Clear processed portion from buffer
                    remaining = len(full_audio) % buffer_size
                    if remaining > 0:
                        self.audio_buffers[session_id] = [full_audio[-remaining:]]
                    else:
                        self.audio_buffers[session_id] = []

                    # Process audio chunk
                    processed_data = self.process_audio_chunk(
                        full_audio[:len(full_audio) - remaining],
                        sample_rate,
                        audio_processor,
                        session_id
                    )

                    # Send processed results
                    emit('processed_audio', {
                        'timestamp': data.get('timestamp'),
                        **processed_data
                    })

            except Exception as e:
                logger.error(f"Audio stream processing error: {e}", exc_info=True)
                emit('error', {
                    'message': str(e),
                    'code': 'PROCESSING_ERROR'
                })

        @self.socketio.on('synthesize_stream')
        def on_synthesize_stream(data):
            """Handle real-time text-to-speech synthesis"""
            session_id = data.get('session_id', request.sid)  # Default to request.sid

            try:
                inference_engine = getattr(current_app, 'inference_engine', None)
                if not inference_engine:
                    emit('error', {
                        'message': 'Inference engine not available',
                        'code': 'ENGINE_UNAVAILABLE'
                    })
                    return

                text = data.get('text', '')
                if not text:
                    emit('error', {
                        'message': 'No text provided',
                        'code': 'INVALID_INPUT'
                    })
                    return

                # Get synthesis parameters
                speaker_id = data.get('speaker_id', 0)
                voice_config = data.get('voice_config', {})
                stream_chunks = data.get('stream_chunks', True)

                logger.info(f"Synthesizing stream for session {session_id}: {text[:50]}...")

                # Synthesize speech (Comment 3: remove unsupported kwargs)
                audio_data = inference_engine.synthesize_speech(
                    text=text,
                    speaker_id=speaker_id
                )

                if torch and isinstance(audio_data, torch.Tensor):
                    audio_data = audio_data.cpu().numpy()
                elif not NUMPY_AVAILABLE:
                    # Without numpy, can't process further
                    emit('error', {
                        'message': 'NumPy required for audio processing',
                        'code': 'NUMPY_UNAVAILABLE'
                    })
                    return

                # Stream audio in chunks if requested
                if stream_chunks:
                    chunk_size = voice_config.get('chunk_size', 1024)
                    for i in range(0, len(audio_data), chunk_size):
                        chunk = audio_data[i:i + chunk_size]
                        emit('synthesized_chunk', {
                            'audio': base64.b64encode(chunk.tobytes()).decode('utf-8'),
                            'chunk_index': i // chunk_size,
                            'is_final': i + chunk_size >= len(audio_data)
                        })
                else:
                    # Send complete audio
                    # Fix config access - use correct nesting
                    sample_rate = current_app.app_config.get('audio', {}).get('sample_rate', 22050)
                    emit('synthesized_audio', {
                        'audio': base64.b64encode(audio_data.tobytes()).decode('utf-8'),
                        'duration': len(audio_data) / sample_rate
                    })

            except Exception as e:
                logger.error(f"Synthesis stream error: {e}", exc_info=True)
                emit('error', {
                    'message': str(e),
                    'code': 'SYNTHESIS_ERROR'
                })

        @self.socketio.on('audio_analysis')
        def on_audio_analysis(data):
            """Perform real-time audio analysis"""
            session_id = data.get('session_id', request.sid)  # Default to request.sid

            try:
                audio_processor = getattr(current_app, 'audio_processor', None)
                if not audio_processor:
                    emit('error', {
                        'message': 'Audio processor not available',
                        'code': 'PROCESSOR_UNAVAILABLE'
                    })
                    return

                # Check dependencies
                if not NUMPY_AVAILABLE or not TORCH_AVAILABLE:
                    emit('error', {
                        'message': 'NumPy and PyTorch required for analysis',
                        'code': 'DEPENDENCIES_UNAVAILABLE'
                    })
                    return

                # Decode audio
                audio_bytes = base64.b64decode(data['audio'])
                audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

                # Convert to tensor
                audio_tensor = torch.from_numpy(audio_array).float()
                if audio_tensor.ndim == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)

                # Perform analysis
                analysis = {}

                # Pitch detection
                if data.get('analyze_pitch', True):
                    # Comment 10: Ensure audio tensor is 1D
                    audio_1d = audio_tensor.squeeze(0) if audio_tensor.ndim > 1 else audio_tensor
                    pitch = audio_processor.extract_pitch(audio_1d)
                    if pitch is not None:
                        # Move to CPU if it's a tensor
                        if isinstance(pitch, torch.Tensor):
                            pitch = pitch.detach().cpu().numpy()
                        elif not isinstance(pitch, np.ndarray):
                            pitch = np.array(pitch)
                        analysis['pitch'] = {
                            'values': pitch.tolist(),
                            'mean': float(np.mean(pitch[pitch > 0])) if len(pitch[pitch > 0]) > 0 else 0
                        }

                # Voice activity detection (Comment 4: correct method name)
                if data.get('analyze_vad', True):
                    # Comment 10: Ensure audio tensor is 1D
                    audio_1d = audio_tensor.squeeze(0) if audio_tensor.ndim > 1 else audio_tensor
                    vad = audio_processor.voice_activity_detection(audio_1d)
                    if vad is not None:
                        # Move to CPU if it's a tensor
                        if isinstance(vad, torch.Tensor):
                            vad = vad.detach().cpu().numpy()
                        elif not isinstance(vad, np.ndarray):
                            vad = np.array(vad)
                        analysis['vad'] = {
                            'segments': vad.tolist(),
                            'voice_ratio': float(np.mean(vad)) if len(vad) > 0 else 0
                        }

                # Spectrogram for visualization
                if data.get('compute_spectrogram', False):
                    # Comment 10: Ensure audio tensor is 1D
                    audio_1d = audio_tensor.squeeze(0) if audio_tensor.ndim > 1 else audio_tensor
                    spec = audio_processor.compute_spectrogram(audio_1d)
                    # Move to CPU if it's a tensor
                    if isinstance(spec, torch.Tensor):
                        spec = spec.detach().cpu()
                    # Send summary instead of full spectrogram for efficiency
                    analysis['spectrogram'] = {
                        'shape': list(spec.shape),
                        'max': float(spec.max()),
                        'min': float(spec.min()),
                        'mean': float(spec.mean())
                    }

                emit('analysis_result', {
                    'timestamp': data.get('timestamp'),
                    'analysis': analysis
                })

            except Exception as e:
                logger.error(f"Audio analysis error: {e}", exc_info=True)
                emit('error', {
                    'message': str(e),
                    'code': 'ANALYSIS_ERROR'
                })

        @self.socketio.on('voice_config')
        def on_voice_config(data):
            """Update voice synthesis configuration for a session"""
            session_id = data.get('session_id', request.sid)  # Default to request.sid
            config = data.get('config', {})

            # Update session config
            if session_id in self.sessions:
                self.sessions[session_id]['config'].update(config)
            else:
                # Initialize session if not exists
                self.sessions[session_id] = {
                    'room': 'default',
                    'config': config,
                    'start_time': time.time()
                }

            emit('config_updated', {
                'status': 'success',
                'config': config,
                'session_id': session_id
            })
            logger.info(f"Updated config for session {session_id}: {config}")

        @self.socketio.on('get_status')
        def on_get_status(data):
            """Get current processing status and capabilities"""
            session_id = data.get('session_id', request.sid) if data else request.sid  # Default to request.sid

            status = {
                'session_id': session_id,
                'session_active': session_id in self.sessions,
                'buffer_size': len(self.audio_buffers.get(session_id, [])),
                'capabilities': self._get_capabilities(),
                'performance': self._get_performance_metrics()
            }

            if session_id in self.sessions:
                status['session_duration'] = time.time() - self.sessions[session_id]['start_time']
                status['config'] = self.sessions[session_id].get('config', {})

            emit('status_update', status)

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
                # Comment 10: Ensure audio tensor is 1D
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

            # Voice activity detection (Comment 4: correct method name)
            if config.get('enable_vad', True):
                # Comment 10: Ensure audio tensor is 1D
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

            # Pitch shifting
            if 'pitch_shift' in config and config['pitch_shift'] != 0:
                # Apply pitch shift (placeholder - implement actual pitch shifting)
                processed_audio = audio_tensor  # TODO: Implement pitch shifting

            # Speed adjustment
            if 'speed' in config and config['speed'] != 1.0:
                # Apply speed adjustment (placeholder - implement actual time stretching)
                processed_audio = audio_tensor  # TODO: Implement time stretching

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
        """Clean up session data"""
        if session_id in self.sessions:
            del self.sessions[session_id]
        if session_id in self.audio_buffers:
            del self.audio_buffers[session_id]