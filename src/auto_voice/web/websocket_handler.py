"""WebSocket handler for real-time audio streaming"""
from flask_socketio import emit, join_room, leave_room
import numpy as np
import base64


class WebSocketHandler:
    """Handles WebSocket connections for real-time audio"""

    def __init__(self, socketio):
        self.socketio = socketio
        self.setup_handlers()

    def setup_handlers(self):
        """Setup WebSocket event handlers"""

        @self.socketio.on('join')
        def on_join(data):
            room = data.get('room', 'default')
            join_room(room)
            emit('status', {'message': f'Joined room {room}'})

        @self.socketio.on('leave')
        def on_leave(data):
            room = data.get('room', 'default')
            leave_room(room)
            emit('status', {'message': f'Left room {room}'})

        @self.socketio.on('audio_stream')
        def on_audio_stream(data):
            """Handle incoming audio stream data"""
            try:
                # Decode base64 audio data
                audio_data = base64.b64decode(data['audio'])
                audio_array = np.frombuffer(audio_data, dtype=np.float32)

                # Process audio (placeholder)
                processed_audio = self.process_audio_chunk(audio_array)

                # Send processed audio back
                emit('processed_audio', {
                    'audio': base64.b64encode(processed_audio.tobytes()).decode('utf-8'),
                    'timestamp': data.get('timestamp')
                })
            except Exception as e:
                emit('error', {'message': str(e)})

        @self.socketio.on('voice_config')
        def on_voice_config(data):
            """Update voice configuration"""
            config = data.get('config', {})
            emit('config_updated', {'status': 'success', 'config': config})

    def process_audio_chunk(self, audio_chunk):
        """Process a chunk of audio data"""
        # Placeholder for audio processing
        return audio_chunk

    def broadcast_to_room(self, room, event, data):
        """Broadcast data to all clients in a room"""
        self.socketio.emit(event, data, room=room)