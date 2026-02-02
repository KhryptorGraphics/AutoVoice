"""Karaoke WebSocket event handlers for real-time audio streaming.

Provides SocketIO namespace /karaoke for bidirectional audio streaming
during live voice conversion sessions. Includes optional training sample
collection for voice profile improvement.
"""
import logging
import os
import struct
import time
from typing import Dict, Any, Optional

import numpy as np
import torch
from flask_socketio import Namespace, emit, join_room, leave_room

from .karaoke_session import KaraokeSession
from .karaoke_api import register_session, update_session_activity, cleanup_session

logger = logging.getLogger(__name__)

# Training sample storage path (configurable via environment)
SAMPLE_STORAGE_PATH = os.environ.get(
    'AUTOVOICE_SAMPLE_STORAGE', '/var/lib/autovoice/training_samples'
)


# ============================================================================
# Usage Analytics (Task 8.4) - Privacy-Respecting Metrics
# ============================================================================

class KaraokeAnalytics:
    """Simple, privacy-respecting usage analytics.

    Only tracks aggregate metrics, no personal data or audio content.
    """

    def __init__(self):
        self._metrics = {
            'total_sessions': 0,
            'total_chunks_processed': 0,
            'total_audio_seconds': 0.0,
            'total_errors': 0,
            'sessions_by_hour': {},  # hour -> count
            'avg_session_duration_s': 0.0,
            'avg_latency_ms': 0.0,
        }
        self._session_count_for_avg = 0
        self._latency_samples = []

    def record_session_start(self):
        """Record a new session started."""
        self._metrics['total_sessions'] += 1
        hour = time.strftime('%Y-%m-%d-%H')
        self._metrics['sessions_by_hour'][hour] = (
            self._metrics['sessions_by_hour'].get(hour, 0) + 1
        )

    def record_session_end(self, duration_s: float, chunks_processed: int):
        """Record session completion."""
        self._metrics['total_chunks_processed'] += chunks_processed
        self._session_count_for_avg += 1

        # Update rolling average duration
        n = self._session_count_for_avg
        old_avg = self._metrics['avg_session_duration_s']
        self._metrics['avg_session_duration_s'] = old_avg + (duration_s - old_avg) / n

    def record_audio_processed(self, seconds: float, latency_ms: float):
        """Record audio chunk processed."""
        self._metrics['total_audio_seconds'] += seconds

        # Keep last 1000 latency samples for rolling average
        self._latency_samples.append(latency_ms)
        if len(self._latency_samples) > 1000:
            self._latency_samples.pop(0)
        self._metrics['avg_latency_ms'] = sum(self._latency_samples) / len(self._latency_samples)

    def record_error(self):
        """Record an error occurrence."""
        self._metrics['total_errors'] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get current aggregate metrics."""
        return {
            'total_sessions': self._metrics['total_sessions'],
            'total_chunks_processed': self._metrics['total_chunks_processed'],
            'total_audio_minutes': round(self._metrics['total_audio_seconds'] / 60, 1),
            'total_errors': self._metrics['total_errors'],
            'avg_session_duration_s': round(self._metrics['avg_session_duration_s'], 1),
            'avg_latency_ms': round(self._metrics['avg_latency_ms'], 1),
            'sessions_last_24h': sum(
                count for hour, count in self._metrics['sessions_by_hour'].items()
                if hour >= time.strftime('%Y-%m-%d', time.gmtime(time.time() - 86400))
            ),
        }


# Global analytics instance
_analytics = KaraokeAnalytics()


def get_karaoke_analytics() -> Dict[str, Any]:
    """Get karaoke usage analytics (for health/monitoring endpoints)."""
    return _analytics.get_metrics()


class KaraokeNamespace(Namespace):
    """SocketIO namespace for karaoke real-time audio streaming.

    Events:
        connect: Client connects to namespace
        disconnect: Client disconnects
        join_session: Join a karaoke session room
        leave_session: Leave a karaoke session room
        start_session: Initialize a conversion session with embedding
        stop_session: Stop and cleanup a session
        audio_chunk: Receive audio chunk from client microphone

    Emits:
        connected: Confirmation of connection
        session_joined: Confirmation of session join
        session_started: Session ready for audio
        converted_audio: Processed audio chunk back to client
        latency_update: Current processing latency
        samples_collected: Number of training samples captured (if enabled)
        error: Error messages
    """

    def __init__(self, namespace: str = '/karaoke'):
        super().__init__(namespace)
        self._sessions: Dict[str, KaraokeSession] = {}
        self._client_sessions: Dict[str, str] = {}  # client_id -> session_id
        self._client_connect_time: Dict[str, float] = {}  # client_id -> connect timestamp
        self._sample_collectors: Dict[str, Any] = {}  # session_id -> SampleCollector

    def on_connect(self):
        """Handle client connection to karaoke namespace."""
        from flask import request
        client_id = request.sid
        self._client_connect_time[client_id] = time.time()
        logger.info(f"Client {client_id[:8]}... connected to /karaoke namespace")
        emit('connected', {'status': 'connected', 'client_id': client_id})

    def on_disconnect(self):
        """Handle client disconnection with graceful cleanup (Task 8.3)."""
        from flask import request
        client_id = request.sid

        # Clean up any active session for this client
        session_id = self._client_sessions.pop(client_id, None)
        if session_id:
            session = self._sessions.pop(session_id, None)
            if session:
                # Record analytics before cleanup
                stats = session.get_stats()
                _analytics.record_session_end(
                    duration_s=stats.get('duration_s', 0),
                    chunks_processed=stats.get('chunks_processed', 0)
                )
                session.stop()
                # Clean up from API tracking
                cleanup_session(session_id, reason='client_disconnect')
                logger.info(
                    f"Session {session_id} cleaned up on disconnect "
                    f"(chunks={stats.get('chunks_processed', 0)}, "
                    f"duration={stats.get('duration_s', 0):.1f}s)"
                )

            # Stop sample collection if active
            collector = self._sample_collectors.pop(session_id, None)
            if collector:
                try:
                    samples = collector.stop_recording()
                    logger.info(
                        f"Sample collection stopped on disconnect: "
                        f"{len(samples)} samples captured for session {session_id}"
                    )
                except Exception as e:
                    logger.warning(f"Error stopping sample collection: {e}")

        # Clean up connect time tracking
        self._client_connect_time.pop(client_id, None)

        logger.info(f"Client {client_id[:8]}... disconnected from /karaoke namespace")

    def on_join_session(self, data: Dict[str, Any]):
        """Handle client joining a karaoke session.

        Args:
            data: Dict with session_id and optional voice_model_id
        """
        session_id = data.get('session_id')
        if not session_id:
            emit('error', {'message': 'session_id is required'})
            return

        join_room(session_id)
        logger.info(f"Client joined karaoke session: {session_id}")
        emit('session_joined', {
            'session_id': session_id,
            'status': 'joined'
        })

    def on_leave_session(self, data: Dict[str, Any]):
        """Handle client leaving a karaoke session.

        Args:
            data: Dict with session_id
        """
        session_id = data.get('session_id')
        if session_id:
            leave_room(session_id)
            logger.info(f"Client left karaoke session: {session_id}")
            emit('session_left', {'session_id': session_id})

    def on_start_session(self, data: Dict[str, Any]):
        """Start a karaoke conversion session.

        Args:
            data: Dict with:
                - session_id: Unique session identifier
                - song_id: ID of uploaded song
                - vocals_path: Path to separated vocals
                - instrumental_path: Path to separated instrumental
                - speaker_embedding: Optional base64-encoded embedding
                - profile_id: Optional voice profile ID for sample collection
                - collect_samples: Optional bool to enable training sample collection
                - pipeline_type: 'realtime' for low-latency or 'quality' for high-fidelity (default: realtime)
        """
        from flask import request

        session_id = data.get('session_id')
        song_id = data.get('song_id')
        vocals_path = data.get('vocals_path')
        instrumental_path = data.get('instrumental_path')
        client_id = request.sid

        # Pipeline selection (realtime/realtime_meanvc for karaoke, quality/quality_seedvc/quality_shortcut for offline)
        pipeline_type = data.get('pipeline_type', 'realtime')
        if pipeline_type not in ('realtime', 'quality', 'quality_seedvc', 'realtime_meanvc', 'quality_shortcut'):
            pipeline_type = 'realtime'  # Default to realtime for live karaoke
        logger.info(f"Session using pipeline_type: {pipeline_type}")

        # Sample collection settings (requires explicit consent)
        profile_id = data.get('profile_id')
        collect_samples = data.get('collect_samples', False)

        if not all([session_id, song_id]):
            emit('error', {'message': 'session_id and song_id are required'})
            _analytics.record_error()
            return

        # Create session
        try:
            session = KaraokeSession(
                session_id=session_id,
                song_id=song_id,
                vocals_path=vocals_path or '',
                instrumental_path=instrumental_path or '',
            )
            # Store user's requested pipeline type for reference/logging
            session._user_pipeline_preference = pipeline_type

            # Set speaker embedding if provided
            if 'speaker_embedding' in data:
                import base64
                embedding_bytes = base64.b64decode(data['speaker_embedding'])
                embedding = torch.from_numpy(
                    np.frombuffer(embedding_bytes, dtype=np.float32)
                )
                session.set_speaker_embedding(embedding)

            session.start()
            self._sessions[session_id] = session
            self._client_sessions[client_id] = session_id

            # Register session for tracking and cleanup (Task 8.3)
            register_session(session_id, song_id, client_id)

            # Record analytics (Task 8.4)
            _analytics.record_session_start()

            # Initialize sample collection if enabled (Task 3.7)
            sample_collection_enabled = False
            if profile_id and collect_samples:
                try:
                    from auto_voice.profiles.sample_collector import SampleCollector

                    collector = SampleCollector(storage_path=SAMPLE_STORAGE_PATH)
                    collector.start_recording(
                        profile_id=profile_id,
                        session_id=session_id,
                        consent_given=True,  # User explicitly opted in
                    )
                    self._sample_collectors[session_id] = collector
                    sample_collection_enabled = True
                    logger.info(
                        f"Sample collection enabled for session {session_id}, "
                        f"profile {profile_id}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to initialize sample collection: {e}. "
                        f"Session will continue without collection."
                    )

            emit('session_started', {
                'session_id': session_id,
                'status': 'ready',
                'sample_collection_enabled': sample_collection_enabled,
            })
            logger.info(f"Karaoke session started: {session_id} for client {client_id[:8]}...")

        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            _analytics.record_error()
            emit('error', {'message': f'Failed to start session: {str(e)}'})

    def on_stop_session(self, data: Dict[str, Any]):
        """Stop a karaoke conversion session.

        Args:
            data: Dict with session_id
        """
        session_id = data.get('session_id')
        if not session_id:
            return

        session = self._sessions.get(session_id)
        if session:
            stats = session.get_stats()
            session.stop()
            del self._sessions[session_id]

            # Stop sample collection if active (Task 3.7)
            samples_collected = 0
            collector = self._sample_collectors.pop(session_id, None)
            if collector:
                try:
                    captured_samples = collector.stop_recording()
                    samples_collected = len(captured_samples)
                    logger.info(
                        f"Sample collection completed for session {session_id}: "
                        f"{samples_collected} samples captured"
                    )
                except Exception as e:
                    logger.warning(f"Error stopping sample collection: {e}")

            emit('session_stopped', {
                'session_id': session_id,
                'stats': stats,
                'samples_collected': samples_collected,
            })

            # Emit separate event for sample collection status
            if samples_collected > 0:
                emit('samples_collected', {
                    'session_id': session_id,
                    'count': samples_collected,
                })

            logger.info(f"Karaoke session stopped: {session_id}")

    def on_audio_chunk(self, data: Dict[str, Any]):
        """Handle incoming audio chunk from client microphone.

        Args:
            data: Dict with:
                - session_id: Session to process chunk in
                - audio: Base64-encoded PCM float32 audio
                - timestamp: Client timestamp for sync
        """
        from flask import request

        session_id = data.get('session_id')
        if not session_id:
            # Try to get from client mapping
            session_id = self._client_sessions.get(request.sid)

        if not session_id:
            emit('error', {'message': 'No active session'})
            _analytics.record_error()
            return

        session = self._sessions.get(session_id)
        if not session or not session.is_active:
            emit('error', {'message': 'Session not active'})
            _analytics.record_error()
            return

        try:
            import base64

            # Update session activity timestamp (Task 8.3)
            update_session_activity(session_id)

            # Decode audio data
            audio_b64 = data.get('audio', '')
            audio_bytes = base64.b64decode(audio_b64)
            audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
            audio_tensor = torch.from_numpy(audio_np)

            # Process through conversion pipeline
            converted = session.process_chunk(audio_tensor)
            latency_ms = session.get_latency_ms()

            # Record analytics (Task 8.4)
            # Assume 24kHz sample rate, calculate seconds from samples
            audio_seconds = len(audio_np) / 24000.0
            _analytics.record_audio_processed(audio_seconds, latency_ms)

            # Feed to sample collector if active (Task 3.7)
            collector = self._sample_collectors.get(session_id)
            if collector:
                try:
                    collector.add_chunk(audio_np, sample_rate=24000)
                except Exception as e:
                    # Sample collection errors shouldn't interrupt conversion
                    logger.debug(f"Sample collection error (non-critical): {e}")

            # Encode output
            converted_np = converted.cpu().numpy().astype(np.float32)
            converted_b64 = base64.b64encode(converted_np.tobytes()).decode('utf-8')

            # Send back converted audio
            emit('converted_audio', {
                'session_id': session_id,
                'audio': converted_b64,
                'timestamp': data.get('timestamp'),
                'latency_ms': latency_ms
            })

        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            _analytics.record_error()
            emit('error', {'message': f'Processing error: {str(e)}'})

    def on_set_speaker_embedding(self, data: Dict[str, Any]):
        """Update speaker embedding for a session.

        Args:
            data: Dict with session_id and speaker_embedding (base64)
        """
        session_id = data.get('session_id')
        session = self._sessions.get(session_id)

        if not session:
            emit('error', {'message': 'Session not found'})
            return

        try:
            import base64
            embedding_bytes = base64.b64decode(data['speaker_embedding'])
            embedding = torch.from_numpy(
                np.frombuffer(embedding_bytes, dtype=np.float32)
            )
            session.set_speaker_embedding(embedding)
            emit('embedding_updated', {'session_id': session_id})
        except Exception as e:
            emit('error', {'message': f'Failed to set embedding: {str(e)}'})


def register_karaoke_namespace(socketio):
    """Register the karaoke namespace with SocketIO.

    Args:
        socketio: Flask-SocketIO instance
    """
    karaoke_ns = KaraokeNamespace('/karaoke')
    socketio.on_namespace(karaoke_ns)
    logger.info("Karaoke WebSocket namespace registered")
    return karaoke_ns
