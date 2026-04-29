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
from flask import current_app, request
from flask_socketio import Namespace, emit, join_room, leave_room

from .karaoke_session import KaraokeSession
from .karaoke_api import (
    cleanup_session,
    load_karaoke_session_snapshot,
    register_session,
    save_karaoke_session_snapshot,
    update_session_activity,
    _load_audio_router_config,
)
from .security import record_structured_audit_event, require_socketio_authorization

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

    def _audit_socket_event(
        self,
        action: str,
        *,
        session_id: str | None = None,
        allowed: bool,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            record_structured_audit_event(
                action,
                'karaoke_session',
                app=current_app,
                actor=f"socketio:{getattr(request, 'sid', 'unknown')}",
                resource_id=session_id,
                details={'allowed': allowed, **dict(details or {})},
            )
        except Exception:
            logger.debug("Failed to record karaoke audit event %s", action, exc_info=True)

    def _authorized_or_emit(
        self,
        payload: Optional[Dict[str, Any]] = None,
        *,
        action: str = 'karaoke.event',
    ) -> bool:
        """Mirror HTTP/default Socket.IO token auth for karaoke events."""
        if require_socketio_authorization(current_app, payload):
            self._audit_socket_event(
                action,
                session_id=(payload or {}).get('session_id') if isinstance(payload, dict) else None,
                allowed=True,
            )
            return True
        self._audit_socket_event(
            action,
            session_id=(payload or {}).get('session_id') if isinstance(payload, dict) else None,
            allowed=False,
        )
        emit('error', {'message': 'authentication required'})
        return False

    def _build_session_snapshot(
        self,
        session: KaraokeSession,
        *,
        collect_samples: bool = False,
        sample_collection_enabled: bool = False,
    ) -> Dict[str, Any]:
        snapshot = session.get_recovery_snapshot()
        snapshot['collect_samples'] = collect_samples
        snapshot['sample_collection_enabled'] = sample_collection_enabled
        snapshot['audio_router_targets'] = _load_audio_router_config()
        return snapshot

    def _persist_session_snapshot(
        self,
        session: KaraokeSession,
        *,
        collect_samples: bool = False,
        sample_collection_enabled: bool = False,
    ) -> None:
        try:
            save_karaoke_session_snapshot(
                self._build_session_snapshot(
                    session,
                    collect_samples=collect_samples,
                    sample_collection_enabled=sample_collection_enabled,
                )
            )
        except Exception as exc:
            logger.debug("Failed to persist karaoke session snapshot %s: %s", session.session_id, exc)

    def on_connect(self, auth: Optional[Dict[str, Any]] = None):
        """Handle client connection to karaoke namespace."""
        if not require_socketio_authorization(current_app, auth):
            self._audit_socket_event('karaoke.connect', allowed=False)
            logger.warning("Rejected unauthorized /karaoke Socket.IO connection")
            return False
        client_id = request.sid
        self._client_connect_time[client_id] = time.time()
        self._audit_socket_event('karaoke.connect', allowed=True)
        logger.info(f"Client {client_id[:8]}... connected to /karaoke namespace")
        emit('connected', {'status': 'connected', 'client_id': client_id})

    def on_disconnect(self):
        """Handle client disconnection with graceful cleanup (Task 8.3)."""
        client_id = request.sid

        # Clean up any active session for this client
        session_id = self._client_sessions.pop(client_id, None)
        if session_id:
            session = self._sessions.pop(session_id, None)
            if session:
                # Record analytics before cleanup
                stats = session.get_stats()
                collect_samples = session_id in self._sample_collectors
                _analytics.record_session_end(
                    duration_s=stats.get('duration_s', 0),
                    chunks_processed=stats.get('chunks_processed', 0)
                )
                session.stop()
                self._persist_session_snapshot(
                    session,
                    collect_samples=collect_samples,
                    sample_collection_enabled=collect_samples,
                )
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
        if not self._authorized_or_emit(data, action='karaoke.join_session'):
            return
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
        if not self._authorized_or_emit(data, action='karaoke.leave_session'):
            return
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
                - pipeline_type: 'realtime' or 'realtime_meanvc' for live conversion (default: realtime)
        """
        if not self._authorized_or_emit(data, action='karaoke.start_session'):
            return

        session_id = data.get('session_id')
        song_id = data.get('song_id')
        vocals_path = data.get('vocals_path')
        instrumental_path = data.get('instrumental_path')
        persisted_session = load_karaoke_session_snapshot(session_id) if session_id else None
        if not vocals_path and persisted_session:
            vocals_path = persisted_session.get('vocals_path')
        if not instrumental_path and persisted_session:
            instrumental_path = persisted_session.get('instrumental_path')
        voice_model_id = data.get('voice_model_id') or (
            persisted_session.get('voice_model_id') if persisted_session else None
        )
        client_id = request.sid

        # Live karaoke only exposes the low-latency pipelines.
        pipeline_type = data.get('pipeline_type') or (
            persisted_session.get('requested_pipeline') if persisted_session else 'realtime'
        )
        if pipeline_type not in ('realtime', 'realtime_meanvc'):
            emit('error', {'message': 'Live karaoke pipeline_type must be realtime or realtime_meanvc'})
            _analytics.record_error()
            return
        logger.info(f"Session using pipeline_type: {pipeline_type}")

        # Sample collection settings (requires explicit consent)
        profile_id = data.get('profile_id') or (
            persisted_session.get('target_profile_id') if persisted_session else None
        )
        collect_samples = data.get(
            'collect_samples',
            persisted_session.get('collect_samples', False) if persisted_session else False,
        )

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

            session._source_voice_model_id = voice_model_id

            # Set speaker embedding if provided
            if 'speaker_embedding' in data:
                import base64
                embedding_bytes = base64.b64decode(data['speaker_embedding'])
                embedding = torch.from_numpy(
                    np.frombuffer(embedding_bytes, dtype=np.float32).copy()
                )
                session.set_speaker_embedding(embedding)
            elif persisted_session and persisted_session.get('speaker_embedding'):
                session.set_speaker_embedding(
                    torch.from_numpy(
                        np.asarray(persisted_session['speaker_embedding'], dtype=np.float32)
                    )
                )
            elif profile_id:
                from flask import current_app
                from auto_voice.storage.voice_profiles import VoiceProfileStore
                from auto_voice.storage.paths import (
                    resolve_data_dir,
                    resolve_profiles_dir,
                    resolve_samples_dir,
                )

                voice_cloner = getattr(current_app, 'voice_cloner', None)
                store = voice_cloner.store if voice_cloner is not None else VoiceProfileStore(
                    profiles_dir=str(resolve_profiles_dir(data_dir=str(resolve_data_dir(current_app.config.get('DATA_DIR'))))),
                    samples_dir=str(resolve_samples_dir(data_dir=str(resolve_data_dir(current_app.config.get('DATA_DIR'))))),
                )
                profile = store.load(profile_id)
                embedding = store.load_speaker_embedding(profile_id)
                if embedding is None:
                    embedding = profile.get('embedding')
                if embedding is None:
                    raise ValueError(
                        f'Profile {profile_id} has no speaker embedding for live conversion'
                    )
                session.set_speaker_embedding(
                    torch.from_numpy(np.asarray(embedding, dtype=np.float32))
                )
                session.voice_model_id = profile_id
                session._target_profile_id = profile_id
                session._target_model_type = profile.get('active_model_type', 'base')
                session._profiles_dir = store.profiles_dir
                full_model_path = os.path.join(store.trained_models_dir, f"{profile_id}_full_model.pt")
                session._full_model_path = full_model_path if os.path.exists(full_model_path) else None
            elif voice_model_id:
                from .karaoke_api import _get_voice_model_registry

                registry = _get_voice_model_registry()
                session.load_voice_model(registry, voice_model_id)
                session._target_model_type = 'registry_model'

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
            self._persist_session_snapshot(
                session,
                collect_samples=collect_samples,
                sample_collection_enabled=sample_collection_enabled,
            )
            audio_router_targets = _load_audio_router_config()

            emit('session_started', {
                'session_id': session_id,
                'status': 'ready',
                'sample_collection_enabled': sample_collection_enabled,
                'target_profile_id': profile_id,
                'active_model_type': getattr(session, '_target_model_type', None),
                'source_voice_model_id': voice_model_id,
                'requested_pipeline': pipeline_type,
                'resolved_pipeline': pipeline_type,
                'runtime_backend': session.pipeline_type,
                'audio_router_targets': {
                    'speaker_device': audio_router_targets.get('speaker_device'),
                    'headphone_device': audio_router_targets.get('headphone_device'),
                },
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
        if not self._authorized_or_emit(data, action='karaoke.stop_session'):
            return
        session_id = data.get('session_id')
        if not session_id:
            return

        session = self._sessions.get(session_id)
        if session:
            stats = session.get_stats()
            collect_samples = session_id in self._sample_collectors
            session.stop()
            del self._sessions[session_id]
            for client_id, mapped_session_id in list(self._client_sessions.items()):
                if mapped_session_id == session_id:
                    self._client_sessions.pop(client_id, None)

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

            self._persist_session_snapshot(
                session,
                collect_samples=collect_samples,
                sample_collection_enabled=collect_samples,
            )
            cleanup_session(session_id, reason='stopped')

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
        if not self._authorized_or_emit(data, action='karaoke.audio_chunk'):
            return

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
            audio_np = np.frombuffer(audio_bytes, dtype=np.float32).copy()
            audio_tensor = torch.from_numpy(audio_np)

            # Process through conversion pipeline
            converted = session.process_chunk(audio_tensor)
            latency_ms = session.get_latency_ms()

            # Record analytics (Task 8.4)
            # Assume 24kHz sample rate, calculate seconds from samples
            audio_seconds = len(audio_np) / 24000.0
            _analytics.record_audio_processed(audio_seconds, latency_ms)
            self._persist_session_snapshot(
                session,
                collect_samples=session_id in self._sample_collectors,
                sample_collection_enabled=session_id in self._sample_collectors,
            )

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
        if not self._authorized_or_emit(data, action='karaoke.set_speaker_embedding'):
            return

        session_id = data.get('session_id')
        session = self._sessions.get(session_id)

        if not session:
            emit('error', {'message': 'Session not found'})
            return

        try:
            import base64
            embedding_bytes = base64.b64decode(data['speaker_embedding'])
            embedding = torch.from_numpy(
                np.frombuffer(embedding_bytes, dtype=np.float32).copy()
            )
            session.set_speaker_embedding(embedding)
            self._persist_session_snapshot(
                session,
                collect_samples=session_id in self._sample_collectors,
                sample_collection_enabled=session_id in self._sample_collectors,
            )
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
