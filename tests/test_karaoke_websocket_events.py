"""Tests for Karaoke WebSocket events.

Tests the /karaoke namespace WebSocket functionality:
- connect/disconnect events
- start_session/stop_session
- audio_chunk streaming
- error handling
- session management
- analytics tracking
"""
import base64
import json
import time
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
import torch


class TestKaraokeNamespaceImport:
    """Test KaraokeNamespace can be imported."""

    def test_import_karaoke_namespace(self):
        """KaraokeNamespace can be imported."""
        from auto_voice.web.karaoke_events import KaraokeNamespace
        assert KaraokeNamespace is not None

    def test_import_register_function(self):
        """register_karaoke_namespace can be imported."""
        from auto_voice.web.karaoke_events import register_karaoke_namespace
        assert register_karaoke_namespace is not None

    def test_import_analytics(self):
        """KaraokeAnalytics can be imported."""
        from auto_voice.web.karaoke_events import KaraokeAnalytics, get_karaoke_analytics
        assert KaraokeAnalytics is not None
        assert get_karaoke_analytics is not None


class TestKaraokeAnalytics:
    """Test KaraokeAnalytics metrics tracking."""

    def test_analytics_init(self):
        """KaraokeAnalytics initializes with zero metrics."""
        from auto_voice.web.karaoke_events import KaraokeAnalytics

        analytics = KaraokeAnalytics()
        metrics = analytics.get_metrics()

        assert metrics['total_sessions'] == 0
        assert metrics['total_chunks_processed'] == 0
        assert metrics['total_errors'] == 0

    def test_record_session_start(self):
        """record_session_start increments session count."""
        from auto_voice.web.karaoke_events import KaraokeAnalytics

        analytics = KaraokeAnalytics()
        analytics.record_session_start()
        analytics.record_session_start()

        metrics = analytics.get_metrics()
        assert metrics['total_sessions'] == 2

    def test_record_session_end(self):
        """record_session_end updates duration and chunk stats."""
        from auto_voice.web.karaoke_events import KaraokeAnalytics

        analytics = KaraokeAnalytics()
        analytics.record_session_end(duration_s=60.0, chunks_processed=100)

        metrics = analytics.get_metrics()
        assert metrics['total_chunks_processed'] == 100
        assert metrics['avg_session_duration_s'] > 0

    def test_record_audio_processed(self):
        """record_audio_processed tracks audio and latency."""
        from auto_voice.web.karaoke_events import KaraokeAnalytics

        analytics = KaraokeAnalytics()
        analytics.record_audio_processed(seconds=60.0, latency_ms=50.0)  # 1 minute
        analytics.record_audio_processed(seconds=60.0, latency_ms=60.0)  # 1 minute

        metrics = analytics.get_metrics()
        assert metrics['total_audio_minutes'] >= 1  # At least 1 minute (rounded)
        assert metrics['avg_latency_ms'] == 55.0

    def test_record_error(self):
        """record_error increments error count."""
        from auto_voice.web.karaoke_events import KaraokeAnalytics

        analytics = KaraokeAnalytics()
        analytics.record_error()
        analytics.record_error()

        metrics = analytics.get_metrics()
        assert metrics['total_errors'] == 2

    def test_get_metrics_returns_dict(self):
        """get_metrics returns complete metrics dict."""
        from auto_voice.web.karaoke_events import KaraokeAnalytics

        analytics = KaraokeAnalytics()
        metrics = analytics.get_metrics()

        assert 'total_sessions' in metrics
        assert 'total_chunks_processed' in metrics
        assert 'total_audio_minutes' in metrics
        assert 'total_errors' in metrics
        assert 'avg_session_duration_s' in metrics
        assert 'avg_latency_ms' in metrics
        assert 'sessions_last_24h' in metrics


class TestKaraokeNamespaceInit:
    """Test KaraokeNamespace initialization."""

    def test_namespace_init(self):
        """KaraokeNamespace initializes with correct namespace."""
        from auto_voice.web.karaoke_events import KaraokeNamespace

        ns = KaraokeNamespace()
        assert ns.namespace == '/karaoke'

    def test_namespace_custom_path(self):
        """KaraokeNamespace accepts custom namespace path."""
        from auto_voice.web.karaoke_events import KaraokeNamespace

        ns = KaraokeNamespace('/custom_karaoke')
        assert ns.namespace == '/custom_karaoke'

    def test_namespace_has_sessions_dict(self):
        """KaraokeNamespace has sessions tracking."""
        from auto_voice.web.karaoke_events import KaraokeNamespace

        ns = KaraokeNamespace()
        assert hasattr(ns, '_sessions')
        assert isinstance(ns._sessions, dict)


class TestConnectDisconnect:
    """Connect/disconnect handlers are covered by context-aware integration tests."""

    def test_context_coverage_lives_in_dedicated_module(self):
        """Keep this unit module free of Flask context placeholders."""
        import tests.test_karaoke_websocket_context as context_tests

        assert hasattr(context_tests, 'test_on_connect_emits_connected')
        assert hasattr(context_tests, 'test_on_disconnect_cleans_up_session_and_collector')


class TestJoinLeaveSession:
    """Test join_session and leave_session events."""

    def test_on_join_session_without_id_emits_error(self):
        """on_join_session emits error without session_id."""
        from auto_voice.web.karaoke_events import KaraokeNamespace

        ns = KaraokeNamespace()

        with patch('auto_voice.web.karaoke_events.emit') as mock_emit:
            ns.on_join_session({})

        mock_emit.assert_called_once()
        assert mock_emit.call_args[0][0] == 'error'

    def test_on_join_session_joins_room(self):
        """on_join_session joins the session room."""
        from auto_voice.web.karaoke_events import KaraokeNamespace

        ns = KaraokeNamespace()

        with patch('auto_voice.web.karaoke_events.join_room') as mock_join:
            with patch('auto_voice.web.karaoke_events.emit'):
                ns.on_join_session({'session_id': 'session-abc'})

        mock_join.assert_called_with('session-abc')

    def test_on_leave_session_leaves_room(self):
        """on_leave_session leaves the session room."""
        from auto_voice.web.karaoke_events import KaraokeNamespace

        ns = KaraokeNamespace()

        with patch('auto_voice.web.karaoke_events.leave_room') as mock_leave:
            with patch('auto_voice.web.karaoke_events.emit'):
                ns.on_leave_session({'session_id': 'session-abc'})

        mock_leave.assert_called_with('session-abc')


class TestStartSession:
    """start_session handler coverage is owned by the context-aware test module."""

    def test_start_session_context_coverage_exists(self):
        """Guard against regressing back to skip-only start_session coverage."""
        import tests.test_karaoke_websocket_context as context_tests

        assert hasattr(context_tests, 'test_on_start_session_missing_params_emits_error')
        assert hasattr(context_tests, 'test_on_start_session_creates_session_and_registers')
        assert hasattr(context_tests, 'test_on_start_session_with_embedding_sets_speaker_embedding')


class TestStopSession:
    """Test stop_session event handler."""

    def test_on_stop_session_stops_and_removes_session(self):
        """on_stop_session stops and removes the session."""
        from auto_voice.web.karaoke_events import KaraokeNamespace

        ns = KaraokeNamespace()

        mock_session = MagicMock()
        mock_session.get_stats.return_value = {'duration_s': 30, 'chunks_processed': 50}
        ns._sessions['session-abc'] = mock_session

        with patch('auto_voice.web.karaoke_events.emit'):
            ns.on_stop_session({'session_id': 'session-abc'})

        mock_session.stop.assert_called_once()
        assert 'session-abc' not in ns._sessions

    def test_on_stop_session_emits_session_stopped(self):
        """on_stop_session emits session_stopped event with stats."""
        from auto_voice.web.karaoke_events import KaraokeNamespace

        ns = KaraokeNamespace()

        mock_session = MagicMock()
        mock_session.get_stats.return_value = {'duration_s': 30, 'chunks_processed': 50}
        ns._sessions['session-abc'] = mock_session

        with patch('auto_voice.web.karaoke_events.emit') as mock_emit:
            ns.on_stop_session({'session_id': 'session-abc'})

        # Find session_stopped emit
        stopped_calls = [c for c in mock_emit.call_args_list if c[0][0] == 'session_stopped']
        assert len(stopped_calls) == 1
        assert stopped_calls[0][0][1]['session_id'] == 'session-abc'
        assert 'stats' in stopped_calls[0][0][1]


class TestAudioChunk:
    """audio_chunk handler coverage is owned by the context-aware test module."""

    def test_audio_chunk_context_coverage_exists(self):
        """Guard against regressing back to skip-only audio_chunk coverage."""
        import tests.test_karaoke_websocket_context as context_tests

        assert hasattr(context_tests, 'test_on_audio_chunk_without_session_emits_error')
        assert hasattr(context_tests, 'test_on_audio_chunk_processes_audio_and_emits_result')
        assert hasattr(context_tests, 'test_on_audio_chunk_processing_error_emits_error')


class TestSetSpeakerEmbedding:
    """Test set_speaker_embedding event handler."""

    def test_on_set_speaker_embedding_no_session_emits_error(self):
        """on_set_speaker_embedding emits error for unknown session."""
        from auto_voice.web.karaoke_events import KaraokeNamespace

        ns = KaraokeNamespace()

        with patch('auto_voice.web.karaoke_events.emit') as mock_emit:
            ns.on_set_speaker_embedding({'session_id': 'unknown'})

        error_calls = [c for c in mock_emit.call_args_list if c[0][0] == 'error']
        assert len(error_calls) > 0

    def test_on_set_speaker_embedding_updates_session(self):
        """on_set_speaker_embedding updates session embedding."""
        from auto_voice.web.karaoke_events import KaraokeNamespace

        ns = KaraokeNamespace()

        mock_session = MagicMock()
        ns._sessions['session-abc'] = mock_session

        embedding = np.random.randn(256).astype(np.float32)
        embedding_b64 = base64.b64encode(embedding.tobytes()).decode('utf-8')

        with patch('auto_voice.web.karaoke_events.emit') as mock_emit:
            ns.on_set_speaker_embedding({
                'session_id': 'session-abc',
                'speaker_embedding': embedding_b64,
            })

        mock_session.set_speaker_embedding.assert_called_once()

    def test_on_set_speaker_embedding_emits_updated(self):
        """on_set_speaker_embedding emits embedding_updated event."""
        from auto_voice.web.karaoke_events import KaraokeNamespace

        ns = KaraokeNamespace()

        mock_session = MagicMock()
        ns._sessions['session-abc'] = mock_session

        embedding = np.random.randn(256).astype(np.float32)
        embedding_b64 = base64.b64encode(embedding.tobytes()).decode('utf-8')

        with patch('auto_voice.web.karaoke_events.emit') as mock_emit:
            ns.on_set_speaker_embedding({
                'session_id': 'session-abc',
                'speaker_embedding': embedding_b64,
            })

        updated_calls = [c for c in mock_emit.call_args_list if c[0][0] == 'embedding_updated']
        assert len(updated_calls) == 1


class TestRegisterNamespace:
    """Test namespace registration."""

    def test_register_karaoke_namespace(self):
        """register_karaoke_namespace registers with SocketIO."""
        from auto_voice.web.karaoke_events import register_karaoke_namespace

        mock_socketio = MagicMock()

        ns = register_karaoke_namespace(mock_socketio)

        mock_socketio.on_namespace.assert_called_once()
        assert ns is not None


class TestSampleCollection:
    """Test training sample collection during karaoke."""

    def test_start_session_sample_collection_context_coverage_exists(self):
        """Guard against regressing back to skip-only sample collection coverage."""
        import tests.test_karaoke_websocket_context as context_tests

        assert hasattr(context_tests, 'test_on_start_session_with_profile_and_sample_collection')

    def test_stop_session_reports_samples_collected(self):
        """stop_session reports number of samples collected."""
        from auto_voice.web.karaoke_events import KaraokeNamespace

        ns = KaraokeNamespace()

        mock_session = MagicMock()
        mock_session.get_stats.return_value = {'duration_s': 60, 'chunks_processed': 100}
        ns._sessions['session-abc'] = mock_session

        mock_collector = MagicMock()
        mock_collector.stop_recording.return_value = ['sample1', 'sample2', 'sample3']
        ns._sample_collectors['session-abc'] = mock_collector

        with patch('auto_voice.web.karaoke_events.emit') as mock_emit:
            ns.on_stop_session({'session_id': 'session-abc'})

        # Check samples_collected in response
        stopped_calls = [c for c in mock_emit.call_args_list if c[0][0] == 'session_stopped']
        assert len(stopped_calls) == 1
        assert stopped_calls[0][0][1]['samples_collected'] == 3

        # Should also emit samples_collected event
        sample_calls = [c for c in mock_emit.call_args_list if c[0][0] == 'samples_collected']
        assert len(sample_calls) == 1
        assert sample_calls[0][0][1]['count'] == 3
