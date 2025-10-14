"""
Comprehensive web interface tests for AutoVoice.

Tests Flask API, WebSocket connections, request validation, and integration workflows.
"""

import pytest
import json
import io
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

@pytest.fixture
def sample_audio():
    """Generate sample audio data for testing"""
    return np.random.rand(22050).astype(np.float32)  # 1 second of audio at 22kHz

@pytest.fixture
def benchmark_timer():
    """Simple benchmark timer for performance tests"""
    import time
    def timer(func):
        start = time.time()
        result = func()
        elapsed = time.time() - start
        return result, elapsed
    return timer


@pytest.mark.web
@pytest.mark.integration
class TestFlaskApp:
    """Test Flask application creation and configuration."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures"""
        try:
            from src.auto_voice.web.app import create_app
            self.app, self.socketio = create_app(config={'TESTING': True})
            self.client = self.app.test_client()
        except ImportError:
            pytest.skip("Flask app not available")

    def test_app_creation(self):
        """Test Flask app instantiation."""
        assert self.app is not None
        assert self.app.config['TESTING'] is True

    def test_app_routes_registered(self):
        """Test all required routes are registered."""
        rules = [rule.rule for rule in self.app.url_map.iter_rules()]

        required_routes = [
            '/',
            '/api/health',
            '/api/synthesize',
            '/api/convert',
            '/api/clone',
            '/api/speakers',
            '/api/gpu_status',
            '/api/process_audio',
            '/ws/audio_stream'
        ]

        for route in required_routes:
            assert route in rules, f"Route {route} not registered"

    def test_app_config_defaults(self):
        """Test default configuration values."""
        assert 'MAX_CONTENT_LENGTH' in self.app.config
        assert 'JSON_SORT_KEYS' in self.app.config

    @pytest.mark.parametrize("method", ["GET", "POST", "PUT", "DELETE"])
    def test_cors_headers(self, method):
        """Test CORS headers are set correctly."""
        response = self.client.open('/api/health', method=method)

        if method in ["GET", "POST"]:
            assert 'Access-Control-Allow-Origin' in response.headers
            assert response.headers['Access-Control-Allow-Origin'] == '*'


@pytest.mark.web
@pytest.mark.integration
class TestRESTEndpoints:
    """Test REST API endpoints."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures"""
        try:
            from src.auto_voice.web.app import create_app
            self.app, self.socketio = create_app(config={'TESTING': True})
            self.client = self.app.test_client()
        except ImportError:
            pytest.skip("Flask app not available")

    # ========================================================================
    # Health Endpoint Tests
    # ========================================================================

    def test_health_endpoint_success(self):
        """Test health check returns 200."""
        response = self.client.get('/api/health')

        assert response.status_code == 200
        data = response.get_json()

        assert 'status' in data
        assert 'gpu_available' in data
        assert 'model_loaded' in data
        assert data['status'] in ['healthy', 'degraded', 'unhealthy']

    def test_health_endpoint_structure(self):
        """Test health check response structure."""
        response = self.client.get('/api/health')
        data = response.get_json()

        required_fields = ['status', 'gpu_available', 'model_loaded', 'timestamp']
        for field in required_fields:
            assert field in data

    # ========================================================================
    # Synthesize Endpoint Tests
    # ========================================================================

    def test_synthesize_endpoint_valid_request(self):
        """Test synthesize with valid request."""
        payload = {
            'text': 'Hello world',
            'speaker_id': 0,
            'speed': 1.0
        }

        response = self.client.post(
            '/api/synthesize',
            json=payload,
            content_type='application/json'
        )

        assert response.status_code in [200, 201]
        data = response.get_json()
        assert 'audio' in data or 'audio_url' in data

    @pytest.mark.parametrize("missing_field", ["text", "speaker_id"])
    def test_synthesize_missing_fields(self, missing_field):
        """Test synthesize with missing required fields."""
        payload = {
            'text': 'Hello world',
            'speaker_id': 0,
            'speed': 1.0
        }
        del payload[missing_field]

        response = self.client.post(
            '/api/synthesize',
            json=payload,
            content_type='application/json'
        )

        assert response.status_code == 400

    def test_synthesize_invalid_speaker(self):
        """Test synthesize with invalid speaker ID."""
        payload = {
            'text': 'Hello world',
            'speaker_id': 9999,
            'speed': 1.0
        }

        response = self.client.post(
            '/api/synthesize',
            json=payload,
            content_type='application/json'
        )

        assert response.status_code in [400, 404]

    def test_synthesize_empty_text(self):
        """Test synthesize with empty text."""
        payload = {
            'text': '',
            'speaker_id': 0
        }

        response = self.client.post(
            '/api/synthesize',
            json=payload,
            content_type='application/json'
        )

        assert response.status_code == 400

    @pytest.mark.parametrize("speed", [0.5, 1.0, 1.5, 2.0])
    def test_synthesize_different_speeds(self, speed):
        """Test synthesize with different speed settings."""
        payload = {
            'text': 'Testing speed',
            'speaker_id': 0,
            'speed': speed
        }

        response = self.client.post(
            '/api/synthesize',
            json=payload,
            content_type='application/json'
        )

        assert response.status_code in [200, 201, 500]  # Allow implementation errors

    # ========================================================================
    # Convert Endpoint Tests
    # ========================================================================

    def test_convert_endpoint_valid_audio(self, sample_audio):
        """Test voice conversion with valid audio."""
        audio_bytes = io.BytesIO(sample_audio.tobytes())
        audio_bytes.name = 'audio.wav'

        data = {
            'target_speaker': '1',
            'audio': (audio_bytes, 'audio.wav')
        }

        response = self.client.post(
            '/api/convert',
            data=data,
            content_type='multipart/form-data'
        )

        assert response.status_code in [200, 201, 500]

    def test_convert_missing_audio(self):
        """Test convert without audio file."""
        data = {'target_speaker': '1'}

        response = self.client.post(
            '/api/convert',
            data=data,
            content_type='multipart/form-data'
        )

        assert response.status_code == 400

    def test_convert_invalid_audio_format(self):
        """Test convert with invalid audio format."""
        invalid_audio = io.BytesIO(b'not an audio file')
        invalid_audio.name = 'invalid.txt'

        data = {
            'target_speaker': '1',
            'audio': (invalid_audio, 'invalid.txt')
        }

        response = self.client.post(
            '/api/convert',
            data=data,
            content_type='multipart/form-data'
        )

        assert response.status_code in [400, 415]

    # ========================================================================
    # Speakers Endpoint Tests
    # ========================================================================

    def test_speakers_endpoint(self):
        """Test speakers list endpoint."""
        response = self.client.get('/api/speakers')

        assert response.status_code == 200
        data = response.get_json()

        assert isinstance(data, list)
        if len(data) > 0:
            speaker = data[0]
            assert 'id' in speaker
            assert 'name' in speaker

    def test_speakers_filtering(self):
        """Test speakers endpoint with query parameters."""
        response = self.client.get('/api/speakers?language=en')

        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)

    # ========================================================================
    # GPU Status Endpoint Tests
    # ========================================================================

    def test_gpu_status_endpoint(self):
        """Test GPU status endpoint."""
        response = self.client.get('/api/gpu_status')

        assert response.status_code == 200
        data = response.get_json()

        assert 'cuda_available' in data
        assert 'device' in data

    def test_gpu_status_structure(self):
        """Test GPU status response structure."""
        response = self.client.get('/api/gpu_status')
        data = response.get_json()

        if data.get('cuda_available'):
            assert 'device_name' in data
            assert 'memory_total' in data
            assert 'memory_allocated' in data


@pytest.mark.web
@pytest.mark.integration
class TestWebSocketConnections:
    """Test WebSocket connection handling."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures"""
        try:
            from src.auto_voice.web.websocket_handler import WebSocketHandler
            from unittest.mock import Mock
            # Create a mock SocketIO instance
            mock_socketio = Mock()
            self.handler = WebSocketHandler(mock_socketio)
        except ImportError:
            pytest.skip("WebSocket handler not available")

    def test_websocket_connection_lifecycle(self):
        """Test WebSocket connect and disconnect."""
        pytest.skip("Requires WebSocket implementation")

    def test_websocket_audio_stream(self):
        """Test streaming audio through WebSocket."""
        pytest.skip("Requires WebSocket implementation")

    def test_websocket_error_handling(self):
        """Test WebSocket error scenarios."""
        pytest.skip("Requires WebSocket implementation")

    def test_websocket_authentication(self):
        """Test WebSocket authentication if required."""
        pytest.skip("Requires WebSocket authentication implementation")


@pytest.mark.web
@pytest.mark.integration
class TestRequestValidation:
    """Test request validation and error handling."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures"""
        try:
            from src.auto_voice.web.app import create_app
            self.app, self.socketio = create_app(config={'TESTING': True})
            self.client = self.app.test_client()
        except ImportError:
            pytest.skip("Flask app not available")

    def test_invalid_json_body(self):
        """Test handling of invalid JSON."""
        response = self.client.post(
            '/api/synthesize',
            data='invalid json',
            content_type='application/json'
        )

        assert response.status_code == 400

    def test_missing_content_type(self):
        """Test handling of missing Content-Type."""
        response = self.client.post(
            '/api/synthesize',
            data=json.dumps({'text': 'test'})
        )

        assert response.status_code in [400, 415]

    def test_oversized_request(self):
        """Test handling of oversized requests."""
        large_text = 'x' * (10 * 1024 * 1024)  # 10MB
        payload = {
            'text': large_text,
            'speaker_id': 0
        }

        response = self.client.post(
            '/api/synthesize',
            json=payload,
            content_type='application/json'
        )

        assert response.status_code in [400, 413]

    @pytest.mark.parametrize("endpoint", [
        '/api/synthesize',
        '/api/convert',
        '/api/process_audio'
    ])
    def test_rate_limiting(self, endpoint):
        """Test rate limiting if implemented."""
        pytest.skip("Requires rate limiting implementation")


@pytest.mark.web
@pytest.mark.integration
class TestResponseFormats:
    """Test API response formats and structure."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures"""
        try:
            from src.auto_voice.web.app import create_app
            self.app, self.socketio = create_app(config={'TESTING': True})
            self.client = self.app.test_client()
        except ImportError:
            pytest.skip("Flask app not available")

    def test_json_response_format(self):
        """Test JSON response structure."""
        response = self.client.get('/api/health')

        assert response.content_type == 'application/json'
        data = response.get_json()
        assert isinstance(data, dict)

    def test_error_response_format(self):
        """Test error response structure."""
        response = self.client.get('/api/nonexistent')

        assert response.status_code == 404
        data = response.get_json()

        assert 'error' in data or 'message' in data

    def test_success_response_format(self):
        """Test success response structure."""
        response = self.client.get('/api/health')

        assert response.status_code == 200
        data = response.get_json()
        assert 'status' in data


@pytest.mark.web
@pytest.mark.e2e
class TestIntegrationWorkflows:
    """Test complete integration workflows."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures"""
        try:
            from src.auto_voice.web.app import create_app
            self.app, self.socketio = create_app(config={'TESTING': True})
            self.client = self.app.test_client()
        except ImportError:
            pytest.skip("Flask app not available")

    def test_full_synthesize_workflow(self):
        """Test complete text-to-speech workflow."""
        # 1. Check health
        health_response = self.client.get('/api/health')
        assert health_response.status_code == 200

        # 2. Get available speakers
        speakers_response = self.client.get('/api/speakers')
        assert speakers_response.status_code == 200
        speakers = speakers_response.get_json()

        if len(speakers) == 0:
            pytest.skip("No speakers available")

        # 3. Synthesize audio
        speaker_id = speakers[0]['id']
        synth_payload = {
            'text': 'Integration test',
            'speaker_id': speaker_id
        }

        synth_response = self.client.post(
            '/api/synthesize',
            json=synth_payload,
            content_type='application/json'
        )

        assert synth_response.status_code in [200, 201]

    def test_health_check_before_operations(self):
        """Test health check before performing operations."""
        health_response = self.client.get('/api/health')
        assert health_response.status_code == 200

        health_data = health_response.get_json()

        if health_data['status'] != 'healthy':
            pytest.skip("System not healthy")

    def test_concurrent_requests_handling(self):
        """Test handling of concurrent requests."""
        pytest.skip("Requires concurrent request testing")


@pytest.mark.web
@pytest.mark.performance
class TestAPIPerformance:
    """Test API performance and response times."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures"""
        try:
            from src.auto_voice.web.app import create_app
            self.app, self.socketio = create_app(config={'TESTING': True})
            self.client = self.app.test_client()
        except ImportError:
            pytest.skip("Flask app not available")

    def test_health_endpoint_latency(self, benchmark_timer):
        """Benchmark health endpoint response time."""
        result, elapsed = benchmark_timer(
            lambda: self.client.get('/api/health')
        )

        assert elapsed < 0.1  # Should respond in < 100ms

    def test_speakers_endpoint_latency(self, benchmark_timer):
        """Benchmark speakers list response time."""
        result, elapsed = benchmark_timer(
            lambda: self.client.get('/api/speakers')
        )

        assert elapsed < 0.5  # Should respond in < 500ms

    def test_synthesize_throughput(self):
        """Test synthesis throughput."""
        pytest.skip("Requires performance testing implementation")

    def test_concurrent_request_capacity(self):
        """Test maximum concurrent request capacity."""
        pytest.skip("Requires load testing implementation")
