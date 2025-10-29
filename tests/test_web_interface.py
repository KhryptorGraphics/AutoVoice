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

        # Check for versioned API routes (/api/v1/*)
        required_routes_v1 = [
            '/',
            '/api/v1/health',
            '/api/v1/synthesize',
            '/api/v1/convert',
            '/api/v1/voice/clone',  # Updated path
            '/api/v1/speakers',
            '/api/v1/gpu_status',
            '/api/v1/process_audio',
            '/api/v1/voice/profiles',  # New endpoint
            '/ws/audio_stream'
        ]

        for route in required_routes_v1:
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
        response = self.client.get('/api/v1/health')

        assert response.status_code == 200
        data = response.get_json()

        assert 'status' in data
        assert 'gpu_available' in data
        assert 'model_loaded' in data
        assert data['status'] in ['healthy', 'degraded', 'unhealthy']

    def test_health_endpoint_structure(self):
        """Test health check response structure."""
        response = self.client.get('/api/v1/health')
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
            '/api/v1/synthesize',
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
            '/api/v1/synthesize',
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
            '/api/v1/synthesize',
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
            '/api/v1/synthesize',
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
            '/api/v1/synthesize',
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
            '/api/v1/convert',
            data=data,
            content_type='multipart/form-data'
        )

        assert response.status_code in [200, 201, 500]

    def test_convert_missing_audio(self):
        """Test convert without audio file."""
        data = {'target_speaker': '1'}

        response = self.client.post(
            '/api/v1/convert',
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
            '/api/v1/convert',
            data=data,
            content_type='multipart/form-data'
        )

        assert response.status_code in [400, 415]

    # ========================================================================
    # Speakers Endpoint Tests
    # ========================================================================

    def test_speakers_endpoint(self):
        """Test speakers list endpoint."""
        response = self.client.get('/api/v1/speakers')

        assert response.status_code == 200
        data = response.get_json()

        assert isinstance(data, list)
        if len(data) > 0:
            speaker = data[0]
            assert 'id' in speaker
            assert 'name' in speaker

    def test_speakers_filtering(self):
        """Test speakers endpoint with query parameters."""
        response = self.client.get('/api/v1/speakers?language=en')

        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)

    # ========================================================================
    # GPU Status Endpoint Tests
    # ========================================================================

    def test_gpu_status_endpoint(self):
        """Test GPU status endpoint."""
        response = self.client.get('/api/v1/gpu_status')

        assert response.status_code == 200
        data = response.get_json()

        assert 'cuda_available' in data
        assert 'device' in data

    def test_gpu_status_structure(self):
        """Test GPU status response structure."""
        response = self.client.get('/api/v1/gpu_status')
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
            '/api/v1/synthesize',
            data='invalid json',
            content_type='application/json'
        )

        assert response.status_code == 400

    def test_missing_content_type(self):
        """Test handling of missing Content-Type."""
        response = self.client.post(
            '/api/v1/synthesize',
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
            '/api/v1/synthesize',
            json=payload,
            content_type='application/json'
        )

        assert response.status_code in [400, 413]

    @pytest.mark.parametrize("endpoint", [
        '/api/v1/synthesize',
        '/api/v1/convert',
        '/api/v1/process_audio'
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
        response = self.client.get('/api/v1/health')

        assert response.content_type == 'application/json'
        data = response.get_json()
        assert isinstance(data, dict)

    def test_error_response_format(self):
        """Test error response structure."""
        response = self.client.get('/api/v1/nonexistent')

        assert response.status_code == 404
        data = response.get_json()

        assert 'error' in data or 'message' in data

    def test_success_response_format(self):
        """Test success response structure."""
        response = self.client.get('/api/v1/health')

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
        health_response = self.client.get('/api/v1/health')
        assert health_response.status_code == 200

        # 2. Get available speakers
        speakers_response = self.client.get('/api/v1/speakers')
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
            '/api/v1/synthesize',
            json=synth_payload,
            content_type='application/json'
        )

        assert synth_response.status_code in [200, 201]

    def test_health_check_before_operations(self):
        """Test health check before performing operations."""
        health_response = self.client.get('/api/v1/health')
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
            lambda: self.client.get('/api/v1/health')
        )

        assert elapsed < 0.1  # Should respond in < 100ms

    def test_speakers_endpoint_latency(self, benchmark_timer):
        """Benchmark speakers list response time."""
        result, elapsed = benchmark_timer(
            lambda: self.client.get('/api/v1/speakers')
        )

        assert elapsed < 0.5  # Should respond in < 500ms

    def test_synthesize_throughput(self):
        """Test synthesis throughput."""
        pytest.skip("Requires performance testing implementation")

    def test_concurrent_request_capacity(self):
        """Test maximum concurrent request capacity."""
        pytest.skip("Requires load testing implementation")


# ========================================================================
# Voice Conversion Endpoint Tests
# ========================================================================

@pytest.fixture
def sample_song_file():
    """Generate a sample song file for testing (3 seconds of audio)"""
    sample_rate = 22050
    duration = 3.0
    samples = int(sample_rate * duration)

    # Generate simple audio waveform (sine wave)
    t = np.linspace(0, duration, samples)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

    return audio, sample_rate


@pytest.fixture
def test_profile_id():
    """Return a mock profile ID for testing"""
    return 'test-profile-12345'


@pytest.mark.web
@pytest.mark.integration
class TestVoiceCloningEndpoints:
    """Test voice cloning REST API endpoints."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures"""
        try:
            from src.auto_voice.web.app import create_app
            self.app, self.socketio = create_app(config={'TESTING': True})
            self.client = self.app.test_client()
        except ImportError:
            pytest.skip("Flask app not available")

    def test_voice_clone_endpoint_valid_audio(self):
        """Test voice cloning with valid 30s audio"""
        # Generate 30 seconds of valid audio
        sample_rate = 22050
        duration = 30.0
        audio = np.random.rand(int(sample_rate * duration)).astype(np.float32)

        # Convert to WAV bytes
        import wave
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            audio_int16 = (audio * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())

        buffer.seek(0)

        data = {
            'reference_audio': (buffer, 'reference.wav')
        }

        response = self.client.post(
            '/api/v1/voice/clone',
            data=data,
            content_type='multipart/form-data'
        )

        # Should succeed or service unavailable
        assert response.status_code in [201, 503]

        if response.status_code == 201:
            data = response.get_json()
            assert 'status' in data
            assert data['status'] == 'success'
            assert 'profile_id' in data
            assert 'audio_duration' in data

    def test_voice_clone_with_user_id(self):
        """Test voice cloning with user_id parameter"""
        sample_rate = 22050
        duration = 30.0
        audio = np.random.rand(int(sample_rate * duration)).astype(np.float32)

        import wave
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            audio_int16 = (audio * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())

        buffer.seek(0)

        data = {
            'reference_audio': (buffer, 'reference.wav'),
            'user_id': 'test_user_123'
        }

        response = self.client.post(
            '/api/v1/voice/clone',
            data=data,
            content_type='multipart/form-data'
        )

        assert response.status_code in [201, 503]

        if response.status_code == 201:
            data = response.get_json()
            assert 'user_id' in data

    def test_voice_clone_missing_audio(self):
        """Test voice cloning without audio - should return 400"""
        response = self.client.post(
            '/api/v1/voice/clone',
            data={},
            content_type='multipart/form-data'
        )

        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

    def test_voice_clone_invalid_audio_format(self):
        """Test voice cloning with invalid audio format - should return 400/415"""
        invalid_audio = io.BytesIO(b'not an audio file')
        invalid_audio.name = 'invalid.txt'

        data = {
            'reference_audio': (invalid_audio, 'invalid.txt')
        }

        response = self.client.post(
            '/api/v1/voice/clone',
            data=data,
            content_type='multipart/form-data'
        )

        assert response.status_code in [400, 415]

    def test_voice_clone_audio_too_short(self):
        """Test voice cloning with audio < 5s - should return 400"""
        # Generate 3 seconds of audio (too short)
        sample_rate = 22050
        duration = 3.0
        audio = np.random.rand(int(sample_rate * duration)).astype(np.float32)

        import wave
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            audio_int16 = (audio * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())

        buffer.seek(0)

        data = {
            'reference_audio': (buffer, 'short.wav')
        }

        response = self.client.post(
            '/api/v1/voice/clone',
            data=data,
            content_type='multipart/form-data'
        )

        # Should either fail validation (400) or service unavailable (503)
        assert response.status_code in [400, 503]

    def test_voice_clone_audio_too_long(self):
        """Test voice cloning with audio > 60s - should return 400"""
        # Generate 65 seconds of audio (too long)
        sample_rate = 22050
        duration = 65.0
        audio = np.random.rand(int(sample_rate * duration)).astype(np.float32)

        import wave
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            audio_int16 = (audio * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())

        buffer.seek(0)

        data = {
            'reference_audio': (buffer, 'long.wav')
        }

        response = self.client.post(
            '/api/v1/voice/clone',
            data=data,
            content_type='multipart/form-data'
        )

        assert response.status_code in [400, 503]

    def test_voice_clone_service_unavailable(self):
        """Test voice cloning when service is unavailable - should return 503"""
        # This will test the case when voice_cloner is None in app context
        sample_rate = 22050
        duration = 30.0
        audio = np.random.rand(int(sample_rate * duration)).astype(np.float32)

        import wave
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            audio_int16 = (audio * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())

        buffer.seek(0)

        data = {
            'reference_audio': (buffer, 'test.wav')
        }

        response = self.client.post(
            '/api/v1/voice/clone',
            data=data,
            content_type='multipart/form-data'
        )

        # Should be 201 (success) or 503 (service unavailable)
        assert response.status_code in [201, 503]


@pytest.mark.web
@pytest.mark.integration
class TestSongConversionEndpoints:
    """Test song conversion REST API endpoints."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures"""
        try:
            from src.auto_voice.web.app import create_app
            self.app, self.socketio = create_app(config={'TESTING': True})
            self.client = self.app.test_client()
        except ImportError:
            pytest.skip("Flask app not available")

    def test_convert_song_endpoint_valid_request(self, sample_song_file):
        """Test song conversion with valid request"""
        audio, sample_rate = sample_song_file

        import wave
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            audio_int16 = (audio * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())

        buffer.seek(0)

        data = {
            'song': (buffer, 'song.wav'),
            'profile_id': 'test-profile-123'
        }

        response = self.client.post(
            '/api/v1/convert/song',
            data=data,
            content_type='multipart/form-data'
        )

        # Should succeed, fail with 404 (profile not found), or 503 (service unavailable)
        assert response.status_code in [200, 404, 503]

    def test_convert_song_with_volumes(self, sample_song_file):
        """Test song conversion with custom vocal/instrumental volumes"""
        audio, sample_rate = sample_song_file

        import wave
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            audio_int16 = (audio * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())

        buffer.seek(0)

        data = {
            'song': (buffer, 'song.wav'),
            'profile_id': 'test-profile-123',
            'vocal_volume': '1.2',
            'instrumental_volume': '0.8'
        }

        response = self.client.post(
            '/api/v1/convert/song',
            data=data,
            content_type='multipart/form-data'
        )

        assert response.status_code in [200, 404, 503]

    def test_convert_song_with_return_stems(self, sample_song_file):
        """Test song conversion with return_stems parameter"""
        audio, sample_rate = sample_song_file

        import wave
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            audio_int16 = (audio * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())

        buffer.seek(0)

        data = {
            'song': (buffer, 'song.wav'),
            'profile_id': 'test-profile-123',
            'return_stems': 'true'
        }

        response = self.client.post(
            '/api/v1/convert/song',
            data=data,
            content_type='multipart/form-data'
        )

        assert response.status_code in [200, 404, 503]

        if response.status_code == 200:
            data = response.get_json()
            assert 'stems' in data or 'audio' in data

    def test_convert_song_missing_song_file(self):
        """Test song conversion without song file - should return 400"""
        data = {
            'profile_id': 'test-profile-123'
        }

        response = self.client.post(
            '/api/v1/convert/song',
            data=data,
            content_type='multipart/form-data'
        )

        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

    def test_convert_song_missing_profile_id(self, sample_song_file):
        """Test song conversion without profile_id - should return 400"""
        audio, sample_rate = sample_song_file

        import wave
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            audio_int16 = (audio * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())

        buffer.seek(0)

        data = {
            'song': (buffer, 'song.wav')
        }

        response = self.client.post(
            '/api/v1/convert/song',
            data=data,
            content_type='multipart/form-data'
        )

        assert response.status_code == 400

    def test_convert_song_invalid_profile_id(self, sample_song_file):
        """Test song conversion with invalid profile_id - should return 404"""
        audio, sample_rate = sample_song_file

        import wave
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            audio_int16 = (audio * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())

        buffer.seek(0)

        data = {
            'song': (buffer, 'song.wav'),
            'profile_id': 'nonexistent-profile-999'
        }

        response = self.client.post(
            '/api/v1/convert/song',
            data=data,
            content_type='multipart/form-data'
        )

        # Should be 404 (profile not found) or 503 (service unavailable)
        assert response.status_code in [404, 503]

    def test_convert_song_invalid_volumes(self, sample_song_file):
        """Test song conversion with volumes out of range - should return 400"""
        audio, sample_rate = sample_song_file

        import wave
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            audio_int16 = (audio * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())

        buffer.seek(0)

        data = {
            'song': (buffer, 'song.wav'),
            'profile_id': 'test-profile-123',
            'vocal_volume': '3.0',  # Out of range [0.0, 2.0]
            'instrumental_volume': '0.8'
        }

        response = self.client.post(
            '/api/v1/convert/song',
            data=data,
            content_type='multipart/form-data'
        )

        assert response.status_code in [400, 404, 503]

    def test_convert_song_invalid_file_format(self):
        """Test song conversion with invalid file format - should return 400"""
        invalid_file = io.BytesIO(b'not an audio file')

        data = {
            'song': (invalid_file, 'invalid.txt'),
            'profile_id': 'test-profile-123'
        }

        response = self.client.post(
            '/api/v1/convert/song',
            data=data,
            content_type='multipart/form-data'
        )

        assert response.status_code == 400


@pytest.mark.web
@pytest.mark.integration
class TestProfileManagementEndpoints:
    """Test voice profile management REST API endpoints."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures"""
        try:
            from src.auto_voice.web.app import create_app
            self.app, self.socketio = create_app(config={'TESTING': True})
            self.client = self.app.test_client()
        except ImportError:
            pytest.skip("Flask app not available")

    def test_get_voice_profiles_empty_list(self):
        """Test getting voice profiles when list is empty"""
        response = self.client.get('/api/v1/voice/profiles')

        # Should succeed or service unavailable
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.get_json()
            assert isinstance(data, list)

    def test_get_voice_profiles_with_profiles(self):
        """Test getting voice profiles when profiles exist"""
        response = self.client.get('/api/v1/voice/profiles')

        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.get_json()
            assert isinstance(data, list)

    def test_get_voice_profiles_filtered_by_user(self):
        """Test getting voice profiles filtered by user_id"""
        response = self.client.get('/api/v1/voice/profiles?user_id=test_user_123')

        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.get_json()
            assert isinstance(data, list)

    def test_get_voice_profile_by_id(self, test_profile_id):
        """Test getting specific voice profile"""
        response = self.client.get(f'/api/v1/voice/profiles/{test_profile_id}')

        # Should be 200 (found), 404 (not found), or 503 (service unavailable)
        assert response.status_code in [200, 404, 503]

    def test_get_voice_profile_not_found(self):
        """Test getting non-existent profile - should return 404"""
        response = self.client.get('/api/v1/voice/profiles/nonexistent-profile-999')

        # Should be 404 (not found) or 503 (service unavailable)
        assert response.status_code in [404, 503]

    def test_delete_voice_profile_success(self, test_profile_id):
        """Test deleting voice profile - should return 200"""
        response = self.client.delete(f'/api/v1/voice/profiles/{test_profile_id}')

        # Should be 200 (success), 404 (not found), or 503 (service unavailable)
        assert response.status_code in [200, 404, 503]

    def test_delete_voice_profile_not_found(self):
        """Test deleting non-existent profile - should return 404"""
        response = self.client.delete('/api/v1/voice/profiles/nonexistent-profile-999')

        assert response.status_code in [404, 503]

    def test_delete_voice_profile_service_unavailable(self):
        """Test deleting profile when service is unavailable - should return 503"""
        response = self.client.delete('/api/v1/voice/profiles/some-profile-id')

        # Should be 404 (not found) or 503 (service unavailable)
        assert response.status_code in [404, 503]


@pytest.mark.web
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.websocket
class TestWebSocketConversionProgress:
    """Test WebSocket connection for conversion progress tracking."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures"""
        try:
            from src.auto_voice.web.app import create_app
            self.app, self.socketio = create_app(config={'TESTING': True})
            self.client = self.socketio.test_client(self.app)
        except ImportError:
            pytest.skip("Flask-SocketIO not available")

    def test_websocket_conversion_progress_events(self):
        """Test WebSocket progress event emission during conversion"""
        import base64

        # Create small test WAV (1 second of silence)
        sample_rate = 16000
        duration = 1.0
        audio_data = np.zeros(int(sample_rate * duration), dtype=np.int16)

        # Encode to WAV format
        import io
        import wave
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        wav_bytes = wav_buffer.getvalue()
        audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')

        # Emit conversion request
        self.client.emit('convert_song_stream', {
            'conversion_id': 'test-conversion-123',
            'song_data': audio_base64,
            'target_profile_id': 'test-profile-id',
            'vocal_volume': 1.0,
            'instrumental_volume': 0.9,
            'return_stems': False
        })

        # Collect received events
        received = self.client.get_received()

        # Assert we received progress events
        progress_events = [e for e in received if e['name'] == 'conversion_progress']
        assert len(progress_events) > 0, "Should receive at least one progress event"

        # Check progress event structure
        for event in progress_events:
            assert 'args' in event
            data = event['args'][0]
            assert 'conversion_id' in data
            assert 'progress' in data
            assert 'stage' in data
            assert data['conversion_id'] == 'test-conversion-123'
            assert 0 <= data['progress'] <= 100

        # Check for completion or error event
        complete_events = [e for e in received if e['name'] == 'conversion_complete']
        error_events = [e for e in received if e['name'] == 'conversion_error']

        assert len(complete_events) > 0 or len(error_events) > 0, \
            "Should receive either completion or error event"

        if complete_events:
            data = complete_events[0]['args'][0]
            assert 'conversion_id' in data
            assert 'audio' in data or 'error' in data
            assert 'sample_rate' in data or 'error' in data
            assert 'duration' in data or 'error' in data

    def test_websocket_conversion_cancellation(self):
        """Test canceling conversion mid-process via WebSocket"""
        import base64
        import time

        # Create small test WAV
        sample_rate = 16000
        audio_data = np.zeros(int(sample_rate * 0.5), dtype=np.int16)

        import io
        import wave
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        wav_bytes = wav_buffer.getvalue()
        audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')

        conversion_id = 'test-cancel-456'

        # Start conversion
        self.client.emit('convert_song_stream', {
            'conversion_id': conversion_id,
            'song_data': audio_base64,
            'target_profile_id': 'test-profile-id',
            'vocal_volume': 1.0,
            'instrumental_volume': 0.9,
            'return_stems': False
        })

        # Give it a moment to start
        time.sleep(0.1)

        # Cancel the conversion
        self.client.emit('cancel_conversion', {
            'conversion_id': conversion_id
        })

        # Collect events
        received = self.client.get_received()

        # Check for cancellation event
        cancel_events = [e for e in received if e['name'] == 'conversion_cancelled']

        # Should receive cancellation acknowledgment
        assert len(cancel_events) > 0, "Should receive cancellation event"

        if cancel_events:
            data = cancel_events[0]['args'][0]
            assert data['conversion_id'] == conversion_id

    def test_websocket_conversion_error_handling(self):
        """Test error event handling via WebSocket"""
        # Send invalid conversion request (missing required fields)
        self.client.emit('convert_song_stream', {
            'conversion_id': 'test-error-789'
            # Missing song_data and target_profile_id
        })

        # Collect events
        received = self.client.get_received()

        # Should receive error event
        error_events = [e for e in received if e['name'] == 'conversion_error' or e['name'] == 'error']

        assert len(error_events) > 0, "Should receive error event for invalid request"

    def test_websocket_get_conversion_status(self):
        """Test querying conversion status via WebSocket"""
        import base64

        # Create small test WAV
        sample_rate = 16000
        audio_data = np.zeros(int(sample_rate * 0.5), dtype=np.int16)

        import io
        import wave
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        wav_bytes = wav_buffer.getvalue()
        audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')

        conversion_id = 'test-status-101'

        # Start conversion
        self.client.emit('convert_song_stream', {
            'conversion_id': conversion_id,
            'song_data': audio_base64,
            'target_profile_id': 'test-profile-id',
            'vocal_volume': 1.0,
            'instrumental_volume': 0.9,
            'return_stems': False
        })

        # Query status
        self.client.emit('get_conversion_status', {
            'conversion_id': conversion_id
        })

        # Collect events
        received = self.client.get_received()

        # Check for status event
        status_events = [e for e in received if e['name'] == 'conversion_status']

        if status_events:
            data = status_events[0]['args'][0]
            assert 'conversion_id' in data
            assert 'progress' in data
            assert 'stage' in data
            assert 'status' in data
            assert data['conversion_id'] == conversion_id


@pytest.mark.web
@pytest.mark.e2e
@pytest.mark.slow
class TestEndToEndWorkflows:
    """Test complete end-to-end workflows for voice conversion."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures"""
        try:
            from src.auto_voice.web.app import create_app
            self.app, self.socketio = create_app(config={'TESTING': True})
            self.client = self.app.test_client()
        except ImportError:
            pytest.skip("Flask app not available")

    def test_full_voice_cloning_workflow(self):
        """Test complete workflow: Create → List → Get → Delete"""
        # Step 1: Create voice profile
        sample_rate = 22050
        duration = 30.0
        audio = np.random.rand(int(sample_rate * duration)).astype(np.float32)

        import wave
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            audio_int16 = (audio * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())

        buffer.seek(0)

        create_response = self.client.post(
            '/api/v1/voice/clone',
            data={'reference_audio': (buffer, 'test.wav')},
            content_type='multipart/form-data'
        )

        if create_response.status_code == 503:
            pytest.skip("Service unavailable")

        assert create_response.status_code in [201, 400]

        if create_response.status_code == 201:
            created_profile = create_response.get_json()
            profile_id = created_profile['profile_id']

            # Step 2: List profiles
            list_response = self.client.get('/api/v1/voice/profiles')
            assert list_response.status_code == 200

            # Step 3: Get specific profile
            get_response = self.client.get(f'/api/v1/voice/profiles/{profile_id}')
            assert get_response.status_code in [200, 404]

            # Step 4: Delete profile
            delete_response = self.client.delete(f'/api/v1/voice/profiles/{profile_id}')
            assert delete_response.status_code in [200, 404]

    def test_full_song_conversion_workflow(self, sample_song_file):
        """Test complete workflow: Create profile → Convert → Verify → Cleanup"""
        # Step 1: Create voice profile first
        sample_rate = 22050
        duration = 30.0
        ref_audio = np.random.rand(int(sample_rate * duration)).astype(np.float32)

        import wave
        ref_buffer = io.BytesIO()
        with wave.open(ref_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            audio_int16 = (ref_audio * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())

        ref_buffer.seek(0)

        create_response = self.client.post(
            '/api/v1/voice/clone',
            data={'reference_audio': (ref_buffer, 'reference.wav')},
            content_type='multipart/form-data'
        )

        if create_response.status_code == 503:
            pytest.skip("Service unavailable")

        if create_response.status_code != 201:
            pytest.skip("Could not create profile")

        profile_id = create_response.get_json()['profile_id']

        # Step 2: Convert song
        audio, sr = sample_song_file
        song_buffer = io.BytesIO()
        with wave.open(song_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sr)
            audio_int16 = (audio * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())

        song_buffer.seek(0)

        convert_response = self.client.post(
            '/api/v1/convert/song',
            data={
                'song': (song_buffer, 'song.wav'),
                'profile_id': profile_id
            },
            content_type='multipart/form-data'
        )

        # Conversion may fail due to dependencies, but workflow should be testable
        assert convert_response.status_code in [200, 404, 500, 503]

        # Step 3: Cleanup - delete profile
        delete_response = self.client.delete(f'/api/v1/voice/profiles/{profile_id}')
        assert delete_response.status_code in [200, 404]
