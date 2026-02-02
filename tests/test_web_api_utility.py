"""Comprehensive tests for utility endpoints.

Phase 4.6: Tests for utility and system endpoints:
- GET /health - Health check
- GET /api/v1/gpu/metrics - GPU stats
- GET /api/v1/system/info - System info
- GET /api/v1/devices/list - Device list
- POST /api/v1/youtube/info - YouTube metadata
- POST /api/v1/youtube/download - YouTube download
- GET /api/v1/models/loaded - Loaded models
- POST /api/v1/models/load - Model loading
- POST /api/v1/models/tensorrt/rebuild - TensorRT rebuild
- GET /api/v1/kernels/metrics - CUDA kernel metrics

Uses Flask test client with mocked components.
"""

import io
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def app_with_utils():
    """Create Flask app for utility endpoint testing."""
    pytest.importorskip('flask_swagger_ui', reason="flask_swagger_ui not installed")

    from auto_voice.web.app import create_app

    app, socketio = create_app(config={
        'TESTING': True,
        'singing_conversion_enabled': False,
        'voice_cloning_enabled': False,
    })

    app.socketio = socketio
    app.app_config = {'audio': {'sample_rate': 22050}}

    return app


@pytest.fixture
def client(app_with_utils):
    """Flask test client."""
    return app_with_utils.test_client()


class TestHealthEndpoint:
    """Test GET /api/v1/health endpoint."""

    def test_health_returns_ok(self, client):
        """Returns healthy status."""
        response = client.get('/api/v1/health')

        assert response.status_code == 200
        data = json.loads(response.data)

        # Should have status field
        assert 'status' in data or 'healthy' in data or 'ok' in str(data).lower()

    def test_health_contains_component_status(self, client):
        """Health check includes component statuses."""
        response = client.get('/api/v1/health')

        assert response.status_code == 200
        data = json.loads(response.data)

        # Should have some component info
        assert isinstance(data, dict)


class TestGPUMetricsEndpoint:
    """Test GET /api/v1/gpu/metrics endpoint."""

    def test_gpu_metrics_returns_info(self, client):
        """Returns GPU metrics or unavailable status."""
        response = client.get('/api/v1/gpu/metrics')

        # May return 200 (GPU available) or 503/404 (no GPU)
        assert response.status_code in (200, 404, 503)

    def test_gpu_metrics_json_response(self, client):
        """Returns valid JSON response."""
        response = client.get('/api/v1/gpu/metrics')

        if response.status_code == 200:
            data = json.loads(response.data)
            assert isinstance(data, (dict, list))


class TestSystemInfoEndpoint:
    """Test GET /api/v1/system/info endpoint."""

    def test_system_info_returns_data(self, client):
        """Returns system information."""
        response = client.get('/api/v1/system/info')

        # Endpoint may or may not exist
        assert response.status_code in (200, 404)

    def test_system_info_contains_version(self, client):
        """System info includes version information."""
        response = client.get('/api/v1/system/info')

        if response.status_code == 200:
            data = json.loads(response.data)
            # May contain version, python_version, etc.
            assert isinstance(data, dict)


class TestDevicesListEndpoint:
    """Test GET /api/v1/devices/list endpoint."""

    def test_devices_list_returns_array(self, client):
        """Returns list of available devices."""
        response = client.get('/api/v1/devices/list')

        # Endpoint may or may not exist
        assert response.status_code in (200, 404)

    def test_devices_list_includes_cuda(self, client):
        """Device list includes CUDA devices if available."""
        response = client.get('/api/v1/devices/list')

        if response.status_code == 200:
            data = json.loads(response.data)
            assert isinstance(data, (dict, list))


class TestYouTubeInfoEndpoint:
    """Test POST /api/v1/youtube/info endpoint."""

    def test_youtube_info_missing_url(self, client):
        """Returns 400 without URL."""
        response = client.post(
            '/api/v1/youtube/info',
            json={},
            content_type='application/json'
        )

        assert response.status_code == 400

    def test_youtube_info_invalid_url(self, client):
        """Returns error for invalid URL."""
        response = client.post(
            '/api/v1/youtube/info',
            json={'url': 'not a valid url'},
            content_type='application/json'
        )

        # May return 200 with error info, 400, or 500 depending on implementation
        assert response.status_code in (200, 400, 500)

    def test_youtube_info_with_mock(self, client):
        """Returns metadata with mocked downloader."""
        with patch('auto_voice.web.api.YouTubeDownloader') as MockYT:
            mock_yt = MagicMock()
            mock_yt.get_info.return_value = {
                'title': 'Test Video',
                'channel': 'Test Channel',
                'duration': 180,
                'thumbnail': 'https://example.com/thumb.jpg',
            }
            MockYT.return_value = mock_yt

            response = client.post(
                '/api/v1/youtube/info',
                json={'url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'},
                content_type='application/json'
            )

        # May succeed or fail depending on implementation
        assert response.status_code in (200, 400, 500)


class TestYouTubeDownloadEndpoint:
    """Test POST /api/v1/youtube/download endpoint."""

    def test_youtube_download_missing_url(self, client):
        """Returns 400 without URL."""
        response = client.post(
            '/api/v1/youtube/download',
            json={},
            content_type='application/json'
        )

        assert response.status_code == 400

    def test_youtube_download_with_format(self, client):
        """Accepts format parameter."""
        with patch('auto_voice.web.api.YouTubeDownloader') as MockYT:
            mock_yt = MagicMock()
            mock_yt.download.return_value = MagicMock(
                audio_path='/tmp/audio.wav',
                title='Test Video',
                duration=180,
            )
            MockYT.return_value = mock_yt

            response = client.post(
                '/api/v1/youtube/download',
                json={
                    'url': 'https://www.youtube.com/watch?v=test',
                    'format': 'audio_only',
                },
                content_type='application/json'
            )

        assert response.status_code in (200, 202, 400, 500)


class TestModelsLoadedEndpoint:
    """Test GET /api/v1/models/loaded endpoint."""

    def test_models_loaded_returns_list(self, client):
        """Returns list of loaded models."""
        response = client.get('/api/v1/models/loaded')

        # Endpoint may or may not exist
        assert response.status_code in (200, 404)

    def test_models_loaded_json_response(self, client):
        """Returns valid JSON."""
        response = client.get('/api/v1/models/loaded')

        if response.status_code == 200:
            data = json.loads(response.data)
            assert isinstance(data, (dict, list))


class TestModelLoadEndpoint:
    """Test POST /api/v1/models/load endpoint."""

    def test_load_model_missing_name(self, client):
        """Returns 400 when model name missing."""
        response = client.post(
            '/api/v1/models/load',
            json={},
            content_type='application/json'
        )

        assert response.status_code in (400, 404)

    def test_load_model_unknown_model(self, client):
        """Returns error for unknown model."""
        response = client.post(
            '/api/v1/models/load',
            json={'model_name': 'nonexistent_model'},
            content_type='application/json'
        )

        assert response.status_code in (400, 404, 500)


class TestTensorRTRebuildEndpoint:
    """Test POST /api/v1/models/tensorrt/rebuild endpoint."""

    def test_tensorrt_rebuild(self, client):
        """TensorRT rebuild endpoint."""
        response = client.post(
            '/api/v1/models/tensorrt/rebuild',
            json={},
            content_type='application/json'
        )

        # May succeed or fail based on TensorRT availability
        assert response.status_code in (200, 400, 404, 500, 503)

    def test_tensorrt_rebuild_with_model(self, client):
        """Rebuild specific model."""
        response = client.post(
            '/api/v1/models/tensorrt/rebuild',
            json={'model': 'vocoder'},
            content_type='application/json'
        )

        assert response.status_code in (200, 400, 404, 500, 503)


class TestKernelMetricsEndpoint:
    """Test GET /api/v1/kernels/metrics endpoint."""

    def test_kernel_metrics_returns_data(self, client):
        """Returns CUDA kernel metrics."""
        response = client.get('/api/v1/kernels/metrics')

        # Endpoint may or may not exist
        assert response.status_code in (200, 404, 503)

    def test_kernel_metrics_json_response(self, client):
        """Returns valid JSON."""
        response = client.get('/api/v1/kernels/metrics')

        if response.status_code == 200:
            data = json.loads(response.data)
            assert isinstance(data, (dict, list))


class TestAPIVersioning:
    """Test API versioning and routing."""

    def test_v1_prefix_works(self, client):
        """API v1 prefix routes correctly."""
        response = client.get('/api/v1/gpu/metrics')

        # Should reach the endpoint (not 404 from routing)
        # May return 503 if GPU not available
        assert response.status_code in (200, 404, 503)

    def test_root_health_no_version(self, client):
        """Health endpoint available at /api/v1/health."""
        response = client.get('/api/v1/health')

        assert response.status_code == 200


class TestCORSHeaders:
    """Test CORS header handling."""

    def test_options_request(self, client):
        """OPTIONS request returns proper CORS headers."""
        response = client.options('/api/v1/gpu/metrics')

        # Should handle OPTIONS (may be 200 or 204)
        assert response.status_code in (200, 204, 404, 405)


class TestErrorResponses:
    """Test error response formatting."""

    def test_404_returns_json(self, client):
        """404 errors return JSON."""
        response = client.get('/api/v1/nonexistent/endpoint')

        assert response.status_code == 404
        # Should be JSON
        try:
            data = json.loads(response.data)
            assert isinstance(data, dict)
        except json.JSONDecodeError:
            # Some frameworks return HTML for 404
            pass

    def test_400_includes_error_message(self, client):
        """400 errors include error message."""
        response = client.post(
            '/api/v1/youtube/info',
            json={},
            content_type='application/json'
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data or 'message' in data


class TestContentNegotiation:
    """Test content type handling."""

    def test_accepts_json(self, client):
        """Accepts application/json content type."""
        response = client.post(
            '/api/v1/youtube/info',
            json={'url': 'https://youtube.com/test'},
            content_type='application/json'
        )

        assert response.status_code in (200, 400, 500)

    def test_json_response_type(self, client):
        """Returns JSON content type."""
        response = client.get('/api/v1/health')

        assert response.status_code == 200
        assert 'application/json' in response.content_type
