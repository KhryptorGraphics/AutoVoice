"""Tests for Flask REST API endpoints."""
import json
import io
import pytest
import numpy as np


class TestHealthEndpoint:
    """Health check endpoint tests."""

    @pytest.mark.smoke
    def test_health_returns_200(self, client):
        resp = client.get('/api/v1/health')
        assert resp.status_code == 200

    @pytest.mark.smoke
    def test_health_has_components(self, client):
        resp = client.get('/api/v1/health')
        data = resp.get_json()
        assert 'components' in data
        assert 'status' in data
        assert data['components']['api']['status'] == 'up'

    def test_health_shows_torch_status(self, client):
        data = client.get('/api/v1/health').get_json()
        assert 'torch' in data['components']
        assert data['components']['torch']['status'] == 'up'

    def test_health_with_full_app(self, client_full):
        resp = client_full.get('/api/v1/health')
        data = resp.get_json()
        assert resp.status_code == 200
        assert data['status'] == 'healthy'
        assert data['components']['singing_pipeline']['status'] == 'up'
        assert data['components']['voice_cloner']['status'] == 'up'


class TestSystemInfo:
    """System info endpoint tests."""

    @pytest.mark.smoke
    def test_system_info_returns_200(self, client):
        resp = client.get('/api/v1/system/info')
        assert resp.status_code == 200

    def test_system_info_has_python_version(self, client):
        data = client.get('/api/v1/system/info').get_json()
        assert 'system' in data
        assert 'python_version' in data['system']

    def test_system_info_shows_dependencies(self, client):
        data = client.get('/api/v1/system/info').get_json()
        assert 'dependencies' in data
        assert data['dependencies']['torch'] is True
        assert data['dependencies']['numpy'] is True


class TestGPUMetrics:
    """GPU metrics endpoint tests."""

    @pytest.mark.cuda
    def test_gpu_metrics_returns_200(self, client):
        resp = client.get('/api/v1/gpu/metrics')
        assert resp.status_code == 200

    @pytest.mark.cuda
    def test_gpu_metrics_shows_device(self, client):
        data = client.get('/api/v1/gpu/metrics').get_json()
        assert data['available'] is True
        assert data['device_count'] >= 1


class TestKernelMetrics:
    """CUDA kernel metrics endpoint tests."""

    def test_kernel_metrics_returns_200(self, client):
        resp = client.get('/api/v1/kernels/metrics')
        assert resp.status_code == 200


class TestVoiceCloneEndpoint:
    """Voice clone API tests."""

    def test_clone_no_file_returns_400(self, client_full):
        resp = client_full.post('/api/v1/voice/clone')
        assert resp.status_code == 400

    def test_clone_empty_filename_returns_400(self, client_full):
        data = {'reference_audio': (io.BytesIO(b''), '')}
        resp = client_full.post('/api/v1/voice/clone', data=data,
                                content_type='multipart/form-data')
        assert resp.status_code == 400

    def test_clone_invalid_extension_returns_400(self, client_full):
        data = {'reference_audio': (io.BytesIO(b'data'), 'test.txt')}
        resp = client_full.post('/api/v1/voice/clone', data=data,
                                content_type='multipart/form-data')
        assert resp.status_code == 400

    def test_clone_service_unavailable_when_disabled(self, client):
        data = {'reference_audio': (io.BytesIO(b'data'), 'test.wav')}
        resp = client.post('/api/v1/voice/clone', data=data,
                           content_type='multipart/form-data')
        assert resp.status_code == 503


class TestVoiceProfilesEndpoint:
    """Voice profiles listing tests."""

    def test_profiles_service_unavailable_when_disabled(self, client):
        resp = client.get('/api/v1/voice/profiles')
        assert resp.status_code == 503

    def test_profiles_returns_list(self, client_full):
        resp = client_full.get('/api/v1/voice/profiles')
        assert resp.status_code == 200
        assert isinstance(resp.get_json(), list)


class TestConvertSongEndpoint:
    """Song conversion endpoint tests."""

    def test_convert_no_file_returns_400(self, client_full):
        resp = client_full.post('/api/v1/convert/song')
        assert resp.status_code == 400

    def test_convert_no_profile_returns_400(self, client_full):
        data = {'song': (io.BytesIO(b'audio'), 'test.wav')}
        resp = client_full.post('/api/v1/convert/song', data=data,
                                content_type='multipart/form-data')
        assert resp.status_code == 400

    def test_convert_invalid_profile_returns_404(self, client_full):
        data = {
            'song': (io.BytesIO(b'audio'), 'test.wav'),
            'profile_id': 'nonexistent-profile-id'
        }
        resp = client_full.post('/api/v1/convert/song', data=data,
                                content_type='multipart/form-data')
        assert resp.status_code == 404

    def test_convert_pipeline_unavailable_returns_503(self, client):
        data = {
            'song': (io.BytesIO(b'audio'), 'test.wav'),
            'profile_id': 'test'
        }
        resp = client.post('/api/v1/convert/song', data=data,
                           content_type='multipart/form-data')
        assert resp.status_code == 503


class TestConvertStatusEndpoint:
    """Conversion status endpoint tests."""

    def test_status_unknown_job_returns_404(self, client_full):
        resp = client_full.get('/api/v1/convert/status/nonexistent-job')
        assert resp.status_code == 404

    def test_status_service_unavailable_when_disabled(self, client):
        resp = client.get('/api/v1/convert/status/any-id')
        assert resp.status_code == 503


class TestConvertDownloadEndpoint:
    """Download endpoint tests."""

    def test_download_unknown_job_returns_404(self, client_full):
        resp = client_full.get('/api/v1/convert/download/nonexistent-job')
        assert resp.status_code == 404


class TestConvertCancelEndpoint:
    """Cancel endpoint tests."""

    def test_cancel_unknown_job_returns_404(self, client_full):
        resp = client_full.post('/api/v1/convert/cancel/nonexistent-job')
        assert resp.status_code == 404


class TestConvertMetricsEndpoint:
    """Conversion metrics endpoint tests."""

    def test_metrics_unknown_job_returns_404(self, client_full):
        resp = client_full.get('/api/v1/convert/metrics/nonexistent-job')
        assert resp.status_code == 404
