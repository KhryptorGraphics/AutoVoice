"""Tests for Flask REST API endpoints."""
import json
import io
import os
import wave
import pytest
import numpy as np


def _save_test_profile(client_full, profile_id: str, *, role: str, has_full_model: bool = False):
    store = client_full.application.voice_cloner.store
    store.save({
        'profile_id': profile_id,
        'name': f'{role}-{profile_id[-4:]}',
        'embedding': np.zeros(256, dtype=np.float32).tolist(),
        'profile_role': role,
        'sample_count': 1,
        'has_trained_model': has_full_model,
    })
    if has_full_model:
        full_model_path = os.path.join(store.trained_models_dir, f'{profile_id}_full_model.pt')
        with open(full_model_path, 'wb') as handle:
            handle.write(b'full-model')
    return store.load(profile_id)


def _write_wav(path: str, audio: np.ndarray, sample_rate: int = 22050) -> None:
    audio_int16 = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_int16 * 32767).astype(np.int16)
    with wave.open(path, 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(audio_int16.tobytes())


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
        assert resp.status_code == 200
        assert isinstance(resp.get_json(), list)

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

    def test_convert_rejects_source_artist_profile(self, client_full):
        profile_id = '00000000-0000-0000-0000-000000000101'
        _save_test_profile(client_full, profile_id, role='source_artist')

        data = {
            'song': (io.BytesIO(b'audio'), 'test.wav'),
            'profile_id': profile_id,
        }
        resp = client_full.post('/api/v1/convert/song', data=data,
                                content_type='multipart/form-data')
        assert resp.status_code == 400
        assert 'target user profile' in resp.get_json()['error'].lower()

    def test_convert_allows_full_model_without_adapter(self, client_full, monkeypatch):
        profile_id = '00000000-0000-0000-0000-000000000102'
        _save_test_profile(client_full, profile_id, role='target_user', has_full_model=True)

        def _fake_convert_song(**kwargs):
            assert kwargs['target_profile_id'] == profile_id
            return {
                'mixed_audio': np.zeros(22050, dtype=np.float32),
                'sample_rate': 22050,
                'duration': 1.0,
                'metadata': {
                    'target_profile_id': profile_id,
                    'active_model_type': 'full_model',
                },
                'f0_contour': np.array([], dtype=np.float32),
                'f0_original': np.array([], dtype=np.float32),
            }

        monkeypatch.setattr(client_full.application, 'job_manager', None, raising=False)
        monkeypatch.setattr(
            client_full.application.singing_conversion_pipeline,
            'convert_song',
            _fake_convert_song,
        )

        data = {
            'song': (io.BytesIO(b'audio'), 'test.wav'),
            'profile_id': profile_id,
        }
        resp = client_full.post('/api/v1/convert/song', data=data,
                                content_type='multipart/form-data')
        assert resp.status_code == 200
        payload = resp.get_json()
        assert payload['active_model_type'] == 'full_model'
        assert payload['adapter_type'] is None


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
        events = client_full.application.state_store.list_audit_events(resource_id="nonexistent-job")
        assert not any(event["metadata"]["event_type"] == "conversion.downloaded" for event in events)

    def test_download_vocal_stem_variant_returns_file(self, client_full, tmp_path, monkeypatch):
        vocals_path = tmp_path / 'vocals.wav'
        _write_wav(str(vocals_path), np.zeros(22050, dtype=np.float32))

        monkeypatch.setattr(
            client_full.application.job_manager,
            'get_job_asset_path',
            lambda job_id, asset='mix': str(vocals_path) if asset == 'vocals' else None,
        )

        resp = client_full.get('/api/v1/convert/download/job-123?variant=vocals')
        assert resp.status_code == 200
        assert resp.mimetype == 'audio/wav'
        events = client_full.application.state_store.list_audit_events(resource_id="job-123")
        assert any(event["metadata"]["event_type"] == "conversion.downloaded" for event in events)

    def test_reassemble_endpoint_returns_mixed_audio(self, client_full, tmp_path, monkeypatch):
        vocals_path = tmp_path / 'vocals.wav'
        instrumental_path = tmp_path / 'instrumental.wav'
        _write_wav(str(vocals_path), np.full(22050, 0.1, dtype=np.float32))
        _write_wav(str(instrumental_path), np.full(22050, 0.05, dtype=np.float32))

        def _get_job_asset_path(job_id, asset='mix'):
            if asset == 'vocals':
                return str(vocals_path)
            if asset == 'instrumental':
                return str(instrumental_path)
            return None

        monkeypatch.setattr(
            client_full.application.job_manager,
            'get_job_asset_path',
            _get_job_asset_path,
        )

        resp = client_full.get('/api/v1/convert/reassemble/job-123')
        assert resp.status_code == 200
        assert resp.mimetype == 'audio/wav'
        events = client_full.application.state_store.list_audit_events(resource_id="job-123")
        assert any(event["metadata"]["event_type"] == "conversion.reassembled" for event in events)


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
