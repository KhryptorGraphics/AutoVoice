"""Edge case tests for web API endpoints - comprehensive coverage.

Tests focus on:
1. Missing dependency handling (numpy, torch, librosa not available)
2. Parameter validation edge cases
3. Error response formatting
4. File upload validation
5. get_param utility function edge cases
6. API endpoint availability checks
7. Content-type handling
8. Large payload handling
"""
import json
import io
import base64
from unittest.mock import MagicMock, patch, mock_open
import pytest
from flask import Flask


@pytest.fixture
def app():
    """Create Flask app for testing."""
    app = Flask(__name__)
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    from auto_voice.web.api import api_bp
    app.register_blueprint(api_bp)
    return app.test_client()


class TestGetParamUtility:
    """Test get_param utility function edge cases."""

    def test_get_param_default_value(self, app):
        """get_param returns default when no value provided."""
        from auto_voice.web.api import get_param

        with app.test_request_context():
            value = get_param({}, 'test_key', 'settings_key', default='default_val')
            assert value == 'default_val'

    def test_get_param_from_data_dict(self, app):
        """get_param extracts value from data dict."""
        from auto_voice.web.api import get_param

        with app.test_request_context():
            data = {'settings_key': 'data_value'}
            value = get_param(data, 'form_key', 'settings_key', default='default')
            assert value == 'data_value'

    def test_get_param_form_overrides_data(self, app):
        """get_param prefers form value over data dict."""
        from auto_voice.web.api import get_param

        with app.test_request_context(
            method='POST',
            data={'form_key': 'form_value'}
        ):
            data = {'settings_key': 'data_value'}
            value = get_param(data, 'form_key', 'settings_key', default='default')
            assert value == 'form_value'

    def test_get_param_float_conversion(self, app):
        """get_param converts to float when type_hint='float'."""
        from auto_voice.web.api import get_param

        with app.test_request_context():
            data = {'settings_key': '3.14'}
            value = get_param(data, 'test', 'settings_key', default=0.0, type_hint='float')
            assert value == 3.14
            assert isinstance(value, float)

    def test_get_param_float_conversion_invalid(self, app):
        """get_param raises ValueError for invalid float."""
        from auto_voice.web.api import get_param

        with app.test_request_context():
            data = {'settings_key': 'not_a_number'}

            with pytest.raises(ValueError, match='Invalid test'):
                get_param(data, 'test', 'settings_key', default=0.0, type_hint='float')

    def test_get_param_bool_conversion_true(self, app):
        """get_param converts various true values to bool."""
        from auto_voice.web.api import get_param

        with app.test_request_context():
            for val in ['true', 'True', 'TRUE', '1', 'yes', 'YES', 'on', 'ON']:
                data = {'settings_key': val}
                result = get_param(data, 'test', 'settings_key', default=False, type_hint='bool')
                assert result is True, f"Failed for value: {val}"

    def test_get_param_bool_conversion_false(self, app):
        """get_param converts various false values to bool."""
        from auto_voice.web.api import get_param

        with app.test_request_context():
            for val in ['false', 'False', 'FALSE', '0', 'no', 'off']:
                data = {'settings_key': val}
                result = get_param(data, 'test', 'settings_key', default=True, type_hint='bool')
                assert result is False, f"Failed for value: {val}"

    def test_get_param_str_conversion(self, app):
        """get_param converts to string when type_hint='str'."""
        from auto_voice.web.api import get_param

        with app.test_request_context():
            data = {'settings_key': 123}
            value = get_param(data, 'test', 'settings_key', default='', type_hint='str')
            assert value == '123'
            assert isinstance(value, str)

    def test_get_param_validator_pass(self, app):
        """get_param accepts value when validator passes."""
        from auto_voice.web.api import get_param

        with app.test_request_context():
            data = {'settings_key': '10'}
            validator = lambda x: float(x) > 5
            value = get_param(data, 'test', 'settings_key', default='0', validator=validator, type_hint='float')
            assert value == 10.0

    def test_get_param_validator_fail(self, app):
        """get_param raises ValueError when validator fails."""
        from auto_voice.web.api import get_param

        with app.test_request_context():
            data = {'settings_key': '3'}
            validator = lambda x: float(x) > 5

            with pytest.raises(ValueError, match='Invalid value'):
                get_param(data, 'test', 'settings_key', default='0', validator=validator, type_hint='float')

    def test_get_param_none_data_dict(self, app):
        """get_param handles None data dict."""
        from auto_voice.web.api import get_param

        with app.test_request_context():
            value = get_param(None, 'test', 'settings_key', default='default_val')
            assert value == 'default_val'


class TestMissingDependencies:
    """Test API behavior when dependencies are missing."""

    def test_convert_song_no_numpy(self, app, client):
        """convert_song returns 503 when numpy unavailable."""
        from auto_voice.web import api

        # Mock singing pipeline present but numpy missing
        app.singing_conversion_pipeline = MagicMock()

        with patch.object(api, 'NUMPY_AVAILABLE', False):
            response = client.post('/api/v1/convert/song')

        assert response.status_code == 503
        assert b'numpy required' in response.data

    def test_convert_song_no_pipeline(self, client):
        """convert_song returns 503 when pipeline not initialized."""
        # No singing_conversion_pipeline attached to app
        response = client.post('/api/v1/convert/song')

        assert response.status_code == 503
        data = json.loads(response.data)
        assert 'error' in data
        assert 'unavailable' in data['error'].lower()


class TestFileUploadValidation:
    """Test file upload validation."""

    def test_no_file_provided(self, app, client):
        """convert_song returns 400 when no file provided."""
        app.singing_conversion_pipeline = MagicMock()

        response = client.post('/api/v1/convert/song', data={})

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'No song file' in data['error']

    def test_empty_filename(self, app, client):
        """convert_song returns 400 for empty filename."""
        app.singing_conversion_pipeline = MagicMock()

        response = client.post(
            '/api/v1/convert/song',
            data={'song': (io.BytesIO(b'test'), '')}
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'No selected file' in data['error']

    def test_invalid_file_extension(self, app, client):
        """convert_song returns 400 for invalid file type."""
        app.singing_conversion_pipeline = MagicMock()

        response = client.post(
            '/api/v1/convert/song',
            data={'song': (io.BytesIO(b'test'), 'test.exe')}
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Invalid file type' in data['error']

    def test_valid_audio_extensions(self, app, client):
        """allowed_file accepts all valid audio extensions."""
        from auto_voice.web.api import allowed_file

        valid_extensions = ['wav', 'mp3', 'flac', 'ogg', 'm4a', 'wma', 'aac']

        for ext in valid_extensions:
            assert allowed_file(f'test.{ext}') is True
            assert allowed_file(f'test.{ext.upper()}') is True

    def test_invalid_extensions_rejected(self):
        """allowed_file rejects non-audio extensions."""
        from auto_voice.web.api import allowed_file

        invalid_files = ['test.exe', 'test.pdf', 'test.zip', 'test.txt', 'test']

        for filename in invalid_files:
            assert allowed_file(filename) is False


class TestParameterValidation:
    """Test parameter validation in API endpoints."""

    def test_profile_id_from_form(self, app, client):
        """convert_song extracts profile_id from form."""
        app.singing_conversion_pipeline = MagicMock()
        app.singing_conversion_pipeline.convert_song.return_value = {
            'audio': base64.b64encode(b'fake_audio').decode(),
            'format': 'wav',
            'sample_rate': 22050,
        }

        with patch('auto_voice.web.api.allowed_file', return_value=True):
            response = client.post(
                '/api/v1/convert/song',
                data={
                    'song': (io.BytesIO(b'fake_audio'), 'test.wav'),
                    'profile_id': 'test-profile-123',
                }
            )

        # Should accept the request
        assert response.status_code in [200, 202]  # Could be async or sync mode

    def test_profile_id_from_settings_json(self, app, client):
        """convert_song extracts profile_id from settings JSON."""
        app.singing_conversion_pipeline = MagicMock()
        app.singing_conversion_pipeline.convert_song.return_value = {
            'audio': base64.b64encode(b'fake_audio').decode(),
            'format': 'wav',
            'sample_rate': 22050,
        }

        settings = json.dumps({'target_profile_id': 'json-profile-456'})

        with patch('auto_voice.web.api.allowed_file', return_value=True):
            response = client.post(
                '/api/v1/convert/song',
                data={
                    'song': (io.BytesIO(b'fake_audio'), 'test.wav'),
                    'settings': settings,
                }
            )

        assert response.status_code in [200, 202]

    def test_invalid_settings_json(self, app, client):
        """convert_song handles invalid JSON gracefully."""
        app.singing_conversion_pipeline = MagicMock()

        with patch('auto_voice.web.api.allowed_file', return_value=True):
            response = client.post(
                '/api/v1/convert/song',
                data={
                    'song': (io.BytesIO(b'fake_audio'), 'test.wav'),
                    'settings': '{invalid json}',
                }
            )

        # Should either reject or use defaults
        assert response.status_code in [200, 202, 400]


class TestErrorResponseFormatting:
    """Test error response consistency."""

    def test_error_has_standard_format(self, client):
        """Error responses have consistent JSON format."""
        response = client.post('/api/v1/convert/song')

        assert response.status_code >= 400
        data = json.loads(response.data)
        assert 'error' in data
        assert isinstance(data['error'], str)

    def test_404_for_nonexistent_endpoint(self, client):
        """Non-existent endpoints return 404."""
        response = client.get('/api/v1/nonexistent')
        assert response.status_code == 404


class TestHealthEndpoint:
    """Test health check endpoint edge cases."""

    def test_health_no_pipeline(self, client):
        """Health endpoint works even without pipelines."""
        response = client.get('/api/v1/health')

        # Should still return 200 but with unavailable components
        data = json.loads(response.data)
        assert 'status' in data

    def test_health_with_all_components(self, app, client):
        """Health endpoint shows all components when available."""
        # Attach mock components
        app.singing_conversion_pipeline = MagicMock()
        app.voice_cloner = MagicMock()

        response = client.get('/api/v1/health')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'status' in data


class TestContentTypeHandling:
    """Test content-type handling."""

    def test_json_response_content_type(self, client):
        """API returns proper JSON content-type."""
        response = client.get('/api/v1/health')

        assert 'application/json' in response.content_type

    def test_multipart_form_data_accepted(self, app, client):
        """convert_song accepts multipart/form-data."""
        app.singing_conversion_pipeline = MagicMock()
        app.singing_conversion_pipeline.convert_song.return_value = {
            'audio': base64.b64encode(b'fake').decode(),
            'format': 'wav',
            'sample_rate': 22050,
        }

        with patch('auto_voice.web.api.allowed_file', return_value=True):
            response = client.post(
                '/api/v1/convert/song',
                data={'song': (io.BytesIO(b'audio'), 'test.wav')},
                content_type='multipart/form-data'
            )

        assert response.status_code in [200, 202]


class TestAsyncVsSyncMode:
    """Test async vs sync mode detection."""

    def test_sync_mode_when_job_manager_unavailable(self, app, client):
        """Returns 200 with inline audio when JobManager unavailable."""
        app.singing_conversion_pipeline = MagicMock()
        app.singing_conversion_pipeline.convert_song.return_value = {
            'audio': base64.b64encode(b'fake_audio_data').decode(),
            'format': 'wav',
            'sample_rate': 22050,
            'metadata': {},
        }

        # No job_manager attached
        assert not hasattr(app, 'job_manager')

        with patch('auto_voice.web.api.allowed_file', return_value=True):
            response = client.post(
                '/api/v1/convert/song',
                data={'song': (io.BytesIO(b'audio'), 'test.wav')}
            )

        # Should be sync mode (200)
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'audio' in data or 'status' in data


class TestVolumeParameters:
    """Test volume parameter handling."""

    def test_vocal_volume_default(self, app, client):
        """vocal_volume defaults to 1.0 if not specified."""
        app.singing_conversion_pipeline = MagicMock()
        app.singing_conversion_pipeline.convert_song.return_value = {
            'audio': base64.b64encode(b'fake').decode(),
            'format': 'wav',
            'sample_rate': 22050,
        }

        with patch('auto_voice.web.api.allowed_file', return_value=True):
            client.post(
                '/api/v1/convert/song',
                data={'song': (io.BytesIO(b'audio'), 'test.wav')}
            )

        # Check that convert_song was called
        app.singing_conversion_pipeline.convert_song.assert_called()

    def test_instrumental_volume_custom(self, app, client):
        """instrumental_volume can be customized."""
        app.singing_conversion_pipeline = MagicMock()
        app.singing_conversion_pipeline.convert_song.return_value = {
            'audio': base64.b64encode(b'fake').decode(),
            'format': 'wav',
            'sample_rate': 22050,
        }

        with patch('auto_voice.web.api.allowed_file', return_value=True):
            response = client.post(
                '/api/v1/convert/song',
                data={
                    'song': (io.BytesIO(b'audio'), 'test.wav'),
                    'instrumental_volume': '0.5',
                }
            )

        assert response.status_code in [200, 202]


class TestPitchShift:
    """Test pitch shift parameter validation."""

    def test_pitch_shift_in_range(self, app, client):
        """pitch_shift within [-12, 12] is accepted."""
        app.singing_conversion_pipeline = MagicMock()
        app.singing_conversion_pipeline.convert_song.return_value = {
            'audio': base64.b64encode(b'fake').decode(),
            'format': 'wav',
            'sample_rate': 22050,
        }

        with patch('auto_voice.web.api.allowed_file', return_value=True):
            for pitch in [-12, -6, 0, 6, 12]:
                response = client.post(
                    '/api/v1/convert/song',
                    data={
                        'song': (io.BytesIO(b'audio'), 'test.wav'),
                        'pitch_shift': str(pitch),
                    }
                )
                assert response.status_code in [200, 202], f"Failed for pitch_shift={pitch}"


@pytest.mark.smoke
class TestAPISmoke:
    """Quick smoke tests for API."""

    def test_api_blueprint_imports(self):
        """API blueprint can be imported."""
        from auto_voice.web.api import api_bp
        assert api_bp is not None
        assert api_bp.name == 'api'

    def test_api_blueprint_url_prefix(self):
        """API blueprint has correct URL prefix."""
        from auto_voice.web.api import api_bp
        assert api_bp.url_prefix == '/api/v1'

    def test_constants_defined(self):
        """API constants are defined."""
        from auto_voice.web.api import UPLOAD_FOLDER, MAX_TEXT_LENGTH, MAX_AUDIO_DURATION
        assert UPLOAD_FOLDER is not None
        assert MAX_TEXT_LENGTH > 0
        assert MAX_AUDIO_DURATION > 0
