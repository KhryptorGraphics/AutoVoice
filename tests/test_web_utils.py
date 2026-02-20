"""Tests for web utilities."""
import pytest
from flask import Flask

from auto_voice.web.utils import (
    allowed_file,
    ALLOWED_AUDIO_EXTENSIONS,
    validation_error_response,
    not_found_response,
    service_unavailable_response,
    error_response,
)


class TestAllowedFile:
    """allowed_file() tests."""

    @pytest.mark.smoke
    def test_wav_allowed(self):
        assert allowed_file('song.wav') is True

    def test_mp3_allowed(self):
        assert allowed_file('song.mp3') is True

    def test_flac_allowed(self):
        assert allowed_file('song.flac') is True

    def test_ogg_allowed(self):
        assert allowed_file('song.ogg') is True

    def test_txt_not_allowed(self):
        assert allowed_file('notes.txt') is False

    def test_py_not_allowed(self):
        assert allowed_file('script.py') is False

    def test_no_extension(self):
        assert allowed_file('noext') is False

    def test_empty_string(self):
        assert allowed_file('') is False

    def test_case_insensitive(self):
        assert allowed_file('song.WAV') is True
        assert allowed_file('song.Mp3') is True

    def test_multiple_dots(self):
        assert allowed_file('my.song.wav') is True


class TestAllowedExtensions:
    """ALLOWED_AUDIO_EXTENSIONS set tests."""

    @pytest.mark.smoke
    def test_is_set(self):
        assert isinstance(ALLOWED_AUDIO_EXTENSIONS, set)

    def test_common_formats_present(self):
        for ext in ['wav', 'mp3', 'flac', 'ogg']:
            assert ext in ALLOWED_AUDIO_EXTENSIONS

    def test_no_dangerous_extensions(self):
        for ext in ['exe', 'sh', 'py', 'js', 'html']:
            assert ext not in ALLOWED_AUDIO_EXTENSIONS


class TestValidationErrorResponse:
    """validation_error_response() tests."""

    @pytest.mark.smoke
    def test_returns_400_status(self, flask_app):
        with flask_app.app_context():
            response, status = validation_error_response("test error")
            assert status == 400

    @pytest.mark.smoke
    def test_returns_json_with_error_field(self, flask_app):
        with flask_app.app_context():
            response, status = validation_error_response("test error")
            data = response.get_json()
            assert 'error' in data
            assert data['error'] == "test error"

    def test_handles_empty_string(self, flask_app):
        with flask_app.app_context():
            response, status = validation_error_response("")
            data = response.get_json()
            assert data['error'] == ""
            assert status == 400

    def test_handles_long_error_message(self, flask_app):
        with flask_app.app_context():
            long_msg = "a" * 1000
            response, status = validation_error_response(long_msg)
            data = response.get_json()
            assert data['error'] == long_msg
            assert status == 400

    def test_handles_special_characters(self, flask_app):
        with flask_app.app_context():
            msg = "Error: user@example.com - field 'name' is required!"
            response, status = validation_error_response(msg)
            data = response.get_json()
            assert data['error'] == msg


class TestNotFoundResponse:
    """not_found_response() tests."""

    @pytest.mark.smoke
    def test_returns_404_status(self, flask_app):
        with flask_app.app_context():
            response, status = not_found_response("Resource not found")
            assert status == 404

    @pytest.mark.smoke
    def test_returns_json_with_error_field(self, flask_app):
        with flask_app.app_context():
            response, status = not_found_response("Profile not found")
            data = response.get_json()
            assert 'error' in data
            assert data['error'] == "Profile not found"

    def test_profile_not_found_message(self, flask_app):
        with flask_app.app_context():
            response, status = not_found_response("Profile not found")
            data = response.get_json()
            assert data['error'] == "Profile not found"
            assert status == 404

    def test_job_not_found_message(self, flask_app):
        with flask_app.app_context():
            response, status = not_found_response("Conversion job not found")
            data = response.get_json()
            assert data['error'] == "Conversion job not found"
            assert status == 404

    def test_handles_empty_string(self, flask_app):
        with flask_app.app_context():
            response, status = not_found_response("")
            data = response.get_json()
            assert data['error'] == ""
            assert status == 404


class TestServiceUnavailableResponse:
    """service_unavailable_response() tests."""

    @pytest.mark.smoke
    def test_returns_503_status(self, flask_app):
        with flask_app.app_context():
            response, status = service_unavailable_response("Service unavailable")
            assert status == 503

    @pytest.mark.smoke
    def test_returns_json_with_error_field(self, flask_app):
        with flask_app.app_context():
            response, status = service_unavailable_response("Pipeline unavailable")
            data = response.get_json()
            assert 'error' in data
            assert data['error'] == "Pipeline unavailable"

    def test_without_message_in_production(self):
        app = Flask(__name__)
        app.config['DEBUG'] = False
        with app.app_context():
            response, status = service_unavailable_response(
                "Service down",
                message="Internal details"
            )
            data = response.get_json()
            assert data['error'] == "Service down"
            assert 'message' not in data
            assert status == 503

    def test_with_message_in_debug_mode(self):
        app = Flask(__name__)
        app.config['DEBUG'] = True
        with app.app_context():
            response, status = service_unavailable_response(
                "Service down",
                message="Internal details"
            )
            data = response.get_json()
            assert data['error'] == "Service down"
            assert data['message'] == "Internal details"
            assert status == 503

    def test_no_message_parameter(self):
        app = Flask(__name__)
        with app.app_context():
            response, status = service_unavailable_response("Service down")
            data = response.get_json()
            assert data['error'] == "Service down"
            assert 'message' not in data
            assert status == 503

    def test_none_message_parameter(self):
        app = Flask(__name__)
        with app.app_context():
            response, status = service_unavailable_response(
                "Service down",
                message=None
            )
            data = response.get_json()
            assert data['error'] == "Service down"
            assert 'message' not in data
            assert status == 503


class TestErrorResponse:
    """error_response() tests."""

    @pytest.mark.smoke
    def test_default_500_status(self, flask_app):
        with flask_app.app_context():
            response, status = error_response("Internal error")
            assert status == 500

    @pytest.mark.smoke
    def test_returns_json_with_error_field(self, flask_app):
        with flask_app.app_context():
            response, status = error_response("Something went wrong")
            data = response.get_json()
            assert 'error' in data
            assert data['error'] == "Something went wrong"

    def test_custom_status_code(self, flask_app):
        with flask_app.app_context():
            response, status = error_response("Custom error", status_code=422)
            assert status == 422
            data = response.get_json()
            assert data['error'] == "Custom error"

    def test_with_additional_fields(self, flask_app):
        with flask_app.app_context():
            response, status = error_response(
                "Validation failed",
                status_code=422,
                field="email",
                reason="Invalid format"
            )
            data = response.get_json()
            assert data['error'] == "Validation failed"
            assert data['field'] == "email"
            assert data['reason'] == "Invalid format"
            assert status == 422

    def test_with_multiple_kwargs(self, flask_app):
        with flask_app.app_context():
            response, status = error_response(
                "Complex error",
                detail="More info",
                code="ERR_001",
                timestamp="2026-02-20T00:00:00Z"
            )
            data = response.get_json()
            assert data['error'] == "Complex error"
            assert data['detail'] == "More info"
            assert data['code'] == "ERR_001"
            assert data['timestamp'] == "2026-02-20T00:00:00Z"
            assert status == 500

    def test_no_kwargs(self, flask_app):
        with flask_app.app_context():
            response, status = error_response("Simple error")
            data = response.get_json()
            assert data == {'error': 'Simple error'}
            assert status == 500

    def test_various_status_codes(self, flask_app):
        with flask_app.app_context():
            for code in [400, 401, 403, 404, 422, 500, 503]:
                response, status = error_response("Error", status_code=code)
                assert status == code
