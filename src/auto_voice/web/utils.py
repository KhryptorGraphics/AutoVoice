"""Shared web utilities for AutoVoice API."""
from typing import Optional, Tuple, Any
from flask import jsonify, current_app, has_app_context

ALLOWED_AUDIO_EXTENSIONS = {
    'wav', 'mp3', 'flac', 'ogg', 'opus', 'aac', 'm4a', 'wma', 'aiff', 'webm'
}


def allowed_file(filename: str) -> bool:
    """Check if filename has an allowed audio extension."""
    if not filename or '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_AUDIO_EXTENSIONS


def validation_error_response(error: str, **kwargs) -> Tuple[Any, int]:
    """Return a validation error response (400 Bad Request).

    Args:
        error: Error message describing the validation failure
        **kwargs: Optional additional JSON fields (e.g. error_code, details)

    Returns:
        Tuple of (JSON response, status code 400)

    Example:
        return validation_error_response("user_id is required")
        # Returns: ({'error': 'user_id is required'}, 400)
    """
    response = {'error': error}
    response.update({key: value for key, value in kwargs.items() if value is not None})
    return jsonify(response), 400


def not_found_response(error: str) -> Tuple[Any, int]:
    """Return a not found error response (404 Not Found).

    Args:
        error: Error message describing what was not found

    Returns:
        Tuple of (JSON response, status code 404)

    Example:
        return not_found_response("Profile not found")
        # Returns: ({'error': 'Profile not found'}, 404)
    """
    return jsonify({'error': error}), 404


def service_unavailable_response(error: str, message: Optional[str] = None) -> Tuple[Any, int]:
    """Return a service unavailable error response (503 Service Unavailable).

    Args:
        error: Error message describing the unavailable service
        message: Optional detailed message (only included in debug mode)

    Returns:
        Tuple of (JSON response, status code 503)

    Example:
        return service_unavailable_response("Voice cloning service unavailable")
        # Returns: ({'error': 'Voice cloning service unavailable'}, 503)

        # With detailed message in debug mode:
        return service_unavailable_response(
            "Conversion failed",
            message=str(exception)
        )
        # Debug mode: ({'error': 'Conversion failed', 'message': '...'}, 503)
        # Production: ({'error': 'Conversion failed'}, 503)
    """
    if _redact_server_errors():
        response = {'error': 'Service unavailable', 'error_code': 'service_unavailable'}
    else:
        response = {'error': error}
    if message and current_app.debug:
        response['message'] = message
    return jsonify(response), 503


def _redact_server_errors() -> bool:
    if not has_app_context() or current_app.config.get("TESTING"):
        return False
    try:
        from .security import response_path_redaction_enabled

        return response_path_redaction_enabled(current_app)
    except Exception:
        return False


def error_response(error: str, status_code: int = 500, **kwargs) -> Tuple[Any, int]:
    """Return a generic error response with custom status code.

    Args:
        error: Error message
        status_code: HTTP status code (default: 500)
        **kwargs: Additional fields to include in response

    Returns:
        Tuple of (JSON response, status code)

    Example:
        return error_response("Internal server error")
        # Returns: ({'error': 'Internal server error'}, 500)

        return error_response("Custom error", status_code=422, detail="validation failed")
        # Returns: ({'error': 'Custom error', 'detail': 'validation failed'}, 422)
    """
    if status_code >= 500 and _redact_server_errors():
        response = {'error': 'Internal server error', 'error_code': 'internal_error'}
        request_id = kwargs.get('request_id')
        if request_id is not None:
            response['request_id'] = request_id
    else:
        response = {'error': error}
        response.update(kwargs)
    return jsonify(response), status_code
