"""Production-mode security helpers for the Flask web surface."""

from __future__ import annotations

import hmac
import os
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from flask import Flask, g, jsonify, request
from werkzeug.utils import secure_filename

from .utils import allowed_file


PUBLIC_PATHS = {
    "/health",
    "/ready",
    "/api/v1/health",
    "/api/v1/ready",
}

YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
    "youtu.be",
    "www.youtu.be",
}


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def public_deployment_enabled(app: Flask | None = None) -> bool:
    environment = os.environ.get("ENVIRONMENT", "").strip().lower()
    config_value = bool(app and app.config.get("AUTOVOICE_PUBLIC_DEPLOYMENT"))
    return config_value or env_bool("AUTOVOICE_PUBLIC_DEPLOYMENT") or environment in {"public", "hosted"}


def api_auth_required(app: Flask) -> bool:
    return public_deployment_enabled(app) or env_bool("AUTOVOICE_REQUIRE_API_AUTH")


def strict_path_sandbox_enabled(app: Flask) -> bool:
    return public_deployment_enabled(app) or env_bool("AUTOVOICE_STRICT_PATH_SANDBOX")


def media_consent_required(app: Flask) -> bool:
    return public_deployment_enabled(app) or env_bool("AUTOVOICE_REQUIRE_MEDIA_CONSENT")


def parse_cors_origins(raw: str | None = None) -> str | list[str]:
    value = raw if raw is not None else os.environ.get("CORS_ORIGINS", "*")
    origins = [origin.strip() for origin in str(value).split(",") if origin.strip()]
    if not origins or origins == ["*"]:
        return "*"
    return origins


def socketio_cors_allowed_origins(app: Flask) -> str | list[str]:
    origins = parse_cors_origins(app.config.get("CORS_ORIGINS"))
    if (
        origins == "*"
        and not app.config.get("TESTING")
        and public_deployment_enabled(app)
        and not env_bool("AUTOVOICE_ALLOW_INSECURE_CORS")
    ):
        raise RuntimeError(
            "CORS_ORIGINS='*' is not allowed when AUTOVOICE_PUBLIC_DEPLOYMENT is enabled. "
            "Set CORS_ORIGINS to explicit origins or AUTOVOICE_ALLOW_INSECURE_CORS=true."
        )
    return origins


def _configured_api_token(app: Flask) -> str | None:
    token = app.config.get("AUTOVOICE_API_TOKEN") or os.environ.get("AUTOVOICE_API_TOKEN")
    if token:
        return str(token)
    return None


def _request_token() -> str | None:
    authorization = request.headers.get("Authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip()
    return (
        request.headers.get("X-AutoVoice-API-Key")
        or request.headers.get("X-API-Key")
        or request.args.get("api_token")
    )


def token_authorized(app: Flask, token: str | None = None) -> bool:
    expected = _configured_api_token(app)
    if not expected:
        return not api_auth_required(app)
    supplied = token if token is not None else _request_token()
    return bool(supplied) and hmac.compare_digest(str(supplied), expected)


def _request_key() -> str:
    token = _request_token()
    if token:
        return f"token:{token[:16]}"
    return f"ip:{request.remote_addr or 'unknown'}"


def _rate_limit_response(limit: int) -> tuple[Any, int]:
    return jsonify({"error": f"rate limit exceeded: {limit} requests per minute"}), 429


def init_production_security(app: Flask) -> None:
    """Install request ID, optional API-token auth, and public-mode rate limiting."""
    app._rate_limit_buckets = defaultdict(deque)

    if api_auth_required(app) and not _configured_api_token(app) and not app.config.get("TESTING"):
        raise RuntimeError(
            "AUTOVOICE_REQUIRE_API_AUTH/AUTOVOICE_PUBLIC_DEPLOYMENT requires AUTOVOICE_API_TOKEN"
        )

    if socketio_cors_allowed_origins(app) == "*" and public_deployment_enabled(app):
        # The helper raises unless explicitly overridden; call it here so HTTP-only
        # test paths also fail fast under unsafe public-mode CORS.
        pass

    @app.before_request
    def enforce_security_controls():
        g.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        if request.method == "OPTIONS" or request.path in PUBLIC_PATHS:
            return None

        if api_auth_required(app) and not token_authorized(app):
            return jsonify({"error": "authentication required"}), 401

        rate_limit_enabled = env_bool("AUTOVOICE_ENABLE_RATE_LIMIT", api_auth_required(app))
        limit = int(os.environ.get("RATE_LIMIT", app.config.get("RATE_LIMIT", 60)) or 60)
        if rate_limit_enabled and limit > 0:
            now = time.time()
            bucket = app._rate_limit_buckets[_request_key()]
            while bucket and now - bucket[0] >= 60:
                bucket.popleft()
            if len(bucket) >= limit:
                return _rate_limit_response(limit)
            bucket.append(now)
        return None

    @app.after_request
    def attach_request_id(response):
        response.headers["X-Request-ID"] = getattr(g, "request_id", uuid.uuid4().hex)
        return response


def require_socketio_authorization(app: Flask, auth: Any | None = None) -> bool:
    if not api_auth_required(app):
        return True
    token = None
    if isinstance(auth, dict):
        token = auth.get("token") or auth.get("api_token")
    if not token:
        token = _request_token()
    return token_authorized(app, token)


def managed_audio_roots(data_dir: str | Path, upload_folder: str | Path | None = None) -> list[Path]:
    roots = [Path(data_dir).expanduser().resolve()]
    if upload_folder:
        roots.append(Path(upload_folder).expanduser().resolve())
    return roots


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_server_audio_path(
    raw_path: Any,
    *,
    data_dir: str | Path,
    upload_folder: str | Path | None = None,
    strict: bool = False,
) -> Path:
    if not raw_path:
        raise ValueError("audio_path is required")
    path = Path(str(raw_path)).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(str(raw_path))
    if not allowed_file(path.name):
        raise ValueError("audio_path must use an allowed audio extension")
    if strict and not any(_is_relative_to(path, root) for root in managed_audio_roots(data_dir, upload_folder)):
        raise PermissionError("audio_path must be under a managed AutoVoice data directory")
    return path


def safe_upload_filename(filename: str) -> str:
    cleaned = secure_filename(filename or "")
    if not cleaned:
        raise ValueError("No file selected")
    if not allowed_file(cleaned):
        raise ValueError("Invalid file type")
    return cleaned


def canonical_youtube_url(raw_url: Any) -> str:
    url = str(raw_url or "").strip()
    if not url:
        raise ValueError("URL cannot be empty")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http(s) YouTube URLs are allowed")
    host = (parsed.hostname or "").lower()
    if host not in YOUTUBE_HOSTS:
        raise ValueError("Only YouTube URLs are allowed")

    video_id = None
    if host.endswith("youtu.be"):
        video_id = parsed.path.strip("/").split("/", 1)[0]
    else:
        video_id = parse_qs(parsed.query).get("v", [None])[0]
        if not video_id and parsed.path.startswith("/shorts/"):
            video_id = parsed.path.split("/", 2)[2].split("/", 1)[0]
    if not video_id or len(video_id) > 64:
        raise ValueError("YouTube URL must include a valid video id")
    return f"https://www.youtube.com/watch?v={video_id}"


def require_media_consent(data: dict[str, Any], app: Flask) -> None:
    if not media_consent_required(app):
        return
    if not data.get("consent_confirmed") or not data.get("source_media_policy_confirmed"):
        raise PermissionError(
            "consent_confirmed and source_media_policy_confirmed are required in public mode"
        )
