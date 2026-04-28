"""Production-mode security helpers for the Flask web surface."""

from __future__ import annotations

import hmac
import hashlib
import os
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from flask import Flask, current_app, g, has_app_context, has_request_context, jsonify, request
from werkzeug.utils import secure_filename

from .utils import allowed_file


PUBLIC_PATHS = {
    "/health",
    "/ready",
    "/api/v1/health",
    "/api/v1/ready",
}

INSECURE_API_TOKENS = {
    "autovoice",
    "autovoice-token",
    "changeme",
    "change-me",
    "default",
    "dev",
    "dev-token",
    "development",
    "password",
    "secret",
    "test",
    "token",
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

_ASSET_SCALAR_FIELDS: dict[str, tuple[str, str]] = {
    "path": ("asset_id", "asset"),
    "file_path": ("file_asset_id", "file"),
    "audio_path": ("audio_asset_id", "audio"),
    "vocals_path": ("vocals_asset_id", "vocals"),
    "instrumental_path": ("instrumental_asset_id", "instrumental"),
    "model_path": ("model_asset_id", "model"),
    "full_model_path": ("full_model_asset_id", "full_model"),
    "tensorrt_engine_path": ("tensorrt_engine_asset_id", "tensorrt_engine"),
    "adapter_path": ("adapter_asset_id", "adapter"),
    "embedding_path": ("embedding_asset_id", "embedding"),
    "runtime_artifact_manifest_path": (
        "runtime_artifact_manifest_asset_id",
        "runtime_artifact_manifest",
    ),
    "primary_reference_audio_path": ("primary_reference_audio_asset_id", "reference_audio"),
    "filtered_audio_path": ("filtered_audio_asset_id", "filtered_audio"),
    "original_path": ("original_asset_id", "source_audio"),
    "audioPath": ("audioAssetId", "audio"),
    "filteredPath": ("filteredAssetId", "filtered_audio"),
}

_ASSET_LIST_FIELDS: dict[str, tuple[str, str]] = {
    "reference_audio_paths": ("reference_audio_asset_ids", "reference_audio"),
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
    return (
        public_deployment_enabled(app)
        or api_auth_required(app)
        or env_bool("AUTOVOICE_STRICT_PATH_SANDBOX")
    )


def media_consent_required(app: Flask) -> bool:
    return public_deployment_enabled(app) or env_bool("AUTOVOICE_REQUIRE_MEDIA_CONSENT")


def _strict_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return False


def response_path_redaction_enabled(app: Flask) -> bool:
    """Hide local filesystem paths from any externally authenticated API surface."""
    return public_deployment_enabled(app) or api_auth_required(app)


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
        and api_auth_required(app)
        and not env_bool("AUTOVOICE_ALLOW_INSECURE_CORS")
    ):
        raise RuntimeError(
            "CORS_ORIGINS='*' is not allowed when API authentication is required. "
            "Set CORS_ORIGINS to explicit origins or AUTOVOICE_ALLOW_INSECURE_CORS=true."
        )
    return origins


def _configured_api_token(app: Flask) -> str | None:
    token = app.config.get("AUTOVOICE_API_TOKEN") or os.environ.get("AUTOVOICE_API_TOKEN")
    if token:
        return str(token)
    return None


def _api_token_is_insecure(token: str) -> bool:
    normalized = token.strip().lower()
    return len(token.strip()) < 32 or normalized in INSECURE_API_TOKENS or "changeme" in normalized


def validate_api_token_configuration(app: Flask) -> None:
    """Fail closed when hosted/public mode uses a missing or placeholder token."""
    if not api_auth_required(app) or app.config.get("TESTING"):
        return

    token = _configured_api_token(app)
    if not token:
        raise RuntimeError(
            "AUTOVOICE_REQUIRE_API_AUTH/AUTOVOICE_PUBLIC_DEPLOYMENT requires AUTOVOICE_API_TOKEN"
        )
    if _api_token_is_insecure(token) and not env_bool("AUTOVOICE_ALLOW_INSECURE_API_TOKEN"):
        raise RuntimeError(
            "Authenticated AutoVoice deployments require a non-placeholder AUTOVOICE_API_TOKEN "
            "with at least 32 characters, or AUTOVOICE_ALLOW_INSECURE_API_TOKEN=true for local testing"
        )


def _request_token() -> str | None:
    authorization = request.headers.get("Authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip()
    header_token = request.headers.get("X-AutoVoice-API-Key") or request.headers.get("X-API-Key")
    if header_token:
        return header_token

    if env_bool("AUTOVOICE_ALLOW_QUERY_API_TOKEN"):
        return request.args.get("api_token")
    return None


def token_authorized(app: Flask, token: str | None = None) -> bool:
    expected = _configured_api_token(app)
    if not expected:
        return not api_auth_required(app)
    supplied = token if token is not None else _request_token()
    return bool(supplied) and hmac.compare_digest(str(supplied), expected)


def _request_key() -> str:
    token = _request_token()
    if token:
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
        return f"token:{digest}"
    return f"ip:{request.remote_addr or 'unknown'}"


def _rate_limit_response(limit: int) -> tuple[Any, int]:
    return jsonify({"error": f"rate limit exceeded: {limit} requests per minute"}), 429


def init_production_security(app: Flask) -> None:
    """Install request ID, optional API-token auth, and public-mode rate limiting."""
    app._rate_limit_buckets = defaultdict(deque)

    validate_api_token_configuration(app)

    if socketio_cors_allowed_origins(app) == "*" and api_auth_required(app):
        # The helper raises unless explicitly overridden; call it here so HTTP-only
        # test paths also fail fast under unsafe authenticated CORS.
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
        origins = parse_cors_origins(app.config.get("CORS_ORIGINS"))
        request_origin = request.headers.get("Origin")
        if isinstance(origins, list) and request_origin in origins:
            response.headers["Access-Control-Allow-Origin"] = request_origin
            response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Credentials"] = "true"
        elif origins == "*" and not public_deployment_enabled(app):
            response.headers["Access-Control-Allow-Origin"] = "*"

        if request.method == "OPTIONS":
            response.headers.setdefault("Access-Control-Allow-Methods", "GET,POST,PUT,PATCH,DELETE,OPTIONS")
            response.headers.setdefault(
                "Access-Control-Allow-Headers",
                "Authorization,Content-Type,X-AutoVoice-API-Key,X-API-Key,X-Request-ID",
            )
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


def resolve_managed_media_path(
    raw_path: Any,
    *,
    data_dir: str | Path,
    upload_folder: str | Path | None = None,
) -> Path | None:
    if not raw_path:
        return None
    try:
        path = Path(str(raw_path)).expanduser().resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    if not path.exists() or not path.is_file():
        return None
    if not any(_is_relative_to(path, root) for root in managed_audio_roots(data_dir, upload_folder)):
        return None
    return path


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


def asset_id_for_path(
    raw_path: Any,
    *,
    asset_kind: str = "asset",
    app: Flask | None = None,
) -> str | None:
    if not raw_path:
        return None
    try:
        normalized_path = str(Path(str(raw_path)).expanduser().resolve())
    except (OSError, RuntimeError, TypeError, ValueError):
        normalized_path = str(raw_path)

    target_app = app
    if target_app is None and has_app_context():
        target_app = current_app._get_current_object()

    secret = (
        (target_app.config.get("AUTOVOICE_API_TOKEN") if target_app else None)
        or (target_app.config.get("SECRET_KEY") if target_app else None)
        or os.environ.get("AUTOVOICE_ASSET_ID_SECRET")
        or "autovoice-asset-id"
    )
    digest = hmac.new(
        str(secret).encode("utf-8"),
        msg=f"{asset_kind}:{normalized_path}".encode("utf-8"),
        digestmod="sha256",
    ).hexdigest()
    return f"{asset_kind}_{digest[:24]}"


def build_asset_reference(
    raw_path: Any,
    *,
    asset_kind: str = "asset",
    app: Flask | None = None,
) -> dict[str, Any] | None:
    asset_id = asset_id_for_path(raw_path, asset_kind=asset_kind, app=app)
    if asset_id is None:
        return None

    resolved_path = None
    size_bytes = None
    try:
        resolved = Path(str(raw_path)).expanduser().resolve()
        resolved_path = str(resolved)
        if resolved.exists() and resolved.is_file():
            size_bytes = resolved.stat().st_size
    except (OSError, RuntimeError, TypeError, ValueError):
        resolved_path = str(raw_path)

    reference: dict[str, Any] = {
        "asset_id": asset_id,
        "kind": asset_kind,
        "filename": Path(str(raw_path)).name or None,
    }
    if size_bytes is not None:
        reference["size_bytes"] = size_bytes

    target_app = app
    if target_app is None and has_app_context():
        target_app = current_app._get_current_object()
    if not (target_app and response_path_redaction_enabled(target_app)):
        reference["path"] = resolved_path
    return reference


def annotate_asset_payload(payload: Any, *, app: Flask | None = None) -> Any:
    target_app = app
    if target_app is None and has_app_context():
        target_app = current_app._get_current_object()
    hide_paths = bool(target_app and response_path_redaction_enabled(target_app))

    def _walk(value: Any) -> Any:
        if isinstance(value, dict):
            annotated: dict[str, Any] = {}
            for key, item in value.items():
                list_mapping = _ASSET_LIST_FIELDS.get(key)
                if list_mapping and isinstance(item, list):
                    asset_field, asset_kind = list_mapping
                    asset_ids = [
                        asset_id_for_path(entry, asset_kind=asset_kind, app=target_app)
                        for entry in item
                        if entry
                    ]
                    annotated[key] = [] if hide_paths else list(item)
                    if asset_ids:
                        annotated[asset_field] = [asset_id for asset_id in asset_ids if asset_id]
                    continue

                scalar_mapping = _ASSET_SCALAR_FIELDS.get(key)
                if scalar_mapping and item:
                    asset_field, asset_kind = scalar_mapping
                    annotated[key] = None if hide_paths else item
                    asset_id = asset_id_for_path(item, asset_kind=asset_kind, app=target_app)
                    if asset_id:
                        annotated[asset_field] = asset_id
                    continue

                annotated[key] = _walk(item)
            return annotated
        if isinstance(value, list):
            return [_walk(item) for item in value]
        return value

    return _walk(value=payload)


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
    if not _strict_bool(data.get("consent_confirmed")) or not _strict_bool(
        data.get("source_media_policy_confirmed")
    ):
        raise PermissionError(
            "consent_confirmed and source_media_policy_confirmed are required in public mode"
        )


def record_structured_audit_event(
    action: str,
    resource_type: str,
    *,
    app: Flask | None = None,
    resource_id: str | None = None,
    asset_paths: list[Any] | None = None,
    asset_kind: str = "asset",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target_app = app
    if target_app is None and has_app_context():
        target_app = current_app._get_current_object()

    asset_refs = [
        reference
        for reference in (
            build_asset_reference(path, asset_kind=asset_kind, app=target_app)
            for path in (asset_paths or [])
        )
        if reference is not None
    ]

    event: dict[str, Any] = {
        "event_id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "mode": "public" if target_app and public_deployment_enabled(target_app) else "local",
        "asset_ids": [reference["asset_id"] for reference in asset_refs],
        "assets": asset_refs,
        "details": dict(details or {}),
    }
    if has_request_context():
        event["request"] = {
            "id": getattr(g, "request_id", None),
            "method": request.method,
            "path": request.path,
            "remote_addr": request.remote_addr,
        }

    state_store = getattr(target_app, "state_store", None) if target_app is not None else None
    if state_store is not None and hasattr(state_store, "record_audit_event"):
        state_store.record_audit_event(event)
    return event


PATH_RESPONSE_FIELDS = {
    "audio_path",
    "audioPath",
    "filtered_audio_path",
    "filteredPath",
    "file_path",
    "model_path",
    "full_model_path",
    "tensorrt_engine_path",
    "adapter_path",
    "embedding_path",
    "primary_reference_audio_path",
    "runtime_artifact_manifest_path",
    "instrumental_path",
    "vocals_path",
    "original_path",
    "path",
}

PATH_LIST_RESPONSE_FIELDS = {
    "reference_audio_paths",
}


def public_asset_payload(app: Flask, state_store: Any, path: Any, *, kind: str, owner_id: str | None = None) -> dict[str, Any]:
    """Return an opaque asset reference and persist the local path server-side."""
    if not path:
        return {}
    asset = state_store.register_asset(str(path), kind=kind, owner_id=owner_id)
    return {"asset_id": asset["asset_id"]}


def redact_public_paths(
    payload: Any,
    app: Flask,
    state_store: Any,
    *,
    owner_id: str | None = None,
    kind: str = "file",
) -> Any:
    """Replace filesystem paths with opaque asset IDs when responses cross an auth boundary."""
    if not response_path_redaction_enabled(app):
        return payload
    if isinstance(payload, list):
        return [redact_public_paths(item, app, state_store, owner_id=owner_id, kind=kind) for item in payload]
    if not isinstance(payload, dict):
        return payload

    redacted: dict[str, Any] = {}
    for key, value in payload.items():
        if key in PATH_LIST_RESPONSE_FIELDS and isinstance(value, list):
            redacted[f"{key}_asset_ids"] = [
                public_asset_payload(app, state_store, item, kind=kind, owner_id=owner_id).get("asset_id")
                for item in value
                if item
            ]
            continue
        if key in PATH_RESPONSE_FIELDS and value:
            asset = public_asset_payload(app, state_store, value, kind=kind, owner_id=owner_id)
            redacted[f"{key}_asset_id"] = asset.get("asset_id")
            continue
        redacted[key] = redact_public_paths(value, app, state_store, owner_id=owner_id, kind=kind)
    return redacted
