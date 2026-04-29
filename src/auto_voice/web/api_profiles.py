"""Profile and sample API routes extracted from the legacy web API module."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

from .security import (
    annotate_asset_payload,
    api_auth_required,
    env_bool,
    public_deployment_enabled,
    redact_public_paths,
    record_structured_audit_event,
    require_media_consent,
    resolve_server_audio_path,
    strict_path_sandbox_enabled,
)

_deps: Dict[str, Any] = {}


def register_profile_sample_routes(api_bp: Blueprint, **deps: Any) -> None:
    """Register the profile/sample route family on the shared API blueprint."""
    _deps.update(deps)
    api_bp.add_url_rule('/voice/clone', view_func=clone_voice, methods=['POST'])
    api_bp.add_url_rule('/voice/profiles', view_func=get_voice_profiles, methods=['GET'])
    api_bp.add_url_rule('/voice/profiles/<profile_id>', view_func=get_voice_profile, methods=['GET'])
    api_bp.add_url_rule('/voice/profiles/<profile_id>', view_func=delete_voice_profile, methods=['DELETE'])
    api_bp.add_url_rule('/voice/profiles/<profile_id>/delete', view_func=delete_voice_profile, methods=['POST'])
    api_bp.add_url_rule('/voice/profiles/<profile_id>/export', view_func=export_voice_profile, methods=['GET'])
    api_bp.add_url_rule('/voice/profiles/<profile_id>/purge', view_func=purge_voice_profile, methods=['DELETE', 'POST'])
    api_bp.add_url_rule('/voice/profiles/<profile_id>/adapters', view_func=get_profile_adapters, methods=['GET'])
    api_bp.add_url_rule('/profiles/<profile_id>/adapters', view_func=get_profile_adapters, methods=['GET'])
    api_bp.add_url_rule('/voice/profiles/<profile_id>/model', view_func=get_profile_model, methods=['GET'])
    api_bp.add_url_rule('/profiles/<profile_id>/model', view_func=get_profile_model, methods=['GET'])
    api_bp.add_url_rule('/voice/profiles/<profile_id>/adapter/select', view_func=select_profile_adapter, methods=['POST'])
    api_bp.add_url_rule('/profiles/<profile_id>/adapter/select', view_func=select_profile_adapter, methods=['POST'])
    api_bp.add_url_rule('/voice/profiles/<profile_id>/adapter/metrics', view_func=get_adapter_metrics, methods=['GET'])
    api_bp.add_url_rule('/profiles/<profile_id>/adapter/metrics', view_func=get_adapter_metrics, methods=['GET'])
    api_bp.add_url_rule('/voice/profiles/<profile_id>/training-status', view_func=get_profile_training_status, methods=['GET'])
    api_bp.add_url_rule('/profiles/<profile_id>/training-status', view_func=get_profile_training_status, methods=['GET'])
    api_bp.add_url_rule('/profiles/<profile_id>/samples', view_func=list_samples, methods=['GET'])
    api_bp.add_url_rule('/profiles/<profile_id>/samples', view_func=upload_sample, methods=['POST'])
    api_bp.add_url_rule('/profiles/<profile_id>/samples/from-path', view_func=add_sample_from_path, methods=['POST'])
    api_bp.add_url_rule('/profiles/<profile_id>/songs', view_func=upload_song, methods=['POST'])
    api_bp.add_url_rule('/separation/<job_id>/status', view_func=get_separation_status, methods=['GET'])
    api_bp.add_url_rule('/profiles/<profile_id>/samples/<sample_id>', view_func=get_sample, methods=['GET'])
    api_bp.add_url_rule('/profiles/<profile_id>/samples/<sample_id>', view_func=delete_sample, methods=['DELETE'])


def _dep(name: str) -> Any:
    return _deps[name]


# In-memory storage for samples (fallback for samples not in VoiceProfileStore)
_profile_samples: Dict[str, Dict[str, Dict[str, Any]]] = {}

# In-memory storage for song separation jobs
_separation_jobs: Dict[str, Dict[str, Any]] = {}


def _form_bool(name: str) -> bool:
    return str(request.form.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _form_media_consent_payload() -> Dict[str, bool]:
    return {
        "consent_confirmed": _form_bool("consent_confirmed"),
        "source_media_policy_confirmed": _form_bool("source_media_policy_confirmed"),
    }


def _ensure_profile_for_song_upload(profile_id: str) -> bool:
    try:
        _dep('ensure_profile_in_store')(profile_id)
        return True
    except _dep('ProfileNotFoundError'):
        pass

    try:
        from auto_voice.profiles.db.models import VoiceProfileDB
        from auto_voice.profiles.db.session import get_db_session

        with get_db_session() as session:
            profile = session.query(VoiceProfileDB).filter_by(id=profile_id).first()
            if not profile:
                return False
            _dep('get_profile_store')().save({
                'profile_id': profile.id,
                'name': profile.name,
                'profile_role': 'target_user',
                'training_status': 'pending',
                'has_trained_model': False,
                'metadata': {
                    'source': 'legacy_db_profile',
                    'user_id': profile.user_id,
                },
            })
        return True
    except Exception:
        return False


def _serialize_training_sample(profile_id: str, training_sample: Any) -> Dict[str, Any]:
    quality_metadata = dict(getattr(training_sample, 'quality_metadata', {}) or {})
    extra_metadata = dict(getattr(training_sample, 'extra_metadata', {}) or {})
    return {
        'id': training_sample.sample_id,
        'sample_id': training_sample.sample_id,
        'profile_id': profile_id,
        'audio_path': training_sample.vocals_path,
        'file_path': training_sample.vocals_path,
        'duration_seconds': training_sample.duration,
        'duration': training_sample.duration,
        'sample_rate': 44100,
        'created': training_sample.created_at,
        'created_at': training_sample.created_at,
        'source_file': training_sample.source_file,
        'filename': os.path.basename(training_sample.vocals_path),
        'metadata': {
            'source_file': training_sample.source_file,
            'instrumental_path': training_sample.instrumental_path,
            **extra_metadata,
        },
        'quality_metadata': quality_metadata,
    }


def _find_training_sample(profile_id: str, sample_id: str):
    store = _dep('get_profile_store')()
    for sample in store.list_training_samples(profile_id):
        if sample.sample_id == sample_id:
            return sample
    legacy_samples = _profile_samples.get(profile_id, {})
    legacy_sample = legacy_samples.get(sample_id)
    if legacy_sample:
        return type('LegacyTrainingSample', (), {
            'sample_id': legacy_sample.get('sample_id', sample_id),
            'vocals_path': legacy_sample.get('vocals_path') or legacy_sample.get('file_path'),
            'duration': legacy_sample.get('duration', legacy_sample.get('duration_seconds', 0.0)),
            'source_file': legacy_sample.get('source_file'),
            'created_at': legacy_sample.get('created_at') or legacy_sample.get('created'),
            'instrumental_path': legacy_sample.get('instrumental_path'),
        })()
    return None


def clone_voice():
    """Clone voice from reference audio to create new voice profile."""
    logger = _dep('logger')
    voice_cloner = getattr(current_app, 'voice_cloner', None)
    if voice_cloner is None:
        logger.warning("Voice cloner service unavailable")
        return jsonify({
            'error': 'Voice cloning service unavailable',
            'message': 'Voice cloner not initialized'
        }), 503

    if 'reference_audio' not in request.files:
        return _dep('validation_error_response')('No reference_audio file provided')

    audio_file = request.files['reference_audio']
    if audio_file.filename == '':
        return _dep('validation_error_response')('No file selected')

    if not _dep('allowed_file')(audio_file.filename):
        return _dep('validation_error_response')('Invalid file format')

    user_id = request.form.get('user_id')
    name = request.form.get('name')
    try:
        require_media_consent(
            {
                'consent_confirmed': request.form.get('consent_confirmed') in ('1', 'true', 'True', 'yes', 'on'),
                'source_media_policy_confirmed': request.form.get('source_media_policy_confirmed') in (
                    '1', 'true', 'True', 'yes', 'on'
                ),
            },
            current_app,
        )
    except PermissionError as exc:
        return _dep('validation_error_response')(str(exc))

    tmp_file = None
    try:
        secure_name = secure_filename(audio_file.filename)
        tmp_file = tempfile.NamedTemporaryFile(
            suffix=os.path.splitext(secure_name)[1], delete=False
        )
        audio_file.save(tmp_file.name)

        result = voice_cloner.create_voice_profile(
            audio=tmp_file.name, user_id=user_id, name=name
        )

        response_data = result.copy()
        response_data.pop('embedding', None)
        response_data['status'] = 'success'
        for field in ['profile_id', 'audio_duration', 'user_id', 'name', 'vocal_range', 'created_at']:
            if field not in response_data:
                response_data[field] = result.get(field)
        _audit_event(
            "voice_profile.created",
            "voice_profile",
            response_data.get("profile_id"),
            {"name": name, "user_id": user_id, "source_filename": secure_name},
        )
        return jsonify(_redact_profile_payload(response_data, response_data.get("profile_id"))), 201

    except _dep('InvalidAudioError') as e:
        logger.warning(f"Invalid audio for voice cloning: {e}")
        return _dep('validation_error_response')(
            'Invalid reference audio',
            message=str(e),
            error_code='invalid_reference_audio'
        )

    except _dep('InsufficientQualityError') as e:
        logger.warning(f"Insufficient audio quality for voice cloning: {e}")
        kwargs = {
            'message': str(e),
            'error_code': getattr(e, 'error_code', 'insufficient_quality')
        }
        if hasattr(e, 'details') and e.details:
            kwargs['details'] = e.details
        return _dep('validation_error_response')('Insufficient audio quality', **kwargs)

    except _dep('InconsistentSamplesError') as e:
        logger.warning(f"Inconsistent samples for voice cloning: {e}")
        kwargs = {
            'message': str(e),
            'error_code': getattr(e, 'error_code', 'inconsistent_samples')
        }
        if hasattr(e, 'details') and e.details:
            kwargs['details'] = e.details
        return _dep('validation_error_response')('Inconsistent audio samples', **kwargs)

    except Exception as e:
        logger.error(f"Voice cloning error: {e}", exc_info=True)
        return _dep('service_unavailable_response')(
            'Temporary service unavailability during voice cloning',
            message=str(e),
        )

    finally:
        if tmp_file and os.path.exists(tmp_file.name):
            try:
                os.unlink(tmp_file.name)
            except OSError:
                pass


def get_voice_profiles():
    """List voice profiles for optional user_id."""
    logger = _dep('logger')
    user_id = request.args.get('user_id')
    try:
        store = _dep('get_profile_store')()
        profiles = store.list_profiles(user_id=user_id)
        clean_profiles = [
            _redact_profile_payload(_dep('serialize_profile_for_response')(profile), profile.get('profile_id'))
            for profile in profiles
        ]
        return jsonify(clean_profiles)
    except Exception as e:
        logger.error(f"Voice cloner list_profiles error: {e}", exc_info=True)
        return _dep('service_unavailable_response')(
            'Temporary service unavailability during profile listing',
            message=str(e),
        )


def get_voice_profile(profile_id):
    """Get specific voice profile by ID."""
    logger = _dep('logger')
    voice_cloner = getattr(current_app, 'voice_cloner', None)
    if voice_cloner is None:
        logger.warning("Voice cloner service unavailable")
        return jsonify({
            'error': 'Voice cloning service unavailable',
            'message': 'Voice cloner not initialized'
        }), 503

    try:
        profile = voice_cloner.load_voice_profile(profile_id)
        if profile is None:
            logger.warning(f"Voice profile not found: {profile_id}")
            return jsonify({
                'error': 'Voice profile not found',
                'profile_id': profile_id
            }), 404

        clean_profile = _dep('serialize_profile_for_response')(profile)
        adapter_artifact = _dep('get_canonical_adapter_artifact')(profile_id, profile)
        if adapter_artifact is not None:
            clean_profile['adapter_path'] = adapter_artifact['path']

        return jsonify(_redact_profile_payload(clean_profile, profile_id))
    except _dep('ProfileNotFoundError'):
        logger.info(f"Voice profile not found via exception: {profile_id}")
        return jsonify({
            'error': 'Voice profile not found',
            'profile_id': profile_id
        }), 404
    except Exception as e:
        logger.error(f"Error loading voice profile {profile_id}: {e}", exc_info=True)
        return _dep('service_unavailable_response')(
            'Temporary service unavailability during profile retrieval',
            message=str(e),
        )


def delete_voice_profile(profile_id):
    """Delete voice profile by ID."""
    logger = _dep('logger')
    voice_cloner = getattr(current_app, 'voice_cloner', None)
    if voice_cloner is None:
        logger.warning("Voice cloner service unavailable")
        return jsonify({
            'error': 'Voice cloning service unavailable',
            'message': 'Voice cloner not initialized'
        }), 503

    try:
        profile = _dep('load_runtime_profile')(profile_id)
        sample_payloads = []
        profile_payload = None
        if profile is not None:
            profile_payload = _dep('serialize_profile_for_response')(profile)
            sample_payloads = [
                _serialize_training_sample(profile_id, sample)
                for sample in _dep('get_profile_store')().list_training_samples(profile_id)
            ]

        deleted = voice_cloner.delete_voice_profile(profile_id)
        if deleted:
            purge_summary = _state_store().purge_profile_state(profile_id) if _state_store() else {}
            _audit_event(
                "voice_profile.deleted",
                "voice_profile",
                profile_id,
                {"profile_id": profile_id, "purge_summary": purge_summary},
                payload={"profile": profile_payload, "samples": sample_payloads},
            )
            logger.info(f"Voice profile deleted: {profile_id}")
            return jsonify({
                'status': 'success',
                'profile_id': profile_id,
                'purge_summary': purge_summary,
            })

        logger.warning(f"Voice profile not found for deletion: {profile_id}")
        return jsonify({
            'error': 'Voice profile not found',
            'profile_id': profile_id
        }), 404
    except _dep('ProfileNotFoundError'):
        logger.info(f"Voice profile not found for deletion via exception: {profile_id}")
        return jsonify({
            'error': 'Voice profile not found',
            'profile_id': profile_id
        }), 404
    except Exception as e:
        logger.error(f"Voice profile deletion error: {e}", exc_info=True)
        return _dep('service_unavailable_response')(
            'Temporary service unavailability during profile deletion',
            message=str(e),
        )


def _state_store():
    getter = _deps.get('get_state_store')
    return getter() if getter else None


_AUDIT_ACTION_OVERRIDES = {
    "voice_profile.created": "clone",
    "voice_profile.deleted": "delete",
    "voice_profile.exported": "export",
    "voice_profile.purged": "delete",
    "training_sample.imported": "import",
    "training_sample.deleted": "delete",
}

_PATH_KEYS = {
    "audio_path",
    "audioPath",
    "filtered_audio_path",
    "filteredPath",
    "file_path",
    "model_path",
    "adapter_path",
    "embedding_path",
    "path",
    "instrumental_path",
    "primary_reference_audio_path",
    "runtime_artifact_manifest_path",
    "full_model_path",
    "tensorrt_engine_path",
    "vocals_path",
    "original_path",
}


def _collect_path_values(payload: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "reference_audio_paths" and isinstance(value, list):
                paths.extend(str(item) for item in value if item)
                continue
            if key in _PATH_KEYS and value:
                paths.append(str(value))
                continue
            paths.extend(_collect_path_values(value))
    elif isinstance(payload, list):
        for item in payload:
            paths.extend(_collect_path_values(item))
    return paths


def _audit_event(
    event_type: str,
    resource_type: str,
    resource_id: str | None,
    metadata: Dict[str, Any],
    *,
    payload: Any = None,
    asset_paths: Optional[list[str]] = None,
) -> None:
    try:
        action = metadata.get("action") or _AUDIT_ACTION_OVERRIDES.get(
            event_type,
            event_type.rsplit(".", 1)[-1],
        )
        details = dict(metadata)
        details["event_type"] = event_type
        paths = _collect_path_values(payload) if payload is not None else []
        if asset_paths:
            paths.extend(str(path) for path in asset_paths if path)
        record_structured_audit_event(
            action,
            resource_type,
            app=current_app,
            resource_id=resource_id,
            asset_paths=paths,
            asset_kind="voice_profile",
            details=details,
        )
    except Exception:
        _dep('logger').warning("Failed to persist audit event %s", event_type, exc_info=True)


def _redact_profile_payload(payload: Dict[str, Any], profile_id: str) -> Dict[str, Any]:
    store = _state_store()
    annotated = annotate_asset_payload(payload, app=current_app)
    if store is None or not public_deployment_enabled(current_app):
        return annotated
    return redact_public_paths(payload, current_app, store, owner_id=profile_id, kind="voice_profile")


def export_voice_profile(profile_id):
    """Export profile metadata and server-side asset references for portability/compliance."""
    profile = _dep('load_runtime_profile')(profile_id)
    if profile is None:
        return _dep('not_found_response')('Voice profile not found')

    store = _dep('get_profile_store')()
    samples = [_serialize_training_sample(profile_id, sample) for sample in store.list_training_samples(profile_id)]
    manifest = {
        "profile": _dep('serialize_profile_for_response')(profile),
        "samples": samples,
        "state": _state_store().export_profile_state(profile_id) if _state_store() else {},
        "assets": _state_store().list_assets(owner_id=profile_id) if _state_store() else [],
        "retention": {
            "audit_records_included": True,
            "audit_records_retained_after_purge": True,
            "purge_scope": "profile_state_and_registered_owned_assets",
        },
        "exported_at": time.time(),
    }
    _audit_event(
        "voice_profile.exported",
        "voice_profile",
        profile_id,
        {"sample_count": len(samples), "asset_count": len(manifest["assets"])},
        payload=manifest,
    )
    return jsonify(_redact_profile_payload(manifest, profile_id))


def purge_voice_profile(profile_id):
    """Delete profile metadata plus registered owned assets from the local deployment."""
    logger = _dep('logger')
    voice_cloner = getattr(current_app, 'voice_cloner', None)
    if voice_cloner is None:
        logger.warning("Voice cloner service unavailable")
        return jsonify({
            'error': 'Voice cloning service unavailable',
            'message': 'Voice cloner not initialized'
        }), 503

    store = _state_store()
    profile = _dep('load_runtime_profile')(profile_id)
    sample_payloads = []
    profile_payload = None
    if profile is not None:
        profile_payload = _dep('serialize_profile_for_response')(profile)
        sample_payloads = [
            _serialize_training_sample(profile_id, sample)
            for sample in _dep('get_profile_store')().list_training_samples(profile_id)
        ]
    purged_assets: list[str] = []
    purge_summary = store.purge_profile_state(profile_id) if store is not None else {}
    if store is not None:
        for asset in store.list_assets(owner_id=profile_id):
            if store.delete_asset(asset["asset_id"]):
                purged_assets.append(asset["asset_id"])

    try:
        deleted = voice_cloner.delete_voice_profile(profile_id)
    except _dep('ProfileNotFoundError'):
        deleted = False

    if not deleted and not purged_assets:
        return jsonify({
            'error': 'Voice profile not found',
            'profile_id': profile_id
        }), 404

    _audit_event(
        "voice_profile.purged",
        "voice_profile",
        profile_id,
        {
            "profile_deleted": bool(deleted),
            "purged_asset_count": len(purged_assets),
            "purge_summary": purge_summary,
        },
        payload={"profile": profile_payload, "samples": sample_payloads},
    )
    return jsonify({
        "status": "success",
        "profile_id": profile_id,
        "profile_deleted": bool(deleted),
        "purged_asset_ids": purged_assets,
        "purge_summary": purge_summary,
        "retention": {
            "audit_records_retained": True,
            "purge_scope": "profile_state_and_registered_owned_assets",
        },
    })


def get_profile_adapters(profile_id):
    """Get available LoRA adapters for a voice profile."""
    profile = _dep('load_runtime_profile')(profile_id)
    if profile is None:
        return _dep('not_found_response')('Voice profile not found')

    adapters = []
    selected = None
    adapter_artifact = _dep('get_canonical_adapter_artifact')(profile_id, profile)
    if adapter_artifact is not None:
        adapters.append({
            'type': adapter_artifact['type'],
            'path': adapter_artifact['path'],
            'size_kb': adapter_artifact['size_kb'],
            'epochs': adapter_artifact['epochs'],
            'loss': adapter_artifact['loss'],
            'precision': adapter_artifact['precision'],
            'config': adapter_artifact['config'],
        })
        selected = adapter_artifact['type']

    return jsonify(_redact_profile_payload({
        'profile_id': profile_id,
        'adapters': adapters,
        'selected': selected,
        'count': len(adapters),
    }, profile_id))


def get_profile_model(profile_id):
    """Get trained model information for a voice profile."""
    logger = _dep('logger')
    if not profile_id or len(profile_id) != 36:
        return jsonify({
            'error': 'Invalid profile ID format',
            'message': 'Profile ID must be a valid UUID (36 characters)'
        }), 400

    profile = _dep('load_runtime_profile')(profile_id)
    if profile is None:
        return jsonify({
            'error': 'Voice profile not found',
            'profile_id': profile_id
        }), 404

    try:
        adapter_manager = _dep('get_adapter_manager')()
        available_artifact_types = adapter_manager.get_available_artifact_types(profile_id)
        has_adapter = "adapter" in available_artifact_types
        adapter_path = adapter_manager.get_adapter_path(profile_id) if has_adapter else None
        full_model_path = adapter_manager.get_full_model_path(profile_id)
        tensorrt_engine_path = adapter_manager.get_tensorrt_engine_path(profile_id)
        serialized_profile = _dep('serialize_profile_for_response')(profile)

        if not available_artifact_types:
            return jsonify(_redact_profile_payload({
                'profile_id': profile_id,
                'has_model': False,
                'has_trained_model': False,
                'has_adapter_model': False,
                'has_full_model': False,
                'model_type': None,
                'model_path': None,
                'active_model_type': serialized_profile['active_model_type'],
                'profile_role': serialized_profile['profile_role'],
                'clean_vocal_seconds': serialized_profile['clean_vocal_seconds'],
                'full_model_eligible': serialized_profile['full_model_eligible'],
                'available_artifact_types': [],
                'message': 'No trained model available for this profile',
            }, profile_id)), 404

        if not has_adapter:
            selected_artifact_type = available_artifact_types[0]
            selected_artifact_path = adapter_manager.get_artifact_path(
                profile_id,
                selected_artifact_type,
            )
            message = {
                'full_model': 'Full model checkpoint available for this profile',
                'tensorrt': 'TensorRT engine available for this profile',
            }.get(selected_artifact_type, 'Trained model available for this profile')

            return jsonify(_redact_profile_payload({
                'profile_id': profile_id,
                'has_model': True,
                'has_trained_model': True,
                'has_adapter_model': False,
                'has_full_model': selected_artifact_type == 'full_model',
                'model_type': selected_artifact_type,
                'model_path': str(selected_artifact_path) if selected_artifact_path else None,
                'active_model_type': serialized_profile['active_model_type'],
                'profile_role': serialized_profile['profile_role'],
                'clean_vocal_seconds': serialized_profile['clean_vocal_seconds'],
                'full_model_eligible': serialized_profile['full_model_eligible'],
                'available_artifact_types': available_artifact_types,
                'full_model_path': str(full_model_path) if full_model_path else None,
                'tensorrt_engine_path': (
                    str(tensorrt_engine_path) if tensorrt_engine_path else None
                ),
                'message': message,
            }, profile_id)), 200

        adapter_info = adapter_manager.get_adapter_info(profile_id)
        embedding_path = adapter_manager.get_embedding_path(profile_id)
        embedding_exists = embedding_path.exists()
        embedding_shape = None

        if embedding_exists:
            import numpy as np
            try:
                embedding = np.load(embedding_path)
                embedding_shape = embedding.shape
            except Exception as e:
                logger.warning(f"Failed to load embedding shape: {e}")

        created_at = None
        if adapter_path and adapter_path.exists():
            import datetime
            created_at = datetime.datetime.fromtimestamp(
                adapter_path.stat().st_mtime
            ).isoformat()

        return jsonify(_redact_profile_payload({
            'profile_id': profile_id,
            'has_model': True,
            'has_trained_model': True,
            'has_adapter_model': True,
            'has_full_model': 'full_model' in available_artifact_types,
            'model_type': 'adapter',
            'active_model_type': serialized_profile['active_model_type'],
            'adapter_path': str(adapter_path) if adapter_path else None,
            'selected_adapter': _dep('get_frontend_adapter_type')(profile),
            'profile_role': serialized_profile['profile_role'],
            'clean_vocal_seconds': serialized_profile['clean_vocal_seconds'],
            'full_model_eligible': serialized_profile['full_model_eligible'],
            'available_artifact_types': available_artifact_types,
            'full_model_path': str(full_model_path) if full_model_path else None,
            'tensorrt_engine_path': str(tensorrt_engine_path) if tensorrt_engine_path else None,
            'adapter_info': {
                'rank': adapter_info.rank if adapter_info else None,
                'alpha': adapter_info.alpha if adapter_info else None,
                'sample_count': adapter_info.sample_count if adapter_info else None,
                'training_epochs': adapter_info.training_epochs if adapter_info else None,
                'loss_final': adapter_info.loss_final if adapter_info else None,
                'profile_name': adapter_info.profile_name if adapter_info else None,
            },
            'embedding_path': str(embedding_path) if embedding_exists else None,
            'embedding_shape': list(embedding_shape) if embedding_shape is not None else None,
            'created_at': created_at,
        }, profile_id))

    except Exception as e:
        logger.error(f"Error getting model info for {profile_id}: {e}", exc_info=True)
        return jsonify({
            'error': 'Failed to retrieve model information',
            'message': str(e)
        }), 500


def select_profile_adapter(profile_id):
    """Select which LoRA adapter to use for voice conversion."""
    logger = _dep('logger')
    data = request.get_json() or {}
    adapter_type = data.get('adapter_type')

    if adapter_type not in ['hq', 'nvfp4', 'unified']:
        return jsonify({
            'error': 'Invalid adapter_type',
            'message': "Must be 'hq', 'nvfp4', or 'unified'"
        }), 400

    try:
        profile = _dep('ensure_profile_in_store')(profile_id)
    except _dep('ProfileNotFoundError'):
        return _dep('not_found_response')('Profile not found')

    adapter_artifact = _dep('get_canonical_adapter_artifact')(profile_id, profile)
    if adapter_artifact is None:
        return jsonify({
            'error': 'Adapter not found',
            'message': f'No trained adapter exists for profile {profile_id}'
        }), 404

    try:
        store = _dep('get_profile_store')()
        profile['selected_adapter'] = adapter_type
        store.save(dict(profile))

        return jsonify(_redact_profile_payload({
            'status': 'success',
            'success': True,
            'profile_id': profile_id,
            'selected': _dep('get_frontend_adapter_type')({'selected_adapter': adapter_type}),
            'selected_adapter': adapter_type,
            'adapter_path': adapter_artifact['path'],
        }, profile_id))
    except Exception as e:
        logger.error(f"Failed to select adapter: {e}", exc_info=True)
        return _dep('error_response')('Failed to select adapter', message=str(e))


def get_adapter_metrics(profile_id):
    """Get detailed metrics for all adapters of a voice profile."""
    profile = _dep('load_runtime_profile')(profile_id)
    if profile is None:
        return _dep('not_found_response')('Voice profile not found')

    metrics = {}
    adapter_artifact = _dep('get_canonical_adapter_artifact')(profile_id, profile)
    if adapter_artifact is not None:
        adapter_type = adapter_artifact['type']
        adapter_info = adapter_artifact['adapter_info']
        config = dict(adapter_artifact['config'])
        param_count = adapter_artifact['parameter_count']
        file_size_mb = adapter_artifact['size_kb'] / 1024
        metrics[adapter_type] = {
            'epochs': adapter_artifact['epochs'],
            'loss': adapter_artifact['loss'],
            'precision': adapter_artifact['precision'],
            'trained_on': None,
            'architecture': {
                'input_dim': config.get('input_dim', 768),
                'hidden_dim': config.get('hidden_dim', 1024),
                'output_dim': config.get('output_dim', 768),
                'num_layers': config.get('num_layers', 6),
                'lora_rank': config.get('lora_rank', adapter_info.rank if adapter_info else 8),
                'lora_alpha': config.get('lora_alpha', adapter_info.alpha if adapter_info else 16),
            },
            'file_size_kb': round(adapter_artifact['size_kb'], 1),
            'parameter_count': param_count,
            'parameter_count_formatted': (
                f"{param_count / 1e6:.2f}M"
                if param_count > 1e6
                else f"{param_count / 1e3:.1f}K" if param_count else 'N/A'
            ),
            'file_path': adapter_artifact['path'],
            'modified_time': adapter_artifact['modified_time'],
            'performance': {
                'relative_quality': 'high',
                'relative_speed': 'normal',
                'memory_estimate_mb': round(file_size_mb, 2),
            }
        }

    return jsonify(_redact_profile_payload({
        'profile_id': profile_id,
        'profile_name': profile.get('name', 'Unknown'),
        'adapters': metrics,
        'adapter_count': len(metrics),
        'recommended': next(iter(metrics.keys()), None),
    }, profile_id))


def get_profile_training_status(profile_id):
    """Get training status for a voice profile."""
    logger = _dep('logger')
    store = _dep('get_profile_store')()

    try:
        if not store.exists(profile_id):
            return jsonify({
                'error': 'Voice profile not found',
                'profile_id': profile_id
            }), 404

        has_trained = store.has_trained_model(profile_id)
        profile = store.load(profile_id)
        training_status = profile.get('training_status', 'pending' if not has_trained else 'ready')

        return jsonify({
            'profile_id': profile_id,
            'has_trained_model': has_trained,
            'training_status': training_status if not has_trained else 'ready',
            'model_version': profile.get('model_version'),
            'profile_role': profile.get('profile_role', 'target_user'),
            'clean_vocal_seconds': profile.get('clean_vocal_seconds', 0.0),
            'full_model_eligible': profile.get('full_model_eligible', False),
        })
    except _dep('ProfileNotFoundError'):
        return jsonify({
            'error': 'Voice profile not found',
            'profile_id': profile_id
        }), 404
    except Exception as e:
        logger.error(f"Training status error for {profile_id}: {e}", exc_info=True)
        return _dep('service_unavailable_response')('Failed to get training status', message=str(e))


def list_samples(profile_id: str):
    """List all samples for a profile."""
    logger = _dep('logger')
    if _dep('load_runtime_profile')(profile_id) is None:
        from auto_voice.profiles.api import list_samples as legacy_list_samples
        return legacy_list_samples(profile_id)

    try:
        store = _dep('get_profile_store')()
        training_samples = store.list_training_samples(profile_id)
        samples = [_serialize_training_sample(profile_id, sample) for sample in training_samples]
        if samples:
            return jsonify(_redact_profile_payload(samples, profile_id))
    except Exception as e:
        logger.warning(f"Failed to get samples from VoiceProfileStore: {e}")

    samples = _profile_samples.get(profile_id, {})
    return jsonify(_redact_profile_payload(list(samples.values()), profile_id))


def upload_sample(profile_id: str):
    """Upload a new training sample for a profile."""
    logger = _dep('logger')
    try:
        file = None
        if 'file' in request.files:
            file = request.files['file']
        elif 'audio' in request.files:
            file = request.files['audio']

        if not file:
            return _dep('validation_error_response')('No file provided (expected "file" or "audio" field)')
        if not file.filename:
            return _dep('validation_error_response')('No file selected')

        if not _dep('allowed_file')(file.filename):
            return _dep('validation_error_response')('Invalid file type')

        try:
            require_media_consent(_form_media_consent_payload(), current_app)
        except PermissionError as exc:
            return _dep('validation_error_response')(str(exc))

        try:
            _dep('ensure_profile_in_store')(profile_id)
        except _dep('ProfileNotFoundError'):
            from auto_voice.profiles.api import upload_sample as legacy_upload_sample
            return legacy_upload_sample(profile_id)

        filename = secure_filename(file.filename)
        upload_dir = os.path.join(_dep('upload_folder'), 'incoming-samples', profile_id)
        os.makedirs(upload_dir, exist_ok=True)

        file_path = os.path.join(upload_dir, f"{uuid.uuid4()}_{filename}")
        file.save(file_path)

        metadata = {}
        if request.form.get('metadata'):
            try:
                metadata = json.loads(request.form.get('metadata'))
            except json.JSONDecodeError:
                pass

        duration = 0.0
        if _dep('soundfile_available'):
            try:
                duration = float(_dep('soundfile').info(file_path).duration)
            except Exception:
                duration = 0.0

        store = _dep('get_profile_store')()
        training_sample = store.add_training_sample(
            profile_id=profile_id,
            vocals_path=file_path,
            source_file=metadata.get('source_file') or filename,
            duration=duration,
            extra_metadata={
                'source': metadata.get('source', 'upload'),
                'provenance': metadata.get('provenance') or filename,
            },
            quality_metadata=metadata.get('quality_metadata'),
        )
        if metadata:
            sample_payload = _serialize_training_sample(profile_id, training_sample)
            sample_payload['metadata'].update(metadata)
        else:
            sample_payload = _serialize_training_sample(profile_id, training_sample)

        logger.info(f"Uploaded sample {training_sample.sample_id} for profile {profile_id}")
        _audit_event(
            "training_sample.imported",
            "training_sample",
            training_sample.sample_id,
            {
                "profile_id": profile_id,
                "sample_id": training_sample.sample_id,
                "source": sample_payload.get("metadata", {}).get("source", "upload"),
            },
            payload=sample_payload,
        )
        return jsonify(_redact_profile_payload(sample_payload, profile_id)), 201
    except Exception as e:
        logger.error(f"Error uploading sample: {e}", exc_info=True)
        return _dep('error_response')(str(e))


def add_sample_from_path(profile_id: str):
    """Add a training sample from an existing file path on the server."""
    logger = _dep('logger')
    try:
        _dep('ensure_profile_in_store')(profile_id)
        data = request.get_json() or {}
        audio_path = data.get('audio_path')
        audio_asset_id = (
            data.get('audio_asset_id')
            or data.get('audioAssetId')
            or data.get('audio_path_asset_id')
            or data.get('asset_id')
        )
        skip_separation = data.get('skip_separation', False)

        try:
            require_media_consent(data, current_app)
            if audio_asset_id:
                asset = (_state_store().get_asset(str(audio_asset_id)) if _state_store() else None)
                if not asset:
                    return _dep('not_found_response')(f'Asset not found: {audio_asset_id}')
                owner_id = asset.get('owner_id')
                if owner_id not in {None, profile_id}:
                    return _dep('validation_error_response')('Asset is not owned by this profile')
                audio_path = asset.get('path')
            elif api_auth_required(current_app) and not env_bool("AUTOVOICE_ALLOW_SERVER_AUDIO_PATH_IMPORT"):
                return _dep('validation_error_response')(
                    'audio_asset_id is required for server-side sample imports when API authentication is enabled'
                )
            resolved_audio_path = resolve_server_audio_path(
                audio_path,
                data_dir=current_app.config.get('DATA_DIR', 'data'),
                upload_folder=_dep('upload_folder'),
                strict=strict_path_sandbox_enabled(current_app),
            )
        except FileNotFoundError:
            return _dep('not_found_response')(f'File not found: {audio_path}')
        except PermissionError as exc:
            return _dep('validation_error_response')(str(exc))
        except ValueError as exc:
            return _dep('validation_error_response')(str(exc))

        audio_path = str(resolved_audio_path)

        filename = os.path.basename(audio_path)
        base_name = os.path.splitext(filename)[0]
        metadata = data.get('metadata', {})
        metadata['source'] = 'youtube_download'
        metadata['original_path'] = audio_path

        if skip_separation:
            upload_dir = os.path.join(_dep('upload_folder'), 'incoming-samples', profile_id)
            os.makedirs(upload_dir, exist_ok=True)
            dest_path = os.path.join(upload_dir, f"{uuid.uuid4()}_{filename}")
            import shutil
            shutil.copy2(audio_path, dest_path)
            logger.info(f"Added sample without separation: {dest_path}")
        else:
            logger.info(f"Running vocal separation on: {audio_path}")
            if not _dep('soundfile_available'):
                return _dep('error_response')('soundfile not available for audio loading')

            audio, sr = _dep('soundfile').read(audio_path)
            if audio.ndim > 1:
                audio = audio.T
            logger.info(f"Loaded audio: {audio.shape}, sr={sr}")

            from auto_voice.audio.separation import VocalSeparator

            if _dep('torch_available'):
                _dep('torch').cuda.empty_cache()
                _dep('torch').cuda.synchronize()

            separator = VocalSeparator(segment=10.0)
            duration_sec = len(audio) / sr if audio.ndim == 1 else audio.shape[-1] / sr
            logger.info(f"Starting vocal separation ({duration_sec:.1f}s audio, 10s segments)...")
            result = separator.separate(audio.T if audio.ndim > 1 else audio, sr)
            vocals = result['vocals']
            instrumental = result['instrumental']

            logger.info(f"Separation complete: vocals={vocals.shape}, instrumental={instrumental.shape}")

            upload_dir = os.path.join(_dep('upload_folder'), 'incoming-samples', profile_id)
            os.makedirs(upload_dir, exist_ok=True)
            sample_prefix = str(uuid.uuid4())
            vocals_filename = f"{sample_prefix}_{base_name}_vocals.wav"
            dest_path = os.path.join(upload_dir, vocals_filename)
            _dep('soundfile').write(dest_path, vocals, sr)
            logger.info(f"Saved vocals to: {dest_path}")

            instrumental_filename = f"{sample_prefix}_{base_name}_instrumental.wav"
            instrumental_path = os.path.join(upload_dir, instrumental_filename)
            _dep('soundfile').write(instrumental_path, instrumental, sr)
            logger.info(f"Saved instrumental to: {instrumental_path}")

            metadata['separated'] = True
            metadata['instrumental_path'] = instrumental_path
            filename = vocals_filename

        duration = None
        if _dep('soundfile_available'):
            try:
                info = _dep('soundfile').info(dest_path)
                duration = info.duration
            except Exception as e:
                logger.warning(f"Could not get duration: {e}")

        store = _dep('get_profile_store')()
        training_sample = store.add_training_sample(
            profile_id=profile_id,
            vocals_path=dest_path,
            instrumental_path=metadata.get('instrumental_path'),
            source_file=metadata.get('original_path') or filename,
            duration=duration or 0.0,
            extra_metadata={
                'source': metadata.get('source', 'youtube_download'),
                'original_path': metadata.get('original_path'),
                'provenance': metadata.get('original_path') or filename,
                'separated': metadata.get('separated', False),
            },
            quality_metadata=metadata.get('quality_metadata'),
        )
        sample_payload = _serialize_training_sample(profile_id, training_sample)
        sample_payload['metadata'].update(metadata)

        logger.info(f"Added sample {training_sample.sample_id} (vocals) from path for profile {profile_id}")
        _audit_event(
            "training_sample.imported",
            "training_sample",
            training_sample.sample_id,
            {
                "profile_id": profile_id,
                "sample_id": training_sample.sample_id,
                "source": sample_payload.get("metadata", {}).get("source", "youtube_download"),
                "skip_separation": bool(skip_separation),
            },
            payload=sample_payload,
        )
        return jsonify(_redact_profile_payload(sample_payload, profile_id)), 201
    except Exception as e:
        logger.error(f"Error adding sample from path: {e}", exc_info=True)
        return _dep('error_response')(str(e))


def upload_song(profile_id: str):
    """Upload a song for vocal separation (uses Demucs to extract vocals for training)."""
    logger = _dep('logger')
    try:
        file = None
        if 'file' in request.files:
            file = request.files['file']
        elif 'audio' in request.files:
            file = request.files['audio']

        if not file:
            return _dep('validation_error_response')('No file provided (expected "file" or "audio" field)')

        if not file.filename:
            return _dep('validation_error_response')('No file selected')

        if not _dep('allowed_file')(file.filename):
            return _dep('validation_error_response')('Invalid file type')

        try:
            require_media_consent(_form_media_consent_payload(), current_app)
        except PermissionError as exc:
            return _dep('validation_error_response')(str(exc))

        if not _ensure_profile_for_song_upload(profile_id):
            return _dep('not_found_response')('Profile not found')

        filename = secure_filename(file.filename)
        job_id = str(uuid.uuid4())
        song_id = str(uuid.uuid4())

        upload_dir = os.path.join(_dep('upload_folder'), 'songs', profile_id)
        os.makedirs(upload_dir, exist_ok=True)

        file_path = os.path.join(upload_dir, f"{song_id}_{filename}")
        file.save(file_path)

        auto_split = request.form.get('auto_split', 'true').lower() == 'true'

        separation_job = {
            'job_id': job_id,
            'job_type': 'song_separation',
            'song_id': song_id,
            'profile_id': profile_id,
            'filename': filename,
            'file_path': file_path,
            'status': 'pending' if auto_split else 'complete',
            'progress': 0,
            'message': 'Queued for vocal separation' if auto_split else 'Uploaded without separation',
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'vocals_path': None,
            'instrumental_path': None,
            'error': None
        }
        _separation_jobs[job_id] = separation_job
        _dep('save_background_job')(separation_job)
        _audit_event(
            "song.uploaded",
            "song",
            song_id,
            {
                "profile_id": profile_id,
                "job_id": job_id,
                "source": "uploaded_song",
                "auto_split": auto_split,
            },
            payload=separation_job,
            asset_paths=[file_path],
        )

        if auto_split:
            app_obj = current_app._get_current_object()

            def run_separation():
                try:
                    _separation_jobs[job_id]['status'] = 'processing'
                    _separation_jobs[job_id]['message'] = 'Separating vocals and instrumental...'
                    _separation_jobs[job_id]['progress'] = 10
                    _dep('save_background_job')(_separation_jobs[job_id])

                    import soundfile as sf
                    from auto_voice.audio.separation import VocalSeparator

                    output_dir = os.path.join(_dep('upload_folder'), 'separated', profile_id, song_id)
                    os.makedirs(output_dir, exist_ok=True)

                    _separation_jobs[job_id]['progress'] = 20
                    _separation_jobs[job_id]['message'] = 'Loading audio...'
                    _dep('save_background_job')(_separation_jobs[job_id])

                    audio, sr = sf.read(file_path)
                    if audio.ndim > 1:
                        audio = audio.T

                    _separation_jobs[job_id]['progress'] = 30
                    _separation_jobs[job_id]['message'] = 'Running vocal separation (this may take a while)...'
                    _dep('save_background_job')(_separation_jobs[job_id])

                    separator = VocalSeparator()
                    result = separator.separate(audio, sr)

                    _separation_jobs[job_id]['progress'] = 80
                    _separation_jobs[job_id]['message'] = 'Saving separated tracks...'
                    _dep('save_background_job')(_separation_jobs[job_id])

                    vocals_path = os.path.join(output_dir, 'vocals.wav')
                    instrumental_path = os.path.join(output_dir, 'instrumental.wav')
                    sf.write(vocals_path, result['vocals'], sr)
                    sf.write(instrumental_path, result['instrumental'], sr)

                    _separation_jobs[job_id]['vocals_path'] = vocals_path
                    _separation_jobs[job_id]['instrumental_path'] = instrumental_path
                    _separation_jobs[job_id]['status'] = 'complete'
                    _separation_jobs[job_id]['progress'] = 100
                    _separation_jobs[job_id]['message'] = 'Separation complete'

                    store = _dep('get_profile_store')()
                    sample = store.add_training_sample(
                        profile_id=profile_id,
                        vocals_path=vocals_path,
                        instrumental_path=instrumental_path,
                        source_file=filename,
                        duration=float(len(result['vocals']) / sr) if sr else 0.0,
                        extra_metadata={
                            'source_song': song_id,
                            'source': 'uploaded_song',
                            'provenance': filename,
                            'separated': True,
                        },
                    )
                    _separation_jobs[job_id]['sample_id'] = sample.sample_id
                    _dep('save_background_job')(_separation_jobs[job_id])
                    record_structured_audit_event(
                        "separate",
                        "song",
                        app=app_obj,
                        resource_id=song_id,
                        asset_paths=[vocals_path, instrumental_path],
                        asset_kind="voice_profile",
                        details={
                            "event_type": "song.separation_completed",
                            "profile_id": profile_id,
                            "job_id": job_id,
                            "song_id": song_id,
                            "sample_id": sample.sample_id,
                            "source": "uploaded_song",
                        },
                    )

                    logger.info(f"Separation complete for song {song_id}, added sample {sample.sample_id}")
                except Exception as e:
                    logger.error(f"Separation failed for job {job_id}: {e}", exc_info=True)
                    _separation_jobs[job_id]['status'] = 'error'
                    _separation_jobs[job_id]['error'] = str(e)
                    _separation_jobs[job_id]['message'] = f'Separation failed: {str(e)}'
                    _dep('save_background_job')(_separation_jobs[job_id])
                    record_structured_audit_event(
                        "separate",
                        "song",
                        app=app_obj,
                        resource_id=song_id,
                        asset_paths=[file_path],
                        asset_kind="voice_profile",
                        details={
                            "event_type": "song.separation_failed",
                            "profile_id": profile_id,
                            "job_id": job_id,
                            "song_id": song_id,
                            "error": str(e),
                        },
                    )

            thread = threading.Thread(target=run_separation, daemon=True)
            thread.start()

        logger.info(f"Uploaded song {song_id} for profile {profile_id}, job {job_id}")
        return jsonify({
            'job_id': job_id,
            'song_id': song_id,
            'status': _separation_jobs[job_id]['status'],
            'message': _separation_jobs[job_id]['message']
        }), 202

    except Exception as e:
        logger.error(f"Error uploading song: {e}", exc_info=True)
        return _dep('error_response')(str(e))


def get_separation_status(job_id: str):
    """Get status of a vocal separation job."""
    job = _separation_jobs.get(job_id) or _dep('get_background_job')(job_id)
    if not job:
        return _dep('not_found_response')('Job not found')
    return jsonify(_redact_profile_payload(job, job.get("profile_id")))


def get_sample(profile_id: str, sample_id: str):
    """Get details of a specific sample."""
    if _dep('load_runtime_profile')(profile_id) is None:
        from auto_voice.profiles.api import get_sample as legacy_get_sample
        return legacy_get_sample(profile_id, sample_id)

    sample = _find_training_sample(profile_id, sample_id)
    if sample is not None:
        return jsonify(_redact_profile_payload(_serialize_training_sample(profile_id, sample), profile_id))

    samples = _profile_samples.get(profile_id, {})
    fallback = samples.get(sample_id)
    if fallback is None:
        return _dep('not_found_response')('Sample not found')
    return jsonify(_redact_profile_payload(fallback, profile_id))


def delete_sample(profile_id: str, sample_id: str):
    """Delete a sample."""
    logger = _dep('logger')
    if _dep('load_runtime_profile')(profile_id) is None:
        from auto_voice.profiles.api import delete_sample as legacy_delete_sample
        return legacy_delete_sample(profile_id, sample_id)

    store = _dep('get_profile_store')()
    sample_payload = None
    sample = _find_training_sample(profile_id, sample_id)
    if sample is not None:
        sample_payload = _serialize_training_sample(profile_id, sample)
    if store.delete_training_sample(profile_id, sample_id):
        _audit_event(
            "training_sample.deleted",
            "training_sample",
            sample_id,
            {"profile_id": profile_id, "sample_id": sample_id},
            payload=sample_payload,
        )
        logger.info(f"Deleted sample {sample_id} from profile {profile_id}")
        return '', 204

    samples = _profile_samples.get(profile_id, {})
    fallback_sample = samples.get(sample_id)
    if fallback_sample is None:
        return _dep('not_found_response')('Sample not found')

    if fallback_sample.get('file_path') and os.path.exists(fallback_sample['file_path']):
        os.remove(fallback_sample['file_path'])

    del _profile_samples[profile_id][sample_id]
    _audit_event(
        "training_sample.deleted",
        "training_sample",
        sample_id,
        {"profile_id": profile_id, "sample_id": sample_id},
        payload=fallback_sample,
    )
    logger.info(f"Deleted fallback sample {sample_id} from profile {profile_id}")
    return '', 204
