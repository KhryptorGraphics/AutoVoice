"""Profile and sample API routes extracted from the legacy web API module."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from typing import Any, Callable, Dict, Optional

from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

_deps: Dict[str, Any] = {}


def register_profile_sample_routes(api_bp: Blueprint, **deps: Any) -> None:
    """Register the profile/sample route family on the shared API blueprint."""
    _deps.update(deps)
    api_bp.add_url_rule('/voice/clone', view_func=clone_voice, methods=['POST'])
    api_bp.add_url_rule('/voice/profiles', view_func=get_voice_profiles, methods=['GET'])
    api_bp.add_url_rule('/voice/profiles/<profile_id>', view_func=get_voice_profile, methods=['GET'])
    api_bp.add_url_rule('/voice/profiles/<profile_id>', view_func=delete_voice_profile, methods=['DELETE'])
    api_bp.add_url_rule('/voice/profiles/<profile_id>/delete', view_func=delete_voice_profile, methods=['POST'])
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
        return jsonify(response_data), 201

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
            _dep('serialize_profile_for_response')(profile)
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

        return jsonify(clean_profile)
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
        deleted = voice_cloner.delete_voice_profile(profile_id)
        if deleted:
            logger.info(f"Voice profile deleted: {profile_id}")
            return jsonify({
                'status': 'success',
                'profile_id': profile_id
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

    return jsonify({
        'profile_id': profile_id,
        'adapters': adapters,
        'selected': selected,
        'count': len(adapters),
    })


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
            return jsonify({
                'profile_id': profile_id,
                'has_model': False,
                'model_type': None,
                'model_path': None,
                'profile_role': serialized_profile['profile_role'],
                'clean_vocal_seconds': serialized_profile['clean_vocal_seconds'],
                'full_model_eligible': serialized_profile['full_model_eligible'],
                'available_artifact_types': [],
                'message': 'No trained model available for this profile',
            }), 404

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

            return jsonify({
                'profile_id': profile_id,
                'has_model': True,
                'model_type': selected_artifact_type,
                'model_path': str(selected_artifact_path) if selected_artifact_path else None,
                'profile_role': serialized_profile['profile_role'],
                'clean_vocal_seconds': serialized_profile['clean_vocal_seconds'],
                'full_model_eligible': serialized_profile['full_model_eligible'],
                'available_artifact_types': available_artifact_types,
                'full_model_path': str(full_model_path) if full_model_path else None,
                'tensorrt_engine_path': (
                    str(tensorrt_engine_path) if tensorrt_engine_path else None
                ),
                'message': message,
            }), 200

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

        return jsonify({
            'profile_id': profile_id,
            'has_model': True,
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
        })

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

        return jsonify({
            'status': 'success',
            'success': True,
            'profile_id': profile_id,
            'selected': _dep('get_frontend_adapter_type')({'selected_adapter': adapter_type}),
            'selected_adapter': adapter_type,
            'adapter_path': adapter_artifact['path'],
        })
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

    return jsonify({
        'profile_id': profile_id,
        'profile_name': profile.get('name', 'Unknown'),
        'adapters': metrics,
        'adapter_count': len(metrics),
        'recommended': next(iter(metrics.keys()), None),
    })


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
            return jsonify(samples)
    except Exception as e:
        logger.warning(f"Failed to get samples from VoiceProfileStore: {e}")

    samples = _profile_samples.get(profile_id, {})
    return jsonify(list(samples.values()))


def upload_sample(profile_id: str):
    """Upload a new training sample for a profile."""
    logger = _dep('logger')
    try:
        try:
            _dep('ensure_profile_in_store')(profile_id)
        except _dep('ProfileNotFoundError'):
            from auto_voice.profiles.api import upload_sample as legacy_upload_sample
            return legacy_upload_sample(profile_id)

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
        return jsonify(sample_payload), 201
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
        skip_separation = data.get('skip_separation', False)

        if not audio_path:
            return _dep('validation_error_response')('audio_path is required')

        if not os.path.exists(audio_path):
            return _dep('not_found_response')(f'File not found: {audio_path}')

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
        return jsonify(sample_payload), 201
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

        if auto_split:
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

                    logger.info(f"Separation complete for song {song_id}, added sample {sample.sample_id}")
                except Exception as e:
                    logger.error(f"Separation failed for job {job_id}: {e}", exc_info=True)
                    _separation_jobs[job_id]['status'] = 'error'
                    _separation_jobs[job_id]['error'] = str(e)
                    _separation_jobs[job_id]['message'] = f'Separation failed: {str(e)}'
                    _dep('save_background_job')(_separation_jobs[job_id])

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
    return jsonify(job)


def get_sample(profile_id: str, sample_id: str):
    """Get details of a specific sample."""
    if _dep('load_runtime_profile')(profile_id) is None:
        from auto_voice.profiles.api import get_sample as legacy_get_sample
        return legacy_get_sample(profile_id, sample_id)

    sample = _find_training_sample(profile_id, sample_id)
    if sample is not None:
        return jsonify(_serialize_training_sample(profile_id, sample))

    samples = _profile_samples.get(profile_id, {})
    fallback = samples.get(sample_id)
    if fallback is None:
        return _dep('not_found_response')('Sample not found')
    return jsonify(fallback)


def delete_sample(profile_id: str, sample_id: str):
    """Delete a sample."""
    logger = _dep('logger')
    if _dep('load_runtime_profile')(profile_id) is None:
        from auto_voice.profiles.api import delete_sample as legacy_delete_sample
        return legacy_delete_sample(profile_id, sample_id)

    store = _dep('get_profile_store')()
    if store.delete_training_sample(profile_id, sample_id):
        logger.info(f"Deleted sample {sample_id} from profile {profile_id}")
        return '', 204

    samples = _profile_samples.get(profile_id, {})
    sample = samples.get(sample_id)
    if sample is None:
        return _dep('not_found_response')('Sample not found')

    if sample.get('file_path') and os.path.exists(sample['file_path']):
        os.remove(sample['file_path'])

    del _profile_samples[profile_id][sample_id]
    logger.info(f"Deleted fallback sample {sample_id} from profile {profile_id}")
    return '', 204
