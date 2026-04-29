"""Training lifecycle API routes extracted from the legacy web API module."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from flask import Blueprint, jsonify, request

_deps: Dict[str, Any] = {}


TRAINING_CONFIG_LIMITS: Dict[str, Dict[str, float]] = {
    'lora_rank': {'min': 1, 'max': 64},
    'lora_alpha': {'min': 1, 'max': 256},
    'lora_dropout': {'min': 0.0, 'max': 0.5},
    'learning_rate': {'min': 1e-6, 'max': 1e-2},
    'batch_size': {'min': 1, 'max': 32},
    'epochs': {'min': 1, 'max': 1000},
    'warmup_steps': {'min': 0, 'max': 10000},
    'max_grad_norm': {'min': 0.1, 'max': 10.0},
    'weight_decay': {'min': 0.0, 'max': 1.0},
    'adam_beta1': {'min': 0.0, 'max': 0.999},
    'adam_beta2': {'min': 0.0, 'max': 0.9999},
    'scheduler_gamma': {'min': 0.5, 'max': 1.0},
    'checkpoint_every_steps': {'min': 0, 'max': 100000},
    'validation_split': {'min': 0.0, 'max': 0.5},
    'early_stopping_patience': {'min': 0, 'max': 100},
    'early_stopping_min_delta': {'min': 0.0, 'max': 1.0},
    'ewc_lambda': {'min': 0.0, 'max': 100000.0},
    'prior_loss_weight': {'min': 0.0, 'max': 1.0},
}

TRAINING_CONFIG_ENUMS = {
    'training_mode': ['lora', 'full'],
    'initialization_mode': ['scratch', 'continue'],
    'precision': ['fp32', 'fp16', 'bf16'],
    'optimizer': ['adamw', 'adam'],
    'scheduler': ['exponential', 'none'],
    'lora_target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'content_encoder'],
}

TRAINING_PRESETS = [
    {
        'id': 'fast_lora',
        'label': 'Fast LoRA',
        'description': 'Quick adapter training for small clean vocal sets.',
        'config': {
            'training_mode': 'lora',
            'initialization_mode': 'scratch',
            'lora_rank': 8,
            'lora_alpha': 16,
            'epochs': 50,
            'learning_rate': 1e-4,
            'batch_size': 4,
            'precision': 'fp16',
            'checkpoint_every_steps': 1000,
        },
    },
    {
        'id': 'quality_lora',
        'label': 'Quality LoRA',
        'description': 'Higher-capacity LoRA for production offline conversion quality.',
        'config': {
            'training_mode': 'lora',
            'initialization_mode': 'scratch',
            'lora_rank': 16,
            'lora_alpha': 32,
            'epochs': 120,
            'learning_rate': 7.5e-5,
            'batch_size': 2,
            'precision': 'fp16',
            'checkpoint_every_steps': 500,
        },
    },
    {
        'id': 'continue_lora',
        'label': 'Continue LoRA',
        'description': 'Resume an existing adapter from its latest checkpoint or artifact.',
        'config': {
            'training_mode': 'lora',
            'initialization_mode': 'continue',
            'epochs': 40,
            'learning_rate': 5e-5,
            'checkpoint_every_steps': 500,
        },
    },
    {
        'id': 'full_model_scratch',
        'label': 'Full Model Scratch',
        'description': 'Dedicated target-user model unlocked after 30 clean vocal minutes.',
        'requires_full_training': True,
        'config': {
            'training_mode': 'full',
            'initialization_mode': 'scratch',
            'epochs': 500,
            'learning_rate': 5e-5,
            'batch_size': 1,
            'precision': 'fp16',
            'checkpoint_every_steps': 500,
        },
    },
    {
        'id': 'continue_full_model',
        'label': 'Continue Full Model',
        'description': 'Resume a dedicated full model from checkpoint or artifact.',
        'requires_full_training': True,
        'requires_existing_full_model': True,
        'config': {
            'training_mode': 'full',
            'initialization_mode': 'continue',
            'epochs': 120,
            'learning_rate': 2.5e-5,
            'batch_size': 1,
            'precision': 'fp16',
            'checkpoint_every_steps': 500,
        },
    },
]


def register_training_routes(api_bp: Blueprint, **deps: Any) -> None:
    """Register the training/checkpoint/retrain route family on the shared API blueprint."""
    _deps.update(deps)
    api_bp.add_url_rule('/training/config-options', view_func=get_training_config_options, methods=['GET'])
    api_bp.add_url_rule('/training/jobs', view_func=list_training_jobs, methods=['GET'])
    api_bp.add_url_rule('/training/jobs', view_func=create_training_job, methods=['POST'])
    api_bp.add_url_rule('/training/jobs/<job_id>', view_func=get_training_job, methods=['GET'])
    api_bp.add_url_rule('/training/jobs/<job_id>/cancel', view_func=cancel_training_job, methods=['POST'])
    api_bp.add_url_rule('/profiles/<profile_id>/checkpoints', view_func=list_checkpoints, methods=['GET'])
    api_bp.add_url_rule(
        '/profiles/<profile_id>/checkpoints/<checkpoint_id>/rollback',
        view_func=rollback_checkpoint,
        methods=['POST'],
    )
    api_bp.add_url_rule(
        '/profiles/<profile_id>/checkpoints/<checkpoint_id>',
        view_func=delete_checkpoint,
        methods=['DELETE'],
    )
    api_bp.add_url_rule('/profiles/<profile_id>/check-retrain', view_func=check_retrain, methods=['POST'])
    api_bp.add_url_rule('/loras/retrain/<profile_id>', view_func=retrain_lora, methods=['POST'])


def _dep(name: str) -> Any:
    return _deps[name]


def _sanitize_job(job: dict) -> dict:
    """Sanitize job dict to ensure valid JSON (no Infinity/NaN)."""
    import math

    def sanitize_value(value: Any) -> Any:
        if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
            return None
        if isinstance(value, dict):
            return {key: sanitize_value(val) for key, val in value.items()}
        if isinstance(value, list):
            return [sanitize_value(item) for item in value]
        return value

    return sanitize_value(dict(job))


def _save_training_job(job: dict) -> dict:
    """Persist a training job to the durable state store."""
    sanitized = _sanitize_job(job)
    _dep('get_state_store')().save_training_job(sanitized)
    return sanitized


def _available_training_devices() -> list[dict]:
    devices = [{'id': 'auto', 'label': 'Auto CUDA device', 'available': True}]
    try:
        import torch

        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                devices.append({
                    'id': f'cuda:{idx}',
                    'label': getattr(props, 'name', f'CUDA {idx}'),
                    'available': True,
                    'memory_total_gb': round(float(props.total_memory) / (1024 ** 3), 2),
                })
        else:
            devices[0]['available'] = False
            devices[0]['reason'] = 'cuda_unavailable'
    except Exception as exc:
        devices[0]['available'] = False
        devices[0]['reason'] = str(exc)
    return devices


def get_training_config_options():
    """Return backend-supported training controls, presets, and validation limits."""
    return jsonify({
        'schema_version': 1,
        'defaults': {
            'training_mode': 'lora',
            'initialization_mode': 'scratch',
            'lora_rank': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.1,
            'lora_target_modules': ['q_proj', 'v_proj', 'content_encoder'],
            'learning_rate': 1e-4,
            'batch_size': 4,
            'epochs': 100,
            'warmup_steps': 100,
            'max_grad_norm': 1.0,
            'preset_id': 'custom',
            'device_id': 'auto',
            'precision': 'fp32',
            'optimizer': 'adamw',
            'weight_decay': 0.01,
            'adam_beta1': 0.8,
            'adam_beta2': 0.99,
            'scheduler': 'exponential',
            'scheduler_gamma': 0.999,
            'checkpoint_every_steps': 1000,
            'validation_split': 0.0,
            'early_stopping_patience': 0,
            'early_stopping_min_delta': 0.0,
            'use_ewc': True,
            'ewc_lambda': 1000.0,
            'use_prior_preservation': False,
            'prior_loss_weight': 0.5,
        },
        'limits': TRAINING_CONFIG_LIMITS,
        'enums': TRAINING_CONFIG_ENUMS,
        'presets': TRAINING_PRESETS,
        'devices': _available_training_devices(),
        'full_model_unlock_minutes': 30,
    })


def _normalize_training_config_payload(config_payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(config_payload, dict):
        return None, 'config must be an object'

    normalized = dict(config_payload)
    normalized.setdefault('training_mode', 'lora')
    normalized.setdefault('initialization_mode', 'scratch')
    normalized.setdefault('precision', 'fp32')
    normalized.setdefault('optimizer', 'adamw')
    normalized.setdefault('scheduler', 'exponential')
    normalized.setdefault('device_id', 'auto')
    normalized.setdefault('preset_id', 'custom')

    for key, allowed in TRAINING_CONFIG_ENUMS.items():
        value = normalized.get(key)
        if value is None:
            continue
        if key == 'lora_target_modules':
            if not isinstance(value, list) or any(item not in allowed for item in value):
                return None, f'{key} must be a list containing only: {", ".join(allowed)}'
        elif value not in allowed:
            return None, f'{key} must be one of: {", ".join(allowed)}'

    device_id = normalized.get('device_id', 'auto')
    if device_id != 'auto':
        try:
            import torch

            if not isinstance(device_id, str) or not device_id.startswith('cuda:'):
                return None, 'device_id must be "auto" or a CUDA device like "cuda:0"'
            device_index = int(device_id.split(':', 1)[1])
            if device_index < 0 or device_index >= torch.cuda.device_count():
                return None, f'device_id is unavailable: {device_id}'
        except ValueError:
            return None, 'device_id must be "auto" or a CUDA device like "cuda:0"'
        except Exception as exc:
            return None, f'Unable to validate device_id: {exc}'

    integer_fields = {'lora_rank', 'lora_alpha', 'batch_size', 'epochs', 'warmup_steps', 'checkpoint_every_steps', 'early_stopping_patience'}
    for key, bounds in TRAINING_CONFIG_LIMITS.items():
        if key not in normalized or normalized[key] is None:
            continue
        try:
            value = int(normalized[key]) if key in integer_fields else float(normalized[key])
        except (TypeError, ValueError):
            return None, f'{key} must be numeric'
        if value < bounds['min'] or value > bounds['max']:
            return None, f'{key} must be between {bounds["min"]} and {bounds["max"]}'
        normalized[key] = value

    for key in ('use_ewc', 'use_prior_preservation'):
        if key in normalized:
            normalized[key] = bool(normalized[key])

    return normalized, None


def _queue_lora_training_job(
    profile_id: str,
    *,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    batch_size: int = 4,
):
    """Queue a LoRA training job using the canonical training backend."""
    from ..training.job_manager import TrainingConfig

    store = _dep('get_profile_store')()
    profile = _dep('ensure_profile_in_store')(profile_id)

    if profile.get('profile_role', 'target_user') != 'target_user':
        raise ValueError(
            'Only target user profiles can be trained. '
            'Source artist profiles are reference voices extracted from songs.'
        )

    samples = store.list_training_samples(profile_id)
    sample_ids = [sample.sample_id for sample in samples]
    if not sample_ids:
        raise ValueError('No training samples found for this profile')

    job_manager = _dep('get_training_job_manager')()
    job = job_manager.create_job(
        profile_id=profile_id,
        sample_ids=sample_ids,
        config=TrainingConfig(
            training_mode='lora',
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
        ),
    )
    job_manager.execute_job(job.job_id)
    return job


def list_training_jobs():
    """List all training jobs, optionally filtered by profile."""
    logger = _dep('logger')
    try:
        profile_id = request.args.get('profile_id')
        job_manager = _dep('get_training_job_manager')()
        jobs = [
            _sanitize_job(_dep('serialize_training_job')(job))
            for job in job_manager.list_jobs(profile_id)
        ]
        return jsonify(jobs)
    except Exception as e:
        logger.error(f"Error listing training jobs: {e}", exc_info=True)
        return _dep('error_response')(str(e))


def create_training_job():
    """Create and start a new training job."""
    logger = _dep('logger')
    try:
        data = request.get_json(silent=True)
        if not data:
            return _dep('validation_error_response')('No JSON data provided')

        profile_id = data.get('profile_id')
        if not profile_id:
            return _dep('validation_error_response')('profile_id is required')

        config_payload = data.get('config') or {}
        normalized_config, config_error = _normalize_training_config_payload(config_payload)
        if config_error:
            return _dep('validation_error_response')(config_error)
        assert normalized_config is not None

        store = _dep('get_profile_store')()
        profile = _dep('ensure_profile_in_store')(profile_id)
        available_samples = store.list_training_samples(profile_id)
        if not available_samples:
            return _dep('validation_error_response')('No training samples found for this profile')
        quality_summary = store.get_training_quality_summary(profile_id)

        if profile.get('profile_role', 'target_user') != 'target_user':
            return _dep('validation_error_response')(
                'Only target user profiles can be trained. '
                'Source artist profiles are reference voices extracted from songs.'
            )

        training_mode = normalized_config['training_mode']
        initialization_mode = normalized_config['initialization_mode']
        job_manager = _dep('get_training_job_manager')()

        from ..training.job_manager import TrainingConfig

        if training_mode == 'full':
            eligibility = job_manager.check_needs_full_model(profile_id)
            allow_existing_full_model = (
                eligibility.get('reason') == 'already_full_model'
                and initialization_mode in {'scratch', 'continue'}
            )
            if not eligibility['needs_full_model'] and not allow_existing_full_model:
                minutes = eligibility.get('clean_vocal_seconds', 0.0) / 60.0
                needed_minutes = eligibility.get('remaining_seconds', 0.0) / 60.0
                return _dep('validation_error_response')(
                    'Full model training is locked until this target profile has '
                    f'30 minutes of clean singing vocals. Current clean vocal duration: '
                    f'{minutes:.1f} minutes. Need {needed_minutes:.1f} more minutes.'
                )
            normalized_config.setdefault('epochs', 500)
            normalized_config.setdefault('learning_rate', 5e-5)
            normalized_config['lora_rank'] = 0
            normalized_config['lora_alpha'] = 0

        if initialization_mode == 'continue':
            has_resume_checkpoint = bool(
                getattr(job_manager, '_find_latest_checkpoint', lambda *_args, **_kwargs: None)(
                    profile_id,
                    training_mode,
                )
            )
            if training_mode == 'full' and not (
                profile.get('has_full_model') or has_resume_checkpoint
            ):
                return _dep('validation_error_response')(
                    'No existing full model is available to continue training for this profile'
                )
            if training_mode == 'lora' and not (
                profile.get('has_adapter_model') or has_resume_checkpoint
            ):
                return _dep('validation_error_response')(
                    'No existing LoRA adapter is available to continue training for this profile'
                )

        sample_ids = data.get('sample_ids') or [sample.sample_id for sample in available_samples]
        sample_ids = [sample_id for sample_id in sample_ids if isinstance(sample_id, str)]
        if not sample_ids:
            return _dep('validation_error_response')('At least one valid sample_id is required')

        selected_samples = [
            sample for sample in available_samples if sample.sample_id in sample_ids
        ]
        trainable_samples = [
            sample for sample in selected_samples
            if (sample.quality_metadata or {}).get('qa_status') in {'pass', 'warn'}
        ]
        if not trainable_samples:
            return _dep('validation_error_response')(
                'No selected training samples passed the quality gates. Upload or re-run QA on clean voiced samples before starting training.',
                details=quality_summary,
                recommendations=quality_summary.get('recommendations'),
            )
        filtered_sample_ids = [sample.sample_id for sample in trainable_samples]
        rejected_sample_ids = sorted(set(sample_ids) - set(filtered_sample_ids))

        training_config = TrainingConfig.from_dict(normalized_config)
        if training_mode == 'full':
            job = job_manager.create_full_model_job(
                profile_id=profile_id,
                config=training_config,
                initialization_mode=initialization_mode,
            )
        else:
            job = job_manager.create_job(
                profile_id=profile_id,
                sample_ids=filtered_sample_ids,
                config=training_config,
            )
        job_manager.execute_job(job.job_id)

        serialized = _sanitize_job(_dep('serialize_training_job')(job))
        if rejected_sample_ids:
            serialized['rejected_sample_ids'] = rejected_sample_ids
            serialized['training_quality_summary'] = quality_summary
        logger.info(f"Created training job {job.job_id} for profile {profile_id}")
        return jsonify(serialized), 201
    except Exception as e:
        logger.error(f"Error creating training job: {e}", exc_info=True)
        return _dep('error_response')(str(e))


def get_training_job(job_id: str):
    """Get details of a specific training job."""
    job = _dep('get_training_job_manager')().get_job(job_id)
    if not job:
        return _dep('not_found_response')('Training job not found')
    return jsonify(_sanitize_job(_dep('serialize_training_job')(job)))


def cancel_training_job(job_id: str):
    """Cancel a training job."""
    logger = _dep('logger')
    job_manager = _dep('get_training_job_manager')()
    job = job_manager.get_job(job_id)
    if job is None:
        return _dep('not_found_response')('Training job not found')
    serialized = _dep('serialize_training_job')(job)
    if serialized['status'] in ('completed', 'failed', 'cancelled'):
        return _dep('validation_error_response')(
            f"Cannot cancel job in {serialized['status']} state"
        )
    if not job_manager.cancel_job(job_id):
        return _dep('validation_error_response')(
            f"Cannot cancel job in {serialized['status']} state"
        )
    logger.info(f"Cancelled training job {job_id}")
    return jsonify(_sanitize_job(_dep('serialize_training_job')(job_manager.get_job(job_id))))


def list_checkpoints(profile_id: str):
    """List all checkpoints for a profile."""
    return jsonify(_dep('get_state_store')().list_checkpoints(profile_id))


def rollback_checkpoint(profile_id: str, checkpoint_id: str):
    """Rollback to a specific checkpoint."""
    logger = _dep('logger')
    checkpoint = _dep('get_state_store')().get_checkpoint(profile_id, checkpoint_id)
    if not checkpoint:
        return _dep('not_found_response')('Checkpoint not found')
    try:
        store = _dep('get_profile_store')()
        profile = store.load(profile_id)
    except _dep('ProfileNotFoundError'):
        return _dep('not_found_response')('Profile not found')

    profile_updates = dict(checkpoint.get('profile_snapshot') or {})
    profile_updates['profile_id'] = profile_id
    profile_updates['model_version'] = (
        checkpoint.get('version') or checkpoint.get('model_version') or checkpoint_id
    )
    if checkpoint.get('active_model_type'):
        profile_updates['active_model_type'] = checkpoint['active_model_type']
    if checkpoint.get('selected_adapter'):
        profile_updates['selected_adapter'] = checkpoint['selected_adapter']
    profile.update(profile_updates)
    store.save(profile)

    state_store = _dep('get_state_store')()
    for entry in state_store.list_checkpoints(profile_id):
        updated_entry = dict(entry)
        updated_entry['is_active'] = entry.get('id') == checkpoint_id
        state_store.save_checkpoint(profile_id, updated_entry)

    for loaded_model in state_store.list_loaded_models():
        if loaded_model.get('profile_id') == profile_id:
            state_store.delete_loaded_model(loaded_model.get('model_type') or loaded_model.get('type'))

    logger.info(f"Rolled back profile {profile_id} to checkpoint {checkpoint_id}")
    return jsonify({
        'status': 'rolled_back',
        'checkpoint': {
            **checkpoint,
            'is_active': True,
        },
        'profile_id': profile_id,
        'active_model_type': profile.get('active_model_type'),
        'model_version': profile.get('model_version'),
    })


def delete_checkpoint(profile_id: str, checkpoint_id: str):
    """Delete a checkpoint."""
    logger = _dep('logger')
    if not _dep('get_state_store')().delete_checkpoint(profile_id, checkpoint_id):
        return _dep('not_found_response')('Checkpoint not found')
    logger.info(f"Deleted checkpoint {checkpoint_id} from profile {profile_id}")
    return '', 204


def check_retrain(profile_id: str):
    """Check if a profile needs retraining and optionally trigger it."""
    logger = _dep('logger')
    try:
        from pathlib import Path
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from scripts.audit_loras import LoRAAuditor

        trigger = request.json.get('trigger', False) if request.is_json else False

        auditor = LoRAAuditor(verbose=False)
        statuses, _ = auditor.audit_all()

        profile_status = None
        for status in statuses:
            if status.profile_id == profile_id:
                profile_status = status
                break

        if not profile_status:
            return _dep('not_found_response')(f'Profile {profile_id} not found')

        needs_retrain = profile_status.needs_retrain or profile_status.needs_training
        reasons = profile_status.issues + profile_status.recommendations

        result = {
            'profile_id': profile_id,
            'needs_retrain': needs_retrain,
            'needs_initial_training': profile_status.needs_training,
            'is_stale': profile_status.is_stale,
            'quality_ok': profile_status.quality_ok,
            'sample_count': profile_status.sample_count,
            'reasons': reasons,
            'training_queued': False,
        }

        if trigger and needs_retrain:
            try:
                job = _dep('queue_lora_training_job')(
                    profile_id=profile_id,
                    epochs=100,
                    batch_size=4,
                    learning_rate=1e-4,
                )

                result['training_queued'] = True
                result['job_id'] = job.job_id
                result['job_status'] = job.status

            except Exception as e:
                logger.warning(f"Failed to queue training for {profile_id}: {e}")
                result['training_error'] = str(e)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Check retrain failed: {e}")
        return _dep('error_response')(str(e))


def retrain_lora(profile_id):
    """Queue LoRA retraining for a profile."""
    logger = _dep('logger')
    try:
        data = request.json or {}
        epochs = data.get('epochs', 100)
        learning_rate = data.get('learning_rate', 1e-4)
        batch_size = data.get('batch_size', 4)
        job = _dep('queue_lora_training_job')(
            profile_id=profile_id,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        return jsonify({
            'job_id': job.job_id,
            'status': job.status,
            'profile_id': profile_id,
            'epochs': epochs,
            'message': f'Retraining queued for profile {profile_id}',
        })

    except Exception as e:
        logger.error(f"Retrain LoRA failed: {e}")
        return _dep('error_response')(str(e))
