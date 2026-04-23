"""REST API endpoints for AutoVoice with comprehensive voice synthesis and audio processing"""
import base64
import io
import os
import json
import logging
import time
import uuid
import threading
from datetime import datetime, timezone
from pathlib import Path
from flask import Blueprint, request, jsonify, current_app, send_file
from werkzeug.utils import secure_filename
from typing import Optional, Dict, Any, List
import tempfile

# Graceful imports with fallback
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except (ImportError, OSError):
    torchaudio = None
    TORCHAUDIO_AVAILABLE = False

try:
    import soundfile
    SOUNDFILE_AVAILABLE = True
except ImportError:
    soundfile = None
    SOUNDFILE_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    librosa = None
    LIBROSA_AVAILABLE = False

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    nr = None
    NOISEREDUCE_AVAILABLE = False

# Import voice cloning exceptions
try:
    from ..inference.voice_cloner import InvalidAudioError, InsufficientQualityError, InconsistentSamplesError
    INVALID_AUDIO_ERROR_AVAILABLE = True
except ImportError:
    InvalidAudioError = None
    InsufficientQualityError = None
    InconsistentSamplesError = None
    INVALID_AUDIO_ERROR_AVAILABLE = False

try:
    from ..storage.voice_profiles import ProfileNotFoundError
    PROFILE_NOT_FOUND_ERROR_AVAILABLE = True
except ImportError:
    ProfileNotFoundError = None
    PROFILE_NOT_FOUND_ERROR_AVAILABLE = False

# Import singing conversion exceptions
try:
    from ..inference.singing_conversion_pipeline import SeparationError, ConversionError
    SINGING_CONVERSION_ERRORS_AVAILABLE = True
except ImportError:
    SeparationError = None
    ConversionError = None
    SINGING_CONVERSION_ERRORS_AVAILABLE = False

# Import YouTube downloader
try:
    from ..audio.youtube_downloader import YouTubeDownloader, YouTubeDownloadResult
    YOUTUBE_DOWNLOADER_AVAILABLE = True
except ImportError:
    YouTubeDownloader = None
    YouTubeDownloadResult = None
    YOUTUBE_DOWNLOADER_AVAILABLE = False

# Import PipelineFactory for status endpoint
try:
    from ..inference.pipeline_factory import PipelineFactory
    PIPELINE_FACTORY_AVAILABLE = True
except ImportError:
    PipelineFactory = None
    PIPELINE_FACTORY_AVAILABLE = False

# Import shared utilities
from .utils import (
    allowed_file,
    ALLOWED_AUDIO_EXTENSIONS,
    validation_error_response,
    not_found_response,
    service_unavailable_response,
    error_response
)
from ..storage.paths import (
    resolve_data_dir,
    resolve_profiles_dir,
    resolve_samples_dir,
    resolve_trained_models_dir,
)
from ..runtime_contract import (
    CANONICAL_LIVE_PIPELINE,
    CANONICAL_OFFLINE_PIPELINE,
    LEGACY_PIPELINES,
    LIVE_PIPELINES,
    OFFLINE_PIPELINES,
    PIPELINE_DEFINITIONS,
)
from .offline_realtime import run_offline_realtime_conversion
from .persistence import (
    DEFAULT_DEVICE_CONFIG,
    DEFAULT_PITCH_CONFIG,
    DEFAULT_SEPARATION_CONFIG,
)
from .api_profiles import (
    register_profile_sample_routes,
    _find_training_sample,
    _profile_samples,
    _separation_jobs,
    clone_voice,
    get_voice_profiles,
    get_voice_profile,
    delete_voice_profile,
    get_profile_adapters,
    get_profile_model,
    select_profile_adapter,
    get_adapter_metrics,
    get_profile_training_status,
    list_samples,
    upload_sample,
    add_sample_from_path,
    upload_song,
    get_separation_status,
    get_sample,
    delete_sample,
)
from .api_diarization import (
    register_diarization_routes,
    _diarization_results,
    _segment_assignments,
    _parse_legacy_segment_key,
    _build_diarization_speaker_summary,
    _create_profile_from_diarized_speaker,
    diarize_audio,
    filter_sample,
    set_profile_speaker_embedding,
    get_profile_speaker_embedding,
    assign_diarization_segment,
    get_profile_segments,
    auto_create_profile_from_diarization,
)
from .api_training import (
    register_training_routes,
    _sanitize_job,
    _save_training_job,
    _queue_lora_training_job,
    list_training_jobs,
    create_training_job,
    get_training_job,
    cancel_training_job,
    list_checkpoints,
    rollback_checkpoint,
    delete_checkpoint,
    check_retrain,
    retrain_lora,
)
from .api_conversion import (
    register_conversion_routes,
    _conversion_history,
    list_conversion_workflows,
    create_conversion_workflow,
    get_conversion_workflow,
    resolve_conversion_workflow_match,
    attach_conversion_workflow_training_job,
    convert_from_workflow,
    convert_song,
    get_conversion_status,
    download_converted_audio,
    reassemble_converted_audio,
    cancel_conversion,
    get_conversion_metrics,
    get_conversion_history,
    delete_conversion_record,
    update_conversion_record,
)
from .api_runtime import (
    register_runtime_routes,
    _presets,
    get_latest_benchmark_dashboard,
    get_latest_release_evidence,
    health_check,
    pipelines_status,
    get_app_settings,
    update_app_settings,
    readiness_check,
    get_metrics_endpoint,
    gpu_metrics,
    kernel_metrics,
    system_info,
    list_devices,
    get_device_config,
    set_device_config,
    list_presets,
    create_preset,
    get_preset,
    update_preset,
    delete_preset,
    get_loaded_models,
    load_model,
    unload_model,
    get_tensorrt_status,
    rebuild_tensorrt,
    build_tensorrt,
    get_separation_config,
    update_separation_config,
    get_pitch_config,
    update_pitch_config,
    get_audio_router_config,
    update_audio_router_config,
)

logger = logging.getLogger(__name__)


class _DaemonExecutor:
    def submit(self, fn, *args: Any, **kwargs: Any):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread


_background_executor = _DaemonExecutor()


# Use /api/v1 prefix for versioned API
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

UPLOAD_FOLDER = '/tmp/autovoice_uploads'
ALLOWED_EXTENSIONS = ALLOWED_AUDIO_EXTENSIONS  # Use shared constant
MAX_TEXT_LENGTH = 5000
MAX_AUDIO_DURATION = 600  # 10 minutes


def get_param(data, form_key, settings_key, default, validator=None, type_hint=None):
    value = data.get(settings_key, default) if data else default
    form_value = request.form.get(form_key)
    if form_value is not None:
        value = form_value

    if value is None:
        if validator and not validator(None):
            raise ValueError(f'Invalid value for {form_key}')
        return None
    
    try:
        if type_hint == 'float':
            value = float(value)
        elif type_hint == 'bool':
            value = str(value).lower() in ('true', '1', 'yes', 'on')
        elif type_hint == 'str':
            value = str(value)
        else:
            value = str(value)

        if validator and not validator(value):
            raise ValueError(f'Invalid value for {form_key}')
        return value
    except ValueError as exc:
        if str(exc).startswith('Invalid value for '):
            raise
        raise ValueError(f'Invalid {form_key}')
    except TypeError:
        raise ValueError(f'Invalid {form_key}')


def _normalize_app_settings_payload(raw_settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return app settings with split pipeline defaults and legacy compatibility."""
    settings = dict(raw_settings or {})
    legacy_pipeline = settings.get('preferred_pipeline')
    offline_pipeline = settings.get('preferred_offline_pipeline')
    live_pipeline = settings.get('preferred_live_pipeline')

    if offline_pipeline not in OFFLINE_PIPELINES:
        offline_pipeline = 'realtime' if legacy_pipeline == 'realtime' else CANONICAL_OFFLINE_PIPELINE

    if live_pipeline not in LIVE_PIPELINES:
        live_pipeline = 'realtime' if legacy_pipeline == 'realtime' else CANONICAL_LIVE_PIPELINE

    settings['preferred_offline_pipeline'] = offline_pipeline
    settings['preferred_live_pipeline'] = live_pipeline
    settings['preferred_pipeline'] = (
        'realtime'
        if offline_pipeline == 'realtime' and live_pipeline == 'realtime'
        else 'quality'
    )
    settings.setdefault('last_updated', None)
    return settings


def _validate_offline_pipeline(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in OFFLINE_PIPELINES:
        raise ValueError(
            'pipeline_type must be one of: realtime, quality, quality_seedvc, quality_shortcut'
        )
    return normalized


def _normalize_preset_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize preset config to the canonical offline conversion contract."""
    normalized = dict(config or {})
    pipeline_type = str(normalized.get('pipeline_type', CANONICAL_OFFLINE_PIPELINE)).strip().lower()
    if pipeline_type not in OFFLINE_PIPELINES:
        raise ValueError(
            'Preset pipeline_type must be one of: realtime, quality, quality_seedvc, quality_shortcut'
        )
    normalized['pipeline_type'] = pipeline_type

    if pipeline_type == 'quality_seedvc' and normalized.get('vocoder_type') == 'hifigan':
        raise ValueError('quality_seedvc presets cannot force the HiFiGAN vocoder')

    if pipeline_type == 'quality_shortcut' and normalized.get('preset') == 'studio':
        raise ValueError('quality_shortcut presets do not support the studio preset tier')

    if pipeline_type == 'realtime' and normalized.get('return_stems') is True:
        raise ValueError('Realtime presets cannot request offline stems')

    return normalized


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _coerce_existing_file_path(path_value: Any) -> Optional[str]:
    """Return a normalized existing file path or None."""
    if isinstance(path_value, Path):
        candidate = path_value
    elif isinstance(path_value, str):
        candidate = Path(path_value)
    elif isinstance(path_value, os.PathLike):
        candidate = Path(os.fspath(path_value))
    else:
        return None

    return str(candidate) if candidate.exists() else None


def _get_state_store():
    state_store = getattr(current_app, 'state_store', None)
    if state_store is None:
        raise RuntimeError('Application state store unavailable')
    return state_store


def _get_conversion_workflow_manager():
    manager = getattr(current_app, 'conversion_workflow_manager', None)
    if manager is None:
        from .conversion_workflows import ConversionWorkflowManager

        manager = ConversionWorkflowManager(current_app._get_current_object())
        current_app.conversion_workflow_manager = manager
    return manager


def _default_runtime_device() -> str:
    return 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'


def _default_gpu_enabled() -> bool:
    return bool(TORCH_AVAILABLE and torch.cuda.is_available())


def _reports_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "reports"


def _engine_inventory() -> List[Dict[str, Any]]:
    engines_dir = (Path(__file__).resolve().parents[1] / "export" / "engines").resolve()
    if not engines_dir.exists():
        return []

    inventory: List[Dict[str, Any]] = []
    for engine_path in sorted(engines_dir.iterdir()):
        if engine_path.suffix not in {".engine", ".plan"}:
            continue
        inventory.append({
            "name": engine_path.name,
            "model": engine_path.stem,
            "path": str(engine_path),
            "size": engine_path.stat().st_size,
            "precision": "fp16" if "fp16" in engine_path.stem.lower() else "fp32",
            "built_at": datetime.fromtimestamp(engine_path.stat().st_mtime, timezone.utc).isoformat(),
        })
    return inventory


def _save_background_job(job: Dict[str, Any]) -> Dict[str, Any]:
    state_store = getattr(current_app, "state_store", None)
    if state_store is not None:
        return state_store.save_background_job(job)
    return job


def _get_background_job(job_id: str) -> Optional[Dict[str, Any]]:
    state_store = getattr(current_app, "state_store", None)
    if state_store is not None:
        return state_store.get_background_job(job_id)
    return None


def _list_background_jobs(job_type: Optional[str] = None) -> List[Dict[str, Any]]:
    state_store = getattr(current_app, "state_store", None)
    if state_store is not None:
        return state_store.list_background_jobs(job_type=job_type)
    return []


def _update_background_job(job_id: str, **updates: Any) -> Dict[str, Any]:
    job = dict(_get_background_job(job_id) or {})
    if not job:
        raise KeyError(f"Background job {job_id} not found")
    job.update(updates)
    _save_background_job(job)
    return job


def _create_background_job(job_type: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    job = {
        "job_id": str(uuid.uuid4()),
        "job_type": job_type,
        "status": "queued",
        "progress": 0,
        "created_at": _utcnow_iso(),
        "started_at": None,
        "completed_at": None,
        "error": None,
        "payload": dict(payload or {}),
        "result": None,
    }
    _save_background_job(job)
    return job


def _submit_background_job(job_id: str, runner, *args: Any, **kwargs: Any) -> None:
    app = current_app._get_current_object()

    def _wrapped() -> None:
        with app.app_context():
            _update_background_job(job_id, status="running", started_at=_utcnow_iso(), progress=5)
            try:
                result = runner(job_id, *args, **kwargs)
                _update_background_job(
                    job_id,
                    status="completed",
                    progress=100,
                    completed_at=_utcnow_iso(),
                    result=result,
                )
            except Exception as exc:
                logger.error("Background job %s failed: %s", job_id, exc, exc_info=True)
                _update_background_job(
                    job_id,
                    status="failed",
                    completed_at=_utcnow_iso(),
                    error=str(exc),
                )

    _background_executor.submit(_wrapped)


def _resolve_trt_paths(model_name: str) -> tuple[Path, Path]:
    engines_dir = (Path(__file__).resolve().parents[1] / "export" / "engines").resolve()
    engines_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = engines_dir / f"{model_name}.onnx"
    engine_path = engines_dir / f"{model_name}.engine"
    return onnx_path, engine_path


def _run_tensorrt_job(job_id: str, *, models: List[str], precision: str, force_rebuild: bool) -> Dict[str, Any]:
    try:
        from ..export.tensorrt_engine import TRT_AVAILABLE, TRTEngineBuilder
    except Exception as exc:
        raise RuntimeError(f"TensorRT export support unavailable: {exc}") from exc

    if not TRT_AVAILABLE or TRTEngineBuilder is None:
        raise RuntimeError("TensorRT is not available in this environment")

    builder = TRTEngineBuilder()
    built: List[Dict[str, Any]] = []
    for index, model_name in enumerate(models, start=1):
        onnx_path, engine_path = _resolve_trt_paths(model_name)
        if not onnx_path.exists():
            raise RuntimeError(f"ONNX source not found for {model_name}: {onnx_path}")
        _update_background_job(
            job_id,
            progress=min(95, int((index - 1) / max(len(models), 1) * 100)),
            result={
                "current_model": model_name,
                "models_requested": list(models),
                "precision": precision,
                "force_rebuild": force_rebuild,
                "engines_built": built,
            },
        )
        if force_rebuild:
            engine = builder.build_engine(str(onnx_path), fp16=(precision == "fp16"), int8=(precision == "int8"))
            builder.save_engine(engine, str(engine_path))
        else:
            builder.load_cached_engine(str(onnx_path), str(engine_path), fp16=(precision == "fp16"), int8=(precision == "int8"))
        built.append({
            "model": model_name,
            "onnx_path": str(onnx_path),
            "engine_path": str(engine_path),
            "precision": precision,
        })
    return {
        "models_requested": list(models),
        "engines_built": built,
        "precision": precision,
        "force_rebuild": force_rebuild,
    }


def _build_loaded_model_entry(model_type: str, model_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    info = dict(model_info or {})
    runtime_backend = info.get("runtime_backend", "pytorch")
    memory_usage = float(info.get("memory_usage") or 0.0)
    entry = {
        "type": model_type,
        "model_type": model_type,
        "name": info.get("name") or model_type.replace("_", " ").title(),
        "path": info.get("path"),
        "loaded": bool(info.get("loaded", True)),
        "loaded_at": info.get("loaded_at"),
        "runtime_backend": runtime_backend,
        "device": info.get("device", _default_runtime_device()),
        "memory_usage": memory_usage,
        "memory_mb": round(memory_usage / (1024 * 1024), 2) if memory_usage else 0.0,
        "status": info.get("status", "loaded"),
        "source": info.get("source", "registry"),
        "artifact_path": info.get("artifact_path"),
    }
    if info.get("profile_id"):
        entry["profile_id"] = info["profile_id"]
    return entry


def _load_separation_config() -> Dict[str, Any]:
    config = _get_state_store().get_separation_config()
    if config.get('device') == DEFAULT_SEPARATION_CONFIG['device'] and _default_runtime_device() == 'cuda':
        config['device'] = 'cuda'
    return config


def _load_pitch_config() -> Dict[str, Any]:
    config = _get_state_store().get_pitch_config()
    if (
        config.get('device') == DEFAULT_PITCH_CONFIG['device']
        and config.get('use_gpu') == DEFAULT_PITCH_CONFIG['use_gpu']
        and _default_gpu_enabled()
    ):
        config['use_gpu'] = True
        config['device'] = 'cuda'
    return config


def _get_data_dir() -> Path:
    return resolve_data_dir(current_app.config.get('DATA_DIR'))


def _get_profile_store():
    from ..storage.voice_profiles import VoiceProfileStore

    data_dir = _get_data_dir()
    return VoiceProfileStore(
        profiles_dir=str(resolve_profiles_dir(data_dir=str(data_dir))),
        samples_dir=str(resolve_samples_dir(data_dir=str(data_dir))),
    )


def _get_adapter_manager():
    from ..models.adapter_manager import AdapterManager, AdapterManagerConfig

    data_dir = _get_data_dir()
    return AdapterManager(AdapterManagerConfig(
        adapters_dir=resolve_trained_models_dir(data_dir=str(data_dir)),
        profiles_dir=resolve_profiles_dir(data_dir=str(data_dir)),
    ))


def _get_training_job_manager():
    manager = getattr(current_app, '_training_job_manager', None)
    if manager is not None:
        return manager

    from ..training.job_manager import TrainingJobManager

    data_dir = _get_data_dir()
    socketio = current_app.extensions.get('socketio') or getattr(current_app, 'socketio', None)
    manager = TrainingJobManager(
        storage_path=data_dir / 'app_state',
        require_gpu=False,
        socketio=socketio,
        profiles_dir=str(resolve_profiles_dir(data_dir=str(data_dir))),
        samples_dir=str(resolve_samples_dir(data_dir=str(data_dir))),
    )
    current_app._training_job_manager = manager
    return manager


def _serialize_training_job(job: Any) -> Dict[str, Any]:
    if isinstance(job, dict):
        return job
    if hasattr(job, 'to_dict'):
        return job.to_dict()
    raise TypeError(f'Unsupported training job type: {type(job)}')


def _load_runtime_profile(profile_id: str) -> Optional[Dict[str, Any]]:
    store = _get_profile_store()
    try:
        return store.load(profile_id)
    except Exception:
        pass

    voice_cloner = getattr(current_app, 'voice_cloner', None)
    if voice_cloner is None:
        return None
    try:
        return voice_cloner.load_voice_profile(profile_id)
    except Exception:
        return None


def _ensure_profile_in_store(profile_id: str) -> Dict[str, Any]:
    from ..storage.voice_profiles import ProfileNotFoundError

    store = _get_profile_store()
    try:
        return store.load(profile_id)
    except Exception:
        pass

    profile = _load_runtime_profile(profile_id)
    if profile is None:
        raise ProfileNotFoundError(f'Profile {profile_id} not found')

    store.save(dict(profile))
    return store.load(profile_id)


def _serialize_profile_for_response(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Strip bulky fields and expose the canonical profile workflow contract."""
    clean_profile = {k: v for k, v in profile.items() if k != 'embedding'}
    sample_count = int(
        clean_profile.get('sample_count', clean_profile.get('training_sample_count', 0))
    )
    clean_vocal_seconds = float(
        clean_profile.get('clean_vocal_seconds', clean_profile.get('total_training_duration', 0.0))
    )
    remaining_seconds = float(
        clean_profile.get(
            'full_model_remaining_seconds',
            max(float(clean_profile.get('full_model_unlock_seconds', 1800.0)) - clean_vocal_seconds, 0.0),
        )
    )

    clean_profile['sample_count'] = sample_count
    clean_profile['training_sample_count'] = sample_count
    clean_profile['profile_role'] = clean_profile.get('profile_role', 'target_user')
    clean_profile['created_from'] = clean_profile.get('created_from', 'manual')
    if clean_profile.get('last_trained_at') and not clean_profile.get('last_trained'):
        clean_profile['last_trained'] = clean_profile['last_trained_at']
    clean_profile['clean_vocal_seconds'] = clean_vocal_seconds
    clean_profile['clean_vocal_minutes'] = round(clean_vocal_seconds / 60.0, 2)
    clean_profile['full_model_remaining_seconds'] = remaining_seconds
    clean_profile['full_model_remaining_minutes'] = round(remaining_seconds / 60.0, 2)
    clean_profile['full_model_unlock_seconds'] = float(
        clean_profile.get('full_model_unlock_seconds', 1800.0)
    )
    clean_profile['full_model_eligible'] = bool(
        clean_profile.get('full_model_eligible', remaining_seconds <= 0.0)
    )
    clean_profile['has_trained_model'] = bool(clean_profile.get('has_trained_model'))
    clean_profile['has_full_model'] = bool(clean_profile.get('has_full_model'))
    clean_profile['has_adapter_model'] = bool(clean_profile.get('has_adapter_model'))
    clean_profile['active_model_type'] = clean_profile.get(
        'active_model_type',
        'full_model' if clean_profile['has_full_model'] else (
            'adapter' if clean_profile['has_trained_model'] else 'base'
        ),
    )
    return clean_profile


def _get_frontend_adapter_type(profile: Optional[Dict[str, Any]]) -> str:
    adapter_type = (profile or {}).get('selected_adapter')
    if adapter_type in {'hq', 'nvfp4'}:
        return adapter_type
    return 'hq'


def _get_canonical_adapter_artifact(profile_id: str, profile: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    adapter_manager = _get_adapter_manager()
    adapter_path = adapter_manager.get_adapter_path(profile_id)
    if adapter_path is None:
        return None

    adapter_type = _get_frontend_adapter_type(profile)
    adapter_info = adapter_manager.get_adapter_info(profile_id)
    checkpoint: Dict[str, Any] = {}
    parameter_count = 0

    if TORCH_AVAILABLE:
        try:
            checkpoint = torch.load(adapter_path, map_location='cpu', weights_only=False)
        except Exception as exc:
            logger.warning(f"Failed to load adapter checkpoint metadata for {adapter_path}: {exc}")

    if checkpoint:
        parameter_count = sum(
            tensor.numel()
            for tensor in checkpoint.values()
            if hasattr(tensor, 'numel')
        )

    stat = adapter_path.stat()
    configured_epochs = adapter_info.training_epochs if adapter_info else 0
    epochs = configured_epochs or 0
    if checkpoint and 'epoch' in checkpoint:
        epochs = int(checkpoint.get('epoch', 0)) + 1

    loss = checkpoint.get('loss')
    if loss is None and adapter_info is not None:
        loss = adapter_info.loss_final

    return {
        'type': adapter_type,
        'path': str(adapter_path),
        'size_kb': stat.st_size / 1024,
        'epochs': epochs,
        'loss': loss,
        'precision': checkpoint.get('precision', 'unified'),
        'config': checkpoint.get('config', {}),
        'parameter_count': parameter_count,
        'modified_time': datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        'adapter_info': adapter_info,
    }


register_profile_sample_routes(
    api_bp,
    logger=logger,
    upload_folder=UPLOAD_FOLDER,
    allowed_file=allowed_file,
    validation_error_response=validation_error_response,
    not_found_response=not_found_response,
    service_unavailable_response=service_unavailable_response,
    error_response=error_response,
    get_profile_store=_get_profile_store,
    get_adapter_manager=_get_adapter_manager,
    load_runtime_profile=_load_runtime_profile,
    ensure_profile_in_store=_ensure_profile_in_store,
    serialize_profile_for_response=_serialize_profile_for_response,
    get_frontend_adapter_type=_get_frontend_adapter_type,
    get_canonical_adapter_artifact=_get_canonical_adapter_artifact,
    save_background_job=_save_background_job,
    get_background_job=_get_background_job,
    soundfile_available=SOUNDFILE_AVAILABLE,
    soundfile=soundfile,
    torch_available=TORCH_AVAILABLE,
    torch=torch,
    InvalidAudioError=InvalidAudioError,
    InsufficientQualityError=InsufficientQualityError,
    InconsistentSamplesError=InconsistentSamplesError,
    ProfileNotFoundError=ProfileNotFoundError,
)

register_diarization_routes(
    api_bp,
    logger=logger,
    allowed_file=allowed_file,
    validation_error_response=validation_error_response,
    not_found_response=not_found_response,
    error_response=error_response,
    get_profile_store=_get_profile_store,
    find_training_sample=_find_training_sample,
)

register_training_routes(
    api_bp,
    logger=logger,
    validation_error_response=validation_error_response,
    not_found_response=not_found_response,
    error_response=error_response,
    get_state_store=_get_state_store,
    get_profile_store=_get_profile_store,
    get_training_job_manager=_get_training_job_manager,
    ensure_profile_in_store=_ensure_profile_in_store,
    serialize_training_job=_serialize_training_job,
    queue_lora_training_job=lambda *args, **kwargs: _queue_lora_training_job(*args, **kwargs),
    ProfileNotFoundError=ProfileNotFoundError,
)
register_conversion_routes(api_bp)
register_runtime_routes(api_bp)


# =============================================================================
# YOUTUBE DOWNLOAD ENDPOINTS
# =============================================================================

# Module-level YouTube downloader instance
_youtube_downloader: Optional['YouTubeDownloader'] = None


def get_youtube_downloader() -> 'YouTubeDownloader':
    """Get or create a YouTubeDownloader instance."""
    global _youtube_downloader
    if _youtube_downloader is None:
        if not YOUTUBE_DOWNLOADER_AVAILABLE:
            raise RuntimeError("YouTube downloader not available")
        output_dir = os.path.join(UPLOAD_FOLDER, 'youtube')
        os.makedirs(output_dir, exist_ok=True)
        _youtube_downloader = YouTubeDownloader(output_dir)
    return _youtube_downloader


@api_bp.route('/youtube/history', methods=['GET'])
def list_youtube_history():
    """List persisted YouTube download history."""
    try:
        limit = request.args.get('limit', type=int)
        return jsonify(_get_state_store().list_youtube_history(limit=limit))
    except Exception as e:
        logger.error(f"Failed to list YouTube history: {e}", exc_info=True)
        return error_response(str(e))


@api_bp.route('/youtube/history', methods=['POST'])
def save_youtube_history():
    """Create or update a persisted YouTube download history item."""
    try:
        data = request.get_json()
        if not data:
            return validation_error_response('No JSON data provided')

        history_item = {
            'id': data.get('id') or f"{int(time.time())}-{uuid.uuid4().hex[:8]}",
            'url': data.get('url'),
            'title': data.get('title'),
            'mainArtist': data.get('mainArtist'),
            'featuredArtists': data.get('featuredArtists', []),
            'hasDiarization': bool(data.get('hasDiarization', False)),
            'numSpeakers': int(data.get('numSpeakers', 0)),
            'timestamp': data.get('timestamp') or _utcnow_iso(),
            'audioPath': data.get('audioPath'),
            'filteredPath': data.get('filteredPath'),
            'videoId': data.get('videoId'),
        }
        _get_state_store().save_youtube_history_item(history_item)
        return jsonify(history_item), 201
    except Exception as e:
        logger.error(f"Failed to save YouTube history: {e}", exc_info=True)
        return error_response(str(e))


@api_bp.route('/youtube/history', methods=['DELETE'])
def clear_youtube_history():
    """Clear persisted YouTube download history."""
    try:
        _get_state_store().clear_youtube_history()
        return '', 204
    except Exception as e:
        logger.error(f"Failed to clear YouTube history: {e}", exc_info=True)
        return error_response(str(e))


@api_bp.route('/youtube/history/<item_id>', methods=['DELETE'])
def delete_youtube_history_item(item_id: str):
    """Delete one persisted YouTube history item."""
    try:
        if not _get_state_store().delete_youtube_history_item(item_id):
            return not_found_response('History item not found')
        return '', 204
    except Exception as e:
        logger.error(f"Failed to delete YouTube history item: {e}", exc_info=True)
        return error_response(str(e))


@api_bp.route('/youtube/info', methods=['POST'])
def youtube_info():
    """Get video information and detected artists without downloading.

    Request (JSON):
        url (str): YouTube video URL (required)

    Returns:
        {
            "success": true/false,
            "title": "Video Title",
            "duration": 180.5,
            "main_artist": "Artist Name",
            "featured_artists": ["Artist 1", "Artist 2"],
            "is_cover": false,
            "original_artist": null,
            "song_title": "Song Name",
            "thumbnail_url": "https://...",
            "video_id": "abc123",
            "error": null
        }
    """
    if not YOUTUBE_DOWNLOADER_AVAILABLE:
        return service_unavailable_response('YouTube downloader not available. Install yt-dlp.')

    data = request.get_json()
    if not data or 'url' not in data:
        return validation_error_response('Missing required field: url')

    url = data['url']
    if not url.strip():
        return validation_error_response('URL cannot be empty')

    try:
        downloader = get_youtube_downloader()
        result = downloader.get_video_info(url)

        return jsonify({
            'success': result.success,
            'title': result.title,
            'duration': result.duration,
            'main_artist': result.main_artist,
            'featured_artists': result.featured_artists,
            'is_cover': result.is_cover,
            'original_artist': result.original_artist,
            'song_title': result.song_title,
            'thumbnail_url': result.thumbnail_url,
            'video_id': result.video_id,
            'error': result.error
        })

    except Exception as e:
        logger.error(f"YouTube info failed: {e}")
        return error_response(str(e))


@api_bp.route('/youtube/download', methods=['POST'])
def youtube_download():
    """Download audio from YouTube video with metadata.

    Request (JSON):
        url (str): YouTube video URL (required)
        format (str): Audio format (wav, mp3, flac) - default: wav
        sample_rate (int): Sample rate - default: 44100
        run_diarization (bool): Run speaker diarization after download - default: false

    Returns:
        {
            "success": true/false,
            "audio_path": "/path/to/audio.wav",
            "title": "Video Title",
            "duration": 180.5,
            "main_artist": "Artist Name",
            "featured_artists": ["Artist 1", "Artist 2"],
            "is_cover": false,
            "video_id": "abc123",
            "diarization_result": {...} (if run_diarization=true),
            "error": null
        }
    """
    if not YOUTUBE_DOWNLOADER_AVAILABLE:
        return service_unavailable_response('YouTube downloader not available. Install yt-dlp.')

    data = request.get_json()
    if not data or 'url' not in data:
        return validation_error_response('Missing required field: url')

    url = data['url']
    if not url.strip():
        return validation_error_response('URL cannot be empty')

    audio_format = data.get('format', 'wav')
    if audio_format not in ['wav', 'mp3', 'flac']:
        return validation_error_response('Invalid format. Must be wav, mp3, or flac')

    sample_rate = data.get('sample_rate', 44100)
    try:
        sample_rate = int(sample_rate)
        if sample_rate not in [16000, 22050, 44100, 48000]:
            return validation_error_response('Invalid sample_rate. Must be 16000, 22050, 44100, or 48000')
    except (ValueError, TypeError):
        return validation_error_response('sample_rate must be an integer')

    run_diarization = data.get('run_diarization', False)
    filter_to_main_artist = data.get('filter_to_main_artist', False)

    try:
        downloader = get_youtube_downloader()
        result = downloader.download(url, audio_format=audio_format, sample_rate=sample_rate)

        if not result.success:
            return jsonify({
                'success': False,
                'error': result.error,
                'title': result.title,
                'video_id': result.video_id
            }), 400

        response = {
            'success': True,
            'audio_path': result.audio_path,
            'title': result.title,
            'duration': result.duration,
            'main_artist': result.main_artist,
            'featured_artists': result.featured_artists,
            'is_cover': result.is_cover,
            'original_artist': result.original_artist,
            'song_title': result.song_title,
            'thumbnail_url': result.thumbnail_url,
            'video_id': result.video_id,
            'error': None
        }

        # Optionally run diarization
        if run_diarization and result.audio_path:
            try:
                from ..audio.speaker_diarization import SpeakerDiarizer
                diarizer = SpeakerDiarizer()
                diarization_result = diarizer.diarize(result.audio_path)
                diarization_id = str(uuid.uuid4())
                speaker_durations = {}
                segments = []
                for seg in diarization_result.segments:
                    speaker_durations[seg.speaker_id] = (
                        speaker_durations.get(seg.speaker_id, 0.0) + seg.duration
                    )
                    segments.append({
                        'speaker_id': seg.speaker_id,
                        'start': seg.start,
                        'end': seg.end,
                        'duration': seg.duration,
                    })

                _diarization_results[diarization_id] = {
                    'audio_path': result.audio_path,
                    'audio_duration': result.duration or 0.0,
                    'num_speakers': diarization_result.num_speakers,
                    'segments': segments,
                    'created_at': time.time(),
                    'metadata': {
                        'source': 'youtube_download',
                        'title': result.title,
                        'video_id': result.video_id,
                        'main_artist': result.main_artist,
                        'featured_artists': result.featured_artists,
                        'song_title': result.song_title,
                        'original_artist': result.original_artist,
                        'thumbnail_url': result.thumbnail_url,
                    },
                }

                response['diarization_result'] = {
                    'diarization_id': diarization_id,
                    'num_speakers': diarization_result.num_speakers,
                    'speaker_durations': speaker_durations,
                    'segments': segments,
                }
                response['diarization_id'] = diarization_id
                response['speaker_durations'] = speaker_durations

                # Filter to main artist only if requested
                if filter_to_main_artist and diarization_result.num_speakers > 1:
                    try:
                        from ..audio.training_filter import TrainingDataFilter

                        # Find the dominant speaker (most speaking time = main artist)
                        speaker_durations = {}
                        for seg in diarization_result.segments:
                            speaker_durations[seg.speaker_id] = speaker_durations.get(seg.speaker_id, 0) + seg.duration
                        main_speaker = max(speaker_durations, key=speaker_durations.get)

                        # Extract only the main speaker's segments
                        filtered_path = result.audio_path.replace('.wav', '_filtered.wav')
                        filter_result = diarizer.extract_speaker_audio(
                            result.audio_path,
                            diarization_result.segments,
                            main_speaker,
                            filtered_path
                        )

                        if filter_result and os.path.exists(filtered_path):
                            response['filtered_audio_path'] = filtered_path
                            response['main_speaker_id'] = main_speaker
                            response['filtered_duration'] = speaker_durations[main_speaker]
                            logger.info(f"Filtered audio to main speaker {main_speaker}: {filtered_path}")
                    except Exception as e:
                        logger.warning(f"Failed to filter to main artist: {e}")
                        response['filter_error'] = str(e)

            except Exception as e:
                logger.warning(f"Diarization failed: {e}")
                response['diarization_error'] = str(e)

        try:
            _get_state_store().save_youtube_history_item({
                'id': f"{int(time.time())}-{result.video_id or uuid.uuid4().hex[:8]}",
                'url': url,
                'title': result.title,
                'mainArtist': result.main_artist,
                'featuredArtists': result.featured_artists,
                'hasDiarization': bool(response.get('diarization_result')),
                'numSpeakers': response.get('diarization_result', {}).get('num_speakers', 0),
                'timestamp': _utcnow_iso(),
                'audioPath': result.audio_path,
                'filteredPath': response.get('filtered_audio_path'),
                'videoId': result.video_id,
            })
        except Exception as history_error:
            logger.warning(f"Failed to persist YouTube history: {history_error}")

        return jsonify(response)

    except Exception as e:
        logger.error(f"YouTube download failed: {e}")
        return error_response(str(e))


# ============================================================================
# LoRA Lifecycle Management Endpoints
# Cross-Context: lora-lifecycle-management_20260201
# ============================================================================


@api_bp.route('/audio/identify-speaker', methods=['POST'])
def identify_speaker():
    """Identify speaker from audio by matching against known profiles.

    Cross-Context Dependencies:
    - speaker-diarization_20260130: WavLM embeddings
    - voice-profile-training_20260124: Profile management

    Request body:
    - file: Audio file
    - threshold: Similarity threshold (default: 0.85)

    Returns:
        profile_id: Matched profile ID or null
        profile_name: Matched profile name or null
        similarity: Best similarity score
        is_match: Whether threshold was met
        all_similarities: Scores for all profiles
    """
    try:
        from ..inference.voice_identifier import get_voice_identifier

        # Get audio file
        if 'file' not in request.files:
            return validation_error_response('No audio file provided')

        audio_file = request.files['file']
        if not audio_file.filename:
            return validation_error_response('Empty filename')

        # Get threshold
        threshold = request.form.get('threshold', 0.85)
        try:
            threshold = float(threshold)
        except ValueError:
            threshold = 0.85

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Identify speaker
            identifier = get_voice_identifier()
            result = identifier.identify_from_file(tmp_path, threshold)

            return jsonify({
                'profile_id': result.profile_id,
                'profile_name': result.profile_name,
                'similarity': result.similarity,
                'is_match': result.is_match,
                'all_similarities': result.all_similarities,
            })
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except Exception as e:
        logger.error(f"Speaker identification failed: {e}")
        return error_response(str(e))


@api_bp.route('/loras/audit', methods=['GET'])
def audit_loras():
    """Audit all LoRA adapters across voice profiles.

    Cross-Context Dependencies:
    - training-inference-integration_20260130: AdapterManager
    - speaker-diarization_20260130: Profile embeddings

    Query params:
    - format: 'json' (default) or 'summary'

    Returns:
        audit_timestamp: When audit was run
        summary: Aggregate statistics
        profiles: Per-profile status details
    """
    try:
        from pathlib import Path
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from scripts.audit_loras import LoRAAuditor

        output_format = request.args.get('format', 'json')

        auditor = LoRAAuditor(verbose=False)
        statuses, summary = auditor.audit_all()

        if output_format == 'summary':
            return jsonify({
                'audit_timestamp': None,
                'total_profiles': summary.total_profiles,
                'profiles_with_adapters': summary.profiles_with_adapters,
                'profiles_needing_training': summary.profiles_needing_training,
                'profiles_needing_retrain': summary.profiles_needing_retrain,
                'stale_adapters': summary.stale_adapters,
                'low_quality_adapters': summary.low_quality_adapters,
                'adapter_types': summary.adapter_types,
            })

        # Full JSON output
        from dataclasses import asdict
        from datetime import datetime

        return jsonify({
            'audit_timestamp': datetime.now().isoformat(),
            'summary': asdict(summary),
            'profiles': [asdict(s) for s in statuses],
        })

    except Exception as e:
        logger.error(f"LoRA audit failed: {e}")
        return error_response(str(e))


@api_bp.route('/convert/analyze', methods=['POST'])
def analyze_conversion():
    """Analyze conversion quality with comprehensive metrics.

    Cross-Context Dependencies:
    - sota-dual-pipeline_20260130: Pipeline implementations
    - lora-lifecycle-management_20260201: Quality thresholds

    Request body:
    - source_audio: Path to source audio
    - converted_audio: Path to converted audio
    - target_profile_id: Target profile ID (optional)
    - methodology: Conversion methodology name

    Returns:
        metrics: All quality metrics
        passes_thresholds: Whether all thresholds met
        recommendations: Improvement suggestions
    """
    try:
        from ..evaluation.conversion_quality_analyzer import ConversionQualityAnalyzer

        data = request.json or {}
        source_audio = data.get('source_audio')
        converted_audio = data.get('converted_audio')
        target_profile_id = data.get('target_profile_id')
        methodology = data.get('methodology', 'unknown')

        if not source_audio or not converted_audio:
            return validation_error_response('source_audio and converted_audio required')

        analyzer = ConversionQualityAnalyzer()
        analysis = analyzer.analyze(
            source_audio=source_audio,
            converted_audio=converted_audio,
            target_profile_id=target_profile_id,
            methodology=methodology,
        )

        return jsonify({
            'methodology': analysis.methodology,
            'metrics': analysis.metrics.to_dict(),
            'quality_score': analysis.metrics.quality_score,
            'passes_thresholds': analysis.passes_thresholds,
            'threshold_failures': analysis.threshold_failures,
            'recommendations': analysis.recommendations,
            'timestamp': analysis.timestamp,
        })

    except Exception as e:
        logger.error(f"Conversion analysis failed: {e}")
        return error_response(str(e))


@api_bp.route('/convert/compare-methodologies', methods=['POST'])
def compare_methodologies():
    """Compare conversion quality across multiple methodologies.

    Request body:
    - source_audio: Path to source audio
    - target_profile_id: Target profile ID
    - converted_outputs: Dict mapping methodology -> converted audio path

    Returns:
        best_methodology: Top-ranked methodology
        rankings: Methodology rankings
        analyses: Per-methodology analysis
    """
    try:
        from ..evaluation.conversion_quality_analyzer import ConversionQualityAnalyzer

        data = request.json or {}
        source_audio = data.get('source_audio')
        target_profile_id = data.get('target_profile_id')
        converted_outputs = data.get('converted_outputs', {})

        if not source_audio or not converted_outputs:
            return validation_error_response('source_audio and converted_outputs required')

        analyzer = ConversionQualityAnalyzer()
        comparison = analyzer.compare_methodologies(
            source_audio=source_audio,
            target_profile_id=target_profile_id,
            methodologies=list(converted_outputs.keys()),
            converted_outputs=converted_outputs,
        )

        return jsonify({
            'best_methodology': comparison.best_methodology,
            'rankings': comparison.rankings,
            'summary': comparison.summary,
            'analyses': {
                m: {
                    'metrics': a.metrics.to_dict(),
                    'passes_thresholds': a.passes_thresholds,
                    'threshold_failures': a.threshold_failures,
                }
                for m, a in comparison.analyses.items()
            },
        })

    except Exception as e:
        logger.error(f"Methodology comparison failed: {e}")
        return error_response(str(e))


# =============================================================================
# Phase 5: Multi-Artist Separation API (lora-lifecycle-management_20260201)
# =============================================================================


@api_bp.route('/audio/separate-artists', methods=['POST'])
def separate_artists():
    """Separate multi-artist audio and route to voice profiles.

    Phase 5: Multi-Artist Separation and Profile Routing

    Pipeline:
    1. Demucs vocal/instrumental separation
    2. WavLM speaker diarization
    3. Match segments to known profiles (or create new)
    4. Return organized artist segments

    Request:
        multipart/form-data with 'audio' file
        Optional JSON fields:
        - num_speakers: Expected number of speakers
        - youtube_url: YouTube URL for metadata extraction
        - auto_create_profiles: Create profiles for unknown artists (default: true)

    Returns:
        {
            "artists": {
                "profile_id": {
                    "profile_name": str,
                    "segments": [{"start": float, "end": float, "duration": float}],
                    "total_duration": float
                }
            },
            "num_artists": int,
            "new_profiles_created": [str],
            "instrumental_available": bool
        }
    """
    try:
        from ..audio.multi_artist_separator import MultiArtistSeparator

        # Get audio file
        if 'audio' not in request.files:
            return validation_error_response('No audio file provided')

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return validation_error_response('Empty filename')

        # Get optional parameters
        num_speakers = request.form.get('num_speakers', type=int)
        auto_create = request.form.get('auto_create_profiles', 'true').lower() == 'true'
        youtube_url = request.form.get('youtube_url')

        # Get YouTube metadata if URL provided
        youtube_metadata = None
        if youtube_url and YOUTUBE_DOWNLOADER_AVAILABLE:
            try:
                downloader = YouTubeDownloader()
                youtube_metadata = downloader.get_metadata(youtube_url)
            except Exception as e:
                logger.warning(f"Failed to get YouTube metadata: {e}")

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Load audio
            waveform, sr = torchaudio.load(tmp_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze(0)
            audio = waveform.numpy()

            # Separate and route
            separator = MultiArtistSeparator(auto_create_profiles=auto_create)
            result = separator.separate_and_route(
                audio=audio,
                sample_rate=sr,
                num_speakers=num_speakers,
                youtube_metadata=youtube_metadata,
                source_file=audio_file.filename,
            )

            # Format response
            artists_response = {}
            for profile_id, segments in result.artists.items():
                artists_response[profile_id] = {
                    'profile_name': segments[0].profile_name if segments else profile_id,
                    'segments': [
                        {
                            'start': s.start,
                            'end': s.end,
                            'duration': s.duration,
                            'similarity': s.similarity,
                        }
                        for s in segments
                    ],
                    'total_duration': sum(s.duration for s in segments),
                }

            return jsonify({
                'artists': artists_response,
                'num_artists': result.num_artists,
                'new_profiles_created': result.new_profiles_created,
                'total_duration': result.total_duration,
                'instrumental_available': True,
            })

        finally:
            os.unlink(tmp_path)

    except Exception as e:
        logger.error(f"Multi-artist separation failed: {e}", exc_info=True)
        return error_response(str(e))


@api_bp.route('/audio/batch-separate', methods=['POST'])
def batch_separate_artists():
    """Process multiple audio files (e.g., an album) for multi-artist separation.

    Phase 5: Batch processing with artist aggregation

    Request:
        multipart/form-data with multiple 'audio' files
        Optional:
        - num_speakers: Expected speakers per file

    Returns:
        {
            "files_processed": int,
            "files_successful": int,
            "artists_found": int,
            "artist_summary": {
                "profile_id": {
                    "profile_name": str,
                    "total_segments": int,
                    "total_duration": float
                }
            }
        }
    """
    try:
        from ..audio.multi_artist_separator import MultiArtistSeparator

        files = request.files.getlist('audio')
        if not files:
            return validation_error_response('No audio files provided')

        num_speakers = request.form.get('num_speakers', type=int)

        # Save files to temp directory
        temp_paths = []
        for audio_file in files:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                audio_file.save(tmp.name)
                temp_paths.append(tmp.name)

        try:
            separator = MultiArtistSeparator()
            result = separator.process_batch(
                audio_files=temp_paths,
                num_speakers=num_speakers,
            )

            return jsonify(result)

        finally:
            for path in temp_paths:
                try:
                    os.unlink(path)
                except Exception:
                    pass

    except Exception as e:
        logger.error(f"Batch separation failed: {e}", exc_info=True)
        return error_response(str(e))


# =============================================================================
# Phase 6: Quality Monitoring API (lora-lifecycle-management_20260201)
# =============================================================================


@api_bp.route('/profiles/<profile_id>/quality-history', methods=['GET'])
def get_profile_quality_history(profile_id: str):
    """Get quality metrics history for a profile.

    Phase 6: Quality Monitoring

    Query Parameters:
        - days: Number of days of history (default: 30)
        - limit: Maximum number of records (optional)

    Returns:
        {
            "profile_id": str,
            "period_days": int,
            "total_metrics": int,
            "statistics": {
                "metric_name": {
                    "mean": float,
                    "std": float,
                    "min": float,
                    "max": float
                }
            },
            "metrics": [QualityMetric],
            "recent_alerts": [QualityAlert]
        }
    """
    try:
        from ..monitoring.quality_monitor import get_quality_monitor

        days = request.args.get('days', 30, type=int)

        monitor = get_quality_monitor()
        history = monitor.get_quality_history(profile_id, days=days)

        return jsonify(history)

    except Exception as e:
        logger.error(f"Get quality history failed: {e}", exc_info=True)
        return error_response(str(e))


@api_bp.route('/profiles/<profile_id>/quality-status', methods=['GET'])
def get_profile_quality_status(profile_id: str):
    """Get current quality status for a profile.

    Phase 6: Quality Monitoring

    Returns:
        {
            "profile_id": str,
            "status": "healthy" | "degraded" | "critical",
            "rolling_averages": {
                "speaker_similarity": float,
                "mcd": float,
                ...
            },
            "thresholds": {...},
            "recommendations": [str],
            "unacknowledged_alerts": int
        }
    """
    try:
        from ..monitoring.quality_monitor import get_quality_monitor

        monitor = get_quality_monitor()
        status = monitor.get_quality_summary(profile_id)

        return jsonify(status)

    except Exception as e:
        logger.error(f"Get quality status failed: {e}", exc_info=True)
        return error_response(str(e))


@api_bp.route('/profiles/<profile_id>/check-degradation', methods=['POST'])
def check_profile_degradation(profile_id: str):
    """Explicitly check for quality degradation.

    Phase 6: Quality Monitoring

    Analyzes recent metrics vs historical baseline to detect degradation.

    Returns:
        {
            "profile_id": str,
            "degradation_detected": bool,
            "alerts": [QualityAlert],
            "recommendation": str | null
        }
    """
    try:
        from ..monitoring.quality_monitor import get_quality_monitor

        monitor = get_quality_monitor()
        result = monitor.detect_degradation(profile_id)

        # If degradation detected, optionally trigger retraining
        auto_retrain = request.json.get('auto_retrain', False) if request.json else False

        if result['degradation_detected'] and auto_retrain:
            try:
                job = _get_training_job_manager().auto_queue_training(profile_id)
                if job:
                    result['retrain_job_id'] = job.job_id
                    result['retrain_queued'] = True
                else:
                    result['retrain_queued'] = False
            except Exception as e:
                logger.warning(f"Failed to queue retrain: {e}")
                result['retrain_queued'] = False

        return jsonify(result)

    except Exception as e:
        logger.error(f"Check degradation failed: {e}", exc_info=True)
        return error_response(str(e))


@api_bp.route('/quality/record', methods=['POST'])
def record_quality_metric():
    """Record a quality metric for a profile.

    Phase 6: Quality Monitoring

    Request Body:
        {
            "profile_id": str (required),
            "speaker_similarity": float (optional),
            "mcd": float (optional),
            "f0_correlation": float (optional),
            "rtf": float (optional),
            "mos": float (optional),
            "conversion_id": str (optional)
        }

    Returns:
        {
            "recorded": bool,
            "alerts": [QualityAlert]
        }
    """
    try:
        from ..monitoring.quality_monitor import get_quality_monitor

        data = request.json
        if not data or 'profile_id' not in data:
            return validation_error_response('profile_id required')

        monitor = get_quality_monitor()
        alerts = monitor.record_metric(
            profile_id=data['profile_id'],
            speaker_similarity=data.get('speaker_similarity'),
            mcd=data.get('mcd'),
            f0_correlation=data.get('f0_correlation'),
            rtf=data.get('rtf'),
            mos=data.get('mos'),
            conversion_id=data.get('conversion_id'),
        )

        return jsonify({
            'recorded': True,
            'alerts': [a.to_dict() for a in alerts],
            'alert_count': len(alerts),
        })

    except Exception as e:
        logger.error(f"Record quality metric failed: {e}", exc_info=True)
        return error_response(str(e))


@api_bp.route('/quality/all-profiles', methods=['GET'])
def get_all_profiles_quality():
    """Get quality status for all monitored profiles.

    Phase 6: Quality Monitoring Dashboard

    Returns:
        [
            {
                "profile_id": str,
                "status": str,
                "rolling_averages": {...},
                ...
            }
        ]
    """
    try:
        from ..monitoring.quality_monitor import get_quality_monitor

        monitor = get_quality_monitor()
        profiles = monitor.get_all_profiles_status()

        return jsonify({
            'profiles': profiles,
            'total': len(profiles),
            'degraded_count': sum(1 for p in profiles if p.get('status') == 'degraded'),
            'critical_count': sum(1 for p in profiles if p.get('status') == 'critical'),
        })

    except Exception as e:
        logger.error(f"Get all profiles quality failed: {e}", exc_info=True)
        return error_response(str(e))
