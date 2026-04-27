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
    _save_diarization_result,
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
from .api_youtube import (
    register_youtube_routes,
    _youtube_downloader,
    get_youtube_downloader,
    list_youtube_history,
    save_youtube_history,
    clear_youtube_history,
    delete_youtube_history_item,
    youtube_info,
    youtube_download,
)
from .api_quality import (
    register_quality_routes,
    identify_speaker,
    audit_loras,
    analyze_conversion,
    compare_methodologies,
    separate_artists,
    batch_separate_artists,
    get_profile_quality_history,
    get_profile_quality_status,
    check_profile_degradation,
    record_quality_metric,
    get_all_profiles_quality,
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
    can_train = clean_profile['profile_role'] == 'target_user' and sample_count > 0
    can_convert = clean_profile['profile_role'] == 'target_user' and clean_profile['has_trained_model']
    clean_profile['readiness'] = {
        'training': {
            'ready': can_train,
            'reason': 'ready' if can_train else (
                'source_artist_profiles_are_not_trainable'
                if clean_profile['profile_role'] == 'source_artist'
                else 'no_trainable_samples'
            ),
            'sample_count': sample_count,
            'clean_vocal_seconds': clean_vocal_seconds,
            'clean_vocal_minutes': round(clean_vocal_seconds / 60.0, 2),
        },
        'conversion': {
            'ready': can_convert,
            'reason': 'ready' if can_convert else (
                'source_artist_profiles_are_not_conversion_targets'
                if clean_profile['profile_role'] == 'source_artist'
                else 'target_profile_not_trained'
            ),
        },
        'live_conversion': {
            'ready': can_convert,
            'reason': 'ready' if can_convert else (
                'source_artist_profiles_are_not_live_targets'
                if clean_profile['profile_role'] == 'source_artist'
                else 'target_profile_not_trained'
            ),
        },
    }
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
    get_state_store=_get_state_store,
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
register_youtube_routes(api_bp)
register_quality_routes(api_bp)
