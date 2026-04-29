"""Runtime, health, device, preset, and model API routes extracted from the legacy API module."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from flask import Blueprint, Response, current_app, jsonify, request

from .security import env_bool, public_deployment_enabled, redact_public_paths


def _root():
    from . import api as api_root

    return api_root


def register_runtime_routes(api_bp: Blueprint) -> None:
    """Register runtime-control and observability routes."""
    api_bp.add_url_rule('/reports/benchmarks/latest', view_func=get_latest_benchmark_dashboard, methods=['GET'])
    api_bp.add_url_rule('/reports/release-evidence/latest', view_func=get_latest_release_evidence, methods=['GET'])
    api_bp.add_url_rule('/health', view_func=health_check, methods=['GET'])
    api_bp.add_url_rule('/pipelines/status', view_func=pipelines_status, methods=['GET'])
    api_bp.add_url_rule('/settings/app', view_func=get_app_settings, methods=['GET'])
    api_bp.add_url_rule('/settings/app', view_func=update_app_settings, methods=['PATCH'])
    api_bp.add_url_rule('/ready', view_func=readiness_check, methods=['GET'])
    api_bp.add_url_rule('/public-commercial/readiness', view_func=public_commercial_readiness, methods=['GET'])
    api_bp.add_url_rule('/metrics', view_func=get_metrics_endpoint, methods=['GET'])
    api_bp.add_url_rule('/gpu/metrics', view_func=gpu_metrics, methods=['GET'])
    api_bp.add_url_rule('/kernels/metrics', view_func=kernel_metrics, methods=['GET'])
    api_bp.add_url_rule('/system/info', view_func=system_info, methods=['GET'])
    api_bp.add_url_rule('/audit/events', view_func=list_audit_events, methods=['GET'])
    api_bp.add_url_rule('/devices/list', view_func=list_devices, methods=['GET'])
    api_bp.add_url_rule('/devices/config', view_func=get_device_config, methods=['GET'])
    api_bp.add_url_rule('/devices/config', view_func=set_device_config, methods=['POST'])
    api_bp.add_url_rule('/presets', view_func=list_presets, methods=['GET'])
    api_bp.add_url_rule('/presets', view_func=create_preset, methods=['POST'])
    api_bp.add_url_rule('/presets/<preset_id>', view_func=get_preset, methods=['GET'])
    api_bp.add_url_rule('/presets/<preset_id>', view_func=update_preset, methods=['PUT', 'PATCH'])
    api_bp.add_url_rule('/presets/<preset_id>', view_func=delete_preset, methods=['DELETE'])
    api_bp.add_url_rule('/models/loaded', view_func=get_loaded_models, methods=['GET'])
    api_bp.add_url_rule('/models/load', view_func=load_model, methods=['POST'])
    api_bp.add_url_rule('/models/unload', view_func=unload_model, methods=['POST'])
    api_bp.add_url_rule('/models/tensorrt/status', view_func=get_tensorrt_status, methods=['GET'])
    api_bp.add_url_rule('/models/tensorrt/rebuild', view_func=rebuild_tensorrt, methods=['POST'])
    api_bp.add_url_rule('/models/tensorrt/build', view_func=build_tensorrt, methods=['POST'])
    api_bp.add_url_rule('/config/separation', view_func=get_separation_config, methods=['GET'])
    api_bp.add_url_rule('/config/separation', view_func=update_separation_config, methods=['POST', 'PATCH'])
    api_bp.add_url_rule('/config/pitch', view_func=get_pitch_config, methods=['GET'])
    api_bp.add_url_rule('/config/pitch', view_func=update_pitch_config, methods=['POST', 'PATCH'])
    api_bp.add_url_rule('/audio/router/config', view_func=get_audio_router_config, methods=['GET'])
    api_bp.add_url_rule('/audio/router/config', view_func=update_audio_router_config, methods=['POST', 'PATCH'])


# Legacy in-process preset mirror kept for compatibility tests and transient
# runtime references. AppStateStore is the canonical source of truth.
_presets: Dict[str, Dict[str, Any]] = {}


_PUBLIC_COMMERCIAL_REQUIREMENTS: tuple[dict[str, str], ...] = (
    {
        "id": "account_auth",
        "bead_id": "AV-3rfd.18.3",
        "title": "Account authentication and authorization",
        "env": "AUTOVOICE_ACCOUNT_AUTH_PROVIDER",
        "reason": "set AUTOVOICE_ACCOUNT_AUTH_PROVIDER to the supported account auth provider",
    },
    {
        "id": "tenant_isolation",
        "bead_id": "AV-3rfd.18.3",
        "title": "Per-user tenant isolation",
        "env": "AUTOVOICE_TENANT_ISOLATION_ENABLED",
        "reason": "set AUTOVOICE_TENANT_ISOLATION_ENABLED=true after isolation tests pass",
    },
    {
        "id": "persistent_quotas",
        "bead_id": "AV-3rfd.18.1",
        "title": "Persistent quotas and abuse controls",
        "env": "AUTOVOICE_QUOTA_BACKEND",
        "reason": "set AUTOVOICE_QUOTA_BACKEND to a persistent backend after quota tests pass",
    },
    {
        "id": "abuse_review",
        "bead_id": "AV-3rfd.18.1",
        "title": "Abuse review workflow",
        "env": "AUTOVOICE_ABUSE_REVIEW_ENABLED",
        "reason": "set AUTOVOICE_ABUSE_REVIEW_ENABLED=true after operator review workflow is active",
    },
    {
        "id": "hosted_evidence",
        "bead_id": "AV-3rfd.18.4",
        "title": "Hosted public-lane release evidence",
        "env": "AUTOVOICE_HOSTED_PUBLIC_EVIDENCE_PATH",
        "reason": "point AUTOVOICE_HOSTED_PUBLIC_EVIDENCE_PATH at current-head hosted evidence",
    },
    {
        "id": "legal_approval",
        "bead_id": "AV-3rfd.18.5",
        "title": "Legal/product approval",
        "env": "AUTOVOICE_LEGAL_APPROVAL_PATH",
        "reason": "point AUTOVOICE_LEGAL_APPROVAL_PATH at signed approval evidence",
    },
    {
        "id": "public_ingress_review",
        "bead_id": "AV-3rfd.18.6",
        "title": "Public ingress threat model and security review",
        "env": "AUTOVOICE_PUBLIC_INGRESS_REVIEW_PATH",
        "reason": "point AUTOVOICE_PUBLIC_INGRESS_REVIEW_PATH at completed threat-model/security-review evidence",
    },
)


def _current_git_sha(root_dir: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root_dir,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _configured_path_exists(env_name: str) -> bool:
    raw_path = os.environ.get(env_name, "").strip()
    return bool(raw_path) and Path(raw_path).expanduser().exists()


def _requirement_satisfied(requirement: dict[str, str]) -> bool:
    env_name = requirement["env"]
    if requirement["id"] == "account_auth":
        provider = os.environ.get(env_name, "").strip().lower()
        return bool(provider) and provider not in {"api-token", "operator-token", "none", "local"}
    if requirement["id"] == "persistent_quotas":
        backend = os.environ.get(env_name, "").strip().lower()
        return bool(backend) and backend not in {"memory", "in-memory", "in_memory", "local", "none"}
    if requirement["id"] in {"tenant_isolation", "abuse_review"}:
        return env_bool(env_name)
    return _configured_path_exists(env_name)


def public_commercial_readiness():
    """Machine-readable public/commercial launch gate.

    This endpoint intentionally tracks public launch blockers separately from
    `/ready`, which is a runtime readiness probe for local/private operation.
    """
    root_dir = Path(__file__).resolve().parents[3]
    blockers: list[dict[str, str]] = []
    satisfied: list[dict[str, str]] = []

    for requirement in _PUBLIC_COMMERCIAL_REQUIREMENTS:
        item = {
            "id": requirement["id"],
            "bead_id": requirement["bead_id"],
            "title": requirement["title"],
            "env": requirement["env"],
            "reason": requirement["reason"],
        }
        if _requirement_satisfied(requirement):
            satisfied.append(item)
        else:
            blockers.append(item)

    ready = not blockers
    payload = {
        "ready": ready,
        "status": "ready" if ready else "blocked",
        "scope": "public_commercial",
        "public_deployment_mode": public_deployment_enabled(current_app),
        "current_git_sha": _current_git_sha(root_dir),
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "blockers": blockers,
        "satisfied": satisfied,
        "closure_rule": (
            "Do not close AV-3rfd.18 until this endpoint is ready and the linked "
            "evidence is archived for the candidate release commit."
        ),
    }
    return jsonify(payload)


def _decorate_evidence_payload(payload: Dict[str, Any], source_path: Path) -> Dict[str, Any]:
    """Add operator-facing provenance fields derived from the real artifact path."""
    root_dir = Path(__file__).resolve().parents[3]
    current_sha = _current_git_sha(root_dir)
    provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
    evidence_sha = payload.get("git_sha") or provenance.get("git_sha")
    decorated = dict(payload)
    decorated["source_path"] = str(source_path.relative_to(root_dir)) if source_path.is_relative_to(root_dir) else str(source_path)
    if current_sha:
        decorated["current_git_sha"] = current_sha
        decorated["current_git_sha_short"] = current_sha[:12]
    if evidence_sha:
        decorated["git_sha"] = evidence_sha
        decorated["git_sha_short"] = str(evidence_sha)[:12]
    if current_sha and evidence_sha:
        decorated["is_stale"] = current_sha != evidence_sha
    return decorated


def get_latest_benchmark_dashboard():
    """Return the latest canonical benchmark dashboard JSON."""
    root = _root()
    dashboard_path = root._reports_dir() / "benchmarks" / "latest" / "benchmark_dashboard.json"
    if not dashboard_path.exists():
        return root.not_found_response('Benchmark dashboard not found')
    payload = json.loads(dashboard_path.read_text(encoding='utf-8'))
    return jsonify(_decorate_evidence_payload(payload, dashboard_path))


def get_latest_release_evidence():
    """Return the latest release evidence JSON."""
    root = _root()
    report_path = root._reports_dir() / "release-evidence" / "latest" / "release_decision.json"
    if not report_path.exists():
        report_path = root._reports_dir() / "benchmarks" / "latest" / "release_evidence.json"
    if not report_path.exists():
        return root.not_found_response('Release evidence not found')
    payload = json.loads(report_path.read_text(encoding='utf-8'))
    return jsonify(_decorate_evidence_payload(payload, report_path))


def list_audit_events():
    """Return structured audit events from the durable local state store."""
    root = _root()
    try:
        limit = request.args.get('limit', default=100, type=int)
        resource_id = request.args.get('resource_id')
        event_type = request.args.get('event_type')
        events = root._get_state_store().list_audit_events(
            resource_id=resource_id,
            event_type=event_type,
            limit=max(1, min(limit or 100, 1000)),
        )
        events = redact_public_paths(events, current_app, root._get_state_store(), kind="audit_asset")
        return jsonify({
            "events": events,
            "count": len(events),
        })
    except Exception as exc:
        root.logger.error("Failed to list audit events: %s", exc, exc_info=True)
        return root.error_response(str(exc))


def health_check():
    """Health check endpoint for liveness/readiness probes."""
    root = _root()
    components = {'api': {'status': 'up'}}
    overall_status = 'healthy'

    if root.TORCH_AVAILABLE:
        components['torch'] = {
            'status': 'up',
            'version': root.torch.__version__,
            'cuda': root.torch.cuda.is_available(),
        }
        if root.torch.cuda.is_available():
            try:
                components['torch']['device'] = root.torch.cuda.get_device_name(0)
            except Exception:
                pass
    else:
        components['torch'] = {'status': 'down', 'cuda': False}
        overall_status = 'degraded'

    voice_cloner = getattr(current_app, 'voice_cloner', None)
    components['voice_cloner'] = {'status': 'up' if voice_cloner else 'down'}
    if not voice_cloner:
        overall_status = 'degraded'

    singing_pipeline = getattr(current_app, 'singing_conversion_pipeline', None)
    components['singing_pipeline'] = {'status': 'up' if singing_pipeline else 'down'}
    if not singing_pipeline:
        overall_status = 'degraded'

    job_manager = getattr(current_app, 'job_manager', None)
    components['job_manager'] = {'status': 'up' if job_manager else 'down'}

    try:
        from ..gpu.cuda_kernels import CUDA_KERNELS_AVAILABLE

        cuda_kernels = CUDA_KERNELS_AVAILABLE
    except ImportError:
        cuda_kernels = False

    return jsonify({
        'status': overall_status,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'components': components,
        'cuda_kernels_available': cuda_kernels,
        'uptime': max(0.0, time.time() - getattr(current_app, 'start_time', time.time())),
        'version': '0.1.0',
    })


def pipelines_status():
    """Get status of all voice conversion pipelines."""
    root = _root()
    if not root.PIPELINE_FACTORY_AVAILABLE:
        root.logger.error("PipelineFactory not available")
        return jsonify({
            'error': 'PipelineFactory unavailable',
            'message': 'Pipeline factory module not loaded',
        }), 503

    try:
        factory = root.PipelineFactory.get_instance()
        pipeline_status = factory.get_status()
        return jsonify({
            'status': 'ok',
            'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'pipelines': pipeline_status,
        }), 200
    except Exception as exc:
        root.logger.error("Error getting pipeline status: %s", exc, exc_info=True)
        return jsonify({
            'error': 'Failed to get pipeline status',
            'message': str(exc),
        }), 503


def get_app_settings():
    """Return durable app-level settings used by the frontend."""
    root = _root()
    try:
        return jsonify(root._normalize_app_settings_payload(root._get_state_store().get_app_settings()))
    except Exception as exc:
        root.logger.error("Error reading app settings: %s", exc, exc_info=True)
        return root.error_response(str(exc))


def update_app_settings():
    """Persist durable app-level settings."""
    root = _root()
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return root.validation_error_response('No JSON object provided')

        updates: Dict[str, Any] = {}
        if 'preferred_pipeline' in data:
            preferred_pipeline = str(data['preferred_pipeline']).strip().lower()
            if preferred_pipeline not in root.LEGACY_PIPELINES:
                return root.validation_error_response(
                    'preferred_pipeline must be one of: realtime, quality'
                )
            updates['preferred_pipeline'] = preferred_pipeline
            if preferred_pipeline == 'realtime':
                updates['preferred_offline_pipeline'] = 'realtime'
                updates['preferred_live_pipeline'] = 'realtime'
            else:
                updates['preferred_offline_pipeline'] = root.CANONICAL_OFFLINE_PIPELINE

        if 'preferred_offline_pipeline' in data:
            preferred_offline_pipeline = str(data['preferred_offline_pipeline']).strip().lower()
            if preferred_offline_pipeline not in root.OFFLINE_PIPELINES:
                return root.validation_error_response(
                    'preferred_offline_pipeline must be one of: realtime, quality, quality_seedvc, quality_shortcut'
                )
            updates['preferred_offline_pipeline'] = preferred_offline_pipeline

        if 'preferred_live_pipeline' in data:
            preferred_live_pipeline = str(data['preferred_live_pipeline']).strip().lower()
            if preferred_live_pipeline not in root.LIVE_PIPELINES:
                return root.validation_error_response(
                    'preferred_live_pipeline must be one of: realtime, realtime_meanvc'
                )
            updates['preferred_live_pipeline'] = preferred_live_pipeline

        if not updates:
            return root.validation_error_response('No supported app settings were provided')

        updates['last_updated'] = root._utcnow_iso()
        settings = root._get_state_store().update_app_settings(updates)
        return jsonify(root._normalize_app_settings_payload(settings))
    except Exception as exc:
        root.logger.error("Error updating app settings: %s", exc, exc_info=True)
        return root.error_response(str(exc))


def readiness_check():
    """Readiness check endpoint for orchestration probes."""
    root = _root()
    ready = True
    components_ready = {}

    if not root.TORCH_AVAILABLE or not root.torch.cuda.is_available():
        ready = False
        components_ready['torch'] = False
    else:
        components_ready['torch'] = True

    voice_cloner = getattr(current_app, 'voice_cloner', None)
    components_ready['voice_cloner'] = voice_cloner is not None
    if not voice_cloner:
        ready = False

    singing_pipeline = getattr(current_app, 'singing_conversion_pipeline', None)
    components_ready['singing_pipeline'] = singing_pipeline is not None
    if not singing_pipeline:
        ready = False

    status_code = 200 if ready else 503
    return jsonify({
        'ready': ready,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'components': components_ready,
    }), status_code


def get_metrics_endpoint():
    """Metrics endpoint for monitoring and dashboards."""
    root = _root()
    accept_header = request.headers.get('Accept', '')
    format_param = request.args.get('format', 'json')

    if format_param == 'prometheus' or 'text/plain' in accept_header:
        try:
            from ..monitoring.prometheus import get_content_type, get_metrics, update_gpu_metrics

            update_gpu_metrics()
            return Response(get_metrics(), mimetype=get_content_type())
        except ImportError:
            return jsonify({
                'error': 'Prometheus metrics not available',
                'message': 'Install prometheus_client to enable metrics export',
            }), 503

    try:
        from ..monitoring.prometheus import get_conversion_analytics

        return jsonify(get_conversion_analytics())
    except Exception as exc:
        root.logger.error("Failed to get conversion analytics: %s", exc, exc_info=True)
        return jsonify({
            'error': 'Failed to retrieve metrics',
            'message': str(exc),
        }), 500


def gpu_metrics():
    """Get GPU utilization and memory metrics."""
    root = _root()
    if not root.TORCH_AVAILABLE or not root.torch.cuda.is_available():
        return jsonify({
            'available': False,
            'device_count': 0,
            'devices': [],
            'message': 'CUDA not available',
        })

    try:
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        devices = []
        for index in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
            except Exception:
                gpu_util = None
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = None
            devices.append({
                'index': index,
                'name': name,
                'memory_total_gb': round(mem_info.total / (1024**3), 2),
                'memory_used_gb': round(mem_info.used / (1024**3), 2),
                'memory_free_gb': round(mem_info.free / (1024**3), 2),
                'utilization_percent': gpu_util,
                'temperature_c': temp,
            })
        pynvml.nvmlShutdown()
        return jsonify({
            'available': True,
            'device_count': device_count,
            'devices': devices,
        })
    except ImportError:
        device_count = root.torch.cuda.device_count()
        devices = []
        for index in range(device_count):
            props = root.torch.cuda.get_device_properties(index)
            mem_allocated = root.torch.cuda.memory_allocated(index)
            mem_reserved = root.torch.cuda.memory_reserved(index)
            devices.append({
                'index': index,
                'name': props.name,
                'memory_total_gb': round(props.total_memory / (1024**3), 2),
                'memory_used_gb': round(mem_allocated / (1024**3), 2),
                'memory_reserved_gb': round(mem_reserved / (1024**3), 2),
                'utilization_percent': None,
                'temperature_c': None,
            })
        return jsonify({
            'available': True,
            'device_count': device_count,
            'devices': devices,
            'note': 'Limited metrics (pynvml not available)',
        })
    except Exception as exc:
        root.logger.warning("GPU metrics partially unavailable: %s", exc)
        device_count = root.torch.cuda.device_count()
        devices = []
        for index in range(device_count):
            props = root.torch.cuda.get_device_properties(index)
            devices.append({
                'index': index,
                'name': props.name,
                'memory_total_gb': round(props.total_memory / (1024**3), 2),
                'utilization_percent': None,
                'temperature_c': None,
            })
        return jsonify({
            'available': True,
            'device_count': device_count,
            'devices': devices,
            'note': f'Some metrics unsupported: {exc}',
        })


def kernel_metrics():
    """Get CUDA kernel performance metrics."""
    root = _root()
    try:
        from ..gpu.cuda_kernels import CUDA_KERNELS_AVAILABLE, get_kernel_metrics

        if CUDA_KERNELS_AVAILABLE:
            metrics = get_kernel_metrics()
            return jsonify(metrics if metrics else [])
    except (ImportError, AttributeError):
        pass

    return jsonify({
        'kernels': [],
        'note': 'Using PyTorch fallbacks - custom CUDA kernel metrics not available',
        'cuda_available': root.TORCH_AVAILABLE and root.torch.cuda.is_available() if root.TORCH_AVAILABLE else False,
    })


def system_info():
    """Get comprehensive system information."""
    root = _root()
    import platform
    import sys

    info = {
        'system': {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_version': sys.version,
        },
        'dependencies': {
            'numpy': root.NUMPY_AVAILABLE,
            'torch': root.TORCH_AVAILABLE,
            'torchaudio': root.TORCHAUDIO_AVAILABLE,
            'soundfile': root.SOUNDFILE_AVAILABLE,
            'librosa': root.LIBROSA_AVAILABLE,
            'noisereduce': root.NOISEREDUCE_AVAILABLE,
        },
    }

    if root.TORCH_AVAILABLE:
        info['torch'] = {
            'version': root.torch.__version__,
            'cuda_available': root.torch.cuda.is_available(),
            'cuda_version': root.torch.version.cuda if root.torch.cuda.is_available() else None,
            'cudnn_version': (
                root.torch.backends.cudnn.version() if root.torch.cuda.is_available() else None
            ),
        }
        if root.torch.cuda.is_available():
            info['torch']['device_name'] = root.torch.cuda.get_device_name(0)
            info['torch']['device_count'] = root.torch.cuda.device_count()

    try:
        from ..gpu.cuda_kernels import CUDA_KERNELS_AVAILABLE

        info['cuda_kernels_available'] = CUDA_KERNELS_AVAILABLE
    except ImportError:
        info['cuda_kernels_available'] = False

    return jsonify(info)


def list_devices():
    """List available audio devices."""
    root = _root()
    try:
        from .audio_router import list_audio_devices

        device_type = request.args.get('type')
        if device_type and device_type not in ('input', 'output'):
            return root.validation_error_response('Invalid type parameter, use "input" or "output"')

        return jsonify(list_audio_devices(device_type=device_type))
    except Exception as exc:
        root.logger.error("Error listing audio devices: %s", exc, exc_info=True)
        return jsonify({
            'error': 'Failed to list audio devices',
            'message': str(exc) if current_app.debug else None,
        }), 500


def get_device_config():
    """Get current audio device configuration."""
    root = _root()
    device_config = root._get_state_store().get_device_config()
    if (
        device_config.get('sample_rate') == root.DEFAULT_DEVICE_CONFIG['sample_rate']
        and current_app.app_config.get('audio', {}).get('sample_rate')
    ):
        device_config = root._get_state_store().update_device_config({
            'sample_rate': current_app.app_config.get('audio', {}).get('sample_rate'),
        })
    current_app._device_config = dict(device_config)
    return jsonify(device_config)


def set_device_config():
    """Set audio device configuration."""
    root = _root()
    try:
        data = request.get_json()
        if data is None:
            return root.validation_error_response('Request body required')

        device_config = root._get_state_store().get_device_config()

        if 'input_device_id' in data:
            input_id = data['input_device_id']
            if input_id is not None and input_id != '':
                from .audio_router import list_audio_devices

                input_devices = list_audio_devices(device_type='input')
                valid_ids = [device['device_id'] for device in input_devices]
                if input_id not in valid_ids:
                    return root.validation_error_response(f'Invalid input device ID: {input_id}')
                device_config['input_device_id'] = str(input_id)
            else:
                device_config['input_device_id'] = None

        if 'output_device_id' in data:
            output_id = data['output_device_id']
            if output_id is not None and output_id != '':
                from .audio_router import list_audio_devices

                output_devices = list_audio_devices(device_type='output')
                valid_ids = [device['device_id'] for device in output_devices]
                if output_id not in valid_ids:
                    return root.validation_error_response(f'Invalid output device ID: {output_id}')
                device_config['output_device_id'] = str(output_id)
            else:
                device_config['output_device_id'] = None

        if 'sample_rate' in data:
            sample_rate = data['sample_rate']
            if isinstance(sample_rate, int) and sample_rate > 0:
                device_config['sample_rate'] = sample_rate
            else:
                return root.validation_error_response('Invalid sample_rate, must be positive integer')

        device_config = root._get_state_store().update_device_config(device_config)
        current_app._device_config = dict(device_config)
        root.logger.info("Device config updated: %s", device_config)
        return jsonify(device_config)
    except Exception as exc:
        root.logger.error("Error setting device config: %s", exc, exc_info=True)
        return jsonify({
            'error': 'Failed to set device configuration',
            'message': str(exc) if current_app.debug else None,
        }), 500


def list_presets():
    """List all user presets."""
    root = _root()
    presets = root._get_state_store().list_presets()
    for preset in presets:
        preset['config'] = root._normalize_preset_config(preset.get('config', {}))
    return jsonify(presets)


def create_preset():
    """Create a new preset."""
    root = _root()
    try:
        data = request.get_json()
        if not data:
            return root.validation_error_response('No JSON data provided')

        name = data.get('name')
        if not name:
            return root.validation_error_response('name is required')

        preset_id = str(root.uuid.uuid4())
        preset = {
            'id': preset_id,
            'name': name,
            'config': root._normalize_preset_config(data.get('config', {})),
            'created_at': root._utcnow_iso(),
            'updated_at': root._utcnow_iso(),
        }
        _presets[preset_id] = preset
        root._get_state_store().save_preset(preset)
        root.logger.info("Created preset %s: %s", preset_id, name)
        return jsonify(preset), 201
    except ValueError as exc:
        return root.validation_error_response(str(exc))
    except Exception as exc:
        root.logger.error("Error creating preset: %s", exc, exc_info=True)
        return root.error_response(str(exc))


def get_preset(preset_id: str):
    """Get a specific preset."""
    root = _root()
    preset = root._get_state_store().get_preset(preset_id)
    if not preset:
        return root.not_found_response('Preset not found')
    preset['config'] = root._normalize_preset_config(preset.get('config', {}))
    return jsonify(preset)


def update_preset(preset_id: str):
    """Update a preset."""
    root = _root()
    preset = root._get_state_store().get_preset(preset_id)
    if not preset:
        return root.not_found_response('Preset not found')

    try:
        data = request.get_json()
        if not data:
            return root.validation_error_response('No JSON data provided')

        if 'name' in data:
            preset['name'] = data['name']
        if 'config' in data:
            preset['config'] = root._normalize_preset_config(data['config'])
        preset['updated_at'] = root._utcnow_iso()

        _presets[preset_id] = preset
        root._get_state_store().save_preset(preset)
        root.logger.info("Updated preset %s", preset_id)
        return jsonify(preset)
    except ValueError as exc:
        return root.validation_error_response(str(exc))
    except Exception as exc:
        root.logger.error("Error updating preset: %s", exc, exc_info=True)
        return root.error_response(str(exc))


def delete_preset(preset_id: str):
    """Delete a preset."""
    root = _root()
    if not root._get_state_store().delete_preset(preset_id):
        return root.not_found_response('Preset not found')
    _presets.pop(preset_id, None)
    root.logger.info("Deleted preset %s", preset_id)
    return '', 204


def get_loaded_models():
    """Get list of currently loaded models."""
    root = _root()
    persisted = {
        entry.get('model_type') or entry.get('type'): entry
        for entry in root._get_state_store().list_loaded_models()
    }

    if root.PIPELINE_FACTORY_AVAILABLE:
        factory = root.PipelineFactory.get_instance()
        for pipeline_type, status in factory.get_status().items():
            if not status.get('loaded'):
                continue
            persisted[pipeline_type] = {
                **persisted.get(pipeline_type, {}),
                'model_type': pipeline_type,
                'type': pipeline_type,
                'name': status.get('description') or pipeline_type,
                'loaded': True,
                'runtime_backend': persisted.get(pipeline_type, {}).get('runtime_backend', 'pytorch'),
                'device': persisted.get(pipeline_type, {}).get('device', root._default_runtime_device()),
                'memory_usage': int((status.get('memory_gb') or 0.0) * 1024 * 1024 * 1024),
                'source': 'pipeline_factory',
                'status': 'loaded',
                'loaded_at': persisted.get(pipeline_type, {}).get('loaded_at'),
            }

    models = [
        root._build_loaded_model_entry(model_type, model_info)
        for model_type, model_info in sorted(persisted.items())
    ]
    return jsonify({'models': models})


def load_model():
    """Load a model."""
    root = _root()
    try:
        data = request.get_json()
        if not data:
            return root.validation_error_response('No JSON data provided')

        model_type = data.get('model_type')
        if not model_type:
            return root.validation_error_response('model_type is required')

        path = data.get('path')
        runtime_backend = data.get('runtime_backend', 'pytorch')
        device = data.get('device', root._default_runtime_device())
        memory_usage = float(data.get('memory_usage', 0.0) or 0.0)

        if model_type in root.PIPELINE_DEFINITIONS:
            if not root.PIPELINE_FACTORY_AVAILABLE:
                return root.service_unavailable_response('Pipeline factory unavailable')
            factory = root.PipelineFactory.get_instance()
            factory.get_pipeline(model_type)
            status = factory.get_status().get(model_type, {})
            runtime_backend = 'pytorch'
            memory_usage = int((status.get('memory_gb') or 0.0) * 1024 * 1024 * 1024)
            path = path or status.get('artifact_path')
            source = 'pipeline_factory'
            display_name = status.get('description') or model_type
        else:
            if path:
                candidate = Path(path)
                if not candidate.exists():
                    return root.not_found_response(f'Model path not found: {path}')
                path = str(candidate)
                if memory_usage <= 0:
                    memory_usage = float(candidate.stat().st_size)
            source = 'registry'
            display_name = data.get('name') or model_type

        model_info = {
            'model_type': model_type,
            'type': model_type,
            'path': path,
            'name': display_name,
            'loaded': True,
            'loaded_at': root._utcnow_iso(),
            'runtime_backend': runtime_backend,
            'device': device,
            'memory_usage': memory_usage,
            'status': 'loaded',
            'source': source,
        }
        root._get_state_store().save_loaded_model(model_type, model_info)
        root.logger.info("Loaded model: %s", model_type)
        return jsonify(root._build_loaded_model_entry(model_type, model_info)), 201
    except Exception as exc:
        root.logger.error("Error loading model: %s", exc, exc_info=True)
        return root.error_response(str(exc))


def unload_model():
    """Unload a model."""
    root = _root()
    try:
        data = request.get_json()
        if not data:
            return root.validation_error_response('No JSON data provided')

        model_type = data.get('model_type')
        if not model_type:
            return root.validation_error_response('model_type is required')

        removed = root._get_state_store().delete_loaded_model(model_type)
        if model_type in root.PIPELINE_DEFINITIONS and root.PIPELINE_FACTORY_AVAILABLE:
            removed = root.PipelineFactory.get_instance().unload_pipeline(model_type) or removed
        if not removed:
            return root.not_found_response('Model not loaded')

        root.logger.info("Unloaded model: %s", model_type)
        return '', 204
    except Exception as exc:
        root.logger.error("Error unloading model: %s", exc, exc_info=True)
        return root.error_response(str(exc))


def get_tensorrt_status():
    """Get TensorRT engine status."""
    root = _root()
    try:
        engines = root._engine_inventory()
        jobs = root._list_background_jobs(job_type='tensorrt_build')
        active_job = next((job for job in jobs if job.get('status') in {'queued', 'running'}), None)
        runtime_available = False
        runtime_version = None
        runtime_error = None
        try:
            from ..export import tensorrt_engine

            runtime_available = bool(tensorrt_engine.TRT_AVAILABLE)
            if runtime_available and tensorrt_engine.trt is not None:
                runtime_version = getattr(tensorrt_engine.trt, "__version__", None)
        except Exception as exc:
            runtime_error = str(exc)

        return jsonify({
            # Backwards compatible engine-inventory flag. New clients should use
            # runtime_available and engines_available to avoid conflating install
            # state with whether optimized engine artifacts have been built.
            'available': len(engines) > 0,
            'runtime_available': runtime_available,
            'runtime_version': runtime_version,
            'runtime_error': runtime_error,
            'engines_available': len(engines) > 0,
            'engines': engines,
            'cuda_available': root.TORCH_AVAILABLE and root.torch.cuda.is_available(),
            'build_in_progress': active_job is not None,
            'active_job': active_job,
            'jobs': jobs[:10],
        })
    except Exception as exc:
        root.logger.error("Error getting TensorRT status: %s", exc, exc_info=True)
        return root.error_response(str(exc))


def rebuild_tensorrt():
    """Rebuild TensorRT engines."""
    root = _root()
    try:
        data = request.get_json() or {}
        precision = data.get('precision', 'fp16')
        models = data.get('models') or [engine['model'] for engine in root._engine_inventory()]
        if not models:
            return root.validation_error_response('No TensorRT models available to rebuild')

        job = root._create_background_job(
            'tensorrt_build',
            {'models': list(models), 'precision': precision, 'force_rebuild': True},
        )
        root._submit_background_job(
            job['job_id'],
            root._run_tensorrt_job,
            models=list(models),
            precision=precision,
            force_rebuild=True,
        )
        root.logger.info("Queued TensorRT rebuild job %s", job['job_id'])
        return jsonify({
            'job_id': job['job_id'],
            'status': 'queued',
            'precision': precision,
            'models': list(models),
        }), 202
    except Exception as exc:
        root.logger.error("Error rebuilding TensorRT: %s", exc, exc_info=True)
        return root.error_response(str(exc))


def build_tensorrt():
    """Build TensorRT engines with options."""
    root = _root()
    try:
        data = request.get_json() or {}
        precision = data.get('precision', 'fp16')
        models = data.get('models', ['encoder', 'decoder', 'vocoder'])
        job = root._create_background_job(
            'tensorrt_build',
            {'models': list(models), 'precision': precision, 'force_rebuild': False},
        )
        root._submit_background_job(
            job['job_id'],
            root._run_tensorrt_job,
            models=list(models),
            precision=precision,
            force_rebuild=False,
        )
        root.logger.info("Queued TensorRT build job %s", job['job_id'])
        return jsonify({
            'job_id': job['job_id'],
            'status': 'queued',
            'precision': precision,
            'models': list(models),
        }), 202
    except Exception as exc:
        root.logger.error("Error building TensorRT: %s", exc, exc_info=True)
        return root.error_response(str(exc))


def get_separation_config():
    """Get vocal separation configuration."""
    root = _root()
    return jsonify(root._load_separation_config())


def update_separation_config():
    """Update vocal separation configuration."""
    root = _root()
    try:
        data = request.get_json()
        if not data:
            return root.validation_error_response('No JSON data provided')

        existing = root._load_separation_config()
        key_map = {
            'model': 'model',
            'stems': 'stems',
            'overlap': 'overlap',
            'segment': 'segment_length',
            'segment_length': 'segment_length',
            'shifts': 'shifts',
            'device': 'device',
        }
        updates = {}
        for key, mapped_key in key_map.items():
            if key in data:
                updates[mapped_key] = data[key]

        config = root._get_state_store().update_separation_config({**existing, **updates})
        root.logger.info("Updated separation config: %s", config)
        return jsonify(config)
    except Exception as exc:
        root.logger.error("Error updating separation config: %s", exc, exc_info=True)
        return root.error_response(str(exc))


def get_pitch_config():
    """Get pitch extraction configuration."""
    root = _root()
    return jsonify(root._load_pitch_config())


def update_pitch_config():
    """Update pitch extraction configuration."""
    root = _root()
    try:
        data = request.get_json()
        if not data:
            return root.validation_error_response('No JSON data provided')

        existing = root._load_pitch_config()
        updates = {}
        for key in ['method', 'hop_length', 'f0_min', 'f0_max', 'threshold', 'use_gpu', 'device']:
            if key in data:
                updates[key] = data[key]
        if 'use_gpu' in data:
            updates['device'] = 'cuda' if data['use_gpu'] else 'cpu'

        config = root._get_state_store().update_pitch_config({**existing, **updates})
        root.logger.info("Updated pitch config: %s", config)
        return jsonify(config)
    except Exception as exc:
        root.logger.error("Error updating pitch config: %s", exc, exc_info=True)
        return root.error_response(str(exc))


def get_audio_router_config():
    """Get audio router configuration."""
    root = _root()
    return jsonify(root._get_state_store().get_audio_router_config())


def update_audio_router_config():
    """Update audio router configuration."""
    root = _root()
    try:
        data = request.get_json()
        if not data:
            return root.validation_error_response('No JSON data provided')

        updates = {}
        for key in [
            'speaker_gain',
            'headphone_gain',
            'voice_gain',
            'instrumental_gain',
            'speaker_enabled',
            'headphone_enabled',
            'speaker_device',
            'headphone_device',
            'sample_rate',
        ]:
            if key in data:
                updates[key] = data[key]

        config = root._get_state_store().update_audio_router_config(updates)
        root.logger.info("Updated audio router config: %s", config)
        return jsonify(config)
    except Exception as exc:
        root.logger.error("Error updating audio router config: %s", exc, exc_info=True)
        return root.error_response(str(exc))
