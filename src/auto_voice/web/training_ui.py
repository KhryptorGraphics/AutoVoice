"""Training-specific API routes for telemetry and live controls."""

from __future__ import annotations

import io
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from flask import Blueprint, jsonify, request, send_file

try:
    import soundfile
    SOUNDFILE_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in dependency validation
    soundfile = None
    SOUNDFILE_AVAILABLE = False

from .api import (
    _get_profile_store,
    _get_training_job_manager,
    _sanitize_job,
    _serialize_training_job,
)
from .utils import error_response, not_found_response, service_unavailable_response, validation_error_response

logger = logging.getLogger(__name__)

training_ui_bp = Blueprint('training_ui', __name__, url_prefix='/api/v1')


def _resolve_preview_sample(job: Any, sample_id: Optional[str] = None) -> Tuple[Optional[Any], Optional[str]]:
    """Return the preferred preview sample object and source path."""
    store = _get_profile_store()
    candidate_ids = []
    if sample_id:
        candidate_ids.append(sample_id)
    candidate_ids.extend(getattr(job, 'sample_ids', []) or [])

    available_samples = list(store.list_training_samples(job.profile_id))
    by_id = {sample.sample_id: sample for sample in available_samples}

    for candidate_id in candidate_ids:
        sample = by_id.get(candidate_id)
        if sample and sample.vocals_path:
            return sample, sample.vocals_path

    for sample in available_samples:
        if sample.vocals_path:
            return sample, sample.vocals_path

    return None, None


@training_ui_bp.route('/training/jobs/<job_id>/pause', methods=['POST'])
def pause_training_job(job_id: str):
    """Pause a running training job."""
    try:
        job_manager = _get_training_job_manager()
        job = job_manager.get_job(job_id)
        if job is None:
            return not_found_response('Training job not found')

        serialized = _sanitize_job(_serialize_training_job(job))
        if serialized['status'] != 'running':
            return validation_error_response(
                f"Cannot pause job in {serialized['status']} state"
            )
        if serialized.get('is_paused'):
            return validation_error_response('Training job is already paused')
        if not job_manager.pause_job(job_id):
            return validation_error_response('Unable to pause training job')

        logger.info("Paused training job %s", job_id)
        return jsonify(_sanitize_job(_serialize_training_job(job_manager.get_job(job_id))))
    except Exception as e:
        logger.error(f"Error pausing training job {job_id}: {e}", exc_info=True)
        return error_response(str(e))


@training_ui_bp.route('/training/jobs/<job_id>/resume', methods=['POST'])
def resume_training_job(job_id: str):
    """Resume a paused training job."""
    try:
        job_manager = _get_training_job_manager()
        job = job_manager.get_job(job_id)
        if job is None:
            return not_found_response('Training job not found')

        serialized = _sanitize_job(_serialize_training_job(job))
        if serialized['status'] != 'running':
            return validation_error_response(
                f"Cannot resume job in {serialized['status']} state"
            )
        if not serialized.get('is_paused'):
            return validation_error_response('Training job is not paused')
        if not job_manager.resume_job(job_id):
            return validation_error_response('Unable to resume training job')

        logger.info("Resumed training job %s", job_id)
        return jsonify(_sanitize_job(_serialize_training_job(job_manager.get_job(job_id))))
    except Exception as e:
        logger.error(f"Error resuming training job {job_id}: {e}", exc_info=True)
        return error_response(str(e))


@training_ui_bp.route('/training/jobs/<job_id>/telemetry', methods=['GET'])
def get_training_telemetry(job_id: str):
    """Return latest runtime telemetry for a training job."""
    try:
        job_manager = _get_training_job_manager()
        job = job_manager.get_job(job_id)
        if job is None:
            return not_found_response('Training job not found')

        sample, sample_path = _resolve_preview_sample(job)
        return jsonify({
            'job': _sanitize_job(_serialize_training_job(job)),
            'runtime_metrics': _sanitize_job(job_manager.get_job_runtime_metrics(job_id)),
            'preview_available': bool(sample_path),
            'preview_sample_id': getattr(sample, 'sample_id', None),
        })
    except Exception as e:
        logger.error(f"Error getting training telemetry for {job_id}: {e}", exc_info=True)
        return error_response(str(e))


@training_ui_bp.route('/training/preview/<job_id>', methods=['POST'])
def generate_training_preview(job_id: str):
    """Return a short preview clip derived from the job's training samples."""
    if not SOUNDFILE_AVAILABLE:
        return service_unavailable_response('soundfile is required for training previews')

    try:
        payload = request.get_json(silent=True) or {}
        sample_id = payload.get('sample_id')
        duration_seconds = float(payload.get('duration_seconds', 4.0))
        offset_seconds = float(payload.get('offset_seconds', 0.0))
        duration_seconds = max(1.0, min(duration_seconds, 12.0))
        offset_seconds = max(0.0, offset_seconds)

        job_manager = _get_training_job_manager()
        job = job_manager.get_job(job_id)
        if job is None:
            return not_found_response('Training job not found')

        sample, sample_path = _resolve_preview_sample(job, sample_id=sample_id)
        if not sample_path:
            return not_found_response('No training sample is available for preview')

        info = soundfile.info(sample_path)
        start_frame = int(offset_seconds * info.samplerate)
        frame_count = int(duration_seconds * info.samplerate)
        audio, sample_rate = soundfile.read(
            sample_path,
            start=start_frame,
            frames=frame_count,
            dtype='float32',
        )
        audio_np = np.asarray(audio, dtype=np.float32)
        if audio_np.size == 0:
            return validation_error_response('Selected training sample has no previewable audio')

        peak = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
        if peak > 0.98:
            audio_np *= 0.98 / peak

        buffer = io.BytesIO()
        soundfile.write(buffer, audio_np, sample_rate, format='WAV')
        buffer.seek(0)

        logger.info(
            "Generated training preview for job %s using sample %s",
            job_id,
            getattr(sample, 'sample_id', 'unknown'),
        )
        return send_file(
            buffer,
            mimetype='audio/wav',
            as_attachment=False,
            download_name=f'{job_id}_preview.wav',
        )
    except ValueError as e:
        return validation_error_response(str(e))
    except Exception as e:
        logger.error(f"Error generating training preview for {job_id}: {e}", exc_info=True)
        return error_response(str(e))
