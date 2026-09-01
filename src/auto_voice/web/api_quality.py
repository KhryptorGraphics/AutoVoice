"""Quality-analysis, utility, and monitoring API routes extracted from the legacy API module."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

from flask import Blueprint, current_app, jsonify, request


def _root():
    from . import api as api_root

    return api_root


def register_quality_routes(api_bp: Blueprint) -> None:
    """Register quality-analysis, utility, and monitoring routes."""
    api_bp.add_url_rule('/audio/identify-speaker', view_func=identify_speaker, methods=['POST'])
    api_bp.add_url_rule('/loras/audit', view_func=audit_loras, methods=['GET'])
    api_bp.add_url_rule('/convert/analyze', view_func=analyze_conversion, methods=['POST'])
    api_bp.add_url_rule('/convert/compare-methodologies', view_func=compare_methodologies, methods=['POST'])
    api_bp.add_url_rule('/audio/separate-artists', view_func=separate_artists, methods=['POST'])
    api_bp.add_url_rule('/audio/batch-separate', view_func=batch_separate_artists, methods=['POST'])
    api_bp.add_url_rule('/profiles/<profile_id>/quality-history', view_func=get_profile_quality_history, methods=['GET'])
    api_bp.add_url_rule('/profiles/<profile_id>/quality-status', view_func=get_profile_quality_status, methods=['GET'])
    api_bp.add_url_rule('/profiles/<profile_id>/check-degradation', view_func=check_profile_degradation, methods=['POST'])
    api_bp.add_url_rule('/quality/record', view_func=record_quality_metric, methods=['POST'])
    api_bp.add_url_rule('/quality/all-profiles', view_func=get_all_profiles_quality, methods=['GET'])
    api_bp.add_url_rule('/quality/conversion-options', view_func=list_quality_conversion_options, methods=['GET'])
    api_bp.add_url_rule('/quality/conversion-analysis', view_func=analyze_conversion_record, methods=['POST'])
    api_bp.add_url_rule('/quality/conversion-comparison', view_func=compare_conversion_records, methods=['POST'])


_COMPLETED_CONVERSION_STATUSES = {'completed', 'complete', 'success'}


def _basename(value: Any, fallback: str = 'Unknown source') -> str:
    if value in (None, ''):
        return fallback
    name = Path(str(value)).name
    return name or str(value) or fallback


def _source_id(record: dict[str, Any]) -> str:
    asset_id = record.get('original_audio_asset_id')
    if asset_id:
        return f"asset:{asset_id}"
    raw_source = record.get('input_file') or record.get('originalFileName') or record.get('id') or 'unknown'
    digest = hashlib.sha256(str(raw_source).encode('utf-8')).hexdigest()[:16]
    return f"input:{digest}"


def _source_label(root, record: dict[str, Any]) -> str:
    asset_id = record.get('original_audio_asset_id')
    if asset_id:
        try:
            asset = root._get_state_store().get_asset(str(asset_id))
            metadata = dict((asset or {}).get('metadata') or {})
            return (
                metadata.get('title')
                or metadata.get('label')
                or metadata.get('filename')
                or _basename((asset or {}).get('path'))
            )
        except Exception:
            pass
    return _basename(record.get('originalFileName') or record.get('input_file'))


def _profile_name(root, profile_id: Any) -> str | None:
    if not profile_id:
        return None
    try:
        profile = root._get_profile_store().load(str(profile_id))
        if profile:
            return profile.get('name') or profile.get('profile_id')
    except Exception:
        return None
    return None


def _record_methodology(record: dict[str, Any]) -> str:
    model = record.get('active_model_type') or 'model'
    adapter = record.get('adapter_type')
    pipeline = record.get('resolved_pipeline') or record.get('pipeline_type') or record.get('preset') or 'conversion'
    parts = [str(pipeline), str(model)]
    if adapter:
        parts.append(str(adapter))
    record_id = str(record.get('id') or '')
    if record_id:
        parts.append(record_id[:8])
    return ' · '.join(parts)


def _resolve_record_source_path(root, record: dict[str, Any]) -> str | None:
    asset_id = record.get('original_audio_asset_id')
    if asset_id:
        try:
            asset = root._get_state_store().get_asset(str(asset_id))
            resolved = root._coerce_existing_file_path((asset or {}).get('path'))
            if resolved:
                return resolved
        except Exception:
            return None
    return root._coerce_existing_file_path(record.get('input_file'))


def _resolve_record_output_path(root, record: dict[str, Any]) -> str | None:
    record_id = str(record.get('id') or '')
    if not record_id:
        return None
    job_manager = getattr(current_app, 'job_manager', None)
    if job_manager is not None and hasattr(job_manager, 'get_job_asset_path'):
        resolved = root._coerce_existing_file_path(job_manager.get_job_asset_path(record_id, 'mix'))
        if resolved:
            return resolved
    resolved = root._coerce_existing_file_path(record.get('result_path'))
    if resolved:
        return resolved
    state_store = root._get_state_store()
    base_dir = getattr(state_store, 'data_dir', None) or 'data'
    return root._coerce_existing_file_path(Path(str(base_dir)) / 'conversions' / record_id / 'mix.wav')


def _quality_records(root) -> list[dict[str, Any]]:
    records = root._get_state_store().list_conversion_history()
    return [
        record
        for record in records
        if str(record.get('status') or '').lower() in _COMPLETED_CONVERSION_STATUSES
        and _resolve_record_source_path(root, record)
        and _resolve_record_output_path(root, record)
    ]


def _serialize_quality_conversion(root, record: dict[str, Any]) -> dict[str, Any]:
    profile_id = record.get('profile_id')
    completed_at = record.get('completed_at') or record.get('timestamp') or record.get('created_at')
    source = _source_id(record)
    methodology = _record_methodology(record)
    label_bits = [
        _source_label(root, record),
        _profile_name(root, profile_id) or str(profile_id or 'unknown profile'),
        methodology,
    ]
    metrics = record.get('quality_metrics') if isinstance(record.get('quality_metrics'), dict) else {}
    quality_score = record.get('quality_score')
    if quality_score is None:
        quality_score = metrics.get('quality_score')
    speaker_similarity = record.get('speaker_similarity')
    if speaker_similarity is None:
        speaker_similarity = metrics.get('speaker_similarity')
    return {
        'id': record.get('id'),
        'source_id': source,
        'source_label': _source_label(root, record),
        'label': ' / '.join(str(bit) for bit in label_bits if bit),
        'methodology': methodology,
        'profile_id': profile_id,
        'profile_name': _profile_name(root, profile_id),
        'status': record.get('status'),
        'completed_at': completed_at,
        'duration': record.get('duration') or record.get('audio_duration_seconds'),
        'rtf': record.get('rtf'),
        'active_model_type': record.get('active_model_type'),
        'adapter_type': record.get('adapter_type'),
        'pipeline_type': record.get('pipeline_type'),
        'resolved_pipeline': record.get('resolved_pipeline'),
        'runtime_backend': record.get('runtime_backend'),
        'quality_score': quality_score,
        'speaker_similarity': speaker_similarity,
        'preset': record.get('preset'),
    }


def _get_quality_record(root, record_id: str) -> dict[str, Any] | None:
    record = root._get_state_store().get_conversion_record(record_id)
    if not record:
        return None
    status = str(record.get('status') or '').lower()
    if status not in _COMPLETED_CONVERSION_STATUSES:
        return None
    if not _resolve_record_source_path(root, record) or not _resolve_record_output_path(root, record):
        return None
    return record

def list_quality_conversion_options():
    """Return path-free conversion records usable by the Quality tab."""
    root = _root()
    try:
        records = _quality_records(root)
        sources: dict[str, dict[str, Any]] = {}
        conversions = []

        for record in records:
            serialized = _serialize_quality_conversion(root, record)
            conversions.append(serialized)
            source = sources.setdefault(
                serialized['source_id'],
                {
                    'id': serialized['source_id'],
                    'label': serialized['source_label'],
                    'conversions': [],
                },
            )
            source['conversions'].append(serialized)

        return jsonify({
            'sources': list(sources.values()),
            'conversions': conversions,
        })
    except Exception as exc:
        root.logger.error("Quality conversion options failed: %s", exc)
        return root.error_response(str(exc))


def analyze_conversion_record():
    """Analyze a completed conversion selected by ID instead of raw file path."""
    root = _root()
    try:
        from ..evaluation.conversion_quality_analyzer import ConversionQualityAnalyzer

        data = request.json or {}
        conversion_id = data.get('conversion_id')
        if not conversion_id:
            return root.validation_error_response('conversion_id is required')

        record = _get_quality_record(root, str(conversion_id))
        if not record:
            return root.not_found_response('Conversion record not found or unavailable')

        source_audio = _resolve_record_source_path(root, record)
        converted_audio = _resolve_record_output_path(root, record)
        if not source_audio or not converted_audio:
            return root.not_found_response('Conversion audio artifact not found')

        analyzer = ConversionQualityAnalyzer()
        analysis = analyzer.analyze(
            source_audio=source_audio,
            converted_audio=converted_audio,
            target_profile_id=data.get('target_profile_id') or record.get('profile_id'),
            methodology=data.get('methodology') or _record_methodology(record),
        )

        return jsonify({
            'conversion': _serialize_quality_conversion(root, record),
            'methodology': analysis.methodology,
            'metrics': analysis.metrics.to_dict(),
            'quality_score': analysis.metrics.quality_score,
            'passes_thresholds': analysis.passes_thresholds,
            'threshold_failures': analysis.threshold_failures,
            'recommendations': analysis.recommendations,
            'timestamp': analysis.timestamp,
        })
    except Exception as exc:
        root.logger.error("Conversion record analysis failed: %s", exc)
        return root.error_response(str(exc))


def compare_conversion_records():
    """Compare completed conversion records selected by ID."""
    root = _root()
    try:
        from ..evaluation.conversion_quality_analyzer import ConversionQualityAnalyzer

        data = request.json or {}
        conversion_ids = [str(item) for item in data.get('conversion_ids', []) if item]
        source_id = data.get('source_id')

        if len(conversion_ids) < 2:
            return root.validation_error_response('at least two conversion_ids are required')

        records = []
        for conversion_id in conversion_ids:
            record = _get_quality_record(root, conversion_id)
            if not record:
                return root.not_found_response(f'Conversion record {conversion_id} not found or unavailable')
            records.append(record)

        source_ids = {_source_id(record) for record in records}
        if source_id and source_ids != {source_id}:
            return root.validation_error_response('selected conversions must match source_id')
        if len(source_ids) != 1:
            return root.validation_error_response('selected conversions must share one source')

        source_audio = _resolve_record_source_path(root, records[0])
        if not source_audio:
            return root.not_found_response('Source audio artifact not found')

        converted_outputs: dict[str, str] = {}
        methodology_to_record: dict[str, str] = {}
        for record in records:
            output_path = _resolve_record_output_path(root, record)
            if not output_path:
                return root.not_found_response(f'Conversion artifact {record.get("id")} not found')
            methodology = _record_methodology(record)
            if methodology in converted_outputs:
                methodology = f"{methodology} · {str(record.get('id'))[:8]}"
            converted_outputs[methodology] = output_path
            methodology_to_record[methodology] = str(record.get('id'))

        analyzer = ConversionQualityAnalyzer()
        comparison = analyzer.compare_methodologies(
            source_audio=source_audio,
            target_profile_id=data.get('target_profile_id') or records[0].get('profile_id'),
            methodologies=list(converted_outputs.keys()),
            converted_outputs=converted_outputs,
        )

        return jsonify({
            'source_id': next(iter(source_ids)),
            'records': [_serialize_quality_conversion(root, record) for record in records],
            'methodology_to_record': methodology_to_record,
            'best_methodology': comparison.best_methodology,
            'rankings': comparison.rankings,
            'summary': comparison.summary,
            'analyses': {
                methodology: {
                    'metrics': analysis.metrics.to_dict(),
                    'passes_thresholds': analysis.passes_thresholds,
                    'threshold_failures': analysis.threshold_failures,
                }
                for methodology, analysis in comparison.analyses.items()
            },
        })
    except Exception as exc:
        root.logger.error("Conversion record comparison failed: %s", exc)
        return root.error_response(str(exc))


def identify_speaker():
    """Identify speaker from audio by matching against known profiles."""
    root = _root()
    try:
        from ..inference.voice_identifier import get_voice_identifier

        if 'file' not in request.files:
            return root.validation_error_response('No audio file provided')

        audio_file = request.files['file']
        if not audio_file.filename:
            return root.validation_error_response('Empty filename')

        threshold = request.form.get('threshold', 0.85)
        try:
            threshold = float(threshold)
        except ValueError:
            threshold = 0.85

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        try:
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
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except Exception as exc:
        root.logger.error("Speaker identification failed: %s", exc)
        return root.error_response(str(exc))


def audit_loras():
    """Audit all LoRA adapters across voice profiles."""
    root = _root()
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

        from dataclasses import asdict
        from datetime import datetime

        return jsonify({
            'audit_timestamp': datetime.now().isoformat(),
            'summary': asdict(summary),
            'profiles': [asdict(status) for status in statuses],
        })
    except Exception as exc:
        root.logger.error("LoRA audit failed: %s", exc)
        return root.error_response(str(exc))


def analyze_conversion():
    """Analyze conversion quality with comprehensive metrics."""
    root = _root()
    try:
        from ..evaluation.conversion_quality_analyzer import ConversionQualityAnalyzer

        data = request.json or {}
        source_audio = data.get('source_audio')
        converted_audio = data.get('converted_audio')
        target_profile_id = data.get('target_profile_id')
        methodology = data.get('methodology', 'unknown')

        if not source_audio or not converted_audio:
            return root.validation_error_response('source_audio and converted_audio required')

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
    except Exception as exc:
        root.logger.error("Conversion analysis failed: %s", exc)
        return root.error_response(str(exc))


def compare_methodologies():
    """Compare conversion quality across multiple methodologies."""
    root = _root()
    try:
        from ..evaluation.conversion_quality_analyzer import ConversionQualityAnalyzer

        data = request.json or {}
        source_audio = data.get('source_audio')
        target_profile_id = data.get('target_profile_id')
        converted_outputs = data.get('converted_outputs', {})

        if not source_audio or not converted_outputs:
            return root.validation_error_response('source_audio and converted_outputs required')

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
                methodology: {
                    'metrics': analysis.metrics.to_dict(),
                    'passes_thresholds': analysis.passes_thresholds,
                    'threshold_failures': analysis.threshold_failures,
                }
                for methodology, analysis in comparison.analyses.items()
            },
        })
    except Exception as exc:
        root.logger.error("Methodology comparison failed: %s", exc)
        return root.error_response(str(exc))


def separate_artists():
    """Separate multi-artist audio and route to voice profiles."""
    root = _root()
    try:
        from ..audio.multi_artist_separator import MultiArtistSeparator

        if 'audio' not in request.files:
            return root.validation_error_response('No audio file provided')

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return root.validation_error_response('Empty filename')

        num_speakers = request.form.get('num_speakers', type=int)
        auto_create = request.form.get('auto_create_profiles', 'true').lower() == 'true'
        youtube_url = request.form.get('youtube_url')

        youtube_metadata = None
        if youtube_url and root.YOUTUBE_DOWNLOADER_AVAILABLE:
            try:
                downloader = root.YouTubeDownloader()
                youtube_metadata = downloader.get_metadata(youtube_url)
            except Exception as exc:
                root.logger.warning("Failed to get YouTube metadata: %s", exc)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        try:
            waveform, sample_rate = root.torchaudio.load(tmp_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze(0)
            audio = waveform.numpy()

            separator = MultiArtistSeparator(auto_create_profiles=auto_create)
            result = separator.separate_and_route(
                audio=audio,
                sample_rate=sample_rate,
                num_speakers=num_speakers,
                youtube_metadata=youtube_metadata,
                source_file=audio_file.filename,
            )

            artists_response = {}
            for profile_id, segments in result.artists.items():
                artists_response[profile_id] = {
                    'profile_name': segments[0].profile_name if segments else profile_id,
                    'segments': [
                        {
                            'start': segment.start,
                            'end': segment.end,
                            'duration': segment.duration,
                            'similarity': segment.similarity,
                        }
                        for segment in segments
                    ],
                    'total_duration': sum(segment.duration for segment in segments),
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
    except Exception as exc:
        root.logger.error("Multi-artist separation failed: %s", exc, exc_info=True)
        return root.error_response(str(exc))


def batch_separate_artists():
    """Process multiple audio files for multi-artist separation."""
    root = _root()
    try:
        from ..audio.multi_artist_separator import MultiArtistSeparator

        files = request.files.getlist('audio')
        if not files:
            return root.validation_error_response('No audio files provided')

        num_speakers = request.form.get('num_speakers', type=int)
        temp_paths = []
        for audio_file in files:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                audio_file.save(tmp.name)
                temp_paths.append(tmp.name)

        try:
            separator = MultiArtistSeparator()
            result = separator.process_batch(audio_files=temp_paths, num_speakers=num_speakers)
            return jsonify(result)
        finally:
            for path in temp_paths:
                try:
                    os.unlink(path)
                except Exception:
                    pass
    except Exception as exc:
        root.logger.error("Batch separation failed: %s", exc, exc_info=True)
        return root.error_response(str(exc))


def get_profile_quality_history(profile_id: str):
    """Get quality metrics history for a profile."""
    root = _root()
    try:
        from ..monitoring.quality_monitor import get_quality_monitor

        days = request.args.get('days', 30, type=int)
        monitor = get_quality_monitor()
        return jsonify(monitor.get_quality_history(profile_id, days=days))
    except Exception as exc:
        root.logger.error("Get quality history failed: %s", exc, exc_info=True)
        return root.error_response(str(exc))


def get_profile_quality_status(profile_id: str):
    """Get current quality status for a profile."""
    root = _root()
    try:
        from ..monitoring.quality_monitor import get_quality_monitor

        monitor = get_quality_monitor()
        return jsonify(monitor.get_quality_summary(profile_id))
    except Exception as exc:
        root.logger.error("Get quality status failed: %s", exc, exc_info=True)
        return root.error_response(str(exc))


def check_profile_degradation(profile_id: str):
    """Explicitly check for quality degradation."""
    root = _root()
    try:
        from ..monitoring.quality_monitor import get_quality_monitor

        monitor = get_quality_monitor()
        result = monitor.detect_degradation(profile_id)
        auto_retrain = request.json.get('auto_retrain', False) if request.json else False

        if result['degradation_detected'] and auto_retrain:
            try:
                job = root._get_training_job_manager().auto_queue_training(profile_id)
                if job:
                    result['retrain_job_id'] = job.job_id
                    result['retrain_queued'] = True
                else:
                    result['retrain_queued'] = False
            except Exception as exc:
                root.logger.warning("Failed to queue retrain: %s", exc)
                result['retrain_queued'] = False

        return jsonify(result)
    except Exception as exc:
        root.logger.error("Check degradation failed: %s", exc, exc_info=True)
        return root.error_response(str(exc))


def record_quality_metric():
    """Record a quality metric for a profile."""
    root = _root()
    try:
        from ..monitoring.quality_monitor import get_quality_monitor

        data = request.json
        if not data or 'profile_id' not in data:
            return root.validation_error_response('profile_id required')

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
            'alerts': [alert.to_dict() for alert in alerts],
            'alert_count': len(alerts),
        })
    except Exception as exc:
        root.logger.error("Record quality metric failed: %s", exc, exc_info=True)
        return root.error_response(str(exc))


def get_all_profiles_quality():
    """Get quality status for all monitored profiles."""
    root = _root()
    try:
        from ..monitoring.quality_monitor import get_quality_monitor

        profiles = get_quality_monitor().get_all_profiles_status()
        return jsonify({
            'profiles': profiles,
            'total': len(profiles),
            'degraded_count': sum(1 for profile in profiles if profile.get('status') == 'degraded'),
            'critical_count': sum(1 for profile in profiles if profile.get('status') == 'critical'),
        })
    except Exception as exc:
        root.logger.error("Get all profiles quality failed: %s", exc, exc_info=True)
        return root.error_response(str(exc))
