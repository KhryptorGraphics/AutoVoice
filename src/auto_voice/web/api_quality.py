"""Quality-analysis, utility, and monitoring API routes extracted from the legacy API module."""

from __future__ import annotations

import os
import tempfile

from flask import Blueprint, jsonify, request


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
