"""YouTube history, metadata, and download API routes extracted from the legacy API module."""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from flask import Blueprint, current_app, jsonify, request

from .security import (
    canonical_youtube_url,
    managed_audio_roots,
    record_structured_audit_event,
    redact_public_paths,
    require_media_consent,
)


def _root():
    from . import api as api_root

    return api_root


def register_youtube_routes(api_bp: Blueprint) -> None:
    """Register YouTube history, metadata, and download routes."""
    api_bp.add_url_rule('/youtube/history', view_func=list_youtube_history, methods=['GET'])
    api_bp.add_url_rule('/youtube/history', view_func=save_youtube_history, methods=['POST'])
    api_bp.add_url_rule('/youtube/history', view_func=clear_youtube_history, methods=['DELETE'])
    api_bp.add_url_rule('/youtube/history/<item_id>', view_func=delete_youtube_history_item, methods=['DELETE'])
    api_bp.add_url_rule('/youtube/history/export', view_func=export_youtube_history, methods=['GET'])
    api_bp.add_url_rule('/youtube/history/purge', view_func=purge_youtube_history, methods=['POST', 'DELETE'])
    api_bp.add_url_rule('/youtube/info', view_func=youtube_info, methods=['POST'])
    api_bp.add_url_rule('/youtube/download', view_func=youtube_download, methods=['POST'])
    api_bp.add_url_rule('/youtube/ingest', view_func=start_youtube_ingest, methods=['POST'])
    api_bp.add_url_rule('/youtube/ingest/<job_id>', view_func=get_youtube_ingest, methods=['GET'])
    api_bp.add_url_rule('/youtube/ingest/<job_id>/confirm', view_func=confirm_youtube_ingest, methods=['POST'])


_youtube_downloader: Optional['YouTubeDownloader'] = None
_INGEST_MATCH_THRESHOLD = 0.72


def get_youtube_downloader() -> 'YouTubeDownloader':
    """Get or create a YouTubeDownloader instance."""
    global _youtube_downloader
    root = _root()
    cached = getattr(root, '_youtube_downloader', _youtube_downloader)
    if cached is None:
        if not root.YOUTUBE_DOWNLOADER_AVAILABLE:
            raise RuntimeError("YouTube downloader not available")
        output_dir = os.path.join(root.UPLOAD_FOLDER, 'youtube')
        os.makedirs(output_dir, exist_ok=True)
        cached = root.YouTubeDownloader(output_dir)
    _youtube_downloader = cached
    root._youtube_downloader = cached
    return cached


def _youtube_ingest_job(job_id: str) -> dict[str, Any] | None:
    root = _root()
    getter = getattr(root, '_get_background_job', None)
    if not getter:
        return None
    job = getter(job_id)
    if not job or job.get('job_type') != 'youtube_ingest':
        return None
    return job


def _update_youtube_ingest_job(job_id: str, **updates: Any) -> dict[str, Any]:
    root = _root()
    updater = getattr(root, '_update_background_job', None)
    if updater:
        return updater(job_id, **updates)
    job = _youtube_ingest_job(job_id)
    if not job:
        raise KeyError(f"YouTube ingest job {job_id} not found")
    job.update(updates)
    root._get_state_store().save_background_job(job)
    return job


def _register_youtube_asset(path: str, *, kind: str, job_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
    state_store = _root()._get_state_store()
    return state_store.register_asset(
        path,
        kind=kind,
        owner_id=job_id,
        metadata={
            'source': 'youtube_ingest',
            'job_id': job_id,
            **metadata,
        },
    )


def _speaker_duration_map(segments: list[dict[str, Any]]) -> dict[str, float]:
    durations: dict[str, float] = {}
    for segment in segments:
        speaker_id = str(segment.get('speaker_id'))
        durations[speaker_id] = durations.get(speaker_id, 0.0) + float(segment.get('duration', 0.0))
    return durations


def _speaker_segments(segments: list[dict[str, Any]], speaker_id: str) -> list[dict[str, Any]]:
    return [segment for segment in segments if segment.get('speaker_id') == speaker_id]


def _suggested_speaker_name(index: int, speaker_id: str, metadata: dict[str, Any]) -> str:
    names = []
    if metadata.get('main_artist'):
        names.append(metadata['main_artist'])
    names.extend(metadata.get('featured_artists') or [])
    if index < len(names) and names[index]:
        return str(names[index])
    return f"Unknown Speaker {index + 1}"


def _build_ingest_suggestions(
    *,
    diarizer,
    profile_store,
    vocals_path: str,
    segments: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    suggestions = []
    speaker_ids = sorted({str(segment['speaker_id']) for segment in segments})
    durations = _speaker_duration_map(segments)

    for index, speaker_id in enumerate(speaker_ids):
        current_segments = _speaker_segments(segments, speaker_id)
        longest = max(current_segments, key=lambda item: float(item.get('duration', 0.0)))
        matches: list[dict[str, Any]] = []
        match_error = None
        try:
            embedding = diarizer.extract_speaker_embedding(
                audio_path=vocals_path,
                start=float(longest['start']),
                end=float(longest['end']),
            )
            matches = profile_store.rank_speaker_embedding_matches(
                embedding,
                profile_role='source_artist',
                limit=5,
            )
        except Exception as exc:
            match_error = str(exc)

        best_match = matches[0] if matches else None
        suggestions.append({
            'speaker_id': speaker_id,
            'suggested_name': _suggested_speaker_name(index, speaker_id, metadata),
            'duration': durations.get(speaker_id, 0.0),
            'segment_count': len(current_segments),
            'matches': matches,
            'recommended_action': (
                'assign_existing'
                if best_match and float(best_match.get('similarity') or 0.0) >= _INGEST_MATCH_THRESHOLD
                else 'create_new'
            ),
            'recommended_profile_id': (
                best_match.get('profile_id')
                if best_match and float(best_match.get('similarity') or 0.0) >= _INGEST_MATCH_THRESHOLD
                else None
            ),
            'identity_confidence': (
                'voice_match'
                if best_match and float(best_match.get('similarity') or 0.0) >= _INGEST_MATCH_THRESHOLD
                else 'metadata_unverified'
            ),
            'match_error': match_error,
        })

    return suggestions


def _run_youtube_ingest(job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    root = _root()
    import soundfile as sf
    from auto_voice.audio.separation import VocalSeparator
    from auto_voice.audio.speaker_diarization import SpeakerDiarizer

    url = payload['url']
    audio_format = payload.get('format', 'wav')
    sample_rate = int(payload.get('sample_rate', 44100))
    metadata: dict[str, Any] = {}

    _update_youtube_ingest_job(job_id, progress=10, stage='download', message='Downloading YouTube audio')
    result = root.get_youtube_downloader().download(url, audio_format=audio_format, sample_rate=sample_rate)
    if not result.success or not result.audio_path:
        raise RuntimeError(result.error or 'YouTube download failed')

    metadata = {
        'url': url,
        'title': result.title,
        'duration': result.duration,
        'main_artist': result.main_artist,
        'featured_artists': result.featured_artists or [],
        'is_cover': result.is_cover,
        'original_artist': result.original_artist,
        'song_title': result.song_title,
        'thumbnail_url': result.thumbnail_url,
        'video_id': result.video_id,
    }
    audio_asset = _register_youtube_asset(
        result.audio_path,
        kind='youtube_audio',
        job_id=job_id,
        metadata={**metadata, 'variant': 'downloaded_audio'},
    )

    _update_youtube_ingest_job(job_id, progress=35, stage='separate', message='Splitting vocals and instrumental')
    audio, sr = sf.read(result.audio_path)
    if getattr(audio, 'ndim', 1) > 1:
        audio = audio.T

    separator = VocalSeparator()
    stems = separator.separate(audio, sr)
    output_dir = Path(root.UPLOAD_FOLDER) / 'youtube_ingest' / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    vocals_path = str(output_dir / 'vocals.wav')
    instrumental_path = str(output_dir / 'instrumental.wav')
    sf.write(vocals_path, stems['vocals'], sr)
    sf.write(instrumental_path, stems['instrumental'], sr)
    vocals_asset = _register_youtube_asset(
        vocals_path,
        kind='youtube_vocals',
        job_id=job_id,
        metadata={**metadata, 'variant': 'vocals'},
    )
    instrumental_asset = _register_youtube_asset(
        instrumental_path,
        kind='youtube_instrumental',
        job_id=job_id,
        metadata={**metadata, 'variant': 'instrumental'},
    )

    _update_youtube_ingest_job(job_id, progress=60, stage='diarize', message='Diarizing vocals')
    diarizer = SpeakerDiarizer()
    diarization = diarizer.diarize(vocals_path)
    diarization_id = str(uuid.uuid4())
    segments = [
        {
            'speaker_id': segment.speaker_id,
            'start': segment.start,
            'end': segment.end,
            'duration': segment.duration,
            'confidence': segment.confidence,
        }
        for segment in diarization.segments
    ]
    speaker_durations = _speaker_duration_map(segments)
    diarization_payload = {
        'diarization_id': diarization_id,
        'audio_path': vocals_path,
        'audio_duration': diarization.audio_duration,
        'num_speakers': diarization.num_speakers,
        'segments': segments,
        'speaker_durations': speaker_durations,
        'created_at': time.time(),
        'metadata': {
            **metadata,
            'source': 'youtube_ingest',
            'job_id': job_id,
            'downloaded_audio_asset_id': audio_asset['asset_id'],
            'vocals_asset_id': vocals_asset['asset_id'],
            'instrumental_asset_id': instrumental_asset['asset_id'],
        },
    }
    root._save_diarization_result(diarization_payload)

    _update_youtube_ingest_job(job_id, progress=82, stage='match', message='Matching speakers to source profiles')
    suggestions = _build_ingest_suggestions(
        diarizer=diarizer,
        profile_store=root._get_profile_store(),
        vocals_path=vocals_path,
        segments=segments,
        metadata=metadata,
    )

    response = {
        'job_id': job_id,
        'url': url,
        'metadata': metadata,
        'assets': {
            'audio': {'path': result.audio_path, 'asset_id': audio_asset['asset_id']},
            'vocals': {'path': vocals_path, 'asset_id': vocals_asset['asset_id']},
            'instrumental': {'path': instrumental_path, 'asset_id': instrumental_asset['asset_id']},
        },
        'diarization_id': diarization_id,
        'diarization_result': {
            'diarization_id': diarization_id,
            'audio_duration': diarization.audio_duration,
            'num_speakers': diarization.num_speakers,
            'segments': segments,
            'speaker_durations': speaker_durations,
        },
        'suggestions': suggestions,
        'review_required': True,
    }

    history_item = {
        'id': f"{int(time.time())}-{result.video_id or job_id[:8]}",
        'url': url,
        'title': result.title,
        'mainArtist': result.main_artist,
        'featuredArtists': result.featured_artists or [],
        'hasDiarization': True,
        'numSpeakers': diarization.num_speakers,
        'timestamp': root._utcnow_iso(),
        'audioPath': result.audio_path,
        'vocalsPath': vocals_path,
        'instrumentalPath': instrumental_path,
        'videoId': result.video_id,
        'ingestJobId': job_id,
    }
    root._get_state_store().save_youtube_history_item(history_item)
    _audit_event(
        'youtube.ingest_completed',
        'youtube_ingest',
        job_id,
        {'video_id': result.video_id, 'title': result.title, 'action': 'ingest'},
        payload=response,
        asset_paths=[result.audio_path, vocals_path, instrumental_path],
    )
    return response


def list_youtube_history():
    """List persisted YouTube download history."""
    root = _root()
    try:
        limit = request.args.get('limit', type=int)
        state_store = root._get_state_store()
        history = state_store.list_youtube_history(limit=limit)
        return jsonify(_redact_youtube_payload(history))
    except Exception as exc:
        root.logger.error("Failed to list YouTube history: %s", exc, exc_info=True)
        return root.error_response(str(exc))


def start_youtube_ingest():
    """Start an async reviewed YouTube ingest pipeline."""
    root = _root()
    try:
        data = request.get_json() or {}
        if not data.get('url'):
            return root.validation_error_response('YouTube URL is required')

        try:
            url = canonical_youtube_url(data.get('url'))
        except ValueError as exc:
            return root.validation_error_response(str(exc))

        try:
            require_media_consent(data, current_app)
        except PermissionError as exc:
            return root.validation_error_response(str(exc))

        audio_format = data.get('format', 'wav')
        if audio_format not in ['wav', 'mp3', 'flac']:
            return root.validation_error_response('Invalid format. Must be wav, mp3, or flac')

        try:
            sample_rate = int(data.get('sample_rate', 44100))
        except (TypeError, ValueError):
            return root.validation_error_response('sample_rate must be an integer')
        if sample_rate not in [16000, 22050, 44100, 48000]:
            return root.validation_error_response(
                'Invalid sample_rate. Must be 16000, 22050, 44100, or 48000'
            )

        payload = {
            'url': url,
            'format': audio_format,
            'sample_rate': sample_rate,
            'separate_vocals': True,
            'run_diarization': True,
            'match_existing_profiles': True,
            'review_required': True,
        }
        job = root._create_background_job('youtube_ingest', payload)
        job.update({
            'stage': 'queued',
            'message': 'Queued YouTube auto-ingest',
        })
        root._save_background_job(job)
        root._submit_background_job(job['job_id'], _run_youtube_ingest, payload)
        _audit_event(
            'youtube.ingest_started',
            'youtube_ingest',
            job['job_id'],
            {'action': 'ingest', 'url': url},
        )
        return jsonify(_redact_youtube_payload(job)), 202
    except Exception as exc:
        root.logger.error("Failed to start YouTube ingest: %s", exc, exc_info=True)
        return root.error_response(str(exc))


def get_youtube_ingest(job_id: str):
    """Return status for a YouTube ingest job."""
    root = _root()
    try:
        job = _youtube_ingest_job(job_id)
        if not job:
            return root.not_found_response('YouTube ingest job not found')
        return jsonify(_redact_youtube_payload(job))
    except Exception as exc:
        root.logger.error("Failed to load YouTube ingest %s: %s", job_id, exc, exc_info=True)
        return root.error_response(str(exc))


def confirm_youtube_ingest(job_id: str):
    """Apply operator-reviewed profile assignments for a completed ingest job."""
    root = _root()
    try:
        job = _youtube_ingest_job(job_id)
        if not job:
            return root.not_found_response('YouTube ingest job not found')
        if job.get('status') != 'completed':
            return root.validation_error_response('YouTube ingest job is not complete')
        if job.get('confirmation'):
            return jsonify(_redact_youtube_payload(job['confirmation']))

        data = request.get_json() or {}
        decisions = data.get('decisions') or []
        if not isinstance(decisions, list) or not decisions:
            return root.validation_error_response('decisions must be a non-empty list')

        result = job.get('result') or {}
        diarization_id = result.get('diarization_id')
        if not diarization_id:
            return root.validation_error_response('Ingest job has no diarization result')

        from auto_voice.audio.speaker_diarization import (
            DiarizationResult,
            SpeakerDiarizer,
            SpeakerSegment,
        )

        diarization_data = root._get_state_store().get_diarization_result(diarization_id)
        if not diarization_data:
            return root.not_found_response('Diarization result not found')

        segments = diarization_data.get('segments') or []
        valid_speakers = {segment.get('speaker_id') for segment in segments}
        store = root._get_profile_store()
        diarizer = SpeakerDiarizer()
        applied = []
        skipped = []

        for decision in decisions:
            if not isinstance(decision, dict):
                return root.validation_error_response('Each decision must be an object')
            speaker_id = decision.get('speaker_id')
            action = decision.get('action')
            if speaker_id not in valid_speakers:
                return root.validation_error_response(f'Unknown speaker_id: {speaker_id}')
            if action == 'skip':
                skipped.append({'speaker_id': speaker_id, 'action': 'skip'})
                continue

            if action == 'create_new':
                name = (decision.get('name') or '').strip()
                if not name:
                    return root.validation_error_response('name is required when action=create_new')
                profile = root._create_profile_from_diarized_speaker(
                    diarization_data=diarization_data,
                    speaker_id=speaker_id,
                    name=name,
                    user_id=decision.get('user_id') or 'system',
                    profile_role='source_artist',
                    extract_segments=True,
                    request_metadata={
                        'source': 'youtube_ingest_review',
                        'ingest_job_id': job_id,
                        'identity_confidence': 'operator_reviewed',
                        **(decision.get('metadata') or {}),
                    },
                )
                applied.append({
                    'speaker_id': speaker_id,
                    'action': 'create_new',
                    'profile_id': profile['profile_id'],
                    'name': profile['name'],
                })
                continue

            if action == 'assign_existing':
                profile_id = decision.get('profile_id')
                if not profile_id:
                    return root.validation_error_response(
                        'profile_id is required when action=assign_existing'
                    )
                if not store.exists(profile_id):
                    return root.not_found_response(f'Profile not found: {profile_id}')

                speaker_segments = _speaker_segments(segments, speaker_id)
                seg_objects = [
                    SpeakerSegment(
                        start=float(segment['start']),
                        end=float(segment['end']),
                        speaker_id=str(segment['speaker_id']),
                        confidence=float(segment.get('confidence', 1.0)),
                    )
                    for segment in speaker_segments
                ]
                temp_result = DiarizationResult(
                    segments=seg_objects,
                    audio_duration=float(diarization_data.get('audio_duration') or 0.0),
                    num_speakers=1,
                )
                extracted_path = diarizer.extract_speaker_audio(
                    audio_path=diarization_data['audio_path'],
                    diarization=temp_result,
                    speaker_id=speaker_id,
                )

                duration = sum(float(segment.get('duration', 0.0)) for segment in speaker_segments)
                sample = store.add_training_sample(
                    profile_id=profile_id,
                    vocals_path=str(extracted_path),
                    duration=duration,
                    source_file=f"youtube_ingest_{job_id}_{speaker_id}",
                    extra_metadata={
                        'source': 'youtube_ingest_review',
                        'ingest_job_id': job_id,
                        'diarization_id': diarization_id,
                        'speaker_id': speaker_id,
                        'identity_confidence': 'operator_reviewed',
                        **(decision.get('metadata') or {}),
                    },
                )
                root._get_state_store().save_diarization_segment_assignment(
                    profile_id,
                    f"{diarization_id}_{speaker_id}",
                    str(extracted_path),
                )
                applied.append({
                    'speaker_id': speaker_id,
                    'action': 'assign_existing',
                    'profile_id': profile_id,
                    'sample_id': sample.sample_id,
                    'duration': duration,
                })
                continue

            return root.validation_error_response(
                'Invalid action. Must be assign_existing, create_new, or skip'
            )

        confirmation = {
            'job_id': job_id,
            'diarization_id': diarization_id,
            'applied': applied,
            'skipped': skipped,
            'status': 'success',
        }
        _update_youtube_ingest_job(job_id, confirmed_at=root._utcnow_iso(), confirmation=confirmation)
        _audit_event(
            'youtube.ingest_confirmed',
            'youtube_ingest',
            job_id,
            {'action': 'confirm', 'applied': len(applied), 'skipped': len(skipped)},
            payload=confirmation,
        )
        return jsonify(_redact_youtube_payload(confirmation))
    except Exception as exc:
        root.logger.error("Failed to confirm YouTube ingest %s: %s", job_id, exc, exc_info=True)
        return root.error_response(str(exc))


def save_youtube_history():
    """Create or update a persisted YouTube download history item."""
    root = _root()
    try:
        data = request.get_json()
        if not data:
            return root.validation_error_response('No JSON data provided')

        try:
            url = canonical_youtube_url(data.get('url')) if data.get('url') else None
        except ValueError as exc:
            return root.validation_error_response(str(exc))

        if data.get('audioPath') or data.get('filteredPath'):
            return root.validation_error_response(
                'audioPath and filteredPath are server-managed fields; use audioAssetId/filteredAssetId'
            )

        history_item = {
            'id': data.get('id') or f"{int(time.time())}-{uuid.uuid4().hex[:8]}",
            'url': url,
            'title': data.get('title'),
            'mainArtist': data.get('mainArtist'),
            'featuredArtists': data.get('featuredArtists', []),
            'hasDiarization': bool(data.get('hasDiarization', False)),
            'numSpeakers': int(data.get('numSpeakers', 0)),
            'timestamp': data.get('timestamp') or root._utcnow_iso(),
            'audioPath': None,
            'filteredPath': None,
            'audioAssetId': data.get('audioAssetId'),
            'filteredAssetId': data.get('filteredAssetId'),
            'videoId': data.get('videoId'),
        }
        state_store = root._get_state_store()
        state_store.save_youtube_history_item(history_item)
        _audit_event("youtube.history.saved", "youtube_history", history_item["id"], {"video_id": history_item.get("videoId")})
        return jsonify(_redact_youtube_payload(history_item)), 201
    except Exception as exc:
        root.logger.error("Failed to save YouTube history: %s", exc, exc_info=True)
        return root.error_response(str(exc))


def clear_youtube_history():
    """Clear persisted YouTube download history."""
    root = _root()
    try:
        state_store = root._get_state_store()
        history = state_store.list_youtube_history()
        state_store.clear_youtube_history()
        _audit_event(
            "youtube.history.cleared",
            "youtube_history",
            "history",
            {"cleared_count": len(history)},
            payload=history,
        )
        return '', 204
    except Exception as exc:
        root.logger.error("Failed to clear YouTube history: %s", exc, exc_info=True)
        return root.error_response(str(exc))


def delete_youtube_history_item(item_id: str):
    """Delete one persisted YouTube history item."""
    root = _root()
    try:
        state_store = root._get_state_store()
        history_item = state_store.get_youtube_history_item(item_id)
        if not state_store.delete_youtube_history_item(item_id):
            return root.not_found_response('History item not found')
        _audit_event(
            "youtube.history.deleted",
            "youtube_history",
            item_id,
            {"history_item_id": item_id},
            payload=history_item,
        )
        return '', 204
    except Exception as exc:
        root.logger.error("Failed to delete YouTube history item: %s", exc, exc_info=True)
        return root.error_response(str(exc))


def export_youtube_history():
    """Export persisted YouTube history with opaque asset references in public mode."""
    root = _root()
    try:
        limit = request.args.get('limit', type=int)
        payload = root._get_state_store().export_youtube_history(limit=limit)
        _audit_event(
            "youtube.history.exported",
            "youtube_history",
            "history",
            {"count": payload.get("count", 0)},
            payload=payload,
        )
        return jsonify(_redact_youtube_payload(payload))
    except Exception as exc:
        root.logger.error("Failed to export YouTube history: %s", exc, exc_info=True)
        return root.error_response(str(exc))


def purge_youtube_history():
    """Purge YouTube history and optionally delete managed downloaded media."""
    root = _root()
    try:
        data = request.get_json(silent=True) or {}
        delete_assets = bool(data.get("delete_assets", True))
        state_store = root._get_state_store()
        payload = state_store.purge_youtube_history(
            delete_assets=delete_assets,
            managed_roots=managed_audio_roots(
                current_app.config.get("DATA_DIR", "data"),
                root.UPLOAD_FOLDER,
            ),
        )
        _audit_event(
            "youtube.history.purged",
            "youtube_history",
            "history",
            {
                "delete_assets": delete_assets,
                "purged_items": payload.get("purged_items", 0),
                "deleted_file_count": len(payload.get("deleted_files", [])),
                "skipped_file_count": len(payload.get("skipped_files", [])),
            },
            asset_paths=payload.get("deleted_files", []),
        )
        return jsonify(_redact_youtube_payload(payload))
    except Exception as exc:
        root.logger.error("Failed to purge YouTube history: %s", exc, exc_info=True)
        return root.error_response(str(exc))


def youtube_info():
    """Get video information and detected artists without downloading."""
    root = _root()
    if not root.YOUTUBE_DOWNLOADER_AVAILABLE:
        return root.service_unavailable_response('YouTube downloader not available. Install yt-dlp.')

    data = request.get_json()
    if not data or 'url' not in data:
        return root.validation_error_response('Missing required field: url')

    try:
        url = canonical_youtube_url(data['url'])
    except ValueError as exc:
        return root.validation_error_response(str(exc))

    try:
        downloader = root.get_youtube_downloader()
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
            'error': result.error,
        })
    except Exception as exc:
        root.logger.error("YouTube info failed: %s", exc)
        return root.error_response(str(exc))


def youtube_download():
    """Download audio from YouTube video with metadata."""
    root = _root()
    if not root.YOUTUBE_DOWNLOADER_AVAILABLE:
        return root.service_unavailable_response('YouTube downloader not available. Install yt-dlp.')

    data = request.get_json()
    if not data or 'url' not in data:
        return root.validation_error_response('Missing required field: url')

    try:
        require_media_consent(data, current_app)
        url = canonical_youtube_url(data['url'])
    except PermissionError as exc:
        return root.validation_error_response(str(exc))
    except ValueError as exc:
        return root.validation_error_response(str(exc))

    audio_format = data.get('format', 'wav')
    if audio_format not in ['wav', 'mp3', 'flac']:
        return root.validation_error_response('Invalid format. Must be wav, mp3, or flac')

    sample_rate = data.get('sample_rate', 44100)
    try:
        sample_rate = int(sample_rate)
        if sample_rate not in [16000, 22050, 44100, 48000]:
            return root.validation_error_response(
                'Invalid sample_rate. Must be 16000, 22050, 44100, or 48000'
            )
    except (ValueError, TypeError):
        return root.validation_error_response('sample_rate must be an integer')

    run_diarization = data.get('run_diarization', False)
    filter_to_main_artist = data.get('filter_to_main_artist', False)

    try:
        downloader = root.get_youtube_downloader()
        result = downloader.download(url, audio_format=audio_format, sample_rate=sample_rate)

        if not result.success:
            return jsonify({
                'success': False,
                'error': result.error,
                'title': result.title,
                'video_id': result.video_id,
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
            'error': None,
        }

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

                root._save_diarization_result({
                    'diarization_id': diarization_id,
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
                })

                response['diarization_result'] = {
                    'diarization_id': diarization_id,
                    'num_speakers': diarization_result.num_speakers,
                    'speaker_durations': speaker_durations,
                    'segments': segments,
                }
                response['diarization_id'] = diarization_id
                response['speaker_durations'] = speaker_durations

                if filter_to_main_artist and diarization_result.num_speakers > 1:
                    try:
                        from ..audio.training_filter import TrainingDataFilter  # noqa: F401

                        speaker_durations = {}
                        for seg in diarization_result.segments:
                            speaker_durations[seg.speaker_id] = (
                                speaker_durations.get(seg.speaker_id, 0) + seg.duration
                            )
                        main_speaker = max(speaker_durations, key=speaker_durations.get)
                        filtered_path = result.audio_path.replace('.wav', '_filtered.wav')
                        filter_result = diarizer.extract_speaker_audio(
                            result.audio_path,
                            diarization_result.segments,
                            main_speaker,
                            filtered_path,
                        )

                        if filter_result and os.path.exists(filtered_path):
                            response['filtered_audio_path'] = filtered_path
                            response['main_speaker_id'] = main_speaker
                            response['filtered_duration'] = speaker_durations[main_speaker]
                            root.logger.info(
                                "Filtered audio to main speaker %s: %s",
                                main_speaker,
                                filtered_path,
                            )
                    except Exception as exc:
                        root.logger.warning("Failed to filter to main artist: %s", exc)
                        response['filter_error'] = str(exc)

            except Exception as exc:
                root.logger.warning("Diarization failed: %s", exc)
                response['diarization_error'] = str(exc)

        try:
            state_store = root._get_state_store()
            history_item = {
                'id': f"{int(time.time())}-{result.video_id or uuid.uuid4().hex[:8]}",
                'url': url,
                'title': result.title,
                'mainArtist': result.main_artist,
                'featuredArtists': result.featured_artists,
                'hasDiarization': bool(response.get('diarization_result')),
                'numSpeakers': response.get('diarization_result', {}).get('num_speakers', 0),
                'timestamp': root._utcnow_iso(),
                'audioPath': result.audio_path,
                'filteredPath': response.get('filtered_audio_path'),
                'videoId': result.video_id,
            }
            state_store.save_youtube_history_item(history_item)
            _audit_event(
                "youtube.downloaded",
                "youtube_audio",
                history_item["id"],
                {"video_id": result.video_id, "title": result.title},
            )
        except Exception as history_error:
            root.logger.warning("Failed to persist YouTube history: %s", history_error)

        return jsonify(_redact_youtube_payload(response))
    except Exception as exc:
        _audit_event("youtube.download_failed", "youtube_audio", None, {"error": str(exc)})
        root.logger.error("YouTube download failed: %s", exc)
        return root.error_response(str(exc))


_AUDIT_ACTION_OVERRIDES = {
    "youtube.downloaded": "download",
    "youtube.download_failed": "download",
    "youtube.ingest_started": "ingest",
    "youtube.ingest_completed": "ingest",
    "youtube.ingest_confirmed": "confirm",
    "youtube.history.saved": "import",
    "youtube.history.deleted": "delete",
    "youtube.history.cleared": "delete",
    "youtube.history.exported": "export",
    "youtube.history.purged": "delete",
}


def _collect_path_values(payload):
    paths = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"audio_path", "filtered_audio_path", "audioPath", "filteredPath", "path"} and value:
                paths.append(str(value))
                continue
            paths.extend(_collect_path_values(value))
    elif isinstance(payload, list):
        for item in payload:
            paths.extend(_collect_path_values(item))
    return paths


def _redact_youtube_payload(payload):
    return redact_public_paths(payload, current_app, _root()._get_state_store(), kind="youtube_audio")


def _audit_event(
    event_type: str,
    resource_type: str,
    resource_id: str | None,
    metadata: dict,
    *,
    payload=None,
    asset_paths=None,
) -> None:
    root = _root()
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
            asset_kind="youtube_audio",
            details=details,
        )
    except Exception:
        root.logger.warning("Failed to persist audit event %s", event_type, exc_info=True)
