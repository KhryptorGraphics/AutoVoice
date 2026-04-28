"""YouTube history, metadata, and download API routes extracted from the legacy API module."""

from __future__ import annotations

import os
import time
import uuid
from typing import Optional

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


_youtube_downloader: Optional['YouTubeDownloader'] = None


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
