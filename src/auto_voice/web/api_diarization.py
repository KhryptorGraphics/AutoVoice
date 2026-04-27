"""Diarization and profile-enrichment API routes extracted from the legacy API module."""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, Optional

from flask import Blueprint, jsonify, request

_deps: Dict[str, Any] = {}


def register_diarization_routes(api_bp: Blueprint, **deps: Any) -> None:
    """Register the diarization/profile-enrichment route family on the shared API blueprint."""
    _deps.update(deps)
    api_bp.add_url_rule('/audio/diarize', view_func=diarize_audio, methods=['POST'])
    api_bp.add_url_rule('/profiles/<profile_id>/samples/<sample_id>/filter', view_func=filter_sample, methods=['POST'])
    api_bp.add_url_rule('/profiles/<profile_id>/speaker-embedding', view_func=set_profile_speaker_embedding, methods=['POST'])
    api_bp.add_url_rule('/profiles/<profile_id>/speaker-embedding', view_func=get_profile_speaker_embedding, methods=['GET'])
    api_bp.add_url_rule('/audio/diarize/assign', view_func=assign_diarization_segment, methods=['POST'])
    api_bp.add_url_rule('/profiles/<profile_id>/segments', view_func=get_profile_segments, methods=['GET'])
    api_bp.add_url_rule('/profiles/auto-create', view_func=auto_create_profile_from_diarization, methods=['POST'])


def _dep(name: str) -> Any:
    return _deps[name]


_diarization_results: Dict[str, Dict[str, Any]] = {}
_segment_assignments: Dict[str, Dict[str, str]] = {}


def _state_store():
    getter = _deps.get('get_state_store')
    if getter is None:
        return None
    try:
        return getter()
    except Exception:
        return None


def _save_diarization_result(diarization_data: Dict[str, Any]) -> Dict[str, Any]:
    """Persist a diarization result while keeping the legacy module cache warm."""
    diarization_id = diarization_data['diarization_id']
    payload = dict(diarization_data)
    store = _state_store()
    if store is not None:
        store.save_diarization_result(payload)
    _diarization_results[diarization_id] = payload
    return payload


def _get_diarization_result(diarization_id: str) -> Optional[Dict[str, Any]]:
    store = _state_store()
    if store is not None:
        result = store.get_diarization_result(diarization_id)
        if result:
            _diarization_results[diarization_id] = result
            return result
    return _diarization_results.get(diarization_id)


def _save_segment_assignment(profile_id: str, segment_key: str, audio_path: str) -> Dict[str, str]:
    store = _state_store()
    if store is not None:
        assignments = store.save_diarization_segment_assignment(profile_id, segment_key, audio_path)
    else:
        assignments = dict(_segment_assignments.get(profile_id, {}))
        assignments[segment_key] = audio_path
    _segment_assignments[profile_id] = dict(assignments)
    return dict(assignments)


def _get_segment_assignments(profile_id: str) -> Dict[str, str]:
    assignments: Dict[str, str] = {}
    store = _state_store()
    if store is not None:
        assignments.update(store.get_diarization_segment_assignments(profile_id))
    assignments.update(_segment_assignments.get(profile_id, {}))
    if assignments:
        _segment_assignments[profile_id] = dict(assignments)
    return assignments


def _parse_legacy_segment_key(segment_key: Optional[str]) -> tuple[Optional[str], Optional[int]]:
    """Parse legacy `<diarization_id>_<segment_index>` keys."""
    if not segment_key:
        return None, None

    try:
        diarization_id, segment_index = segment_key.rsplit('_', 1)
        return diarization_id, int(segment_index)
    except (ValueError, AttributeError):
        return None, None


def _build_diarization_speaker_summary(segments: list[dict]) -> list[dict]:
    """Build a compact per-speaker summary for diarization responses."""
    speaker_map: dict[str, dict[str, Any]] = {}
    for segment in segments:
        speaker_id = segment['speaker_id']
        entry = speaker_map.setdefault(
            speaker_id,
            {
                'speaker_id': speaker_id,
                'duration': 0.0,
                'segment_count': 0,
                'confidence': [],
            },
        )
        entry['duration'] += float(segment.get('duration', 0.0))
        entry['segment_count'] += 1
        if 'confidence' in segment and segment['confidence'] is not None:
            entry['confidence'].append(float(segment['confidence']))

    summary = []
    for speaker_id in sorted(speaker_map):
        entry = speaker_map[speaker_id]
        confidences = entry.pop('confidence')
        entry['avg_confidence'] = (
            float(sum(confidences) / len(confidences)) if confidences else None
        )
        summary.append(entry)
    return summary


def _create_profile_from_diarized_speaker(
    *,
    diarization_data: dict,
    speaker_id: str,
    name: str,
    user_id: str = 'system',
    profile_role: str = 'source_artist',
    extract_segments: bool = True,
    request_metadata: Optional[dict] = None,
):
    """Create a profile from one speaker within a stored diarization result."""
    from auto_voice.audio.speaker_diarization import SpeakerDiarizer, DiarizationResult, SpeakerSegment

    segments = diarization_data.get('segments', [])
    speaker_segments = [s for s in segments if s['speaker_id'] == speaker_id]
    if not speaker_segments:
        raise ValueError(f'No segments found for speaker {speaker_id}')

    audio_path = diarization_data.get('audio_path')
    if not audio_path or not os.path.exists(audio_path):
        raise FileNotFoundError('Original audio not found')

    diarizer = SpeakerDiarizer()
    longest_segment = max(speaker_segments, key=lambda s: s['end'] - s['start'])
    embedding = diarizer.extract_speaker_embedding(
        audio_path=audio_path,
        start=longest_segment['start'],
        end=longest_segment['end'],
    )

    speaker_duration = sum(s['end'] - s['start'] for s in speaker_segments)
    profile_metadata = {
        'source_audio_path': audio_path,
        'source_diarization_id': diarization_data.get('diarization_id'),
        'source_speaker_id': speaker_id,
        'source_speaker_duration': speaker_duration,
    }
    profile_metadata.update(diarization_data.get('metadata') or {})
    profile_metadata.update(request_metadata or {})

    audio_segments = []
    if extract_segments:
        seg_objects = [
            SpeakerSegment(
                start=s['start'],
                end=s['end'],
                speaker_id=s['speaker_id'],
                confidence=s.get('confidence', 1.0),
            )
            for s in speaker_segments
        ]
        temp_result = DiarizationResult(
            segments=seg_objects,
            audio_duration=diarization_data.get('audio_duration', 0),
            num_speakers=diarization_data.get('num_speakers', 1),
        )
        extracted_path = diarizer.extract_speaker_audio(
            audio_path=audio_path,
            diarization=temp_result,
            speaker_id=speaker_id,
        )
        if extracted_path:
            audio_segments.append(str(extracted_path))

    store = _dep('get_profile_store')()
    profile_id = store.create_profile_from_diarization(
        name=name,
        speaker_embedding=embedding,
        user_id=user_id,
        audio_segments=audio_segments,
        profile_role=profile_role,
        metadata=profile_metadata,
    )

    return {
        'profile_id': profile_id,
        'name': name,
        'speaker_id': speaker_id,
        'profile_role': profile_role,
        'num_segments': len(speaker_segments),
        'total_duration': speaker_duration,
        'embedding_dim': len(embedding),
        'metadata': profile_metadata,
    }


def diarize_audio():
    """Run speaker diarization on uploaded audio."""
    logger = _dep('logger')
    try:
        from auto_voice.audio.speaker_diarization import SpeakerDiarizer

        if 'file' in request.files or 'audio' in request.files:
            file = request.files.get('file') or request.files.get('audio')
            if not file.filename:
                return _dep('validation_error_response')('No file selected')
            if not _dep('allowed_file')(file.filename):
                return _dep('validation_error_response')('Invalid file type')

            import tempfile
            temp_dir = tempfile.mkdtemp()
            audio_path = os.path.join(temp_dir, file.filename)
            file.save(audio_path)
        elif request.is_json:
            data = request.get_json()
            audio_path = data.get('audio_path')
            if not audio_path or not os.path.exists(audio_path):
                return _dep('validation_error_response')('audio_path not found')
        else:
            return _dep('validation_error_response')('Provide file upload or audio_path')

        num_speakers = None
        if request.is_json:
            num_speakers = request.get_json().get('num_speakers')

        diarizer = SpeakerDiarizer()
        result = diarizer.diarize(audio_path, num_speakers=num_speakers)
        diarization_id = str(uuid.uuid4())

        segments = [
            {
                'start': seg.start,
                'end': seg.end,
                'speaker_id': seg.speaker_id,
                'duration': seg.duration,
                'confidence': seg.confidence,
            }
            for seg in result.segments
        ]

        response = {
            'diarization_id': diarization_id,
            'audio_duration': result.audio_duration,
            'num_speakers': result.num_speakers,
            'segments': segments,
            'speakers': _build_diarization_speaker_summary(segments),
            'speaker_durations': {
                speaker_id: result.get_speaker_total_duration(speaker_id)
                for speaker_id in result.get_all_speaker_ids()
            },
        }
        if segments:
            response['segment_key'] = f'{diarization_id}_0'

        _save_diarization_result({
            'diarization_id': diarization_id,
            'audio_path': audio_path,
            'audio_duration': result.audio_duration,
            'num_speakers': result.num_speakers,
            'segments': segments,
            'speakers': response['speakers'],
            'created_at': time.time(),
        })

        logger.info(f"Diarization {diarization_id} complete: {result.num_speakers} speakers detected")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Diarization error: {e}", exc_info=True)
        return _dep('error_response')(str(e))


def filter_sample(profile_id: str, sample_id: str):
    """Filter a training sample to only include target speaker vocals."""
    logger = _dep('logger')
    try:
        from auto_voice.audio.training_filter import TrainingDataFilter

        sample = _dep('find_training_sample')(profile_id, sample_id)
        if sample is None:
            return _dep('not_found_response')('Sample not found')

        audio_path = sample.vocals_path
        if not audio_path or not os.path.exists(audio_path):
            return _dep('not_found_response')('Sample audio file not found')

        store = _dep('get_profile_store')()
        embedding = store.load_speaker_embedding(profile_id)
        if embedding is None:
            return _dep('validation_error_response')(
                'Profile has no speaker embedding. Upload a sample first to create one.'
            )

        data = request.get_json() or {}
        similarity_threshold = data.get('similarity_threshold', 0.7)

        filter_obj = TrainingDataFilter()
        output_path, metadata = filter_obj.filter_training_audio(
            audio_path=audio_path,
            target_embedding=embedding,
            similarity_threshold=similarity_threshold,
        )

        response = {
            'sample_id': sample_id,
            'original_path': audio_path,
            'filtered_path': str(output_path),
            'original_duration': metadata['original_duration'],
            'filtered_duration': metadata['filtered_duration'],
            'num_segments': metadata['num_segments'],
            'purity': metadata['purity'],
            'status': metadata['status'],
        }

        logger.info(f"Filtered sample {sample_id}: {metadata['filtered_duration']:.1f}s extracted")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Filter error: {e}", exc_info=True)
        return _dep('error_response')(str(e))


def set_profile_speaker_embedding(profile_id: str):
    """Set or update the speaker embedding for a profile."""
    logger = _dep('logger')
    try:
        from auto_voice.audio.speaker_diarization import SpeakerDiarizer

        store = _dep('get_profile_store')()
        if not store.exists(profile_id):
            return _dep('not_found_response')('Profile not found')

        data = request.get_json() or {}
        if data.get('use_samples', False):
            samples = store.list_training_samples(profile_id)
            if not samples:
                return _dep('validation_error_response')('No training samples to compute embedding from')
            audio_path = samples[0].vocals_path
        elif 'audio_path' in data:
            audio_path = data['audio_path']
            if not os.path.exists(audio_path):
                return _dep('validation_error_response')('Audio file not found')
        else:
            return _dep('validation_error_response')('Provide audio_path or set use_samples=true')

        diarizer = SpeakerDiarizer()
        embedding = diarizer.extract_speaker_embedding(audio_path)
        store.save_speaker_embedding(profile_id, embedding)

        logger.info(f"Set speaker embedding for profile {profile_id}")
        return jsonify({
            'profile_id': profile_id,
            'embedding_dim': len(embedding),
            'source': audio_path,
            'status': 'success',
        })
    except Exception as e:
        logger.error(f"Error setting speaker embedding: {e}", exc_info=True)
        return _dep('error_response')(str(e))


def get_profile_speaker_embedding(profile_id: str):
    """Check if profile has a speaker embedding."""
    logger = _dep('logger')
    try:
        store = _dep('get_profile_store')()
        if not store.exists(profile_id):
            return _dep('not_found_response')('Profile not found')

        embedding = store.load_speaker_embedding(profile_id)
        return jsonify({
            'profile_id': profile_id,
            'has_embedding': embedding is not None,
            'embedding_dim': len(embedding) if embedding is not None else None,
        })
    except Exception as e:
        logger.error(f"Error getting speaker embedding: {e}", exc_info=True)
        return _dep('error_response')(str(e))


def assign_diarization_segment():
    """Assign a diarization segment to an existing profile."""
    logger = _dep('logger')
    try:
        from auto_voice.audio.speaker_diarization import SpeakerDiarizer, DiarizationResult, SpeakerSegment

        data = request.get_json()
        if not data:
            return _dep('validation_error_response')('No JSON data provided')

        diarization_id = data.get('diarization_id')
        segment_index = data.get('segment_index')
        legacy_segment_key = data.get('segment_key')
        profile_id = data.get('profile_id')

        if (diarization_id is None or segment_index is None) and legacy_segment_key:
            parsed_diarization_id, parsed_segment_index = _parse_legacy_segment_key(legacy_segment_key)
            diarization_id = diarization_id or parsed_diarization_id
            if segment_index is None:
                segment_index = parsed_segment_index

        if not all([diarization_id, segment_index is not None, profile_id]):
            return _dep('validation_error_response')('Required: diarization_id, segment_index, profile_id')

        diarization_data = _get_diarization_result(diarization_id)
        if not diarization_data:
            return _dep('not_found_response')('Diarization result not found or expired')

        segments = diarization_data.get('segments', [])
        if segment_index < 0 or segment_index >= len(segments):
            return _dep('validation_error_response')(f'Invalid segment_index: {segment_index}')

        segment = segments[segment_index]
        store = _dep('get_profile_store')()
        if not store.exists(profile_id):
            return _dep('not_found_response')('Profile not found')

        extract_audio = data.get('extract_audio', True)
        extracted_path = None
        if extract_audio:
            audio_path = diarization_data.get('audio_path')
            if audio_path and os.path.exists(audio_path):
                diarizer = SpeakerDiarizer()
                seg_obj = SpeakerSegment(
                    start=segment['start'],
                    end=segment['end'],
                    speaker_id=segment['speaker_id'],
                    confidence=segment.get('confidence', 1.0),
                )
                temp_result = DiarizationResult(
                    segments=[seg_obj],
                    audio_duration=diarization_data.get('audio_duration', 0),
                    num_speakers=1,
                )
                extracted_path = diarizer.extract_speaker_audio(
                    audio_path=audio_path,
                    diarization=temp_result,
                    speaker_id=segment['speaker_id'],
                )
                if extracted_path:
                    from scipy.io import wavfile
                    sr, audio_data = wavfile.read(str(extracted_path))
                    duration = len(audio_data) / sr
                    store.add_training_sample(
                        profile_id=profile_id,
                        vocals_path=str(extracted_path),
                        duration=duration,
                        source_file=f"diarization_{diarization_id}_seg{segment_index}",
                    )

        segment_key = f"{diarization_id}_{segment_index}"
        _save_segment_assignment(profile_id, segment_key, str(extracted_path) if extracted_path else "")

        logger.info(f"Assigned segment {segment_index} from {diarization_id} to profile {profile_id}")
        return jsonify({
            'status': 'success',
            'profile_id': profile_id,
            'segment_index': segment_index,
            'extracted_path': str(extracted_path) if extracted_path else None,
            'segment': segment,
        })
    except Exception as e:
        logger.error(f"Error assigning segment: {e}", exc_info=True)
        return _dep('error_response')(str(e))


def get_profile_segments(profile_id: str):
    """Get all audio segments assigned to a profile."""
    logger = _dep('logger')
    try:
        store = _dep('get_profile_store')()
        if not store.exists(profile_id):
            return _dep('not_found_response')('Profile not found')

        samples = store.list_training_samples(profile_id)
        sample_segments = [
            {
                'type': 'training_sample',
                'sample_id': s.sample_id,
                'vocals_path': s.vocals_path,
                'duration': s.duration,
                'source_file': s.source_file,
                'created_at': s.created_at,
            }
            for s in samples
        ]

        assignments = _get_segment_assignments(profile_id)
        assignment_segments = [
            {'type': 'diarization_assignment', 'segment_key': key, 'audio_path': path}
            for key, path in assignments.items()
        ]

        total_duration = sum(s.duration for s in samples)
        return jsonify({
            'profile_id': profile_id,
            'total_segments': len(sample_segments) + len(assignment_segments),
            'total_duration': total_duration,
            'training_samples': sample_segments,
            'diarization_assignments': assignment_segments,
        })
    except Exception as e:
        logger.error(f"Error getting profile segments: {e}", exc_info=True)
        return _dep('error_response')(str(e))


def auto_create_profile_from_diarization():
    """Create a new profile from diarization results."""
    logger = _dep('logger')
    try:
        data = request.get_json()
        if not data:
            return _dep('validation_error_response')('No JSON data provided')

        diarization_id = data.get('diarization_id')
        legacy_segment_key = data.get('segment_key')
        speaker_id = data.get('speaker_id')
        name = data.get('name')
        artist_names = data.get('artist_names')
        create_all = bool(data.get('create_all'))

        if not diarization_id and legacy_segment_key:
            diarization_id, _ = _parse_legacy_segment_key(legacy_segment_key)
        if not diarization_id:
            return _dep('validation_error_response')('Required: diarization_id')
        if not create_all and not artist_names and not all([speaker_id, name]):
            return _dep('validation_error_response')(
                'Required: diarization_id plus speaker_id/name, artist_names, or create_all'
            )

        diarization_data = _get_diarization_result(diarization_id)
        if not diarization_data:
            return _dep('not_found_response')('Diarization result not found or expired')

        user_id = data.get('user_id', 'system')
        profile_role = data.get('profile_role', 'source_artist')
        request_metadata = data.get('metadata') or {}
        extract_segments = data.get('extract_segments', True)

        if artist_names or create_all:
            segments = diarization_data.get('segments', [])
            speaker_ids = sorted({segment['speaker_id'] for segment in segments})
            profiles = []
            names = list(artist_names or [])
            for index, current_speaker_id in enumerate(speaker_ids, start=1):
                speaker_name = (
                    names[index - 1]
                    if index - 1 < len(names) and names[index - 1]
                    else f"Artist {current_speaker_id.replace('_', ' ').title()}"
                )
                profiles.append(
                    _create_profile_from_diarized_speaker(
                        diarization_data=diarization_data,
                        speaker_id=current_speaker_id,
                        name=speaker_name,
                        user_id=user_id,
                        profile_role=profile_role,
                        extract_segments=extract_segments,
                        request_metadata=request_metadata,
                    )
                )

            if not profiles:
                return _dep('validation_error_response')('No artist profiles could be created from diarization')

            logger.info("Auto-created %s profiles from diarization %s", len(profiles), diarization_id)
            return jsonify({
                'status': 'success',
                'profiles': profiles,
                'diarization_id': diarization_id,
            }), 201

        profile_data = _create_profile_from_diarized_speaker(
            diarization_data=diarization_data,
            speaker_id=speaker_id,
            name=name,
            user_id=user_id,
            profile_role=profile_role,
            extract_segments=extract_segments,
            request_metadata=request_metadata,
        )

        logger.info("Auto-created profile '%s' (%s) from diarization", name, profile_data['profile_id'])
        return jsonify({**profile_data, 'status': 'success'}), 201
    except ValueError as e:
        logger.error(f"Error auto-creating profile: {e}", exc_info=True)
        return _dep('validation_error_response')(str(e))
    except FileNotFoundError as e:
        logger.error(f"Error auto-creating profile: {e}", exc_info=True)
        return _dep('not_found_response')(str(e))
    except Exception as e:
        logger.error(f"Error auto-creating profile: {e}", exc_info=True)
        return _dep('error_response')(str(e))
