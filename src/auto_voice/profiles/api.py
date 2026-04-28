"""REST API endpoints for voice profile management.

Provides CRUD operations for voice profiles:
- POST /api/v1/profiles - Create new profile
- GET /api/v1/profiles - List profiles by user_id
- GET /api/v1/profiles/{id} - Get profile by ID
- PUT /api/v1/profiles/{id} - Update profile
- DELETE /api/v1/profiles/{id} - Delete profile

Training sample endpoints:
- POST /api/v1/profiles/{id}/samples - Upload audio sample
- GET /api/v1/profiles/{id}/samples - List profile samples
- GET /api/v1/profiles/{id}/samples/{sample_id} - Get sample
- DELETE /api/v1/profiles/{id}/samples/{sample_id} - Delete sample
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import soundfile as sf
from flask import Blueprint, current_app, jsonify, request

from auto_voice.profiles.db.models import TrainingSampleDB, VoiceProfileDB
from auto_voice.profiles.db.session import get_db_session
from auto_voice.web.utils import (
    error_response,
    not_found_response,
    validation_error_response,
)

logger = logging.getLogger(__name__)

profiles_bp = Blueprint("profiles", __name__, url_prefix="/api/v1/profiles")


@profiles_bp.route("", methods=["POST"])
def create_profile():
    """Create a new voice profile.

    Request JSON:
        user_id (required): User identifier
        name (required): Profile display name
        settings (optional): Profile settings dict

    Returns:
        201: Created profile data
        400: Validation error
    """
    data = request.get_json()

    if not data:
        return validation_error_response("Request body required")

    user_id = data.get("user_id")
    if not user_id:
        return validation_error_response("user_id is required")

    name = data.get("name")
    if not name:
        return validation_error_response("name is required")

    if not name.strip():
        return validation_error_response("name cannot be empty")

    try:
        with get_db_session() as session:
            profile = VoiceProfileDB(
                user_id=user_id,
                name=name.strip(),
                settings=data.get("settings"),
            )
            session.add(profile)
            session.flush()  # Get ID before commit

            result = profile.to_dict()
            logger.info(f"Created profile {profile.id} for user {user_id}")
            return jsonify(result), 201
    except Exception as e:
        logger.error(f"Failed to create profile: {e}")
        return error_response(str(e))


@profiles_bp.route("", methods=["GET"])
def list_profiles():
    """List profiles for a user.

    Query params:
        user_id (required): User identifier

    Returns:
        200: Array of profile data
        400: Missing user_id
    """
    user_id = request.args.get("user_id")

    if not user_id:
        return validation_error_response("user_id query parameter is required")

    try:
        with get_db_session() as session:
            profiles = (
                session.query(VoiceProfileDB)
                .filter_by(user_id=user_id)
                .order_by(VoiceProfileDB.created.desc())
                .all()
            )
            return jsonify([p.to_dict() for p in profiles]), 200
    except Exception as e:
        logger.error(f"Failed to list profiles: {e}")
        return error_response(str(e))


@profiles_bp.route("/<profile_id>", methods=["GET"])
def get_profile(profile_id: str):
    """Get a profile by ID.

    Returns:
        200: Profile data
        404: Profile not found
    """
    try:
        with get_db_session() as session:
            profile = session.query(VoiceProfileDB).filter_by(id=profile_id).first()

            if not profile:
                return not_found_response("Profile not found")

            return jsonify(profile.to_dict()), 200
    except Exception as e:
        logger.error(f"Failed to get profile {profile_id}: {e}")
        return error_response(str(e))


@profiles_bp.route("/<profile_id>", methods=["PUT"])
def update_profile(profile_id: str):
    """Update a profile.

    Request JSON (all optional):
        name: New profile name
        settings: New settings dict
        model_version: Model version string
        model_path: Path to model file

    Returns:
        200: Updated profile data
        400: Validation error
        404: Profile not found
    """
    data = request.get_json()

    if not data:
        return validation_error_response("Request body required")

    try:
        with get_db_session() as session:
            profile = session.query(VoiceProfileDB).filter_by(id=profile_id).first()

            if not profile:
                return not_found_response("Profile not found")

            # Update allowed fields
            if "name" in data:
                name = data["name"]
                if not name or not name.strip():
                    return validation_error_response("name cannot be empty")
                profile.name = name.strip()

            if "settings" in data:
                profile.settings = data["settings"]

            if "model_version" in data:
                profile.model_version = data["model_version"]

            if "model_path" in data:
                profile.model_path = data["model_path"]

            # Note: user_id is immutable, ignore if provided
            profile.updated = datetime.now(timezone.utc)

            session.flush()
            result = profile.to_dict()
            logger.info(f"Updated profile {profile_id}")
            return jsonify(result), 200
    except Exception as e:
        logger.error(f"Failed to update profile {profile_id}: {e}")
        return error_response(str(e))


@profiles_bp.route("/<profile_id>", methods=["DELETE"])
def delete_profile(profile_id: str):
    """Delete a profile and all associated samples.

    Returns:
        204: Successfully deleted
        404: Profile not found
    """
    try:
        with get_db_session() as session:
            profile = session.query(VoiceProfileDB).filter_by(id=profile_id).first()

            if not profile:
                return not_found_response("Profile not found")

            session.delete(profile)
            logger.info(f"Deleted profile {profile_id}")
            return "", 204
    except Exception as e:
        logger.error(f"Failed to delete profile {profile_id}: {e}")
        return error_response(str(e))


# =============================================================================
# Training Sample Endpoints
# =============================================================================


def get_storage_path() -> Path:
    """Get the configured sample storage path."""
    path = current_app.config.get("SAMPLE_STORAGE_PATH", "/data/autovoice/samples")
    return Path(path)


def get_profile_sample_dir(profile_id: str) -> Path:
    """Get the sample directory for a profile."""
    return get_storage_path() / profile_id


def upload_sample(profile_id: str):
    """Upload an audio sample for a profile.

    Request:
        multipart/form-data with:
        - audio (required): Audio file (WAV, MP3, FLAC, etc.)
        - metadata (optional): JSON string with additional metadata

    Returns:
        201: Created sample data
        400: Validation error
        404: Profile not found
    """
    try:
        with get_db_session() as session:
            # Check profile exists
            profile = session.query(VoiceProfileDB).filter_by(id=profile_id).first()
            if not profile:
                return not_found_response("Profile not found")

            # Check for audio file. The shared /api/v1/profiles surface documents
            # "file"; keep accepting legacy "audio" for older clients.
            audio_file = request.files.get("audio") or request.files.get("file")
            if audio_file is None:
                return validation_error_response('audio file is required')
            if not audio_file.filename:
                return validation_error_response("audio file is required")

            # Create profile sample directory
            sample_dir = get_profile_sample_dir(profile_id)
            sample_dir.mkdir(parents=True, exist_ok=True)

            # Generate unique filename
            sample_id = str(uuid4())
            ext = Path(audio_file.filename).suffix or ".wav"
            filename = f"{sample_id}{ext}"
            audio_path = sample_dir / filename

            # Save file temporarily to validate
            audio_file.save(str(audio_path))

            # Validate and get audio info
            try:
                info = sf.info(str(audio_path))
                duration_seconds = info.duration
                sample_rate = info.samplerate
            except Exception as e:
                # Invalid audio file - clean up and return error
                os.remove(audio_path)
                return validation_error_response(f"Invalid audio file: {e}")

            # Parse optional metadata
            extra_metadata = None
            if "metadata" in request.form:
                try:
                    extra_metadata = json.loads(request.form["metadata"])
                except json.JSONDecodeError:
                    extra_metadata = None

            # Create sample record
            sample = TrainingSampleDB(
                id=sample_id,
                profile_id=profile_id,
                audio_path=str(audio_path),
                duration_seconds=duration_seconds,
                sample_rate=sample_rate,
                extra_metadata=extra_metadata,
            )
            session.add(sample)

            # Increment profile sample count
            profile.samples_count += 1
            profile.updated = datetime.now(timezone.utc)

            session.flush()
            result = sample.to_dict()
            logger.info(f"Uploaded sample {sample_id} for profile {profile_id}")
            return jsonify(result), 201

    except Exception as e:
        logger.error(f"Failed to upload sample for profile {profile_id}: {e}")
        return error_response(str(e))


def list_samples(profile_id: str):
    """List all samples for a profile.

    Returns:
        200: Array of sample data
        404: Profile not found
    """
    try:
        with get_db_session() as session:
            # Check profile exists
            profile = session.query(VoiceProfileDB).filter_by(id=profile_id).first()
            if not profile:
                return not_found_response("Profile not found")

            samples = (
                session.query(TrainingSampleDB)
                .filter_by(profile_id=profile_id)
                .order_by(TrainingSampleDB.created.desc())
                .all()
            )
            return jsonify([s.to_dict() for s in samples]), 200

    except Exception as e:
        logger.error(f"Failed to list samples for profile {profile_id}: {e}")
        return error_response(str(e))


def get_sample(profile_id: str, sample_id: str):
    """Get a sample by ID.

    Returns:
        200: Sample data
        404: Sample or profile not found
    """
    try:
        with get_db_session() as session:
            # Check profile exists
            profile = session.query(VoiceProfileDB).filter_by(id=profile_id).first()
            if not profile:
                return not_found_response("Profile not found")

            sample = (
                session.query(TrainingSampleDB)
                .filter_by(id=sample_id, profile_id=profile_id)
                .first()
            )
            if not sample:
                return not_found_response("Sample not found")

            return jsonify(sample.to_dict()), 200

    except Exception as e:
        logger.error(f"Failed to get sample {sample_id}: {e}")
        return error_response(str(e))


def delete_sample(profile_id: str, sample_id: str):
    """Delete a sample.

    Returns:
        204: Successfully deleted
        404: Sample or profile not found
    """
    try:
        with get_db_session() as session:
            # Check profile exists
            profile = session.query(VoiceProfileDB).filter_by(id=profile_id).first()
            if not profile:
                return not_found_response("Profile not found")

            sample = (
                session.query(TrainingSampleDB)
                .filter_by(id=sample_id, profile_id=profile_id)
                .first()
            )
            if not sample:
                return not_found_response("Sample not found")

            # Delete file from disk
            audio_path = sample.audio_path
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)

            # Delete from DB
            session.delete(sample)

            # Decrement profile sample count
            if profile.samples_count > 0:
                profile.samples_count -= 1
            profile.updated = datetime.now(timezone.utc)

            logger.info(f"Deleted sample {sample_id} from profile {profile_id}")
            return "", 204

    except Exception as e:
        logger.error(f"Failed to delete sample {sample_id}: {e}")
        return error_response(str(e))
