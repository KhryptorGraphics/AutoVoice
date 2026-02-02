"""API documentation generator and Swagger UI integration."""
import json
import yaml
from typing import Dict, Any
from flask import Blueprint, jsonify, current_app
from flask_swagger_ui import get_swaggerui_blueprint

from .openapi_spec import create_openapi_spec


# Create documentation blueprint
docs_bp = Blueprint('docs', __name__)

# Swagger UI configuration
SWAGGER_URL = '/docs'
API_SPEC_URL = '/api/v1/openapi.json'

swagger_ui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_SPEC_URL,
    config={
        'app_name': "AutoVoice API",
        'defaultModelsExpandDepth': 3,
        'defaultModelExpandDepth': 3,
        'docExpansion': 'list',
        'filter': True,
        'showExtensions': True,
        'showCommonExtensions': True,
        'tryItOutEnabled': True
    }
)


@docs_bp.route('/openapi.json', methods=['GET'])
def get_openapi_json():
    """Return OpenAPI spec as JSON."""
    spec = create_openapi_spec()

    # Add endpoint documentation by introspecting Flask routes
    _add_endpoint_docs(spec)

    return jsonify(spec.to_dict())


@docs_bp.route('/openapi.yaml', methods=['GET'])
def get_openapi_yaml():
    """Return OpenAPI spec as YAML."""
    spec = create_openapi_spec()
    _add_endpoint_docs(spec)

    yaml_str = yaml.dump(spec.to_dict(), default_flow_style=False, sort_keys=False)
    return yaml_str, 200, {'Content-Type': 'text/yaml'}


def _add_endpoint_docs(spec):
    """Add endpoint documentation to OpenAPI spec."""

    # Conversion endpoints
    spec.path(
        path="/api/v1/convert/song",
        operations={
            "post": {
                "tags": ["Conversion"],
                "summary": "Convert song to target voice",
                "description": """
Convert a song using singing voice conversion pipeline. Supports both synchronous
and asynchronous processing modes.

**Async Mode** (when JobManager enabled):
- Returns HTTP 202 with job_id
- Poll GET /api/v1/convert/status/{job_id} for progress
- Download from GET /api/v1/convert/download/{job_id} when complete

**Sync Mode** (when JobManager disabled):
- Returns HTTP 200 with inline base64 audio
                """,
                "requestBody": {
                    "required": True,
                    "content": {
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "required": ["song", "profile_id"],
                                "properties": {
                                    "song": {
                                        "type": "string",
                                        "format": "binary",
                                        "description": "Audio file to convert (WAV, MP3, FLAC, OGG)"
                                    },
                                    "profile_id": {
                                        "type": "string",
                                        "description": "Target voice profile ID"
                                    },
                                    "settings": {
                                        "type": "string",
                                        "description": "JSON-encoded conversion settings"
                                    },
                                    "vocal_volume": {
                                        "type": "number",
                                        "minimum": 0.0,
                                        "maximum": 2.0,
                                        "default": 1.0,
                                        "description": "Vocal volume multiplier"
                                    },
                                    "instrumental_volume": {
                                        "type": "number",
                                        "minimum": 0.0,
                                        "maximum": 2.0,
                                        "default": 0.9,
                                        "description": "Instrumental volume multiplier"
                                    },
                                    "pitch_shift": {
                                        "type": "number",
                                        "minimum": -12,
                                        "maximum": 12,
                                        "default": 0,
                                        "description": "Pitch shift in semitones"
                                    },
                                    "return_stems": {
                                        "type": "boolean",
                                        "default": False,
                                        "description": "Return separate vocal/instrumental stems"
                                    },
                                    "output_quality": {
                                        "type": "string",
                                        "enum": ["draft", "fast", "balanced", "high", "studio"],
                                        "default": "balanced",
                                        "description": "Output quality preset"
                                    },
                                    "adapter_type": {
                                        "type": "string",
                                        "enum": ["hq", "nvfp4", "unified"],
                                        "description": "LoRA adapter type (default: unified)"
                                    },
                                    "pipeline_type": {
                                        "type": "string",
                                        "enum": ["realtime", "quality", "quality_seedvc"],
                                        "default": "quality",
                                        "description": "Processing pipeline type"
                                    }
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Conversion completed (sync mode)",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ConversionResult"}
                            }
                        }
                    },
                    "202": {
                        "description": "Conversion job queued (async mode)",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/AsyncJobResponse"}
                            }
                        }
                    },
                    "400": {
                        "description": "Invalid request",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        }
                    },
                    "404": {
                        "description": "Profile not found or no trained model",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        }
                    },
                    "503": {
                        "description": "Service unavailable",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        }
                    }
                }
            }
        }
    )

    spec.path(
        path="/api/v1/convert/status/{job_id}",
        operations={
            "get": {
                "tags": ["Conversion"],
                "summary": "Get conversion job status",
                "description": "Poll for conversion job status and progress",
                "parameters": [
                    {
                        "name": "job_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Job identifier"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Job status",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/JobStatus"}
                            }
                        }
                    },
                    "404": {
                        "description": "Job not found",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        }
                    }
                }
            }
        }
    )

    spec.path(
        path="/api/v1/convert/download/{job_id}",
        operations={
            "get": {
                "tags": ["Conversion"],
                "summary": "Download conversion result",
                "description": "Download completed conversion audio file",
                "parameters": [
                    {
                        "name": "job_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Job identifier"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Audio file",
                        "content": {
                            "audio/wav": {
                                "schema": {
                                    "type": "string",
                                    "format": "binary"
                                }
                            }
                        }
                    },
                    "404": {
                        "description": "Job not found",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        }
                    }
                }
            }
        }
    )

    spec.path(
        path="/api/v1/convert/cancel/{job_id}",
        operations={
            "post": {
                "tags": ["Conversion"],
                "summary": "Cancel conversion job",
                "description": "Cancel a queued or running conversion job",
                "parameters": [
                    {
                        "name": "job_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Job identifier"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Job cancelled",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "message": {"type": "string"},
                                        "job_id": {"type": "string"}
                                    }
                                }
                            }
                        }
                    },
                    "404": {
                        "description": "Job not found",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        }
                    }
                }
            }
        }
    )

    # Voice profile endpoints
    spec.path(
        path="/api/v1/voice/clone",
        operations={
            "post": {
                "tags": ["Voice Profiles"],
                "summary": "Create voice profile from samples",
                "description": "Upload audio samples to create a new voice profile",
                "requestBody": {
                    "required": True,
                    "content": {
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "required": ["samples", "name"],
                                "properties": {
                                    "samples": {
                                        "type": "array",
                                        "items": {
                                            "type": "string",
                                            "format": "binary"
                                        },
                                        "description": "Audio samples (minimum 3, recommended 5-10)"
                                    },
                                    "name": {
                                        "type": "string",
                                        "description": "Voice profile name"
                                    }
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Voice profile created",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/VoiceProfile"}
                            }
                        }
                    },
                    "400": {
                        "description": "Invalid samples or insufficient quality",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        }
                    }
                }
            }
        }
    )

    spec.path(
        path="/api/v1/voice/profiles",
        operations={
            "get": {
                "tags": ["Voice Profiles"],
                "summary": "List voice profiles",
                "description": "Get all available voice profiles",
                "responses": {
                    "200": {
                        "description": "List of voice profiles",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "profiles": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/VoiceProfile"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    )

    spec.path(
        path="/api/v1/voice/profiles/{profile_id}",
        operations={
            "get": {
                "tags": ["Voice Profiles"],
                "summary": "Get voice profile details",
                "parameters": [
                    {
                        "name": "profile_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"}
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Profile details",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/VoiceProfile"}
                            }
                        }
                    },
                    "404": {
                        "description": "Profile not found",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        }
                    }
                }
            },
            "delete": {
                "tags": ["Voice Profiles"],
                "summary": "Delete voice profile",
                "parameters": [
                    {
                        "name": "profile_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"}
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Profile deleted",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "message": {"type": "string"}
                                    }
                                }
                            }
                        }
                    },
                    "404": {
                        "description": "Profile not found",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        }
                    }
                }
            }
        }
    )

    # Health check
    spec.path(
        path="/api/v1/health",
        operations={
            "get": {
                "tags": ["System"],
                "summary": "System health check",
                "description": "Get system health status and component availability",
                "responses": {
                    "200": {
                        "description": "Health status",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/HealthCheck"}
                            }
                        }
                    }
                }
            }
        }
    )

    # Training endpoints
    spec.path(
        path="/api/v1/training/jobs",
        operations={
            "get": {
                "tags": ["Training"],
                "summary": "List training jobs",
                "description": "Get all training jobs with optional status filter",
                "parameters": [
                    {
                        "name": "status",
                        "in": "query",
                        "required": False,
                        "schema": {
                            "type": "string",
                            "enum": ["queued", "training", "completed", "failed"]
                        }
                    }
                ],
                "responses": {
                    "200": {
                        "description": "List of training jobs",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "jobs": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/TrainingJob"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "post": {
                "tags": ["Training"],
                "summary": "Create training job",
                "description": "Start training a voice model for a profile",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["profile_id"],
                                "properties": {
                                    "profile_id": {
                                        "type": "string",
                                        "description": "Profile to train"
                                    },
                                    "epochs": {
                                        "type": "integer",
                                        "minimum": 10,
                                        "maximum": 1000,
                                        "default": 100,
                                        "description": "Number of training epochs"
                                    },
                                    "batch_size": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": 64,
                                        "default": 8,
                                        "description": "Training batch size"
                                    },
                                    "learning_rate": {
                                        "type": "number",
                                        "minimum": 0.00001,
                                        "maximum": 0.01,
                                        "default": 0.0001,
                                        "description": "Learning rate"
                                    },
                                    "adapter_type": {
                                        "type": "string",
                                        "enum": ["hq", "nvfp4", "unified"],
                                        "default": "unified",
                                        "description": "Adapter type to train"
                                    }
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "202": {
                        "description": "Training job created",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/TrainingJob"}
                            }
                        }
                    },
                    "400": {
                        "description": "Invalid request",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        }
                    }
                }
            }
        }
    )

    # Audio processing endpoints
    spec.path(
        path="/api/v1/audio/diarize",
        operations={
            "post": {
                "tags": ["Audio Processing"],
                "summary": "Speaker diarization",
                "description": "Detect and separate multiple speakers in audio",
                "requestBody": {
                    "required": True,
                    "content": {
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "required": ["audio"],
                                "properties": {
                                    "audio": {
                                        "type": "string",
                                        "format": "binary",
                                        "description": "Audio file to diarize"
                                    },
                                    "num_speakers": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": 10,
                                        "description": "Expected number of speakers (auto-detect if not specified)"
                                    }
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Diarization result",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/DiarizationResult"}
                            }
                        }
                    },
                    "400": {
                        "description": "Invalid audio",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        }
                    }
                }
            }
        }
    )

    # YouTube endpoints
    spec.path(
        path="/api/v1/youtube/info",
        operations={
            "post": {
                "tags": ["YouTube"],
                "summary": "Get YouTube video info",
                "description": "Fetch metadata for a YouTube video",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["url"],
                                "properties": {
                                    "url": {
                                        "type": "string",
                                        "description": "YouTube video URL"
                                    }
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Video information",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/YouTubeInfo"}
                            }
                        }
                    },
                    "400": {
                        "description": "Invalid URL",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        }
                    }
                }
            }
        }
    )

    spec.path(
        path="/api/v1/youtube/download",
        operations={
            "post": {
                "tags": ["YouTube"],
                "summary": "Download YouTube audio",
                "description": "Download and process YouTube video audio",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["url"],
                                "properties": {
                                    "url": {
                                        "type": "string",
                                        "description": "YouTube video URL"
                                    },
                                    "profile_id": {
                                        "type": "string",
                                        "description": "Profile to associate with download"
                                    }
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Download result",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/YouTubeDownloadResult"}
                            }
                        }
                    },
                    "400": {
                        "description": "Invalid request",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        }
                    }
                }
            }
        }
    )

    # GPU metrics
    spec.path(
        path="/api/v1/gpu/metrics",
        operations={
            "get": {
                "tags": ["System"],
                "summary": "GPU metrics",
                "description": "Get current GPU utilization and memory metrics",
                "responses": {
                    "200": {
                        "description": "GPU metrics",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "gpus": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/GPUMetrics"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    )
