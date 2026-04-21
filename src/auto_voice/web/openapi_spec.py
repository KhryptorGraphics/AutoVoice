"""OpenAPI 3.0 specification generator for AutoVoice API."""
from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin
from apispec_webframeworks.flask import FlaskPlugin
from marshmallow import Schema, fields


# Define common schemas
class ErrorSchema(Schema):
    """Standard error response."""
    error = fields.Str(required=True, metadata={"description": "Error message"})
    message = fields.Str(metadata={"description": "Detailed error description"})


class JobStatusSchema(Schema):
    """Job status response."""
    job_id = fields.Str(required=True, metadata={"description": "Unique job identifier"})
    status = fields.Str(required=True, metadata={"description": "Job status: queued, processing, completed, failed, cancelled"})
    progress = fields.Int(metadata={"description": "Progress percentage (0-100)"})
    message = fields.Str(metadata={"description": "Status message"})
    created_at = fields.DateTime(metadata={"description": "Job creation timestamp"})
    updated_at = fields.DateTime(metadata={"description": "Last update timestamp"})


class ConversionSettingsSchema(Schema):
    """Song conversion settings."""
    target_profile_id = fields.Str(required=True, metadata={"description": "Target voice profile ID"})
    vocal_volume = fields.Float(metadata={"description": "Vocal volume multiplier (0.0-2.0, default: 1.0)"})
    instrumental_volume = fields.Float(metadata={"description": "Instrumental volume multiplier (0.0-2.0, default: 0.9)"})
    pitch_shift = fields.Float(metadata={"description": "Pitch shift in semitones (-12 to 12, default: 0)"})
    return_stems = fields.Bool(metadata={"description": "Return separate vocal/instrumental stems (default: false)"})
    output_quality = fields.Str(metadata={"description": "Quality preset: draft, fast, balanced, high, studio (default: balanced)"})
    adapter_type = fields.Str(metadata={"description": "LoRA adapter type: hq, nvfp4, unified (default: unified)"})
    pipeline_type = fields.Str(metadata={"description": "Pipeline type: realtime, quality, quality_seedvc, quality_shortcut (default: quality_seedvc; realtime/quality_shortcut are non-canonical variants)"})


class ConversionResultSchema(Schema):
    """Song conversion result (sync mode)."""
    status = fields.Str(required=True, metadata={"description": "Conversion status"})
    job_id = fields.Str(required=True, metadata={"description": "Job identifier"})
    audio = fields.Str(metadata={"description": "Base64-encoded audio data"})
    format = fields.Str(metadata={"description": "Audio format (wav)"})
    sample_rate = fields.Int(metadata={"description": "Sample rate in Hz"})
    duration = fields.Float(metadata={"description": "Duration in seconds"})
    metadata = fields.Dict(metadata={"description": "Conversion metadata"})
    f0_contour = fields.List(fields.Float(), metadata={"description": "Pitch contour in Hz (optional)"})
    f0_times = fields.List(fields.Float(), metadata={"description": "Time points for pitch contour (optional)"})


class AsyncJobResponseSchema(Schema):
    """Async job creation response."""
    status = fields.Str(required=True, metadata={"description": "Job status (queued)"})
    job_id = fields.Str(required=True, metadata={"description": "Unique job identifier"})
    websocket_room = fields.Str(metadata={"description": "WebSocket room ID for progress updates"})
    message = fields.Str(metadata={"description": "Instructions for receiving updates"})


class VoiceProfileSchema(Schema):
    """Voice profile metadata."""
    profile_id = fields.Str(required=True, metadata={"description": "Unique profile identifier"})
    name = fields.Str(required=True, metadata={"description": "Profile display name"})
    created_at = fields.DateTime(metadata={"description": "Creation timestamp"})
    sample_count = fields.Int(metadata={"description": "Number of training samples"})
    embedding_dim = fields.Int(metadata={"description": "Embedding dimension"})
    selected_adapter = fields.Str(metadata={"description": "Currently selected adapter: hq, nvfp4, unified"})
    metadata = fields.Dict(metadata={"description": "Additional metadata"})


class TrainingSampleSchema(Schema):
    """Training sample metadata."""
    sample_id = fields.Str(required=True, metadata={"description": "Unique sample identifier"})
    filename = fields.Str(required=True, metadata={"description": "Original filename"})
    duration = fields.Float(metadata={"description": "Duration in seconds"})
    sample_rate = fields.Int(metadata={"description": "Sample rate in Hz"})
    created_at = fields.DateTime(metadata={"description": "Upload timestamp"})


class TrainingJobSchema(Schema):
    """Training job status."""
    job_id = fields.Str(required=True, metadata={"description": "Unique job identifier"})
    profile_id = fields.Str(required=True, metadata={"description": "Target profile ID"})
    status = fields.Str(required=True, metadata={"description": "Job status: queued, training, completed, failed"})
    progress = fields.Int(metadata={"description": "Progress percentage (0-100)"})
    epoch = fields.Int(metadata={"description": "Current epoch"})
    total_epochs = fields.Int(metadata={"description": "Total epochs"})
    loss = fields.Float(metadata={"description": "Current loss value"})
    created_at = fields.DateTime(metadata={"description": "Job creation timestamp"})


class HealthCheckSchema(Schema):
    """System health status."""
    status = fields.Str(required=True, metadata={"description": "Overall status: healthy, degraded, unhealthy"})
    components = fields.Dict(metadata={"description": "Component-level health status"})
    timestamp = fields.DateTime(metadata={"description": "Health check timestamp"})


class GPUMetricsSchema(Schema):
    """GPU metrics."""
    gpu_id = fields.Int(metadata={"description": "GPU device ID"})
    name = fields.Str(metadata={"description": "GPU model name"})
    memory_used = fields.Int(metadata={"description": "Memory used in bytes"})
    memory_total = fields.Int(metadata={"description": "Total memory in bytes"})
    memory_percent = fields.Float(metadata={"description": "Memory usage percentage"})
    utilization = fields.Float(metadata={"description": "GPU utilization percentage"})
    temperature = fields.Float(metadata={"description": "GPU temperature in Celsius"})


class DiarizerSegmentSchema(Schema):
    """Speaker diarization segment."""
    speaker_id = fields.Str(required=True, metadata={"description": "Speaker identifier (SPEAKER_00, SPEAKER_01, etc.)"})
    start_time = fields.Float(required=True, metadata={"description": "Segment start time in seconds"})
    end_time = fields.Float(required=True, metadata={"description": "Segment end time in seconds"})
    confidence = fields.Float(metadata={"description": "Confidence score (0-1)"})


class DiarizationResultSchema(Schema):
    """Diarization result."""
    job_id = fields.Str(required=True, metadata={"description": "Job identifier"})
    segments = fields.List(fields.Nested(DiarizerSegmentSchema), metadata={"description": "Speaker segments"})
    num_speakers = fields.Int(metadata={"description": "Number of detected speakers"})
    total_duration = fields.Float(metadata={"description": "Total audio duration in seconds"})


class YouTubeInfoSchema(Schema):
    """YouTube video information."""
    title = fields.Str(metadata={"description": "Video title"})
    duration = fields.Float(metadata={"description": "Duration in seconds"})
    artist = fields.Str(metadata={"description": "Artist name (if detected)"})
    is_featured_artist = fields.Bool(metadata={"description": "Whether artist is in featured list"})
    thumbnail_url = fields.Str(metadata={"description": "Thumbnail URL"})


class YouTubeDownloadResultSchema(Schema):
    """YouTube download result."""
    job_id = fields.Str(required=True, metadata={"description": "Job identifier"})
    status = fields.Str(required=True, metadata={"description": "Download status"})
    audio_path = fields.Str(metadata={"description": "Path to downloaded audio file"})
    metadata = fields.Nested(YouTubeInfoSchema, metadata={"description": "Video metadata"})


def create_openapi_spec():
    """Create OpenAPI 3.0 specification for AutoVoice API."""
    spec = APISpec(
        title="AutoVoice API",
        version="1.0.0",
        openapi_version="3.0.2",
        info={
            "description": """
# AutoVoice REST API

GPU-accelerated singing voice conversion and TTS system with real-time processing capabilities.

## Features

- **Voice Conversion**: Convert songs to target voice profiles with high-fidelity quality
- **Voice Cloning**: Create voice profiles from audio samples
- **Training**: Train custom voice models with LoRA adapters
- **Real-time Processing**: Low-latency voice conversion for live karaoke
- **Speaker Diarization**: Separate and identify multiple speakers
- **YouTube Integration**: Download and process YouTube audio

## Pipeline Types

- **realtime**: Canonical fast/live path for karaoke and offline realtime conversion
- **quality_seedvc**: Canonical offline quality path with Seed-VC DiT-CFM (44.1kHz)
- **quality**: Experimental CoMoSVC offline path
- **quality_shortcut**: Experimental Seed-VC shortcut offline path

## Adapter Types

- **hq**: High-quality LoRA adapter (balanced quality/speed)
- **nvfp4**: Fast 4-bit quantized adapter (low latency)
- **unified**: New unified adapter format (recommended)

## Authentication

Currently no authentication required. Production deployments should add API key authentication.

## Rate Limiting

No rate limiting currently enforced. Production deployments should implement rate limiting.

## WebSocket Support

Real-time progress updates available via Socket.IO:
- Connect to Socket.IO endpoint
- Join room with job_id to receive progress events
- Event types: `job_progress`, `training_progress`, `separation_progress`
            """,
            "contact": {
                "name": "AutoVoice Support",
                "url": "https://github.com/yourusername/autovoice"
            },
            "license": {
                "name": "MIT",
                "url": "https://opensource.org/licenses/MIT"
            }
        },
        servers=[
            {"url": "http://localhost:5000", "description": "Development server"},
            {"url": "http://localhost:10000", "description": "Production server"}
        ],
        plugins=[FlaskPlugin(), MarshmallowPlugin()],
    )

    # Register schemas
    spec.components.schema("Error", schema=ErrorSchema)
    spec.components.schema("JobStatus", schema=JobStatusSchema)
    spec.components.schema("ConversionSettings", schema=ConversionSettingsSchema)
    spec.components.schema("ConversionResult", schema=ConversionResultSchema)
    spec.components.schema("AsyncJobResponse", schema=AsyncJobResponseSchema)
    spec.components.schema("VoiceProfile", schema=VoiceProfileSchema)
    spec.components.schema("TrainingSample", schema=TrainingSampleSchema)
    spec.components.schema("TrainingJob", schema=TrainingJobSchema)
    spec.components.schema("HealthCheck", schema=HealthCheckSchema)
    spec.components.schema("GPUMetrics", schema=GPUMetricsSchema)
    spec.components.schema("DiarizerSegment", schema=DiarizerSegmentSchema)
    spec.components.schema("DiarizationResult", schema=DiarizationResultSchema)
    spec.components.schema("YouTubeInfo", schema=YouTubeInfoSchema)
    spec.components.schema("YouTubeDownloadResult", schema=YouTubeDownloadResultSchema)

    # Define common tags
    spec.tag({
        "name": "Conversion",
        "description": "Song and voice conversion operations"
    })
    spec.tag({
        "name": "Voice Profiles",
        "description": "Voice profile management and creation"
    })
    spec.tag({
        "name": "Training",
        "description": "Model training operations"
    })
    spec.tag({
        "name": "Audio Processing",
        "description": "Audio analysis and processing utilities"
    })
    spec.tag({
        "name": "System",
        "description": "System health and monitoring"
    })
    spec.tag({
        "name": "Configuration",
        "description": "System configuration management"
    })
    spec.tag({
        "name": "YouTube",
        "description": "YouTube download and processing"
    })

    return spec
