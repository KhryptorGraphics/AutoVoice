# AutoVoice API Reference

Complete API reference for AutoVoice voice synthesis (TTS) and singing voice conversion.

## Table of Contents

1. [Introduction](#1-introduction)
2. [TTS Endpoints](#2-tts-endpoints)
3. [System Information Endpoints](#3-system-information-endpoints)
4. [Voice Conversion Endpoints](#4-voice-conversion-endpoints)
5. [WebSocket/Socket.IO API](#5-websocketsocketio-api)
6. [Error Codes](#6-error-codes)
7. [Python SDK](#7-python-sdk)
8. [JavaScript SDK](#8-javascript-sdk)
9. [Best Practices](#9-best-practices)

## 1. Introduction

### Overview

AutoVoice provides a unified REST API for both text-to-speech synthesis and singing voice conversion. The API is designed for ease of use, scalability, and production deployment.

**Base URL**: `http://localhost:5000/api/v1`

**Authentication**: None (currently open access)

**Rate Limiting**: None (recommended for production deployment)

**Response Format**: JSON

### API Reference Summary

| Endpoint | Method | Purpose | Auth Required |
|----------|--------|---------|---------------|
| `/health` | GET | General health status | No |
| `/health/live` | GET | Liveness probe | No |
| `/health/ready` | GET | Readiness probe | No |
| `/metrics` | GET | Prometheus metrics | No |
| `/api/v1/health` | GET | API-specific health check | No |
| `/api/v1/synthesize` | POST | Text-to-speech synthesis | No |
| `/api/v1/analyze` | POST | Quick audio analysis | No |
| `/api/v1/models/info` | GET | Get model information | No |
| `/api/v1/config` | GET | Get API configuration | No |
| `/api/v1/config` | POST | Update runtime config | No |
| `/api/v1/speakers` | GET | List available speakers | No |
| `/api/v1/gpu_status` | GET | Get GPU status | No |
| `/api/v1/process_audio` | POST | Process audio file | No |
| `/api/v1/voice/clone` | POST | Create voice profile | No |
| `/api/v1/voice/profiles` | GET | List voice profiles | No |
| `/api/v1/voice/profiles/{id}` | GET | Get voice profile details | No |
| `/api/v1/voice/profiles/{id}` | DELETE | Delete voice profile | No |
| `/api/v1/convert/song` | POST | Convert song to target voice | No |
| `/api/v1/convert/status/{conversion_id}` | GET | Get conversion status | No |
| `/api/v1/convert/download/{conversion_id}/converted.wav` | GET | Download converted audio file | No |

### Quick Start

```bash
# Text-to-speech
curl -X POST http://localhost:5000/api/v1/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "speaker_id": 0}'

# Voice cloning
curl -X POST http://localhost:5000/api/v1/voice/clone \
  -F "audio=@my_voice.wav" \
  -F "user_id=user123"

# Song conversion
curl -X POST http://localhost:5000/api/v1/convert/song \
  -F "song=@song.mp3" \
  -F "profile_id=profile-uuid"
```

### API Versions

- **v1** (current): Initial release with TTS and voice conversion

### Backward Compatibility

For backward compatibility, legacy `/api/*` routes automatically redirect (HTTP 307) to their `/api/v1/*` equivalents. For example:
- `/api/synthesize` → `/api/v1/synthesize`
- `/api/voice/clone` → `/api/v1/voice/clone`
- `/api/convert/song` → `/api/v1/convert/song`

**Recommendation**: Update your clients to use the versioned `/api/v1/*` paths directly to avoid redirect overhead and ensure forward compatibility.

### System Requirements

- **GPU**: NVIDIA GPU with compute capability 7.0+ (recommended)
- **Memory**: 4-8GB VRAM for voice conversion, 2-4GB for TTS
- **Audio Formats**: WAV, MP3, FLAC, OGG

---

## 2. TTS Endpoints

### Health and Metrics

#### GET /health

General system health check endpoint.

**Endpoint**: `GET /health`

**Response**:

**Status**: `200 OK`

**Body**:
```json
{
  "status": "healthy",
  "components": {
    "gpu_available": true,
    "model_loaded": true,
    "api": true,
    "synthesizer": true,
    "voice_cloner": true,
    "singing_conversion_pipeline": true
  },
  "system": {
    "memory_percent": 45.2,
    "cpu_percent": 12.5,
    "gpu": {
      "available": true,
      "device_count": 1
    }
  }
}
```

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Always `"healthy"` for this endpoint |
| `components` | object | Component availability status |
| `components.gpu_available` | boolean | Whether GPU is available |
| `components.model_loaded` | boolean | Whether voice model is loaded |
| `components.api` | boolean | Whether API is operational (always `true`) |
| `components.synthesizer` | boolean | Whether synthesizer is initialized |
| `components.voice_cloner` | boolean | Whether voice cloner is initialized |
| `components.singing_conversion_pipeline` | boolean | Whether singing conversion pipeline is initialized |
| `system` | object | System metrics (optional fields) |
| `system.memory_percent` | number | Memory usage percentage (if psutil available) |
| `system.cpu_percent` | number | CPU usage percentage (if psutil available) |
| `system.gpu` | object | GPU status details (if GPU manager available) |
| `system.gpu.available` | boolean | Whether CUDA is available |
| `system.gpu.device_count` | number | Number of GPU devices |

**Note**: The `system` object and its fields are optional and only included when the respective monitoring libraries (psutil, GPU manager) are available.

**Example**:

```bash
curl http://localhost:5000/health
```

---

#### GET /health/live

Liveness probe - checks if the application is running.

**Endpoint**: `GET /health/live`

**Response**:

**Status**: `200 OK`

**Body**:
```json
{
  "status": "alive"
}
```

**Example**:

```bash
curl http://localhost:5000/health/live
```

---

#### GET /health/ready

Readiness probe - checks if the application is ready to serve traffic.

**Endpoint**: `GET /health/ready`

**Response (Ready)**:

**Status**: `200 OK`

**Body**:
```json
{
  "status": "ready",
  "components": {
    "model": "ready",
    "gpu": "available",
    "synthesizer": "ready",
    "voice_cloner": "ready",
    "singing_conversion_pipeline": "ready"
  }
}
```

**Response (Not Ready)**:

**Status**: `503 Service Unavailable`

**Body**:
```json
{
  "status": "not_ready",
  "components": {
    "model": "not_initialized",
    "gpu": "unavailable",
    "synthesizer": "not_initialized",
    "voice_cloner": "not_initialized",
    "singing_conversion_pipeline": "not_initialized"
  }
}
```

**Component Status Values**:

| Component | Possible Values | Critical for Readiness |
|-----------|----------------|------------------------|
| `model` | `"ready"`, `"not_ready"`, `"not_initialized"` | ✅ Yes |
| `gpu` | `"available"`, `"unavailable"` | ❌ No (optional) |
| `synthesizer` | `"ready"`, `"not_initialized"` | ✅ Yes |
| `voice_cloner` | `"ready"`, `"not_initialized"` | ❌ No (optional) |
| `singing_conversion_pipeline` | `"ready"`, `"not_initialized"` | ❌ No (optional) |

**Readiness Logic**:
- Returns `200 OK` with `status: "ready"` only when **all critical components** (`model`, `synthesizer`) are ready.
- Returns `503 Service Unavailable` with `status: "not_ready"` if any critical component is not ready.
- Optional components (`gpu`, `voice_cloner`, `singing_conversion_pipeline`) do not affect readiness status.

**Example**:

```bash
curl http://localhost:5000/health/ready
```

---

#### GET /metrics

Prometheus metrics endpoint for monitoring and observability.

**Endpoint**: `GET /metrics`

**Response (Enabled)**:

**Status**: `200 OK`

**Content-Type**: `text/plain; version=0.0.4; charset=utf-8` (Prometheus text format)

**Body**:
```
# HELP autovoice_requests_total Total number of requests
# TYPE autovoice_requests_total counter
autovoice_requests_total{endpoint="/api/v1/synthesize"} 1234

# HELP autovoice_request_duration_seconds Request duration in seconds
# TYPE autovoice_request_duration_seconds histogram
autovoice_request_duration_seconds_bucket{le="0.1"} 100
autovoice_request_duration_seconds_bucket{le="0.5"} 450
autovoice_request_duration_seconds_bucket{le="1.0"} 800
autovoice_request_duration_seconds_bucket{le="+Inf"} 1234
autovoice_request_duration_seconds_sum 456.78
autovoice_request_duration_seconds_count 1234

# HELP autovoice_gpu_memory_allocated_bytes GPU memory allocated in bytes
# TYPE autovoice_gpu_memory_allocated_bytes gauge
autovoice_gpu_memory_allocated_bytes 4294967296
```

**Response (Disabled)**:

**Status**: `503 Service Unavailable`

**Content-Type**: `text/plain`

**Body** (plain text):
```
Metrics not enabled
```

**Authentication**:
- Authentication is **disabled by default** for the `/metrics` endpoint
- For production deployments, it is **strongly recommended** to enforce authentication at the reverse proxy level (e.g., nginx, Apache, or cloud load balancer)
- See the Prometheus scrape configuration example below for integration

**Prometheus Scrape Configuration**:

See `config/prometheus.yml` for a complete example. Basic configuration:

```yaml
scrape_configs:
  - job_name: 'autovoice'
    scrape_interval: 10s
    static_configs:
      - targets: ['auto-voice-app:5000']
    metrics_path: '/metrics'
    scheme: http
    # For authenticated endpoints, add:
    # basic_auth:
    #   username: 'prometheus'
    #   password: 'your-password'
```

**Example**:

```bash
# Fetch metrics
curl http://localhost:5000/metrics

# With authentication (if configured at reverse proxy)
curl -u prometheus:password http://localhost:5000/metrics
```

**Note**: The Content-Type header for successful responses is determined by the metrics collector's `get_content_type()` method. The disabled response returns plain text.

---

### GET /api/v1/health

API-specific health check endpoint with detailed service information.

**Endpoint**: `GET /api/v1/health`

**Response**:

**Status**: `200 OK`

**Body**:
```json
{
  "status": "healthy",
  "gpu_available": true,
  "model_loaded": true,
  "timestamp": "2024-01-15T10:00:00.000000+00:00",
  "service": "AutoVoice API",
  "endpoints": {
    "synthesize": true,
    "process_audio": true,
    "analyze": true
  },
  "dependencies": {
    "numpy": true,
    "torch": true,
    "torchaudio": true
  }
}
```

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Overall API health status: `"healthy"` or `"degraded"` |
| `gpu_available` | boolean | Whether GPU/CUDA is available |
| `model_loaded` | boolean | Whether the voice model is loaded |
| `timestamp` | string | ISO 8601 timestamp with timezone |
| `service` | string | Service identifier (always `"AutoVoice API"`) |
| `endpoints` | object | Availability of API endpoints |
| `endpoints.synthesize` | boolean | Whether TTS synthesis is available |
| `endpoints.process_audio` | boolean | Whether audio processing is available |
| `endpoints.analyze` | boolean | Whether audio analysis is available |
| `dependencies` | object | Status of required Python dependencies |
| `dependencies.numpy` | boolean | Whether numpy is available |
| `dependencies.torch` | boolean | Whether PyTorch is available |
| `dependencies.torchaudio` | boolean | Whether torchaudio is available |

**Status Values**:
- `"healthy"`: All critical components (inference engine, audio processor) are operational
- `"degraded"`: One or more critical components are unavailable

**Comparison with `/health`**:
- `/health` - General application health with system metrics (memory, CPU, GPU details)
- `/api/v1/health` - API-specific health with endpoint capabilities and dependencies

**Example**:

```bash
curl http://localhost:5000/api/v1/health
```

---

### POST /synthesize

Synthesize speech from text.

**Endpoint**: `POST /api/v1/synthesize`

**Content-Type**: `application/json`

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Text to synthesize (max 5000 characters) |
| `speaker_id` | integer | Yes | Speaker ID for multi-speaker models (0-based) |
| `voice_config` | object | No | Voice synthesis parameters (optional) |
| `voice_config.temperature` | float | No | Sampling temperature (not currently applied) |
| `voice_config.speed` | float | No | Speech speed multiplier (not currently applied) |
| `voice_config.pitch` | float | No | Pitch adjustment (not currently applied) |

#### Request Example

**cURL**:
```bash
curl -X POST http://localhost:5000/api/v1/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, world!",
    "speaker_id": 0,
    "voice_config": {
      "temperature": 0.8,
      "speed": 1.0,
      "pitch": 1.0
    }
  }'
```

**Python**:
```python
import requests
import base64

url = "http://localhost:5000/api/v1/synthesize"
data = {
    "text": "Hello, world!",
    "speaker_id": 0,
    "voice_config": {
        "temperature": 0.8,
        "speed": 1.0,
        "pitch": 1.0
    }
}

response = requests.post(url, json=data)
result = response.json()

# Decode audio
audio_bytes = base64.b64decode(result['audio'])
with open('output.wav', 'wb') as f:
    f.write(audio_bytes)

print(f"Duration: {result['duration']:.2f} seconds")
```

**JavaScript**:
```javascript
const data = {
  text: 'Hello, world!',
  speaker_id: 0,
  voice_config: {
    temperature: 0.8,
    speed: 1.0,
    pitch: 1.0
  }
};

fetch('http://localhost:5000/api/v1/synthesize', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(data)
})
.then(response => response.json())
.then(result => {
  // Decode base64 audio
  const audioBytes = atob(result.audio);
  const audioArray = new Uint8Array(audioBytes.length);
  for (let i = 0; i < audioBytes.length; i++) {
    audioArray[i] = audioBytes.charCodeAt(i);
  }

  const blob = new Blob([audioArray], { type: 'audio/wav' });
  const audioUrl = URL.createObjectURL(blob);

  console.log('Duration:', result.duration);
  console.log('Audio URL:', audioUrl);
});
```

#### Response

**Status**: `200 OK`

**Body**:
```json
{
  "status": "success",
  "audio": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA...",
  "format": "wav",
  "sample_rate": 22050,
  "duration": 1.5,
  "metadata": {
    "text_length": 13,
    "speaker_id": 0,
    "voice_config": {
      "temperature": 0.8,
      "speed": 1.0,
      "pitch": 1.0
    }
  }
}
```

#### Error Responses

**Invalid speaker_id**:
```json
{
  "error": "Invalid speaker_id. Must be between 0 and 0"
}
```
**Status**: `404 Not Found`

**Text too long**:
```json
{
  "error": "Text too long. Maximum 5000 characters allowed"
}
```
**Status**: `400 Bad Request`

**Service unavailable**:
```json
{
  "error": "Voice synthesis service unavailable",
  "message": "Inference engine not initialized"
}
```
**Status**: `503 Service Unavailable`

---

### POST /process_audio

Process audio file or data.

**Endpoint**: `POST /api/v1/process_audio`

**Content-Type**: `multipart/form-data` or `application/json`

#### Request Parameters

**Option 1: File Upload (multipart/form-data)**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `audio` | file | Yes | Audio file (WAV, MP3, FLAC, OGG) |
| `processing_config` | JSON string | No | Processing parameters |

**Option 2: Base64 Audio (application/json)**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `audio_data` | string | Yes | Base64-encoded audio data |
| `processing_config` | object | No | Processing parameters |

**Processing Config Options**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_vad` | boolean | false | Voice activity detection |
| `enable_denoising` | boolean | false | Noise reduction |
| `enable_pitch_extraction` | boolean | false | Extract pitch contour |

#### Request Example

**cURL (File Upload)**:
```bash
curl -X POST http://localhost:5000/api/v1/process_audio \
  -F "audio=@input.wav" \
  -F 'processing_config={"enable_vad": true, "enable_pitch_extraction": true}'
```

**Python (File Upload)**:
```python
import requests

url = "http://localhost:5000/api/v1/process_audio"
files = {"audio": open("input.wav", "rb")}
data = {
    "processing_config": '{"enable_vad": true, "enable_pitch_extraction": true}'
}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Duration: {result['metadata']['duration']:.2f} seconds")
if 'pitch' in result['features']:
    print(f"Mean pitch: {result['features']['pitch']['mean_hz']:.2f} Hz")
```

**Python (Base64 Audio)**:
```python
import requests
import base64

url = "http://localhost:5000/api/v1/process_audio"

with open("input.wav", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode('utf-8')

data = {
    "audio_data": audio_base64,
    "processing_config": {
        "enable_vad": True,
        "enable_pitch_extraction": True
    }
}

response = requests.post(url, json=data)
result = response.json()
```

#### Response

**Status**: `200 OK`

**Body**:
```json
{
  "status": "success",
  "metadata": {
    "duration": 3.5,
    "sample_rate": 22050,
    "channels": 1,
    "format": "wav"
  },
  "features": {
    "vad": {
      "voice_activity_ratio": 0.85,
      "speech_segments": [
        {"start": 0.2, "end": 1.5},
        {"start": 1.8, "end": 3.3}
      ]
    },
    "pitch": {
      "mean_hz": 220.0,
      "std_hz": 35.2,
      "min_hz": 180.0,
      "max_hz": 280.0,
      "contour": [220.5, 225.3, 230.1, "..."]
    }
  }
}
```

#### Error Responses

**Invalid audio format**:
```json
{
  "error": "Invalid file format. Allowed: wav, mp3, flac, ogg"
}
```
**Status**: `400 Bad Request`

**Audio too long**:
```json
{
  "error": "Audio too long. Maximum duration: 600 seconds"
}
```
**Status**: `400 Bad Request`

---

## 3. System Information Endpoints

### POST /api/v1/analyze

Quick audio analysis without full processing.

**Endpoint**: `POST /api/v1/analyze`

**Content-Type**: `application/json`

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `audio_data` | string | Yes | Base64-encoded audio data |

#### Response

**Status**: `200 OK`

**Body**:
```json
{
  "status": "success",
  "analysis": {
    "duration": 3.5,
    "sample_rate": 22050,
    "channels": 1,
    "samples": 77175,
    "statistics": {
      "mean": 0.0012,
      "std": 0.145,
      "min": -0.98,
      "max": 0.97,
      "rms": 0.145
    },
    "pitch": {
      "mean_hz": 220.5,
      "std_hz": 35.2,
      "min_hz": 180.0,
      "max_hz": 280.0
    }
  }
}
```

#### Error Responses

**No audio data**:
```json
{
  "error": "No audio data provided"
}
```
**Status**: `400 Bad Request`

**Service unavailable**:
```json
{
  "error": "Audio analysis service unavailable",
  "message": "torchaudio not installed"
}
```
**Status**: `503 Service Unavailable`

**Example**:

```bash
# Encode audio to base64
AUDIO_BASE64=$(base64 -w 0 audio.wav)

curl -X POST http://localhost:5000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d "{\"audio_data\": \"$AUDIO_BASE64\"}"
```

---

### GET /api/v1/models/info

Get information about available models and their capabilities.

**Endpoint**: `GET /api/v1/models/info`

#### Response

**Status**: `200 OK`

**Body**:
```json
{
  "status": "available",
  "models": [
    {
      "name": "vits",
      "version": "1.0",
      "type": "tts",
      "capabilities": {
        "multi_speaker": true,
        "num_speakers": 100,
        "style_transfer": false,
        "tensorrt": false,
        "device": "cuda:0"
      },
      "parameters": {
        "max_length": 1000,
        "temperature_range": [0.1, 2.0],
        "speed_range": [0.5, 2.0],
        "pitch_range": [-12, 12]
      }
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:5000/api/v1/models/info
```

---

### GET /api/v1/config

Get current API configuration (sanitized).

**Endpoint**: `GET /api/v1/config`

#### Response

**Status**: `200 OK`

**Body**:
```json
{
  "audio": {
    "sample_rate": 22050,
    "channels": 1,
    "formats": ["wav", "mp3", "flac", "ogg"]
  },
  "limits": {
    "max_text_length": 1000,
    "max_audio_duration": 600,
    "max_file_size": 16777216
  },
  "processing": {
    "vad_enabled": true,
    "denoising_enabled": true,
    "pitch_extraction_enabled": true
  }
}
```

**Example**:

```bash
curl http://localhost:5000/api/v1/config
```

---

### POST /api/v1/config

Update voice synthesis parameters (runtime only, not persistent).

**Endpoint**: `POST /api/v1/config`

**Content-Type**: `application/json`

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `temperature` | number | No | Synthesis temperature (0.1-2.0) |
| `speed` | number | No | Speech speed multiplier (0.5-2.0) |
| `pitch` | number | No | Pitch shift in semitones (-12 to 12) |
| `speaker_id` | integer | No | Default speaker ID |

**Note**: Only the parameters listed above can be updated. Changes are runtime-only and not persisted.

#### Response

**Status**: `200 OK`

**Body**:
```json
{
  "status": "success",
  "message": "Configuration updated",
  "updates": {
    "temperature": 0.8,
    "speed": 1.2
  }
}
```

#### Error Responses

**No valid parameters**:
```json
{
  "error": "No valid configuration parameters provided"
}
```
**Status**: `400 Bad Request`

**Example**:

```bash
curl -X POST http://localhost:5000/api/v1/config \
  -H "Content-Type: application/json" \
  -d '{"temperature": 0.8, "speed": 1.2}'
```

---

### GET /api/v1/speakers

Get list of available speakers.

**Endpoint**: `GET /api/v1/speakers`

#### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `language` | string | No | Filter by language code (e.g., "en", "es") |

#### Response

**Status**: `200 OK`

**Body**:
```json
[
  {
    "id": 0,
    "name": "Speaker 0",
    "language": "en",
    "gender": "neutral",
    "description": "Generated speaker voice 0"
  },
  {
    "id": 1,
    "name": "Speaker 1",
    "language": "en",
    "gender": "neutral",
    "description": "Generated speaker voice 1"
  }
]
```

**Example**:

```bash
# Get all speakers
curl http://localhost:5000/api/v1/speakers

# Filter by language
curl "http://localhost:5000/api/v1/speakers?language=en"
```

---

### GET /api/v1/gpu_status

Get GPU status and utilization information.

**Endpoint**: `GET /api/v1/gpu_status`

#### Response

**Status**: `200 OK`

**Body (CUDA available)**:
```json
{
  "cuda_available": true,
  "device": "cuda",
  "device_count": 1,
  "device_name": "NVIDIA GeForce RTX 3090",
  "memory_total": 25769803776,
  "memory_allocated": 4294967296,
  "memory_reserved": 5368709120,
  "memory_free": 21474836480
}
```

**Body (CUDA unavailable)**:
```json
{
  "cuda_available": false,
  "device": "cpu",
  "device_count": 0
}
```

**Example**:

```bash
curl http://localhost:5000/api/v1/gpu_status
```

---

## 4. Voice Conversion Endpoints

### POST /voice/clone

Create a voice profile from an audio sample.

**Endpoint**: `POST /api/v1/voice/clone`

**Content-Type**: `multipart/form-data`

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `audio` | file | Yes | Audio file (WAV, MP3, FLAC, OGG) containing voice sample |
| `user_id` | string | Yes | Unique identifier for the user |
| `profile_name` | string | No | Human-readable name for the profile |
| `description` | string | No | Optional description of the voice profile |

**Audio Requirements**:
- **Duration**: 30-60 seconds recommended
- **Quality**: Clean vocals, minimal background noise
- **SNR**: >10 dB recommended (>15 dB optimal)
- **Content**: Singing or speaking in natural voice

#### Request Example

**cURL**:
```bash
curl -X POST http://localhost:5000/api/v1/voice/clone \
  -F "audio=@my_voice.wav" \
  -F "user_id=user123" \
  -F "profile_name=My Singing Voice" \
  -F "description=Natural singing voice recorded in studio"
```

**Python**:
```python
import requests

url = "http://localhost:5000/api/v1/voice/clone"
files = {"audio": open("my_voice.wav", "rb")}
data = {
    "user_id": "user123",
    "profile_name": "My Singing Voice",
    "description": "Natural singing voice"
}

response = requests.post(url, files=files, data=data)
profile = response.json()

print(f"Profile ID: {profile['profile_id']}")
print(f"Vocal Range: {profile['vocal_range']['min_note']} - {profile['vocal_range']['max_note']}")
print(f"SNR: {profile['quality_metrics']['snr_db']:.2f} dB")
```

**JavaScript**:
```javascript
const formData = new FormData();
formData.append('audio', audioFile);
formData.append('user_id', 'user123');
formData.append('profile_name', 'My Singing Voice');

fetch('http://localhost:5000/api/v1/voice/clone', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(profile => {
  console.log('Profile ID:', profile.profile_id);
  console.log('Vocal Range:', `${profile.vocal_range.min_note} - ${profile.vocal_range.max_note}`);
  console.log('SNR:', profile.quality_metrics.snr_db.toFixed(2), 'dB');
});
```

#### Response

**Status**: `201 Created`

**Body**:
```json
{
  "profile_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user123",
  "profile_name": "My Singing Voice",
  "description": "Natural singing voice",
  "created_at": "2024-01-15T10:00:00Z",
  "audio_info": {
    "duration_seconds": 45.5,
    "sample_rate": 44100,
    "format": "wav",
    "file_size_bytes": 4000000
  },
  "vocal_range": {
    "min_pitch_hz": 196.0,
    "max_pitch_hz": 587.3,
    "min_note": "G3",
    "max_note": "D5",
    "range_semitones": 17
  },
  "quality_metrics": {
    "snr_db": 18.5,
    "quality_score": 0.92
  },
  "speaker_embedding": {
    "dimensions": 256,
    "norm": 12.45
  }
}
```

#### Error Responses

**Invalid audio duration**:
```json
{
  "error": "invalid_audio_duration",
  "message": "Audio sample too short. Minimum 15 seconds required.",
  "details": {
    "duration_seconds": 8.5,
    "minimum_required": 15,
    "recommended": "30-60 seconds"
  }
}
```
**Status**: `400 Bad Request`

**Poor audio quality**:
```json
{
  "error": "poor_audio_quality",
  "message": "Audio quality insufficient for voice cloning",
  "details": {
    "snr_db": 5.2,
    "minimum_required": 10,
    "issues": ["High background noise", "Low vocal clarity"]
  }
}
```
**Status**: `400 Bad Request`

---

### GET /voice/profiles

List all voice profiles for a user.

**Endpoint**: `GET /api/v1/voice/profiles`

#### Query Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `user_id` | string | Yes | - | User identifier |
| `page` | integer | No | 1 | Page number (1-based) |
| `per_page` | integer | No | 20 | Profiles per page (max 100) |
| `sort_by` | string | No | created_at | Sort field: `created_at`, `profile_name`, `quality_score` |
| `order` | string | No | desc | Sort order: `asc`, `desc` |

#### Request Example

**cURL**:
```bash
curl "http://localhost:5000/api/v1/voice/profiles?user_id=user123&page=1&per_page=10"
```

**Python**:
```python
import requests

url = "http://localhost:5000/api/v1/voice/profiles"
params = {
    "user_id": "user123",
    "page": 1,
    "per_page": 10,
    "sort_by": "created_at",
    "order": "desc"
}

response = requests.get(url, params=params)
result = response.json()

print(f"Total profiles: {result['total']}")
for profile in result['profiles']:
    print(f"  - {profile['profile_name']} ({profile['profile_id']})")
```

**JavaScript**:
```javascript
const params = new URLSearchParams({
  user_id: 'user123',
  page: '1',
  per_page: '10'
});

fetch(`http://localhost:5000/api/v1/voice/profiles?${params}`)
  .then(response => response.json())
  .then(result => {
    console.log('Total profiles:', result.total);
    result.profiles.forEach(profile => {
      console.log(`  - ${profile.profile_name} (${profile.profile_id})`);
    });
  });
```

#### Response

**Status**: `200 OK`

**Body**:
```json
{
  "profiles": [
    {
      "profile_id": "550e8400-e29b-41d4-a716-446655440000",
      "profile_name": "My Singing Voice",
      "created_at": "2024-01-15T10:00:00Z",
      "vocal_range": {
        "min_note": "G3",
        "max_note": "D5",
        "range_semitones": 17
      },
      "quality_metrics": {
        "snr_db": 18.5,
        "quality_score": 0.92
      }
    },
    {
      "profile_id": "660e8400-e29b-41d4-a716-446655440001",
      "profile_name": "Alto Voice",
      "created_at": "2024-01-14T15:30:00Z",
      "vocal_range": {
        "min_note": "F3",
        "max_note": "C5",
        "range_semitones": 18
      },
      "quality_metrics": {
        "snr_db": 16.2,
        "quality_score": 0.87
      }
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 10,
    "total": 2,
    "total_pages": 1
  }
}
```

---

### GET /voice/profiles/{profile_id}

Get detailed information about a specific voice profile.

**Endpoint**: `GET /api/v1/voice/profiles/{profile_id}`

#### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `profile_id` | string | UUID of the voice profile |

#### Request Example

**cURL**:
```bash
curl http://localhost:5000/api/v1/voice/profiles/550e8400-e29b-41d4-a716-446655440000
```

**Python**:
```python
import requests

profile_id = "550e8400-e29b-41d4-a716-446655440000"
url = f"http://localhost:5000/api/v1/voice/profiles/{profile_id}"

response = requests.get(url)
profile = response.json()

print(f"Profile: {profile['profile_name']}")
print(f"Vocal Range: {profile['vocal_range']['min_note']} - {profile['vocal_range']['max_note']}")
print(f"Quality Score: {profile['quality_metrics']['quality_score']:.2f}")
```

#### Response

**Status**: `200 OK`

**Body**:
```json
{
  "profile_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user123",
  "profile_name": "My Singing Voice",
  "description": "Natural singing voice",
  "created_at": "2024-01-15T10:00:00Z",
  "audio_info": {
    "duration_seconds": 45.5,
    "sample_rate": 44100,
    "format": "wav",
    "file_size_bytes": 4000000
  },
  "vocal_range": {
    "min_pitch_hz": 196.0,
    "max_pitch_hz": 587.3,
    "min_note": "G3",
    "max_note": "D5",
    "range_semitones": 17
  },
  "quality_metrics": {
    "snr_db": 18.5,
    "quality_score": 0.92
  },
  "speaker_embedding": {
    "dimensions": 256,
    "norm": 12.45
  },
  "usage_stats": {
    "total_conversions": 15,
    "last_used": "2024-01-15T12:00:00Z"
  }
}
```

#### Error Responses

**Profile not found**:
```json
{
  "error": "profile_not_found",
  "message": "Voice profile not found",
  "profile_id": "550e8400-e29b-41d4-a716-446655440000"
}
```
**Status**: `404 Not Found`

---

### DELETE /voice/profiles/{profile_id}

Delete a voice profile.

**Endpoint**: `DELETE /api/v1/voice/profiles/{profile_id}`

#### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `profile_id` | string | UUID of the voice profile |

#### Request Example

**cURL**:
```bash
curl -X DELETE http://localhost:5000/api/v1/voice/profiles/550e8400-e29b-41d4-a716-446655440000
```

**Python**:
```python
import requests

profile_id = "550e8400-e29b-41d4-a716-446655440000"
url = f"http://localhost:5000/api/v1/voice/profiles/{profile_id}"

response = requests.delete(url)
result = response.json()

print(f"Status: {result['status']}")
```

#### Response

**Status**: `200 OK`

**Body**:
```json
{
  "status": "success",
  "message": "Voice profile deleted successfully",
  "profile_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

#### Error Responses

**Profile not found**:
```json
{
  "error": "profile_not_found",
  "message": "Voice profile not found",
  "profile_id": "550e8400-e29b-41d4-a716-446655440000"
}
```
**Status**: `404 Not Found`

---

### POST /convert/song

Convert a song to a target voice.

**Endpoint**: `POST /api/v1/convert/song`

**Content-Type**: `multipart/form-data`

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `song` | file | Yes | - | Audio file to convert (WAV, MP3, FLAC, OGG) |
| `profile_id` | string | Yes | - | UUID of target voice profile |
| `vocal_volume` | float | No | 1.0 | Volume of converted vocals (0.0-2.0) |
| `instrumental_volume` | float | No | 0.9 | Volume of instrumental (0.0-2.0) |
| `pitch_shift_semitones` | integer | No | 0 | Pitch shift in semitones (�12) |
| `temperature` | float | No | 1.0 | Expressiveness control (0.5-2.0) |
| `quality_preset` | string | No | balanced | Quality preset: `fast`, `balanced`, `quality` |
| `return_stems` | boolean | No | false | Return separated vocals/instrumental |

> **Note**: The request parameter is `profile_id`. The response metadata includes `target_profile_id` for clarity.

**Quality Presets**:

| Preset | Speed | Quality | Use Case |
|--------|-------|---------|----------|
| `fast` | ~0.5x real-time | Good | Quick previews, testing |
| `balanced` | ~1x real-time | Very Good | General use, production |
| `quality` | ~2x real-time | Excellent | Final masters, critical quality |

#### Request Example

**cURL**:
```bash
curl -X POST http://localhost:5000/api/v1/convert/song \
  -F "song=@song.mp3" \
  -F "profile_id=550e8400-e29b-41d4-a716-446655440000" \
  -F "vocal_volume=1.0" \
  -F "instrumental_volume=0.9" \
  -F "pitch_shift_semitones=0" \
  -F "temperature=1.0" \
  -F "quality_preset=balanced" \
  -F "return_stems=true"
```

**Python**:
```python
import requests

url = "http://localhost:5000/api/v1/convert/song"
files = {"song": open("song.mp3", "rb")}
data = {
    "profile_id": "550e8400-e29b-41d4-a716-446655440000",
    "vocal_volume": 1.0,
    "instrumental_volume": 0.9,
    "pitch_shift_semitones": 0,
    "temperature": 1.0,
    "quality_preset": "balanced",
    "return_stems": True
}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Conversion ID: {result['conversion_id']}")
print(f"Status: {result['status']}")
```

**JavaScript**:
```javascript
const formData = new FormData();
formData.append('song', songFile);
formData.append('profile_id', '550e8400-e29b-41d4-a716-446655440000');
formData.append('vocal_volume', '1.0');
formData.append('instrumental_volume', '0.9');
formData.append('quality_preset', 'balanced');

fetch('http://localhost:5000/api/v1/convert/song', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(result => {
  console.log('Conversion ID:', result.conversion_id);
  console.log('Status:', result.status);
});
```

#### Response

**Status**: `202 Accepted`

**Body**:
```json
{
  "conversion_id": "conv-770e8400-e29b-41d4-a716-446655440002",
  "status": "queued",
  "message": "Song conversion started",
  "estimated_time_seconds": 90,
  "status_url": "/api/v1/convert/status/conv-770e8400-e29b-41d4-a716-446655440002"
}
```

**Response Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `conversion_id` | string | UUID of the conversion job |
| `status` | string | Initial status (always `queued` for async conversions) |
| `message` | string | Human-readable status message |
| `estimated_time_seconds` | integer | Estimated time to completion (seconds) |
| `status_url` | string | URL to poll for conversion status, see [GET /api/v1/convert/status/{conversion_id}](#get-convertstatusconversion_id) |

**Next Steps**:

1. **Poll for Status**: Use the `status_url` to check conversion progress. Recommended: exponential backoff starting at 1s, max 30s between requests.
   - See [GET /api/v1/convert/status/{conversion_id}](#get-convertstatusconversion_id) for detailed documentation

2. **Download Result**: When status is `succeeded`, use the `download_url` from the status response to retrieve the converted audio.
   - See [GET /api/v1/convert/download/{conversion_id}/converted.wav](#get-convertdownloadconversion_idconvertedwav) for detailed documentation

**Example Async Workflow**:

```python
import requests
import time

# Step 1: Submit conversion
response = requests.post(
    "http://localhost:5000/api/v1/convert/song",
    files={"song": open("song.mp3", "rb")},
    data={"profile_id": "550e8400-e29b-41d4-a716-446655440000"}
)
result = response.json()
conversion_id = result["conversion_id"]

# Step 2: Poll for completion
status_url = f"http://localhost:5000{result['status_url']}"
wait_time = 1

while True:
    status_response = requests.get(status_url)
    status_data = status_response.json()

    if status_data["status"] == "succeeded":
        download_url = status_data["download_url"]
        break
    elif status_data["status"] == "failed":
        raise Exception(f"Conversion failed: {status_data['message']}")

    time.sleep(wait_time)
    wait_time = min(wait_time * 1.5, 30)  # Exponential backoff

# Step 3: Download result
download_response = requests.get(f"http://localhost:5000{download_url}")
with open("converted.wav", "wb") as f:
    f.write(download_response.content)

print("Conversion complete!")
```

#### Error Responses

**Invalid profile**:
```json
{
  "error": "profile_not_found",
  "message": "Target voice profile not found",
  "target_profile_id": "550e8400-e29b-41d4-a716-446655440000"
}
```
**Status**: `404 Not Found`

**Invalid parameters**:
```json
{
  "error": "invalid_parameters",
  "message": "Invalid parameter values",
  "details": {
    "pitch_shift_semitones": "Must be between -12 and 12",
    "vocal_volume": "Must be between 0.0 and 2.0"
  }
}
```
**Status**: `400 Bad Request`

---

### GET /convert/status/{conversion_id}

Get the current status of a song conversion job. Use this endpoint to poll for completion after submitting a conversion with `POST /api/v1/convert/song`.

**Endpoint**: `GET /api/v1/convert/status/{conversion_id}`

#### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `conversion_id` | string | UUID of the conversion job (returned by POST /api/v1/convert/song) |

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `conversion_id` | string | UUID of the conversion job |
| `status` | string | Current status: `queued`, `running`, `succeeded`, `failed` |
| `progress` | integer | Completion percentage (0-100), only present when `status` is `running` |
| `stage` | string | Current processing stage, only present when `status` is `running` |
| `download_url` | string | URL to download converted audio, only present when `status` is `succeeded` |
| `error` | string | Error code, only present when `status` is `failed` |
| `message` | string | Human-readable error message, only present when `status` is `failed` |
| `created_at` | string | ISO 8601 timestamp when conversion was created |
| `updated_at` | string | ISO 8601 timestamp of last status update |
| `estimated_time_seconds` | integer | Estimated time to completion (seconds), only present when `status` is `queued` or `running` |

#### Response Examples

**Queued State**:

**Status**: `200 OK`

**Body**:
```json
{
  "conversion_id": "conv-770e8400-e29b-41d4-a716-446655440002",
  "status": "queued",
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-15T10:00:00Z",
  "estimated_time_seconds": 90
}
```

**Running State**:

**Status**: `200 OK`

**Body**:
```json
{
  "conversion_id": "conv-770e8400-e29b-41d4-a716-446655440002",
  "status": "running",
  "progress": 45,
  "stage": "voice_conversion",
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-15T10:00:35Z",
  "estimated_time_seconds": 50
}
```

**Processing Stages**:
- `vocal_separation` - Separating vocals from instrumental
- `pitch_extraction` - Extracting pitch contour
- `voice_conversion` - Converting voice characteristics
- `audio_synthesis` - Synthesizing final audio
- `mixing` - Mixing converted vocals with instrumental

**Succeeded State**:

**Status**: `200 OK`

**Body**:
```json
{
  "conversion_id": "conv-770e8400-e29b-41d4-a716-446655440002",
  "status": "succeeded",
  "download_url": "/api/v1/convert/download/conv-770e8400-e29b-41d4-a716-446655440002/converted.wav",
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-15T10:01:25Z"
}
```

**Failed State**:

**Status**: `200 OK`

**Body**:
```json
{
  "conversion_id": "conv-770e8400-e29b-41d4-a716-446655440002",
  "status": "failed",
  "error": "separation_failed",
  "message": "Vocal separation failed: audio quality insufficient",
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-15T10:00:45Z"
}
```

#### Request Examples

**cURL**:
```bash
curl http://localhost:5000/api/v1/convert/status/conv-770e8400-e29b-41d4-a716-446655440002
```

**Python**:
```python
import requests
import time

conversion_id = "conv-770e8400-e29b-41d4-a716-446655440002"
url = f"http://localhost:5000/api/v1/convert/status/{conversion_id}"

# Poll for completion with exponential backoff
wait_time = 1
max_wait = 30

while True:
    response = requests.get(url)
    result = response.json()

    status = result['status']
    print(f"Status: {status}")

    if status == 'running':
        progress = result.get('progress', 0)
        stage = result.get('stage', 'processing')
        print(f"  Progress: {progress}% - {stage}")

    if status == 'succeeded':
        print(f"  Download URL: {result['download_url']}")
        break
    elif status == 'failed':
        print(f"  Error: {result['error']} - {result['message']}")
        break

    # Exponential backoff
    time.sleep(wait_time)
    wait_time = min(wait_time * 1.5, max_wait)
```

**JavaScript**:
```javascript
const conversionId = 'conv-770e8400-e29b-41d4-a716-446655440002';
const url = `http://localhost:5000/api/v1/convert/status/${conversionId}`;

// Poll for completion with exponential backoff
async function pollStatus() {
  let waitTime = 1000; // Start with 1 second
  const maxWait = 30000; // Max 30 seconds

  while (true) {
    const response = await fetch(url);
    const result = await response.json();

    console.log(`Status: ${result.status}`);

    if (result.status === 'running') {
      console.log(`  Progress: ${result.progress}% - ${result.stage}`);
    }

    if (result.status === 'succeeded') {
      console.log(`  Download URL: ${result.download_url}`);
      return result.download_url;
    } else if (result.status === 'failed') {
      console.error(`  Error: ${result.error} - ${result.message}`);
      throw new Error(result.message);
    }

    // Exponential backoff
    await new Promise(resolve => setTimeout(resolve, waitTime));
    waitTime = Math.min(waitTime * 1.5, maxWait);
  }
}

pollStatus().then(downloadUrl => {
  console.log('Conversion complete:', downloadUrl);
}).catch(error => {
  console.error('Conversion failed:', error);
});
```

#### Error Responses

**Conversion not found**:
```json
{
  "error": "conversion_not_found",
  "message": "Conversion job not found",
  "conversion_id": "conv-770e8400-e29b-41d4-a716-446655440002"
}
```
**Status**: `404 Not Found`

**Invalid conversion ID format**:
```json
{
  "error": "invalid_id",
  "message": "Invalid conversion ID format",
  "conversion_id": "invalid-id"
}
```
**Status**: `400 Bad Request`

---

### GET /convert/download/{conversion_id}/converted.wav

Download the converted audio file for a completed conversion. Only available when the conversion status is `succeeded`.

**Endpoint**: `GET /api/v1/convert/download/{conversion_id}/converted.wav`

#### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `conversion_id` | string | UUID of the conversion job |

#### Response

**Status**: `200 OK`

**Headers**:
```
Content-Type: audio/wav
Content-Disposition: attachment; filename="converted.wav"
Content-Length: <file-size-bytes>
```

**Body**: Binary audio data (WAV format)

#### Request Examples

**cURL**:
```bash
# Download and save to file
curl http://localhost:5000/api/v1/convert/download/conv-770e8400-e29b-41d4-a716-446655440002/converted.wav \
  -o converted.wav

# Download with progress indicator
curl -# http://localhost:5000/api/v1/convert/download/conv-770e8400-e29b-41d4-a716-446655440002/converted.wav \
  -o converted.wav
```

**Python**:
```python
import requests

conversion_id = "conv-770e8400-e29b-41d4-a716-446655440002"
url = f"http://localhost:5000/api/v1/convert/download/{conversion_id}/converted.wav"

# Download with streaming to handle large files
response = requests.get(url, stream=True)
response.raise_for_status()

# Save to file with progress
total_size = int(response.headers.get('content-length', 0))
downloaded = 0

with open('converted.wav', 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                progress = (downloaded / total_size) * 100
                print(f"Download progress: {progress:.1f}%", end='\r')

print("\nDownload complete!")
```

**JavaScript (Node.js)**:
```javascript
const fs = require('fs');
const https = require('https');

const conversionId = 'conv-770e8400-e29b-41d4-a716-446655440002';
const url = `http://localhost:5000/api/v1/convert/download/${conversionId}/converted.wav`;

// Download with progress
const file = fs.createWriteStream('converted.wav');
https.get(url, (response) => {
  const totalSize = parseInt(response.headers['content-length'], 10);
  let downloaded = 0;

  response.on('data', (chunk) => {
    downloaded += chunk.length;
    file.write(chunk);

    const progress = (downloaded / totalSize) * 100;
    process.stdout.write(`Download progress: ${progress.toFixed(1)}%\r`);
  });

  response.on('end', () => {
    file.end();
    console.log('\nDownload complete!');
  });
}).on('error', (error) => {
  console.error('Download failed:', error);
});
```

**JavaScript (Browser)**:
```javascript
const conversionId = 'conv-770e8400-e29b-41d4-a716-446655440002';
const url = `http://localhost:5000/api/v1/convert/download/${conversionId}/converted.wav`;

// Download and trigger browser download
fetch(url)
  .then(response => {
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    return response.blob();
  })
  .then(blob => {
    // Create download link
    const downloadUrl = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = downloadUrl;
    a.download = 'converted.wav';
    document.body.appendChild(a);
    a.click();

    // Cleanup
    window.URL.revokeObjectURL(downloadUrl);
    document.body.removeChild(a);

    console.log('Download complete!');
  })
  .catch(error => {
    console.error('Download failed:', error);
  });
```

#### Error Responses

**Conversion not found**:
```json
{
  "error": "conversion_not_found",
  "message": "Conversion job not found",
  "conversion_id": "conv-770e8400-e29b-41d4-a716-446655440002"
}
```
**Status**: `404 Not Found`

**Conversion not ready**:
```json
{
  "error": "not_ready",
  "message": "Conversion is not complete. Current status: running",
  "status": "running",
  "conversion_id": "conv-770e8400-e29b-41d4-a716-446655440002"
}
```
**Status**: `409 Conflict`

**Conversion failed**:
```json
{
  "error": "conversion_error",
  "message": "Conversion failed and no output is available",
  "conversion_id": "conv-770e8400-e29b-41d4-a716-446655440002"
}
```
**Status**: `409 Conflict`

---

> **Asynchronous Workflow**: The `/api/v1/convert/song` endpoint supports asynchronous conversion with status polling and separate download. After submitting a conversion, use the `status_url` to poll for completion (recommended: exponential backoff starting at 1s, max 30s). When status is `succeeded`, use the `download_url` to retrieve the converted audio. See [GET /api/v1/convert/status/{conversion_id}](#get-convertstatusconversion_id) and [GET /api/v1/convert/download/{conversion_id}/converted.wav](#get-convertdownloadconversion_idconvertedwav) for details.

---

## 5. WebSocket/Socket.IO API

Real-time communication using Socket.IO for streaming audio, synthesis, and conversion progress updates.

**Connection**: Socket.IO connects to the same port as the HTTP API (default: 5000)

**Endpoint**: `ws://<host>:<port>/socket.io/` (or use Socket.IO client library)

**Note**: AutoVoice uses Flask-SocketIO, not raw WebSocket. Use a Socket.IO client library for proper connection handling.

### Connection Examples

**JavaScript (using socket.io-client)**:
```javascript
// Install: npm install socket.io-client
import io from 'socket.io-client';

const socket = io('http://localhost:5000');

socket.on('connect', () => {
  console.log('Connected to AutoVoice Socket.IO');
  console.log('Socket ID:', socket.id);
});

socket.on('disconnect', () => {
  console.log('Disconnected from server');
});

socket.on('error', (error) => {
  console.error('Socket error:', error);
});
```

**Python (using python-socketio)**:
```python
# Install: pip install python-socketio[client]
import socketio

sio = socketio.Client()

@sio.event
def connect():
    print('Connected to AutoVoice Socket.IO')
    print('Session ID:', sio.sid)

@sio.event
def disconnect():
    print('Disconnected from server')

@sio.event
def error(data):
    print('Socket error:', data)

# Connect to server
sio.connect('http://localhost:5000')
```

### Socket.IO Events

#### Client → Server Events

##### `audio_stream`
Stream audio data for real-time processing.

**Payload**:
```json
{
  "audio_data": "<base64-encoded-audio>",
  "sample_rate": 44100,
  "chunk_size": 1024
}
```

**Response Event**: `audio_processed`

**Example (JavaScript)**:
```javascript
const audioBase64 = btoa(String.fromCharCode(...audioBuffer));
socket.emit('audio_stream', {
  audio_data: audioBase64,
  sample_rate: 44100,
  chunk_size: 1024
});

socket.on('audio_processed', (data) => {
  console.log('Audio processed:', data);
});
```

---

##### `synthesize_stream`
Stream text-to-speech synthesis.

**Payload**:
```json
{
  "text": "Hello, world!",
  "speaker_id": 0
}
```

**Response Event**: `synthesis_complete`

**Example (JavaScript)**:
```javascript
socket.emit('synthesize_stream', {
  text: 'Hello, this is a streaming synthesis test.',
  speaker_id: 0
});

socket.on('synthesis_complete', (data) => {
  console.log('Synthesis complete');
  // data.audio contains base64-encoded audio
  const audioBlob = base64ToBlob(data.audio, 'audio/wav');
  playAudio(audioBlob);
});
```

**Example (Python)**:
```python
@sio.on('synthesis_complete')
def on_synthesis_complete(data):
    print('Synthesis complete')
    audio_base64 = data['audio']
    audio_bytes = base64.b64decode(audio_base64)
    # Save or play audio_bytes

sio.emit('synthesize_stream', {
    'text': 'Hello, this is a streaming synthesis test.',
    'speaker_id': 0
})
```

---

##### `convert_song_stream`
Stream song conversion with real-time progress updates.

**Payload**:
```json
{
  "conversion_id": "optional-custom-id",
  "song_data": "<base64-encoded-audio>",
  "song_mime": "audio/wav",
  "song_filename": "song.wav",
  "target_profile_id": "profile-uuid",
  "vocal_volume": 1.0,
  "instrumental_volume": 0.9,
  "return_stems": false
}
```

**Response Events**:
- `conversion_progress` (multiple times during processing)
- `conversion_complete` (on success)
- `conversion_error` (on failure)

**Example (JavaScript)**:
```javascript
const conversionId = 'my-conversion-' + Date.now();

socket.emit('convert_song_stream', {
  conversion_id: conversionId,
  song_data: songBase64,
  song_mime: 'audio/wav',
  song_filename: 'song.wav',
  target_profile_id: 'profile-abc123',
  vocal_volume: 1.0,
  instrumental_volume: 0.9,
  return_stems: false
});

socket.on('conversion_progress', (data) => {
  console.log(`Progress: ${data.progress}% - ${data.stage}`);
  updateProgressBar(data.progress);
});

socket.on('conversion_complete', (data) => {
  console.log('Conversion complete!');
  console.log('Converted audio:', data.converted_audio_base64);
});

socket.on('conversion_error', (data) => {
  console.error('Conversion failed:', data.error);
});
```

**Example (Python)**:
```python
import base64

@sio.on('conversion_progress')
def on_progress(data):
    print(f"Progress: {data['progress']}% - {data['stage']}")

@sio.on('conversion_complete')
def on_complete(data):
    print('Conversion complete!')
    audio_bytes = base64.b64decode(data['converted_audio_base64'])
    with open('converted.wav', 'wb') as f:
        f.write(audio_bytes)

@sio.on('conversion_error')
def on_error(data):
    print(f"Conversion failed: {data['error']}")

# Read and encode song
with open('song.wav', 'rb') as f:
    song_bytes = f.read()
    song_base64 = base64.b64encode(song_bytes).decode('utf-8')

sio.emit('convert_song_stream', {
    'conversion_id': 'my-conversion-123',
    'song_data': song_base64,
    'song_mime': 'audio/wav',
    'song_filename': 'song.wav',
    'target_profile_id': 'profile-abc123',
    'vocal_volume': 1.0,
    'instrumental_volume': 0.9,
    'return_stems': False
})
```

---

##### `audio_analysis`
Analyze audio in real-time.

**Payload**:
```json
{
  "audio_data": "<base64-encoded-audio>",
  "sample_rate": 44100
}
```

**Response Event**: `analysis_complete`

---

##### `get_status`
Get server status and capabilities.

**Payload**: None

**Response Event**: `status`

**Example**:
```javascript
socket.emit('get_status');

socket.on('status', (data) => {
  console.log('Server capabilities:', data.capabilities);
  console.log('Performance metrics:', data.metrics);
});
```

---

##### `join` / `leave`
Join or leave a room for targeted event broadcasting.

**Payload**:
```json
{
  "room": "room-name"
}
```

**Response Events**: `joined` / `left`

---

#### Server → Client Events

##### `audio_processed`
Audio stream processing result.

**Payload**:
```json
{
  "status": "success",
  "results": { ... }
}
```

---

##### `synthesis_complete`
TTS synthesis completed.

**Payload**:
```json
{
  "audio": "<base64-encoded-audio>",
  "text": "synthesized text",
  "speaker_id": 0
}
```

---

##### `conversion_progress`
Song conversion progress update.

**Payload**:
```json
{
  "conversion_id": "my-conversion-123",
  "progress": 65,
  "stage": "voice_conversion",
  "timestamp": 1234567890.123
}
```

---

##### `conversion_complete`
Song conversion completed successfully.

**Payload**:
```json
{
  "conversion_id": "my-conversion-123",
  "converted_audio_base64": "<base64-audio>",
  "vocals_base64": "<base64-audio>",
  "instrumental_base64": "<base64-audio>",
  "quality_metrics": { ... }
}
```

---

##### `conversion_error`
Song conversion failed.

**Payload**:
```json
{
  "conversion_id": "my-conversion-123",
  "error": "Error message",
  "code": "ERROR_CODE"
}
```

---

##### `analysis_complete`
Audio analysis completed.

**Payload**:
```json
{
  "analysis": {
    "pitch": { ... },
    "energy": { ... }
  },
  "sample_rate": 44100
}
```

---

##### `status`
Server status and capabilities.

**Payload**:
```json
{
  "capabilities": {
    "tts": true,
    "voice_cloning": true,
    "singing_conversion": true
  },
  "metrics": {
    "active_connections": 5,
    "gpu_utilization": 45.2
  },
  "timestamp": 1234567890.123
}
```

---

##### `error`
General error event.

**Payload**:
```json
{
  "message": "Error description"
}
```

---

## 6. Error Codes

### General Error Codes

| Code | HTTP Status | Description | Resolution |
|------|-------------|-------------|------------|
| `invalid_request` | 400 | Malformed request | Check request format |
| `missing_parameter` | 400 | Required parameter missing | Include all required fields |
| `invalid_parameter` | 400 | Invalid parameter value | Check parameter constraints |
| `resource_not_found` | 404 | Requested resource not found | Verify resource ID |
| `method_not_allowed` | 405 | HTTP method not supported | Use correct HTTP method |
| `internal_error` | 500 | Internal server error | Contact support |
| `service_unavailable` | 503 | Service temporarily unavailable | Retry later |

### TTS Error Codes

| Code | HTTP Status | Description | Resolution |
|------|-------------|-------------|------------|
| `text_too_long` | 400 | Text exceeds maximum length | Reduce text to <5000 characters |
| `invalid_speaker_id` | 404 | Speaker ID not available | Use valid speaker ID (0-based) |
| `synthesis_failed` | 500 | Voice synthesis failed | Check model status, retry |
| `audio_processing_failed` | 500 | Audio processing error | Check audio format, retry |

### Voice Conversion Error Codes

| Code | HTTP Status | Description | Resolution |
|------|-------------|-------------|------------|
| `profile_not_found` | 404 | Voice profile not found | Verify profile ID |
| `invalid_audio_format` | 400 | Unsupported audio format | Use WAV, MP3, FLAC, or OGG |
| `invalid_audio_duration` | 400 | Audio duration invalid | Use 30-60s for profiles |
| `poor_audio_quality` | 400 | Audio quality insufficient | Improve SNR, reduce noise |
| `separation_failed` | 500 | Vocal separation failed | Check audio quality |
| `pitch_extraction_failed` | 500 | Pitch extraction failed | Ensure clear vocals |
| `conversion_error` | 500 | Voice conversion failed | Check model status |
| `file_not_found` | 404 | Result file not found | Check conversion status |
| `conversion_incomplete` | 409 | Conversion not complete | Wait for completion |

### Error Response Format

All errors follow this format:

```json
{
  "error": "error_code",
  "message": "Human-readable error message",
  "details": {
    "field": "Additional context",
    "suggestions": ["Actionable suggestions"]
  }
}
```

---

## 7. Python SDK

### Installation

```bash
pip install autovoice-sdk
```

### Usage Examples

#### TTS

```python
from autovoice import AutoVoiceClient

# Initialize client
client = AutoVoiceClient(base_url="http://localhost:5000")

# Synthesize speech
audio = client.synthesize(
    text="Hello, world!",
    speaker_id=0,
    temperature=0.8
)

# Save to file
audio.save("output.wav")

# Process audio
result = client.process_audio(
    audio_path="input.wav",
    enable_vad=True,
    enable_pitch_extraction=True
)

print(f"Duration: {result.duration:.2f}s")
print(f"Mean pitch: {result.features.pitch.mean_hz:.2f} Hz")
```

#### Voice Conversion

```python
from autovoice import AutoVoiceClient

# Initialize client
client = AutoVoiceClient(base_url="http://localhost:5000")

# Create voice profile
profile = client.voice.create_profile(
    audio_path="my_voice.wav",
    user_id="user123",
    profile_name="My Singing Voice"
)

print(f"Profile ID: {profile.profile_id}")
print(f"Vocal Range: {profile.vocal_range.min_note} - {profile.vocal_range.max_note}")

# List profiles
profiles = client.voice.list_profiles(user_id="user123")
for p in profiles:
    print(f"  - {p.profile_name} ({p.profile_id})")

# Convert song
conversion = client.voice.convert_song(
    song_path="song.mp3",
    target_profile_id=profile.profile_id,
    quality_preset="balanced",
    return_stems=True
)

# Wait for completion
result = conversion.wait_for_completion(progress_callback=lambda p: print(f"Progress: {p}%"))

# Download result
result.download_converted("converted.wav")
result.download_vocals("vocals.wav")
result.download_instrumental("instrumental.wav")

# Quality metrics
metrics = result.quality_metrics
print(f"Pitch RMSE: {metrics.pitch_accuracy.rmse_hz:.2f} Hz")
print(f"Speaker Similarity: {metrics.speaker_similarity.cosine_similarity:.2f}")
```

### SDK Reference

#### AutoVoiceClient

```python
class AutoVoiceClient:
    def __init__(self, base_url: str = "http://localhost:5000"):
        """Initialize AutoVoice client"""

    def health(self) -> HealthStatus:
        """Get system health status"""

    def synthesize(
        self,
        text: str,
        speaker_id: int,
        temperature: float = 0.8,
        speed: float = 1.0,
        pitch: float = 1.0
    ) -> Audio:
        """Synthesize speech from text"""

    def process_audio(
        self,
        audio_path: str,
        enable_vad: bool = False,
        enable_denoising: bool = False,
        enable_pitch_extraction: bool = False
    ) -> ProcessingResult:
        """Process audio file"""

    @property
    def voice(self) -> VoiceConversion:
        """Voice conversion API"""
```

#### VoiceConversion

```python
class VoiceConversion:
    def create_profile(
        self,
        audio_path: str,
        user_id: str,
        profile_name: Optional[str] = None,
        description: Optional[str] = None
    ) -> VoiceProfile:
        """Create voice profile"""

    def list_profiles(
        self,
        user_id: str,
        page: int = 1,
        per_page: int = 20
    ) -> List[VoiceProfile]:
        """List voice profiles"""

    def get_profile(self, profile_id: str) -> VoiceProfile:
        """Get voice profile details"""

    def delete_profile(self, profile_id: str) -> bool:
        """Delete voice profile"""

    def convert_song(
        self,
        song_path: str,
        target_profile_id: str,
        vocal_volume: float = 1.0,
        instrumental_volume: float = 0.9,
        pitch_shift_semitones: int = 0,
        temperature: float = 1.0,
        quality_preset: str = "balanced",
        return_stems: bool = False
    ) -> Conversion:
        """Convert song to target voice"""
```

---

## 8. JavaScript SDK

### Installation

```bash
npm install autovoice-sdk
```

### Usage Examples

#### TTS

```javascript
import { AutoVoiceClient } from 'autovoice-sdk';

// Initialize client
const client = new AutoVoiceClient({ baseUrl: 'http://localhost:5000' });

// Synthesize speech
const audio = await client.synthesize({
  text: 'Hello, world!',
  speakerId: 0,
  temperature: 0.8
});

// Download audio
audio.download('output.wav');

// Process audio
const result = await client.processAudio({
  audioFile: inputFile,
  enableVAD: true,
  enablePitchExtraction: true
});

console.log(`Duration: ${result.duration.toFixed(2)}s`);
console.log(`Mean pitch: ${result.features.pitch.meanHz.toFixed(2)} Hz`);
```

#### Voice Conversion

```javascript
import { AutoVoiceClient } from 'autovoice-sdk';

// Initialize client
const client = new AutoVoiceClient({ baseUrl: 'http://localhost:5000' });

// Create voice profile
const profile = await client.voice.createProfile({
  audioFile: voiceFile,
  userId: 'user123',
  profileName: 'My Singing Voice'
});

console.log(`Profile ID: ${profile.profileId}`);
console.log(`Vocal Range: ${profile.vocalRange.minNote} - ${profile.vocalRange.maxNote}`);

// List profiles
const profiles = await client.voice.listProfiles({ userId: 'user123' });
profiles.forEach(p => {
  console.log(`  - ${p.profileName} (${p.profileId})`);
});

// Convert song
const conversion = await client.voice.convertSong({
  songFile: songFile,
  targetProfileId: profile.profileId,
  qualityPreset: 'balanced',
  returnStems: true
});

// Wait for completion with progress
const result = await conversion.waitForCompletion({
  onProgress: (progress) => console.log(`Progress: ${progress}%`)
});

// Download results
await result.downloadConverted('converted.wav');
await result.downloadVocals('vocals.wav');
await result.downloadInstrumental('instrumental.wav');

// Quality metrics
const metrics = result.qualityMetrics;
console.log(`Pitch RMSE: ${metrics.pitchAccuracy.rmseHz.toFixed(2)} Hz`);
console.log(`Speaker Similarity: ${metrics.speakerSimilarity.cosineSimilarity.toFixed(2)}`);
```

### SDK Reference

#### AutoVoiceClient

```typescript
class AutoVoiceClient {
  constructor(options: { baseUrl: string });

  health(): Promise<HealthStatus>;

  synthesize(params: {
    text: string;
    speakerId: number;
    temperature?: number;
    speed?: number;
    pitch?: number;
  }): Promise<Audio>;

  processAudio(params: {
    audioFile: File;
    enableVAD?: boolean;
    enableDenoising?: boolean;
    enablePitchExtraction?: boolean;
  }): Promise<ProcessingResult>;

  get voice(): VoiceConversion;
}
```

#### VoiceConversion

```typescript
class VoiceConversion {
  createProfile(params: {
    audioFile: File;
    userId: string;
    profileName?: string;
    description?: string;
  }): Promise<VoiceProfile>;

  listProfiles(params: {
    userId: string;
    page?: number;
    perPage?: number;
  }): Promise<VoiceProfile[]>;

  getProfile(profileId: string): Promise<VoiceProfile>;

  deleteProfile(profileId: string): Promise<boolean>;

  convertSong(params: {
    songFile: File;
    targetProfileId: string;
    vocalVolume?: number;
    instrumentalVolume?: number;
    pitchShiftSemitones?: number;
    temperature?: number;
    qualityPreset?: 'fast' | 'balanced' | 'quality';
    returnStems?: boolean;
  }): Promise<Conversion>;
}
```

---

## 9. Best Practices

### General Guidelines

1. **Error Handling**
   - Always check response status codes
   - Implement retry logic with exponential backoff
   - Log errors for debugging
   - Handle rate limiting gracefully

2. **Authentication** (Future)
   - Store API keys securely
   - Rotate keys regularly
   - Use environment variables for keys
   - Never expose keys in client-side code

3. **Performance**
   - Use quality presets appropriately
   - Enable caching where possible
   - Batch operations when available
   - Monitor resource usage

### TTS Best Practices

1. **Text Input**
   - Keep text under 5000 characters
   - Use proper punctuation for natural prosody
   - Split long text into sentences
   - Avoid special characters that may not render

2. **Audio Output**
   - Cache synthesized audio for repeated use
   - Use appropriate sample rates
   - Monitor audio quality metrics

### Voice Conversion Best Practices

1. **Voice Profile Creation**
   - Use 30-60 second audio samples
   - Ensure clean vocals with minimal background noise
   - Aim for SNR >15 dB
   - Record in natural voice without effects
   - Include variety in pitch range

2. **Song Conversion**
   - Use appropriate quality preset for use case
   - Test with fast preset before quality preset
   - Monitor quality metrics (pitch RMSE, similarity)
   - Keep pitch shift within �12 semitones
   - Use return_stems for debugging

3. **Quality Optimization**
   - Target pitch RMSE <10 Hz
   - Target speaker similarity >0.85
   - Monitor processing time vs quality trade-offs
   - Use balanced preset for most cases

4. **Resource Management**
   - Monitor GPU memory usage
   - Clean up completed conversions
   - Use batch processing for multiple songs
   - Implement queue management for high load

### Production Deployment

1. **Infrastructure**
   - Use load balancing for horizontal scaling
   - Implement health check monitoring
   - Set up logging and metrics
   - Use GPU instances for optimal performance

2. **Security**
   - Implement authentication and authorization
   - Use HTTPS for all communications
   - Validate and sanitize all inputs
   - Set rate limits per user/IP

3. **Monitoring**
   - Track API response times
   - Monitor error rates
   - Track resource utilization
   - Set up alerting for critical issues

4. **Caching**
   - Cache voice profiles
   - Cache frequently used conversions
   - Implement CDN for audio delivery
   - Use Redis for distributed caching

### Rate Limiting (Recommended)

```yaml
# Example rate limit configuration
rate_limits:
  tts:
    synthesize: 100 requests/minute
    process_audio: 50 requests/minute

  voice_conversion:
    create_profile: 10 requests/hour
    convert_song: 20 requests/hour

  general:
    per_ip: 1000 requests/hour
    per_user: 5000 requests/hour
```

### Quality Targets

**Voice Profile Quality**:
- Duration: 30-60 seconds (optimal 45s)
- SNR: >15 dB (minimum 10 dB)
- Quality Score: >0.85 (minimum 0.70)

**Conversion Quality**:
- Pitch RMSE: <10 Hz (imperceptible to listeners)
- Speaker Similarity: >0.85 (85% match)
- MCD: <6.0 dB (natural sounding)
- STOI: >0.9 (highly intelligible)

### Example Production Configuration

```python
# config.py
import os

class ProductionConfig:
    # API Configuration
    BASE_URL = os.getenv('AUTOVOICE_API_URL', 'https://api.autovoice.com')
    API_VERSION = 'v1'

    # Rate Limiting
    RATE_LIMIT_ENABLED = True
    RATE_LIMIT_STORAGE = 'redis://localhost:6379/0'

    # Caching
    CACHE_ENABLED = True
    CACHE_TYPE = 'redis'
    CACHE_REDIS_URL = 'redis://localhost:6379/1'
    CACHE_DEFAULT_TIMEOUT = 3600  # 1 hour

    # GPU Configuration
    CUDA_VISIBLE_DEVICES = '0,1'  # Use GPUs 0 and 1
    GPU_MEMORY_FRACTION = 0.8  # Use 80% of GPU memory

    # Quality Presets
    DEFAULT_QUALITY_PRESET = 'balanced'
    QUALITY_PRESET_TIMEOUT = {
        'fast': 120,      # 2 minutes
        'balanced': 300,  # 5 minutes
        'quality': 600    # 10 minutes
    }

    # Monitoring
    PROMETHEUS_ENABLED = True
    PROMETHEUS_PORT = 9090
    LOG_LEVEL = 'INFO'

    # Security
    CORS_ENABLED = True
    CORS_ORIGINS = ['https://app.autovoice.com']
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB
```

---

## Support

For issues, questions, or feature requests:

- **Documentation**: [https://autovoice.readthedocs.io](https://autovoice.readthedocs.io)
- **GitHub Issues**: [https://github.com/autovoice/autovoice/issues](https://github.com/autovoice/autovoice/issues)
- **Discord**: [https://discord.gg/autovoice](https://discord.gg/autovoice)
- **Email**: support@autovoice.com

---

**Version**: v1.0.0
**Last Updated**: 2024-01-15
**License**: MIT
