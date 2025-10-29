# AutoVoice API Reference

Complete API reference for AutoVoice voice synthesis (TTS) and singing voice conversion.

## Table of Contents

1. [Introduction](#1-introduction)
2. [TTS Endpoints](#2-tts-endpoints)
3. [Voice Conversion Endpoints](#3-voice-conversion-endpoints)
4. [WebSocket API](#4-websocket-api)
5. [Error Codes](#5-error-codes)
6. [Python SDK](#6-python-sdk)
7. [JavaScript SDK](#7-javascript-sdk)
8. [Best Practices](#8-best-practices)

## 1. Introduction

### Overview

AutoVoice provides a unified REST API for both text-to-speech synthesis and singing voice conversion. The API is designed for ease of use, scalability, and production deployment.

**Base URL**: `http://localhost:5000/api/v1`

**Authentication**: None (currently open access)

**Rate Limiting**: None (recommended for production deployment)

**Response Format**: JSON

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
  -F "target_profile_id=profile-uuid"
```

### API Versions

- **v1** (current): Initial release with TTS and voice conversion

### System Requirements

- **GPU**: NVIDIA GPU with compute capability 7.0+ (recommended)
- **Memory**: 4-8GB VRAM for voice conversion, 2-4GB for TTS
- **Audio Formats**: WAV, MP3, FLAC, OGG

---

## 2. TTS Endpoints

### GET /health

System health check endpoint.

**Endpoint**: `GET /api/v1/health`

#### Response

**Status**: `200 OK`

**Body**:
```json
{
  "status": "healthy",
  "gpu_available": true,
  "model_loaded": true,
  "timestamp": "2024-01-15T10:30:00Z",
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

#### Status Values

| Status | Description |
|--------|-------------|
| `healthy` | All systems operational |
| `degraded` | Some services unavailable |
| `unhealthy` | Critical services down |

#### Example

**cURL**:
```bash
curl http://localhost:5000/api/v1/health
```

**Python**:
```python
import requests

response = requests.get("http://localhost:5000/api/v1/health")
status = response.json()
print(f"Status: {status['status']}")
print(f"GPU Available: {status['gpu_available']}")
```

**JavaScript**:
```javascript
fetch('http://localhost:5000/api/v1/health')
  .then(response => response.json())
  .then(data => {
    console.log('Status:', data.status);
    console.log('GPU Available:', data.gpu_available);
  });
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

## 3. Voice Conversion Endpoints

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
| `target_profile_id` | string | Yes | - | UUID of target voice profile |
| `vocal_volume` | float | No | 1.0 | Volume of converted vocals (0.0-2.0) |
| `instrumental_volume` | float | No | 0.9 | Volume of instrumental (0.0-2.0) |
| `pitch_shift_semitones` | integer | No | 0 | Pitch shift in semitones (�12) |
| `temperature` | float | No | 1.0 | Expressiveness control (0.5-2.0) |
| `quality_preset` | string | No | balanced | Quality preset: `fast`, `balanced`, `quality` |
| `return_stems` | boolean | No | false | Return separated vocals/instrumental |

> **Note**: For backward compatibility, the server also accepts `profile_id` (deprecated); prefer `target_profile_id`.

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
  -F "target_profile_id=550e8400-e29b-41d4-a716-446655440000" \
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
    "target_profile_id": "550e8400-e29b-41d4-a716-446655440000",
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
formData.append('target_profile_id', '550e8400-e29b-41d4-a716-446655440000');
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

Check the status of a song conversion.

**Endpoint**: `GET /api/v1/convert/status/{conversion_id}`

#### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `conversion_id` | string | Conversion task ID |

#### Request Example

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

while True:
    response = requests.get(url)
    status = response.json()

    print(f"Status: {status['status']} ({status['progress']}%)")

    if status['status'] in ['completed', 'failed']:
        break

    time.sleep(2)

if status['status'] == 'completed':
    print(f"Download URL: {status['result']['converted_audio_url']}")
```

**JavaScript**:
```javascript
async function checkStatus(conversionId) {
  const url = `http://localhost:5000/api/v1/convert/status/${conversionId}`;

  while (true) {
    const response = await fetch(url);
    const status = await response.json();

    console.log(`Status: ${status.status} (${status.progress}%)`);

    if (status.status === 'completed' || status.status === 'failed') {
      return status;
    }

    await new Promise(resolve => setTimeout(resolve, 2000));
  }
}

const result = await checkStatus('conv-770e8400-...');
console.log('Download URL:', result.result.converted_audio_url);
```

#### Response (Processing)

**Status**: `200 OK`

**Body**:
```json
{
  "conversion_id": "conv-770e8400-e29b-41d4-a716-446655440002",
  "status": "processing",
  "progress": 65,
  "current_stage": "voice_conversion",
  "stage_details": {
    "stage": 3,
    "total_stages": 4,
    "stage_name": "Voice Conversion",
    "stage_progress": 62.5
  },
  "created_at": "2024-01-15T11:00:00Z",
  "estimated_time_remaining_seconds": 15
}
```

#### Response (Completed)

**Status**: `200 OK`

**Body**:
```json
{
  "conversion_id": "conv-770e8400-e29b-41d4-a716-446655440002",
  "status": "completed",
  "progress": 100,
  "created_at": "2024-01-15T11:00:00Z",
  "completed_at": "2024-01-15T11:01:30Z",
  "processing_time_seconds": 90,
  "result": {
    "converted_audio_url": "/api/v1/convert/download/conv-770e8400-e29b-41d4-a716-446655440002/converted.wav",
    "stems": {
      "vocals_url": "/api/v1/convert/download/conv-770e8400-e29b-41d4-a716-446655440002/vocals.wav",
      "instrumental_url": "/api/v1/convert/download/conv-770e8400-e29b-41d4-a716-446655440002/instrumental.wav"
    },
    "metadata": {
      "original_duration_seconds": 180.5,
      "converted_duration_seconds": 180.5,
      "sample_rate": 44100,
      "format": "wav"
    },
    "quality_metrics": {
      "pitch_accuracy": {
        "rmse_hz": 8.2,
        "rmse_log2": 0.15,
        "correlation": 0.94
      },
      "speaker_similarity": {
        "cosine_similarity": 0.88,
        "embedding_distance": 0.24
      },
      "f0_statistics": {
        "mean_hz": 261.6,
        "std_hz": 45.3,
        "min_hz": 196.0,
        "max_hz": 392.0,
        "min_note": "G3",
        "max_note": "G4"
      }
    }
  }
}
```

#### Response (Failed)

**Status**: `200 OK`

**Body**:
```json
{
  "conversion_id": "conv-770e8400-e29b-41d4-a716-446655440002",
  "status": "failed",
  "progress": 45,
  "created_at": "2024-01-15T11:00:00Z",
  "failed_at": "2024-01-15T11:00:45Z",
  "error": {
    "code": "pitch_extraction_failed",
    "message": "Failed to extract pitch contour from vocals",
    "details": {
      "reason": "Insufficient vocal clarity",
      "suggestions": [
        "Use higher quality source audio",
        "Try different separation model",
        "Ensure vocals are prominent in mix"
      ]
    }
  }
}
```

---

### GET /convert/download/{conversion_id}/{file_type}

Download converted audio or stems.

**Endpoint**: `GET /api/v1/convert/download/{conversion_id}/{file_type}`

#### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `conversion_id` | string | Conversion task ID |
| `file_type` | string | File type: `converted.wav`, `vocals.wav`, `instrumental.wav` |

#### Request Example

**cURL**:
```bash
# Download converted audio
curl http://localhost:5000/api/v1/convert/download/conv-770e8400-e29b-41d4-a716-446655440002/converted.wav \
  -o converted.wav

# Download stems
curl http://localhost:5000/api/v1/convert/download/conv-770e8400-e29b-41d4-a716-446655440002/vocals.wav \
  -o vocals.wav
```

**Python**:
```python
import requests

conversion_id = "conv-770e8400-e29b-41d4-a716-446655440002"
base_url = f"http://localhost:5000/api/v1/convert/download/{conversion_id}"

# Download converted audio
response = requests.get(f"{base_url}/converted.wav")
with open("converted.wav", "wb") as f:
    f.write(response.content)

# Download stems
response = requests.get(f"{base_url}/vocals.wav")
with open("vocals.wav", "wb") as f:
    f.write(response.content)

response = requests.get(f"{base_url}/instrumental.wav")
with open("instrumental.wav", "wb") as f:
    f.write(response.content)
```

**JavaScript**:
```javascript
const conversionId = 'conv-770e8400-e29b-41d4-a716-446655440002';
const baseUrl = `http://localhost:5000/api/v1/convert/download/${conversionId}`;

// Download converted audio
fetch(`${baseUrl}/converted.wav`)
  .then(response => response.blob())
  .then(blob => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'converted.wav';
    a.click();
  });
```

#### Response

**Status**: `200 OK`

**Content-Type**: `audio/wav`

**Body**: Binary audio data

#### Error Responses

**File not found**:
```json
{
  "error": "file_not_found",
  "message": "Requested file not found",
  "conversion_id": "conv-770e8400-...",
  "file_type": "converted.wav"
}
```
**Status**: `404 Not Found`

**Conversion not complete**:
```json
{
  "error": "conversion_incomplete",
  "message": "Conversion not yet completed",
  "conversion_id": "conv-770e8400-...",
  "current_status": "processing"
}
```
**Status**: `409 Conflict`

---

## 4. WebSocket API

Real-time progress updates for song conversion.

### Connection

**Endpoint**: `ws://localhost:5000/ws/conversion/{conversion_id}`

#### Connection Example

**JavaScript**:
```javascript
const conversionId = 'conv-770e8400-e29b-41d4-a716-446655440002';
const ws = new WebSocket(`ws://localhost:5000/ws/conversion/${conversionId}`);

ws.onopen = () => {
  console.log('WebSocket connected');
};

ws.onmessage = (event) => {
  const update = JSON.parse(event.data);
  console.log(`Progress: ${update.progress}% - ${update.current_stage}`);

  if (update.status === 'completed') {
    console.log('Conversion completed!');
    console.log('Download URL:', update.result.converted_audio_url);
    ws.close();
  } else if (update.status === 'failed') {
    console.error('Conversion failed:', update.error.message);
    ws.close();
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('WebSocket disconnected');
};
```

**Python**:
```python
import websocket
import json

def on_message(ws, message):
    update = json.loads(message)
    print(f"Progress: {update['progress']}% - {update['current_stage']}")

    if update['status'] == 'completed':
        print('Conversion completed!')
        print('Download URL:', update['result']['converted_audio_url'])
        ws.close()
    elif update['status'] == 'failed':
        print('Conversion failed:', update['error']['message'])
        ws.close()

def on_error(ws, error):
    print('WebSocket error:', error)

def on_close(ws, close_status_code, close_msg):
    print('WebSocket disconnected')

def on_open(ws):
    print('WebSocket connected')

conversion_id = 'conv-770e8400-e29b-41d4-a716-446655440002'
ws_url = f"ws://localhost:5000/ws/conversion/{conversion_id}"

ws = websocket.WebSocketApp(
    ws_url,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close,
    on_open=on_open
)

ws.run_forever()
```

### Message Format

#### Progress Update

```json
{
  "conversion_id": "conv-770e8400-e29b-41d4-a716-446655440002",
  "status": "processing",
  "progress": 65,
  "current_stage": "voice_conversion",
  "stage_details": {
    "stage": 3,
    "total_stages": 4,
    "stage_name": "Voice Conversion",
    "stage_progress": 62.5
  },
  "estimated_time_remaining_seconds": 15
}
```

#### Completion

```json
{
  "conversion_id": "conv-770e8400-e29b-41d4-a716-446655440002",
  "status": "completed",
  "progress": 100,
  "completed_at": "2024-01-15T11:01:30Z",
  "result": {
    "converted_audio_url": "/api/v1/convert/download/conv-770e8400-e29b-41d4-a716-446655440002/converted.wav",
    "quality_metrics": {
      "pitch_accuracy": {
        "rmse_hz": 8.2,
        "correlation": 0.94
      },
      "speaker_similarity": {
        "cosine_similarity": 0.88
      }
    }
  }
}
```

#### Failure

```json
{
  "conversion_id": "conv-770e8400-e29b-41d4-a716-446655440002",
  "status": "failed",
  "progress": 45,
  "failed_at": "2024-01-15T11:00:45Z",
  "error": {
    "code": "pitch_extraction_failed",
    "message": "Failed to extract pitch contour from vocals"
  }
}
```

---

## 5. Error Codes

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
| `conversion_failed` | 500 | Voice conversion failed | Check model status |
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

## 6. Python SDK

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

## 7. JavaScript SDK

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

## 8. Best Practices

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
