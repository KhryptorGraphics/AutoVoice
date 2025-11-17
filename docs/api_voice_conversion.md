# AutoVoice API Reference - Voice Conversion

**Note**: This document provides voice conversion-specific API documentation. For the complete consolidated API reference covering both TTS and voice conversion, see [`docs/api-documentation.md`](api-documentation.md) which is the canonical source of truth for all API endpoints.

Complete API reference for AutoVoice singing voice conversion endpoints.

## 1. Introduction

### Overview

The AutoVoice Voice Conversion API provides RESTful endpoints for creating voice profiles and converting songs to different voices while preserving the original pitch, timing, and musical expression.

**Base URL**: `http://localhost:5000/api/v1`

**Authentication**: None (currently open access)

**Rate Limiting**: None (add resource-based limits for production)

**Response Format**: JSON

### Quick Start

```bash
# Create a voice profile
curl -X POST http://localhost:5000/api/v1/voice/clone \
  -F "audio=@my_voice.wav" \
  -F "user_id=user123"

# Convert a song
curl -X POST http://localhost:5000/api/v1/convert/song \
  -F "song=@song.mp3" \
  -F "target_profile_id=profile-uuid"
```

### API Versions

- **v1** (current): Initial release with voice cloning and song conversion

## 2. Voice Profile Endpoints

### POST /voice/clone

Create a voice profile from an audio sample.

**Endpoint**: `POST /api/v1/voice/clone`

**Content-Type**: `multipart/form-data`

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `audio` | file | Yes | Audio file (WAV, MP3, FLAC, OGG) containing voice sample |
| `user_id` | string | Yes | Unique identifier for the user |
| `profile_name` | string | No | Human-readable name for the profile (default: timestamp) |
| `description` | string | No | Optional description of the voice profile |

#### Request Example

**cURL**:
```bash
curl -X POST http://localhost:5000/api/v1/voice/clone \
  -F "audio=@my_voice.wav" \
  -F "user_id=user123" \
  -F "profile_name=My Singing Voice" \
  -F "description=My natural singing voice recorded in studio"
```

**Python**:
```python
import requests

url = "http://localhost:5000/api/v1/voice/clone"
files = {"audio": open("my_voice.wav", "rb")}
data = {
    "user_id": "user123",
    "profile_name": "My Singing Voice",
    "description": "My natural singing voice"
}

response = requests.post(url, files=files, data=data)
profile = response.json()
print(f"Profile created: {profile['profile_id']}")
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
.then(data => console.log('Profile created:', data.profile_id));
```

#### Response

**Status**: `201 Created`

**Body**:
```json
{
  "profile_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user123",
  "profile_name": "My Singing Voice",
  "description": "My natural singing voice",
  "created_at": "2024-01-15T10:30:00Z",
  "audio_info": {
    "duration_seconds": 45.2,
    "sample_rate": 44100,
    "channels": 1,
    "format": "wav"
  },
  "vocal_range": {
    "min_pitch_hz": 130.8,
    "max_pitch_hz": 392.0,
    "min_note": "C3",
    "max_note": "G4",
    "range_semitones": 19
  },
  "quality_metrics": {
    "snr_db": 22.5,
    "quality_score": 0.87
  },
  "embedding_info": {
    "model": "resemblyzer",
    "dimensions": 256
  }
}
```

#### Error Responses

**400 Bad Request** - Invalid input:
```json
{
  "error": "invalid_audio",
  "message": "Audio file too short (5.2s). Minimum 30 seconds required.",
  "details": {
    "duration": 5.2,
    "minimum_required": 30
  }
}
```

**413 Payload Too Large** - File too large:
```json
{
  "error": "file_too_large",
  "message": "Audio file exceeds maximum size of 100MB",
  "details": {
    "file_size_mb": 150,
    "max_size_mb": 100
  }
}
```

**422 Unprocessable Entity** - Low quality audio:
```json
{
  "error": "low_quality_audio",
  "message": "Audio quality insufficient for voice cloning",
  "details": {
    "snr_db": 8.5,
    "minimum_snr_db": 10.0,
    "suggestions": [
      "Record in quieter environment",
      "Move closer to microphone",
      "Use better recording equipment"
    ]
  }
}
```

### GET /voice/profiles

Retrieve all voice profiles for a user.

**Endpoint**: `GET /api/v1/voice/profiles`

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `user_id` | string | Yes | User identifier to filter profiles |
| `limit` | integer | No | Maximum number of profiles to return (default: 100) |
| `offset` | integer | No | Number of profiles to skip for pagination (default: 0) |
| `sort_by` | string | No | Sort field: `created_at`, `name`, `duration` (default: `created_at`) |
| `order` | string | No | Sort order: `asc`, `desc` (default: `desc`) |

#### Request Example

**cURL**:
```bash
curl -X GET "http://localhost:5000/api/v1/voice/profiles?user_id=user123&limit=10&sort_by=created_at&order=desc"
```

**Python**:
```python
import requests

url = "http://localhost:5000/api/v1/voice/profiles"
params = {
    "user_id": "user123",
    "limit": 10,
    "sort_by": "created_at",
    "order": "desc"
}

response = requests.get(url, params=params)
profiles = response.json()
print(f"Found {len(profiles['profiles'])} profiles")
```

**JavaScript**:
```javascript
const params = new URLSearchParams({
  user_id: 'user123',
  limit: 10,
  sort_by: 'created_at',
  order: 'desc'
});

fetch(`http://localhost:5000/api/v1/voice/profiles?${params}`)
  .then(response => response.json())
  .then(data => console.log(`Found ${data.profiles.length} profiles`));
```

#### Response

**Status**: `200 OK`

**Body**:
```json
{
  "profiles": [
    {
      "profile_id": "550e8400-e29b-41d4-a716-446655440000",
      "user_id": "user123",
      "profile_name": "My Singing Voice",
      "description": "My natural singing voice",
      "created_at": "2024-01-15T10:30:00Z",
      "audio_info": {
        "duration_seconds": 45.2,
        "sample_rate": 44100
      },
      "vocal_range": {
        "min_note": "C3",
        "max_note": "G4",
        "range_semitones": 19
      }
    },
    {
      "profile_id": "660e8400-e29b-41d4-a716-446655440001",
      "user_id": "user123",
      "profile_name": "Alternative Voice",
      "created_at": "2024-01-14T15:20:00Z",
      "audio_info": {
        "duration_seconds": 60.0,
        "sample_rate": 44100
      },
      "vocal_range": {
        "min_note": "D3",
        "max_note": "A4",
        "range_semitones": 17
      }
    }
  ],
  "pagination": {
    "total": 2,
    "limit": 10,
    "offset": 0,
    "has_more": false
  }
}
```

### GET /voice/profiles/{profile_id}

Retrieve details for a specific voice profile.

**Endpoint**: `GET /api/v1/voice/profiles/{profile_id}`

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `profile_id` | string | Yes | UUID of the voice profile (path parameter) |

#### Request Example

**cURL**:
```bash
curl -X GET "http://localhost:5000/api/v1/voice/profiles/550e8400-e29b-41d4-a716-446655440000"
```

**Python**:
```python
import requests

profile_id = "550e8400-e29b-41d4-a716-446655440000"
url = f"http://localhost:5000/api/v1/voice/profiles/{profile_id}"

response = requests.get(url)
profile = response.json()
print(f"Profile: {profile['profile_name']}")
```

**JavaScript**:
```javascript
const profileId = '550e8400-e29b-41d4-a716-446655440000';
fetch(`http://localhost:5000/api/v1/voice/profiles/${profileId}`)
  .then(response => response.json())
  .then(profile => console.log('Profile:', profile.profile_name));
```

#### Response

**Status**: `200 OK`

**Body**: Same as single profile object in GET /voice/profiles response

#### Error Responses

**404 Not Found** - Profile doesn't exist:
```json
{
  "error": "profile_not_found",
  "message": "Voice profile not found",
  "details": {
    "profile_id": "550e8400-e29b-41d4-a716-446655440000"
  }
}
```

### DELETE /voice/profiles/{profile_id}

Delete a voice profile.

**Endpoint**: `DELETE /api/v1/voice/profiles/{profile_id}`

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `profile_id` | string | Yes | UUID of the voice profile to delete (path parameter) |
| `user_id` | string | Yes | User ID for authorization (query parameter) |

#### Request Example

**cURL**:
```bash
curl -X DELETE "http://localhost:5000/api/v1/voice/profiles/550e8400-e29b-41d4-a716-446655440000?user_id=user123"
```

**Python**:
```python
import requests

profile_id = "550e8400-e29b-41d4-a716-446655440000"
url = f"http://localhost:5000/api/v1/voice/profiles/{profile_id}"
params = {"user_id": "user123"}

response = requests.delete(url, params=params)
print(f"Status: {response.status_code}")
```

**JavaScript**:
```javascript
const profileId = '550e8400-e29b-41d4-a716-446655440000';
fetch(`http://localhost:5000/api/v1/voice/profiles/${profileId}?user_id=user123`, {
  method: 'DELETE'
})
.then(response => console.log('Deleted:', response.ok));
```

#### Response

**Status**: `204 No Content`

**Body**: Empty

#### Error Responses

**403 Forbidden** - User doesn't own profile:
```json
{
  "error": "unauthorized",
  "message": "User is not authorized to delete this profile",
  "details": {
    "profile_id": "550e8400-e29b-41d4-a716-446655440000",
    "user_id": "user123"
  }
}
```

**404 Not Found** - Profile doesn't exist:
```json
{
  "error": "profile_not_found",
  "message": "Voice profile not found"
}
```

## 3. Song Conversion Endpoints

### POST /convert/song

Convert a song to a target voice while preserving pitch and timing.

**Endpoint**: `POST /api/v1/convert/song`

**Content-Type**: `multipart/form-data`

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `song` | file | Yes | Audio file to convert (MP3, WAV, FLAC, OGG) |
| `target_profile_id` | string | Yes | UUID of target voice profile |
| `vocal_volume` | float | No | Volume of converted vocals (0.0-2.0, default: 1.0) |
| `instrumental_volume` | float | No | Volume of instrumental (0.0-2.0, default: 0.9) |
| `pitch_shift_semitones` | integer | No | Pitch shift in semitones (±12, default: 0) |
| `temperature` | float | No | Expressiveness control (0.5-2.0, default: 1.0) |
| `quality_preset` | string | No | Quality preset: `fast`, `balanced`, `quality` (default: `balanced`) |
| `return_stems` | boolean | No | Return separated vocals/instrumental (default: false) |
| `use_cache` | boolean | No | Use cached separation if available (default: true) |

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
  -F "return_stems=false"
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
    "return_stems": False
}

response = requests.post(url, files=files, data=data)
result = response.json()
print(f"Conversion ID: {result['conversion_id']}")
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
.then(data => console.log('Conversion ID:', data.conversion_id));
```

#### Response

**Status**: `202 Accepted`

**Body**:
```json
{
  "conversion_id": "conv-770e8400-e29b-41d4-a716-446655440002",
  "status": "processing",
  "message": "Song conversion started",
  "created_at": "2024-01-15T11:00:00Z",
  "estimated_time_seconds": 45,
  "progress_url": "/api/v1/convert/status/conv-770e8400-e29b-41d4-a716-446655440002",
  "websocket_url": "ws://localhost:5000/ws/conversion/conv-770e8400-e29b-41d4-a716-446655440002"
}
```

#### Processing Pipeline Stages

1. **Separation (0-25%)**: Separating vocals from instrumental using Demucs
2. **Pitch Extraction (25-40%)**: Extracting pitch contour with torchcrepe
3. **Voice Conversion (40-80%)**: Converting vocals to target voice
4. **Mixing (80-100%)**: Mixing converted vocals with instrumental

#### Error Responses

**400 Bad Request** - Invalid parameters:
```json
{
  "error": "invalid_parameters",
  "message": "Pitch shift out of range",
  "details": {
    "pitch_shift_semitones": 15,
    "valid_range": [-12, 12]
  }
}
```

**404 Not Found** - Profile not found:
```json
{
  "error": "profile_not_found",
  "message": "Target voice profile not found",
  "details": {
    "target_profile_id": "550e8400-e29b-41d4-a716-446655440000"
  }
}
```

**413 Payload Too Large** - File too large:
```json
{
  "error": "file_too_large",
  "message": "Song file exceeds maximum size of 100MB"
}
```

**422 Unprocessable Entity** - Processing failed:
```json
{
  "error": "separation_failed",
  "message": "Failed to separate vocals from instrumental",
  "details": {
    "reason": "No vocals detected in audio",
    "suggestions": [
      "Ensure song contains vocals",
      "Check audio quality"
    ]
  }
}
```

### GET /convert/status/{conversion_id}

Get the status of a song conversion.

**Endpoint**: `GET /api/v1/convert/status/{conversion_id}`

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `conversion_id` | string | Yes | Conversion ID (path parameter) |

#### Request Example

**cURL**:
```bash
curl -X GET "http://localhost:5000/api/v1/convert/status/conv-770e8400-e29b-41d4-a716-446655440002"
```

**Python**:
```python
import requests

conversion_id = "conv-770e8400-e29b-41d4-a716-446655440002"
url = f"http://localhost:5000/api/v1/convert/status/{conversion_id}"

response = requests.get(url)
status = response.json()
print(f"Progress: {status['progress']}%")
```

**JavaScript**:
```javascript
const conversionId = 'conv-770e8400-e29b-41d4-a716-446655440002';
fetch(`http://localhost:5000/api/v1/convert/status/${conversionId}`)
  .then(response => response.json())
  .then(status => console.log('Progress:', status.progress + '%'));
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

### GET /convert/download/{conversion_id}/{file_type}

Download converted audio or stems.

**Endpoint**: `GET /api/v1/convert/download/{conversion_id}/{file_type}`

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `conversion_id` | string | Yes | Conversion ID (path parameter) |
| `file_type` | string | Yes | File type: `converted.wav`, `vocals.wav`, `instrumental.wav` |

#### Request Example

**cURL**:
```bash
curl -X GET "http://localhost:5000/api/v1/convert/download/conv-770e8400-e29b-41d4-a716-446655440002/converted.wav" \
  -o converted_song.wav
```

**Python**:
```python
import requests

conversion_id = "conv-770e8400-e29b-41d4-a716-446655440002"
url = f"http://localhost:5000/api/v1/convert/download/{conversion_id}/converted.wav"

response = requests.get(url)
with open("converted_song.wav", "wb") as f:
    f.write(response.content)
```

**JavaScript**:
```javascript
const conversionId = 'conv-770e8400-e29b-41d4-a716-446655440002';
fetch(`http://localhost:5000/api/v1/convert/download/${conversionId}/converted.wav`)
  .then(response => response.blob())
  .then(blob => {
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'converted_song.wav';
    a.click();
  });
```

#### Response

**Status**: `200 OK`

**Content-Type**: `audio/wav`

**Headers**:
```
Content-Disposition: attachment; filename="converted.wav"
Content-Length: 31457280
```

**Body**: Binary audio data

#### Error Responses

**404 Not Found** - File not available:
```json
{
  "error": "file_not_found",
  "message": "Converted audio file not available",
  "details": {
    "conversion_id": "conv-770e8400-e29b-41d4-a716-446655440002",
    "file_type": "converted.wav"
  }
}
```

**410 Gone** - File expired:
```json
{
  "error": "file_expired",
  "message": "Converted audio file has expired",
  "details": {
    "expired_at": "2024-01-16T11:00:00Z",
    "retention_hours": 24
  }
}
```

## 4. WebSocket API

### Connection

Establish WebSocket connection for real-time conversion progress.

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
  const data = JSON.parse(event.data);
  console.log('Progress:', data.progress + '%');

  if (data.status === 'completed') {
    console.log('Download URL:', data.result.converted_audio_url);
    ws.close();
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('WebSocket closed');
};
```

**Python**:
```python
import asyncio
import websockets
import json

async def monitor_conversion(conversion_id):
    uri = f"ws://localhost:5000/ws/conversion/{conversion_id}"

    async with websockets.connect(uri) as websocket:
        async for message in websocket:
            data = json.loads(message)
            print(f"Progress: {data['progress']}%")

            if data['status'] == 'completed':
                print(f"Download URL: {data['result']['converted_audio_url']}")
                break
            elif data['status'] == 'failed':
                print(f"Error: {data['error']['message']}")
                break

asyncio.run(monitor_conversion("conv-770e8400-e29b-41d4-a716-446655440002"))
```

### Message Types

#### Progress Update

Sent periodically during processing (every 1-2 seconds).

```json
{
  "type": "progress",
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
  "timestamp": "2024-01-15T11:00:45Z"
}
```

#### Completion Message

Sent when conversion completes successfully.

```json
{
  "type": "completed",
  "conversion_id": "conv-770e8400-e29b-41d4-a716-446655440002",
  "status": "completed",
  "progress": 100,
  "result": {
    "converted_audio_url": "/api/v1/convert/download/conv-770e8400-e29b-41d4-a716-446655440002/converted.wav",
    "quality_metrics": {
      "pitch_accuracy": {
        "rmse_hz": 8.2
      },
      "speaker_similarity": {
        "cosine_similarity": 0.88
      }
    }
  },
  "timestamp": "2024-01-15T11:01:30Z"
}
```

#### Error Message

Sent when conversion fails.

```json
{
  "type": "error",
  "conversion_id": "conv-770e8400-e29b-41d4-a716-446655440002",
  "status": "failed",
  "error": {
    "code": "pitch_extraction_failed",
    "message": "Failed to extract pitch contour from vocals",
    "details": {
      "reason": "Insufficient vocal clarity"
    }
  },
  "timestamp": "2024-01-15T11:00:45Z"
}
```

## 5. Python SDK

### Installation

```bash
pip install auto-voice
```

### Voice Cloning

```python
from auto_voice.inference import VoiceCloner

# Initialize cloner
cloner = VoiceCloner(device='cuda')

# Create voice profile
profile = cloner.create_voice_profile(
    audio='my_voice.wav',
    user_id='user123'
)

print(f"Profile ID: {profile['profile_id']}")
print(f"Vocal Range: {profile['vocal_range']['min_note']} - {profile['vocal_range']['max_note']}")

# Create multi-sample profile
samples = ['voice1.wav', 'voice2.wav', 'voice3.wav']
profile = cloner.create_voice_profile_from_multiple_samples(
    audio_paths=samples,
    user_id='user123'
)
```

### Song Conversion

```python
from auto_voice.inference import SingingConversionPipeline

# Initialize pipeline
pipeline = SingingConversionPipeline(
    device='cuda',
    quality_preset='balanced'
)

# Convert song
result = pipeline.convert_song(
    song_path='song.mp3',
    target_profile_id='550e8400-e29b-41d4-a716-446655440000',
    vocal_volume=1.0,
    instrumental_volume=0.9,
    pitch_shift_semitones=0,
    temperature=1.0,
    return_stems=True
)

print(f"Conversion ID: {result['conversion_id']}")
print(f"Output: {result['output_path']}")
print(f"Quality Metrics: {result['quality_metrics']}")

# Batch conversion
songs = ['song1.mp3', 'song2.mp3', 'song3.mp3']
for song in songs:
    result = pipeline.convert_song(
        song_path=song,
        target_profile_id='550e8400-e29b-41d4-a716-446655440000'
    )
    print(f"Converted: {song} -> {result['output_path']}")
```

### Quality Evaluation

```python
from auto_voice.utils.quality_metrics import QualityMetricsAggregator

# Initialize metrics
metrics = QualityMetricsAggregator()

# Compute metrics
quality = metrics.compute_all_metrics(
    original_audio='original.wav',
    converted_audio='converted.wav',
    reference_f0='reference_f0.npy',
    converted_f0='converted_f0.npy',
    target_embedding='target_embedding.npy',
    converted_embedding='converted_embedding.npy'
)

print(f"Pitch RMSE (Hz): {quality['pitch_accuracy']['rmse_hz']:.2f}")
print(f"Speaker Similarity: {quality['speaker_similarity']['cosine_similarity']:.2f}")
print(f"Overall Quality: {quality['overall_quality_score']:.2f}")
```

## 6. JavaScript SDK

### Installation

```bash
npm install autovoice-sdk
```

### Voice Cloning

```javascript
import { VoiceClient } from 'autovoice-sdk';

// Initialize client
const client = new VoiceClient({
  baseUrl: 'http://localhost:5000/api/v1'
});

// Create voice profile
const profile = await client.createVoiceProfile({
  audio: audioFile,
  userId: 'user123',
  profileName: 'My Singing Voice'
});

console.log('Profile ID:', profile.profileId);
console.log('Vocal Range:', profile.vocalRange);

// List profiles
const profiles = await client.getVoiceProfiles({
  userId: 'user123',
  limit: 10
});

console.log('Found profiles:', profiles.profiles.length);

// Delete profile
await client.deleteVoiceProfile({
  profileId: profile.profileId,
  userId: 'user123'
});
```

### Song Conversion

```javascript
import { ConversionClient } from 'autovoice-sdk';

// Initialize client
const client = new ConversionClient({
  baseUrl: 'http://localhost:5000/api/v1'
});

// Convert song
const conversion = await client.convertSong({
  song: songFile,
  targetProfileId: '550e8400-e29b-41d4-a716-446655440000',
  vocalVolume: 1.0,
  instrumentalVolume: 0.9,
  qualityPreset: 'balanced'
});

console.log('Conversion ID:', conversion.conversionId);

// Monitor progress
const status = await client.getConversionStatus(conversion.conversionId);
console.log('Progress:', status.progress + '%');

// Download result
const audioBlob = await client.downloadConversion(conversion.conversionId);
const url = URL.createObjectURL(audioBlob);
```

### WebSocket Monitoring

```javascript
import { ConversionMonitor } from 'autovoice-sdk';

// Create monitor
const monitor = new ConversionMonitor({
  conversionId: 'conv-770e8400-e29b-41d4-a716-446655440002',
  websocketUrl: 'ws://localhost:5000'
});

// Listen for progress
monitor.on('progress', (data) => {
  console.log('Progress:', data.progress + '%');
  console.log('Stage:', data.currentStage);
});

// Listen for completion
monitor.on('completed', (data) => {
  console.log('Download URL:', data.result.convertedAudioUrl);
  console.log('Quality:', data.result.qualityMetrics);
});

// Listen for errors
monitor.on('error', (error) => {
  console.error('Error:', error.message);
});

// Start monitoring
monitor.connect();
```

## 7. Error Codes Reference

### Client Errors (4xx)

| Code | HTTP Status | Description | Resolution |
|------|-------------|-------------|------------|
| `invalid_audio` | 400 | Audio file format invalid or corrupted | Use supported format (WAV, MP3, FLAC, OGG) |
| `audio_too_short` | 400 | Audio duration below minimum | Record at least 30 seconds |
| `invalid_parameters` | 400 | Request parameters out of range | Check parameter constraints |
| `missing_required_field` | 400 | Required field not provided | Include all required fields |
| `unauthorized` | 403 | User not authorized for operation | Verify user owns resource |
| `profile_not_found` | 404 | Voice profile doesn't exist | Check profile ID is correct |
| `conversion_not_found` | 404 | Conversion doesn't exist | Check conversion ID is correct |
| `file_not_found` | 404 | Converted file not available | Ensure conversion completed |
| `file_expired` | 410 | Converted file has expired | Files expire after 24 hours |
| `file_too_large` | 413 | File exceeds size limit | Use file under 100MB |
| `low_quality_audio` | 422 | Audio quality insufficient | Improve recording quality |
| `separation_failed` | 422 | Vocal separation failed | Ensure song has clear vocals |
| `pitch_extraction_failed` | 422 | Pitch detection failed | Use higher quality source |
| `conversion_error` | 422 | Voice conversion failed | Check error details |

### Server Errors (5xx)

| Code | HTTP Status | Description | Resolution |
|------|-------------|-------------|------------|
| `internal_error` | 500 | Unexpected server error | Retry request, contact support |
| `gpu_error` | 500 | GPU processing error | Check GPU availability |
| `out_of_memory` | 500 | Insufficient memory | Use shorter audio, try CPU |
| `model_load_error` | 500 | Failed to load model | Check model files exist |
| `service_unavailable` | 503 | Service temporarily down | Wait and retry |

## 8. Best Practices

### Voice Profile Creation

**Duration**:
- Record 45-60 seconds for optimal quality
- Minimum 30 seconds, shorter samples produce lower quality
- Longer samples (up to 60s) improve robustness

**Recording Quality**:
- Use quiet environment with minimal background noise
- Maintain consistent distance from microphone (6-12 inches)
- Avoid clipping and distortion
- Use lossless formats (WAV, FLAC) when possible
- Target SNR >15 dB for good quality

**Content**:
- Speak or sing naturally with variation
- Include different pitches and dynamics
- Avoid monotone delivery
- Use conversational tone or natural singing

**Multi-Sample Profiles**:
- Create profiles from 2-5 samples for best results
- Use different content for each sample
- Ensure consistent recording setup
- System averages embeddings for robustness

### Song Conversion

**Source Material**:
- Use high-quality source songs (lossless preferred)
- Ensure vocals are clear and prominent in mix
- Avoid heavily processed vocals (heavy autotune, extreme effects)
- Solo vocals work better than complex arrangements

**Quality Settings**:
- Test with `fast` preset first for quick results
- Use `balanced` preset for general use
- Use `quality` preset for final versions

**Pitch Matching**:
- Check if song key matches your vocal range
- Use `pitch_shift_semitones` to transpose song
- Typical ranges:
  - Female: C4-G5 (262-784 Hz)
  - Male: C3-G4 (130-392 Hz)

**Volume Balancing**:
- Start with `vocal_volume=1.0`, `instrumental_volume=0.9`
- Adjust based on desired mix balance
- Lower instrumental for vocal clarity
- Lower vocals if song has strong backing track

### API Usage

**Polling vs WebSocket**:
- Use WebSocket for real-time progress in user-facing apps
- Use polling for batch processing or background jobs
- WebSocket recommended for better UX

**Caching**:
- Enable `use_cache=true` for repeated conversions
- Cache significantly speeds up separation step
- Disable for different separation models

**Error Handling**:
- Always check status codes and error messages
- Retry transient errors (5xx) with exponential backoff
- Don't retry client errors (4xx) without fixing issue
- Implement timeout for long-running conversions

**File Management**:
- Download and store results within 24 hours
- Converted files expire after retention period
- Clean up old conversions to save storage

### Performance Optimization

**Batch Processing**:
- Use SDK batch conversion for multiple songs
- Process songs in parallel when possible
- Reuse same profile for multiple songs

**Quality vs Speed**:
- `fast` preset: ~0.5x real-time (15-30s for 30s song)
- `balanced` preset: ~1x real-time (30-60s for 30s song)
- `quality` preset: ~2x real-time (60-120s for 30s song)

**GPU Usage**:
- GPU provides 10-50x speedup over CPU
- Check GPU availability before processing
- Monitor GPU memory for long songs

**TensorRT Optimization**:
- Enable TensorRT for 2-3x additional speedup
- Requires model conversion (one-time setup)
- Best for production deployments

## 9. Rate Limiting (Planned)

**Future Implementation**:
- Rate limiting per user/API key
- Concurrent conversion limits
- Storage quota management
- Usage analytics and reporting

**Current Status**:
- No rate limiting implemented
- Consider resource-based limits for production
- Monitor usage patterns

## 10. Changelog

### v1.0.0 (Current)
- Initial API release
- Voice profile creation and management
- Song conversion with quality presets
- WebSocket progress monitoring
- Quality metrics reporting

### Planned Features
- Authentication and API keys
- Rate limiting and quotas
- Advanced separation models
- Custom model training
- Multi-language support
- Real-time voice conversion

## 11. Support and Resources

**Documentation**:
- User Guide: `docs/voice_conversion_guide.md`
- Model Architecture: `docs/model_architecture.md`
- Operations Guide: `docs/runbook.md`

**Examples**:
- Interactive Notebooks: `examples/voice_cloning_demo.ipynb`, `examples/song_conversion_demo.ipynb`
- Demo Scripts: `examples/demo_voice_conversion.py`, `examples/demo_batch_conversion.py`

**Support**:
- GitHub Issues: Report bugs and feature requests
- Documentation: Comprehensive guides and references
- Community: Join discussions and share experiences
