# Voice Profile API Reference

> Historical note: this file is a narrow API reference and may omit newer role-based behavior. For the current MVP flow, start with [docs/api/README.md](./api/README.md) and [user-guide-voice-profiles.md](./user-guide-voice-profiles.md).

API documentation for voice profile management, training samples, and continuous learning.

**Base URL:** `/api/v1`

---

## Voice Profiles

### Create Voice Profile (Clone Voice)

Create a new voice profile from reference audio.

```
POST /voice/clone
Content-Type: multipart/form-data
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `reference_audio` | file | Yes | Audio file (WAV, MP3, FLAC, OGG, M4A) with voice sample |
| `user_id` | string | No | Optional user identifier to associate with profile |

**Response (201 Created):**

```json
{
  "status": "success",
  "profile_id": "uuid-string",
  "user_id": "optional-user-id",
  "audio_duration": 15.5,
  "vocal_range": {"low": 120.0, "high": 450.0},
  "created_at": "2026-01-30T12:00:00Z"
}
```

**Error Responses:**

| Code | Error | Description |
|------|-------|-------------|
| 400 | `invalid_reference_audio` | Audio file is corrupt or unsupported format |
| 400 | `insufficient_quality` | Audio quality too low (SNR, clarity) |
| 400 | `inconsistent_samples` | Multiple voice characteristics detected |
| 503 | Service unavailable | Voice cloner not initialized |

---

### List Voice Profiles

```
GET /voice/profiles
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_id` | string | Filter profiles by user ID |

**Response (200 OK):**

```json
[
  {
    "profile_id": "uuid-1",
    "user_id": "user-123",
    "name": "My Voice",
    "created_at": "2026-01-30T12:00:00Z",
    "samples_count": 25,
    "model_version": "v3"
  }
]
```

---

### Get Voice Profile

```
GET /voice/profiles/{profile_id}
```

**Response (200 OK):**

```json
{
  "profile_id": "uuid-string",
  "user_id": "user-123",
  "name": "My Voice",
  "created_at": "2026-01-30T12:00:00Z",
  "samples_count": 25,
  "model_version": "v3",
  "vocal_range": {"low": 120.0, "high": 450.0},
  "training_history": [
    {"version": "v1", "samples": 10, "date": "2026-01-25"},
    {"version": "v2", "samples": 18, "date": "2026-01-28"},
    {"version": "v3", "samples": 25, "date": "2026-01-30"}
  ]
}
```

**Error Responses:**

| Code | Error | Description |
|------|-------|-------------|
| 404 | Profile not found | No profile with given ID |
| 503 | Service unavailable | Voice cloner not initialized |

---

### Delete Voice Profile

```
DELETE /voice/profiles/{profile_id}
```

**Response (200 OK):**

```json
{
  "status": "success",
  "profile_id": "uuid-string"
}
```

---

## Training Samples

### List Samples

```
GET /profiles/{profile_id}/samples
```

**Response (200 OK):**

```json
[
  {
    "sample_id": "uuid-1",
    "profile_id": "profile-uuid",
    "filename": "song-phrase.wav",
    "created_at": "2026-01-30T12:00:00Z",
    "duration": 8.5,
    "metadata": {
      "source": "karaoke_session",
      "song_title": "Bohemian Rhapsody"
    }
  }
]
```

---

### Upload Sample

```
POST /profiles/{profile_id}/samples
Content-Type: multipart/form-data
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Audio file with training sample |
| `metadata` | JSON string | No | Additional metadata (source, song info) |

**Response (201 Created):**

```json
{
  "sample_id": "uuid-string",
  "profile_id": "profile-uuid",
  "filename": "sample.wav",
  "file_path": "/uploads/samples/profile-uuid/uuid_sample.wav",
  "created_at": "2026-01-30T12:00:00Z",
  "duration": null,
  "metadata": {}
}
```

---

### Get Sample

```
GET /profiles/{profile_id}/samples/{sample_id}
```

**Response (200 OK):** Same as upload response.

---

### Delete Sample

```
DELETE /profiles/{profile_id}/samples/{sample_id}
```

**Response (200 OK):**

```json
{
  "status": "success",
  "sample_id": "uuid-string"
}
```

---

## Training Jobs

### List Training Jobs

```
GET /training/jobs
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `profile_id` | string | Filter jobs by profile ID |

**Response (200 OK):**

```json
[
  {
    "job_id": "uuid-1",
    "profile_id": "profile-uuid",
    "status": "completed",
    "created_at": "2026-01-30T10:00:00Z",
    "started_at": "2026-01-30T10:01:00Z",
    "completed_at": "2026-01-30T10:15:00Z",
    "progress": 100,
    "sample_ids": ["sample-1", "sample-2"],
    "config": {
      "lora_rank": 8,
      "lora_alpha": 16,
      "learning_rate": 0.0001,
      "epochs": 10
    },
    "error": null,
    "results": {
      "final_loss": 0.023,
      "model_version": "v4"
    }
  }
]
```

---

### Create Training Job

```
POST /training/jobs
Content-Type: application/json
```

**Request Body:**

```json
{
  "profile_id": "profile-uuid",
  "sample_ids": ["sample-1", "sample-2", "sample-3"],
  "config": {
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "learning_rate": 0.0001,
    "epochs": 10,
    "use_ewc": true,
    "ewc_lambda": 1000.0
  }
}
```

**Training Config Options:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `lora_rank` | int | 8 | LoRA adapter rank (4-64) |
| `lora_alpha` | int | 16 | LoRA scaling factor |
| `lora_dropout` | float | 0.1 | Dropout rate for LoRA layers |
| `learning_rate` | float | 1e-4 | Optimizer learning rate |
| `epochs` | int | 10 | Training epochs per job |
| `use_ewc` | bool | true | Enable Elastic Weight Consolidation |
| `ewc_lambda` | float | 1000.0 | EWC regularization strength |
| `batch_size` | int | 4 | Training batch size |
| `gradient_accumulation` | int | 1 | Gradient accumulation steps |

**Response (201 Created):**

```json
{
  "job_id": "uuid-string",
  "profile_id": "profile-uuid",
  "status": "pending",
  "created_at": "2026-01-30T12:00:00Z",
  "progress": 0
}
```

---

### Get Training Job

```
GET /training/jobs/{job_id}
```

**Response (200 OK):** Same structure as list response (single job).

**Job Status Values:**

| Status | Description |
|--------|-------------|
| `pending` | Job queued, waiting for GPU |
| `running` | Training in progress |
| `completed` | Training finished successfully |
| `failed` | Training failed with error |
| `cancelled` | Job cancelled by user |

---

### Cancel Training Job

```
POST /training/jobs/{job_id}/cancel
```

**Response (200 OK):**

```json
{
  "job_id": "uuid-string",
  "status": "cancelled",
  "completed_at": "2026-01-30T12:05:00Z"
}
```

**Error Responses:**

| Code | Error | Description |
|------|-------|-------------|
| 400 | Cannot cancel | Job already completed/failed/cancelled |
| 404 | Job not found | No job with given ID |

---

## Model Checkpoints

### List Checkpoints

```
GET /profiles/{profile_id}/checkpoints
```

**Response (200 OK):**

```json
[
  {
    "checkpoint_id": "v3",
    "profile_id": "profile-uuid",
    "created_at": "2026-01-30T12:00:00Z",
    "samples_count": 25,
    "metrics": {
      "loss": 0.023,
      "speaker_similarity": 0.92
    }
  }
]
```

---

### Rollback to Checkpoint

```
POST /profiles/{profile_id}/checkpoints/{checkpoint_id}/rollback
```

**Response (200 OK):**

```json
{
  "status": "success",
  "profile_id": "profile-uuid",
  "rolled_back_to": "v2",
  "current_version": "v2"
}
```

---

### Delete Checkpoint

```
DELETE /profiles/{profile_id}/checkpoints/{checkpoint_id}
```

**Response (200 OK):**

```json
{
  "status": "success",
  "checkpoint_id": "v2"
}
```

---

## WebSocket Events

Real-time updates via WebSocket connection at `/socket.io`.

### Training Progress

```json
{
  "event": "training_progress",
  "data": {
    "job_id": "uuid-string",
    "profile_id": "profile-uuid",
    "status": "running",
    "progress": 45,
    "current_epoch": 5,
    "total_epochs": 10,
    "current_loss": 0.034
  }
}
```

### Training Complete

```json
{
  "event": "training_complete",
  "data": {
    "job_id": "uuid-string",
    "profile_id": "profile-uuid",
    "status": "completed",
    "model_version": "v4",
    "metrics": {
      "final_loss": 0.023,
      "speaker_similarity": 0.92
    }
  }
}
```

---

## Error Response Format

All errors follow this format:

```json
{
  "error": "Human-readable error message",
  "message": "Technical details (debug mode only)",
  "error_code": "machine_readable_code",
  "details": {}
}
```

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| `/voice/clone` | 10/minute |
| `/training/jobs` (POST) | 5/minute |
| All other endpoints | 60/minute |

---

*Generated for AutoVoice v1.0 - Voice Profile & Continuous Training*
