# AutoVoice API Documentation

This directory documents the current single-user MVP API surface.

## Canonical API Shape

- Base URL: `http://localhost:5000/api/v1`
- Canonical voice profile surface: `/api/v1/voice/profiles/*`
- Compatibility surface still present for one-release migration paths: `/api/v1/profiles/*`
- Sample CRUD remains on `/api/v1/profiles/*`, but the shared web API handler is the single registered route owner.
- Canonical non-karaoke Socket.IO namespace: `/`
- Dedicated live namespace: `/karaoke`

The backend does not expose a separate `/training` namespace.

## Current Profile Model

- `source_artist`: extracted from uploaded songs after separation and diarization
- `target_user`: user-owned singing profile used for LoRA or full-model training

Only `target_user` profiles are trainable. Full-model training unlocks after `30 minutes` of clean target-user vocals.

## Documentation Files

- [tutorials.md](./tutorials.md): workflow examples for conversion, training, karaoke, and YouTube ingestion
- [websocket-events.md](./websocket-events.md): realtime event matrix and payload examples
- [INDEX.md](./INDEX.md): endpoint index and workflow summary
- [SUMMARY.md](./SUMMARY.md): docs implementation status and validation notes
- [postman_collection.json](./postman_collection.json): API request collection

## Endpoint Groups

### Conversion

- `POST /convert/song`
- `GET /convert/status/{job_id}`
- `GET /convert/download/{job_id}`
- `POST /convert/cancel/{job_id}`
- `GET /convert/metrics/{job_id}`
- `GET /convert/history`
- `PATCH /convert/history/{id}`
- `DELETE /convert/history/{id}`
- `GET /convert/reassemble/{job_id}`

### Voice Profiles

- `POST /voice/clone`
- `GET /voice/profiles`
- `GET /voice/profiles/{id}`
- `DELETE /voice/profiles/{id}`
- `GET /voice/profiles/{id}/adapters`
- `GET /voice/profiles/{id}/model`
- `POST /voice/profiles/{id}/adapter/select`
- `GET /voice/profiles/{id}/adapter/metrics`
- `GET /voice/profiles/{id}/training-status`
- `POST /profiles/auto-create`

### Training

- `GET /training/jobs`
- `POST /training/jobs`
- `GET /training/jobs/{id}`
- `POST /training/jobs/{id}/cancel`
- `POST /training/jobs/{id}/pause`
- `POST /training/jobs/{id}/resume`
- `GET /training/jobs/{id}/telemetry`
- `POST /training/preview/{id}`

### Samples and Audio Processing

- `GET|POST /profiles/{id}/samples`
- `POST /profiles/{id}/samples/from-path`
- `GET|DELETE /profiles/{id}/samples/{sample_id}`
- `POST /profiles/{id}/samples/{sample_id}/filter`
- `GET /profiles/{id}/segments`
- `POST /profiles/{id}/songs`
- `POST /audio/diarize`
- `POST /audio/diarize/assign`
- `GET|POST /audio/router/config`

### YouTube and System

- `POST /youtube/info`
- `POST /youtube/download`
- `GET /health`
- `GET /gpu/metrics`
- `GET /kernels/metrics`
- `GET /system/info`
- `GET /devices/list`
- `GET|POST /devices/config`
- `GET|POST /config/separation`
- `GET|POST /config/pitch`
- `GET|POST|PUT|DELETE /presets`
- `GET|POST /models/*`

## Training Request Shape

Training creation uses a nested `config` object:

```json
{
  "profile_id": "profile_550e8400",
  "sample_ids": ["sample-a", "sample-b"],
  "config": {
    "training_mode": "lora",
    "epochs": 100,
    "batch_size": 8,
    "learning_rate": 0.0001,
    "adapter_type": "unified"
  }
}
```

If `sample_ids` is omitted, the backend uses all available samples for the profile.

## WebSocket Summary

Default namespace (`/`):

- room handshake: `join_job`, `leave_job`, `joined_job`, `left_job`
- conversion events: `job_created`, `job_progress`, `job_completed`, `job_failed`
- frontend conversion aliases: `conversion_progress`, `conversion_complete`, `conversion_error`
- training canonical events: `training.started`, `training.progress`, `training.completed`, `training.failed`, `training.paused`, `training.resumed`, `training.cancelled`
- compatibility training aliases: `training_progress`, `training_complete`, `training_error`, `training_paused`, `training_resumed`, `training_cancelled`

Dedicated live namespace (`/karaoke`):

- live session control and audio streaming only

## Generated Docs

- Swagger UI: `http://localhost:5000/docs`
- OpenAPI JSON: `http://localhost:5000/api/v1/openapi.json`
- OpenAPI YAML: `http://localhost:5000/api/v1/openapi.yaml`
