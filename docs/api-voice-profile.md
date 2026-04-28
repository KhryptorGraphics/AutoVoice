# Voice Profile And Training API

> Secondary reference: for the canonical API index, start with [docs/api/README.md](./api/README.md) and [docs/api/websocket-events.md](./api/websocket-events.md).

This document describes the current single-user MVP contract for voice profiles, training samples, and training jobs.

## Route Ownership

AutoVoice currently has two profile route families:

- Canonical profile identity and runtime state routes live under `/api/v1/voice/profiles/*`
- Compatibility and ancillary profile workflows still live under `/api/v1/profiles/*`

Current ownership split:

| Surface | Canonical Route Family | Notes |
| --- | --- | --- |
| clone a new profile | `/api/v1/voice/clone` | Current create flow |
| list profiles | `/api/v1/voice/profiles` | Canonical |
| get profile | `/api/v1/voice/profiles/{profile_id}` | Canonical |
| delete profile | `/api/v1/voice/profiles/{profile_id}` | Canonical |
| adapter/model/training status | `/api/v1/voice/profiles/{profile_id}/...` | Canonical |
| training samples | `/api/v1/profiles/{profile_id}/samples...` | Active compatibility path; implemented by the shared web API handler |
| checkpoints and quality history | `/api/v1/profiles/{profile_id}/...` | Legacy-but-active |
| auto-create from diarization | `/api/v1/profiles/auto-create` | Legacy-but-active |

There is no separate `/api/v1/voice/profiles/{profile_id}/samples` surface yet. Sample management remains on `/api/v1/profiles/*`, but duplicate same-method sample handlers are not registered; the shared web API handler owns the route and delegates to the DB-backed compatibility functions only when a profile is not present in the runtime store.

## Canonical Voice Profile Routes

### Create Voice Profile

`POST /api/v1/voice/clone`

Create a new voice profile from uploaded reference audio.

Common form fields:

- `reference_audio`: input audio file
- `user_id`: optional user identifier
- other role/source metadata may be accepted by the current backend implementation

### List Voice Profiles

`GET /api/v1/voice/profiles`

Optional query parameters:

- `user_id`: filter profiles by owner

### Get Voice Profile

`GET /api/v1/voice/profiles/{profile_id}`

Returns the serialized profile plus runtime metadata when available, such as selected adapter and adapter artifact information.

### Delete Voice Profile

`DELETE /api/v1/voice/profiles/{profile_id}`

Deletes the profile and associated stored state.

### Get Profile Adapters

`GET /api/v1/voice/profiles/{profile_id}/adapters`

Lists adapter artifacts available to the runtime and the currently selected adapter.

### Select Active Adapter

`POST /api/v1/voice/profiles/{profile_id}/adapter/select`

JSON body:

```json
{
  "adapter_type": "unified"
}
```

Accepted values today:

- `hq`
- `nvfp4`
- `unified`

Missing profiles return `404`.

### Get Adapter Metrics

`GET /api/v1/voice/profiles/{profile_id}/adapter/metrics`

Returns training/runtime metadata for available adapters.

### Get Runtime Model Info

`GET /api/v1/voice/profiles/{profile_id}/model`

Returns current runtime-facing model information, including whether a trained artifact exists and any embedding metadata.

### Get Profile Training Status

`GET /api/v1/voice/profiles/{profile_id}/training-status`

Returns summary state such as:

- `has_trained_model`
- `training_status`
- `model_version`

## Training Sample Routes

Training sample management is still exposed under `/api/v1/profiles/*`.

### List Samples

`GET /api/v1/profiles/{profile_id}/samples`

### Upload Sample

`POST /api/v1/profiles/{profile_id}/samples`

Accepted file field names:

- `file`
- `audio`

Optional form field:

- `metadata`: JSON-encoded metadata object
- `consent_confirmed` and `source_media_policy_confirmed`: required when `AUTOVOICE_REQUIRE_MEDIA_CONSENT=true`

### Add Sample From Server Path

`POST /api/v1/profiles/{profile_id}/samples/from-path`

JSON body:

```json
{
  "audio_path": "/absolute/path/on/server.wav",
  "skip_separation": false,
  "metadata": {}
}
```

When `AUTOVOICE_REQUIRE_MEDIA_CONSENT=true`, the JSON body must also include `consent_confirmed: true` and `source_media_policy_confirmed: true`.

### Get Or Delete A Sample

- `GET /api/v1/profiles/{profile_id}/samples/{sample_id}`
- `DELETE /api/v1/profiles/{profile_id}/samples/{sample_id}`

### Filter A Sample

`POST /api/v1/profiles/{profile_id}/samples/{sample_id}/filter`

## Training Job Routes

Training jobs use the dedicated `/api/v1/training/jobs/*` REST surface.

### List Training Jobs

`GET /api/v1/training/jobs`

Optional query parameter:

- `profile_id`

### Create Training Job

`POST /api/v1/training/jobs`

Example body:

```json
{
  "profile_id": "profile-id",
  "sample_ids": ["sample-1", "sample-2"],
  "config": {
    "training_mode": "lora",
    "epochs": 100,
    "batch_size": 8,
    "learning_rate": 0.0001
  }
}
```

Current rules:

- `profile_id` is required
- `config` must be an object when present
- `training_mode` must be `lora` or `full`
- only `target_user` profiles can be trained
- if `sample_ids` is omitted, the backend uses all available training samples for the profile

### Get Training Job

`GET /api/v1/training/jobs/{job_id}`

### Cancel Training Job

`POST /api/v1/training/jobs/{job_id}/cancel`

### Pause, Resume, Telemetry, And Preview

- `POST /api/v1/training/jobs/{job_id}/pause`
- `POST /api/v1/training/jobs/{job_id}/resume`
- `GET /api/v1/training/jobs/{job_id}/telemetry`
- `POST /api/v1/training/preview/{job_id}`

These routes are the current live training-control surface documented in OpenAPI and used by the frontend training monitor.

## WebSocket Contract

Training and conversion updates share the default Socket.IO namespace: `/`.

- conversion uses explicit room subscription via `join_job` and `leave_job`
- training does not currently have a dedicated room handshake
- training events should be filtered client-side by `job_id`

Canonical training events:

- `training.started`
- `training.progress`
- `training.completed`
- `training.failed`
- `training.paused`
- `training.resumed`
- `training.cancelled`

Compatibility aliases still emitted for active UI paths:

- `training_progress`
- `training_complete`
- `training_error`
- `training_paused`
- `training_resumed`
- `training_cancelled`

There is no separate `/training` Socket.IO namespace in the current backend.

## App Settings

The frontend’s durable default-pipeline settings live at:

- `GET /api/v1/settings/app`
- `PATCH /api/v1/settings/app`

Current stable fields:

- `preferred_offline_pipeline`
- `preferred_live_pipeline`

Legacy compatibility input:

- `preferred_pipeline`

## Related References

- [docs/api/README.md](./api/README.md)
- [docs/api/websocket-events.md](./api/websocket-events.md)
- [docs/current-truth.md](./current-truth.md)
