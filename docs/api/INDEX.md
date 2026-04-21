# AutoVoice API Index

**Current contract date:** 2026-04-21

## Quick Links

- Swagger UI: `http://localhost:5000/docs`
- OpenAPI JSON: `http://localhost:5000/api/v1/openapi.json`
- OpenAPI YAML: `http://localhost:5000/api/v1/openapi.yaml`
- Realtime event matrix: [websocket-events.md](./websocket-events.md)

## Canonical Surfaces

- REST base: `/api/v1`
- Canonical profile routes: `/api/v1/voice/profiles/*`
- Compatibility-only sample/profile helpers: `/api/v1/profiles/*`
- Default Socket.IO namespace: `/`
- Karaoke Socket.IO namespace: `/karaoke`

There is no `/training` Socket.IO namespace in the current backend.

## Endpoint Index

### Conversion

- `POST /api/v1/convert/song`
- `GET /api/v1/convert/status/{job_id}`
- `GET /api/v1/convert/download/{job_id}`
- `POST /api/v1/convert/cancel/{job_id}`
- `GET /api/v1/convert/metrics/{job_id}`
- `GET /api/v1/convert/history`
- `PATCH /api/v1/convert/history/{record_id}`
- `DELETE /api/v1/convert/history/{record_id}`
- `GET /api/v1/convert/reassemble/{job_id}`

### Voice Profiles

- `POST /api/v1/voice/clone`
- `GET /api/v1/voice/profiles`
- `GET /api/v1/voice/profiles/{profile_id}`
- `DELETE /api/v1/voice/profiles/{profile_id}`
- `GET /api/v1/voice/profiles/{profile_id}/adapters`
- `GET /api/v1/voice/profiles/{profile_id}/model`
- `POST /api/v1/voice/profiles/{profile_id}/adapter/select`
- `GET /api/v1/voice/profiles/{profile_id}/adapter/metrics`
- `GET /api/v1/voice/profiles/{profile_id}/training-status`
- `POST /api/v1/profiles/auto-create`

### Training

- `GET /api/v1/training/jobs`
- `POST /api/v1/training/jobs`
- `GET /api/v1/training/jobs/{job_id}`
- `POST /api/v1/training/jobs/{job_id}/cancel`
- `POST /api/v1/training/jobs/{job_id}/pause`
- `POST /api/v1/training/jobs/{job_id}/resume`
- `GET /api/v1/training/jobs/{job_id}/telemetry`
- `POST /api/v1/training/preview/{job_id}`

### Sample and Audio Utilities

- `GET|POST /api/v1/profiles/{profile_id}/samples`
- `POST /api/v1/profiles/{profile_id}/samples/from-path`
- `GET|DELETE /api/v1/profiles/{profile_id}/samples/{sample_id}`
- `POST /api/v1/profiles/{profile_id}/samples/{sample_id}/filter`
- `GET /api/v1/profiles/{profile_id}/segments`
- `POST /api/v1/profiles/{profile_id}/songs`
- `POST /api/v1/audio/diarize`
- `POST /api/v1/audio/diarize/assign`
- `GET|POST /api/v1/audio/router/config`

### YouTube and System

- `POST /api/v1/youtube/info`
- `POST /api/v1/youtube/download`
- `GET /api/v1/health`
- `GET /api/v1/gpu/metrics`
- `GET /api/v1/kernels/metrics`
- `GET /api/v1/system/info`
- `GET /api/v1/devices/list`
- `GET|POST /api/v1/devices/config`
- `GET|POST /api/v1/config/separation`
- `GET|POST /api/v1/config/pitch`
- `GET|POST|PUT|DELETE /api/v1/presets`
- `GET|POST /api/v1/models/*`

## Workflow Reference

### Convert a Song

1. `POST /api/v1/convert/song`
2. watch `conversion_progress` on `/` or poll `GET /api/v1/convert/status/{job_id}`
3. `GET /api/v1/convert/download/{job_id}`

### Train a Voice Profile

1. `POST /api/v1/voice/clone`
2. `POST /api/v1/training/jobs`
3. watch `training_progress` on `/` and filter by `job_id`
4. optional controls: `pause`, `resume`, `telemetry`, `preview`
5. `GET /api/v1/voice/profiles/{profile_id}/training-status`

### Live Karaoke

1. preflight and session setup via karaoke REST endpoints
2. connect to `/karaoke`
3. stream live session events there only

## Notes

- Conversion jobs support explicit room joins via `join_job` and `leave_job`.
- Training events are emitted on the default namespace today; clients should filter by `job_id`.
- Use [README.md](./README.md) as the higher-level entrypoint and [websocket-events.md](./websocket-events.md) for payload details.
