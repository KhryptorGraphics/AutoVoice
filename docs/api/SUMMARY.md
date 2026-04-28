# API Docs Summary

This summary tracks the current maintained API documentation scope, not the older 2026-02 rollout snapshot.

## Current Truth

- Canonical REST base is `/api/v1`
- Canonical profile surface is `/api/v1/voice/profiles/*`
- Compatibility helper routes still exist under `/api/v1/profiles/*`
- `/api/v1/profiles/{id}/samples*` routes have one registered handler family; legacy DB functions are callable compatibility delegates, not duplicate Flask rules.
- Default Socket.IO namespace (`/`) carries conversion and training events
- `/karaoke` is the dedicated live-session namespace
- The maintained training REST surface includes:
  - `GET /training/jobs`
  - `POST /training/jobs`
  - `GET /training/jobs/{job_id}`
  - `POST /training/jobs/{job_id}/cancel`
  - `POST /training/jobs/{job_id}/pause`
  - `POST /training/jobs/{job_id}/resume`
  - `GET /training/jobs/{job_id}/telemetry`
  - `POST /training/preview/{job_id}`

## Validation Expectations

The docs are considered in sync when all of the following stay true:

- `/api/v1/openapi.json` returns a valid OpenAPI 3 document
- `/api/v1/openapi.yaml` is parseable YAML
- `/docs` serves Swagger UI
- the generated spec includes the training control and preview routes
- markdown docs do not describe a separate `/training` Socket.IO namespace

## Related Files

- [README.md](./README.md)
- [INDEX.md](./INDEX.md)
- [websocket-events.md](./websocket-events.md)
- [tutorials.md](./tutorials.md)
- [../../src/auto_voice/web/api_docs.py](../../src/auto_voice/web/api_docs.py)
- [../../tests/test_api_docs_contract.py](../../tests/test_api_docs_contract.py)
