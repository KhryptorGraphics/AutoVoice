# Current Truth

This document is the operator-facing summary of what is canonical in the repo today.

## Product Boundary

AutoVoice currently targets a reliable single-user, local-first workflow:

- create and manage target-user voice profiles
- ingest source-artist material from uploaded songs
- train target-user profiles
- run offline conversion jobs
- run live karaoke sessions
- persist local product state under `DATA_DIR`

## Canonical Runtime

- backend entrypoint: `autovoice serve`
- frontend entrypoint: `frontend/` Vite app
- canonical REST base: `/api/v1`
- canonical profile routes: `/api/v1/voice/profiles/*`
- compatibility helper routes: `/api/v1/profiles/*`
- canonical non-karaoke Socket.IO namespace: `/`
- dedicated live namespace: `/karaoke`

There is no separate `/training` Socket.IO namespace in the current backend.

## Canonical Docs

Read these first:

- [../README.md](../README.md)
- [README.md](./README.md)
- [api/README.md](./api/README.md)
- [user-guide-voice-profiles.md](./user-guide-voice-profiles.md)
- [troubleshooting.md](./troubleshooting.md)

## Historical Artifacts

These are useful for archaeology but are not the current product spec:

- `conductor/`
- `ORCHESTRATOR*.md`
- date-stamped coverage and readiness reports under `docs/`
- older swarm/claude-flow helper scripts and notes

Use those only after validating against the canonical docs and live code paths.

## Current Verification Entry Points

- backend contract slice: targeted pytest suites in `tests/`
- generated API docs: `/api/v1/openapi.json`, `/api/v1/openapi.yaml`, `/docs`
- frontend build: `cd frontend && npm run build`
