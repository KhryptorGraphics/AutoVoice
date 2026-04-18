# AutoVoice

AutoVoice is a local-first singing voice conversion and karaoke stack. The current target is a reliable single-user MVP: one backend entrypoint, one frontend entrypoint, one Socket.IO realtime path, and durable local metadata for presets, conversion history, training jobs, checkpoints, and YouTube download history.

## Current Scope

- Offline song conversion to a trained voice profile
- Live karaoke sessions over Socket.IO
- Voice profile and training job management
- Persistent local product state under `DATA_DIR`
- React frontend for the supported MVP flows

## Canonical Docs

Start here when onboarding or validating current behavior:

- [Docs Index](docs/README.md)
- [Voice Profiles User Guide](docs/user-guide-voice-profiles.md)
- [Frontend Persistence Boundaries](docs/frontend-persistence-boundaries.md)
- [API Documentation](docs/api/README.md)
- [Troubleshooting](docs/troubleshooting.md)

## Canonical Runtime

Backend:

```bash
python -m pip install -e .
autovoice serve --host 0.0.0.0 --port 5000
```

Frontend:

```bash
cd frontend
npm ci
npm run dev
```

By default the frontend proxies `/api` and `/socket.io` to `http://localhost:5000`. Override with `VITE_BACKEND_URL` if needed.

Supported local CUDA targets:

- `x86_64` Linux with CUDA-capable NVIDIA GPUs
- `aarch64` Linux with CUDA-capable NVIDIA GPUs, including NVIDIA Thor / `sm_110`

## Data and Config

- Default config file: `config/gpu_config.yaml`
- Default data dir: `data/`
- Override data dir with `DATA_DIR=/path/to/data`
- Set `SECRET_KEY` in the environment for local runs if you are not using the secrets manager path

Durable app metadata is stored under:

```text
data/app_state/
```

## Docker

Local container runtime uses the same backend entrypoint:

```bash
docker compose up --build
```

Backend health is exposed at:

```text
/api/v1/health
```

## Repo Map

- `src/auto_voice/`: backend application code
- `src/auto_voice/web/`: Flask API, Socket.IO, karaoke, persistence
- `frontend/`: React/Vite frontend
- `config/`: runtime configuration
- `tests/`: backend and integration tests
- `docs/`, `conductor/`, `reports/`, `academic-research/`: supporting docs and research artifacts

## What Is Canonical vs Experimental

Canonical for the MVP:

- `autovoice serve`
- `/api/v1/*` REST API
- Socket.IO over `/socket.io`
- `/karaoke` namespace for live audio sessions
- JSON-backed local state in `data/app_state/`
- Source-artist profiles extracted from uploaded songs
- Target-user profiles trained with LoRA first and promoted to full-model training after 30 minutes of clean user vocals

Still experimental or research-heavy:

- model benchmarking and alternative pipelines outside the primary UI flows
- large model assets and generated artifacts stored in-repo
- broader deployment hardening beyond a trusted local single-user setup

## Verification

Backend sanity checks:

```bash
python -m compileall src tests
pytest tests/test_web_utils.py tests/test_web_api.py tests/test_pipeline_status_api.py
```

Frontend sanity checks:

```bash
cd frontend
npm run build
```
