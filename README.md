# AutoVoice

AutoVoice is a local-first singing voice conversion and karaoke stack. The current target is a reliable single-user MVP: one backend entrypoint, one frontend entrypoint, one Socket.IO realtime path, and durable local metadata for presets, conversion history, training jobs, checkpoints, and YouTube download history.

## Current Scope

- Offline song conversion to a trained voice profile
- Live karaoke sessions over Socket.IO
- Browser sing-along capture for recording a local/LAN user's microphone while
  playing the source song through that user's browser output device
- Voice profile and training job management
- Persistent local product state under `DATA_DIR`
- React frontend for the supported MVP flows

## Canonical Docs

Start here when onboarding or validating current behavior:

- [Docs Index](docs/README.md)
- [Current Truth](docs/current-truth.md)
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

Local HTTPS for browser microphone capture:

```bash
scripts/local_https_dev.sh
```

This starts the backend on `https://<host>:5443` and the frontend on
`https://<host>:3443` with a self-signed certificate in `.local/https/`. Use it
for same-machine HTTPS testing now and for trusted LAN browser testing after
the browser accepts or trusts the generated certificate. Override the advertised
browser host with `AUTOVOICE_LAN_HOST=<lan-ip-or-hostname>` when needed.

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
- `docs/`: canonical product, API, and operator docs
- `docs/repo-hygiene.md`: canonical source, fixture, artifact, and release-evidence policy
- `conductor/`, `ORCHESTRATOR*.md`, root completion summaries, `reports/`, `output/`, `academic-research/`: historical planning, research, generated output, or release-evidence artifacts

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
- large model assets that are still intentionally shipped in-repo for local-first evaluation
- broader deployment hardening beyond a trusted local single-user setup
- public multi-user or commercial launch without account isolation, quotas, abuse review, retention/export/deletion policy, current-head full hardware evidence, and legal/policy approval
- generated dependency trees and report outputs under `node_modules/`, `reports/`, and `output/`
  are not canonical source and should remain ignored/untracked

## Verification

Backend sanity checks:

```bash
python -m pip install -e .[dev]
python -c "import auto_voice"
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m compileall src tests
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/test_web_utils.py tests/test_web_api.py tests/test_pipeline_status_api.py -q
scripts/validate_compose_config.sh
```

Frontend sanity checks:

```bash
cd frontend
npm run build
```
