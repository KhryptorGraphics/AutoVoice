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
- canonical swarm entrypoint: `autovoice swarm`
- frontend entrypoint: `frontend/` Vite app
- canonical REST base: `/api/v1`
- canonical profile routes: `/api/v1/voice/profiles/*`
- compatibility helper routes: `/api/v1/profiles/*`
- canonical non-karaoke Socket.IO namespace: `/`
- dedicated live namespace: `/karaoke`
- canonical offline pipeline: `quality_seedvc`
- canonical fast/live pipeline: `realtime`
- experimental pipelines: `quality`, `quality_shortcut`, `realtime_meanvc`
- experimental quality upgrades remain behind the evidence gate defined in
  `config/experimental_evidence.json` and validated by
  `python scripts/validate_experimental_evidence.py`

## Vendor Model Repos

The nested model directories are canonical vendor repos tracked as gitlinks/submodules:

- `models/hq-svc`
- `models/meanvc`
- `models/seed-vc`

Initialize or refresh them with:

```bash
git submodule update --init --recursive
```

Audit their parent-repo contract and local hygiene with:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src /home/kp/anaconda3/envs/autovoice-thor/bin/python \
  scripts/audit_vendor_repos.py
```

Use `--require-clean` only when you explicitly want local dirt in those vendor repos to fail the audit.
Local experiments or runtime artifacts inside nested vendor repos are not parent-repo product changes until
they are committed in the vendor repo and the parent gitlink is updated intentionally.

## Canonical Training SSIM

There is one supported SSIM implementation on the current training path:

- `src/auto_voice/models/so_vits_svc.py::_ssim_loss`

There is no separate canonical `src/auto_voice/training/ssim_loss.py` module.

There is no separate `/training` Socket.IO namespace in the current backend.

## Canonical Docs

Read these first:

- [../README.md](../README.md)
- [README.md](./README.md)
- [api/README.md](./api/README.md)
- [user-guide-voice-profiles.md](./user-guide-voice-profiles.md)
- [troubleshooting.md](./troubleshooting.md)

## Governance And Swarm State

- `bd` is the canonical task and planning source of truth.
- GitNexus is the canonical code-intelligence layer for repo exploration and impact analysis.
- `config/swarm_config.yaml`, `config/agent_contexts.yaml`, `config/swarm_manifests/*.yaml`, and
  `autovoice swarm ...` define the canonical DAG-based swarm runner.
- `scripts/launch_swarms.sh` and `scripts/swarm_orchestrator.py` are compatibility wrappers around
  the repo-native swarm runner, not separate orchestration systems.
- MemKraft is now the preferred durable swarm memory backend for `autovoice swarm` runs when the
  Python package is installed. Run ledgers under `DATA_DIR/swarm_runs/` remain the canonical
  execution record, and `DATA_DIR/swarm_memory/` stores the MemKraft channel/task/agent context.
- GitNexus remains the required code-context input; do not treat swarm memory as a replacement for
  fresh code-graph inspection.

## Historical Artifacts

These are useful for archaeology but are not the current product spec:

- `conductor/`
- `ORCHESTRATOR*.md`
- date-stamped coverage and readiness reports under `docs/`
- generated runtime reports under `reports/` and `output/reports/`
- older swarm/claude-flow helper scripts and notes

Use those only after validating against the canonical docs and live code paths.

## Current Verification Entry Points

- backend contract slice: targeted pytest suites in `tests/`, run with
  `PYTHONNOUSERSITE=1`, `PYTHONPATH=src`, and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`
- canonical local test entrypoint: `./run_tests.sh`
- generated API docs: `/api/v1/openapi.json`, `/api/v1/openapi.yaml`, `/docs`
- frontend build: `cd frontend && npm run build`
- release-candidate validation: `python scripts/validate_release_candidate.py --base-url http://127.0.0.1:5000`
- experimental evidence validation: `python scripts/validate_experimental_evidence.py`
- Jetson/TensorRT validation: `bash scripts/validate_cuda_stack.sh --pipeline all --output-dir reports/platform`
