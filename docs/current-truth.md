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

## Readiness Vocabulary And Current Status

Use these terms consistently. Avoid the unqualified phrase "production-ready" in
release notes or operator docs because it hides the support boundary.

| Tier | Current status | Boundary |
| --- | --- | --- |
| Local-only single-user | Supported MVP target | Trusted operator machine, local `DATA_DIR`, no external users, no public ingress. |
| Private/operator-controlled hosted | Conditional | Requires API auth, explicit CORS origins, media-consent gates, current-head health/readiness/preflight proof, and operator-owned media. |
| Public multi-user | Not ready | Requires account auth, per-user isolation, quotas, abuse review, retention/export/deletion policy, and current-head full hardware evidence. |
| Commercial launch | Not ready | Requires the public multi-user controls plus legal/policy approval for voice, likeness, copyright, biometric privacy, and platform terms. |

The latest local evidence source SHA is
`fb7b4c123b2d0d50735deabef6ecc8754a52351e`. A readiness claim must cite
artifacts whose embedded git SHA matches the candidate commit being released;
rerun the evidence commands after the final release commit is selected. During
the 2026-04-29 local-only pass, the immutable completion matrix at
`reports/completion/20260429T113605Z-fb7b4c12/completion_matrix.json` passed
with `ok: true` for local/no-Docker lanes, and the release-grade benchmark
evidence under `reports/local-evidence/20260429T113605Z-fb7b4c12/benchmarks/`
matched the same commit. `reports/benchmarks/latest/benchmark_dashboard.json`
and `reports/benchmarks/latest/release_evidence.json` also validated against
that source SHA with `scripts/validate_benchmark_dashboard.py --current-git-sha
--release-grade`.

The same pass ran the supported local completion matrix in `autovoice-thor`.
It passed GitNexus refresh, skip-audit policy, backend contract smoke tests,
compose-config validation without Docker deployment, benchmark dashboard build
and validation, experimental evidence validation, local hosted preflight,
frontend lint/typecheck/build, and the frontend browser smoke lane. Real audio
E2E, real compose, release-candidate compose, Jetson CUDA/TensorRT, TensorRT
engine, and TensorRT checkpoint parity lanes remain explicit skipped lanes unless
their hardware/deployment scope is enabled. MeanVC performance remains
experimental and is no longer a default local-only readiness gate; run it
explicitly with `AUTOVOICE_MEANVC_FULL=1` and the prepared MeanVC assets when
promoting that lane.

Do not treat every `latest` pointer as authoritative. `reports/completion/latest/
completion_matrix.json` still references `9c6a056378df7585c453ecbb4d1f964345287436`.
Treat that pointer as historical until deliberately republished for the candidate
commit. `reports/release-evidence/latest/release_decision.json` now references
`fb7b4c123b2d0d50735deabef6ecc8754a52351e`, but release decisions are still
scoped by the local/no-Docker support boundary unless hardware/deployment lanes
are explicitly enabled.

The latest closeout and post-release quality plan are not contradictory when read
with this vocabulary: AutoVoice has meaningful local/private deployment proof and
post-release quality work, but public/commercial production release remains blocked
until the release-candidate matrix, benchmark evidence, full supported pytest lanes,
and hardware/model lanes are current-head green or explicitly gated.

## Canonical Runtime

- backend entrypoint: `autovoice serve`
- canonical swarm entrypoint: `autovoice swarm`
- frontend entrypoint: `frontend/` Vite app
- canonical REST base: `/api/v1`
- canonical durable app-state store for training jobs, presets, and conversion history: `AppStateStore`
- canonical profile routes: `/api/v1/voice/profiles/*`
- compatibility helper routes: `/api/v1/profiles/*`
- canonical non-karaoke Socket.IO namespace: `/`
- dedicated live namespace: `/karaoke`
- canonical offline pipeline: `quality_seedvc`
- canonical fast/live pipeline: `realtime`
- experimental pipelines: `quality`, `quality_shortcut`, `realtime_meanvc`
- supported local train/serve contract: LoRA and full-model training artifacts
  are packaged for the canonical `realtime` serving path only. The canonical
  `quality_seedvc` offline path is reference-audio driven and must not be treated
  as consuming trained LoRA/full-model artifacts.
- canonical training feature contract: ContentVec content embeddings are 768
  dims, RMVPE/PitchEncoder pitch embeddings are 768 dims, and speaker embeddings
  are 256 dims. CoMoSVC training jobs and regression tests must use that contract.
- experimental quality upgrades remain behind the evidence gate defined in
  `config/experimental_evidence.json` and validated by
  `python scripts/validate_experimental_evidence.py`
- benchmark/release promotion evidence is defined by `config/benchmark_suites.json` and
  validated by `python scripts/validate_benchmark_dashboard.py`
- production completion evidence is written by `python scripts/run_completion_matrix.py`
  under `reports/completion/latest/`

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
- [swarm-operator-contract.md](./swarm-operator-contract.md)
- [api/README.md](./api/README.md)
- [user-guide-voice-profiles.md](./user-guide-voice-profiles.md)
- [troubleshooting.md](./troubleshooting.md)

## Governance And Swarm State

- `bd` is the canonical task and planning source of truth.
- GitNexus is the canonical code-intelligence layer for repo exploration and impact analysis.
- `config/swarm_config.yaml`, `config/agent_contexts.yaml`, `config/swarm_manifests/*.yaml`, and
  `autovoice swarm ...` define the canonical DAG-based swarm runner.
- `docs/swarm-operator-contract.md` is the canonical operator contract for run taxonomy, required
  GitNexus inputs, required MemKraft writes, artifact paths, and lane completion rules.
- `scripts/launch_swarms.sh` and `scripts/swarm_orchestrator.py` are compatibility wrappers around
  the repo-native swarm runner, not separate orchestration systems.
- MemKraft is now the preferred durable swarm memory backend for `autovoice swarm` runs when the
  Python package is installed. Run ledgers under `DATA_DIR/swarm_runs/` remain the canonical
  execution record, and `DATA_DIR/swarm_memory/` stores the MemKraft channel/task/agent context.
- `autovoice swarm status` reports live ledger state before completion, and `autovoice swarm cancel`,
  `autovoice swarm resume`, and `autovoice swarm retry` are the canonical run-control commands.
- GitNexus remains the required code-context input; do not treat swarm memory as a replacement for
  fresh code-graph inspection.

## Historical Artifacts

These are useful for archaeology but are not the current product spec:

- `conductor/`
- `ORCHESTRATOR*.md`
- date-stamped coverage and readiness reports under `docs/`
- generated runtime reports under `reports/`, `output/`, and `output/reports/`
- older swarm/claude-flow helper scripts and notes

Use those only after validating against the canonical docs and live code paths.

## Artifact And Fixture Policy

See [repo-hygiene.md](./repo-hygiene.md) for the full policy. The short version:

- canonical source lives in `src/`, `frontend/`, `config/`, `scripts/`, `tests/`,
  `docs/`, and explicitly tracked fixtures
- canonical evidence is generated under ignored `reports/` paths and is valid for
  release claims only when its embedded git SHA matches the candidate commit
- generated media/model outputs under `output/` are not tracked; do not add new
  generated media/model artifacts without documenting the owner, purpose, and
  retention rule as an explicit fixture exception
- voice-profile export includes profile metadata, training samples, profile-scoped
  app-state, registered owned asset references, and recent audit events; purge
  removes profile-linked app-state plus registered owned asset references while
  retaining durable audit records
- local runtime state, scratch outputs, and swarm run artifacts belong in ignored
  report/data paths, not in the root repository

## Current Verification Entry Points

- backend contract slice: targeted pytest suites in `tests/`, run with
  `PYTHONNOUSERSITE=1`, `PYTHONPATH=src`, and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`
- canonical local test entrypoint: `./run_tests.sh`
- generated API docs: `/api/v1/openapi.json`, `/api/v1/openapi.yaml`, `/docs`
- frontend build: `cd frontend && npm run build`
- release-candidate validation: `python scripts/validate_release_candidate.py --base-url http://127.0.0.1:10001 --wait-seconds 180`
  This validates `/api/v1/health`, `/ready`, and `/api/v1/metrics`, and verifies benchmark evidence schema/provenance against `HEAD` or `GITHUB_SHA`.
- production completion matrix: `python scripts/run_completion_matrix.py`
  Use `--full` on a capable Jetson/compose/hosted runner; local smoke mode records unavailable frontend, compose, and hardware lanes as explicit skipped lanes.
- full hardware RC evidence preflight: `python scripts/preflight_full_hardware_rc.py --output reports/release_candidates/AV-j4cd/preflight.json --benchmark-report <current-head-benchmark-report.json>`
- full hardware RC evidence bundle: `python scripts/run_full_hardware_rc.py --benchmark-report <current-head-benchmark-report.json>`
- `<current-head-benchmark-report.json>` must be the raw `comprehensive_report.json`; the RC runner derives `release_evidence.json`.
- local Jetson release decision without Docker deployment: `python scripts/run_full_hardware_rc.py --bead-id local-jetson --deployment-base-url http://127.0.0.1:10600 --local-base-url http://127.0.0.1:10600 --no-require-hosted-probes --no-require-production-smoke-stems --no-run-real-compose --no-run-full-hosted-preflight --no-require-docker --require-clean-head --benchmark-report <current-head-benchmark-report.json>`
- benchmark dashboard contract validation: `python scripts/validate_benchmark_dashboard.py`
- hosted deployment preflight: `python scripts/validate_hosted_deployment.py --hostname autovoice.giggahost.com`
- experimental evidence validation: `python scripts/validate_experimental_evidence.py`
- Jetson/TensorRT validation: `bash scripts/validate_cuda_stack.sh --pipeline all --output-dir reports/platform`
