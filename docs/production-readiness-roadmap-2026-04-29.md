# AutoVoice Production Readiness Roadmap

Generated from the 2026-04-29 audit pass using GitNexus, MemKraft search, local repo inspection, and four read-only specialist audit lanes.

## Current Truth

AutoVoice is close to a supported single-user, local-only operator release, but it is not release-cleared because the evidence stack is stale or incomplete. Public or commercial deployment remains a separate future track.

Verified current facts:

- Current branch: `main`
- Current HEAD during audit: `b29f6b43cbab76ca90ce2bb888f00229bd91af8e`
- Correct Python environment: `autovoice-thor`, Python `3.12.12`
- Beads status before this roadmap: zero open issues despite remaining readiness gaps
- GitNexus graph refresh: `19,920` symbols, `51,626` relationships, `300` flows
- GitNexus embedding refresh: blocked by a local Kuzu lock from active GitNexus MCP holders
- Frontend lint: passed
- Experimental evidence validation: passed
- Benchmark dashboard validation: failed because `reports/benchmarks/latest/benchmark_dashboard.json` and `reports/benchmarks/latest/release_evidence.json` are missing
- Completion matrix latest: `ok=false`, stale SHA `9c6a056378df7585c453ecbb4d1f964345287436`
- Release decision latest: stale SHA `f0d37ef01f2accd031cc134b84ea8e00501a82e2`, blockers include tegrastats and unexecuted hardware lanes

Implementation update from the 2026-04-29 local-only pass:

- `reports/benchmarks/latest/benchmark_dashboard.json` and `release_evidence.json`
  were regenerated locally for the then-current worktree baseline.
- `scripts/validate_benchmark_dashboard.py --current-git-sha --release-grade` passes.
- Completion matrix and hardware release-decision evidence remain stale and must
  be regenerated on the target Jetson/local runner for the selected candidate
  commit before any broader release claim.

## Orchestration Contract

Use beads for all work. Each sprint starts by claiming its bead, querying GitNexus for affected flows, and storing a MemKraft note with the sprint goal, files touched, validation run, and remaining risks.

Use four-agent swarm batches to avoid the local thread limit:

- Research swarm: security, ML/training, frontend/operator UX, QA/evidence
- Development swarm: disjoint write scopes only
- Review swarm: security, Python, TypeScript, QA/evidence
- Testing swarm: backend contracts, frontend build/lint/typecheck, release evidence, hardware/local RC

Do not run Docker deployment work for the local-only track. Docker, hosted Apache, public ingress, and commercial governance belong to the optional public/commercial track unless explicitly re-scoped.

## Phase 0 - Backlog And Evidence Baseline

Goal: replace closed-but-incomplete roadmap state with open executable work.

Sprint 0.1 - Reopen the real backlog

- Create one new epic for readiness gap closure.
- Add child beads for every sprint below.
- Link acceptance criteria to concrete validation commands and artifact paths.
- Store the roadmap summary in MemKraft.

Sprint 0.2 - Current-head evidence baseline

- Run `conda run -n autovoice-thor python --version`.
- Run backend targeted contract tests in `autovoice-thor`.
- Run frontend `npm run lint`, `npm run typecheck`, and `npm run build`.
- Run `conda run -n autovoice-thor python scripts/validate_experimental_evidence.py`.
- Run `conda run -n autovoice-thor python scripts/validate_benchmark_dashboard.py --current-git-sha`.
- Record all failures as bead comments before fixes.

Done when the project has open beads matching the real work and a current-head baseline artifact set.

## Phase 1 - Local Release Evidence Closure

Goal: make the local-only release decision reproducible from current HEAD.

Sprint 1.1 - Restore canonical benchmark artifacts

- Regenerate or repair `reports/benchmarks/latest/benchmark_dashboard.json`.
- Regenerate or repair `reports/benchmarks/latest/release_evidence.json`.
- Ensure `reports/benchmarks/latest/comprehensive_report.json` is either input evidence or explicitly noncanonical for validator purposes.
- Make `scripts/validate_benchmark_dashboard.py --current-git-sha` pass on current HEAD.

Sprint 1.2 - Regenerate no-Docker local RC evidence

- Run the local Jetson release decision path with `--no-require-docker`, `--no-run-real-compose`, and local base URLs.
- Ensure completion matrix and release decision embed current HEAD.
- Mark hosted/compose lanes as intentionally out of scope for local-only, not silent skips.

Sprint 1.3 - Reconcile operator docs

- Update `docs/current-truth.md` so current SHA references are current or intentionally absent.
- Ensure optimistic readiness docs point back to current-truth instead of making standalone production claims.
- Remove contradictions between completion matrix, release decision, and quality UX docs.

Done when `latest/` evidence and immutable SHA-named evidence agree on HEAD and local-only scope.

## Phase 2 - Training And Model Contract Closure

Goal: make training artifacts, runtime manifests, and supported pipelines tell one story.

Sprint 2.1 - Pick the supported training-serving contract

- Decide the local MVP contract: either trained artifacts support only the `realtime` lane, or `quality_seedvc` must consume trained-profile artifacts in a meaningful tested way.
- Encode that contract in runtime manifests and readiness checks.
- Block impossible offline promotion states in API responses and UI copy.

Sprint 2.2 - Align feature dimensions

- Resolve `CoMoSVCDecoder(pitch_dim=256)` versus `SoVitsSvc(pitch_dim=768)` contract drift.
- Retire, gate, or fix the noncanonical model path from readiness-critical tests.
- Add regression tests that fail if the selected training model and feature encoder dimensions diverge.

Sprint 2.3 - Repair sample-quality fixtures

- Replace invalid test fixtures with minimal voiced samples that satisfy current QA policy.
- Add explicit positive and negative tests for training sample QA.
- Ensure profile storage, training job creation, and LoRA/full-model lifecycle tests use valid fixtures.

Done when training lifecycle tests are green under one declared local MVP contract.

## Phase 3 - Benchmark And Quality Gate Closure

Goal: enforce one benchmark policy for canonical pipelines.

Sprint 3.1 - Normalize metric policy

- Decide release semantics for `speaker_similarity_mean`, `pitch_corr_mean`, `mcd_mean`, and latency.
- Stop disabling MCD gating for canonical pipelines unless an explicit exemption with basis is emitted and validated.
- Update benchmark tests to match the chosen policy.

Sprint 3.2 - Regenerate release-grade quality evidence

- Run benchmark generation with release-quality fixtures.
- Require nonzero pitch correlation and plausible MCD for canonical lanes or a documented validator-approved exemption.
- Keep HQ-SVC, MeanVC, TensorRT, and shortcut pipelines experimental unless promoted by evidence.

Done when the benchmark validator passes and the operator status page can render real green/blocked status from current artifacts.

## Phase 4 - Local Operator UX Closure

Goal: make the UI reflect real local readiness and avoid flows that break in auth-enabled mode.

Sprint 4.1 - Evidence panel contract

- Align `/api/v1/reports/benchmarks/latest`, `/api/v1/reports/release-evidence/latest`, and `SystemStatusPage` with the same latest JSON files.
- Show current SHA, stale SHA, failed lanes, skipped lanes, and next actions.
- Make missing evidence a clear blocked state, not ambiguous unavailability.

Sprint 4.2 - Ingest consent and asset-ID UX

- Add frontend consent controls for uploads, YouTube ingest, voice clone, and conversion workflow inputs.
- Send `consent_confirmed` and `source_media_policy_confirmed` where backend policy requires them.
- Make YouTube add-to-profile use `audio_asset_id` in auth/public mode and fall back to raw `audio_path` only in trusted local mode.

Sprint 4.3 - Training UX truthfulness

- Either implement backend-backed filtered upload or change copy so users understand filtering is not done during upload.
- Add an operator token status/entry surface for auth-enabled local/private runs.

Done when frontend lint, typecheck, build, and focused operator UI tests pass.

## Phase 5 - Security And Governance Hardening

Goal: make private/operator-hosted mode defensible while keeping local-only simple.

Sprint 5.1 - Socket.IO auth parity

- Apply API-token authorization to the `/karaoke` Socket.IO namespace when auth is required.
- Require auth for connect and session-mutating events.
- Add regression tests for allowed and denied websocket auth.

Sprint 5.2 - Export, purge, and audit semantics

- Make profile purge delete all owned registered assets under managed roots.
- Document and test retained audit records.
- Ensure profile export returns a complete inventory of profile-linked data and retained audit status.

Sprint 5.3 - Production-safe error surfaces

- Replace raw exception strings in auth/public responses with stable user-safe errors.
- Keep detailed exceptions in server logs with request IDs.
- Add tests that internal paths and tool output do not leak in auth-enabled mode.

Done when private/operator-hosted mode can pass auth, CORS, rate-limit, websocket, export/purge, and error-redaction tests.

## Phase 6 - Reproducibility And Repo Hygiene

Goal: reduce drift and make the next audit cheaper.

Sprint 6.1 - Dependency lock strategy

- Choose a lock mechanism for `autovoice-thor` such as `conda-lock` or generated explicit conda specs.
- Reconcile broad backend runtime pins with reproducible local installs.
- Keep frontend `package-lock.json` as the canonical frontend lock.

Sprint 6.2 - GitNexus embedding unblock

- Close stale GitNexus MCP/index holders or run embedding refresh in a clean shell.
- Regenerate GitNexus with embeddings.
- Commit only intentional metadata changes.

Sprint 6.3 - Historical artifact cleanup

- Archive or label root-level historical summaries and generated coverage JSON.
- Keep generated media, reports, and model outputs ignored unless explicitly documented as fixtures.
- Make `docs/current-truth.md` the single operator-facing truth index.

Done when repo hygiene checks, GitNexus graph, and docs agree without historical readiness noise.

## Phase 7 - Optional Public/Commercial Track

Only start this after local-only release evidence is green or if the scope changes.

Required work:

- Account auth and per-user storage isolation
- Persistent quotas and job abuse controls
- Public ingress threat model
- External deletion/export guarantees
- Biometric, copyright, source-media, and impersonation policy approval
- Moderation/review workflows for prohibited content
- Hosted observability, rollback, and incident playbooks

Done when public/commercial launch blockers are closed with current-head evidence and legal/product approval.

## Final Release Gate

The epic is done only when all required local-only beads are closed and these gates pass:

- `conda run -n autovoice-thor python -m compileall src tests`
- Targeted backend contract tests in `autovoice-thor`
- Full supported pytest suite or explicitly scoped local MVP suite with documented skips
- `cd frontend && npm run lint && npm run typecheck && npm run build`
- `conda run -n autovoice-thor python scripts/validate_experimental_evidence.py`
- `conda run -n autovoice-thor python scripts/validate_benchmark_dashboard.py --current-git-sha`
- No-Docker local Jetson RC evidence bundle with current HEAD
- `bd ready` shows no unblocked required local-only work
- MemKraft contains final sprint handoff and residual-risk note
