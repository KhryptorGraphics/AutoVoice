# Repository Hygiene And Artifact Policy

This policy separates canonical source from local evidence, fixtures, and
historical artifacts. It is intentionally conservative: do not delete or move a
tracked artifact unless you have confirmed it is not referenced by tests,
documentation, release evidence, or another active worker.

## Artifact Classes

| Class | Canonical location | Track in git? | Rule |
| --- | --- | --- | --- |
| Source and docs | `src/`, `frontend/`, `config/`, `scripts/`, `tests/`, `docs/` | Yes | Current product truth and executable contracts. |
| Canonical test fixtures | `tests/fixtures/`, explicitly referenced `tests/quality_samples/`, explicitly referenced `test_audio/` samples | Yes, only when small or intentionally real-audio | Must be named, referenced by tests, and kept stable. Invalid fixtures must only be used by rejection tests. |
| Vendor model repos | `models/hq-svc`, `models/meanvc`, `models/seed-vc` gitlinks/submodules | Yes as gitlinks | Parent repo tracks the gitlink contract. Local dirt inside vendor repos is not an AutoVoice source change. |
| Runtime state | `DATA_DIR`, `data/`, logs, checkpoints, app-state, uploads, generated model outputs | No | Keep ignored and operator-owned. Do not move runtime state into source paths. |
| Release evidence | `reports/completion/`, `reports/benchmarks/`, `reports/release_candidates/`, `reports/release-evidence/`, `reports/platform/` | No | Archive under immutable timestamp/SHA paths; `latest/` is a mutable convenience copy. |
| Historical reports | date-stamped docs, `conductor/`, `ORCHESTRATOR*.md`, root completion summaries | Usually yes until an explicit cleanup issue removes them | Archaeology only; never current-product truth without checking canonical docs and live code. |
| Scratch outputs | `output/`, local screenshots, temporary audio/model files, swarm run artifacts | No | Store under ignored report/data/scratch paths and delete when no longer needed. |

## Current Repository Notes

- The canonical docs are indexed from [docs/README.md](./README.md) and
  [current-truth.md](./current-truth.md).
- `reports/` is ignored on purpose. Release claims must reference the immutable
  evidence path and the embedded git SHA, not just `reports/*/latest`.
- Existing tracked real-audio samples under `tests/quality_samples/` and
  `test_audio/` are fixtures or historical quality samples. Keep them stable
  unless a dedicated fixture cleanup issue replaces them with smaller,
  deterministic alternatives and updates tests.
- Generated outputs under `output/` are ignored and must stay untracked unless a
  dedicated fixture exception documents the owner, test reference, and retention
  rule.
- Root-level generated summaries from earlier workstreams are historical reports.
  Prefer moving future summaries into `reports/` or `docs/` with a clear status
  note instead of adding more root artifacts.

## Adding New Fixtures

Before adding a fixture:

1. Prefer synthetic deterministic data generated inside the test.
2. If real audio is required, keep the shortest useful clip and document why
   synthetic data is insufficient.
3. Add a test reference in the same change; unreferenced fixtures are artifacts.
4. Confirm rights/consent for any voice or commercial media.
5. Keep expected outputs small. Large outputs should be regenerated during
   release or benchmark runs and stored under ignored `reports/` paths.

## Cleaning Existing Artifacts

Use this order for cleanup work:

1. Prove whether the path is referenced with `rg` and targeted tests.
2. If the path is generated evidence, move future writes to ignored `reports/`
   or `DATA_DIR` locations instead of committing the output.
3. If a tracked fixture is still required, document it in this policy or in the
   owning test.
4. If a tracked artifact is unreferenced and noncanonical, remove it in a
   dedicated cleanup change with before/after size notes and targeted test
   output.

This policy is enforced by `run_repo_boundary_audit`; tracked files under
`reports/` or `output/` should fail release validation unless they are moved to a
documented fixture location.
