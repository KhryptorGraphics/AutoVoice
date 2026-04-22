# Swarm Operator Contract

This document is the canonical operator contract for `autovoice swarm`.

## Scope

Sprint 0 standardizes the repo-native swarm runner around one run taxonomy:

`program -> phase -> sprint -> lane -> artifact`

The current canonical values for the next-phase program are:

- `program`: `next-phase-perfection`
- `phase`: `hardening`
- `sprint`: `sprint-0`
- `lanes`: `research`, `development`, `review`, `testing`

## Required Inputs Per Lane

Every lane starts with GitNexus and uses `bd` as backlog context.

Research lane:

- `gitnexus query`: swarm runner, manifests, memory, docs
- `gitnexus context`: `execute_manifest`, `SwarmMemoryBackend`, lane-owning symbols
- artifacts: backlog snapshot, GitNexus status, sprint brief

Development lane:

- `gitnexus query`: symbols or processes being edited
- `gitnexus context`: owning modules before refactors
- artifacts: code changes, config changes, implementation notes

Review lane:

- `gitnexus query`: affected contracts and public interfaces
- `gitnexus context`: changed symbols and route owners
- artifacts: review notes, contract findings, residual risk summary

Testing lane:

- `gitnexus query`: impacted lifecycle or platform flows
- `gitnexus context`: test owners and changed runtime code
- artifacts: test commands, outputs, failure notes, handoff summary

## Required MemKraft Writes

Every run and lane must write these categories into MemKraft or the file fallback:

- `sprint_brief`
- `assumptions_and_decisions`
- `findings`
- `artifacts_produced`
- `test_outcomes`
- `handoff_summary`

The runner seeds these categories in run metadata so every manifest run shares the same schema.

## Run And Channel IDs

Top-level runs are created by the operator with an explicit run id:

```bash
autovoice swarm --data-dir data run --manifest config/swarm_manifests/full.yaml --run-id sprint0-bootstrap
```

Nested lane runs derive their ids from the parent run id and must reuse the same `--data-dir`:

- parent: `sprint0-bootstrap`
- research child: `sprint0-bootstrap-research`
- development child: `sprint0-bootstrap-development`

The runner also exports run-scoped identifiers into task environments:

- `AUTOVOICE_SWARM_RUN_ID`
- `AUTOVOICE_SWARM_PARENT_RUN_ID`
- `AUTOVOICE_SWARM_DATA_DIR`
- `AUTOVOICE_SWARM_CHANNEL_ID`
- `AUTOVOICE_SWARM_TASK_KEY`
- `AUTOVOICE_SWARM_LANE_KEY`
- `AUTOVOICE_SWARM_AGENT_KEY`
- `AUTOVOICE_SWARM_ARTIFACT_ROOT`

## Artifact Paths

Canonical operator-visible files live under:

- run snapshot: `DATA_DIR/swarm_runs/<run-id>/manifest.snapshot.json`
- live ledger: `DATA_DIR/swarm_runs/<run-id>/ledger.json`
- completion summary: `DATA_DIR/swarm_runs/<run-id>/completion.json`
- task logs: `DATA_DIR/swarm_runs/<run-id>/tasks/*.log`
- task artifact roots: `DATA_DIR/swarm_runs/<run-id>/artifacts/<lane>/<task-id>/`
- MemKraft or fallback memory: `DATA_DIR/swarm_memory/`

## Lane Completion Rules

A lane is complete only when all of the following are true:

1. GitNexus query/context work is captured in the run artifacts or notes.
2. Required MemKraft categories were written for the lane.
3. Produced artifacts are stored under the run-scoped artifact root.
4. The lane completion status is visible through `ledger.json` and `completion.json`.
5. Follow-up work is filed in `bd` before the session ends.

## Canonical Commands

```bash
autovoice swarm validate --manifest config/swarm_manifests/full.yaml
autovoice swarm --data-dir data run --manifest config/swarm_manifests/full.yaml --run-id sprint0-bootstrap --dry-run
autovoice swarm --data-dir data status --run-id sprint0-bootstrap
```
