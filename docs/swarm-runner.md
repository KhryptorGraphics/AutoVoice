# Swarm Runner

`autovoice swarm` is the canonical repo-native swarm runner.

It executes explicit YAML manifests as a dependency-aware DAG, persists run state under
`DATA_DIR/swarm_runs/`, and writes:

- `manifest.snapshot.json`
- `ledger.json`
- `completion.json`
- per-task logs under `tasks/`
- optional MemKraft channel/task/agent state under `DATA_DIR/swarm_memory/`

## Commands

```bash
autovoice swarm validate --manifest config/swarm_manifests/development.yaml
autovoice swarm run --manifest config/swarm_manifests/full.yaml --run-id rc-dry-run --dry-run
autovoice swarm status --run-id rc-dry-run
```

## Canonical Inputs

- backlog context: `bd`
- code-intelligence context: GitNexus
- durable swarm memory: MemKraft
- fallback if MemKraft import is unavailable: file-backed JSON under `DATA_DIR/swarm_memory/fallback/`

Each manifest can optionally declare `parallelism`, `lane`, and `role` metadata. The runner preserves
backward compatibility with older manifests that omit those fields.

## Compatibility Wrappers

- `scripts/launch_swarms.sh`
- `scripts/swarm_orchestrator.py`

Those wrappers call `autovoice swarm`; they are no longer separate orchestration runtimes.
