# Swarm Runner

`autovoice swarm` is the canonical repo-native swarm runner.

It executes explicit YAML manifests, persists run state under `DATA_DIR/swarm_runs/`, and writes:

- `manifest.snapshot.json`
- `ledger.json`
- `completion.json`
- per-task logs under `tasks/`

## Commands

```bash
autovoice swarm validate --manifest config/swarm_manifests/development.yaml
autovoice swarm run --manifest config/swarm_manifests/full.yaml --run-id rc-dry-run --dry-run
autovoice swarm status --run-id rc-dry-run
```

## Canonical Inputs

- backlog context: `bd`
- code-intelligence context: GitNexus
- memory fallback: the available `memory` MCP

A dedicated MemKraft MCP server is not installed in this workspace today, so manifests and ledgers are the durable swarm memory layer.

## Compatibility Wrappers

- `scripts/launch_swarms.sh`
- `scripts/swarm_orchestrator.py`

Those wrappers call `autovoice swarm`; they are no longer separate orchestration runtimes.
