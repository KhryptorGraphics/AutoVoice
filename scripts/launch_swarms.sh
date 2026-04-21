#!/bin/bash
# Canonical launcher for the repo-native swarm manifest runner.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
source "$SCRIPT_DIR/common_env.sh"
autovoice_activate_env

usage() {
    cat <<EOF
Usage: $(basename "$0") [all|research|development|review|testing] [--dry-run] [--run-id ID]
       $(basename "$0") status <run-id>

Examples:
  ./scripts/launch_swarms.sh
  ./scripts/launch_swarms.sh research --dry-run
  ./scripts/launch_swarms.sh status run-123
EOF
}

COMMAND="${1:-all}"
shift || true

case "$COMMAND" in
    all|research|development|review|testing)
        exec "$PYTHON" "$SCRIPT_DIR/swarm_orchestrator.py" --swarm "$COMMAND" "$@"
        ;;
    status)
        RUN_ID="${1:-}"
        if [[ -z "$RUN_ID" ]]; then
            echo "status requires a run id" >&2
            usage
            exit 1
        fi
        exec "$PYTHON" "$SCRIPT_DIR/swarm_orchestrator.py" --status "$RUN_ID"
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo "Unknown command: $COMMAND" >&2
        usage
        exit 1
        ;;
esac
