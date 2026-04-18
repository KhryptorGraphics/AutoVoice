#!/bin/bash
# Validate the canonical CUDA/TensorRT/runtime stack and capture latency artifacts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
autovoice_activate_env

DRY_RUN=0
PIPELINE="all"
OUTPUT_DIR="${AUTOVOICE_PROJECT_ROOT}/reports/platform"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --pipeline)
            PIPELINE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

run_cmd() {
    echo "+ $*"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        "$@"
    fi
}

AUDIT_PATH="$OUTPUT_DIR/dependency-audit.json"
LATENCY_PATH="$OUTPUT_DIR/${PIPELINE}-latency-report.md"

echo "=== AutoVoice CUDA Stack Validation ==="
echo "Project: $AUTOVOICE_PROJECT_ROOT"
echo "Python: $PYTHON"
echo "Pipeline: $PIPELINE"

run_cmd "$PYTHON" "$SCRIPT_DIR/verify_dependencies.py" \
    --json \
    --require-env \
    --require-tensorrt \
    --output "$AUDIT_PATH"

run_cmd "$PYTHON" "$SCRIPT_DIR/performance_validation.py" \
    --pipeline "$PIPELINE" \
    --duration 10 \
    --warmup 1 \
    --runs 1 \
    --quiet \
    --output "$LATENCY_PATH"

echo "Validation artifacts:"
echo "  - $AUDIT_PATH"
echo "  - $LATENCY_PATH"
