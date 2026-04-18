#!/bin/bash
# Convenience wrapper for pipeline latency profiling in the canonical env.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
autovoice_activate_env

PIPELINE="${1:-quality_seedvc}"
shift || true

echo "=== AutoVoice Inference Latency Profile ==="
echo "Python: $PYTHON"
echo "Pipeline: $PIPELINE"

"$PYTHON" "$SCRIPT_DIR/performance_validation.py" --pipeline "$PIPELINE" "$@"
