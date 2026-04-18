#!/bin/bash
# Build CUDA extensions and run tests
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

source "$SCRIPT_DIR/common_env.sh"
autovoice_activate_env

echo "=== AutoVoice Build & Test ==="
echo "Python: $PYTHON"

# Build CUDA extensions
echo "Building CUDA extensions..."
"$PYTHON" setup.py build_ext --inplace 2>&1 || echo "CUDA extension build skipped (fallback available)"

# Verify runtime dependencies first
echo "Verifying dependency stack..."
"$PYTHON" scripts/verify_dependencies.py --require-env --require-tensorrt

# Verify CUDA bindings
echo "Verifying bindings..."
"$PYTHON" scripts/verify_bindings.py

# Run tests
echo "Running tests..."
"$PYTHON" -m pytest tests/ -v --tb=short

echo "=== Build & Test Complete ==="
