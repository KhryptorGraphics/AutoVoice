#!/bin/bash
# Build ARM64-sensitive source dependencies inside the canonical AutoVoice env.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
autovoice_activate_env

echo "=== AutoVoice Source Dependency Build ==="
echo "Python: $PYTHON"

"$PYTHON" -m pip install --no-user --upgrade pip setuptools wheel
"$PYTHON" -m pip install --no-user --no-binary=:all: pyworld pesq
"$PYTHON" -m pip install --no-user flask-swagger-ui pystoi local-attention

echo "=== Verifying source dependencies ==="
"$PYTHON" "$SCRIPT_DIR/verify_dependencies.py" --require-env
