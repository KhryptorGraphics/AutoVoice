#!/bin/bash
# Build CUDA extensions and run tests
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=== AutoVoice Build & Test ==="

# Activate environment
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate autovoice-thor 2>/dev/null || true
fi

export PYTHONNOUSERSITE=1
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="11.0"

# Build CUDA extensions
echo "Building CUDA extensions..."
python setup.py build_ext --inplace 2>&1 || echo "CUDA extension build skipped (fallback available)"

# Verify CUDA bindings
echo "Verifying bindings..."
python scripts/verify_bindings.py

# Run tests
echo "Running tests..."
python -m pytest tests/ -v --tb=short

echo "=== Build & Test Complete ==="
