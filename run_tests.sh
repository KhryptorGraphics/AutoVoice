#!/bin/bash
# AutoVoice test runner
# Usage: ./run_tests.sh [mode]
# Modes: smoke, fast, all, coverage, coverage_remaining, cuda

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda env if available
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate autovoice-thor 2>/dev/null || true
fi

export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"
export PYTHONNOUSERSITE=1

MODE="${1:-fast}"

case "$MODE" in
    smoke)
        echo "Running smoke tests..."
        python -m pytest tests/ -m smoke -v --tb=short
        ;;
    fast)
        echo "Running fast tests (excluding slow)..."
        python -m pytest tests/ -m "not slow and not very_slow" -v --tb=short
        ;;
    all)
        echo "Running all tests..."
        python -m pytest tests/ -v --tb=short
        ;;
    coverage)
        echo "Running tests with coverage..."
        python -m coverage erase
        python -m coverage run -m pytest -p no:cov tests/ -v
        python -m coverage report -m
        python -m coverage html
        echo "Coverage report: htmlcov/index.html"
        ;;
    coverage_remaining)
        echo "Running remaining-module coverage gate..."
        python scripts/run_remaining_module_coverage.py
        ;;
    cuda)
        echo "Running CUDA tests..."
        python -m pytest tests/ -m cuda -v --tb=short
        ;;
    *)
        echo "Usage: $0 {smoke|fast|all|coverage|coverage_remaining|cuda}"
        exit 1
        ;;
esac
