#!/bin/bash
# AutoVoice CUDA Bindings Test Runner
# Quick script to run various test configurations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_msg() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Banner
echo "================================================================================================"
echo "                    AutoVoice CUDA Bindings Test Suite"
echo "================================================================================================"
echo ""

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    print_error "pytest not found. Install with: pip install pytest pytest-cov"
    exit 1
fi

# Check if CUDA extension is built
print_msg "Checking CUDA extension..."
if python -c "import cuda_kernels" 2>/dev/null || python -c "from auto_voice import cuda_kernels" 2>/dev/null; then
    print_success "CUDA extension found"
else
    print_warning "CUDA extension not found. Build with: pip install -e ."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Parse command line arguments
case "${1:-all}" in
    smoke|s)
        print_msg "Running smoke tests (fast validation)..."
        pytest tests/test_bindings_smoke.py -v
        ;;

    integration|i)
        print_msg "Running integration tests..."
        pytest tests/test_bindings_integration.py -v -m integration
        ;;

    performance|p)
        print_msg "Running performance benchmarks..."
        pytest tests/test_bindings_performance.py -v -m performance -s
        ;;

    fast|f)
        print_msg "Running fast tests (excluding slow)..."
        pytest tests/ -v -m "not slow"
        ;;

    all|a)
        print_msg "Running complete test suite..."
        pytest tests/ -v
        ;;

    coverage|c)
        print_msg "Running tests with coverage..."
        pytest tests/ -v --cov=src/cuda_kernels --cov-report=html --cov-report=term
        print_success "Coverage report generated in htmlcov/"
        ;;

    markers|m)
        print_msg "Available test markers:"
        pytest --markers | grep -A 1 "^@pytest.mark"
        ;;

    list|l)
        print_msg "Available tests:"
        pytest tests/ --collect-only -q
        ;;

    help|h|*)
        echo "Usage: $0 [option]"
        echo ""
        echo "Options:"
        echo "  smoke, s         Run smoke tests only (< 30s)"
        echo "  integration, i   Run integration tests (1-5 min)"
        echo "  performance, p   Run performance benchmarks (2-10 min)"
        echo "  fast, f          Run fast tests, exclude slow (3-5 min)"
        echo "  all, a           Run complete test suite (10-15 min)"
        echo "  coverage, c      Run with coverage report"
        echo "  markers, m       Show available test markers"
        echo "  list, l          List all available tests"
        echo "  help, h          Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 smoke         # Quick validation"
        echo "  $0 fast          # Standard pre-commit check"
        echo "  $0 coverage      # Full suite with coverage"
        echo ""
        echo "Advanced usage:"
        echo "  pytest tests/test_bindings_smoke.py::test_function_callable -v"
        echo "  pytest tests/ -k 'pitch_detection' -v"
        echo "  pytest tests/ -m cuda -v"
        exit 0
        ;;
esac

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    print_success "All tests passed!"
    echo ""
else
    echo ""
    print_error "Some tests failed. Check output above."
    echo ""
    exit 1
fi
