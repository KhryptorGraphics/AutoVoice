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
echo "                    AutoVoice Test Suite"
echo "================================================================================================"
echo ""

# Detect Docker Compose command
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker compose"
elif docker-compose --version &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker-compose"
else
    DOCKER_COMPOSE_CMD="docker compose"  # Default fallback
fi
print_msg "Using Docker Compose: $DOCKER_COMPOSE_CMD"

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
    # Non-interactive behavior for CI and environment-driven control
    if [ -n "$CI" ]; then
        print_error "CI environment detected. CUDA extension is required."
        exit 1
    elif [ "${ALLOW_NO_CUDA:-0}" = "1" ]; then
        print_warning "ALLOW_NO_CUDA=1 set. Continuing without CUDA extension. Some tests may be skipped."
    else
        print_warning "Continuing without CUDA extension. Some tests may be skipped."
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
        pytest tests/test_bindings_integration.py -v --tb=short
        ;;

    performance|p)
        print_msg "Running performance benchmarks..."
        pytest tests/test_performance.py -v -m performance -s
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
        pytest tests/ -v --cov=src/cuda_kernels --cov=src/auto_voice --cov-report=html --cov-report=term --cov-report=json
        print_success "Coverage report generated in htmlcov/"
        print_msg "Generating coverage analysis report..."
        if [ -f "scripts/analyze_coverage.py" ]; then
            # Run analysis but don't fail on exit code (analysis may exit non-zero if coverage < 80%)
            python ./scripts/analyze_coverage.py || true
            print_success "Coverage analysis report generated in docs/coverage_analysis_report.md"
            if [ -f "logs/coverage_gaps.json" ]; then
                print_msg "Coverage gaps analysis: logs/coverage_gaps.json"
            fi
        else
            print_warning "scripts/analyze_coverage.py not found, skipping analysis"
        fi
        ;;

    phase2|p2)
        print_msg "Running Phase 2 execution..."
        ./scripts/phase2_execute.sh
        ;;

    phase3|p3)
        print_msg "Running Phase 3 execution (Docker validation and E2E tests)..."
        ./scripts/phase3_execute.sh
        ;;

    docker-build|db)
        print_msg "Building Docker image..."
        docker build -t autovoice:latest .
        ;;

    docker-test|dt)
        print_msg "Testing Docker deployment..."
        $DOCKER_COMPOSE_CMD up -d
        sleep 10
        curl -f http://localhost:5000/health && print_success "Docker deployment working" || print_error "Docker deployment failed"
        $DOCKER_COMPOSE_CMD down
        ;;

    api-validate|av)
        print_msg "Validating API endpoints..."
        ./scripts/validate_api_endpoints.sh
        ;;

    websocket-test|wt)
        print_msg "Testing WebSocket connection..."
        python scripts/test_websocket_connection.py
        ;;

    validate|v)
        print_msg "Running quick test validation..."
        ./scripts/quick_test_check.sh
        ;;

    rerun|r)
        print_msg "Rerunning failed tests..."
        ./scripts/rerun_failed_tests.sh
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
        echo "Unit & Integration Tests:"
        echo "  smoke, s         Run smoke tests only (< 30s)"
        echo "  integration, i   Run integration tests (1-5 min)"
        echo "  performance, p   Run performance benchmarks (2-10 min)"
        echo "  fast, f          Run fast tests, exclude slow (3-5 min)"
        echo "  all, a           Run complete test suite (10-15 min)"
        echo "  coverage, c      Run with coverage report and analysis (generates JSON gaps)"
        echo ""
        echo "Phase Execution:"
        echo "  phase2, p2       Run Phase 2 execution (complete test suite)"
        echo "  phase3, p3       Run Phase 3 execution (Docker + E2E tests, 2-4 hours)"
        echo ""
        echo "Docker & Deployment:"
        echo "  docker-build, db Build Docker image"
        echo "  docker-test, dt  Test Docker Compose deployment"
        echo "  api-validate, av Validate all API endpoints"
        echo "  websocket-test, wt Test WebSocket/Socket.IO connection"
        echo ""
        echo "Utilities:"
        echo "  validate, v      Run quick test validation (pre-flight check)"
        echo "  rerun, r         Rerun failed tests from last run"
        echo "  markers, m       Show available test markers"
        echo "  list, l          List all available tests"
        echo "  help, h          Show this help message"
        echo ""
        echo "Recommended Workflow:"
        echo "  1. $0 validate   # Quick validation (< 1 min)"
        echo "  2. $0 phase2     # Full Phase 2 execution (10-15 min)"
        echo "  3. $0 phase3     # Docker validation and E2E tests (2-4 hours)"
        echo ""
        echo "Examples:"
        echo "  $0 smoke         # Quick validation"
        echo "  $0 fast          # Standard pre-commit check"
        echo "  $0 coverage      # Full suite with coverage and analysis"
        echo "  $0 phase3        # Complete Docker and API validation"
        echo "  $0 api-validate  # Test all REST API endpoints"
        echo ""
        echo "Environment Variables:"
        echo "  CI=1             # Fail if CUDA extension missing (for CI)"
        echo "  ALLOW_NO_CUDA=1  # Continue without CUDA extension"
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
