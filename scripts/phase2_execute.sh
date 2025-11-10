#!/bin/bash
# Phase 2 Execution Script: Execute Core Test Suite and Validate Functionality
# This script orchestrates the complete Phase 2 test execution process

set -e
set -o pipefail
set -u

# Set project root and change to it
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Define timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Ensure logs directory exists
mkdir -p logs

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Python-based float comparison (replaces bc dependency)
py_compare() {
    python3 - "$1" "$2" "$3" <<'PY'
import sys
a, op, b = float(sys.argv[1]), sys.argv[2], float(sys.argv[3])
if op == ">=": print(int(a >= b))
elif op == ">": print(int(a > b))
elif op == "<=": print(int(a <= b))
elif op == "<": print(int(a < b))
elif op == "==": print(int(a == b))
else: print(0)
PY
}

# Python-based float arithmetic (replaces bc dependency)
py_calc() {
    python3 - "$@" <<'PY'
import sys
print(eval(sys.argv[1]))
PY
}

# Initialize test result counters (before any parsing)
SMOKE_PASSED=0
SMOKE_FAILED=0
SMOKE_SKIPPED=0
INTEGRATION_PASSED=0
INTEGRATION_FAILED=0
INTEGRATION_SKIPPED=0
PERFORMANCE_PASSED=0
PERFORMANCE_FAILED=0
PERFORMANCE_SKIPPED=0
AUDIO_PASSED=0
AUDIO_FAILED=0
AUDIO_SKIPPED=0
MODEL_PASSED=0
MODEL_FAILED=0
MODEL_SKIPPED=0
INFERENCE_PASSED=0
INFERENCE_FAILED=0
INFERENCE_SKIPPED=0
FULL_PASSED=0
FULL_FAILED=0
FULL_SKIPPED=0
COVERAGE_PERCENT=0
SMOKE_DURATION=0
INTEGRATION_DURATION=0
PERFORMANCE_DURATION=0
AUDIO_DURATION=0
MODEL_DURATION=0
INFERENCE_DURATION=0
FULL_DURATION=0

# Pre-flight validation
print_header "Phase 2: Execute Core Test Suite and Validate Functionality"
echo "Started at: $(date)"
echo "Timestamp: $TIMESTAMP"
echo "Project root: $PROJECT_ROOT"
echo ""

if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
    print_warning "No conda environment active. Recommended: autovoice_py312"
else
    if [[ "$CONDA_DEFAULT_ENV" != "autovoice_py312" ]]; then
        print_warning "Conda environment: $CONDA_DEFAULT_ENV (recommended: autovoice_py312)"
    else
        print_success "Conda environment: $CONDA_DEFAULT_ENV"
    fi
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
if [[ "$PYTHON_VERSION" != 3.12.* ]]; then
    print_warning "Python version: $PYTHON_VERSION (recommended: 3.12.x)"
else
    print_success "Python version: $PYTHON_VERSION"
fi

# Check PyTorch installation
if python -c "import torch; print('PyTorch version:', torch.__version__)" 2>/dev/null; then
    PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    print_success "PyTorch installed: $PYTORCH_VERSION"
else
    print_error "PyTorch not found. Please run Phase 1 setup."
    exit 1
fi

# Check CUDA extensions
if python -c "import cuda_kernels" 2>/dev/null || python -c "from auto_voice import cuda_kernels" 2>/dev/null; then
    print_success "CUDA extensions available"
else
    print_warning "CUDA extensions not found. Some tests may be skipped."
fi

# Check pytest availability
if command -v pytest >/dev/null 2>&1; then
    print_success "pytest available"
else
    print_error "pytest not found. Please install pytest."
    exit 1
fi

# Check CUDA availability
if python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
    CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
    if [[ "$CUDA_AVAILABLE" == "True" ]]; then
        GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown GPU")
        print_success "CUDA available: $GPU_NAME"
    else
        print_warning "CUDA not available. GPU tests will be skipped."
    fi
else
    print_warning "Cannot check CUDA availability."
fi

echo ""

# Step 1: Run Smoke Tests
print_header "Step 1/6: Running Smoke Tests (7 tests, <30s)"
echo "Command: pytest tests/test_bindings_smoke.py -v --tb=short"
echo "Started at: $(date)"
echo ""

SMOKE_START=$(date +%s)
set +e
pytest tests/test_bindings_smoke.py -v --tb=short 2>&1 | tee logs/smoke_tests_${TIMESTAMP}.log
SMOKE_EXIT=${PIPESTATUS[0]}
set -e
SMOKE_END=$(date +%s)
SMOKE_DURATION=$((SMOKE_END - SMOKE_START))

echo ""
echo "Smoke tests completed in ${SMOKE_DURATION}s"

# Parse smoke test results
if [[ -f logs/smoke_tests_${TIMESTAMP}.log ]]; then
    SMOKE_RESULTS=$(grep -E "passed|failed|skipped" logs/smoke_tests_${TIMESTAMP}.log | tail -1)
    if [[ -n "$SMOKE_RESULTS" ]]; then
        SMOKE_PASSED=$(echo "$SMOKE_RESULTS" | grep -o "[0-9]\+ passed" | awk '{print $1}' || echo "0")
        SMOKE_FAILED=$(echo "$SMOKE_RESULTS" | grep -o "[0-9]\+ failed" | awk '{print $1}' || echo "0")
        SMOKE_SKIPPED=$(echo "$SMOKE_RESULTS" | grep -o "[0-9]\+ skipped" | awk '{print $1}' || echo "0")

        if [[ "$SMOKE_FAILED" -eq 0 ]]; then
            print_success "Smoke tests: ${SMOKE_PASSED} passed, ${SMOKE_FAILED} failed, ${SMOKE_SKIPPED} skipped"
        else
            print_error "Smoke tests: ${SMOKE_PASSED} passed, ${SMOKE_FAILED} failed, ${SMOKE_SKIPPED} skipped"
            print_warning "Critical failures detected. Consider fixing before continuing."
        fi
    else
        SMOKE_PASSED=0
        SMOKE_FAILED=0
        SMOKE_SKIPPED=0
        SMOKE_DURATION=0
        print_warning "Could not parse smoke test results"
    fi
else
    SMOKE_PASSED=0
    SMOKE_FAILED=0
    SMOKE_SKIPPED=0
    SMOKE_DURATION=0
    print_error "Smoke test log not found"
fi

echo ""

# Step 2: Run Integration Tests
print_header "Step 2/6: Running Integration Tests (9 tests, 1-5min)"
echo "Command: pytest tests/test_bindings_integration.py -v --tb=short"
echo "Started at: $(date)"
echo ""

INTEGRATION_START=$(date +%s)
set +e
pytest tests/test_bindings_integration.py -v --tb=short 2>&1 | tee logs/integration_tests_${TIMESTAMP}.log
INTEGRATION_EXIT=${PIPESTATUS[0]}
set -e
INTEGRATION_END=$(date +%s)
INTEGRATION_DURATION=$((INTEGRATION_END - INTEGRATION_START))

echo ""
echo "Integration tests completed in ${INTEGRATION_DURATION}s"

# Parse integration test results
if [[ -f logs/integration_tests_${TIMESTAMP}.log ]]; then
    INTEGRATION_RESULTS=$(grep -E "passed|failed|skipped" logs/integration_tests_${TIMESTAMP}.log | tail -1)
    if [[ -n "$INTEGRATION_RESULTS" ]]; then
        INTEGRATION_PASSED=$(echo "$INTEGRATION_RESULTS" | grep -o "[0-9]\+ passed" | awk '{print $1}' || echo "0")
        INTEGRATION_FAILED=$(echo "$INTEGRATION_RESULTS" | grep -o "[0-9]\+ failed" | awk '{print $1}' || echo "0")
        INTEGRATION_SKIPPED=$(echo "$INTEGRATION_RESULTS" | grep -o "[0-9]\+ skipped" | awk '{print $1}' || echo "0")

        if [[ "$INTEGRATION_FAILED" -eq 0 ]]; then
            print_success "Integration tests: ${INTEGRATION_PASSED} passed, ${INTEGRATION_FAILED} failed, ${INTEGRATION_SKIPPED} skipped"
        else
            print_warning "Integration tests: ${INTEGRATION_PASSED} passed, ${INTEGRATION_FAILED} failed, ${INTEGRATION_SKIPPED} skipped"
        fi
    else
        INTEGRATION_PASSED=0
        INTEGRATION_FAILED=0
        INTEGRATION_SKIPPED=0
        INTEGRATION_DURATION=0
        print_warning "Could not parse integration test results"
    fi
else
        INTEGRATION_PASSED=0
        INTEGRATION_FAILED=0
        INTEGRATION_SKIPPED=0
        INTEGRATION_DURATION=0
    print_error "Integration test log not found"
fi

echo ""

# Step 3: Run Core Component Tests
print_header "Step 3/6: Running Core Component Tests"
echo "Started at: $(date)"
echo ""

# Audio Processor Tests
echo "Running Audio Processor tests..."
AUDIO_START=$(date +%s)
set +e
pytest tests/test_audio_processor.py -v --tb=short 2>&1 | tee logs/audio_processor_${TIMESTAMP}.log
AUDIO_EXIT=${PIPESTATUS[0]}
set -e
AUDIO_END=$(date +%s)
AUDIO_DURATION=$((AUDIO_END - AUDIO_START))

if [[ -f logs/audio_processor_${TIMESTAMP}.log ]]; then
    AUDIO_RESULTS=$(grep -E "passed|failed|skipped" logs/audio_processor_${TIMESTAMP}.log | tail -1)
    if [[ -n "$AUDIO_RESULTS" ]]; then
        AUDIO_PASSED=$(echo "$AUDIO_RESULTS" | grep -o "[0-9]\+ passed" | awk '{print $1}' || echo "0")
        AUDIO_FAILED=$(echo "$AUDIO_RESULTS" | grep -o "[0-9]\+ failed" | awk '{print $1}' || echo "0")
        AUDIO_SKIPPED=$(echo "$AUDIO_RESULTS" | grep -o "[0-9]\+ skipped" | awk '{print $1}' || echo "0")
        print_info "Audio Processor: ${AUDIO_PASSED} passed, ${AUDIO_FAILED} failed, ${AUDIO_SKIPPED} skipped (${AUDIO_DURATION}s)"
    fi
fi

# Model Tests
echo "Running Model tests..."
MODEL_START=$(date +%s)
set +e
pytest tests/test_models.py -v --tb=short 2>&1 | tee logs/models_${TIMESTAMP}.log
MODEL_EXIT=${PIPESTATUS[0]}
set -e
MODEL_END=$(date +%s)
MODEL_DURATION=$((MODEL_END - MODEL_START))

if [[ -f logs/models_${TIMESTAMP}.log ]]; then
    MODEL_RESULTS=$(grep -E "passed|failed|skipped" logs/models_${TIMESTAMP}.log | tail -1)
    if [[ -n "$MODEL_RESULTS" ]]; then
        MODEL_PASSED=$(echo "$MODEL_RESULTS" | grep -o "[0-9]\+ passed" | awk '{print $1}' || echo "0")
        MODEL_FAILED=$(echo "$MODEL_RESULTS" | grep -o "[0-9]\+ failed" | awk '{print $1}' || echo "0")
        MODEL_SKIPPED=$(echo "$MODEL_RESULTS" | grep -o "[0-9]\+ skipped" | awk '{print $1}' || echo "0")
        print_info "Models: ${MODEL_PASSED} passed, ${MODEL_FAILED} failed, ${MODEL_SKIPPED} skipped (${MODEL_DURATION}s)"
    fi
fi

# Inference Tests
echo "Running Inference tests..."
INFERENCE_START=$(date +%s)
set +e
pytest tests/test_inference.py -v --tb=short 2>&1 | tee logs/inference_${TIMESTAMP}.log
INFERENCE_EXIT=${PIPESTATUS[0]}
set -e
INFERENCE_END=$(date +%s)
INFERENCE_DURATION=$((INFERENCE_END - INFERENCE_START))

if [[ -f logs/inference_${TIMESTAMP}.log ]]; then
    INFERENCE_RESULTS=$(grep -E "passed|failed|skipped" logs/inference_${TIMESTAMP}.log | tail -1)
    if [[ -n "$INFERENCE_RESULTS" ]]; then
        INFERENCE_PASSED=$(echo "$INFERENCE_RESULTS" | grep -o "[0-9]\+ passed" | awk '{print $1}' || echo "0")
        INFERENCE_FAILED=$(echo "$INFERENCE_RESULTS" | grep -o "[0-9]\+ failed" | awk '{print $1}' || echo "0")
        INFERENCE_SKIPPED=$(echo "$INFERENCE_RESULTS" | grep -o "[0-9]\+ skipped" | awk '{print $1}' || echo "0")
        print_info "Inference: ${INFERENCE_PASSED} passed, ${INFERENCE_FAILED} failed, ${INFERENCE_SKIPPED} skipped (${INFERENCE_DURATION}s)"
    fi
fi

echo ""

# Step 4: Run Full Test Suite with Coverage
print_header "Step 4/6: Running Full Test Suite with Coverage (151+ tests)"
echo "Command: pytest tests/ -v --cov=src/auto_voice --cov-report=html --cov-report=term-missing --cov-report=json --tb=short --durations=10"
echo "Started at: $(date)"
echo ""

FULL_START=$(date +%s)
set +e
pytest tests/ -v --cov=src/auto_voice --cov-report=html --cov-report=term-missing --cov-report=json --tb=short --durations=10 2>&1 | tee logs/full_suite_${TIMESTAMP}.log
FULL_EXIT=${PIPESTATUS[0]}
set -e
FULL_END=$(date +%s)
FULL_DURATION=$((FULL_END - FULL_START))

echo ""
echo "Full test suite completed in ${FULL_DURATION}s"

# Parse full suite results
if [[ -f logs/full_suite_${TIMESTAMP}.log ]]; then
    FULL_RESULTS=$(grep -E "passed|failed|skipped" logs/full_suite_${TIMESTAMP}.log | tail -1)
    if [[ -n "$FULL_RESULTS" ]]; then
        FULL_PASSED=$(echo "$FULL_RESULTS" | grep -o "[0-9]\+ passed" | awk '{print $1}' || echo "0")
        FULL_FAILED=$(echo "$FULL_RESULTS" | grep -o "[0-9]\+ failed" | awk '{print $1}' || echo "0")
        FULL_SKIPPED=$(echo "$FULL_RESULTS" | grep -o "[0-9]\+ skipped" | awk '{print $1}' || echo "0")
        FULL_TOTAL=$((FULL_PASSED + FULL_FAILED + FULL_SKIPPED))

        # Parse coverage percentage
        COVERAGE_LINE=$(grep "TOTAL" logs/full_suite_${TIMESTAMP}.log | tail -1)
        if [[ -n "$COVERAGE_LINE" ]]; then
            COVERAGE_PERCENT=$(echo "$COVERAGE_LINE" | awk '{print $NF}' | sed 's/%//')
        else
            COVERAGE_PERCENT="0"
        fi

        print_success "Full suite: ${FULL_TOTAL} tests (${FULL_PASSED} passed, ${FULL_FAILED} failed, ${FULL_SKIPPED} skipped)"
        print_success "Coverage: ${COVERAGE_PERCENT}% (target: 80%)"

        if [[ $(py_compare "$COVERAGE_PERCENT" ">=" "80") -eq 1 ]]; then
            print_success "Coverage target met! ✅"
        elif [ "$(python3 -c "print(float(${COVERAGE_PERCENT}) >= 70)")" = "True"  ]; then
            print_warning "Coverage close to target (${COVERAGE_PERCENT}%)"
        else
            print_error "Coverage below target (${COVERAGE_PERCENT}%)"
        fi
    else
        print_warning "Could not parse full suite results"
    fi
else
    print_error "Full suite log not found"
fi

echo ""

# Step 5: Analyze Results
print_header "Step 5/6: Analyzing Results"

# Calculate total statistics
TOTAL_PASSED=$((SMOKE_PASSED + INTEGRATION_PASSED + PERFORMANCE_PASSED + AUDIO_PASSED + MODEL_PASSED + INFERENCE_PASSED))
TOTAL_FAILED=$((SMOKE_FAILED + INTEGRATION_FAILED + PERFORMANCE_FAILED + AUDIO_FAILED + MODEL_FAILED + INFERENCE_FAILED))
TOTAL_SKIPPED=$((SMOKE_SKIPPED + INTEGRATION_SKIPPED + PERFORMANCE_SKIPPED + AUDIO_SKIPPED + MODEL_SKIPPED + INFERENCE_SKIPPED))
TOTAL_TESTS=$((TOTAL_PASSED + TOTAL_FAILED + TOTAL_SKIPPED))

if [[ $TOTAL_TESTS -gt 0 ]]; then
    PASS_RATE=$(( (TOTAL_PASSED * 100) / TOTAL_TESTS ))
else
    PASS_RATE=0
fi

print_info "Total tests executed: $TOTAL_TESTS"
print_info "Passed: $TOTAL_PASSED (${PASS_RATE}%)"
print_info "Failed: $TOTAL_FAILED"
print_info "Skipped: $TOTAL_SKIPPED"

# Identify critical failures
CRITICAL_FAILURES=0
if [[ $SMOKE_FAILED -gt 0 ]]; then
    CRITICAL_FAILURES=$((CRITICAL_FAILURES + SMOKE_FAILED))
    print_error "Critical failures in smoke tests: $SMOKE_FAILED"
fi

# Extract top 10 slowest tests
if [[ -f logs/full_suite_${TIMESTAMP}.log ]]; then
    echo ""
    print_info "Top 10 slowest tests:"
    grep -A 10 "slowest test durations" logs/full_suite_${TIMESTAMP}.log | tail -10 || echo "No duration data available"
fi

echo ""

# Step 6: Generate Reports
print_header "Step 6/6: Generating Reports"

echo "Generating Phase 2 completion report..."
./scripts/generate_phase2_report.sh "${TIMESTAMP}"

echo "Analyzing coverage gaps..."
python ./scripts/analyze_coverage.py

echo ""
print_success "Reports generated successfully!"
echo ""
echo "📊 Phase 2 Completion Report: PHASE2_COMPLETION_REPORT.md"
echo "📈 Coverage HTML Report: htmlcov/index.html"
echo "📋 Coverage Analysis: docs/coverage_analysis_report.md"

# Summary display
echo ""
print_header "Phase 2 Execution Summary"

if [[ $TOTAL_FAILED -eq 0 && $CRITICAL_FAILURES -eq 0 ]]; then
    if [[ $(py_compare "$COVERAGE_PERCENT" ">=" "80") -eq 1 ]]; then
        print_success "Phase 2 Complete! ✅"
        print_success "All tests passed, coverage target met."
        echo ""
        print_info "Next: Ready for Phase 3 (Docker validation)"
    else
        print_warning "Phase 2 Complete with Coverage Gap ⚠️"
        print_warning "All tests passed but coverage (${COVERAGE_PERCENT}%) below target (80%)"
        echo ""
        print_info "Next: Add tests for uncovered code (see docs/coverage_analysis_report.md)"
    fi
elif [[ $CRITICAL_FAILURES -gt 0 ]]; then
    print_error "Phase 2 Failed ❌"
    print_error "Critical failures detected: $CRITICAL_FAILURES"
    echo ""
    print_info "Next: Fix critical failures and re-run Phase 2"
    print_info "Command: ./scripts/rerun_failed_tests.sh"
else
    print_warning "Phase 2 Partially Complete ⚠️"
    print_warning "$TOTAL_FAILED tests failed, $CRITICAL_FAILURES critical"
    echo ""
    print_info "Next: Review failures and fix issues"
    print_info "Command: ./scripts/rerun_failed_tests.sh"
fi

echo ""
echo "Key Metrics:"
echo "  Total Tests: $TOTAL_TESTS"
echo "  Passed: $TOTAL_PASSED (${PASS_RATE}%)"
echo "  Failed: $TOTAL_FAILED"
echo "  Skipped: $TOTAL_SKIPPED"
echo "  Coverage: ${COVERAGE_PERCENT}% (target: 80%)"
# Calculate total duration across all steps
TOTAL_DURATION=$((SMOKE_DURATION + INTEGRATION_DURATION + PERFORMANCE_DURATION + AUDIO_DURATION + MODEL_DURATION + INFERENCE_DURATION + FULL_DURATION))
echo "  Duration: ${TOTAL_DURATION}s (Smoke: ${SMOKE_DURATION}s, Integration: ${INTEGRATION_DURATION}s, Performance: ${PERFORMANCE_DURATION}s, Audio: ${AUDIO_DURATION}s, Model: ${MODEL_DURATION}s, Inference: ${INFERENCE_DURATION}s, Full: ${FULL_DURATION}s)"

echo ""
echo "Completed at: $(date)"
echo "Log files saved in: logs/"
echo ""

# Exit with appropriate code
if [[ $CRITICAL_FAILURES -gt 0 ]]; then
    exit 1
elif [[ $TOTAL_FAILED -gt 0 ]]; then
    exit 2
else
    exit 0
fi
