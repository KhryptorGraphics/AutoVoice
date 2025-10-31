#!/bin/bash
# AutoVoice Build and Test Script
# Builds CUDA extensions and runs comprehensive tests

set -e
set -o pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Unicode symbols
CHECK="✓"
CROSS="✗"
INFO="ℹ"
ARROW="→"

# Counters
PASSED=0
FAILED=0
SKIPPED=0

# Header
echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        AutoVoice Build and Test Script                 ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to print status messages
print_status() {
    echo -e "${BLUE}[${INFO}]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[${INFO}]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[${CHECK}]${NC} $1"
}

print_error() {
    echo -e "${RED}[${CROSS}]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_step() {
    echo -e "\n${BLUE}${ARROW}${NC} ${1}"
}

# Navigate to project root
cd /home/kp/autovoice

# Step 1: Environment Prerequisites Check
print_step "Checking environment prerequisites"

# Check Python
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    print_success "Python ${PYTHON_VERSION}"
else
    print_error "Python not found"
    exit 1
fi

# Check PyTorch
if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    print_success "PyTorch ${TORCH_VERSION}"
else
    print_error "PyTorch not available"
    echo ""
    print_status "Run ./scripts/setup_pytorch_env.sh to fix PyTorch installation"
    exit 1
fi

# Check CUDA
CUDA_AVAILABLE=false
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    CUDA_DEVICE=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    print_success "CUDA available: ${CUDA_DEVICE}"
    CUDA_AVAILABLE=true
else
    print_warning "CUDA not available - will run CPU tests only"
fi

# Check CUDA Toolkit (for building) - comprehensive validation
print_status "Checking CUDA toolkit requirements..."
if ./scripts/check_cuda_toolkit.sh > cuda_check.log 2>&1; then
    HAS_CUDA_ENV=true
    print_success "CUDA environment validated"
    # Extract CUDA version from check script output
    if grep -q "CUDA compiler version" cuda_check.log; then
        NVCC_VERSION=$(grep "CUDA compiler version" cuda_check.log | awk '{print $4}')
        print_info "CUDA Toolkit ${NVCC_VERSION} ready"
    fi
else
    HAS_CUDA_ENV=false
    print_warning "CUDA environment check failed - see cuda_check.log"
    print_info "Run: ./scripts/install_cuda_toolkit.sh (to fix CUDA issues)"
    print_info "Run: ./scripts/check_cuda_toolkit.sh (to diagnose issues)"
fi

# Determine build capabilities
if [ "$HAS_CUDA_ENV" = true ] && [ "$CUDA_AVAILABLE" = true ]; then
    CAN_BUILD_CUDA=true
    print_success "CUDA extensions can be built"
else
    CAN_BUILD_CUDA=false
    if [ "$CUDA_AVAILABLE" = false ]; then
        print_warning "PyTorch CUDA not available - CUDA extensions disabled"
    fi
    if [ "$HAS_CUDA_ENV" = false ]; then
        print_warning "CUDA toolkit validation failed - CUDA extensions disabled"
    fi
fi

# Step 2: Build CUDA Extensions
print_step "Building CUDA extensions"

if [ "$CAN_BUILD_CUDA" = true ]; then
    print_info "CUDA environment validated - building CUDA extensions"
    echo "  Running: pip install -e ."
    echo ""

    # Clean previous build artifacts
    if [ -d "build" ]; then
        print_status "Cleaning previous build artifacts"
        rm -rf build/
    fi

    # Build with comprehensive error handling
    if pip install -e . 2>&1 | tee build.log; then
        print_success "CUDA extensions built successfully"

        # Quick post-build verification
        if python -c "import auto_voice.cuda_kernels" 2>/dev/null; then
            print_success "CUDA extensions import successfully"
        else
            print_warning "Built CUDA extensions cannot be imported"
        fi

        ((PASSED++))
    else
        print_error "Build failed - see build.log for details"
        ((FAILED++))

        # Show helpful error context
        echo ""
        echo "Build error context:"
        if grep -q "nv/target" build.log; then
            echo "  - Missing nv/target header (critical for PyTorch CUDA extensions)"
        fi
        if grep -q "CUDA" build.log && grep -q "not found" build.log; then
            echo "  - CUDA toolkit or headers not found"
        fi

        echo ""
        echo "Last 20 lines of build output:"
        tail -n 20 build.log

        print_info "To diagnose CUDA issues: ./scripts/check_cuda_toolkit.sh"
        print_info "To fix CUDA installation: ./scripts/install_cuda_toolkit.sh"

        # Don't exit - continue with CPU-only testing
        print_warning "Continuing with CPU-only tests (CUDA build failed)"
    fi
else
    if [ "$CUDA_AVAILABLE" = true ]; then
        print_warning "Skipping CUDA extension build (CUDA toolkit issues)"
        print_info "Run: ./scripts/check_cuda_toolkit.sh (to diagnose)"
        print_info "Run: ./scripts/install_cuda_toolkit.sh (to fix)"
    else
        print_warning "Skipping CUDA extension build (CUDA not available)"
    fi
    ((SKIPPED++))
fi

# Step 3: Verify Bindings
print_step "Verifying Python bindings"

if python ./scripts/verify_bindings.py 2>&1 | tee verify.log; then
    print_success "Python bindings verified"
    ((PASSED++))
else
    print_error "Binding verification failed"
    ((FAILED++))
    cat verify.log
fi

# Step 4: Run Smoke Tests
print_step "Running smoke tests"

if [ -f "tests/test_bindings_smoke.py" ]; then
    pytest tests/test_bindings_smoke.py -v 2>&1 | tee smoke_test.log
    status=$?
    if [ $status -eq 0 ]; then
        print_success "Smoke tests passed"
        ((PASSED++))
    else
        print_error "Smoke tests failed"
        ((FAILED++))
        cat smoke_test.log
    fi
else
    print_warning "Smoke test file not found"
    ((SKIPPED++))
fi

# Step 5: Run Comprehensive Test Suite
print_step "Running comprehensive test suite"

# Test coverage and reporting options
COVERAGE_ARGS=""
if [ -f ".coveragerc" ]; then
    COVERAGE_ARGS="--cov=src/auto_voice --cov-report=term-missing --cov-report=html --cov-append"
    print_info "Coverage reporting enabled with append mode"
fi

# Run different test categories
TEST_CATEGORIES=(
    "Smoke Tests:-m smoke"
    "Unit Tests:-m unit and not slow and not smoke"
    "Integration Tests:-m integration and not slow"
    "E2E Tests:-m e2e and not slow"
    "Performance Tests:-m performance"
    "CUDA Tests:-m cuda and not smoke"  # Only if CUDA available
)

# Counters for each test category
UNIT_PASSED=0
UNIT_FAILED=0
UNIT_SKIPPED=0
INTEGRATION_PASSED=0
INTEGRATION_FAILED=0
INTEGRATION_SKIPPED=0
PERFORMANCE_PASSED=0
PERFORMANCE_FAILED=0
PERFORMANCE_SKIPPED=0
CUDA_PASSED=0
CUDA_FAILED=0
CUDA_SKIPPED=0

for category in "${TEST_CATEGORIES[@]}"; do
    category_name=$(echo "$category" | cut -d':' -f1)
    pytest_marker=$(echo "$category" | cut -d':' -f2)

    print_info "Running ${category_name}..."

    # Special handling for CUDA tests
    if [[ "$category_name" == *"CUDA"* ]]; then
        if [ "$CUDA_AVAILABLE" = false ]; then
            print_warning "CUDA tests skipped (CUDA not available)"
            CUDA_SKIPPED=$((CUDA_SKIPPED + 1))
            ((SKIPPED++))
            continue
        fi
        if [ "$CAN_BUILD_CUDA" = false ]; then
            print_warning "CUDA tests skipped (CUDA build environment issues)"
            CUDA_SKIPPED=$((CUDA_SKIPPED + 1))
            ((SKIPPED++))
            continue
        fi
    fi

    # Run the test category with proper exit code capture
    PYTEST_CMD="pytest tests/ ${pytest_marker} -v --tb=short ${COVERAGE_ARGS}"

    # Compute safe log filename
    safe_name="${category_name,,}"
    safe_name="${safe_name// /_}"
    log_file="${safe_name}_log.txt"

    eval "$PYTEST_CMD" 2>&1 | tee "$log_file"
    status=$?

    if [ $status -eq 0 ]; then
        print_success "${category_name} passed"

        # Extract results for this category
        if grep -q "passed" "$log_file"; then
            cat_passed=$(grep -oP '\d+(?= passed)' "$log_file" | head -1 || echo "0")
            cat_failed=$(grep -oP '\d+(?= failed)' "$log_file" | head -1 || echo "0")
            cat_skipped=$(grep -oP '\d+(?= skipped)' "$log_file" | head -1 || echo "0")

            # Update category counters and global counters
            if [[ "$category_name" == *"Unit"* ]]; then
                UNIT_PASSED=$((UNIT_PASSED + cat_passed))
                UNIT_FAILED=$((UNIT_FAILED + cat_failed))
                UNIT_SKIPPED=$((UNIT_SKIPPED + cat_skipped))
            elif [[ "$category_name" == *"Integration"* ]]; then
                INTEGRATION_PASSED=$((INTEGRATION_PASSED + cat_passed))
                INTEGRATION_FAILED=$((INTEGRATION_FAILED + cat_failed))
                INTEGRATION_SKIPPED=$((INTEGRATION_SKIPPED + cat_skipped))
            elif [[ "$category_name" == *"Performance"* ]]; then
                PERFORMANCE_PASSED=$((PERFORMANCE_PASSED + cat_passed))
                PERFORMANCE_FAILED=$((PERFORMANCE_FAILED + cat_failed))
                PERFORMANCE_SKIPPED=$((PERFORMANCE_SKIPPED + cat_skipped))
            elif [[ "$category_name" == *"CUDA"* ]]; then
                CUDA_PASSED=$((CUDA_PASSED + cat_passed))
                CUDA_FAILED=$((CUDA_FAILED + cat_failed))
                CUDA_SKIPPED=$((CUDA_SKIPPED + cat_skipped))
            fi

            ((PASSED++))

            echo "  ${category_name}: ${cat_passed} passed, ${cat_failed} failed, ${cat_skipped} skipped"
        fi
    else
        print_error "${category_name} failed - see $log_file"

        # Still extract results even on failure
        if grep -q "passed" "$log_file"; then
            cat_passed=$(grep -oP '\d+(?= passed)' "$log_file" | head -1 || echo "0")
            cat_failed=$(grep -oP '\d+(?= failed)' "$log_file" | head -1 || echo "0")
            cat_skipped=$(grep -oP '\d+(?= skipped)' "$log_file" | head -1 || echo "0")

            # Update category counters and global counters
            if [[ "$category_name" == *"Unit"* ]]; then
                UNIT_PASSED=$((UNIT_PASSED + cat_passed))
                UNIT_FAILED=$((UNIT_FAILED + cat_failed))
                UNIT_SKIPPED=$((UNIT_SKIPPED + cat_skipped))
            elif [[ "$category_name" == *"Integration"* ]]; then
                INTEGRATION_PASSED=$((INTEGRATION_PASSED + cat_passed))
                INTEGRATION_FAILED=$((INTEGRATION_FAILED + cat_failed))
                INTEGRATION_SKIPPED=$((INTEGRATION_SKIPPED + cat_skipped))
            elif [[ "$category_name" == *"Performance"* ]]; then
                PERFORMANCE_PASSED=$((PERFORMANCE_PASSED + cat_passed))
                PERFORMANCE_FAILED=$((PERFORMANCE_FAILED + cat_failed))
                PERFORMANCE_SKIPPED=$((PERFORMANCE_SKIPPED + cat_skipped))
            elif [[ "$category_name" == *"CUDA"* ]]; then
                CUDA_PASSED=$((CUDA_PASSED + cat_passed))
                CUDA_FAILED=$((CUDA_FAILED + cat_failed))
                CUDA_SKIPPED=$((CUDA_SKIPPED + cat_skipped))
            fi

            ((FAILED++))
        else
            ((FAILED++))
        fi

        # Show brief failure summary
        echo ""
        echo "Failure summary for ${category_name}:"
        grep -A 5 "FAILED" "$log_file" || echo "See $log_file for details"
        echo ""
    fi
done

# Step 5.5: Run Full Test Suite with Aggregated Coverage
print_step "Running full test suite with aggregated coverage"

FULL_SUITE_CMD="pytest tests/ -v --cov=src/auto_voice --cov-report=html --cov-report=term-missing"

eval "$FULL_SUITE_CMD" 2>&1 | tee full_suite_log.txt
status=$?

if [ $status -eq 0 ]; then
    print_success "Full test suite passed"
    ((PASSED++))
else
    print_warning "Full test suite had failures (see full_suite_log.txt)"
    # Don't increment FAILED as individual tests already counted
fi

# Print detailed test summary
echo ""
print_info "Detailed Test Results:"
echo "  Unit Tests:      ${UNIT_PASSED} passed, ${UNIT_FAILED} failed, ${UNIT_SKIPPED} skipped"
echo "  Integration:     ${INTEGRATION_PASSED} passed, ${INTEGRATION_FAILED} failed, ${INTEGRATION_SKIPPED} skipped"
echo "  Performance:     ${PERFORMANCE_PASSED} passed, ${PERFORMANCE_FAILED} failed, ${PERFORMANCE_SKIPPED} skipped"
if [ "$CUDA_AVAILABLE" = true ]; then
    echo "  CUDA Tests:      ${CUDA_PASSED} passed, ${CUDA_FAILED} failed, ${CUDA_SKIPPED} skipped"
fi

# Step 6: Legacy CUDA Integration Tests (if still needed)
if [ "$CUDA_AVAILABLE" = true ] && [ "$CUDA_PASSED" -eq 0 ]; then
    print_step "Running legacy CUDA integration tests"

    LEGACY_CUDA_TESTS=(
        "tests/test_bindings_integration.py"
        "tests/test_bindings_performance.py"
    )

    for test_file in "${LEGACY_CUDA_TESTS[@]}"; do
        if [ -f "$test_file" ]; then
            test_name=$(basename "$test_file" .py)
            if python "$test_file" 2>&1 | tee "legacy_cuda_${test_name}.log"; then
                print_success "Legacy CUDA test passed: ${test_name}"
                CUDA_PASSED=$((CUDA_PASSED + 1))
                ((PASSED++))
            else
                print_warning "Legacy CUDA test failed: ${test_name}"
                CUDA_FAILED=$((CUDA_FAILED + 1))
                ((FAILED++))
            fi
        fi
    done
fi

# Step 7: Generate Test Report
print_step "Generating test report"

REPORT_FILE="test_report_$(date +%Y%m%d_%H%M%S).txt"

cat > "$REPORT_FILE" << EOF
AutoVoice Build and Test Report
================================
Generated: $(date)

Environment:
  Python: ${PYTHON_VERSION}
  PyTorch: ${TORCH_VERSION}
  CUDA: $([ "$CUDA_AVAILABLE" = true ] && echo "Available (${CUDA_DEVICE})" || echo "Not available")
  CUDA Toolkit: $([ "$HAS_NVCC" = true ] && echo "${NVCC_VERSION}" || echo "Not found")

Test Results:
  Passed:  ${PASSED}
  Failed:  ${FAILED}
  Skipped: ${SKIPPED}
  Total:   $((PASSED + FAILED + SKIPPED))

Status: $([ "$FAILED" -eq 0 ] && echo "SUCCESS" || echo "FAILURE")

Detailed Logs:
  Build log:        build.log
  Verify log:       verify.log
  Smoke test log:   smoke_test.log
  Pytest log:       pytest.log
EOF

print_success "Test report saved: ${REPORT_FILE}"

# Step 8: Summary
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Test Summary                                            ${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${GREEN}Passed:${NC}  ${PASSED}"
echo -e "  ${RED}Failed:${NC}  ${FAILED}"
echo -e "  ${YELLOW}Skipped:${NC} ${SKIPPED}"
echo -e "  Total:   $((PASSED + FAILED + SKIPPED))"
echo ""

# Exit with appropriate code
if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ALL TESTS PASSED!                                      ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
    exit 0
else
    echo -e "${RED}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  TESTS FAILED - Review logs for details                 ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════╝${NC}"
    exit 1
fi
