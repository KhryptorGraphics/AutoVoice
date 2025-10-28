#!/bin/bash
# AutoVoice Build and Test Script
# Builds CUDA extensions and runs comprehensive tests

set -e

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

# Check CUDA Toolkit (for building)
if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
    print_success "CUDA Toolkit ${NVCC_VERSION}"
    HAS_NVCC=true
else
    print_warning "CUDA Toolkit not found - cannot build GPU extensions"
    HAS_NVCC=false
fi

# Step 2: Build CUDA Extensions
print_step "Building CUDA extensions"

if [ "$HAS_NVCC" = true ]; then
    echo "  Running: pip install -e ."
    echo ""

    # Clean previous build artifacts
    if [ -d "build" ]; then
        print_status "Cleaning previous build artifacts"
        rm -rf build/
    fi

    # Build
    if pip install -e . 2>&1 | tee build.log; then
        print_success "CUDA extensions built successfully"
        ((PASSED++))
    else
        print_error "Build failed - see build.log for details"
        ((FAILED++))

        # Show last 20 lines of error
        echo ""
        echo "Last 20 lines of build output:"
        tail -n 20 build.log

        exit 1
    fi
else
    print_warning "Skipping CUDA extension build (no nvcc)"
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
    if python tests/test_bindings_smoke.py 2>&1 | tee smoke_test.log; then
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

# Step 5: Run Unit Tests
print_step "Running unit tests"

# Find all test files
TEST_FILES=(
    "tests/test_pitch_extraction.py"
    "tests/test_singing_analysis.py"
    "tests/test_source_separator.py"
)

echo "Test files to run:"
for test_file in "${TEST_FILES[@]}"; do
    if [ -f "$test_file" ]; then
        echo "  - $test_file"
    fi
done
echo ""

# Run tests with pytest
if command -v pytest &> /dev/null; then
    PYTEST_ARGS="-v --tb=short"

    # Add markers based on CUDA availability
    if [ "$CUDA_AVAILABLE" = false ]; then
        PYTEST_ARGS="$PYTEST_ARGS -m 'not cuda'"
        print_warning "Running CPU tests only (CUDA not available)"
    fi

    # Run pytest
    if pytest $PYTEST_ARGS 2>&1 | tee pytest.log; then
        print_success "Unit tests passed"
        ((PASSED++))

        # Extract test summary
        if grep -q "passed" pytest.log; then
            TESTS_PASSED=$(grep -oP '\d+(?= passed)' pytest.log | head -1)
            TESTS_FAILED=$(grep -oP '\d+(?= failed)' pytest.log | head -1 || echo "0")
            TESTS_SKIPPED=$(grep -oP '\d+(?= skipped)' pytest.log | head -1 || echo "0")

            echo "  Tests passed: ${TESTS_PASSED}"
            [ "$TESTS_FAILED" != "0" ] && echo "  Tests failed: ${TESTS_FAILED}"
            [ "$TESTS_SKIPPED" != "0" ] && echo "  Tests skipped: ${TESTS_SKIPPED}"
        fi
    else
        print_error "Unit tests failed"
        ((FAILED++))

        # Show failure summary
        echo ""
        echo "Test failure summary:"
        grep -A 10 "FAILED" pytest.log || echo "See pytest.log for details"
    fi
else
    print_warning "pytest not installed - skipping unit tests"
    print_status "Install pytest: pip install pytest"
    ((SKIPPED++))
fi

# Step 6: Run Integration Tests (if CUDA available)
if [ "$CUDA_AVAILABLE" = true ]; then
    print_step "Running CUDA integration tests"

    CUDA_TEST_FILES=(
        "tests/test_pitch_extraction.py::TestSingingPitchExtractor::test_extract_f0_realtime_cuda"
    )

    for test_path in "${CUDA_TEST_FILES[@]}"; do
        test_name=$(echo "$test_path" | awk -F'::' '{print $NF}')

        if pytest "$test_path" -v 2>&1 | tee "cuda_test_${test_name}.log"; then
            print_success "CUDA test passed: ${test_name}"
            ((PASSED++))
        else
            print_warning "CUDA test failed: ${test_name}"
            ((FAILED++))
        fi
    done
else
    print_warning "Skipping CUDA integration tests (CUDA not available)"
    ((SKIPPED++))
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
