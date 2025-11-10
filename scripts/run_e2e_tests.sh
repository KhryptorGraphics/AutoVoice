#!/bin/bash
# End-to-End Test Execution and Reporting Script
# Purpose: Execute comprehensive E2E test suite and generate detailed report
# Usage: ./scripts/run_e2e_tests.sh [--quick|--full|--quality]

set -e
set -o pipefail

# Configuration
TEST_DIR="tests"
REPORT_DIR="validation_results/e2e"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="${REPORT_DIR}/e2e_test_report_${TIMESTAMP}.md"
JSON_FILE="${REPORT_DIR}/e2e_test_results_${TIMESTAMP}.json"
MODE=${1:---full}

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Helper functions
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_info() { echo -e "${BLUE}ℹ${NC} $1"; }

# Create report directory
mkdir -p "$REPORT_DIR"

echo "=== AutoVoice End-to-End Test Execution ==="
echo "Mode: $MODE"
echo "Timestamp: $TIMESTAMP"
echo ""

# Pre-flight checks
echo "[Pre-flight] Checking environment..."

# Check Python
if ! command -v python &> /dev/null; then
    print_error "Python not found"
    exit 1
fi
print_success "Python $(python --version 2>&1 | awk '{print $2}') found"

# Check pytest
if ! python -c "import pytest" 2>/dev/null; then
    print_error "pytest not installed"
    exit 1
fi
print_success "pytest found"

# Check pytest-json-report
if ! python -c "import pytest_jsonreport" 2>/dev/null; then
    print_warning "pytest-json-report not installed, installing..."
    pip install pytest-json-report --quiet || print_warning "Failed to install pytest-json-report"
fi

# Check CUDA
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [ "$CUDA_AVAILABLE" = "True" ]; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
    print_success "CUDA available: $GPU_NAME"
else
    print_warning "CUDA not available (tests will use CPU fallback)"
fi

# Check test data
if [ ! -d "tests/data/benchmark" ]; then
    print_warning "Test data not found, generating..."
    if [ -f "scripts/generate_benchmark_test_data.py" ]; then
        python scripts/generate_benchmark_test_data.py
    fi
fi

echo ""
echo "[Execution] Running E2E tests..."

# Determine pytest arguments based on mode
case "$MODE" in
    --quick)
        PYTEST_ARGS="-v -m 'e2e and not slow'"
        ;;
    --quality)
        PYTEST_ARGS="-v -m 'quality'"
        ;;
    --full|*)
        PYTEST_ARGS="-v -m 'e2e'"
        ;;
esac

# Run tests with JSON report
# Disable exit on error temporarily to capture exit code and generate report
set +e
START_TIME=$(date +%s)
pytest tests/test_end_to_end.py $PYTEST_ARGS --tb=short --color=yes \
    --json-report --json-report-file="$JSON_FILE" --json-report-indent=2 \
    2>&1 | tee "${REPORT_DIR}/pytest_output_${TIMESTAMP}.log"
EXIT_CODE=${PIPESTATUS[0]}
set -e

if [ $EXIT_CODE -eq 0 ]; then
    TEST_RESULT="PASSED"
else
    TEST_RESULT="FAILED"
fi
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))
DURATION_SEC=$((DURATION % 60))

echo ""
echo "[Report] Generating test report..."

# Parse test results from JSON report if available
if [ -f "$JSON_FILE" ] && command -v jq &> /dev/null; then
    print_info "Parsing JSON report with jq..."
    TOTAL_TESTS=$(jq -r '.summary.total // 0' "$JSON_FILE")
    PASSED_TESTS=$(jq -r '.summary.passed // 0' "$JSON_FILE")
    FAILED_TESTS=$(jq -r '.summary.failed // 0' "$JSON_FILE")
    SKIPPED_TESTS=$(jq -r '.summary.skipped // 0' "$JSON_FILE")
    JSON_DURATION=$(jq -r '.duration // 0' "$JSON_FILE")

    print_success "Parsed results from JSON report"
else
    # Fallback to parsing pytest output
    print_warning "JSON report not available or jq not found, parsing text output..."
    TOTAL_TESTS=$(grep -oP '\d+(?= passed)' "${REPORT_DIR}/pytest_output_${TIMESTAMP}.log" | head -1 || echo "0")
    PASSED_TESTS=$(grep -oP '\d+(?= passed)' "${REPORT_DIR}/pytest_output_${TIMESTAMP}.log" | head -1 || echo "0")
    FAILED_TESTS=$(grep -oP '\d+(?= failed)' "${REPORT_DIR}/pytest_output_${TIMESTAMP}.log" | head -1 || echo "0")
    SKIPPED_TESTS=$(grep -oP '\d+(?= skipped)' "${REPORT_DIR}/pytest_output_${TIMESTAMP}.log" | head -1 || echo "0")
    JSON_DURATION="N/A"
fi

# Fail explicitly if no tests were found
if [ "$TOTAL_TESTS" -eq 0 ]; then
    print_error "No tests were executed! Check test markers and pytest configuration."
    EXIT_CODE=1
    TEST_RESULT="FAILED"
fi

# Collect environment details
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "N/A")
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'N/A')" 2>/dev/null || echo "N/A")

# Generate Markdown report
cat > "$REPORT_FILE" << EOF
# End-to-End Test Execution Report

**Date**: $(date '+%Y-%m-%d %H:%M:%S')
**Mode**: $MODE
**Duration**: ${DURATION_MIN}m ${DURATION_SEC}s (${DURATION}s)
**Environment**:
- Python: $PYTHON_VERSION
- PyTorch: $PYTORCH_VERSION
- CUDA: $CUDA_VERSION
- GPU: $GPU_NAME
- CUDA Available: $CUDA_AVAILABLE

## Executive Summary

- **Total Tests**: $TOTAL_TESTS
- **Passed**: $PASSED_TESTS ✓
- **Failed**: $FAILED_TESTS ✗
- **Skipped**: $SKIPPED_TESTS ⊘
- **Success Rate**: $(python -c "print(f'{($PASSED_TESTS / max($TOTAL_TESTS, 1)) * 100:.1f}%')" 2>/dev/null || echo "N/A")
- **Overall Status**: $TEST_RESULT

## Test Results

**Detailed Outputs:**
- Text log: \`${REPORT_DIR}/pytest_output_${TIMESTAMP}.log\`
- JSON report: \`${JSON_FILE}\`

## Quality Gates

| Metric | Target | Status |
|--------|--------|--------|
| Pitch Accuracy (RMSE Hz) | <10.0 | See test output |
| Speaker Similarity | >85% | See test output |
| Overall Quality Score | >0.75 | See test output |
| MOS Estimation | >4.0 | See test output |
| STOI Score | >0.9 | See test output |

## Recommendations

$(if [ "$TEST_RESULT" = "PASSED" ]; then
    echo "- ✓ All tests passed - Ready for production"
    echo "- ✓ Quality gates met"
else
    echo "- ✗ Some tests failed - Review failures before deployment"
    echo "- ⚠ Check test output for details"
fi)

## Next Steps

1. Review detailed test output
2. $(if [ "$TEST_RESULT" = "PASSED" ]; then echo "Deploy to staging environment"; else echo "Fix failing tests and rerun"; fi)
3. Monitor production metrics

---

**Report Generated**: $(date '+%Y-%m-%d %H:%M:%S')
**Artifacts:**
- Markdown Report: \`$REPORT_FILE\`
- JSON Report: \`$JSON_FILE\`
- Test Output Log: \`${REPORT_DIR}/pytest_output_${TIMESTAMP}.log\`
EOF

print_success "Markdown report saved to: $REPORT_FILE"
if [ -f "$JSON_FILE" ]; then
    print_success "JSON report saved to: $JSON_FILE"
fi

echo ""
echo "=== Execution Summary ==="
echo "Total Tests: $TOTAL_TESTS"
echo "Passed: $PASSED_TESTS"
echo "Failed: $FAILED_TESTS"
echo "Skipped: $SKIPPED_TESTS"
echo "Duration: ${DURATION_MIN}m ${DURATION_SEC}s"
echo "Status: $TEST_RESULT"
echo ""

if [ "$TEST_RESULT" = "PASSED" ]; then
    print_success "All E2E tests PASSED"
else
    print_error "Some E2E tests FAILED"
fi

exit $EXIT_CODE

