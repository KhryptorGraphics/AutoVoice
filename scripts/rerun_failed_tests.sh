#!/bin/bash
# Rerun Failed Tests Script
# Automatically detects and reruns previously failed tests

set -e
set -o pipefail

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

print_header "Rerun Failed Tests"
echo "Started at: $(date)"
echo ""

# Check pytest availability
if ! command -v pytest &> /dev/null; then
    print_error "pytest not found. Install with: pip install pytest"
    exit 1
fi

# Parse command line options
USE_LASTFAILED=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --lf|--last-failed)
            USE_LASTFAILED=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --lf, --last-failed  Use pytest's lastfailed cache (recommended)"
            echo "  -v, --verbose        Show verbose output"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                   # Rerun from most recent full_suite log"
            echo "  $0 --lf              # Use pytest's lastfailed cache"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Strategy 1: Use pytest's lastfailed cache if available and requested
LASTFAILED_CACHE=".pytest_cache/v/cache/lastfailed"
FAILED_TESTS=()

if [[ "$USE_LASTFAILED" == true ]] && [[ -f "$LASTFAILED_CACHE" ]]; then
    print_info "Using pytest's lastfailed cache..."

    # Parse lastfailed cache with Python
    mapfile -t FAILED_TESTS < <(python3 - <<'PY'
import json
import sys
try:
    with open('.pytest_cache/v/cache/lastfailed', 'r') as f:
        data = json.load(f)
        for key in data.keys():
            print(key)
except Exception as e:
    sys.stderr.write(f"Error reading lastfailed cache: {e}\n")
    sys.exit(1)
PY
)

    if [[ ${#FAILED_TESTS[@]} -eq 0 ]]; then
        print_success "No failed tests in lastfailed cache!"
        print_info "All tests passed in last run."
        exit 0
    fi

    print_info "Found ${#FAILED_TESTS[@]} failed test(s) from cache"

else
    # Strategy 2: Parse most recent full_suite log
    print_info "Searching for recent test failures in logs..."

    # Only look at full_suite logs to avoid partial results
    LATEST_LOG=$(ls -t logs/full_suite_*.log 2>/dev/null | head -1)

    if [[ -z "$LATEST_LOG" ]]; then
        print_error "No full_suite logs found in logs/ directory"
        print_info "Run tests first: ./scripts/phase2_execute.sh"
        print_info "Or use --lf flag to use pytest's lastfailed cache"
        exit 1
    fi

    print_info "Checking log: $LATEST_LOG"

    # Extract failed test node IDs (format: tests/path.py::TestClass::test_name)
    mapfile -t FAILED_TESTS < <(grep "^FAILED " "$LATEST_LOG" | awk '{print $2}' | grep -v "^$")

    if [[ ${#FAILED_TESTS[@]} -eq 0 ]]; then
        print_success "No failed tests found in recent logs!"
        print_info "All tests passed in last run."
        exit 0
    fi

    print_info "Found ${#FAILED_TESTS[@]} failed test(s) from log"
fi

# Display failed tests
echo ""
print_warning "Failed tests to rerun:"
for test in "${FAILED_TESTS[@]}"; do
    echo "  - $test"
done

echo ""

# Ask for confirmation (unless CI mode)
if [[ -z "${CI:-}" && -z "${ALLOW_NO_CUDA:-}" ]]; then
    read -p "Rerun these tests? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Cancelled by user"
        exit 0
    fi
fi

# Create rerun log directory
mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RERUN_LOG="logs/rerun_failed_${TIMESTAMP}.log"

# Rerun failed tests
print_header "Rerunning Failed Tests"
echo "Logging to: $RERUN_LOG"
echo ""

START_TIME=$(date +%s)

# Rerun each test individually and track results (avoid subshell)
TESTS_PASSED=0
TESTS_FAILED=0
STILL_FAILING=()

for test in "${FAILED_TESTS[@]}"; do
    if [[ -n "$test" ]]; then
        print_info "Running: $test"

        # Run the test (disable set -e temporarily)
        set +e
        if [[ "$VERBOSE" == true ]]; then
            pytest "$test" -v --tb=short 2>&1 | tee -a "$RERUN_LOG"
        else
            pytest "$test" -v --tb=short >> "$RERUN_LOG" 2>&1
        fi
        TEST_EXIT=$?
        set -e

        if [[ $TEST_EXIT -eq 0 ]]; then
            print_success "✅ Now passing: $test"
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            print_error "❌ Still failing: $test"
            TESTS_FAILED=$((TESTS_FAILED + 1))
            STILL_FAILING+=("$test")
        fi

        echo ""
    fi
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
print_info "Rerun completed in ${DURATION}s"

echo ""
print_header "Rerun Results"

print_info "Tests rerun: ${#FAILED_TESTS[@]}"
print_info "Now passing: $TESTS_PASSED"
print_info "Still failing: $TESTS_FAILED"

echo ""

if [[ $TESTS_FAILED -eq 0 ]]; then
    print_success "All previously failed tests now pass! ✅"
    echo ""
    print_info "Next steps:"
    print_info "  1. Commit your fixes"
    print_info "  2. Run full suite: ./scripts/phase2_execute.sh"
    echo ""
    exit 0
else
    print_error "Some tests still failing ❌"
    echo ""

    print_info "Still failing tests:"
    for test in "${STILL_FAILING[@]}"; do
        echo "  - $test"
    done

    echo ""
    print_info "Next steps:"
    print_info "  1. Review failure details in: $RERUN_LOG"
    print_info "  2. Debug specific tests: pytest <test_path> -vv"
    print_info "  3. Fix issues and rerun: $0"
    echo ""

    exit 1
fi
