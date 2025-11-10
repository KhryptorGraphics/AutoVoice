#!/bin/bash
# Quick Test Check Script
# Fast validation to verify tests pass without full suite run

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

print_header "Quick Test Validation"
echo "Started at: $(date)"
echo ""

# Check pytest availability
if ! command -v pytest &> /dev/null; then
    print_error "pytest not found. Install with: pip install pytest"
    exit 1
fi

# Parse command line options
VERBOSE=false
SPECIFIC_TEST=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -t|--test)
            SPECIFIC_TEST="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose    Show detailed test output"
            echo "  -t, --test PATH  Run specific test file or pattern"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run smoke tests only"
            echo "  $0 -v                                 # Verbose output"
            echo "  $0 -t tests/test_bindings_smoke.py   # Specific test file"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run validation
if [[ -n "$SPECIFIC_TEST" ]]; then
    print_info "Running specific test: $SPECIFIC_TEST"

    if [[ "$VERBOSE" == true ]]; then
        pytest "$SPECIFIC_TEST" -v --tb=short
    else
        pytest "$SPECIFIC_TEST" -q
    fi

    if [[ $? -eq 0 ]]; then
        print_success "Test passed!"
        exit 0
    else
        print_error "Test failed!"
        exit 1
    fi
else
    # Pre-flight validations
    print_header "Pre-Flight Validations"

    DISCOVERY_OK=true
    MARKERS_OK=true
    COVERAGE_OK=true
    VALIDATION_WARNINGS=0
    VALIDATION_ERRORS=0

    # 1. Test Discovery Check
    print_info "Checking test discovery..."
    DISCOVERY_OUTPUT=$(pytest tests/ --collect-only -q 2>&1)
    DISCOVERY_EXIT=$?

    if [[ $DISCOVERY_EXIT -eq 0 ]]; then
        TEST_COUNT=$(echo "$DISCOVERY_OUTPUT" | grep -E "^[0-9]+ test" | awk '{print $1}')
        if [[ -n "$TEST_COUNT" && "$TEST_COUNT" -gt 0 ]]; then
            print_success "Test discovery: $TEST_COUNT tests found"
        else
            print_warning "Test discovery: No tests found or count unclear"
            DISCOVERY_OK=false
            VALIDATION_WARNINGS=$((VALIDATION_WARNINGS + 1))
        fi
    else
        if echo "$DISCOVERY_OUTPUT" | grep -qi "error"; then
            print_error "Test discovery: Collection errors found"
            DISCOVERY_OK=false
            VALIDATION_ERRORS=$((VALIDATION_ERRORS + 1))
            if [[ "$VERBOSE" == true ]]; then
                echo "$DISCOVERY_OUTPUT" | grep -i "error" | head -5
            fi
        else
            print_warning "Test discovery: Unexpected exit code $DISCOVERY_EXIT"
            DISCOVERY_OK=false
            VALIDATION_WARNINGS=$((VALIDATION_WARNINGS + 1))
        fi
    fi

    # 2. Marker Validation Check
    print_info "Checking pytest markers..."
    MARKER_OUTPUT=$(pytest --markers 2>&1)
    if [[ $? -eq 0 ]]; then
        # More robust marker detection - look for lines containing @pytest.mark regardless of indentation
        CUSTOM_MARKERS=$(echo "$MARKER_OUTPUT" | grep -E "@pytest\.mark\." | wc -l)
        print_success "Markers: $CUSTOM_MARKERS custom markers registered"
    else
        print_warning "Markers: Could not retrieve marker list"
        VALIDATION_WARNINGS=$((VALIDATION_WARNINGS + 1))
    fi

    # Check for unknown marker warnings in collection (robust parsing)
    UNKNOWN_MARKERS=$(pytest tests/ --collect-only 2>&1 | grep -i "unknown marker" || true)
    if [[ -n "$UNKNOWN_MARKERS" ]]; then
        MARKERS_OK=false
        VALIDATION_WARNINGS=$((VALIDATION_WARNINGS + 1))
        if [[ "$VERBOSE" == true ]]; then
            echo "$UNKNOWN_MARKERS" | head -3
        fi
    else
        # More robust marker validation - collect all custom markers from test files
        CUSTOM_MARKERS_FOUND=$(grep -r "@pytest.mark\." tests/ 2>/dev/null | grep -v "\.git" | wc -l || echo "0")
        if [[ "$CUSTOM_MARKERS_FOUND" -gt 0 ]]; then
            print_success "Custom markers: Found $CUSTOM_MARKERS_FOUND marker usages"
        else
            print_info "No custom markers detected in test files"
        fi
    fi

    # 3. Coverage Plugin Check (robust detection)
    print_info "Checking coverage plugin..."

    # First check if pytest-cov is available via import
    COV_PLUGIN_AVAILABLE=false
    if python3 -c "import importlib.util; print(importlib.util.find_spec('pytest_cov') is not None)" 2>/dev/null | grep -q "True"; then
        COV_PLUGIN_AVAILABLE=true
    fi

    # Also check if pytest recognizes --cov option
    COV_OPTION_AVAILABLE=false
    if pytest --help 2>/dev/null | grep -q -- "--cov"; then
        COV_OPTION_AVAILABLE=true
    fi

    if [[ "$COV_PLUGIN_AVAILABLE" == true ]] && [[ "$COV_OPTION_AVAILABLE" == true ]]; then
        print_success "Coverage plugin: pytest-cov installed and recognized"
        COVERAGE_OK=true
    elif [[ "$COV_PLUGIN_AVAILABLE" == true ]]; then
        print_warning "Coverage plugin: Installed but not recognized by pytest"
        COVERAGE_OK=false
        VALIDATION_WARNINGS=$((VALIDATION_WARNINGS + 1))
    else
        # Fallback check: try running coverage and see if it works
        COV_TEST_OUTPUT=$(pytest tests/test_bindings_smoke.py::test_cuda_kernels_import --cov=src/auto_voice --cov-report=term -q 2>&1 || true)

        if echo "$COV_TEST_OUTPUT" | grep -qi "coverage"; then
            print_success "Coverage plugin: Working (via alternative method)"
            COVERAGE_OK=true
        else
            if echo "$COV_TEST_OUTPUT" | grep -qi "no module named.*coverage\|unknown option.*cov"; then
                print_error "Coverage plugin: Not installed or not working"
                COVERAGE_OK=false
                VALIDATION_ERRORS=$((VALIDATION_ERRORS + 1))
                echo "Install with: pip install pytest-cov"
            else
                print_warning "Coverage plugin: Status unclear"
                COVERAGE_OK=false
                VALIDATION_WARNINGS=$((VALIDATION_WARNINGS + 1))
            fi
        fi
    fi

    # Validation Summary
    echo ""
    print_header "Validation Summary"
    echo "Discovery: $([ "$DISCOVERY_OK" = true ] && echo "✅" || echo "⚠️")"
    echo "Markers:   $([ "$MARKERS_OK" = true ] && echo "✅" || echo "⚠️")"
    echo "Coverage:  $([ "$COVERAGE_OK" = true ] && echo "✅" || echo "❌")"
    echo ""

    if [[ $VALIDATION_ERRORS -gt 0 ]]; then
        print_error "Pre-flight validation failed with $VALIDATION_ERRORS critical error(s)"
        print_info "Fix errors before running tests"
        exit 1
    elif [[ $VALIDATION_WARNINGS -gt 0 ]]; then
        print_warning "Pre-flight validation completed with $VALIDATION_WARNINGS warning(s)"
        print_info "Continuing with smoke tests..."
    else
        print_success "All pre-flight validations passed"
    fi

    echo ""

    # Run smoke tests by default (fast validation)
    print_header "Smoke Tests"
    print_info "Running smoke tests (fast validation <30s)..."
    echo ""

    START_TIME=$(date +%s)

    if [[ "$VERBOSE" == true ]]; then
        pytest tests/test_bindings_smoke.py -v --tb=short
    else
        pytest tests/test_bindings_smoke.py -q
    fi

    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo ""
    print_info "Completed in ${DURATION}s"

    if [[ $EXIT_CODE -eq 0 ]]; then
        print_success "Quick validation passed! ✅"
        echo ""
        print_info "All critical functionality working"
        print_info "Ready for full test suite: ./scripts/phase2_execute.sh"

        # Exit with appropriate code based on validation warnings
        if [[ $VALIDATION_WARNINGS -gt 0 ]]; then
            print_warning "Note: $VALIDATION_WARNINGS validation warning(s) detected"
            exit 2  # Success with warnings
        else
            exit 0  # Complete success
        fi
    else
        print_error "Quick validation failed! ❌"
        echo ""
        print_info "Fix issues before running full suite"
        print_info "Debug with: pytest tests/test_bindings_smoke.py -v"
        exit 1
    fi
fi
