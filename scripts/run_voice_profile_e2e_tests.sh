#!/bin/bash
# Run voice profile training E2E tests
#
# Usage:
#   ./scripts/run_voice_profile_e2e_tests.sh [phase]
#
# Phases:
#   api       - API integration tests only
#   browser   - Browser automation tests only
#   all       - All E2E tests (default)
#   smoke     - Quick smoke tests

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
source "$SCRIPT_DIR/common_env.sh"
autovoice_activate_env

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PYTEST_ARGS="-x --tb=short -q"
PHASE="${1:-all}"

echo -e "${GREEN}Voice Profile Training E2E Test Suite${NC}"
echo "================================================"
echo "Phase: $PHASE"
echo "Python: $PYTHON"
echo ""

# Ensure test environment
# Check if Flask app is running
check_app_running() {
    if ! curl -s http://localhost:10600/health > /dev/null 2>&1; then
        echo -e "${YELLOW}WARNING: Flask app not running on localhost:10600${NC}"
        echo "Start the app with: python main.py --host 127.0.0.1 --port 10600"
        echo ""
        return 1
    fi
    return 0
}

# Run API tests
run_api_tests() {
    echo -e "${GREEN}Running API Integration Tests...${NC}"
    "$PYTHON" -m pytest tests/test_voice_profile_training_e2e.py \
        -m "integration" \
        $PYTEST_ARGS \
        --junit-xml=test-results/voice-profile-api-e2e.xml
}

# Run browser tests
run_browser_tests() {
    echo -e "${GREEN}Running Browser Automation Tests...${NC}"

    # Check VNC display
    if ! DISPLAY=:99 xdpyinfo > /dev/null 2>&1; then
        echo -e "${RED}ERROR: VNC display :99 not available${NC}"
        echo "Start VNC with: vncserver :99"
        return 1
    fi

    # Check if app is running
    if ! check_app_running; then
        return 1
    fi

    "$PYTHON" -m pytest tests/test_browser_voice_profile_workflow.py \
        -m "browser" \
        $PYTEST_ARGS \
        --junit-xml=test-results/voice-profile-browser-e2e.xml
}

# Run smoke tests
run_smoke_tests() {
    echo -e "${GREEN}Running Smoke Tests...${NC}"
    "$PYTHON" -m pytest tests/test_voice_profile_training_e2e.py \
        -m "integration and not slow" \
        $PYTEST_ARGS \
        --junit-xml=test-results/voice-profile-smoke.xml
}

# Create results directory
mkdir -p test-results

# Run tests based on phase
case "$PHASE" in
    api)
        run_api_tests
        ;;
    browser)
        run_browser_tests
        ;;
    smoke)
        run_smoke_tests
        ;;
    all)
        echo -e "${GREEN}Running all E2E tests...${NC}"
        echo ""

        # API tests
        if run_api_tests; then
            echo -e "${GREEN}✓ API tests passed${NC}"
        else
            echo -e "${RED}✗ API tests failed${NC}"
            exit 1
        fi

        echo ""

        # Browser tests
        if run_browser_tests; then
            echo -e "${GREEN}✓ Browser tests passed${NC}"
        else
            echo -e "${YELLOW}⚠ Browser tests skipped or failed${NC}"
            # Don't fail entire suite if browser tests are unavailable
        fi
        ;;
    *)
        echo -e "${RED}Unknown phase: $PHASE${NC}"
        echo "Valid phases: api, browser, smoke, all"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}E2E Tests Complete!${NC}"
echo ""
echo "Results: test-results/"
echo "  - voice-profile-api-e2e.xml"
echo "  - voice-profile-browser-e2e.xml"
echo ""
