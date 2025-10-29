#!/bin/bash
# Run API E2E Tests for AutoVoice
# Implements Comment 12 requirements

set -e

echo "================================================"
echo "AutoVoice API E2E Test Suite"
echo "================================================"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create validation results directory
mkdir -p validation_results

echo -e "${YELLOW}Step 1: Checking prerequisites...${NC}"

# Check if pytest is installed
if ! python -c "import pytest" 2>/dev/null; then
    echo -e "${RED}Error: pytest not installed${NC}"
    echo "Install with: pip install pytest pytest-cov"
    exit 1
fi

# Check if requests is installed
if ! python -c "import requests" 2>/dev/null; then
    echo -e "${RED}Error: requests not installed${NC}"
    echo "Install with: pip install requests"
    exit 1
fi

echo -e "${GREEN}✓ Prerequisites satisfied${NC}"

echo -e "\n${YELLOW}Step 2: Running API E2E tests...${NC}"

# Run tests with detailed output
if pytest tests/test_api_e2e_validation.py -v -s --tb=short \
    --junit-xml=test-results/api_e2e_results.xml \
    --html=test-results/api_e2e_report.html --self-contained-html; then
    echo -e "\n${GREEN}✓ All API E2E tests passed!${NC}"
else
    echo -e "\n${RED}✗ Some tests failed. Check output above.${NC}"
    exit 1
fi

echo -e "\n${YELLOW}Step 3: Validation results summary${NC}"

# Count validation result files
RESULT_COUNT=$(ls -1 validation_results/*_results.json 2>/dev/null | wc -l)
echo -e "${GREEN}Generated ${RESULT_COUNT} validation result files${NC}"

echo -e "\n${YELLOW}Validation results saved to:${NC}"
ls -1 validation_results/*.json 2>/dev/null || echo "No results generated"

echo -e "\n${GREEN}================================================"
echo "API E2E Test Suite Complete"
echo "================================================${NC}"

echo -e "\nTest reports available at:"
echo "  - JUnit XML: test-results/api_e2e_results.xml"
echo "  - HTML: test-results/api_e2e_report.html"
echo "  - Validation results: validation_results/"
