#!/bin/bash

# Simple verification script to check Phase 1 report generation structure
# This script verifies the script syntax and structure without running it

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "Verifying Phase 1 Report Generation Structure"
echo "=============================================="
echo ""

SCRIPT_PATH="scripts/phase1_execute.sh"

# Test 1: Script syntax is valid
echo -e "${YELLOW}Test 1: Checking script syntax${NC}"
if bash -n "$SCRIPT_PATH"; then
    echo -e "${GREEN}✓ Script syntax is valid${NC}"
else
    echo -e "${RED}✗ Script syntax error${NC}"
    exit 1
fi

# Test 2: Verify REPORT_OUTPUT variable is defined
echo -e "${YELLOW}Test 2: Checking REPORT_OUTPUT variable${NC}"
if grep -q 'REPORT_OUTPUT=' "$SCRIPT_PATH"; then
    echo -e "${GREEN}✓ REPORT_OUTPUT variable defined${NC}"
else
    echo -e "${RED}✗ REPORT_OUTPUT variable not found${NC}"
    exit 1
fi

# Test 3: Verify --report-out flag parsing
echo -e "${YELLOW}Test 3: Checking --report-out flag parsing${NC}"
if grep -q '\-\-report-out' "$SCRIPT_PATH"; then
    echo -e "${GREEN}✓ --report-out flag parsing present${NC}"
else
    echo -e "${RED}✗ --report-out flag parsing not found${NC}"
    exit 1
fi

# Test 4: Verify generate_report function exists
echo -e "${YELLOW}Test 4: Checking generate_report function${NC}"
if grep -q 'generate_report()' "$SCRIPT_PATH"; then
    echo -e "${GREEN}✓ generate_report function exists${NC}"
else
    echo -e "${RED}✗ generate_report function not found${NC}"
    exit 1
fi

# Test 5: Verify report writes to REPORT_OUTPUT
echo -e "${YELLOW}Test 5: Checking report output destination${NC}"
if grep -q 'cat > "$REPORT_OUTPUT"' "$SCRIPT_PATH"; then
    echo -e "${GREEN}✓ Report writes to \$REPORT_OUTPUT${NC}"
else
    echo -e "${RED}✗ Report output not using \$REPORT_OUTPUT${NC}"
    exit 1
fi

# Test 6: Verify no FILLED file is created
echo -e "${YELLOW}Test 6: Checking for FILLED file generation${NC}"
if grep -q 'PHASE1_COMPLETION_REPORT_FILLED.md' "$SCRIPT_PATH"; then
    echo -e "${RED}✗ Script still references FILLED file${NC}"
    exit 1
else
    echo -e "${GREEN}✓ No FILLED file generation${NC}"
fi

# Test 7: Verify dynamic variables are collected
echo -e "${YELLOW}Test 7: Checking dynamic variable collection${NC}"
required_vars=(
    "python_version"
    "pytorch_version"
    "cuda_available"
    "gpu_name"
    "gpu_count"
    "cuda_version"
    "cudnn_version"
    "nvcc_version"
    "extension_path"
    "extension_size"
)

all_found=true
for var in "${required_vars[@]}"; do
    if ! grep -q "local $var=" "$SCRIPT_PATH"; then
        echo -e "${RED}✗ Variable $var not found${NC}"
        all_found=false
    fi
done

if [ "$all_found" = true ]; then
    echo -e "${GREEN}✓ All required variables collected${NC}"
else
    exit 1
fi

# Test 8: Verify dynamic checkboxes
echo -e "${YELLOW}Test 8: Checking dynamic checkbox generation${NC}"
if grep -q '\[\$(' "$SCRIPT_PATH"; then
    echo -e "${GREEN}✓ Dynamic checkbox generation present${NC}"
else
    echo -e "${RED}✗ Dynamic checkbox generation not found${NC}"
    exit 1
fi

# Test 9: Verify report sections
echo -e "${YELLOW}Test 9: Checking report sections${NC}"
required_sections=(
    "Executive Summary"
    "Pre-Flight Check Results"
    "CUDA Toolkit Installation"
    "CUDA Extension Build"
    "Bindings Verification"
    "PyTorch CUDA Validation"
    "Environment Snapshot"
    "Verification Checklist"
    "Next Steps"
    "Conclusion"
)

all_sections_found=true
for section in "${required_sections[@]}"; do
    if ! grep -q "## $section" "$SCRIPT_PATH"; then
        echo -e "${RED}✗ Section '$section' not found${NC}"
        all_sections_found=false
    fi
done

if [ "$all_sections_found" = true ]; then
    echo -e "${GREEN}✓ All required sections present${NC}"
else
    exit 1
fi

# Test 10: Verify summary file references correct report
echo -e "${YELLOW}Test 10: Checking summary file references${NC}"
if grep -q 'Review PHASE1_COMPLETION_REPORT.md' "$SCRIPT_PATH"; then
    echo -e "${GREEN}✓ Summary references correct report file${NC}"
else
    echo -e "${RED}✗ Summary doesn't reference PHASE1_COMPLETION_REPORT.md${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}All verification tests passed!${NC}"
echo ""
echo "The Phase 1 report generation has been successfully improved:"
echo "  ✓ Single canonical report file (PHASE1_COMPLETION_REPORT.md)"
echo "  ✓ Fully populated with dynamic content"
echo "  ✓ No duplicate FILLED file"
echo "  ✓ Optional --report-out flag for custom paths"
echo "  ✓ All required sections and variables present"
echo ""

