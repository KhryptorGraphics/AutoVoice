#!/bin/bash
# Local workflow testing script
# Mimics GitHub Actions environment for local validation

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔍 Starting Local Validation Workflow${NC}"
echo "================================================"

# Configuration
PYTHON_VERSION="3.10"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VALIDATION_DIR="$PROJECT_ROOT/validation_results"

cd "$PROJECT_ROOT"

# Step 1: Environment check
echo -e "\n${YELLOW}Step 1: Environment Check${NC}"
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'false')"
echo "Working directory: $PROJECT_ROOT"

# Step 2: Create directories
echo -e "\n${YELLOW}Step 2: Creating Directories${NC}"
mkdir -p "$VALIDATION_DIR"/{tests,quality,integration,docs}
mkdir -p tests/data/validation
echo "✓ Directories created"

# Step 3: Pre-task hook
echo -e "\n${YELLOW}Step 3: Pre-Task Hook${NC}"
if command -v npx &> /dev/null; then
    npx claude-flow@alpha hooks pre-task \
        --description "Local validation workflow" \
        --tags "local,validation,testing" || echo "⚠ Hook failed (non-critical)"
else
    echo "⚠ npx not found, skipping hook"
fi

# Step 4: Generate test data
echo -e "\n${YELLOW}Step 4: Generating Test Data${NC}"
if [ -f "tests/data/validation/generate_test_data.py" ]; then
    timeout 300 python tests/data/validation/generate_test_data.py || {
        echo -e "${RED}✗ Test data generation failed${NC}"
        exit 1
    }
    echo "✓ Test data generated"
else
    echo "⚠ Test data generator not found, skipping"
fi

# Step 5: Run system validation tests
echo -e "\n${YELLOW}Step 5: System Validation Tests${NC}"
pytest tests/test_system_validation.py \
    -v \
    --tb=short \
    --json-report \
    --json-report-file="$VALIDATION_DIR/tests/test_results.json" \
    --cov=src/auto_voice \
    --cov-report=html:"$VALIDATION_DIR/tests/coverage_html" \
    --cov-report=json:"$VALIDATION_DIR/tests/coverage.json" \
    --cov-report=term \
    --maxfail=5 \
    --timeout=300 \
    -n auto || {
        echo -e "${RED}✗ Some tests failed${NC}"
        TEST_FAILED=1
    }

# Step 6: Code quality validation
echo -e "\n${YELLOW}Step 6: Code Quality Validation${NC}"
python scripts/validate_code_quality.py \
    --output "$VALIDATION_DIR/quality/quality_report.json" || {
        echo -e "${RED}✗ Code quality issues found${NC}"
        QUALITY_FAILED=1
    }

# Step 7: Pylint
echo -e "\n${YELLOW}Step 7: Pylint Analysis${NC}"
pylint src/auto_voice \
    --output-format=json:"$VALIDATION_DIR/quality/pylint.json",colorized \
    --rcfile=.pylintrc \
    --exit-zero || true

# Step 8: Flake8
echo -e "\n${YELLOW}Step 8: Flake8 Analysis${NC}"
flake8 src/auto_voice tests \
    --format=json \
    --output-file="$VALIDATION_DIR/quality/flake8.json" \
    --exit-zero || true

# Step 9: Type checking
echo -e "\n${YELLOW}Step 9: Type Checking${NC}"
mypy src/auto_voice \
    --json-report "$VALIDATION_DIR/quality/mypy" \
    --html-report "$VALIDATION_DIR/quality/mypy_html" \
    --ignore-missing-imports || {
        echo -e "${RED}✗ Type checking found issues${NC}"
        MYPY_FAILED=1
    }

# Step 10: Security analysis
echo -e "\n${YELLOW}Step 10: Security Analysis${NC}"
bandit -r src/auto_voice \
    -f json \
    -o "$VALIDATION_DIR/quality/bandit.json" \
    --exit-zero || true

# Step 11: Complexity metrics
echo -e "\n${YELLOW}Step 11: Complexity Metrics${NC}"
radon cc src/auto_voice -a -j > "$VALIDATION_DIR/quality/complexity.json" || true
radon mi src/auto_voice -j > "$VALIDATION_DIR/quality/maintainability.json" || true

# Step 12: Integration validation
echo -e "\n${YELLOW}Step 12: Integration Validation${NC}"
python scripts/validate_integration.py \
    --output "$VALIDATION_DIR/integration/integration_report.json" || {
        echo -e "${RED}✗ Integration validation issues found${NC}"
        INTEGRATION_FAILED=1
    }

# Step 13: Documentation validation
echo -e "\n${YELLOW}Step 13: Documentation Validation${NC}"
python scripts/validate_documentation.py \
    --output "$VALIDATION_DIR/docs/docs_report.json" || {
        echo -e "${RED}✗ Documentation validation issues found${NC}"
        DOCS_FAILED=1
    }

# Step 14: Post-task hook
echo -e "\n${YELLOW}Step 14: Post-Task Hook${NC}"
if command -v npx &> /dev/null; then
    npx claude-flow@alpha hooks post-task \
        --task-id "local-validation-$$" \
        --status "completed" || echo "⚠ Hook failed (non-critical)"
else
    echo "⚠ npx not found, skipping hook"
fi

# Step 15: Generate report
echo -e "\n${YELLOW}Step 15: Generating Validation Report${NC}"
python scripts/generate_validation_report.py \
    --input-dir "$VALIDATION_DIR" \
    --output FINAL_VALIDATION_REPORT.md \
    --format markdown || {
        echo -e "${RED}✗ Report generation failed${NC}"
        exit 1
    }

python scripts/generate_validation_report.py \
    --input-dir "$VALIDATION_DIR" \
    --output "$VALIDATION_DIR/summary.json" \
    --format json || true

# Step 16: Check validation targets
echo -e "\n${YELLOW}Step 16: Checking Validation Targets${NC}"
if grep -q "❌ FAILED" FINAL_VALIDATION_REPORT.md; then
    echo -e "${RED}✗ Validation targets not met${NC}"
    echo ""
    echo "Failure details:"
    grep -A 5 "❌ FAILED" FINAL_VALIDATION_REPORT.md || true
    VALIDATION_FAILED=1
else
    echo -e "${GREEN}✓ All validation targets met${NC}"
fi

# Step 17: Session end hook
echo -e "\n${YELLOW}Step 17: Session End Hook${NC}"
if command -v npx &> /dev/null; then
    npx claude-flow@alpha hooks session-end \
        --session-id "local-validation-$$" \
        --export-metrics true || echo "⚠ Hook failed (non-critical)"
else
    echo "⚠ npx not found, skipping hook"
fi

# Summary
echo ""
echo "================================================"
echo -e "${GREEN}📊 Validation Summary${NC}"
echo "================================================"
echo "Results directory: $VALIDATION_DIR"
echo "Full report: FINAL_VALIDATION_REPORT.md"
echo ""

# Display report summary
if [ -f "FINAL_VALIDATION_REPORT.md" ]; then
    echo "Report preview:"
    head -n 30 FINAL_VALIDATION_REPORT.md
    echo ""
    echo "... (see FINAL_VALIDATION_REPORT.md for full report)"
fi

# Exit code
if [ -n "$TEST_FAILED" ] || [ -n "$QUALITY_FAILED" ] || \
   [ -n "$MYPY_FAILED" ] || [ -n "$INTEGRATION_FAILED" ] || \
   [ -n "$DOCS_FAILED" ] || [ -n "$VALIDATION_FAILED" ]; then
    echo ""
    echo -e "${RED}❌ Validation completed with failures${NC}"
    exit 1
else
    echo ""
    echo -e "${GREEN}✅ Validation completed successfully${NC}"
    exit 0
fi
