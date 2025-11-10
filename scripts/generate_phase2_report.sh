#!/bin/bash
# Phase 2 Report Generation Script
# Aggregates test results and generates comprehensive Phase 2 completion report

set -e
set -o pipefail
set -u

# Validate timestamp argument
if [[ $# -lt 1 ]] || [[ -z "${1:-}" ]]; then
    echo "Usage: $0 TIMESTAMP"
    echo "Example: $0 20251101_103500"
    exit 1
fi

# Accept timestamp as argument
TIMESTAMP=$1

# Set project root and directories
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="${PROJECT_ROOT}/logs"
REPORT_PATH="${PROJECT_ROOT}/PHASE2_COMPLETION_REPORT.md"

# Ensure logs directory exists
mkdir -p "$LOG_DIR"

echo "Generating Phase 2 completion report..."
echo "Timestamp: $TIMESTAMP"
echo "Log directory: $LOG_DIR"
echo "Report path: $REPORT_PATH"

# Validate that expected log files exist
EXPECTED_LOGS=(
    "${LOG_DIR}/smoke_tests_${TIMESTAMP}.log"
    "${LOG_DIR}/integration_tests_${TIMESTAMP}.log"
    "${LOG_DIR}/full_suite_${TIMESTAMP}.log"
)

MISSING_LOGS=()
for log_file in "${EXPECTED_LOGS[@]}"; do
    if [[ ! -f "$log_file" ]]; then
        MISSING_LOGS+=("$log_file")
    fi
done

if [[ ${#MISSING_LOGS[@]} -gt 0 ]]; then
    echo "ERROR: Expected log files are missing:"
    for missing in "${MISSING_LOGS[@]}"; do
        echo "  - $missing"
    done
    echo ""
    echo "Please ensure Phase 2 execution completed successfully before generating the report."
    exit 1
fi

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
    python3 - "$1" <<'PY'
import sys
print(eval(sys.argv[1]))
PY
}

# Function to extract test results from log file
extract_results() {
    local log_file=$1
    if [[ ! -f "$log_file" ]]; then
        echo "0 0 0"
        return
    fi

    local results=$(grep -E "passed|failed|skipped" "$log_file" | tail -1)
    if [[ -z "$results" ]]; then
        echo "0 0 0"
        return
    fi

    local passed=$(echo "$results" | grep -o "[0-9]\+ passed" | awk '{print $1}' || echo "0")
    local failed=$(echo "$results" | grep -o "[0-9]\+ failed" | awk '{print $1}' || echo "0")
    local skipped=$(echo "$results" | grep -o "[0-9]\+ skipped" | awk '{print $1}' || echo "0")

    echo "$passed $failed $skipped"
}

# Parse smoke test results
echo "Parsing smoke test results..."
SMOKE_LOG="${LOG_DIR}/smoke_tests_${TIMESTAMP}.log"
read SMOKE_PASSED SMOKE_FAILED SMOKE_SKIPPED <<< $(extract_results "$SMOKE_LOG")
SMOKE_DURATION=$(grep -o "completed in [0-9]\+s" "$SMOKE_LOG" | awk '{print $3}' | sed 's/s//' || echo "0")

# Parse integration test results
echo "Parsing integration test results..."
INTEGRATION_LOG="${LOG_DIR}/integration_tests_${TIMESTAMP}.log"
read INTEGRATION_PASSED INTEGRATION_FAILED INTEGRATION_SKIPPED <<< $(extract_results "$INTEGRATION_LOG")
INTEGRATION_DURATION=$(grep -o "completed in [0-9]\+s" "$INTEGRATION_LOG" | awk '{print $3}' | sed 's/s//' || echo "0")

# Parse core component results
echo "Parsing core component results..."
AUDIO_LOG="${LOG_DIR}/audio_processor_${TIMESTAMP}.log"
read AUDIO_PASSED AUDIO_FAILED AUDIO_SKIPPED <<< $(extract_results "$AUDIO_LOG")
AUDIO_DURATION=$(grep -o "completed in [0-9]\+s" "$AUDIO_LOG" | awk '{print $3}' | sed 's/s//' || echo "0")

MODEL_LOG="${LOG_DIR}/models_${TIMESTAMP}.log"
read MODEL_PASSED MODEL_FAILED MODEL_SKIPPED <<< $(extract_results "$MODEL_LOG")
MODEL_DURATION=$(grep -o "completed in [0-9]\+s" "$MODEL_LOG" | awk '{print $3}' | sed 's/s//' || echo "0")

INFERENCE_LOG="${LOG_DIR}/inference_${TIMESTAMP}.log"
read INFERENCE_PASSED INFERENCE_FAILED INFERENCE_SKIPPED <<< $(extract_results "$INFERENCE_LOG")
INFERENCE_DURATION=$(grep -o "completed in [0-9]\+s" "$INFERENCE_LOG" | awk '{print $3}' | sed 's/s//' || echo "0")

# Parse full suite results
echo "Parsing full suite results..."
FULL_LOG="${LOG_DIR}/full_suite_${TIMESTAMP}.log"
read FULL_PASSED FULL_FAILED FULL_SKIPPED <<< $(extract_results "$FULL_LOG")
FULL_DURATION=$(grep -o "completed in [0-9]\+s" "$FULL_LOG" | awk '{print $3}' | sed 's/s//' || echo "0")
FULL_TOTAL=$((FULL_PASSED + FULL_FAILED + FULL_SKIPPED))

# Parse coverage percentage
COVERAGE_PERCENT="0"
if [[ -f "$FULL_LOG" ]]; then
    COVERAGE_LINE=$(grep "TOTAL" "$FULL_LOG" | tail -1)
    if [[ -n "$COVERAGE_LINE" ]]; then
        COVERAGE_PERCENT=$(echo "$COVERAGE_LINE" | awk '{print $NF}' | sed 's/%//')
    fi
fi

# Calculate totals
TOTAL_PASSED=$((SMOKE_PASSED + INTEGRATION_PASSED + AUDIO_PASSED + MODEL_PASSED + INFERENCE_PASSED))
TOTAL_FAILED=$((SMOKE_FAILED + INTEGRATION_FAILED + AUDIO_FAILED + MODEL_FAILED + INFERENCE_FAILED))
TOTAL_SKIPPED=$((SMOKE_SKIPPED + INTEGRATION_SKIPPED + AUDIO_SKIPPED + MODEL_SKIPPED + INFERENCE_SKIPPED))
TOTAL_TESTS=$((TOTAL_PASSED + TOTAL_FAILED + TOTAL_SKIPPED))

if [[ $TOTAL_TESTS -gt 0 ]]; then
    PASS_RATE=$(( (TOTAL_PASSED * 100) / TOTAL_TESTS ))
else
    PASS_RATE=0
fi

# Parse coverage data from coverage.json
echo "Parsing coverage data..."
MODULES_ABOVE_80=0
MODULES_BELOW_80=0
TOP_LOW_MODULES=""
TOP_HIGH_MODULES=""
AUDIO_COVERAGE="0.0"
MODELS_COVERAGE="0.0"
INFERENCE_COVERAGE="0.0"
TRAINING_COVERAGE="0.0"
WEB_COVERAGE="0.0"
CUDA_COVERAGE="0.0"
CORE_COVERAGE="0.0"

if [[ -f "coverage.json" ]]; then
    # Use Python to parse JSON and extract module coverage
    python3 -c "
import json
import sys
from collections import defaultdict

try:
    with open('coverage.json', 'r') as f:
        data = json.load(f)

    files = data.get('files', {})
    modules = []

    # Component-level aggregation
    components = defaultdict(lambda: {'covered': 0, 'total': 0})

    for file_path, file_data in files.items():
        summary = file_data.get('summary', {})
        covered = summary.get('covered_lines', 0)
        total = summary.get('num_statements', 0)

        if total > 0:
            coverage_pct = (covered / total) * 100
            modules.append({
                'path': file_path,
                'coverage': coverage_pct,
                'covered': covered,
                'total': total
            })

            # Categorize by component
            if 'audio' in file_path.lower():
                components['audio']['covered'] += covered
                components['audio']['total'] += total
            elif 'model' in file_path.lower():
                components['models']['covered'] += covered
                components['models']['total'] += total
            elif 'inference' in file_path.lower():
                components['inference']['covered'] += covered
                components['inference']['total'] += total
            elif 'training' in file_path.lower():
                components['training']['covered'] += covered
                components['training']['total'] += total
            elif 'web' in file_path.lower():
                components['web']['covered'] += covered
                components['web']['total'] += total
            elif 'cuda' in file_path.lower():
                components['cuda']['covered'] += covered
                components['cuda']['total'] += total
            else:
                components['core']['covered'] += covered
                components['core']['total'] += total

    # Sort by coverage
    modules.sort(key=lambda x: x['coverage'])

    # Count modules above/below 80%
    above_80 = [m for m in modules if m['coverage'] >= 80]
    below_80 = [m for m in modules if m['coverage'] < 80]

    print(f'{len(above_80)} {len(below_80)}')

    # Top 5 lowest coverage modules
    for i, m in enumerate(modules[:5]):
        print(f'{m[\"path\"]}|{m[\"coverage\"]:.1f}%|{m[\"covered\"]}|{m[\"total\"]}')

    print('---')

    # Top 5 highest coverage modules
    for i, m in enumerate(modules[-5:][::-1]):
        print(f'{m[\"path\"]}|{m[\"coverage\"]:.1f}%|{m[\"covered\"]}|{m[\"total\"]}')

    print('---')

    # Component coverage
    for comp_name in ['audio', 'models', 'inference', 'training', 'web', 'cuda', 'core']:
        comp = components[comp_name]
        if comp['total'] > 0:
            comp_cov = (comp['covered'] / comp['total']) * 100
            print(f'{comp_name}|{comp_cov:.1f}')
        else:
            print(f'{comp_name}|0.0')

except Exception as e:
    print(f'0 0')
    for _ in range(10):
        print('Error parsing coverage data|0.0%|0|0')
    print('---')
    for _ in range(7):
        print('error|0.0')
" > /tmp/coverage_data.txt

    read MODULES_ABOVE_80 MODULES_BELOW_80 <<< $(head -1 /tmp/coverage_data.txt)

    # Extract top modules
    TOP_LOW_MODULES=$(sed -n '2,6p' /tmp/coverage_data.txt | tr '\n' ';' | sed 's/;$//')
    TOP_HIGH_MODULES=$(sed -n '8,12p' /tmp/coverage_data.txt | tr '\n' ';' | sed 's/;$//')

    # Extract component coverage
    AUDIO_COVERAGE=$(sed -n '14p' /tmp/coverage_data.txt | cut -d'|' -f2)
    MODELS_COVERAGE=$(sed -n '15p' /tmp/coverage_data.txt | cut -d'|' -f2)
    INFERENCE_COVERAGE=$(sed -n '16p' /tmp/coverage_data.txt | cut -d'|' -f2)
    TRAINING_COVERAGE=$(sed -n '17p' /tmp/coverage_data.txt | cut -d'|' -f2)
    WEB_COVERAGE=$(sed -n '18p' /tmp/coverage_data.txt | cut -d'|' -f2)
    CUDA_COVERAGE=$(sed -n '19p' /tmp/coverage_data.txt | cut -d'|' -f2)
    CORE_COVERAGE=$(sed -n '20p' /tmp/coverage_data.txt | cut -d'|' -f2)
fi

# Identify critical failures
echo "Identifying critical failures..."
CRITICAL_FAILURES=0
FAILURE_CATEGORIES=""

# Check for import errors
IMPORT_ERRORS=$(grep -h -i "importerror\|modulenotfound" ${LOG_DIR}/*_${TIMESTAMP}.log 2>/dev/null | wc -l || echo "0")
if [[ $IMPORT_ERRORS -gt 0 ]]; then
    FAILURE_CATEGORIES="${FAILURE_CATEGORIES}Import Errors: $IMPORT_ERRORS, "
    CRITICAL_FAILURES=$((CRITICAL_FAILURES + IMPORT_ERRORS))
fi

# Check for CUDA errors
CUDA_ERRORS=$(grep -h -i "cuda\|runtimeerror.*gpu" ${LOG_DIR}/*_${TIMESTAMP}.log 2>/dev/null | wc -l || echo "0")
if [[ $CUDA_ERRORS -gt 0 ]]; then
    FAILURE_CATEGORIES="${FAILURE_CATEGORIES}CUDA Errors: $CUDA_ERRORS, "
fi

# Check for assertion errors
ASSERTION_ERRORS=$(grep -h -i "assertionerror" ${LOG_DIR}/*_${TIMESTAMP}.log 2>/dev/null | wc -l || echo "0")
if [[ $ASSERTION_ERRORS -gt 0 ]]; then
    FAILURE_CATEGORIES="${FAILURE_CATEGORIES}Assertion Errors: $ASSERTION_ERRORS, "
fi

# Remove trailing comma and space
FAILURE_CATEGORIES=$(echo "$FAILURE_CATEGORIES" | sed 's/, $//')

# Extract top 10 slowest tests
TOP_SLOW_TESTS=""
if [[ -f "$FULL_LOG" ]]; then
    TOP_SLOW_TESTS=$(grep -A 10 "slowest test durations" "$FULL_LOG" | tail -10 | sed 's/^ *//' | tr '\n' ';' | sed 's/;$//')
fi

# Calculate total duration (sum all test step durations)
TOTAL_DURATION=$((SMOKE_DURATION + INTEGRATION_DURATION + AUDIO_DURATION + MODEL_DURATION + INFERENCE_DURATION + FULL_DURATION))

# Determine overall status
if [[ $TOTAL_FAILED -eq 0 && $CRITICAL_FAILURES -eq 0 ]]; then
    if [[ $(py_compare "$COVERAGE_PERCENT" ">=" "80" 2>/dev/null || echo "0") -eq 1 ]]; then
        OVERALL_STATUS="Success"
    else
        OVERALL_STATUS="Partial Success"
    fi
else
    OVERALL_STATUS="Failed"
fi

# Generate the report
echo "Generating markdown report..."

cat > "$REPORT_PATH" << EOF
# Phase 2 Completion Report: Execute Core Test Suite and Validate Functionality

**Date**: $(date +"%Y-%m-%d %H:%M:%S")
**Duration**: ${TOTAL_DURATION} seconds
**Overall Status**: ${OVERALL_STATUS}
**Test Execution Time**: ${TOTAL_DURATION} seconds
**Coverage Achieved**: ${COVERAGE_PERCENT}%

## Executive Summary

Phase 2 execution completed with the following key metrics:

- **Total tests executed**: ${TOTAL_TESTS}
- **Tests passed**: ${TOTAL_PASSED} (${PASS_RATE}%)
- **Tests failed**: ${TOTAL_FAILED}
- **Tests skipped**: ${TOTAL_SKIPPED}
- **Coverage achieved**: ${COVERAGE_PERCENT}% (target: 80%)

**Overall assessment**: ${OVERALL_STATUS}

EOF

if [[ $CRITICAL_FAILURES -gt 0 ]]; then
    cat >> "$REPORT_PATH" << EOF
**Critical issues summary**: ${CRITICAL_FAILURES} critical failures detected

EOF
fi

cat >> "$REPORT_PATH" << EOF
## Test Execution Results

### Smoke Tests (7 tests, <30s)

**Command executed**: \`pytest tests/test_bindings_smoke.py -v --tb=short\`

**Results**: ${SMOKE_PASSED} passed, ${SMOKE_FAILED} failed, ${SMOKE_SKIPPED} skipped
**Duration**: ${SMOKE_DURATION} seconds
**Status**: $([[ $SMOKE_FAILED -eq 0 ]] && echo "✅ All passed" || echo "❌ Critical failures")

### Integration Tests (9 tests, 1-5min)

**Command executed**: \`pytest tests/test_bindings_integration.py -v --tb=short\`

**Results**: ${INTEGRATION_PASSED} passed, ${INTEGRATION_FAILED} failed, ${INTEGRATION_SKIPPED} skipped
**Duration**: ${INTEGRATION_DURATION} seconds
**Status**: $([[ $INTEGRATION_FAILED -eq 0 ]] && echo "✅ All passed" || echo "⚠️ Some failed")

### Audio Processor Tests

**Command executed**: \`pytest tests/test_audio_processor.py -v --tb=short\`

**Results**: ${AUDIO_PASSED} passed, ${AUDIO_FAILED} failed, ${AUDIO_SKIPPED} skipped
**Duration**: ${AUDIO_DURATION} seconds

### Model Tests

**Command executed**: \`pytest tests/test_models.py -v --tb=short\`

**Results**: ${MODEL_PASSED} passed, ${MODEL_FAILED} failed, ${MODEL_SKIPPED} skipped
**Duration**: ${MODEL_DURATION} seconds

### Inference Tests

**Command executed**: \`pytest tests/test_inference.py -v --tb=short\`

**Results**: ${INFERENCE_PASSED} passed, ${INFERENCE_FAILED} failed, ${INFERENCE_SKIPPED} skipped
**Duration**: ${INFERENCE_DURATION} seconds

### Full Test Suite

**Command executed**: \`pytest tests/ -v --cov=src/auto_voice --cov-report=html --cov-report=term-missing --cov-report=json --tb=short --durations=10\`

**Total tests collected**: ${FULL_TOTAL}
**Results**: ${FULL_PASSED} passed, ${FULL_FAILED} failed, ${FULL_SKIPPED} skipped
**Duration**: ${FULL_DURATION} seconds
**Overall status**: $([[ $FULL_FAILED -eq 0 ]] && echo "✅ All passed" || echo "❌ Some failed")

## Coverage Analysis

### Overall Coverage

**Overall Coverage**: ${COVERAGE_PERCENT}% (target: 80%)
**Coverage Status**: $( [[ $(py_compare "$COVERAGE_PERCENT" ">=" "80" 2>/dev/null || echo "0") -eq 1 ]] && echo "✅ Target met" || echo "❌ Below target" )

### Coverage by Component

- Audio Processing (\`audio/\`): ${AUDIO_COVERAGE}%
- Model Architectures (\`models/\`): ${MODELS_COVERAGE}%
- Inference Engines (\`inference/\`): ${INFERENCE_COVERAGE}%
- Training Pipeline (\`training/\`): ${TRAINING_COVERAGE}%
- Web Interface (\`web/\`): ${WEB_COVERAGE}%
- CUDA Kernels (\`cuda_kernels/\`): ${CUDA_COVERAGE}%
- Core/Utilities: ${CORE_COVERAGE}%

### Modules Below Threshold

| Module | Coverage | Gap to 80% |
|--------|----------|------------|
EOF

# Add modules below 80%
if [[ -n "$TOP_LOW_MODULES" ]]; then
    IFS=';' read -ra MODULES <<< "$TOP_LOW_MODULES"
    for module in "${MODULES[@]}"; do
        if [[ -n "$module" && "$module" != "Error parsing coverage data|0.0%|0|0" ]]; then
            IFS='|' read -r path coverage covered total <<< "$module"
            # Strip % sign from coverage before arithmetic
            coverage_num="${coverage%\%}"
            gap=$(py_calc "80 - $coverage_num" 2>/dev/null || echo "0")
            printf "| %s | %s | %.1f%% |\n" "$path" "$coverage" "$gap" >> "$REPORT_PATH"
        fi
    done
fi

cat >> "$REPORT_PATH" << EOF

### Top 5 Modules with Highest Coverage

| Module | Coverage | Covered Lines | Total Lines |
|--------|----------|---------------|-------------|
EOF

# Add modules with highest coverage
if [[ -n "$TOP_HIGH_MODULES" ]]; then
    IFS=';' read -ra MODULES <<< "$TOP_HIGH_MODULES"
    for module in "${MODULES[@]}"; do
        if [[ -n "$module" && "$module" != "Error parsing coverage data|0.0%|0|0" ]]; then
            IFS='|' read -r path coverage covered total <<< "$module"
            printf "| %s | %s | %s | %s |\n" "$path" "$coverage" "$covered" "$total" >> "$REPORT_PATH"
        fi
    done
fi

cat >> "$REPORT_PATH" << EOF

**Coverage Report Location**: \`htmlcov/index.html\`
**Coverage Analysis Report**: \`docs/coverage_analysis_report.md\`

## Critical Failures Analysis

**Total Failures**: ${TOTAL_FAILED}
**Critical Failures**: ${CRITICAL_FAILURES} (smoke tests or marked critical)

### Failure Categories

- Import Errors: ${IMPORT_ERRORS}
- CUDA Errors: ${CUDA_ERRORS}
- Assertion Errors: ${ASSERTION_ERRORS}

### Detailed Failure Reports

EOF

# Extract and add detailed failure information
if [[ $TOTAL_FAILED -gt 0 ]]; then
    echo "Extracting detailed failure information..."

    # Parse failed tests from logs
    python3 -c "
import re
import sys

failed_tests = []

# Parse all log files for failures
log_files = [
    '${LOG_DIR}/smoke_tests_${TIMESTAMP}.log',
    '${LOG_DIR}/integration_tests_${TIMESTAMP}.log',
    '${LOG_DIR}/audio_processor_${TIMESTAMP}.log',
    '${LOG_DIR}/models_${TIMESTAMP}.log',
    '${LOG_DIR}/inference_${TIMESTAMP}.log',
    '${LOG_DIR}/full_suite_${TIMESTAMP}.log'
]

for log_file in log_files:
    try:
        with open(log_file, 'r') as f:
            content = f.read()

        # Find FAILED test lines
        failed_pattern = r'FAILED ([\w/:.]+) - (.+?)(?=\n|$)'
        matches = re.findall(failed_pattern, content)

        for test_id, error_msg in matches:
            # Extract just the first line of error
            error_first_line = error_msg.split('\n')[0].strip()[:100]
            failed_tests.append((test_id, error_first_line))

    except FileNotFoundError:
        pass

# Print unique failures
seen = set()
for test_id, error in failed_tests:
    if test_id not in seen:
        print(f'{test_id}|||{error}')
        seen.add(test_id)
" > /tmp/failed_tests.txt

    cat >> "$REPORT_PATH" << 'EOF'

| Test ID | Error Summary |
|---------|---------------|
EOF

    if [[ -s /tmp/failed_tests.txt ]]; then
        while IFS='|||' read -r test_id error; do
            printf "| %s | %s |\n" "$test_id" "$error" >> "$REPORT_PATH"
        done < /tmp/failed_tests.txt
    else
        echo "| No detailed failure information available | See log files for details |" >> "$REPORT_PATH"
    fi
else
    cat >> "$REPORT_PATH" << 'EOF'

No test failures detected. All tests passed successfully!
EOF
fi

cat >> "$REPORT_PATH" << EOF

## Performance Metrics

**Total Execution Time**: ${TOTAL_DURATION} seconds (${TOTAL_DURATION} minutes)
**Average Test Duration**: $([[ $FULL_TOTAL -gt 0 ]] && py_calc "$TOTAL_DURATION / $FULL_TOTAL" 2>/dev/null || echo "0") seconds

### Top 10 Slowest Tests

| Rank | Test Name | Duration (s) | File |
|------|-----------|--------------|------|
EOF

# Add slowest tests
if [[ -n "$TOP_SLOW_TESTS" ]]; then
    IFS=';' read -ra TESTS <<< "$TOP_SLOW_TESTS"
    rank=1
    for test in "${TESTS[@]}"; do
        if [[ -n "$test" ]]; then
            printf "| %d | %s | | |\n" "$rank" "$test" >> "$REPORT_PATH"
            ((rank++))
        fi
    done
fi

cat >> "$REPORT_PATH" << EOF

## Recommendations

### Immediate Actions

EOF

if [[ $CRITICAL_FAILURES -gt 0 ]]; then
    cat >> "$REPORT_PATH" << EOF
- **Fix critical failures**: ${CRITICAL_FAILURES} critical failures detected in smoke tests
- Review failure logs and address import/CUDA issues
- Re-run smoke tests before proceeding

EOF
fi

if [[ $(py_compare "$COVERAGE_PERCENT" "<" "80" 2>/dev/null || echo "0") -eq 1 ]]; then
    cat >> "$REPORT_PATH" << EOF
- **Address coverage gaps**: Current coverage ${COVERAGE_PERCENT}% below 80% target
- Review \`docs/coverage_analysis_report.md\` for specific recommendations
- Add tests for modules with low coverage

EOF
fi

cat >> "$REPORT_PATH" << EOF
### Next Phase Readiness

**Ready for Phase 3**: $([[ $TOTAL_FAILED -eq 0 && $CRITICAL_FAILURES -eq 0 && $(py_compare "$COVERAGE_PERCENT" ">=" "80" 2>/dev/null || echo "0") -eq 1 ]] && echo "✅ Yes" || echo "❌ No")

EOF

if [[ $TOTAL_FAILED -gt 0 || $CRITICAL_FAILURES -gt 0 ]]; then
    cat >> "$REPORT_PATH" << EOF
**Blockers**: ${TOTAL_FAILED} test failures, ${CRITICAL_FAILURES} critical failures

EOF
fi

cat >> "$REPORT_PATH" << EOF
## Appendices

### Appendix A: Test Execution Logs

- Smoke tests: \`logs/smoke_tests_${TIMESTAMP}.log\`
- Integration tests: \`logs/integration_tests_${TIMESTAMP}.log\`
- Audio processor: \`logs/audio_processor_${TIMESTAMP}.log\`
- Models: \`logs/models_${TIMESTAMP}.log\`
- Inference: \`logs/inference_${TIMESTAMP}.log\`
- Full suite: \`logs/full_suite_${TIMESTAMP}.log\`

### Appendix B: Coverage Reports

- HTML report: \`htmlcov/index.html\`
- JSON data: \`coverage.json\`
- Analysis: \`docs/coverage_analysis_report.md\`
- Gaps: \`logs/coverage_gaps.json\`

### Appendix C: Environment Snapshot

- Python version: $(python --version 2>&1)
- PyTorch version: $(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Not available")
- CUDA version: $(python -c "import torch; print('CUDA', torch.version.cuda)" 2>/dev/null || echo "Not available")
- GPU: $(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Not available")
- Conda environment: ${CONDA_DEFAULT_ENV:-Not set}

### Appendix D: Summary Files

- JSON summary: \`logs/phase2_summary_${TIMESTAMP}.json\`
- Plain text summary: \`logs/phase2_summary_${TIMESTAMP}.txt\`

---

**Report Generated**: $(date +"%Y-%m-%d %H:%M:%S")
**Generated By**: scripts/generate_phase2_report.sh
**Phase 2 Status**: ${OVERALL_STATUS}
**Ready for Phase 3**: $([[ $TOTAL_FAILED -eq 0 && $CRITICAL_FAILURES -eq 0 && $(py_compare "$COVERAGE_PERCENT" ">=" "80" 2>/dev/null || echo "0") -eq 1 ]] && echo "Yes" || echo "No")
EOF

# Generate JSON summary
echo "Generating JSON summary..."
cat > "${LOG_DIR}/phase2_summary_${TIMESTAMP}.json" << EOF
{
  "timestamp": "${TIMESTAMP}",
  "duration_seconds": ${TOTAL_DURATION},
  "total_tests": ${TOTAL_TESTS},
  "passed": ${TOTAL_PASSED},
  "failed": ${TOTAL_FAILED},
  "skipped": ${TOTAL_SKIPPED},
  "pass_rate": ${PASS_RATE},
  "coverage_percent": ${COVERAGE_PERCENT},
  "critical_failures": ${CRITICAL_FAILURES},
  "status": "${OVERALL_STATUS,,}",
  "smoke_tests": {
    "passed": ${SMOKE_PASSED},
    "failed": ${SMOKE_FAILED},
    "skipped": ${SMOKE_SKIPPED}
  },
  "integration_tests": {
    "passed": ${INTEGRATION_PASSED},
    "failed": ${INTEGRATION_FAILED},
    "skipped": ${INTEGRATION_SKIPPED}
  },
  "coverage": {
    "modules_above_80": ${MODULES_ABOVE_80},
    "modules_below_80": ${MODULES_BELOW_80}
  }
}
EOF

# Generate plain text summary
echo "Generating plain text summary..."
cat > "${LOG_DIR}/phase2_summary_${TIMESTAMP}.txt" << EOF
Phase 2: ${TOTAL_TESTS}/${TOTAL_PASSED} tests passed (${PASS_RATE}% pass rate, ${COVERAGE_PERCENT}% coverage)
Status: ${OVERALL_STATUS}
Critical failures: ${CRITICAL_FAILURES}
EOF

if [[ $TOTAL_FAILED -gt 0 ]]; then
    echo "Failed tests: ${TOTAL_FAILED}" >> "${LOG_DIR}/phase2_summary_${TIMESTAMP}.txt"
fi

echo ""
echo "Phase 2 report generated: $REPORT_PATH"
echo "JSON summary: ${LOG_DIR}/phase2_summary_${TIMESTAMP}.json"
echo "Plain text summary: ${LOG_DIR}/phase2_summary_${TIMESTAMP}.txt"
