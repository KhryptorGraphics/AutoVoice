#!/bin/bash
# Full Validation Orchestrator Script
# Coordinates all validation tests and generates comprehensive report

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/validation_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="$RESULTS_DIR/validation_summary_${TIMESTAMP}.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Exit codes
EXIT_SUCCESS=0
EXIT_FAILURE=1

# Validation status tracking
TOTAL_VALIDATIONS=0
PASSED_VALIDATIONS=0
FAILED_VALIDATIONS=0
SKIPPED_VALIDATIONS=0

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$SUMMARY_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$SUMMARY_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$SUMMARY_FILE"
}

log_section() {
    echo "" | tee -a "$SUMMARY_FILE"
    echo "========================================" | tee -a "$SUMMARY_FILE"
    echo "$1" | tee -a "$SUMMARY_FILE"
    echo "========================================" | tee -a "$SUMMARY_FILE"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1" | tee -a "$SUMMARY_FILE"
}

log_failure() {
    echo -e "${RED}✗${NC} $1" | tee -a "$SUMMARY_FILE"
}

log_skip() {
    echo -e "${YELLOW}⊘${NC} $1" | tee -a "$SUMMARY_FILE"
}

# Run validation step
run_validation() {
    local name="$1"
    local command="$2"
    local optional="${3:-false}"

    TOTAL_VALIDATIONS=$((TOTAL_VALIDATIONS + 1))

    log_section "$name"
    log_info "Running: $command"

    local start_time=$(date +%s)

    if eval "$command" 2>&1 | tee -a "$SUMMARY_FILE"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        PASSED_VALIDATIONS=$((PASSED_VALIDATIONS + 1))
        log_success "$name completed successfully (${duration}s)"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))

        if [ "$optional" = true ]; then
            SKIPPED_VALIDATIONS=$((SKIPPED_VALIDATIONS + 1))
            log_skip "$name failed but is optional (${duration}s)"
            return 0
        else
            FAILED_VALIDATIONS=$((FAILED_VALIDATIONS + 1))
            log_failure "$name failed (${duration}s)"
            return 1
        fi
    fi
}

# Main validation orchestration
main() {
    # Initialize
    log_section "Full Validation Suite - $(date)"
    log_info "Project: AutoVoice"
    log_info "Root: $PROJECT_ROOT"
    log_info "Results: $RESULTS_DIR"

    # Create validation results directory
    log_info "Creating validation results directory"
    mkdir -p "$RESULTS_DIR"

    # Change to project root
    cd "$PROJECT_ROOT"

    # Check Python environment
    log_section "Environment Check"
    log_info "Python version: $(python --version 2>&1)"
    log_info "Working directory: $(pwd)"

    if [ -d "venv" ]; then
        log_info "Virtual environment detected"
    else
        log_warn "No virtual environment detected - using system Python"
    fi

    # Check for GPU
    GPU_AVAILABLE=false
    if command -v nvidia-smi &> /dev/null; then
        log_info "NVIDIA GPU detected"
        GPU_AVAILABLE=true
    else
        log_warn "No NVIDIA GPU detected - GPU tests will be skipped"
    fi

    # Generate test data
    if [ -f "tests/data/validation/generate_test_data.py" ]; then
        run_validation \
            "Test Data Generation" \
            "python tests/data/validation/generate_test_data.py" \
            false || true
    else
        log_skip "Test data generation script not found"
        SKIPPED_VALIDATIONS=$((SKIPPED_VALIDATIONS + 1))
    fi

    # Run system validation tests
    if [ -f "tests/test_system_validation.py" ]; then
        run_validation \
            "System Validation Tests" \
            "pytest tests/test_system_validation.py -v --tb=short" \
            false || true
    else
        log_skip "System validation tests not found"
        SKIPPED_VALIDATIONS=$((SKIPPED_VALIDATIONS + 1))
    fi

    # Run code quality validation
    if [ -f "scripts/validate_code_quality.py" ]; then
        run_validation \
            "Code Quality Validation" \
            "python scripts/validate_code_quality.py" \
            false || true
    else
        log_skip "Code quality validation script not found"
        SKIPPED_VALIDATIONS=$((SKIPPED_VALIDATIONS + 1))
    fi

    # Run integration validation
    if [ -f "scripts/validate_integration.py" ]; then
        run_validation \
            "Integration Validation" \
            "python scripts/validate_integration.py" \
            false || true
    else
        log_skip "Integration validation script not found"
        SKIPPED_VALIDATIONS=$((SKIPPED_VALIDATIONS + 1))
    fi

    # Run documentation validation
    if [ -f "scripts/validate_documentation.py" ]; then
        run_validation \
            "Documentation Validation" \
            "python scripts/validate_documentation.py" \
            false || true
    else
        log_skip "Documentation validation script not found"
        SKIPPED_VALIDATIONS=$((SKIPPED_VALIDATIONS + 1))
    fi

    # Run Docker validation (optional if no GPU)
    if [ -f "scripts/test_docker_deployment.sh" ]; then
        if [ "$GPU_AVAILABLE" = false ]; then
            run_validation \
                "Docker Deployment Validation" \
                "bash scripts/test_docker_deployment.sh" \
                true || true
        else
            run_validation \
                "Docker Deployment Validation" \
                "bash scripts/test_docker_deployment.sh" \
                false || true
        fi
    else
        log_skip "Docker validation script not found"
        SKIPPED_VALIDATIONS=$((SKIPPED_VALIDATIONS + 1))
    fi

    # Run quality evaluation if available
    if [ -f "examples/evaluate_voice_conversion.py" ]; then
        run_validation \
            "Voice Quality Evaluation" \
            "python examples/evaluate_voice_conversion.py --quick" \
            true || true
    else
        log_skip "Quality evaluation script not found"
        SKIPPED_VALIDATIONS=$((SKIPPED_VALIDATIONS + 1))
    fi

    # Generate final validation report
    if [ -f "scripts/generate_validation_report.py" ]; then
        run_validation \
            "Validation Report Generation" \
            "python scripts/generate_validation_report.py" \
            false || true
    else
        log_skip "Validation report generator not found"
        SKIPPED_VALIDATIONS=$((SKIPPED_VALIDATIONS + 1))
    fi

    # Final summary
    log_section "Validation Suite Summary"
    log_info "Total Validations: $TOTAL_VALIDATIONS"
    log_success "Passed: $PASSED_VALIDATIONS"
    log_failure "Failed: $FAILED_VALIDATIONS"
    log_skip "Skipped: $SKIPPED_VALIDATIONS"

    local success_rate=0
    if [ $TOTAL_VALIDATIONS -gt 0 ]; then
        success_rate=$((PASSED_VALIDATIONS * 100 / TOTAL_VALIDATIONS))
    fi

    log_info "Success Rate: ${success_rate}%"
    log_info "Summary saved to: $SUMMARY_FILE"

    # Determine exit code
    if [ $FAILED_VALIDATIONS -eq 0 ]; then
        log_section "VALIDATION SUITE PASSED"
        return $EXIT_SUCCESS
    else
        log_section "VALIDATION SUITE FAILED"
        return $EXIT_FAILURE
    fi
}

# Run main function
main
exit $?
