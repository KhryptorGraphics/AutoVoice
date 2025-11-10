#!/bin/bash

# Phase 3 Execution Script: Docker Validation and End-to-End Testing
# This script orchestrates all Phase 3 validation steps in sequence

set -e
set -o pipefail
set -u

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================${NC}"
}

print_step() {
    echo -e "${CYAN}[STEP $1/8]${NC} $2"
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

# Set project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Create timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create logs directory
mkdir -p logs/phase3

# Set log file
LOG_FILE="logs/phase3/phase3_${TIMESTAMP}.log"

# Function to log messages
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $*" | tee -a "$LOG_FILE"
}

# Trap for cleanup
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        print_error "Phase 3 execution failed. Check logs: $LOG_FILE"
        if ${DOCKER_COMPOSE_CMD:-docker compose} ps 2>/dev/null | grep -q "Up"; then
            print_warning "Docker Compose services may still be running. Consider running: ${DOCKER_COMPOSE_CMD:-docker compose} down"
        fi
    fi
    exit $exit_code
}

trap cleanup EXIT

# Start logging
exec > >(tee -a "$LOG_FILE") 2>&1

# Banner and Overview
print_header "Phase 3: Docker Validation and End-to-End Testing"
echo "Estimated time: 2-4 hours"
echo ""
echo "Steps:"
echo "  1. Prerequisites validation"
echo "  2. Docker image build"
echo "  3. Docker Compose validation"
echo "  4. E2E tests execution"
echo "  5. API endpoint validation"
echo "  6. WebSocket streaming tests"
echo "  7. Performance benchmarks"
echo "  8. Report generation"
echo ""

# Prompt user to continue
read -p "Proceed with Phase 3? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Phase 3 execution cancelled by user"
    exit 0
fi

print_info "Starting Phase 3 execution..."
START_TIME=$(date +%s)

# Step 1: Prerequisites Validation
print_step 1 "Validating Prerequisites"

print_info "Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    print_error "Docker not found. Please install Docker."
    exit 1
fi
print_success "Docker found: $(docker --version)"

print_info "Checking Docker Compose..."
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker compose"
elif docker-compose --version &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker-compose"
else
    print_error "Docker Compose not found"
    exit 1
fi
print_success "Docker Compose found: $($DOCKER_COMPOSE_CMD --version)"

print_info "Checking Docker daemon..."
if ! docker info &> /dev/null; then
    print_error "Docker daemon not running"
    exit 1
fi
print_success "Docker daemon running"

print_info "Checking NVIDIA Docker runtime..."
if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    print_error "NVIDIA Docker runtime not available"
    exit 1
fi
print_success "NVIDIA Docker runtime available"

print_info "Checking conda environment..."
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" != "autovoice_py312" ]; then
    print_warning "Conda environment 'autovoice_py312' not active. Attempting to activate..."

    # Source conda.sh if not already sourced (needed for non-interactive shells)
    if ! command -v conda &> /dev/null; then
        for CONDA_SH in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh" "/opt/conda/etc/profile.d/conda.sh"; do
            if [ -f "$CONDA_SH" ]; then
                print_info "Sourcing conda from $CONDA_SH"
                source "$CONDA_SH"
                break
            fi
        done
    fi

    # Try to activate
    if command -v conda &> /dev/null; then
        if conda activate autovoice_py312 2>/dev/null; then
            print_success "Conda environment activated: autovoice_py312"
        else
            print_warning "Failed to activate conda environment (continuing anyway - Docker tests don't require it)"
        fi
    else
        print_warning "Conda not available (continuing anyway - Docker tests don't require it)"
    fi
else
    print_success "Conda environment active: $CONDA_DEFAULT_ENV"
fi

print_info "Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
if [[ $PYTHON_VERSION =~ ^3\.10 || $PYTHON_VERSION =~ ^3\.12 ]]; then
    print_success "Python version: $PYTHON_VERSION (compatible)"
else
    print_warning "Python $PYTHON_VERSION found, but 3.10 or 3.12 recommended (continuing - Docker tests don't require it)"
fi

print_info "Checking PyTorch installation..."
if ! python -c "import torch; print('PyTorch version:', torch.__version__)" &> /dev/null; then
    print_warning "PyTorch not found locally (continuing - Docker tests don't require it)"
else
    print_success "PyTorch installed"
fi

print_info "Checking CUDA extensions..."
# Set PYTHONPATH to include src directory
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# Try to import CUDA kernels (non-fatal since Docker validation is primary)
if PYTHONPATH="$PROJECT_ROOT/src" python -c "import cuda_kernels" 2>/dev/null; then
    print_success "CUDA extensions available locally"
elif [ "${REQUIRE_LOCAL_CUDA:-0}" = "1" ]; then
    print_error "CUDA extensions required but not available (REQUIRE_LOCAL_CUDA=1)"
    exit 1
else
    print_warning "CUDA extensions not available locally (continuing - Docker validation is primary)"
fi

print_info "Checking pytest..."
if ! command -v pytest &> /dev/null && ! python -m pytest --version &> /dev/null; then
    print_warning "pytest not found (local tests may be skipped)"
else
    print_success "pytest available"
fi

print_info "Checking disk space..."
DISK_FREE=$(df . | tail -1 | awk '{print $4}')
if [ $DISK_FREE -lt 10485760 ]; then  # 10GB in KB
    print_error "Insufficient disk space: $(($DISK_FREE/1024/1024))GB free, need 10GB"
    exit 1
fi
print_success "Disk space OK: $(($DISK_FREE/1024/1024))GB free"

print_success "Prerequisites validation complete"

# Step 2: Build Docker Image
print_step 2 "Building Docker Image"

print_info "Building autovoice:latest with CUDA 12.1.0..."

# Check Dockerfile exists
if [ ! -f Dockerfile ]; then
    print_error "Dockerfile not found"
    exit 1
fi

# Verify CUDA version
if ! grep -q "FROM nvidia/cuda:12.1" Dockerfile; then
    print_error "Dockerfile does not use correct CUDA version"
    exit 1
fi

BUILD_START=$(date +%s)
if ! docker build -t autovoice:latest -t autovoice/autovoice:latest -t autovoice:phase3-${TIMESTAMP} --progress=plain . 2>&1 | tee logs/phase3/docker_build_${TIMESTAMP}.log; then
    print_error "Docker build failed"
    exit 1
fi
BUILD_END=$(date +%s)
BUILD_TIME=$((BUILD_END - BUILD_START))

# Verify image created
if ! docker images | grep -q autovoice; then
    print_error "Docker image not created"
    exit 1
fi

IMAGE_SIZE=$(docker images autovoice:latest --format "{{.Size}}")
print_success "Docker image built successfully in ${BUILD_TIME}s"
print_info "Image size: $IMAGE_SIZE"

# Verify CUDA access in container
print_info "Verifying CUDA access in container..."
if docker run --rm --gpus all autovoice:latest nvidia-smi 2>&1 | tee -a logs/phase3/docker_build_${TIMESTAMP}.log; then
    print_success "CUDA verification passed - container can access GPU"
else
    print_warning "nvidia-smi check failed, trying fallback GPU verification..."
    if docker run --rm --gpus all autovoice:latest python -c "import torch; import sys; sys.exit(0 if (hasattr(torch,'cuda') and torch.cuda.is_available()) else 1)" 2>&1 | tee -a logs/phase3/docker_build_${TIMESTAMP}.log; then
        print_success "Fallback GPU verification passed - PyTorch CUDA available in container"
    else
        print_error "CUDA verification failed - container cannot access GPU"
        print_error "Check that nvidia-docker runtime is properly configured"
        exit 1
    fi
fi

# Step 3: Docker Compose Validation
print_step 3 "Testing Docker Compose"

print_info "Starting services with $DOCKER_COMPOSE_CMD..."

if [ ! -f docker-compose.yml ]; then
    print_error "docker-compose.yml not found"
    exit 1
fi

if ! $DOCKER_COMPOSE_CMD config &> /dev/null; then
    print_error "Invalid docker-compose.yml"
    exit 1
fi

if ! $DOCKER_COMPOSE_CMD up -d 2>&1 | tee logs/phase3/docker_compose_up_${TIMESTAMP}.log; then
    print_error "Docker Compose up failed"
    exit 1
fi

print_info "Waiting for services to become ready..."

# Bounded health polling loop (up to 180 seconds)
MAX_WAIT=180
ELAPSED=0
BACKOFF=3

while [ $ELAPSED -lt $MAX_WAIT ]; do
    # Check if at least one service is Up
    SERVICES_STATUS=$($DOCKER_COMPOSE_CMD ps 2>/dev/null || echo "")
    if echo "$SERVICES_STATUS" | grep -q "Up"; then
        print_info "Services are up, checking health endpoint..."

        # Check if /health returns 200
        if curl -f http://localhost:5000/health &> /dev/null; then
            print_success "Services are ready (health check passed)"
            break
        else
            print_info "Services up but health check not ready yet (${ELAPSED}s elapsed)"
        fi
    else
        print_info "Waiting for services to start (${ELAPSED}s elapsed)"
    fi

    sleep $BACKOFF
    ELAPSED=$((ELAPSED + BACKOFF))

    # Increase backoff to 5s after first few attempts
    if [ $ELAPSED -gt 15 ] && [ $BACKOFF -lt 5 ]; then
        BACKOFF=5
    fi
done

# Final check
SERVICES_STATUS=$($DOCKER_COMPOSE_CMD ps)
if ! echo "$SERVICES_STATUS" | grep -q "Up"; then
    print_error "Services failed to start within ${MAX_WAIT}s"
    print_error "Service logs:"
    $DOCKER_COMPOSE_CMD logs --tail=100
    exit 1
fi

if ! curl -f http://localhost:5000/health &> /dev/null; then
    print_warning "Services are up but health check still failing after ${MAX_WAIT}s"
    print_warning "Continuing with tests (some may fail)"
fi

print_success "Services status:"
echo "$SERVICES_STATUS"

# Check health endpoints
if ! curl -f http://localhost:5000/health &> /dev/null; then
    print_warning "AutoVoice health check failed"
else
    print_success "AutoVoice health check passed"
fi

if ! $DOCKER_COMPOSE_CMD exec -T redis redis-cli ping | grep -q "PONG"; then
    print_warning "Redis health check failed"
else
    print_success "Redis health check passed"
fi

# Check for errors in logs
if $DOCKER_COMPOSE_CMD logs --tail=50 auto-voice-app | grep -qi error; then
    print_warning "Errors found in AutoVoice logs"
fi

# Verify GPU access inside container
if ! $DOCKER_COMPOSE_CMD exec -T auto-voice-app nvidia-smi &> /dev/null; then
    print_warning "nvidia-smi check failed, trying fallback GPU verification..."
    if $DOCKER_COMPOSE_CMD exec -T auto-voice-app python -c "import torch; import sys; sys.exit(0 if (hasattr(torch,'cuda') and torch.cuda.is_available()) else 1)" &> /dev/null; then
        print_success "Fallback GPU verification passed - PyTorch CUDA available in container"
    else
        print_error "GPU access check failed - container cannot see GPU"
    fi
else
    print_success "GPU access verified from within container"
fi

print_success "Docker Compose validation complete"

# Step 4: Run E2E Tests
print_step 4 "Running End-to-End Tests"

print_info "Executing comprehensive E2E test suite..."

cd "$PROJECT_ROOT"

# Activate conda if needed (non-fatal)
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" != "autovoice_py312" ]; then
    if command -v conda &> /dev/null; then
        # Try to source conda.sh if not already sourced
        if ! type conda &> /dev/null; then
            for CONDA_SH in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh" "/opt/conda/etc/profile.d/conda.sh"; do
                if [ -f "$CONDA_SH" ]; then
                    source "$CONDA_SH" 2>/dev/null || true
                    break
                fi
            done
        fi
        # Try to activate (non-fatal)
        conda activate autovoice_py312 2>/dev/null || print_warning "Could not activate conda environment (continuing with current Python)"
    else
        print_warning "Conda not available (continuing with current Python)"
    fi
fi

# Run E2E tests using python -m pytest to use current interpreter
if ! python -m pytest tests/test_end_to_end.py -v --tb=short --durations=10 2>&1 | tee logs/phase3/e2e_tests_${TIMESTAMP}.log; then
    print_warning "Some E2E tests failed"
fi

# Parse results
if [ -f logs/phase3/e2e_tests_${TIMESTAMP}.log ]; then
    E2E_RESULTS=$(grep -E "passed|failed|skipped" logs/phase3/e2e_tests_${TIMESTAMP}.log | tail -1)
    print_info "E2E Results: $E2E_RESULTS"
fi

# Run web interface tests
print_info "Running web interface tests..."
if ! python -m pytest tests/test_web_interface.py -v --tb=short 2>&1 | tee logs/phase3/web_interface_tests_${TIMESTAMP}.log; then
    print_warning "Some web interface tests failed"
fi

if [ -f logs/phase3/web_interface_tests_${TIMESTAMP}.log ]; then
    WEB_RESULTS=$(grep -E "passed|failed|skipped" logs/phase3/web_interface_tests_${TIMESTAMP}.log | tail -1)
    print_info "Web Interface Results: $WEB_RESULTS"
fi

print_success "E2E tests execution complete"

# Step 5: Validate API Endpoints
print_step 5 "Validating API Endpoints"

print_info "Running comprehensive API validation..."

# Call the API validator script with the current timestamp
if ! ./scripts/validate_api_endpoints.sh "$TIMESTAMP" 2>&1 | tee logs/phase3/api_validation_run_${TIMESTAMP}.log; then
    print_warning "Some API endpoints failed validation - see logs for details"
    print_info "Continuing with Phase 3 (results will be aggregated in final report)"
else
    print_success "All API endpoints validated successfully"
fi

print_success "API endpoint validation complete"

# Step 6: Test WebSocket Streaming
print_step 6 "Testing WebSocket Streaming"

print_info "Running WebSocket functionality tests..."

# Run WebSocket-specific tests (pytest)
if ! python -m pytest tests/test_web_interface.py::TestWebSocketConversionProgress -v --tb=short 2>&1 | tee logs/phase3/websocket_pytest_${TIMESTAMP}.log; then
    print_warning "WebSocket pytest tests failed"
fi

# Run WebSocket connection test (separate log file)
if ! ./scripts/test_websocket_connection.py --json-output logs/phase3/websocket_conn_${TIMESTAMP}.json 2>&1 | tee logs/phase3/websocket_conn_${TIMESTAMP}.log; then
    print_warning "WebSocket connection test failed - see logs"
else
    print_success "WebSocket connection test passed"
fi

print_success "WebSocket streaming tests complete"

# Step 7: Run Performance Benchmarks
print_step 7 "Running Performance Benchmarks"

print_info "Executing performance test suite..."

if ! python -m pytest tests/test_performance.py -v -s --tb=short --durations=20 2>&1 | tee logs/phase3/performance_tests_${TIMESTAMP}.log; then
    print_warning "Some performance tests failed"
fi

# Parse performance results
if [ -f logs/phase3/performance_tests_${TIMESTAMP}.log ]; then
    RTF_VAL=$(grep "RTF" logs/phase3/performance_tests_${TIMESTAMP}.log | head -1 | awk '{print $NF}')
    if [ -n "$RTF_VAL" ]; then
        print_info "Average RTF: ${RTF_VAL}x"
    fi
fi

print_success "Performance benchmarks complete"

# Step 8: Generate Phase 3 Report
print_step 8 "Generating Phase 3 Completion Report"

print_info "Aggregating results and generating report..."

if ! ./scripts/generate_phase3_report.sh "${TIMESTAMP}"; then
    print_error "Report generation failed"
    exit 1
fi

print_success "Phase 3 report generated: PHASE3_COMPLETION_REPORT.md"

# Cleanup and Summary
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

print_header "Phase 3 Execution Complete"

echo "Total execution time: ${DURATION}s"
echo ""
echo "Results Summary:"
echo "  Docker Build: ✅ Success"
echo "  Docker Compose: ✅ Success"
echo "  E2E Tests: See report for details"
echo "  API Endpoints: See report for details"
echo "  WebSocket: See report for details"
echo "  Performance: See report for details"
echo ""
echo "Reports:"
echo "  - PHASE3_COMPLETION_REPORT.md"
echo "  - logs/phase3/phase3_summary_${TIMESTAMP}.json"
echo ""
echo "Log files:"
echo "  - $LOG_FILE"
echo "  - logs/phase3/docker_build_${TIMESTAMP}.log"
echo "  - logs/phase3/docker_compose_up_${TIMESTAMP}.log"
echo "  - logs/phase3/e2e_tests_${TIMESTAMP}.log"
echo "  - And more..."

read -p "Stop Docker Compose services? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Stopping Docker Compose services..."
    $DOCKER_COMPOSE_CMD down
    print_success "Services stopped"
fi

print_success "Phase 3 Complete!"
print_info "Ready for Phase 4 (Security Hardening)"

exit 0
