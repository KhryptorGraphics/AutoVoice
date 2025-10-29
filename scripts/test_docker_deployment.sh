#!/bin/bash
# Docker Deployment Validation Script
# Validates Docker image build, container startup, and API functionality

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_NAME="autovoice:validation"
CONTAINER_NAME="autovoice_validation_$$"
API_PORT=5000
STARTUP_TIMEOUT=60
LOG_FILE="validation_results/docker_validation.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_section() {
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "$1" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    if [ ! -z "${CONTAINER_ID:-}" ]; then
        docker stop $CONTAINER_ID >/dev/null 2>&1 || true
        docker rm $CONTAINER_ID >/dev/null 2>&1 || true
        log_info "Container removed"
    fi
}

# Set trap for cleanup
trap cleanup EXIT

# Create validation results directory
mkdir -p "$(dirname "$LOG_FILE")"

# Initialize log file
log_section "Docker Deployment Validation - $(date)"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed or not in PATH"
    exit 1
fi

log_info "Docker version: $(docker --version)"

# Check if Dockerfile exists
if [ ! -f "$PROJECT_ROOT/Dockerfile" ]; then
    log_error "Dockerfile not found at $PROJECT_ROOT/Dockerfile"
    exit 1
fi

# Check for GPU support
GPU_AVAILABLE=false
if command -v nvidia-smi &> /dev/null; then
    log_info "NVIDIA GPU detected"
    GPU_AVAILABLE=true
    log_info "GPU Info:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee -a "$LOG_FILE"
    GPU_FLAGS="--gpus all"
else
    log_warn "No NVIDIA GPU detected - skipping GPU-specific tests"
    GPU_FLAGS=""
fi

# Build Docker image
log_section "Building Docker Image"
log_info "Building image: $IMAGE_NAME"
cd "$PROJECT_ROOT"

if ! docker build -t "$IMAGE_NAME" -f "$PROJECT_ROOT/Dockerfile" "$PROJECT_ROOT" 2>&1 | tee -a "$LOG_FILE"; then
    log_error "Docker build failed"
    exit 1
fi

log_info "Docker image built successfully"
IMAGE_SIZE=$(docker images "$IMAGE_NAME" --format "{{.Size}}")
log_info "Image size: $IMAGE_SIZE"

# Run container
log_section "Starting Container"

if [ "$GPU_AVAILABLE" = true ]; then
    log_info "Starting container with GPU support"
else
    log_info "Starting container without GPU support"
fi

CONTAINER_ID=$(docker run -d $GPU_FLAGS -p $API_PORT:5000 \
    -e LOG_LEVEL=INFO \
    -e FLASK_ENV=production \
    --name $CONTAINER_NAME \
    $IMAGE_NAME 2>&1 | tee -a "$LOG_FILE")

if [ -z "$CONTAINER_ID" ]; then
    log_error "Failed to start container"
    exit 1
fi

log_info "Container started: $CONTAINER_NAME (ID: $CONTAINER_ID)"

# Wait for container startup
log_section "Waiting for Service Startup"
log_info "Waiting up to $STARTUP_TIMEOUT seconds for service to be ready..."

ELAPSED=0
READY=false

while [ $ELAPSED -lt $STARTUP_TIMEOUT ]; do
    if curl -sf "http://localhost:$API_PORT/health" >/dev/null 2>&1; then
        READY=true
        break
    fi

    # Check if container is still running
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        log_error "Container stopped unexpectedly"
        log_info "Container logs:"
        docker logs "$CONTAINER_ID" 2>&1 | tail -n 50 | tee -a "$LOG_FILE"
        exit 1
    fi

    sleep 2
    ELAPSED=$((ELAPSED + 2))
    echo -n "." | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"

if [ "$READY" = false ]; then
    log_error "Service failed to start within $STARTUP_TIMEOUT seconds"
    log_info "Container logs:"
    docker logs "$CONTAINER_ID" 2>&1 | tail -n 50 | tee -a "$LOG_FILE"
    exit 1
fi

log_info "Service is ready (startup time: ${ELAPSED}s)"

# Test health endpoints
log_section "Testing Health Endpoints"

# Test /health (mandatory)
log_info "Testing GET /health"
if ! HEALTH_RESPONSE=$(curl -sf "http://localhost:$API_PORT/health" 2>&1); then
    log_error "Health check failed"
    echo "$HEALTH_RESPONSE" | tee -a "$LOG_FILE"
    exit 1
fi
echo "$HEALTH_RESPONSE" | tee -a "$LOG_FILE"
log_info "✓ /health endpoint OK"

# Test /health/live (optional)
log_info "Testing GET /health/live (optional)"
if LIVE_RESPONSE=$(curl -sf "http://localhost:$API_PORT/health/live" 2>&1); then
    echo "$LIVE_RESPONSE" | tee -a "$LOG_FILE"
    log_info "✓ /health/live endpoint OK"
else
    log_warn "/health/live endpoint not available (optional)"
fi

# Test /health/ready (optional)
log_info "Testing GET /health/ready (optional)"
if READY_RESPONSE=$(curl -sf "http://localhost:$API_PORT/health/ready" 2>&1); then
    echo "$READY_RESPONSE" | tee -a "$LOG_FILE"
    log_info "✓ /health/ready endpoint OK"
else
    log_warn "/health/ready endpoint not available (optional)"
fi

# Test GPU status endpoint
if [ "$GPU_AVAILABLE" = true ]; then
    log_section "Testing GPU Status Endpoint"
    log_info "Testing GET /api/v1/gpu_status"

    if ! GPU_STATUS=$(curl -sf "http://localhost:$API_PORT/api/v1/gpu_status" 2>&1); then
        log_error "GPU status check failed"
        echo "$GPU_STATUS" | tee -a "$LOG_FILE"
        exit 1
    fi

    echo "$GPU_STATUS" | tee -a "$LOG_FILE"

    # Verify CUDA is available
    if echo "$GPU_STATUS" | grep -q '"cuda_available".*true'; then
        log_info "✓ CUDA is available in container"
    else
        log_error "CUDA is not available in container"
        exit 1
    fi
fi

# Test API endpoints
log_section "Testing API Endpoints"

log_info "Testing GET /api/v1/voice/profiles"
if ! PROFILES_RESPONSE=$(curl -sf "http://localhost:$API_PORT/api/v1/voice/profiles" 2>&1); then
    log_error "Voice profiles endpoint failed"
    echo "$PROFILES_RESPONSE" | tee -a "$LOG_FILE"
    exit 1
fi

echo "$PROFILES_RESPONSE" | tee -a "$LOG_FILE"
log_info "✓ /api/v1/voice/profiles endpoint OK"

# Check GPU utilization via nvidia-smi in container
if [ "$GPU_AVAILABLE" = true ]; then
    log_section "GPU Utilization Check"
    log_info "Executing nvidia-smi in container"

    if docker exec "$CONTAINER_ID" nvidia-smi 2>&1 | tee -a "$LOG_FILE"; then
        log_info "✓ nvidia-smi executed successfully in container"

        log_info "GPU Memory Usage:"
        docker exec "$CONTAINER_ID" nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>&1 | tee -a "$LOG_FILE"

        log_info "GPU Utilization:"
        docker exec "$CONTAINER_ID" nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>&1 | tee -a "$LOG_FILE"
    else
        log_warn "Could not execute nvidia-smi in container"
    fi
fi

# Check for errors in logs
log_section "Container Log Analysis"
log_info "Checking for errors in container logs"

ERROR_LINES=$(docker logs "$CONTAINER_ID" 2>&1 | grep -i "error" | head -n 10 || true)

if [ -n "$ERROR_LINES" ]; then
    log_warn "Found error messages in logs:"
    echo "$ERROR_LINES" | tee -a "$LOG_FILE"
else
    log_info "No error messages found in logs"
fi

log_info "First 10 lines of container logs:"
docker logs "$CONTAINER_ID" 2>&1 | head -n 10 | tee -a "$LOG_FILE"

log_info "Last 10 lines of container logs:"
docker logs "$CONTAINER_ID" 2>&1 | tail -n 10 | tee -a "$LOG_FILE"

# Final summary
log_section "Validation Summary"
log_info "All Docker deployment validations passed successfully"
log_info "Image: $IMAGE_NAME"
log_info "Container: $CONTAINER_NAME"
log_info "Port: $API_PORT"
log_info "GPU Support: $GPU_AVAILABLE"
log_info "Results saved to: $LOG_FILE"

exit 0
