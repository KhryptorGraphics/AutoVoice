#!/bin/bash
# Health Check Endpoint Validation Script
# Purpose: Automated testing of all health check endpoints
# Usage: ./scripts/validate_health_checks.sh [base_url]

# Configuration
BASE_URL=${1:-http://localhost:5000}
TIMEOUT=10
VERBOSE=${VERBOSE:-false}
MAX_RETRIES=5
RETRY_DELAY=2  # Initial delay in seconds (exponential backoff)

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

result_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}PASS${NC}"
    else
        echo -e "${RED}FAIL${NC}"
    fi
}

# Retry helper function with exponential backoff
retry_with_backoff() {
    local endpoint=$1
    local max_retries=$MAX_RETRIES
    local delay=$RETRY_DELAY
    local attempt=1

    while [ $attempt -le $max_retries ]; do
        if [ $attempt -gt 1 ]; then
            print_info "Retry attempt $attempt/$max_retries for $endpoint (waiting ${delay}s)..."
            sleep $delay
            delay=$((delay * 2))  # Exponential backoff
        fi

        # Try the endpoint
        local status_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time $TIMEOUT "${BASE_URL}${endpoint}" 2>/dev/null || echo "000")

        if [ "$status_code" = "200" ] || [ "$status_code" = "503" ]; then
            return 0  # Success
        fi

        attempt=$((attempt + 1))
    done

    return 1  # All retries failed
}

# Test functions
test_health_endpoint() {
    local response
    local status_code
    local headers
    local start_time
    local end_time
    local duration

    # Retry with exponential backoff before actual test
    if ! retry_with_backoff "/health"; then
        print_error "Health endpoint not responding after $MAX_RETRIES retries"
        return 1
    fi

    # Capture timing using curl's built-in timing
    local timing_output
    start_time=$(python -c "import time; print(int(time.monotonic() * 1000))" 2>/dev/null || date +%s%3N)

    # Create temp files for headers and body
    local tmp_headers=$(mktemp)
    local tmp_body=$(mktemp)

    # Capture headers and body separately with timing
    timing_output=$(curl -s -w "@-" -D "$tmp_headers" -o "$tmp_body" --max-time $TIMEOUT "${BASE_URL}/health" <<'EOF'
{
  "time_total": "%{time_total}",
  "http_code": "%{http_code}",
  "exitcode": "%{exitcode}"
}
EOF
2>/dev/null || echo '{"time_total": "0.000", "http_code": "000", "exitcode": "1"}')

    status_code=$(echo "$timing_output" | grep -o '"http_code": "[^"]*' | cut -d'"' -f4)
    local body=$(cat "$tmp_body")
    headers=$(cat "$tmp_headers")
    local curl_time=$(echo "$timing_output" | grep -o '"time_total": "[^"]*' | cut -d'"' -f4)
    end_time=$(python -c "import time; print(int(time.monotonic() * 1000))" 2>/dev/null || date +%s%3N)

    # Calculate duration in milliseconds
    if [[ "$curl_time" =~ ^[0-9]*\.?[0-9]+$ ]]; then
        duration=$(echo "scale=0; $curl_time * 1000 / 1" | bc 2>/dev/null || echo "1000")
    else
        duration=$(python -c "import time; print(int(time.monotonic() * 1000 - $start_time))" 2>/dev/null || echo "1000")
    fi

    # Clean up temp files
    rm -f "$tmp_headers" "$tmp_body"

    # Check status code
    if [ "$status_code" != "200" ]; then
        print_error "Status: $status_code (expected 200)"
        return 1
    fi
    print_success "Status: 200 OK"

    # Check response time threshold (1000ms for health endpoint)
    if [ $duration -gt 1000 ]; then
        print_error "Response time: ${duration}ms (exceeds 1000ms threshold)"
        return 1
    fi
    print_success "Response time: ${duration}ms"

    # Check Content-Type header
    local content_type=$(echo "$headers" | grep -i "^content-type:" | awk '{print $2}' | tr -d '\r\n')
    if [[ ! "$content_type" =~ application/json ]]; then
        print_error "Content-Type: $content_type (expected application/json)"
        return 1
    fi
    print_success "Content-Type: $content_type"

    # Validate JSON schema
    if command -v jq &> /dev/null; then
        # Check required fields exist
        local status=$(echo "$body" | jq -r '.status // empty')
        if [ -z "$status" ]; then
            print_error "Missing required field: .status"
            return 1
        fi
        if [ "$(echo "$body" | jq -r 'type')" != "object" ]; then
            print_error "Response is not a JSON object"
            return 1
        fi
        print_success "Service status: $status"

        # Check GPU field
        local gpu_available=$(echo "$body" | jq -r '.gpu.available // empty')
        if [ -z "$gpu_available" ]; then
            print_warning "Missing field: .gpu.available"
        else
            if [ "$(echo "$body" | jq -r '.gpu.available | type')" != "boolean" ]; then
                print_error ".gpu.available is not a boolean"
                return 1
            fi
            print_success "GPU available: $gpu_available"
        fi

        # Check models object exists
        if ! echo "$body" | jq -e '.models' > /dev/null 2>&1; then
            print_warning "Missing object: .models"
        else
            print_success "Models object present"
        fi

        # Check components object exists
        if ! echo "$body" | jq -e '.components' > /dev/null 2>&1; then
            print_warning "Missing object: .components"
        else
            print_success "Components object present"
        fi
    else
        print_warning "jq not available, skipping schema validation"
    fi

    return 0
}

test_liveness_endpoint() {
    local response
    local status_code
    local headers
    local start_time
    local end_time
    local duration

    # Capture timing using curl's built-in timing
    local timing_output
    start_time=$(python -c "import time; print(int(time.monotonic() * 1000))" 2>/dev/null || date +%s%3N)

    # Create temp files for headers and body
    local tmp_headers=$(mktemp)
    local tmp_body=$(mktemp)

    # Capture headers and body separately with timing
    timing_output=$(curl -s -w "@-" -D "$tmp_headers" -o "$tmp_body" --max-time $TIMEOUT "${BASE_URL}/health/live" <<'EOF'
{
  "time_total": "%{time_total}",
  "http_code": "%{http_code}",
  "exitcode": "%{exitcode}"
}
EOF
2>/dev/null || echo '{"time_total": "0.000", "http_code": "000", "exitcode": "1"}')

    status_code=$(echo "$timing_output" | grep -o '"http_code": "[^"]*' | cut -d'"' -f4)
    local body=$(cat "$tmp_body")
    headers=$(cat "$tmp_headers")
    local curl_time=$(echo "$timing_output" | grep -o '"time_total": "[^"]*' | cut -d'"' -f4)
    end_time=$(python -c "import time; print(int(time.monotonic() * 1000))" 2>/dev/null || date +%s%3N)

    # Calculate duration in milliseconds
    if [[ "$curl_time" =~ ^[0-9]*\.?[0-9]+$ ]]; then
        duration=$(echo "scale=0; $curl_time * 1000 / 1" | bc 2>/dev/null || echo "500")
    else
        duration=$(python -c "import time; print(int(time.monotonic() * 1000 - $start_time))" 2>/dev/null || echo "500")
    fi

    # Clean up temp files
    rm -f "$tmp_headers" "$tmp_body"

    # Check status code
    if [ "$status_code" != "200" ]; then
        print_error "Status: $status_code (expected 200)"
        return 1
    fi
    print_success "Status: 200 OK"

    # Check response time threshold (500ms for liveness)
    if [ $duration -gt 500 ]; then
        print_error "Response time: ${duration}ms (exceeds 500ms threshold)"
        return 1
    fi
    print_success "Response time: ${duration}ms"

    # Check Content-Type header
    local content_type=$(echo "$headers" | grep -i "^content-type:" | awk '{print $2}' | tr -d '\r\n')
    if [[ ! "$content_type" =~ application/json ]]; then
        print_error "Content-Type: $content_type (expected application/json)"
        return 1
    fi
    print_success "Content-Type: $content_type"

    # Validate JSON schema
    if command -v jq &> /dev/null; then
        local status=$(echo "$body" | jq -r '.status // empty')
        if [ "$status" != "alive" ]; then
            print_error "Expected .status == 'alive', got: $status"
            return 1
        fi
        print_success "Liveness status: alive"
    else
        print_warning "jq not available, skipping schema validation"
    fi

    return 0
}

test_readiness_endpoint() {
    local response
    local status_code
    local headers
    local start_time
    local end_time
    local duration

    # Retry with exponential backoff before actual test
    if ! retry_with_backoff "/health/ready"; then
        print_error "Readiness endpoint not responding after $MAX_RETRIES retries"
        return 1
    fi

    # Capture timing using curl's built-in timing
    local timing_output
    start_time=$(python -c "import time; print(int(time.monotonic() * 1000))" 2>/dev/null || date +%s%3N)

    # Create temp files for headers and body
    local tmp_headers=$(mktemp)
    local tmp_body=$(mktemp)

    # Capture headers and body separately with timing
    timing_output=$(curl -s -w "@-" -D "$tmp_headers" -o "$tmp_body" --max-time $TIMEOUT "${BASE_URL}/health/ready" <<'EOF'
{
  "time_total": "%{time_total}",
  "http_code": "%{http_code}",
  "exitcode": "%{exitcode}"
}
EOF
2>/dev/null || echo '{"time_total": "0.000", "http_code": "000", "exitcode": "1"}')

    status_code=$(echo "$timing_output" | grep -o '"http_code": "[^"]*' | cut -d'"' -f4)
    local body=$(cat "$tmp_body")
    headers=$(cat "$tmp_headers")
    local curl_time=$(echo "$timing_output" | grep -o '"time_total": "[^"]*' | cut -d'"' -f4)
    end_time=$(python -c "import time; print(int(time.monotonic() * 1000))" 2>/dev/null || date +%s%3N)

    # Calculate duration in milliseconds
    if [[ "$curl_time" =~ ^[0-9]*\.?[0-9]+$ ]]; then
        duration=$(echo "scale=0; $curl_time * 1000 / 1" | bc 2>/dev/null || echo "1000")
    else
        duration=$(python -c "import time; print(int(time.monotonic() * 1000 - $start_time))" 2>/dev/null || echo "1000")
    fi

    # Clean up temp files
    rm -f "$tmp_headers" "$tmp_body"

    # Check status code (200 or 503 are acceptable)
    if [ "$status_code" != "200" ] && [ "$status_code" != "503" ]; then
        print_error "Status: $status_code (expected 200 or 503)"
        return 1
    fi
    print_success "Status: $status_code"

    # Check response time threshold (1000ms for readiness)
    if [ "$duration" -gt 1000 ]; then
        print_error "Response time: ${duration}ms (exceeds 1000ms threshold)"
        return 1
    fi
    print_success "Response time: ${duration}ms"

    # Check Content-Type header
    local content_type=$(echo "$headers" | grep -i "^content-type:" | awk '{print $2}' | tr -d '\r\n')
    if [[ ! "$content_type" =~ application/json ]]; then
        print_error "Content-Type: $content_type (expected application/json)"
        return 1
    fi
    print_success "Content-Type: $content_type"

    # Validate JSON schema
    if command -v jq &> /dev/null; then
        local status=$(echo "$body" | jq -r '.status // empty')
        if [ -z "$status" ]; then
            print_error "Missing required field: .status"
            return 1
        fi
        print_info "Readiness status: $status"

        # If status is 200, check critical components
        if [ "$status_code" = "200" ]; then
            print_success "Service is ready"

            # Check for components object
            if echo "$body" | jq -e '.components' > /dev/null 2>&1; then
                # Check model component if it exists
                local model_ready=$(echo "$body" | jq -r '.components.model // empty')
                if [ -n "$model_ready" ]; then
                    if [ "$model_ready" = "true" ]; then
                        print_success "Model component: ready"
                    else
                        print_warning "Model component: not ready"
                    fi
                fi

                # Check synthesizer component if it exists
                local synth_ready=$(echo "$body" | jq -r '.components.synthesizer // empty')
                if [ -n "$synth_ready" ]; then
                    if [ "$synth_ready" = "true" ]; then
                        print_success "Synthesizer component: ready"
                    else
                        print_warning "Synthesizer component: not ready"
                    fi
                fi
            fi
        else
            print_warning "Service is not ready (503)"
        fi
    else
        print_warning "jq not available, skipping schema validation"
    fi

    return 0
}

test_docker_health_check() {
    if command -v docker &> /dev/null; then
        if docker ps --filter name=auto_voice_app --format '{{.Names}}' | grep -q auto_voice_app; then
            local health_status=$(docker inspect auto_voice_app --format='{{.State.Health.Status}}' 2>/dev/null || echo "unknown")
            
            if [ "$health_status" = "healthy" ]; then
                print_success "Container status: healthy"
                return 0
            elif [ "$health_status" = "unknown" ]; then
                print_warning "Container health check not configured"
                return 0
            else
                print_warning "Container status: $health_status"
                return 1
            fi
        else
            print_info "Docker container not running (skipping)"
            return 0
        fi
    else
        print_info "Docker not available (skipping)"
        return 0
    fi
}

# Main execution
echo "=== AutoVoice Health Check Validation ==="
echo "Base URL: $BASE_URL"
echo ""

# Test 1: Main health endpoint
echo "[1/4] Testing /health endpoint..."
test_health_endpoint
HEALTH_RESULT=$?
echo ""

# Test 2: Liveness probe
echo "[2/4] Testing /health/live endpoint..."
test_liveness_endpoint
LIVE_RESULT=$?
echo ""

# Test 3: Readiness probe
echo "[3/4] Testing /health/ready endpoint..."
test_readiness_endpoint
READY_RESULT=$?
echo ""

# Test 4: Docker health check
echo "[4/4] Testing Docker health check..."
test_docker_health_check
DOCKER_RESULT=$?
echo ""

# Summary
echo "=== Validation Summary ==="
echo "Main health check: $(result_status $HEALTH_RESULT)"
echo "Liveness probe: $(result_status $LIVE_RESULT)"
echo "Readiness probe: $(result_status $READY_RESULT)"
echo "Docker health: $(result_status $DOCKER_RESULT)"
echo ""

# Exit code
if [ $HEALTH_RESULT -eq 0 ] && [ $LIVE_RESULT -eq 0 ] && [ $READY_RESULT -eq 0 ]; then
    print_success "All health checks PASSED"
    exit 0
else
    print_error "Some health checks FAILED"
    exit 1
fi

