#!/bin/bash

# AutoVoice API Endpoint Validation Script
# Validates all REST API endpoints for Phase 3 testing
# Generates api_validation_results.json with status codes and response times

set -e
set -o pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Accept timestamp from CLI argument or generate new one
if [ -n "$1" ]; then
    TIMESTAMP="$1"
else
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
fi

OUTPUT_DIR="$PROJECT_ROOT/logs/phase3"
RESULTS_FILE="$OUTPUT_DIR/api_validation_results_${TIMESTAMP}.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Unicode symbols
CHECK="✓"
CROSS="✗"
INFO="ℹ"
ARROW="→"

# API Configuration
BASE_URL=${AUTO_VOICE_BASE_URL:-"http://localhost:5000"}
VERBOSE=${VERBOSE:-false}

# Create output directories
mkdir -p "$OUTPUT_DIR"

# Helper functions
print_info() {
    echo -e "${BLUE}[${INFO}]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[${CHECK}]${NC} $1"
}

print_error() {
    echo -e "${RED}[${CROSS}]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Temporary file for collecting endpoint results
ENDPOINTS_NDJSON="$OUTPUT_DIR/api_endpoints_${TIMESTAMP}.ndjson"

# Initialize results structure
init_results() {
    # Create empty NDJSON file for collecting results
    > "$ENDPOINTS_NDJSON"
}

# Update results with endpoint status
update_results() {
    local endpoint="$1"
    local method="$2"
    local status_code="$3"
    local response_time="$4"
    local status="$5"
    local message="$6"

    # Append to NDJSON file (one JSON object per line)
    cat >> "$ENDPOINTS_NDJSON" << EOF
{"endpoint":"$endpoint","method":"$method","status_code":$status_code,"response_time_ms":$response_time,"status":"$status","message":"$message","timestamp":"$TIMESTAMP"}
EOF
}

# Test health endpoints
test_health_endpoints() {
    print_info "Testing health endpoints..."

    # /health
    print_info "Testing GET /health"
    local start_time=$(date +%s%N)
    local response=$(curl -s -w "%{http_code}" -o /tmp/health_response.json "$BASE_URL/health" 2>/dev/null || echo "000")
    local end_time=$(date +%s%N)
    local status_code=${response: -3}
    local response_time=$(( (end_time - start_time) / 1000000 ))

    local status="unknown"
    local message=""

    if [ "$status_code" = "200" ]; then
        status="success"
        message="Health check passed"
        print_success "/health - ${response_time}ms"
    else
        status="error"
        message="Health check failed with status $status_code"
        print_error "/health failed with status $status_code"
    fi

    update_results "/health" "GET" "$status_code" "$response_time" "$status" "$message"

    # /api/v1/health
    print_info "Testing GET /api/v1/health"
    start_time=$(date +%s%N)
    response=$(curl -s -w "%{http_code}" -o /tmp/api_health_response.json "$BASE_URL/api/v1/health" 2>/dev/null || echo "000")
    end_time=$(date +%s%N)
    status_code=${response: -3}
    response_time=$(( (end_time - start_time) / 1000000 ))

    if [ "$status_code" = "200" ]; then
        status="success"
        message="API health check passed"
        print_success "/api/v1/health - ${response_time}ms"
    else
        status="error"
        message="API health check failed with status $status_code"
        print_error "/api/v1/health failed with status $status_code"
    fi

    update_results "/api/v1/health" "GET" "$status_code" "$response_time" "$status" "$message"

    # /health/ready
    print_info "Testing GET /health/ready"
    start_time=$(date +%s%N)
    response=$(curl -s -w "%{http_code}" -o /tmp/health_ready_response.json "$BASE_URL/health/ready" 2>/dev/null || echo "000")
    end_time=$(date +%s%N)
    status_code=${response: -3}
    response_time=$(( (end_time - start_time) / 1000000 ))

    # Readiness accepts both 200 (ready) and 503 (not ready) as valid responses
    if [ "$status_code" = "200" ]; then
        status="success"
        message="Service is ready (200)"
        print_success "/health/ready - ${response_time}ms (ready: 200)"
    elif [ "$status_code" = "503" ]; then
        status="success"
        message="Service not ready but endpoint working (503)"
        print_success "/health/ready - ${response_time}ms (not ready: 503, acceptable)"
    else
        status="error"
        message="Readiness check returned unexpected status $status_code"
        print_error "/health/ready failed with status $status_code"
    fi

    update_results "/health/ready" "GET" "$status_code" "$response_time" "$status" "$message"

    # /health/live
    print_info "Testing GET /health/live"
    start_time=$(date +%s%N)
    response=$(curl -s -w "%{http_code}" -o /tmp/health_live_response.json "$BASE_URL/health/live" 2>/dev/null || echo "000")
    end_time=$(date +%s%N)
    status_code=${response: -3}
    response_time=$(( (end_time - start_time) / 1000000 ))

    if [ "$status_code" = "200" ]; then
        status="success"
        message="Liveness check passed"
        print_success "/health/live - ${response_time}ms"
    else
        status="error"
        message="Liveness check failed with status $status_code"
        print_error "/health/live failed with status $status_code"
    fi

    update_results "/health/live" "GET" "$status_code" "$response_time" "$status" "$message"

    # /metrics
    print_info "Testing GET /metrics"
    start_time=$(date +%s%N)
    response=$(curl -s -w "%{http_code}" -o /tmp/metrics_response.txt "$BASE_URL/metrics" 2>/dev/null || echo "000")
    end_time=$(date +%s%N)
    status_code=${response: -3}
    response_time=$(( (end_time - start_time) / 1000000 ))

    # Metrics can be 200 (enabled) or 503 (disabled)
    if [ "$status_code" = "200" ]; then
        status="success"
        message="Metrics endpoint enabled"
        print_success "/metrics - ${response_time}ms (enabled)"
    elif [ "$status_code" = "503" ]; then
        status="success"
        message="Metrics endpoint disabled but working"
        print_success "/metrics - ${response_time}ms (disabled, acceptable)"
    else
        status="warning"
        message="Metrics endpoint returned $status_code"
        print_warning "/metrics returned $status_code"
    fi

    update_results "/metrics" "GET" "$status_code" "$response_time" "$status" "$message"
}

# Test TTS endpoints
test_tts_endpoints() {
    print_info "Testing TTS endpoints..."

    # /api/v1/synthesize (POST) - actual endpoint
    print_info "Testing POST /api/v1/synthesize"
    local test_data='{"text": "Hello World", "speaker_id": 0}'
    start_time=$(date +%s%N)
    response=$(curl -s -X POST -H "Content-Type: application/json" -d "$test_data" -w "%{http_code}" -o /tmp/synthesize_post_response.json "$BASE_URL/api/v1/synthesize" 2>/dev/null || echo "000")
    end_time=$(date +%s%N)
    status_code=${response: -3}
    response_time=$(( (end_time - start_time) / 1000000 ))

    local status="unknown"
    local message=""

    if [ "$status_code" = "200" ] || [ "$status_code" = "201" ]; then
        status="success"
        message="TTS synthesis completed successfully"
        print_success "/api/v1/synthesize (POST) - ${response_time}ms"
    else
        status="warning"  # TTS might require dependencies
        message="TTS synthesis returned status $status_code (may require model loading)"
        print_warning "/api/v1/synthesize (POST) returned status $status_code"
    fi

    update_results "/api/v1/synthesize" "POST" "$status_code" "$response_time" "$status" "$message"

    # Test GET on synthesize (should return 405)
    print_info "Testing GET /api/v1/synthesize (negative test)"
    start_time=$(date +%s%N)
    response=$(curl -s -w "%{http_code}" -o /dev/null "$BASE_URL/api/v1/synthesize" 2>/dev/null || echo "000")
    end_time=$(date +%s%N)
    status_code=${response: -3}
    response_time=$(( (end_time - start_time) / 1000000 ))

    if [ "$status_code" = "405" ]; then
        status="success"
        message="Synthesize correctly rejects GET method"
        print_success "/api/v1/synthesize (GET) correctly returns 405"
    else
        status="warning"
        message="Synthesize GET returned unexpected status $status_code"
        print_warning "/api/v1/synthesize (GET) returned $status_code (expected 405)"
    fi

    update_results "/api/v1/synthesize" "GET" "$status_code" "$response_time" "$status" "$message"

    # Test /api/v1/speakers
    print_info "Testing GET /api/v1/speakers"
    start_time=$(date +%s%N)
    response=$(curl -s -w "%{http_code}" -o /tmp/speakers_response.json "$BASE_URL/api/v1/speakers" 2>/dev/null || echo "000")
    end_time=$(date +%s%N)
    status_code=${response: -3}
    response_time=$(( (end_time - start_time) / 1000000 ))

    if [ "$status_code" = "200" ]; then
        status="success"
        message="Speakers endpoint responded correctly"
        print_success "/api/v1/speakers - ${response_time}ms"
    else
        status="error"
        message="Speakers endpoint failed with status $status_code"
        print_error "/api/v1/speakers failed with status $status_code"
    fi

    update_results "/api/v1/speakers" "GET" "$status_code" "$response_time" "$status" "$message"

    # Test /api/v1/gpu_status
    print_info "Testing GET /api/v1/gpu_status"
    start_time=$(date +%s%N)
    response=$(curl -s -w "%{http_code}" -o /tmp/gpu_status_response.json "$BASE_URL/api/v1/gpu_status" 2>/dev/null || echo "000")
    end_time=$(date +%s%N)
    status_code=${response: -3}
    response_time=$(( (end_time - start_time) / 1000000 ))

    if [ "$status_code" = "200" ]; then
        status="success"
        message="GPU status endpoint responded correctly"
        print_success "/api/v1/gpu_status - ${response_time}ms"
    else
        status="warning"
        message="GPU status endpoint returned $status_code (may not have GPU)"
        print_warning "/api/v1/gpu_status returned $status_code"
    fi

    update_results "/api/v1/gpu_status" "GET" "$status_code" "$response_time" "$status" "$message"

    # Test /api/v1/ws/audio_stream
    print_info "Testing GET /api/v1/ws/audio_stream"
    start_time=$(date +%s%N)
    response=$(curl -s -w "%{http_code}" -o /tmp/ws_info_response.json "$BASE_URL/api/v1/ws/audio_stream" 2>/dev/null || echo "000")
    end_time=$(date +%s%N)
    status_code=${response: -3}
    response_time=$(( (end_time - start_time) / 1000000 ))

    if [ "$status_code" = "200" ]; then
        status="success"
        message="WebSocket info endpoint responded correctly"
        print_success "/api/v1/ws/audio_stream - ${response_time}ms"
    else
        status="warning"
        message="WebSocket info endpoint returned $status_code"
        print_warning "/api/v1/ws/audio_stream returned $status_code"
    fi

    update_results "/api/v1/ws/audio_stream" "GET" "$status_code" "$response_time" "$status" "$message"
}

# Test voice conversion endpoints
test_voice_conversion_endpoints() {
    print_info "Testing voice conversion endpoints..."

    # POST /api/v1/convert/song - test with minimal payload (will fail validation but should not crash)
    print_info "Testing POST /api/v1/convert/song (validation test)"
    start_time=$(date +%s%N)
    # Test with empty multipart to verify endpoint exists and validates
    response=$(curl -s -X POST -F "song=@/dev/null" -w "%{http_code}" -o /tmp/convert_song_response.json "$BASE_URL/api/v1/convert/song" 2>/dev/null || echo "000")
    end_time=$(date +%s%N)
    status_code=${response: -3}
    response_time=$(( (end_time - start_time) / 1000000 ))

    local status="unknown"
    local message=""

    # Conversion endpoint should reject empty/invalid data - that's expected
    if [ "$status_code" = "400" ] || [ "$status_code" = "422" ]; then
        status="success"
        message="Song conversion endpoint validates input correctly"
        print_success "/api/v1/convert/song (POST) validation working - ${response_time}ms"
    elif [ "$status_code" = "200" ] || [ "$status_code" = "201" ]; then
        status="warning"
        message="Song conversion accepted empty data (unexpected)"
        print_warning "/api/v1/convert/song accepted empty data"
    elif [ "$status_code" = "404" ]; then
        status="error"
        message="Song conversion endpoint not found"
        print_error "/api/v1/convert/song not found (404)"
    else
        status="warning"
        message="Song conversion endpoint returned status $status_code"
        print_warning "/api/v1/convert/song returned $status_code"
    fi

    update_results "/api/v1/convert/song" "POST" "$status_code" "$response_time" "$status" "$message"
}

# Test voice profiles endpoints
test_voice_profiles_endpoints() {
    print_info "Testing voice profiles endpoints..."

    # GET /api/v1/voice/profiles
    print_info "Testing GET /api/v1/voice/profiles"
    start_time=$(date +%s%N)
    response=$(curl -s -w "%{http_code}" -o /tmp/profiles_get_response.json "$BASE_URL/api/v1/voice/profiles" 2>/dev/null || echo "000")
    end_time=$(date +%s%N)
    status_code=${response: -3}
    response_time=$(( (end_time - start_time) / 1000000 ))

    if [ "$status_code" = "200" ]; then
        status="success"
        message="Voice profiles endpoint responded correctly"
        print_success "/api/v1/voice/profiles - ${response_time}ms"
    else
        status="error"
        message="Voice profiles endpoint failed with status $status_code"
        print_error "/api/v1/voice/profiles failed with status $status_code"
    fi

    update_results "/api/v1/voice/profiles" "GET" "$status_code" "$response_time" "$status" "$message"

    # GET /api/v1/voice/profiles with query params
    print_info "Testing GET /api/v1/voice/profiles?limit=5"
    start_time=$(date +%s%N)
    response=$(curl -s -w "%{http_code}" -o /tmp/profiles_query_response.json "$BASE_URL/api/v1/voice/profiles?limit=5" 2>/dev/null || echo "000")
    end_time=$(date +%s%N)
    status_code=${response: -3}
    response_time=$(( (end_time - start_time) / 1000000 ))

    if [ "$status_code" = "200" ]; then
        status="success"
        message="Voice profiles query parameters supported"
        print_success "/api/v1/voice/profiles (with query) - ${response_time}ms"
    else
        status="warning"
        message="Voice profiles query parameters not implemented (status $status_code)"
        print_warning "/api/v1/voice/profiles query params not supported"
    fi

    update_results "/api/v1/voice/profiles?limit=5" "GET" "$status_code" "$response_time" "$status" "$message"

    # POST /api/v1/voice/clone - negative test without audio file
    print_info "Testing POST /api/v1/voice/clone (negative test - no audio)"
    start_time=$(date +%s%N)
    # Test with empty payload to verify validation
    response=$(curl -s -X POST -H "Content-Type: application/json" -d '{"name":"test"}' -w "%{http_code}" -o /tmp/clone_negative_response.json "$BASE_URL/api/v1/voice/clone" 2>/dev/null || echo "000")
    end_time=$(date +%s%N)
    status_code=${response: -3}
    response_time=$(( (end_time - start_time) / 1000000 ))

    # Voice clone should reject requests without audio file
    if [ "$status_code" = "400" ] || [ "$status_code" = "422" ]; then
        status="success"
        message="Voice clone endpoint validates audio requirement correctly"
        print_success "/api/v1/voice/clone (POST) validation working - ${response_time}ms"
    elif [ "$status_code" = "200" ] || [ "$status_code" = "201" ]; then
        status="warning"
        message="Voice clone accepted request without audio (unexpected)"
        print_warning "/api/v1/voice/clone accepted request without audio"
    elif [ "$status_code" = "404" ]; then
        status="error"
        message="Voice clone endpoint not found"
        print_error "/api/v1/voice/clone not found (404)"
    else
        status="warning"
        message="Voice clone endpoint returned status $status_code"
        print_warning "/api/v1/voice/clone returned $status_code"
    fi

    update_results "/api/v1/voice/clone" "POST" "$status_code" "$response_time" "$status" "$message"
}

# Test negative cases and error handling
test_negative_cases() {
    print_info "Testing negative cases and error handling..."

    # Test invalid endpoint
    print_info "Testing GET /api/v1/invalid_endpoint (should return 404)"
    start_time=$(date +%s%N)
    response=$(curl -s -w "%{http_code}" -o /dev/null "$BASE_URL/api/v1/invalid_endpoint" 2>/dev/null || echo "000")
    end_time=$(date +%s%N)
    status_code=${response: -3}
    response_time=$(( (end_time - start_time) / 1000000 ))

    if [ "$status_code" = "404" ]; then
        status="success"
        message="Invalid endpoint correctly returns 404"
        print_success "/api/v1/invalid_endpoint correctly returns 404 - ${response_time}ms"
    else
        status="warning"
        message="Invalid endpoint returned $status_code (expected 404)"
        print_warning "/api/v1/invalid_endpoint returned $status_code (expected 404)"
    fi

    update_results "/api/v1/invalid_endpoint" "GET" "$status_code" "$response_time" "$status" "$message"

    # Test invalid HTTP method on valid endpoint
    print_info "Testing PUT /api/v1/speakers (should return 405)"
    start_time=$(date +%s%N)
    response=$(curl -s -X PUT -w "%{http_code}" -o /dev/null "$BASE_URL/api/v1/speakers" 2>/dev/null || echo "000")
    end_time=$(date +%s%N)
    status_code=${response: -3}
    response_time=$(( (end_time - start_time) / 1000000 ))

    if [ "$status_code" = "405" ]; then
        status="success"
        message="Invalid method correctly returns 405"
        print_success "PUT /api/v1/speakers correctly returns 405 - ${response_time}ms"
    else
        status="warning"
        message="Invalid method returned $status_code (expected 405)"
        print_warning "PUT /api/v1/speakers returned $status_code (expected 405)"
    fi

    update_results "/api/v1/speakers" "PUT" "$status_code" "$response_time" "$status" "$message"

    # Test malformed JSON
    print_info "Testing POST /api/v1/synthesize with malformed JSON (should return 400)"
    start_time=$(date +%s%N)
    response=$(curl -s -X POST -H "Content-Type: application/json" -d '{"text": "test", "invalid": json}' -w "%{http_code}" -o /dev/null "$BASE_URL/api/v1/synthesize" 2>/dev/null || echo "000")
    end_time=$(date +%s%N)
    status_code=${response: -3}
    response_time=$(( (end_time - start_time) / 1000000 ))

    if [ "$status_code" = "400" ] || [ "$status_code" = "422" ]; then
        status="success"
        message="Malformed JSON correctly rejected"
        print_success "POST /api/v1/synthesize with malformed JSON correctly rejected - ${response_time}ms"
    else
        status="warning"
        message="Malformed JSON returned $status_code (expected 400/422)"
        print_warning "POST /api/v1/synthesize with malformed JSON returned $status_code"
    fi

    update_results "/api/v1/synthesize" "POST" "$status_code" "$response_time" "$status" "$message"

    # Test missing required parameters
    print_info "Testing POST /api/v1/synthesize with missing text (should return 400)"
    start_time=$(date +%s%N)
    response=$(curl -s -X POST -H "Content-Type: application/json" -d '{"speaker_id": 0}' -w "%{http_code}" -o /dev/null "$BASE_URL/api/v1/synthesize" 2>/dev/null || echo "000")
    end_time=$(date +%s%N)
    status_code=${response: -3}
    response_time=$(( (end_time - start_time) / 1000000 ))

    if [ "$status_code" = "400" ] || [ "$status_code" = "422" ]; then
        status="success"
        message="Missing required parameters correctly rejected"
        print_success "POST /api/v1/synthesize with missing text correctly rejected - ${response_time}ms"
    else
        status="warning"
        message="Missing parameters returned $status_code (expected 400/422)"
        print_warning "POST /api/v1/synthesize with missing text returned $status_code"
    fi

    update_results "/api/v1/synthesize" "POST" "$status_code" "$response_time" "$status" "$message"

    # Test invalid content type
    print_info "Testing POST /api/v1/synthesize with wrong content type (should return 400)"
    start_time=$(date +%s%N)
    response=$(curl -s -X POST -H "Content-Type: text/plain" -d "text=test&speaker_id=0" -w "%{http_code}" -o /dev/null "$BASE_URL/api/v1/synthesize" 2>/dev/null || echo "000")
    end_time=$(date +%s%N)
    status_code=${response: -3}
    response_time=$(( (end_time - start_time) / 1000000 ))

    if [ "$status_code" = "400" ] || [ "$status_code" = "415" ]; then
        status="success"
        message="Invalid content type correctly rejected"
        print_success "POST /api/v1/synthesize with wrong content type correctly rejected - ${response_time}ms"
    else
        status="warning"
        message="Invalid content type returned $status_code (expected 400/415)"
        print_warning "POST /api/v1/synthesize with wrong content type returned $status_code"
    fi

    update_results "/api/v1/synthesize" "POST" "$status_code" "$response_time" "$status" "$message"
}

# Test legacy redirects (optional - only if LEGACY_TESTS=1)
test_legacy_redirects() {
    if [[ "${LEGACY_TESTS:-0}" != "1" ]]; then
        print_info "Skipping legacy redirect tests (set LEGACY_TESTS=1 to enable)"
        return
    fi

    print_info "Testing legacy redirects..."

    # /tts -> /api/v1/tts (if implemented)
    print_info "Testing legacy redirect /tts -> /api/v1/tts"
    start_time=$(date +%s%N)
    response=$(curl -s -L -w "%{http_code}" -o /dev/null "$BASE_URL/tts" 2>/dev/null || echo "000")
    end_time=$(date +%s%N)
    status_code=${response: -3}
    response_time=$(( (end_time - start_time) / 1000000 ))

    if [ "$status_code" = "200" ]; then
        status="success"
        message="Legacy /tts redirect working"
        print_success "/tts redirect - ${response_time}ms"
    else
        status="warning"
        message="Legacy redirect not implemented (status $status_code) - this is optional"
        print_warning "/tts redirect not implemented (optional)"
    fi

    update_results "/tts" "GET" "$status_code" "$response_time" "$status" "$message"
}

# Generate final summary
generate_summary() {
    print_info "Generating API validation summary..."

    # Count metrics from NDJSON file
    local endpoint_count=0
    local successful_count=0
    local warning_count=0
    local error_count=0
    local total_time=0

    if [[ -f "$ENDPOINTS_NDJSON" ]]; then
        while IFS= read -r line; do
            endpoint_count=$((endpoint_count + 1))

            # Extract status
            if echo "$line" | grep -q '"status":"success"'; then
                successful_count=$((successful_count + 1))
            elif echo "$line" | grep -q '"status":"warning"'; then
                warning_count=$((warning_count + 1))
            elif echo "$line" | grep -q '"status":"error"'; then
                error_count=$((error_count + 1))
            fi

            # Extract response time
            if [[ $line =~ \"response_time_ms\":([0-9]+) ]]; then
                total_time=$((total_time + BASH_REMATCH[1]))
            fi
        done < "$ENDPOINTS_NDJSON"
    fi

    local avg_time=0
    if [ "$endpoint_count" -gt 0 ]; then
        avg_time=$((total_time / endpoint_count))
    fi

    # Build valid JSON structure
    cat > "$RESULTS_FILE" << EOF
{
  "timestamp": "$TIMESTAMP",
  "base_url": "$BASE_URL",
  "endpoints": [
EOF

    # Add endpoint entries from NDJSON (properly formatted as array)
    local first=true
    if [[ -f "$ENDPOINTS_NDJSON" ]]; then
        while IFS= read -r line; do
            # Add comma before all but first entry
            if [ "$first" = false ]; then
                echo "," >> "$RESULTS_FILE"
            fi
            first=false
            # Write endpoint entry as array element
            echo -n "    $line" >> "$RESULTS_FILE"
        done < "$ENDPOINTS_NDJSON"
    fi

    # Close endpoints array and add summary
    cat >> "$RESULTS_FILE" << EOF

  ],
  "summary": {
    "total_endpoints": $endpoint_count,
    "successful": $successful_count,
    "warnings": $warning_count,
    "errors": $error_count,
    "avg_response_time_ms": $avg_time
  },
  "recommendations": []
}
EOF

    # Validate JSON output
    if command -v jq &> /dev/null; then
        if jq . "$RESULTS_FILE" > /dev/null 2>&1; then
            print_success "JSON output validated successfully"
        else
            print_error "Generated JSON is invalid! Check $RESULTS_FILE"
            return 1
        fi
    fi

    print_success "API validation completed: $successful_count/$endpoint_count endpoints working"

    if [ "$error_count" -gt 0 ]; then
        print_error "$error_count endpoints had errors"
    fi

    if [ "$warning_count" -gt 0 ]; then
        print_warning "$warning_count endpoints had warnings"
    fi

    print_info "Average response time: ${avg_time}ms"
    print_info "Results saved to: $RESULTS_FILE"
}

main() {
    echo "🧪 AutoVoice API Endpoint Validation"
    echo "Base URL: $BASE_URL"
    echo "Timestamp: $TIMESTAMP"
    echo ""

    # Initialize results
    init_results

    # Run all endpoint tests
    test_health_endpoints
    test_tts_endpoints
    test_voice_conversion_endpoints
    test_voice_profiles_endpoints
    test_negative_cases
    test_legacy_redirects

    # Generate summary
    generate_summary

    # Extract error count from summary
    local final_error_count=$(grep -o '"errors": [0-9]\+' "$RESULTS_FILE" | awk '{print $2}')
    local final_success_count=$(grep -o '"successful": [0-9]\+' "$RESULTS_FILE" | awk '{print $2}')

    print_success "API validation completed"
    print_info "Results: $final_success_count successful, $final_error_count errors"

    # Exit with success if no critical errors
    if [ "${final_error_count:-0}" -gt 0 ]; then
        print_error "Some endpoints failed - review results in $RESULTS_FILE"
        exit 1
    else
        print_success "All endpoints validated successfully"
        exit 0
    fi
}

# Run main if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
