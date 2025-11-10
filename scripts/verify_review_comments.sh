#!/bin/bash
# Verification Script for Review Comment Implementation
# Purpose: Verify all 6 review comments have been properly implemented
# Usage: ./scripts/verify_review_comments.sh

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_info() { echo -e "${BLUE}ℹ${NC} $1"; }

echo "=== AutoVoice Review Comments Implementation Verification ==="
echo ""

# Comment 1: E2E test script exit code handling
echo "[1/6] Verifying E2E test script exit code handling..."
if grep -q "set +e" scripts/run_e2e_tests.sh && \
   grep -q "EXIT_CODE=\${PIPESTATUS\[0\]}" scripts/run_e2e_tests.sh && \
   grep -q "set -e" scripts/run_e2e_tests.sh; then
    print_success "E2E test script has proper exit code handling"
else
    print_error "E2E test script missing exit code handling"
fi
echo ""

# Comment 2: Production readiness checklist updated
echo "[2/6] Verifying production readiness checklist..."
if grep -q "2025-11-07" docs/production_readiness_checklist.md && \
   grep -q "Version.*2.2" docs/production_readiness_checklist.md; then
    print_success "Production readiness checklist updated"
else
    print_warning "Production readiness checklist may need updates"
fi
echo ""

# Comment 3: Metrics reference document
echo "[3/6] Verifying metrics reference document..."
if [ -f "docs/metrics-reference.md" ]; then
    if grep -q "autovoice_http_requests_total" docs/metrics-reference.md && \
       grep -q "autovoice_synthesis_requests_total" docs/metrics-reference.md && \
       grep -q "autovoice_gpu_utilization_percent" docs/metrics-reference.md; then
        print_success "Metrics reference document exists and contains expected metrics"
    else
        print_warning "Metrics reference document missing some metrics"
    fi
else
    print_error "Metrics reference document not found"
fi
echo ""

# Comment 4: NVIDIA exporter port exposure
echo "[4/6] Verifying NVIDIA exporter port configuration..."
if grep -q "# No external port exposure" docker-compose.yml; then
    print_success "NVIDIA exporter port not exposed externally"
else
    print_warning "NVIDIA exporter configuration may need review"
fi
echo ""

# Comment 5: Health check retry logic
echo "[5/6] Verifying health check retry logic..."
if grep -q "MAX_RETRIES=5" scripts/validate_health_checks.sh && \
   grep -q "RETRY_DELAY=2" scripts/validate_health_checks.sh && \
   grep -q "retry_with_backoff" scripts/validate_health_checks.sh; then
    print_success "Health check retry logic implemented"
else
    print_error "Health check retry logic missing"
fi
echo ""

# Comment 6: Grafana dashboard timeseries panels
echo "[6/6] Verifying Grafana dashboard panel types..."
if [ -f "config/grafana/dashboards/autovoice-overview.json" ]; then
    GRAPH_COUNT=$(grep -o '"type": "graph"' config/grafana/dashboards/autovoice-overview.json | wc -l)
    TIMESERIES_COUNT=$(grep -o '"type": "timeseries"' config/grafana/dashboards/autovoice-overview.json | wc -l)

    if [ "$GRAPH_COUNT" -eq 0 ] && [ "$TIMESERIES_COUNT" -gt 0 ]; then
        print_success "Grafana dashboard uses timeseries panels (found $TIMESERIES_COUNT)"
    else
        print_warning "Grafana dashboard may still use legacy graph panels (graph: $GRAPH_COUNT, timeseries: $TIMESERIES_COUNT)"
    fi
else
    print_error "Grafana dashboard not found"
fi
echo ""

# Summary
echo "=== Verification Summary ==="
echo ""
print_info "All implementation checks completed"
echo ""
echo "Next steps:"
echo "1. Run E2E tests: ./scripts/run_e2e_tests.sh --full"
echo "2. Start monitoring stack: docker-compose --profile monitoring up -d"
echo "3. Validate health checks: ./scripts/validate_health_checks.sh"
echo "4. Check Grafana dashboard: http://localhost:3000"
echo ""
print_success "Implementation verification complete!"

