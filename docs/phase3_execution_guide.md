# Phase 3 Execution Guide

## Overview

Phase 3 validates AutoVoice's containerized deployment, API functionality, and end-to-end integration through comprehensive Docker builds, service orchestration, and endpoint testing.

**Estimated Time**: 2-4 hours  
**Prerequisites**: Docker, Docker Compose, NVIDIA Docker runtime, Python 3.12, Conda

## Prerequisites

### System Requirements

1. **Docker & Docker Compose**
   ```bash
   docker --version  # >= 20.10
   docker compose version  # >= 2.0 (or docker-compose >= 1.29)
   ```

2. **NVIDIA Docker Runtime**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

3. **Python Environment**
   ```bash
   conda activate autovoice_py312
   python --version  # 3.12.x
   ```

4. **Disk Space**
   - Minimum: 10 GB free
   - Recommended: 20 GB free

5. **Network**
   - Ports 5000, 6379 available
   - Internet access for Docker image pulls

### Software Dependencies

```bash
# Install required Python packages
pip install pytest pytest-cov python-socketio

# Verify CUDA extensions (optional for Phase 3)
PYTHONPATH=src python -c "import cuda_kernels"
```

## Automated Execution

### Quick Start

```bash
# Run complete Phase 3 validation
./scripts/phase3_execute.sh
```

### Using run_tests.sh

```bash
# Full Phase 3 execution
./run_tests.sh phase3

# Individual components
./run_tests.sh docker-build    # Build Docker image only
./run_tests.sh docker-test      # Test Docker Compose
./run_tests.sh api-validate     # Validate API endpoints
./run_tests.sh websocket-test   # Test WebSocket connection
```

## Manual Execution

### Step 1: Prerequisites Validation

```bash
# Check Docker
docker info

# Check NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Check Python environment
conda activate autovoice_py312
python --version
```

### Step 2: Build Docker Image

```bash
# Build with CUDA 12.1
docker build -t autovoice:latest .

# Verify build
docker images | grep autovoice

# Verify CUDA access in container
docker run --rm --gpus all autovoice:latest nvidia-smi
```

### Step 3: Start Services with Docker Compose

```bash
# Start services
docker compose up -d

# Check service status
docker compose ps

# View logs
docker compose logs -f auto-voice-app
```

### Step 4: Wait for Services to be Ready

```bash
# Poll health endpoint (up to 180 seconds)
for i in {1..60}; do
  if curl -f http://localhost:5000/health; then
    echo "Services ready"
    break
  fi
  sleep 3
done
```

### Step 5: Run End-to-End Tests

```bash
# E2E tests
pytest tests/test_end_to_end.py -v --tb=short

# Web interface tests
pytest tests/test_web_interface.py -v --tb=short
```

### Step 6: Validate API Endpoints

```bash
# Run API validation script
./scripts/validate_api_endpoints.sh

# Check results
cat logs/phase3/api_validation_results_*.json
```

### Step 7: Test WebSocket Connection

```bash
# Run WebSocket test
python scripts/test_websocket_connection.py --url http://localhost:5000

# Or with JSON output
python scripts/test_websocket_connection.py --json-output ws_results.json
```

### Step 8: Run Performance Tests

```bash
# Performance benchmarks
pytest tests/test_performance.py -v -s --tb=short
```

### Step 9: Generate Report

```bash
# Generate Phase 3 completion report
./scripts/generate_phase3_report.sh [TIMESTAMP]

# View report
cat PHASE3_COMPLETION_REPORT.md
```

### Step 10: Cleanup

```bash
# Stop services
docker compose down

# Optional: Remove images
docker rmi autovoice:latest
```

## Success Indicators

### Docker Build
- ✅ Image builds without errors
- ✅ Image size reasonable (< 10 GB)
- ✅ `nvidia-smi` works inside container

### Service Orchestration
- ✅ All services show "Up" status
- ✅ `/health` returns 200
- ✅ `/health/ready` returns 200 or 503
- ✅ No critical errors in logs

### API Validation
- ✅ All health endpoints respond correctly
- ✅ `/api/v1/synthesize` accepts POST requests
- ✅ `/api/v1/speakers` returns speaker list
- ✅ `/api/v1/gpu_status` returns GPU info
- ✅ `/api/v1/voice/profiles` returns profiles

### WebSocket Testing
- ✅ Socket.IO connection established
- ✅ `connected` event received
- ✅ `get_status` request succeeds
- ✅ `status` event received
- ✅ Clean disconnect

### End-to-End Tests
- ✅ E2E tests pass (or acceptable failure rate)
- ✅ Web interface tests pass
- ✅ Performance tests complete

## Troubleshooting

### Docker Build Fails

**Problem**: Build fails with CUDA errors
```bash
# Solution: Verify CUDA base image
grep "FROM nvidia/cuda" Dockerfile
# Should be: FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
```

**Problem**: Out of disk space
```bash
# Solution: Clean up Docker
docker system prune -a
```

### Services Won't Start

**Problem**: Port already in use
```bash
# Solution: Check what's using the port
sudo lsof -i :5000
# Kill the process or change port in docker-compose.yml
```

**Problem**: GPU not accessible
```bash
# Solution: Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Health Checks Fail

**Problem**: `/health` returns 503
```bash
# Solution: Check service logs
docker compose logs auto-voice-app | tail -50
# Look for model loading errors or dependency issues
```

**Problem**: Services timeout during startup
```bash
# Solution: Increase wait time or check resources
docker stats
# Ensure sufficient memory and CPU
```

### API Validation Fails

**Problem**: Endpoints return 404
```bash
# Solution: Verify service is running
curl http://localhost:5000/health
# Check API routes in logs
docker compose logs auto-voice-app | grep "Registered route"
```

**Problem**: Endpoints return 500
```bash
# Solution: Check application logs
docker compose logs auto-voice-app | grep ERROR
```

### WebSocket Tests Fail

**Problem**: Connection refused
```bash
# Solution: Verify Socket.IO is enabled
curl http://localhost:5000/socket.io/
# Should return Socket.IO handshake info
```

**Problem**: Events not received
```bash
# Solution: Check WebSocket handler logs
docker compose logs auto-voice-app | grep WebSocket
```

## Best Practices

1. **Run in Clean Environment**: Start with fresh Docker state
2. **Monitor Resources**: Watch CPU, memory, and GPU usage
3. **Save Logs**: Keep logs for debugging and analysis
4. **Incremental Testing**: Test each component before moving to next
5. **Document Issues**: Note any failures or warnings for investigation

## Environment Variables

```bash
# Optional: Require local CUDA extensions
export REQUIRE_LOCAL_CUDA=1

# Optional: Custom base URL for API tests
export AUTO_VOICE_BASE_URL=http://localhost:5000

# Optional: Verbose output
export VERBOSE=true
```

## Next Steps

After successful Phase 3 completion:

1. Review `PHASE3_COMPLETION_REPORT.md`
2. Address any warnings or failures
3. Proceed to Phase 4 (Security Hardening)
4. Deploy to staging environment

---

*For issues or questions, refer to the main README.md or open an issue on GitHub.*

