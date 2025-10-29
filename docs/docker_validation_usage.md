# Docker Deployment Validation Script

## Overview

`scripts/test_docker_deployment.sh` is a comprehensive validation script for testing AutoVoice Docker deployments. It automates the build, deployment, health check, and API testing workflow.

## Features

- **Docker Image Build**: Builds the AutoVoice Docker image from Dockerfile
- **GPU Detection**: Automatically detects and uses NVIDIA Docker runtime if available
- **Container Management**: Starts container with proper GPU flags and environment
- **Health Checks**: Validates all health endpoints
  - `/health/live` - Liveness probe
  - `/health/ready` - Readiness probe
  - `/api/v1/health` - Comprehensive health check
- **GPU Status**: Queries GPU availability and utilization
- **API Testing**: Tests core API endpoints
- **Automatic Cleanup**: Stops and removes container on exit

## Prerequisites

- Docker installed and running
- NVIDIA Docker runtime (optional, for GPU mode)
- `jq` for JSON parsing
- `curl` for API testing

## Usage

### Basic Usage

```bash
./scripts/test_docker_deployment.sh
```

### With Custom Test Audio

```bash
TEST_AUDIO=/path/to/audio.wav ./scripts/test_docker_deployment.sh
```

### Environment Variables

- `TEST_AUDIO`: Path to test audio file (default: `tests/data/test_audio.wav`)
- Can be set inline or exported

## What It Tests

### 1. Docker Environment
- Docker availability
- NVIDIA Docker runtime detection
- Image build process

### 2. Container Lifecycle
- Container startup
- Port binding (5000:5000)
- Environment configuration
- GPU device mounting

### 3. Health Endpoints
- **Liveness** (`/health/live`): Basic service availability
- **Readiness** (`/health/ready`): Component initialization status
- **Health** (`/api/v1/health`): Comprehensive health information

### 4. GPU Status
- CUDA availability inside container
- GPU device information
- Memory usage
- Utilization metrics via `nvidia-smi`

### 5. API Endpoints
- `/api/v1/config` - Configuration information
- `/api/v1/speakers` - Available speakers
- `/api/v1/models/info` - Model information
- `/api/v1/convert` - Voice conversion (if test audio available)

## Output

The script provides colored output for easy reading:
- **GREEN**: Success messages
- **YELLOW**: Warnings (non-critical issues)
- **RED**: Errors (critical failures)

### Example Output

```
=== Docker Deployment Validation ===

[INFO] Checking Docker...
[INFO] Checking NVIDIA Docker runtime...
[INFO] NVIDIA Docker runtime detected
[INFO] Building Docker image: autovoice:validation
[INFO] Starting container...
[INFO] Container ID: abc123...

=== Running Health Checks ===

[INFO] Checking /health/live...
[INFO] ✓ Liveness check passed
[INFO] Checking /health/ready...
[INFO] ✓ Readiness check passed
{
  "status": "ready",
  "timestamp": "2025-10-28T...",
  "components": {
    "inference_engine": true,
    "audio_processor": true
  }
}

=== Checking GPU Status ===

[INFO] Querying /api/v1/gpu_status...
[INFO] ✓ CUDA is available in container
{
  "cuda_available": true,
  "device": "cuda",
  "device_count": 1,
  "device_name": "NVIDIA GeForce RTX 4090",
  "memory_total": 25757220864,
  "memory_allocated": 0,
  "memory_free": 25757220864
}

[INFO] Running nvidia-smi inside container...
0, NVIDIA GeForce RTX 4090, 0 %, 0 MiB, 24564 MiB
[INFO] ✓ GPU accessible from container

=== Validation Summary ===

[INFO] Container: autovoice_validation_12345
[INFO] Image: autovoice:validation
[INFO] API Port: 5000
[INFO] GPU Mode: Enabled

[INFO] All validation checks completed successfully! ✓
```

## Exit Codes

- **0**: All validation checks passed
- **1**: One or more validation checks failed

## Cleanup

The script automatically cleans up the container on exit (success or failure) using a trap handler. No manual cleanup required.

## Integration with CI/CD

This script can be integrated into CI/CD pipelines:

```yaml
# Example GitLab CI
test_docker:
  stage: test
  script:
    - ./scripts/test_docker_deployment.sh
  tags:
    - docker
    - gpu  # Optional: for GPU testing
```

```yaml
# Example GitHub Actions
- name: Test Docker Deployment
  run: ./scripts/test_docker_deployment.sh
```

## Troubleshooting

### Container Fails to Start
- Check Docker logs: `docker logs <container_id>`
- Verify Dockerfile builds successfully
- Check port 5000 is not in use

### GPU Not Available
- Verify NVIDIA Docker runtime installation
- Check GPU drivers: `nvidia-smi`
- Ensure `--gpus all` flag is supported

### Health Checks Failing
- Increase `MAX_WAIT` timeout in script
- Check application logs for initialization errors
- Verify all dependencies are in Docker image

### API Tests Failing
- Ensure services are fully initialized
- Check Flask app configuration
- Verify API routes are registered correctly

## Related Files

- `/home/kp/autovoice/Dockerfile` - Docker image definition
- `/home/kp/autovoice/docker-compose.yml` - Compose configuration
- `/home/kp/autovoice/src/auto_voice/web/api.py` - Health endpoints implementation

## Health Endpoint Specifications

### Liveness Probe: `/health/live`
```json
{
  "status": "live",
  "timestamp": 1698765432.123
}
```

Returns **200 OK** if service is running.

### Readiness Probe: `/health/ready`
```json
{
  "status": "ready",
  "timestamp": "2025-10-28T12:34:56.789Z",
  "components": {
    "inference_engine": true,
    "audio_processor": true
  }
}
```

Returns:
- **200 OK** if all components are ready
- **503 Service Unavailable** if components are initializing

### GPU Status: `/api/v1/gpu_status`
```json
{
  "cuda_available": true,
  "device": "cuda",
  "device_count": 1,
  "device_name": "NVIDIA GeForce RTX 4090",
  "memory_total": 25757220864,
  "memory_allocated": 1024000,
  "memory_reserved": 2048000,
  "memory_free": 25755148864
}
```

## Best Practices

1. **Run Before Deployment**: Execute this script before deploying to production
2. **CI/CD Integration**: Add as a required test stage in pipelines
3. **GPU Testing**: Test both CPU and GPU modes
4. **Load Testing**: Combine with load testing tools for production readiness
5. **Monitoring**: Monitor health endpoints in production with Kubernetes probes

## Future Enhancements

- Add load testing with concurrent requests
- Implement WebSocket connection testing
- Add performance benchmarking
- Support custom health check thresholds
- Generate detailed validation reports
