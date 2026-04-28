# AutoVoice Deployment Verification Checklist

Use this checklist as a reference for deployment verification. A deployment is
release-ready only within the support boundary defined in
[current-truth.md](./current-truth.md), and public/commercial launch requires
additional governance and current-head evidence.

## Pre-Deployment

### Docker Environment
- [ ] Docker installed and running: `docker --version`
- [ ] nvidia-docker runtime configured: `docker run --rm --runtime=nvidia nvidia/cuda:13.0-base nvidia-smi`
- [ ] Models downloaded to `models/pretrained/`:
  - [ ] `sovits5.0_main_1500.pth`
  - [ ] `hifigan_ljspeech.ckpt`
  - [ ] `hubert-soft-0d54a1f4.pt`

### Configuration
- [ ] `.env` file created from `.env.example`
- [ ] `SECRET_KEY` changed from default
- [ ] `HOST_PORT` configured (default: 10001)
- [ ] GPU device selection set: `CUDA_VISIBLE_DEVICES=0`
- [ ] Worker count tuned: `MAX_WORKERS=4`

## Build Verification

### Docker Build
```bash
# Build image
docker-compose build

# Expected output:
# - Multi-stage build completes
# - No errors in dependency installation
# - Image tagged as autovoice:latest
```

- [ ] Backend image built successfully
- [ ] Frontend image built successfully
- [ ] No build errors or warnings

### Image Inspection
```bash
# Check image size
docker images autovoice:latest

# Inspect image
docker inspect autovoice:latest | jq '.[0].Config.Env'
```

- [ ] Image size reasonable (<5GB)
- [ ] Environment variables set correctly
- [ ] CUDA_HOME and TORCH_CUDA_ARCH_LIST configured

## Startup Verification

### Container Start
```bash
# Start services
docker-compose up -d

# Check status
docker-compose ps

# Expected: backend and frontend "Up" and healthy
```

- [ ] Backend container starts without errors
- [ ] Frontend container starts without errors
- [ ] Both containers show "healthy" status after start_period (60s)

### GPU Access
```bash
# Verify GPU is accessible
docker exec autovoice-backend nvidia-smi

# Check PyTorch CUDA
docker exec autovoice-backend python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

- [ ] nvidia-smi shows GPU information
- [ ] PyTorch detects CUDA
- [ ] Correct GPU device name displayed

### Logs
```bash
# Check startup logs
docker-compose logs backend | head -50

# Look for:
# - "AutoVoice Flask app created"
# - "KaraokeManager initialized"
# - "Singing conversion pipeline initialized"
# - "Voice cloner initialized"
# - "JobManager initialized"
# - "Starting AutoVoice on 0.0.0.0:5000"
```

- [ ] No errors in startup logs
- [ ] All ML components initialized
- [ ] Server started successfully

## Endpoint Verification

### Health Check
```bash
curl -s http://localhost:10001/api/v1/health | jq

# Expected response:
# {
#   "status": "healthy",
#   "timestamp": "...",
#   "components": {
#     "api": {"status": "up"},
#     "torch": {"status": "up", "cuda": true, ...},
#     "voice_cloner": {"status": "up"},
#     "singing_pipeline": {"status": "up"},
#     "job_manager": {"status": "up"}
#   },
#   "cuda_kernels_available": true,
#   "version": "0.1.0"
# }
```

- [ ] `/health` returns 200 OK
- [ ] `status` is "healthy"
- [ ] All components show "up"
- [ ] `cuda` is true

### Readiness Check
```bash
curl -s http://localhost:10001/api/v1/ready | jq

# Expected response:
# {
#   "ready": true,
#   "timestamp": "...",
#   "components": {
#     "torch": true,
#     "voice_cloner": true,
#     "singing_pipeline": true
#   }
# }
```

- [ ] `/ready` returns 200 OK
- [ ] `ready` is true
- [ ] All components are true

### GPU Metrics
```bash
curl -s http://localhost:10001/api/v1/gpu/metrics | jq

# Expected: GPU device info with memory usage
```

- [ ] `/gpu/metrics` returns GPU information
- [ ] Memory usage shown in GB
- [ ] Utilization and temperature available (if pynvml installed)

### Prometheus Metrics
```bash
curl -s http://localhost:10001/api/v1/metrics | head -20

# Expected: Prometheus text format metrics
# HELP autovoice_conversions_total ...
# TYPE autovoice_conversions_total counter
# ...
```

- [ ] `/metrics` returns Prometheus format
- [ ] Metrics include conversion, GPU, HTTP request counters
- [ ] No errors in metric export

## Monitoring Stack (Optional)

### Start Monitoring
```bash
# Start with monitoring profile
docker-compose --profile monitoring up -d

# Verify services
docker-compose ps
```

- [ ] Prometheus container started
- [ ] Grafana container started
- [ ] Both containers healthy

### Prometheus
```bash
# Check Prometheus targets
curl -s http://localhost:10002/api/v1/targets | jq '.data.activeTargets[] | {job, health, lastError}'

# Expected: autovoice target "up"
```

- [ ] Prometheus accessible at port 10002
- [ ] AutoVoice target scraping successfully
- [ ] No scrape errors

### Grafana
```bash
# Access Grafana
open http://localhost:10003

# Login with admin/admin (or configured credentials)
# Navigate to Dashboards > AutoVoice Monitoring
```

- [ ] Grafana accessible at port 10003
- [ ] Login successful
- [ ] AutoVoice dashboard loads
- [ ] Panels show data (GPU memory, conversions, etc.)

## Functional Testing

### Voice Cloning
```bash
# Test voice cloning endpoint (requires test audio file)
curl -X POST http://localhost:10001/api/v1/voice/clone \
  -F "profile_name=test" \
  -F "audio=@test_audio.wav" \
  -v

# Expected: 200 OK with profile created
```

- [ ] Voice cloning endpoint accessible
- [ ] Profile creation successful
- [ ] No GPU memory errors

### Voice Conversion
```bash
# Test conversion (requires profile and audio)
curl -X POST http://localhost:10001/api/v1/convert/singing \
  -F "source_audio=@source.wav" \
  -F "target_profile_id=test" \
  -v

# Expected: Job ID returned
```

- [ ] Conversion job submitted successfully
- [ ] Job status queryable
- [ ] Conversion completes without errors

### Frontend
```bash
# Access frontend
open http://localhost:3000
```

- [ ] Frontend loads successfully
- [ ] API communication working
- [ ] UI functional (upload, convert, etc.)

## Resilience Testing

### Graceful Shutdown
```bash
# Stop container gracefully
docker-compose stop backend

# Check logs for shutdown message
docker-compose logs backend | tail -20

# Expected:
# - "Received signal 15, initiating graceful shutdown..."
# - "Stopping job manager..."
# - "Clearing GPU memory..."
# - "Shutdown complete"
```

- [ ] Container stops gracefully
- [ ] GPU memory cleaned up
- [ ] No force-kill required

### Restart After Crash
```bash
# Kill container (simulate crash)
docker kill autovoice-backend

# Wait for auto-restart
sleep 10

# Check status
docker-compose ps

# Expected: Container restarted automatically
```

- [ ] Container restarts automatically
- [ ] Health check passes after restart
- [ ] ML components re-initialize

### Resource Limits
```bash
# Check resource usage
docker stats autovoice-backend --no-stream

# Monitor during conversion
watch -n 1 docker stats autovoice-backend
```

- [ ] Memory usage stays within limits (32GB max)
- [ ] No OOM kills
- [ ] GPU memory managed correctly

## Performance Validation

### Latency
```bash
# Measure health check latency
time curl http://localhost:10001/api/v1/health > /dev/null

# Expected: < 500ms
```

- [ ] Health check responds quickly (<500ms)
- [ ] Ready check responds quickly (<500ms)
- [ ] Metrics endpoint responds (<1s)

### Throughput
```bash
# Run performance tests
docker exec autovoice-backend pytest tests/ -m performance -v
```

- [ ] Performance tests pass
- [ ] RTF (Real-Time Factor) < 1.0 for real-time conversion
- [ ] GPU utilization >50% during conversion

## Security Validation

### Configuration
- [ ] `SECRET_KEY` is not default value
- [ ] Grafana admin password changed
- [ ] CORS origins restricted (not `*` in production)
- [ ] Container runs as non-root user

### Network
- [ ] Ports properly exposed (10001, 10002, 10003)
- [ ] Internal network isolated (autovoice-net)
- [ ] HTTPS reverse proxy configured (if external access)

### Data
- [ ] Persistent volumes configured
- [ ] Model files mounted read-only
- [ ] Secrets not in environment variables (use .env)

## Documentation

- [ ] Deployment guide reviewed
- [ ] Team trained on monitoring dashboards
- [ ] Runbook created for common issues
- [ ] Backup and restore procedures documented

## Production Readiness Checklist

### Critical (Must Pass)
- [ ] Health and readiness endpoints working
- [ ] GPU access verified
- [ ] ML components initialized
- [ ] Graceful shutdown working
- [ ] Auto-restart configured

### Important (Should Pass)
- [ ] Monitoring stack deployed
- [ ] Prometheus scraping metrics
- [ ] Grafana dashboards configured
- [ ] Resource limits set
- [ ] Logs configured with rotation

### Nice-to-Have (Optional)
- [ ] Systemd service for auto-start on boot
- [ ] Alert rules configured
- [ ] Load testing completed
- [ ] Backup automation set up

## Sign-off

Date: _______________

Verified by: _______________

Notes:
_______________________________________________
_______________________________________________
_______________________________________________
