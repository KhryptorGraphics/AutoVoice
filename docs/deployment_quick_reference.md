# AutoVoice Deployment Quick Reference Card

**Version**: 1.0 | **Last Updated**: 2025-10-27

This quick reference provides essential commands and checks for deploying AutoVoice.

---

## Pre-flight Checks ✈️

```bash
# 1. Check GPU
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
# Required: Compute capability ≥ 7.0, Driver 535+

# 2. Check CUDA
nvcc --version
# Required: CUDA 11.8+ or 12.1 recommended

# 3. Check PyTorch
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
# Required: CUDA available = True

# 4. Check Python
python --version
# Required: 3.8, 3.9, or 3.10

# 5. Check Build Tools
gcc --version && cmake --version && ninja --version
# Required: gcc 7+, cmake 3.10+, ninja 1.10+
```

---

## Quick Install (5 minutes)

### Method 1: Docker (Recommended)

```bash
# Pull and run
docker run -d \
    --name autovoice \
    --gpus all \
    -p 5000:5000 \
    -v $(pwd)/models:/app/models \
    -e LOG_LEVEL=INFO \
    autovoice:latest

# Health check
curl http://localhost:5000/health
```

### Method 2: From Source

```bash
# Clone
git clone https://github.com/autovoice/autovoice.git
cd autovoice

# Install PyTorch with CUDA
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Build CUDA extensions
./scripts/build.sh

# Test
./scripts/test.sh

# Run
python main.py
```

---

## Essential Commands

### Build

```bash
# Standard build
python setup.py build_ext --inplace

# Build for specific GPU
TORCH_CUDA_ARCH_LIST="80;86" python setup.py build_ext --inplace

# Clean and rebuild
python setup.py clean --all
python setup.py build_ext --inplace

# Verbose build (for debugging)
python setup.py build_ext --inplace --verbose
```

### Test

```bash
# Full test suite
./scripts/test.sh

# Quick smoke test
python -c "from auto_voice.audio.processor import AudioProcessor; print('OK')"

# GPU test
python -c "import torch; from auto_voice.audio.processor import AudioProcessor; p = AudioProcessor(device='cuda' if torch.cuda.is_available() else 'cpu'); print(f'Device: {p.device}')"

# Run pytest
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/auto_voice --cov-report=html
```

### Docker

```bash
# Build image
docker build -t autovoice:latest .

# Run with GPU
docker run --gpus all -p 5000:5000 autovoice:latest

# Run with environment variables
docker run --gpus all -p 5000:5000 \
    -e LOG_LEVEL=DEBUG \
    -e CUDA_VISIBLE_DEVICES=0 \
    autovoice:latest

# View logs
docker logs -f autovoice

# Shell into container
docker exec -it autovoice bash

# Stop and remove
docker stop autovoice && docker rm autovoice
```

### Monitoring

```bash
# Health check
curl http://localhost:5000/health/live
curl http://localhost:5000/health/ready

# Metrics
curl http://localhost:5000/metrics

# GPU monitoring
watch -n 1 nvidia-smi

# Detailed GPU metrics
nvidia-smi dmon -s pucvmet

# Application logs
docker logs -f autovoice | jq '.'  # Pretty-print JSON logs
```

---

## Environment Variables

```bash
# Core
export CUDA_HOME=/usr/local/cuda
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="70;75;80;86;89"

# Application
export LOG_LEVEL=INFO                # DEBUG, INFO, WARNING, ERROR
export LOG_FORMAT=json               # json, text
export FLASK_ENV=production          # development, production
export PROMETHEUS_ENABLED=true       # Enable metrics

# Build
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

---

## Troubleshooting Quick Fixes

### "CUDA not available"

```bash
# Check driver
nvidia-smi
# If fails, install: sudo apt-get install -y nvidia-driver-535

# Check CUDA toolkit
nvcc --version
# If fails, install CUDA toolkit

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

### "nvcc not found"

```bash
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
```

### "CUDA version mismatch"

```bash
# Check versions
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"
nvcc --version

# Install matching PyTorch
# For CUDA 12.1:
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8:
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

### "Out of memory"

```bash
# Check GPU memory
nvidia-smi

# Use smaller batch size (edit config)
# Or enable CPU fallback
export AUTOVOICE_CPU_FALLBACK=true
```

### "libcudart.so not found"

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# Make permanent: Add to ~/.bashrc
```

### Docker can't access GPU

```bash
# Test nvidia-docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If fails, reinstall nvidia-docker
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

## Performance Tuning

### GPU Optimization

```bash
# Enable TensorRT (if installed)
export AUTOVOICE_TENSORRT_ENABLED=true

# Set optimal batch size (edit config)
# RTX 3090: batch_size=32
# A100: batch_size=64

# Use specific GPU
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0 only

# Multiple GPUs (not fully supported yet)
export CUDA_VISIBLE_DEVICES=0,1  # Use GPUs 0 and 1
```

### Application Tuning

```bash
# Increase workers (for CPU-bound tasks)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:5000 main:app

# Adjust log level (reduce I/O)
export LOG_LEVEL=WARNING

# Disable metrics (slight performance gain)
export PROMETHEUS_ENABLED=false
```

---

## Production Checklist

### Pre-deployment

- [ ] GPU compute capability ≥ 7.0
- [ ] NVIDIA driver 535+ installed
- [ ] CUDA 11.8+ or 12.1 installed
- [ ] PyTorch CUDA version matches system CUDA
- [ ] Build completes without errors
- [ ] All tests pass (`./scripts/test.sh`)
- [ ] Health check responds (`/health`)

### Security

- [ ] Running as non-root user
- [ ] No hardcoded secrets
- [ ] TLS/SSL enabled for public endpoints
- [ ] Rate limiting configured
- [ ] Security scanning passed (Trivy)

### Monitoring

- [ ] Health checks configured (`/health/live`, `/health/ready`)
- [ ] Metrics endpoint accessible (`/metrics`)
- [ ] Prometheus scraping working
- [ ] Grafana dashboards loaded
- [ ] Alerts configured (high error rate, GPU OOM)

### Documentation

- [ ] Deployment runbook updated
- [ ] Rollback procedure documented
- [ ] On-call contacts listed
- [ ] Known issues documented

---

## Load Testing

```bash
# Install tools
sudo apt-get install -y apache2-utils

# Simple load test
ab -n 100 -c 10 -p request.json -T application/json http://localhost:5000/api/v1/synthesize

# request.json:
# {"text": "Load test", "speaker_id": "default"}

# Monitor during load test
watch -n 1 nvidia-smi
```

---

## Rollback Procedure

### Docker Rollback

```bash
# 1. Stop current
docker stop autovoice

# 2. Start previous version
docker run -d --name autovoice --gpus all -p 5000:5000 autovoice:v1.0.0

# 3. Verify
curl http://localhost:5000/health
```

### Source Rollback

```bash
# 1. Checkout previous version
git fetch --tags
git checkout v1.0.0

# 2. Rebuild
pip uninstall -y auto_voice
pip install -e .

# 3. Restart
systemctl restart autovoice
```

---

## Monitoring Queries

### Prometheus Queries

```promql
# Request rate
rate(http_requests_total[5m])

# Error rate
rate(http_requests_total{status=~"5.."}[5m])

# P95 latency
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# GPU utilization
nvidia_gpu_utilization_percentage

# GPU memory usage
nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes
```

### Alert Thresholds

```yaml
# High error rate
rate(http_requests_total{status=~"5.."}[5m]) > 0.05  # 5% errors

# High latency
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5  # 500ms

# GPU memory
nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes > 0.9  # 90% full

# Service down
up{job="autovoice"} == 0
```

---

## Support Resources

| Resource | Location |
|----------|----------|
| **Full Deployment Guide** | `/home/kp/autovoice/docs/deployment_guide.md` |
| **Production Checklist** | `/home/kp/autovoice/docs/production_readiness_checklist.md` |
| **Troubleshooting** | `/home/kp/autovoice/docs/deployment_guide.md#troubleshooting` |
| **GitHub Issues** | https://github.com/autovoice/autovoice/issues |
| **Documentation** | `/home/kp/autovoice/README.md` |

---

## Common Ports

| Port | Service | Protocol |
|------|---------|----------|
| 5000 | AutoVoice API & Metrics | HTTP |
| 8080 | WebSocket Streaming | WS |
| 9090 | Prometheus | HTTP |
| 3000 | Grafana | HTTP |

---

## Version Compatibility

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.8, 3.9, 3.10 | Tested in CI |
| PyTorch | 2.0.0 - 2.2.0 | With CUDA support |
| CUDA | 11.8+ or 12.1 | 12.1 recommended |
| Driver | 525+ | 535+ recommended |
| cuDNN | 8.6.0+ | Bundled with PyTorch |

---

## One-liner Commands

```bash
# Full setup and test
git clone https://github.com/autovoice/autovoice.git && cd autovoice && pip install -r requirements.txt && ./scripts/build.sh && ./scripts/test.sh

# Docker deploy with monitoring
docker-compose --profile monitoring up -d

# Check everything is working
curl -f http://localhost:5000/health && nvidia-smi && docker ps

# View all logs
docker-compose logs -f

# Complete teardown
docker-compose down -v && docker rmi autovoice:latest

# Emergency stop
docker stop $(docker ps -q --filter ancestor=autovoice:latest)
```

---

**Keep this card handy for quick deployment and troubleshooting!**

---

**Document Version**: 1.0
**For**: AutoVoice v0.1.0
**Last Updated**: 2025-10-27
