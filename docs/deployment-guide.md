# AutoVoice Production Deployment Guide

This guide covers deploying AutoVoice in production on Jetson Thor with Docker and monitoring.

## Prerequisites

- Jetson Thor device (aarch64, CUDA 13.0, 64GB GPU)
- Docker installed with nvidia-docker runtime
- Pre-trained models downloaded to `models/pretrained/`

## Installation

### Docker Runtime Setup

Ensure nvidia-docker runtime is configured:

```bash
# Test nvidia-docker
nvidia-docker run --rm nvidia/cuda:13.0-base nvidia-smi

# If not working, install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Optional: Auto-start on Boot

To start AutoVoice automatically on system boot:

```bash
# Copy systemd service file
sudo cp config/systemd/autovoice.service /etc/systemd/system/

# Enable service
sudo systemctl enable autovoice.service

# Start now
sudo systemctl start autovoice.service

# Check status
sudo systemctl status autovoice.service
```

## Quick Start

### 1. Environment Configuration

Copy the environment template and customize:

```bash
cp .env.example .env
# Edit .env with your configuration
```

Key settings to verify:
- `HOST_PORT` - Web server port (default: 10001)
- `SECRET_KEY` - Change from default in production
- `CUDA_VISIBLE_DEVICES` - GPU selection
- `MAX_WORKERS` - Concurrent job workers

### 2. Build and Start

```bash
# Build the image
docker-compose build

# Start the service
docker-compose up -d

# Check logs
docker-compose logs -f autovoice
```

### 3. Verify Deployment

```bash
# Health check
curl http://localhost:10001/api/v1/health

# Readiness check
curl http://localhost:10001/api/v1/ready

# GPU metrics
curl http://localhost:10001/api/v1/gpu/metrics
```

> **Note**: If you encounter errors during deployment, see the [Troubleshooting Guide](troubleshooting.md) for common issues and solutions.

## Monitoring Setup

### Enable Prometheus and Grafana

```bash
# Start with monitoring stack
docker-compose --profile monitoring up -d

# Access Grafana
open http://localhost:10003
# Default credentials: admin/admin (change after first login)
```

### Metrics Endpoints

- **Application**: http://localhost:10001/api/v1/metrics
- **Prometheus**: http://localhost:10002
- **Grafana**: http://localhost:10003

### Grafana Dashboard

The AutoVoice dashboard includes:
- GPU memory usage and utilization
- Conversion rate and duration (p50/p95)
- Job queue status
- HTTP request metrics

## Production Checklist

### Security

- [ ] Change `SECRET_KEY` in `.env`
- [ ] Change Grafana admin password
- [ ] Configure CORS origins (restrict `CORS_ORIGINS`)
- [ ] Enable API rate limiting
- [ ] Review firewall rules for exposed ports
- [ ] Use HTTPS reverse proxy (nginx/traefik)

### Performance

- [ ] Verify GPU memory limits (`MAX_MEMORY_FRACTION=0.9`)
- [ ] Tune worker count (`MAX_WORKERS=4` for 64GB GPU)
- [ ] Configure job TTL settings
- [ ] Set resource limits in docker-compose.yml

### Reliability

- [ ] Test health check endpoints (`/health`, `/ready`)
- [ ] Verify graceful shutdown (SIGTERM handling)
- [ ] Configure automatic restart (`restart: unless-stopped`)
- [ ] Set up log rotation (max-size: 10m, max-file: 3)
- [ ] Test container restart and GPU cleanup

### Monitoring

- [ ] Prometheus scraping AutoVoice metrics
- [ ] Grafana dashboard configured
- [ ] Alert rules for GPU memory/utilization
- [ ] Alert rules for error rates
- [ ] Log aggregation (optional: ELK/Loki)

## Docker Commands

### Basic Operations

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Restart AutoVoice
docker-compose restart autovoice

# View logs (last 100 lines, follow)
docker-compose logs -f --tail=100 autovoice

# Check container status
docker-compose ps
```

### Resource Management

```bash
# GPU access verification
docker exec autovoice nvidia-smi

# GPU memory from inside container
docker exec autovoice python -c "import torch; print(torch.cuda.get_device_properties(0))"

# Container resource usage
docker stats autovoice
```

### Debugging

```bash
# Shell access
docker exec -it autovoice /bin/bash

# Run tests inside container
docker exec autovoice pytest tests/ -v

# Check Python environment
docker exec autovoice pip list
```

## Scaling

### Horizontal Scaling (Multiple Instances)

For load balancing across multiple GPUs:

```bash
# Scale to 2 instances (requires 2 GPUs)
docker-compose up -d --scale autovoice=2

# Use nginx/traefik for load balancing
```

### Vertical Scaling (Resource Limits)

Edit `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 32G  # Increase for larger models
    reservations:
      memory: 8G
```

## Backup and Restore

### Data Persistence

Persistent volumes:
- `autovoice-profiles` - Voice profiles
- `autovoice-uploads` - User uploads
- `autovoice-outputs` - Conversion outputs
- `autovoice-logs` - Application logs

```bash
# Backup volumes
docker run --rm -v autovoice-profiles:/data -v $(pwd):/backup \
  ubuntu tar czf /backup/profiles-backup.tar.gz /data

# Restore volumes
docker run --rm -v autovoice-profiles:/data -v $(pwd):/backup \
  ubuntu tar xzf /backup/profiles-backup.tar.gz -C /
```

### Model Files

```bash
# Models are mounted read-only from host
# Backup the models directory
tar czf models-backup.tar.gz models/pretrained/
```

## Troubleshooting

> **Comprehensive Guide**: For detailed troubleshooting steps, error solutions, and diagnostic workflows, see the [Troubleshooting Guide](troubleshooting.md).

This section covers quick deployment-specific issues. For GPU errors, model loading failures, audio processing issues, and more, refer to the full troubleshooting guide.

### Container Won't Start

```bash
# Check logs
docker-compose logs autovoice

# Verify GPU access
nvidia-docker run --rm nvidia/cuda:13.0-base nvidia-smi

# Check runtime configuration
docker info | grep nvidia
```

### GPU Not Detected

```bash
# Verify nvidia-docker runtime
docker run --rm --runtime=nvidia nvidia/cuda:13.0-base nvidia-smi

# Check CUDA in container
docker exec autovoice python -c "import torch; print(torch.cuda.is_available())"
```

### Health Check Failing

```bash
# Check health status
docker inspect autovoice | jq '.[0].State.Health'

# Manual health check
curl -v http://localhost:10001/api/v1/health

# Check component status
docker exec autovoice python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Memory Issues

```bash
# Check GPU memory
docker exec autovoice nvidia-smi

# Clear GPU cache
docker exec autovoice python -c "import torch; torch.cuda.empty_cache()"

# Restart with memory cleanup
docker-compose restart autovoice
```

## Updates and Maintenance

### Updating the Application

```bash
# Pull latest code
git pull

# Rebuild image
docker-compose build

# Restart with new image
docker-compose up -d
```

### Model Updates

```bash
# Download new models to models/pretrained/
# Update config/gpu_config.yaml with new checkpoint names
# Restart container
docker-compose restart autovoice
```

### Database Migrations (if using external DB)

```bash
# Run migrations
docker exec autovoice alembic upgrade head
```

## Performance Tuning

### Jetson Thor Specific

- **GPU Memory Fraction**: Set `MAX_MEMORY_FRACTION=0.9` for 64GB GPU
- **Worker Count**: Start with `MAX_WORKERS=4`, monitor GPU utilization
- **CUDA Architecture**: Verify `TORCH_CUDA_ARCH_LIST=11.0` for SM 11.0
- **Memory Cleanup**: Graceful shutdown handles GPU cache clearing

### Benchmarking

```bash
# Run performance tests
docker exec autovoice pytest tests/ -m performance -v

# Monitor during load
watch -n 1 'docker exec autovoice nvidia-smi'
```

## Support

For issues:
1. Check the [Troubleshooting Guide](troubleshooting.md) for common errors and solutions
2. Check logs: `docker-compose logs -f autovoice`
3. Verify health: `curl http://localhost:10001/api/v1/health`
4. Check GPU: `docker exec autovoice nvidia-smi`
5. Review metrics: http://localhost:10003 (Grafana)
