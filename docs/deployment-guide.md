# AutoVoice Deployment Guide

**Version**: 1.0 | **Last Updated**: 2025-11-01

This comprehensive guide provides step-by-step instructions for deploying AutoVoice in production environments, from prerequisites to monitoring and troubleshooting.

## Table of Contents

1. [Introduction & Overview](#introduction--overview)
2. [Prerequisites & Requirements](#prerequisites--requirements)
3. [Pre-Deployment Validation](#pre-deployment-validation)
4. [Installation Methods](#installation-methods)
5. [Cloud Provider Deployments](#cloud-provider-deployments)
6. [Configuration](#configuration)
7. [Security Hardening](#security-hardening)
8. [Monitoring & Observability](#monitoring--observability)
9. [Testing the Deployment](#testing-the-deployment)
10. [Troubleshooting](#troubleshooting)
11. [Performance Optimization](#performance-optimization)
12. [Rollback Procedures](#rollback-procedures)
13. [Production Checklist](#production-checklist)
14. [Monitoring Queries & Alerts](#monitoring-queries--alerts)
15. [Support & Resources](#support--resources)

---

## Introduction & Overview

### Purpose and Scope

This deployment guide is designed for DevOps engineers, SREs, and developers responsible for deploying and maintaining AutoVoice in production environments. It covers:

- **Hardware and software prerequisites**
- **Multiple deployment methods** (Docker, source, cloud providers)
- **Security best practices** and hardening
- **Monitoring and observability** setup
- **Performance optimization** strategies
- **Troubleshooting** common issues

### Target Audience

- **DevOps Engineers**: Responsible for infrastructure and deployment automation
- **Site Reliability Engineers (SREs)**: Ensuring system reliability and performance
- **Developers**: Building and integrating AutoVoice into applications

### Deployment Options Overview

AutoVoice supports three primary deployment methods:

1. **Docker Deployment** (Recommended): Containerized deployment with GPU support
2. **From Source**: Direct installation on host system
3. **Cloud Providers**: AWS, GCP, Azure with managed services

---

## Prerequisites & Requirements

### Hardware Requirements

**GPU Requirements:**
- **Minimum**: NVIDIA GPU with compute capability 7.0+ (Volta architecture)
- **Recommended**: Compute capability 8.0+ (Ampere) or 8.9 (Ada Lovelace)
- **VRAM**:
  - Minimum: 8GB for basic TTS
  - Recommended: 16GB+ for voice conversion
  - Enterprise: 24GB+ for high-volume production

**Supported GPU Models:**
- **Budget/Development**: NVIDIA T4 (16GB VRAM, compute 7.5)
- **Production**: NVIDIA RTX 3090/4090 (24GB VRAM, compute 8.6/8.9)
- **Enterprise**: NVIDIA A100 (40GB/80GB VRAM, compute 8.0)

**CPU and Memory:**
- **CPU**: 4+ cores recommended (8+ for high concurrency)
- **RAM**: 16GB minimum, 32GB+ recommended
- **Storage**: 50GB+ for models and data

### Software Requirements

**Operating System:**
- Ubuntu 20.04 LTS or 22.04 LTS (recommended)
- Debian 11+
- CentOS 8+ / RHEL 8+
- Other Linux distributions with CUDA support

**NVIDIA Driver:**
- **Minimum**: 525+ (for CUDA 11.8)
- **Recommended**: 535+ (for CUDA 12.1)

**CUDA Toolkit:**
- **Minimum**: 11.8
- **Recommended**: 12.1

**Python:**
- **Supported**: 3.8, 3.9, 3.10, 3.11, 3.12
- **Recommended**: 3.10 or 3.11

**PyTorch:**
- **Minimum**: 2.0.0 with CUDA support
- **Recommended**: 2.2.x (supports 2.0–2.2)

**Docker (for containerized deployment):**
- Docker 20.10+ with nvidia-docker2 runtime
- Docker Compose 1.29+ (optional, for multi-service deployment)

### Network Requirements

**Required Ports:**
- **5000**: Main application API (HTTP) and WebSocket (Socket.IO)
- **9090**: Prometheus metrics (monitoring)
- **3000**: Grafana dashboard (monitoring, optional)

**Note**: WebSocket connections use Socket.IO on the same port as the HTTP API (default 5000). No separate WebSocket port is required.

**Firewall Configuration:**
```bash
# Allow application port
sudo ufw allow 5000/tcp

# Allow monitoring ports (if using Prometheus/Grafana)
sudo ufw allow 9090/tcp
sudo ufw allow 3000/tcp
```

### Access Requirements

**System Access:**
- sudo/root access for driver and CUDA installation
- docker group membership for Docker deployment
- Network access to PyPI and Docker Hub

**Cloud Access (if deploying to cloud):**
- AWS: IAM credentials with EC2, ECS permissions
- GCP: Service account with Compute Engine permissions
- Azure: Subscription with VM creation permissions

---

## Pre-Deployment Validation

Before deploying AutoVoice, validate that all prerequisites are met.

### GPU Validation

**Check GPU availability:**
```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
```

**Expected output:**
```
name, compute_cap, driver_version
NVIDIA RTX 3090, 8.6, 535.129.03
```

**Requirements:**
- Compute capability ≥ 7.0
- Driver version ≥ 535 (or 525+ for CUDA 11.8)


### CUDA Validation

**Check CUDA installation:**
```bash
nvcc --version
```

**Expected output:**
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Aug_15_22:02:13_PDT_2023
Cuda compilation tools, release 12.1, V12.1.105
```

**Requirements:**
- CUDA 11.8+ or 12.1+ installed
- nvcc accessible in PATH

**If CUDA is not installed:**
```bash
# Install CUDA Toolkit 12.1 (Ubuntu)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-12-1

# Add to PATH
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
```

### PyTorch Validation

**Check PyTorch CUDA support:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

**Expected output:**
```
PyTorch: 2.2.2+cu121
CUDA available: True
CUDA version: 12.1
```

**If PyTorch is not installed or CUDA is not available:**
```bash
# Install PyTorch with CUDA 12.1 support (recommended)
python -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu121

# Or for CUDA 11.8
python -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu118
```

### Build Tools Validation

**Check build tools:**
```bash
gcc --version
cmake --version
ninja --version
```

**Requirements:**
- gcc 7+ (9+ recommended)
- cmake 3.10+ (3.18+ recommended)
- ninja 1.10+

**Install build tools (Ubuntu):**
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build
```

---

## Installation Methods

### Method 1: Docker Deployment (Recommended)

Docker deployment is the recommended method for production as it provides:
- Consistent environment across deployments
- Easy rollback and version management
- Isolation from host system
- Simplified dependency management

#### Step 1: Install Docker and NVIDIA Container Toolkit

**1.1 Install Docker:**
```bash
# Install Docker using the official script
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to the docker group (requires logout/login to take effect)
sudo usermod -aG docker $USER

# Verify Docker installation
docker --version
```

**1.2 Install NVIDIA Container Toolkit:**

The NVIDIA Container Toolkit enables Docker containers to access NVIDIA GPUs. This replaces the older `nvidia-docker2` package.

```bash
# Configure the repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
   && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install the NVIDIA Container Toolkit packages
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Verify installation
nvidia-ctk --version
```

**1.3 Configure Docker to use NVIDIA runtime:**

```bash
# Configure the Docker daemon to use nvidia runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker to apply changes
sudo systemctl restart docker

# Verify Docker can see the nvidia runtime
docker info | grep -i runtime
# Should show: Runtimes: nvidia runc
```

**1.4 Validate GPU access in Docker:**

```bash
# Test GPU access with CUDA base image
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Expected output: nvidia-smi output showing your GPU(s)
# If this fails, check:
# - NVIDIA drivers are installed: nvidia-smi (on host)
# - Docker daemon was restarted after toolkit installation
# - User is in docker group: groups $USER
```

**Troubleshooting GPU access:**

If `docker run --gpus all` fails with "could not select device driver":
```bash
# Check NVIDIA driver on host
nvidia-smi

# Reconfigure and restart
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Check Docker daemon logs
sudo journalctl -u docker.service | tail -50
```

#### Step 2: Pull AutoVoice Docker Image

```bash
# Pull latest image
docker pull autovoice/autovoice:latest

# Or pull specific version
docker pull autovoice/autovoice:v1.0.0
```

#### Step 3: Run AutoVoice Container

```bash
# Run with GPU support
docker run -d \
  --name autovoice \
  --gpus all \
  -p 5000:5000 \
  -p 9090:9090 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -e LOG_LEVEL=INFO \
  -e CUDA_VISIBLE_DEVICES=0 \
  --restart unless-stopped \
  autovoice/autovoice:latest

# Check logs
docker logs -f autovoice
```

#### Step 4: Validate Deployment

```bash
# Health check with pretty printing
curl http://localhost:5000/health | jq .

# Expected response includes nested components and system info:
# {
#   "status": "healthy",
#   "components": {
#     "gpu_available": true,
#     "model_loaded": true,
#     "api": true,
#     "synthesizer": true,
#     "voice_cloner": true,
#     "singing_conversion_pipeline": true
#   },
#   "system": {
#     "memory_percent": 45.2,
#     "cpu_percent": 12.5,
#     "gpu": {"available": true, "device_count": 1}
#   }
# }

# Test TTS endpoint
curl -X POST http://localhost:5000/api/v1/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a test.", "speaker_id": 0}'
```

### Method 2: From Source

Installing from source provides more control and is useful for development or custom deployments.

#### Step 1: Clone Repository

```bash
git clone https://github.com/autovoice/autovoice.git
cd autovoice
```

#### Step 2: Install PyTorch with CUDA Support

```bash
# Install PyTorch with CUDA 12.1 (recommended)
python -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu121

# Or for CUDA 11.8
python -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu118
```

#### Step 3: Install Dependencies

```bash
# Install Python dependencies
python -m pip install -r requirements.txt
```

#### Step 4: Build CUDA Extensions

```bash
# Build CUDA extensions
python -m pip install -e .

# Or use the build script
./scripts/build.sh
```

#### Step 5: Run Tests

```bash
# Run full test suite
./scripts/test.sh

# Or run pytest directly
pytest tests/ -v
```

#### Step 6: Start Application

```bash
# Start application
python main.py

# Or use the run script
./scripts/run.sh
```

### Method 3: Docker Compose with Monitoring

For production deployments with full monitoring stack (Prometheus + Grafana).

#### Step 1: Create docker-compose.yml

```yaml
version: '3.8'

services:
  autovoice:
    image: autovoice/autovoice:latest
    container_name: autovoice
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ports:
      - "5000:5000"
      - "9090:9090"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - LOG_LEVEL=INFO
      - CUDA_VISIBLE_DEVICES=0
      - PROMETHEUS_ENABLED=true
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  redis:
    image: redis:7-alpine
    container_name: autovoice-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    container_name: autovoice-prometheus
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: autovoice-grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    restart: unless-stopped
    depends_on:
      - prometheus

volumes:
  redis-data:
  prometheus-data:
  grafana-data:
```

#### Step 2: Create prometheus.yml

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'autovoice'
    static_configs:
      - targets: ['autovoice:9090']
    metrics_path: '/metrics'
```

#### Step 3: Start Services

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f autovoice

# Stop services
docker-compose down
```

#### Step 4: Access Services

- **AutoVoice API**: http://localhost:5000
- **Prometheus**: http://localhost:9091
- **Grafana**: http://localhost:3000 (admin/admin)

---

## Cloud Provider Deployments

### AWS Deployment

#### Option 1: EC2 with GPU

**Step 1: Launch EC2 Instance**

```bash
# Launch p3.2xlarge instance (V100 GPU)
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type p3.2xlarge \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxx \
  --subnet-id subnet-xxxxxxxx \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=autovoice-prod}]'
```

**Recommended Instance Types:**
- **Development**: g4dn.xlarge (T4, 16GB VRAM) - $0.526/hr
- **Production**: p3.2xlarge (V100, 16GB VRAM) - $3.06/hr
- **High-Performance**: p4d.24xlarge (8x A100, 320GB VRAM) - $32.77/hr

**Step 2: Install NVIDIA Drivers and CUDA**

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@<instance-ip>

# Install NVIDIA drivers
sudo apt-get update
sudo apt-get install -y nvidia-driver-535

# Reboot
sudo reboot

# Verify GPU
nvidia-smi
```

**Step 3: Deploy AutoVoice**

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install nvidia-docker2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Run AutoVoice
docker run -d --name autovoice --gpus all -p 5000:5000 autovoice/autovoice:latest
```

#### Option 2: ECS with GPU

**Step 1: Create ECS Cluster**

```bash
# Create cluster
aws ecs create-cluster --cluster-name autovoice-cluster

# Register task definition (see task-definition.json below)
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

**task-definition.json:**
```json
{
  "family": "autovoice",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["EC2"],
  "cpu": "4096",
  "memory": "16384",
  "containerDefinitions": [
    {
      "name": "autovoice",
      "image": "autovoice/autovoice:latest",
      "portMappings": [
        {
          "containerPort": 5000,
          "protocol": "tcp"
        }
      ],
      "resourceRequirements": [
        {
          "type": "GPU",
          "value": "1"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/autovoice",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```


### GCP Deployment

#### Compute Engine with GPU

```bash
# Create instance with T4 GPU
gcloud compute instances create autovoice-prod \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE \
  --metadata=startup-script='#!/bin/bash
    curl https://get.docker.com | sh
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
    apt-get update && apt-get install -y nvidia-docker2
    systemctl restart docker
    docker run -d --name autovoice --gpus all -p 5000:5000 autovoice/autovoice:latest'

# SSH into instance
gcloud compute ssh autovoice-prod --zone=us-central1-a
```

### Azure Deployment

#### VM with GPU

```bash
# Create resource group
az group create --name autovoice-rg --location eastus

# Create VM with NC6 (K80 GPU)
az vm create \
  --resource-group autovoice-rg \
  --name autovoice-vm \
  --image UbuntuLTS \
  --size Standard_NC6 \
  --admin-username azureuser \
  --generate-ssh-keys

# Install NVIDIA drivers (after SSH)
az vm extension set \
  --resource-group autovoice-rg \
  --vm-name autovoice-vm \
  --name NvidiaGpuDriverLinux \
  --publisher Microsoft.HpcCompute \
  --version 1.3
```

---

## Configuration

### Environment Variables

**Core Configuration:**
```bash
# Application
LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR
WORKERS=4                         # Number of worker processes
HOST=0.0.0.0                      # Bind address
PORT=5000                         # Application port

# GPU Configuration
CUDA_VISIBLE_DEVICES=0            # GPU device ID (0,1,2 for multi-GPU)
CUDA_LAUNCH_BLOCKING=0            # Set to 1 for debugging
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Memory management

# Model Configuration
MODEL_PATH=/app/models            # Path to model files
MODEL_CACHE_DIR=/app/cache        # Model cache directory
BATCH_SIZE=4                      # Inference batch size
MAX_SEQUENCE_LENGTH=1024          # Maximum sequence length

# Performance
ENABLE_TENSORRT=true              # Enable TensorRT optimization
PRECISION=fp16                    # fp32, fp16, int8
NUM_THREADS=8                     # CPU threads for data loading

# Monitoring
PROMETHEUS_ENABLED=true           # Enable Prometheus metrics
METRICS_PORT=9090                 # Metrics endpoint port
HEALTH_CHECK_INTERVAL=30          # Health check interval (seconds)

# Security
API_KEY_REQUIRED=false            # Require API key authentication
RATE_LIMIT_ENABLED=true           # Enable rate limiting
MAX_REQUESTS_PER_MINUTE=60        # Rate limit threshold
```

### Configuration Files

**logging_config.yaml:**
```yaml
version: 1
disable_existing_loggers: false

formatters:
  json:
    class: pythonjsonlogger.jsonlogger.JsonFormatter
    format: '%(asctime)s %(name)s %(levelname)s %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    formatter: json
    stream: ext://sys.stdout

  file:
    class: logging.handlers.RotatingFileHandler
    formatter: json
    filename: /app/logs/autovoice.log
    maxBytes: 104857600  # 100MB
    backupCount: 10

root:
  level: INFO
  handlers: [console, file]
```

### GPU Selection

**Single GPU:**
```bash
# Use GPU 0
export CUDA_VISIBLE_DEVICES=0
```

**Multi-GPU (model parallelism):**
```bash
# Use GPUs 0 and 1
export CUDA_VISIBLE_DEVICES=0,1
```

**CPU-only mode:**
```bash
# Disable GPU
export CUDA_VISIBLE_DEVICES=-1
```

---

## Security Hardening

### Container Security

**Non-root user (already implemented in Dockerfile):**
```dockerfile
# Create non-root user
RUN useradd -m -u 1000 autovoice
USER autovoice
```

**Read-only filesystem:**
```bash
docker run -d \
  --read-only \
  --tmpfs /tmp \
  --tmpfs /app/cache \
  --gpus all \
  -p 5000:5000 \
  autovoice/autovoice:latest
```

### TLS/SSL Configuration

**Using nginx as reverse proxy:**
```nginx
server {
    listen 443 ssl http2;
    server_name autovoice.example.com;

    ssl_certificate /etc/ssl/certs/autovoice.crt;
    ssl_certificate_key /etc/ssl/private/autovoice.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Secrets Management

**Never hardcode credentials:**
```bash
# Use environment variables
export API_KEY=$(cat /run/secrets/api_key)

# Or use Docker secrets
docker secret create api_key api_key.txt
docker service create \
  --name autovoice \
  --secret api_key \
  autovoice/autovoice:latest
```

### Rate Limiting

**Application-level rate limiting:**
```python
# config.yaml
rate_limiting:
  enabled: true
  requests_per_minute: 60
  burst: 10
```

### Network Security

**Firewall rules:**
```bash
# Allow only necessary ports
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

---

## Monitoring & Observability

### Health Check Endpoints

**Liveness probe:**
```bash
curl http://localhost:5000/health/live
# Response: {"status": "alive"}
# Status: 200 OK
```

**Readiness probe:**
```bash
curl http://localhost:5000/health/ready
# Response when ready (200 OK):
# {
#   "status": "ready",
#   "components": {
#     "model": "ready",
#     "gpu": "available",
#     "synthesizer": "ready",
#     "voice_cloner": "ready",
#     "singing_conversion_pipeline": "ready"
#   }
# }
#
# Response when not ready (503 Service Unavailable):
# {
#   "status": "not_ready",
#   "components": {
#     "model": "not_initialized",
#     "gpu": "unavailable",
#     "synthesizer": "not_initialized",
#     "voice_cloner": "not_initialized",
#     "singing_conversion_pipeline": "not_initialized"
#   }
# }
```

**Main health check:**
```bash
curl http://localhost:5000/health | jq .
# Response:
# {
#   "status": "healthy",
#   "components": {
#     "gpu_available": true,
#     "model_loaded": true,
#     "api": true,
#     "synthesizer": true,
#     "voice_cloner": true,
#     "singing_conversion_pipeline": true
#   },
#   "system": {
#     "memory_percent": 45.2,
#     "cpu_percent": 12.5,
#     "gpu": {
#       "available": true,
#       "device_count": 1
#     }
#   }
# }
```


### Prometheus Metrics

**Metrics endpoint:**
```bash
curl http://localhost:9090/metrics
```

**Key metrics:**
- `autovoice_requests_total`: Total number of requests
- `autovoice_request_duration_seconds`: Request latency histogram
- `autovoice_errors_total`: Total number of errors
- `autovoice_gpu_utilization`: GPU utilization percentage
- `autovoice_gpu_memory_used_bytes`: GPU memory usage
- `autovoice_model_inference_duration_seconds`: Model inference time

### GPU Monitoring

**Monitor GPU usage:**
```bash
# Real-time monitoring
nvidia-smi -l 1

# Detailed query
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 1
```

### Log Aggregation

**Structured JSON logs:**
```json
{
  "timestamp": "2025-11-01T10:30:45.123Z",
  "level": "INFO",
  "message": "Request processed successfully",
  "request_id": "abc123",
  "duration_ms": 150,
  "gpu_id": 0
}
```

**View logs:**
```bash
# Docker logs
docker logs -f autovoice

# Filter by level
docker logs autovoice 2>&1 | grep ERROR

# Follow with timestamps
docker logs -f --timestamps autovoice
```

---

## Testing the Deployment

### Health Check Validation

```bash
# Test liveness
curl -f http://localhost:5000/health/live || echo "Liveness check failed"

# Test readiness
curl -f http://localhost:5000/health/ready || echo "Readiness check failed"

# Test main health endpoint
curl http://localhost:5000/health | jq .
```

### API Endpoint Testing

**TTS synthesis:**
```bash
curl -X POST http://localhost:5000/api/v1/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test of the text-to-speech system.",
    "speaker_id": 0,
    "speed": 1.0
  }' \
  --output test_audio.wav
```

**Voice conversion:**
```bash
# Create voice profile
curl -X POST http://localhost:5000/api/v1/voice/clone \
  -F "audio=@sample_voice.wav" \
  -F "name=test_voice"

# List profiles
curl http://localhost:5000/api/v1/voice/profiles
```

### Load Testing

**Using apache2-utils (ab):**
```bash
# Install ab
sudo apt-get install -y apache2-utils

# Run load test (100 requests, 10 concurrent)
ab -n 100 -c 10 -p request.json -T application/json \
  http://localhost:5000/api/v1/synthesize
```

**request.json:**
```json
{"text": "Load test message", "speaker_id": 0}
```

### GPU Utilization Verification

```bash
# Monitor GPU during load test
watch -n 1 nvidia-smi

# Expected: GPU utilization 70-95% during active requests
```

### Performance Benchmarking

```bash
# Run benchmark script
./scripts/benchmark.sh

# Expected output:
# TTS latency: 100-200ms (p50), 200-400ms (p99)
# Voice conversion: 2-5s (p50), 5-10s (p99)
# GPU memory: 4-8GB used
```

---

## Troubleshooting

### Common Issues and Solutions

**Issue: "CUDA not available"**
```bash
# Check driver
nvidia-smi

# Check CUDA
nvcc --version

# Check PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# Solution: Reinstall PyTorch with CUDA support (2.0–2.2 supported)
python -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu121
```

**Issue: "nvcc not found"**
```bash
# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Make permanent
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

**Issue: "CUDA version mismatch"**
```bash
# Check versions
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# Reinstall PyTorch to match CUDA version (2.0–2.2 supported)
pip uninstall torch torchvision torchaudio
python -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu121
```

**Issue: "Out of memory"**
```bash
# Reduce batch size
export BATCH_SIZE=1

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Monitor memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1
```

**Issue: "libcudart.so not found"**
```bash
# Add CUDA lib to library path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
sudo ldconfig
```

**Issue: Docker GPU access fails**
```bash
# Verify nvidia-docker2 installed
dpkg -l | grep nvidia-docker2

# Restart Docker
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**Issue: Build failures**
```bash
# Clean and rebuild
python setup.py clean --all
python -m pip install -e . --force-reinstall --no-deps

# Build for specific GPU architecture
TORCH_CUDA_ARCH_LIST="80" python -m pip install -e .
```

**Issue: Import errors**
```bash
# Verify build succeeded
ls -la build/lib*/auto_voice/cuda_kernels*.so

# Check PYTHONPATH
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Reinstall
python -m pip install -e . --force-reinstall
```

### Diagnostic Commands

**Collect diagnostic information:**
```bash
# System info
uname -a
lsb_release -a

# GPU info
nvidia-smi
nvcc --version

# Python/PyTorch info
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU available: {torch.cuda.is_available()}')"

# Docker info (if using Docker)
docker --version
docker info | grep -i runtime

# Application logs
docker logs autovoice --tail 100
```

---

## Performance Optimization

### GPU Optimization

**TensorRT optimization:**
```bash
# Enable TensorRT
export ENABLE_TENSORRT=true

# Use INT8 quantization
export PRECISION=int8
```

**Batch size tuning:**
```bash
# Increase batch size for throughput
export BATCH_SIZE=8

# Monitor GPU memory
nvidia-smi --query-gpu=memory.used --format=csv -l 1
```

**CUDA architecture selection:**
```bash
# Build for specific architecture (faster compilation)
TORCH_CUDA_ARCH_LIST="80" python -m pip install -e .

# Build for multiple architectures (broader compatibility)
TORCH_CUDA_ARCH_LIST="70;75;80;86" python -m pip install -e .
```

### Application Tuning

**Worker processes:**
```bash
# Increase workers for higher concurrency
export WORKERS=8

# Rule of thumb: 2 * CPU cores
```

**Log level:**
```bash
# Reduce logging overhead in production
export LOG_LEVEL=WARNING
```

**Metrics sampling:**
```bash
# Reduce metrics overhead
export METRICS_SAMPLE_RATE=0.1  # Sample 10% of requests
```

### Mixed Precision (AMP)

**Enable automatic mixed precision:**
```python
# config.yaml
performance:
  enable_amp: true
  precision: fp16
```

**Expected benefits:**
- 2x faster inference
- 50% less GPU memory
- Minimal accuracy loss

---

## Rollback Procedures

### Docker Rollback

**Step 1: Identify previous version**
```bash
# List images
docker images autovoice/autovoice

# Example output:
# autovoice/autovoice  v1.1.0  abc123  2 days ago   5GB
# autovoice/autovoice  v1.0.0  def456  1 week ago   4.8GB
```

**Step 2: Stop current container**
```bash
docker stop autovoice
docker rm autovoice
```

**Step 3: Start previous version**
```bash
docker run -d \
  --name autovoice \
  --gpus all \
  -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  autovoice/autovoice:v1.0.0
```

**Step 4: Verify rollback**
```bash
curl http://localhost:5000/health
docker logs autovoice
```


### Source Rollback

**Step 1: Identify previous commit**
```bash
git log --oneline -10
# Example: abc1234 v1.0.0 release
```

**Step 2: Checkout previous version**
```bash
git checkout abc1234
```

**Step 3: Rebuild**
```bash
python setup.py clean --all
python -m pip install -e . --force-reinstall
```

**Step 4: Restart application**
```bash
sudo systemctl restart autovoice
# Or: ./scripts/run.sh
```

**Step 5: Verify rollback**
```bash
curl http://localhost:5000/health
./scripts/test.sh
```

### Rollback Decision Criteria

**When to rollback:**
- Error rate > 5% for 5+ minutes
- Latency p99 > 2x baseline for 10+ minutes
- GPU memory leak detected
- Critical security vulnerability
- Data corruption detected

**When NOT to rollback:**
- Temporary spike in errors (< 2 minutes)
- Single user reports issue (investigate first)
- Non-critical feature broken
- Performance degradation < 20%

---

## Production Checklist

### Pre-Deployment Checklist

- [ ] **GPU validated**: Compute capability ≥ 7.0, Driver 535+
- [ ] **CUDA installed**: Version 11.8+ or 12.1+
- [ ] **PyTorch validated**: CUDA support confirmed
- [ ] **Build tools installed**: gcc, cmake, ninja
- [ ] **Tests passing**: All unit and integration tests pass
- [ ] **Docker tested**: GPU access verified in container
- [ ] **Models downloaded**: All required model weights present
- [ ] **Configuration reviewed**: Environment variables set correctly
- [ ] **Secrets secured**: No hardcoded credentials
- [ ] **Firewall configured**: Only necessary ports open

### Security Checklist

- [ ] **Non-root user**: Container runs as non-root
- [ ] **TLS enabled**: HTTPS configured with valid certificate
- [ ] **Rate limiting**: Enabled and tested
- [ ] **Input validation**: All endpoints validate input
- [ ] **Secrets management**: Using environment variables or secrets manager
- [ ] **Network security**: Firewall rules configured
- [ ] **Container scanning**: Trivy or similar tool run
- [ ] **Dependency audit**: No known vulnerabilities
- [ ] **Logging sanitized**: No sensitive data in logs
- [ ] **Access control**: Authentication/authorization implemented

### Monitoring Checklist

- [ ] **Health checks**: /health, /health/live, /health/ready working
- [ ] **Metrics endpoint**: /metrics accessible
- [ ] **Prometheus configured**: Scraping metrics successfully
- [ ] **Grafana dashboards**: Imported and displaying data
- [ ] **Alerts configured**: Error rate, latency, GPU memory alerts set
- [ ] **Log aggregation**: Logs being collected and searchable
- [ ] **GPU monitoring**: nvidia-smi or equivalent monitoring GPU
- [ ] **Uptime monitoring**: External monitoring service configured
- [ ] **On-call rotation**: Team members assigned and notified
- [ ] **Runbook updated**: Troubleshooting steps documented

### Documentation Checklist

- [ ] **Deployment guide**: Up-to-date and validated
- [ ] **API documentation**: All endpoints documented
- [ ] **Runbook**: Troubleshooting steps documented
- [ ] **Architecture diagram**: System architecture documented
- [ ] **Configuration guide**: All environment variables documented
- [ ] **Rollback procedure**: Tested and documented
- [ ] **Disaster recovery**: Backup and restore procedures documented
- [ ] **Contact information**: On-call contacts and escalation path documented
- [ ] **Change log**: Recent changes documented
- [ ] **Known issues**: Current limitations documented

---

## Monitoring Queries & Alerts

### Prometheus Queries

**Request rate:**
```promql
# Requests per second
rate(autovoice_requests_total[5m])

# Requests per minute
rate(autovoice_requests_total[1m]) * 60
```

**Error rate:**
```promql
# Error percentage
(rate(autovoice_errors_total[5m]) / rate(autovoice_requests_total[5m])) * 100

# Errors per minute
rate(autovoice_errors_total[1m]) * 60
```

**Latency:**
```promql
# p50 latency
histogram_quantile(0.50, rate(autovoice_request_duration_seconds_bucket[5m]))

# p95 latency
histogram_quantile(0.95, rate(autovoice_request_duration_seconds_bucket[5m]))

# p99 latency
histogram_quantile(0.99, rate(autovoice_request_duration_seconds_bucket[5m]))
```

**GPU utilization:**
```promql
# GPU utilization percentage
autovoice_gpu_utilization

# GPU memory usage
autovoice_gpu_memory_used_bytes / autovoice_gpu_memory_total_bytes * 100
```

### Alert Rules

**prometheus_alerts.yml:**
```yaml
groups:
  - name: autovoice_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: (rate(autovoice_errors_total[5m]) / rate(autovoice_requests_total[5m])) * 100 > 5
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }}% (threshold: 5%)"

      - alert: HighLatency
        expr: histogram_quantile(0.99, rate(autovoice_request_duration_seconds_bucket[5m])) > 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
          description: "p99 latency is {{ $value }}s (threshold: 0.5s)"

      - alert: GPUMemoryHigh
        expr: (autovoice_gpu_memory_used_bytes / autovoice_gpu_memory_total_bytes) * 100 > 90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU memory usage high"
          description: "GPU memory usage is {{ $value }}% (threshold: 90%)"

      - alert: ServiceDown
        expr: up{job="autovoice"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "AutoVoice service is down"
          description: "AutoVoice has been down for more than 2 minutes"
```

### Grafana Dashboard

**Key panels:**
1. **Request Rate**: Line graph of requests/second
2. **Error Rate**: Line graph of error percentage
3. **Latency**: Multi-line graph (p50, p95, p99)
4. **GPU Utilization**: Gauge showing current utilization
5. **GPU Memory**: Gauge showing memory usage
6. **Active Requests**: Counter of in-flight requests
7. **Model Inference Time**: Histogram of inference duration

**Import dashboard:**
```bash
# Import from Grafana dashboard ID (if published)
# Or create custom dashboard with above panels
```

---

## Support & Resources

### Documentation

- **README**: [README.md](../README.md) - Quick start and overview
- **API Documentation**: [api-documentation.md](api-documentation.md) - Complete API reference
- **Runbook**: [runbook.md](runbook.md) - Operational procedures
- **Architecture**: [architecture.md](architecture.md) - System design
- **Contributing**: [CONTRIBUTING.md](../CONTRIBUTING.md) - Development guidelines

### Community

- **GitHub Issues**: [https://github.com/autovoice/autovoice/issues](https://github.com/autovoice/autovoice/issues)
- **GitHub Discussions**: [https://github.com/autovoice/autovoice/discussions](https://github.com/autovoice/autovoice/discussions)
- **Documentation Site**: [https://autovoice.readthedocs.io](https://autovoice.readthedocs.io)

### Commercial Support

For enterprise support, SLA agreements, and custom development:
- **Email**: support@autovoice.io
- **Website**: https://autovoice.io/enterprise

### Reporting Issues

When reporting issues, include:

1. **Environment information**:
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.is_available()}')"
   nvidia-smi
   nvcc --version
   ```

2. **Error logs**:
   ```bash
   docker logs autovoice --tail 100
   # Or application logs from /app/logs/
   ```

3. **Steps to reproduce**: Clear description of how to reproduce the issue

4. **Expected vs actual behavior**: What you expected to happen vs what actually happened

5. **Configuration**: Relevant environment variables and configuration files

---

## Conclusion

This deployment guide provides comprehensive instructions for deploying AutoVoice in production environments. Key takeaways:

- **Docker deployment is recommended** for consistency and ease of management
- **GPU requirements are critical**: Ensure compute capability ≥ 7.0 and Driver 535+
- **Monitoring is essential**: Set up Prometheus, Grafana, and alerts before going live
- **Security must be prioritized**: Use TLS, rate limiting, and secrets management
- **Testing is mandatory**: Validate health checks, API endpoints, and performance before production
- **Have a rollback plan**: Test rollback procedures before you need them

For questions or issues not covered in this guide, consult the [Support & Resources](#support--resources) section.

---

**Document Version**: 1.0
**Last Updated**: 2025-11-01
**Maintained By**: AutoVoice Team

