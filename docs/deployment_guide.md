# AutoVoice CUDA Extension Deployment Guide

This guide provides step-by-step instructions for deploying the AutoVoice CUDA extension in production environments.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Building from Source](#building-from-source)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Testing the Deployment](#testing-the-deployment)
- [Troubleshooting](#troubleshooting)
- [Rolling Back](#rolling-back)
- [Production Monitoring](#production-monitoring)
- [Performance Optimization](#performance-optimization)

---

## Prerequisites

### Hardware Requirements

#### GPU Requirements
- **Minimum**: NVIDIA GPU with compute capability 7.0 (Volta architecture)
- **Recommended**: Compute capability 8.0+ (Ampere) or 8.9 (Ada Lovelace)
- **Supported Architectures**:
  - Volta (V100): Compute capability 7.0
  - Turing (RTX 20xx, T4): Compute capability 7.5
  - Ampere (A100, RTX 30xx): Compute capability 8.0/8.6
  - Ada Lovelace (RTX 40xx): Compute capability 8.9

Check your GPU compute capability:
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

#### Memory Requirements
- **Minimum VRAM**: 4GB
- **Recommended VRAM**: 8GB or more
- **System RAM**: 8GB minimum, 16GB+ recommended
- **Disk Space**: 10GB for dependencies and models

### Software Requirements

#### CUDA Toolkit
- **Minimum**: CUDA 11.8
- **Recommended**: CUDA 12.2 or 12.9
- **cuDNN**: 8.6.0 or later (compatible with CUDA version)

Install CUDA Toolkit:
```bash
# Ubuntu 22.04 example
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-2
```

Verify CUDA installation:
```bash
nvcc --version
nvidia-smi
```

#### NVIDIA Driver
- **Minimum**: Driver 525+ (for CUDA 12.x)
- **Recommended**: Driver 535+ (stable production release)

Install NVIDIA Driver:
```bash
# Ubuntu
sudo apt-get update
sudo apt-get install -y nvidia-driver-535

# Verify
nvidia-smi
```

#### Python Environment
- **Python**: 3.8, 3.9, or 3.10
- **pip**: Latest version recommended
- **virtualenv** or **conda**: For isolated environments

#### Build Tools
```bash
# Ubuntu/Debian
sudo apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    libsndfile1-dev \
    ffmpeg

# Verify
gcc --version
cmake --version
ninja --version
```

---

## Environment Setup

### 1. Create Virtual Environment

```bash
# Using virtualenv
python3.10 -m venv venv
source venv/bin/activate

# Or using conda
conda create -n autovoice python=3.10
conda activate autovoice
```

### 2. Set Environment Variables

Create a `.env` file or export variables:

```bash
# CUDA Configuration
export CUDA_HOME=/usr/local/cuda
export CUDA_VISIBLE_DEVICES=0  # GPU device ID(s)
export TORCH_CUDA_ARCH_LIST="70;75;80;86;89"  # Target architectures

# Build Configuration
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CFLAGS="-I$CUDA_HOME/include"
export LDFLAGS="-L$CUDA_HOME/lib64"

# Application Configuration
export LOG_LEVEL=INFO
export LOG_FORMAT=json
export FLASK_ENV=production
export PROMETHEUS_ENABLED=true
```

### 3. Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    libsndfile1 \
    ffmpeg \
    portaudio19-dev \
    libportaudio2 \
    curl \
    ca-certificates
```

---

## Building from Source

### Method 1: Standard Build (Recommended)

#### Step 1: Clone Repository
```bash
git clone https://github.com/autovoice/autovoice.git
cd autovoice
```

#### Step 2: Install Python Dependencies
```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support FIRST
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

#### Step 3: Install Other Dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Build CUDA Extensions
```bash
# Method A: Using build script (recommended)
chmod +x scripts/build.sh
./scripts/build.sh

# Method B: Using setup.py directly
python setup.py build_ext --inplace

# Method C: Install in editable mode (for development)
pip install -e .
```

**Build Options**:
```bash
# Build for specific architectures
TORCH_CUDA_ARCH_LIST="80;86" python setup.py build_ext --inplace

# Build with verbose output
python setup.py build_ext --inplace --verbose

# Build without CUDA (CPU-only, not recommended for production)
SKIP_CUDA_BUILD=1 pip install -e .
```

#### Step 5: Verify Build
```bash
# Run verification script
./scripts/test.sh

# Or manual verification
python -c "
import torch
import auto_voice
from auto_voice.audio.processor import AudioProcessor

print(f'AutoVoice version: {auto_voice.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

# Try importing CUDA kernels
try:
    import auto_voice.cuda_kernels
    print('CUDA kernels imported successfully')
except ImportError as e:
    print(f'CUDA kernels import failed: {e}')

# Create processor
processor = AudioProcessor(device='cuda' if torch.cuda.is_available() else 'cpu')
print(f'AudioProcessor created on device: {processor.device}')
"
```

### Method 2: Docker Build (Production)

#### Step 1: Verify Docker and nvidia-docker
```bash
# Check Docker
docker --version

# Check NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu22.04 nvidia-smi
```

If nvidia-docker is not installed:
```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### Step 2: Build Docker Image
```bash
# Clone repository
git clone https://github.com/autovoice/autovoice.git
cd autovoice

# Build image with build args
docker build \
    --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
    --build-arg VCS_REF=$(git rev-parse --short HEAD) \
    --build-arg VERSION=$(git describe --tags --always) \
    -t autovoice:latest \
    -t autovoice:$(git describe --tags --always) \
    .

# Build for specific CUDA architectures
docker build \
    --build-arg TORCH_CUDA_ARCH_LIST="80;86;89" \
    -t autovoice:latest \
    .
```

#### Step 3: Test Docker Image
```bash
# Run test container
docker run --rm --gpus all autovoice:latest python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
"

# Run health check
docker run --rm --gpus all -p 5000:5000 autovoice:latest python -c "
from auto_voice.audio.processor import AudioProcessor
processor = AudioProcessor(device='cuda')
print('Health check passed')
"
```

---

## Docker Deployment

### Method 1: Docker Run (Simple)

```bash
# Basic deployment
docker run -d \
    --name autovoice \
    --gpus all \
    -p 5000:5000 \
    -v /path/to/models:/app/models \
    -v /path/to/data:/app/data \
    -e LOG_LEVEL=INFO \
    -e CUDA_VISIBLE_DEVICES=0 \
    --restart unless-stopped \
    autovoice:latest

# Check logs
docker logs -f autovoice

# Check health
curl http://localhost:5000/health
```

### Method 2: Docker Compose (Recommended)

#### Step 1: Create docker-compose.yml

```yaml
version: '3.8'

services:
  autovoice:
    image: autovoice:latest
    container_name: autovoice
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - LOG_LEVEL=INFO
      - LOG_FORMAT=json
      - FLASK_ENV=production
      - PROMETHEUS_ENABLED=true
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    restart: unless-stopped
    profiles:
      - monitoring

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    volumes:
      - ./config/grafana:/etc/grafana/provisioning
      - grafana_data:/var/lib/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    restart: unless-stopped
    profiles:
      - monitoring

volumes:
  prometheus_data:
  grafana_data:
```

#### Step 2: Deploy with Docker Compose

```bash
# Deploy application only
docker-compose up -d

# Deploy with monitoring
docker-compose --profile monitoring up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f autovoice

# Stop deployment
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

---

## Cloud Deployment

### AWS Deployment

#### Prerequisites
- AWS CLI installed and configured
- EC2 GPU instance (g4dn, g5, p3, p4 series)
- ECS or EKS cluster (for container orchestration)

#### Option A: EC2 Direct Deployment

```bash
# 1. Launch GPU instance (example: g4dn.xlarge)
aws ec2 run-instances \
    --image-id ami-0c7217cdde317cfec \
    --instance-type g4dn.xlarge \
    --key-name your-key-pair \
    --security-groups your-security-group \
    --user-data file://user-data.sh

# 2. user-data.sh content:
#!/bin/bash
# Install NVIDIA drivers
sudo apt-get update
sudo apt-get install -y nvidia-driver-535

# Install Docker and nvidia-docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Pull and run AutoVoice
sudo docker run -d \
    --name autovoice \
    --gpus all \
    -p 5000:5000 \
    --restart unless-stopped \
    autovoice:latest
```

#### Option B: ECS Deployment

Create ECS task definition (`task-definition.json`):

```json
{
  "family": "autovoice",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["EC2"],
  "containerDefinitions": [
    {
      "name": "autovoice",
      "image": "your-ecr-repo/autovoice:latest",
      "cpu": 4096,
      "memory": 8192,
      "essential": true,
      "portMappings": [
        {
          "containerPort": 5000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "LOG_LEVEL", "value": "INFO"},
        {"name": "FLASK_ENV", "value": "production"}
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

Deploy to ECS:
```bash
# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service \
    --cluster your-cluster \
    --service-name autovoice \
    --task-definition autovoice \
    --desired-count 2 \
    --launch-type EC2
```

### GCP Deployment

#### Prerequisites
- gcloud CLI installed
- GCP project with Compute Engine API enabled
- GPU quota in target region

#### Create GPU Instance with Container

```bash
# Create instance with NVIDIA GPU
gcloud compute instances create-with-container autovoice-vm \
    --container-image=gcr.io/your-project/autovoice:latest \
    --container-restart-policy=always \
    --container-env=LOG_LEVEL=INFO,FLASK_ENV=production \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=50GB \
    --zone=us-central1-a

# Install GPU drivers on first boot (automatic with GCP's GPU-optimized images)
gcloud compute instances create autovoice-vm \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB \
    --zone=us-central1-a \
    --metadata startup-script='#!/bin/bash
    # Install NVIDIA drivers
    curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get install -y cuda-drivers-535

    # Install Docker
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh

    # Install nvidia-docker
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker

    # Run AutoVoice
    sudo docker run -d --name autovoice --gpus all -p 5000:5000 gcr.io/your-project/autovoice:latest
    '
```

### Azure Deployment

#### Prerequisites
- Azure CLI installed
- NC-series or ND-series VM quota

#### Deploy with Azure Container Instances

```bash
# Create resource group
az group create --name autovoice-rg --location eastus

# Create container instance with GPU
az container create \
    --resource-group autovoice-rg \
    --name autovoice \
    --image your-acr.azurecr.io/autovoice:latest \
    --cpu 4 \
    --memory 8 \
    --gpu-count 1 \
    --gpu-sku V100 \
    --ports 5000 \
    --environment-variables LOG_LEVEL=INFO FLASK_ENV=production \
    --restart-policy Always
```

---

## Testing the Deployment

### 1. Health Check Tests

```bash
# Basic health check
curl -f http://localhost:5000/health || echo "Health check failed"

# Liveness probe
curl -f http://localhost:5000/health/live

# Readiness probe
curl -f http://localhost:5000/health/ready

# Metrics endpoint
curl http://localhost:5000/metrics
```

### 2. Functional Tests

```bash
# Test synthesis endpoint
curl -X POST http://localhost:5000/api/v1/synthesize \
    -H "Content-Type: application/json" \
    -d '{
        "text": "Hello, this is a test of the AutoVoice system.",
        "speaker_id": "default"
    }' \
    -o test_output.wav

# Verify output file
file test_output.wav
```

### 3. Load Testing

```bash
# Install Apache Bench
sudo apt-get install -y apache2-utils

# Run load test (100 requests, 10 concurrent)
ab -n 100 -c 10 -p request.json -T application/json \
    http://localhost:5000/api/v1/synthesize

# request.json content:
{
  "text": "Load testing the AutoVoice system.",
  "speaker_id": "default"
}
```

### 4. GPU Utilization Test

```bash
# Monitor GPU during load test
watch -n 1 nvidia-smi

# Check CUDA operations
docker exec autovoice python -c "
import torch
from auto_voice.audio.processor import AudioProcessor

processor = AudioProcessor(device='cuda')
print(f'Device: {processor.device}')
print(f'Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB')
print(f'Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB')
"
```

---

## Troubleshooting

### Issue 1: CUDA Not Available

**Symptoms**:
- `torch.cuda.is_available()` returns `False`
- "CUDA is not available" warnings during build

**Solutions**:

```bash
# 1. Verify NVIDIA driver
nvidia-smi
# If command fails, install/reinstall driver

# 2. Check CUDA installation
nvcc --version
ls -la /usr/local/cuda

# 3. Verify PyTorch CUDA support
python -c "import torch; print(torch.version.cuda)"

# 4. Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# 5. Check environment variables
echo $CUDA_HOME
echo $LD_LIBRARY_PATH
```

### Issue 2: CUDA Kernel Build Failures

**Symptoms**:
- Build fails with "nvcc not found"
- Compiler errors during extension build

**Solutions**:

```bash
# 1. Ensure nvcc is in PATH
export PATH=/usr/local/cuda/bin:$PATH
which nvcc

# 2. Verify CUDA_HOME is set
export CUDA_HOME=/usr/local/cuda
echo $CUDA_HOME

# 3. Check CUDA libraries
ldconfig -p | grep cuda

# 4. Try building with verbose output
python setup.py build_ext --inplace --verbose 2>&1 | tee build.log

# 5. Build for specific architecture only
TORCH_CUDA_ARCH_LIST="80" python setup.py build_ext --inplace
```

### Issue 3: Docker Container Can't Access GPU

**Symptoms**:
- nvidia-smi fails in container
- CUDA not available in Docker

**Solutions**:

```bash
# 1. Test nvidia-docker installation
docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu22.04 nvidia-smi

# 2. Reinstall NVIDIA Container Toolkit
sudo apt-get purge -y nvidia-container-toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 3. Check Docker runtime
docker info | grep -i runtime

# 4. Use explicit runtime flag
docker run --runtime=nvidia --rm nvidia/cuda:12.9.0-base-ubuntu22.04 nvidia-smi

# 5. Verify docker-compose GPU configuration
docker-compose config
```

### Issue 4: Out of Memory (OOM) Errors

**Symptoms**:
- "CUDA out of memory" errors
- Process killed by OOM killer

**Solutions**:

```bash
# 1. Check GPU memory usage
nvidia-smi

# 2. Reduce batch size or model size in configuration

# 3. Enable CPU fallback
export AUTOVOICE_CPU_FALLBACK=true

# 4. Monitor memory during operation
watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.free --format=csv'

# 5. Increase GPU memory if possible (use larger instance type)
```

### Issue 5: Slow Performance

**Symptoms**:
- Synthesis takes longer than expected
- GPU utilization is low

**Solutions**:

```bash
# 1. Check GPU utilization
nvidia-smi dmon

# 2. Profile GPU operations
python -m torch.utils.bottleneck script.py

# 3. Enable TensorRT optimization (if available)
export AUTOVOICE_TENSORRT_ENABLED=true

# 4. Check for CPU bottlenecks
htop

# 5. Verify CUDA graphs are being used
docker logs autovoice | grep "CUDA graph"
```

### Issue 6: Import Errors

**Symptoms**:
- `ImportError: cannot import name 'cuda_kernels'`
- `ModuleNotFoundError: No module named 'auto_voice'`

**Solutions**:

```bash
# 1. Verify PYTHONPATH
echo $PYTHONPATH
export PYTHONPATH=/app/src:$PYTHONPATH

# 2. Reinstall package
pip uninstall -y auto_voice
pip install -e .

# 3. Check extension build
ls -la build/lib*/auto_voice/

# 4. Rebuild extensions
python setup.py clean --all
python setup.py build_ext --inplace

# 5. Test imports manually
python -c "import sys; sys.path.insert(0, 'src'); import auto_voice; print(auto_voice.__file__)"
```

---

## Rolling Back

### Rollback Procedures

#### Docker Rollback

```bash
# 1. Stop current deployment
docker stop autovoice

# 2. Remove current container
docker rm autovoice

# 3. Run previous version
docker run -d \
    --name autovoice \
    --gpus all \
    -p 5000:5000 \
    autovoice:v1.0.0  # Previous stable version

# Or with docker-compose
docker-compose down
git checkout v1.0.0
docker-compose up -d
```

#### Source Rollback

```bash
# 1. Checkout previous version
git fetch --tags
git checkout v1.0.0

# 2. Rebuild
pip uninstall -y auto_voice
pip install -e .

# 3. Restart service
systemctl restart autovoice  # If using systemd
```

#### Cloud Rollback

```bash
# AWS ECS
aws ecs update-service \
    --cluster your-cluster \
    --service autovoice \
    --task-definition autovoice:previous-version

# GCP
gcloud compute instances update-container autovoice-vm \
    --container-image=gcr.io/your-project/autovoice:v1.0.0

# Azure
az container create \
    --resource-group autovoice-rg \
    --name autovoice \
    --image your-acr.azurecr.io/autovoice:v1.0.0 \
    # ... other parameters
```

---

## Production Monitoring

### Metrics to Monitor

#### Application Metrics
- Request rate (requests/second)
- Response latency (p50, p95, p99)
- Error rate (4xx, 5xx errors)
- Active connections

#### GPU Metrics
- GPU utilization (%)
- GPU memory usage (MB)
- GPU temperature (°C)
- Power usage (W)

#### System Metrics
- CPU utilization (%)
- RAM usage (MB)
- Disk I/O
- Network I/O

### Monitoring Setup

#### Prometheus Configuration

Create `config/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'autovoice'
    static_configs:
      - targets: ['autovoice:5000']
    metrics_path: '/metrics'

  - job_name: 'nvidia-gpu'
    static_configs:
      - targets: ['node-exporter:9100']
```

#### Grafana Dashboards

Import pre-built dashboards or create custom ones:

1. **AutoVoice Application Dashboard**
   - Request rate and latency
   - Error rate
   - WebSocket connections

2. **GPU Monitoring Dashboard**
   - GPU utilization over time
   - Memory usage
   - Temperature trends

3. **System Resources Dashboard**
   - CPU and RAM usage
   - Disk and network I/O

### Alerting Rules

Create `config/prometheus/alerts.yml`:

```yaml
groups:
  - name: autovoice_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors/second"

      - alert: GPUMemoryHigh
        expr: nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU memory usage is high"
          description: "GPU memory usage is {{ $value }}%"

      - alert: ServiceDown
        expr: up{job="autovoice"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "AutoVoice service is down"
```

---

## Performance Optimization

### GPU Optimization

#### 1. Enable TensorRT (if available)

```bash
# Install TensorRT
pip install tensorrt~=8.6.0 --index-url https://pypi.nvidia.com

# Enable in configuration
export AUTOVOICE_TENSORRT_ENABLED=true
```

#### 2. Use Mixed Precision Training

```python
# Automatic Mixed Precision (AMP) is already enabled
# Verify in logs:
docker logs autovoice | grep "AMP enabled"
```

#### 3. Optimize Batch Size

```python
# Tune batch size for your GPU
# Edit config/audio_config.yaml
batch_size: 32  # Adjust based on GPU memory
```

### Application Optimization

#### 1. Enable CUDA Graphs

```bash
# CUDA graphs are already implemented
# Verify usage in logs
docker logs autovoice | grep "CUDA graph"
```

#### 2. Use Async Processing

```python
# WebSocket streaming is already async
# For batch processing, use async endpoints
```

#### 3. Connection Pooling

```bash
# Configure gunicorn workers
gunicorn -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:5000 main:app
```

### Network Optimization

#### 1. Enable Compression

```python
# In Flask configuration
COMPRESS_MIMETYPES = ['application/json', 'audio/wav']
COMPRESS_LEVEL = 6
```

#### 2. CDN for Static Assets

```bash
# Use CloudFront, Cloud CDN, or Azure CDN for model files
```

#### 3. Load Balancing

```bash
# Use NGINX or cloud load balancers
upstream autovoice {
    least_conn;
    server autovoice1:5000;
    server autovoice2:5000;
}
```

---

## Best Practices

### Security
1. ✅ Always run containers as non-root user
2. ✅ Use secrets management (AWS Secrets Manager, HashiCorp Vault)
3. ✅ Enable TLS/SSL for production endpoints
4. ✅ Implement rate limiting
5. ✅ Regular security updates and scanning

### Reliability
1. ✅ Implement health checks
2. ✅ Use auto-restart policies
3. ✅ Set up proper monitoring and alerting
4. ✅ Plan for graceful degradation (CPU fallback)
5. ✅ Regular backups of models and data

### Performance
1. ✅ Profile before optimizing
2. ✅ Use appropriate batch sizes
3. ✅ Enable CUDA graphs for repeated operations
4. ✅ Monitor GPU utilization
5. ✅ Use TensorRT when possible

### Maintainability
1. ✅ Version control all configurations
2. ✅ Document all customizations
3. ✅ Use infrastructure as code
4. ✅ Regular updates and patching
5. ✅ Comprehensive logging

---

## Appendix

### A. Environment Variable Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device IDs to use |
| `TORCH_CUDA_ARCH_LIST` | `70;75;80;86;89` | Target CUDA architectures |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FORMAT` | `json` | Log format (json/text) |
| `FLASK_ENV` | `production` | Flask environment |
| `PROMETHEUS_ENABLED` | `true` | Enable metrics endpoint |

### B. Port Reference

| Port | Service | Description |
|------|---------|-------------|
| 5000 | AutoVoice API | Main application and metrics |
| 8080 | WebSocket | WebSocket streaming |
| 9090 | Prometheus | Metrics collection |
| 3000 | Grafana | Monitoring dashboards |

### C. GPU Requirements by Model

| Model Size | Min VRAM | Recommended | Compute Capability |
|------------|----------|-------------|-------------------|
| Small | 2GB | 4GB | 7.0+ |
| Medium | 4GB | 6GB | 7.5+ |
| Large | 6GB | 8GB | 8.0+ |
| XLarge | 8GB | 12GB | 8.6+ |

### D. Support Resources

- **Documentation**: https://autovoice.readthedocs.io
- **GitHub Issues**: https://github.com/autovoice/autovoice/issues
- **Discussions**: https://github.com/autovoice/autovoice/discussions
- **Slack/Discord**: [Community link]

---

**Document Version**: 1.0
**Last Updated**: 2025-10-27
**Authors**: AutoVoice Team
