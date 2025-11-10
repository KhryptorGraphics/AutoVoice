# AutoVoice Operations Runbook

Comprehensive operations guide for deploying, monitoring, and maintaining AutoVoice in production environments.

## 1. System Overview

### Architecture Components

**Core Services**:
- **Web Server** (`auto_voice/web/app.py`): Flask application serving API and UI
- **TTS Engine** (`auto_voice/inference/engine.py`): Text-to-speech processing
- **Voice Conversion** (`auto_voice/inference/singing_conversion_pipeline.py`): Voice cloning and song conversion
- **WebSocket Handler** (`auto_voice/web/websocket_handler.py`): Real-time progress updates

**Background Workers**:
- **Audio Processing Queue**: Handles TTS and voice conversion jobs
- **Model Loading**: Lazy loading of ML models on demand
- **Cache Manager**: Manages temporary files and conversion artifacts

**Data Stores**:
- **Voice Profiles**: Speaker embeddings and metadata (file-based storage)
- **Conversion Cache**: Cached vocal separations and intermediate results
- **Session Data**: WebSocket connections and job status

### System Requirements

**Production Environment**:
- **CPU**: 8+ cores recommended for multi-tenant workloads
- **Memory**: 16GB+ RAM (8GB minimum)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (CUDA 11.8+)
- **Storage**: 50GB+ SSD for models and cache
- **Network**: 100Mbps+ for file uploads/downloads

**Software Dependencies**:
- **Python**: 3.10+ (CPython recommended)
- **CUDA**: 11.8 or 12.1 for GPU acceleration
- **cuDNN**: 8.x compatible with CUDA version
- **FFmpeg**: 4.4+ for audio processing
- **System Libraries**: libsndfile, sox, portaudio

## 2. Deployment

### 2.1 Initial Setup

**Environment Preparation**:
```bash
# Clone repository
git clone https://github.com/yourorg/autovoice.git
cd autovoice

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models
python scripts/download_models.py
```

**Configuration**:
```yaml
# config/production.yaml
server:
  host: 0.0.0.0
  port: 5000
  workers: 4
  timeout: 300

gpu:
  device: cuda
  allow_fallback: true
  max_batch_size: 16
  enable_tensorrt: true

storage:
  voice_profiles_dir: /data/voice_profiles
  cache_dir: /data/cache
  temp_dir: /tmp/autovoice
  max_cache_size_gb: 20

quality:
  default_preset: balanced
  allow_fast_preset: true
  allow_quality_preset: true

limits:
  max_audio_size_mb: 100
  max_profile_duration_s: 120
  max_song_duration_s: 600
  concurrent_conversions: 5

cache:
  separation_ttl_hours: 24
  conversion_ttl_hours: 24
  voice_profile_ttl_days: 90
```

**Database Initialization** (if using DB):
```bash
# Initialize database schema
python scripts/init_db.py --config config/production.yaml

# Create admin user
python scripts/create_user.py --username admin --role admin
```

### 2.2 Service Deployment

**Systemd Service** (Linux):
```ini
# /etc/systemd/system/autovoice.service
[Unit]
Description=AutoVoice AI Service
After=network.target

[Service]
Type=simple
User=autovoice
WorkingDirectory=/opt/autovoice
Environment="PATH=/opt/autovoice/venv/bin"
Environment="CUDA_VISIBLE_DEVICES=0"
ExecStart=/opt/autovoice/venv/bin/python -m auto_voice.web.app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Start Service**:
```bash
# Enable and start service
sudo systemctl enable autovoice
sudo systemctl start autovoice

# Check status
sudo systemctl status autovoice
```

**Docker Deployment**:
```dockerfile
# Simplified Dockerfile example (see root Dockerfile for full multi-stage build)
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip \
    ffmpeg libsndfile1 sox \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download models
RUN python scripts/download_models.py

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "-m", "auto_voice.web.app"]
```

**Note**: This is a simplified example. For production deployment, use the multi-stage Dockerfile in the repository root which includes CUDA 12.1.0 development stage for building extensions and a smaller runtime stage. See `Dockerfile` for the complete implementation.

**Docker Compose**:
```yaml
# docker-compose.yml
version: '3.8'

services:
  autovoice:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - voice_profiles:/data/voice_profiles
      - cache:/data/cache
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

volumes:
  voice_profiles:
  cache:
```

**Deploy with Docker**:
```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f autovoice

# Stop
docker-compose down
```

### 2.3 Load Balancing

**Nginx Configuration**:
```nginx
# /etc/nginx/sites-available/autovoice
upstream autovoice_backend {
    least_conn;
    server 127.0.0.1:5000 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:5001 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:5002 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name autovoice.example.com;

    # WebSocket support
    location /ws/ {
        proxy_pass http://autovoice_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 300s;
    }

    # API endpoints
    location /api/ {
        proxy_pass http://autovoice_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        client_max_body_size 100M;
        proxy_read_timeout 300s;
    }

    # Static files
    location /static/ {
        alias /opt/autovoice/src/auto_voice/web/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    # Health check
    location /health {
        proxy_pass http://autovoice_backend;
        access_log off;
    }
}
```

## 3. Monitoring

### 3.1 Health Checks

**Health Endpoint**:
```python
# auto_voice/web/health.py
from flask import Blueprint, jsonify
import torch

health_bp = Blueprint('health', __name__)

@health_bp.route('/health', methods=['GET'])
def health_check():
    """System health check"""
    status = {
        'status': 'healthy',
        'gpu': {
            'available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        },
        'models': {
            'tts_loaded': tts_engine is not None,
            'vc_loaded': vc_pipeline is not None
        },
        'storage': {
            'cache_size_gb': get_cache_size() / (1024**3)
        }
    }

    if torch.cuda.is_available():
        status['gpu']['memory_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
        status['gpu']['memory_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)

    return jsonify(status), 200
```

**Monitoring Script**:
```bash
#!/bin/bash
# scripts/health_check.sh

# Check service status
curl -f http://localhost:5000/health || exit 1

# Check GPU availability
nvidia-smi || exit 1

# Check disk space
CACHE_SIZE=$(du -sh /data/cache | cut -f1)
echo "Cache size: $CACHE_SIZE"

# Check memory usage
FREE_MEM=$(free -g | awk '/^Mem:/{print $4}')
if [ $FREE_MEM -lt 4 ]; then
    echo "WARNING: Low memory (<4GB free)"
fi

echo "Health check passed"
```

### 3.2 Metrics

**Prometheus Metrics**:
```python
# auto_voice/web/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
tts_requests_total = Counter(
    'autovoice_tts_requests_total',
    'Total TTS requests',
    ['status']
)

vc_requests_total = Counter(
    'autovoice_vc_requests_total',
    'Total voice conversion requests',
    ['status']
)

# Processing time
tts_duration_seconds = Histogram(
    'autovoice_tts_duration_seconds',
    'TTS processing duration',
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

vc_duration_seconds = Histogram(
    'autovoice_vc_duration_seconds',
    'Voice conversion duration',
    buckets=[5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
)

# System metrics
gpu_memory_used_bytes = Gauge(
    'autovoice_gpu_memory_used_bytes',
    'GPU memory usage in bytes'
)

active_conversions = Gauge(
    'autovoice_active_conversions',
    'Number of active conversions'
)
```

**Grafana Dashboard Configuration**:
```json
{
  "dashboard": {
    "title": "AutoVoice Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(autovoice_tts_requests_total[5m])",
            "legendFormat": "TTS"
          },
          {
            "expr": "rate(autovoice_vc_requests_total[5m])",
            "legendFormat": "Voice Conversion"
          }
        ]
      },
      {
        "title": "Processing Duration (p95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, autovoice_tts_duration_seconds)",
            "legendFormat": "TTS p95"
          },
          {
            "expr": "histogram_quantile(0.95, autovoice_vc_duration_seconds)",
            "legendFormat": "VC p95"
          }
        ]
      },
      {
        "title": "GPU Memory Usage",
        "targets": [
          {
            "expr": "autovoice_gpu_memory_used_bytes / 1024 / 1024 / 1024",
            "legendFormat": "GPU Memory (GB)"
          }
        ]
      },
      {
        "title": "Active Conversions",
        "targets": [
          {
            "expr": "autovoice_active_conversions",
            "legendFormat": "Active Jobs"
          }
        ]
      }
    ]
  }
}
```

### 3.3 Logging

**Structured Logging Configuration**:
```python
# config/logging.yaml
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
    filename: /var/log/autovoice/app.log
    maxBytes: 104857600  # 100MB
    backupCount: 10

  error_file:
    class: logging.handlers.RotatingFileHandler
    formatter: json
    filename: /var/log/autovoice/error.log
    maxBytes: 104857600
    backupCount: 10
    level: ERROR

root:
  level: INFO
  handlers: [console, file, error_file]

loggers:
  auto_voice:
    level: DEBUG
    handlers: [console, file]
    propagate: false

  werkzeug:
    level: WARNING
```

**Log Analysis**:
```bash
# Search for errors
grep '"levelname": "ERROR"' /var/log/autovoice/app.log | jq .

# Count requests by endpoint
cat /var/log/autovoice/app.log | jq -r '.endpoint' | sort | uniq -c

# Average processing time
cat /var/log/autovoice/app.log | jq -r 'select(.processing_time != null) | .processing_time' | awk '{sum+=$1; count++} END {print sum/count}'
```

## 4. Troubleshooting

### 4.0 PyTorch and CUDA Environment Issues

#### Issue: Python 3.13 Import Segfaults

**Symptoms**:
```
Segmentation fault (core dumped)
Fatal Python error: Segmentation fault
```

**Diagnosis**:
```bash
# Check Python version
python --version

# Check PyTorch version
python -c "import torch; print(torch.__version__)"
```

**Resolution**:
```bash
# Solution 1: Use Python 3.12 or earlier (recommended)
conda create -n autovoice python=3.12 -y
conda activate autovoice

# Solution 2: Use PyTorch 2.7+ with Python 3.13 (experimental)
pip install torch>=2.7.0 --index-url https://download.pytorch.org/whl/cu121
```

#### Issue: libtorch_global_deps.so Errors

**Symptoms**:
```
ImportError: libtorch_global_deps.so: cannot open shared object file
OSError: libcudart.so.12: cannot open shared object file
```

**Diagnosis**:
```bash
# Check library paths
echo $LD_LIBRARY_PATH

# Find PyTorch libraries
find $(python -c "import torch; print(torch.__path__[0])") -name "*.so"
```

**Resolution**:
```bash
# Add PyTorch lib directory to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(python -c "import torch; print(torch.__path__[0])")/lib:$LD_LIBRARY_PATH

# Make permanent by adding to ~/.bashrc
echo 'export LD_LIBRARY_PATH=$(python -c "import torch; print(torch.__path__[0])" 2>/dev/null)/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```

#### Issue: PyTorch CUDA Version Mismatch

**Symptoms**:
```
RuntimeError: CUDA version mismatch: PyTorch was compiled with CUDA 11.8 but system has CUDA 12.1
The detected CUDA version (12.1) mismatches the version that was used to compile PyTorch (11.8)
```

**Diagnosis**:
```bash
# Check system CUDA version
nvcc --version

# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
```

**Resolution**:
```bash
# Reinstall PyTorch with matching CUDA version

# For CUDA 12.1 (recommended)
pip uninstall torch torchvision torchaudio -y
pip install torch==2.5.1 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip uninstall torch torchvision torchaudio -y
pip install torch==2.5.1 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### Issue: CUDA Extension Build Header Errors

**Symptoms**:
```
fatal error: cuda_runtime.h: No such file or directory
error: command 'nvcc' failed with exit status 1
```

**Diagnosis**:
```bash
# Check if CUDA toolkit is installed
which nvcc

# Check CUDA_HOME
echo $CUDA_HOME

# Check for CUDA headers
ls /usr/local/cuda/include/cuda_runtime.h
```

**Resolution**:
```bash
# Install CUDA toolkit (if missing)
./scripts/install_cuda_toolkit.sh

# Or manually set CUDA_HOME
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Rebuild extensions
python setup.py clean --all
python setup.py build_ext --inplace
```

#### Automated Environment Setup

**Use provided scripts for automated fixes**:

```bash
# Complete environment setup (Python 3.12 + PyTorch + CUDA)
./scripts/setup_pytorch_env.sh

# Build and test everything
./scripts/build_and_test.sh

# Quick verification
./scripts/verify_bindings.py

# Install CUDA toolkit if needed
./scripts/install_cuda_toolkit.sh
```

**Script locations**:
- `scripts/setup_pytorch_env.sh` - Automated PyTorch environment setup
- `scripts/build_and_test.sh` - Build CUDA extensions and run tests
- `scripts/verify_bindings.py` - Verify Python bindings and CUDA availability
- `scripts/install_cuda_toolkit.sh` - Install system CUDA toolkit

### 4.1 Common Issues

#### Issue: GPU Out of Memory

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Diagnosis**:
```bash
# Check GPU memory usage
nvidia-smi

# Check active processes
ps aux | grep python

# Check conversion queue
curl http://localhost:5000/api/v1/queue/status
```

**Resolution**:
```python
# 1. Clear GPU cache
import torch
torch.cuda.empty_cache()

# 2. Reduce batch size in config
# config/production.yaml
gpu:
  max_batch_size: 8  # Reduce from 16

# 3. Use fast preset for large files
quality:
  default_preset: fast

# 4. Restart service to clear fragmented memory
sudo systemctl restart autovoice
```

#### Issue: Slow Processing

**Symptoms**:
- Conversions taking >2x expected time
- High CPU usage, low GPU usage
- Timeout errors

**Diagnosis**:
```bash
# Check GPU utilization
nvidia-smi dmon -s u

# Profile application
python -m cProfile -o profile.stats -m auto_voice.web.app

# Analyze profile
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"
```

**Resolution**:
```bash
# 1. Verify GPU is being used
python -c "import torch; print(torch.cuda.is_available())"

# 2. Check TensorRT optimization
ls models/*.trt  # Should exist if enabled

# 3. Enable TensorRT
python scripts/convert_to_tensorrt.py

# 4. Optimize worker count
# config/production.yaml
server:
  workers: 4  # Match CPU cores

# 5. Clear old cache
find /data/cache -mtime +7 -delete
```

#### Issue: Voice Conversion Quality

**Symptoms**:
- Robotic or unnatural sound
- Poor pitch accuracy
- Low speaker similarity

**Diagnosis**:
```python
# Check quality metrics
from auto_voice.utils.quality_metrics import QualityMetricsAggregator

metrics = QualityMetricsAggregator()
quality = metrics.compute_all_metrics(
    original_audio='original.wav',
    converted_audio='converted.wav',
    reference_f0='reference_f0.npy',
    converted_f0='converted_f0.npy'
)

print(f"Pitch RMSE (Hz): {quality['pitch_accuracy']['rmse_hz']}")
print(f"Speaker Similarity: {quality['speaker_similarity']['cosine_similarity']}")
```

**Resolution**:
```bash
# 1. Use quality preset
curl -X POST http://localhost:5000/api/v1/convert/song \
  -F "quality_preset=quality"

# 2. Improve voice profile quality
# - Record in quiet environment
# - Use 45-60 second samples
# - Create multi-sample profiles

# 3. Use better source audio
# - Lossless formats (WAV, FLAC)
# - Clear, prominent vocals
# - Minimal processing/effects

# 4. Check separation quality
python scripts/test_separation.py --song test.mp3
```

#### Issue: WebSocket Connection Errors

**Symptoms**:
```
WebSocket connection failed
Error: 1006 Abnormal Closure
```

**Diagnosis**:
```bash
# Check WebSocket endpoint
wscat -c ws://localhost:5000/ws/conversion/test-id

# Check nginx WebSocket config
nginx -t

# Check firewall
sudo ufw status
```

**Resolution**:
```nginx
# Fix nginx WebSocket configuration
location /ws/ {
    proxy_pass http://autovoice_backend;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 300s;
}

# Reload nginx
sudo systemctl reload nginx
```

### 4.2 Performance Tuning

**GPU Optimization**:
```yaml
# config/gpu_tuning.yaml
gpu:
  # Enable TensorRT for 2-3x speedup
  enable_tensorrt: true

  # Use FP16 for 2x memory reduction
  mixed_precision: true

  # Optimize batch size for GPU
  max_batch_size: 16

  # Enable CUDA graphs for repeated operations
  cuda_graphs: true

  # Pin memory for faster data transfer
  pin_memory: true
```

**Cache Tuning**:
```yaml
# config/cache_tuning.yaml
cache:
  # Increase separation cache TTL
  separation_ttl_hours: 48

  # Pre-warm frequently used models
  prewarm_models: true

  # Enable aggressive caching
  cache_intermediate_results: true

  # Set max cache size
  max_cache_size_gb: 50
```

**Concurrency Tuning**:
```yaml
# config/concurrency_tuning.yaml
limits:
  # Increase concurrent conversions
  concurrent_conversions: 10

  # Enable request queuing
  queue_enabled: true
  max_queue_size: 100

  # Worker pool size
  worker_threads: 8
```

## 5. Maintenance

### 5.1 Regular Tasks

**Daily**:
```bash
# Check service health
./scripts/health_check.sh

# Review error logs
tail -n 100 /var/log/autovoice/error.log

# Monitor disk usage
df -h /data
```

**Weekly**:
```bash
# Clean old cache files
find /data/cache -mtime +7 -delete

# Rotate logs
logrotate /etc/logrotate.d/autovoice

# Review metrics
curl http://localhost:5000/metrics | grep autovoice
```

**Monthly**:
```bash
# Update dependencies
pip install --upgrade -r requirements.txt

# Backup voice profiles
tar -czf voice_profiles_$(date +%Y%m%d).tar.gz /data/voice_profiles

# Review and optimize models
python scripts/model_pruning.py
```

### 5.2 Backup and Recovery

**Backup Strategy**:
```bash
#!/bin/bash
# scripts/backup.sh

BACKUP_DIR=/backups/autovoice
DATE=$(date +%Y%m%d_%H%M%S)

# Backup voice profiles
tar -czf $BACKUP_DIR/profiles_$DATE.tar.gz /data/voice_profiles

# Backup configuration
cp -r config $BACKUP_DIR/config_$DATE

# Backup models (if custom trained)
tar -czf $BACKUP_DIR/models_$DATE.tar.gz models/

# Rotate old backups (keep 30 days)
find $BACKUP_DIR -mtime +30 -delete

echo "Backup completed: $DATE"
```

**Recovery Procedure**:
```bash
#!/bin/bash
# scripts/restore.sh

BACKUP_FILE=$1

# Stop service
sudo systemctl stop autovoice

# Restore voice profiles
tar -xzf $BACKUP_FILE -C /

# Verify restoration
ls -lh /data/voice_profiles

# Start service
sudo systemctl start autovoice

# Verify health
curl http://localhost:5000/health
```

### 5.3 Updates and Upgrades

**Update Procedure**:
```bash
#!/bin/bash
# scripts/update.sh

# 1. Backup current state
./scripts/backup.sh

# 2. Pull latest code
git fetch origin
git checkout v1.2.0  # Replace with target version

# 3. Update dependencies
source venv/bin/activate
pip install --upgrade -r requirements.txt

# 4. Run database migrations (if applicable)
python scripts/migrate_db.py

# 5. Download new models
python scripts/download_models.py --version 1.2.0

# 6. Test in staging
ENVIRONMENT=staging python -m auto_voice.web.app &
sleep 10
curl http://localhost:5001/health

# 7. Stop staging
kill %1

# 8. Rolling restart production
sudo systemctl restart autovoice

# 9. Verify production
curl http://localhost:5000/health
./scripts/health_check.sh

echo "Update completed successfully"
```

## 6. Security

### 6.1 Authentication

**API Key Authentication**:
```python
# auto_voice/web/auth.py
from functools import wraps
from flask import request, jsonify

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')

        if not api_key:
            return jsonify({'error': 'API key required'}), 401

        if not validate_api_key(api_key):
            return jsonify({'error': 'Invalid API key'}), 403

        return f(*args, **kwargs)

    return decorated_function

@app.route('/api/v1/convert/song', methods=['POST'])
@require_api_key
def convert_song():
    # Conversion logic
    pass
```

### 6.2 Rate Limiting

**Flask-Limiter Configuration**:
```python
# auto_voice/web/app.py
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour"],
    storage_uri="redis://localhost:6379"
)

@app.route('/api/v1/voice/clone', methods=['POST'])
@limiter.limit("10 per hour")
def clone_voice():
    # Voice cloning logic
    pass

@app.route('/api/v1/convert/song', methods=['POST'])
@limiter.limit("20 per hour")
def convert_song():
    # Conversion logic
    pass
```

### 6.3 Input Validation

**File Upload Validation**:
```python
# auto_voice/web/validators.py
import magic

ALLOWED_MIME_TYPES = {
    'audio/wav',
    'audio/mpeg',
    'audio/flac',
    'audio/ogg'
}

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

def validate_audio_file(file) -> bool:
    """Validate uploaded audio file"""
    # Check file size
    file.seek(0, 2)  # Seek to end
    size = file.tell()
    file.seek(0)  # Reset

    if size > MAX_FILE_SIZE:
        raise ValueError(f"File too large: {size} bytes (max {MAX_FILE_SIZE})")

    # Check MIME type
    mime = magic.from_buffer(file.read(1024), mime=True)
    file.seek(0)

    if mime not in ALLOWED_MIME_TYPES:
        raise ValueError(f"Invalid MIME type: {mime}")

    return True
```

## 7. Disaster Recovery

### 7.1 Incident Response

**Severity Levels**:
- **P0 - Critical**: Service completely down, data loss
- **P1 - High**: Major functionality broken, performance degraded >50%
- **P2 - Medium**: Minor functionality issues, performance degraded <50%
- **P3 - Low**: Cosmetic issues, no functionality impact

**Response Procedures**:

**P0 - Service Down**:
```bash
# 1. Check service status
sudo systemctl status autovoice

# 2. Check logs for errors
tail -n 500 /var/log/autovoice/error.log

# 3. Attempt restart
sudo systemctl restart autovoice

# 4. If restart fails, restore from backup
./scripts/restore.sh /backups/autovoice/latest.tar.gz

# 5. Notify stakeholders
./scripts/send_alert.sh "AutoVoice service restored"
```

**P1 - GPU Failure**:
```bash
# 1. Verify GPU status
nvidia-smi

# 2. Reset GPU
sudo nvidia-smi --gpu-reset

# 3. Enable CPU fallback
# config/production.yaml
gpu:
  allow_fallback: true

# 4. Restart service
sudo systemctl restart autovoice

# 5. Schedule maintenance window for GPU replacement
```

### 7.2 Rollback Procedures

**Version Rollback**:
```bash
#!/bin/bash
# scripts/rollback.sh

# 1. Stop service
sudo systemctl stop autovoice

# 2. Checkout previous version
git checkout v1.1.0

# 3. Restore previous dependencies
pip install -r requirements.txt

# 4. Restore previous models
tar -xzf /backups/models_v1.1.0.tar.gz -C models/

# 5. Start service
sudo systemctl start autovoice

# 6. Verify
./scripts/health_check.sh
```

## 8. Voice Conversion Operations

### 8.1 Voice Profile Management

**Create Voice Profile**:
```bash
# CLI
python -m auto_voice.cli voice-clone \
  --audio my_voice.wav \
  --user-id user123 \
  --name "My Singing Voice"

# API
curl -X POST http://localhost:5000/api/v1/voice/clone \
  -F "audio=@my_voice.wav" \
  -F "user_id=user123"
```

**List Profiles**:
```bash
# Get all profiles for user
curl "http://localhost:5000/api/v1/voice/profiles?user_id=user123"
```

**Delete Profile**:
```bash
# Delete specific profile
curl -X DELETE "http://localhost:5000/api/v1/voice/profiles/550e8400-e29b-41d4-a716-446655440000?user_id=user123"
```

### 8.2 Song Conversion Operations

**Start Conversion**:
```bash
# Convert song
curl -X POST http://localhost:5000/api/v1/convert/song \
  -F "song=@song.mp3" \
  -F "target_profile_id=550e8400-e29b-41d4-a716-446655440000" \
  -F "quality_preset=balanced"
```

**Monitor Progress**:
```bash
# Check conversion status
curl "http://localhost:5000/api/v1/convert/status/conv-770e8400-e29b-41d4-a716-446655440002"

# Output:
# {
#   "status": "processing",
#   "progress": 65,
#   "current_stage": "voice_conversion"
# }
```

**Download Results**:
```bash
# Download converted song
curl "http://localhost:5000/api/v1/convert/download/conv-770e8400-e29b-41d4-a716-446655440002/converted.wav" \
  -o converted_song.wav
```

### 8.3 Quality Monitoring

**Check Conversion Quality**:
```python
# scripts/check_quality.py
from auto_voice.utils.quality_metrics import QualityMetricsAggregator

metrics = QualityMetricsAggregator()

# Compute metrics
quality = metrics.compute_all_metrics(
    original_audio='original.wav',
    converted_audio='converted.wav',
    reference_f0='reference_f0.npy',
    converted_f0='converted_f0.npy',
    target_embedding='target.npy',
    converted_embedding='converted.npy'
)

# Check against targets
if quality['pitch_accuracy']['rmse_hz'] > 10.0:
    print("WARNING: Pitch RMSE exceeds target (10 Hz)")

if quality['speaker_similarity']['cosine_similarity'] < 0.85:
    print("WARNING: Speaker similarity below target (0.85)")

if quality['overall_quality_score'] < 0.75:
    print("WARNING: Overall quality below target (0.75)")
```

**Quality Dashboard**:
```bash
# Generate quality report
python scripts/quality_report.py \
  --conversions /data/conversions \
  --output /var/www/html/quality_report.html

# View in browser
# http://autovoice.example.com/quality_report.html
```

### 8.4 Cache Management

**Clear Separation Cache**:
```bash
# Clear all cached separations
find /data/cache/separations -type f -delete

# Clear separations older than 7 days
find /data/cache/separations -mtime +7 -delete
```

**Clear Conversion Cache**:
```bash
# Clear all cached conversions
find /data/cache/conversions -type f -delete

# Clear by user
find /data/cache/conversions -path "*user123*" -delete
```

**Optimize Cache Size**:
```python
# scripts/optimize_cache.py
import os
import shutil

def optimize_cache(cache_dir: str, max_size_gb: float):
    """Remove oldest cache files to stay under max size"""
    # Get all cache files with timestamps
    files = []
    for root, dirs, filenames in os.walk(cache_dir):
        for filename in filenames:
            path = os.path.join(root, filename)
            files.append((path, os.path.getmtime(path), os.path.getsize(path)))

    # Sort by modification time (oldest first)
    files.sort(key=lambda x: x[1])

    # Calculate total size
    total_size = sum(f[2] for f in files)
    max_bytes = max_size_gb * 1024**3

    # Remove oldest files until under limit
    for path, mtime, size in files:
        if total_size <= max_bytes:
            break

        os.remove(path)
        total_size -= size
        print(f"Removed: {path} ({size / 1024**2:.2f} MB)")

if __name__ == '__main__':
    optimize_cache('/data/cache', max_size_gb=20.0)
```

## 9. Incident Examples

### 9.1 Case Study: GPU Memory Leak

**Incident**: GPU memory gradually increasing, eventually OOM

**Investigation**:
```bash
# Monitor GPU memory over time
watch -n 1 nvidia-smi

# Identify memory growth pattern
# Memory increases after each conversion, not fully released
```

**Root Cause**: Cached tensors not being cleared between conversions

**Fix**:
```python
# auto_voice/inference/singing_conversion_pipeline.py

def convert_song(self, ...):
    try:
        # Conversion logic
        result = self._convert(...)
        return result
    finally:
        # Clear GPU cache after each conversion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

**Prevention**:
- Added GPU memory monitoring to metrics
- Set up alerts for >80% GPU memory usage
- Implemented automatic cache clearing

### 9.2 Case Study: Slow WebSocket Connections

**Incident**: WebSocket connections timing out for long conversions

**Investigation**:
```bash
# Check nginx timeout settings
grep -r "timeout" /etc/nginx/sites-enabled/

# Found: proxy_read_timeout 60s (too short)
```

**Root Cause**: Nginx WebSocket timeout too short for long conversions

**Fix**:
```nginx
location /ws/ {
    proxy_pass http://autovoice_backend;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 300s;  # Increased to 5 minutes
}
```

**Prevention**:
- Updated deployment documentation with correct timeout
- Added automated configuration validation
- Set up monitoring for WebSocket connection failures

## 10. Appendix

### 10.1 Configuration Reference

**Complete Configuration Template**:
```yaml
# config/production.yaml
server:
  host: 0.0.0.0
  port: 5000
  workers: 4
  timeout: 300
  debug: false

gpu:
  device: cuda
  device_id: 0
  allow_fallback: true
  max_batch_size: 16
  enable_tensorrt: true
  mixed_precision: true
  pin_memory: true

storage:
  voice_profiles_dir: /data/voice_profiles
  cache_dir: /data/cache
  temp_dir: /tmp/autovoice
  models_dir: /opt/autovoice/models
  max_cache_size_gb: 20

quality:
  default_preset: balanced
  allow_fast_preset: true
  allow_quality_preset: true

  presets:
    fast:
      separation_model: htdemucs
      separation_shifts: 1
      vocoder_quality: medium

    balanced:
      separation_model: htdemucs_ft
      separation_shifts: 2
      vocoder_quality: high

    quality:
      separation_model: htdemucs_ft
      separation_shifts: 3
      vocoder_quality: highest

limits:
  max_audio_size_mb: 100
  max_profile_duration_s: 120
  min_profile_duration_s: 30
  max_song_duration_s: 600
  concurrent_conversions: 5
  requests_per_hour: 100

cache:
  separation_ttl_hours: 24
  conversion_ttl_hours: 24
  voice_profile_ttl_days: 90
  cleanup_interval_hours: 6

logging:
  level: INFO
  format: json
  file: /var/log/autovoice/app.log
  max_bytes: 104857600
  backup_count: 10

monitoring:
  prometheus_enabled: true
  prometheus_port: 9090
  health_check_interval: 60

security:
  api_key_required: true
  rate_limiting_enabled: true
  allowed_origins: ["https://autovoice.example.com"]
```

### 10.2 Command Reference

**Service Management**:
```bash
# Start service
sudo systemctl start autovoice

# Stop service
sudo systemctl stop autovoice

# Restart service
sudo systemctl restart autovoice

# Reload configuration
sudo systemctl reload autovoice

# Check status
sudo systemctl status autovoice

# View logs
journalctl -u autovoice -f
```

**Diagnostic Commands**:
```bash
# Check GPU status
nvidia-smi

# Check disk usage
df -h

# Check memory
free -h

# Check network
netstat -tulpn | grep 5000

# Check processes
ps aux | grep autovoice

# Check logs
tail -f /var/log/autovoice/app.log
```

**Maintenance Commands**:
```bash
# Clear cache
rm -rf /data/cache/*

# Backup voice profiles
tar -czf profiles_backup.tar.gz /data/voice_profiles

# Restore voice profiles
tar -xzf profiles_backup.tar.gz -C /

# Update dependencies
pip install --upgrade -r requirements.txt

# Download models
python scripts/download_models.py
```

### 10.3 Contact Information

**Support Channels**:
- **Emergency**: oncall@example.com
- **Operations**: ops@example.com
- **Development**: dev@example.com
- **Documentation**: https://docs.autovoice.example.com

**Escalation Path**:
1. On-call Engineer
2. DevOps Lead
3. Engineering Manager
4. VP Engineering
