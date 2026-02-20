# Production Deployment Readiness Checklist

**Date:** 2026-02-02
**Status:** DRAFT (Pending coverage validation)
**Trigger Condition:** Overall project coverage ≥80% after Cycle 2

---

## Pre-Deployment Verification

### Coverage Metrics ✅ (if triggered)
- [ ] Overall project coverage: ≥80%
- [ ] Inference module coverage: ≥85%
- [ ] Database module coverage: ≥70%
- [ ] Audio processing coverage: ≥70%
- [ ] Web API coverage: ≥75%
- [ ] Test pass rate: ≥95%

### Code Quality
- [ ] All beads issues closed (current: 44 closed, 0 open) ✅
- [ ] No blocking bugs (P0/P1)
- [ ] Static analysis passing (ruff, mypy if used)
- [ ] Security scan clean (no critical vulnerabilities)

### Testing Infrastructure
- [ ] All tests passing on CI/CD
- [ ] Integration tests covering E2E workflows
- [ ] Performance benchmarks within acceptable range
- [ ] Load testing completed (if applicable)

---

## Deployment Checklist

### 1. Environment Configuration

#### Production Environment Variables
```bash
# Database
AUTOVOICE_DB_TYPE=mysql
MYSQL_HOST=localhost
MYSQL_USER=autovoice_user
MYSQL_PASSWORD=<secure_password>
MYSQL_DATABASE=autovoice_production

# Model Paths
MODEL_CHECKPOINT_DIR=/var/autovoice/models
LORA_WEIGHTS_DIR=/var/autovoice/loras
SPEAKER_EMBEDDINGS_DIR=/var/autovoice/embeddings

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=https://yourdomain.com
SECRET_KEY=<generate_secure_key>

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_ARCH_LIST=11.0

# Feature Flags
ENABLE_TENSORRT=true
ENABLE_BROWSER_VOICE_PROFILES=true
ENABLE_KARAOKE_MODE=true
```

#### System Dependencies
```bash
# CUDA 13.0 (Jetson Thor)
export CUDA_HOME=/usr/local/cuda-13.0
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Conda environment
conda create -n autovoice_prod python=3.10
conda activate autovoice_prod
pip install -r requirements.txt
pip install torch==2.6.0.dev --index-url https://download.pytorch.org/whl/cu130
```

### 2. Database Setup

#### MySQL Schema Initialization
```bash
# Create production database
mysql -u root -p << EOF
CREATE DATABASE IF NOT EXISTS autovoice_production CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'autovoice_user'@'localhost' IDENTIFIED BY '<secure_password>';
GRANT ALL PRIVILEGES ON autovoice_production.* TO 'autovoice_user'@'localhost';
FLUSH PRIVILEGES;
EOF

# Apply migrations
cd /home/kp/repo2/autovoice
python -m src.auto_voice.db.migrations.init_db
```

#### Backup Strategy
- [ ] Automated daily backups configured
- [ ] Backup retention: 30 days
- [ ] Test restoration procedure documented

### 3. Model Deployment

#### Model Artifacts
- [ ] HQ-SVC checkpoint downloaded
- [ ] Voice embeddings models (WavLM, ECAPA) available
- [ ] Vocoder models (NSF-HiFiGAN) present
- [ ] TensorRT engines compiled (optional)

#### Model Loading Test
```bash
python -c "
from src.auto_voice.inference.pipeline import VoiceConversionPipeline
pipeline = VoiceConversionPipeline()
print('Pipeline loaded successfully')
"
```

### 4. Service Configuration

#### Systemd Service Unit
```ini
# /etc/systemd/system/autovoice-api.service
[Unit]
Description=AutoVoice API Service
After=network.target mysql.service

[Service]
Type=simple
User=autovoice
Group=autovoice
WorkingDirectory=/home/kp/repo2/autovoice
Environment="PATH=/home/kp/anaconda3/envs/autovoice_prod/bin:/usr/local/cuda-13.0/bin"
Environment="LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64"
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="AUTOVOICE_DB_TYPE=mysql"
ExecStart=/home/kp/anaconda3/envs/autovoice_prod/bin/python -m src.auto_voice.web.app
Restart=on-failure
RestartSec=10s

# Resource limits
MemoryLimit=16G
CPUQuota=400%

[Install]
WantedBy=multi-user.target
```

#### Enable and Start Service
```bash
sudo systemctl daemon-reload
sudo systemctl enable autovoice-api
sudo systemctl start autovoice-api
sudo systemctl status autovoice-api
```

### 5. Reverse Proxy Configuration

#### Nginx Configuration
```nginx
# /etc/nginx/sites-available/autovoice
server {
    listen 443 ssl http2;
    server_name autovoice.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/autovoice.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/autovoice.yourdomain.com/privkey.pem;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;

    # API proxy
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support (for realtime features)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Timeouts for long-running inference
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    # File upload limits
    client_max_body_size 100M;
}
```

### 6. Monitoring and Observability

#### Prometheus Metrics
```yaml
# /etc/prometheus/prometheus.yml
scrape_configs:
  - job_name: 'autovoice'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

#### Grafana Dashboards
- [ ] API request rate and latency
- [ ] Voice conversion throughput (audios/hour)
- [ ] GPU utilization and memory
- [ ] Database query performance
- [ ] Error rate and types

#### Logging
```python
# Configure structured logging
import logging
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ]
)
```

### 7. Security Hardening

#### Firewall Rules
```bash
# Allow only necessary ports
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP (redirect to HTTPS)
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

#### API Authentication
- [ ] JWT token-based authentication implemented
- [ ] Rate limiting configured (e.g., 100 req/min per user)
- [ ] API key management system in place
- [ ] CORS properly configured for allowed origins

#### Secrets Management
- [ ] Environment variables stored securely (not in code)
- [ ] Database passwords rotated regularly
- [ ] TLS certificates auto-renewed (Let's Encrypt)

### 8. Performance Optimization

#### TensorRT Optimization (Optional)
```bash
# Convert models to TensorRT for faster inference
python -m src.auto_voice.export.tensorrt_engine \
    --checkpoint ckpts/hq_svc_24khz/best.pt \
    --output models/hq_svc_24khz.trt \
    --precision fp16
```

#### Model Caching
- [ ] Speaker embeddings cached in Redis/memory
- [ ] Frequently used LoRA weights preloaded
- [ ] WavLM model loaded once on startup

#### Connection Pooling
- [ ] Database connection pool configured (10-20 connections)
- [ ] Redis connection pool for caching

### 9. Documentation

#### API Documentation
- [ ] OpenAPI/Swagger spec generated
- [ ] Example requests/responses documented
- [ ] Authentication flow explained
- [ ] Error codes and troubleshooting guide

#### Operations Runbook
- [ ] Deployment procedure documented
- [ ] Rollback procedure tested
- [ ] Common issues and solutions listed
- [ ] Contact information for on-call team

### 10. Rollout Strategy

#### Phased Rollout
1. **Internal testing (Week 1)**
   - Deploy to staging environment
   - Team testing with real workloads
   - Monitor for issues

2. **Beta release (Week 2)**
   - Limited user group (10-50 users)
   - Gather feedback on performance/quality
   - Fix any critical issues

3. **General availability (Week 3)**
   - Full production release
   - Monitor metrics closely for first 48 hours
   - Be ready for quick rollback if needed

#### Rollback Plan
```bash
# Quick rollback to previous version
cd /home/kp/repo2/autovoice
git checkout <previous_tag>
sudo systemctl restart autovoice-api

# Database rollback (if schema changed)
mysql autovoice_production < backups/pre_deployment_backup.sql
```

---

## Post-Deployment Validation

### Smoke Tests (Run immediately after deployment)
```bash
# Health check
curl https://autovoice.yourdomain.com/api/health

# Speaker profile creation
curl -X POST https://autovoice.yourdomain.com/api/speakers \
  -H "Authorization: Bearer $TOKEN" \
  -F "audio=@test_voice.wav" \
  -F "name=Test Speaker"

# Voice conversion
curl -X POST https://autovoice.yourdomain.com/api/convert \
  -H "Authorization: Bearer $TOKEN" \
  -F "source_audio=@input.wav" \
  -F "target_speaker_id=123" \
  -F "output_format=wav"
```

### Success Criteria
- [ ] All health checks passing
- [ ] Voice conversion completing in <10s (24kHz, 10s audio)
- [ ] Speaker profile creation working
- [ ] Browser voice profile workflow functional
- [ ] Karaoke mode operational (if enabled)
- [ ] No memory leaks (monitor over 24 hours)
- [ ] Error rate <0.1%

---

## Track Updates

### Close Completed Tracks
```bash
cd /home/kp/repo2/autovoice

# Update tracks.md
# Mark coverage-report-generation_20260201 as complete
# Mark production-deployment-prep_20260201 as in-progress

git add conductor/tracks.md conductor/PRODUCTION_DEPLOYMENT_READINESS.md
git commit -m "chore: Mark coverage track complete, begin production deployment

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
git push origin main
```

### Beads Cleanup
```bash
# Verify all issues closed
bd stats

# Sync with remote
bd sync
```

---

## Sign-Off

### Technical Approval
- [ ] Tech Lead: _________________ Date: _______
- [ ] QA Lead: __________________ Date: _______
- [ ] Security Lead: _____________ Date: _______

### Deployment Authorization
- [ ] Product Owner: _____________ Date: _______
- [ ] Engineering Manager: _______ Date: _______

---

**Status:** Ready for deployment pending final approvals
**Next Action:** Monitor coverage validation results, proceed if ≥80%
