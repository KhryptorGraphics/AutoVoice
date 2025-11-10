# Production Deployment Runbook - AutoVoice

**Version**: 1.0
**Last Updated**: 2025-11-01
**Purpose**: Step-by-step guide for deploying AutoVoice to production environments

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Pre-Deployment Checklist](#pre-deployment-checklist)
3. [Deployment Steps](#deployment-steps)
4. [Start Monitoring Stack](#start-monitoring-stack)
5. [Functional Smoke Tests](#functional-smoke-tests)
6. [Performance Validation](#performance-validation)
7. [Security Hardening](#security-hardening)
8. [Backup Procedures](#backup-procedures)
9. [Rollback Procedure](#rollback-procedure)
10. [Post-Deployment Validation](#post-deployment-validation)
11. [Monitoring and Alerts](#monitoring-and-alerts)
12. [Troubleshooting](#troubleshooting)
13. [Maintenance Schedule](#maintenance-schedule)
14. [Support Contacts](#support-contacts)
15. [Appendix](#appendix)

---

## Executive Summary

### Readiness Status

AutoVoice is **production-ready** with the following deployment options:

- **Docker Compose**: Recommended for single-server deployments
- **Docker Swarm**: For multi-node orchestration
- **Kubernetes**: For enterprise-scale deployments (requires custom manifests)

### Prerequisites

**Hardware:**
- NVIDIA GPU with CUDA 11.8+ support (minimum 8GB VRAM)
- 16GB+ system RAM
- 50GB+ available disk space
- Multi-core CPU (4+ cores recommended)

**Software:**
- Docker 24.0+ with Compose v2
- NVIDIA Container Toolkit
- Linux kernel 5.x+ (Ubuntu 20.04+ or equivalent)

**Network:**
- Ports 5000 (API), 8080 (WebSocket), 3000 (Grafana), 9090 (Prometheus)
- Outbound internet access for model downloads (initial setup)

### Deployment Timeline

- **Preparation**: 30 minutes
- **Deployment**: 15 minutes
- **Validation**: 30 minutes
- **Total**: ~75 minutes

---

## Pre-Deployment Checklist

### Infrastructure Requirements

- [ ] NVIDIA GPU available and tested (`nvidia-smi` works)
- [ ] Docker and Docker Compose installed
- [ ] NVIDIA Container Toolkit installed and configured
- [ ] Sufficient disk space (50GB+ free)
- [ ] Network ports available (5000, 8080, 3000, 9090, 6379)
- [ ] Firewall rules configured

### Software Requirements

- [ ] Repository cloned: `git clone https://github.com/khryptorgraphics/autovoice.git`
- [ ] Environment file created: `.env` (see Appendix)
- [ ] SSL certificates obtained (if using HTTPS)
- [ ] Backup storage configured

### Validation Completed

- [ ] Health check validation passed: `./scripts/validate_health_checks.sh`
- [ ] E2E tests passed: `./scripts/run_e2e_tests.sh --quick`
- [ ] Production readiness checklist reviewed: `docs/production_readiness_checklist.md`

---

## Deployment Steps

### Step 1: Environment Preparation

**Verify GPU and drivers:**
```bash
# Check NVIDIA driver
nvidia-smi

# Expected output: GPU info, driver version, CUDA version
# Minimum: CUDA 11.8, Driver 520+

# Test Docker GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

**Verify Docker Compose:**
```bash
# Check Docker Compose version
docker compose version

# Expected: Docker Compose version v2.x.x or higher
```

### Step 2: Configuration

**Create environment file:**
```bash
cd /path/to/autovoice

# Create .env file with required variables
cat > .env << 'EOF'
# Application
LOG_LEVEL=INFO
LOG_FORMAT=json
FLASK_ENV=production

# GPU
CUDA_VISIBLE_DEVICES=0

# Monitoring
PROMETHEUS_ENABLED=true
METRICS_PORT=5000

# Grafana
GRAFANA_ADMIN_PASSWORD=<strong-password-here>

# Redis
REDIS_PASSWORD=<strong-password-here>
EOF

# Edit the file to set strong passwords
nano .env
```

**Important**: Replace `<strong-password-here>` with actual strong passwords before deployment.

**Verify configuration:**
```bash
# Check syntax
docker compose config

# Should output valid YAML without errors
```

### Step 3: Start Core Services

**Deploy AutoVoice application and Redis:**
```bash
# Start core services (without monitoring profile)
docker compose up -d

# Verify services are starting
docker compose ps

# Expected output:
# auto_voice_app        starting
# auto_voice_redis      running (healthy)
```

**Wait for health checks:**
```bash
# Health check has start_period of 40s
echo "Waiting for services to become healthy (40s start period)..."
sleep 45

# Check container health status
docker compose ps

# Expected output:
# auto_voice_app        running (healthy)
# auto_voice_redis      running (healthy)
```

**Monitor startup logs:**
```bash
# Follow application logs during startup
docker compose logs -f auto-voice-app

# Look for:
# ✓ "Model loaded successfully"
# ✓ "GPU initialized"
# ✓ "Server started on port 5000"
# ✓ No ERROR or CRITICAL messages

# Press Ctrl+C to stop following logs
```

### Step 4: Health Validation

**Run automated health check validation:**
```bash
# Execute comprehensive health check validation
./scripts/validate_health_checks.sh

# Expected output:
# === AutoVoice Health Check Validation ===
# [1/4] Testing /health endpoint...
# ✓ Status: 200 OK
# ✓ Response time: XXXms
# ✓ Content-Type: application/json
# ✓ Service status: healthy
# ✓ GPU available: true
# ✓ Models object present
# ✓ Components object present
#
# [2/4] Testing /health/live endpoint...
# ✓ Status: 200 OK
# ✓ Response time: XXXms
# ✓ Liveness status: alive
#
# [3/4] Testing /health/ready endpoint...
# ✓ Status: 200
# ✓ Service is ready
#
# [4/4] Testing Docker health check...
# ✓ Container status: healthy
#
# === Validation Summary ===
# ✓ All health checks PASSED
```

**Manual health check verification:**
```bash
# Test main health endpoint
curl -s http://localhost:5000/health | jq .

# Expected response:
# {
#   "status": "healthy",
#   "gpu": {
#     "available": true,
#     "device_count": 1,
#     "memory_allocated_gb": 2.5,
#     "memory_reserved_gb": 3.0
#   },
#   "models": {
#     "tts_loaded": true,
#     "vc_loaded": true
#   },
#   "components": {
#     "synthesizer": true,
#     "vocoder": true
#   }
# }

# Test liveness probe
curl -s http://localhost:5000/health/live | jq .
# Expected: {"status": "alive"}

# Test readiness probe
curl -s http://localhost:5000/health/ready | jq .
# Expected: {"status": "ready", "components": {...}}
```

**Verify GPU access:**
```bash
# Check GPU is accessible from container
docker exec auto_voice_app nvidia-smi

# Should show GPU information and current usage
```

---

## Start Monitoring Stack

### Step 5: Deploy Monitoring Services

**Start Prometheus and Grafana:**
```bash
# Start monitoring stack
docker compose --profile monitoring up -d

# Verify all services running
docker compose ps

# Expected output:
# auto_voice_app        running (healthy)
# auto_voice_redis      running (healthy)
# auto_voice_prometheus running
# auto_voice_grafana    running
```

**Verify Prometheus targets:**
```bash
# Check Prometheus is scraping metrics
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health}'

# Expected output:
# {
#   "job": "autovoice",
#   "health": "up"
# }
```

**Access Prometheus UI:**
- URL: http://localhost:9090
- Navigate to Status → Targets
- Verify "autovoice" target is UP
- Query test: `autovoice_synthesis_requests_total`

### Step 6: Configure Grafana

**Initial login:**
- URL: http://localhost:3000
- Username: `admin`
- Password: (from `GRAFANA_ADMIN_PASSWORD` in `.env`)

**Verify datasource:**
1. Navigate to Configuration → Data Sources
2. Verify "Prometheus" datasource exists
3. Click "Test" → Should show "Data source is working"

**Verify dashboards:**
1. Navigate to Dashboards → Browse
2. Verify "AutoVoice Monitoring Dashboard" is present
3. Open dashboard and verify panels are loading data
4. Check for:
   - HTTP Requests per Second (should show data)
   - GPU Utilization (should show current GPU usage)
   - Synthesis Duration (should show metrics)

**If dashboards are missing:**
```bash
# Check Grafana logs
docker logs auto_voice_grafana | grep -i provision

# Should see:
# "Provisioning dashboards"
# "Dashboard provisioned"

# Restart Grafana if needed
docker compose restart grafana
```

---

## Functional Smoke Tests

### Step 7: API Smoke Tests

**Test TTS synthesis endpoint:**
```bash
# Simple TTS request
curl -X POST http://localhost:5000/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a production test.",
    "speaker_id": "default"
  }' \
  --output test_output.wav

# Verify:
# - HTTP 200 response
# - test_output.wav file created
# - File size > 0 bytes

# Check audio file
file test_output.wav
# Expected: "RIFF (little-endian) data, WAVE audio"

# Play audio (if available)
# aplay test_output.wav  # Linux
# afplay test_output.wav # macOS
```

**Test metrics endpoint:**
```bash
# Check metrics are being exported
curl -s http://localhost:5000/metrics | grep autovoice

# Expected output (sample):
# autovoice_synthesis_requests_total{speaker_id="default",success="true"} 1
# autovoice_http_requests_total{method="POST",endpoint="/synthesize",status="200"} 1
# autovoice_gpu_memory_used_bytes{device_id="0"} 4294967296
# autovoice_gpu_utilization_percent{device_id="0"} 45.5
```

**Test WebSocket endpoint (optional):**
```bash
# Install wscat if needed: npm install -g wscat
wscat -c ws://localhost:8080/ws

# Send test message:
# {"action": "ping"}

# Expected response:
# {"action": "pong", "timestamp": "..."}
```

### Step 8: Log Verification

**Check application logs:**
```bash
# View recent logs
docker compose logs --tail=100 auto-voice-app

# Look for:
# ✓ No ERROR or CRITICAL messages
# ✓ Successful synthesis operations
# ✓ Metrics being recorded
# ✓ GPU operations completing

# Check for warnings
docker compose logs auto-voice-app | grep -i warn

# Investigate any unexpected warnings
```

**Check Redis logs:**
```bash
docker compose logs auto-voice-redis

# Should show:
# - Redis server started
# - Ready to accept connections
# - No connection errors
```

---

## Performance Validation

### Step 9: Run Performance Tests

**Quick E2E validation:**
```bash
# Run quick E2E test suite
./scripts/run_e2e_tests.sh --quick

# Expected output:
# ✓ All tests PASSED
# ✓ Quality gates met
# Report saved to: validation_results/e2e/e2e_test_report_*.md
```

**Monitor GPU during tests:**
```bash
# In a separate terminal, watch GPU usage
watch -n 1 nvidia-smi

# Observe:
# - GPU utilization: 60-90% during synthesis
# - Memory usage: stable, no leaks
# - Temperature: within safe limits (<85°C)
```

**Check performance metrics:**
```bash
# Query Prometheus for latency
curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.95,rate(autovoice_synthesis_duration_seconds_bucket[5m]))' | jq '.data.result[0].value[1]'

# Expected: <0.5 (p95 latency under 500ms for TTS)

# Check error rate
curl -s 'http://localhost:9090/api/v1/query?query=rate(autovoice_synthesis_requests_total{success="false"}[5m])/rate(autovoice_synthesis_requests_total[5m])*100' | jq '.data.result[0].value[1]'

# Expected: <1 (error rate under 1%)
```

---

## Security Hardening

### Step 10: Secure the Deployment

**Change default passwords:**
```bash
# Update .env file
nano .env

# Change:
# - GRAFANA_ADMIN_PASSWORD (use strong password)
# - REDIS_PASSWORD (if Redis auth is enabled)

# Restart services
docker compose restart grafana redis
```

**Configure firewall:**
```bash
# Allow only necessary ports
sudo ufw allow 5000/tcp   # API
sudo ufw allow 8080/tcp   # WebSocket
sudo ufw allow 3000/tcp   # Grafana (restrict to admin IPs)
sudo ufw allow 9090/tcp   # Prometheus (restrict to admin IPs)

# Deny direct Redis access from external
sudo ufw deny 6379/tcp

# Enable firewall
sudo ufw enable
```

**Enable TLS/HTTPS (recommended):**
```bash
# Option 1: Use reverse proxy (nginx, Caddy)
# Option 2: Configure Flask with SSL certificates

# Example nginx config:
# server {
#     listen 443 ssl;
#     server_name autovoice.example.com;
#     ssl_certificate /path/to/cert.pem;
#     ssl_certificate_key /path/to/key.pem;
#     location / {
#         proxy_pass http://localhost:5000;
#     }
# }
```

**Restrict Grafana access:**
```bash
# Edit docker-compose.yml to bind Grafana to localhost only
# ports:
#   - "127.0.0.1:3000:3000"

# Access via SSH tunnel:
# ssh -L 3000:localhost:3000 user@server
```

**Set resource limits:**
```bash
# Already configured in docker-compose.yml:
# - CPU: 4 cores
# - Memory: 8GB
# - GPU: 1 device

# Verify limits are applied
docker inspect auto_voice_app | jq '.[0].HostConfig.Resources'
```

---

## Backup Procedures

### Step 11: Configure Backups

**Backup locations:**
- Configuration: `.env`, `docker-compose.yml`, `config/`
- Data: `data/`, `models/`
- Logs: `logs/`
- Volumes: Docker volumes (redis-data, prometheus-data, grafana-data)

**Create backup script:**
```bash
#!/bin/bash
# backup.sh - AutoVoice backup script

BACKUP_DIR="/backup/autovoice/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup configuration
cp -r .env docker-compose.yml config/ "$BACKUP_DIR/"

# Backup data and models
cp -r data/ models/ "$BACKUP_DIR/"

# Backup Docker volumes
docker run --rm -v auto_voice_redis_data:/data -v "$BACKUP_DIR":/backup alpine tar czf /backup/redis-data.tar.gz -C /data .
docker run --rm -v auto_voice_prometheus_data:/data -v "$BACKUP_DIR":/backup alpine tar czf /backup/prometheus-data.tar.gz -C /data .
docker run --rm -v auto_voice_grafana_data:/data -v "$BACKUP_DIR":/backup alpine tar czf /backup/grafana-data.tar.gz -C /data .

echo "Backup completed: $BACKUP_DIR"
```

**Schedule automated backups:**
```bash
# Add to crontab
crontab -e

# Daily backup at 2 AM
0 2 * * * /path/to/autovoice/backup.sh

# Weekly cleanup (keep last 30 days)
0 3 * * 0 find /backup/autovoice -type d -mtime +30 -exec rm -rf {} +
```

**Test backup restoration:**
```bash
# Stop services
docker compose down

# Restore configuration
cp -r /backup/autovoice/YYYYMMDD_HHMMSS/.env .
cp -r /backup/autovoice/YYYYMMDD_HHMMSS/config/ .

# Restore volumes
docker run --rm -v auto_voice_redis_data:/data -v /backup/autovoice/YYYYMMDD_HHMMSS:/backup alpine tar xzf /backup/redis-data.tar.gz -C /data

# Restart services
docker compose up -d
```

---

## Rollback Procedure

### Step 12: Rollback to Previous Version

**Scenario**: Deployment fails or critical issues discovered

**Immediate rollback steps:**

1. **Stop current deployment:**
```bash
docker compose down
```

2. **Restore previous configuration:**
```bash
# Restore from backup
cp /backup/autovoice/PREVIOUS_BACKUP/.env .
cp /backup/autovoice/PREVIOUS_BACKUP/docker-compose.yml .
cp -r /backup/autovoice/PREVIOUS_BACKUP/config/ .
```

3. **Restore previous Docker image:**
```bash
# If you tagged previous version
docker tag autovoice/autovoice:previous autovoice/autovoice:latest

# Or pull specific version
docker pull autovoice/autovoice:v1.0.0
docker tag autovoice/autovoice:v1.0.0 autovoice/autovoice:latest
```

4. **Restore data volumes (if needed):**
```bash
# Only if data corruption occurred
docker run --rm -v auto_voice_redis_data:/data -v /backup/autovoice/PREVIOUS_BACKUP:/backup alpine tar xzf /backup/redis-data.tar.gz -C /data
```

5. **Restart services:**
```bash
docker compose up -d
```

6. **Validate rollback:**
```bash
# Run health checks
./scripts/validate_health_checks.sh

# Check logs
docker compose logs --tail=50 auto-voice-app

# Test API
curl -s http://localhost:5000/health | jq .
```

**Post-rollback actions:**
- Document the issue that caused rollback
- Review logs for root cause
- Create incident report
- Plan fix for next deployment

---

## Post-Deployment Validation

### Step 13: Immediate Validation (0-1 hour)

**Health monitoring:**
```bash
# Monitor health endpoints every 5 minutes
watch -n 300 './scripts/validate_health_checks.sh'

# Watch for errors in logs
docker compose logs -f auto-voice-app | grep -i error
```

**Performance baseline:**
```bash
# Capture initial metrics
curl -s http://localhost:9090/api/v1/query?query=autovoice_synthesis_duration_seconds | jq . > baseline_metrics.json

# Monitor GPU
nvidia-smi dmon -s u -c 60  # Monitor for 1 hour
```

**Smoke test suite:**
```bash
# Run comprehensive smoke tests
for i in {1..10}; do
  curl -X POST http://localhost:5000/synthesize \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"Test $i\", \"speaker_id\": \"default\"}" \
    -o /dev/null -w "Request $i: %{http_code} %{time_total}s\n"
  sleep 5
done
```

### Step 14: 24-Hour Validation

**Metrics to monitor:**
- [ ] Error rate < 1%
- [ ] p95 latency < 500ms
- [ ] GPU utilization 60-80%
- [ ] Memory usage stable (no leaks)
- [ ] No service restarts
- [ ] Disk space sufficient

**Grafana dashboard checks:**
1. Open AutoVoice dashboard
2. Set time range to "Last 24 hours"
3. Verify all panels show healthy metrics
4. Check for anomalies or spikes

**Log analysis:**
```bash
# Count errors in last 24 hours
docker compose logs --since 24h auto-voice-app | grep -c ERROR

# Should be 0 or very low

# Check for memory warnings
docker compose logs --since 24h auto-voice-app | grep -i "memory\|oom"

# Should be empty
```

---

## Monitoring and Alerts

### Key Metrics and Thresholds

**Critical Alerts** (immediate action required):
- Service down (health check fails)
- Error rate > 5%
- p95 latency > 1s
- GPU memory > 95%
- Disk space < 10%

**Warning Alerts** (investigate within 1 hour):
- Error rate > 2%
- p95 latency > 500ms
- GPU utilization > 90%
- Memory usage increasing trend

**Monitoring dashboards:**
- **AutoVoice Overview**: http://localhost:3000/d/autovoice
- **Prometheus Targets**: http://localhost:9090/targets
- **Prometheus Alerts**: http://localhost:9090/alerts

**Alert configuration:**
See `docs/monitoring-guide.md` for detailed alert rules and Prometheus configuration.

**On-call procedures:**
1. Check Grafana dashboards for anomalies
2. Review application logs: `docker compose logs auto-voice-app`
3. Check GPU status: `nvidia-smi`
4. Verify network connectivity
5. Escalate if issue persists > 15 minutes

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: Container fails to start with GPU error

**Symptoms:**
```
Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
```

**Solution:**
```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

#### Issue: Health check fails with 503

**Symptoms:**
```
✗ Status: 503 (expected 200)
```

**Solution:**
```bash
# Check if model is loading
docker compose logs auto-voice-app | grep -i "model"

# If model download is in progress, wait
# If model failed to load, check disk space and permissions

# Check GPU availability
docker exec auto_voice_app nvidia-smi

# Restart service
docker compose restart auto-voice-app
```

#### Issue: High latency (p95 > 1s)

**Symptoms:**
- Slow API responses
- Grafana shows high latency

**Solution:**
```bash
# Check GPU utilization
nvidia-smi

# If GPU is at 100%, scale horizontally or upgrade GPU
# If GPU is low, check CPU bottleneck

# Check for memory leaks
docker stats auto_voice_app

# Restart if memory is high
docker compose restart auto-voice-app
```

#### Issue: Grafana dashboards show no data

**Symptoms:**
- Empty panels in Grafana
- "No data" messages

**Solution:**
```bash
# Check Prometheus is scraping
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[0].health'

# Should return "up"

# Check metrics endpoint
curl http://localhost:5000/metrics | grep autovoice

# If no metrics, check PROMETHEUS_ENABLED=true in .env

# Verify Grafana datasource
curl -u admin:$GRAFANA_ADMIN_PASSWORD http://localhost:3000/api/datasources

# Restart Grafana
docker compose restart grafana
```

#### Issue: Out of disk space

**Symptoms:**
```
Error: No space left on device
```

**Solution:**
```bash
# Check disk usage
df -h

# Clean Docker resources
docker system prune -a --volumes

# Clean old logs
find ./logs -name "*.log" -mtime +7 -delete

# Rotate logs
docker compose logs --tail=1000 auto-voice-app > logs/app_recent.log
docker compose restart auto-voice-app
```

#### Issue: Redis connection errors

**Symptoms:**
```
Error: Connection refused to Redis
```

**Solution:**
```bash
# Check Redis is running
docker compose ps redis

# Check Redis logs
docker compose logs redis

# Test Redis connection
docker exec auto_voice_redis redis-cli ping
# Should return "PONG"

# Restart Redis
docker compose restart redis
```

### Debug Commands

**View all container logs:**
```bash
docker compose logs --tail=100 -f
```

**Execute commands in container:**
```bash
docker exec -it auto_voice_app bash
```

**Check container resource usage:**
```bash
docker stats
```

**Inspect container configuration:**
```bash
docker inspect auto_voice_app | jq .
```

**Test network connectivity:**
```bash
docker exec auto_voice_app curl -s http://localhost:5000/health
```

---

## Maintenance Schedule

### Daily Tasks

- [ ] Check Grafana dashboards for anomalies
- [ ] Review error logs: `docker compose logs auto-voice-app | grep ERROR`
- [ ] Verify health checks: `./scripts/validate_health_checks.sh`
- [ ] Monitor disk space: `df -h`

### Weekly Tasks

- [ ] Review performance metrics (latency, throughput, error rate)
- [ ] Check for Docker image updates
- [ ] Rotate logs: `docker compose logs --tail=10000 > logs/weekly_backup.log`
- [ ] Test backup restoration procedure
- [ ] Review security logs

### Monthly Tasks

- [ ] Update dependencies and Docker images
- [ ] Review and update alert thresholds
- [ ] Capacity planning review (GPU, CPU, memory, disk)
- [ ] Security audit (passwords, firewall rules, certificates)
- [ ] Performance benchmarking: `./scripts/run_e2e_tests.sh --full`
- [ ] Documentation review and updates

### Quarterly Tasks

- [ ] Disaster recovery drill (full backup and restore)
- [ ] Security vulnerability scan
- [ ] Review and update runbook
- [ ] Stakeholder review meeting
- [ ] Evaluate new features and upgrades

---

## Support Contacts

### Internal Team

- **DevOps Lead**: devops@example.com
- **ML Engineering**: ml-team@example.com
- **On-Call Rotation**: oncall@example.com

### External Resources

- **GitHub Issues**: https://github.com/khryptorgraphics/autovoice/issues
- **Documentation**: https://github.com/khryptorgraphics/autovoice/tree/main/docs
- **Monitoring Guide**: `docs/monitoring-guide.md`
- **Production Checklist**: `docs/production_readiness_checklist.md`

### Escalation Path

1. **Level 1** (0-15 min): On-call engineer investigates
2. **Level 2** (15-30 min): DevOps lead engaged
3. **Level 3** (30+ min): ML engineering team and management notified

---

## Appendix

### A. Environment Variables Reference

**Application Settings:**
```bash
LOG_LEVEL=INFO              # Logging level (DEBUG, INFO, WARN, ERROR)
LOG_FORMAT=json             # Log format (json, text)
FLASK_ENV=production        # Flask environment
PYTHONPATH=/app/src         # Python path
```

**GPU Settings:**
```bash
CUDA_VISIBLE_DEVICES=0      # GPU device IDs (0, 1, 0,1, etc.)
NVIDIA_VISIBLE_DEVICES=all  # NVIDIA runtime setting
NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

**Monitoring:**
```bash
PROMETHEUS_ENABLED=true     # Enable Prometheus metrics
METRICS_PORT=5000           # Metrics endpoint port
```

**Security:**
```bash
GRAFANA_ADMIN_PASSWORD=<strong-password>
REDIS_PASSWORD=<strong-password>  # If Redis auth enabled
```

### B. Port Reference

| Port | Service | Purpose | External Access |
|------|---------|---------|-----------------|
| 5000 | AutoVoice API | REST API and metrics | Yes |
| 8080 | WebSocket | Real-time communication | Yes |
| 6379 | Redis | Cache and queue | No (internal only) |
| 9090 | Prometheus | Metrics collection | Admin only |
| 3000 | Grafana | Monitoring dashboards | Admin only |

### C. Volume Reference

| Volume | Purpose | Backup Priority | Size Estimate |
|--------|---------|-----------------|---------------|
| `./data` | Application data | High | 1-10 GB |
| `./models` | ML models | High | 5-20 GB |
| `./logs` | Application logs | Medium | 1-5 GB |
| `./config` | Configuration | Critical | < 100 MB |
| `redis-data` | Redis persistence | Medium | 100 MB - 1 GB |
| `prometheus-data` | Metrics history | Low | 1-10 GB |
| `grafana-data` | Dashboards | Low | < 100 MB |

### D. Useful Commands Cheat Sheet

```bash
# Start services
docker compose up -d
docker compose --profile monitoring up -d

# Stop services
docker compose down
docker compose down -v  # Also remove volumes

# View logs
docker compose logs -f auto-voice-app
docker compose logs --tail=100 --since=1h

# Restart service
docker compose restart auto-voice-app

# Update and restart
docker compose pull
docker compose up -d

# Health check
curl http://localhost:5000/health | jq .

# Metrics
curl http://localhost:5000/metrics | grep autovoice

# GPU status
nvidia-smi
watch -n 1 nvidia-smi

# Container stats
docker stats

# Clean up
docker system prune -a
docker volume prune
```

### E. Related Documentation

- **Monitoring Guide**: `docs/monitoring-guide.md` - Detailed monitoring setup and PromQL queries
- **Production Readiness Checklist**: `docs/production_readiness_checklist.md` - Pre-deployment validation
- **API Documentation**: `docs/api.md` - API endpoints and usage
- **Architecture Overview**: `docs/architecture.md` - System design and components

---

**Document Version**: 1.0
**Last Updated**: 2025-11-01
**Next Review**: 2025-12-01
**Maintained By**: DevOps Team
